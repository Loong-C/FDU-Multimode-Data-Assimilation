"""Reconstruct reviewed Fuzhou strokes in their broadcast camera and evaluate withheld frames."""

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .camera import Camera, fit_camera
from .court import DOUBLES_HALF_WIDTH, net_height
from .flight import DEFAULT_DRAG, GRAVITY
from .posterior import PosteriorModel, fit_map, flight_states
from .posterior_sampling import sample_posterior
from .reconstruct import ROOT, write_json

PIXEL_SD = 2.5  # coordinate localization scale (px), set before testing
FOOT_SD = 1.0  # includes foot placement, arm reach and contact-frame error (m)
DRAG = {"fixed_drag": DEFAULT_DRAG, "joint_drag": None, "vacuum": 0.0}
STATE = ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s")


class StrokeModel(PosteriorModel):
    """The observed segment ends at another racket contact, with unknown positive height."""

    def __init__(self, camera, times, pixels, *, horizon, player, receiver, fixed_k, sigma_px=PIXEL_SD):
        constraints = {} if player is None else {"player_xy": (player, FOOT_SD)}
        super().__init__(
            camera, times, pixels, sigma_px=sigma_px, fixed_k=fixed_k, constraints=constraints
        )
        if not np.isfinite(horizon) or horizon < self.times[-1]:
            raise ValueError("The next contact must be at or after the last trusted detection")
        if receiver is not None and (np.shape(receiver) != (2,) or not np.isfinite(receiver).all()):
            raise ValueError("Expected a finite receiver foot proxy in court coordinates")
        self.receiver = receiver
        self.airborne_until = horizon
        self.grid = np.unique(np.r_[self.grid, horizon])
        self.indices = np.searchsorted(self.grid, self.times)
        self.end_index = np.searchsorted(self.grid, horizon)

    def end_position(self, x):
        theta, k = self.unpack(x)
        return theta[:3] + self._states(*theta[3:], k)[self.end_index, :3]

    def final_height(self, x):
        return self.end_position(x)[2]

    def _evaluate(self, x, strict):
        r = super()._evaluate(x, strict)
        if r is None:
            return None
        end = self.end_position(x)
        if strict and end[2] < -1e-8:
            return None
        if self.receiver is not None:
            r = np.r_[r, (end[:2] - self.receiver) / FOOT_SD]
        return r


@dataclass
class Shot:
    meta: dict
    camera: Camera
    frame: np.ndarray
    t: np.ndarray
    uv: np.ndarray
    role: np.ndarray

    @property
    def horizon(self):
        return (self.meta["end_frame_exclusive"] - self.meta["start_frame"]) / self.meta["fps"]

    @property
    def key(self):
        return {k: self.meta[k] for k in ("shot_id", "rally_id", "split", "shot_type")}


def inputs(directory):
    def load(name):
        return json.loads((directory / name).read_text(encoding="utf-8"))

    cameras = {r: Camera.from_dict(v) for r, v in load("cameras.json")["cameras"].items()}
    grouped = defaultdict(list)
    with (directory / "observations.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            grouped[row["shot_id"]].append(row)
    shots = []
    for meta in load("shots.json")["shots"]:
        rows = sorted(grouped[meta["shot_id"]], key=lambda r: int(r["frame_index"]))
        frame = np.array([int(r["frame_index"]) for r in rows])
        if not np.array_equal(frame, np.arange(meta["start_frame"], meta["end_frame_exclusive"])):
            raise ValueError(f"Incomplete frame audit for {meta['shot_id']}")
        t = (frame - frame[0]) / meta["fps"]
        uv = np.array([[float(r["u_px"] or "nan"), float(r["v_px"] or "nan")] for r in rows])
        role = np.array([r["role"] for r in rows])
        if not np.isfinite(uv[np.isin(role, ["fit", "holdout"])]).all() or not (role == "holdout").any():
            raise ValueError(f"Invalid observed pixels for {meta['shot_id']}")
        shots.append(Shot(meta, cameras[meta["rally_id"]], frame, t, uv, role))
    return shots, load("camera-checks.json")


def ground_xy(camera, uv):
    c, rays = camera.rays(np.asarray(uv))
    return (c - rays * (c[2] / rays[2]))[:2]


def infer(shot, drag, *, camera=None, keep=None, pixels=None, shift=0.0, anchors=True):
    camera = camera or shot.camera
    used = (shot.role == "fit") if keep is None else keep.copy()
    used &= shot.t >= shift - 1e-10
    player = shot.meta["player_xy_m"] if anchors else None
    receiver = shot.meta["receiver_xy_m"] if anchors else None
    if camera is not shot.camera and anchors:
        # Keep the measured feet pixels fixed when changing calibration.
        if player is not None:
            player = ground_xy(camera, shot.camera.project([*player, 0.0]))
        if receiver is not None:
            receiver = ground_xy(camera, shot.camera.project([*receiver, 0.0]))
    model = StrokeModel(
        camera,
        np.maximum(shot.t[used] - shift, 0),
        (shot.uv if pixels is None else pixels)[used],
        horizon=shot.horizon - shift,
        player=player,
        receiver=receiver,
        fixed_k=DRAG[drag],
    )
    return model, fit_map(model, max_nfev=1500)


def rms(deltas):
    return float(np.sqrt(np.mean(np.sum(deltas**2, axis=1))))


def metrics(shot, model, fit, shift=0.0):
    theta, k = model.unpack(fit.x)
    selected = shot.t >= shift - 1e-10
    states = np.full((len(shot.t), 6), np.nan)
    states[selected] = flight_states(theta, np.maximum(shot.t[selected] - shift, 0), k)
    pred = np.full_like(shot.uv, np.nan)
    pred[selected] = model.camera.project(states[selected, :3])
    scores = {}
    for subset in ("fit", "holdout"):
        used = (shot.role == subset) & selected
        scores[subset + "_count"] = int(used.sum())
        scores[subset + "_rmse_px"] = rms(pred[used] - shot.uv[used]) if used.any() else None
    return scores, states, pred


def net_check(shot, theta, k, shift=0.0):
    horizon = shot.horizon - shift
    end = flight_states(theta, [horizon], k)[0]
    crossing = None
    if theta[0] * end[0] < 0:
        crossing = brentq(lambda t: flight_states(theta, [t], k)[0, 0], 0.0, horizon)
    y = clearance = None
    if crossing is not None:
        _, y, z = flight_states(theta, [crossing], k)[0, :3]
        if abs(y) <= DOUBLES_HALF_WIDTH:
            clearance = float(z - net_height(y))
    issues = []
    if crossing is None:
        issues.append("no_net_crossing")
    elif clearance is None:
        issues.append("outside_net_posts")
    elif clearance < 0:
        issues.append("below_net")
    for name, z in (("launch", theta[2]), ("next_contact", end[2])):
        if z < 0.15 or z > 4.0:
            issues.append(name + "_height_outside_screen")
    if shot.meta["player_xy_m"] is not None and theta[0] * shot.meta["player_xy_m"][0] <= 0:
        issues.append("wrong_hitter_half")
    if shot.meta["receiver_xy_m"] is not None and end[0] * shot.meta["receiver_xy_m"][0] <= 0:
        issues.append("wrong_receiver_half")
    return dict(
        next_contact_z_m=end[2],
        next_contact_x_m=end[0],
        next_contact_y_m=end[1],
        net_time_s=crossing,
        net_y_m=y,
        net_clearance_m=clearance,
        physics_screen=";".join(issues) if issues else "passes",
    )


def perturb_camera(shot, checks):
    data = [p for p in checks[shot.meta["rally_id"]]["landmarks"] if p["split"] == "train"]
    xyz = np.array([p["world_m"] for p in data])
    pts = np.array([p["pixel"] for p in data])
    rng = np.random.default_rng(6017)
    return fit_camera(xyz, pts + rng.normal(0, 1, pts.shape), shot.camera.image_size)


def variants(shot, drag, base, checks):
    fitmask = shot.role == "fit"
    sparse = fitmask.copy()
    sparse[np.flatnonzero(fitmask)[1:-1:3]] = False
    rng = np.random.default_rng(6009 + shot.meta["start_frame"])
    scenarios = [(candidate, candidate, {}) for candidate in DRAG]
    scenarios += [
        ("no_foot_proxies", drag, {"anchors": False}),
        ("court_train_jitter_1px", drag, {"camera": perturb_camera(shot, checks)}),
        ("launch_1_frame_earlier", drag, {"shift": -1 / shot.meta["fps"]}),
        ("launch_1_frame_later", drag, {"shift": 1 / shot.meta["fps"]}),
        ("drop_every_third_interior_detection", drag, {"keep": sparse}),
    ]
    for i in range(3):
        scenarios.append(
            (f"pixel_jitter_{i+1}", drag, {"pixels": shot.uv + rng.normal(0, PIXEL_SD, shot.uv.shape)})
        )
    _, ref, _ = metrics(shot, *base)
    rows = []
    for name, candidate, kwargs in scenarios:
        row = dict(**shot.key, variant=name)
        try:
            model, fit = base if name == drag else infer(shot, candidate, **kwargs)
            score, path, _ = metrics(shot, model, fit, kwargs.get("shift", 0.0))
            ok = np.isfinite(path).all(axis=1)
            theta, k = model.unpack(fit.x)
            row.update(
                status="ok",
                observations_used=len(model.times),
                **score,
                displacement_rms_m=rms(path[ok, :3] - ref[ok, :3]),
                launch_speed_m_s=np.linalg.norm(theta[3:]),
                launch_z_m=theta[2],
                k_m_inv=k,
                error="",
                **net_check(shot, theta, k, kwargs.get("shift", 0.0)),
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            row.update(status="failed", error=str(exc))
        rows.append(row)
    return rows


def matched_simulation(shot, base, drag):
    """Conditional parametric bootstrap with independent integration, exact observed-frame mask."""
    model, fit = base
    theta, k = model.unpack(fit.x)

    def rhs(_, state):
        v = state[3:]
        return np.r_[v, -k * np.linalg.norm(v) * v - [0.0, 0.0, GRAVITY]]

    sol = solve_ivp(
        rhs, (0.0, shot.horizon), theta, method="DOP853", dense_output=True, rtol=1e-10, atol=1e-11
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    truth = sol.sol(shot.t).T
    ideal = shot.camera.project(truth[:, :3])
    w, h = shot.camera.image_size
    intopicture = (ideal[:, 0] >= 0) & (ideal[:, 0] < w) & (ideal[:, 1] >= 0) & (ideal[:, 1] < h)
    compare_role = np.where(intopicture, shot.role, "missing")
    real = Shot(shot.meta, shot.camera, shot.frame, shot.t, shot.uv, compare_role)
    real_fit = base if np.array_equal(compare_role, shot.role) else infer(real, drag)
    real_score, _, _ = metrics(real, *real_fit)
    rows = []
    for seed in (6101, 6102, 6103, 6104, 6105):
        rng = np.random.default_rng([seed, shot.meta["start_frame"]])
        uv = ideal + rng.normal(0, PIXEL_SD, ideal.shape)
        # Simulate a detector at the image border with the same valid frame mask.
        invalid = ~((uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h))
        while np.any(invalid & intopicture):
            idx = invalid & intopicture
            uv[idx] = ideal[idx] + rng.normal(0, PIXEL_SD, (idx.sum(), 2))
            invalid = ~((uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h))
        aux = dict(shot.meta)
        for key, xy in (("player_xy_m", theta[:2]), ("receiver_xy_m", sol.sol(shot.horizon)[:2])):
            if aux[key] is not None:
                aux[key] = xy + rng.normal(0, FOOT_SD, 2)
        sim = Shot(aux, shot.camera, shot.frame, shot.t, uv, compare_role)
        recovered = infer(sim, drag)
        score, path, _ = metrics(sim, *recovered)
        rows.append(
            dict(
                **shot.key,
                seed=seed,
                fit_count=score["fit_count"],
                holdout_count=score["holdout_count"],
                real_holdout_rmse_px=real_score["holdout_rmse_px"],
                sim_holdout_rmse_px=score["holdout_rmse_px"],
                sim_position_rmse_m=rms(path[:, :3] - truth[:, :3]),
            )
        )
    return rows


def write_table(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def posterior_intervals(shot, model, fit):
    """Stage-five chains conditional on this shot's camera, contact frames and observation model."""
    row = dict(**shot.key, seed=6301 + shot.meta["start_frame"])
    try:
        for draws in (1200, 2400):
            result = sample_posterior(model, fit, seed=row["seed"], chains=4, warmup=400, draws=draws)
            d = result.diagnostics
            row.update(
                chains=4,
                draws_per_chain=draws,
                warmup_per_chain=400,
                max_rhat=d["max_rhat"],
                min_bulk_ess=d["min_bulk_ess"],
                min_tail_ess=d["min_tail_ess"],
            )
            if d["converged"]:
                break
        if not d["converged"]:
            return dict(**row, status="diagnostics_incomplete")
        states = result.chains.reshape(-1, len(fit.x))
        variables = {
            "launch_x_m": states[:, 0],
            "launch_z_m": states[:, 2],
            "speed_m_s": np.linalg.norm(states[:, 3:6], axis=1),
        }
        if model.fixed_k is None:
            variables["k_m_inv"] = np.exp(states[:, 6])
        for key, values in variables.items():
            for q, value in zip(("p05", "p50", "p95"), np.quantile(values, [0.05, 0.5, 0.95])):
                row[f"{key}_{q}"] = value
        return dict(**row, status="converged")
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        return dict(**row, status="sampling_failed", error=str(exc))


def run(directory, out):
    shots, checks = inputs(directory)
    dev, test = ([s for s in shots if s.meta["split"] == k] for k in ("development", "test"))
    if not dev or not test or {s.meta["rally_id"] for s in dev} & {s.meta["rally_id"] for s in test}:
        raise ValueError("Require disjoint development and test rallies")
    # Select on development only. Test observations first enter fitting/scoring after selection.
    tuning = {}
    cache = {}
    for drag in ("fixed_drag", "joint_drag"):
        values = []
        for shot in dev:
            fit = infer(shot, drag)
            cache[(shot.meta["shot_id"], drag)] = fit
            values.append(metrics(shot, *fit)[0]["holdout_rmse_px"])
        tuning[drag] = dict(per_stroke_holdout_rmse_px=values, mean=float(np.mean(values)))
    selected = min(tuning, key=lambda name: tuning[name]["mean"])
    fits, residuals, tracks, sensitivity, sims, posteriors = [], [], [], [], [], []
    for shot in dev + test:
        key = (shot.meta["shot_id"], selected)
        base = cache[key] if key in cache else infer(shot, selected)
        posteriors.append((shot, *base))
        model, fit = base
        theta, k = model.unpack(fit.x)
        scores, path, uv = metrics(shot, *base)
        row = dict(
            **shot.key,
            model=selected,
            **scores,
            **dict(zip(STATE, theta)),
            k_m_inv=k,
            speed_m_s=np.linalg.norm(theta[3:]),
            **net_check(shot, theta, k),
        )
        for i, frame in enumerate(shot.frame):
            tracks.append(
                dict(
                    **shot.key,
                    frame_index=frame,
                    t_s=shot.t[i],
                    **dict(zip(STATE, path[i])),
                    u_fit_px=uv[i, 0],
                    v_fit_px=uv[i, 1],
                )
            )
            if shot.role[i] in ("fit", "holdout"):
                err = uv[i] - shot.uv[i]
                residuals.append(
                    dict(
                        **shot.key,
                        frame_index=frame,
                        role=shot.role[i],
                        t_s=shot.t[i],
                        u_px=shot.uv[i, 0],
                        v_px=shot.uv[i, 1],
                        u_fit_px=uv[i, 0],
                        v_fit_px=uv[i, 1],
                        du_px=err[0],
                        dv_px=err[1],
                        error_px=np.linalg.norm(err),
                    )
                )
        sensitivity.extend(variants(shot, selected, base, checks))
        sims.extend(matched_simulation(shot, base, selected))
        # Local linearized scales supplement model/input sensitivity; they condition on the camera and priors.
        row.update(
            launch_z_local_sd_m=np.sqrt(fit.cov[2, 2]),
            initial_x_local_sd_m=np.sqrt(fit.cov[0, 0]),
            speed_local_sd_m_s=np.sqrt(theta[3:] @ fit.cov[3:6, 3:6] @ theta[3:])
            / np.linalg.norm(theta[3:]),
            log_k_local_sd=np.sqrt(fit.cov[-1, -1]) if len(fit.x) == 7 else "",
            converged_starts=sum(s.success for s in fit.starts),
            gn_condition=fit.condition,
        )
        fits.append(row)
    summary = {}
    for split in ("development", "test"):
        held = [r for r in residuals if r["split"] == split and r["role"] == "holdout"]
        rows = [r for r in fits if r["split"] == split]
        summary[split] = dict(
            shots=len(rows),
            heldout_pixels=len(held),
            heldout_rmse_px=float(np.sqrt(np.mean([r["error_px"] ** 2 for r in held]))),
            median_shot_holdout_rmse_px=float(np.median([r["holdout_rmse_px"] for r in rows])),
            physics_pass=sum(r["physics_screen"] == "passes" for r in rows),
        )
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("fits", fits),
        ("trajectories", tracks),
        ("residuals", residuals),
        ("sensitivity", sensitivity),
        ("matched-simulation", sims),
    ):
        write_table(out / (name + ".csv"), rows)
    write_json(
        out / "study.json",
        dict(
            primary_model=selected,
            development_rule="minimum mean per-stroke held-out pixel RMS; fixed wins a tie",
            development_ids=[s.meta["shot_id"] for s in dev],
            test_ids=[s.meta["shot_id"] for s in test],
            sigma_px=PIXEL_SD,
            foot_proxy_sigma_m=FOOT_SD,
            development_comparison=tuning,
            summary=summary,
            flight_end="Next racket contact; its ground-plane foot proxy is soft. No landing time/position was assumed.",
            physics_screen="Post-fit consistency: cross x=0 between posts above the nominal net, both contacts on the respective players' halves, both heights between 0.15 and 4 m. The height interval is a screening convention.",
            uncertainty="Local curvature conditions on the model, A-camera and contact times; it ignores physical truncation at boundaries. posterior.csv reports conditional 5th to 95th percentiles only for chains with max rank R-hat <1.01 and min bulk/tail ESS >=400. The fixed k, if selected, has no data-estimated interval.",
            sensitivity="Separate refits. Camera jitter perturbs training landmarks only. Time jitter shifts the first contact, fixes the next, and compares shared original timestamps.",
            synthetic_control="Conditional on the real MAP launch; DOP853, same camera/timestamps/visibility split, new 2.5 px Gaussian detections and 1 m foot proxies. 3D errors refer to this synthetic model only.",
        ),
    )
    print(
        f"{len(shots)} real strokes, selected {selected}, test heldout RMS {summary['test']['heldout_rmse_px']:.2f} px -> {out}"
    )
    write_table(out / "posterior.csv", [posterior_intervals(*run) for run in posteriors])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "results/stage6")
    parser.add_argument("--out", type=Path, default=ROOT / "results/stage6")
    args = parser.parse_args()
    run(args.data, args.out)
