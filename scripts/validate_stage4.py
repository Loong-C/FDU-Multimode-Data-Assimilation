"""Cross-check the stage-four exports using DOP853 and OpenCV reference projections."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from badminton.camera import Camera
from badminton.court import DOUBLES_HALF_WIDTH, HALF_LENGTH, SINGLES_HALF_WIDTH, net_height
from badminton.inverse import fit_track
from badminton.inverse_cases import iter_cases
from badminton.inverse_study import fit_observations
from badminton.reconstruct import read_observations


FIELDS = ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s")
SCORE_ATOL = 1e-4


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def read_grouped(path):
    rows = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            rows[row["case"]].append(row)
    return rows


def theta(row, prefix=""):
    return np.array([float(row[prefix+field]) for field in FIELDS])


def rms(difference):
    return float(np.sqrt(np.mean(np.linalg.norm(difference, axis=1)**2)))


def reference_flight(initial, k):
    def rhs(_, state):
        v = state[3:]
        return np.r_[v, -k*np.linalg.norm(v)*v - [0, 0, 9.81]]

    def ground(_, state):
        return state[2]

    def apex(_, state):
        return state[5]

    ground.terminal = True
    ground.direction = apex.direction = -1
    sol = solve_ivp(rhs, (0., 12.), initial, method="DOP853", events=(ground, apex),
                    rtol=1e-11, atol=1e-12, dense_output=True)
    require(sol.success and len(sol.t_events[0]) == 1, "Reference flight did not reach the floor")
    height = sol.y_events[1][0, 2] if len(sol.y_events[1]) else initial[2]
    return sol.sol, sol.t_events[0][0], sol.y_events[0][0, :2], height


def project(camera, xyz):
    pixels, _ = cv2.projectPoints(np.ascontiguousarray(xyz), cv2.Rodrigues(camera.R)[0], camera.t, camera.K,
                                  np.array([camera.k1, 0., 0., 0., 0.]))
    return pixels.reshape(-1, 2)


def check_exports(directory, cases):
    scores = read_grouped(directory / "scores.csv")
    starts = read_grouped(directory / "multistart.csv")
    observations = read_grouped(directory / "observations.csv")
    require(len(cases) == 50 and set(cases) == scores.keys() == starts.keys() == observations.keys(),
            "Case IDs/counts disagree")
    maxima, clean_errors = defaultdict(float), []
    samples = {}
    for name, case in cases.items():
        require(len(scores[name]) == 1, f"Duplicate score for {name}")
        row = scores[name][0]
        require((row["view"], row["condition"], row["shot"]) == (case.view, case.condition, case.shot)
                and int(row["seed"]) == case.seed and int(row["robust"]) == case.robust
                and float(row["k_m_inv"]) == case.k and float(row["sigma_px"]) == case.sigma_px
                and int(row["video_frames"]) == case.n_video_frames
                and int(row["visible_frames"]) == case.n_visible_frames, f"Case metadata mismatch for {name}")
        estimate = theta(row)
        data = observations[name]
        times = np.array([float(r["t_s"]) for r in data])
        pixels = np.array([[float(r["u_px"]), float(r["v_px"])] for r in data])
        frames = np.array([int(r["frame"]) for r in data])
        require(len(data) == int(row["observations"]) and int(row["bad_detections"]) == case.outlier_mask.sum(),
                f"Detection counts disagree for {name}")
        np.testing.assert_array_equal(frames, case.frame_index)
        np.testing.assert_array_equal(times, frames / 25)
        np.testing.assert_array_equal(pixels, case.uv)
        samples[name] = (frames, pixels)

        candidates = starts[name]
        require([int(r["start"]) for r in candidates] == [1, 2, 3], f"Missing starts for {name}")
        viable = [r for r in candidates if r["converged"] == "1"]
        require(len(viable) == int(row["converged_starts"]), f"Incorrect success count for {name}")
        require(len(np.unique([theta(r, "seed_") for r in candidates], axis=0)) == 3,
                f"Starts are identical for {name}")
        np.testing.assert_array_equal(estimate, theta(min(viable, key=lambda r: float(r["cost"]))))
        for candidate in viable:
            fit_state = reference_flight(theta(candidate), case.k)[0]
            predicted = project(case.camera, fit_state(times)[:3].T)
            np.testing.assert_allclose(rms(predicted-pixels), float(candidate["observed_rmse_px"]),
                                       rtol=0, atol=SCORE_ATOL)

        # The reference uses DOP853, event roots, uniform quadrature and OpenCV.
        # The production fit and scorer use LSODA, RK4 and the Camera class.
        pred, landing_s, landing_xy, apex = reference_flight(estimate, case.k)
        truth, true_s, true_xy, true_apex = reference_flight(case.truth_theta, case.k)
        uv = project(case.camera, pred(times)[:3].T)
        uv_clean = project(case.camera, truth(times)[:3].T)
        stop = min(landing_s, true_s)
        t = np.linspace(0, stop, 2001)
        delta = pred(t)[:3] - truth(t)[:3]
        state_rmse = np.sqrt(np.trapezoid(np.sum(delta**2, axis=0), t) / stop)
        measured = {
            "position_rmse_m": state_rmse, "landing_xy_error_m": np.linalg.norm(landing_xy - true_xy),
            "landing_time_error_s": abs(landing_s - true_s), "apex_height_error_m": abs(apex - true_apex),
            "launch_position_error_m": np.linalg.norm(estimate[:3]-case.truth_theta[:3]),
            "launch_velocity_error_m_s": np.linalg.norm(estimate[3:]-case.truth_theta[3:]),
            "scored_duration_s": stop, "observed_rmse_px": rms(uv - pixels),
            "clean_image_rmse_px": rms(uv - uv_clean),
        }
        for field, value in measured.items():
            deviation = abs(float(row[field]) - value)
            # Small integration/sampling differences are expected (RK4 at 4 ms vs DOP853).
            require(deviation < SCORE_ATOL, f"{name} {field} discrepancy {deviation}")
            maxima[field] = max(maxima[field], float(deviation))
        if case.condition == "clean":
            clean_errors.append([measured["launch_position_error_m"], measured["launch_velocity_error_m_s"]])
    clean = np.max(clean_errors, axis=0)
    require(len(clean_errors) == 5 and clean[0] < 1e-4 and clean[1] < 1e-3, "Clean launches not recovered")
    return samples, dict(cases=len(cases), starts=sum(map(len, starts.values())),
                         observations=sum(map(len, observations.values())), score_absolute_tolerance=SCORE_ATOL,
                         max_score_discrepancy=dict(maxima),
                         clean_max_launch_error_m=float(clean[0]), clean_max_velocity_error_m_s=float(clean[1]))


def check_pairs(samples):
    for seed in (401, 402, 403):
        def get(cond):
            return samples[f"clear_oblique_{cond}_{seed}"]

        full_frames, full = get("noise")
        np.testing.assert_array_equal(full_frames, np.arange(55))
        clean = samples["clear_oblique_clean_0"][1]
        np.testing.assert_allclose(get("high_noise")[1]-clean, 3*(full-clean), atol=1e-10, rtol=0)
        for cond, count in (("dropout", 40), ("occlusion", 40), ("short", 4), ("medium", 12)):
            frames, pixels = get(cond)
            require(len(frames) == count, f"Incorrect {cond} count")
            np.testing.assert_array_equal(full[frames], pixels)
        np.testing.assert_array_equal(get("occlusion")[0], np.r_[0:15, 30:55])
        np.testing.assert_array_equal(get("short")[0], np.arange(4))
        dirty = get("outliers")[1]
        np.testing.assert_array_equal(dirty, get("outliers_robust")[1])
        shift = np.linalg.norm(dirty-full, axis=1)
        changed = shift[shift > 0]
        require(len(changed) == 6 and np.all((changed >= 35) & (changed <= 75)), "Incorrect false detections")
    return {"paired_seeds": 3, "baseline_frames": 55, "retained_dropout_or_occlusion": 40,
            "identical_detections_with_swapped_loss": 55, "false_detections_per_seed": 6}


def check_ambiguity(directory, cases):
    report = json.loads((directory / "ambiguity.json").read_text(encoding="utf-8"))
    case = cases[report["case_name"]]
    best = np.asarray(report["best_theta"])
    alternative = np.asarray(report["selected"]["theta"])
    pixels, touchdowns = [], []
    for initial in (best, alternative):
        state, end, xy, _ = reference_flight(initial, case.k)
        crossing = brentq(lambda t: state(t)[0], 0, end)
        y, z = state(crossing)[1:3]
        require(abs(y) <= DOUBLES_HALF_WIDTH and z > net_height(y) and initial[0]*xy[0] < 0
                and abs(xy[0]) <= HALF_LENGTH and abs(xy[1]) <= SINGLES_HALF_WIDTH
                and abs(initial[0]) <= HALF_LENGTH and abs(initial[1]) <= SINGLES_HALF_WIDTH,
                "An ambiguity example falls outside the singles court or hits the net")
        pixels.append(project(case.camera, state(case.times)[:3].T))
        touchdowns.append(xy)
    short_rmse = [rms(p - case.uv) for p in pixels]
    gap = rms(pixels[0] - pixels[1])
    landing_gap = np.linalg.norm(touchdowns[0]-touchdowns[1])
    np.testing.assert_allclose([*short_rmse, gap, landing_gap],
                               [report["best_short_rmse_px"], report["selected"]["short_rmse_px"],
                                report["selected"]["prediction_gap_px"], report["selected"]["landing_gap_m"]],
                               rtol=0, atol=1e-4)
    require(max(short_rmse) < 1.5 and gap < 1 and landing_gap > 1, "Counterexample too weak")
    full = cases[report["paired_full_case"]]
    refit = fit_track(full.camera, full.times, full.uv, k=full.k, sigma_px=full.sigma_px,
                      launch_height_m=alternative[2])
    state = reference_flight(refit.theta, full.k)[0]
    full_error = rms(project(full.camera, state(full.times)[:3].T)-full.uv)
    np.testing.assert_allclose(full_error, report["full_refit_at_selected_height_rmse_px"], atol=1e-4, rtol=0)
    require(full_error > 3, "Full-arc detections do not distinguish the profile")
    return dict(short_observations=len(case.times), short_rmse_px=short_rmse, prediction_gap_px=gap,
                landing_gap_m=float(landing_gap), full_fixed_height_refit_rmse_px=full_error)


def check_input_and_camera(directory, cases):
    stage3 = ROOT / "results/stage3"
    raw = json.loads((stage3 / "camera.json").read_text(encoding="utf-8"))
    fuzhou = cases["clear_fuzhou_noise_401"]
    np.testing.assert_allclose(fuzhou.camera.P, raw["P"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(fuzhou.camera.center, raw["center_world_m"], rtol=0, atol=1e-10)
    state = reference_flight(fuzhou.truth_theta, fuzhou.k)[0]
    coords = project(fuzhou.camera, state(np.arange(55)/25)[:3].T)
    physical = np.flatnonzero(np.all((coords >= 0) & (coords < raw["image_size"]), axis=1))
    np.testing.assert_array_equal(fuzhou.frame_index, physical)
    require(len(physical) == 25, "Unexpected calibrated field of view")

    # Exercise the normal CSV reader with stage-three trajectory files and stage-four exports.
    camera = Camera.from_dict(json.loads((stage3 / "virtual-camera.json").read_text(encoding="utf-8")))
    times, pixels = read_observations(stage3 / "virtual-trajectories.csv", track="clear")
    sampled = np.isclose(25*times, np.rint(25*times), rtol=0, atol=1e-10)
    t, uv = times[sampled], pixels[sampled]
    kept = (t >= 0.12) & ((t < 0.4) | (t > 0.8))
    # This object has no truth, flight length or shot-type attributes.
    bare = SimpleNamespace(camera=camera, times=t[kept], uv=uv[kept], k=fuzhou.k, sigma_px=0.01, robust=False)
    selected = fit_observations(bare)
    missing = uv.copy()
    missing[~kept] = np.nan
    gaps = fit_track(camera, t, missing, k=bare.k, sigma_px=bare.sigma_px)
    np.testing.assert_allclose(selected.theta, gaps.theta, rtol=0, atol=1e-8)
    original = cases["clear_oblique_clean_0"].truth_theta
    pos, vel = np.linalg.norm(gaps.theta[:3]-original[:3]), np.linalg.norm(gaps.theta[3:]-original[3:])
    require(pos < 1e-3 and vel < 1e-2 and t[kept][0] > 0.1, "Launch time was reset at first detection")
    ft, fu = read_observations(directory / "observations.csv", track=fuzhou.name)
    np.testing.assert_array_equal(ft, fuzhou.times)
    np.testing.assert_array_equal(fu, fuzhou.uv)
    partial = bare.uv.copy()
    partial[0, 0] = np.nan
    invalid = [(bare.times[:2], bare.uv[:2], 1.), (bare.times, partial, 1.),
               (bare.times[::-1], bare.uv, 1.), (bare.times-1., bare.uv, 1.), (bare.times, bare.uv, 0.)]
    for bad_times, bad_pixels, bad_sigma in invalid:
        try:
            fit_track(camera, bad_times, bad_pixels, sigma_px=bad_sigma)
        except ValueError:
            continue
        raise AssertionError("Invalid time, pixel or weight accepted")
    return dict(fuzhou_in_sensor_frames=len(physical), fuzhou_total_frames=55,
                stage3_delayed_first_detection_s=float(t[kept][0]), stage3_retained_observations=int(kept.sum()),
                invalid_inputs_rejected=len(invalid), launch_error_m=float(pos), launch_velocity_error_m_s=float(vel))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "results/stage4")
    parser.add_argument("--out", type=Path, default=ROOT / "results/stage4/validation.json")
    args = parser.parse_args()
    cases = {c.name: c for c in iter_cases()}
    samples, exports = check_exports(args.data, cases)
    report = dict(exports=exports, pairings=check_pairs(samples), ambiguity=check_ambiguity(args.data, cases),
                  interfaces=check_input_and_camera(args.data, cases), passed=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Verified {exports['cases']} cases, {exports['starts']} starts and {exports['observations']} detections.")


if __name__ == "__main__":
    main()
