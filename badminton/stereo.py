"""Evaluate a monocular shuttle fit on a synchronized independent camera."""

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .camera import Camera
from .flight import DEFAULT_DRAG
from .inverse import fit_track
from .posterior import fit_map, flight_states, landing_event
from .real_study import StrokeModel
from .reconstruct import read_observations, write_json


@dataclass(frozen=True)
class Clock:
    """Affine map taking a timestamp on camera B to the timestamp on A."""

    scale: float
    offset: float

    def to_a(self, b):
        return self.scale*np.asarray(b) + self.offset


def synchronize(anchors):
    """Fit with independently annotated sync events and score held-out events."""
    anchors = list(anchors)
    try:
        a = np.array([float(r["t_a_s"]) for r in anchors])
        b = np.array([float(r["t_b_s"]) for r in anchors])
        groups = np.array([r["split"].strip() for r in anchors], dtype=str)
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Sync CSV needs t_a_s,t_b_s,split in every row") from exc
    used = groups == "fit"
    if not np.isfinite([a, b]).all() or not np.isin(groups, ["fit", "check"]).all():
        raise ValueError("Sync anchors require finite timestamps and a fit or check split")
    if len(np.unique(a)) != len(a) or len(np.unique(b)) != len(b) or np.any(np.diff(a[np.argsort(b)]) <= 0):
        raise ValueError("Sync anchors must describe distinct events in the same order on both cameras")
    if used.sum() < 2:
        raise ValueError("Need at least two distinct fit anchors for clock offset and drift")
    ac, bc = a[used].mean(), b[used].mean()
    db = b[used] - bc
    scale = np.dot(db, a[used]-ac) / np.dot(db, db)
    clock = Clock(float(scale), float(ac-scale*bc))
    residual = clock.to_a(b) - a
    report = dict(method="independent_sync_events", scale_b_to_a=clock.scale, offset_a_s=clock.offset,
                  fit_anchor_range_b_s=[float(b[used].min()), float(b[used].max())],
                  fit_anchors=int(used.sum()), fit_rmse_s=float(np.sqrt(np.mean(residual[used]**2))),
                  check_anchors=int((~used).sum()),
                  check_rmse_s=float(np.sqrt(np.mean(residual[~used]**2))) if (~used).any() else None)
    return clock, report


def _rays_midpoint(a, b, ua, ub, min_angle_deg):
    ca, ra = a.rays(ua)
    cb, rb = b.rays(ub)
    # The acute angle detects ill-conditioned nearly parallel and opposing rays.
    angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(ra, rb)), 0, 1))))
    if angle < min_angle_deg:
        raise ValueError("weak_parallax")
    distances = np.linalg.lstsq(np.column_stack((ra, -rb)), cb-ca, rcond=None)[0]
    if np.min(distances) <= 0:
        raise ValueError("behind_camera")
    pa, pb = ca+distances[0]*ra, cb+distances[1]*rb
    return (pa+pb)/2, angle, float(np.linalg.norm(pa-pb))


def triangulate(a, b, ua, ub, *, min_angle_deg=2., max_pair_reprojection_px=6.):
    """Return a ray-midpoint reference and its independent geometric quality measures."""
    if not (0 < min_angle_deg < 90 and 0 < max_pair_reprojection_px < np.inf):
        raise ValueError("Triangulation needs a positive pixel tolerance and a convergence angle between 0 and 90")
    xyz, angle, gap = _rays_midpoint(a, b, ua, ub, min_angle_deg)
    ea = float(np.linalg.norm(a.project(xyz)-ua))
    eb = float(np.linalg.norm(b.project(xyz)-ub))
    if max(ea, eb) > max_pair_reprojection_px:
        raise ValueError("inconsistent_rays")
    return xyz, angle, gap, ea, eb


def _at_time(t, times, uv, max_gap):
    """Linearly interpolate camera A only across short gaps inside this stroke."""
    j = int(np.searchsorted(times, t))
    for i in (j-1, j):
        if 0 <= i < len(times) and abs(times[i]-t) < 1e-8:
            left = i-1 if i and times[i]-times[i-1] <= max_gap else i
            right = i+1 if i+1 < len(times) and times[i+1]-times[i] <= max_gap else i
            derivative = (uv[right]-uv[left])/(times[right]-times[left]) if left != right else None
            return uv[i], derivative, 1.
    if j == 0 or j == len(times):
        raise ValueError("no_bracket")
    dt = times[j]-times[j-1]
    if dt > max_gap:
        raise ValueError("long_a_gap")
    w = (t-times[j-1])/dt
    return (1-w)*uv[j-1]+w*uv[j], (uv[j]-uv[j-1])/dt, (1-w)**2+w*w


def _reference_error(a, b, ua, ub, velocity_a, variance_ratio, sigma_a, sigma_b, time_sd):
    """First-order propagation with independent pixel errors and relative clock jitter."""
    pair = np.r_[ua, ub]
    jac = np.empty((3, 4))
    for i in range(4):
        dv = np.eye(4)[i]*.05
        plus, minus = pair+dv, pair-dv
        p = _rays_midpoint(a, b, plus[:2], plus[2:], 1e-6)[0]
        m = _rays_midpoint(a, b, minus[:2], minus[2:], 1e-6)[0]
        jac[:, i] = (p-m)/.1
    variances = np.repeat([sigma_a**2*variance_ratio, sigma_b**2], 2)
    pixel_var = (jac**2) @ variances
    pix_rms = float(np.sqrt(pixel_var.sum()))
    if velocity_a is None and time_sd:
        return dict(pixel_rms_m=pix_rms)
    time_effect = (jac[:, :2]@velocity_a)*time_sd if velocity_a is not None else np.zeros(3)
    sd = np.sqrt(pixel_var + time_effect**2)
    return dict(pixel_rms_m=pix_rms, sync_rms_m=float(np.linalg.norm(time_effect)),
                reference_rms_m=float(np.linalg.norm(sd)), sd_x_m=sd[0], sd_y_m=sd[1], sd_z_m=sd[2])


def _visible(camera, times, uv):
    t, uv = np.asarray(times, dtype=float), np.asarray(uv, dtype=float)
    if t.ndim != 1 or uv.shape != (len(t), 2) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Timestamps must be finite, ordered, distinct and paired with pixel rows")
    seen = ~np.isnan(uv).all(axis=1)
    t, uv = t[seen], uv[seen]
    if not np.isfinite(uv).all() or np.any(uv < 0) or np.any(uv >= camera.image_size):
        raise ValueError("Each stream needs ordered distinct timestamps and visible native-image pixel pairs")
    return t, uv


def _rmse(rows, key):
    errors = [r[key] for r in rows if key in r]
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else None


@dataclass(frozen=True)
class Settings:
    launch_a_s: float
    stop_a_s: float
    model: str = "fixed_drag_baseline"
    k: float | None = None
    player_xy_m: list | None = None
    receiver_xy_m: list | None = None
    sigma_a_px: float = 2.5
    sigma_b_px: float = 1.
    sync_sigma_s: float = .005
    clock_source: str = "anchors"
    max_gap_a_s: float = .065
    min_angle_deg: float = 2.
    max_pair_reprojection_px: float = 6.

    def __post_init__(self):
        scales = (self.sigma_a_px, self.sigma_b_px, self.max_gap_a_s, self.max_pair_reprojection_px)
        if (not np.isfinite([*scales, self.launch_a_s, self.stop_a_s, self.sync_sigma_s, self.min_angle_deg]).all()
                or min(scales) <= 0 or self.stop_a_s <= self.launch_a_s or self.sync_sigma_s < 0
                or not 0 < self.min_angle_deg < 90):
            raise ValueError("Invalid segment, uncertainty scale or triangulation threshold")
        if self.model not in ("fixed_drag_baseline", "fixed_drag_map", "joint_drag"):
            raise ValueError("Choose fixed_drag_baseline, fixed_drag_map or joint_drag")
        if self.clock_source not in ("anchors", "published_shared"):
            raise ValueError("Choose anchors or published_shared for clock_source")
        if self.k is not None and (self.model == "joint_drag" or not np.isfinite(self.k) or self.k < 0):
            raise ValueError("Only fixed-drag models accept a finite nonnegative k")
        for xy in (self.player_xy_m, self.receiver_xy_m):
            if xy is not None and (self.model == "fixed_drag_baseline" or np.shape(xy) != (2,)
                                   or not np.isfinite(xy).all()):
                raise ValueError("Foot proxies require two finite court coordinates and a MAP model")


@dataclass(frozen=True)
class SingleViewFit:
    times: np.ndarray
    pixels: np.ndarray
    theta: np.ndarray
    k: float
    landing: float
    rmse: float


def _fit_a(camera_a, t_a, uv_a, options):
    t_a, uv_a = _visible(camera_a, t_a, uv_a)
    take_a = (t_a >= options.launch_a_s) & (t_a < options.stop_a_s)
    a_times, a_uv = t_a[take_a], uv_a[take_a]
    a_elapsed = a_times-options.launch_a_s
    k = DEFAULT_DRAG if options.k is None else options.k
    if options.model == "fixed_drag_baseline":
        theta = fit_track(camera_a, a_elapsed, a_uv, k=k, sigma_px=options.sigma_a_px).theta
    else:
        model = StrokeModel(camera_a, a_elapsed, a_uv, horizon=options.stop_a_s-options.launch_a_s,
                            player=options.player_xy_m, receiver=options.receiver_xy_m,
                            fixed_k=None if options.model == "joint_drag" else k, sigma_px=options.sigma_a_px)
        theta, k = model.unpack(fit_map(model, max_nfev=1500).x)
    t_landing, _ = landing_event(theta, k)
    a_error = np.linalg.norm(camera_a.project(flight_states(theta, a_elapsed, k)[:, :3])-a_uv, axis=1)
    return SingleViewFit(a_times, a_uv, theta, k, t_landing, float(np.sqrt(np.mean(a_error**2))))


def _score(camera_a, camera_b, fit, t_b, uv_b, anchors, options):
    if options.clock_source == "published_shared":
        if anchors is not None:
            raise ValueError("A published shared timebase takes no fitted sync anchors")
        clock = Clock(1., 0.)
        sync = dict(method="published_shared", scale_b_to_a=1., offset_a_s=0., fit_anchors=0,
                    check_anchors=0, fit_anchor_range_b_s=None, fit_rmse_s=None, check_rmse_s=None)
    else:
        if anchors is None:
            raise ValueError("Provide an anchor CSV to estimate clock offset and drift")
        clock, sync = synchronize(anchors)
    t_b, uv_b = _visible(camera_b, t_b, uv_b)
    mapped = clock.to_a(t_b)
    take_b = (mapped >= options.launch_a_s) & (mapped < options.stop_a_s)
    if not take_b.any():
        raise ValueError("No visible B detections within the synchronized stroke interval")
    elapsed = mapped[take_b]-options.launch_a_s
    predicted = flight_states(fit.theta, elapsed, fit.k)[:, :3]
    rows = []
    for tb, ta, ub, time, pred in zip(t_b[take_b], mapped[take_b], uv_b[take_b], elapsed, predicted):
        row = dict(t_b_s=tb, aligned_a_s=ta, t_since_launch_s=time, u_b_px=ub[0], v_b_px=ub[1])
        if time > fit.landing+1e-8:
            row["prediction"] = "after_predicted_landing"
        else:
            try:
                projected = camera_b.project(pred)
                row.update(prediction="projected", u_pred_px=projected[0], v_pred_px=projected[1],
                           b_error_px=float(np.linalg.norm(projected-ub)),
                           pred_in_image=int(np.all((projected >= 0) & (projected < camera_b.image_size))))
            except ValueError:
                row["prediction"] = "outside_projection_domain"
        try:
            ua, va, weight = _at_time(ta, fit.times, fit.pixels, options.max_gap_a_s)
            ref, angle, gap, ea, eb = triangulate(camera_a, camera_b, ua, ub, min_angle_deg=options.min_angle_deg,
                                                  max_pair_reprojection_px=options.max_pair_reprojection_px)
        except ValueError as err:
            row["reference"] = str(err)
        else:
            row.update(reference="triangulated", x_ref_m=ref[0], y_ref_m=ref[1], z_ref_m=ref[2],
                       angle_deg=angle, ray_gap_m=gap, tri_a_px=ea, tri_b_px=eb)
            if time <= fit.landing+1e-8:
                row["reference_error_m"] = float(np.linalg.norm(pred-ref))
            try:
                row.update(_reference_error(camera_a, camera_b, ua, ub, va, weight, options.sigma_a_px,
                                            options.sigma_b_px, options.sync_sigma_s))
            except ValueError:
                row["uncertainty_status"] = "geometry_unstable_at_pixel_perturbation"
        rows.append(row)
    extrapolated = None
    if sync["fit_anchor_range_b_s"] is not None:
        lo, hi = sync["fit_anchor_range_b_s"]
        extrapolated = int(np.sum((t_b[take_b] < lo) | (t_b[take_b] > hi)))
    angles = [r["angle_deg"] for r in rows if r["reference"] == "triangulated"]
    baseline = float(np.linalg.norm(camera_a.center - camera_b.center))
    report = dict(clock=sync, launch_a_s=options.launch_a_s, stop_a_s=options.stop_a_s, model=options.model,
                  a_fit_k_m_inv=float(fit.k),
                  player_xy_m=None if options.player_xy_m is None else np.asarray(options.player_xy_m).tolist(),
                  receiver_xy_m=None if options.receiver_xy_m is None else np.asarray(options.receiver_xy_m).tolist(),
                  a_training_points=len(fit.times), a_fit_theta=fit.theta.tolist(), predicted_landing_s=fit.landing,
                  a_fit_rmse_px=fit.rmse, b_held_out_points=len(rows),
                  b_sync_extrapolated=extrapolated,
                  prediction_counts=dict(Counter(r["prediction"] for r in rows)),
                  b_rmse_px=_rmse(rows,"b_error_px"), camera_baseline_m=baseline,
                  reference_counts=dict(Counter(r["reference"] for r in rows)),
                  accepted_angle_range_deg=[min(angles), max(angles)] if angles else None,
                  a_fit_to_stereo_compared=sum("reference_error_m" in r for r in rows),
                  a_fit_to_stereo_rmse_m=_rmse(rows,"reference_error_m"),
                  reference_pixel_rms_m=_rmse(rows,"pixel_rms_m"), reference_clock_rms_m=_rmse(rows,"sync_rms_m"),
                  reference_combined_rms_m=_rmse(rows,"reference_rms_m"),
                  reference_uncertainty_count=sum("reference_rms_m" in r for r in rows),
                  sigma_a_px=options.sigma_a_px, sigma_b_px=options.sigma_b_px, sync_sigma_s=options.sync_sigma_s,
                  max_gap_a_s=options.max_gap_a_s, min_angle_deg=options.min_angle_deg,
                  max_pair_reprojection_px=options.max_pair_reprojection_px)
    return report, rows


def evaluate(camera_a, camera_b, t_a, uv_a, t_b, uv_b, anchors=None, **settings):
    """Fit exclusively to A, then evaluate B at its own capture times."""
    opts = Settings(**settings)
    fit = _fit_a(camera_a, t_a, uv_a, opts)
    return _score(camera_a, camera_b, fit, t_b, uv_b, anchors, opts)


def run(config_path, out):
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    paths = {key: config_path.parent / config.pop(key) for key in
             ("camera_a", "camera_b", "observations_a", "observations_b")}
    anchor_path = config.pop("anchors", None)
    if paths["observations_a"].resolve() == paths["observations_b"].resolve():
        raise ValueError("Independent views require separate observation files")

    def camera(key):
        data = json.loads(paths[key].read_text(encoding="utf-8-sig"))
        if not {"image_size", "focal_px", "R", "t", "k1"}.issubset(data):
            raise ValueError(f"{key} must be a stage-three camera.json containing image_size, focal_px, R, t, k1")
        return Camera.from_dict(data)

    opts = Settings(**config)
    a = camera("camera_a")
    fit = _fit_a(a, *read_observations(paths["observations_a"]), opts)
    # The A fit is complete before opening camera B's calibration or detections.
    b = camera("camera_b")
    tb, ub = read_observations(paths["observations_b"])
    anchors = None
    if anchor_path is not None:
        with (config_path.parent/anchor_path).open(encoding="utf-8-sig", newline="") as f:
            anchors = list(csv.DictReader(f))
    report, rows = _score(a, b, fit, tb, ub, anchors, opts)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/"stereo.json", report)
    with (out/"stereo-points.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(f, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = run(args.config, args.out)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, csv.Error) as exc:
        parser.exit(2, f"Invalid stereo input: {exc}\n")
    print(f"Independent B-view RMSE {result['b_rmse_px']} px -> {args.out}")
