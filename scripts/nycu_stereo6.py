"""Evaluate a frozen monocular fit on three preselected NYCU synchronized camera trials."""

import argparse
import configparser
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from fetch_nycu6 import RAW, ROOT, TRIALS, check_files, PINNED

sys.path.insert(0, str(ROOT))
from badminton.camera import Camera
from badminton.posterior import flight_states
from badminton.reconstruct import write_json, write_rows
from badminton.stereo import Settings, _fit_a, _score


class SourceCamera:
    """Original OpenCV camera. Native scores include radial and tangential distortion."""

    def __init__(self, cfg, size):
        config = configparser.ConfigParser()
        config.read(cfg, encoding="utf-8")
        source = {k: np.asarray(json.loads(v)) for k, v in config["Other"].items()}
        self.K = source["ks"]
        self.dist = source["dist"].ravel()
        ext = source["extrinsic_mat"]
        np.testing.assert_allclose(source["newcameramtx"] @ ext, source["projection_mat"], atol=1e-6)
        # The published rotation has float32 rounding. Only orthonormalize within that tolerance.
        u, singular, vh = np.linalg.svd(ext[:, :3])
        np.testing.assert_allclose(singular, 1., atol=1e-6, rtol=0)
        self.R = u @ vh
        self.t = ext[:, 3].copy()
        self.rvec = cv2.Rodrigues(self.R)[0]
        self.image_size = size
        self.ideal = Camera(size, np.sqrt(self.K[0, 0]*self.K[1, 1]), self.R, self.t)
        self.center = self.ideal.center

    def project(self, xyz):
        world = np.asarray(xyz, dtype=float)
        flat = np.ascontiguousarray(world.reshape(-1, 3))
        if np.any(flat @ self.R[2] + self.t[2] <= 0.):
            raise ValueError("Point behind camera")
        uv = cv2.projectPoints(flat, self.rvec, self.t, self.K, self.dist)[0]
        return uv.reshape((*world.shape[:-1], 2))

    def rectify(self, uv):
        uv = np.asarray(uv, dtype=float)
        points = np.ascontiguousarray(uv.reshape(-1, 1, 2))
        ideal = cv2.undistortPoints(points, self.K, self.dist, P=self.ideal.K,
                                    criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 35, 1e-10))
        return ideal.reshape(uv.shape)

    def rays(self, uv):
        return self.ideal.rays(self.rectify(uv))


def stream(directory, side):
    prefix = f"CameraReader{side}"
    cap = cv2.VideoCapture(str(directory/f"{prefix}.avi"))
    try:
        size = tuple(int(cap.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    if size != (1440, 1080) or fps != 120:
        raise ValueError(f"Unexpected native video size or rate in {directory}/{prefix}")
    with (directory/f"{prefix}_ball.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    frames = np.asarray([int(r["Frame"]) for r in rows])
    times = np.asarray([float(r["Timestamp"]) for r in rows])
    if not np.array_equal(frames, np.arange(n)) or not np.allclose(times, frames/fps, rtol=0, atol=1e-12):
        raise ValueError("Published native frame or timestamp mismatch")
    visible = np.array([int(r["Visibility"]) == 1 for r in rows])
    uv = np.array([[float(row[k]) for k in ("X", "Y")] for row in rows])
    return SourceCamera(directory/f"{prefix}.cfg", size), frames, times, visible, uv


def rmse(vectors):
    return float(np.sqrt(np.mean(np.sum(np.asarray(vectors)**2, axis=1))))


def compare(raw, trial, points_dir):
    path = raw/"vicon_dataset"/trial
    left, frames, times, seen_a, ua = stream(path, "L")
    # Preserve the entire author's clip time span, including its final frame in the half-open API.
    opts = Settings(launch_a_s=0., stop_a_s=np.nextafter(times[-1], np.inf), model="fixed_drag_map",
                    clock_source="published_shared", sigma_a_px=2.5, sigma_b_px=2.5,
                    sync_sigma_s=1/120/np.sqrt(12), max_gap_a_s=1.5/120)
    fit = _fit_a(left.ideal, times[seen_a], left.rectify(ua[seen_a]), opts)
    predicted = flight_states(fit.theta, times, fit.k)[:, :3]
    uap = left.project(predicted)

    right, frames_b, times_b, seen_b, ub = stream(path, "R")
    if not np.array_equal(frames, frames_b) or not np.array_equal(times, times_b):
        raise ValueError(f"The published paired frame identities differ in {trial}")
    report, scored = _score(left.ideal, right, fit, times_b[seen_b], ub[seen_b], None, opts)
    rows_b = {round(row["t_b_s"]*120): row for row in scored}
    rows = []
    for i in frames:
        item = rows_b.get(i, {})
        rows.append(dict(frame_index=int(i), t_s=times[i],
                         a_u_px=ua[i, 0] if seen_a[i] else "", a_v_px=ua[i, 1] if seen_a[i] else "",
                         a_pred_u_px=uap[i, 0], a_pred_v_px=uap[i, 1],
                         b_u_px=ub[i, 0] if seen_b[i] else "", b_v_px=ub[i, 1] if seen_b[i] else "",
                         b_pred_u_px=item.get("u_pred_px", ""), b_pred_v_px=item.get("v_pred_px", ""),
                         prediction=item.get("prediction", "missing_label"), b_in_image=item.get("pred_in_image", ""),
                         reference=item.get("reference", ""), reference_error_m=item.get("reference_error_m", ""),
                         x_ref_m=item.get("x_ref_m", ""), y_ref_m=item.get("y_ref_m", ""),
                         z_ref_m=item.get("z_ref_m", ""), ray_angle_deg=item.get("angle_deg", ""),
                         ray_gap_m=item.get("ray_gap_m", ""), reference_rms_m=item.get("reference_rms_m", "")))
    write_rows(points_dir/f"{trial}.csv", rows)
    ub_errors = np.array([r["b_error_px"] for r in scored if "b_error_px" in r])
    angles = np.array([r["angle_deg"] for r in scored if "angle_deg" in r])
    report.update(trial=trial, n_frames=len(frames), source_fingerprint_sha256=PINNED[trial],
                  a_native_rmse_px=rmse((uap-ua)[seen_a]), b_scored_points=len(ub_errors),
                  b_scored_out_of_image=sum(r.get("pred_in_image") == 0 for r in scored),
                  b_median_px=float(np.median(ub_errors)) if len(ub_errors) else None,
                  b_max_px=float(np.max(ub_errors)) if len(ub_errors) else None,
                  median_ray_angle_deg=float(np.median(angles)) if len(angles) else None,
                  camera_baseline_m=float(np.linalg.norm(left.ideal.center-right.ideal.center)))
    return report


def pooled(reports, metric, count):
    values = [(r[metric], r[count]) for r in reports if r.get(metric) is not None and r[count]]
    return float(np.sqrt(np.average([value**2 for value, _ in values], weights=[n for _, n in values]))) if values else None


def run(raw, out, points_dir):
    check_files(raw)
    points_dir.mkdir(parents=True, exist_ok=True)
    successes, failures = [], {}
    for trial in TRIALS:
        try:
            successes.append(compare(raw, trial, points_dir))
        except (ValueError, cv2.error, OSError, AssertionError) as exc:
            failures[trial] = str(exc)
    report = dict(source="NYCU Shuttlecock Trajectories Dataset, vicon_dataset (two synchronized 120 Hz camera videos per flight)",
                  source_url="https://drive.google.com/drive/folders/1KaPYvxVkpDieZUxfACg5bTdc8senjDDC",
                  protocol="First three numbered trials specified before reconstruction. Each is a separate recorded flight. Full source clips. Fixed-drag MAP selected on the Fuzhou development rallies. No NYCU tuning, player constraints or right-view alignment. The left and right source labels are independently collected native pixels.",
                  timebase="Authors' synchronized frame counters and equal timestamps. No clock fit; the timing SD of 1/sqrt(12) frames is a scenario for the triangulation error budget, not a measured synchronization accuracy.",
                  geometry="Published metric world-to-camera extrinsic_mat, ks and five OpenCV distortion coefficients. Left-fit pixel units use a centred square-pixel virtual image with focal length sqrt(fx*fy); left native and all right-view scores use original distorted images.",
                  reference="Recomputed two-camera ray-midpoint reference shares left labels with the fit. The 6px per-view consistency gate uses rectified left and native right pixel coordinates. Reference uncertainty propagates assumed 2.5px localization and clock jitter, conditional on the published calibration. Vicon has a separate time origin, so neither Vicon nor Model3D is used to fit or score.",
                  denominator="Every labelled right frame in the left-defined clip is counted. All projectable pre-impact predictions, including off-image predictions, enter B RMSE. Failed projections and timestamps after predicted landing are counted separately.",
                  selected_trials=list(TRIALS), failures=failures, trials=successes,
                  pooled=dict(completed_trials=len(successes),
                              frames=sum(r["n_frames"] for r in successes),
                              b_labelled=sum(r["b_held_out_points"] for r in successes),
                              b_scored=sum(r["b_scored_points"] for r in successes),
                              b_rmse_px=pooled(successes, "b_rmse_px", "b_scored_points"),
                              compared_to_stereo=sum(r["a_fit_to_stereo_compared"] for r in successes),
                              a_fit_to_stereo_rmse_m=pooled(successes,"a_fit_to_stereo_rmse_m","a_fit_to_stereo_compared"),
                              a_native_rmse_px=pooled(successes,"a_native_rmse_px","a_training_points")))
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--out", type=Path, default=ROOT/"results/stage6/nycu-stereo.json")
    parser.add_argument("--points-dir", type=Path, default=ROOT/"artifacts/stage6/nycu")
    args = parser.parse_args()
    result = run(args.raw, args.out, args.points_dir)
    print(result["pooled"])
    if result["failures"]:
        raise SystemExit(result["failures"])
