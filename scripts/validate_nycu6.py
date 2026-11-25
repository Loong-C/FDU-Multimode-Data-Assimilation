"""Independently audit NYCU native frame identities, held-out scores and stereo reference."""

import argparse
from collections import Counter
import configparser
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[1]
TRIALS = ("001", "002", "003")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def rms(vectors):
    return float(np.sqrt(np.mean(np.sum(np.asarray(vectors)**2, axis=1))))


@dataclass
class NativeView:
    K: np.ndarray
    distortion: np.ndarray
    extrinsic: np.ndarray
    rvec: np.ndarray
    centre: np.ndarray
    seconds: np.ndarray
    seen: np.ndarray
    pixels: np.ndarray
    normalized: np.ndarray
    size: tuple
    max_pts_error_s: float

    def project(self, xyz):
        return cv2.projectPoints(np.ascontiguousarray(xyz, dtype=float), self.rvec, self.extrinsic[:, 3],
                                 self.K, self.distortion)[0].reshape(-1, 2)


def load_view(folder, side):
    name = f"CameraReader{side}"
    cfg = configparser.ConfigParser()
    cfg.read(folder/f"{name}.cfg", encoding="utf-8")
    section = cfg["Other"]
    matrix, distortion, rectified_K, projection = [
        np.array(json.loads(section[k]), dtype=float)
        for k in ("ks", "dist", "newcameramtx", "projection_mat")]
    # Recover the extrinsic from the rectified projection, independently of the adapter.
    extrinsic = np.linalg.solve(rectified_K, projection)
    np.testing.assert_allclose(extrinsic, json.loads(section["extrinsic_mat"]), atol=2e-6, rtol=0)
    rvec = cv2.Rodrigues(extrinsic[:, :3])[0]
    centre = np.linalg.solve(extrinsic[:, :3], -extrinsic[:, 3])
    with (folder/f"{name}_ball.csv").open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    frame = np.array([int(r["Frame"]) for r in records])
    seconds = np.array([float(r["Timestamp"]) for r in records])
    seen = np.array([int(r["Visibility"]) == 1 for r in records])
    pixels = np.array([[float(r[d]) for d in ("X","Y")] for r in records])
    normalized = cv2.undistortPoints(pixels.reshape(-1, 1, 2), matrix, distortion)[:, 0]
    # Refine the default inverse against the native forward projection to subpixel tolerance.
    origin = np.zeros(3)
    for _ in range(60):
        directions = np.c_[normalized, np.ones(len(pixels))]
        reprojection = cv2.projectPoints(directions, origin, origin, matrix, distortion)[0][:, 0]
        delta = pixels-reprojection
        if np.max(np.abs(delta)) < 1e-9:
            break
        normalized += delta / np.diag(matrix)[:2]
    else:
        raise ValueError(f"Native lens inversion failed in {folder}/{name}")
    capture = cv2.VideoCapture(str(folder/f"{name}.avi"))
    pts = []
    try:
        check(capture.isOpened(), f"Cannot decode {folder}/{name}.avi")
        rate = capture.get(cv2.CAP_PROP_FPS)
        check(rate == 120., f"Unexpected video rate in {folder}/{name}")
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        check(size == (1440, 1080), f"Unexpected image size in {folder}/{name}")
        while True:
            ok, image = capture.read()
            if not ok:
                break
            check(image.shape[:2] == size[::-1], f"Image size changed in {folder}/{name}")
            pts.append(capture.get(cv2.CAP_PROP_POS_MSEC)/1000.)
            check(capture.get(cv2.CAP_PROP_POS_FRAMES) == len(pts), f"Unordered decoder frame in {folder}/{name}")
    finally:
        capture.release()
    np.testing.assert_array_equal(frame, np.arange(len(pts)))
    np.testing.assert_allclose(seconds, frame/rate, atol=1e-12, rtol=0)
    np.testing.assert_allclose(seconds, pts, atol=1e-7, rtol=0)
    return NativeView(matrix, distortion, extrinsic, rvec, centre, seconds, seen, pixels, normalized, size,
                      float(np.max(np.abs(seconds - pts))))


def integrate(theta, k, seconds):
    def rhs(_, state):
        v = state[3:]
        return np.r_[v, -k*np.linalg.norm(v)*v - [0., 0., 9.81]]
    sol = solve_ivp(rhs, (0., seconds[-1]), theta, t_eval=seconds, method="DOP853", rtol=1e-12, atol=1e-13)
    check(sol.success, sol.message)
    return sol.y[:3].T


def stereo_references(a, b):
    # Compare the inhomogeneous closest-ray solution with homogeneous OpenCV DLT.
    homogeneous = cv2.triangulatePoints(a.extrinsic, b.extrinsic, a.normalized.T, b.normalized.T)
    dlt = (homogeneous[:3]/homogeneous[3]).T
    identities = np.eye(3)
    lines = []
    for view in (a, b):
        rays = np.linalg.solve(view.extrinsic[:, :3], np.c_[view.normalized, np.ones(len(dlt))].T).T
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        lines.append(rays)
    angle = np.rad2deg(np.arccos(np.clip(abs(np.sum(lines[0]*lines[1], axis=1)), 0., 1.)))
    normal_a, normal_b = [identities - np.einsum("ni,nj->nij", rays, rays) for rays in lines]
    midpoint = np.linalg.solve(normal_a+normal_b, (normal_a @ a.centre + normal_b @ b.centre)[..., None])[..., 0]
    both_forward = ((np.sum((midpoint-a.centre)*lines[0], axis=1)>0) &
                    (np.sum((midpoint-b.centre)*lines[1], axis=1)>0))
    return midpoint, dlt, angle, both_forward


def audit_trial(path, metadata, points_dir):
    a, b = (load_view(path, side) for side in ("L", "R"))
    # This release contains fully paired labels in these three preselected source clips.
    np.testing.assert_array_equal(a.seconds, b.seconds)
    check(a.seen.all() and b.seen.all(), f"Source visibility changed in {path.name}")
    n = len(a.seconds)
    frames = np.arange(n)
    with (points_dir/f"{path.name}.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    np.testing.assert_array_equal([int(r["frame_index"]) for r in rows], frames)
    np.testing.assert_allclose([float(r["t_s"]) for r in rows], a.seconds, atol=1e-12, rtol=0)
    for view, name in ((a,"a"), (b,"b")):
        np.testing.assert_array_equal([[float(r[f"{name}_{d}_px"]) for d in ("u", "v")] for r in rows], view.pixels)
    check(metadata["clock"]["method"] == "published_shared" and metadata["clock"]["fit_anchors"] == 0,
          "NYCU uses published synchronized frame times")
    check(metadata["launch_a_s"] == 0. and a.seconds[-1] < metadata["stop_a_s"] <= a.seconds[-1]+1/120,
          "The scoring interval must cover the entire original clip")
    xyz = integrate(metadata["a_fit_theta"], metadata["a_fit_k_m_inv"], a.seconds)
    pixels = b.project(xyz)
    depth_b = xyz @ b.extrinsic[2, :3] + b.extrinsic[2, 3]
    expected_status = np.where(xyz[:, 2] < -1e-8, "after_predicted_landing",
                               np.where(depth_b > 0., "projected", "outside_projection_domain"))
    np.testing.assert_array_equal([r["prediction"] for r in rows], expected_status)
    scored = expected_status == "projected"
    np.testing.assert_equal(dict(Counter(expected_status)), metadata["prediction_counts"])
    expected_count = int(scored.sum())
    check(metadata["b_scored_points"] == expected_count and metadata["b_held_out_points"] == n, "Wrong B denominator")
    in_image = np.all((pixels >= 0.) & (pixels < b.size), axis=1)
    check(metadata["b_scored_out_of_image"] == int((scored & ~in_image).sum()), "Off-image B prediction lost")
    shipped = np.array([[float(r["b_pred_u_px"]), float(r["b_pred_v_px"])] for i,r in enumerate(rows) if scored[i]])
    np.testing.assert_allclose(pixels[scored], shipped, atol=.002, rtol=0)
    b_rmse = rms(pixels[scored]-b.pixels[scored])
    np.testing.assert_allclose(b_rmse, metadata["b_rmse_px"], atol=.001, rtol=0)
    np.testing.assert_allclose(rms(a.project(xyz)-a.pixels), metadata["a_native_rmse_px"], atol=.001, rtol=0)

    midpoint, dlt, angles, forward = stereo_references(a, b)
    camera_xyz = midpoint @ a.extrinsic[:, :3].T + a.extrinsic[:, 3]
    focal = np.sqrt(a.K[0, 0]*a.K[1, 1])
    left_ideal_error = focal*np.linalg.norm(camera_xyz[:, :2]/camera_xyz[:, 2:]-a.normalized, axis=1)
    left_native_error = np.linalg.norm(a.project(midpoint)-a.pixels, axis=1)
    right_native_error = np.linalg.norm(b.project(midpoint)-b.pixels, axis=1)
    geometric = (angles >= metadata["min_angle_deg"]) & forward
    tol = metadata["max_pair_reprojection_px"]
    passed = geometric & (np.maximum(left_ideal_error, right_native_error) <= tol)
    native_gate = geometric & (np.maximum(left_native_error, right_native_error) <= tol)
    np.testing.assert_array_equal(passed, [r["reference"] == "triangulated" for r in rows])
    shipped_ref = np.array([[float(r[d+"_ref_m"]) for d in ("x","y","z")] for i,r in enumerate(rows) if passed[i]])
    np.testing.assert_allclose(midpoint[passed], shipped_ref, atol=3e-5, rtol=0)
    compared = passed & (expected_status != "after_predicted_landing")
    check(metadata["a_fit_to_stereo_compared"] == int(compared.sum()), "Wrong 3D comparison denominator")
    fit_to_midpoint = rms(xyz[compared]-midpoint[compared])
    np.testing.assert_allclose(fit_to_midpoint, metadata["a_fit_to_stereo_rmse_m"], atol=5e-5, rtol=0)
    result = dict(trial=path.name, source_frame_ids=[0, n-1], paired_source_frames=n,
                  decoded_frames_left=n, decoded_frames_right=n, right_visible=n, scored_right=expected_count,
                  right_unscored=n-expected_count, right_native_rmse_px=b_rmse,
                  max_native_prediction_difference_px=float(np.max(np.linalg.norm(pixels[scored]-shipped,axis=1))),
                  max_csv_to_decoded_pts_error_s=max(a.max_pts_error_s, b.max_pts_error_s),
                  stereo_accepted=int(passed.sum()), stereo_excluded=int(n-passed.sum()),
                  native_both_gate_decisions_different=int((native_gate != passed).sum()),
                  compared_to_reference=int(compared.sum()), fit_to_midpoint_rmse_m=fit_to_midpoint,
                  dlt_vs_midpoint_rmse_m=rms(dlt[passed]-midpoint[passed]),
                  dlt_vs_midpoint_max_m=float(np.linalg.norm(dlt[passed]-midpoint[passed],axis=1).max()),
                  fit_to_dlt_rmse_m=rms(xyz[compared]-dlt[compared]))
    return result


def audit(raw, scores, points_dir):
    report = json.loads(scores.read_text(encoding="utf-8"))
    check(report["selected_trials"] == list(TRIALS) and not report["failures"], "All prespecified trials are needed")
    trials = {d["trial"]: d for d in report["trials"]}
    check(set(trials) == set(TRIALS) and len(report["trials"]) == len(TRIALS), "Missing or duplicated scored trial")
    checks = [audit_trial(raw/"vicon_dataset"/trial, trials[trial], points_dir) for trial in TRIALS]
    frames = sum(c["paired_source_frames"] for c in checks)
    scored = sum(c["scored_right"] for c in checks)
    n_reference = sum(c["compared_to_reference"] for c in checks)
    b_error = float(np.sqrt(sum(c["right_native_rmse_px"]**2*c["scored_right"] for c in checks)/scored))
    spatial = float(np.sqrt(sum(c["fit_to_midpoint_rmse_m"]**2*c["compared_to_reference"] for c in checks)/n_reference))
    pooled = report["pooled"]
    check(frames == pooled["frames"] == pooled["b_labelled"] and scored == pooled["b_scored"] and
          n_reference == pooled["compared_to_stereo"], "Pooled denominator mismatch")
    np.testing.assert_allclose(b_error, pooled["b_rmse_px"], atol=.001, rtol=0)
    np.testing.assert_allclose(spatial, pooled["a_fit_to_stereo_rmse_m"], atol=5e-5, rtol=0)
    return dict(passed=True, evidence="Independent recomputation on original paired labels and decoded videos. "
                "Only the frozen A launch and drag come from the estimator. Integration uses DOP853. "
                "Native OpenCV projections are reconstructed from ks, distortion, newcameramtx and projection_mat. "
                "Stereo references are independently recomputed using ray normal equations and OpenCV homogeneous DLT.",
                clock="Decoded frame timestamps confirm the published common 120 Hz index. Physical shutter "
                      "synchronization uncertainty is not observable from these nominal timestamps.",
                paired_source_frames=frames, scored_right=scored, native_b_pooled_rmse_px=b_error,
                compared_to_stereo=n_reference, pooled_fit_to_midpoint_rmse_m=spatial, trials=checks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT/"data/raw/nycu-stereo6")
    parser.add_argument("--scores", type=Path, default=ROOT/"results/stage6/nycu-stereo.json")
    parser.add_argument("--points-dir", type=Path, default=ROOT/"artifacts/stage6/nycu")
    parser.add_argument("--out", type=Path, default=ROOT/"results/stage6/nycu-validation.json")
    args = parser.parse_args()
    result = audit(args.raw, args.scores, args.points_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Independent NYCU audit passed on {result['paired_source_frames']} pairs -> {args.out}")
