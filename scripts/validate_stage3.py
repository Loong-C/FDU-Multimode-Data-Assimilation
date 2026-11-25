"""Independently validate synthetic pinhole geometry, calibration and forward flight."""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from badminton.camera import Camera, dlt_diagnostics, fit_camera, look_at_camera, pixel_rays, project_points
from badminton.forward import project_flight


SIZE = (1280, 720)
CENTER = np.array([-13.0, -2.1, 6.7])
BASE = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
ROT = BASE @ Rotation.from_euler("zyx", [-9, -19, 1], degrees=True).as_matrix()
TRANS = -ROT @ CENTER
K = np.array([[970.0, 0, 639.5], [0, 970.0, 359.5], [0, 0, 1]])
TRAIN = np.array([(x, y, 0.0) for x in (-6.7, -1.98, 1.98, 6.7) for y in (-3.05, 3.05)]
                 + [(0, -3.05, 1.55), (0, 3.05, 1.55), (-2, 1.5, 3.4), (2.8, -1.4, 4.2)])
HOLDOUT = np.array([[-4.8, -2.3, 2.7], [-2.7, 1.7, 0], [1.3, -1.4, 4.1],
                    [5.0, 0.8, 1.7], [-3.1, 1.4, 5.1], [3.8, -2.2, 0]])


def reference_pixels(points, k1=0.0):
    pixels, _ = cv2.projectPoints(points, cv2.Rodrigues(ROT)[0], TRANS, K,
                                  np.array([k1, 0, 0, 0, 0], dtype=float))
    return pixels.reshape(-1, 2)


def max_distance(a, b):
    return float(np.max(np.linalg.norm(a - b, axis=-1)))


def compare_rays(center, directions, world):
    delta = world - center
    along_ray = np.sum(delta * directions, axis=1, keepdims=True)
    if np.any(along_ray <= 0):
        raise AssertionError("Ray points away from the calibration scene")
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1, rtol=0, atol=1e-12)
    return max_distance(world, center + along_ray * directions)


def general_projection():
    # Exercise unequal focal lengths, skew, off-centre principal point, sign and scale.
    intrinsics = np.array([[1043., 11., 570.], [0., 782., 331.], [0., 0., 1.]])
    p = intrinsics @ np.column_stack((ROT, TRANS))
    q = (HOLDOUT - CENTER) @ ROT.T
    u = 570 + (1043*q[:, 0] + 11*q[:, 1])/q[:, 2]
    v = 331 + 782*q[:, 1]/q[:, 2]
    expected = np.column_stack((u, v))
    errors, distances = [], []
    for scale in (1.0, -0.003, 1e9):
        errors.append(max_distance(project_points(scale*p, HOLDOUT), expected))
        center, directions = pixel_rays(scale*p, expected)
        np.testing.assert_allclose(center, CENTER, rtol=0, atol=1e-10)
        distances.append(compare_rays(center, directions, HOLDOUT))
    return {"max_projection_error_px": max(errors), "max_ray_distance_m": max(distances)}


def fitted_cases():
    reports = []
    for k1 in (0.0, -0.12, 0.15):
        camera = fit_camera(TRAIN, reference_pixels(TRAIN, k1), SIZE, fit_k1=(k1 != 0))
        test_uv = reference_pixels(HOLDOUT, k1)
        measured = camera.project(HOLDOUT)
        restored = Camera.from_dict(json.loads(json.dumps(camera.to_dict())))
        np.testing.assert_allclose(restored.project(HOLDOUT), measured)
        center, directions = restored.rays(test_uv)
        reports.append(dict(k1_true=k1, k1_fitted=camera.k1,
                            focal_error_px=abs(camera.focal_px-970), center_error_m=float(np.linalg.norm(center-CENTER)),
                            heldout_max_error_px=max_distance(measured, test_uv),
                            heldout_max_ray_distance_m=compare_rays(center, directions, HOLDOUT)))
    return reports


def planar_ambiguity():
    planar = TRAIN[TRAIN[:, 2] == 0]
    uv = reference_pixels(planar)
    diagnostics = dlt_diagnostics(planar, uv)
    rejected = False
    try:
        fit_camera(planar, uv, SIZE)
    except ValueError:
        rejected = True
    # Adding any column perturbation multiplied by world z leaves every floor observation unchanged.
    p = K @ np.column_stack((ROT, TRANS))
    other = p.copy()
    other[:, 2] += [140, -80, 0.1]
    return {"planar_fit_rejected": rejected, "planar_design_rank": diagnostics["design_rank"],
            "noncoplanar_design_rank": dlt_diagnostics(TRAIN, reference_pixels(TRAIN))["design_rank"],
            "floor_max_difference_px": max_distance(project_points(other, planar), project_points(p, planar)),
            "off_floor_max_difference_px": max_distance(project_points(other, HOLDOUT), project_points(p, HOLDOUT))}


def forward_visibility():
    cam = look_at_camera([0, 0, 1], [1, 0, 1], 800, SIZE)
    shot = project_flight(cam, [-1, 0, 1], [4, .8, 1.4], k=0, dt=.05)
    fly = shot.flight
    front = fly.position[:, 0] > 1e-9
    np.testing.assert_array_equal(shot.in_front, front)
    np.testing.assert_array_equal(shot.projection_valid, front)
    if not np.isnan(shot.uv[~front]).all():
        raise AssertionError("Behind-camera samples need missing pixels")
    x, y, z = fly.position[front].T
    uv = np.column_stack((639.5-800*y/x, 359.5-800*(z-1)/x))
    np.testing.assert_allclose(uv, shot.uv[front], atol=1e-9, rtol=0)
    on_sensor = (uv[:, 0] >= 0) & (uv[:, 0] < SIZE[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < SIZE[1])
    np.testing.assert_array_equal(shot.in_image[front], on_sensor)
    return {"behind_or_on_plane": int((~front).sum()), "in_front_out_of_image": int((~on_sensor).sum()),
            "in_image": int(on_sensor.sum())}


def check_export(directory):
    # Compare the physics against the archived stage-two results, then use OpenCV on those exact samples.
    with (ROOT / "results/stage2/trajectories.csv").open(encoding="utf-8", newline="") as src:
        reference = [r for r in csv.DictReader(src) if "_k" not in r["run_id"]]
    camera = json.loads((directory / "virtual-camera.json").read_text(encoding="utf-8"))
    with (directory / "virtual-trajectories.csv").open(encoding="utf-8", newline="") as src:
        rows = list(csv.DictReader(src))
    ids = [r["shot"] for r in rows]
    if ids != [r["run_id"] for r in reference]:
        raise AssertionError("Stage-two and stage-three sample IDs disagree")
    np.testing.assert_allclose([float(r["t_s"]) for r in rows], [float(r["t_s"]) for r in reference],
                               rtol=1e-12, atol=1e-12)
    world = np.array([[float(r[k]) for k in ("x_m", "y_m", "z_m")] for r in reference])
    xyz = np.array([[float(r[k]) for k in ("x_m", "y_m", "z_m")] for r in rows])
    np.testing.assert_allclose(world, xyz, atol=1e-10, rtol=1e-12)
    expected, _ = cv2.projectPoints(world, cv2.Rodrigues(np.array(camera["R"]))[0],
                                    np.array(camera["t"]), np.array(camera["K"]),
                                    np.array([camera["k1"], 0, 0, 0, 0]))
    uv = np.array([[float(r["u_px"]), float(r["v_px"])] for r in rows])
    return {"samples": len(rows), "shot_count": len(set(ids)),
            "in_image": sum(int(r["in_image"]) for r in rows),
            "max_opencv_projection_error_px": max_distance(uv, expected.reshape(-1, 2))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "results/stage3", help="default five-shot forward export")
    parser.add_argument("--out", type=Path, default=ROOT / "results/stage3/validation.json")
    args = parser.parse_args()
    general = general_projection()
    fits = fitted_cases()
    planar = planar_ambiguity()
    visibility = forward_visibility()
    forward = check_export(args.data)
    checks = {
        "general_projective_camera": max(general.values()) < 1e-9,
        "noiseless_heldout_fit": all(case["heldout_max_error_px"] < 1e-6 and case["focal_error_px"] < 1e-5
                                     and case["center_error_m"] < 1e-6 for case in fits),
        "radial_inverse": all(abs(case["k1_true"]-case["k1_fitted"]) < 1e-7
                              and case["heldout_max_ray_distance_m"] < 1e-6 for case in fits),
        "planar_degeneracy": planar["planar_fit_rejected"] and planar["planar_design_rank"] < 11
                             and planar["noncoplanar_design_rank"] == 11
                             and planar["floor_max_difference_px"] < 1e-9 and planar["off_floor_max_difference_px"] > 1,
        "visibility": min(visibility.values()) > 0,
        "forward_export": forward["max_opencv_projection_error_px"] < 1e-8 and forward["samples"] == forward["in_image"],
    }
    report = {"passed": all(checks.values()), "checks": checks, "general_projection": general,
              "synthetic_fit": {"training_count": len(TRAIN), "heldout_count": len(HOLDOUT), "cases": fits},
              "coplanar_ambiguity": planar, "visibility": visibility, "exported_virtual_camera": forward}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(f"Failed checks: {[name for name, ok in checks.items() if not ok]}")
    print(f"Verified synthetic projection, rays, calibration, visibility and {forward['samples']} samples -> {args.out}")


if __name__ == "__main__":
    main()
