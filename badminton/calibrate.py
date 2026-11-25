"""Calibrate one annotated camera segment and evaluate independent image points."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .camera import dlt_diagnostics, fit_camera
from .court import HALF_LENGTH, SINGLES_HALF_WIDTH

ROOT = Path(__file__).resolve().parents[1]


def errors_by_point(camera, landmarks, model, role, frame):
    xyz = np.array([p["world_m"] for p in landmarks], dtype=float)
    reference = np.array([p["pixel"] for p in landmarks], dtype=float)
    predicted = camera.project(xyz)
    delta = predicted - reference
    return [
        dict(model=model, group=role, frame=frame, point_id=p["id"], z_m=z,
             u_observed=uv[0], v_observed=uv[1], u_projected=pr[0], v_projected=pr[1],
             du_px=d[0], dv_px=d[1], error_px=float(np.linalg.norm(d)))
        for p, z, uv, pr, d in zip(landmarks, xyz[:, 2], reference, predicted, delta)
    ]


def metrics(rows):
    errors = np.array([row["error_px"] for row in rows])
    return {"n": len(rows), "rmse_px": float(np.sqrt(np.mean(errors**2))),
            "median_px": float(np.median(errors)), "max_px": float(np.max(errors))}


def perturbation_check(world, pixels, image_size, baseline):
    """Effect of independent 1 px annotation noise on pose and off-plane projections."""
    seed, repeats = 301, 50
    rng = np.random.default_rng(seed)
    heights = [0.0, 1.55, 3.0, 6.0]
    probes = np.array([(x, y, z) for z in heights for x in (-HALF_LENGTH, 0, HALF_LENGTH)
                       for y in (-SINGLES_HALF_WIDTH, 0, SINGLES_HALF_WIDTH)])
    reference = baseline.project(probes)
    focal, centers, displacements = [], [], []
    for _ in range(repeats):
        cam = fit_camera(world, pixels + rng.normal(0, 1, size=pixels.shape), image_size)
        focal.append(cam.K[0, 0])
        centers.append(np.linalg.norm(cam.center - baseline.center))
        displacements.append(np.linalg.norm(cam.project(probes) - reference, axis=1))
    shifts = np.array(displacements)
    return dict(seed=seed, refits=repeats, noise_std_per_coordinate_px=1.0,
                focal_px_percentiles_5_50_95=np.percentile(focal, [5, 50, 95]).tolist(),
                center_displacement_p95_m=float(np.percentile(centers, 95)),
                probe_x_m=[-HALF_LENGTH, 0, HALF_LENGTH],
                probe_y_m=[-SINGLES_HALF_WIDTH, 0, SINGLES_HALF_WIDTH],
                image_displacement_p95_px_by_height_m={
                    str(z): float(np.percentile(shifts[:, probes[:, 2] == z], 95)) for z in heights})


def calibrate(config):
    image_size = tuple(config["image_size"])
    first = config["segment"]["start_frame"]
    stop = config["segment"]["end_frame_exclusive"]
    source_frame = config["calibration_frame"]
    if not 0 <= first <= source_frame < stop:
        raise ValueError("Calibration frame is outside the annotated segment")
    points = config["landmarks"]
    by_id = {row["id"]: row for row in points}
    train = [row for row in points if row["split"] == "train"]
    holdout = [row for row in points if row["split"] == "holdout"]
    if len(by_id) != len(points) or len(train) + len(holdout) != len(points) or not holdout:
        raise ValueError("Each distinct landmark needs a training or holdout role")
    world = np.array([row["world_m"] for row in train], dtype=float)
    pixels = np.array([row["pixel"] for row in train], dtype=float)
    ground = world[:, 2] == 0.0

    baseline = fit_camera(world, pixels, image_size)
    radial = fit_camera(world, pixels, image_size, fit_k1=True)
    report = {"video": config["video"], "segment": config["segment"],
              "calibration_frame": source_frame, "primary_model": "pinhole",
              "models": {}, "temporal": {},
              "dlt": {"training": dlt_diagnostics(world, pixels)},
              "annotation_perturbation": perturbation_check(world, pixels, image_size, baseline)}
    if np.count_nonzero(ground) >= 6:
        report["dlt"]["ground_training_only"] = dlt_diagnostics(world[ground], pixels[ground])
    rows = []
    for name, camera in (("pinhole", baseline), ("radial_k1", radial)):
        training = errors_by_point(camera, train, name, "train", source_frame)
        held = errors_by_point(camera, holdout, name, "holdout", source_frame)
        rows.extend(training + held)
        entry = {"focal_px": float(camera.K[0, 0]), "k1": float(camera.k1),
                 "center_m": camera.center.tolist(), "train": metrics(training), "holdout": metrics(held)}
        for kind, subset in (("ground", [r for r in held if r["z_m"] == 0]),
                             ("elevated", [r for r in held if r["z_m"] > 0])):
            if subset:
                entry[f"holdout_{kind}"] = metrics(subset)
        # A known floor height permits a metric backprojection check on unused landmarks.
        floor_pts = [p for p in holdout if p["world_m"][2] == 0]
        if floor_pts:
            center, rays = camera.rays([p["pixel"] for p in floor_pts])
            if np.any(np.abs(rays[:, 2]) < 1e-10):
                raise ValueError("Held-out floor ray is parallel to the ground")
            distance = -center[2] / rays[:, 2]
            if np.any(distance <= 0):
                raise ValueError("Held-out floor point lies behind the camera")
            xy = center[:2] + distance[:, None] * rays[:, :2]
            diff = xy - np.array([p["world_m"][:2] for p in floor_pts])
            entry["holdout_ground_ray_rmse_m"] = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
        report["models"][name] = entry

    for check in config["temporal_checks"]:
        frame = check["frame"]
        if frame == source_frame or not first <= frame < stop:
            raise ValueError("Temporal check must use another frame within the calibrated segment")
        landmarks = [dict(by_id[p["id"]], pixel=p["pixel"]) for p in check["observations"]]
        temporal = errors_by_point(baseline, landmarks, "pinhole", "other_frame", frame)
        rows.extend(temporal)
        report["temporal"][str(frame)] = metrics(temporal)
    return baseline, rows, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage3-fuzhou.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results/stage3")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    camera, rows, report = calibrate(config)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, obj in (("camera.json", camera.to_dict()), ("calibration.json", report)):
        (args.out / name).write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                                     encoding="utf-8")
    with (args.out / "reprojection.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    held = report["models"]["pinhole"]["holdout"]
    print(f"{held['n']} held-out landmarks, RMS reprojection error {held['rmse_px']:.3f} px -> {args.out}")


if __name__ == "__main__":
    main()
