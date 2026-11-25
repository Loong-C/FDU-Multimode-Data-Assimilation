"""Prepare fixed rally splits, measured cameras and timed observations for stage six."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera, fit_camera
from .reconstruct import write_json, write_rows
from scripts.inspect_stage3_view import ANCHORS, camera_change, paint_intersection, peak_offset


ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT / "artifacts/stage1"
STAGE6_CONFIG = ROOT / "configs/stage6-fuzhou.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def indexed_csv(path, key):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        indexed = {}
        for row in csv.DictReader(stream):
            i = int(row[key])
            if i in indexed:
                raise ValueError(f"Duplicate {key}={i} in {path}")
            indexed[i] = row
        return indexed


def read_gray(capture, frame):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = capture.read()
    if not ok:
        raise ValueError(f"Missing video frame {frame}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def register_net(reference, gray, pixel):
    """Remeasure displacement of a visually identified net attachment or lower hem."""
    x, y = np.rint(pixel).astype(int)
    template = reference[y-13:y+14, x-13:x+14]
    scores = cv2.matchTemplate(gray[y-19:y+20, x-19:x+20], template, cv2.TM_CCOEFF_NORMED)
    _, quality, _, (u, v) = cv2.minMaxLoc(scores)
    if quality < 0.90 or u in (0, 12) or v in (0, 12):
        raise ValueError(f"Cannot register net patch centred at {pixel}")
    dx = u-6 + peak_offset(scores[v, u-1:u+2])
    dy = v-6 + peak_offset(scores[v-1:v+2, u])
    return np.round(np.asarray(pixel) + [dx, dy], 2).tolist(), round(quality, 4)


def camera_error(camera, points):
    world = [p["world_m"] for p in points]
    distances = np.linalg.norm(camera.project(world) - [p["pixel"] for p in points], axis=1)
    return {"n": len(points), "rmse_px": round(float(np.sqrt(np.mean(distances**2))), 4),
            "max_px": round(float(distances.max()), 4)}


def measure_camera(path, info, fps, geometry, net_reference, original_camera):
    capture = cv2.VideoCapture(str(path))
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        size = tuple(int(capture.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        if (frames != info["source_frames"] or capture.get(cv2.CAP_PROP_FPS) != fps
                or list(size) != geometry["image_size"]):
            raise ValueError(f"Source video differs from manifest: {path}")
        stop = info["end_frame_exclusive"]
        # A median of evenly spaced decoded frames suppresses foreground occlusions.
        samples = np.linspace(0, stop-1, 15, dtype=int).tolist()
        gray = np.median([read_gray(capture, i) for i in samples], axis=0).astype(np.uint8)
        points = []
        for source in geometry["landmarks"]:
            point = {k: source[k] for k in ("id", "world_m", "split")}
            if "seed_px" in source:
                point["pixel"] = paint_intersection(gray, source)
                point["method"] = "local_paint_centrelines"
            else:
                point["pixel"], point["patch_correlation"] = register_net(net_reference, gray, source["pixel"])
                point["method"] = "registered_visually_checked_net_patch"
            points.append(point)
        train = [p for p in points if p["split"] == "train"]
        holdout = [p for p in points if p["split"] == "holdout"]
        camera = fit_camera([p["world_m"] for p in train], [p["pixel"] for p in train], size)
        by_id = {p["id"]: p for p in points}
        anchors = np.rint([by_id[k]["pixel"] for k in ANCHORS]).astype(int)
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        motion = []
        for i in range(frames):
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Decode failed at {path.name} frame {i}")
            motion.append(camera_change(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), gray, anchors))
        stable = motion[:stop]
        if any("scale" not in row for row in stable):
            raise ValueError(f"Unresolved camera geometry inside {path.name}'s main view")
        summary = {
            "source_video": path.relative_to(ROOT).as_posix(),
            "decoded_frames": frames, "stable_frame_range_half_open": [0, stop],
            "median_reference_frames": samples,
            "method": "Independent pinhole fit per rally. Floor paint intersections were remeasured on a median image and net attachments checked against the manually picked stage-three frame. A 27x27 pixel patch localizes the post or seam. The lower net edge uses nominal elevation 0.79 m.",
            "landmarks": points, "train": camera_error(camera, train), "holdout": camera_error(camera, holdout),
            "original_stage3_camera_on_holdout": camera_error(original_camera, holdout),
            "motion_anchor_ids": list(ANCHORS),
            "min_matched_anchors": min(row["matched"] for row in stable),
            "max_median_anchor_displacement_px": round(max(row["median_displacement_px"] for row in stable), 4),
            "max_centre_translation_px": round(max(row["centre_translation_px"] for row in stable), 4),
            "max_abs_scale_change": round(max(abs(row["scale"]-1) for row in stable), 7),
            "outside_stable_interval": {str(i): row for i, row in enumerate(motion) if i >= stop},
        }
        if summary["max_centre_translation_px"] > 2.0 or summary["max_abs_scale_change"] > 0.005:
            raise ValueError(f"Possible camera motion inside {path.name}'s main view; segment and recalibrate")
        return camera, summary
    finally:
        capture.release()


def foot_proxy(event, camera):
    uv = [float(event[k]) for k in ("player_x_px", "player_y_px")]
    origin, direction = camera.rays(uv)
    distance = -origin[2] / direction[2]
    if distance <= 0 or not np.isfinite(distance):
        raise ValueError(f"Player annotation misses the floor: {uv}")
    return uv, np.round((origin + distance * direction)[:2], 4).tolist()


def prepare_shots(rally, selection, info, fps, camera, video):
    events = indexed_csv(STAGE1 / rally / "events.csv", "shot_index")
    original = indexed_csv(ROOT / "data/raw/tracknetv3/Professional/match4/csv" / f"{rally}_ball.csv", "Frame")
    frames = indexed_csv(STAGE1 / rally / "frames.csv", "frame_index")
    shots, observations = [], []
    previous_end = -1
    for shot_index, review in selection["shots"].items():
        index = int(shot_index)
        event, following = events[index], events[index + 1]
        start, end = int(event["frame_index"]), int(following["frame_index"])
        if not previous_end <= start < end < info["end_frame_exclusive"] or following["player"] == event["player"]:
            raise ValueError(f"Invalid contact boundaries or players for {rally} #{index}")
        previous_end = end
        shot_id = f"{rally}_{index:02d}"
        rows = []
        for i in range(start, end):
            row = frames[i]
            if int(row["source_frame_index"]) != i:
                raise ValueError(f"Unexpected source frame mapping in {rally} at {i}")
            label = original.get(i)
            # Preserve explicitly missing and absent labels separately in the provenance.
            visible = label is not None and int(label["Visibility"]) > 0 and (int(label["X"]), int(label["Y"])) != (0, 0)
            uv = (int(label["X"]), int(label["Y"])) if visible else ("", "")
            if tuple(str(v) for v in uv) != (row["x_px"], row["y_px"]):
                raise ValueError(f"Stage-one coordinates differ from released CSV: {rally} frame {i}")
            fold = "holdout" if (i-start) % 4 == 2 else "fit"
            rows.append(dict(shot_id=shot_id, rally_id=rally, frame_index=i, t_s=round((i-start)/fps, 6),
                             u_px=uv[0], v_px=uv[1], role=fold if visible else "missing",
                             scheduled_role=fold, source_time_s=round(i/fps, 6),
                             source_visibility=row["source_visibility"], missing_reason=row["missing_reason"]))
        player_uv, player_xy = foot_proxy(event, camera)
        receiver_uv, receiver_xy = foot_proxy(following, camera)
        shots.append(dict(shot_id=shot_id, rally_id=rally, shot_index=index, shot_type=event["shot_type_en"],
                          shot_type_zh=event["shot_type"], split=selection["split"], fps=fps,
                          start_frame=start, end_frame_exclusive=end, source_video=video,
                          start_shuttleset_frame=int(event["shuttleset_frame"]),
                          next_shuttleset_frame=int(following["shuttleset_frame"]),
                          player=event["player"], receiver=following["player"],
                          player_uv_px=player_uv, player_xy_m=player_xy,
                          receiver_uv_px=receiver_uv, receiver_xy_m=receiver_xy,
                          frame_counts=dict(Counter(row["role"] for row in rows)), **review))
        observations.extend(rows)
    return shots, observations


def build(out):
    config = read_json(STAGE6_CONFIG)
    manifest = read_json(STAGE1 / "manifest.json")
    geometry = read_json(ROOT / "configs/stage3-fuzhou.json")
    original_camera = Camera.from_dict(read_json(ROOT / "results/stage3/camera.json"))
    capture = cv2.VideoCapture(str(ROOT / geometry["video"]))
    try:
        net_reference = read_gray(capture, geometry["calibration_frame"])
    finally:
        capture.release()
    shots, observations, cameras, checks = [], [], {}, {}
    for rally, selection in config["rallies"].items():
        info = manifest["rallies"][rally]
        for filename, digest in info["inputs"].items():
            if hashlib.sha256((ROOT / "data/raw" / filename).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Unexpected source content: {filename}")
        video = f"data/raw/tracknetv3/Professional/match4/video/{rally}.mp4"
        camera, checks[rally] = measure_camera(ROOT/video, info, manifest["fps"], geometry,
                                              net_reference, original_camera)
        cameras[rally] = camera.to_dict()
        batch, rows = prepare_shots(rally, selection, info, manifest["fps"], camera, video)
        shots.extend(batch)
        observations.extend(rows)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/"cameras.json", {"cameras": cameras})
    write_json(out/"camera-checks.json", checks)
    write_json(out/"shots.json", {**{k: v for k, v in config.items() if k != "rallies"},
               "frame_convention": "Zero-based local MP4 frame, [start_frame,end_frame_exclusive), t_s measured from start_frame. The right boundary is the next stroke, not ground impact.",
               "foot_proxy": "ShuttleSet player-location image annotations at each player's own stroke, intersected with the floor plane. Position is a weak contact proxy affected by reach, lunges and jumping; independent metre-scale contact/landing measurements are unavailable.",
               "shots": shots})
    write_rows(out/"observations.csv", observations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"results/stage6")
    build(parser.parse_args().out)
