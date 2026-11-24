"""Independently verify the stage-one CSVs, encoded videos and ball-to-image alignment."""
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/raw/tracknetv3/Professional/match4"
OUTPUT = ROOT / "artifacts/stage1"
SIZE = (1280, 720)
FPS = 25
LAGS = range(-2, 3)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def open_video(path):
    cap = cv2.VideoCapture(str(path))
    size = tuple(int(cap.get(prop)) for prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or size != SIZE or abs(fps - FPS) > 0.001:
        cap.release()
        raise ValueError(f"{path}: expected {SIZE} at {FPS} fps, got {size} at {fps:.6f}")
    return cap


def thumbnail(gray):
    return cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)


def ball_patch(gray, point):
    x, y = point
    return gray[y - 2:y + 3, x - 2:x + 3]


def read_source(rally, labels, source_count, end):
    visible = {idx: (int(row["X"]), int(row["Y"])) for idx, row in labels.items()
               if int(row["Visibility"]) != 0}
    # Use exactly the same moving, visible ball centers at each lag. Slow/standing
    # balls provide little information about which video frame matches the label.
    centers = {}
    for i in sorted(visible):
        x, y = visible[i]
        if (2 <= i < end - 2 and 2 <= x < SIZE[0] - 2 and 2 <= y < SIZE[1] - 2
                and i - 1 in visible and i + 1 in visible
                and math.dist(visible[i - 1], visible[i + 1]) >= 16):
            centers[i] = (x, y)
    check(centers, "empty moving-ball sample")
    samples = {lag: [] for lag in LAGS}
    source = []
    background_frames = []
    cap = open_video(INPUT / "video" / f"{rally}.mp4")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i = len(source)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            source.append(thumbnail(gray))
            if i >= end:
                continue
            if i % 20 == 0:
                background_frames.append(gray)
            for lag in LAGS:
                center = centers.get(i - lag)
                if center is not None:
                    samples[lag].append(ball_patch(gray, center).copy())
    finally:
        cap.release()
    check(len(source) == source_count, f"expected {source_count} raw frames, got {len(source)}")
    background = np.median(background_frames, axis=0).astype(np.int16)
    reference = np.array([ball_patch(background, center) for center in centers.values()])
    scores = {}
    for lag in LAGS:
        patches = np.array(samples[lag])
        check(patches.shape == reference.shape, f"different sample sets at lag {lag}")
        peaks = np.maximum(patches.astype(np.int16) - reference, 0).max(axis=(1, 2))
        scores[lag] = {"median": round(float(np.median(peaks)), 2),
                       "mean": round(float(np.mean(peaks)), 2)}
    check(all(scores[0]["median"] > scores[k]["median"] for k in LAGS if k),
          f"moving-ball brightness prefers a time shift: {scores}")
    return source, {"n_shared_centers": len(centers), "positive_gray_residual": scores}


def validate_frames(rows, labels, end, events):
    check(len(rows) == end, f"frames.csv has {len(rows)} rows, expected {end}")
    by_frame = {int(event["frame_index"]): event for event in events}
    missing = {}
    for i, row in enumerate(rows):
        check(int(row["frame_index"]) == i, f"frames.csv out of order at {i}")
        check(abs(float(row["timestamp_s"]) - i / FPS) < 1e-5, f"timestamp at {i}")
        check(int(row["source_frame_index"]) == i, f"clip frame at {i}")
        label = labels.get(i)
        absent = label is None or int(label["Visibility"]) == 0
        event = by_frame.get(i, {})
        check(row["shot_index"] == event.get("shot_index", "")
              and row["shot_type"] == event.get("shot_type", ""), f"event linkage at {i}")
        check(row["source_visibility"] == (label["Visibility"] if label else ""), f"visibility at {i}")
        check(row["observation_origin"] == ("human_annotation" if label else ""), f"provenance at {i}")
        value = row["is_missing"].lower()
        check(value in ("0", "1", "false", "true") and (value in ("1", "true")) == absent,
              f"missing flag disagrees with raw data at {i}")
        check(row["confidence"] == "", f"unexpected observation confidence at {i}")
        if absent:
            reason = row["missing_reason"]
            check(row["x_px"] == row["y_px"] == ""
                  and reason == ("not_visible" if label else "no_label_row"), f"missing data at {i}")
            missing[reason] = missing.get(reason, 0) + 1
        else:
            point = (float(row["x_px"]), float(row["y_px"]))
            check(row["missing_reason"] == "", f"unexpected missing reason at {i}")
            check(point == (float(label["X"]), float(label["Y"])), f"wrong ball center at {i}")
            check(0 <= point[0] < SIZE[0] and 0 <= point[1] < SIZE[1], f"ball bounds at {i}")
    return missing


def validate_events(rally, events, config, end):
    game, a, b = map(int, rally.split("_"))
    original = [r for r in read_csv(ROOT / "data/raw/shuttleset" / f"set{game}.csv")
                if (int(r["roundscore_A"]), int(r["roundscore_B"])) == (a, b) and r["ball_round"]]
    check(len(events) == len(original), "event count disagrees with ShuttleSet")
    by_shot = {}
    last_frame = -1
    for row, raw in zip(events, original):
        i = int(row["frame_index"])
        shot = int(row["shot_index"])
        check(last_frame < i < end, f"event order at {i}")
        last_frame = i
        check(shot == int(float(raw["ball_round"])), f"event index at {i}")
        original_frame = int(float(raw["frame_num"]))
        mapped_frame = original_frame - config["source_frame_at_local_zero"]
        reviewed = config["reviewed_contacts"]
        check((int(row["shuttleset_frame"]), int(row["mapped_frame"])) == (original_frame, mapped_frame),
              f"ShuttleSet mapping at {i}")
        check(i == reviewed.get(str(shot), mapped_frame), f"reviewed event frame at {i}")
        check(row["timing_method"] == ("video_review" if str(shot) in reviewed else "rally_offset"),
              f"timing provenance at {i}")
        check(abs(float(row["timestamp_s"]) - i / FPS) < 1e-5, f"event timestamp at {i}")
        check(row["shot_type"] == raw["type"] and row["player"] == raw["player"], f"ShuttleSet label at {i}")
        by_shot[shot] = row
    return by_shot


def temporal_distance(actual, source, i):
    # Compare the same moving pixels at the nominal index and its neighbours.
    # Downsampling reduces sensitivity to the separate MP4 encode of each product.
    prev = max(i - 1, 0)
    nxt = min(i + 1, len(source) - 1)
    frames = np.asarray([source[prev], source[i], source[nxt]], dtype=np.int16)[:, 20:160]
    moving = np.ptp(frames, axis=0) >= 8
    if np.count_nonzero(moving) < 30:
        return None
    err = np.abs(frames[:, moving] - actual[20:160][moving])
    score = np.minimum(err, 30).mean(axis=1)
    return score


def court_marks(configs):
    actual = json.loads((OUTPUT / "court.json").read_text(encoding="utf-8"))
    raw = next(r for r in read_csv(ROOT / "data/raw/shuttleset/homography.csv")
               if r["video"] == configs["match"])
    marks = []
    for name in ("upleft", "upright", "downright", "downleft"):
        xy = [float(raw[f"{name}_{axis}"]) for axis in ("x", "y")]
        check(actual["court_corners_px"][name] == xy, f"court corner {name}")
        marks.append((xy, (255, 218, 36)))
    for name, xy in configs["net_post_tops_px"].items():
        check(actual["net_post_tops_px"][name] == xy, f"net post {name}")
        marks.append((xy, (224, 85, 255)))
    return marks


def validate_video(path, source, start, stop, marks=()):
    cap = open_video(path)
    errors = []
    shifts = []
    boundaries = {}
    try:
        for i in range(start, stop):
            ok, frame = cap.read()
            check(ok, f"{path.name} ends before frame {i}")
            expected_ms = (i - start) * 1000 / FPS
            check(abs(cap.get(cv2.CAP_PROP_POS_MSEC) - expected_ms) < 1, f"{path.name}: PTS at {i}")
            for (px, py), color in marks:
                x, y = round(px), round(py)
                patch = frame[y - 6:y + 7, x - 6:x + 7].astype(np.int16)
                color_error = np.abs(patch - color).max(axis=2)
                check(np.count_nonzero(color_error < 70) >= 3, f"overlay marker at {x},{y} frame {i}")
            gray = thumbnail(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            errors.append(float(cv2.absdiff(gray, source[i]).mean()))
            distance = temporal_distance(gray, source, i)
            if distance is not None:
                shifts.append(distance)
                if i in (start, stop - 1):
                    boundaries[i] = [round(float(s), 2) for s in distance]
        check(not cap.read()[0], f"{path.name} contains extra frames")
    finally:
        cap.release()
    limit = 17 if marks else 4
    check(max(errors) < limit, f"{path.name}: high source-image error {max(errors):.2f}")
    check(shifts, f"{path.name} has no motion to compare")
    means = np.mean(shifts, axis=0)
    check(means[1] < min(means[0], means[2]), f"{path.name}: neighbouring source frame matches better")
    if not marks:
        for i, distance in boundaries.items():
            check(distance[1] <= min(distance) + 0.5, f"{path.name}: wrong boundary frame {i}: {distance}")
    return {"frames_decoded": stop - start, "fps": FPS, "size": SIZE,
            "source_gray_mae_max": round(max(errors), 2),
            "motion_error_lags_minus1_zero_plus1": np.round(means, 2).tolist(),
            "boundary_motion_error": boundaries}


def validate_rally(rally, config, marks):
    source_count, end = config["source_frames"], config["end_frame_exclusive"]
    check(0 < end <= source_count, f"invalid selected range [0, {end})")
    directory = OUTPUT / rally
    raw_labels = read_csv(INPUT / "csv" / f"{rally}_ball.csv")
    labels = {int(row["Frame"]): row for row in raw_labels}
    check(len(labels) == len(raw_labels), "duplicate raw frame")
    source, alignment = read_source(rally, labels, source_count, end)
    event_rows = read_csv(directory / "events.csv")
    events = validate_events(rally, event_rows, config, end)
    missing = validate_frames(read_csv(directory / "frames.csv"), labels, end, event_rows)
    videos = {"rally.mp4": validate_video(directory / "rally.mp4", source, 0, end),
              "overlay.mp4": validate_video(directory / "overlay.mp4", source, 0, end, marks)}
    shots = read_csv(directory / "shots.csv")
    check(len(shots) == len(config["export_strokes"]), "incorrect number of shot clips")
    files = {path.relative_to(directory).as_posix() for path in (directory / "shots").glob("*.mp4")}
    check({row["file"] for row in shots} == files, "shot index and files differ")
    for row, shot in zip(shots, config["export_strokes"]):
        start, stop, filename = int(row["start_frame"]), int(row["end_frame_exclusive"]), row["file"]
        check((start, stop) == (int(events[shot]["frame_index"]), int(events[shot + 1]["frame_index"])),
              f"wrong bounds for {filename}")
        check(int(row["shot_index"]) == shot and row["shot_type"] == events[shot]["shot_type"],
              f"wrong stroke label on {filename}")
        check(filename not in videos and 0 <= start < stop <= end, f"invalid clip {filename}")
        videos[filename] = validate_video(directory / filename, source, start, stop)
    return {"source_frames_decoded": source_count, "usable_frames": end, "missing": missing,
            "events": len(events), "ball_alignment": alignment, "videos": videos}


def main():
    configs = json.loads((ROOT / "configs/fuzhou2019.json").read_text(encoding="utf-8"))
    report = {"passed": False, "rallies": {}}
    subject = "court"
    try:
        marks = court_marks(configs)
        for subject, config in configs["rallies"].items():
            report["rallies"][subject] = validate_rally(subject, config, marks)
        report["passed"] = True
    except (ValueError, OSError, KeyError, cv2.error, StopIteration) as exc:
        report["error"] = f"{subject}: {exc}"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "validation.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(report["error"])
    print(f"Validated {len(report['rallies'])} rallies. Results in {target}")


if __name__ == "__main__":
    main()
