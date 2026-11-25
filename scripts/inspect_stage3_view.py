"""Remeasure court markings and screen the selected Fuzhou rally for camera changes."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage3-fuzhou.json"
ANCHORS = (
    "far_baseline_outer_left", "far_baseline_outer_right",
    "near_baseline_outer_left", "near_baseline_outer_right",
    "near_short_service_outer_left", "near_short_service_outer_right",
    "far_short_service_outer_left", "far_short_service_outer_right",
)


def peak_offset(values):
    left, peak, right = map(float, values)
    curvature = 2 * (2 * peak - left - right)
    return (right - left) / curvature if curvature > 0 else 0.0


def white_peak(profile, centre, radius=5):
    """Locate the bright paint stripe across a single image scanline."""
    start = int(round(centre)) - radius
    values = profile[start:start + 2 * radius + 1].astype(float)
    j = int(np.argmax(values))
    if j == 0 or j == len(values) - 1 or values[j] - np.median(values) < 12:
        return None
    return start + j + peak_offset(values[j - 1:j + 2])


def paint_intersection(gray, landmark):
    """Fit two independent local paint ridges near a manually identified intersection."""
    key = landmark["id"]
    x, y = landmark["seed_px"]
    offsets = list(range(-21, -6)) + list(range(7, 22))
    horizontal = []
    vertical = []
    # Approximate directions of the visible stripes guide only the local search window.
    slope = -0.28 if key.endswith("left") else 0.28 if key.endswith("right") else 0.0
    for d in offsets:
        horizontal_exists = not ((key.endswith("outer_left") and d < 0)
                                 or (key.endswith("outer_right") and d > 0))
        if horizontal_exists:
            v = white_peak(gray[:, x + d], y)
            if v is not None:
                horizontal.append((x + d, v))
        vertical_exists = not ((key.startswith("near_baseline") and d > 0)
                               or (key.startswith("far_baseline") and d < 0)
                               or (key == "near_short_service_middle" and d < 0)
                               or (key == "far_short_service_middle" and d > 0))
        # The far doubles rear service stripe and the net tape overlap in this view.
        if vertical_exists and not 318 <= y + d <= 328:
            u = white_peak(gray[y + d, :], x + slope * d)
            if u is not None:
                vertical.append((y + d, u))
    if min(len(horizontal), len(vertical)) < 3:
        raise ValueError(f"Insufficient visible stripe near {key}")
    m, b = np.polyfit(*np.asarray(horizontal).T, 1)
    n, c = np.polyfit(*np.asarray(vertical).T, 1)
    u = (c + n * b) / (1 - n * m)
    return [round(float(u), 2), round(float(m * u + b), 2)]


def camera_change(gray, reference, anchors):
    """Match local background patches only; units are decoded image pixels."""
    before, after = [], []
    for x, y in anchors:
        template = reference[y-14:y+15, x-14:x+15]
        area = gray[y-25:y+26, x-25:x+26]
        scores = cv2.matchTemplate(area, template, cv2.TM_CCOEFF_NORMED)
        _, quality, _, (u, v) = cv2.minMaxLoc(scores)
        if quality < 0.80 or not (0 < u < scores.shape[1]-1 and 0 < v < scores.shape[0]-1):
            continue
        # Interpolate the peak in x and y to measure subpixel translation.
        dx = peak_offset(scores[v, u-1:u+2])
        dy = peak_offset(scores[v-1:v+2, u])
        before.append((x, y))
        after.append((x+u-11+dx, y+v-11+dy))
    if len(before) < 4:
        return {"matched": len(before)}
    # A similarity fit summarizes coherent pan and zoom across well-spaced anchors.
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    centre = np.array([gray.shape[1]/2, gray.shape[0]/2])
    transform, _ = cv2.estimateAffinePartial2D(before-centre, after-centre, method=cv2.LMEDS)
    if transform is None:
        return {"matched": len(before)}
    return dict(matched=len(before), median_displacement_px=float(np.median(np.linalg.norm(after-before, axis=1))),
                centre_translation_px=float(np.linalg.norm(transform[:, 2])),
                scale=float(np.linalg.norm(transform[:, 0])))


def inspect(config):
    path = ROOT / config["video"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != config["source_sha256"]:
        raise ValueError(f"Unexpected source video: {path}")
    points = {p["id"]: p for p in config["landmarks"]}
    anchors = np.rint([points[key]["pixel"] for key in ANCHORS]).astype(int)
    checks = {sample["frame"]: sample["observations"] for sample in config["temporal_checks"]}
    reference_index = config["calibration_frame"]
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, reference_index)
        ok, image = cap.read()
        if not ok:
            raise OSError(path)
        reference = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        measurements = {}
        local_changes = []
        while True:
            ok, image = cap.read()
            if not ok:
                break
            i = len(local_changes)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if i == reference_index:
                measurements[i] = {key: paint_intersection(gray, point) for key, point in points.items()
                                   if point["method"] == "local_paint_centrelines"}
            elif i in checks:
                measurements[i] = {p["id"]: paint_intersection(gray, points[p["id"]]) for p in checks[i]}
            local_changes.append(camera_change(gray, reference, anchors))
    finally:
        cap.release()
    first, stop = (config["segment"][k] for k in ("start_frame", "end_frame_exclusive"))
    stable = local_changes[first:stop]
    if len(stable) != stop-first or any("scale" not in m for m in stable):
        raise ValueError("Selected interval contains unresolved camera motion or insufficient imagery")
    reread_error = {}
    for i, measured in measurements.items():
        stored = points if i == reference_index else {p["id"]: p for p in checks[i]}
        reread_error[i] = max(float(np.linalg.norm(np.asarray(xy) - stored[key]["pixel"]))
                              for key, xy in measured.items())
    return dict(
        decoded_frames=len(local_changes), checked_segment=config["segment"],
        anchor_ids=ANCHORS, min_matched_anchors=min(m["matched"] for m in stable),
        max_median_anchor_displacement_px=max(m["median_displacement_px"] for m in stable),
        max_centre_translation_px=max(m["centre_translation_px"] for m in stable),
        max_abs_scale_change=max(abs(m["scale"]-1) for m in stable),
        outside_segment={i: m for i, m in enumerate(local_changes) if not first <= i < stop},
        remeasured_pixel_centres=measurements,
        max_stored_paint_discrepancy_px_by_frame=reread_error,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"data/work/stage3_observations/view-check.json")
    args = parser.parse_args()
    report = inspect(json.loads(CONFIG.read_text(encoding="utf-8")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.out)
