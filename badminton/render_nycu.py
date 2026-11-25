"""Show a left-only 3D fit on paired NYCU recordings at the published frame times."""

import argparse
import csv
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

from .render_real import FIT, HOLDOUT, MODEL, ROOT, label, video_writer


def pixels(rows, prefix):
    return np.array([[float(row[f"{prefix}_{axis}_px"] or "nan") for axis in ("u", "v")] for row in rows])


def panel(image, observations, projected, i, view):
    scale = 600 / image.shape[1]
    picture = cv2.resize(image, (600, 450), interpolation=cv2.INTER_AREA)
    obs = observations[i] * scale
    pred = projected[i] * scale
    path = projected[:i+1] * scale
    for start, end in zip(path, path[1:]):
        if np.isfinite([start, end]).all():
            cv2.line(picture, tuple(start.round().astype(int)), tuple(end.round().astype(int)), MODEL, 1, cv2.LINE_AA)
    color = FIT if view == "a" else HOLDOUT
    if np.isfinite(pred).all():
        cv2.drawMarker(picture, tuple(pred.round().astype(int)), MODEL, cv2.MARKER_CROSS, 15, 2, cv2.LINE_AA)
    if np.isfinite(obs).all():
        cv2.circle(picture, tuple(obs.round().astype(int)), 6, color, 2, cv2.LINE_AA)
    distance = np.linalg.norm(pred - obs) / scale
    score = f"{distance:.1f} px at native resolution" if np.isfinite(distance) else "No scored pixel pair"
    title = np.full((66, 600, 3), (33, 36, 38), np.uint8)
    label(title, "A  |  Left camera, used for fit" if view == "a" else "B  |  Right camera, held out",
          12, 26, color=color, scale=.68)
    label(title, score, 12, 54, scale=.61)
    return np.vstack((title, picture))


def render(data, raw, out):
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    with video_writer(out / "stereo-replay.mp4", 1200, 556, 30) as stream:
        for sample in ("001", "002", "003"):
            with (data / f"{sample}.csv").open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            tracks = {view: (pixels(rows, view), pixels(rows, view+"_pred")) for view in ("a", "b")}
            with ExitStack() as stack:
                readers = []
                for side in ("L", "R"):
                    cap = cv2.VideoCapture(str(raw / sample / f"CameraReader{side}.avi"))
                    stack.callback(cap.release)
                    if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(rows):
                        raise ValueError(f"{sample}/{side}: frame count does not match the benchmark")
                    readers.append(cap)
                for i, row in enumerate(rows):
                    if int(row["frame_index"]) != i:
                        raise ValueError(f"{sample}: unexpected frame index {row['frame_index']}")
                    panels = []
                    for cap, view in zip(readers, ("a", "b")):
                        ok, frame = cap.read()
                        if not ok or frame.shape[:2] != (1080, 1440):
                            raise ValueError(f"{sample}/{view}: unexpected source frame at {i}")
                        panels.append(panel(frame, *tracks[view], i, view))
                    header = np.full((40, 1200, 3), (25, 28, 30), np.uint8)
                    label(header, f'NYCU   {sample}   |   frame {i:02d}   |   t = {float(row["t_s"]):.3f} s',
                          12, 27)
                    label(header, "+ 3D reprojection", 604, 27, color=MODEL)
                    label(header, "O annotated centre", 825, 27)
                    output = np.vstack((header, np.hstack(panels)))
                    stream.write(output.tobytes())
                    count += 1
                    if sample == "002" and i == len(rows)//2:
                        if not cv2.imwrite(str(out / "independent-view.jpg"), output):
                            raise OSError("Could not write paired-camera still")
    print(f"{count} real stereo frame pairs, 120 fps played at 30 fps -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/stage6/nycu")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/nycu-stereo6/vicon_dataset")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/stage6/nycu")
    args = parser.parse_args()
    render(args.data, args.raw, args.out)
