"""Replay fitted single strokes on their original broadcast frames."""

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import cv2
import imageio_ffmpeg
import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure

from .plots import COLORS, STYLE, _court

ROOT = Path(__file__).resolve().parents[1]
MODEL, FIT, HOLDOUT = (46, 221, 251), (110, 231, 100), (237, 80, 239)


def group_rows(file):
    by_shot = defaultdict(list)
    with file.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            by_shot[row["shot_id"]].append(row)
    for rows in by_shot.values():
        rows.sort(key=lambda row: int(row["frame_index"]))
    return by_shot


def label(image, string, x, y, *, scale=.63, color=(240, 240, 240)):
    cv2.putText(image, string, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def stroke_label(shot):
    kind = shot.get("shot_type_en", shot["shot_type"]).replace("_", " ")
    return f'{shot["rally_id"]}  /  #{shot["shot_index"]} {kind}'


def draw_frame(frame, uv, i, obs):
    points = np.rint(uv * 16).astype(np.int32).reshape((-1, 1, 2))
    ghost = frame.copy()
    cv2.polylines(ghost, [points], False, MODEL, 2, cv2.LINE_AA, shift=4)
    painted = cv2.addWeighted(ghost, .45, frame, .55, 0)
    cv2.polylines(painted, [points[:i+1]], False, MODEL, 2, cv2.LINE_AA, shift=4)
    model_pt = tuple(np.rint(uv[i]).astype(int))
    cv2.drawMarker(painted, model_pt, (15, 31, 31), cv2.MARKER_CROSS, 13, 3, cv2.LINE_AA)
    cv2.drawMarker(painted, model_pt, MODEL, cv2.MARKER_CROSS, 13, 1, cv2.LINE_AA)
    if obs["role"] in ("fit", "holdout"):
        measured = tuple(int(round(float(obs[key]))) for key in ("u_px", "v_px"))
        color = HOLDOUT if obs["role"] == "holdout" else FIT
        cv2.circle(painted, measured, 8, (30, 30, 30), 3, cv2.LINE_AA)
        cv2.circle(painted, measured, 8, color, 2, cv2.LINE_AA)
    return painted


def sample_panel(original, painted, uv, obs, shot, passes):
    centre = np.rint([float(obs["u_px"]), float(obs["v_px"])]).astype(int)
    h, w = original.shape[:2]
    # Use the same raw-pixel crop in both insets, centred on the reviewed observation.
    x, y = np.clip(centre - 46, (0, 0), (w-92, h-92))
    left = cv2.resize(painted, (640, 360), interpolation=cv2.INTER_AREA)
    side = np.full((360, 360, 3), (34, 38, 40), np.uint8)
    pair = (original, painted)
    for col, img in enumerate(pair):
        zoom = cv2.resize(img[y:y+92, x:x+92], (168, 168), interpolation=cv2.INTER_NEAREST)
        side[77:245, 8+col*178:176+col*178] = zoom
    label(side, "Source", 12, 65)
    label(side, "Reprojection", 185, 65)
    distance = np.linalg.norm(uv - [float(obs["u_px"]), float(obs["v_px"])])
    label(side, f'Held-out frame {obs["frame_index"]}', 12, 276)
    label(side, f'Pixel distance   {distance:.1f} px', 12, 305)
    title = np.full((40, 1000, 3), (29, 33, 35), np.uint8)
    title_text = f'{shot["split"].upper()}   {stroke_label(shot)}'
    label(title, title_text, 12, 27, scale=.72)
    label(title, "3D checks passed" if passes else "3D checks failed", 648, 27,
          color=FIT if passes else MODEL)
    return np.vstack((title, np.hstack((left, side))))


def trajectory_figure(study, shots, tracks, passing):
    with mpl.rc_context(STYLE):
        fig = Figure(figsize=(12, 3.1 * ((len(shots)+2)//3)), layout="constrained")
        fig.suptitle("Fuzhou 2019   |   Single-camera 3D estimates\n"
                     "Blue: passes physical checks     Orange: fails physical checks     Dot: launch",
                     fontsize=13, fontweight="bold")
        for i, shot in enumerate(shots):
            ax = fig.add_subplot((len(shots)+2)//3, 3, i+1, projection="3d", computed_zorder=False)
            xyz = np.array([[float(r[k]) for k in ("x_m", "y_m", "z_m")] for r in tracks[shot["shot_id"]]])
            _court(ax, [SimpleNamespace(position=xyz)])
            ax.set_box_aspect(ax.get_box_aspect(), zoom=.88)
            color = COLORS[0 if shot["shot_id"] in passing else 1]
            ax.plot(*xyz.T, color=color, lw=2.4, zorder=4)
            ax.scatter(*xyz[0], marker="o", s=26, c=color, edgecolors="white", zorder=5)
            ax.set_title(stroke_label(shot), fontsize=10)
        fig.savefig(study / "real-trajectories.png", dpi=145)


@contextmanager
def video_writer(destination, width, height, fps):
    with NamedTemporaryFile(dir=destination.parent, suffix=".mp4", delete=False) as stream:
        temp = Path(stream.name)
    try:
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo",
                   "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "fast",
                   "-movflags", "+faststart", str(temp)]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            try:
                yield encoder.stdin
            finally:
                encoder.stdin.close()
            if encoder.wait():
                raise OSError("FFmpeg could not encode the real-shot video")
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)


def render(study, out):
    shots = json.loads((study / "shots.json").read_text(encoding="utf-8"))["shots"]
    tracks = group_rows(study / "trajectories.csv")
    observations = group_rows(study / "observations.csv")
    with (study / "fits.csv").open(encoding="utf-8", newline="") as stream:
        passing = {r["shot_id"] for r in csv.DictReader(stream) if r["physics_screen"] == "passes"}
    # Compare the first test clear and lob, independently of the fitted error.
    examples = {next(s["shot_id"] for s in shots if s["split"] == "test" and s["shot_type"] == kind)
                for kind in ("clear", "lob")}
    out.mkdir(parents=True, exist_ok=True)
    panels, count = [], 0
    capture = cv2.VideoCapture(str(ROOT / shots[0]["source_video"]))
    width, height = [int(capture.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]
    capture.release()
    fps = shots[0]["fps"]
    with video_writer(out / "reprojection.mp4", width, height, fps) as stream:
        for shot in shots:
            sid = shot["shot_id"]
            rows, obs = tracks[sid], observations[sid]
            frames = list(range(shot["start_frame"], shot["end_frame_exclusive"]))
            if any([int(r["frame_index"]) for r in table] != frames for table in (rows, obs)):
                raise ValueError(f"{sid}: frame indices disagree")
            uv = np.array([[float(r[k]) for k in ("u_fit_px", "v_fit_px")] for r in rows])
            held = [j for j, r in enumerate(obs) if r["role"] == "holdout"
                    and 46 <= float(r["u_px"]) <= width-46 and 46 <= float(r["v_px"]) <= height-46]
            example = min(held, key=lambda j: abs(j-(len(obs)-1)/2)) if sid in examples and held else None
            capture = cv2.VideoCapture(str(ROOT / shot["source_video"]))
            try:
                if capture.get(cv2.CAP_PROP_FPS) != fps or not capture.set(cv2.CAP_PROP_POS_FRAMES, frames[0]):
                    raise ValueError(f"{sid}: cannot seek or frame rate changed")
                for i, frame_no in enumerate(frames):
                    ok, original = capture.read()
                    if not ok or original.shape[:2] != (height, width):
                        raise OSError(f"{sid}: source frame {frame_no} missing or size changed")
                    painted = draw_frame(original, uv, i, obs[i])
                    if i == example:
                        panels.append(sample_panel(original, painted, uv[i], obs[i], shot, sid in passing))
                    cv2.rectangle(painted, (0, 0), (width, 34), (29, 33, 35), -1)
                    check = "3D pass" if sid in passing else "3D flag"
                    label(painted, f'{stroke_label(shot)}   |   {shot["split"]}   |   f{frame_no}   |   {check}',
                          12, 23)
                    label(painted, "+ model", width-478, 23, color=MODEL)
                    label(painted, "O fit", width-363, 23, color=FIT)
                    label(painted, "O held-out", width-279, 23, color=HOLDOUT)
                    label(painted, obs[i]["role"], width-118, 23)
                    stream.write(painted.tobytes())
                    count += 1
            finally:
                capture.release()
    if len(panels) != len(examples) or not cv2.imwrite(str(out / "reprojection.jpg"), np.vstack(panels)):
        raise OSError("Could not write representative held-out frames")
    trajectory_figure(study, shots, tracks, passing)
    print(f"{len(shots)} strokes, {count} original frames -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "results/stage6")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/stage6")
    args = parser.parse_args()
    render(args.study, args.out)


if __name__ == "__main__":
    main()
