"""Draw a calibrated court on the source frames belonging to one camera segment."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import imageio_ffmpeg
import numpy as np

from .camera import Camera
from .court import court_wireframe

ROOT = Path(__file__).resolve().parents[1]
FLOOR, NET, HOLDOUT = (240, 217, 27), (24, 200, 255), (220, 75, 245)


def draw_geometry(frame, paths):
    overlay = frame.copy()
    for name, uv in paths.items():
        color = NET if name.startswith(("net_", "post_")) else FLOOR
        # Subpixel coordinates keep the projected lines thin enough to see the paint underneath.
        points = np.rint(16 * uv).astype(np.int32)
        cv2.polylines(overlay, [points], False, color, 2, cv2.LINE_AA, shift=4)
    return cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)


def comparison_image(frame, overlay, paths, holdout, dest, index):
    marked = overlay.copy()
    for p in holdout:
        point = tuple(np.rint(p["pixel"]).astype(int))
        cv2.circle(marked, point, 7, HOLDOUT, 2, cv2.LINE_AA)
    geometry = np.concatenate(list(paths.values()), axis=0)
    h, w = frame.shape[:2]
    x0, y0 = np.maximum(0, np.floor(geometry.min(axis=0) - [55, 45])).astype(int)
    x1, y1 = np.minimum([w, h], np.ceil(geometry.max(axis=0) + [55, 33])).astype(int)
    panels = []
    for im, label in ((frame, f"Broadcast | frame {index}"), (marked, "Projected court | held-out points in magenta")):
        crop = im[y0:y1, x0:x1]
        panel = cv2.copyMakeBorder(crop, 42, 0, 0, 0, cv2.BORDER_CONSTANT, value=(29, 32, 33))
        cv2.putText(panel, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .67, (242, 242, 242), 1, cv2.LINE_AA)
        panels.append(panel)
    if not cv2.imwrite(str(dest), cv2.hconcat(panels), [cv2.IMWRITE_JPEG_QUALITY, 96]):
        raise OSError(f"Cannot write {dest}")


def render(config, camera, out):
    src = ROOT / config["video"]
    if not src.is_file():
        raise FileNotFoundError(f"Run scripts/fetch_tracknetv3.py to obtain {src}")
    if hashlib.sha256(src.read_bytes()).hexdigest() != config["source_sha256"]:
        raise ValueError(f"Video does not match the annotated source: {src}")
    start = config["segment"]["start_frame"]
    stop = config["segment"]["end_frame_exclusive"]
    capture = cv2.VideoCapture(str(src))
    if not capture.isOpened():
        raise OSError(f"Cannot open {src}. Fetch the stage-one TrackNetV3 sample first.")
    out.mkdir(parents=True, exist_ok=True)
    paths = {name: camera.project(xyz) for name, xyz in court_wireframe().items()}
    w, h = camera.image_size
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        video_size = tuple(int(capture.get(p)) for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        if video_size != (w, h) or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Camera and video metadata disagree")
        if not 0 <= start <= config["calibration_frame"] < stop <= int(capture.get(cv2.CAP_PROP_FRAME_COUNT)):
            raise ValueError("Invalid camera segment or calibration frame")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        with NamedTemporaryFile(suffix=".mp4", dir=out, delete=False) as f:
            temp = Path(f.name)
        try:
            command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                       "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
                       "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
                       "-preset", "fast", "-movflags", "+faststart", str(temp)]
            with subprocess.Popen(command, stdin=subprocess.PIPE) as enc:
                try:
                    for i in range(start, stop):
                        ok, original = capture.read()
                        if not ok:
                            raise OSError(f"Could not decode source frame {i}")
                        overlay = draw_geometry(original, paths)
                        if i == config["calibration_frame"]:
                            comparison_image(original, overlay, paths,
                                             [p for p in config["landmarks"] if p["split"] == "holdout"],
                                             out / "reprojection.jpg", i)
                        cv2.rectangle(overlay, (0, 0), (w, 29), (28, 28, 31), -1)
                        cv2.putText(overlay, f"Court reprojection   |   f{i}   {i/fps:.2f}s",
                                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (239, 239, 239), 1, cv2.LINE_AA)
                        enc.stdin.write(overlay.tobytes())
                finally:
                    enc.stdin.close()
                if enc.wait():
                    raise OSError("FFmpeg could not encode the reprojection video")
            temp.replace(out / "reprojection.mp4")
        finally:
            temp.unlink(missing_ok=True)
    finally:
        capture.release()
    print(f"{stop-start} reprojected source frames -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage3-fuzhou.json")
    parser.add_argument("--camera", type=Path, default=ROOT / "results/stage3/camera.json")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/stage3")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    camera = Camera.from_dict(json.loads(args.camera.read_text(encoding="utf-8")))
    render(config, camera, args.out)


if __name__ == "__main__":
    main()
