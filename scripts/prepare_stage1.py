"""Build the aligned Fuzhou singles sample from the pinned local source files."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import deque
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import cv2
import imageio_ffmpeg
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FRAME_FIELDS = (
    "match_id", "rally_id", "frame_index", "source_frame_index", "timestamp_s", "x_px", "y_px",
    "confidence", "is_missing", "missing_reason", "source_visibility", "observation_source",
    "observation_origin", "shot_index", "shot_type",
)
EVENT_FIELDS = (
    "match_id", "rally_id", "shot_index", "frame_index", "timestamp_s", "player", "shot_type",
    "shot_type_en", "event_source", "event_origin", "shuttleset_rally", "shuttleset_frame",
    "mapped_frame", "timing_method", "player_x_px", "player_y_px", "opponent_x_px",
    "opponent_y_px",
)
TYPES = {
    "發短球": "short_service", "發長球": "long_service", "挑球": "lob",
    "長球": "clear", "殺球": "smash", "點扣": "wrist_smash", "切球": "drop",
    "過度切球": "passive_drop", "過渡切球": "passive_drop", "放小球": "net_shot",
    "擋小球": "return_net", "勾球": "cross_court_net_shot", "推球": "push",
    "後場抽平球": "back_court_drive", "平球": "drive", "撲球": "rush",
    "防守回抽": "defensive_return_drive", "防守回挑": "defensive_return_lob",
}


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows, fieldnames=None):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def publish_output(staging, out):
    """Publish verified files with the destination directory's Windows access rules."""
    files = [path for path in staging.rglob("*") if path.is_file()]
    for path in sorted(files, key=lambda p: p.name == "manifest.json"):
        target = out / path.relative_to(staging)
        target.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temp = Path(stream.name)
        try:
            shutil.copyfile(path, temp)
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)


def build_events(match_id, rally, settings, stroke_file, fps):
    _, a, b = map(int, rally.split("_"))
    rows = [r for r in read_rows(stroke_file)
            if (int(r["roundscore_A"]), int(r["roundscore_B"])) == (a, b)]
    if not rows or len({r["rally"] for r in rows}) != 1:
        raise ValueError(f"{rally}: score does not identify exactly one annotated rally")

    reviewed = settings["reviewed_contacts"]
    events = []
    for shot, row in enumerate(rows, 1):
        broadcast_frame = int(float(row["frame_num"]))
        mapped = broadcast_frame - settings["source_frame_at_local_zero"]
        frame = reviewed.get(str(shot), mapped)
        if (float(row["ball_round"]) != shot or not 0 <= frame < settings["end_frame_exclusive"]
                or (events and frame <= events[-1]["frame_index"])):
            raise ValueError(f"{rally}: invalid event sequence at stroke {shot}")
        events.append(dict(
            match_id=match_id, rally_id=rally, shot_index=shot, frame_index=frame, timestamp_s=f"{frame/fps:.6f}",
            player=row["player"], shot_type=row["type"], shot_type_en=TYPES[row["type"]],
            event_source="ShuttleSet", event_origin="human_annotation",
            shuttleset_rally=row["rally"], shuttleset_frame=broadcast_frame, mapped_frame=mapped,
            timing_method="video_review" if str(shot) in reviewed else "rally_offset",
            player_x_px=row["player_location_x"], player_y_px=row["player_location_y"],
            opponent_x_px=row["opponent_location_x"], opponent_y_px=row["opponent_location_y"],
        ))
    return events


def build_frames(match_id, rally, label_file, settings, width, height, fps, events):
    labels = {}
    for row in read_rows(label_file):
        index, visible, x, y = (int(row[k]) for k in ("Frame", "Visibility", "X", "Y"))
        if not 0 <= index < settings["source_frames"] or index in labels:
            raise ValueError(f"{label_file}: duplicate or out-of-range index {index}")
        if visible not in (0, 1) or (visible and not (0 <= x < width and 0 <= y < height)):
            raise ValueError(f"{label_file}: invalid observation at {index}")
        labels[index] = (visible, x, y)

    hits = {e["frame_index"]: e for e in events}
    frames = []
    for i in range(settings["end_frame_exclusive"]):
        visible, x, y = labels.get(i, (0, None, None))
        reason = "" if visible else ("not_visible" if i in labels else "no_label_row")
        event = hits.get(i, {})
        frames.append(dict(
            match_id=match_id, rally_id=rally, frame_index=i, source_frame_index=i, timestamp_s=f"{i/fps:.6f}",
            x_px=x if visible else "", y_px=y if visible else "", confidence="",
            is_missing=int(not visible), missing_reason=reason,
            source_visibility=visible if i in labels else "",
            observation_source="Shuttlecock Trajectory Dataset" if i in labels else "",
            observation_origin="human_annotation" if i in labels else "",
            shot_index=event.get("shot_index", ""), shot_type=event.get("shot_type", ""),
        ))
    return frames


def build_court(path, config):
    [row] = [r for r in read_rows(path) if r["video"] == config["match"]]
    corners = {name: [float(row[f"{name}_x"]), float(row[f"{name}_y"])]
               for name in ("upleft", "upright", "downright", "downleft")}
    return dict(image_size_px=[config["width"], config["height"]], court_corners_px=corners,
                court_source="ShuttleSet/homography.csv", net_post_tops_px=config["net_post_tops_px"],
                net_source=config["net_reference"])


def extract_video(source, target, first, stop, fps):
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-map", "0:v:0", "-vf",
        f"trim=start_frame={first}:end_frame={stop},setpts=N/({fps}*TB)",
        "-r", str(fps), "-fps_mode", "cfr", "-an", "-c:v", "libx264", "-crf", "15", "-preset", "fast",
        "-movflags", "+faststart", str(target),
    ], check=True)


def draw_overlay(image, row, trail, court, events, fps):
    corners = np.rint(list(court["court_corners_px"].values())).astype(np.int32)
    cv2.polylines(image, [corners], True, (255, 218, 36), 2, cv2.LINE_AA)
    for point in corners:
        cv2.drawMarker(image, tuple(point), (255, 218, 36), cv2.MARKER_CROSS, 14, 2)
    for point in court["net_post_tops_px"].values():
        cv2.drawMarker(image, tuple(point), (224, 85, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
    if row["is_missing"]:
        trail.clear()
    else:
        point = (row["x_px"], row["y_px"])
        trail.append(point)
        cv2.polylines(image, [np.asarray(trail, dtype=np.int32)], False, (24, 196, 255), 2)
        cv2.circle(image, point, 10, (0, 238, 255), 2, cv2.LINE_AA)
    i = row["frame_index"]
    cv2.rectangle(image, (0, 0), (image.shape[1], 32), (25, 25, 25), -1)
    message = f"{row['rally_id']}   f{i}   {i/fps:.2f}s"
    if row["is_missing"]:
        message += f"   |   {row['missing_reason']}"
    recent = [e for e in events if 0 <= i - e["frame_index"] < 6]
    if recent:
        hit = recent[-1]
        message += f"   |   #{hit['shot_index']} {hit['player']} {hit['shot_type_en']} @ f{hit['frame_index']}"
    cv2.putText(image, message, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .65,
                (245, 245, 245), 1, cv2.LINE_AA)


def render_overlay(source, dest, frames, events, court, config, settings):
    width, height = config["width"], config["height"]
    fps = config["fps"]
    capture = cv2.VideoCapture(str(source))
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
               "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
               "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-crf", "19",
               "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)]
    try:
        if (not capture.isOpened() or abs(capture.get(cv2.CAP_PROP_FPS)-fps) > .001
                or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != settings["source_frames"]):
            raise ValueError(f"{source}: unexpected video properties")
        trail = deque(maxlen=10)
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            try:
                for row in frames:
                    ok, image = capture.read()
                    if not ok or image.shape[:2] != (height, width):
                        raise ValueError(f"{source}: cannot read frame {row['frame_index']}")
                    draw_overlay(image, row, trail, court, events, fps)
                    encoder.stdin.write(image.tobytes())
            finally:
                encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError(f"Overlay encoder failed for {source}")
    finally:
        capture.release()


def prepare(raw, out, config):
    tracknet = raw / "tracknetv3" / config["tracknet_match"]
    shuttle = raw / "shuttleset"
    court = build_court(shuttle / "homography.csv", config)
    write_json(out / "court.json", court)
    manifest = dict(schema_version=1, match=config["match"], players={"A": "Kento Momota",
                    "B": "Chou Tien Chen"}, fps=config["fps"], width=config["width"],
                    height=config["height"], rallies={})
    for rally, settings in config["rallies"].items():
        dest = out / rally
        (dest / "shots").mkdir(parents=True)
        source = tracknet / "video" / f"{rally}.mp4"
        labels = tracknet / "csv" / f"{rally}_ball.csv"
        stroke_file = shuttle / f"set{rally.split('_')[0]}.csv"
        events = build_events(config["match"], rally, settings, stroke_file, config["fps"])
        frames = build_frames(config["match"], rally, labels, settings, config["width"], config["height"],
                              config["fps"], events)
        write_rows(dest / "events.csv", events, EVENT_FIELDS)
        write_rows(dest / "frames.csv", frames, FRAME_FIELDS)
        if len(frames) == settings["source_frames"]:
            shutil.copyfile(source, dest / "rally.mp4")
        else:
            extract_video(source, dest / "rally.mp4", 0, len(frames), config["fps"])
        render_overlay(source, dest / "overlay.mp4", frames, events, court, config, settings)

        shots = []
        for number in settings["export_strokes"]:
            hit, next_hit = events[number-1:number+1]
            first, stop = hit["frame_index"], next_hit["frame_index"]
            name = f"shots/{number:02}_{hit['shot_type_en']}.mp4"
            extract_video(source, dest / name, first, stop, config["fps"])
            shots.append(dict(shot_index=number, shot_type=hit["shot_type"], player=hit["player"],
                              start_frame=first, end_frame_exclusive=stop, file=name))
        write_rows(dest / "shots.csv", shots)
        missing = sum(row["is_missing"] for row in frames)
        manifest["rallies"][rally] = dict(
            start_frame=0, end_frame_exclusive=len(frames), source_frames=settings["source_frames"],
            source_frame_at_local_zero=settings["source_frame_at_local_zero"], visible=len(frames)-missing,
            not_visible=sum(row["missing_reason"] == "not_visible" for row in frames),
            no_label_row=sum(row["missing_reason"] == "no_label_row" for row in frames),
            events=len(events), shot_clips=len(shots),
            inputs={p.relative_to(raw).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (source, labels, stroke_file, shuttle / "homography.csv")})
    preview = config["overlay_preview"]
    clip = cv2.VideoCapture(str(out / preview["rally"] / "overlay.mp4"))
    try:
        clip.set(cv2.CAP_PROP_POS_FRAMES, preview["frame"])
        ok, image = clip.read()
        if not ok or not cv2.imwrite(str(out / "alignment.jpg"), image):
            raise OSError("Unable to extract the alignment preview")
    finally:
        clip.release()
    write_json(out / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/stage1")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/fuzhou2019.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as tmp:
        staging = Path(tmp)
        prepare(args.raw, staging, config)
        publish_output(staging, args.out)
    print(args.out / "manifest.json")


if __name__ == "__main__":
    main()
