"""Normalize the local BFMD example and three MonoTrack input rallies."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2

from fetch_monotrack import DIGESTS, SAMPLE
from prepare_stage1 import EVENT_FIELDS, FRAME_FIELDS, ROOT, build_frames, publish_output, write_json, write_rows


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def video_info(video):
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open {video}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if image.shape[:2] != (height, width):
                raise ValueError(f"Resolution change in {video} at frame {count}")
            count += 1
        if count == 0 or count != declared or fps <= 0:
            raise ValueError(f"Unexpected frame count or timebase in {video}")
        return dict(width=width, height=height, fps=fps, frames=count)
    finally:
        capture.release()


def export_rally(dest, info, frames, events):
    if len(frames) != info["frames"]:
        raise ValueError(f"{dest.name}: frames table does not cover video")
    visible = sum(1 for row in frames if not row["is_missing"])
    dest.mkdir(parents=True, exist_ok=True)
    write_rows(dest / "frames.csv", frames, FRAME_FIELDS)
    write_rows(dest / "events.csv", events, EVENT_FIELDS)
    return dict(**info, visible=visible, missing=len(frames)-visible, events=len(events))


def prepare_bfmd(raw, out):
    match_id = "BFMD/Malaysia_Open_2025_Naraoka_Antonsen"
    rally = "malaysia_game1_rally0"
    match = raw / "bfmd/malaysia"
    video = raw / "bfmd/demo_petronas.mp4"
    info = video_info(video)
    interval = read_json(match / "metadata.json")["games"]["game1"]["rallies"][0]
    start, stop = interval["start"], interval["end"]
    if (start, stop) != (10317, 10477) or stop - start != info["frames"]:
        raise ValueError("BFMD example and matched source interval disagree")

    timeline = read_json(match / "shot_type.json")
    labels = defaultdict(set)
    for annotation in timeline["annotations"]:
        for item in annotation["result"]:
            if item.get("origin") != "manual":
                continue
            value = item["value"]
            for span in value["ranges"]:
                frame = span["start"]
                if start <= frame < stop and frame == span["end"]:
                    labels[frame].update(value["timelinelabels"])

    fps = info["fps"]
    hits = {}
    for frame, tags in sorted(labels.items()):
        if "hit" in tags:
            [shot_type] = tags - {"hit"}
            local = frame - start
            hits[frame] = dict(
                match_id=match_id, rally_id=rally, shot_index=len(hits)+1, frame_index=local,
                timestamp_s=f"{local/fps:.6f}", player="", shot_type=shot_type,
                shot_type_en=shot_type, event_source="BFMD/shot_type.json",
                event_origin="human_annotation", mapped_frame=local,
                timing_method="broadcast_frame_minus_clip_start",
            )
    if len(hits) != 5:
        raise ValueError("BFMD first rally should have five manually typed contacts")

    shuttle = read_json(match / "shuttle.json")
    [prediction] = shuttle["predictions"]
    [track] = prediction["result"]
    boxes = {}
    for box in track["value"]["sequence"]:
        frame = box["frame"]
        if start <= frame < stop:
            if frame in boxes:
                raise ValueError(f"Duplicate BFMD model record at {frame}")
            boxes[frame] = box

    width, height = info["width"], info["height"]
    frames = []
    for absolute in range(start, stop):
        box = boxes.get(absolute)
        enabled = box is not None and box["enabled"]
        local = absolute - start
        hit = hits.get(absolute, {})
        x = y = reason = ""
        if enabled:
            x = (box["x"] + box["width"]/2) * width/100
            y = (box["y"] + box["height"]/2) * height/100
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"BFMD center outside image at {absolute}")
            x, y = f"{x:.6f}", f"{y:.6f}"
        else:
            reason = "no_prediction_row" if box is None else "prediction_disabled"
        frames.append(dict(
            match_id=match_id, rally_id=rally, frame_index=local, source_frame_index=absolute,
            timestamp_s=f"{local/fps:.6f}", x_px=x, y_px=y, confidence="",
            is_missing=int(not enabled), missing_reason=reason, source_visibility="",
            observation_source="BFMD/shuttle.json:tracknet" if box else "",
            observation_origin="model_prediction" if box else "",
            shot_index=hit.get("shot_index", ""), shot_type=hit.get("shot_type", ""),
        ))
    summary = export_rally(out / "bfmd" / rally, info, frames, list(hits.values()))
    if summary["visible"] != 143 or summary["missing"] != 17:
        raise ValueError("Unexpected BFMD detection count")
    summary.update(
        input_video=video.relative_to(raw).as_posix(), event_coverage="human_annotated",
        source_frame_axis="full_match", source_start=start, source_end_exclusive=stop,
        model_box_center_formula="x_px=(x+width/2)*image_width/100; y_px=(y+height/2)*image_height/100",
        source_frame_indexing=shuttle["_frame_indexing"], confidence_available=False,
        source_visibility_available=False, player_identity_used=False,
        auxiliary_net_hit_source_frames=[frame for frame, tags in labels.items() if "net_hit" in tags],
    )
    return {rally: summary}


def prepare_monotrack(raw, out):
    directory = raw / "monotrack" / SAMPLE
    result = {}
    for name in sorted(DIGESTS):
        if not name.endswith(".mp4"):
            continue
        video = directory / name
        rally = video.stem
        info = video_info(video)
        labels = directory / "csv" / f"{rally}_ball.csv"
        bounds = dict(source_frames=info["frames"], end_frame_exclusive=info["frames"])
        frames = build_frames(SAMPLE.as_posix(), rally, labels, bounds, info["width"], info["height"], info["fps"], [])
        if any(row["missing_reason"] == "no_label_row" for row in frames):
            raise ValueError(f"Incomplete MonoTrack input labels in {rally}")
        summary = export_rally(out / "monotrack" / rally, info, frames, [])
        summary.update(input_video=video.relative_to(raw).as_posix(),
                       input_labels=labels.relative_to(raw).as_posix(),
                       source_frame_axis="rally_clip", event_coverage="not_available_in_base_dataset")
        result[rally] = summary
    if sum(r["frames"] for r in result.values()) != 1952 or sum(r["visible"] for r in result.values()) != 1795:
        raise ValueError("Unexpected MonoTrack base sample counts")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/stage1_supplements")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as tmp:
        staging = Path(tmp)
        manifest = dict(schema_version=1, bfmd=prepare_bfmd(args.raw, staging),
                        monotrack=prepare_monotrack(args.raw, staging))
        write_json(staging / "manifest.json", manifest)
        publish_output(staging, args.out)
    print(args.out / "manifest.json")


if __name__ == "__main__":
    main()
