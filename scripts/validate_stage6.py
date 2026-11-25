"""Audit broadcast source frames, blind-fit isolation, reference integration and replay."""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from badminton.real_study import STATE, infer, inputs
from badminton.render_real import group_rows
from validate_stage4 import project, reference_flight


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate(study, media):
    shots, _ = inputs(study)
    summary = read_json(study / "study.json")
    selection = read_json(ROOT / "configs/stage6-fuzhou.json")["rallies"]
    manifest = read_json(ROOT / "artifacts/stage1/manifest.json")
    recorded = read_csv(study / "fits.csv")
    fits = {r["shot_id"]: r for r in recorded}
    tracks = group_rows(study / "trajectories.csv")
    obs = group_rows(study / "observations.csv")
    residuals = group_rows(study / "residuals.csv")
    require(len(fits) == len(shots) == len(recorded) and fits.keys() == tracks.keys() == obs.keys() ==
            residuals.keys() == {s.meta["shot_id"] for s in shots}, "Inconsistent set of shots")
    replay = cv2.VideoCapture(str(media / "reprojection.mp4"))
    require(replay.isOpened(), "Could not open the replay")
    source_pixels = {}
    maximum = defaultdict(float)
    counts, seen, held_errors = Counter(), set(), defaultdict(list)
    try:
        for shot in shots:
            meta, camera = shot.meta, shot.camera
            sid, rally = meta["shot_id"], meta["rally_id"]
            fit, points, audit = fits[sid], tracks[sid], obs[sid]
            require(meta["split"] == selection[rally]["split"] and
                    str(meta["shot_index"]) in selection[rally]["shots"], f"{sid}: split or shot changed")
            src = ROOT / meta["source_video"]
            if rally not in source_pixels:
                relative = src.relative_to(ROOT / "data/raw").as_posix()
                sha = hashlib.sha256(src.read_bytes()).hexdigest()
                require(sha == manifest["rallies"][rally]["inputs"][relative], f"{rally}: source hash changed")
                source_pixels[rally] = {int(r["frame_index"]): r for r in read_csv(
                    ROOT / "artifacts/stage1" / rally / "frames.csv")}
            cap = cv2.VideoCapture(str(src))
            try:
                require(cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) == meta["fps"] ==
                        replay.get(cv2.CAP_PROP_FPS), f"{sid}: invalid video or frame rate")
                require(cap.set(cv2.CAP_PROP_POS_FRAMES, int(shot.frame[0])), f"{sid}: seek failed")
                require(np.array_equal([int(r["frame_index"]) for r in points], shot.frame),
                        f"{sid}: trajectory frame indices changed")
                theta = np.array([float(fit[k]) for k in STATE])
                k_drag = float(fit["k_m_inv"])
                solution, ground_time, _, _ = reference_flight(theta, k_drag)
                require(ground_time >= shot.t[-1], f"{sid}: segment extends past modeled ground contact")
                reference = solution(shot.t).T
                states = np.array([[float(r[k]) for k in STATE] for r in points])
                np.testing.assert_allclose(states, reference, atol=.00004, rtol=0)
                ref_uv = project(camera, reference[:, :3])
                fitted_uv = np.array([[float(r[k]) for k in ("u_fit_px", "v_fit_px")] for r in points])
                np.testing.assert_allclose(ref_uv, fitted_uv, atol=.001, rtol=0)
                maximum["reference_state_abs"] = max(maximum["reference_state_abs"], float(abs(states-reference).max()))
                maximum["reference_projection_px"] = max(maximum["reference_projection_px"],
                                                         float(np.linalg.norm(ref_uv-fitted_uv, axis=1).max()))
                exported = {int(r["frame_index"]): r for r in residuals[sid]}
                require(len(exported) == len(residuals[sid]), f"{sid}: duplicate residual")
                per_role = defaultdict(list)
                for i, (frame, row) in enumerate(zip(shot.frame, audit)):
                    require((rally, frame) not in seen, f"{sid}: duplicate source frame")
                    seen.add((rally, frame))
                    np.testing.assert_allclose([float(row["t_s"]), float(points[i]["t_s"])], shot.t[i], atol=1e-7)
                    upstream = source_pixels[rally][frame]
                    if upstream["is_missing"] == "1":
                        require(row["role"] == "missing" and not row["u_px"] and not row["v_px"],
                                f"{sid}: filled missing observation at {frame}")
                    elif row["role"] != "excluded":
                        expected_role = "holdout" if i % 4 == 2 else "fit"
                        require(row["role"] == expected_role, f"{sid}: mask changed at {frame}")
                        np.testing.assert_allclose(shot.uv[i], [float(upstream["x_px"]), float(upstream["y_px"])],
                                                   atol=1e-8, rtol=0)
                    if row["role"] in ("fit", "holdout"):
                        error = np.linalg.norm(ref_uv[i] - shot.uv[i])
                        record = exported.pop(frame)
                        require(record["role"] == row["role"], f"{sid}: residual subset changed")
                        np.testing.assert_allclose(error, float(record["error_px"]), atol=.001)
                        per_role[row["role"]].append(error)
                        if row["role"] == "holdout":
                            held_errors[meta["split"]].append(error)
                    counts[row["role"]] += 1
                    ok, original = cap.read()
                    played, rendered = replay.read()
                    require(ok and played and original.shape == rendered.shape ==
                            (camera.image_size[1], camera.image_size[0], 3), f"{sid}: replay mismatch at {frame}")
                    # Exclude the title bar. A difference cap allows thin overlays and H.264 recompression.
                    delta = cv2.absdiff(original[36::3, ::3], rendered[36::3, ::3])
                    mae = float(np.mean(delta))
                    require(mae < 5, f"{sid}: source and replay differ at {frame}, mean {mae:.2f}")
                    maximum["replay_frame_mean_abs_8bit"] = max(maximum["replay_frame_mean_abs_8bit"], mae)
                require(not exported, f"{sid}: unexpected residuals")
                for role, errors in per_role.items():
                    np.testing.assert_allclose(np.sqrt(np.mean(np.square(errors))), float(fit[role+"_rmse_px"]), atol=.001)
            finally:
                cap.release()
            if meta["split"] == "test":
                # Remove every held-out pixel. The model fit must reproduce the saved initial state and drag.
                masked = shot.uv.copy()
                masked[shot.role != "fit"] = np.nan
                model, result = infer(replace(shot, uv=masked), summary["primary_model"])
                refit, refit_k = model.unpack(result.x)
                np.testing.assert_allclose(refit, theta, atol=1e-5, rtol=0)
                np.testing.assert_allclose(refit_k, k_drag, atol=1e-7, rtol=0)
        require(not replay.read()[0], "Unexpected trailing replay frames")
    finally:
        replay.release()
    require(len(shots) == sum(len(r["shots"]) for r in selection.values()), "Frozen selections missing")
    for split, errors in held_errors.items():
        expected = summary["summary"][split]
        require(len(errors) == expected["heldout_pixels"], f"{split}: held-out count changed")
        np.testing.assert_allclose(np.sqrt(np.mean(np.square(errors))), expected["heldout_rmse_px"], atol=.001)
    return {"passed": True, "source_videos_hash_verified": len(source_pixels), "strokes": len(shots),
            "decoded_source_and_replay_frames": len(seen), "observations_by_role": dict(counts),
            "test_strokes_refit_with_heldout_pixels_removed": sum(s.meta["split"] == "test" for s in shots),
            "max_differences": dict(maximum)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "results/stage6")
    parser.add_argument("--media", type=Path, default=ROOT / "artifacts/stage6")
    args = parser.parse_args()
    report = validate(args.study, args.media)
    (args.study / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f'Validated {report["strokes"]} strokes and {report["decoded_source_and_replay_frames"]} replay frames')
