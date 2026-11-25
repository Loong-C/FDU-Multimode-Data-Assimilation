"""Paired single-camera experiments for evaluating single-shot reconstruction.

Frame zero is the known launch time. The synthetic detector reports pixels at
25 fps, while the independent reference flight retains the RK4 grid, apex and
landing. Only times, uv, camera and the stated k / noise model go to the fit.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .camera import Camera, look_at_camera
from .forward import project_flight, virtual_camera


ROOT = Path(__file__).resolve().parents[1]
FPS = 25
TRUTH_STEP_S = 0.004
SEEDS = (401, 402, 403)


@dataclass(frozen=True)
class Case:
    name: str
    shot: str
    view: str
    condition: str
    seed: int
    camera: Camera
    times: np.ndarray
    uv: np.ndarray
    k: float
    sigma_px: float
    robust: bool
    frame_index: np.ndarray
    n_video_frames: int
    n_visible_frames: int
    noise_std_px: float
    # The following fields are for checking the fit, never for initializing it.
    clean_uv: np.ndarray
    outlier_mask: np.ndarray
    truth_time: np.ndarray
    truth_xyz: np.ndarray
    truth_theta: np.ndarray


def _make_case(shot, view, condition, seed, camera, track, k):
    flight = track.flight
    # Localized apex / landing samples belong to the truth, not the detector.
    frame_no = np.rint(flight.time * FPS).astype(int)
    on_grid = np.isclose(flight.time, frame_no / FPS, atol=1e-11, rtol=0)
    visible = track.in_image[on_grid]
    frames = frame_no[on_grid]
    clean = track.uv[on_grid]
    noise_std = 0.0 if condition == "clean" else 3.0 if condition == "high_noise" else 1.0

    # Index noise by original frame so crops, masks, views and robust/nonrobust
    # fits use the same random pixel shifts at the same physical time.
    noise_seed, mask_seed, outlier_seed = np.random.SeedSequence(seed).spawn(3)
    uv = clean + noise_std * np.random.default_rng(noise_seed).normal(size=clean.shape)
    kept = visible.copy()
    outliers = np.zeros(len(kept), dtype=bool)
    if condition == "dropout":
        # Match the 15-frame occlusion in count and preserve both endpoints.
        eligible = np.flatnonzero(kept)[1:-1]
        removed = np.random.default_rng(mask_seed).choice(eligible, size=15, replace=False)
        kept[removed] = False
    elif condition == "occlusion":
        kept &= (frames < 15) | (frames >= 30)
    elif condition == "short":
        kept &= frames < 4
    elif condition == "medium":
        kept &= frames < 12
    elif condition in ("outliers", "outliers_robust"):
        rng = np.random.default_rng(outlier_seed)
        selected = rng.choice(np.flatnonzero(kept), size=6, replace=False)
        angle = rng.uniform(0, 2 * np.pi, size=len(selected))
        offsets = rng.uniform(35, 75, size=(len(selected), 1)) * np.column_stack((np.cos(angle), np.sin(angle)))
        trial = uv[selected] + offsets
        # Reflect a displacement at the sensor edge to keep the false detection in view.
        offsets[(trial < 0) | (trial >= camera.image_size)] *= -1
        uv[selected] += offsets
        outliers[selected] = True
    kept &= np.all((uv >= 0) & (uv < camera.image_size), axis=1)
    if kept.sum() < 4:
        raise ValueError(f"Fewer than four detector frames in {shot}/{view}/{condition}/{seed}")
    return Case(
        name=f"{shot}_{view}_{condition}_{seed}", shot=shot, view=view, condition=condition,
        seed=seed, camera=camera, times=frames[kept] / FPS, uv=uv[kept], k=k,
        # A small positive numerical scale permits a weighted fit of exact pixels.
        sigma_px=max(noise_std, 0.01), robust=(condition == "outliers_robust"),
        frame_index=frames[kept], n_video_frames=len(frames), n_visible_frames=int(visible.sum()),
        noise_std_px=noise_std, clean_uv=clean[kept], outlier_mask=outliers[kept],
        truth_time=flight.time, truth_xyz=flight.position,
        truth_theta=np.r_[flight.position[0], flight.velocity[0]],
    )


def iter_cases():
    """Yield the fixed study matrix, including three paired repetitions of each clear stressor."""
    config = json.loads((ROOT / "configs/stage2-shots.json").read_text(encoding="utf-8"))
    shots, k = config["shots"], config["k_m_inv"]
    oblique = virtual_camera()
    # Rotate the oblique camera about its target's vertical axis, retaining
    # distance, elevation, focal length and framing of the target.
    target = np.array([-1.0, 0.0, 1.85])
    radius = np.linalg.norm(oblique.center[:2] - target[:2])
    endline = look_at_camera((target[0] - radius, target[1], oblique.center[2]),
                             target, oblique.focal_px, oblique.image_size)
    fuzhou = Camera.from_dict(json.loads((ROOT / "results/stage3/camera.json").read_text(encoding="utf-8")))
    cameras = {"oblique": oblique, "endline": endline, "fuzhou": fuzhou}
    matrix = [(shot, "oblique", "clean", 0) for shot in shots]
    matrix.extend((shot, "oblique", "noise", seed) for shot in shots for seed in SEEDS)
    for seed in SEEDS:
        matrix.extend(("clear", "oblique", condition, seed) for condition in
                      ("high_noise", "dropout", "occlusion", "outliers", "outliers_robust",
                       "short", "medium", "alternate"))
        matrix.extend(("clear", view, "noise", seed) for view in ("endline", "fuzhou"))
    tracks = {}
    for shot, view, condition, seed in matrix:
        alternate = condition == "alternate"
        key = (shot, view, alternate)
        if key not in tracks:
            launch = shots[shot]
            if alternate:
                launch = {"position_m": [-4.5, 1.1, 2.15], "velocity_m_s": [28.0, -5.0, 21.0]}
            tracks[key] = project_flight(cameras[view], launch["position_m"], launch["velocity_m_s"],
                                         k=k, dt=TRUTH_STEP_S)
        yield _make_case(shot, view, condition, seed, cameras[view], tracks[key], k)
