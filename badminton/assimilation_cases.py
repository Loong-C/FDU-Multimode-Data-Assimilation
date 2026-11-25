"""Paired noisy camera and auxiliary measurements for stage-five simulation."""

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .camera import Camera
from .flight import Trajectory, simulate
from .forward import virtual_camera


ROOT = Path(__file__).resolve().parents[1]
# Four fixed physical truths: one matches the fixed-drag model exactly.
TRUE_K = {"clear": 0.175, "lift": 0.280, "drop": 0.217, "drive": 0.250}
FPS = 25
SHORT_FRAMES = 6
# Same mask and measured values for every analysis of a given flight.
# prefix controls the viewing window; tuple controls the observation model.
SETTINGS = {
    "short_broad_k": ((), None, 0.90),
    "short_pixels": ((), None, 0.35),
    "short_player": (("player_xy",), None, 0.35),
    "short_height": (("height",), None, 0.35),
    "short_launch": (("player_xy", "height"), None, 0.35),
    "short_all": (("player_xy", "height", "landing_xy", "landing_time"), None, 0.35),
    "full_pixels": ((), None, 0.35),
    "full_launch": (("player_xy", "height"), None, 0.35),
    "full_all": (("player_xy", "height", "landing_xy", "landing_time"), None, 0.35),
    "full_fixed": (("player_xy", "height", "landing_xy", "landing_time"), 0.217, 0.35),
    "full_ballistic": (("player_xy", "height", "landing_xy", "landing_time"), 0.0, 0.35),
}


@dataclass(frozen=True)
class Case:
    shot: str
    seed: int
    camera: Camera
    times: np.ndarray
    uv: np.ndarray
    R: np.ndarray
    constraints: dict
    # Available only to experiment scoring and visualization.
    truth_theta: np.ndarray
    truth_k: float
    flight: Trajectory


@lru_cache(maxsize=len(TRUE_K))
def _reference(shot):
    config = json.loads((ROOT / "configs/stage2-shots.json").read_text(encoding="utf-8"))
    initial = config["shots"][shot]
    state = np.r_[initial["position_m"], initial["velocity_m_s"]]
    flight = simulate(state[:3], state[3:], k=TRUE_K[shot], dt=0.002)
    camera = virtual_camera()
    # Use complete 25 fps frames with an observable airborne shuttle.
    frames = np.arange(int(np.floor(flight.time[-1]*FPS)) + 1)
    times = frames / FPS
    xyz = np.column_stack([np.interp(times, flight.time, axis) for axis in flight.position.T])
    pixels = camera.project(xyz)
    return camera, times, pixels, state, flight


def make_case(shot, seed):
    camera, times, clean, state, flight = _reference(shot)
    camera_seed, auxiliary_seed = np.random.SeedSequence([seed, list(TRUE_K).index(shot)]).spawn(2)
    rng = np.random.default_rng(camera_seed)
    # Every fourth frame has a less certain detection; u and v errors correlate.
    scale = np.where(np.arange(len(times)) % 4 == 3, 1.6, 1.0)
    R = scale[:, None, None]**2 * np.array([[1.0, 0.30], [0.30, 1.44]])
    uv = clean + np.einsum("nij,nj->ni", np.linalg.cholesky(R), rng.normal(size=clean.shape))
    keep = np.all((clean >= 0) & (clean < camera.image_size) & (uv >= 0) & (uv < camera.image_size), axis=1)
    # A player-position observation proxies the horizontal contact position.
    # Its 0.35 m scale includes player-to-contact offset and localization uncertainty.
    aux = np.random.default_rng(auxiliary_seed)
    constraints = {"player_xy": (state[:2] + aux.normal(0, 0.35, 2), 0.35),
                   "height": (state[2] + aux.normal(0, 0.20), 0.20),
                   "landing_xy": (flight.position[-1, :2] + aux.normal(0, 0.18, 2), 0.18),
                   "landing_time": (flight.time[-1] + aux.normal(0, 0.035), 0.035)}
    return Case(shot, seed, camera, times[keep], uv[keep], R[keep], constraints, state, TRUE_K[shot], flight)


def make_model(case, setting):
    from .posterior import PosteriorModel

    keys, fixed, log_k_sd = SETTINGS[setting]
    window = case.times < SHORT_FRAMES / FPS if setting.startswith("short_") else slice(None)
    return PosteriorModel(case.camera, case.times[window], case.uv[window], R=case.R[window],
                          fixed_k=fixed, log_k_sd=log_k_sd, constraints={k: case.constraints[k] for k in keys})
