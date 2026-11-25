"""Project simulated shuttle flights into a virtual or calibrated camera."""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure
import numpy as np

from .camera import Camera, look_at_camera
from .court import DOUBLES_HALF_WIDTH, HALF_LENGTH, court_wireframe
from .flight import DEFAULT_DRAG, Trajectory, simulate
from .plots import COLORS, SHOT_NAMES, STYLE, _text


ROOT = Path(__file__).resolve().parents[1]


def virtual_camera():
    return look_at_camera(center=(-14, -12, 8), target=(-1, 0, 1.85),
                          focal_px=1120, image_size=(1280, 720))


def _image_samples(camera, xyz):
    camera_xyz = (xyz - camera.center) @ camera.R.T
    depth = camera_xyz[:, 2]
    front = depth > 1e-9
    valid = front.copy()
    if camera.k1 < 0:
        # The first radial fold bounds the invertible branch of the model.
        radial_squared = np.sum(camera_xyz[front, :2] ** 2, axis=1) / depth[front] ** 2
        valid[front] = 1 + 3 * camera.k1 * radial_squared > 0
    uv = np.full((len(xyz), 2), np.nan)
    if valid.any():
        uv[valid] = camera.project(xyz[valid])
    width, height = camera.image_size
    inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return uv, depth, front, valid, inside


@dataclass
class ImageTrack:
    flight: Trajectory
    uv: np.ndarray
    depth: np.ndarray
    in_front: np.ndarray
    projection_valid: np.ndarray
    in_image: np.ndarray


def project_flight(camera, position, velocity, *, k=DEFAULT_DRAG, dt=0.01):
    """Return every physics sample, including its camera depth and image visibility."""
    flight = simulate(position, velocity, k=k, dt=dt)
    return ImageTrack(flight, *_image_samples(camera, flight.position))


def write_samples(path, tracks):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("shot", "t_s", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s",
                         "u_px", "v_px", "camera_depth_m", "in_front", "projection_valid", "in_image"))
        for name, track in tracks.items():
            f = track.flight
            for i, t in enumerate(f.time):
                uv = track.uv[i] if track.projection_valid[i] else ("", "")
                writer.writerow((name, t, *f.position[i], *f.velocity[i], *uv, track.depth[i],
                                 int(track.in_front[i]), int(track.projection_valid[i]), int(track.in_image[i])))


def render(path, camera, tracks):
    with mpl.rc_context(STYLE):
        fig = Figure(figsize=(10, 6.4), layout="constrained")
        ax = fig.subplots()
        width, height = camera.image_size
        ax.set(xlim=(-0.5, width-0.5), ylim=(height-0.5, -0.5), aspect="equal",
               xlabel="u (px)", ylabel="v (px)", facecolor="#f6f9fa")
        ax.set_title(_text("模拟球路的像素投影", "Simulated shuttle tracks in the image"),
                     loc="left", fontsize=17, weight="bold", color="#1a3341", pad=12)
        half_width = DOUBLES_HALF_WIDTH
        floor = np.array([(-HALF_LENGTH, -half_width, 0), (HALF_LENGTH, -half_width, 0),
                          (HALF_LENGTH, half_width, 0), (-HALF_LENGTH, half_width, 0)])
        # Sample even straight world segments so a loaded radial lens curves their image.
        boundary = np.concatenate([np.linspace(a, b, 65) for a, b in zip(floor, np.roll(floor, -1, axis=0))])
        floor_uv = _image_samples(camera, boundary)[0]
        if np.isfinite(floor_uv).all():
            ax.fill(*floor_uv.T, color="#e0eee7", zorder=0)
        for name, xyz in court_wireframe().items():
            vertical = name.startswith(("net", "post"))
            uv = _image_samples(camera, xyz)[0]
            ax.plot(*uv.T, color="#445f67" if vertical else "#819a8e", lw=1.4 if vertical else 0.9)
        for i, (name, track) in enumerate(tracks.items()):
            uv = track.uv
            color = COLORS[i % len(COLORS)]
            label = _text(*SHOT_NAMES.get(name, (name, name)))
            ax.plot(*uv.T, color=color, lw=2.2, label=label, zorder=4)
            ax.scatter(*uv[-1], s=28, color=color, edgecolor="white", lw=0.6, zorder=5)
        ax.legend(loc="upper right", frameon=True, framealpha=0.92)
        fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=Path, help="JSON camera from Camera.to_dict()")
    parser.add_argument("--position", type=float, nargs=3, metavar=("X", "Y", "Z"), help="launch position in m")
    parser.add_argument("--velocity", type=float, nargs=3, metavar=("VX", "VY", "VZ"), help="launch velocity in m/s")
    parser.add_argument("--k", type=float, help="quadratic drag coefficient in 1/m")
    parser.add_argument("--dt", type=float, help="physics step in seconds")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if (args.position is None) != (args.velocity is None):
        parser.error("--position and --velocity must be supplied together")
    custom = any(value is not None for value in (args.camera, args.position, args.k, args.dt))
    out = args.out or ROOT / ("artifacts/stage3/custom" if custom else "results/stage3")
    config = json.loads((ROOT / "configs/stage2-shots.json").read_text(encoding="utf-8"))
    shots = config["shots"] if args.position is None else {
        "custom": {"position_m": args.position, "velocity_m_s": args.velocity}
    }
    camera = Camera.from_dict(json.loads(args.camera.read_text(encoding="utf-8"))) if args.camera else virtual_camera()
    k = config["k_m_inv"] if args.k is None else args.k
    dt = config["dt_s"] if args.dt is None else args.dt
    try:
        tracks = {name: project_flight(camera, shot["position_m"], shot["velocity_m_s"], k=k, dt=dt)
                  for name, shot in shots.items()}
    except ValueError as exc:
        parser.error(str(exc))
    out.mkdir(parents=True, exist_ok=True)
    (out / "virtual-camera.json").write_text(json.dumps(camera.to_dict(), indent=2, allow_nan=False) + "\n",
                                              encoding="utf-8")
    write_samples(out / "virtual-trajectories.csv", tracks)
    render(out / "virtual-projection.png", camera, tracks)
    print(f"{len(tracks)} simulated flight(s), {sum(len(t.uv) for t in tracks.values())} samples -> {out}")


if __name__ == "__main__":
    main()
