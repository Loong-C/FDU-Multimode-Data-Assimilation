"""Reconstruct a single free flight using a calibrated camera and timed image centres."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .camera import Camera
from .flight import DEFAULT_DRAG, simulate
from .inverse import fit_track, predict_positions

ROOT = Path(__file__).resolve().parents[1]


def read_observations(path, track=None):
    """Read visible detections, preserving time since launch across missing rows."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"t_s", "u_px", "v_px"}.issubset(reader.fieldnames or []):
            raise ValueError("Observation CSV needs t_s,u_px,v_px")
        key = next((k for k in ("case", "shot") if k in reader.fieldnames), None)
        if track is not None and key is None:
            raise ValueError("--track requires a case or shot column")
        rows = [r for r in reader if track is None or r[key] == track]
    if key and track is None and len({r[key] for r in rows}) > 1:
        raise ValueError("Select one case or shot using --track")
    if not rows:
        raise ValueError("No matching observations")
    times, pixels = [], []
    for row in rows:
        if row.get("in_image", "1").strip().lower() in ("0", "false"):
            continue
        uv = np.array([float(row[k].strip() or "nan") for k in ("u_px", "v_px")])
        if np.isnan(uv).all():
            continue
        if not np.isfinite(uv).all():
            raise ValueError(f"Incomplete or non-finite pixel pair at t={row['t_s']}")
        times.append(float(row["t_s"]))
        pixels.append(uv)
    return np.asarray(times), np.asarray(pixels).reshape(-1, 2)


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True, help="CSV with t_s (since launch), u_px, v_px")
    parser.add_argument("--track", help="value of the case or shot column in a multi-track CSV")
    parser.add_argument("--camera", type=Path, required=True, help="stage-three camera JSON")
    parser.add_argument("--k", type=float, default=DEFAULT_DRAG, help="fixed drag coefficient (1/m)")
    parser.add_argument("--sigma-px", type=float, default=1.0, help="per-coordinate measurement standard deviation")
    parser.add_argument("--robust", action="store_true", help="reduce the influence of false detections")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/stage4/reconstructed")
    args = parser.parse_args()
    try:
        camera = Camera.from_dict(json.loads(args.camera.read_text(encoding="utf-8")))
        times, uv = read_observations(args.observations, args.track)
        fit = fit_track(camera, times, uv, k=args.k, sigma_px=args.sigma_px, robust=args.robust)
        theta = fit.theta
        flight = simulate(theta[:3], theta[3:], k=args.k, dt=0.002)
        predicted = camera.project(predict_positions(theta, times, args.k))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    args.out.mkdir(parents=True, exist_ok=True)
    rmse = float(np.sqrt(np.mean(np.sum((uv - predicted)**2, axis=1))))
    write_json(args.out / "fit.json", dict(position_m=theta[:3].tolist(), velocity_m_s=theta[3:].tolist(),
               observed_points=len(times), observed_rmse_px=rmse, k_m_inv=args.k, sigma_px=args.sigma_px,
               robust=args.robust, landing_time_s=float(flight.time[-1]), landing_xy_m=flight.position[-1, :2].tolist(),
               runs=[dict(initial=s.initial.tolist(), theta=s.theta.tolist(), cost=float(s.cost),
                          success=bool(s.success), nfev=int(s.nfev)) for s in fit.starts]))
    write_rows(args.out / "trajectory.csv", [dict(t_s=t, x_m=x, y_m=y, z_m=z)
               for t, (x, y, z) in zip(flight.time, flight.position)])
    write_rows(args.out / "reprojection.csv", [dict(t_s=t, u_px=u, v_px=v, u_fit_px=p, v_fit_px=q)
               for t, (u, v), (p, q) in zip(times, uv, predicted)])
    print(f"{len(times)} detections, reprojection RMS {rmse:.3f} px -> {args.out}")


if __name__ == "__main__":
    main()
