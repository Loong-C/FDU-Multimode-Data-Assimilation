"""Simulate a single stroke or reproduce the stage-two court experiments."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .court import DOUBLES_HALF_WIDTH, HALF_LENGTH, SINGLES_HALF_WIDTH, net_height
from .flight import DEFAULT_DRAG, simulate
from .plots import render_figures, render_trajectories


ROOT = Path(__file__).resolve().parents[1]


def summarize(run_id, shot, flight, k, dt):
    """Report unconstrained free flight and its intersection with the net plane."""
    start = flight.position[0]
    landing = flight.position[-1]
    x = flight.position[:, 0]
    crossing = {}
    if x[0] != x[-1] and min(x[0], x[-1]) <= 0 <= max(x[0], x[-1]):
        order = slice(None) if x[-1] > x[0] else slice(None, None, -1)
        t, y, z = (float(np.interp(0.0, x[order], values[order]))
                   for values in (flight.time, flight.position[:, 1], flight.position[:, 2]))
        crossing = {"net_plane_t_s": t, "net_plane_y_m": y, "net_plane_z_m": z}
        if abs(y) <= DOUBLES_HALF_WIDTH:
            crossing["net_clearance_m"] = z - float(net_height(y))
    in_singles = abs(landing[0]) <= HALF_LENGTH and abs(landing[1]) <= SINGLES_HALF_WIDTH
    opposite = in_singles and np.sign(start[0]) * np.sign(landing[0]) < 0
    return {
        "run_id": run_id, "shot": shot, "k_m_inv": k, "dt_s": dt,
        "initial_x_m": start[0], "initial_y_m": start[1], "initial_z_m": start[2],
        "initial_vx_m_s": flight.velocity[0, 0], "initial_vy_m_s": flight.velocity[0, 1],
        "initial_vz_m_s": flight.velocity[0, 2], "initial_speed_m_s": flight.speed[0],
        "apex_t_s": flight.apex_time, "apex_x_m": flight.apex_position[0],
        "apex_y_m": flight.apex_position[1], "apex_z_m": flight.apex_position[2],
        "landing_t_s": flight.time[-1], "landing_x_m": landing[0],
        "landing_y_m": landing[1], "landing_z_m": landing[2],
        "landing_speed_m_s": flight.speed[-1], "landing_in_opposite_singles": bool(opposite),
        "net_plane_t_s": "", "net_plane_y_m": "", "net_plane_z_m": "", "net_clearance_m": "",
        **crossing,
    }


def write_data(out, flights, summaries):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    with (out / "trajectories.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("run_id", "t_s", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s", "speed_m_s"))
        for run_id, flight in flights.items():
            samples = np.column_stack((flight.time, flight.position, flight.velocity, flight.speed))
            writer.writerows((run_id, *sample) for sample in samples)


def examples(config_path, out):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    k = config["k_m_inv"]
    dt = config["dt_s"]
    shots = {}
    summaries = []
    for name, initial in config["shots"].items():
        flight = simulate(initial["position_m"], initial["velocity_m_s"], k=k, dt=dt)
        shots[name] = flight
        summaries.append(summarize(name, name, flight, k, dt))

    reference = config["comparison_shot"]
    initial = config["shots"][reference]
    flights = dict(shots)
    comparisons = {}
    for drag in config["comparison_k_m_inv"]:
        if drag == k:
            comparisons[drag] = shots[reference]
            continue
        run_id = f"{reference}_k{drag:g}"
        flight = simulate(initial["position_m"], initial["velocity_m_s"], k=drag, dt=dt)
        comparisons[drag] = flights[run_id] = flight
        summaries.append(summarize(run_id, reference, flight, drag, dt))

    write_data(out, flights, summaries)
    render_figures(out, shots, comparisons)
    print(f"{len(shots)} shot types, {len(flights)} simulated trajectories -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    batch = commands.add_parser("examples", help="generate the five strokes and drag comparison")
    batch.add_argument("--config", type=Path, default=ROOT / "configs/stage2-shots.json")
    batch.add_argument("--out", type=Path, default=ROOT / "results/stage2")
    single = commands.add_parser("shot", help="simulate until first ground contact")
    single.add_argument("--position", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"), help="metres")
    single.add_argument("--velocity", type=float, nargs=3, required=True, metavar=("VX", "VY", "VZ"), help="m/s")
    single.add_argument("--k", type=float, default=DEFAULT_DRAG, help="quadratic drag coefficient in 1/m")
    single.add_argument("--dt", type=float, default=0.01, help="integration step in seconds")
    single.add_argument("--out", type=Path, default=ROOT / "artifacts/stage2/custom")
    args = parser.parse_args()
    try:
        if args.command == "examples":
            examples(args.config, args.out)
        else:
            flight = simulate(args.position, args.velocity, k=args.k, dt=args.dt)
            summary = summarize("custom", "custom", flight, args.k, args.dt)
            write_data(args.out, {"custom": flight}, [summary])
            render_trajectories(args.out / "trajectory.png", {"custom": flight})
            print(f"Flight {flight.time[-1]:.4f} s, apex {flight.apex_position[2]:.4f} m, "
                  f"landing {np.round(flight.position[-1], 4)} m -> {args.out}")
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
