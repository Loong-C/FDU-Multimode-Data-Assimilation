"""Estimate one shot's posterior from image detections and optional independent measurements."""

import argparse
import json
from pathlib import Path

import numpy as np

from .camera import Camera
from .posterior import PosteriorModel, fit_map, flight_states, landing_event
from .posterior_sampling import sample_posterior
from .reconstruct import ROOT, read_observations, write_json, write_rows


STATE = ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s")


def interval(values, mass):
    tail = (1 - mass) / 2
    low, median, high = np.quantile(values, [tail, 0.5, 1-tail], axis=0)
    return low, median, high


def export_prediction(out, model, fit, samples, *, mass=0.90, seed=5201):
    """Propagate an equal number of draws from each chain through first ground contact."""
    chains = samples.chains
    rng = np.random.default_rng(seed)
    flights, chosen = [], []
    for chain_id, chain in enumerate(chains):
        indices = np.sort(rng.choice(len(chain), min(200, len(chain)), replace=False))
        for i in indices:
            theta, k = model.unpack(chain[i])
            end, xy = landing_event(theta, k)
            chosen.append((theta, k, end))
            flights.append(dict(chain=chain_id+1, draw=int(i)+1, **dict(zip(STATE, theta)),
                                k_m_inv=k, landing_time_s=end, landing_x_m=xy[0], landing_y_m=xy[1]))
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / "flights.csv", flights)
    # Direct parameter quantiles can use every retained state, including repeated MH draws.
    launch = chains.reshape(-1, chains.shape[-1])
    marginal_draws = dict(zip(STATE, launch[:, :6].T))
    marginal_draws["k_m_inv"] = np.exp(launch[:, 6]) if model.fixed_k is None else [model.fixed_k]
    for key in ("landing_time_s", "landing_x_m", "landing_y_m"):
        marginal_draws[key] = [f[key] for f in flights]
    summaries = {}
    for key, vals in marginal_draws.items():
        low, median, high = interval(vals, mass)
        summaries[key] = dict(low=low, median=median, high=high)

    # These pointwise flight-state intervals condition on the shuttle still being airborne.
    times = np.arange(0., max(end for _, _, end in chosen), 0.04)
    values = np.full((len(chosen), len(times), 6), np.nan)
    for i, (theta, k, end) in enumerate(chosen):
        alive = times < end
        if alive.any():
            values[i, alive] = flight_states(theta, times[alive], k)
    bands = []
    for j, t in enumerate(times):
        airborne = values[:, j, :]
        airborne = airborne[np.isfinite(airborne).all(axis=1)]
        row = dict(t_s=t, airborne_draws=len(airborne), airborne_probability=len(airborne)/len(chosen))
        for name, column in zip(("low", "median", "high"), interval(airborne, mass)):
            row.update((f"{field}_{name}", v) for field, v in zip(STATE, column))
        bands.append(row)
    if bands:
        write_rows(out / "trajectory-intervals.csv", bands)
    map_theta, map_k = model.unpack(fit.x)
    write_json(out / "posterior.json", dict(interval_mass=mass, propagated_draws=len(chosen),
               interval_status="chain_checks_passed" if samples.diagnostics["converged"] else "exploratory_unconverged",
               retained_draws=int(chains.shape[0]*chains.shape[1]),
               parameterization="position_m, velocity_m_s, log(k) if estimated",
               map_position_m=map_theta[:3].tolist(), map_velocity_m_s=map_theta[3:].tolist(),
               map_k_m_inv=map_k, fixed_k=model.fixed_k, marginal_intervals=summaries,
               chain_diagnostics=samples.diagnostics))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True, help="t_s,u_px,v_px since contact")
    parser.add_argument("--track", help="case or shot in a multi-shot CSV")
    parser.add_argument("--camera", type=Path, required=True)
    noise = parser.add_mutually_exclusive_group()
    noise.add_argument("--sigma-px", type=float, default=1.0, help="independent coordinate standard deviation")
    noise.add_argument("--pixel-covariance", type=Path, help="JSON 2x2 or one 2x2 covariance per retained detection")
    parser.add_argument("--fixed-k", type=float, help="hold drag (1/m) fixed, default jointly estimates it")
    parser.add_argument("--constraints", type=Path, help="JSON with optional player_xy, height, landing_xy, landing_time")
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--draws", type=int, default=800, help="retained draws per chain")
    parser.add_argument("--seed", type=int, default=5201)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/stage5/posterior")
    args = parser.parse_args()
    try:
        camera = Camera.from_dict(json.loads(args.camera.read_text(encoding="utf-8")))
        times, uv = read_observations(args.observations, args.track)
        constraints = json.loads(args.constraints.read_text(encoding="utf-8")) if args.constraints else None
        cov = json.loads(args.pixel_covariance.read_text(encoding="utf-8")) if args.pixel_covariance else None
        model = PosteriorModel(camera, times, uv, sigma_px=args.sigma_px, R=cov,
                               fixed_k=args.fixed_k, constraints=constraints)
        fit = fit_map(model)
        draws = sample_posterior(model, fit, seed=args.seed, warmup=args.warmup, draws=args.draws)
        export_prediction(args.out, model, fit, draws, seed=args.seed)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"{len(times)} detections, max R-hat {draws.diagnostics['max_rhat']:.3f}, "
          f"diagnostics passed: {draws.diagnostics['converged']} -> {args.out}")


if __name__ == "__main__":
    main()
