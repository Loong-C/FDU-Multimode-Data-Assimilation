"""Check the stage-five posterior against independent dynamics, projections and exported intervals."""

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from badminton.assimilation_cases import TRUE_K, make_case
from badminton.inverse import PRIOR_MEAN, PRIOR_STD
from badminton.posterior import PosteriorModel, fit_map, flight_states, landing_event
from badminton.posterior_sampling import sample_posterior
from badminton.reconstruct import write_json
from validate_stage4 import project, reference_flight


def model_check():
    case = make_case("clear", 7101)
    subset = slice(1, -6, 3)
    t, uv, cov = case.times[subset], case.uv[subset], case.R[subset]
    model = PosteriorModel(case.camera, t, uv, R=cov, constraints=case.constraints)
    # No coordinates below are provided to the MAP estimator.
    probes = [case.truth_theta, case.truth_theta+[.025, -.01, .02, -.1, .17, -.2]]
    drag = [case.truth_k, case.truth_k*1.015]
    discrepancies = []
    for theta, k in zip(probes, drag):
        f, end, xy, _ = reference_flight(theta, k)
        states = f(t).T
        err = project(case.camera, states[:, :3])-uv
        cost = np.einsum("ni,nij,nj->", err, np.linalg.inv(cov), err)
        cost += np.sum(((theta-PRIOR_MEAN)/PRIOR_STD)**2) + (np.log(k/.217)/.35)**2
        for name, value in (("player_xy", theta[:2]), ("height", theta[2]),
                            ("landing_xy", xy), ("landing_time", end)):
            obs, sd = case.constraints[name]
            cost += np.sum(((value-obs)/sd)**2)
        x = np.r_[theta, np.log(k)]
        discrepancies.append(abs(model.log_prob(x) + cost/2))
        np.testing.assert_allclose(states, flight_states(theta, t, k), atol=2e-6, rtol=0)
        t_hat, xy_hat = landing_event(theta, k)
        np.testing.assert_allclose(np.r_[t_hat, xy_hat], np.r_[end, xy], atol=2e-6, rtol=0)
    if max(discrepancies) > 2e-4:
        raise AssertionError(f"Joint log posterior disagrees with independent likelihood by {max(discrepancies)}")

    invalid = np.r_[case.truth_theta, np.log(case.truth_k)]
    invalid[2], invalid[5] = 0.2, -12.
    assert model.log_prob(invalid) == -np.inf
    missing = uv.copy()
    missing[2] = np.nan
    retain = np.isfinite(missing).all(axis=1)
    by_nan = PosteriorModel(case.camera, t, missing, R=cov, fixed_k=.217)
    by_rows = PosteriorModel(case.camera, t[retain], uv[retain], R=cov[retain], fixed_k=.217)
    np.testing.assert_allclose(by_nan.log_prob(case.truth_theta), by_rows.log_prob(case.truth_theta))
    try:
        PosteriorModel(case.camera, t, uv, R=[[1, 2], [2, 1]])
    except ValueError:
        pass
    else:
        raise AssertionError("Non-positive-definite R was accepted")
    fit = fit_map(model)
    assert fit.rank == 7 and np.isfinite(model.log_prob(fit.x)) and np.all(np.linalg.eigvalsh(fit.cov) > 0)
    assert np.isclose(min(s.cost for s in fit.starts if s.success), -model.log_prob(fit.x))
    return dict(max_abs_log_prob_discrepancy=max(discrepancies), joint_dimension=fit.rank,
                starts=len(fit.starts), observation_covariance="correlated, varying by frame",
                soft_measurements_checked=list(case.constraints))


def sampler_check():
    # A separate tractable density checks the MH correction and ability to visit two basins.
    centers = np.zeros((2, 6))
    centers[:, 0] = [-3.5, 3.5]

    def log_density(x):
        return float(np.logaddexp.reduce(-0.5*np.sum((x-centers)**2, axis=1)) - np.log(2))

    starts = [SimpleNamespace(x=mu, cost=-log_density(mu), success=True) for mu in centers]
    fit = SimpleNamespace(x=centers[0], cov=np.eye(6), starts=starts)
    # Local quadratic residuals give the proposal the curvature of the nearest mode.
    # Metropolis acceptance still uses the exact mixture density.
    model = SimpleNamespace(log_prob=log_density, lower=np.full(6, -30.), upper=np.full(6, 30.),
                            residual=lambda x: x - centers[int(x[0] > 0)])
    samples = sample_posterior(model, fit, seed=518, warmup=80, draws=700)
    values = samples.chains.reshape(-1, 6)
    positive = float(np.mean(values[:, 0] > 0))
    screen = samples.diagnostics["mode_screen"]
    assert abs(positive-.5) < .06 and np.max(abs(values[:, 1:].mean(axis=0))) < .14
    assert screen["competitive_basins"] == 2 and len(screen["marginal_kde_peak_locations"][0]) == 2
    return dict(target="equal-weight 6D Gaussian mixture", known_positive_mass=.5,
                sampled_positive_mass=positive, identified_marginal_peaks=screen["marginal_kde_peak_locations"][0])


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def check_mcmc(directory):
    summary = json.loads((directory / "mcmc.json").read_text(encoding="utf-8"))
    land = read_csv(directory / "mcmc-flight-draws.csv")
    bands = read_csv(directory / "mcmc-bands.csv")
    maxima = []
    with np.load(directory / "mcmc-chains.npz", allow_pickle=False) as chains:
        for setting in ("short_launch", "full_launch"):
            params = chains[setting]
            diag, report = summary["sampling"][setting], summary["propagation"][setting]
            assert params.shape == (4, diag["draws_per_chain"], 7) and diag["converged"]
            assert diag["max_rhat"] < 1.01 and min(diag["min_bulk_ess"], diag["min_tail_ess"]) >= 400
            rows = [r for r in land if r["setting"] == setting]
            ks = np.exp(params[:, :, 6]).ravel()
            reference = np.quantile(ks, [.05, .5, .95])
            np.testing.assert_allclose(report["parameter_intervals"]["k_m_inv"], reference)
            assert len(rows) == report["retained_draws"]
            for label, field in (("flight_s", "landing_t_s"), ("landing_x_m", "landing_x_m"),
                                 ("landing_y_m", "landing_y_m")):
                np.testing.assert_allclose(report["landing_intervals"][label],
                                           np.quantile([float(r[field]) for r in rows], [.05, .5, .95]))
            drawn = params[[int(r["chain"]) for r in rows], [int(r["iteration"]) for r in rows]]
            for r, x in zip(rows[::101], drawn[::101]):
                _, end, xy, _ = reference_flight(x[:6], np.exp(x[6]))
                err = max(abs(end - float(r["landing_t_s"])),
                          np.max(abs(xy - [float(r["landing_x_m"]), float(r["landing_y_m"])])))
                maxima.append(float(err))
            for j, state in enumerate(("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s")):
                first = next(r for r in bands if r["setting"] == setting and r["state"] == state
                             and float(r["t_s"]) == 0.)
                np.testing.assert_allclose([float(first[k]) for k in ("q05", "median", "q95")],
                                           np.quantile(drawn[:, j], [.05, .5, .95]))
    assert max(maxima) < 2e-6
    return dict(representative_cases=2, independently_integrated_draws=len(maxima),
                max_abs_landing_component_discrepancy=max(maxima))


def check_study(directory):
    report = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
    runs = read_csv(directory / "comparison-runs.csv")
    bounds = read_csv(directory / "intervals.csv")
    keys = {(r["group"], r["shot"], r["seed"], r["setting"]) for r in runs}
    assert len(keys) == len(runs) and len(keys)+len(report["failures"]) == 136
    paired_seeds, cover_seeds = set(report["paired_seeds"]), set(report["coverage_seeds"])
    assert paired_seeds.isdisjoint(cover_seeds) and len(cover_seeds) == 12
    # Reconstruct every MAP value using DOP853, a separate touchdown finder, and OpenCV.
    truths = {}
    largest = 0.
    for row in runs:
        case = make_case(row["shot"], int(row["seed"]))
        theta = np.array([float(row[k]) for k in ("x0", "y0", "z0", "vx0", "vy0", "vz0")])
        f, end, xy, _ = reference_flight(theta, float(row["map_k"]))
        if case.shot not in truths:
            truths[case.shot] = reference_flight(case.truth_theta, case.truth_k)
        true, true_end, true_xy, _ = truths[case.shot]
        t = np.linspace(0, min(end, true_end), 201)
        delta = f(t)[:3]-true(t)[:3]
        e3 = np.sqrt(np.trapezoid(np.sum(delta**2, axis=0), t)/t[-1])
        ev = np.linalg.norm(f(report["probe_time_s"])[3:]-true(report["probe_time_s"])[3:])
        retained = slice(None, int(row["detections"]))
        duv = project(case.camera, f(case.times[retained])[:3].T)-case.uv[retained]
        mah = np.einsum("ni,nij,nj->", duv, np.linalg.inv(case.R[retained]), duv)
        actual = [e3, ev, np.linalg.norm(xy-true_xy), np.sqrt(mah/(2*len(duv)))]
        saved = [float(row[k]) for k in ("position_rmse_m","probe_velocity_error_m_s","landing_error_m","white_pixel_rms")]
        largest = max(largest, float(np.max(np.abs(np.array(saved)-actual))))
    assert largest < 2e-4, largest
    for r in bounds:
        shot = r["shot"]
        true, end, xy, _ = truths[shot]
        s = true(report["probe_time_s"])
        known = dict(zip(("x", "y", "z", "vx", "vy", "vz"), s))
        known.update(speed=np.linalg.norm(s[3:]), k=TRUE_K[shot],
                     landing_x=xy[0], landing_y=xy[1], landing_time=end)
        np.testing.assert_allclose(float(r["truth"]), known[r["quantity"]], rtol=0, atol=2e-5)
        quantiles = np.array([float(r[k]) for k in ("q025", "q05", "median", "q95", "q975")])
        assert np.all(np.diff(quantiles) >= 0)
        np.testing.assert_allclose(float(r["width90"]), quantiles[3]-quantiles[1])
    for coverage in report["coverage"]:
        low, high = ("q05", "q95") if coverage["nominal"] == .9 else ("q025", "q975")
        used = [r for r in bounds if r["group"] == "coverage" and r["quantity"] == coverage["quantity"]
                and int(r["usable"])]
        count = sum(float(r[low]) <= float(r["truth"]) <= float(r[high]) for r in used)
        assert coverage["usable"] == len(used) and coverage["covered"] == count and coverage["attempted"] == 48
        if used:
            np.testing.assert_allclose(count / len(used), coverage["coverage"])
            n, z = len(used), 1.95996398454
            p = count/n
            center = (p+z*z/(2*n))/(1+z*z/n)
            radius = z*np.sqrt((p*(1-p)+z*z/(4*n))/n)/(1+z*z/n)
            np.testing.assert_allclose([center-radius, center+radius], coverage["wilson95"], atol=1e-10)
    # Compare both a fitted parameter and a nonlinear prediction across samplers.
    mcmc = json.loads((directory / "mcmc.json").read_text(encoding="utf-8"))
    endpoint_discrepancies = {}
    for setting, sampled in mcmc["propagation"].items():
        for field, target in (("k",sampled["parameter_intervals"]["k_m_inv"]),
                              ("landing_x",sampled["landing_intervals"]["landing_x_m"])):
            row = next(r for r in bounds if (r["group"], r["shot"], r["seed"], r["setting"], r["quantity"])
                       == ("paired", "clear", "5100", setting, field))
            approx = np.array([float(row[k]) for k in ("q05", "median", "q95")])
            fraction = float(np.max(abs(approx-target))/(target[2]-target[0]))
            endpoint_discrepancies[f"{setting}_{field}"] = fraction
            assert int(row["usable"]) and fraction < .25, endpoint_discrepancies
    modes = json.loads((directory / "modes.json").read_text(encoding="utf-8"))
    assert modes["landing_gap_m"] > 1 and modes["profiled_cost_rise_above_secondary"] > 0
    fitted_landings = [reference_flight(np.array(m["x"][:6]), m["k_m_inv"])[2] for m in modes["modes"]]
    np.testing.assert_allclose(np.linalg.norm(fitted_landings[0]-fitted_landings[1]), modes["landing_gap_m"],
                               atol=2e-5, rtol=0)
    return dict(completed_fits=len(runs), failed_fits=len(report["failures"]),
                separate_coverage_design_trials=48, max_abs_map_metric_discrepancy=largest,
                coverage_counts_checked=len(report["coverage"]),
                sampler_quantile_discrepancy_as_fraction_of_90pct_width=endpoint_discrepancies)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "results/stage5")
    parser.add_argument("--out", type=Path, default=ROOT / "results/stage5/validation.json")
    args = parser.parse_args()
    report = dict(model=model_check(), sampler_control=sampler_check(), representative_mcmc=check_mcmc(args.data),
                  study=check_study(args.data), passed=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    print(f"Stage-five checks passed -> {args.out}")


if __name__ == "__main__":
    main()
