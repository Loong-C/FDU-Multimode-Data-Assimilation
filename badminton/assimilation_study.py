"""Paired model comparisons and held-out repeated-noise coverage of marginal intervals."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import binomtest, truncnorm

from .assimilation_cases import ROOT, SETTINGS, TRUE_K, make_case, make_model
from .flight import GRAVITY
from .posterior import fit_map, flight_states, landing_event, local_curvature


PROBE = 0.24
MEASURES = ("x", "y", "z", "vx", "vy", "vz", "speed", "k", "landing_x", "landing_y", "landing_time")
ABLATION_SEEDS = (5100, 5101)
COVERAGE_SEEDS = range(6300, 6312)
MIN_ESS = 60


def propagate(theta, k):
    """Propagate a batch to 0.24 s and first downward contact, with 5 ms RK4."""
    state = theta.copy()
    count = len(theta)
    end = np.full((count, 3), np.nan)
    probe = np.full_like(theta, np.nan)
    grounded = (state[:, 2] == 0) & (state[:, 5] <= 0)
    end[grounded] = np.column_stack((state[grounded, :2], np.zeros(grounded.sum())))
    # The broad drag ablation can place a few draws much further from the
    # reference flights. Subdivide the common step for those speeds.
    rate = max(np.max(k*np.linalg.norm(state[:, 3:], axis=1)), np.sqrt(GRAVITY*k.max()))
    h = 0.005/max(1, int(np.ceil(2*0.005*rate/2.4)))
    def rhs(s):
        v = s[:, 3:]
        a = -k[:, None]*np.linalg.norm(v, axis=1)[:, None]*v
        a[:, 2] -= GRAVITY
        return np.column_stack((v, a))
    for i in range(1, round(30/h)+1):
        a = rhs(state)
        b = rhs(state + h/2*a)
        c = rhs(state + h/2*b)
        nxt = state + h/6*(a + 2*b + 2*c + rhs(state+h*c))
        if i == round(PROBE/h):
            probe = nxt.copy()
        hit = ~grounded & (nxt[:, 2] <= 0) & (nxt[:, 5] < 0)
        if hit.any():
            u = np.clip(state[hit, 2]/(state[hit, 2]-nxt[hit, 2]), 0, 1)
            end[hit, :2] = state[hit, :2] + u[:, None]*(nxt[hit, :2]-state[hit, :2])
            end[hit, 2] = h*(i-1+u)
            grounded |= hit
        state = nxt
        if grounded.all() and i >= round(PROBE/h):
            break
    if not np.isfinite(end).all():
        raise ValueError("Posterior flight has no bracketed landing")
    # Probe values refer to airborne states. Release from the data-driven
    # posterior can imply an earlier contact in an especially weak window.
    probe[end[:, 2] < PROBE] = np.nan
    return np.column_stack((probe, np.linalg.norm(probe[:, 3:], axis=1), k, end))


def mode_proposals(model, fit):
    """Include separated local optima up to four log-density units below the best."""
    centers = [(fit.x, fit.cov)]
    best_logp = model.log_prob(fit.x)
    for m in sorted((m for m in fit.starts if m.success), key=lambda m: m.cost):
        if best_logp + m.cost > 4 or any(
                np.linalg.norm(np.linalg.solve(np.linalg.cholesky(cov), m.x-mu)) <= 3 for mu,cov in centers):
            continue
        centers.append((m.x, local_curvature(model, m.x)[0]))
    return centers


def vacuum_draws(model, fit, rng):
    """Importance proposal for a misspecified ballistic model at the airborne boundary.

    With no drag, final height is affine in the state. Sample the local quadratic
    density conditional on nonnegative final height; the exact model supplies
    the weights, including the other physical bounds and the landing measurement.
    """
    x = fit.x
    step = np.maximum(np.abs(x), 1.)*2e-5
    lo, hi = np.maximum(x-step, model.lower), np.minimum(x+step, model.upper)
    columns = []
    for j in range(len(x)):
        left, right = x.copy(), x.copy()
        left[j], right[j] = lo[j], hi[j]
        columns.append((model.residual(right)-model.residual(left))/(hi[j]-lo[j]))
    gradient = np.array(columns) @ model.residual(x)
    mu = x-fit.cov@gradient
    a = np.array([0.,0.,1.,0.,0.,model.times[-1]])
    mean_h = model.final_height(x) + a@(mu-x)
    sd_h = np.sqrt(a @ fit.cov @ a)
    gain = fit.cov@a/(sd_h*sd_h)
    L = np.linalg.cholesky(fit.cov)
    for n in (1024, 4096):
        h = truncnorm.rvs(-mean_h/sd_h, np.inf, loc=mean_h, scale=sd_h, size=n, random_state=rng)
        noise = rng.normal(size=(n,len(x)))@L.T
        samples = mu + noise + (h-mean_h-noise@a)[:,None]*gain
        z = (samples-mu)@np.linalg.inv(L).T
        logw = np.array([model.log_prob(v) for v in samples]) + .5*np.sum(z*z,axis=1)
        good = np.isfinite(logw)
        if good.any():
            w = np.exp(logw[good]-logsumexp(logw[good]))
            ess = 1/(w@w)
            if ess >= MIN_ESS:
                break
    if not good.any():
        raise ValueError("Ballistic proposal found no physically feasible draws")
    return samples[good],w,dict(proposal_draws=n, total_proposals=n if n==1024 else n+1024,
                                mode_components=1, proposal_components=1, feasible_fraction=good.mean(),
                                ess=float(ess), largest_weight=w.max())


def posterior_draws(model, fit, seed, coverage):
    """Importance-sample the exact posterior with full mixture-density weights."""
    rng = np.random.default_rng(seed)
    dim, nu = len(fit.x), 8
    modes = mode_proposals(model, fit)
    if model.fixed_k == 0 and len(modes) == 1:
        return vacuum_draws(model, fit, rng)
    means = [mu for mu,_ in modes]
    factors = [np.linalg.cholesky(C) for _,C in modes]
    probability = np.full(len(modes), 1/len(modes))
    curved = False
    considered = 0
    joint = model.fixed_k is None
    sizes = (2048, 4096, 8192) if coverage else ((2048, 4096) if joint else (640, 2560))
    for n in sizes:
        if not curved and (joint or considered):
            from .posterior_sampling import ridge_proposal
            ridge = ridge_proposal(model, fit)
            # Keep a broad component for every competitive mode in addition to
            # the curved conditional proposals. Weight using the entire mixture.
            means.extend(ridge.means)
            factors.extend(ridge.factors)
            probability = np.r_[.1*probability, .9*ridge.weights]
            curved = True
        # A second, independent larger run replaces the pilot if its ESS is low.
        considered += n
        labels = rng.choice(len(means), size=n, p=probability) if len(means)>1 else np.zeros(n, dtype=int)
        u = rng.normal(size=(n, dim))/np.sqrt(rng.chisquare(nu, n)[:,None]/nu)
        x = np.empty_like(u)
        for j, (mu, L) in enumerate(zip(means, factors)):
            x[labels == j] = mu+u[labels == j]@L.T
        densities = []
        for mu, L, prob in zip(means, factors, probability):
            if prob == 0:
                continue
            z = (x-mu)@np.linalg.inv(L).T
            densities.append(np.log(prob)-np.linalg.slogdet(L)[1]-(nu+dim)/2*np.log1p(np.sum(z*z,axis=1)/nu))
        # The common Student-t normalizing constant cancels.
        log_mix = logsumexp(densities, axis=0)
        logw = np.array([model.log_prob(z) for z in x])-log_mix
        good = np.isfinite(logw)
        if not good.any():
            continue
        w = np.exp(logw[good]-logsumexp(logw[good]))
        ess = float(1/np.dot(w,w))
        if ess >= MIN_ESS:
            break
    if not good.any():
        raise ValueError("No importance sample fell in posterior support")
    return x[good],w,dict(proposal_draws=n, total_proposals=considered, mode_components=len(modes),
                          proposal_components=len(means), feasible_fraction=good.mean(),
                          ess=ess, largest_weight=w.max())


def quantiles(values, weights):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    w = w/w.sum()
    return np.interp([0.025, 0.05, 0.50, 0.95, 0.975], np.cumsum(w)-0.5*w, v)


def truth_values(case):
    r = case.flight
    states = np.column_stack((r.position, r.velocity))
    probe = np.array([np.interp(PROBE, r.time, a) for a in states.T])
    return np.r_[probe, np.linalg.norm(probe[3:]), case.truth_k, r.position[-1, :2], r.time[-1]]


def evaluate(case, setting, group):
    model = make_model(case, setting)
    fit = fit_map(model)
    # Monte Carlo stream is independent of every measurement stream.
    seed = [81053, case.seed, list(TRUE_K).index(case.shot), list(SETTINGS).index(setting)]
    draws, w, checks = posterior_draws(model, fit, seed, coverage=group == "coverage")
    k = np.exp(draws[:, 6]) if model.fixed_k is None else np.full(len(draws), model.fixed_k)
    # At most 8192 * 1e-14 posterior mass is omitted from propagation.
    keep = w >= 1e-14
    sampled = propagate(draws[keep, :6], k[keep])
    w = w[keep] / w[keep].sum()
    true = truth_values(case)
    theta, khat = model.unpack(fit.x)
    t_end, end_xy = landing_event(theta, khat)
    residual = model.residual(fit.x)[:len(model.times)*2]
    grid = np.linspace(0, min(t_end, case.flight.time[-1]), 201)
    path = flight_states(theta, grid, khat)
    ref = np.column_stack([np.interp(grid, case.flight.time, a) for a in case.flight.position.T])
    position_rmse = np.sqrt(np.trapezoid(np.sum((path[:, :3]-ref)**2, axis=1), grid)/grid[-1])
    probe = flight_states(theta, [PROBE], khat)[0]
    # Screen independent optimizer starts for distinct modes of comparable mass density.
    # This local search is complemented by the representative multi-chain runs.
    successful = [m for m in fit.starts if m.success]
    near = [m for m in successful if m.cost-min(s.cost for s in successful) <= 2]
    separation = max(float(np.linalg.norm(np.linalg.solve(np.linalg.cholesky(fit.cov), m.x-fit.x))) for m in near)
    row = dict(group=group, shot=case.shot, seed=case.seed, setting=setting,
               detections=len(model.times), true_k=case.truth_k, map_k=khat,
               k_error=abs(khat-case.truth_k), position_rmse_m=position_rmse,
               probe_velocity_error_m_s=np.linalg.norm(probe[3:]-true[3:6]),
               landing_error_m=np.linalg.norm(end_xy-case.flight.position[-1, :2]),
               white_pixel_rms=np.sqrt(np.mean(residual**2)), rank=fit.rank, condition=fit.condition,
               converged_starts=len(successful), nearby_mode_max_mahalanobis=separation,
               **{k:v for k,v in zip(("x0", "y0", "z0", "vx0", "vy0", "vz0"), theta)}, **checks)
    intervals = []
    for i, field in enumerate(MEASURES):
        if field == "k" and model.fixed_k is not None:
            continue
        valid = np.isfinite(sampled[:, i]) & (w > 0)
        mass = w[valid].sum()
        if mass == 0:
            continue
        q = quantiles(sampled[valid, i], w[valid])
        # Reject printed intervals as reliable for effective support too small
        # or a probe after appreciable posterior probability of touchdown.
        valid_ess = mass**2 / np.sum(w[valid]**2)
        intervals.append(dict(group=group, shot=case.shot, seed=case.seed, setting=setting, quantity=field,
                              truth=true[i], q025=q[0], q05=q[1], median=q[2], q95=q[3], q975=q[4],
                              width90=q[3]-q[1], width95=q[4]-q[0], supports_probe_mass=mass,
                              usable=int(valid_ess >= MIN_ESS and mass >= .99)))
    return row, intervals


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(runs, intervals, failures):
    # Paired experiments share physical truths and both noisy measurement streams.
    metrics = ("position_rmse_m", "probe_velocity_error_m_s", "k_error", "landing_error_m", "white_pixel_rms")
    comparison, per_run = {}, {}
    for setting in SETTINGS:
        selected = [r for r in runs if r["group"] == "paired" and r["setting"] == setting]
        it = [i for i in intervals if i["group"] == "paired" and i["setting"] == setting and i["usable"]]
        desc = {"attempted": len(TRUE_K)*len(ABLATION_SEEDS), "completed": len(selected)}
        if selected:
            desc.update({m: float(np.median([r[m] for r in selected])) for m in metrics})
        for r in selected:
            per_run[(setting, r["shot"], r["seed"])] = {m: r[m] for m in metrics}
        for family, fields in {"position": ("x","y","z"), "velocity": ("vx","vy","vz"),
                               "k": ("k",), "landing_xy": ("landing_x","landing_y")}.items():
            widths = {}
            for r in it:
                if r["quantity"] in fields:
                    widths.setdefault((r["shot"], r["seed"]), []).append(r["width90"])
            vals = [np.mean(v) for v in widths.values() if len(v) == len(fields)]
            for (shot, seed), v in widths.items():
                if len(v) == len(fields):
                    per_run[(setting, shot, seed)][family+"_width90"] = np.mean(v)
            desc[family+"_width90"] = float(np.median(vals)) if vals else None
            desc[family+"_interval_runs"] = len(vals)
        comparison[setting] = desc
    # Matched changes use precisely the same shot, pixel noise and external
    # measurements on both sides, including when one interval failed its ESS check.
    paired_effects = {}
    for base, alternative in (("short_broad_k", "short_pixels"), ("short_pixels", "short_player"),
                              ("short_pixels", "short_height"), ("short_pixels", "short_launch"),
                              ("short_launch", "short_all"), ("short_launch", "full_launch"),
                              ("full_launch", "full_all"), ("full_fixed", "full_all"),
                              ("full_ballistic", "full_all")):
        effects = {}
        for m in (*metrics, "position_width90", "velocity_width90", "k_width90", "landing_xy_width90"):
            pairs = []
            for shot in TRUE_K:
                for seed in ABLATION_SEEDS:
                    a = per_run.get((base, shot, seed), {})
                    b = per_run.get((alternative, shot, seed), {})
                    if m in a and m in b:
                        pairs.append(a[m] - b[m])
            if pairs:
                effects[m] = dict(n=len(pairs), median_paired_reduction=float(np.median(pairs)))
        paired_effects[f"{base}_to_{alternative}"] = effects
    coverage = []
    n = len(TRUE_K)*len(COVERAGE_SEEDS)
    for field in MEASURES:
        rows = [r for r in intervals if r["group"] == "coverage" and r["quantity"] == field and r["usable"]]
        for pct, low, high in ((90,"q05","q95"), (95,"q025","q975")):
            count = sum(r[low] <= r["truth"] <= r[high] for r in rows)
            limits = binomtest(int(count), len(rows)).proportion_ci(method="wilson") if rows else None
            coverage.append(dict(quantity=field, nominal=pct/100, attempted=n, usable=len(rows), covered=int(count),
                                 coverage=count/len(rows) if rows else None,
                                 bounds_over_all_attempts=[count/n, (count+n-len(rows))/n],
                                 wilson95=[limits.low, limits.high] if limits else None))
    return dict(probe_time_s=PROBE, min_importance_ess=MIN_ESS,
                paired_seeds=list(ABLATION_SEEDS), coverage_seeds=list(COVERAGE_SEEDS),
                coverage_setting="full_all", paired=comparison, paired_effects=paired_effects,
                coverage=coverage, failures=failures)


def evaluate_block(task):
    group, shot, seeds, settings = task
    runs, intervals, failures = [], [], []
    for seed in seeds:
        case = make_case(shot, seed)
        for setting in settings:
            try:
                row, bounds = evaluate(case, setting, group)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                failures.append(dict(group=group, shot=shot, seed=seed, setting=setting, reason=str(exc)))
                continue
            runs.append(row)
            intervals.extend(bounds)
    print(f"{group} {shot} complete", flush=True)
    return runs, intervals, failures


def run(out, workers=1):
    tasks = [(group, shot, seeds, settings)
             for group, seeds, settings in (("paired", ABLATION_SEEDS, tuple(SETTINGS)),
                                            ("coverage", COVERAGE_SEEDS, ("full_all",)))
             for shot in TRUE_K]
    if workers == 1:
        blocks = list(map(evaluate_block, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            blocks = list(pool.map(evaluate_block, tasks))
    runs, intervals, failures = ([row for block in blocks for row in block[j]] for j in range(3))
    if not runs or not intervals:
        raise RuntimeError(f"No usable fits: {failures[:1]}")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "comparison-runs.csv", runs)
    write_csv(out / "intervals.csv", intervals)
    report = summarize(runs, intervals, failures)
    (out / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{len(runs)} fits completed, {len(failures)} failed -> {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/stage5")
    parser.add_argument("--workers", type=int, default=1, help="independent shot blocks (default: serial)")
    args = parser.parse_args()
    run(args.out, args.workers)
