"""Metropolis sampling in local covariance coordinates, with independent chain diagnostics."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.special import logsumexp, ndtri
from scipy.stats import gaussian_kde, rankdata


@dataclass
class PosteriorSamples:
    chains: np.ndarray
    log_prob: np.ndarray
    diagnostics: dict


def _scaled_factor(cov):
    sd = np.sqrt(np.diag(cov))
    if not np.isfinite(sd).all() or np.any(sd <= 0):
        raise ValueError("A finite, positive proposal variance is required")
    corr = cov / np.outer(sd, sd)
    values, vectors = np.linalg.eigh((corr + corr.T) / 2)
    return sd[:, None] * vectors * np.sqrt(np.maximum(values, 1e-10))


def _normal_ranks(chains):
    ranks = rankdata(chains.ravel()).reshape(chains.shape)
    return ndtri((ranks - 3/8) / (ranks.size + 1/4))


def _rhat(split):
    n = split.shape[1]
    within = np.var(split, axis=1, ddof=1).mean()
    if within <= 0:
        return np.inf
    return float(np.sqrt((n-1)/n + np.var(split.mean(axis=1), ddof=1)/within))


def _ess(split):
    """Multi-chain autocovariances and Geyer's initial positive monotone pair sums."""
    m, n = split.shape
    within = np.var(split, axis=1, ddof=1).mean()
    variance = (n-1)/n * within + np.var(split.mean(axis=1), ddof=1)
    if within <= 0 or variance <= 0:
        return 0.
    centred = split - split.mean(axis=1, keepdims=True)
    freqs = np.fft.rfft(centred, n=2*n, axis=1)
    acov = np.fft.irfft(abs(freqs)**2, n=2*n, axis=1)[:, :n].mean(axis=0) / n
    rho = 1 - (within - acov)/variance
    rho[0] = 1.
    pairs = rho[:2*(n//2)].reshape(-1, 2).sum(axis=1)
    negative = np.flatnonzero(pairs <= 0)
    positive = pairs[:negative[0]] if len(negative) else pairs
    tau = max(1., 2*np.minimum.accumulate(positive).sum() - 1)
    return float(m*n/tau)


def _diagnostics(array):
    half = array.shape[1]//2
    split = np.concatenate((array[:, :half], array[:, -half:]), axis=0)
    rank = _normal_ranks(split)
    folded = _normal_ranks(np.abs(split - np.median(split)))
    tail = min(_ess(split <= q) for q in np.quantile(split, [.05, .95]))
    return max(_rhat(rank), _rhat(folded)), _ess(rank), tail


def _marginal_peaks(values):
    """Screen smoothed marginals; a density peak is not itself a joint posterior mode."""
    low, high = np.quantile(values, [.005, .995])
    if high <= low:
        return []
    grid = np.linspace(low, high, 192)
    # The fixed bandwidth (0.35 marginal SD) reduces low-ESS binning artefacts.
    density = gaussian_kde(values, bw_method=.35)(grid)
    peaks = find_peaks(density, prominence=.12*density.max(), distance=10)[0]
    return grid[peaks].tolist()


def _mode_centres(fit, factor):
    """Distinct competitive fit basins in MAP-whitened coordinates."""
    centres, counts, costs = [], [], []
    for start in sorted(fit.starts, key=lambda s: s.cost):
        if not start.success or not np.isfinite(start.cost):
            continue
        z = np.linalg.solve(factor, start.x-fit.x)
        gaps = np.linalg.norm(np.asarray(centres)-z, axis=1) if centres else []
        if len(gaps) and np.min(gaps) < 1.:
            counts[np.argmin(gaps)] += 1
        else:
            centres.append(z)
            costs.append(float(start.cost))
            counts.append(1)
    if not centres:
        raise ValueError("No converged MAP starts")
    competitive = [z for z, cost in zip(centres, costs) if cost-costs[0] <= 12]
    return np.asarray(competitive), [dict(cost=c, successful_starts=n,
                                         distance_from_best=float(np.linalg.norm(z)))
                                    for z, c, n in zip(centres, costs, counts)]


class _Proposal:
    """Student-t mixture following the curved conditional MAP ridge.

    The independent proposal includes component weights, volumes and overlapping
    density tails in its Hastings correction. Its fit never changes during MCMC.
    """
    def __init__(self, model, fit, centres, factor):
        means, factors, weights = [], [], []
        d = len(fit.x)
        axis = 6 if d == 7 else 2
        remaining = np.arange(d) != axis
        sd = np.sqrt(fit.cov[axis, axis])
        for centre in centres:
            base = fit.x + factor@centre
            axis_grid = np.linspace(max(model.lower[axis]+1e-7, base[axis]-4*sd),
                                    min(model.upper[axis]-1e-7, base[axis]+4*sd), 21)
            modes, sigmas, costs = [], [], []
            seed = base.copy()
            # Start at the middle and march to neighboring conditional modes.
            for indices in (range(10, 21), range(9, -1, -1)):
                seed = base.copy()
                for j in indices:
                    seed[axis] = axis_grid[j]
                    def residual(free):
                        state = seed.copy()
                        state[remaining] = free
                        return model.residual(state)
                    opt = least_squares(residual, seed[remaining],
                                        bounds=(model.lower[remaining], model.upper[remaining]),
                                        x_scale="jac", max_nfev=80)
                    seed[remaining] = opt.x
                    _, singular, vt = np.linalg.svd(opt.jac, full_matrices=False)
                    covariance = (vt.T / singular**2) @ vt
                    modes.append(seed.copy())
                    sigmas.append(np.linalg.cholesky(covariance))
                    costs.append(opt.cost)
            order = np.argsort(np.array(modes)[:, axis])
            modes = np.array(modes)[order]
            slopes = np.gradient(modes, axis_grid, axis=0)
            spacing = axis_grid[1] - axis_grid[0]
            for j, i in enumerate(order):
                local = np.zeros((d, d))
                local[remaining, :d-1] = sigmas[i]
                local[:, -1] = .70*spacing*slopes[j]
                means.append(modes[j])
                factors.append(local)
                weights.append(-costs[i] + np.log(np.diag(sigmas[i])).sum() + np.log(spacing))
        self.means = np.array(means)
        self.factors = np.array(factors)
        self.precision_roots = np.linalg.inv(self.factors)
        self.weights = np.exp(weights - logsumexp(weights))
        self.constants = weights - logsumexp(weights) - np.linalg.slogdet(self.factors)[1]
        self.d = d

    def draw(self, rng):
        idx = rng.choice(len(self.means), p=self.weights)
        return self.means[idx] + self.factors[idx] @ rng.normal(size=self.d)/np.sqrt(rng.chisquare(8)/8)

    def logpdf(self, x):
        deltas = np.einsum("nij,nj->ni", self.precision_roots, x-self.means)
        kernel = -.5*(8+self.d)*np.log1p(np.sum(deltas**2, axis=1)/8)
        return float(logsumexp(self.constants+kernel))


def sample_posterior(model, fit=None, *, seed=5201, chains=4, warmup=400, draws=800):
    """Draw from model.log_prob in native coordinates, including log drag when estimated.

    Alternate an independence MH proposal (a mixture of Student-t densities along
    conditional mode profiles) with a symmetric random walk. Both kernels are fixed throughout each run.
    """
    if chains < 2 or warmup < 0 or draws < 8:
        raise ValueError("Need at least two chains, eight draws and nonnegative warmup")
    if fit is None:
        from .posterior import fit_map
        fit = fit_map(model)
    d = len(fit.x)
    factor = _scaled_factor(fit.cov)
    centres, basins = _mode_centres(fit, factor)
    proposal = _Proposal(model, fit, centres, factor)

    paths = np.empty((chains, draws, d))
    posterior = np.empty((chains, draws))
    acceptance, initial = [], []
    for c, sequence in enumerate(np.random.SeedSequence(seed).spawn(chains)):
        rng = np.random.default_rng(sequence)
        # Spread starts over distinct competing MAP basins and jitter by local SDs.
        origin = centres[c % len(centres)]
        for _ in range(200):
            z = origin + 1.7*rng.normal(size=d)
            lp = model.log_prob(fit.x + factor@z)
            if np.isfinite(lp):
                break
        else:
            raise ValueError("Unable to initialize a chain in the posterior support")
        initial.append((fit.x + factor@z).tolist())
        current = fit.x + factor@z
        q = proposal.logpdf(current)
        accepted = 0
        for i in range(-warmup, draws):
            independence = rng.random() < .85
            if independence:
                other = proposal.draw(rng)
            else:
                other = current + factor@((2.38/np.sqrt(d))*rng.normal(size=d))
            proposal_lp = model.log_prob(other)
            other_q = proposal.logpdf(other)
            correction = q-other_q if independence else 0.
            if np.log(rng.random()) < proposal_lp-lp+correction:
                current, lp, q = other, proposal_lp, other_q
                accepted += i >= 0
            if i >= 0:
                paths[c, i] = current
                posterior[c, i] = lp
        acceptance.append(accepted/draws)

    combined = np.concatenate((paths, posterior[:, :, None]), axis=2)
    rhat, bulk, tail = np.array([_diagnostics(v) for v in np.moveaxis(combined, 2, 0)]).T
    if not np.isfinite(rhat).all():
        raise ValueError("Chain variance is zero; convergence diagnostics are undefined")
    peaks = [_marginal_peaks(v.ravel()) for v in np.moveaxis(paths, 2, 0)]
    diagnostics = dict(seed=seed, chains=chains, warmup_per_chain=warmup, draws_per_chain=draws,
                       diagnostic_variables=["x0_m", "y0_m", "z0_m", "vx0_m_s", "vy0_m_s", "vz0_m_s"]
                                            + (["log_k"] if d == 7 else []) + ["log_posterior"],
                       initial_points=initial, acceptance_rate=acceptance,
                       rhat=rhat.tolist(), ess_bulk=bulk.tolist(), ess_tail=tail.tolist(),
                       max_rhat=float(max(rhat)), min_bulk_ess=float(min(bulk)), min_tail_ess=float(min(tail)),
                       converged=bool(np.max(rhat)<1.01 and min(np.min(bulk), np.min(tail))>=400),
                       mode_screen=dict(map_basins=basins, competitive_basins=len(centres),
                                        marginal_kde_peak_locations=peaks,
                                        chain_parameter_medians=np.median(paths, axis=1).tolist()))
    return PosteriorSamples(paths, posterior, diagnostics)
