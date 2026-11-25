"""Single-stroke posterior density in launch state and natural log drag coefficient."""

from dataclasses import dataclass
from functools import lru_cache
from math import log

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import brentq, least_squares

from .flight import DEFAULT_DRAG, GRAVITY, vacuum_landing_time
from .inverse import LOWER, UPPER, PRIOR_MEAN, PRIOR_STD, _motion, _seeds


# Event integration uses fine output near contact and cubic Hermite interpolation
# of the two LSODA states bracketing the first downward crossing.
_EVENT_GRID = np.r_[np.arange(0, .10, .005), np.arange(.10, 4.11, .025)]


def _integrate(v, grid, k):
    initial = np.r_[np.zeros(3), v]
    if k == 0:
        t = grid[:, None]
        acc = np.array([0., 0., -GRAVITY])
        return np.column_stack((t*v + .5*t*t*acc, v+t*acc))
    if len(grid) == 1:
        return initial[None, :]
    return odeint(_motion, initial, grid, args=(k,), rtol=2e-9, atol=3e-11)


def _check_flight(theta, k):
    state = np.asarray(theta, dtype=float)
    if state.shape != (6,) or not np.isfinite(state).all() or not np.isfinite(k) or k < 0:
        raise ValueError("Expected a finite six-component launch and nonnegative drag")
    return state


def flight_states(theta, times, k=DEFAULT_DRAG):
    """Positions and velocities at requested seconds after launch (free flight ODE)."""
    theta = _check_flight(theta, k)
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError("Expected a vector of finite nonnegative times")
    grid, indices = np.unique(np.r_[0., times], return_inverse=True)
    states = _integrate(theta[3:], grid, k)[indices[1:]]
    states[:, :3] += theta[:3]
    return states


def _landing(theta, k, grid, states):
    if theta[2] == 0 and theta[5] <= 0:
        return 0., theta[:2].copy()
    if k == 0:
        t = vacuum_landing_time(theta[:3], theta[3:])
        return t, theta[:2]+t*theta[3:5]
    crossed = np.flatnonzero((states[1:, 2] + theta[2] <= 0) & (states[1:, 5] <= 0))
    if not len(crossed):
        # Outside the usual badminton flight durations, continue until bracketed.
        end = grid[-1]
        while end < 128:
            following = np.linspace(end, 2*end, int(np.ceil(end/.025))+1)
            extended = _integrate(states[-1, 3:], following-end, k)
            extended[:, :3] += states[-1, :3]
            hits = np.flatnonzero(extended[1:, 2]+theta[2] <= 0)
            if len(hits):
                grid, states, crossed = following, extended, hits
                break
            end, states = following[-1], extended
        else:
            raise ValueError("Cannot bracket a first ground contact within 128 seconds")
    i = int(crossed[0])
    h = grid[i+1] - grid[i]
    left, right = states[i], states[i+1]
    if abs(theta[2]+right[2]) <= 1e-13:
        return float(grid[i+1]), theta[:2]+right[:2]
    # Position = a + b*s + c*s^2 + d*s^3, where s in [0,1].
    a = theta[:3]+left[:3]
    b = h*left[3:]
    delta = right[:3]-left[:3]
    c = 3*delta-h*(2*left[3:]+right[3:])
    d = -2*delta+h*(left[3:]+right[3:])
    if a[2] == 0:
        # For an upward launch very near the floor the second root may occur
        # inside the first output interval. Remove the root at the launch.
        s = brentq(lambda u: b[2]+u*(c[2]+u*d[2]), 0., 1.)
    else:
        s = brentq(lambda u: a[2]+u*(b[2]+u*(c[2]+u*d[2])), 0., 1.)
    return float(grid[i]+h*s), (a + s*(b+s*(c+s*d)))[:2]


def landing_event(theta, k=DEFAULT_DRAG):
    """Return the first downward ground contact as (seconds, world xy in metres)."""
    theta = _check_flight(theta, k)
    if theta[2] < 0:
        raise ValueError("Launch height must be nonnegative")
    return _landing(theta, k, _EVENT_GRID, _integrate(theta[3:], _EVENT_GRID, k))


class PosteriorModel:
    """Truncated Gaussian launch prior and Gaussian log-k prior with Gaussian measurements.

    R may be one 2x2 covariance or one 2x2 covariance per observation. Otherwise
    sigma_px supplies standard deviations (scalar, per-frame or per-coordinate).
    Each external measurement in constraints is a (value, standard deviation) pair.
    """

    def __init__(self, camera, times, uv, *, sigma_px=1., R=None, fixed_k=None,
                 constraints=None, k_prior_median=DEFAULT_DRAG, log_k_sd=.35):
        times, uv = np.asarray(times, dtype=float), np.asarray(uv, dtype=float)
        if (times.ndim != 1 or uv.shape != (len(times), 2) or not np.isfinite(times).all()
                or np.any(times < 0) or np.any(np.diff(times) <= 0)):
            raise ValueError("Expected ordered distinct times since launch and paired pixel coordinates")
        valid = np.isfinite(uv).all(axis=1)
        if valid.sum() < 3 or np.any(~valid & ~np.isnan(uv).all(axis=1)):
            raise ValueError("At least three complete detections are required; mark gaps with paired NaNs")
        if R is None:
            sig = np.asarray(sigma_px, dtype=float)
            if sig.shape == times.shape:
                sig = sig[:, None]
            sig = np.broadcast_to(sig, uv.shape)[valid]
            if np.any(sig <= 0) or not np.isfinite(sig).all():
                raise ValueError("Pixel standard deviations must be finite and positive")
            self.whitener = np.eye(2)[None, :, :] / sig[:, :, None]
        else:
            cov = np.broadcast_to(np.asarray(R, dtype=float), (len(times), 2, 2))[valid]
            if not np.isfinite(cov).all() or not np.allclose(cov, cov.transpose(0, 2, 1), atol=1e-12, rtol=1e-12):
                raise ValueError("Pixel covariance must be finite and symmetric")
            try:
                self.whitener = np.linalg.inv(np.linalg.cholesky(cov))
            except np.linalg.LinAlgError as exc:
                raise ValueError("Pixel covariance must be positive definite") from exc
        if fixed_k is not None and (not np.isfinite(fixed_k) or fixed_k < 0):
            raise ValueError("Fixed drag must be finite and nonnegative")
        if not (np.isfinite(k_prior_median) and k_prior_median > 0 and np.isfinite(log_k_sd) and log_k_sd > 0):
            raise ValueError("Positive finite median k and log-k prior deviation are required")
        self.camera, self.times, self.uv, self.fixed_k = camera, times[valid], uv[valid], fixed_k
        self.airborne_until = self.times[-1]
        self.mean, self.std = PRIOR_MEAN.copy(), PRIOR_STD.copy()
        self.lower, self.upper = LOWER.copy(), UPPER.copy()
        if fixed_k is None:
            mu = log(k_prior_median)
            self.mean, self.std = np.r_[self.mean, mu], np.r_[self.std, log_k_sd]
            self.lower, self.upper = np.r_[self.lower, mu-6*log_k_sd], np.r_[self.upper, mu+6*log_k_sd]
        self.constraints = {}
        dimensions = {"player_xy": 2, "height": 1, "landing_xy": 2, "landing_time": 1}
        if constraints is not None and not isinstance(constraints, dict):
            raise ValueError("constraints must be a mapping from measurement names to [value, standard deviation]")
        for name, pair in (constraints or {}).items():
            if name not in dimensions:
                raise ValueError(f"Unknown measurement {name}")
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"{name} needs a [value, standard deviation] pair")
            value, sd = (np.atleast_1d(np.asarray(x, dtype=float)) for x in pair)
            if value.shape != (dimensions[name],):
                raise ValueError(f"Unexpected shape for {name} measurement")
            sd = np.broadcast_to(sd, value.shape)
            if not np.isfinite(value).all() or not np.isfinite(sd).all() or np.any(sd <= 0):
                raise ValueError(f"{name} needs finite measurements and positive standard deviations")
            self.constraints[name] = (value, sd)
        self.has_landing = bool({"landing_xy", "landing_time"} & self.constraints.keys())
        self.grid = np.unique(np.r_[0., self.times, _EVENT_GRID if self.has_landing else []])
        self.indices = np.searchsorted(self.grid, self.times)
        self._principal = camera.K[:2, 2]
        # Position perturbations in a finite-difference Jacobian share the same flight displacements.
        self._states = lru_cache(maxsize=24)(lambda vx, vy, vz, k: _integrate(np.array([vx, vy, vz]), self.grid, k))

    def unpack(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != self.mean.shape:
            raise ValueError("Unexpected posterior parameter dimension")
        return x[:6], self.fixed_k if self.fixed_k is not None else np.exp(x[6])

    def _evaluate(self, x, strict):
        theta, k = self.unpack(x)
        states = self._states(*theta[3:], k)
        world = states[self.indices, :3]+theta[:3]
        q = world @ self.camera.R.T + self.camera.t
        if strict and (world[-1, 2] < -1e-9 or np.any(q[:, 2] <= 1e-8)):
            return None
        xy = q[:, :2]/np.maximum(q[:, 2:], 1e-8)
        if self.camera.k1:
            kr2 = self.camera.k1*np.sum(xy*xy, axis=1, keepdims=True)
            if strict and np.any(1+3*kr2 <= 0):
                return None
            xy *= 1+kr2
        d = self.camera.focal_px*xy+self._principal-self.uv
        parts = [np.einsum("nij,nj->ni", self.whitener, d).ravel(), (x-self.mean)/self.std]
        predicted = {"player_xy": theta[:2], "height": theta[2]}
        if self.has_landing:
            predicted["landing_time"], predicted["landing_xy"] = _landing(theta, k, self.grid, states)
        for name, (value, sd) in self.constraints.items():
            parts.append((predicted[name]-value)/sd)
        return np.concatenate(parts)

    def residual(self, x):
        """Whitened pixel, prior and external-measurement residuals, smooth near the MAP."""
        return self._evaluate(np.asarray(x, dtype=float), False)

    def log_prob(self, x):
        """Log posterior up to a common constant in (r0,v0,log k) coordinates."""
        x = np.asarray(x, dtype=float)
        if (x.shape != self.mean.shape or not np.isfinite(x).all()
                or np.any(x < self.lower) or np.any(x > self.upper)):
            return -np.inf
        r = self._evaluate(x, True)
        return -.5*float(r@r) if r is not None else -np.inf

    def final_height(self, x):
        theta, k = self.unpack(x)
        return theta[2]+self._states(*theta[3:], k)[self.indices[-1], 2]


@dataclass
class Mode:
    initial: np.ndarray
    x: np.ndarray
    cost: float
    success: bool


@dataclass
class MAP:
    x: np.ndarray
    cov: np.ndarray
    starts: list
    rank: int
    condition: float


def fit_map(model, *, max_nfev=180):
    """Multistart MAP and Gauss–Newton covariance in the original parameter coordinates.

    condition is the Gauss–Newton precision condition number in prior-standardized
    coordinates. Physical feasibility is enforced on each candidate and by log_prob.
    """
    starts = []
    for i, seed6 in enumerate(_seeds(model.camera, model.times, model.uv)):
        seed = seed6 if model.fixed_k is not None else np.r_[seed6, model.mean[-1]+(i-1)*model.std[-1]]
        options = dict(x_scale=model.std, diff_step=1e-5, ftol=2e-9, xtol=2e-9, gtol=1e-7, max_nfev=max_nfev)
        opt = least_squares(model.residual, seed, bounds=(model.lower, model.upper), **options)
        x = opt.x
        if model.final_height(x) < -1e-9:
            # The first downward touchdown must not precede a trusted detection.
            # Reparameterize launch vz as a fraction of the permissible interval.
            # This retains feasible solutions when an unconstrained fit descends
            # so early that even the maximum launch height would be insufficient.
            def decode(p):
                state = p.copy()
                low_v, high_v = model.lower[5], model.upper[5]
                if model.fixed_k == 0:
                    t = model.airborne_until
                    minimum = np.clip(.5*GRAVITY*t-state[2]/t, low_v, high_v)
                else:
                    def height(vz):
                        state[5] = vz
                        return model.final_height(state)

                    if height(low_v) >= 0:
                        minimum = low_v
                    elif height(high_v) <= 0:
                        minimum = high_v
                    else:
                        minimum = brentq(height, low_v, high_v, xtol=1e-10)
                state[5] = minimum+p[5]*(high_v-minimum)
                return state

            def bounded_residual(p):
                state = decode(p)
                return np.r_[model.residual(state), min(model.final_height(state), 0.)/.01]

            low, high = model.lower.copy(), model.upper.copy()
            low[5], high[5] = 0., 1.
            p = x.copy()
            p[5] = .001
            options["x_scale"] = model.std.copy()
            options["x_scale"][5] = .25
            opt = least_squares(bounded_residual, p, bounds=(low, high), **options)
            x = decode(opt.x)
        cost = -model.log_prob(x)
        starts.append(Mode(seed.copy(), x.copy(), cost, bool(opt.success and np.isfinite(cost))))
    valid = [m for m in starts if m.success]
    if not valid:
        raise ValueError("No feasible MAP candidate converged")
    best = min(valid, key=lambda m: m.cost).x
    cov, rank, condition = local_curvature(model, best)
    return MAP(best.copy(), cov, starts, rank, condition)


def local_curvature(model, x):
    """Return local Gauss–Newton covariance, rank and prior-scaled precision condition."""
    x = np.asarray(x, dtype=float)
    step = np.maximum(np.abs(x), 1.)*2e-5
    # Central differences, shortened at parameter bounds. The covariance describes
    # local likelihood/prior curvature; sampling separately respects truncation.
    lo, hi = np.maximum(x-step, model.lower), np.minimum(x+step, model.upper)
    columns = []
    for j in range(len(x)):
        plus, minus = x.copy(), x.copy()
        plus[j], minus[j] = hi[j], lo[j]
        columns.append((model.residual(plus)-model.residual(minus))/(hi[j]-lo[j]))
    jac = np.column_stack(columns)*model.std
    _, s, vt = np.linalg.svd(jac, full_matrices=False)
    rank = int(np.linalg.matrix_rank(jac))
    if rank != len(x):
        raise ValueError("Local posterior curvature is rank deficient")
    scaled = (vt.T/s) @ (vt.T/s).T
    cov = model.std[:, None]*scaled*model.std
    return cov, rank, float((s[0]/s[-1])**2)
