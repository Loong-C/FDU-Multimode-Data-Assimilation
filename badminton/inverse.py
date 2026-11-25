"""Estimate one shuttle launch from timestamped monocular observations in pixels."""

from dataclasses import dataclass
from functools import lru_cache
from math import hypot

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import brentq, least_squares

from .court import DOUBLES_HALF_WIDTH, HALF_LENGTH
from .flight import DEFAULT_DRAG, GRAVITY


# Bounds allow contact beyond the painted court. Velocity components are in m/s.
LOWER = np.array([-HALF_LENGTH-3, -DOUBLES_HALF_WIDTH-2.5, 0, -130, -130, -130])
UPPER = np.array([HALF_LENGTH+3, DOUBLES_HALF_WIDTH+2.5, 5.5, 130, 130, 130])
PRIOR_MEAN = np.array([0.0, 0, 2, 0, 0, 0])
PRIOR_STD = np.array([8.0, 5, 3, 55, 55, 55])


def _motion(state, time, k):
    # Scalar RHS keeps the overhead of repeated short LSODA integrations small.
    vx, vy, vz = state[3:]
    drag = -k * hypot(vx, vy, vz)
    return vx, vy, vz, drag*vx, drag*vy, drag*vz-GRAVITY


def _displacement(velocity, grid, k):
    if len(grid) == 1:
        return np.zeros((1, 3))
    state = np.r_[np.zeros(3), velocity]
    return odeint(_motion, state, grid, args=(k,), rtol=2e-10, atol=1e-11)[:, :3]


def _time_grid(times, k):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError("Expected a vector of finite times since launch in seconds")
    if not np.isscalar(k) or not np.isfinite(k) or k < 0:
        raise ValueError("k must be finite and nonnegative")
    grid, indices = np.unique(np.r_[0., times], return_inverse=True)
    return grid, indices[1:]


def predict_positions(theta, times, k=DEFAULT_DRAG):
    """Sample the flight ODE at times since launch, including an extrapolation below ground."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (6,) or not np.isfinite(theta).all():
        raise ValueError("theta must contain a finite initial position and velocity")
    grid, indices = _time_grid(times, k)
    return theta[:3] + _displacement(theta[3:], grid, k)[indices]


@dataclass
class StartResult:
    initial: np.ndarray
    theta: np.ndarray
    cost: float
    success: bool
    nfev: int
    rmse_px: float
    message: str


@dataclass
class FitResult:
    theta: np.ndarray
    starts: list


def _seeds(camera, times, pixels):
    center, rays = camera.rays(pixels)
    # A local robust angular velocity uses only early image measurements.
    window = min(len(times), max(3, min(6, np.searchsorted(times, times[0]+0.20))))
    i, j = np.triu_indices(window, k=1)
    angular = np.median((rays[j]-rays[i])/(times[j]-times[i])[:, None], axis=0)
    depth_hint = max(np.linalg.norm(center-[0, 0, 2]), 1.)
    for height, radial_speed in ((0.8, -9.), (2.0, 6.), (3.4, 24.)):
        depth = depth_hint
        if abs(rays[0, 2]) > 1e-8:
            intersection = (height-center[2])/rays[0, 2]
            if intersection > 0:
                depth = np.clip(intersection, 0.3*depth_hint, 2*depth_hint)
        r_first = center + depth*rays[0]
        velocity = depth*angular + radial_speed*rays[0]
        velocity[2] += GRAVITY*times[0]
        position = r_first - times[0]*velocity
        position[2] += 0.5*GRAVITY*times[0]**2
        yield np.clip(np.r_[position, velocity], LOWER+1e-5, UPPER-1e-5)


def fit_track(camera, times, uv, *, k=DEFAULT_DRAG, sigma_px=1.0, robust=False,
              launch_height_m=None):
    """Fit launch position and velocity from at least three ordered detections.

    Times are seconds from a known contact time. Drop detections or mark both
    coordinates NaN to represent gaps. sigma_px is the per-coordinate standard
    deviation in pixels (scalar, one per frame, or N x 2). Fixed launch height
    optionally profiles competing depth hypotheses at a user-specified height.
    """
    times = np.asarray(times, dtype=float)
    uv = np.asarray(uv, dtype=float)
    if times.ndim != 1 or uv.shape != (len(times), 2):
        raise ValueError("Provide a time and a pixel pair for each frame")
    if not np.isfinite(times).all() or np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError("Observation times must be finite, nonnegative and strictly increasing")
    detected = np.isfinite(uv).all(axis=1)
    if np.any(~detected & ~np.isnan(uv).all(axis=1)) or np.count_nonzero(detected) < 3:
        raise ValueError("At least three pixel pairs are required; use two NaNs for missing points")
    sigma = np.asarray(sigma_px, dtype=float)
    if sigma.shape == times.shape:
        sigma = sigma[:, None]
    sigma = np.broadcast_to(sigma, uv.shape)[detected]
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("Observed pixel standard deviations must be finite and positive")
    times, uv = times[detected], uv[detected]
    grid, indices = _time_grid(times, k)
    fixed = launch_height_m is not None
    if fixed and not (LOWER[2] <= launch_height_m <= UPPER[2]):
        raise ValueError("Profile height is outside the allowed launch-height interval")
    free = np.ones(6, dtype=bool)
    free[2] = not fixed
    principal = camera.K[:2, 2]

    @lru_cache(maxsize=16)
    def displacement(vx, vy, vz):
        return _displacement((vx, vy, vz), grid, k)[indices]

    def positions(theta):
        return theta[:3] + displacement(*theta[3:])

    def unpack(x):
        theta = np.empty(6)
        theta[free] = x
        if fixed:
            theta[2] = launch_height_m
        return theta

    def residual(theta):
        world = positions(theta)
        camera_xyz = world @ camera.R.T + camera.t
        depth = camera_xyz[:, 2]
        # A finite extension and a depth barrier let trial steps return to the camera frustum.
        q = camera_xyz[:, :2] / np.maximum(depth, 1e-4)[:, None]
        if camera.k1:
            q *= 1+camera.k1*np.sum(q*q, axis=1, keepdims=True)
        error = (principal + camera.focal_px*q - uv)/sigma
        if robust:
            # Signed square root of the soft-L1 cost at 3 sigma. The remaining
            # residual blocks stay quadratic and contribute their usual Jacobian.
            error *= np.sqrt(2/(1 + np.hypot(1., error/3)))
        prior = (theta-PRIOR_MEAN)/PRIOR_STD
        return np.r_[error.ravel(), prior,
                     np.minimum(depth-1e-4, 0.)/0.001]

    @lru_cache(maxsize=48)
    def floor_velocity(height, vx, vy):
        # Minimum launch vz for which z(last detection) >= 0 at this height
        # and horizontal speed. No detection is forced to lie on the floor.
        def final_height(vz):
            return height + displacement(vx, vy, vz)[-1, 2]

        if final_height(LOWER[5]) >= 0:
            return LOWER[5]
        if final_height(UPPER[5]) < 0:
            return UPPER[5]
        return brentq(final_height, LOWER[5], UPPER[5], xtol=1e-9)

    def constrained_state(x):
        theta = unpack(x)
        lower = floor_velocity(theta[2], theta[3], theta[4])
        theta[5] = lower + (x[-1]-1.)*(UPPER[5]-lower)
        return theta

    def constrained_residual(x):
        theta = constrained_state(x)
        # Only needed if even the largest admissible vz cannot reach the last
        # detection for trial horizontal speed and height. Reject such final fits.
        return np.r_[residual(theta), min(0., positions(theta)[-1, 2])/0.01]

    options = dict(ftol=1e-9, xtol=1e-9, gtol=1e-8, max_nfev=180, diff_step=1e-5)
    starts = []
    for seed in _seeds(camera, times, uv):
        if fixed:
            seed[2] = launch_height_m
        low, high = LOWER[free], UPPER[free]
        scale = PRIOR_STD[free]
        opt = least_squares(lambda x: residual(unpack(x)), seed[free], bounds=(low, high),
                            x_scale=scale, **options)
        nfev = opt.nfev
        theta = unpack(opt.x)
        if positions(theta)[-1, 2] < -1e-8:
            # Constrain only when the unconstrained fit lands before the last
            # detection. The bounded fraction parameter avoids a stiff floor
            # penalty. Flight has no interior height minimum, so testing the two
            # endpoints suffices. The last observation may be freely airborne.
            lower = floor_velocity(theta[2], theta[3], theta[4])
            fraction = (theta[5]-lower)/max(UPPER[5]-lower, 1e-8)
            x = opt.x.copy()
            x[-1] = 1 + np.clip(fraction, 0.001, 0.999)
            low[-1], high[-1], scale[-1] = 1., 2., 0.25
            opt = least_squares(constrained_residual, x, bounds=(low, high),
                                x_scale=scale, **options)
            nfev += opt.nfev
            theta = constrained_state(opt.x)
        world = positions(theta)
        message = opt.message
        success = bool(opt.success)
        rmse = float("inf")
        try:
            error = camera.project(world)-uv
            rmse = float(np.sqrt(np.mean(np.sum(error*error, axis=1))))
            if world[-1, 2] < -1e-7:
                raise ValueError("A detection lies below the floor")
        except ValueError as exc:
            success, message = False, str(exc)
        starts.append(StartResult(seed.copy(), theta, float(opt.cost), success,
                                  int(nfev), rmse, message))
    valid = [start for start in starts if start.success]
    if not valid:
        raise ValueError("No converged physically admissible launch: " + starts[0].message)
    return FitResult(min(valid, key=lambda start: start.cost).theta.copy(), starts)
