"""Free flight in metres and seconds, with the ground at z = 0."""

from dataclasses import dataclass
from math import hypot, sqrt

import numpy as np
from scipy.optimize import brentq


GRAVITY = 9.81
DEFAULT_DRAG = 0.217


@dataclass
class Trajectory:
    """Samples from launch through landing, including the apex."""

    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    apex_index: int

    @property
    def speed(self):
        return np.linalg.norm(self.velocity, axis=1)

    @property
    def apex_time(self):
        return float(self.time[self.apex_index])

    @property
    def apex_position(self):
        return self.position[self.apex_index]


def _launch(position, velocity):
    r = np.asarray(position, dtype=float)
    v = np.asarray(velocity, dtype=float)
    if r.shape != (3,) or v.shape != (3,) or not (np.isfinite(r).all() and np.isfinite(v).all()):
        raise ValueError("position and velocity must be finite three-component vectors")
    if r[2] < 0:
        raise ValueError("launch height must be at or above the ground")
    return r, v


def vacuum_state(position, velocity, time):
    """Exact projectile position and velocity at a scalar or vector of times."""
    r, v = _launch(position, velocity)
    t = np.asarray(time, dtype=float)
    if t.ndim > 1 or not np.isfinite(t).all() or np.any(t < 0):
        raise ValueError("time must be a nonnegative finite scalar or vector")
    acceleration = np.array([0.0, 0.0, -GRAVITY])
    t = t[..., None]
    return r + t * v + 0.5 * t**2 * acceleration, v + t * acceleration


def vacuum_landing_time(position, velocity):
    """Time of the first downward ground contact for the exact projectile."""
    r, v = _launch(position, velocity)
    h, vz = r[2], v[2]
    discriminant_root = hypot(vz, sqrt(2 * GRAVITY * h))
    if vz < 0:
        # Rationalized root avoids cancellation for fast downward launches.
        return float(2 * h / (discriminant_root - vz))
    return float((vz + discriminant_root) / GRAVITY)


def _derivative(state, k):
    velocity = state[3:]
    acceleration = (-k * np.linalg.norm(velocity)) * velocity
    acceleration[2] -= GRAVITY
    return np.concatenate((velocity, acceleration))


def _rk4(state, h, k):
    a = _derivative(state, k)
    b = _derivative(state + (0.5 * h) * a, k)
    c = _derivative(state + (0.5 * h) * b, k)
    d = _derivative(state + h * c, k)
    return state + (h / 6) * (a + 2 * b + 2 * c + d)


def _event_offset(state, k, index, start, stop):
    # Reintegrate from the same left state for every candidate time. Linear
    # interpolation of the height would dominate the error in the landing point.
    return brentq(lambda h: _rk4(state, h, k)[index], start, stop,
                  xtol=np.finfo(float).tiny, maxiter=256)


def simulate(position, velocity, *, k=DEFAULT_DRAG, dt=0.01):
    """Integrate dr/dt = v, dv/dt = -g e_z - k |v| v until first landing.

    k is in inverse metres and dt in seconds. Regular samples are dt apart;
    the apex and first downward ground contact have their own localized times.
    A launch from ground level with nonpositive vertical velocity ends at t=0.
    """
    r, v = _launch(position, velocity)
    if not np.isscalar(k) or not np.isfinite(k) or k < 0:
        raise ValueError("k must be finite and nonnegative")
    if not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    k, dt = float(k), float(dt)
    state = np.concatenate((r, v))
    times, states = [0.0], [state]
    apex_index = 0
    if r[2] == 0 and v[2] <= 0:
        return Trajectory(np.asarray(times), r[None, :].copy(), v[None, :].copy(), apex_index)

    # The largest drag Jacobian eigenvalue in magnitude is 2 k |v|. Speed
    # is bounded by max(launch speed, terminal speed). Stay inside classical
    # RK4's negative-real stability interval (approximately [-2.785, 0]).
    if 2 * dt * max(k * hypot(*v), sqrt(k * GRAVITY)) > 2.78:
        raise ValueError("dt is too large for this drag coefficient and speed; reduce dt")

    def record(time, sample):
        # An event may fall on a regular sample to floating-point precision.
        if time == times[-1]:
            states[-1] = sample
        else:
            times.append(time)
            states.append(sample)

    for step in range(1, 1_000_001):
        t = (step - 1) * dt
        next_time = step * dt
        next_state = _rk4(state, dt, k)
        if not np.isfinite(next_state).all():
            raise ArithmeticError("RK4 produced a nonfinite state; check launch values and dt")
        lower = 0.0
        if state[5] > 0 and next_state[5] <= 0:
            lower = _event_offset(state, k, 5, 0.0, dt)
            apex = _rk4(state, lower, k)
            apex[5] = 0.0
            record(min(next_time, t + lower), apex)
            apex_index = len(times) - 1
        if next_state[2] <= 0 and next_state[5] < 0:
            # An upward launch from ground can have both events in one step.
            # Bracketing after the apex excludes the initial ground-level root.
            offset = _event_offset(state, k, 2, lower, dt)
            landing = _rk4(state, offset, k)
            landing[2] = 0.0
            record(min(next_time, t + offset), landing)
            break
        if next_time > times[-1]:
            record(next_time, next_state)
        state = next_state
    else:
        raise ValueError("flight exceeded one million integration steps; use a larger stable dt")

    samples = np.asarray(states)
    return Trajectory(np.asarray(times), samples[:, :3], samples[:, 3:], apex_index)
