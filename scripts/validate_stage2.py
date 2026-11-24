"""Check free-flight dynamics against analytic and independent numerical solutions."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from badminton.flight import simulate, vacuum_landing_time, vacuum_state


G = 9.81
K = 0.217


def check_samples(flight, position, velocity):
    n = len(flight.time)
    if flight.position.shape != (n, 3) or flight.velocity.shape != (n, 3) or flight.speed.shape != (n,):
        raise ValueError("Time, position, velocity and speed sample shapes disagree")
    if n < 2 or flight.time[0] != 0 or not np.all(np.diff(flight.time) > 0):
        raise ValueError("Flight must have an increasing time axis starting at zero")
    if not all(np.isfinite(a).all() for a in (flight.time, flight.position, flight.velocity, flight.speed,
                                                flight.apex_position, flight.apex_time)):
        raise ValueError("Nonfinite flight sample")
    np.testing.assert_allclose(flight.position[0], position, rtol=0, atol=1e-12)
    np.testing.assert_allclose(flight.velocity[0], velocity, rtol=0, atol=1e-12)
    np.testing.assert_allclose(flight.speed, np.linalg.norm(flight.velocity, axis=1), rtol=1e-12, atol=1e-12)
    if (min(flight.position[:, 2]) < -1e-10 or abs(flight.position[-1, 2]) > 1e-10
            or flight.velocity[-1, 2] >= 0 or not 0 <= flight.apex_time <= flight.time[-1]):
        raise ValueError("Invalid airborne interval, apex time or ground event")


def state_errors(flight, reference_position, reference_velocity, apex_time, apex, landing_time, landing):
    return {
        "max_position_error_m": float(np.linalg.norm(flight.position - reference_position, axis=1).max()),
        "max_velocity_error_m_s": float(np.linalg.norm(flight.velocity - reference_velocity, axis=1).max()),
        "apex_time_error_s": abs(float(flight.apex_time) - apex_time),
        "apex_position_error_m": float(np.linalg.norm(flight.apex_position - apex)),
        "landing_time_error_s": abs(float(flight.time[-1]) - landing_time),
        "landing_position_error_m": float(np.linalg.norm(flight.position[-1] - landing)),
    }


def vacuum_case(position, velocity, dt):
    p = np.asarray(position, dtype=float)
    v = np.asarray(velocity, dtype=float)
    acceleration = np.array([0.0, 0.0, -G])
    flight = simulate(p.copy(), v.copy(), k=0, dt=dt)
    check_samples(flight, p, v)

    def position_at(t):
        return p + v * t + 0.5 * acceleration * t**2

    t = flight.time[:, None]
    apex = max(0.0, v[2] / G)
    landing = (v[2] + math.sqrt(v[2] ** 2 + 2 * G * p[2])) / G
    expected_position, expected_velocity = position_at(t), v + t * acceleration
    errors = state_errors(flight, expected_position, expected_velocity, apex, position_at(apex),
                          landing, position_at(landing))
    model_position, model_velocity = vacuum_state(p, v, flight.time)
    touchdown, impact_velocity = vacuum_state(p, v, landing)
    if model_position.shape != expected_position.shape or model_velocity.shape != expected_velocity.shape:
        raise ValueError("Analytic model must return one 3-vector per sample time")
    if touchdown.shape != (3,) or impact_velocity.shape != (3,):
        raise ValueError("Analytic model at a scalar time must return two 3-vectors")
    errors.update(
        analytic_model_max_position_error_m=float(np.linalg.norm(model_position - expected_position, axis=1).max()),
        analytic_model_max_velocity_error_m_s=float(np.linalg.norm(model_velocity - expected_velocity, axis=1).max()),
        analytic_model_landing_time_error_s=abs(float(vacuum_landing_time(p, v)) - landing),
        analytic_model_touchdown_error_m=float(np.linalg.norm(touchdown - position_at(landing))),
        analytic_model_impact_velocity_error_m_s=float(np.linalg.norm(impact_velocity - (v + acceleration * landing))),
    )
    return errors


def immediate_ground_contact():
    position = [1.0, -2.0, 0.0]
    for vz in (0.0, -3.0):
        velocity = [4.0, 0.5, vz]
        for k in (0, K):
            flight = simulate(position, velocity, k=k, dt=0.03)
            if (not np.array_equal(flight.time, [0.0]) or not np.array_equal(flight.position, [position])
                    or not np.array_equal(flight.velocity, [velocity]) or flight.apex_time != 0
                    or not np.array_equal(flight.apex_position, position)
                    or not np.allclose(flight.speed, [np.linalg.norm(velocity)], rtol=1e-12, atol=0)):
                return False
        if vacuum_landing_time(position, velocity) != 0:
            return False
    return True


def vertical_drag_case():
    # dv/dt = -g - k v² on ascent and -g + k v² on descent.
    position = [1.0, -2.0, 1.4]
    velocity = [0.0, 0.0, 12.0]
    flight = simulate(position, velocity, k=K, dt=0.005)
    check_samples(flight, position, velocity)
    terminal = math.sqrt(G / K)
    frequency = math.sqrt(G * K)
    launch_angle = math.atan(velocity[2] / terminal)
    apex_time = launch_angle / frequency
    height = position[2] - math.log(math.cos(launch_angle)) / K
    landing_time = apex_time + math.acosh(math.exp(K * height)) / frequency

    angle = launch_angle - frequency * np.minimum(flight.time, apex_time)
    fall_time = np.maximum(flight.time - apex_time, 0)
    z = height + (np.log(np.cos(angle)) - np.log(np.cosh(frequency * fall_time))) / K
    vz = terminal * (np.tan(angle) - np.tanh(frequency * fall_time))
    p = np.tile(position, (len(z), 1))
    p[:, 2] = z
    v = np.zeros_like(p)
    v[:, 2] = vz
    return state_errors(flight, p, v, apex_time, [*position[:2], height], landing_time, [*position[:2], 0.0])


def step_study():
    position = [-5.4, -1.0, 2.4]
    velocity = [23.0, 3.0, 17.0]

    def dynamics(_t, state):
        v = state[3:]
        acceleration = -K * np.linalg.norm(v) * v
        acceleration[2] -= G
        return np.r_[v, acceleration]

    def apex(_t, state):
        return state[5]

    def ground(_t, state):
        return state[2]

    apex.direction = ground.direction = -1
    ground.terminal = True
    reference = solve_ivp(dynamics, (0, 10), position + velocity, method="DOP853",
                          rtol=1e-12, atol=1e-13, max_step=0.02, dense_output=True, events=(apex, ground))
    if not reference.success or any(len(t) != 1 for t in reference.t_events):
        raise RuntimeError("Independent reference did not resolve both apex and landing")
    reference_apex, reference_landing = (states[0, :3] for states in reference.y_events)
    apex_time, landing_time = (float(t[0]) for t in reference.t_events)
    rows = []
    for dt in (0.04, 0.02, 0.01, 0.005):
        flight = simulate(position, velocity, k=K, dt=dt)
        check_samples(flight, position, velocity)
        # The event error is compared separately. Compare path samples only on the common flight interval.
        common = flight.time <= landing_time
        states = reference.sol(flight.time[common]).T
        rows.append({
            "dt_s": dt, "samples": len(flight.time),
            "landing_time_s": float(flight.time[-1]), "landing_x_m": float(flight.position[-1, 0]),
            "landing_y_m": float(flight.position[-1, 1]),
            "landing_time_error_s": abs(float(flight.time[-1]) - landing_time),
            "landing_error_m": float(np.linalg.norm(flight.position[-1] - reference_landing)),
            "apex_time_error_s": abs(flight.apex_time - apex_time),
            "apex_error_m": float(np.linalg.norm(flight.apex_position - reference_apex)),
            "max_position_error_m": float(np.linalg.norm(flight.position[common] - states[:, :3], axis=1).max()),
            "max_velocity_error_m_s": float(np.linalg.norm(flight.velocity[common] - states[:, 3:], axis=1).max()),
        })
    reference_info = {
        "initial_position_m": position, "initial_velocity_m_s": velocity, "drag_per_m": K,
        "method": "SciPy DOP853", "rtol": 1e-12, "atol": 1e-13, "max_step_s": 0.02,
        "apex_time_s": apex_time, "apex_position_m": reference_apex.tolist(),
        "landing_time_s": landing_time, "landing_position_m": reference_landing.tolist(),
    }
    return reference_info, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/stage2")
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    cases = {"rising": ([-3.5, 1.2, 2.1], [8.0, -2.3, 6.4], 0.031),
             "horizontal": ([0.3, -0.7, 2.2], [6.3, 2.0, 0.0], 0.031),
             "descending": ([1.2, 0.5, 3.1], [-5.3, 1.2, -4.8], 0.031),
             "rising_from_ground": ([0.0, -1.0, 0.0], [2.0, 1.0, 8.0], 0.031),
             "apex_and_landing_in_first_step": ([1.0, -1.0, 0.0], [2.0, -1.0, 0.5], 0.2)}
    vacuum = {name: vacuum_case(p, v, dt) for name, (p, v, dt) in cases.items()}
    vertical = vertical_drag_case()
    reference, steps = step_study()
    order = [math.log2(a["landing_error_m"] / b["landing_error_m"]) for a, b in zip(steps, steps[1:])]
    checks = {
        "vacuum_all_errors_below_1e-8_SI": all(max(errors.values()) < 1e-8 for errors in vacuum.values()),
        "immediate_ground_contact_zero_or_negative_vertical_speed": immediate_ground_contact(),
        "vertical_drag_all_errors_below_1e-6_SI": max(vertical.values()) < 1e-6,
        "landing_error_order_above_3_on_last_two_halvings": min(order[-2:]) > 3.0,
        "dt_0.01_position_error_below_1e-4_m": steps[2]["max_position_error_m"] < 1e-4,
        "dt_0.01_landing_time_error_below_1e-5_s": steps[2]["landing_time_error_s"] < 1e-5,
    }
    report = {"passed": all(checks.values()), "checks": checks, "vacuum_errors": vacuum,
              "vertical_drag_errors": vertical, "step_study_reference": reference,
              "step_study_landing_error_orders": order}
    with (out / "steps.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, list(steps[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(steps)
    (out / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(f"Validation failed: {[key for key, ok in checks.items() if not ok]}")
    print(f"Verified {len(cases)} vacuum cases, vertical drag, and {len(steps)} fixed steps. Results in {out}")


if __name__ == "__main__":
    main()
