"""Probe distinct posterior modes of one six-frame drop shot using numerical curvature."""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .assimilation_cases import make_case, make_model
from .posterior import fit_map, flight_states, landing_event
from .reconstruct import ROOT, write_json


def _curvature(model, x):
    # Whiten the local coordinates with the Gauss–Newton metric for comparable
    # difference steps. Differentiate the actual log density including its
    # residual-curvature term, which Gauss–Newton would omit away from an exact fit.
    eps = np.maximum(1., abs(x))*2e-5
    dx = np.diag(eps)
    jac = np.column_stack([(model.residual(x+d)-model.residual(x-d))/(2*e)
                           for d, e in zip(dx, eps)])
    _, s, vt = np.linalg.svd(jac, full_matrices=False)
    axes = vt/s[:, None]
    fits = []
    for h in (.004, .008):
        base = -model.log_prob(x)
        cost = lambda d: -model.log_prob(x+d)
        steps = axes*h
        gradient = np.array([(cost(d)-cost(-d))/(2*h) for d in steps])
        hess = np.zeros((len(x), len(x)))
        for i, a in enumerate(steps):
            hess[i, i] = (cost(a)+cost(-a)-2*base)/h**2
            for j in range(i):
                b = steps[j]
                hess[i, j] = hess[j, i] = (cost(a+b)-cost(a-b)-cost(b-a)+cost(-a-b))/(4*h*h)
        fits.append(dict(step=h, gradient_norm=float(np.linalg.norm(gradient)),
                         eigenvalues=np.linalg.eigvalsh(hess).tolist()))
    return fits, axes


def _refit(model, x):
    opt = least_squares(model.residual, x, bounds=(model.lower, model.upper),
                        x_scale=model.std, ftol=1e-11, gtol=1e-8, xtol=1e-11,
                        max_nfev=300, diff_step=3e-6)
    if not opt.success or not np.isfinite(model.log_prob(opt.x)):
        raise ValueError("A local posterior refinement did not converge inside the support")
    return opt.x


def _profile(model, modes):
    # Fix launch vx in m/s, then reestimate every other launch component and k.
    # Use both branches independently and continue the fit in both directions.
    # The resulting conditional optima are numerical candidates, not a proof
    # that every solution on this slice or each connecting path has been found.
    free = np.arange(len(modes[0])) != 3
    values = np.linspace(modes[1][3], modes[0][3], 25)
    best = np.full(len(values), np.inf)
    for order in (range(len(values)), reversed(range(len(values)))):
        previous = None
        for i in order:
            def unpack(y):
                x = np.empty(len(free))
                x[3], x[free] = values[i], y
                return x

            candidates = [m[free] for m in modes]
            if previous is not None:
                candidates.append(previous)
            solutions = []
            for start in candidates:
                opt = least_squares(lambda y: model.residual(unpack(y)), start,
                                    bounds=(model.lower[free], model.upper[free]),
                                    x_scale=model.std[free], diff_step=1e-5,
                                    ftol=1e-9, xtol=1e-9, gtol=1e-7, max_nfev=180)
                if opt.success:
                    solutions.append((-model.log_prob(unpack(opt.x)), opt.x))
            if not solutions:
                raise ValueError("Profile slice lacks a converged candidate")
            cost, previous = min(solutions, key=lambda row: row[0])
            best[i] = min(best[i], cost)
    if not np.isfinite(best).all():
        raise ValueError("Profile slice lacks a feasible candidate")
    return [dict(vx_m_s=float(v), cost=float(c)) for v, c in zip(values, best)]


def diagnose():
    case = make_case("drop", 5100)
    model = make_model(case, "short_launch")
    fit = fit_map(model)
    viable = [s.x for s in fit.starts if s.success]
    modes = []
    for start in sorted(viable, key=lambda x: -model.log_prob(x)):
        point = _refit(model, start)
        if all(np.linalg.norm((point-other)/model.std) > .01 for other in modes):
            modes.append(point)
    if len(modes) != 2:
        raise ValueError(f"Expected two distinct reproducible mode candidates, found {len(modes)}")
    mode_rows, projected, landings = [], [], []
    rng = np.random.default_rng(5169)
    for m in modes:
        theta, k = model.unpack(m)
        pixels = model.camera.project(flight_states(theta, model.times, k)[:, :3])
        end, xy = landing_event(theta, k)
        projected.append(pixels)
        landings.append(xy)
        curvature, axes = _curvature(model, m)
        perturbations = []
        for _ in range(6):
            neighbor = m+(rng.normal(size=len(m))*.15) @ axes
            point = _refit(model, neighbor)
            perturbations.append(float(np.linalg.norm((point-m)/model.std)))
        mode_rows.append(dict(x=m.tolist(), k_m_inv=float(k), cost=float(-model.log_prob(m)),
                              image_rmse_px=float(np.sqrt(np.mean(np.sum((pixels-model.uv)**2, axis=1)))),
                              landing_xy_m=xy.tolist(), landing_time_s=end,
                              original_seeds_returning=int(sum(np.linalg.norm((x-m)/model.std)<.01 for x in viable)),
                              curvature_in_GN_whitened_coordinates=curvature,
                              max_refit_distance_in_prior_sd=max(perturbations)))
    profile = _profile(model, modes)
    ridge = max(profile, key=lambda r: r["cost"])
    all_measurements = make_model(case, "short_all")
    after = fit_map(all_measurements)
    return dict(case="drop", seed=5100, setting="short_launch", observed_frames=len(model.times),
                observed_span_s=float(np.ptp(model.times)), perturbations_per_mode=6,
                modes=mode_rows, landing_gap_m=float(np.linalg.norm(landings[0]-landings[1])),
                pairwise_image_gap_rms_px=float(np.sqrt(np.mean(np.sum((projected[0]-projected[1])**2, axis=1)))),
                profile_fixed_vx=profile, profile_highest_cost=ridge,
                profiled_cost_rise_above_secondary=float(ridge["cost"]-mode_rows[1]["cost"]),
                with_landing_measurements=dict(multistart_costs=[s.cost for s in after.starts],
                    max_launch_vx_distance_m_s=max(abs(s.x[3]-after.x[3]) for s in after.starts)),
                interpretation="Local density and finite profile evidence in launch/log-k coordinates. "
                               "Start counts and peak density ratios do not measure posterior mass. "
                               "Stroke direction and in-court landing are not supplied.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/stage5/modes.json")
    out = parser.parse_args().out
    report = diagnose()
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, report)
    print(f"Two six-frame mode candidates, predicted landings {report['landing_gap_m']:.2f} m apart -> {out}")
