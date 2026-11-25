"""Reproduce the monocular single-stroke recovery and identifiability experiments."""

import argparse
from pathlib import Path

import numpy as np

from .court import HALF_LENGTH, SINGLES_HALF_WIDTH
from .flight import simulate
from .inverse import fit_track, predict_positions
from .inverse_cases import iter_cases
from .inverse_plots import render_results
from .reconstruct import ROOT, write_json, write_rows
from .simulate import summarize

STATE_FIELDS = ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s")


def rms(vectors):
    return float(np.sqrt(np.mean(np.sum(vectors**2, axis=1))))


def fit_observations(case, height=None):
    # This is the only interface between generated observations and the solver.
    return fit_track(case.camera, case.times, case.uv, k=case.k, sigma_px=case.sigma_px,
                     robust=case.robust, launch_height_m=height)


def score(case, fit):
    theta = fit.theta
    flight = simulate(theta[:3], theta[3:], k=case.k, dt=0.002)
    predicted_uv = case.camera.project(predict_positions(theta, case.times, case.k))
    # Score the complete shared airborne interval, including times lacking detections.
    # Score touchdown separately so an early landing is visible in the results.
    stop = min(flight.time[-1], case.truth_time[-1])
    t = np.r_[case.truth_time[case.truth_time < stop], stop]
    truth = np.column_stack([np.interp(t, case.truth_time, c) for c in case.truth_xyz.T])
    delta = predict_positions(theta, t, case.k) - truth
    error_3d = np.sqrt(np.trapezoid(np.sum(delta**2, axis=1), t) / stop)
    successful = [s for s in fit.starts if s.success]
    row = dict(case=case.name, shot=case.shot, view=case.view, condition=case.condition,
               seed=case.seed, k_m_inv=case.k, sigma_px=case.sigma_px, robust=int(case.robust),
               video_frames=case.n_video_frames, visible_frames=case.n_visible_frames,
               observations=len(case.times), bad_detections=int(np.sum(case.outlier_mask)),
               first_observation_s=case.times[0], last_observation_s=case.times[-1],
               observed_rmse_px=rms(predicted_uv-case.uv),
               clean_image_rmse_px=rms(predicted_uv-case.clean_uv), position_rmse_m=float(error_3d),
               launch_position_error_m=np.linalg.norm(theta[:3]-case.truth_theta[:3]),
               launch_velocity_error_m_s=np.linalg.norm(theta[3:]-case.truth_theta[3:]),
               apex_height_error_m=abs(flight.apex_position[2]-case.truth_xyz[:, 2].max()),
               landing_xy_error_m=np.linalg.norm(flight.position[-1, :2]-case.truth_xyz[-1, :2]),
               landing_time_error_s=abs(flight.time[-1]-case.truth_time[-1]),
               scored_duration_s=stop, converged_starts=len(successful),
               max_start_position_gap_m=max(np.linalg.norm(s.theta[:3]-theta[:3]) for s in successful))
    row.update(zip(STATE_FIELDS, theta))
    track = dict(theta=theta, times=flight.time, xyz=flight.position, uv_fit=predicted_uv,
                 position_rmse_m=row["position_rmse_m"], observed_rmse_px=row["observed_rmse_px"])
    return row, track


def compare_depth_hypotheses(case, track, paired_full):
    """Find two plausible strokes using only the first four detections and court geometry."""
    rows, eligible = [], []
    for h in np.arange(0.5, 4.1, 0.5):
        fit = fit_observations(case, height=h)
        flight = simulate(fit.theta[:3], fit.theta[3:], k=case.k, dt=0.002)
        uv = case.camera.project(predict_positions(fit.theta, case.times, case.k))
        err = rms(uv-case.uv)
        geom = summarize(case.name, case.shot, flight, case.k, 0.002)
        landing_gap = float(np.linalg.norm(track["xyz"][-1, :2]-flight.position[-1, :2]))
        shift = rms(uv-track["uv_fit"])
        on_court = bool(geom["landing_in_opposite_singles"] and geom.get("net_clearance_m", 0) > 0)
        on_court &= bool(abs(fit.theta[0]) <= HALF_LENGTH and abs(fit.theta[1]) <= SINGLES_HALF_WIDTH)
        admissible = on_court and err <= 1.5*case.sigma_px and shift <= case.sigma_px
        entry = dict(height_m=float(h), theta=fit.theta.tolist(), short_rmse_px=err,
                     prediction_gap_px=shift, landing_gap_m=landing_gap,
                     launch_gap_m=float(np.linalg.norm(track["theta"][:3]-fit.theta[:3])),
                     opposite_singles_and_clears_net=on_court, eligible=bool(admissible))
        rows.append(entry)
        if admissible:
            eligible.append((entry, flight, uv))
    if not eligible:
        raise ValueError("No profiled trajectory fits the short observations within the noise allowance")
    # Select by disagreement in predicted landing, without the reference 3-D flight.
    selected, flight, uv = max(eligible, key=lambda item: item[0]["landing_gap_m"])
    # Only after selection, check the same height hypothesis against the full measurements.
    full = fit_observations(paired_full, height=selected["height_m"])
    full_uv = paired_full.camera.project(predict_positions(full.theta, paired_full.times, paired_full.k))
    report = dict(case_name=case.name, paired_full_case=paired_full.name,
                  best_theta=track["theta"].tolist(), best_short_rmse_px=track["observed_rmse_px"],
                  selected=selected, full_refit_at_selected_height_rmse_px=rms(full_uv-paired_full.uv),
                  profiles=rows)
    plot = dict(case_name=case.name, alternative_xyz=flight.position, alternative_times=flight.time,
                alternative_uv=uv, alternative_rmse_px=selected["short_rmse_px"])
    return report, plot


def run(out):
    cases = list(iter_cases())
    rows, tracks, starts, detections = [], [], [], []
    for case in cases:
        fit = fit_observations(case)
        row, track = score(case, fit)
        rows.append(row)
        tracks.append(track)
        for i, s in enumerate(fit.starts):
            record = dict(case=case.name, start=i+1, converged=int(s.success), cost=s.cost,
                          observed_rmse_px=s.rmse_px if np.isfinite(s.rmse_px) else "", nfev=s.nfev)
            record.update(("seed_"+field, v) for field, v in zip(STATE_FIELDS, s.initial))
            record.update(zip(STATE_FIELDS, s.theta))
            starts.append(record)
        detections.extend(dict(case=case.name, frame=int(frame), t_s=t, u_px=u, v_px=v)
                          for frame, t, (u, v) in zip(case.frame_index, case.times, case.uv))
    lookup = {c.name: (c, t) for c, t in zip(cases, tracks)}
    short, track = lookup["clear_oblique_short_401"]
    report, comparison = compare_depth_hypotheses(short, track, lookup["clear_oblique_noise_401"][0])
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / "scores.csv", rows)
    write_rows(out / "multistart.csv", starts)
    write_rows(out / "observations.csv", detections)
    write_json(out / "ambiguity.json", report)
    render_results(out, cases, tracks, comparison)
    print(f"{len(rows)} recovery cases, {len(starts)} starts, {len(detections)} detections -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/stage4")
    run(parser.parse_args().out)
