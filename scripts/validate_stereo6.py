"""Exercise independent-view scoring on synthetic asynchronous cameras with known geometry."""

import argparse
import csv
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from badminton.camera import fit_camera, look_at_camera
from badminton.forward import virtual_camera
from badminton.reconstruct import write_json
from badminton.stereo import Settings, _reference_error, evaluate, run, synchronize, triangulate
from validate_stage4 import reference_flight, project


def calibration(camera, rng):
    world = np.array([[x, y, 0.] for x in (-6.7, -5.94, -1.98, 1.98, 5.94, 6.7)
                      for y in (-3.05, -2.59, 2.59, 3.05)] + [[0., 3.05, 1.55], [0., -3.05, 1.55]])
    reserved = np.array([2, 4, 6, 10, 12, 15, 16, 21, 23])
    train = np.setdiff1d(np.arange(len(world)), reserved)
    truth = project(camera, world)
    uv = truth + rng.normal(0., .20, truth.shape)
    fit = fit_camera(world[train], uv[train], camera.image_size)
    error = np.linalg.norm(fit.project(world[reserved])-uv[reserved], axis=1)
    return fit, float(np.sqrt(np.mean(error**2)))


def write_pixels(path, times, points):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s","u_px","v_px"])
        writer.writerows([[t, *uv] for t, uv in zip(times, points)])


def validate():
    rng = np.random.default_rng(601)
    true_a = virtual_camera()
    true_b = look_at_camera((3., 16., 7.), (-1., 0., 2.), 1050., (1280, 720))
    a, a_cal = calibration(true_a, rng)
    b, b_cal = calibration(true_b, rng)
    theta = np.array([-5.3, -1.7, 2.6, 33., 8., 25.])
    state, _, _, _ = reference_flight(theta, 0.217)
    # Independent native clocks and rates. Stop before physical landing to mimic a return.
    alpha, beta, onset, duration = 1.0013, 1.532, 8.292, 1.48
    a_elapsed = .009 + np.arange(45)/25.
    b_elapsed = .014 + np.arange(107)*alpha/(60000/1001)
    take = (a_elapsed < .54) | (a_elapsed > .79)
    a_elapsed = a_elapsed[take]
    b_elapsed = np.delete(b_elapsed, [22, 31, 52, 53, 56, 68])
    ta, tb = onset + a_elapsed, (onset+b_elapsed-beta)/alpha
    ua = project(true_a, state(a_elapsed)[:3].T)
    ub = project(true_b, state(b_elapsed)[:3].T)
    assert np.all((ua >= 0) & (ua < a.image_size)) and np.all((ub >= 0) & (ub < b.image_size))
    noisy_a = ua + rng.normal(0., .8, ua.shape)
    noisy_b = ub + rng.normal(0., 1.2, ub.shape)
    anchors = [dict(t_b_s=t, t_a_s=alpha*t+beta+e, split=split) for t,e,split in (
        (1., .0005, "fit"), (6.5, -.001, "check"), (14., -.0005, "fit"), (20., .0002, "check"),
        (45., .0001, "fit"))]
    kw = dict(launch_a_s=onset, stop_a_s=onset+duration, sigma_a_px=.8, sigma_b_px=1.2,
              sync_sigma_s=.003, max_gap_a_s=.060)
    report, rows = evaluate(a,b,ta,noisy_a,tb,noisy_b,anchors,**kw)
    assert report["clock"]["check_anchors"] == 2 and report["clock"]["check_rmse_s"] < .002
    assert abs(report["clock"]["scale_b_to_a"]-alpha) < 3e-5
    assert report["a_training_points"] == (a_elapsed < duration).sum()
    assert report["b_held_out_points"] == (b_elapsed < duration).sum()
    assert report["prediction_counts"] == {"projected": len(rows)}
    assert report["reference_counts"].get("long_a_gap",0) >= 10
    assert report["b_rmse_px"] < 5 and report["a_fit_to_stereo_rmse_m"] < .12
    tri = [r for r in rows if r["reference"] == "triangulated"]
    truth_xyz = state(np.array([alpha*r["t_b_s"]+beta-onset for r in tri]))[:3].T
    xyz = np.array([[r[k] for k in ("x_ref_m","y_ref_m","z_ref_m")] for r in tri])
    true_tri_rmse = np.sqrt(np.mean(np.sum((xyz-truth_xyz)**2, axis=1)))
    assert true_tri_rmse < .07

    # A changed B stream can change the score but cannot change the estimated state.
    wrong_b, _ = evaluate(a,b,ta,noisy_a,tb,noisy_b+[18.,0.],anchors,**kw)
    wrong_anchors = [dict(r,t_a_s=r["t_a_s"]+.04) for r in anchors]
    late, _ = evaluate(a,b,ta,noisy_a,tb,noisy_b,wrong_anchors,**kw)
    np.testing.assert_array_equal(report["a_fit_theta"], wrong_b["a_fit_theta"])
    np.testing.assert_array_equal(report["a_fit_theta"], late["a_fit_theta"])
    assert wrong_b["b_rmse_px"] > 3*report["b_rmse_px"] and late["b_rmse_px"] > 3*report["b_rmse_px"]

    # The same A-only MAP model used on the real match, with optional soft A-only feet.
    # A published hardware-shared timebase uses no invented fitting or check anchors.
    proxies = dict(player_xy_m=(theta[:2]+[.2,-.3]).tolist(),
                   receiver_xy_m=(state(duration)[:2]+[-.4,.1]).tolist())
    map_kw = dict(kw, model="joint_drag", clock_source="published_shared", **proxies)
    shared_times = alpha*tb+beta
    joint, _ = evaluate(a,b,ta,noisy_a,shared_times,noisy_b,**map_kw)
    joint_shift, _ = evaluate(a,b,ta,noisy_a,shared_times,noisy_b+[18.,0.], **map_kw)
    fixed_map, _ = evaluate(a,b,ta,noisy_a,shared_times,noisy_b,**dict(map_kw, model="fixed_drag_map"))
    np.testing.assert_array_equal(joint["a_fit_theta"], joint_shift["a_fit_theta"])
    assert joint["a_fit_k_m_inv"] == joint_shift["a_fit_k_m_inv"]
    assert joint["prediction_counts"] == {"projected": len(rows)} and fixed_map["b_rmse_px"] < 5
    assert joint_shift["b_rmse_px"] > 3*joint["b_rmse_px"]
    assert joint["clock"]["check_rmse_s"] is None and joint["clock"]["fit_anchors"] == 0
    for invalid in (dict(model="joint_drag",k=.4),dict(min_angle_deg=0.),dict(model="unknown"),
                    dict(receiver_xy_m=[1.]),dict(sync_sigma_s=-.1)):
        try:
            Settings(**dict(kw,**invalid))
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid settings accepted: {invalid}")
    for malformed in ([], anchors[:1], anchors+[anchors[0]], [dict(t_a_s="?",t_b_s=0,split="fit")],
                      [dict(r,split="typo") for r in anchors]):
        try:
            synchronize(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid sync anchors accepted")

    exact = [dict(r,t_a_s=alpha*r["t_b_s"]+beta) for r in anchors]
    clean, _ = evaluate(true_a,true_b,ta,ua,tb,ub,exact,**dict(kw,sigma_a_px=.02))
    assert clean["b_rmse_px"] < 1e-3
    for poor_pair in (("weak_parallax",true_a,true_a.project(theta[:3])),
                      ("behind_camera",look_at_camera((4,13,7),(9,16,9),900,(1280,720)),None)):
        reason, test_b, uv_b = poor_pair
        if uv_b is None:
            # A line through the test camera to the shuttle, pointing in the opposite direction.
            uv_b = test_b.project(2*test_b.center-theta[:3])
        try:
            triangulate(true_a,test_b,true_a.project(theta[:3]),uv_b)
        except ValueError as exc:
            assert str(exc) == reason
        else:
            raise AssertionError(f"Failed to reject {reason}")

    t = .27
    p = state(t)[:3]
    pixel_a, pixel_b = true_a.project(p), true_b.project(p)
    velocity = np.diff(project(true_a,state([t-.0001,t+.0001])[:3].T),axis=0)[0]/.0002
    predicted = _reference_error(true_a,true_b,pixel_a,pixel_b,velocity,1.,.8,1.2,.003)
    jitter = rng.normal(0,.003,400)
    pa = project(true_a,state(t+jitter)[:3].T) + rng.normal(0,.8,(len(jitter),2))
    pb = pixel_b + rng.normal(0,1.2,(len(jitter),2))
    mc = [triangulate(true_a,true_b,x,y,max_pair_reprojection_px=100.)[0] for x,y in zip(pa,pb)]
    empirical = float(np.sqrt(np.mean(np.sum((mc-p)**2,axis=1))))
    assert .80 < empirical/predicted["reference_rms_m"] < 1.20

    # Round-trip the user entry point and verify time-base preservation across missing rows.
    with TemporaryDirectory() as temp:
        folder = Path(temp)
        for name, camera in (("a",a), ("b",b)):
            write_json(folder/f"camera-{name}.json",camera.to_dict())
        write_pixels(folder/"a.csv",ta,noisy_a)
        write_pixels(folder/"b.csv",tb,noisy_b)
        with (folder/"sync.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, list(anchors[0]))
            writer.writeheader()
            writer.writerows(anchors)
        config = dict(camera_a="camera-a.json",camera_b="camera-b.json",observations_a="a.csv",
                      observations_b="b.csv",anchors="sync.csv",**kw)
        write_json(folder/"config.json",config)
        run(folder/"config.json",folder/"output")
        loaded = json.loads((folder/"output/stereo.json").read_text(encoding="utf-8"))
        np.testing.assert_allclose(loaded["b_rmse_px"], report["b_rmse_px"], atol=1e-9)
        with (folder/"b.csv").open("a", encoding="utf-8", newline="") as stream:
            stream.write("300,,\n")
        np.testing.assert_allclose(run(folder/"config.json",folder/"output")["b_rmse_px"], report["b_rmse_px"])
        with (folder/"b.csv").open("a", encoding="utf-8", newline="") as stream:
            stream.write("301,100,\n")
        try:
            run(folder/"config.json",folder/"output")
        except ValueError as exc:
            assert "Incomplete" in str(exc)
        else:
            raise AssertionError("Incomplete pixel pair accepted")
    return dict(data="synthetic, independently calibrated simulated cameras", seed=601, passed=True,
                native_fps_a=25.,native_fps_b=60000/1001,true_b_to_a_scale=alpha,
                true_b_to_a_offset_s=beta,
                calibration_heldout_rmse_px={"a":a_cal,"b":b_cal}, noisy_evaluation=report,
                stereo_to_known_truth_rmse_m=float(true_tri_rmse), exact_input_b_rmse_px=clean["b_rmse_px"],
                b_score_with_18px_label_shift_px=wrong_b["b_rmse_px"],
                b_score_with_40ms_clock_error_px=late["b_rmse_px"], independent_a_estimate_unchanged=True,
                map_validation=dict(joint_drag_b_rmse_px=joint["b_rmse_px"],fixed_drag_map_b_rmse_px=fixed_map["b_rmse_px"],
                                    a_joint_drag_m_inv=joint["a_fit_k_m_inv"], independent_a_state_and_drag=True,
                                    joint_b_rmse_with_18px_shift=joint_shift["b_rmse_px"]),
                uncertainty_check=dict(draws=len(jitter), empirical_rms_m=empirical, **predicted))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"results/stage6/stereo-validation.json")
    out = parser.parse_args().out
    result = validate()
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    print(f"Synthetic independent-view checks passed -> {out}")
