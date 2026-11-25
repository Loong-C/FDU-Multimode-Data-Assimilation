"""Reproduce the two representative MCMC experiments and plot their predictions."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from .assimilation_cases import make_case, make_model
from .court import HALF_LENGTH, SINGLES_HALF_WIDTH
from .plots import STYLE, _axes_style, _court, _text
from .posterior import fit_map, flight_states, landing_event
from .posterior_sampling import sample_posterior


ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ("short_launch", "full_launch")
NAMES = [_text("前 6 帧", "First 6 frames"), _text("全程 58 帧", "All 58 frames")]
COLORS = ("#a75124", "#0b7881")
TRUTH = "#142f44"
PARAMS = ("x0_m", "y0_m", "z0_m", "vx0_m_s", "vy0_m_s", "vz0_m_s", "k_m_inv")
STATE = ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s")
LEVELS = (.05, .5, .95)


def _rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _quantiles(values, names):
    return dict(zip(names, np.quantile(values, LEVELS, axis=0).T.tolist()))


def _predict(model, samples, grid, setting):
    chains, draws, _ = samples.chains.shape
    landings, states, curves, rows = [], [], [], []
    # Equal-spaced retained draws per chain. Full chains are used for parameter summaries.
    indices = np.linspace(0, draws-1, min(450, draws), dtype=int)
    for c in range(chains):
        for i in indices:
            theta, k = model.unpack(samples.chains[c, i])
            end, xy = landing_event(theta, k)
            path = flight_states(theta, grid, k)
            path[grid > end] = np.nan
            states.append(path)
            landings.append([end, *xy])
            rows.append(dict(setting=setting, chain=c, iteration=int(i), landing_t_s=end,
                             landing_x_m=xy[0], landing_y_m=xy[1]))
    # Select equal numbers of paths from each chain for the visible flight ensemble.
    for c in range(chains):
        for i in np.linspace(0, draws-1, 8, dtype=int):
            theta, k = model.unpack(samples.chains[c, i])
            end, xy = landing_event(theta, k)
            xyz = flight_states(theta, np.linspace(0, end, 96), k)[:, :3]
            xyz[-1] = [*xy, 0.]
            curves.append(xyz)
    landings, states = np.array(landings), np.array(states)
    survival = np.isfinite(states[:, :, 0]).mean(axis=0)
    band = np.full((3, len(grid), 6), np.nan)
    valid = survival > 0
    band[:, valid] = np.nanquantile(states[:, valid], LEVELS, axis=0)
    return dict(curves=curves, landing=landings, survival=survival, band=band, rows=rows)


def _title(fig, title, subtitle):
    fig.suptitle(title, x=.06, y=.97, ha="left", color=TRUTH, fontsize=19, weight="bold")
    fig.text(.06, .885, subtitle, color="#425867", fontsize=10)


def _ensemble(out, case, predictions):
    fig = Figure(figsize=(12, 8.1))
    _title(fig, _text("同一记高远球的后验轨迹与落点", "Posterior flights and landings of one clear"),
           _text("相同的击球位置与高度软约束。深色线和星号标记模拟真值，两列落点使用相同坐标范围。",
                 "Both windows use the same soft contact measurements. Dark curves and stars mark truth. Landing axes share limits."))
    grid = fig.add_gridspec(2, 2, left=.075, right=.955, bottom=.105, top=.815,
                            hspace=.35, wspace=.28, height_ratios=(1.3, 1))
    all_curves = [case.flight.position] + [xyz for pred in predictions for xyz in pred["curves"]]
    cloud = np.vstack([r["landing"][:, 1:] for r in predictions] + [case.flight.position[-1, :2]])
    middle = .5*(cloud.max(axis=0)+cloud.min(axis=0))
    radius = np.maximum(.10, .57*np.ptp(cloud, axis=0))
    for j, (prediction, color, name) in enumerate(zip(predictions, COLORS, NAMES)):
        scene = fig.add_subplot(grid[0, j], projection="3d", computed_zorder=False)
        _court(scene, [SimpleNamespace(position=xyz) for xyz in all_curves])
        for xyz in prediction["curves"]:
            scene.plot(*xyz.T, color=color, alpha=.28, lw=1., zorder=4)
        scene.plot(*case.flight.position.T, color=TRUTH, lw=2, zorder=5)
        scene.set_title(name + _text(" · 三维轨迹", " · 3D flight"), fontsize=12, pad=0)
        ax = fig.add_subplot(grid[1, j])
        _axes_style(ax)
        ax.scatter(*prediction["landing"][:, 1:].T, color=color, s=6, alpha=.22, linewidths=0)
        ax.plot([0, HALF_LENGTH, HALF_LENGTH, 0, 0],
                [-SINGLES_HALF_WIDTH, -SINGLES_HALF_WIDTH, SINGLES_HALF_WIDTH, SINGLES_HALF_WIDTH, -SINGLES_HALF_WIDTH],
                lw=.9, color="#81968d")
        ax.scatter(*case.flight.position[-1, :2], color=TRUTH, s=110, marker="*", zorder=4)
        ax.set(xlabel="x (m)", ylabel="y (m)", aspect="equal",
               xlim=(middle[0]-radius[0], middle[0]+radius[0]),
               ylim=(middle[1]-radius[1], middle[1]+radius[1]))
        ax.set_title(_text("预测落点与对侧单打区", "Predicted landings and far singles court"), fontsize=12)
        if j == 1:
            zoom = ax.inset_axes([.54, .49, .40, .43])
            zoom.scatter(*prediction["landing"][:, 1:].T, color=color, s=3, alpha=.21, linewidths=0)
            zoom.scatter(*case.flight.position[-1, :2], color=TRUTH, s=45, marker="*")
            zoom.set_title(_text("局部放大 (m)", "Detail (m)"), fontsize=8, pad=3)
            zoom.tick_params(labelsize=7, length=2)
    fig.savefig(out/"posterior-ensemble.png", dpi=150)


def _bands(out, grid, case, predictions):
    fig = Figure(figsize=(12, 7.2))
    _title(fig, _text("位置与速度的 90% 后验可信区间", "90% credible intervals for position and velocity"),
           _text("各时刻取仍在飞行样本的 5%–95% 分位数，仅显示两组至少 95% 样本均在飞行的时段。",
                 "Pointwise intervals condition on flight. Both groups have at least 95% of trajectories airborne at each displayed time."))
    handles = [Patch(facecolor=c, alpha=.35, label=n) for c, n in zip(COLORS, NAMES)]
    handles.append(Line2D([], [], color=TRUTH, linestyle="--", label=_text("模拟真值", "Simulated truth")))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.05,.85), ncol=3, frameon=False)
    panels = fig.subplots(2, 3, sharex=True)
    fig.subplots_adjust(left=.075, right=.965, top=.74, bottom=.09, hspace=.45, wspace=.32)
    truth = np.column_stack((case.flight.position, case.flight.velocity))
    valid = np.logical_and.reduce([p["survival"] >= .95 for p in predictions])
    for j, ax in enumerate(panels.flat):
        _axes_style(ax)
        for p, color in zip(predictions, COLORS):
            lo, med, hi = p["band"][:, valid, j]
            ax.fill_between(grid[valid], lo, hi, color=color, alpha=.29)
            ax.plot(grid[valid], med, lw=1.1, color=color)
        ax.plot(grid[valid], np.interp(grid[valid], case.flight.time, truth[:,j]), linestyle="--",
                color=TRUTH, lw=1.4)
        ax.set(xlim=(0, grid[valid][-1]), ylabel=STATE[j].split("_")[0] + (" (m)" if j<3 else " (m/s)"))
        if j >= 3:
            ax.set_xlabel(_text("离拍后时间 (s)", "Time from launch (s)"))
    fig.savefig(out/"posterior-position.png", dpi=150)


def run(out, draws=3600, warmup=500):
    from .posterior_sampling import _diagnostics
    out.mkdir(parents=True, exist_ok=True)
    case = make_case("clear", 5100)
    grid = np.arange(0, 3.001, .04)
    report = dict(quantiles=list(LEVELS), case="clear_5100", sampling={}, propagation={})
    matrices, predictions, landing_rows, band_rows = {}, [], [], []
    for setting in SETTINGS:
        model = make_model(case, setting)
        fit = fit_map(model)
        samples = sample_posterior(model, fit, warmup=warmup, draws=draws, seed=5201)
        report["sampling"][setting] = samples.diagnostics
        matrices[setting] = samples.chains
        matrices[setting+"_log_posterior"] = samples.log_prob
        pred = _predict(model, samples, grid, setting)
        predictions.append(pred)
        landing_rows.extend(pred["rows"])
        for j,t in enumerate(grid):
            if pred["survival"][j] < .95:
                continue
            for name, interval in zip(STATE, pred["band"][:, j].T):
                band_rows.append(dict(setting=setting, t_s=t, state=name, airborne_fraction=pred["survival"][j],
                                      q05=interval[0], median=interval[1], q95=interval[2]))
        flat = samples.chains.reshape(-1, samples.chains.shape[-1]).copy()
        flat[:, 6] = np.exp(flat[:, 6])
        land = pred["landing"].reshape(samples.chains.shape[0], -1, 3)
        report["propagation"][setting] = dict(observations=len(model.times), retained_draws=len(pred["rows"]),
            parameter_intervals=_quantiles(flat, PARAMS), landing_intervals=_quantiles(pred["landing"],
                                                ("flight_s", "landing_x_m", "landing_y_m")),
            derived_diagnostics={key: dict(zip(("rhat","ess_bulk","ess_tail"), _diagnostics(var)))
                                 for key,var in zip(("flight_s","landing_x_m","landing_y_m"), land.T.swapaxes(1,2))})
    with mpl.rc_context(STYLE):
        _ensemble(out, case, predictions)
        _bands(out, grid, case, predictions)
    np.savez_compressed(out/"mcmc-chains.npz", **matrices)
    _rows(out/"mcmc-flight-draws.csv", landing_rows)
    _rows(out/"mcmc-bands.csv", band_rows)
    (out/"mcmc.json").write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+"\n",
                                 encoding="utf-8")
    for setting, d in report["sampling"].items():
        print(f"{setting} max R-hat {d['max_rhat']:.4f}, min bulk/tail ESS {d['min_bulk_ess']:.0f}/{d['min_tail_ess']:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"results/stage5")
    parser.add_argument("--draws", type=int, default=3600)
    parser.add_argument("--warmup", type=int, default=500)
    args = parser.parse_args()
    run(args.out, args.draws, args.warmup)
