"""Compare simulated ground truth, reconstructed flights and the actual image evidence."""

from types import SimpleNamespace

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from .plots import STYLE, _axes_style, _court, _text


TRUTH = dict(color="#173f50", linewidth=2.8)
FIT = dict(color="#c86421", linewidth=2.1, linestyle=(0, (4, 2)))
ALTERNATIVE = dict(color="#6c48a0", linewidth=2.1, linestyle="-.")
OBSERVED = dict(marker="o", markersize=4.5, markerfacecolor="white", markeredgewidth=1.1,
                color="#467967", linestyle="none")


def _canvas(title, summary, show_alternative):
    fig = Figure(figsize=(11.5, 7.4))
    fig.suptitle(title, x=0.055, y=0.963, ha="left", fontsize=20, color="#173746", weight="bold")
    fig.text(0.056, 0.875, summary, color="#405965", fontsize=11)
    legend = [(_text("模拟真值", "Ground truth"), TRUTH),
              (_text("重建轨迹", "Fit"), FIT)]
    if show_alternative:
        legend.append((_text("另一组可行初态", "Alternative fit"), ALTERNATIVE))
    legend.append((_text("观测球心", "Image detections"), OBSERVED))
    handles = [Line2D([], [], label=label, **style) for label, style in legend]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.05, 0.85),
               ncol=len(handles), frameon=False, handlelength=2.8, columnspacing=2.2)

    grid = fig.add_gridspec(2, 2, left=0.06, right=0.96, top=0.72, bottom=0.11,
                            width_ratios=(1.15, 1), height_ratios=(1.48, 1),
                            wspace=0.20, hspace=0.66)
    scene = fig.add_subplot(grid[:, 0], projection="3d", computed_zorder=False)
    pixels, detail = fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])
    for ax in (pixels, detail):
        _axes_style(ax)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    pixels.set(xlabel="u (px)", ylabel="v (px)", aspect="equal", adjustable="datalim")
    pixels.invert_yaxis()
    return fig, scene, pixels, detail


def _world(scene, paths):
    _court(scene, [SimpleNamespace(position=xyz) for xyz, _ in paths])
    for xyz, style in paths:
        scene.plot(*xyz.T, **style, zorder=4)
    scene.set_title(_text("三维球路 · 场地坐标", "Complete flights in court coordinates"),
                    loc="left", fontsize=12)


def _reconstruction(out, case, track):
    times, points = np.asarray(case.times), np.asarray(case.uv)
    fitted = np.asarray(track["uv_fit"])
    summary = _text(
        f"{len(points)} 帧观测   ·   三维位置 RMSE {track['position_rmse_m']:.3f} m"
        f"   ·   重投影 RMSE {track['observed_rmse_px']:.2f} px",
        f"{len(points)} detections   ·   3D position RMSE {track['position_rmse_m']:.3f} m"
        f"   ·   Reprojection RMSE {track['observed_rmse_px']:.2f} px",
    )
    fig, scene, pixels, residuals = _canvas(
        _text("从带噪声的像素坐标重建高远球", "Reconstructing a clear from noisy detections"),
        summary, False)
    _world(scene, [(case.truth_xyz, TRUTH), (track["xyz"], FIT)])
    pixels.plot(*case.camera.project(case.truth_xyz).T, **TRUTH)
    pixels.plot(*case.camera.project(track["xyz"]).T, **FIT)
    pixels.plot(*points.T, **OBSERVED)
    pixels.set_title(_text("摄像机投影与观测点", "Camera projections and detections"),
                     loc="left", fontsize=12)

    errors = np.linalg.norm(fitted - points, axis=1)
    residuals.plot(times, errors, color=OBSERVED["color"], linewidth=1.1, marker="o", markersize=2.8)
    residuals.set(xlabel=_text("离拍后时间 (s)", "Time from launch (s)"),
                  ylabel=_text("距离 (px)", "Distance (px)"), ylim=(0, None))
    residuals.set_title(_text("各帧球心与重投影的距离", "Distance from each detection to the fitted pixel"),
                        loc="left", fontsize=12)
    fig.savefig(out / "reconstruction.png", dpi=160)


def _ambiguity(out, case, track, comparison):
    times = np.asarray(case.times)
    main, alt = np.asarray(track["xyz"]), np.asarray(comparison["alternative_xyz"])
    landing_gap = np.linalg.norm(main[-1, :2] - alt[-1, :2])
    summary = _text(
        f"观测窗口 {times[-1] - times[0]:.2f} s（{len(times)} 帧）   ·   两解重投影 RMSE "
        f"{track['observed_rmse_px']:.2f} / {comparison['alternative_rmse_px']:.2f} px"
        f"   ·   两个预测落点相距 {landing_gap:.2f} m",
        f"{len(times)} detections over {times[-1] - times[0]:.2f} s   ·   Reprojection RMSE "
        f"{track['observed_rmse_px']:.2f} / {comparison['alternative_rmse_px']:.2f} px"
        f"   ·   Predicted landings {landing_gap:.2f} m apart",
    )
    fig, scene, pixels, heights = _canvas(
        _text("短时间观测下的两条可行球路", "Two plausible flights from a short observation window"),
        summary, True)
    _world(scene, [(case.truth_xyz, TRUTH), (main, FIT), (alt, ALTERNATIVE)])
    pixels.plot(*np.asarray(track["uv_fit"]).T, **FIT)
    pixels.plot(*np.asarray(comparison["alternative_uv"]).T, **ALTERNATIVE)
    pixels.plot(*np.asarray(case.uv).T, **OBSERVED)
    pixels.set_title(_text("观测窗口内的像素轨迹（局部放大）", "Image tracks over the observed window (zoom)"),
                     loc="left", fontsize=12)

    heights.axvspan(times[0], times[-1], color="#cee2dd", lw=0,
                    label=_text("观测窗口", "Observed window"))
    heights.plot(case.truth_time, case.truth_xyz[:, 2], **TRUTH)
    heights.plot(track["times"], main[:, 2], **FIT)
    heights.plot(comparison["alternative_times"], alt[:, 2], **ALTERNATIVE)
    heights.set(xlabel=_text("离拍后时间 (s)", "Time from launch (s)"),
                ylabel=_text("高度 (m)", "Height (m)"), xlim=(0, None), ylim=(0, None))
    heights.set_title(_text("从离拍到落地的高度", "Height from launch to landing"),
                      loc="left", fontsize=12)
    heights.legend(loc="upper right", frameon=False, fontsize=8)
    fig.savefig(out / "ambiguity.png", dpi=160)


def render_results(out, cases, tracks, ambiguity):
    """Render fixed benchmark examples so different runs compare the same observations."""
    results = {case.name: (case, track) for case, track in zip(cases, tracks, strict=True)}
    out.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(STYLE):
        _reconstruction(out, *results["clear_oblique_noise_401"])
        _ambiguity(out, *results[ambiguity["case_name"]], ambiguity)
