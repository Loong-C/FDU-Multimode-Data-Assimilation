"""Static figures for simulated flights. All positions use the court's metre frame."""

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from .court import DOUBLES_HALF_WIDTH, HALF_LENGTH, NET_DEPTH, NET_POST_HEIGHT, floor_lines, net_height


COLORS = ("#126a8a", "#cd6f16", "#874bad", "#268363", "#c13d49")
SHOT_NAMES = {"clear": ("高远球", "Clear"), "lift": ("挑球", "Lift"),
              "drop": ("吊球", "Drop"), "drive": ("平抽球", "Drive"),
              "smash": ("杀球", "Smash"), "custom": ("自定义击球", "Custom shot")}
_FONTS = {font.name for font in font_manager.fontManager.ttflist}
_CJK = next((name for name in ("Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC",
                                "Source Han Sans SC", "SimHei") if name in _FONTS), None)
STYLE = {"font.family": _CJK or "DejaVu Sans", "font.size": 10, "axes.unicode_minus": False,
         "axes.labelcolor": "#334957", "xtick.color": "#506370", "ytick.color": "#506370",
         "axes.edgecolor": "#bbc7c9", "savefig.facecolor": "white"}


def _text(chinese, english):
    return chinese if _CJK else english


def _court(ax, flights):
    hx, hy = HALF_LENGTH, DOUBLES_HALF_WIDTH
    floor = [(-hx, -hy, 0), (hx, -hy, 0), (hx, hy, 0), (-hx, hy, 0)]
    ax.add_collection3d(Poly3DCollection([floor], facecolors="#c6e3d9", edgecolors="none", zorder=0))
    for start, end in floor_lines().values():
        ax.plot(*zip(start, end), color="#688f83", lw=0.85, zorder=1)
    y = np.linspace(-hy, hy, 33)
    top = net_height(y)
    upper = np.column_stack((np.zeros_like(y), y, top))
    lower = upper - (0, 0, NET_DEPTH)
    ax.add_collection3d(Poly3DCollection([np.vstack((upper, lower[::-1]))], facecolors="#668798",
                                         alpha=0.25, edgecolors="none", zorder=2))
    ax.plot(*upper.T, color="#425d69", lw=1.5, zorder=3)
    for post in (-hy, hy):
        ax.plot([0, 0], [post, post], [0, NET_POST_HEIGHT], color="#425d69", lw=1.5, zorder=3)

    points = np.vstack([floor, (0., 0., NET_POST_HEIGHT), *(f.position for f in flights)])
    lo, hi = points.min(axis=0), points.max(axis=0)
    pad = (hi - lo) * [0.035, 0.08, 0.08]
    lo[:2] -= pad[:2]
    hi += pad
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]),
           xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
    ax.set_box_aspect(hi - lo, zoom=1.03)
    ax.set_proj_type("ortho")
    ax.view_init(elev=23, azim=-65)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(4, integer=True))
        axis.pane.set_facecolor((0.94, 0.96, 0.96, 0.22))
        axis.pane.set_edgecolor("#dde5e7")
    ax.tick_params(labelsize=8, pad=0)
    ax.grid(False)


def _axes_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e3e9ea", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)


def _header(fig, title, labels, colors, styles):
    fig.suptitle(title, x=0.055, y=0.96, ha="left", fontsize=18, color="#172e3a", weight="bold")
    lines = [Line2D([], [], color=color, linestyle=style, lw=2.8)
             for color, style in zip(colors, styles)]
    fig.legend(lines, labels, loc="upper left", bbox_to_anchor=(0.05, 0.885),
               ncol=len(labels), frameon=False, handlelength=2.4, columnspacing=2.0)


def _line3d(ax, flight, color, style="-"):
    p = flight.position
    ax.plot(*p.T, color=color, lw=2.2, linestyle=style, zorder=4)
    ax.scatter(*p[-1], s=20, color=color, edgecolors="white", linewidths=0.5, zorder=5)


def render_trajectories(path, shots):
    """Render one or several labelled simulated flights, returning the PNG path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = [COLORS[i % len(COLORS)] for i in range(len(shots))]
    labels = [_text(*SHOT_NAMES.get(name, (name, name))) for name in shots]
    with mpl.rc_context(STYLE):
        fig = Figure(figsize=(12.0, 5.7))
        title = (_text("五种击球的模拟轨迹", "Simulated trajectories of five shots")
                 if set(shots) == set(SHOT_NAMES) - {"custom"}
                 else _text("模拟击球轨迹", "Simulated shuttle flight"))
        _header(fig, title, labels, colors, ["-"] * len(shots))
        grid = fig.add_gridspec(1, 2, width_ratios=(1.2, 1), left=0.05, right=0.97,
                                bottom=0.15, top=0.74, wspace=0.13)
        scene = fig.add_subplot(grid[0], projection="3d", computed_zorder=False)
        _court(scene, shots.values())
        scene.set_title(_text("三维球场 · 各轴等比例", "Court in 3D · equal axis scale"), fontsize=12)
        profile = fig.add_subplot(grid[1])
        _axes_style(profile)
        for flight, color in zip(shots.values(), colors):
            _line3d(scene, flight, color)
            p = flight.position
            profile.plot(p[:, 0], p[:, 2], color=color, lw=2.2)
            profile.scatter(p[-1, 0], p[-1, 2], s=22, color=color, clip_on=False, zorder=4)
        profile.plot([0, 0], [NET_POST_HEIGHT - NET_DEPTH, NET_POST_HEIGHT], color="#405966", lw=3)
        xs = np.concatenate([[-HALF_LENGTH, HALF_LENGTH], *[f.position[:, 0] for f in shots.values()]])
        pad = np.ptp(xs) * 0.035
        profile.set(xlim=(xs.min() - pad, xs.max() + pad), ylim=(0, None), xlabel="x (m)", ylabel="z (m)")
        profile.set_title(_text("纵向剖面 · 圆点为落点", "Side view · circles mark landings"), fontsize=12)
        fig.savefig(path, dpi=160)
    return path


def render_figures(out_dir, shots, comparisons):
    """Write the shot and drag figures; each comparison has identical launch conditions."""
    out_dir = Path(out_dir)
    shot_path = render_trajectories(out_dir / "court-trajectories.png", shots)
    drag_path = out_dir / "drag-comparison.png"
    cases = sorted(comparisons.items())
    colors = ["#455467" if k == 0 else COLORS[i % len(COLORS)] for i, (k, _) in enumerate(cases)]
    styles = ["--" if k == 0 else "-" for k, _ in cases]
    labels = [f"$k={k:g}$ " + (r"$\mathrm{m}^{-1}$" if k else _text("（无阻力）", "(no drag)"))
              for k, _ in cases]
    with mpl.rc_context(STYLE):
        fig = Figure(figsize=(12.0, 6.7))
        _header(fig, _text("相同击球初态下的阻力对照", "Effect of drag at the same launch conditions"),
                labels, colors, styles)
        grid = fig.add_gridspec(2, 2, width_ratios=(1.12, 1), left=0.05, right=0.97,
                                bottom=0.10, top=0.75, wspace=0.18, hspace=0.52)
        scene = fig.add_subplot(grid[:, 0], projection="3d", computed_zorder=False)
        scene_cases = {k for k, _ in cases if k > 0} or {0}
        _court(scene, [flight for k, flight in cases if k in scene_cases])
        scene_title = (_text("三维球场 · 有阻力的轨迹", "Court in 3D · flights with drag")
                       if 0 not in scene_cases else _text("三维球场 · 无阻力轨迹", "Court in 3D · no drag"))
        scene.set_title(scene_title, fontsize=12)
        profile = fig.add_subplot(grid[0, 1])
        speed = fig.add_subplot(grid[1, 1])
        for ax in (profile, speed):
            _axes_style(ax)
        profile.axvspan(-HALF_LENGTH, HALF_LENGTH, color="#c6e3d9", alpha=.60, lw=0)
        for (k, flight), color, style in zip(cases, colors, styles):
            if k in scene_cases:
                _line3d(scene, flight, color, style)
            p = flight.position
            profile.plot(p[:, 0], p[:, 2], color=color, lw=2, ls=style)
            profile.scatter(p[-1, 0], p[-1, 2], s=17, color=color, clip_on=False, zorder=4)
            speed.plot(flight.time, flight.speed, color=color, lw=2, ls=style)
        profile.set(xlabel="x (m)", ylabel="z (m)", ylim=(0, None))
        profile.set_title(_text("完整纵向剖面 · 阴影为球场范围", "Full side view · shading marks court length"), fontsize=12)
        speed.set(xlabel=_text("飞行时间 (s)", "Time (s)"), ylabel=_text("速率 (m/s)", "Speed (m/s)"),
                  ylim=(0, None), xlim=(0, None))
        fig.savefig(drag_path, dpi=160)
    return shot_path, drag_path
