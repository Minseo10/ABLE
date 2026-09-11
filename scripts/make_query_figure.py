"""Explain what a query does to the limb: the torque profile, a real trace, the model.

`python scripts/make_query_figure.py --out figures`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from kinder.envs.dynamic3d.limb_utils import PYBULLET_TIMESTEP  # noqa: E402

from limb_baseline.human import (  # noqa: E402
    DEFAULT_PRIOR,
    active_motions,
    default_magnitudes,
)
from limb_baseline.queries import DEFAULT_QUERY_MODEL, QuerySimulator  # noqa: E402

KEEP = "#008D63"
INK = "#141C1B"
FAINT = "#6E7A78"
GRID = "#E2E7E6"
RED = "#D42A2A"
BLUE = "#1A5FB4"
FREE = "#F2F5F4"
ZONE = "#E6EEF7"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": FAINT,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": FAINT,
        "ytick.color": FAINT,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def trace(seed: int, motion_index: int, target: float):
    """Run one real query and return its reach over time, with the true bound."""
    rng = np.random.default_rng(seed)
    theta = DEFAULT_PRIOR.sample(rng)
    sim = QuerySimulator(theta, DEFAULT_QUERY_MODEL)
    reaches: list[float] = []
    try:
        outcome = sim.ask(
            motion_index, target, rng, on_step=lambda _s, r: reaches.append(r)
        )
    finally:
        sim.close()
    bound = abs(theta[motion_index] * default_magnitudes()[motion_index])
    return np.array(reaches), bound, outcome


def panel_torque(ax, model, bound: float) -> None:
    """Effort against the end-of-range torque, and where the two balance."""
    q = np.linspace(bound - 0.55, bound + 0.16, 800)
    resist = np.array([-model.restoring_torque(x, bound) for x in q])
    rest = bound - model.zone_width
    stop = rest + model.effort_mean / model.stiffness

    ax.axvspan(q[0], rest, color=FREE, lw=0)
    ax.axvspan(rest, bound, color=ZONE, lw=0)
    ax.plot(q, resist, color=BLUE, lw=2.0)
    ax.axhline(model.effort_mean, color=KEEP, lw=2.0)
    ax.plot([stop], [model.effort_mean], "o", ms=7, mfc="white", mec=INK, mew=1.6, zorder=5)
    ax.vlines(stop, 0, model.effort_mean, color=INK, lw=1.0, ls=":")
    ax.vlines(bound, 0, 16, color=INK, lw=1.2, ls="--")

    ax.annotate(
        r"stops at $q^{*}$", xy=(stop, model.effort_mean), xytext=(stop - 0.32, 7.2),
        color=INK, fontsize=9,
        arrowprops=dict(arrowstyle="->", color=INK, lw=0.9),
    )
    ax.text(q[0] + 0.02, model.effort_mean + 0.5, r"effort $u$", color=KEEP, fontsize=9)
    ax.text(q[0] + 0.02, 9.6, r"resistance $-\tau_{RoM}(q)$", color=BLUE, fontsize=9)
    ax.text(rest - 0.13, 13.6, "free\nswing", color=FAINT, fontsize=8, ha="center")
    ax.text((rest + bound) / 2, 13.6, "end-range\nspring", color=BLUE, fontsize=8, ha="center")
    ax.text(bound + 0.02, 13.6, "hard\nstop", color=INK, fontsize=8, ha="left")
    ax.set_xticks([rest, stop, bound])
    ax.set_xticklabels([r"$b-w$", r"$q^{*}$", r"$b$"], color=INK)
    ax.set_xlim(q[0], q[-1])
    ax.set_ylim(0, 16)
    ax.set_xlabel("joint angle  $q$  (rad)")
    ax.set_ylabel("torque  (N·m)")
    ax.set_title("a.  the person pushes until it hurts", fontsize=10, loc="left", pad=10)


def panel_trace(ax, reaches, bound: float, outcome, model) -> None:
    """One recorded query: the joint creeping up and settling short of the bound."""
    t = np.arange(len(reaches)) * PYBULLET_TIMESTEP
    top = max(outcome.target, bound)
    ax.plot(t, reaches, color=KEEP, lw=1.8)
    ax.axhline(bound, color=INK, lw=1.2, ls="--")
    # Stops short of the right edge so the short-fall arrow has clear space.
    ax.hlines(outcome.target, 0, t[-1] * 0.80, color=RED, lw=1.2)
    ax.text(t[-1], bound + 0.06, r"true bound $b=R\theta$", color=INK, fontsize=8, ha="right")
    ax.text(t[0], outcome.target - 0.16, "target", color=RED, fontsize=8)
    mark = t[-1] * 0.93
    ax.annotate("", xy=(mark, bound), xytext=(mark, reaches[-1]),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1))
    ax.text(mark - 0.6, (bound + reaches[-1]) / 2, r"$w-u/k$", color=INK,
            fontsize=9, ha="right", va="center")
    ax.plot([t[-1]], [reaches[-1]], "o", ms=6, mfc="white", mec=KEEP, mew=1.6)
    ax.text(t[-1] * 0.55, reaches[-1] - 0.42, r"stops at $q^{*}$", color=KEEP, fontsize=9)
    ax.set_xlabel("time  (s)")
    ax.set_ylabel("reach  (rad)")
    ax.set_xlim(0, t[-1] * 1.02)
    ax.set_ylim(0, top + 0.3)
    ax.set_title("b.  one recorded query", fontsize=10, loc="left", pad=10)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def panel_math(ax, model) -> None:
    """The three equations the two plots are about."""
    ax.axis("off")
    rows = [
        (0.86, "constant effort against an end-range spring",
         r"$M\ddot q + C \;=\; u \;+\; \tau_{RoM}(q) \;-\; d\,\dot q$"),
        (0.55, "so they stop short of their own bound",
         r"$q^{*} \;=\; b \;-\; (w - u/k)$"),
        (0.24, r"which the robot reads as a noisy look at $b = R\theta$",
         r"$y \;=\; q^{*} + (w - \mu_u/k) \;=\; R\,\theta + \varepsilon$"),
    ]
    for y, caption, equation in rows:
        ax.text(0.0, y + 0.09, caption, fontsize=9, color=FAINT, va="center")
        ax.text(0.0, y, equation, fontsize=14, color=INK, va="center")
    ax.text(0.0, 0.03,
            f"$w={model.zone_width}$,  $k={model.stiffness:.0f}$,  "
            f"$\\mu_u={model.effort_mean:.0f}$,  $\\sigma_u={model.effort_stddev:.0f}$"
            f"   $\\Rightarrow$   short fall ${model.short_fall:.2f}$ rad,  "
            f"$\\varepsilon \\sim \\mathcal{{N}}(0,\\ {model.tau:.2f}^2)$",
            fontsize=9, color=FAINT, va="center")
    ax.set_title("c.  the query model", fontsize=10, loc="left", pad=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--motion", type=str, default="shoulder_abduction")
    parser.add_argument("--target", type=float, default=3.10)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()

    model = DEFAULT_QUERY_MODEL
    index = active_motions().index(args.motion)
    reaches, bound, outcome = trace(args.seed, index, args.target)
    print(f"{args.motion}: bound {bound:.3f}  effort {outcome.effort:.2f}  "
          f"stopped {outcome.answer:.3f}  predicted {bound - model.zone_width + outcome.effort / model.stiffness:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.15]})
    panel_torque(axes[0], model, bound)
    panel_trace(axes[1], reaches, bound, outcome, model)
    panel_math(axes[2], model)
    for ax in axes[:2]:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        path = args.out / f"query_model.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
