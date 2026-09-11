"""Cartoon of the four pipeline stages, for talks.

`python scripts/make_pipeline_figure.py --out figures`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Arc,
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
)

KEEP = "#008D63"
INK = "#141C1B"
FAINT = "#6E7A78"
GRID = "#E2E7E6"
GHOST = "#B6C0BE"
RED = "#D42A2A"
BLUE = "#1A5FB4"
SKIN = "#F2C9A0"
SKIN_DARK = "#E0A97A"
PAPER = "#F2F5F4"

plt.rcParams.update(
    {
        "font.size": 10,
        "text.color": INK,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def panel(ax, number, title, subtitle):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.15, 0.15), 9.7, 9.7,
            boxstyle="round,pad=0.0,rounding_size=0.35",
            fc="white", ec=GRID, lw=1.4, zorder=0,
        )
    )
    ax.add_patch(Circle((0.95, 8.95), 0.42, fc=KEEP, ec="none", zorder=3))
    ax.text(0.95, 8.95, str(number), ha="center", va="center",
            color="white", fontsize=13, fontweight="bold", zorder=4)
    ax.text(1.6, 9.1, title, ha="left", va="center", fontsize=13, fontweight="bold")
    ax.text(1.6, 8.6, subtitle, ha="left", va="center", fontsize=9.5, color=FAINT)


def bed(ax, x0, x1, y):
    ax.add_patch(
        FancyBboxPatch(
            (x0, y - 0.45), x1 - x0, 0.45,
            boxstyle="round,pad=0.0,rounding_size=0.18",
            fc=PAPER, ec=GRID, lw=1.2, zorder=1,
        )
    )


def arm(ax, shoulder, upper_deg, fore_deg, color=SKIN_DARK, s=1.0, lw=6, ls="-",
        zorder=5):
    """Upper arm + forearm from the shoulder, angles from +x. Returns the hand."""
    u, f = np.radians(upper_deg), np.radians(fore_deg)
    elbow = (shoulder[0] + 1.55 * s * np.cos(u), shoulder[1] + 1.55 * s * np.sin(u))
    hand = (elbow[0] + 1.35 * s * np.cos(f), elbow[1] + 1.35 * s * np.sin(f))
    for a, b in ((shoulder, elbow), (elbow, hand)):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls,
                solid_capstyle="round", zorder=zorder)
    if ls == "-":
        for p in (shoulder, elbow):
            ax.add_patch(Circle(p, 0.16 * s, fc="white", ec=color, lw=2,
                                zorder=zorder + 1))
        ax.add_patch(Circle(hand, 0.2 * s, fc=color, ec="none", zorder=zorder + 1))
    return hand


def person(ax, shoulder, upper_deg, fore_deg, s=1.0, **kw):
    """Supine person, head left of the shoulder, body to the right."""
    sx, sy = shoulder
    ax.add_patch(Circle((sx - 0.85 * s, sy - 0.30 * s), 0.45 * s, fc=SKIN, ec=INK,
                        lw=1.4, zorder=3))
    ax.add_patch(
        FancyBboxPatch(
            (sx - 0.3 * s, sy - 0.9 * s), 2.9 * s, 1.05 * s,
            boxstyle="round,pad=0.0,rounding_size=0.25",
            fc=SKIN, ec=INK, lw=1.4, zorder=2,
        )
    )
    return arm(ax, shoulder, upper_deg, fore_deg, s=s, lw=6 * s, **kw)


def robot(ax, base, s=1.0, color=BLUE, reach=None):
    """Little mobile arm, reaching for `reach`."""
    x, y = base
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.75 * s, y), 1.5 * s, 0.55 * s,
            boxstyle="round,pad=0.0,rounding_size=0.15",
            fc=color, ec="none", zorder=3,
        )
    )
    ax.plot([x, x], [y + 0.5 * s, y + 1.7 * s], color=color, lw=7 * s,
            solid_capstyle="round", zorder=3)
    shoulder = (x, y + 1.7 * s)
    reach = reach if reach is not None else (x - 1.8 * s, y + 2.4 * s)
    mid = ((shoulder[0] + reach[0]) / 2, max(shoulder[1], reach[1]) + 0.9 * s)
    for a, b in ((shoulder, mid), (mid, reach)):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=5.5 * s,
                solid_capstyle="round", zorder=6)
    for p in (shoulder, mid):
        ax.add_patch(Circle(p, 0.15 * s, fc="white", ec=color, lw=2, zorder=7))
    ax.add_patch(Circle(reach, 0.2 * s, fc="white", ec=color, lw=2.5, zorder=7))
    return mid


def bubble(ax, center, size, text, tail, ec=INK, fontsize=10):
    """Rounded speech box with a triangular tail pointing at `tail`."""
    (cx, cy), (w, h) = center, size
    ax.add_patch(Polygon([(cx - 0.15, cy - h / 2 + 0.05), (cx + 0.5, cy - h / 2 + 0.05),
                          tail], closed=True, fc="white", ec=ec, lw=1.4, zorder=8))
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.0,rounding_size=0.25",
            fc="white", ec=ec, lw=1.4, zorder=9,
        )
    )
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, zorder=10)


def arrow(ax, a, b, color=INK, lw=2.0, rad=0.0, zorder=6):
    ax.add_patch(
        FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=15, lw=lw, color=color,
                        connectionstyle=f"arc3,rad={rad}", zorder=zorder,
                        shrinkA=0, shrinkB=0)
    )


# ----------------------------------------------------------------- stage 1

def stage_ask(ax):
    panel(ax, 1, "Ask", "they move their own arm")
    bed(ax, 1.9, 7.0, 2.7)
    sh = (3.5, 3.6)
    arm(ax, sh, 78, 52, s=1.2, lw=7, color=GHOST, ls="--", zorder=2)
    person(ax, sh, 38, 14, s=1.2)
    ax.add_patch(Arc(sh, 4.0, 4.0, theta1=38, theta2=78, color=RED, lw=1.8,
                     ls=(0, (3, 2)), zorder=7))
    ax.text(6.4, 7.6, "target", color=FAINT, fontsize=10.5, ha="center")
    arrow(ax, (5.9, 7.4), (5.15, 6.9), color=FAINT, lw=1.4)
    ax.text(8.5, 4.2, "stops\nshort", color=RED, fontsize=10.5, ha="center",
            fontweight="bold")
    arrow(ax, (7.85, 4.5), (7.05, 5.05), color=RED, lw=1.5, rad=0.2)
    ax.text(1.0, 1.6, "ask: an angle", fontsize=10.5, color=FAINT)
    ax.text(1.0, 0.95, "answer: where they stop", fontsize=10.5, color=FAINT)


# ----------------------------------------------------------------- stage 2

def stage_belief(ax):
    panel(ax, 2, "Believe", "the answer sharpens the guess")
    x = np.linspace(0, 10, 400)
    base = 2.4

    def gauss(mu, sd, h):
        return base + h * np.exp(-0.5 * ((x - mu) / sd) ** 2)

    prior, post = gauss(5.7, 1.7, 2.3), gauss(4.7, 0.68, 3.9)
    ax.plot(x, prior, color=FAINT, lw=2, ls="--")
    ax.fill_between(x, base, prior, color=FAINT, alpha=0.10)
    ax.plot(x, post, color=KEEP, lw=2.6)
    ax.fill_between(x, base, post, color=KEEP, alpha=0.18)
    ax.plot([0.8, 9.3], [base, base], color=INK, lw=1.6)
    ax.text(9.3, 1.75, r"$\theta$: how far the arm goes", ha="right", fontsize=10.5,
            color=FAINT)

    ax.plot([4.7, 4.7], [base, 6.9], color=RED, lw=1.6, ls=(0, (3, 2)), zorder=2)
    ax.add_patch(Circle((4.7, base), 0.17, fc=RED, ec="none", zorder=6))
    ax.text(4.7, 7.25, "answer", color=RED, fontsize=10.5, ha="center",
            fontweight="bold")

    ax.text(8.3, 3.9, "before", color=FAINT, fontsize=10.5, ha="center")
    ax.text(2.2, 5.6, "after", color=KEEP, fontsize=11, ha="center", fontweight="bold")
    arrow(ax, (2.75, 5.35), (3.75, 4.4), color=KEEP, lw=1.5)
    ax.text(5.05, 0.95, "$K$ answers, $K$ updates", fontsize=10.5, ha="center")


# ----------------------------------------------------------------- stage 3

def stage_rehearse(ax):
    panel(ax, 3, "Rehearse", "offline, before the real task")
    top, bot = 7.6, 5.2
    for i, d in enumerate((0.34, 0.17, 0.0)):
        ax.add_patch(
            FancyBboxPatch(
                (0.75 + d, bot + d), 2.5, top - bot,
                boxstyle="round,pad=0.0,rounding_size=0.2",
                fc="white", ec=FAINT if d else INK, lw=1.3, zorder=2 + i,
            )
        )
    ax.add_patch(Circle((1.25, 6.05), 0.22, fc=SKIN, ec=INK, lw=1.2, zorder=6))
    arm(ax, (1.85, 6.0), 52, 26, s=0.45, lw=3.5, zorder=6)
    ax.text(2.05, 4.75, "tasks drawn\nfrom the belief", fontsize=9.5, color=FAINT,
            ha="center", va="top")

    ax.add_patch(
        FancyBboxPatch(
            (3.85, bot), 2.4, top - bot,
            boxstyle="round,pad=0.0,rounding_size=0.2",
            fc=PAPER, ec=INK, lw=1.4, zorder=2,
        )
    )
    t = np.linspace(0, 1, 200)
    ax.plot(4.2 + 1.7 * t, 6.9 - 1.1 * t + 0.4 * np.sin(13 * t) * (1 - t),
            color=FAINT, lw=1.6, zorder=3)
    ax.text(5.05, 4.75, "planner\nsolves each", fontsize=9.5, color=FAINT,
            ha="center", va="top")

    ax.add_patch(
        FancyBboxPatch(
            (6.85, bot), 2.4, top - bot,
            boxstyle="round,pad=0.0,rounding_size=0.2",
            fc="white", ec=KEEP, lw=2.2, zorder=2,
        )
    )
    ax.text(8.05, 6.4, "$g$", fontsize=18, color=KEEP, ha="center", va="center",
            fontweight="bold", zorder=3)
    ax.text(8.05, 4.75, "fits the\nsolutions", fontsize=9.5, color=KEEP, ha="center",
            va="top")
    arrow(ax, (3.35, 6.4), (3.75, 6.4), lw=1.8)
    arrow(ax, (6.35, 6.4), (6.75, 6.4), lw=1.8)

    ax.text(5.05, 3.2, "goal + belief $\\rightarrow$ torques", fontsize=10.5,
            ha="center", va="top")
    t = np.linspace(0, 1, 120)
    ax.plot(2.9 + 3.2 * t, 1.5 + 0.4 * np.sin(4.2 * t + 0.4), color=KEEP, lw=2.4)
    for c in (0.02, 0.34, 0.66, 0.98):
        ax.add_patch(Circle((2.9 + 3.2 * c, 1.5 + 0.4 * np.sin(4.2 * c + 0.4)),
                            0.14, fc=KEEP, ec="none"))
    ax.text(6.5, 1.45, r"$\psi$", fontsize=13, color=KEEP, va="center")


# ----------------------------------------------------------------- stage 4

def stage_do(ax):
    panel(ax, 4, "Do", "the robot moves the arm")
    bed(ax, 1.4, 6.4, 3.05)
    sh = (3.0, 4.1)
    arm(ax, sh, 72, 48, s=1.15, lw=7, color=GHOST, ls="--", zorder=2)
    person(ax, sh, 36, 12, s=1.15)
    ax.text(2.1, 7.7, "goal", color=FAINT, fontsize=10.5, ha="center")
    arrow(ax, (2.6, 7.5), (3.9, 6.9), color=FAINT, lw=1.4, rad=-0.2)
    robot(ax, (8.6, 3.0), s=0.8, reach=(6.15, 5.5))
    bubble(ax, (7.1, 7.9), (3.0, 1.1), "starts from $\\psi$", tail=(7.9, 6.7), ec=KEEP)

    ax.add_patch(
        FancyBboxPatch(
            (0.55, 0.45), 8.9, 2.0,
            boxstyle="round,pad=0.0,rounding_size=0.25",
            fc=PAPER, ec=GRID, lw=1.2, zorder=1,
        )
    )
    ax.text(0.95, 2.0, "plan time", fontsize=10.5, va="center")
    ax.plot([1.0, 5.4], [1.35, 1.35], color=FAINT, lw=8, solid_capstyle="round",
            zorder=2)
    ax.plot([1.0, 2.4], [0.8, 0.8], color=KEEP, lw=8, solid_capstyle="round", zorder=2)
    ax.text(5.65, 1.35, "from scratch", fontsize=9.5, color=FAINT, va="center")
    ax.text(2.65, 0.8, "warm start", fontsize=9.5, color=KEEP, va="center",
            fontweight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.5))
    fig.subplots_adjust(left=0.012, right=0.988, top=0.98, bottom=0.02, wspace=0.28)
    for ax, draw in zip(axes, (stage_ask, stage_belief, stage_rehearse, stage_do)):
        draw(ax)

    for i, label in enumerate((r"$(q,\ r)$", r"$b'(\theta)$", r"$\psi = g(x)$")):
        left, right = axes[i].get_position(), axes[i + 1].get_position()
        y = left.y0 + 0.42 * left.height
        fig.add_artist(
            FancyArrowPatch(
                (left.x1 - 0.003, y), (right.x0 + 0.003, y), transform=fig.transFigure,
                arrowstyle="-|>", mutation_scale=18, lw=2.4, color=KEEP,
            )
        )
        fig.text((left.x1 + right.x0) / 2, y + 0.025, label, ha="center", va="bottom",
                 fontsize=12, color=KEEP)

    args.out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        path = args.out / f"pipeline.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
