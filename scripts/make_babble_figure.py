"""Draw how babble makes a training row: push at random, keep where it landed.

`python scripts/make_babble_figure.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Wedge  # noqa: E402

KEEP = "#008D63"
DROP = "#AAB4B2"
WARN = "#C25A34"
INK = "#141C1B"
FAINT = "#6E7A78"
GRID = "#E2E7E6"

UPPER_ARM = 2.9
FOREARM = 2.5

# Shoulder angle, elbow angle. The limb sweeps up and out as the push goes in.
START = (-34.0, -30.0)
LANDED = (16.0, -12.0)
# The person's true range of motion, as a shoulder-angle span.
ROM = (-60.0, 30.0)


def _joints(base: np.ndarray, shoulder: float, elbow: float) -> tuple:
    """Elbow and hand positions of the two-link limb, in drawing coordinates."""
    a, b = np.radians(shoulder), np.radians(shoulder + elbow)
    elbow_at = base + UPPER_ARM * np.array([np.cos(a), np.sin(a)])
    return elbow_at, elbow_at + FOREARM * np.array([np.cos(b), np.sin(b)])


def _limb(ax, base, shoulder, elbow, color=INK, lw=5.0, alpha=1.0, z=3):
    """One pose of the limb: two segments, a shoulder, an elbow and a hand."""
    elbow_at, hand = _joints(base, shoulder, elbow)
    for start, end in ((base, elbow_at), (elbow_at, hand)):
        ax.plot(*zip(start, end), color=color, lw=lw, alpha=alpha,
                solid_capstyle="round", zorder=z)
    for point, size in ((base, 0.23), (elbow_at, 0.2), (hand, 0.17)):
        ax.add_patch(Circle(point, size, color=color, alpha=alpha, zorder=z + 1))
    return elbow_at, hand


def _gripper(ax, elbow_at, hand, color=KEEP):
    """The robot's hand, clamped onto the forearm for the whole episode."""
    along = (hand - elbow_at) / np.linalg.norm(hand - elbow_at)
    normal = np.array([-along[1], along[0]])
    middle = elbow_at + 0.5 * (hand - elbow_at)
    palm = middle - 0.62 * normal
    ax.plot(*zip(palm - 0.42 * along, palm + 0.42 * along), color=color, lw=3.4,
            solid_capstyle="round", zorder=6)
    for side in (1, -1):
        root = palm + 0.42 * side * along
        ax.plot(*zip(root, root + 1.02 * normal), color=color, lw=3.4,
                solid_capstyle="round", zorder=6)


def _rom(ax, base, label=False):
    """The wedge the limb may move in, and the forbidden ground on either side."""
    reach = UPPER_ARM + FOREARM
    ax.add_patch(Wedge(base, reach + 0.5, ROM[0], ROM[1], color=KEEP, alpha=0.07, zorder=0))
    for edge in ROM:
        end = base + (reach + 0.5) * np.array(
            [np.cos(np.radians(edge)), np.sin(np.radians(edge))]
        )
        ax.plot(*zip(base, end), color=WARN, lw=1.3, ls="--", alpha=0.75, zorder=1)
    if label:
        # Along the upper bound itself, just outside it, where nothing else sits.
        angle = np.radians(ROM[1])
        along = np.array([np.cos(angle), np.sin(angle)])
        seat = base + 3.5 * along + 0.42 * np.array([-along[1], along[0]])
        ax.text(*seat, "range of motion", color=FAINT, fontsize=10, ha="center",
                rotation=ROM[1], rotation_mode="anchor")


def _arrow(ax, start, end, color, style="-|>", lw=2.0, ls="-", z=5, rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, color=color, lw=lw,
                                 ls=ls, mutation_scale=15, zorder=z, shrinkA=0,
                                 shrinkB=0, connectionstyle=f"arc3,rad={rad}"))


def _stage(ax, x, y, index, text):
    """The numbered caption under one stage of the method."""
    ax.add_patch(Circle((x, y), 0.34, facecolor=INK, edgecolor="none", zorder=8))
    ax.text(x, y, str(index), color="white", fontsize=11, weight="bold",
            ha="center", va="center", zorder=9)
    ax.text(x + 0.6, y, text, color=INK, fontsize=11.5, ha="left", va="center")


def figure_method(out: Path) -> None:
    """Three drawings: push at random, throw away the illegal pushes, relabel."""
    fig, ax = plt.subplots(figsize=(13.8, 5.6))
    ax.set_xlim(0, 30.5)
    ax.set_ylim(-4.4, 9.0)
    ax.set_aspect("equal")
    ax.axis("off")

    bases = [np.array([2.4, 3.6]), np.array([12.4, 3.6]), np.array([22.4, 3.6])]
    spread = (-28.0, -6.0, 16.0)

    # --- 1. Grasp the limb and push it with random torques. ---
    base = bases[0]
    _rom(ax, base, label=True)
    elbow_at, hand = _limb(ax, base, *START)
    _gripper(ax, elbow_at, hand)
    for shoulder in spread:
        _limb(ax, base, shoulder, START[1], color=DROP, lw=2.4, alpha=0.5, z=2)
    ax.text(base[0] + 0.2, base[1] + 4.6, "random torques", color=INK, fontsize=11)

    # --- 2. A push that leaves the range of motion is rolled back and redrawn. ---
    base = bases[1]
    _rom(ax, base)
    _limb(ax, base, *START, color=DROP, lw=2.4, alpha=0.45, z=2)
    for shoulder, ok in ((46.0, False), (16.0, True)):
        colour = INK if ok else WARN
        elbow_at, tip = _limb(ax, base, shoulder, START[1], color=colour,
                              lw=5.0 if ok else 3.0, alpha=1.0 if ok else 0.85)
        if ok:
            _gripper(ax, elbow_at, tip)
        ax.text(tip[0] + 0.62, tip[1] + 0.16, "✓" if ok else "✕",
                color=KEEP if ok else WARN,
                fontsize=14, weight="bold", ha="center", va="center")
    ax.text(base[0] + 0.2, base[1] + 4.6, "past the limit: redraw", color=WARN,
            fontsize=11)
    ax.text(base[0] + 0.2, base[1] + 3.95, "inside: keep it", color=KEEP, fontsize=11)

    # --- 3. Whatever it reached becomes the goal that push was aiming at. ---
    base = bases[2]
    _rom(ax, base)
    _, start_hand = _limb(ax, base, *START, color=DROP, lw=3.0, alpha=0.5, z=2)
    elbow_at, hand = _limb(ax, base, *LANDED)
    _gripper(ax, elbow_at, hand)
    _arrow(ax, start_hand, hand, KEEP, lw=2.4, rad=-0.36)
    ax.add_patch(Circle(hand, 0.46, facecolor="none", edgecolor=KEEP, lw=2.2, zorder=7))
    ax.text(hand[0] + 0.85, hand[1] + 0.1, "goal", color=KEEP, fontsize=11.5,
            weight="bold", va="center")
    ax.text(start_hand[0] + 0.15, start_hand[1] - 0.8, "start", color=FAINT,
            fontsize=10.5, ha="center")
    ax.text(base[0] + 0.2, base[1] + 4.6, "goal chosen after the fact", color=FAINT,
            fontsize=11)

    for x in (9.5, 19.5):
        _arrow(ax, (x, 3.6), (x + 1.3, 3.6), FAINT, lw=2.2)

    _stage(ax, 1.7, -1.6, 1, "push at random")
    _stage(ax, 11.7, -1.6, 2, "drop what breaks a limit")
    _stage(ax, 21.7, -1.6, 3, "relabel where it landed")

    box = FancyBboxPatch((8.6, -4.05), 13.2, 0.95, boxstyle="round,pad=0.2",
                         facecolor="#F4F7F6", edgecolor=GRID, zorder=7)
    ax.add_patch(box)
    ax.text(15.2, -3.57, "training row:  start, goal, belief  →  the first torques",
            color=INK, fontsize=11, ha="center", va="center", zorder=8)

    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(out / f"babble_method.{suffix}", dpi=300, bbox_inches="tight")
        print(f"wrote {out / f'babble_method.{suffix}'}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("figures/babble_method"))
    args = parser.parse_args()
    figure_method(args.out)


if __name__ == "__main__":
    main()
