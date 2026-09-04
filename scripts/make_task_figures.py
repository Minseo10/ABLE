"""Render the P(task) figures, regenerating the samples so they rebuild from the repo alone.

`python scripts/make_task_figures.py --out figures`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from kinder.envs.dynamic3d.limb_utils import (  # noqa: E402
    LIMB_JOINT_INFOS,
    joint_position_distance,
)

from limb_baseline.human import (  # noqa: E402
    active_joints,
    active_motions,
    DEFAULT_PRIOR,
    theta_to_joint_limits,
)
from limb_baseline.tasks import DEFAULT_TASK_DISTRIBUTION as D  # noqa: E402
from limb_baseline.variant import scene_config  # noqa: E402

KEEP = "#008D63"
DROP = "#AAB4B2"
INK = "#141C1B"
FAINT = "#6E7A78"
GRID = "#E2E7E6"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": FAINT,
        "axes.labelcolor": INK,
        "axes.labelsize": 11,
        "text.color": INK,
        "xtick.color": FAINT,
        "ytick.color": FAINT,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def sample(num_people: int, draws_per_person: int, seed: int = 7):
    """Draw tasks the way `TaskDistribution` does, keeping the rejects too.

    Returns each draw's separation, whether it was accepted, and its displacement.
    """
    rng = np.random.default_rng(seed)
    nominal = np.asarray(
        scene_config().scene.limb_init_joint_positions, float
    )
    seps, keeps, deltas = [], [], []
    for _ in range(num_people):
        theta = DEFAULT_PRIOR.sample(rng)
        lower, upper = theta_to_joint_limits(theta)
        margin = 0.5 * (1.0 - D.inner_fraction) * (upper - lower)
        inner_low, inner_high = lower + margin, upper - margin
        for _ in range(draws_per_person):
            init, goal = nominal.copy(), nominal.copy()
            for j in active_joints():
                init[j] = rng.uniform(inner_low[j], inner_high[j])
                goal[j] = rng.uniform(inner_low[j], inner_high[j])
            separation = joint_position_distance(
                LIMB_JOINT_INFOS, list(init), list(goal)
            )
            kept = D.min_separation <= separation <= D.max_separation
            seps.append(separation)
            keeps.append(kept)
            deltas.append([goal[j] - init[j] for j in active_joints()])
    return np.array(seps), np.array(keeps, dtype=bool), np.array(deltas)


def figure_separation(seps, keeps, out: Path) -> None:
    """Histogram of task separation, with the accepted band marked."""
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    edges = np.arange(0.0, 5.55, 0.15)
    counts, _ = np.histogram(seps, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    colors = [
        KEEP if D.min_separation <= c <= D.max_separation else DROP for c in centers
    ]
    ax.axvspan(D.min_separation, D.max_separation, color=KEEP, alpha=0.09, zorder=0)
    ax.bar(centers, counts, width=0.13, color=colors, zorder=2)
    for edge in (D.min_separation, D.max_separation):
        ax.axvline(edge, color=KEEP, lw=1.3, ls="--", zorder=3)

    below = float((seps < D.min_separation).mean())
    above = float((seps > D.max_separation).mean())
    inside = 1.0 - below - above
    top = counts.max()
    ax.text(1.0, top * 0.97, f"kept  {inside:.1%}", color=KEEP, fontweight="bold", ha="center")
    ax.text(3.3, top * 0.97, f"too far  {above:.1%}", color=FAINT, ha="center")
    ax.text(0.24, top * 0.55, f"too close\n{below:.1%}", color=FAINT, ha="center", fontsize=9)

    ax.set_xlabel("separation (rad)")
    ax.set_ylabel("draws")
    ax.set_xlim(0, 5.4)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save(fig, out, "separation_histogram")


def figure_displacement(deltas, keeps, out: Path) -> None:
    """Per-joint displacement, where the L1 cap shows as a diamond boundary."""
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    kept, dropped = deltas[keeps], deltas[~keeps]
    ax.scatter(dropped[:, 0], dropped[:, 2], s=3, color=DROP, alpha=0.4, lw=0)
    ax.scatter(kept[:, 0], kept[:, 2], s=4, color=KEEP, alpha=0.7, lw=0)

    for radius, color, style in ((D.max_separation, KEEP, "-"), (D.min_separation, INK, "--")):
        ax.plot(
            [radius, 0, -radius, 0, radius], [0, radius, 0, -radius, 0],
            color=color, ls=style, lw=1.5, zorder=5,
        )

    ax.set_xlabel("Δ shoulder flexion (rad)")
    ax.set_ylabel("Δ elbow flexion (rad)")
    limit = 3.0
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.axhline(0, color=GRID, lw=0.9)
    ax.axvline(0, color=GRID, lw=0.9)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=KEEP, ms=5, label="kept"),
        plt.Line2D([], [], marker="o", ls="", color=DROP, ms=5, label="rejected"),
        plt.Line2D([], [], color=KEEP, lw=1.5, label="|Δ| sum = 1.5"),
        plt.Line2D([], [], color=INK, lw=1.5, ls="--", label="|Δ| sum = 0.5"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    save(fig, out, "displacement_l1_ball")


def figure_theta_ranges(out: Path) -> None:
    """The per-joint range at three values of theta, and the window tasks come from."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    thetas = (0.70, 0.95, 1.20)
    labels, y = [], 0.0
    for i, motion in enumerate(active_motions()):
        j = active_joints()[i]
        for t in thetas:
            theta = np.full(len(active_motions()), 0.95)
            theta[i] = t
            lower, upper = theta_to_joint_limits(theta)
            margin = 0.5 * (1.0 - D.inner_fraction) * (upper[j] - lower[j])
            ax.barh(y, upper[j] - lower[j], left=lower[j], height=0.62,
                    color=DROP, alpha=0.5, zorder=2)
            ax.barh(y, (upper[j] - margin) - (lower[j] + margin),
                    left=lower[j] + margin, height=0.62, color=KEEP, zorder=3)
            labels.append(f"{motion.replace('_', ' ')}  θ={t:.2f}")
            y -= 1.0
        y -= 0.5

    ax.axvline(0.0, color=FAINT, lw=0.9, zorder=1)
    ax.set_yticks([-i - 0.5 * (i // 3) for i in range(len(labels))])
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("joint angle (rad)")
    ax.set_ylabel("motion × theta")
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [
        plt.Line2D([], [], marker="s", ls="", color=KEEP, ms=8,
                   label="inner 60% — tasks drawn here"),
        plt.Line2D([], [], marker="s", ls="", color=DROP, ms=8, alpha=0.5,
                   label="rest of the range"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    save(fig, out, "theta_ranges")


def save(fig, out: Path, stem: str) -> None:
    """Write one figure as both PNG and SVG."""
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        path = out / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--num-people", type=int, default=600)
    parser.add_argument("--draws-per-person", type=int, default=12)
    args = parser.parse_args()

    seps, keeps, deltas = sample(args.num_people, args.draws_per_person)
    print(f"{len(seps):,} draws · kept {keeps.mean():.1%}")
    figure_separation(seps, keeps, args.out)
    figure_displacement(deltas, keeps, args.out)
    figure_theta_ranges(args.out)


if __name__ == "__main__":
    main()
