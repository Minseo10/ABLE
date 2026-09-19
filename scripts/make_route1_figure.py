"""Draw Route 1: a critic per stream, trained backward from MPC outcomes.

Top row: how the critics are trained. Bottom row: how a stream uses one at plan time.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "figures"

STREAM, VALUE, CRITIC, OUTCOME = "#cfe2f3", "#ffffff", "#e6d0f0", "#fce5cd"


def box(ax, xy, text, color, width=2.4, height=0.9, style="square", fontsize=10):
    """One labelled node."""
    return ax.text(
        *xy, text, ha="center", va="center", fontsize=fontsize,
        bbox=dict(boxstyle=f"{style},pad=0.45", fc=color, ec="#222222", lw=1.3),
    )


def arrow(ax, a, b, color="#555555", style="-|>", label="", dashed=False, rad=0.0, label_xy=None):
    """Edge between two nodes, with an optional label at its midpoint."""
    ax.annotate(
        "", xy=(0, 0), xytext=(0, 0),
        arrowprops=dict(
            arrowstyle=style, color=color, lw=1.4, linestyle="--" if dashed else "-",
            patchA=a.get_bbox_patch(), patchB=b.get_bbox_patch(),
            connectionstyle=f"arc3,rad={rad}", mutation_scale=15,
        ),
        xycoords=b, textcoords=a,
    )
    if label:
        (xa, ya), (xb, yb) = a.get_position(), b.get_position()
        x, y = label_xy or ((xa + xb) / 2, (ya + yb) / 2 + 0.85)
        ax.text(x, y, label, ha="center", va="bottom", fontsize=9, color=color)


def training(ax) -> None:
    """The chain, its critics, and the backward pass that labels them."""
    chain = [
        (1.5, "sample-grasp", STREAM), (4.0, "?g\ngrasp", VALUE),
        (6.5, "sample-base-pose", STREAM), (9.0, "?q\nbase pose", VALUE),
        (11.5, "plan-limb-motion\n(MPC)", STREAM), (14.2, "outcome\nsuccess, seconds", OUTCOME),
    ]
    nodes = [box(ax, (x, 3.2), text, color, style="round" if color == VALUE else "square")
             for x, text, color in chain]
    for a, b in zip(nodes, nodes[1:]):
        arrow(ax, a, b)

    critics = [
        (4.0, "V_g(g)\nC_g(g)"),
        (9.0, "V_q(q, g)\nC_q(q, g)"),
        (12.2, "V_mpc(psi, g, q)\nC_mpc(psi, g, q)"),
    ]
    heads = [box(ax, (x, 1.2), text, CRITIC, fontsize=9.5) for x, text in critics]
    arrow(ax, nodes[5], heads[2], color="#a64d79", style="-|>", dashed=True,
          label="label: success, seconds", rad=-0.3, label_xy=(14.6, 2.0))
    arrow(ax, heads[2], heads[1], color="#a64d79", style="-|>", dashed=True,
          label="average over psi")
    arrow(ax, heads[1], heads[0], color="#a64d79", style="-|>", dashed=True,
          label="average over q")
    ax.text(0.2, 4.4, "Training: labels flow backward from what MPC did",
            fontsize=13, ha="left", va="center")
    ax.text(0.2, 0.1, "Every stream call is a row: successes and failures both count.",
            fontsize=9.5, ha="left", color="#555555")
    ax.set_xlim(0, 16.2)
    ax.set_ylim(-0.2, 4.9)
    ax.axis("off")


def plan_time(ax) -> None:
    """One call of sample-base-pose: draw, score, then yield in order."""
    rng = np.random.default_rng(0)
    scores = rng.uniform(0.05, 0.95, 8)
    prior = box(ax, (1.8, 2.4), "hand-written\nsample-base-pose", STREAM)
    ax.text(4.9, 3.7, "draw N candidates", fontsize=9.5, ha="center", color="#555555")
    points = ax.scatter(
        rng.uniform(4.2, 5.6, 8), rng.uniform(2.7, 3.4, 8), s=90,
        c=scores, cmap="viridis", vmin=0, vmax=1, edgecolor="#222222", zorder=3,
    )
    critic = box(ax, (8.0, 2.4), "V_q(q, g) / C_q(q, g)\nscore each candidate", CRITIC, fontsize=9.5)
    order = box(ax, (12.2, 2.4), "yield best first\nskip near-duplicates", STREAM, fontsize=9.5)
    arrow(ax, prior, critic, rad=0.0)
    arrow(ax, critic, order)
    plt.colorbar(points, ax=ax, fraction=0.025, pad=0.01, label="predicted score")
    ax.text(0.2, 4.4, "Plan time: the prior proposes, the critic reorders",
            fontsize=13, ha="left", va="center")
    ax.text(0.2, 0.1, "Mixing in unscored samples keeps every base pose reachable.",
            fontsize=9.5, ha="left", color="#555555")
    ax.set_xlim(0, 16.2)
    ax.set_ylim(-0.2, 4.9)
    ax.axis("off")


def main() -> None:
    """Write the Route 1 figure."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    training(axes[0])
    plan_time(axes[1])
    fig.suptitle("Route 1: a critic per stream, learned from planner logs", fontsize=15, x=0.02, ha="left")
    fig.savefig(OUT / "route1_critics.png", dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT / 'route1_critics.png'}")


if __name__ == "__main__":
    main()
