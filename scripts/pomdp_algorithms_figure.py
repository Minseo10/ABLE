"""Schematic of POUCT and POMCPOW on the limb task (hand-laid, not a real search).

    python scripts/pomdp_algorithms_figure.py --out figures/pomdp/algorithms.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyBboxPatch

ACTION, BELIEF, CHOSEN, LIMIT = "#4C72B0", "#DD8452", "#C44E52", "#8172B2"
START, GOAL = 15.0, 125.0  # elbow angles (deg) at the limb's start and goal


def angle(fraction: float) -> float:
    return np.radians(START + fraction * (GOAL - START))


def node(ax, xy, text, color, kind="action", width=2.2, size=8.5, bold=False):
    x, y = xy
    if kind == "belief":
        patch = plt.Circle(xy, 0.42, fc="white", ec=color, lw=2.2 if bold else 1.6, zorder=3)
    else:
        patch = FancyBboxPatch((x - width / 2, y - 0.3), width, 0.6,
                               boxstyle="round,pad=0.05,rounding_size=0.15", fc=color,
                               ec=CHOSEN if bold else color, lw=2.4 if bold else 1, zorder=3)
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=size, zorder=4,
            color="black" if kind == "belief" else "white", fontweight="bold" if bold else None)


def edge(ax, a, b, text=None, color="0.45", bold=False, dy=(0.42, 0.32), left=False):
    ax.annotate("", xy=(b[0], b[1] + dy[1]), xytext=(a[0], a[1] - dy[0]),
                arrowprops=dict(arrowstyle="-|>", color=CHOSEN if bold else color,
                                lw=2 if bold else 1.1), zorder=2)
    if text:
        ax.text((a[0] + b[0]) / 2 + (-0.12 if left else 0.12), (a[1] + b[1]) / 2, text,
                fontsize=7.5, color="0.25", ha="right" if left else "left", va="center")


def particles(ax, xy, weights, color=BELIEF):
    """Dots under a belief node; size shows weight."""
    xs = xy[0] + np.linspace(-0.55, 0.55, len(weights))
    ax.scatter(xs, np.full(len(weights), xy[1] - 0.62), s=np.asarray(weights) * 28 + 4,
               c=color, zorder=3, edgecolors="none")


def limb_panel(ax, limits, true_limit):
    """The task: one elbow, the believed limits as rays, the targets each solver tries."""
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot([-1.25, 0], [0, 0], color="0.3", lw=6, solid_capstyle="round")  # upper arm
    ax.add_patch(plt.Circle((0, 0), 0.06, color="0.3", zorder=5))
    for lim in limits:  # one ray per particle: where that person stops
        a = angle(lim)
        ax.plot([0, 1.2 * np.cos(a)], [0, 1.2 * np.sin(a)], color=LIMIT, alpha=0.18, lw=1)
    a = angle(true_limit)
    ax.plot([0, 1.25 * np.cos(a)], [0, 1.25 * np.sin(a)], color="black", ls="--", lw=1.4)
    ax.text(1.27 * np.cos(a), 1.27 * np.sin(a), "true limit (unknown)", fontsize=8,
            ha="left", va="bottom")
    ax.add_patch(Arc((0, 0), 2.0, 2.0, theta1=START, theta2=GOAL, color="0.6", lw=1, ls=":"))
    for fraction, name, color in ((0, "start", "0.3"), (1, "goal", "green")):
        a = angle(fraction)
        ax.plot([0, np.cos(a)], [0, np.sin(a)], color=color, lw=5, alpha=0.8,
                solid_capstyle="round")
        ax.text(1.12 * np.cos(a), 1.12 * np.sin(a), name, fontsize=9, ha="center",
                va="center", color=color, fontweight="bold")
    for fraction, marker, color, dx in ((1.0, "s", ACTION, 0.0),):
        a = angle(fraction)
        ax.scatter(np.cos(a), np.sin(a), marker=marker, s=60, color=color, zorder=6)
    for fraction in (0.25, 0.5, 0.75):  # POUCT with 4 waypoints
        a = angle(fraction)
        ax.scatter(np.cos(a), np.sin(a), marker="s", s=45, facecolor="white",
                   edgecolor=ACTION, lw=1.5, zorder=6)
    for fraction in (0.38, 0.62, 0.71, 0.86):  # POMCPOW: drawn as the tree widens
        a = angle(fraction)
        ax.scatter(0.88 * np.cos(a), 0.88 * np.sin(a), marker="D", s=40, color=CHOSEN,
                   zorder=6)
    ax.set_xlim(-1.4, 1.6)
    ax.set_ylim(-0.35, 1.45)
    handles = [
        plt.Line2D([], [], color=LIMIT, alpha=0.5, label="limit of one belief particle"),
        plt.Line2D([], [], marker="s", ls="", color=ACTION, label="POUCT target: goal (w1)"),
        plt.Line2D([], [], marker="s", ls="", mfc="white", mec=ACTION,
                   label="+ fixed waypoints (w4)"),
        plt.Line2D([], [], marker="D", ls="", color=CHOSEN,
                   label="POMCPOW: targets sampled\nanywhere on the line"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5, frameon=False,
              bbox_to_anchor=(1.12, -0.05))
    ax.set_title("(a) Limb task: move the elbow from start to goal; where is the limit?",
                 fontsize=10, loc="left")


def shared_top(ax, x0):
    """Root -> move_base -> grasp, common to both trees."""
    root, mb0, mb1 = (x0, 9.0), (x0 - 1.4, 7.3), (x0 + 1.6, 7.3)
    node(ax, root, "h₀", BELIEF, "belief", bold=True)
    particles(ax, root, np.ones(9))
    ax.text(root[0] + 0.6, root[1], "start: θ ~ belief\n(particles)", fontsize=7.5,
            va="center")
    node(ax, mb0, "move_base(b0)", ACTION, bold=True)
    node(ax, mb1, "move_base(b1)", ACTION)
    edge(ax, root, mb0, bold=True, dy=(0.95, 0.32))
    edge(ax, root, mb1, dy=(0.95, 0.32))
    ax.text(mb1[0], mb1[1] - 0.5, "N=40  Q=4.1", fontsize=7, ha="center", va="top",
            color="0.3")
    ax.text(mb0[0] + 1.2, mb0[1] - 0.15, "N=310\nQ=5.0", fontsize=7, ha="left",
            va="top", color="0.3")
    b1, gr = (mb0[0], 6.0), (mb0[0], 4.9)
    node(ax, b1, "", BELIEF, "belief", bold=True)
    edge(ax, mb0, b1, "base observed", bold=True, dy=(0.3, 0.42), left=True)
    node(ax, gr, "grasp(g0)", ACTION, bold=True)
    edge(ax, b1, gr, bold=True)
    return gr


def pouct_panel(ax):
    ax.axis("off")
    ax.set_xlim(-0.9, 11.6)
    ax.set_ylim(-0.7, 9.6)
    gr = shared_top(ax, 4.0)
    hb = (gr[0], 3.8)
    node(ax, hb, "", BELIEF, "belief", bold=True)
    edge(ax, gr, hb, "grasped", bold=True, dy=(0.3, 0.42), left=True)
    ml, st = (hb[0] - 1.0, 2.5), (hb[0] + 1.6, 2.5)
    node(ax, ml, "move_limb(goal)", ACTION, bold=True, width=2.5)
    node(ax, st, "stop", ACTION, width=1.2)
    edge(ax, hb, ml, bold=True)
    edge(ax, hb, st)
    ax.text(ml[0] - 1.35, ml[1], "N=260\nQ=5.2", fontsize=7, ha="right", va="center",
            color="0.3")
    ax.text(st[0] + 0.7, st[1], "N=50\nQ=-1", fontsize=7, ha="left", va="center",
            color="0.3")
    clear, limit = (ml[0] - 1.1, 0.8), (ml[0] + 1.3, 0.8)
    node(ax, clear, "", BELIEF, "belief", bold=True)
    node(ax, limit, "", BELIEF, "belief")
    edge(ax, ml, clear, "Clear", bold=True, dy=(0.3, 0.42), left=True)
    edge(ax, ml, limit, "Limit(bin)", dy=(0.3, 0.42))
    ax.text(clear[0], clear[1] - 0.7, "arrived: +goal reward\n(θ allowed the goal)",
            fontsize=7, ha="center", va="top")
    ax.text(limit[0], limit[1] - 0.7, "stopped early at θ's limit:\nprogress forfeited",
            fontsize=7, ha="center", va="top")
    ax.text(7.0, 9.3,
            "Every simulation:\n"
            "1. draw θ (a person) from the root belief\n"
            "2. descend: pick the action with max\n"
            "    Q + c·√(log N / n); try untried ones first\n"
            "3. simulate it (kinematic model): next state,\n"
            "    observation, reward\n"
            "4. new node → random rollout to estimate value\n"
            "5. back up the return: update N and Q\n\n"
            "Fixed, finite action set: limb targets are\n"
            "the goal (w1) or K evenly spaced waypoints.\n"
            "Nodes keep counts only; no particles below\n"
            "the root (POMCP keeps unweighted ones).",
            fontsize=7.8, va="top", bbox=dict(fc="0.96", ec="0.8"))
    ax.set_title("(b) POUCT: discrete targets, UCB tree search", fontsize=10, loc="left")


def pomcpow_panel(ax):
    ax.axis("off")
    ax.set_xlim(-0.9, 11.6)
    ax.set_ylim(-0.7, 9.6)
    gr = shared_top(ax, 4.0)
    hb = (gr[0], 3.8)
    node(ax, hb, "", BELIEF, "belief", bold=True)
    edge(ax, gr, hb, "grasped", bold=True, dy=(0.3, 0.42), left=True)
    children = [("stop", -2.4, 1.0, False), ("to(1.0)", -1.05, 1.3, False),
                ("to(0.62)", 0.45, 1.45, True), ("to(0.71)", 1.95, 1.3, False)]
    for text, dx, width, bold in children:
        xy = (hb[0] + dx, 2.5)
        node(ax, xy, text, ACTION, width=width, size=8, bold=bold)
        edge(ax, hb, xy, bold=bold)
    ax.text(hb[0] + 0.5, 3.45, "widen while #children ≤ k·N^α", fontsize=7,
            color=CHOSEN, ha="left")
    chosen = (hb[0] + 0.45, 2.5)
    clear, limit = (chosen[0] - 0.9, 0.8), (chosen[0] + 1.1, 0.8)
    node(ax, clear, "", BELIEF, "belief", bold=True)
    node(ax, limit, "", BELIEF, "belief")
    edge(ax, chosen, clear, "Clear", bold=True, dy=(0.3, 0.42), left=True)
    edge(ax, chosen, limit, "Limit", dy=(0.3, 0.42))
    particles(ax, clear, [1, 1, 1, 1, 1, 1])
    particles(ax, limit, [0.2, 1.0, 0.5])
    ax.text(clear[0], clear[1] - 0.95, "safe stop short of\nmost limits", fontsize=7,
            ha="center", va="top")
    ax.text(limit[0], limit[1] - 0.95, "particles weighted\nby p(o | s′)", fontsize=7,
            ha="center", va="top")
    ax.text(7.0, 9.3,
            "Same loop as POUCT, plus:\n"
            "• action widening: a node gets a new action\n"
            "   only while it has ≤ k·N^α children; a limb\n"
            "   target is drawn anywhere on the start→goal\n"
            "   line (failed MPC targets are blocked)\n"
            "• observation widening: a new observation\n"
            "   child only while ≤ k_o·N^α_o; otherwise an\n"
            "   existing one is revisited\n"
            "• each observation node keeps weighted\n"
            "   particles; the next state is drawn from\n"
            "   them, so the belief is updated in the tree\n\n"
            "Finds partial targets (e.g. 0.62) that a\n"
            "fixed grid can only approximate.",
            fontsize=7.8, va="top", bbox=dict(fc="0.96", ec="0.8"))
    ax.set_title("(c) POMCPOW: continuous targets via progressive widening", fontsize=10,
                 loc="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, default=Path("figures/pomdp/algorithms.png"))
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    limits = np.clip(rng.normal(1.0, 0.22, 40), 0.45, 1.25)

    fig = plt.figure(figsize=(16, 11))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.75, 1.5], hspace=0.08, wspace=0.03)
    limb_panel(fig.add_subplot(grid[0, :]), limits, true_limit=0.8)
    pouct_panel(fig.add_subplot(grid[1, 0]))
    pomcpow_panel(fig.add_subplot(grid[1, 1]))
    fig.text(0.5, 0.015, "Blue: actions; orange circles: belief (history) nodes; dots: "
             "particles (size = weight); red: the preferred path. Counts are illustrative.",
             ha="center", fontsize=8.5, color="0.3")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".svg"), bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
