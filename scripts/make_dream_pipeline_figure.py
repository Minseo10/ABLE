"""The dream pipeline (limb_baseline.dream_pipeline) as one diagram, in the style of
figures/dreaming_loop.png: two episodes, their stages, and the cache they share. The
range of motion is drawn as a wedge: wide when uncertain, narrower as the belief learns.

    python scripts/make_dream_pipeline_figure.py --out figures/dream_pipeline.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Wedge  # noqa: E402

GREY = "#8C8C8C"
BELIEF = "#2CA02C"
DREAM = "#D62728"
REAL = "#FF7F0E"
MEMORY = "#1F77B4"
INK = "#222222"
ROW = "#EEF3F8"


def wedge(ax, x, y, span, color, r=0.62, label=None, alpha=0.35):
    """The believed range of motion: an arc sector `span` degrees wide."""
    ax.add_patch(Wedge((x, y), r, 90 - span / 2, 90 + span / 2, width=r * 0.45,
                       facecolor=color, edgecolor=color, alpha=alpha, lw=1.5, zorder=2))
    if label:
        ax.text(x, y - 0.25, label, ha="center", va="top", color=color, fontsize=11,
                linespacing=1.3)


def box(ax, x, y, w, h, text, color, dashed=False, bold=False, size=10.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", lw=1.6,
        edgecolor=color, facecolor="white", ls="--" if dashed else "-", zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color,
            fontsize=size, fontweight="bold" if bold else "normal", zorder=3,
            linespacing=1.35)


def arrow(ax, a, b, color=INK, dashed=False, rad=0.0, lw=1.6):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=15, lw=lw, color=color,
        ls="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}", zorder=1))


def title(ax, x, y, text, color, size=13):
    ax.text(x, y, text, ha="left", va="center", color=color, fontsize=size,
            fontweight="bold")


def cache_rows(ax, x, y, rows, width=6.2):
    for i, (key, value, color) in enumerate(rows):
        yy = y - i * 0.46
        ax.add_patch(FancyBboxPatch((x, yy - 0.19), width, 0.38, boxstyle="round,pad=0.01",
                                    facecolor=ROW, edgecolor="none", zorder=1))
        ax.text(x + 0.15, yy, key, va="center", fontsize=10.5, color=color)
        ax.text(x + width - 0.15, yy, value, va="center", ha="right", fontsize=10.5,
                color=color)


def draw(out: Path) -> None:
    """16:9, for a slide: episode 1 on the left, episode 2 on the right."""
    fig, ax = plt.subplots(figsize=(20, 11.25))
    # Cropped to the content, 16:9.
    ax.set_xlim(0, 29.2)
    ax.set_ylim(2.4, 18.8)
    ax.set_aspect("equal")
    ax.axis("off")

    # ------------------------------------------------------------------ episode 1
    title(ax, 0.3, 17.4, "Episode 1", INK, size=19)

    # 1-2) information gathering and human model learning, off in current runs
    title(ax, 0.3, 16.55, "1) Info gathering  2) Human model learning", GREY)
    ax.text(8.4, 16.55, "(commented out in current runs)", color=GREY, fontsize=11,
            style="italic", va="center")
    wedge(ax, 1.3, 15.05, 150, GREY, label="prior")
    box(ax, 2.6, 15.0, 3.1, 0.9, "ask queries\n(push to a target angle)", GREY, dashed=True)
    box(ax, 6.3, 15.0, 3.1, 0.9, "update belief\nfrom the answers", GREY, dashed=True)
    wedge(ax, 10.7, 15.05, 120, BELIEF, label="belief")
    arrow(ax, (2.05, 15.45), (2.6, 15.45), GREY, dashed=True)
    arrow(ax, (5.7, 15.45), (6.3, 15.45), GREY, dashed=True)
    arrow(ax, (9.4, 15.45), (9.95, 15.45), GREY, dashed=True)

    # 3) dreams
    title(ax, 0.3, 14.05, "3) Dreams  (x n_dreams, real belief untouched)", DREAM)
    wedge(ax, 1.3, 11.8, 120, BELIEF, label="belief\n(Gaussian + floor)")
    for i, (y, span) in enumerate(((12.75, 105), (11.55, 90), (10.35, 70))):
        arrow(ax, (2.05, 12.2), (4.7, y + 0.35), DREAM, rad=-0.15 if i == 0 else 0.1 * i)
        wedge(ax, 5.45, y, span, DREAM, r=0.5,
              label=("imagined 1", "imagined 2", "imagined n")[i])
    ax.text(0.3, 9.95, "one dream:", color=DREAM, fontsize=11, fontweight="bold")
    ax.text(0.3, 9.7,
            "pretend person = 1 belief particle\n"
            "pick 1 of 6 goals, plan under belief\n"
            "imagine executing on pretend person\n"
            "-> belief copy, floor raised",
            color=DREAM, fontsize=10.5, va="top", linespacing=1.4)
    # One arrow per dream: its plan, filed under the belief that dream imagined.
    for start, row in ((13.1, 13.0), (11.9, 12.54), (10.7, 12.08)):
        arrow(ax, (6.15, start), (10.45, row), MEMORY)
    ax.text(8.1, 13.25, "file its plan\nunder this belief", ha="center", fontsize=10,
            color=MEMORY)
    ax.text(6.2, 9.7, "no 2nd search: the plan is refiled,\n"
                      "search again only if the aim point moves",
            fontsize=10, color=DREAM, va="top")

    # cache
    title(ax, 10.5, 14.05, "cache  (kept for all episodes)", MEMORY)
    cache_rows(ax, 10.5, 13.0, [
        ("(goal 3, imagined 1)", "plan", DREAM),
        ("(goal 1, imagined 2)", "plan", DREAM),
        ("...   (imagined n)", "", DREAM),
        ("(goal 3, belief)   <- real step", "plan", REAL),
    ])
    ax.text(10.5, 10.95,
            "key = goal (0.1 rad) + belief (mean, std, floor)\n"
            "lookup = nearest belief, each number within 0.15\n"
            "learned: (pose, goal, belief) -> torques, refit per success",
            fontsize=10, color=GREY, va="top", linespacing=1.4)

    # 4) real execution
    title(ax, 0.3, 7.6, "4) Real execution", REAL)
    wedge(ax, 1.3, 5.75, 120, BELIEF, label="belief")
    box(ax, 2.55, 5.65, 1.5, 0.95, "goal 3", REAL)
    box(ax, 4.55, 5.4, 3.5, 1.45,
        "look up cache (or model)\nhit -> replay in sim\nmiss -> search (<= 600 s)",
        MEMORY, size=10.5)
    box(ax, 8.55, 5.65, 1.4, 0.95, "plan", REAL, bold=True)
    box(ax, 10.45, 5.45, 3.4, 1.35, "execute on TRUE person\nupdate belief from the path",
        REAL, size=10.5)
    wedge(ax, 15.3, 5.75, 85, BELIEF, label="belief_updated\n(narrower)")
    for a, b in ((2.05, 2.55), (4.05, 4.55), (8.05, 8.55), (9.95, 10.45), (13.85, 14.55)):
        arrow(ax, (a, 6.12), (b, 6.12))
    ax.text(2.55, 5.05, "goal beyond belief -> aim 0.05 rad inside the edge",
            fontsize=10, color=REAL, va="top")
    ax.text(10.45, 5.05, "stayed inside -> floor raised\ncrossed -> limit located",
            fontsize=10, color=BELIEF, va="top", linespacing=1.4)
    arrow(ax, (9.25, 6.6), (12.3, 11.45), REAL, dashed=True, rad=-0.25)
    ax.text(9.7, 8.3, "store (goal 3, belief) -> plan\nif reached safely", fontsize=10,
            color=REAL)

    # divider between the episodes
    ax.plot([17.6, 17.6], [3.0, 17.9], color="#BBBBBB", lw=1.0)

    # ------------------------------------------------------------------ episode 2
    title(ax, 18.2, 17.4, "Episode 2", INK, size=19)
    title(ax, 18.2, 16.55, "3) Dreams, from belief_updated", DREAM)
    wedge(ax, 19.4, 14.45, 85, BELIEF)
    ax.text(19.4, 15.35, "belief = belief_updated", ha="center", color=BELIEF, fontsize=11)
    for i, (y, span) in enumerate(((15.1, 75), (13.75, 60))):
        arrow(ax, (20.15, 14.85), (22.8, y + 0.35), DREAM, rad=0.05 - 0.1 * i)
        wedge(ax, 23.55, y, span, DREAM, r=0.5, label=("imagined 1", "imagined 2")[i])
    # the belief carries over from episode 1
    arrow(ax, (16.2, 6.5), (18.8, 13.95), BELIEF, dashed=True)

    title(ax, 18.2, 10.3, "4) Real execution", REAL)
    box(ax, 18.9, 7.6, 5.4, 1.6,
        "look up cache[(goal, belief_updated)]\nhit if an episode-1 imagined belief\nis within 0.15 of it",
        MEMORY, size=10.5)
    box(ax, 24.8, 7.9, 1.5, 1.0, "plan", REAL, bold=True)
    box(ax, 26.8, 7.7, 2.3, 1.4, "execute on\nTRUE person,\nupdate", REAL, size=10.5)
    arrow(ax, (24.3, 8.4), (24.8, 8.4))
    arrow(ax, (26.3, 8.4), (26.8, 8.4))
    # dashed pay-off line, as in dreaming_loop.png: episode-1 dream rows used here
    ax.plot([16.85, 17.2, 17.2, 21.6], [12.1, 12.1, 11.4, 11.4], color=MEMORY, ls="--",
            lw=1.4)
    arrow(ax, (21.6, 11.4), (21.6, 9.25), MEMORY, dashed=True)
    ax.text(21.8, 11.55, "episode-1 dream rows pay off in episode 2", fontsize=10.5,
            color=MEMORY, va="bottom")
    ax.text(18.9, 7.1, "plain: no cache, no dreams, search every time", fontsize=10.5,
            color=GREY, va="top")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".svg"))
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("figures/dream_pipeline.png"))
    draw(parser.parse_args().out)


if __name__ == "__main__":
    main()
