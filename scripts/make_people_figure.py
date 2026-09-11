"""Where the two people studied sit in the population's range of motion.

`python scripts/make_people_figure.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from limb_baseline.human import DEFAULT_PRIOR, active_motions  # noqa: E402

KEEP = "#008D63"
DROP = "#AAB4B2"
WARN = "#C25A34"
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

# The seeds the two collected runs drew their person from.
PEOPLE = {"person 1": 0, "person 2": 22}


def figure_people(out: Path) -> None:
    """The prior over `theta`, with each person's three motions marked on it."""
    prior = DEFAULT_PRIOR
    grid = np.linspace(prior.low - 0.03, prior.high + 0.03, 400)
    density = np.exp(-0.5 * ((grid - prior.mean) / prior.stddev) ** 2)
    density /= density.max()
    motions = [m.replace("_", " ") for m in active_motions()]

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.fill_between(grid, density, color=DROP, alpha=0.30, lw=0)
    ax.plot(grid, density, color=FAINT, lw=1.3)
    ax.axvline(prior.mean, color=FAINT, lw=1.1, ls="--")
    ax.text(prior.mean, -0.075, "population mean", color=FAINT, fontsize=9, ha="center")
    for edge in (prior.low, prior.high):
        ax.axvline(edge, color=WARN, lw=1.1, ls=":", alpha=0.8)

    height = lambda t: float(  # noqa: E731
        np.exp(-0.5 * ((t - prior.mean) / prior.stddev) ** 2)
    )
    for (name, seed), colour in zip(PEOPLE.items(), (INK, KEEP), strict=True):
        theta = prior.sample(np.random.default_rng(seed))
        ax.plot(theta, [height(t) for t in theta], "o", ms=10, color=colour,
                ls="none", label=f"{name} (mean {theta.mean():.2f})", zorder=4)
        # Two motions can land close together; stagger the labels so they stay legible.
        for index, (value, motion) in enumerate(zip(theta, motions, strict=True)):
            ax.annotate(motion, (value, height(value)), textcoords="offset points",
                        xytext=(0, 13 + 15 * (index % 2)), ha="center", fontsize=8,
                        color=colour)

    ax.set_xlabel("$\\theta$ (range of motion scale)")
    ax.set_ylabel("prior density")
    ax.set_yticks([])
    ax.set_ylim(-0.12, 1.30)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")

    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(out / f"people.{suffix}", dpi=300, bbox_inches="tight")
        print(f"wrote {out / f'people.{suffix}'}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    figure_people(args.out)


if __name__ == "__main__":
    main()
