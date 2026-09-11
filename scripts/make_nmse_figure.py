"""Write out the NMSE definition the model-error figure reports.

`python scripts/make_nmse_figure.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KEEP = "#008D63"
WARN = "#C25A34"
INK = "#141C1B"
FAINT = "#6E7A78"
GRID = "#E2E7E6"

EQUATION = (
    r"$\mathrm{NMSE}\;=\;\frac{\mathrm{MSE}(\hat{\psi})}{\mathrm{MSE}(\psi=0)}"
    r"\;=\;\frac{\sum_{i}\left\|\hat{\psi}_{i}-\psi_{i}\right\|^{2}}"
    r"{\sum_{i}\left\|\psi_{i}\right\|^{2}}$"
)


def figure_nmse(out: Path) -> None:
    """The equation, and what each side of 1.0 means."""
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.66, EQUATION, fontsize=25, color=INK, ha="center", va="center")
    # The denominator is the MPC's own default nominal, not an arbitrary scale.
    ax.text(0.5, 0.30, "denominator: the error of the zeros the MPC starts from",
            fontsize=10.5, color=FAINT, ha="center", va="center")
    for x, colour, text in (
        (0.16, KEEP, "< 1  better"),
        (0.5, FAINT, "1  no better"),
        (0.84, WARN, "> 1  worse"),
    ):
        ax.text(x, 0.10, text, fontsize=11, color=colour, ha="center", va="center")
    ax.plot([0.06, 0.94], [0.20, 0.20], color=GRID, lw=1.0)

    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(out / f"nmse.{suffix}", dpi=300, bbox_inches="tight")
        print(f"wrote {out / f'nmse.{suffix}'}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    figure_nmse(args.out)


if __name__ == "__main__":
    main()
