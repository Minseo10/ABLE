"""Compare the personalization pipelines round by round.

`python scripts/make_personalize_figures.py --run-dir data/personalize/run`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#141C1B"
FAINT = "#6E7A78"
GRID = "#E2E7E6"
# One colour per arm, assigned in the order the arms appear in rounds.csv.
PALETTE = ("#008D63", "#C25A34", "#3F6FA8", "#8A5FA8")

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": FAINT,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": FAINT,
        "ytick.color": FAINT,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

PANELS = (
    ("success", "goal reached", lambda g: g["goal_reached"].mean()),
    ("plan_time", "planning time (s)", lambda g: g["plan_time"].mean()),
    ("mpc_steps", "MPC simulator steps",
     lambda g: g[g["goal_reached"]]["mpc_simulator_steps"].mean()),
    ("belief_error", "max |belief - theta|", lambda g: g["belief_error"].mean()),
)


def curves(frame: pd.DataFrame, out: Path) -> None:
    """Four metrics against round, one line per arm."""
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.0 * len(PANELS), 3.4))
    versions = list(dict.fromkeys(frame["version"]))
    for axis, (_, label, metric) in zip(axes, PANELS, strict=True):
        for colour, version in zip(PALETTE, versions, strict=False):
            rows = frame[frame["version"] == version]
            points = rows.groupby("round").apply(metric, include_groups=False)
            axis.plot(points.index, points.values, "o-", color=colour, label=version)
        axis.set_xlabel("round")
        axis.set_ylabel(label)
        axis.grid(color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(out / f"personalize.{suffix}", dpi=300, bbox_inches="tight")
        print(f"wrote {out / f'personalize.{suffix}'}")
    plt.close(fig)


def cost(frame: pd.DataFrame, out: Path) -> None:
    """What each arm bought against the demonstrations it spent getting there."""
    fig, axis = plt.subplots(figsize=(5.4, 3.6))
    for colour, version in zip(PALETTE, dict.fromkeys(frame["version"]), strict=False):
        rows = frame[frame["version"] == version]
        by_round = rows.groupby("round")
        axis.plot(by_round["num_train_rows"].max().values,
                  by_round["goal_reached"].mean().values,
                  "o-", color=colour, label=version)
    axis.set_xlabel("personal training rows")
    axis.set_ylabel("goal reached")
    axis.set_ylim(0.0, 1.0)
    axis.grid(color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(out / f"personalize_cost.{suffix}", dpi=300, bbox_inches="tight")
        print(f"wrote {out / f'personalize_cost.{suffix}'}")
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    frame = pd.read_csv(args.run_dir / "rounds.csv")
    out = args.out or Path("figures") / "personalize" / args.run_dir.name
    curves(frame, out)
    cost(frame, out)


if __name__ == "__main__":
    main()
