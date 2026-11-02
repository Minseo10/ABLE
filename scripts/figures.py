"""Per-episode curves (mean over people) and summary.csv (second-half means) for one experiment.

    python scripts/figures.py data/friction_full --out figures/friction_full
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# Fixed colour per condition across figures.
COLORS = {"plain": "#2a78d6", "learned": "#eb6834", "prior_only": "#1baf7a"}
MARKERS = {"plain": "o", "learned": "s", "prior_only": "^"}
LABELS = {"plain": "plain", "learned": "ABLE (learned)", "prior_only": "prior only"}
INK, MUTED, GRID = "#1f2328", "#59636e", "#e6e8eb"


def load(directory: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(directory.glob("*_p*.csv"))]
    if not frames:
        raise SystemExit(f"no runs in {directory}")
    frame = pd.concat(frames, ignore_index=True)
    frame["success"] = frame["success"].astype(str).str.lower().eq("true").astype(float)
    return frame


def plot(frame: pd.DataFrame, out: Path) -> None:
    unit = frame["effort_unit"].iloc[0].replace("_", " ")
    panels = [("plan_seconds", "planning time (s)"), ("effort", unit), ("success", "success rate")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    for ax, (column, title) in zip(axes, panels):
        for cond in [c for c in COLORS if c in set(frame.condition)]:
            curve = frame[frame.condition == cond].groupby("episode")[column].mean()
            ax.plot(curve.index, curve.values, color=COLORS[cond], lw=2, marker=MARKERS[cond], ms=4,
                    label=LABELS[cond])
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("episode", color=MUTED, fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes[2].set_ylim(-0.05, 1.05)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    task, people = frame.task.iloc[0], frame.person.nunique()
    fig.suptitle(f"{task}: mean over {people} people", color=INK, fontsize=11, x=0.01, ha="left")
    fig.savefig(out / "curves.png", dpi=160)
    plt.close(fig)


def summary(frame: pd.DataFrame, out: Path) -> pd.DataFrame:
    late = frame[frame.episode >= frame.episode.max() / 2]
    table = late.groupby("condition")[["plan_seconds", "effort", "success"]].mean().round(3)
    table["people"] = frame.groupby("condition").person.nunique()
    table.to_csv(out / "summary.csv")
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or Path("figures") / args.data.name
    out.mkdir(parents=True, exist_ok=True)
    frame = load(args.data)
    plot(frame, out)
    print(summary(frame, out))


if __name__ == "__main__":
    main()
