"""Plan time per task iteration, plain PDDLStream against the memorized stream.

One panel per experiment (the same task repeated, and the task with small variation);
`--csv` may be given more than once. Hollow markers are iterations whose plan did not
reach the goal.

    python scripts/make_memo_figures.py --csv data/memo/same_task.csv \
        --csv data/memo/varied_task.csv --out figures/memo/plan_time.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#141C1B"
FAINT = "#6E7A78"
GRID = "#E2E7E6"
COLORS = {"plain": "#AAB4B2", "warm": "#008D63"}
LABELS = {"plain": "plain (PDDLStream as is)", "warm": "warm (memorized plan-limb-motion)"}

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
        "legend.frameon": False,
    }
)


def _title(frame: pd.DataFrame) -> str:
    variation = float(frame["variation"].iloc[0])
    n = int(frame["iteration"].max()) + 1
    if variation <= 0.0:
        return f"Same task, {n} times"
    what = "goal and start" if bool(frame["vary_init"].iloc[0]) else "goal"
    return f"{n} tasks, {what} within {variation:g} rad of one another"


def draw(ax: plt.Axes, frame: pd.DataFrame, log_scale: bool) -> None:
    """One experiment: plan time against iteration, one line per condition."""
    for condition in ("plain", "warm"):
        rows = frame[frame["condition"] == condition].sort_values("iteration")
        if rows.empty:
            continue
        color = COLORS[condition]
        x = rows["iteration"].to_numpy()
        y = rows["plan_time"].to_numpy(dtype=float)
        reached = rows["goal_reached"].astype(str).str.lower().eq("true").to_numpy()
        solved = y[reached]
        label = LABELS[condition]
        if len(solved):
            median = float(np.median(solved))
            label += f", median {median:.0f} s" if median >= 10 else f", median {median:.2f} s"
        ax.plot(x, y, "-", color=color, lw=1.8, label=label, zorder=2)
        ax.scatter(x[reached], y[reached], s=28, color=color, zorder=3)
        ax.scatter(
            x[~reached], y[~reached], s=34, facecolor="white", edgecolor=color,
            lw=1.5, zorder=3,
        )
    ax.set_xlabel("task iteration")
    ax.set_ylabel("planning time (s)")
    ax.set_title(_title(frame), fontsize=10, loc="left")
    ax.set_xticks(sorted(frame["iteration"].unique()))
    if log_scale:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(fontsize=8, loc="best")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", action="store_true", help="Log-scale the time axis.")
    args = parser.parse_args()

    frames = [pd.read_csv(path) for path in args.csv]
    fig, axes = plt.subplots(
        1, len(frames), figsize=(5.4 * len(frames), 3.8), squeeze=False
    )
    for ax, frame in zip(axes[0], frames):
        draw(ax, frame, args.log)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    fig.savefig(args.out.with_suffix(".svg"))
    print(f"Wrote {args.out}")

    for frame in frames:
        print(f"\n{_title(frame)}")
        summary = (
            frame.groupby("condition")
            .agg(
                episodes=("iteration", "count"),
                reached=("goal_reached", lambda s: s.astype(str).str.lower().eq("true").mean()),
                plan_time_median=("plan_time", "median"),
                plan_time_mean=("plan_time", "mean"),
                memo_hits=("memo_hits", "sum") if "memo_hits" in frame else ("iteration", "size"),
            )
            .round(2)
        )
        print(summary.to_string())


if __name__ == "__main__":
    main()
