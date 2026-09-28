"""The notebook's section 5 figure for the dream pipeline, per person and averaged.

Panels, left to right: (a) per-person only, the task against the person's range on each
active joint (true range, believed range at the first and last episode, goals, start);
belief width per episode with revisions circled; planning effort (MPC rollouts); planning
time; final distance to the original goal, the notebook's error.

One line per (condition, model, dreams): colour by condition and model, dashed for no
dreams. With several people in the CSVs, the averaged figure draws the mean over people
with a standard-error band; `--per-person` also writes one figure per person, where
hollow markers are episodes that did not reach their goal and dotted lines mark revisions.
Dream rows in `*_dreams.csv` are ignored.

    python scripts/make_dream_figures.py --csv data/dream/main/*.csv \
        --out figures/dream/main/summary.png --per-person
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
TRUE = "#9AA5A3"
# By (condition, model family).
COLORS = {"plain": "#8C8C8C", "cache": "#1F77B4", "ridge": "#D62728", "mlp": "#6A3D9A",
          "ridge_step": "#FF7F0E", "mlp_step": "#2CA02C", "net": "#17BECF", "net_step": "#0B5563"}
EFFORT_LABELS = {
    "rollouts": "MPC rollouts used",
    "simulator_steps": "simulator steps used",
    "mpc_steps": "MPC steps used",
}

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


def _truthy(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.lower().eq("true").to_numpy()


def _family(condition: str, model: str, seed_mode: str = "plan") -> str:
    if condition != "learned":
        return condition
    family = str(model).split(":")[0] if str(model).split(":")[0] in ("ridge", "net") else "mlp"
    return f"{family}_step" if seed_mode == "step" else family


def _series(frame: pd.DataFrame) -> list[tuple[str, dict, pd.DataFrame]]:
    """One (label, style, rows) per line: plain, then cache, ridge, MLP; no dreams first."""
    out = []
    keys = ["condition", "model", "n_dreams", "seed_mode"]
    for (condition, model, n_dreams, seed_mode), rows in frame.groupby(keys):
        family = _family(condition, model, seed_mode)
        if condition == "plain":
            label = "plain (no memory)"
        else:
            what = "cache" if condition == "cache" else f"learned {model}"
            if family.endswith("_step"):
                what += " (step seed)"
            label = f"{what}, experience only" if n_dreams == 0 else f"{what}, {n_dreams} dreams/ep"
        style = {"color": COLORS[family],
                 "ls": "--" if condition != "plain" and n_dreams == 0 else "-"}
        out.append(((list(COLORS).index(family), int(n_dreams), label), label, style, rows))
    out.sort(key=lambda s: s[0])
    return [(label, style, rows) for _, label, style, rows in out]


def _finish(ax, frame, ylabel: str, log_scale: bool) -> None:
    ax.set_xlabel("episode")
    ax.set_ylabel(ylabel)
    episodes = sorted(frame["episode"].unique())
    ax.set_xticks(episodes[:: max(1, len(episodes) // 10)])
    if log_scale:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw_person(ax, frame: pd.DataFrame, field: str, log_scale: bool, ylabel: str,
                revisions: bool = True) -> None:
    """One person: every line as measured, hollow where the goal was not reached."""
    for label, style, rows in _series(frame):
        rows = rows.sort_values("episode")
        x = rows["episode"].to_numpy()
        y = rows[field].to_numpy(dtype=float)
        reached = _truthy(rows["goal_reached"])
        ax.plot(x, y, lw=1.6, label=label, zorder=2, **style)
        ax.scatter(x[reached], y[reached], s=20, color=style["color"], zorder=3)
        ax.scatter(x[~reached], y[~reached], s=26, facecolor="white",
                   edgecolor=style["color"], lw=1.3, zorder=3)
    if revisions and "belief_revised" in frame:
        for episode in frame.loc[_truthy(frame["belief_revised"]), "episode"].unique():
            ax.axvline(episode, color="k", lw=0.5, ls=":", alpha=0.5)
    _finish(ax, frame, ylabel, log_scale)


def draw_mean(ax, frame: pd.DataFrame, field: str, log_scale: bool, ylabel: str) -> None:
    """Across people: the mean per episode and a standard-error band."""
    for label, style, rows in _series(frame):
        stats = rows.groupby("episode")[field].agg(["mean", "sem", "count"])
        x = stats.index.to_numpy()
        m = stats["mean"].to_numpy(dtype=float)
        se = np.nan_to_num(stats["sem"].to_numpy(dtype=float))
        people = rows["person"].nunique()
        ax.plot(x, m, lw=1.6, marker="o", ms=3, label=f"{label} (n={people})", zorder=2,
                **style)
        low = np.maximum(m - se, 1e-9) if log_scale else m - se
        ax.fill_between(x, low, m + se, color=style["color"], alpha=0.12, lw=0)
    _finish(ax, frame, ylabel, log_scale)


def draw_task(ax, frame: pd.DataFrame) -> None:
    """Panel (a): each active joint as a row, the task placed against the ranges."""
    first = frame.sort_values("episode").groupby("episode").head(1)
    row = first.iloc[0]

    def vec(value) -> np.ndarray:
        return np.array([float(v) for v in str(value).split()])

    true_lo, true_hi = vec(row["true_lower_active"]), vec(row["true_upper_active"])
    start = vec(row["start_active"])
    goals = np.vstack([vec(g) for g in frame["goal_active"].unique()])
    joints = len(true_lo)
    for j in range(joints):
        y = joints - 1 - j
        ax.plot([true_lo[j], true_hi[j]], [y, y], color=TRUE, lw=9, solid_capstyle="butt",
                zorder=1, label="true range" if j == 0 else None)
        # Believed range at the first episode, and at the last for each condition.
        b0_lo, b0_hi = vec(row["believed_lower_active"]), vec(row["believed_upper_active"])
        ax.plot([b0_lo[j], b0_hi[j]], [y + 0.22, y + 0.22], color="k", lw=1.2,
                label="believed, first episode" if j == 0 else None)
        for k, (label, style, rows) in enumerate(_series(frame)):
            last = rows.sort_values("episode").iloc[-1]
            lo, hi = vec(last["believed_lower_active"]), vec(last["believed_upper_active"])
            offset = -0.2 - 0.06 * k
            ax.plot([lo[j], hi[j]], [y + offset] * 2, lw=1.0, color=style["color"],
                    ls=style["ls"])
        ax.scatter(goals[:, j], [y] * len(goals), marker="*", s=90, color="#FF7F0E",
                   edgecolor="k", lw=0.4, zorder=4, label="goals" if j == 0 else None)
        ax.scatter([start[j]], [y], marker="o", s=30, color="k", zorder=5,
                   label="start" if j == 0 else None)
    ax.set_yticks(range(joints))
    ax.set_yticklabels([f"active joint {joints - i}" for i in range(joints)])
    ax.set_xlabel("joint angle (rad)")
    ax.set_ylim(-1.0, joints - 0.4)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# One row per method, each against plain: its experience-only and its dreaming line.
GROUPS = (
    ("Cache", lambda f: f["condition"].eq("cache")),
    ("Learned: ridge", lambda f: f["condition"].eq("learned")
     & f["model"].astype(str).str.startswith("ridge")),
    ("Learned: MLP", lambda f: f["condition"].eq("learned")
     & f["model"].astype(str).str.startswith("mlp")),
    ("Learned: new MLP", lambda f: f["condition"].eq("learned")
     & f["model"].astype(str).str.startswith("net")),
)


def figure(frame: pd.DataFrame, out: Path, effort: str, per_person: bool, title: str) -> None:
    """One row per method (cache, ridge, MLP), each drawn with plain for reference; the
    columns are belief, effort, time and distance, sharing a y-axis down each column.
    Per person, the task panel spans the rows on the left."""
    has_distance = "goal_distance" in frame and frame["goal_distance"].notna().any()
    panels = [("belief_width", False, "belief width (sum of std)",
               "Belief about the range of motion"),
              (effort, True, EFFORT_LABELS[effort], "Planning effort per episode"),
              ("plan_time", False, "planning time (s)", "Planning time per episode"),]
    if has_distance:
        panels.append(("goal_distance", False, "distance to goal (rad)",
                       "Distance left to the original goal"))
    plain = frame[frame["condition"].eq("plain")]
    groups = [(name, frame[pick(frame)]) for name, pick in GROUPS]
    groups = [(name, rows) for name, rows in groups if len(rows)]
    if not groups:
        groups = [("Plain", frame.iloc[0:0])]
    cols = len(panels) + per_person
    fig = plt.figure(figsize=(4.4 * cols + 3.2, 3.3 * len(groups) + 0.9))
    grid = fig.add_gridspec(len(groups), cols, width_ratios=[1.25] * per_person + [1] * len(panels))
    if per_person:
        task = fig.add_subplot(grid[:, 0])
        draw_task(task, frame)
        task.set_title("(a) Task against the range of motion\n"
                       "coloured ticks: believed range at the last episode",
                       fontsize=10, loc="left")
    columns: list[list] = [[] for _ in panels]
    for r, (name, rows) in enumerate(groups):
        sub = pd.concat([plain, rows], ignore_index=True)
        for c, (field, log_scale, ylabel, panel) in enumerate(panels):
            ax = fig.add_subplot(grid[r, c + per_person])
            if per_person:
                draw_person(ax, sub, field, log_scale, ylabel)
            else:
                draw_mean(ax, sub, field, log_scale, ylabel)
            if r == 0:
                ax.set_title(panel, fontsize=10.5, loc="left")
            if r < len(groups) - 1:
                ax.set_xlabel("")
            columns[c].append(ax)
            if c == 0:
                ax.text(-0.36, 0.5, name, transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=12, fontweight="bold",
                        color=INK)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.03, 0.5),
                  fontsize=8.5, frameon=False)
    # Same y-range down each column, so the rows compare at a glance.
    for axes in columns:
        low = min(a.get_ylim()[0] for a in axes)
        high = max(a.get_ylim()[1] for a in axes)
        for a in axes:
            a.set_ylim(low, high)
    if title:
        fig.suptitle(title, fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    print(f"Wrote {out}")


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Per line, averaged over people: success, effort, time, distance, hits, dreams."""
    frame = frame.copy()
    frame["reached"] = _truthy(frame["goal_reached"])
    agg = {
        "people": ("person", "nunique"),
        "episodes": ("episode", "count"),
        "reached": ("reached", "mean"),
        "rollouts_mean": ("rollouts", "mean"),
        "rollouts_median": ("rollouts", "median"),
        "plan_time_mean": ("plan_time", "mean"),
        "plan_time_median": ("plan_time", "median"),
    }
    for column, how in (("goal_distance", "mean"), ("goal_beyond_belief", "sum"),
                        ("memo_hits", "sum"), ("policy_reached", "sum"),
                        ("dream_time", "sum"), ("dreams_planned", "sum"),
                        ("dreams_reached", "sum")):
        if column in frame:
            agg[f"{column}_{how}"] = (column, how)
    keys = ["condition", "model", "n_dreams", "seed_mode"]
    return frame.groupby(keys).agg(**agg).round(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, nargs="+", action="extend", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--effort", choices=sorted(EFFORT_LABELS), default="rollouts")
    parser.add_argument("--per-person", action="store_true",
                        help="Also write <out stem>_person<id>.png per person.")
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Draw only the first this many episodes of every run.")
    args = parser.parse_args()

    paths = [p for p in args.csv if not p.stem.endswith("_dreams")]
    frame = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if "model" not in frame:
        frame["model"] = ""
    frame["model"] = frame["model"].fillna("").astype(str)
    # Older CSVs predate the seed mode; they ran in plan mode.
    if "seed_mode" not in frame:
        frame["seed_mode"] = "plan"
    frame["seed_mode"] = frame["seed_mode"].fillna("plan").astype(str)
    if args.max_episodes is not None:
        frame = frame[frame["episode"] < args.max_episodes]
    people = sorted(frame["person"].unique())
    title = args.title or f"Mean over {len(people)} people, band = standard error"
    figure(frame, args.out, args.effort, per_person=False, title=title)
    if args.per_person:
        for person in people:
            out = args.out.with_name(f"{args.out.stem}_person{person}{args.out.suffix}")
            figure(frame[frame["person"] == person], out, args.effort, per_person=True,
                   title=f"Person {person}")
    print()
    print(summarize(frame).to_string())


if __name__ == "__main__":
    main()
