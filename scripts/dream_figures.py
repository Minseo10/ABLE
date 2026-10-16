"""Figures for the dream pipeline's runs.

    python scripts/dream_figures.py summary --csv data/dream/fresh_goals/*_p??.csv \
        --out figures/dream/fresh_goals/summary.png --per-person
    python scripts/dream_figures.py curves --logs data/dream/warmstart_compare/*_4_step_p*_learning.jsonl
    python scripts/dream_figures.py step [--data data/dream/warmstart_compare]
    python scripts/dream_figures.py samples [--person 24]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# The look of every figure but the summary, which keeps its own (SUMMARY_STYLE).
INK, INK2, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
STYLE = {
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
    "lines.linewidth": 2, "figure.dpi": 150,
}


# --- summary ----------------------------------------------------------------------------

SUMMARY_INK = "#141C1B"
FAINT = "#6E7A78"
SUMMARY_GRID = "#E2E7E6"
TRUE = "#9AA5A3"
# By (condition, model family).
COLORS = {"plain": "#8C8C8C", "cache": "#1F77B4", "ridge": "#D62728", "mlp": "#6A3D9A",
          "ridge_step": "#FF7F0E", "mlp_step": "#2CA02C", "net": "#17BECF", "net_step": "#0B5563",
          "gen_flow_step": "#4a3aa7", "gen_ddpm_step": "#e87ba4"}
EFFORT_LABELS = {
    "rollouts": "MPC rollouts used",
    "simulator_steps": "simulator steps used",
    "mpc_steps": "MPC steps used",
}

SUMMARY_STYLE = {
    "font.size": 10,
    "axes.edgecolor": FAINT,
    "axes.labelcolor": SUMMARY_INK,
    "axes.labelsize": 11,
    "text.color": SUMMARY_INK,
    "xtick.color": FAINT,
    "ytick.color": FAINT,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "legend.frameon": False,
}


def _truthy(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.lower().eq("true").to_numpy()


def _family(condition: str, model: str, seed_mode: str = "plan") -> str:
    if condition != "learned":
        return condition
    kind = str(model).split(":")
    family = (f"gen_{kind[1]}" if kind[0] == "gen" else kind[0] if kind[0] in ("ridge", "net")
              else "mlp")
    return f"{family}_step" if seed_mode == "step" else family


def _model_name(model: str) -> str:
    """A model spec as the figures name it."""
    parts = str(model).split(":")
    if parts[0] == "net":
        return f"MLP {parts[1].replace('x', '×')}" + (" + vel/torque" if parts[-1] == "hist" else "")
    if parts[0] == "mlp" and len(parts) > 1:
        return f"MLP {parts[1].replace('x', '×')}"
    if parts[0] == "gen":
        return {"flow": "flow matching", "ddpm": "diffusion"}.get(parts[1], parts[1])
    return f"learned {model}"


PLANNER_NAMES = {"pddlstream": "PDDLStream", "pouct": "POUCT", "pomcp": "POMCP"}


def _series(frame: pd.DataFrame) -> list[tuple[str, dict, pd.DataFrame]]:
    """One (label, style, rows) per line: plain, then cache, ridge, MLP; no dreams first."""
    out = []
    frame = frame.assign(seed_rollouts=frame.get("seed_rollouts", 1))
    # Older CSVs have no planner column; they all used PDDLStream.
    planners = frame.get("planner", pd.Series("pddlstream", index=frame.index))
    frame = frame.assign(planner=planners.fillna("pddlstream"))
    several = frame["planner"].nunique() > 1
    keys = ["planner", "condition", "model", "n_dreams", "seed_mode", "seed_rollouts"]
    for (planner, condition, model, n_dreams, seed_mode, candidates), rows in frame.groupby(keys):
        family = _family(condition, model, seed_mode)
        if condition == "plain":
            label = "plain (no memory)"
        else:
            what = "cache" if condition == "cache" else _model_name(model)
            if family.endswith("_step"):
                what += " (step seed)"
            label = f"{what}, experience only" if n_dreams == 0 else f"{what}, {n_dreams} dreams/ep"
            if candidates > 1:
                label += f", {candidates} candidates"
        style = {"color": COLORS[family],
                 "ls": ":" if candidates > 1 and family.startswith("net") else
                       "--" if condition != "plain" and n_dreams == 0 else "-"}
        if several:
            label = f"{PLANNER_NAMES.get(planner, planner)}: {label}"
            if planner != "pddlstream":
                style["ls"] = "-."
        order = (list(COLORS).index(family), int(n_dreams), int(candidates), planner, label)
        out.append((order, label, style, rows))
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
    ax.grid(axis="y", color=SUMMARY_GRID, lw=0.8)
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
    ("Learned: MLP 128×128", lambda f: f["condition"].eq("learned")
     & f["model"].astype(str).str.startswith("mlp")),
    ("Learned: MLP 256×256 + vel/torque", lambda f: f["condition"].eq("learned")
     & f["model"].astype(str).str.startswith("net")),
    ("Learned: flow matching", lambda f: f["condition"].eq("learned")
     & f["model"].astype(str).str.startswith("gen:flow")),
    ("Learned: diffusion", lambda f: f["condition"].eq("learned")
     & f["model"].astype(str).str.startswith("gen:ddpm")),
)


def summary_figure(frame: pd.DataFrame, out: Path, effort: str, per_person: bool, title: str) -> None:
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
                        color=SUMMARY_INK)
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


def summary(args) -> None:
    """The notebook's section 5 figure, per person and averaged.

    Panels, left to right: (a) per-person only, the task against the person's range on each
    active joint (true range, believed range at the first and last episode, goals, start);
    belief width per episode with revisions circled; planning effort (MPC rollouts); planning
    time; final distance to the original goal, the notebook's error.

    One line per (condition, model, dreams): colour by condition and model, dashed for no
    dreams. With several people in the CSVs, the averaged figure draws the mean over people
    with a standard-error band; `--per-person` also writes one figure per person, where
    hollow markers are episodes that did not reach their goal and dotted lines mark
    revisions. Dream rows in `*_dreams.csv` are ignored.
    """
    plt.rcParams.update(SUMMARY_STYLE)
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
    summary_figure(frame, args.out, args.effort, per_person=False, title=title)
    if args.per_person:
        for person in people:
            out = args.out.with_name(f"{args.out.stem}_person{person}{args.out.suffix}")
            summary_figure(frame[frame["person"] == person], out, args.effort, per_person=True,
                   title=f"Person {person}")
    print()
    print(summarize(frame).to_string())


# --- curves -----------------------------------------------------------------------------

CURVE_MODELS = {"net": "MLP 256×256 + vel/torque", "net_k8": "MLP 256×256 + vel/torque (8 candidates)",
          "gen_flow_k8": "flow matching (8 candidates)", "gen_ddpm_k8": "diffusion (8 candidates)",
          "mlp": "MLP 128×128"}
TRAIN, VALIDATION, RECENT, NEXT, REAL, DREAM = (
    "#2a78d6", "#eb6834", "#e87ba4", "#1baf7a", "#1baf7a", "#4a3aa7")


def load_refits(paths) -> pd.DataFrame:
    rows = []
    for path in paths:
        match = re.match(r"(.+?)_(\d+)(_step)?(_k\d+)?_p(\d+)_learning", Path(path).stem)
        family, person = match.group(1) + (match.group(4) or ""), int(match.group(5))
        backfill = Path(path).with_name(Path(path).stem + "_backfill.jsonl")
        extra = ([json.loads(line) for line in open(backfill)] if backfill.exists() else [])
        for i, line in enumerate(open(path, encoding="utf-8")):
            row = json.loads(line)
            if i < len(extra):
                row.update({k: extra[i][k] for k in ("val_recent_nmse", "curve")})
            rows.append({**row, "family": family, "person": person})
    return pd.DataFrame(rows)


def per_episode(frame, column, role=None):
    """Mean over people of each person's mean per episode, and its standard error."""
    f = frame if role is None else frame[frame.role.eq("real") == (role == "real")]
    per_person = f.groupby(["person", "episode"])[column].mean().unstack(0)
    return per_person.mean(1), per_person.sem(1)


def band(ax, x, mean, se, color, label, style="-"):
    ax.plot(x, mean, color=color, ls=style, marker="o", ms=3, label=label)
    ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.15, lw=0)


def finish_nmse(ax, title, ylabel):
    ax.axhline(1, color=INK2, lw=1, ls=":")
    ax.set_yscale("log")
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)


def overview(frame, out):
    families = [f for f in CURVE_MODELS if f in set(frame.family)]
    fig, axes = plt.subplots(len(families), 3, figsize=(14, 3.9 * len(families)), squeeze=False)
    for row, family in zip(axes, families):
        f = frame[frame.family == family]
        name = CURVE_MODELS[family]

        ax = row[0]
        m, se = per_episode(f, "train_nmse")
        band(ax, m.index, m, se, TRAIN, "training error (rows it trained on)")
        if "val_nmse" in f and f.val_nmse.notna().any():
            m, se = per_episode(f.dropna(subset=["val_nmse"]), "val_nmse")
            band(ax, m.index, m, se, VALIDATION, "validation error (1/6 of trajectories set aside)")
        m, se = per_episode(f.dropna(subset=["new_nmse"]), "new_nmse")
        band(ax, m.index, m, se, NEXT, "test error (new rows, before the model trains on them)")
        ax.set_xlabel("episode")
        finish_nmse(ax, "Learning curve", "nMSE (1 = predicting the average)")
        ax.annotate(name, xy=(-0.2, 0.5), xycoords="axes fraction", rotation=90, ha="center",
                    va="center", fontsize=12, fontweight="bold", color=INK)

        ax = row[1]
        g = f.dropna(subset=["new_nmse"])
        for role, color, label in (("real", REAL, "test error on real-episode rows"),
                                   ("dream", DREAM, "test error on dream rows")):
            m, se = per_episode(g, "new_nmse", role)
            band(ax, m.index, m, se, color, label)
        ax.set_xlabel("episode")
        finish_nmse(ax, "Test error: real episodes vs dreams", "nMSE on new rows, before training on them")

        ax = row[2]
        real = g.role.eq("real")
        ax.scatter(g[~real].rows, g[~real].new_nmse, s=9, color=DREAM, alpha=0.4, lw=0, label="from a dream")
        ax.scatter(g[real].rows, g[real].new_nmse, s=14, color=REAL, alpha=0.8, lw=0, label="from a real episode")
        ax.set_xscale("log")
        ax.set_xlabel("rows the model had trained on")
        finish_nmse(ax, "Test error at every refit, every person", "test error (nMSE)")
        ax.legend(fontsize=8, markerscale=2)
    fig.suptitle("Learned models' error per episode (lines: mean over people, band: standard error)",
                 x=0.01, y=0.995, ha="left", va="top", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.09 / len(families)))
    fig.savefig(out / "learning_curves.png")


def curves_row(frame, out):
    """Only the learning curves, one panel per model side by side, on a shared y-axis."""
    families = [f for f in CURVE_MODELS if f in set(frame.family)]
    fig, axes = plt.subplots(1, len(families), figsize=(4.6 * len(families), 4.2), sharey=True,
                             squeeze=False)
    for ax, family in zip(axes[0], families):
        f = frame[frame.family == family]
        m, se = per_episode(f, "train_nmse")
        band(ax, m.index, m, se, TRAIN, "training error")
        if "val_nmse" in f and f.val_nmse.notna().any():
            m, se = per_episode(f.dropna(subset=["val_nmse"]), "val_nmse")
            band(ax, m.index, m, se, VALIDATION, "validation error")
        m, se = per_episode(f.dropna(subset=["new_nmse"]), "new_nmse")
        band(ax, m.index, m, se, NEXT, "test error")
        ax.axhline(1, color=INK2, lw=1, ls=":")
        ax.set_yscale("log")
        ax.set_title(CURVE_MODELS[family][0].upper() + CURVE_MODELS[family][1:], loc="left")
        ax.set_xlabel("episode")
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("nMSE (1 = predicting the average)")
    fig.suptitle("Learning curves per model (lines: mean over people, band: standard error)",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "learning_curves_row.png")


def inner_curves(frame, out, episodes=(1, 10, 19)):
    f = frame[(frame.family == "net") & frame.role.eq("real")]
    if f.empty or "curve" not in f:
        return
    people = sorted(f.person.unique())
    fig, axes = plt.subplots(len(episodes), len(people), figsize=(2.9 * len(people), 2.4 * len(episodes)),
                             sharex=True, sharey=True, squeeze=False)
    for i, episode in enumerate(episodes):
        for j, person in enumerate(people):
            ax = axes[i][j]
            rows = f[(f.person == person) & (f.episode == episode)]
            if rows.empty:
                ax.text(0.5, 0.5, "no refit\n(episode failed)", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8.5, color=INK2)
            else:
                c = np.array([[np.nan if v is None else v for v in (list(h) + [None])[:4]]
                              for h in rows.iloc[-1].curve], dtype=float)
                ax.plot(c[:, 0], c[:, 1], color=TRAIN, label="training error")
                ax.plot(c[:, 0], c[:, 2], color=VALIDATION, label="validation error")
                best = rows.iloc[-1].best_step
                if best == best:
                    ax.axvline(best, color=INK2, lw=1, ls=":", label="weights kept")
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(f"person {person}", loc="left")
            if j == 0:
                ax.set_ylabel(f"episode {episode}\nMSE (standardized)")
            if i == len(episodes) - 1:
                ax.set_xlabel("training step")
    axes[0][0].legend(fontsize=7.5)
    fig.suptitle("MLP 256×256 + vel/torque: training inside one refit after a real episode", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out / "learning_curves_inner.png")


def curves(args):
    """Learning curves of the learned models, from `<run>_learning.jsonl`.

    Each line is one refit (after a solved dream or real episode): `new_nmse` is the test
    error: the error on the rows just added, before fitting them (data the model has not
    seen), `train_nmse` the fit's error on all its rows, `val_nmse` the new model's held-out
    error, and `curve` its (step, train, held-out) losses during that refit.
    """
    plt.rcParams.update(STYLE)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame = load_refits(args.logs)
    overview(frame, out)
    curves_row(frame, out)
    inner_curves(frame, out)
    summary = frame.groupby(["family", "role"])[["new_nmse", "train_nmse"]].median()
    print(f"{len(frame)} refits\n{summary.round(3)}")


# --- step -------------------------------------------------------------------------------

# tag: (label, colour, line style); only the tags with data are drawn.
LINES = {"plain_0": ("plain (no memory)", "#8C8C8C", "-"),
         "mlp_0_step": ("MLP 128×128, 0 dreams", "#eb6834", "--"),
         "net_0_step": ("MLP 256×256 + vel/torque, 0 dreams", "#2a78d6", "--"),
         "cache_4_step": ("cache, 4 dreams", "#1baf7a", "-"),
         "mlp_4_step": ("MLP 128×128, 4 dreams", "#eb6834", "-"),
         "net_4_step": ("MLP 256×256 + vel/torque, 4 dreams", "#2a78d6", "-"),
         "net_4_step_k8": ("MLP 256×256 + vel/torque, 8 candidates", "#86b6ef", "-"),
         "gen_flow_4_step_k8": ("flow matching, 8 candidates", "#4a3aa7", "-"),
         "gen_ddpm_4_step_k8": ("diffusion, 8 candidates", "#e87ba4", "-")}


def step(args):
    """Step-mode warm start: planning effort for plain, the cache and the learned models.

    MPC rollouts are compared (deterministic given the task, unlike planning time, which
    depends on machine load); planning time only between runs made together.
    """
    plt.rcParams.update(STYLE)
    out = args.out or Path("figures/dream") / args.data.name / "step_mode.png"
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(args.data / "*_p??.csv"))])
    d = d[d.tag.isin(LINES)]
    lines = {tag: LINES[tag] for tag in LINES if tag in set(d.tag)}
    d["accepted"] = d.seeded_steps / (d.seeded_steps + d.full_steps)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    ax = axes[0]
    people = sorted(d.person.unique())
    width = 0.8 / len(lines)
    for i, (tag, (label, color, style)) in enumerate(lines.items()):
        means = d[(d.tag == tag) & (d.episode > 0)].groupby("person").rollouts.mean()
        ax.bar(np.arange(len(people)) + (i - (len(lines) - 1) / 2) * width, means.loc[people],
               width * 0.92, color=color, label=label, alpha=0.55 if style == "--" else 1.0)
    ax.set_xticks(range(len(people)), [f"p{p}" for p in people])
    ax.set_ylabel("MPC rollouts per episode (mean, ep. 1–19)")
    ax.set_title("Planning effort per person", loc="left")
    ax.set_xlabel("person")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    for tag, (label, color, style) in lines.items():
        per = d[d.tag == tag].groupby("episode").rollouts
        m, se = per.mean(), per.sem()
        ax.plot(m.index, m, color=color, ls=style, marker="o", ms=3, label=label)
        ax.fill_between(m.index, m - se, m + se, color=color, alpha=0.12, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("episode")
    ax.set_ylabel("MPC rollouts (mean over 5 people)")
    ax.set_title("Planning effort per episode (band: SE)", loc="left")
    ax.legend(fontsize=7.5, loc="lower left")

    ax = axes[2]
    for tag, (label, color, style) in lines.items():
        if tag == "plain_0":
            continue
        per = d[d.tag == tag].groupby("episode").accepted
        m, se = 100 * per.mean(), 100 * per.sem()
        ax.plot(m.index, m, color=color, ls=style, marker="o", ms=3, label=label)
        ax.fill_between(m.index, m - se, m + se, color=color, alpha=0.12, lw=0)
    ax.set_ylim(0, 100)
    ax.set_xlabel("episode")
    ax.set_ylabel("MPC steps taking the model's seed (%)")
    ax.set_title("How often the MPC takes the seed", loc="left")
    ax.text(0.02, 0.97, "colours as in the middle panel", transform=ax.transAxes, va="top", fontsize=8, color=INK2)

    fig.suptitle(args.title,
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(out)


# --- samples ----------------------------------------------------------------------------

CUT, DRAWS, H, J = 14, 30, 12, 7
CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]
FLOW, DDPM, REG = "#4a3aa7", "#e87ba4", "#eb6834"


def samples(args):
    """What the generative models draw under varied geometry, for one person.

    Models are fit on the run's episodes before `CUT`; the figure uses the first control
    step of each motion. Top: torque plans projected on the two main directions of the
    recorded plans, coloured by arm configuration (k-means on the robot's starting joints):
    recorded plans, then 30 draws per held-out motion from flow matching, diffusion, and
    regression with Gaussian noise at its residual scale. Bottom: torques over the horizon
    for two held-out motions with different arm configurations.
    """
    plt.rcParams.update(STYLE)
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    sys.path.insert(0, str(Path(__file__).parent))
    from dream_eval import load_run

    from limb_baseline.warmstart import make_regressor

    x, y, meta, _, _ = load_run(args.data, f"net_4_step_p{args.person}")
    train = (meta.episode < CUT).to_numpy()
    first = x[:, 25] == 0
    x0, y0, held = x[first], y[first], ~train[first]

    robot = x0[:, 6:13]
    k = max((2, 3, 4), key=lambda n: silhouette_score(
        robot, KMeans(n, n_init=5, random_state=0).fit_predict(robot)))
    config = KMeans(k, n_init=5, random_state=0).fit_predict(robot)
    mean, std = y0.mean(0), y0.std(0) + 1e-9
    pca = PCA(2).fit((y0 - mean) / std)
    project = lambda plans: pca.transform((plans.reshape(-1, y.shape[1]) - mean) / std)

    models = {name: make_regressor(spec).fit(x[train], y[train]) for name, spec in (
        ("flow matching", "gen:flow:hist"), ("diffusion", "gen:ddpm:hist"),
        ("regression", "net:256x256:d0.3:hist"))}
    reg = models["regression"]
    scale = (y[train] - reg.predict(x[train])).std(0)
    rng = np.random.default_rng(0)
    draws = {
        "flow matching": models["flow matching"].sample(x0[held], DRAWS, seed=0),
        "diffusion": models["diffusion"].sample(x0[held], DRAWS, seed=0),
        "MLP 256×256 + vel/torque, noisy copies": reg.predict(x0[held])[:, None]
        + rng.normal(size=(held.sum(), DRAWS, y.shape[1])) * scale,
    }

    fig = plt.figure(figsize=(16, 8.6))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.15, 1])
    recorded = project(y0)
    limits = [np.percentile(recorded[:, i], [1, 99]) for i in (0, 1)]
    pad = [0.35 * (hi - lo) for lo, hi in limits]

    ax = fig.add_subplot(grid[0, 0])
    for c in range(k):
        m = config == c
        ax.scatter(recorded[m, 0], recorded[m, 1], s=22, color=CLUSTER_COLORS[c], alpha=0.85, lw=0,
                   label=f"arm configuration {c + 1} ({m.sum()} motions)")
    ax.set_title(f"Recorded plans, all {len(y0)} motions", loc="left")
    ax.set_ylabel("plan direction 2")
    ax.legend(fontsize=7.5, loc="best")
    axes = [ax]
    for i, (name, samples) in enumerate(draws.items(), start=1):
        ax = fig.add_subplot(grid[0, i], sharex=axes[0], sharey=axes[0])
        z = project(samples).reshape(held.sum(), DRAWS, 2)
        for c in range(k):
            m = config[held] == c
            ax.scatter(z[m, :, 0], z[m, :, 1], s=5, color=CLUSTER_COLORS[c], alpha=0.22, lw=0)
            ax.scatter(recorded[held][m, 0], recorded[held][m, 1], s=30, color=CLUSTER_COLORS[c],
                       edgecolor=INK, lw=0.9, zorder=3)
        ax.set_title(f"{name[0].upper() + name[1:]}: {DRAWS} draws", loc="left", fontsize=9.5)
        axes.append(ax)
    axes[1].scatter([], [], s=30, color="white", edgecolor=INK, lw=0.9, label="recorded plan")
    axes[1].scatter([], [], s=8, color=INK2, alpha=0.5, lw=0, label="model draws")
    axes[1].legend(fontsize=7.5, loc="best")
    for ax in axes:
        ax.set_xlim(limits[0][0] - pad[0], limits[0][1] + pad[0])
        ax.set_ylim(limits[1][0] - pad[1], limits[1][1] + pad[1])
        ax.set_xlabel("plan direction 1")
    axes[0].set_ylabel("plan direction 2")

    sizes = np.bincount(config[held], minlength=k)
    chosen = [int(np.where(held & (config == c))[0][0]) for c in np.argsort(sizes)[::-1][:2]]
    steps = np.arange(H)
    for col, index in enumerate(chosen):
        plan = y0[index].reshape(H, J)
        position = int(np.where(np.where(held)[0] == index)[0][0])
        joint = int(np.argmax(np.abs(plan).mean(0)))
        for half, (name, color) in enumerate((("flow matching", FLOW), ("diffusion", DDPM))):
            ax = fig.add_subplot(grid[1, 2 * col + half])
            for s in draws[name][position].reshape(DRAWS, H, J):
                ax.plot(steps, s[:, joint], color=color, lw=0.8, alpha=0.4)
            ax.plot([], [], color=color, lw=1.2, label=f"{name} draws")
            ax.plot(steps, reg.predict(x0[index:index + 1]).reshape(H, J)[:, joint], color=REG, lw=2,
                    ls="--", label="MLP 256×256 + vel/torque")
            ax.plot(steps, plan[:, joint], color=INK, lw=2.4, label="recorded plan")
            ax.set_title(f"Arm configuration {config[index] + 1}, joint {joint}: {name}", loc="left",
                         color=CLUSTER_COLORS[config[index]])
            ax.set_xlabel("control step")
            if col == 0 and half == 0:
                ax.set_ylabel("torque")
            ax.legend(fontsize=7.5)

    fig.suptitle(f"Torque plans under varied geometry, person {args.person}: recorded vs generated "
                 f"(first step of each motion; models fit on episodes 0–{CUT - 1})",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path("figures/dream") / Path(args.data).name / "generative_samples.png"
    fig.savefig(out)
    print(out, "| configurations:", k, "| held-out motions per configuration:", sizes.tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def add(function):
        sub = commands.add_parser(function.__name__, description=function.__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
        sub.set_defaults(function=function)
        return sub

    sub = add(summary)
    sub.add_argument("--csv", type=Path, nargs="+", action="extend", required=True)
    sub.add_argument("--out", type=Path, required=True)
    sub.add_argument("--effort", choices=sorted(EFFORT_LABELS), default="rollouts")
    sub.add_argument("--per-person", action="store_true",
                     help="Also write <out stem>_person<id>.png per person.")
    sub.add_argument("--title", type=str, default="")
    sub.add_argument("--max-episodes", type=int, default=None,
                     help="Draw only the first this many episodes of every run.")
    sub = add(curves)
    sub.add_argument("--logs", nargs="+", required=True)
    sub.add_argument("--out", default="figures/dream/warmstart_compare")
    sub = add(step)
    sub.add_argument("--data", type=Path, default=Path("data/dream/warmstart_compare"))
    sub.add_argument("--out", type=Path, default=None, help="Defaults to figures/dream/<data>/step_mode.png.")
    sub.add_argument("--title", default="Seeding every MPC step (step mode), 5 people × 20 episodes")
    sub = add(samples)
    sub.add_argument("--person", type=int, default=24)
    sub.add_argument("--data", default="data/dream/generative_models")
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
