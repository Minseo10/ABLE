"""Figures for the PuckPushing complexity sweep and the planner profile.

Reads data/friction/complexity/*.json (one per episode) and profile.json, and writes
figures/friction/complexity/*.png plus a CSV summary.

    python scripts/friction/figures.py
"""

from __future__ import annotations

import collections
import csv
import glob
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DATA = Path("data/friction/complexity")
OUT = Path("figures/friction/complexity")
LEVELS = ["L1", "L2", "L3", "L4"]
LEVEL_LABELS = {
    "L1": "L1\n0 holes, r=0.40",
    "L2": "L2\n2 holes, r=0.30",
    "L3": "L3\n4 holes, r=0.25",
    "L4": "L4\n4 holes, r=0.20",
}
BUDGETS = [300, 1000, 3000]
# Ordinal blue ramp (light to dark) for the budgets.
BUDGET_COLORS = {300: "#86b6ef", 1000: "#2a78d6", 3000: "#104281"}
# Categorical slots in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
OUTCOMES = ["goal", "hole", "stuck", "timeout"]
SURFACE = "#fcfcfb"
INK = "#2b2b29"
MUTED = "#6b6a63"


def style(ax) -> None:
    """Recessive axes and grid."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def load_runs() -> list[dict]:
    """All episode logs."""
    runs = []
    for path in glob.glob(str(DATA / "*_seed*.json")):
        with open(path, encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


def group(runs: list[dict]) -> dict:
    """Runs by (env, level, sims)."""
    groups: dict = collections.defaultdict(list)
    for run in runs:
        groups[(run["env"], run["level"], run["sims"])].append(run)
    return groups


def mean_sem(values) -> tuple[float, float]:
    """Mean and standard error."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return float("nan"), 0.0
    sem = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return float(arr.mean()), float(sem)


def per_push_plan(run: dict) -> float:
    """Mean planning seconds per push in an episode."""
    return run["total_plan_seconds"] / max(len(run["steps"]), 1)


def write_summary(groups: dict) -> None:
    """CSV with one row per (env, level, budget)."""
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["env", "level", "sims", "episodes", "success_rate"]
            + OUTCOMES
            + [
                "pushes_success_mean",
                "plan_s_per_push_mean",
                "plan_s_per_episode_mean",
                "return_mean",
            ]
        )
        for (env, level, sims), runs in sorted(groups.items()):
            counts = collections.Counter(r["outcome"] for r in runs)
            wins = [len(r["steps"]) for r in runs if r["outcome"] == "goal"]
            writer.writerow(
                [env, level, sims, len(runs), counts["goal"] / len(runs)]
                + [counts[o] for o in OUTCOMES]
                + [
                    f"{np.mean(wins):.2f}" if wins else "",
                    f"{np.mean([per_push_plan(r) for r in runs]):.2f}",
                    f"{np.mean([r['total_plan_seconds'] for r in runs]):.1f}",
                    f"{np.mean([r['total_reward'] for r in runs]):.2f}",
                ]
            )


def fig_sr_and_time(groups: dict) -> None:
    """Success rate, planning time per push and per episode, vs level."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    x = np.arange(len(LEVELS))
    for row, env in enumerate(["2d", "3d"]):
        for sims in BUDGETS:
            color = BUDGET_COLORS[sims]
            sr, t_push, t_push_e, t_ep, t_ep_e = [], [], [], [], []
            for level in LEVELS:
                runs = groups.get((env, level, sims), [])
                sr.append(np.mean([r["outcome"] == "goal" for r in runs]) if runs else np.nan)
                m, e = mean_sem([per_push_plan(r) for r in runs])
                t_push.append(m)
                t_push_e.append(e)
                m, e = mean_sem([r["total_plan_seconds"] for r in runs])
                t_ep.append(m)
                t_ep_e.append(e)
            label = f"{sims} sims"
            axes[row, 0].plot(x, sr, "-o", color=color, lw=2, ms=8, label=label)
            axes[row, 1].errorbar(
                x, t_push, yerr=t_push_e, fmt="-o", color=color, lw=2, ms=8, capsize=3, label=label
            )
            axes[row, 2].errorbar(
                x, t_ep, yerr=t_ep_e, fmt="-o", color=color, lw=2, ms=8, capsize=3, label=label
            )
        axes[row, 0].set_ylim(-0.03, 1.03)
        axes[row, 0].set_ylabel(f"{env.upper()} env\nsuccess rate", color=INK)
        axes[row, 1].set_ylabel("planning time per push (s)", color=INK)
        axes[row, 2].set_ylabel("planning time per episode (s)", color=INK)
        axes[row, 1].set_yscale("log")
        axes[row, 2].set_yscale("log")
    for ax in axes.flat:
        style(ax)
        ax.set_xticks(x, [LEVEL_LABELS[lv] for lv in LEVELS])
    axes[0, 0].legend(frameon=False, fontsize=8, title="POMCPOW budget", title_fontsize=8)
    fig.suptitle(
        "PuckPushing: plain POMCPOW vs. problem complexity (10 seeds per point, mean ± s.e.)",
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(OUT / "sr_and_planning_time.png", dpi=130, facecolor="white")
    plt.close(fig)


def fig_sr_vs_time(groups: dict) -> None:
    """Success rate against mean planning time per episode; one line per level."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, env in zip(axes, ["2d", "3d"]):
        for k, level in enumerate(LEVELS):
            pts = []
            for sims in BUDGETS:
                runs = groups.get((env, level, sims), [])
                if runs:
                    pts.append(
                        (
                            np.mean([r["total_plan_seconds"] for r in runs]),
                            np.mean([r["outcome"] == "goal" for r in runs]),
                        )
                    )
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-o", color=SERIES[k], lw=2, ms=8, label=level)
            ax.annotate(level, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        color=INK, fontsize=8, va="center")
        ax.set_xscale("log")
        ax.set_xlabel("planning time per episode (s); points = 300 / 1000 / 3000 sims", color=INK)
        ax.set_title(f"{env.upper()} env", color=INK, fontsize=10)
        style(ax)
    axes[0].set_ylabel("success rate", color=INK)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].legend(frameon=False, fontsize=8, title="complexity", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "sr_vs_planning_time.png", dpi=130, facecolor="white")
    plt.close(fig)


def fig_outcomes(groups: dict) -> None:
    """Outcome counts and pushes per success, per level and budget."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
    for row, env in enumerate(["2d", "3d"]):
        ax = axes[row, 0]
        labels, bottoms = [], np.zeros(len(LEVELS) * len(BUDGETS))
        for level in LEVELS:
            for sims in BUDGETS:
                labels.append(f"{level}\n{sims}")
        for k, outcome in enumerate(OUTCOMES):
            vals = []
            for level in LEVELS:
                for sims in BUDGETS:
                    runs = groups.get((env, level, sims), [])
                    vals.append(sum(r["outcome"] == outcome for r in runs))
            vals = np.array(vals, dtype=float)
            ax.bar(np.arange(len(vals)), vals, bottom=bottoms, color=SERIES[k],
                   edgecolor=SURFACE, linewidth=2, label=outcome, width=0.8)
            bottoms += vals
        ax.set_xticks(np.arange(len(labels)), labels, fontsize=7)
        ax.set_ylabel(f"{env.upper()} env\nepisodes", color=INK)
        style(ax)
        ax = axes[row, 1]
        x = np.arange(len(LEVELS))
        for sims in BUDGETS:
            means, errs = [], []
            for level in LEVELS:
                m, e = mean_sem(
                    [len(r["steps"]) for r in groups.get((env, level, sims), [])
                     if r["outcome"] == "goal"]
                )
                means.append(m)
                errs.append(e)
            ax.errorbar(x, means, yerr=errs, fmt="-o", color=BUDGET_COLORS[sims], lw=2,
                        ms=8, capsize=3, label=f"{sims} sims")
        ax.axhline(3, color=MUTED, ls="--", lw=1)
        ax.text(len(LEVELS) - 1, 3.2, "greedy, true friction: 3", color=MUTED, fontsize=8, ha="right")
        ax.set_xticks(x, [LEVEL_LABELS[lv] for lv in LEVELS])
        ax.set_ylabel("pushes per successful episode", color=INK)
        style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=4, loc="lower center",
                      bbox_to_anchor=(0.5, 1.0))
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "outcomes_and_pushes.png", dpi=130, facecolor="white")
    plt.close(fig)


def fig_profile() -> None:
    """Where planning time goes, and how deep the search tree gets."""
    path = DATA / "profile.json"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        prof = json.load(f)
    levels = [lv for lv in LEVELS if lv in prof]
    parts = [
        "leaf value (greedy rollout)",
        "transition (slide + drive)",
        "propose (sample feasible push)",
        "observe + likelihood",
        "tree bookkeeping (POMCPOW itself)",
        "belief update",
        "env step (simulate push)",
    ]
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5.4), gridspec_kw={"width_ratios": [1, 1.3, 1]}
    )
    ax = axes[0]
    x = np.arange(len(levels))
    bottoms = np.zeros(len(levels))
    for k, part in enumerate(parts):
        vals = np.array([prof[lv]["parts"][part] for lv in levels])
        ax.bar(x, vals, bottom=bottoms, color=SERIES[k], edgecolor=SURFACE, linewidth=2,
               width=0.6, label=part)
        bottoms += vals
    ax.set_xticks(x, levels)
    ax.set_ylabel("seconds in one episode (cProfile)", color=INK)
    ax.set_title("Time by planner part (1000 sims/push, seed 0)", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.08))
    style(ax)

    ax = axes[1]
    inner = [
        ("rollout: speed_to_travel", "rollout: speed search (14 slides per target)"),
        ("PuckWorld.push_pose", "push feasibility (push_pose)"),
        ("rollout: puck_path_clear", "rollout: line-of-sight checks"),
        ("PuckWorld.drive_distance", "robot drive distance"),
        ("rollout: path_to_goal", "rollout: path lookup"),
    ]
    level = "L2" if "L2" in prof else levels[0]
    plan = prof[level]["plan_seconds"]
    vals = [prof[level]["inner"][key] / plan * 100 for key, _ in inner]
    y = np.arange(len(inner))[::-1]
    ax.barh(y, vals, color=SERIES[0], height=0.6)
    for yy, v in zip(y, vals):
        ax.text(v + 1, yy, f"{v:.0f}%", va="center", color=INK, fontsize=8)
    ax.set_yticks(y, [label for _, label in inner], fontsize=8)
    ax.set_xlabel(f"% of planning time ({level}; overlapping)", color=INK)
    ax.set_title(
        f"Inside the planner: {prof[level]['calls']['slide'] / 1e3:.0f}k slides per episode",
        color=INK, fontsize=10,
    )
    style(ax)
    ax.grid(axis="x", color="#e6e5e0")

    ax = axes[2]
    for k, lv in enumerate(levels):
        trees = prof[lv]["trees"]
        depth = max(len(t["decisions_by_depth"]) for t in trees)
        mat = np.zeros((len(trees), depth))
        for i, t in enumerate(trees):
            mat[i, : len(t["decisions_by_depth"])] = t["decisions_by_depth"]
        frac = mat.mean(0) / mat[:, 0].mean()
        keep = frac > 0
        ax.plot(np.arange(1, depth + 1)[keep], frac[keep], "-o", color=SERIES[k], lw=2,
                ms=8, label=lv)
    ax.set_yscale("log")
    ax.set_xlabel("tree depth (pushes ahead)", color=INK)
    ax.set_ylabel("fraction of simulations deciding there", color=INK)
    ax.set_title(
        f"Search depth (root has ~"
        f"{np.mean([t['root_actions'] for t in prof[level]['trees']]):.0f} actions)",
        color=INK, fontsize=10,
    )
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "planner_profile.png", dpi=130, facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    groups = group(load_runs())
    write_summary(groups)
    fig_sr_and_time(groups)
    fig_sr_vs_time(groups)
    fig_outcomes(groups)
    fig_profile()
    print(f"{sum(len(v) for v in groups.values())} episodes -> {OUT}")


if __name__ == "__main__":
    main()
