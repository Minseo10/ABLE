"""Figures for scripts/eval_warmstart_models.py.

    python scripts/make_warmstart_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("data/dream/warmstart_models/results.json")
FIG = Path("figures/dream/warmstart_models")
COLORS = {
    "deployed MLP 128×128": "#eb6834",
    "deployed ridge": "#eda100",
    "new MLP, old inputs": "#1baf7a",
    "new MLP + history": "#2a78d6",
}
INK, INK2, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
    "lines.linewidth": 2, "figure.dpi": 150,
})


def heldout(rows, cap=1.5):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    rng = np.random.default_rng(0)
    summary = {}
    for i, (model, color) in enumerate(COLORS.items()):
        mine = [r for r in rows if r["model"] == model and r["protocol"] == "heldout"]
        test = np.array([r["test"] for r in mine])
        train = np.array([r["train"] for r in mine])
        jitter = rng.uniform(-0.12, 0.12, len(test))
        ax.scatter(np.minimum(test, cap), i + jitter, s=14, color=color, alpha=0.55, lw=0)
        ax.plot([np.median(test)] * 2, [i - 0.28, i + 0.28], color=color, lw=3)
        ax.scatter(np.median(train), i, marker="|", s=250, color=INK, lw=1.6, zorder=3)
        ax.text(cap + 0.04, i, f"{np.median(test):.2f}", va="center", fontsize=9, color=INK)
        summary[model] = {"test_median": float(np.median(test)), "train_median": float(np.median(train)),
                          "test_iqr": [float(v) for v in np.percentile(test, [25, 75])],
                          "runs_above_1": int((test > 1).sum()), "runs": len(test)}
    ax.set_yticks(range(len(COLORS)), list(COLORS))
    ax.invert_yaxis()
    ax.set_xlim(0, cap + 0.2)
    ax.set_xlabel(f"nMSE on each run's last 20% of trajectories (dots: runs, clipped at {cap})")
    ax.scatter([], [], marker="|", s=120, color=INK, label="train (median)")
    ax.plot([], [], color=INK2, lw=3, label="held-out (median)")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.suptitle("Held-out error per model, 20 four-dream runs", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "1_heldout.png")
    return summary


def online(rows):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    summary = {}
    for model, color in COLORS.items():
        mine = [r for r in rows if r["model"] == model and r["protocol"] == "online"]
        seen = sorted({r["seen"] for r in mine})
        seen = [n for n in seen if sum(r["seen"] == n for r in mine) >= 10]
        per = [np.array([r["test"] for r in mine if r["seen"] == n]) for n in seen]
        med = [np.median(p) for p in per]
        lo, hi = [np.percentile(p, 25) for p in per], [np.percentile(p, 75) for p in per]
        ax.fill_between(seen, lo, hi, color=color, alpha=0.12, lw=0)
        ax.plot(seen, med, color=color, marker="o", ms=4, label=model)
        summary[model] = dict(zip(map(str, seen), map(float, med)))
        above = [(n, m) for n, m in zip(seen, med) if m > 1.6]
        if above:
            n, m = max(above, key=lambda a: a[1])
            ax.annotate(f"up to {m:.0f}", (above[-1][0], 1.6), xytext=(0, -12), textcoords="offset points",
                        ha="center", fontsize=8, color=color)
    ax.axhline(1.0, color=INK2, lw=1, ls=":")
    ax.text(seen[-1], 1.02, "predicting the mean", ha="right", va="bottom", fontsize=8, color=INK2)
    ax.set_ylim(0, 1.6)
    ax.set_xlabel("trajectories seen by the learner")
    ax.set_ylabel("nMSE on the next 5 trajectories")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title("Online: error on upcoming trajectories as data accrues (median, IQR)",
                 loc="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "2_online.png")
    return summary


def fit_time(rows):
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    summary = {}
    for model, color in COLORS.items():
        mine = [r for r in rows if r["model"] == model]
        ax.scatter([r["rows"] for r in mine], [r["fit_seconds"] for r in mine], s=10,
                   color=color, alpha=0.6, lw=0, label=model)
        summary[model] = float(np.median([r["fit_seconds"] for r in mine]))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training rows")
    ax.set_ylabel("refit time, one CPU core (s)")
    ax.legend(fontsize=8, markerscale=1.8, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("Refit cost per episode", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "3_fit_time.png")
    return summary


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    rows = json.load(open(RESULTS))
    summary = {"heldout": heldout(rows), "online": online(rows), "fit_seconds": fit_time(rows)}
    json.dump(summary, open(RESULTS.with_name("summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
