"""Step-mode warm start: planning effort for plain, the deployed MLP and the new model.

MPC rollouts are compared (deterministic given the task, unlike planning time, which
depends on machine load); planning time only between runs made together.

    python scripts/make_step_compare_figure.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("data/dream/warmstart_compare")
OUT = Path("figures/dream/warmstart_compare/step_mode.png")
LINES = {"plain_0": ("plain (no model)", "#8C8C8C"),
         "mlp_0_step": ("deployed MLP, step seed", "#eb6834"),
         "net_0_step": ("new MLP + history, step seed", "#2a78d6")}
INK, INK2, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
    "lines.linewidth": 2, "figure.dpi": 150,
})


def main():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(DATA / "*_p*.csv")) if "dreams" not in f])
    d = d[d.tag.isin(LINES)]
    d["accepted"] = d.seeded_steps / (d.seeded_steps + d.full_steps)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    ax = axes[0]
    people = sorted(d.person.unique())
    width = 0.26
    for i, (tag, (label, color)) in enumerate(LINES.items()):
        means = d[(d.tag == tag) & (d.episode > 0)].groupby("person").rollouts.mean()
        ax.bar(np.arange(len(people)) + (i - 1) * width, means.loc[people], width * 0.92,
               color=color, label=label)
    ax.set_xticks(range(len(people)), [f"p{p}" for p in people])
    ax.set_ylabel("MPC rollouts per episode (mean, ep. 1–19)")
    ax.set_title("Planning effort per person", loc="left")
    ax.set_xlabel("person")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    for tag, (label, color) in LINES.items():
        per = d[d.tag == tag].groupby("episode").rollouts
        m, se = per.mean(), per.sem()
        ax.plot(m.index, m, color=color, marker="o", ms=3, label=label)
        ax.fill_between(m.index, m - se, m + se, color=color, alpha=0.12, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("episode")
    ax.set_ylabel("MPC rollouts (mean over 5 people)")
    ax.set_title("Planning effort per episode (band: SE)", loc="left")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[2]
    for tag, (label, color) in list(LINES.items())[1:]:
        per = d[d.tag == tag].groupby("episode").accepted
        m, se = 100 * per.mean(), 100 * per.sem()
        ax.plot(m.index, m, color=color, marker="o", ms=3, label=label)
        ax.fill_between(m.index, m - se, m + se, color=color, alpha=0.12, lw=0)
    ax.set_ylim(0, 100)
    ax.set_xlabel("episode")
    ax.set_ylabel("MPC steps taking the model's seed (%)")
    ax.set_title("How often the MPC uses the model", loc="left")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Zero-dream learned models seeding every MPC step, 5 people × 20 episodes",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
