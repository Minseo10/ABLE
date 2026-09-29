"""Learning curves of the dream pipeline's learned models, from `<run>_learning.jsonl`.

Each line is one refit (after a solved dream or real episode): `new_nmse` is the error on
the rows just added, before fitting them (the model's error on data it has not seen),
`train_nmse` the fit's error on all its rows, `val_nmse` the new model's held-out error,
and `curve` its (step, train, held-out) losses during that refit.

    python scripts/make_learning_curves.py --logs data/dream/warmstart_compare/*_4_step_p*_learning.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = {"mlp": ("deployed MLP 128×128", "#eb6834"), "net": ("new MLP + history", "#2a78d6")}
INK, INK2, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
    "lines.linewidth": 2, "figure.dpi": 150,
})


def load(paths) -> pd.DataFrame:
    rows = []
    for path in paths:
        match = re.match(r"(\w+?)_(\d+)(_step)?_p(\d+)_learning", Path(path).stem)
        family, person = match.group(1), int(match.group(4))
        for line in open(path, encoding="utf-8"):
            rows.append({**json.loads(line), "family": family, "person": person})
    return pd.DataFrame(rows)


def per_episode(frame, column, role=None):
    """Mean over people of each person's mean per episode, and its standard error."""
    f = frame if role is None else frame[frame.role.eq("real") == (role == "real")]
    per_person = f.groupby(["person", "episode"])[column].mean().unstack(0)
    return per_person.mean(1), per_person.sem(1)


def band(ax, x, mean, se, color, style, label):
    ax.plot(x, mean, color=color, ls=style, marker="o", ms=3, label=label)
    ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.12, lw=0)


def overview(frame, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    ax = axes[0]
    for family, (name, color) in MODELS.items():
        f = frame[frame.family == family]
        if f.empty:
            continue
        m, se = per_episode(f.dropna(subset=["new_nmse"]), "new_nmse")
        band(ax, m.index, m, se, color, "-", f"{name}: unseen rows")
        m, se = per_episode(f, "train_nmse")
        band(ax, m.index, m, se, color, "--", f"{name}: train")
        if "val_nmse" in f and f.val_nmse.notna().any():
            m, se = per_episode(f.dropna(subset=["val_nmse"]), "val_nmse")
            band(ax, m.index, m, se, color, ":", f"{name}: held-out")
    ax.axhline(1, color=INK2, lw=1, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("episode")
    ax.set_ylabel("nMSE (1 = predicting the mean)")
    ax.set_title("Error per episode (mean over people, band: SE)", loc="left")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    for family, (name, color) in MODELS.items():
        f = frame[(frame.family == family)].dropna(subset=["new_nmse"])
        for role, style in (("real", "-"), ("dream", "--")):
            if f.empty:
                continue
            m, se = per_episode(f, "new_nmse", role)
            band(ax, m.index, m, se, color, style, f"{name}: {role} rows")
    ax.axhline(1, color=INK2, lw=1, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("episode")
    ax.set_ylabel("nMSE on rows before fitting them")
    ax.set_title("Unseen-row error: real episodes vs dreams", loc="left")
    ax.legend(fontsize=7.5)

    ax = axes[2]
    for family, (name, color) in MODELS.items():
        f = frame[frame.family == family].dropna(subset=["new_nmse"])
        ax.scatter(f.rows, f.new_nmse, s=8, color=color, alpha=0.35, lw=0, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1, color=INK2, lw=1, ls=":")
    ax.set_xlabel("training rows at that refit")
    ax.set_ylabel("nMSE on the rows added next")
    ax.set_title("Every refit, every person", loc="left")
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png")


def inner_curves(frame, out, episodes=(1, 5, 10, 19)):
    f = frame[(frame.family == "net") & frame.role.eq("real")]
    if f.empty or "curve" not in f:
        return
    people = sorted(f.person.unique())
    fig, axes = plt.subplots(1, len(people), figsize=(3.2 * len(people), 3.2), sharey=True, squeeze=False)
    shades = plt.cm.Blues(np.linspace(0.45, 1.0, len(episodes)))
    for ax, person in zip(axes[0], people):
        for shade, episode in zip(shades, episodes):
            rows = f[(f.person == person) & (f.episode == episode)]
            if rows.empty:
                continue
            curve = np.array([(s, t, np.nan if v is None else v) for s, t, v in rows.iloc[-1].curve])
            ax.plot(curve[:, 0], curve[:, 1], color=shade, ls="--", lw=1.4)
            ax.plot(curve[:, 0], curve[:, 2], color=shade, lw=1.6, label=f"episode {episode}")
        ax.set_title(f"person {person}", loc="left")
        ax.set_xlabel("optimizer step")
        ax.set_yscale("log")
    axes[0][0].set_ylabel("MSE (standardized)")
    axes[0][0].legend(fontsize=7.5, title="solid: held-out\ndashed: train", title_fontsize=7.5)
    fig.suptitle("New model, inside one refit after a real episode", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out / "learning_curves_inner.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--out", default="figures/dream/warmstart_compare")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame = load(args.logs)
    overview(frame, out)
    inner_curves(frame, out)
    summary = frame.groupby(["family", "role"])[["new_nmse", "train_nmse"]].median()
    print(f"{len(frame)} refits\n{summary.round(3)}")


if __name__ == "__main__":
    main()
