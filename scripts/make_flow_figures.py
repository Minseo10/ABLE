"""Figures for scripts/eval_flow_policy.py: learning curves, data scaling, best-of-K, samples.

    python scripts/make_flow_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent))
from eval_flow_policy import FIG, H, J, KS, OUT, Regression, best_of_k, data  # noqa: E402

FLOW, REG, CTRL = "#2a78d6", "#eb6834", "#6b6a64"
INK, INK2, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
SPLITS = {"train": "--", "val": "-", "test": ":"}
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
    "lines.linewidth": 1.8, "figure.dpi": 150,
})


def curves(ax, history, color, best_epoch, title, ylabel):
    epochs = [h["epoch"] for h in history]
    for split, style in SPLITS.items():
        ax.plot(epochs, [h[split] for h in history], color=color, ls=style, label=split)
    ax.axvline(best_epoch, color=GRID, lw=1.2, zorder=0)
    ax.set_title(title, loc="left")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)


def suptitle(fig, text):
    fig.suptitle(text, x=0.01, ha="left", fontweight="bold", color=INK)


def learning_curves(search):
    chosen = min(search["flow"], key=lambda r: r["val"]["energy"])
    cfg = chosen["config"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    curves(axes[0], chosen["history"], FLOW, chosen["best_epoch"],
           f"Flow {cfg['width']}×{cfg['depth']}, dropout {cfg['dropout']}",
           "flow-matching loss")
    reg = search["regression"]
    curves(axes[1], reg["history"], REG, reg["best_epoch"], "Regression MLP 256×2, dropout 0.3",
           "MSE (standardized)")
    suptitle(fig, "Learning curves, all training data (grey line: epoch kept)")
    fig.tight_layout()
    fig.savefig(FIG / "1_learning_curves.png")


def scaling_curves(scaling):
    fracs = sorted({r["fraction"] for r in scaling})
    fig, axes = plt.subplots(2, len(fracs), figsize=(3 * len(fracs), 5.4), sharey="row",
                             squeeze=False)
    for j, frac in enumerate(fracs):
        row = next(r for r in scaling if r["fraction"] == frac and r["seed"] == 0)
        for i, (name, color, ylabel) in enumerate(
                [("flow", FLOW, "flow-matching loss"), ("regression", REG, "MSE (standardized)")]):
            ax = axes[i, j]
            curves(ax, row[name]["history"], color, row[name]["best_epoch"],
                   f"{name}, {row['segments']} traj.", ylabel if j == 0 else "")
            if j:
                ax.get_legend().remove()
    suptitle(fig, "Learning curves at each training-set size (seed 0)")
    fig.tight_layout()
    fig.savefig(FIG / "2_learning_curves_by_size.png")


def scaling_plot(scaling):
    segs = sorted({r["segments"] for r in scaling})
    agg = lambda key: [np.median([key(r) for r in scaling if r["segments"] == s]) for s in segs]
    panels = [
        ("One plan vs recorded plan (nMSE)", [
            ("flow, deterministic", FLOW, "-", lambda r: r["flow"]["test"]["nmse_det"]),
            ("regression", REG, "-", lambda r: r["regression"]["test"]["nmse_det"]),
            ("flow, train", FLOW, "--", lambda r: r["flow"]["train_nmse_det"]),
            ("regression, train", REG, "--", lambda r: r["regression"]["train_nmse_det"])]),
        ("Energy score (lower is better)", [
            ("flow, 16 samples", FLOW, "-", lambda r: r["flow"]["test"]["energy"]),
            ("regression", REG, "-", lambda r: r["regression"]["test"]["energy"])]),
        ("Best of 8 flow samples (nMSE)", [
            ("flow, best of 8", FLOW, "-", lambda r: r["flow"]["test"]["best_of_k"]["8"]),
            ("regression", REG, "-", lambda r: r["regression"]["test"]["nmse_det"])]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, (title, series) in zip(axes, panels):
        for label, color, style, key in series:
            ys = agg(key)
            ax.plot(segs, ys, color=color, ls=style, marker="o", ms=4, label=label)
            if style == "-":
                offset = 7 if color == REG else -7
                ax.annotate(f"{ys[-1]:.2f}", (segs[-1], ys[-1]), xytext=(5, offset),
                            textcoords="offset points", va="center", fontsize=8, color=INK2)
        ax.set_xscale("log")
        ax.set_xlim(min(segs) * 0.8, max(segs) * 1.6)
        ax.set_ylim(bottom=0)
        ax.set_title(title, loc="left")
        ax.set_xlabel("training trajectories (log)")
        ax.legend(fontsize=7.5)
    suptitle(fig, "Test metrics vs training-set size (median of 2 seeds; dashed = train)")
    fig.tight_layout()
    fig.savefig(FIG / "3_data_scaling.png")


def best_of_k_plot(models, sets, var):
    (xtr, ytr), (xte, yte) = sets["train"], sets["test"]
    flow, reg = models["flow"], models["regression"]
    samples = flow.sample(xte, k=max(KS), seed=0)
    flow_k = best_of_k(samples, yte, var)
    std = np.sqrt(var)
    control = []
    for k in KS:
        centers = KMeans(k, n_init=3, random_state=0).fit(ytr / std).cluster_centers_ * std
        control.append(float((((yte[:, None] - centers[None]) ** 2) / var).mean(-1).min(1).mean()))
    reg_err = float((((reg.predict(xte) - yte) ** 2) / var).mean())
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.plot(KS, control, color=CTRL, ls=":", marker="o", ms=4, label="input-free K-means, best of K")
    ax.plot(KS, [flow_k[k] for k in KS], color=FLOW, marker="o", ms=4, label="flow, best of K samples")
    ax.axhline(reg_err, color=REG, label="regression (one prediction)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(KS, [str(k) for k in KS])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("number of samples K")
    ax.set_ylabel("test nMSE")
    ax.legend(fontsize=8)
    ax.set_title("Does sampling reach the recorded plan?", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "4_best_of_k.png")
    return flow_k, control, reg_err


def ambiguous_groups(x, y):
    keys = np.round(x, 5)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    groups = [np.where(inverse == g)[0] for g in np.where(counts >= 3)[0]]
    return [g for g in groups if (((y[g] - y[g].mean(0)) ** 2).mean() > 0)]


def multimodality(models, x, y, var):
    """Per group of rows with identical inputs: how close samples come to each recorded plan
    (coverage) and how close each sample is to some recorded plan (precision)."""
    flow, reg = models["flow"], models["regression"]
    groups = ambiguous_groups(x, y)
    err = lambda a, b: (((a[:, None] - b[None]) ** 2) / var).mean(-1)
    stats = {"flow": [], "regression": [], "group mean": []}
    for g in groups:
        recorded = y[g]
        samples = flow.sample(x[g[:1]], k=16, seed=0)[0]
        pred = reg.predict(x[g[:1]])
        mean = recorded.mean(0, keepdims=True)
        for name, cand in (("flow", samples), ("regression", pred), ("group mean", mean)):
            e = err(cand, recorded)
            stats[name].append((e.min(0).mean(), e.min(1).mean()))
    table = {k: dict(zip(("coverage", "precision"), np.mean(v, 0).round(3))) for k, v in stats.items()}
    return groups, table


def samples_plot(models, x, y, groups):
    flow, reg = models["flow"], models["regression"]
    g = max(groups, key=lambda g: ((y[g] - y[g].mean(0)) ** 2).sum())
    recorded = y[g].reshape(-1, H, J)
    joints = np.argsort(recorded.std(0).mean(0))[::-1][:3]
    samples = flow.sample(x[g[:1]], k=16, seed=0)[0].reshape(-1, H, J)
    pred = reg.predict(x[g[:1]]).reshape(H, J)
    fig, axes = plt.subplots(1, len(joints), figsize=(10, 3.2))
    steps = np.arange(H)
    for ax, j in zip(axes, joints):
        for i, r in enumerate(recorded):
            ax.plot(steps, r[:, j], color=INK, lw=2.2, alpha=0.8, label="recorded plans" if i == 0 else None)
        for i, s in enumerate(samples):
            ax.plot(steps, s[:, j], color=FLOW, lw=1, alpha=0.5, label="flow samples" if i == 0 else None)
        ax.plot(steps, pred[:, j], color=REG, lw=2.2, label="regression")
        ax.set_title(f"joint {j}", loc="left")
        ax.set_xlabel("control step")
    axes[0].set_ylabel("torque")
    axes[0].legend(fontsize=8)
    suptitle(fig, f"One input, {len(g)} different recorded plans: samples vs regression")
    fig.tight_layout()
    fig.savefig(FIG / "5_samples.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    search = json.load(open(OUT / "search.json"))
    learning_curves(search)
    summary = {"search": [{"config": r["config"], "best_epoch": r["best_epoch"],
                           **{k: r["val"][k] for k in ("nmse_det", "energy")}}
                          for r in search["flow"]],
               "sizes": search["sizes"]}
    if (OUT / "scaling.json").exists():
        scaling = json.load(open(OUT / "scaling.json"))
        scaling_curves(scaling)
        scaling_plot(scaling)
    if (OUT / "models_full.pt").exists():
        sys.modules["__main__"].Regression = Regression
        models = torch.load(OUT / "models_full.pt", weights_only=False)
        sets, _, _ = data()
        var = sets["train"][1].var(0)
        summary["best_of_k"] = dict(zip(("flow", "control", "regression"),
                                        best_of_k_plot(models, sets, var)))
        x = np.vstack([sets[k][0] for k in sets])
        y = np.vstack([sets[k][1] for k in sets])
        groups, summary["multimodality"] = multimodality(models, x, y, var)
        summary["groups"] = len(groups)
        samples_plot(models, x, y, groups)
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != "search"}, indent=1, default=float))


if __name__ == "__main__":
    main()
