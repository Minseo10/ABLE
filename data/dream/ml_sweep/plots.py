"""Figures for the warm-start ML sweep: diagnostic ladder, learning curves, scaling, multimodal."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "figs")
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
    "lines.linewidth": 2, "figure.dpi": 150,
})
rows = {g: [json.loads(l) for l in open(HERE / f"results/{g}.jsonl")]
        for g in ("current", "capacity", "structure", "regularize", "scaling", "multimodal")}
RUN4 = lambda d: d.startswith("run:") and "_4_" in d


def pick(grid, ds_ok, feats, **cfg):
    return [r for r in rows[grid] if ds_ok(r["ds"]) and r["feats"] == feats
            and all(r["cfg"].get(k) == v for k, v in cfg.items())]


def med(rs, key):
    return float(np.median([r[key] for r in rs]))


# 1) Diagnostic ladder: train vs val per model, per-run (4-dream) data.
ladder = [
    ("MLP 128×128 (lbfgs)", pick("current", RUN4, "base", spec="mlp:128x128:a0.001")),
    ("ridge", pick("current", RUN4, "base", spec="ridge")),
    ("MLP 4×1024, no reg.", pick("capacity", RUN4, "base", model="torch", width=1024, depth=4, arch="mlp")),
    ("MLP 256×256, no reg.", pick("structure", RUN4, "base", model="torch", width=256, depth=2, cp=None)),
    ("extra-trees", pick("structure", RUN4, "base", model="et")),
    ("MLP 256×256 + vel/torque", pick("structure", RUN4, "base+vel+prev", model="torch", width=256, cp=None)),
    ("ridge + vel/torque", pick("structure", RUN4, "base+vel+prev", model="ridge")),
    ("extra-trees + vel/torque", pick("structure", RUN4, "base+vel+prev", model="et")),
    ("MLP 256×256 + vel/torque + dropout 0.3", pick("regularize", RUN4, "base+vel+prev", width=256, dropout=0.3, noise=None)),
]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
cap = 1.2
for i, (name, rs) in enumerate(ladder):
    tr, va = med(rs, "train"), med(rs, "val")
    ax.plot([tr, min(va, cap)], [i, i], color=GRID, lw=3, zorder=1, solid_capstyle="round")
    ax.scatter(tr, i, s=46, color=BLUE, zorder=3, edgecolor="white", linewidth=1.5)
    ax.scatter(min(va, cap), i, s=46, color=ORANGE, zorder=3, edgecolor="white", linewidth=1.5,
               marker=">" if va > cap else "o")
    ax.text(min(max(va, tr), cap) + 0.03, i, f"{va:.2f}", va="center", fontsize=8.5, color=INK2)
ax.axvspan(0.34, 0.41, color="#b7b6ae", alpha=0.25, lw=0, zorder=0)
ax.text(0.375, -0.75, "ambiguity floor", ha="center", va="bottom", fontsize=8, color=INK2)
ax.set_yticks(range(len(ladder)), [n for n, _ in ladder])
ax.invert_yaxis()
ax.set_xlim(0, cap + 0.12)
ax.set_ylim(len(ladder) + 0.3, -1.0)
ax.set_xlabel("nMSE (0 perfect, 1 = mean) — median of five 4-dream runs")
ax.scatter([], [], color=BLUE, s=40, label="train")
ax.scatter([], [], color=ORANGE, s=40, label="held-out (later trajectories)")
ax.legend(loc="lower right", fontsize=8.5)
fig.suptitle("Train vs held-out loss: overfitting, then a floor", x=0.01, ha="left", fontweight="bold", color=INK)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(OUT / "1_train_vs_val.png")

# 2) Learning curves on one run: big unregularized vs regularized.
ds = "run:mlp_4_p29"
curves = [
    ("MLP 4×1024, no reg.", pick("capacity", lambda d: d == ds, "base", width=1024, depth=4, arch="mlp")[0], BLUE),
    ("MLP 256×256 + vel/torque + dropout 0.3",
     pick("regularize", lambda d: d == ds, "base+vel+prev", width=256, dropout=0.3, noise=None, seed=None)[0], ORANGE),
]
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
for ax, (name, r, c) in zip(axes, curves):
    e, tr, va = np.array(r["curve"]).T
    ax.plot(e, tr, color=c, ls="--", label="train")
    ax.plot(e, va, color=c, label="held-out")
    ax.set_title(name, loc="left", fontsize=10)
    ax.set_xlabel("epoch")
    ax.legend(fontsize=8.5)
    ax.set_ylim(0, 1.3)
axes[0].set_ylabel("MSE (standardized)")
fig.suptitle(f"Learning curves, {ds.split(':')[1]}", x=0.01, ha="left", fontweight="bold", color=INK)
fig.tight_layout()
fig.savefig(OUT / "2_learning_curves.png")

# 3) Data scaling (pooled, fixed held-out set).
fig, ax = plt.subplots(figsize=(6, 3.6))
series = [("MLP 3×256 + vel/torque", "base+vel+prev", dict(width=256), BLUE),
          ("MLP 3×256, current inputs", "base", dict(width=256), ORANGE),
          ("ridge + vel/torque", "base+vel+prev", dict(model="ridge"), AQUA)]
for name, feats, cfg, c in series:
    pts = {}
    for r in pick("scaling", lambda d: True, feats, **cfg):
        key = "best_val_std" if r["cfg"]["model"] == "torch" else "val"
        pts.setdefault(r["n_seg"] // 10 * 10, []).append((r["n_seg"], r[key]))
    xs = sorted(pts)
    x = [np.mean([p[0] for p in pts[k]]) for k in xs]
    y = [np.median([p[1] for p in pts[k]]) for k in xs]
    ax.plot(x, y, color=c, marker="o", ms=5, label=name)
    ax.text(x[-1] * 1.08, y[-1], f"{y[-1]:.2f}", va="center", fontsize=8.5, color=INK2)
ax.axhspan(0.34, 0.41, color="#b7b6ae", alpha=0.25, lw=0)
ax.set_xscale("log")
ax.set_xlim(20, 1400)
ax.set_ylim(0, 1.1)
ax.axvspan(45, 70, color=YELLOW, alpha=0.12, lw=0)
ax.text(56, 0.03, "one person's run", ha="center", fontsize=8, color=INK2)
ax.set_xlabel("training trajectories (pooled over people, log scale)")
ax.set_ylabel("held-out nMSE")
ax.legend(fontsize=8.5, loc="upper right")
ax.set_title("More data helps, flattening toward the floor", loc="left")
fig.tight_layout()
fig.savefig(OUT / "3_data_scaling.png")

# 4) Multimodal: oracle best-of-K and top-1 vs K, against input-free K-means.
ks = [1, 2, 4, 8, 16]
ctrl = [float(l.split("=")[-1]) for l in open(HERE / "control_kmeans.txt")]
fig, ax = plt.subplots(figsize=(6, 3.6))
def mm(k, key):
    cfg = {"K": k} if k == 1 else {"K": k, "mix": "mdn"}
    rs = pick("multimodal", lambda d: d == "pool:all", "base+vel+prev", **cfg)
    return med(rs, key)
ax.plot(ks, ctrl, color=INK2, marker="o", ms=5, ls=":", label="input-free K-means, best of K")
ax.plot(ks, [mm(k, "val_oracle_std") for k in ks], color=BLUE, marker="o", ms=5, label="MDN, best of K (oracle)")
ax.plot(ks, [mm(k, "val") for k in ks], color=ORANGE, marker="o", ms=5, label="MDN, most-likely mode (deployable)")
ax.set_xscale("log", base=2)
ax.set_xticks(ks, [str(k) for k in ks])
ax.set_ylim(0, 1.05)
ax.set_xlabel("number of modes K")
ax.set_ylabel("held-out nMSE (pooled)")
ax.legend(fontsize=8.5, loc="lower left")
ax.set_title("Multiple modes don't help the deployable prediction", loc="left")
fig.tight_layout()
fig.savefig(OUT / "4_multimodal.png")
print("wrote", *sorted(p.name for p in OUT.glob("*.png")))
