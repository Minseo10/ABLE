"""Train / validation / test error of a dream run's learned model, per line (ABLE vs no dreaming).

Reads `<run>_learning.jsonl`; each real episode's refit gives one point per person:
- train nMSE: the refit's error on all its rows;
- validation nMSE: its error on the sixth of segments held out (the weights kept are those
  with the lowest held-out loss);
- test nMSE: the error on the real episode's rows before the model trains on them;
- train / validation loss: the model's own training loss (flow matching: velocity MSE) at
  the weights kept, and its curve over the training steps.

    python scripts/dream_losses.py data/dream/pomcpow_full figures/dream/pomcpow_full
"""

import glob
import json
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LINES = {4: ("ABLE", "-"), 0: ("No dreaming", "--")}
COLORS = {"train": "#2A78D6", "validation": "#E8642C", "test": "#1BAF7A"}
BINS = [(0, 4), (5, 9), (10, 14), (15, 19)]


def load(data: Path) -> pd.DataFrame:
    rows = []
    for path in glob.glob(str(data / "*_learning.jsonl")):
        name = Path(path).name
        dreams = int(re.search(r"_(\d+)_step", name).group(1))
        person = int(re.search(r"_p(\d+)_learning", name).group(1))
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("role") != "real":
                continue
            curve = np.array(r.get("curve") or [], float).reshape(-1, 3)
            kept = curve[curve[:, 0] == r.get("best_step")] if len(curve) else curve
            rows.append({
                "line": dreams, "person": person, "episode": r["episode"], "rows": r["rows"],
                "train_nmse": r.get("train_nmse"), "val_nmse": r.get("val_nmse"),
                "test_nmse": r.get("new_nmse"),
                "train_loss": kept[0, 1] if len(kept) else np.nan,
                "val_loss": r.get("val_loss"), "curve": curve,
            })
    return pd.DataFrame(rows).astype({"val_nmse": float, "test_nmse": float, "val_loss": float})


def per_episode(f: pd.DataFrame, column: str):
    g = f.dropna(subset=[column]).groupby("episode")[column]
    return g.mean(), g.std(ddof=1) / np.sqrt(g.count())


def band(ax, f, column, color, ls, label):
    m, se = per_episode(f, column)
    if m.empty:
        return
    ax.plot(m.index, m, ls=ls, color=color, marker="o", ms=3, lw=1.6, label=label)
    ax.fill_between(m.index, m - se.fillna(0), m + se.fillna(0), color=color, alpha=0.12, lw=0)


def figure(frame: pd.DataFrame, out: Path, inner=(5, 19)) -> None:
    fig, axes = plt.subplots(1, 2 + len(inner), figsize=(5.2 * (2 + len(inner)), 5.2))
    for dreams, (name, ls) in LINES.items():
        f = frame[frame.line == dreams]
        for kind in ("train", "validation", "test"):
            column = {"validation": "val"}.get(kind, kind) + "_nmse"
            band(axes[0], f, column, COLORS[kind], ls, f"{name}: {kind}")
        for kind in ("train", "validation"):
            column = {"validation": "val"}.get(kind, kind) + "_loss"
            band(axes[1], f, column, COLORS[kind], ls, f"{name}: {kind}")
        for ax, episode in zip(axes[2:], inner):
            curves = [c for c in f[f.episode == episode].curve if len(c)]
            if not curves:
                continue
            c = np.nanmean(np.stack([c for c in curves if c.shape == curves[0].shape]), 0)
            ax.plot(c[:, 0], c[:, 1], ls=ls, color=COLORS["train"], label=f"{name}: train")
            ax.plot(c[:, 0], c[:, 2], ls=ls, color=COLORS["validation"],
                    label=f"{name}: validation")
    axes[0].axhline(1, color="0.5", lw=1, ls=":")
    titles = ["(a) Torque-plan error (nMSE, 1 = predicting the average)",
              "(b) Flow-matching loss at the weights kept"]
    titles += [f"({chr(99 + i)}) Training inside the refit after episode {e}"
               for i, e in enumerate(inner)]
    for i, (ax, title) in enumerate(zip(axes, titles)):
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.set_xlabel("training step" if i >= 2 else "episode")
        if i < 2:
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    axes[0].set_ylabel("nMSE")
    axes[1].set_ylabel("loss (standardized)")
    axes[2].set_ylabel("loss (standardized), mean over people")
    people = frame.person.nunique()
    fig.suptitle(f"Learned model (flow matching) error: lines = mean over {people} people, "
                 "band = standard error", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out / f"learning_losses.{ext}", dpi=150)


def table(frame: pd.DataFrame, out: Path) -> pd.DataFrame:
    columns = ["rows", "train_nmse", "val_nmse", "test_nmse", "train_loss", "val_loss"]
    parts = []
    for dreams, (name, _) in LINES.items():
        for lo, hi in BINS:
            f = frame[(frame.line == dreams) & frame.episode.between(lo, hi)]
            if f.empty:
                continue
            row = f[columns].median().to_dict()
            parts.append({"line": name, "episodes": f"{lo}-{hi}", "refits": len(f), **row})
    t = pd.DataFrame(parts).round(3)
    t.to_csv(out / "learning_losses.csv", index=False)
    (out / "learning_losses.md").write_text(t.to_markdown(index=False) + "\n")
    return t


def main() -> None:
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    frame = load(data)
    figure(frame, out)
    print("medians over people x episodes in each bin")
    print(table(frame, out).to_string(index=False))


if __name__ == "__main__":
    main()
