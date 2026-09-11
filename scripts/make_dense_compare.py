"""Compare what the warm-start model is trained to predict, on one common test set.

`python scripts/make_dense_compare.py`.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from limb_baseline.dataset import load_dataset, split_column  # noqa: E402
from limb_baseline.warmstart import make_regressors  # noqa: E402

KEEP = "#008D63"
DROP = "#AAB4B2"
WARN = "#C25A34"
INK = "#141C1B"
FAINT = "#6E7A78"
GRID = "#E2E7E6"

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
    }
)

TESTS = ("at step 0", "at steps 0-9", "at every step")


def _nmse(pred, y) -> float:
    return float(np.mean((pred - y) ** 2) / np.mean(y**2))


def measure(run: Path) -> tuple[dict, dict, list]:
    """NMSE per (what the model predicts, what it is tested on), and the cutoff sweep."""
    warnings.filterwarnings("ignore")
    X, Y, meta = load_dataset(run / "demos.csv")
    groups = meta[split_column(meta)].to_numpy()
    step = meta["segment"].to_numpy()
    unique = np.unique(groups)
    np.random.default_rng(0).shuffle(unique)
    held = set(unique[: max(1, int(round(0.2 * len(unique))))].tolist())
    test = np.array([g in held for g in groups])
    train = ~test
    plain = np.arange(X.shape[1] - 1)  # every feature but the step index

    where = {"at step 0": test & (step == 0),
             "at steps 0-9": test & (step < 10),
             "at every step": test}
    variants = {
        "first step only": (train & (step == 0), plain),
        "every step": (train, plain),
    }
    scores, sizes = {}, {}
    for name, (mask, cols) in variants.items():
        model = make_regressors()["mlp"].fit(X[np.ix_(mask, cols)], Y[mask])
        sizes[name] = int(mask.sum())
        scores[name] = [_nmse(model.predict(X[np.ix_(w, cols)]), Y[w])
                        for w in where.values()]
        print(f"{name:<20} rows={sizes[name]:>6}  " +
              "  ".join(f"{t}={v:.3f}" for t, v in zip(TESTS, scores[name])))

    whole = whole_motion(X, Y, meta, groups, step, held)

    sweep = []
    for cutoff in (1, 5, 10, 20, 10**6):
        mask = train & (step < cutoff)
        model = make_regressors()["mlp"].fit(X[np.ix_(mask, plain)], Y[mask])
        sweep.append((cutoff, int(mask.sum()),
                      _nmse(model.predict(X[np.ix_(where["at step 0"], plain)]),
                            Y[where["at step 0"]])))
        print(f"cutoff<{cutoff:<8} rows={sweep[-1][1]:>6}  first step={sweep[-1][2]:.3f}")
    return scores, sizes, sweep, whole


def whole_motion(X, Y, meta, groups, step, held, points: int = 8, joints: int = 7):
    """Predict the entire trajectory from the start pose, resampled to `points`."""
    task = meta["task_index"].to_numpy()
    rows_x, rows_y, rows_g, lengths = [], [], [], []
    for one in np.unique(task):
        order = np.where(task == one)[0]
        order = order[np.argsort(step[order])]
        # Each row's first control point is the torque at that step.
        traj = Y[order][:, :joints]
        if len(traj) < 2:
            continue
        src, dst = np.linspace(0, 1, len(traj)), np.linspace(0, 1, points)
        rows_y.append(np.stack([np.interp(dst, src, traj[:, j]) for j in range(joints)],
                               axis=1).ravel())
        rows_x.append(X[order[0]])
        rows_g.append(groups[order[0]])
        lengths.append(len(traj))
    xs, ys = np.array(rows_x), np.array(rows_y)
    gs, ns = np.array(rows_g), np.array(lengths, dtype=float)
    te = np.array([g in held for g in gs])
    model = make_regressors()["mlp"].fit(xs[~te], ys[~te])
    shape = _nmse(model.predict(xs[te]), ys[te])
    clock = make_regressors()["ridge"].fit(xs[~te], ns[~te].reshape(-1, 1))
    predicted = clock.predict(xs[te]).ravel()
    return {"shape": shape, "duration": _nmse(predicted, ns[te]),
            "steps_off": float(np.abs(predicted - ns[te]).mean()),
            "steps_mean": float(ns[te].mean()), "episodes": int(len(xs))}


def figure(scores: dict, sizes: dict, sweep: list, whole: dict, out: Path) -> None:
    """What each training target costs, and what open-loop prediction costs on top."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.6, 3.9))
    fig.subplots_adjust(wspace=0.36)

    names = list(scores)
    x = np.arange(len(TESTS))
    for i, (name, colour) in enumerate(zip(names, (DROP, KEEP), strict=True)):
        offset = (i - 0.5) * 0.3
        ax1.bar(x + offset, scores[name], 0.28, color=colour,
                label=f"{name} ({sizes[name]:,} rows)")
        for xi, v in zip(x + offset, scores[name]):
            ax1.text(xi, min(v, 1.28) + 0.02, f"{v:.2f}", ha="center", fontsize=8,
                     color=INK)
    ax1.axhline(1.0, color=WARN, lw=1.2, ls="--")
    ax1.set_xticks(x, TESTS, fontsize=9)
    ax1.set_ylim(0, 1.4)
    ax1.set_ylabel("prediction error (NMSE)")
    ax1.set_xlabel("state the model is queried at")
    ax1.grid(axis="y", color=GRID, lw=0.8)
    ax1.set_axisbelow(True)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax1.set_title("What the model is trained to predict", loc="left", fontsize=11)

    # The open-loop variant predicts a different label, so it gets its own axes.
    labels = ["one horizon,\nqueried each step", "the whole motion,\nqueried once",
              "its motion\nlength"]
    values = [scores["every step"][2], whole["shape"], whole["duration"]]
    ax2.bar(np.arange(3), values, 0.5, color=[KEEP, INK, FAINT])
    for i, v in enumerate(values):
        ax2.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax2.axhline(1.0, color=WARN, lw=1.2, ls="--")
    ax2.set_xticks(np.arange(3), labels, fontsize=8.5)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("prediction error (NMSE)")
    ax2.grid(axis="y", color=GRID, lw=0.8)
    ax2.set_axisbelow(True)
    ax2.text(1.0, whole["shape"] + 0.10,
             f"needs the length too:\n±{whole['steps_off']:.0f} of {whole['steps_mean']:.0f} steps",
             ha="center", fontsize=8, color=WARN)
    ax2.set_title("Open loop costs a clock", loc="left", fontsize=11)

    rows = [s[1] for s in sweep]
    first = [s[2] for s in sweep]
    ax3.plot(rows, first, marker="o", ms=5, lw=1.8, color=KEEP)
    for (cutoff, n, v) in sweep:
        label = "all steps" if cutoff > 1000 else f"steps < {cutoff}"
        ax3.annotate(label, (n, v), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=8.5, color=FAINT)
    ax3.set_xscale("log")
    ax3.set_xlabel("training rows")
    ax3.set_ylabel("error at step 0 (NMSE)")
    ax3.grid(color=GRID, lw=0.8)
    ax3.set_axisbelow(True)
    ax3.set_title("Keeping late steps helps step 0", loc="left", fontsize=11)

    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(out / f"dense_compare.{suffix}", dpi=300, bbox_inches="tight")
        print(f"wrote {out / f'dense_compare.{suffix}'}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("data/pddlstream/dense22t"))
    parser.add_argument("--out", type=Path, default=Path("figures/dense"))
    args = parser.parse_args()
    figure(*measure(args.run), args.out)


if __name__ == "__main__":
    main()
