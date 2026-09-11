"""Render a collection run's figures.

`python scripts/make_figures.py --run-dir data/babble/isolated_left_arm`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from limb_baseline.dataset import (  # noqa: E402
    load_dataset,
    train_test_split_by_person,
)

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

ROWS = ("pddlstream", "belief", "warm_start", "baseline", "oracle_cold", "oracle")
ROW_LABELS = {
    "pddlstream": "plain",
    "belief": "query",
    "warm_start": "warm",
    "baseline": "both",
    "oracle_cold": "oracle cold",
    "oracle": "oracle",
}


def _present(evaluation: pd.DataFrame) -> list[str]:
    """The baseline rows this evaluation actually ran, in table order."""
    seen = set(evaluation.row)
    return [r for r in ROWS if r in seen]


LABELS = {
    "ridge": "ridge",
    "mlp": "neural net",
    "gradient_boosting": "boosted trees",
    "nearest_neighbors": "nearest neighbours",
}


def _save(fig, stem: str, out: Path) -> None:
    """Write one figure as both PNG and SVG."""
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(out / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
        print(f"wrote {out / f'{stem}.{suffix}'}")
    plt.close(fig)


def figure_models(report: dict, out: Path) -> None:
    """Prediction error on people the model trained on versus people it has not.

    Divided by the zero predictor's error, so runs whose torques differ in size are
    on one scale: 1.0 is no better than the zeros the MPC already starts from.
    """
    order = sorted(report, key=lambda k: report[k]["test_mse"])
    test_zero = report[order[0]]["zero_baseline_mse"]
    train_zero = report[order[0]].get("zero_baseline_train_mse") or test_zero
    train = [report[k]["train_mse"] / train_zero for k in order]
    test = [report[k]["test_mse"] / test_zero for k in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x - 0.19, train, 0.36, color=DROP, label="training data")
    ax.bar(x + 0.19, test, 0.36, color=KEEP, label="test data")
    ax.set_ylim(0, 1.18 * max(test + [1.0]))
    for i, value in enumerate(train):
        if value < 0.01:
            ax.text(i - 0.19, 0.01, "0.00", ha="center", fontsize=9, color=FAINT)
    ax.set_xticks(x, [LABELS.get(k, k) for k in order])
    ax.set_ylabel("prediction error (NMSE)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title("Prediction error", loc="left", fontsize=11)
    _save(fig, "model_error", out)


def figure_signal(path: str, out: Path) -> None:
    """How much the saved torques say about where the limb ended up."""
    from sklearn.cross_decomposition import CCA

    X, Y, meta = load_dataset(path)
    xtr, ytr, xte, yte = train_test_split_by_person(X, Y, meta, seed=0)
    disp = lambda a: a[:, 13:19] - a[:, 0:6]  # noqa: E731
    cca = CCA(n_components=3).fit(disp(xtr), ytr)
    corr = lambda a, b, i: float(  # noqa: E731
        np.corrcoef(*np.array(cca.transform(disp(a), b))[:, :, i])[0, 1]
    )
    x = np.arange(3)
    ins = [corr(xtr, ytr, i) for i in x]
    outs = [corr(xte, yte, i) for i in x]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(x - 0.19, ins, 0.36, color=DROP, label="training data")
    ax.bar(x + 0.19, outs, 0.36, color=KEEP, label="test data")
    for i, (a, b) in enumerate(zip(ins, outs, strict=True)):
        ax.text(i - 0.19, a + 0.008, f"{a:.2f}", ha="center", fontsize=9, color=FAINT)
        ax.text(i + 0.19, b + 0.008, f"{b:.2f}", ha="center", fontsize=9, color=KEEP)
    ax.set_xticks(x, ["1st pattern", "2nd", "3rd"])
    ax.set_ylabel("correlation")
    ax.set_ylim(0, 1.15 * max(ins + outs))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Torque\u2013goal correlation", loc="left", fontsize=11)
    _save(fig, "signal", out)


FAILURES = [
    # (substring of the raw reason, who it belongs to, short name)
    ("no limb motion was attempted", "planner", "Search timeout"),
    ("ran out of control steps", "mpc", "Control step budget spent"),
    ("stalled at an error", "mpc", "MPC error diverged"),
    ("diverged", "mpc", "MPC error diverged"),
    ("loaded a limb joint past", "mpc", "Human torque limit violation"),
    ("more torque than it has", "mpc", "Robot torque limit violation"),
    ("anatomical limit", "mpc", "Human RoM limit violation"),
    ("replayed open-loop to", "mpc", "Replay: goal tolerance"),
    ("replayed open-loop past the person", "mpc", "Replay: human torque limit"),
    ("replayed open-loop past the robot", "mpc", "Replay: robot torque limit"),
    ("replayed open-loop past the limb", "mpc", "Replay: RoM limit"),
    ("replayed open-loop into", "mpc", "Replay: collision"),
]


def _order(counts: pd.DataFrame):
    """Sort key: the MPC's own error first, replay rows last, else commonest first."""
    def key(reason: str) -> tuple[int, float]:
        if reason.startswith("MPC"):
            rank = 0
        elif reason.startswith("Replay"):
            rank = 2
        else:
            rank = 1
        return (rank, -counts.loc[reason].sum())

    return key


def _failure(reason: str) -> tuple[str, str]:
    """Map a raw failure string to (owner, short name)."""
    for needle, owner, short in FAILURES:
        if needle in reason:
            return owner, short
    return "other", reason[:40]


def _wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson 95% interval for a success rate."""
    if n == 0:
        return 0.0, 0.0
    p, z = k / n, 1.96
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return max(0.0, c - h), min(1.0, c + h)


def figure_eval(evaluation: pd.DataFrame, out: Path) -> None:
    """Success rate per baseline, and the controller effort behind it."""
    rows = _present(evaluation)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.8))

    tops = []
    for i, row in enumerate(rows):
        sub = evaluation[evaluation.row == row]
        k, n = int(sub.goal_reached.sum()), len(sub)
        low, high = _wilson(k, n)
        tops.append(high)
        ax1.bar(i, k / n, 0.55, color=DROP if row == "pddlstream" else KEEP)
        ax1.errorbar(i, k / n, yerr=[[k / n - low], [high - k / n]], color=INK,
                     capsize=4, lw=1.1)
        ax1.text(i, high + 0.012, f"{k}/{n}", ha="center", fontsize=9)
    ax1.set_xticks(range(len(rows)))
    ax1.set_xticklabels([ROW_LABELS[r] for r in rows], fontsize=9)
    ax1.set_ylabel("solved")
    ax1.set_ylim(0, 1.16 * max(tops))
    ax1.grid(axis="y", color=GRID, lw=0.8)
    ax1.set_axisbelow(True)
    ax1.set_title("Tasks solved (95% ranges)", loc="left", fontsize=11)

    steps = evaluation.pivot_table(index=["person", "task_index"], columns="row",
                                   values="mpc_simulator_steps")
    solved = evaluation.pivot_table(index=["person", "task_index"], columns="row",
                                    values="goal_reached")
    both = steps[(solved.pddlstream == 1) & (solved.warm_start == 1)] / 1000
    lim = [0, 1.06 * max(both.max())]
    ax2.plot(lim, lim, color=FAINT, lw=1, ls="--")
    ax2.scatter(both.pddlstream, both.warm_start, s=42, color=KEEP,
                edgecolor="white", zorder=3)
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.set_xlabel("plain (k steps)")
    ax2.set_ylabel("warm start (k steps)")
    ax2.grid(color=GRID, lw=0.8)
    ax2.set_axisbelow(True)
    ax2.set_title(f"Controller effort, {len(both)} shared tasks",
                  loc="left", fontsize=11)
    _save(fig, "evaluation", out)


def figure_failures(evaluation: pd.DataFrame, out: Path) -> None:
    """Table image: what each failure was, and whose stage it happened in."""
    rows = _present(evaluation)
    lost = evaluation[~evaluation.goal_reached].copy()
    owned = lost.failure_reason.fillna("").map(_failure)
    lost["owner"] = [o for o, _ in owned]
    lost["reason"] = [r for _, r in owned]
    counts = pd.crosstab(lost.reason, lost.row)
    for row in rows:
        if row not in counts:
            counts[row] = 0
    owner_of = dict(zip(lost.reason, lost.owner, strict=True))

    lines: list[tuple[str, str, list[str]]] = []
    for owner, title in (("planner", "planner"), ("mpc", "mpc"), ("other", "other")):
        group = [r for r in counts.index if owner_of[r] == owner]
        if not group:
            continue
        lines.append(("group", title, []))
        for reason in sorted(group, key=_order(counts)):
            lines.append(("row", reason, [str(counts.loc[reason, r]) for r in rows]))
    totals = counts.sum()
    lines.append(("total", "Total failures", [str(totals[r]) for r in rows]))

    xs = [0.60 + 0.40 * i / max(len(rows) - 1, 1) for i in range(len(rows))]
    fig, ax = plt.subplots(figsize=(7.6, 0.23 * len(lines) + 0.75))
    ax.axis("off")
    top = len(lines) + 1.3
    ax.text(0, top, "failure", fontsize=9, color=FAINT)
    for x, row in zip(xs, rows, strict=True):
        ax.text(x, top, ROW_LABELS[row], fontsize=9, color=FAINT, ha="right")
    ax.plot([0, 1.02], [top - 0.42, top - 0.42], color=FAINT, lw=0.9)
    for i, (kind, label, values) in enumerate(lines):
        y = len(lines) - i
        if kind == "group":
            ax.text(0, y, label, fontsize=8.5, color=FAINT, style="italic")
            continue
        weight = "semibold" if kind == "total" else "normal"
        if kind == "total":
            ax.plot([0, 1.02], [y + 0.5, y + 0.5], color=GRID, lw=0.9)
        ax.text(0.03, y, label, fontsize=9, color=INK, weight=weight)
        for x, value in zip(xs, values, strict=True):
            ax.text(x, y, value, fontsize=9, color=INK, ha="right", weight=weight)
    ax.set_xlim(-0.01, 1.03)
    ax.set_ylim(0.4, top + 0.5)
    ax.set_title("Why the failures failed", loc="left", fontsize=11, y=1.0)
    _save(fig, "failures", out)


def figure_margin(evaluation: pd.DataFrame, out: Path) -> None:
    """How close each baseline drove the limb to the person's true limit."""
    rows = _present(evaluation)
    data = [evaluation[evaluation.row == r].rom_margin.dropna().to_numpy()
            for r in rows]
    if not any(len(d) for d in data):
        return
    top = max(d.max() for d in data if len(d))
    low = min(d.min() for d in data if len(d))
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.axhline(0, color=WARN, lw=1.4, ls="--")
    ax.text(len(rows) - 0.45, 0, "  true limit", color=WARN, va="center", fontsize=9)
    for i, values in enumerate(data):
        if not len(values):
            continue
        ax.boxplot(values, positions=[i], widths=0.5, showfliers=False,
                   medianprops={"color": INK, "lw": 1.4},
                   boxprops={"color": FAINT}, whiskerprops={"color": FAINT},
                   capprops={"color": FAINT})
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(values)), values, s=14,
                   color=KEEP, alpha=0.45, zorder=3, edgecolors="none")
        ax.text(i, top * 1.06, f"n={len(values)}", ha="center", fontsize=8,
                color=FAINT)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([ROW_LABELS[r] for r in rows], fontsize=9)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(min(low, 0) - 0.03, top * 1.14)
    ax.set_ylabel("clearance (rad)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Clearance from the true range of motion", loc="left", fontsize=11)
    _save(fig, "rom_margin", out)


def figure_data(paths: dict[str, str], out: Path) -> None:
    """Where each collection method's rows sit, side by side."""
    from scipy.spatial import ConvexHull
    from sklearn.decomposition import PCA

    names = list(paths)
    loaded = {n: load_dataset(paths[n]) for n in names}
    disp = {n: loaded[n][0][:, 13:19] - loaded[n][0][:, 0:6] for n in names}
    labels = {n: loaded[n][1] for n in names}
    colors = dict(zip(names, (KEEP, WARN), strict=True))

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6))
    fig.subplots_adjust(wspace=0.32)
    for n in names:
        tag = f"{n} ({len(disp[n]):,} rows)"
        axes[0].hist(np.linalg.norm(disp[n], axis=1), bins=30, density=True,
                     histtype="step", lw=1.6, color=colors[n], label=tag)
        axes[1].hist(np.sqrt((labels[n] ** 2).mean(1)), bins=30, density=True,
                     histtype="step", lw=1.6, color=colors[n], label=tag)
    axes[0].set_xlabel("goal distance (rad)")
    axes[0].set_ylabel("density")
    axes[0].set_title("How far the goal is", loc="left", fontsize=11)
    axes[1].set_xlabel("torque magnitude (RMS)")
    axes[1].set_title("How hard the torques push", loc="left", fontsize=11)

    # Only the joints theta bounds: the rest are passive slop babble alone moves.
    from limb_baseline.human import active_joints

    active = list(active_joints())
    pca = PCA(n_components=2).fit(np.vstack([disp[n][:, active] for n in names]))
    for n in names:
        z = pca.transform(disp[n][:, active])
        axes[2].scatter(z[:, 0], z[:, 1], s=9, alpha=0.28, color=colors[n],
                        edgecolors="none")
        loop = np.append(ConvexHull(z).vertices, ConvexHull(z).vertices[0])
        axes[2].plot(z[loop, 0], z[loop, 1], color=colors[n], lw=1.6)
        axes[2].scatter([], [], s=36, color=colors[n], label=n)
    axes[2].set_xlabel("displacement PC1")
    axes[2].set_ylabel("PC2")
    axes[2].set_title("Which motions were seen (active joints)",
                      loc="left", fontsize=11)
    for ax in axes:
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8)
    _save(fig, "data_distribution", out)


def figure_learning_curve(path: str, out: Path, repeats: int = 5) -> None:
    """Training and test error against how many rows the model was given.

    Averaged over `repeats` draws of each subset, with a band of one standard
    deviation; 1.0 is the zero predictor the MPC already starts from.
    """
    from limb_baseline.warmstart import make_regressors

    X, Y, meta = load_dataset(path)
    xtr, ytr, xte, yte = train_test_split_by_person(X, Y, meta, seed=0)
    fractions = np.linspace(0.15, 1.0, 8)
    counts = [max(int(f * len(xtr)), 5) for f in fractions]
    test_zero = float(np.mean(yte ** 2))
    names = list(make_regressors())

    fig, axes = plt.subplots(1, len(names), figsize=(4.0 * len(names), 3.5), sharey=True)
    top = 1.0
    for ax, name in zip(np.atleast_1d(axes), names, strict=True):
        train = np.zeros((repeats, len(counts)))
        test = np.zeros_like(train)
        for r in range(repeats):
            order = np.random.default_rng(r).permutation(len(xtr))
            for k, take in enumerate(counts):
                rows = order[:take]
                model = make_regressors()[name].fit(xtr[rows], ytr[rows])
                train[r, k] = float(np.mean((model.predict(xtr[rows]) - ytr[rows]) ** 2)
                                    ) / float(np.mean(ytr[rows] ** 2))
                test[r, k] = float(np.mean((model.predict(xte) - yte) ** 2)) / test_zero
        ax.axhline(1.0, color=FAINT, lw=1.0, ls=":", zorder=1)
        for values, colour, label in ((train, DROP, "training data"),
                                      (test, KEEP, "test data")):
            mean, spread = values.mean(0), values.std(0)
            ax.fill_between(counts, mean - spread, mean + spread, color=colour,
                            alpha=0.18, lw=0)
            ax.plot(counts, mean, marker="o", ms=4, lw=1.8, color=colour, label=label)
            top = max(top, float((mean + spread).max()))
        ax.set_xlabel("training rows")
        ax.set_title(LABELS.get(name, name), loc="left", fontsize=11)
        ax.grid(color=GRID, lw=0.8, which="both")
        ax.set_axisbelow(True)
    # Log: the early spike is several times the zero predictor and would flatten the rest.
    ticks = [t for t in (0.125, 0.25, 0.5, 1, 2, 4, 8) if t <= 1.6 * top]
    axes_list = list(np.atleast_1d(axes))
    axes_list[0].set_ylabel("prediction error (NMSE)")
    axes_list[0].set_yscale("log")
    axes_list[0].set_yticks(ticks, [f"{t:g}" for t in ticks])
    axes_list[0].set_ylim(0.1, 1.15 * top)
    axes_list[0].legend(frameon=False, fontsize=9, loc="lower left")
    _save(fig, "learning_curve", out)


def figure_learning(path: str, out: Path) -> None:
    """Whether more rows would help, and what the iterative fits do while training."""
    from limb_baseline.warmstart import make_regressors

    X, Y, meta = load_dataset(path)
    xtr, ytr, xte, yte = train_test_split_by_person(X, Y, meta, seed=0)
    fractions = np.linspace(0.15, 1.0, 7)
    order = np.random.default_rng(0).permutation(len(xtr))
    palette = {"ridge": KEEP, "mlp": WARN, "gradient_boosting": INK,
               "nearest_neighbors": FAINT}

    zero = float(np.mean(yte ** 2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 3.8))
    fig.subplots_adjust(wspace=0.3)
    for name in make_regressors():
        curve = []
        for fraction in fractions:
            take = order[: max(int(fraction * len(xtr)), 5)]
            model = make_regressors()[name].fit(xtr[take], ytr[take])
            curve.append(float(np.mean((model.predict(xte) - yte) ** 2)) / zero)
        ax1.plot(fractions * len(xtr), curve, marker="o", ms=4, lw=1.6,
                 color=palette[name], label=LABELS.get(name, name))
    ax1.set_xlabel("training rows")
    ax1.set_ylabel("test error (NMSE)")
    ax1.grid(color=GRID, lw=0.8)
    ax1.set_axisbelow(True)
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title("Does more data help", loc="left", fontsize=11)

    # lbfgs keeps no loss curve, so the panel carries the boosted trees alone then.
    mlp = make_regressors()["mlp"].fit(xtr, ytr).named_steps["mlpregressor"]
    if hasattr(mlp, "loss_curve_"):
        ax2.plot(np.arange(1, len(mlp.loss_curve_) + 1), mlp.loss_curve_,
                 color=WARN, lw=1.6)
        ax2.set_ylabel("neural net training loss", color=WARN)
        ax2.tick_params(axis="y", colors=WARN)
    else:
        ax2.set_yticks([])
    ax2.set_xlabel("iteration")

    gbt = make_regressors()["gradient_boosting"].fit(xtr, ytr)
    stages = [list(e.staged_predict(xte)) for e in gbt.estimators_]
    depth = min(len(s) for s in stages)
    errs = [float(np.mean([(stages[j][i] - yte[:, j]) ** 2
                           for j in range(len(stages))])) / zero
            for i in range(depth)]
    twin = ax2.twinx()
    twin.plot(np.arange(1, depth + 1), errs, color=INK, lw=1.6)
    twin.set_ylabel("boosted trees test error (NMSE)", color=INK)
    twin.grid(False)
    ax2.grid(color=GRID, lw=0.8)
    ax2.set_axisbelow(True)
    ax2.set_title("Per-iteration fits", loc="left", fontsize=11)
    _save(fig, "learning", out)


def figure_timing(evaluation: pd.DataFrame, out: Path) -> None:
    """Table image: what each baseline cost in search time and controller work."""
    rows = _present(evaluation)
    heads = ["solved", "plan s (won)", "plan s (all)", "sim steps / call"]
    lines = []
    for row in rows:
        sub = evaluation[evaluation.row == row]
        won = sub[sub.goal_reached]
        lines.append((ROW_LABELS[row], [
            f"{len(won)}/{len(sub)}",
            f"{won.plan_time.mean():.1f}",
            f"{sub.plan_time.mean():.1f}",
            f"{(won.mpc_simulator_steps / won.num_segments).mean():,.0f}",
        ]))

    xs = [0.34 + 0.22 * i for i in range(len(heads))]
    fig, ax = plt.subplots(figsize=(8.6, 0.28 * len(lines) + 1.0))
    ax.axis("off")
    top = len(lines) + 1.3
    ax.text(0, top, "row", fontsize=9, color=FAINT)
    for x, head in zip(xs, heads, strict=True):
        ax.text(x, top, head, fontsize=9, color=FAINT, ha="right")
    ax.plot([0, 1.0], [top - 0.42, top - 0.42], color=FAINT, lw=0.9)
    for i, (label, values) in enumerate(lines):
        y = len(lines) - i
        ax.text(0.02, y, label, fontsize=9, color=INK)
        for x, value in zip(xs, values, strict=True):
            ax.text(x, y, value, fontsize=9, color=INK, ha="right")
    ax.set_xlim(-0.01, 1.02)
    ax.set_ylim(0.4, top + 0.5)
    ax.set_title("Planning and controller cost", loc="left", fontsize=11, y=1.0)
    _save(fig, "timing", out)


def print_tables(report: dict, demos: pd.DataFrame, attempts: pd.DataFrame,
                 evaluation: pd.DataFrame | None) -> None:
    """The same numbers as the figures, as text."""
    print("\n=== collection ===")
    print(f"episodes {len(attempts):,} | rows {len(demos):,} | "
          f"people {demos.person.nunique()} | "
          f"rows per person {demos.groupby('person').size().median():.0f} (median)")
    moved = f"mean distance moved {attempts[attempts.solved].separation.mean():.3f}"
    # Redraws are a babble counter; a planner run has search time in its place.
    if "num_redraws" in attempts:
        print(f"safety redraws {int(attempts.num_redraws.sum()):,} | {moved}")
    else:
        print(f"median plan time {attempts.plan_time.median():.0f}s | {moved}")

    print("\n=== prediction error ===")
    print(f"{'model':<20}{'training':>12}{'test':>13}{'vs zero predictor':>21}")
    baseline = next(iter(report.values()))["zero_baseline_mse"]
    for k in sorted(report, key=lambda k: report[k]["test_mse"]):
        r = report[k]
        gain = 100 * (baseline - r["test_mse"]) / baseline
        print(f"{LABELS.get(k, k):<20}{r['train_mse']:>12.5f}"
              f"{r['test_mse']:>13.5f}{gain:>20.1f}%")

    if evaluation is None:
        return
    print("\n=== evaluation ===")
    # Averaged over solved tasks only: a failed task records no controller steps,
    # so a per-episode mean just counts how many failures a row happened to have.
    for row in _present(evaluation):
        s = evaluation[evaluation.row == row]
        won = s[s.goal_reached]
        print(f"{row:<12}solved {len(won)}/{len(s)}  "
              f"when solved: plan {won.plan_time.mean():6.1f}s  "
              f"controller steps {won.mpc_simulator_steps.mean():,.0f}  "
              f"RoM violations {int(s.rom_violation_steps.sum())}"
              + (f"  min clearance {s.rom_margin.min():.3f} rad"
                 if "rom_margin" in s and s.rom_margin.notna().any() else ""))
    print("\n=== why the failures failed ===")
    lost = evaluation[~evaluation.goal_reached].copy()
    owned = lost.failure_reason.fillna("").map(_failure)
    lost["owner"] = [o for o, _ in owned]
    lost["reason"] = [r for _, r in owned]
    counts = pd.crosstab(lost.reason, lost.row)
    columns = _present(evaluation)
    for row in columns:
        if row not in counts:
            counts[row] = 0
    owner_of = dict(zip(lost.reason, lost.owner, strict=True))
    header = "".join(f"{ROW_LABELS[r]:>12}" for r in columns)
    print(f"{'failure':<34}{header}")
    for owner, title in (("planner", "planner"), ("mpc", "mpc"), ("other", "other")):
        group = [r for r in counts.index if owner_of[r] == owner]
        if not group:
            continue
        print(f"-- {title}'s fault")
        for reason in sorted(group, key=_order(counts)):
            cells = "".join(f"{counts.loc[reason, r]:>12}" for r in columns)
            print(f"{reason:<34}{cells}")

    p = evaluation.pivot_table(index=["person", "task_index"], columns="row",
                               values="goal_reached")
    a = int(((p.pddlstream == 1) & (p.warm_start == 0)).sum())
    b = int(((p.pddlstream == 0) & (p.warm_start == 1)).sum())
    print(f"tasks that flipped: {a} lost, {b} won, "
          f"p = {binomtest(min(a, b), a + b, 0.5).pvalue:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="A collect run directory, e.g. data/babble/isolated_left_arm.")
    parser.add_argument("--compare-dir", type=Path, default=None,
                        help="Second run directory to plot the data distribution against.")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Defaults to figures/<method>/<tag>, so each run keeps its own folder."
    )
    args = parser.parse_args()

    run = args.run_dir
    out = args.out or Path("figures") / run.parent.name / run.name
    report = json.loads((run / "fit.json").read_text())
    demos = pd.read_csv(run / "demos.csv")
    attempts = pd.read_csv(run / "attempts.csv")
    eval_path = run / "eval.csv"
    evaluation = pd.read_csv(eval_path) if eval_path.exists() else None

    figure_models(report, out)
    figure_signal(str(run / "demos.csv"), out)
    figure_learning(str(run / "demos.csv"), out)
    figure_learning_curve(str(run / "demos.csv"), out)
    if args.compare_dir:
        figure_data({run.parent.name: str(run / "demos.csv"),
                     args.compare_dir.parent.name: str(args.compare_dir / "demos.csv")}, out)
    if evaluation is not None:
        figure_eval(evaluation, out)
        figure_failures(evaluation, out)
        figure_timing(evaluation, out)
        if "rom_margin" in evaluation:
            figure_margin(evaluation, out)
    print_tables(report, demos, attempts, evaluation)


if __name__ == "__main__":
    main()
