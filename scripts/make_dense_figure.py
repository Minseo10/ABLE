"""Why one row per control step dilutes the label: the push decays, the rows do not.

`python scripts/make_dense_figure.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from limb_baseline.dataset import load_dataset  # noqa: E402

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


def _save(fig, stem: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(out / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
        print(f"wrote {out / f'{stem}.{suffix}'}")
    plt.close(fig)


def figure_torque(dense: Path, out: Path, max_step: int = 80) -> None:
    """How hard the robot pushes at each point of the motion, with the spread."""
    _, labels, meta = load_dataset(dense / "demos.csv")
    step = meta["segment"].to_numpy()
    energy = (labels ** 2).mean(axis=1)
    steps = np.arange(max_step)
    mid = np.array([np.median(energy[step == s]) if (step == s).any() else np.nan
                    for s in steps])
    low = np.array([np.percentile(energy[step == s], 25) if (step == s).any() else np.nan
                    for s in steps])
    high = np.array([np.percentile(energy[step == s], 75) if (step == s).any() else np.nan
                     for s in steps])
    mean = np.array([energy[step == s].mean() if (step == s).any() else np.nan
                     for s in steps])

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.fill_between(steps, low, high, color=KEEP, alpha=0.16, lw=0, label="middle half")
    ax.plot(steps, mid, color=KEEP, lw=2.0, label="median")
    ax.plot(steps, mean, color=INK, lw=1.4, ls="--", label="mean")
    ax.set_yscale("log")
    ax.set_xlabel("control step")
    ax.set_ylabel("torque size (mean $\\psi^2$)")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("The push decays along the trajectory", loc="left", fontsize=11)
    _save(fig, "torque_size", out)


def figure_dense(dense: Path, sparse: Path, out: Path, max_step: int = 60) -> None:
    """Label energy and row count per step, and what that costs the fit."""
    _, labels, meta = load_dataset(dense / "demos.csv")
    step = meta["segment"].to_numpy()
    steps = np.arange(max_step)
    energy = np.array([float((labels[step == s] ** 2).mean())
                       if (step == s).any() else np.nan for s in steps])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 3.8))
    fig.subplots_adjust(wspace=0.32)

    ax1.plot(steps, energy, color=KEEP, lw=2.0)
    ax1.set_yscale("log")
    ax1.set_xlabel("control step")
    ax1.set_ylabel("torque size (mean $\\psi^2$)")
    ax1.grid(color=GRID, lw=0.8)
    ax1.set_axisbelow(True)
    ax1.set_title("The push decays along the trajectory", loc="left", fontsize=11)

    reports = {"predict only first": sparse, "predict whole": dense}
    names, values = [], []
    for title, run in reports.items():
        report = json.loads((run / "fit.json").read_text())
        best = min(report, key=lambda k: report[k]["test_mse"])
        names.append(title)
        values.append(report[best]["test_mse"] / report[best]["zero_baseline_mse"])
    x = np.arange(len(names))
    ax2.bar(x, values, 0.5, color=[DROP, KEEP])
    for i, v in enumerate(values):
        ax2.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9, color=INK)
    ax2.axhline(1.0, color=WARN, lw=1.2, ls="--")
    ax2.set_xticks(x, names, fontsize=9)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("prediction error (NMSE)")
    ax2.grid(axis="y", color=GRID, lw=0.8)
    ax2.set_axisbelow(True)
    ax2.set_title("Denser data, worse fit", loc="left", fontsize=11)
    _save(fig, "dense_labels", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", type=Path, default=Path("data/pddlstream/dense22"))
    parser.add_argument("--sparse", type=Path, default=Path("data/pddlstream/person22"))
    parser.add_argument("--out", type=Path, default=Path("figures/dense"))
    args = parser.parse_args()
    figure_dense(args.dense, args.sparse, args.out)
    figure_torque(args.dense, args.out)


if __name__ == "__main__":
    main()
