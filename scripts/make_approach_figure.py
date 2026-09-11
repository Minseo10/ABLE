"""Score every warm-start design against the same target: the executed torques.

`python scripts/make_approach_figure.py`.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
import numpy as np
from kinder_pddlstream_planning.limbrepositioning3d.stream import MPCConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from limb_baseline.dataset import load_dataset, split_column  # noqa: E402
from limb_baseline.warmstart import control_basis, make_regressors  # noqa: E402

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

JOINTS, POINTS = 7, 8


def episodes_from(run: Path):
    """One entry per episode: its start pose, every state, and the torques flown."""
    features, labels, meta = load_dataset(run / "demos.csv")
    step, task = meta["segment"].to_numpy(), meta["task_index"].to_numpy()
    groups = meta[split_column(meta)].to_numpy()
    out = []
    for one in np.unique(task):
        index = np.where(task == one)[0]
        index = index[np.argsort(step[index])]
        if len(index) < 2:
            continue
        out.append({"x0": features[index[0]], "xs": features[index],
                    "psi": labels[index], "truth": labels[index][:, :JOINTS],
                    "group": groups[index[0]]})
    return out


def _resample(traj, count):
    src, dst = np.linspace(0, 1, len(traj)), np.linspace(0, 1, count)
    return np.stack([np.interp(dst, src, traj[:, j]) for j in range(JOINTS)], axis=1)


def measure(run: Path) -> dict:
    """NMSE of each design against the executed torque at every control step."""
    warnings.filterwarnings("ignore")
    mpc = MPCConfig()
    basis = control_basis(mpc)
    episodes = episodes_from(run)
    unique = np.unique([e["group"] for e in episodes])
    np.random.default_rng(0).shuffle(unique)
    held = set(unique[: max(1, int(round(0.2 * len(unique))))].tolist())
    train = [e for e in episodes if e["group"] not in held]
    test = [e for e in episodes if e["group"] in held]
    starts = np.array([e["x0"] for e in train])

    once = make_regressors()["mlp"].fit(starts, np.array([e["psi"][0] for e in train]))
    every = make_regressors()["mlp"].fit(np.vstack([e["xs"] for e in train]),
                                         np.vstack([e["psi"] for e in train]))
    motion = make_regressors()["mlp"].fit(
        starts, np.array([_resample(e["truth"], POINTS).ravel() for e in train]))
    clock = make_regressors()["ridge"].fit(
        starts, np.array([len(e["truth"]) for e in train], dtype=float))

    def whole(entry, true_length: bool):
        points = motion.predict(entry["x0"].reshape(1, -1))[0].reshape(POINTS, JOINTS)
        count = (len(entry["truth"]) if true_length else
                 max(int(round(float(np.ravel(
                     clock.predict(entry["x0"].reshape(1, -1)))[0]))), 2))
        return _resample(points, count)

    designs = {
        "A\none horizon,\npredicted once":
            lambda e: basis @ once.predict(e["x0"].reshape(1, -1))[0].reshape(
                mpc.num_control_points, JOINTS),
        "B\none horizon,\npredicted every step":
            lambda e: every.predict(e["xs"])[:, :JOINTS],
        "C\nwhole motion,\npredicted length":
            lambda e: whole(e, False),
    }

    def score(predict, limit=None):
        top = bottom = 0.0
        for entry in test:
            truth = entry["truth"] if limit is None else entry["truth"][:limit]
            pred = np.atleast_2d(predict(entry))[: len(truth)]
            if len(pred) < len(truth):
                pred = np.vstack([pred, np.zeros((len(truth) - len(pred), JOINTS))])
            top += float(((pred - truth) ** 2).sum())
            bottom += float((truth ** 2).sum())
        return top / bottom

    results = {n: (score(f), score(f, mpc.horizon)) for n, f in designs.items()}
    for name, (a, b) in results.items():
        print(f"{name.replace(chr(10), ' '):<40} all={a:.3f}  first {mpc.horizon}={b:.3f}")
    return results


def figure(results: dict, out: Path) -> None:
    """One bar per design, on the whole motion and on the first horizon."""
    names = list(results)
    x = np.arange(len(names))
    every = [results[n][0] for n in names]
    early = [results[n][1] for n in names]

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.bar(x - 0.2, early, 0.38, color=KEEP, label="first 12 steps (one horizon)")
    ax.bar(x + 0.2, every, 0.38, color=DROP, label="every step of the motion")
    for xi, value in zip(np.r_[x - 0.2, x + 0.2], early + every):
        ax.text(xi, value + 0.012, f"{value:.2f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, names, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("torque error (NMSE)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Scored on the torques the robot actually flew", loc="left",
                 fontsize=11)

    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(out / f"approaches.{suffix}", dpi=300, bbox_inches="tight")
        print(f"wrote {out / f'approaches.{suffix}'}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("data/pddlstream/dense22"))
    parser.add_argument("--out", type=Path, default=Path("figures/dense"))
    args = parser.parse_args()
    figure(measure(args.run), args.out)


if __name__ == "__main__":
    main()
