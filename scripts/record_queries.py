"""Record one video per query and plot how the belief moves.

`python scripts/record_queries.py --out videos --fig figures`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import matplotlib
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from limb_baseline.belief import Belief, pooled_tau, refit  # noqa: E402
from limb_baseline.human import (  # noqa: E402
    active_motions,
    DEFAULT_PRIOR,
    default_magnitudes,
)
from limb_baseline.pipeline import QueryPolicy  # noqa: E402
from limb_baseline.queries import DEFAULT_QUERY_MODEL, QuerySimulator  # noqa: E402

KEEP = "#008D63"
DROP = "#AAB4B2"
INK = "#141C1B"
FAINT = "#6E7A78"
GRID = "#E2E7E6"
RED = "#D42A2A"
BLUE = "#1A5FB4"

# Each motion swings in its own plane, so each gets its own camera yaw and distance.
MOTION_VIEW = {
    "shoulder_flexion": (90.0, 2.05, [0.0, -0.05, -0.05]),
    "shoulder_abduction": (0.0, 2.05, [0.0, -0.05, -0.05]),
    "elbow_flexion": (50.0, 1.35, [0.0, -0.05, -0.28]),
}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": FAINT,
        "axes.labelcolor": INK,
        "axes.labelsize": 10,
        "text.color": INK,
        "xtick.color": FAINT,
        "ytick.color": FAINT,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


class Recorder:
    """Grabs frames from the query simulator and annotates them."""

    def __init__(self, sim: QuerySimulator, size=(640, 480), stride: int = 10) -> None:
        self.sim = sim
        self.width, self.height = size
        self.stride = stride
        self.frames: list[np.ndarray] = []
        self.projection = p.computeProjectionMatrixFOV(
            45.0, self.width / self.height, 0.05, 5.0,
            physicsClientId=sim.physics_client_id,
        )

    def begin(self, motion: str, target: float, bound: float) -> None:
        self.frames = []
        self.effort = 0.0
        self.motion, self.target, self.bound = motion, target, bound
        yaw, distance, target = MOTION_VIEW[motion]
        self.view = p.computeViewMatrixFromYawPitchRoll(
            target, distance, yaw, -12.0, 0.0, 2,
            physicsClientId=self.sim.physics_client_id,
        )

    def capture(self, step: int, reach: float) -> None:
        """Frame callback for `QuerySimulator.ask`."""
        if step % self.stride:
            return
        _, _, rgb, _, _ = p.getCameraImage(
            self.width, self.height, self.view, self.projection,
            renderer=p.ER_TINY_RENDERER, physicsClientId=self.sim.physics_client_id,
        )
        frame = np.asarray(rgb, dtype=np.uint8).reshape(self.height, self.width, 4)
        self.frames.append((frame[:, :, :3].copy(), reach))

    def _annotate(self, frame: np.ndarray, reach: float) -> np.ndarray:
        """Draw the motion, the numbers, and a gauge of reach against the bound."""
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        draw.text((14, 12), self.motion.replace("_", " "), fill=INK)
        draw.text((14, 28), f"target {self.target:.2f} rad", fill=FAINT)
        draw.text((14, 44), f"reach  {reach:.2f} rad", fill=KEEP)
        draw.text((14, 60), f"effort {self.effort:.1f} N·m", fill=FAINT)

        left, right, top = 14, self.width - 14, self.height - 34
        draw.rectangle([left, top, right, top + 12], outline=DROP)
        # The axis runs to whichever of target and bound is larger, so neither marker clips.
        span = max(self.target, self.bound)
        scale = (right - left) / max(span, 1e-6)
        filled = left + scale * float(np.clip(reach, 0.0, span))
        draw.rectangle([left, top, filled, top + 12], fill=KEEP)
        wall = left + scale * self.bound
        draw.line([wall, top - 7, wall, top + 19], fill=BLUE, width=3)
        label_x = max(left, min(wall - 52, right - 116))
        draw.text((label_x, top + 22), "their bound (hidden)", fill=BLUE)
        goal = left + scale * self.target
        draw.line([goal, top - 7, goal, top + 19], fill=RED, width=3)
        draw.text((max(left, min(goal - 18, right - 36)), top - 21), "target", fill=RED)
        draw.text((left, top - 21), "0", fill=FAINT)
        return np.asarray(image)

    def save(self, path: Path, fps: int = 30) -> None:
        """Annotate and encode. Annotation waits for the effort, drawn inside `ask`."""
        path.parent.mkdir(parents=True, exist_ok=True)
        annotated = [self._annotate(frame, reach) for frame, reach in self.frames]
        iio.imwrite(path, np.stack(annotated), fps=fps, codec="libx264")
        print(f"wrote {path}  ({len(annotated)} frames)")


def run(args) -> None:
    """Ask the queries, record each one, and collect the belief after every round."""
    rng = np.random.default_rng(args.seed)
    theta = DEFAULT_PRIOR.sample(rng)
    model = DEFAULT_QUERY_MODEL
    policy = QueryPolicy(num_queries=args.num_queries)
    sim = QuerySimulator(theta, model)
    recorder = Recorder(sim, stride=args.stride)
    base = Belief.from_prior(DEFAULT_PRIOR)
    belief = base

    floors = np.zeros(len(active_motions()))
    answers: list[list[float]] = [[] for _ in active_motions()]
    tau = model.tau
    means = [belief.mean.copy()]
    stddevs = [belief.stddev.copy()]
    queried: list[int] = []
    reached: list[bool] = []
    try:
        for round_index in range(policy.num_queries):
            motion_index, target = policy.choose(belief, rng, floors)
            motion = active_motions()[motion_index]
            bound = abs(theta[motion_index] * default_magnitudes()[motion_index])
            recorder.begin(motion, target, bound)
            outcome = sim.ask(motion_index, target, rng, on_step=recorder.capture)
            recorder.effort = outcome.effort
            recorder.save(args.out / f"query_{round_index + 1}_{motion}.mp4", args.fps)

            plausible = outcome.peak_velocity < model.max_velocity
            if plausible and outcome.reached_target:
                floors[motion_index] = outcome.answer + model.short_fall
            elif plausible and outcome.settled:
                # Repeat answers about a motion measure the noise, so refold them all.
                answers[motion_index].append(outcome.answer)
                tau = pooled_tau(answers, model.tau)
                belief = refit(base, answers, tau, model.short_fall)
            means.append(belief.mean.copy())
            stddevs.append(belief.stddev.copy())
            queried.append(motion_index)
            reached.append(outcome.reached_target)
            print(
                f"  round {round_index + 1}: {motion} target {target:.2f} "
                f"stopped {outcome.answer:.3f} (bound {bound:.3f}) tau {tau:.4f}"
            )
    finally:
        sim.close()

    # One folder per seed, so a sweep does not overwrite its own figures.
    plot_belief(
        np.array(means), np.array(stddevs), queried, reached, theta, tau, model.tau,
        args.fig / "belief_update" / f"seed{args.seed}",
    )


def plot_belief(means, stddevs, queried, reached, theta, tau, assumed,
                out: Path) -> None:
    """One panel per motion, stepping through only the queries about that motion."""
    fig, axes = plt.subplots(len(active_motions()), 1, figsize=(6.4, 6.0))
    for i, (ax, motion) in enumerate(zip(axes, active_motions(), strict=True)):
        rounds = [r for r, m in enumerate(queried) if m == i]
        xs = np.arange(len(rounds) + 1)
        ys = np.array([means[0, i]] + [means[r + 1, i] for r in rounds])
        ss = np.array([stddevs[0, i]] + [stddevs[r + 1, i] for r in rounds])
        ax.fill_between(xs, ys - ss, ys + ss, color=KEEP, alpha=0.16, lw=0)
        ax.plot(xs, ys, color=KEEP, lw=1.8)
        for k, r in enumerate(rounds, start=1):
            ax.plot(k, ys[k], "o", ms=6, mec=KEEP, mew=1.5,
                    mfc="white" if reached[r] else KEEP)
        ax.axhline(theta[i], color=INK, lw=1.2, ls="--")
        ax.text(0.02, 0.86, motion.replace("_", " "), transform=ax.transAxes,
                fontsize=9, color=FAINT,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        ax.set_xticks(xs)
        ax.set_xticklabels(["prior"] + [f"q{r + 1}" for r in rounds])
        ax.set_xlim(-0.25, max(len(rounds), 1) + 0.25)
        ax.set_ylabel("theta scale")
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1].set_xlabel("query about this motion")
    fig.text(0.5, -0.01,
             f"noise measured from repeat answers:  "
             f"$\\hat{{\\tau}} = {tau:.3f}$ rad   (assumed $\\sigma_u/k = {assumed:.3f}$)",
             ha="center", fontsize=9, color=FAINT)
    handles = [
        plt.Line2D([], [], color=KEEP, lw=1.8, label="belief mean ±1σ"),
        plt.Line2D([], [], color=INK, lw=1.2, ls="--", label="true theta"),
        plt.Line2D([], [], marker="o", ls="", mfc=KEEP, mec=KEEP, ms=6, label="answered"),
        plt.Line2D([], [], marker="o", ls="", mfc="white", mec=KEEP, mew=1.5, ms=6,
                   label="reached target (censored)"),
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(out / f"belief_update.{suffix}", dpi=300, bbox_inches="tight")
        print(f"wrote {out / f'belief_update.{suffix}'}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--num-queries", type=int, default=6)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path("videos"))
    parser.add_argument("--fig", type=Path, default=Path("figures"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
