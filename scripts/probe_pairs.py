"""Run MPC on a grid of (grasp, base) pairs, one row per pair.

Stage 1 of doc/downstream-aware-streams.md: does the pair, not just the task, decide
whether the limb motion succeeds?

Needs the stream-learning planner on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from functools import partial

from kinder_pddlstream_planning.limbrepositioning3d.run import (
    _logged_gen_fn,
    build_stream_context,
    create_env,
    reset_to_start,
)
from kinder_pddlstream_planning.limbrepositioning3d.stream import (
    LimbConf,
    MPCConfig,
    plan_grasp_motion,
    plan_limb_motion,
    sample_base_pose,
    sample_grasp,
)

LIMB = "limb"


@dataclass
class Row:
    """One (grasp, base) pair and what MPC made of it."""

    variant: str
    seed: int
    goal_scale: float
    torque_limit: float
    grasp: int
    base: int
    slide: float
    roll: float
    base_x: float
    base_y: float
    base_rot: float
    saturation: float
    reached: bool
    failure: str
    best_error: float
    control_steps: int
    human_torque_ratio: float
    robot_share_ratio: float
    robot_torque_ratio: float
    rom_margin: float
    seconds: float


def _context(args, seed: int):
    """A reset environment and stream context for one worker."""
    sim = create_env(
        args.variant, muscle_tone=args.muscle_tone, goal_scale=args.goal_scale
    )
    sim.reset(seed=seed)
    ctx = build_stream_context(
        sim,
        motion_seed=seed,
        human_torque_limit=args.torque_limit,
        mpc=MPCConfig(
            num_rollouts=args.num_rollouts, max_control_steps=args.max_control_steps
        ),
    )
    reset_to_start(ctx)
    return sim, ctx


def _pair_rows(job) -> list[dict]:
    """Probe every base of one grasp, in a single process."""
    args, seed, grasp_index = job
    sim, ctx = _context(args, seed)
    rows: list[dict] = []
    try:
        scene = sim.config.scene
        init_conf = LimbConf(tuple(scene.limb_init_joint_positions))
        goal_conf = LimbConf(tuple(scene.limb_goal_joint_positions))
        # Through the logging wrapper, so each call's notes land in ctx.log.
        streams = {
            name: _logged_gen_fn(ctx, name, partial(fn, ctx))
            for name, fn in [
                ("sample-grasp", sample_grasp),
                ("sample-base-pose", sample_base_pose),
                ("plan-grasp-motion", plan_grasp_motion),
                ("plan-limb-motion", plan_limb_motion),
            ]
        }
        grasps = itertools.islice(streams["sample-grasp"](LIMB), grasp_index + 1)
        (grasp,) = list(grasps)[grasp_index]
        bases = streams["sample-base-pose"](LIMB, grasp, init_conf, goal_conf)
        for base_index, (base_conf,) in enumerate(
            itertools.islice(bases, args.num_bases)
        ):
            saturation = ctx.log.records[-1].get("saturation", float("nan"))
            start = time.time()
            motion = next(
                streams["plan-grasp-motion"](LIMB, grasp, base_conf, init_conf), None
            )
            if motion is None:
                note = dict(ctx.log.records[-1])
                rows.append(
                    asdict(
                        Row(
                            args.variant, seed, args.goal_scale, args.torque_limit,
                            grasp_index, base_index, grasp.slide, grasp.roll, base_conf.x,
                            base_conf.y, base_conf.rot, saturation, False,
                            note.get("failure", "no arm path"), float("nan"), 0,
                            float("nan"), float("nan"), float("nan"), float("nan"),
                            time.time() - start,
                        )
                    )
                )
                continue
            _, state = motion
            result = next(streams["plan-limb-motion"](state, goal_conf), None)
            note = dict(ctx.log.records[-1])
            rows.append(
                asdict(
                    Row(
                        args.variant, seed, args.goal_scale, args.torque_limit,
                        grasp_index, base_index, grasp.slide, grasp.roll, base_conf.x, base_conf.y,
                        base_conf.rot, saturation, result is not None,
                        note.get("failure", ""), note.get("best_error", float("nan")),
                        note.get("control_steps", 0),
                        note.get("human_torque_ratio", float("nan")),
                        note.get("robot_share_ratio", float("nan")),
                        note.get("robot_torque_ratio", float("nan")),
                        note.get("rom_margin", float("nan")),
                        time.time() - start,
                    )
                )
            )
    finally:
        sim.close()
    return rows


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="isolated-right-arm")
    parser.add_argument("--muscle-tone", default="none")
    parser.add_argument("--torque-limit", type=float, default=50.0)
    parser.add_argument("--goal-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-grasps", type=int, default=4)
    parser.add_argument("--num-bases", type=int, default=3)
    parser.add_argument("--num-rollouts", type=int, default=32)
    parser.add_argument("--max-control-steps", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    jobs = [(args, args.seed, index) for index in range(args.num_grasps)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = [row for batch in pool.map(_pair_rows, jobs) for row in batch]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    reached = sum(row["reached"] for row in rows)
    print(f"{args.variant} tau={args.torque_limit}: {reached}/{len(rows)} reached")


if __name__ == "__main__":
    main()
