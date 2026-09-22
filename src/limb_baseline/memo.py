"""Memorize the first `plan-limb-motion` result and replay it on later calls.

After the notebook's section 4: stages 1-2 skipped, planner given the true range of
motion, no model fitted. A replay is verified open-loop and falls back to the MPC when it
misses; `--cache-mode naive` returns it unverified.

    python -m limb_baseline.memo --condition warm --variation 0 --num-tasks 20 \
        --out data/memo/same_task.csv
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
from kinder.envs.dynamic3d.limb_utils import LIMB_JOINT_INFOS, joint_position_distance
from kinder_pddlstream_planning.limbrepositioning3d import run as planner_run
from kinder_pddlstream_planning.limbrepositioning3d import stream as planner_stream
from kinder_pddlstream_planning.limbrepositioning3d.stream import (
    LimbStreamContext,
    MPCConfig,
    TorqueTrajectory,
    plan_limb_motion,
)
from kinder_pddlstream_planning.limbrepositioning3d.utils import (
    CoupledState,
    advance_corrected,
    capture_state,
    commanded_robot_torque,
    exceeds_human_torque_limits,
    exceeds_robot_torque_limits,
    human_in_collision,
    limb_error,
    limb_out_of_limits,
    restore_state,
    saved_sim_state,
)

from limb_baseline.dataset import append_row
from limb_baseline.human import DEFAULT_PRIOR, active_joints, theta_to_joint_limits
from limb_baseline.planner import PlannerSettings, run_episode
from limb_baseline.tasks import DEFAULT_TASK_DISTRIBUTION, Task
from limb_baseline.variant import DEFAULT_VARIANT, VARIANTS
from limb_baseline.variant import current as current_variant
from limb_baseline.variant import select as select_variant
from limb_baseline.worker import isolate_worker, task_sampler

CONDITIONS = ("plain", "warm")
CACHE_MODES = ("validated", "naive")


@dataclass
class LimbMotionMemo:
    """One memorized `plan-limb-motion` answer, and how often it was usable."""

    mode: str = "validated"
    trajectory: TorqueTrajectory | None = None
    end_state: CoupledState | None = None
    # Per-episode counters; `reset()` clears them before each plan.
    calls: int = 0
    hits: int = 0
    replay_misses: int = 0
    mpc_calls: int = 0
    seconds_in_stream: float = 0.0
    last_miss_reason: str = ""

    def reset(self) -> None:
        """Forget the counters, not the memorized plan."""
        self.calls = self.hits = self.replay_misses = self.mpc_calls = 0
        self.seconds_in_stream = 0.0
        self.last_miss_reason = ""

    @property
    def filled(self) -> bool:
        return self.trajectory is not None

    def _replay(
        self, ctx: LimbStreamContext, s1: CoupledState, q2
    ) -> tuple[TorqueTrajectory, CoupledState] | str:
        """Replay the memorized torques from `s1`; the new trajectory, or why it failed.

        The same open-loop check `plan_limb_motion` runs on its own MPC result, so a
        memorized plan is held to the same standard as a fresh one.
        """
        assert self.trajectory is not None
        sim = ctx.sim
        goal = np.asarray(q2.positions, dtype=np.float64)
        if not ctx.in_believed_limits(list(q2.positions)):
            return "the target is outside the limb's joint limits"
        robot_torques = [list(t) for t in self.trajectory.robot_torques]
        human_torques: list[list[float]] = []
        robot_induced: list[list[float]] = []
        commanded: list[list[float]] = []
        with saved_sim_state(sim):
            restore_state(sim, s1)
            limb_path = [list(sim.limb.get_joint_positions())]
            for torque in robot_torques:
                step_command = commanded_robot_torque(sim, torque)
                commanded.append(list(step_command))
                human_torque = advance_corrected(sim, torque)
                human_torques.append(list(human_torque.total))
                robot_induced.append(list(human_torque.robot))
                limb_path.append(list(sim.limb.get_joint_positions()))
                if limb_out_of_limits(ctx):
                    return "replayed the memorized plan past the limb's anatomical limits"
                if exceeds_human_torque_limits(ctx, human_torque):
                    return "replayed the memorized plan past the person's torque limits"
                if exceeds_robot_torque_limits(ctx, step_command):
                    return "replayed the memorized plan past the robot's torque limits"
            s2 = capture_state(sim)
            error = limb_error(ctx, s2.limb_positions, goal)
            if error >= ctx.goal_atol:
                return f"replayed the memorized plan to {error:.3f}, outside the tolerance"
            if ctx.resting_penetration and human_in_collision(ctx):
                return "replayed the memorized plan into the person or the furniture"
        return (
            TorqueTrajectory(
                robot_torques,
                human_torques,
                robot_induced,
                limb_path,
                commanded,
                [0.0] * len(robot_torques),
            ),
            s2,
        )

    def stream(
        self, ctx: LimbStreamContext, s1: CoupledState, q2
    ) -> Iterator[tuple[TorqueTrajectory, CoupledState]]:
        """Drop-in for `plan_limb_motion`: the memo first, the MPC only when it must.

        Time is charged before each yield: PDDLStream keeps the generator open, so a
        `finally` would only run at garbage collection, after the episode's row is out.
        """
        started = time.perf_counter()
        self.calls += 1
        if self.filled:
            if self.mode == "naive":
                assert self.trajectory is not None and self.end_state is not None
                self.hits += 1
                print("plan-limb-motion (memo): returned the memorized plan blindly.")
                self.seconds_in_stream += time.perf_counter() - started
                yield self.trajectory, self.end_state
                return
            replayed = self._replay(ctx, s1, q2)
            if not isinstance(replayed, str):
                self.hits += 1
                print(
                    "plan-limb-motion (memo): the memorized plan replays to the goal; "
                    f"{len(replayed[0].robot_torques)} steps, no MPC."
                )
                self.seconds_in_stream += time.perf_counter() - started
                yield replayed
                return
            self.replay_misses += 1
            self.last_miss_reason = replayed
            print(f"plan-limb-motion (memo): {replayed}; falling back to the MPC.")
        self.mpc_calls += 1
        for trajectory, s2 in plan_limb_motion(ctx, s1, q2):
            if not self.filled:
                self.trajectory, self.end_state = trajectory, s2
                print(
                    "plan-limb-motion (memo): memorized a plan of "
                    f"{len(trajectory.robot_torques)} steps."
                )
            self.seconds_in_stream += time.perf_counter() - started
            yield trajectory, s2
            started = time.perf_counter()
        self.seconds_in_stream += time.perf_counter() - started


# The seed the MPC's sampling noise draws from on the next construction, when set.
MPC_NOISE_SEED: int | None = None
_STOCK_MPC = planner_stream.PredictiveSamplingMPC


class _ReseededMPC(_STOCK_MPC):  # type: ignore[misc,valid-type]
    """The planner's MPC, with its noise drawn from `MPC_NOISE_SEED` instead.

    `LimbStreamContext.motion_seed` seeds grasp, base, IK and MPC noise alike. Repeating
    one task needs the first three fixed (so the grasped state, and with it the memo's
    replay, repeats) while the MPC still searches afresh each iteration.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if MPC_NOISE_SEED is not None:
            self._rng = np.random.default_rng(MPC_NOISE_SEED)


def install_mpc_reseeding() -> None:
    """Have every MPC the planner builds draw its noise from `MPC_NOISE_SEED`."""
    planner_stream.PredictiveSamplingMPC = _ReseededMPC


def install(memo: LimbMotionMemo) -> None:
    """Route every `plan-limb-motion` call this process makes through `memo`.

    `create_problem` looks the stream function up by name in its module at call time,
    so rebinding that name is enough; nothing in the planner package changes on disk.
    """
    planner_run.plan_limb_motion = memo.stream


def uninstall() -> None:
    """Put the stock stream back."""
    planner_run.plan_limb_motion = plan_limb_motion


# ----------------------------------------------------------------------------- tasks


def nominal_task(theta: np.ndarray, seed: int) -> Task:
    """The one task both conditions repeat: reachable for this person, fixed by `seed`."""
    return task_sampler().sample(np.random.default_rng(seed), theta, theta)


def perturbed_task(
    nominal: Task,
    theta: np.ndarray,
    rng: np.random.Generator,
    variation: float,
    vary_init: bool,
    max_draws: int = 200,
) -> Task:
    """`nominal` with its goal (and optionally its start) nudged by up to `variation` rad.

    Each active joint moves by an independent uniform draw in `[-variation, variation]`,
    clipped to the person's range of motion. The draw is rejected if the separation
    leaves the task distribution's band or the nominal base pose can no longer carry it,
    so every variant is as solvable as the original.
    """
    sampler = task_sampler()
    lower, upper = theta_to_joint_limits(theta)
    distribution = DEFAULT_TASK_DISTRIBUTION
    for _ in range(max_draws):
        goal = np.asarray(nominal.limb_goal, dtype=np.float64)
        init = np.asarray(nominal.limb_init, dtype=np.float64)
        for j in active_joints():
            goal[j] = np.clip(goal[j] + rng.uniform(-variation, variation), lower[j], upper[j])
            if vary_init:
                init[j] = np.clip(
                    init[j] + rng.uniform(-variation, variation), lower[j], upper[j]
                )
        separation = joint_position_distance(LIMB_JOINT_INFOS, list(init), list(goal))
        if not distribution.min_separation <= separation <= distribution.max_separation:
            continue
        if nominal.base_pose is not None and not sampler._reachable_from(  # noqa: SLF001
            nominal.base_pose, init, goal
        ):
            continue
        return dataclasses.replace(
            nominal, limb_init=tuple(init), limb_goal=tuple(goal)
        )
    raise RuntimeError(
        f"no reachable perturbation of the nominal task within {variation} rad "
        f"in {max_draws} draws"
    )


def task_shift(task: Task, nominal: Task) -> float:
    """How far `task`'s goal sits from the nominal goal, in the environment's metric."""
    return joint_position_distance(
        LIMB_JOINT_INFOS, list(task.limb_goal), list(nominal.limb_goal)
    )


# ------------------------------------------------------------------------ experiment


def run_sequence(
    condition: str,
    num_tasks: int,
    variation: float,
    vary_init: bool,
    person: int,
    task_seed: int,
    cache_mode: str,
    settings: PlannerSettings,
    out: Path,
    tag: str,
) -> list[dict]:
    """Plan `num_tasks` episodes in order in this one process, appending a row each.

    The memo lives across iterations, which is the whole point, so the sequence cannot
    be spread over workers the way `evaluate` spreads independent episodes.
    """
    isolate_worker()
    theta = DEFAULT_PRIOR.sample(np.random.default_rng(person))
    nominal = nominal_task(theta, task_seed)
    memo = LimbMotionMemo(mode=cache_mode)
    if condition == "warm":
        install(memo)
    else:
        uninstall()
    install_mpc_reseeding()

    records = []
    for iteration in range(num_tasks):
        rng = np.random.default_rng(task_seed * 1000 + iteration)
        task = (
            nominal
            if variation <= 0.0
            else perturbed_task(nominal, theta, rng, variation, vary_init)
        )
        memo.reset()
        global MPC_NOISE_SEED  # pylint: disable=global-statement
        MPC_NOISE_SEED = task_seed * 1000 + iteration
        record = {
            "tag": tag,
            "condition": condition,
            "cache_mode": cache_mode if condition == "warm" else "",
            "variation": variation,
            "vary_init": vary_init,
            "variant": current_variant().name,
            "person": person,
            "task_seed": task_seed,
            "iteration": iteration,
            "mpc_seed": MPC_NOISE_SEED,
            "separation": round(task.separation, 4),
            "goal_shift": round(task_shift(task, nominal), 4),
        }
        try:
            # One geometry seed for every iteration, so grasp, base pose and IK repeat
            # and the memo has a state to replay from; only the MPC noise (above) is
            # fresh, so the plain condition shows its own spread.
            result = run_episode(
                task,
                theta,
                belief=None,
                believed_theta=theta,
                warm_start=None,
                seed=task_seed,
                settings=settings,
            )
            record.update(
                {
                    "solved": result.solved,
                    "goal_reached": result.goal_reached,
                    "plan_time": round(result.plan_time, 2),
                    "execute_time": round(result.execute_time, 2),
                    "stream_seconds": round(memo.seconds_in_stream, 2),
                    "stream_calls": memo.calls,
                    "memo_hits": memo.hits,
                    "memo_replay_misses": memo.replay_misses,
                    "mpc_calls": memo.mpc_calls,
                    "memo_filled": memo.filled,
                    "num_control_steps": result.num_control_steps,
                    "rom_violation_steps": result.rom_violation_steps,
                    "failure_reason": result.failure_reason,
                    "memo_miss_reason": memo.last_miss_reason,
                }
            )
        except Exception as exc:  # pylint: disable=broad-except
            record.update(
                {
                    "solved": False,
                    "goal_reached": False,
                    "plan_time": float("nan"),
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
            traceback.print_exc()
        append_row(out, record)
        records.append(record)
        print(
            f"[{tag}/{condition} {iteration + 1}/{num_tasks}] "
            f"plan={record.get('plan_time')}s reached={record.get('goal_reached')} "
            f"hits={record.get('memo_hits', 0)} mpc={record.get('mpc_calls', 0)} "
            f"{record.get('failure_reason', '')}",
            flush=True,
        )
    uninstall()
    return records


def main() -> None:
    """CLI entry point: one condition, one task sequence, one CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument(
        "--variation", type=float, default=0.0,
        help="Radians each active goal joint may move from the nominal task; 0 repeats "
             "one identical task.",
    )
    parser.add_argument("--vary-init", action="store_true",
                        help="Also nudge the limb's start, not only its goal.")
    parser.add_argument("--cache-mode", choices=CACHE_MODES, default="validated")
    parser.add_argument("--variant", choices=VARIANTS, default=DEFAULT_VARIANT)
    parser.add_argument("--person", type=int, default=22,
                        help="Seed of the one person whose true range of motion is used.")
    parser.add_argument("--task-seed", type=int, default=0)
    parser.add_argument("--max-time", type=float, default=600.0)
    parser.add_argument("--num-rollouts", type=int, default=MPCConfig.num_rollouts)
    parser.add_argument("--max-control-steps", type=int, default=MPCConfig.max_control_steps)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, "1")

    select_variant(args.variant)
    mpc = dataclasses.replace(
        MPCConfig(),
        num_rollouts=args.num_rollouts,
        max_control_steps=args.max_control_steps,
    )
    settings = PlannerSettings(max_time=args.max_time, mpc=mpc)
    out = args.out.resolve()
    tag = args.tag or ("same_task" if args.variation <= 0.0 else f"varied_{args.variation:g}")
    records = run_sequence(
        args.condition, args.num_tasks, args.variation, args.vary_init, args.person,
        args.task_seed, args.cache_mode, settings, out, tag,
    )
    times = [r["plan_time"] for r in records if r.get("solved")]
    print(
        f"\n{args.condition}: {sum(bool(r.get('goal_reached')) for r in records)}/"
        f"{len(records)} reached the goal; median plan time of solved episodes "
        f"{np.median(times) if times else float('nan'):.1f} s\nWritten to {out}"
    )


if __name__ == "__main__":
    main()
