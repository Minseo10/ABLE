"""The limb planner: fixed grasp and base, then one seeded MPC run from the limb's start to its goal."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from able.planners.deadline import PlanningTimeout
from able.planners.mpc import MPCPlanner, SeedStats
from able.tasks.limb.belief import LimbBelief
from able.tasks.limb.env.human import active_joints
from able.tasks.limb.env.sim import (
    DEFAULT_PLANNER_SETTINGS,
    ParticleLimits,
    PlannerSettings,
    joint_limits,
    open_episode,
)
from able.tasks.limb.planning.setup import FixedSetup, approach, wrist_grasp
from able.tasks.limb.planning.stream import (
    LimbConf,
    check_human_joint_limits,
    check_human_torque_limits,
    check_robot_torque_limits,
    plan_base_motion,
    plan_limb_motion,
)

if TYPE_CHECKING:
    from able.tasks.limb.task import LimbProblem, LimbState

RETARGET_MARGIN = 0.05  # rad inside the believed edge
CHECKS = (check_human_joint_limits, check_human_torque_limits, check_robot_torque_limits)


@dataclass
class LimbPlan:
    sim: Any
    ctx: Any
    actions: list
    task: LimbProblem  # after retargeting


def retarget(task: LimbProblem, limits: ParticleLimits) -> tuple[LimbProblem, bool]:
    """`task` with its goal moved just inside the believed range if it lies beyond."""
    goal = np.asarray(task.limb_goal, dtype=np.float64)
    if limits(goal):
        return task, False
    lower, upper = (np.asarray(b) for b in joint_limits(limits.quantile_theta()))
    beyond = [j for j in active_joints() if not lower[j] <= goal[j] <= upper[j]]
    margin, moved = RETARGET_MARGIN, goal
    for _ in range(10):
        moved = goal.copy()
        for j in beyond or active_joints():
            low, high = min(lower[j] + margin, upper[j]), max(upper[j] - margin, lower[j])
            moved[j] = np.clip(goal[j], low, high)
        if limits(moved):
            break
        margin += RETARGET_MARGIN
    return dataclasses.replace(task, limb_goal=tuple(moved)), True


def solve_limb_motion(ctx, state, target: LimbConf):
    """One MPC run; (trajectory, end state) or None if it fails a check."""
    outputs = plan_limb_motion(ctx, state, target)
    try:
        solved = next(iter(outputs), None)
    finally:
        outputs.close()
    if solved is None or not all(check(ctx, state, target, *solved) for check in CHECKS):
        return None
    return solved


@dataclass(kw_only=True)
class LimbPlanner(MPCPlanner):
    """MPC only: fixed grasp and base, then one seeded MPC run init -> goal; the deadline is `settings.max_time`."""

    setup: FixedSetup
    settings: PlannerSettings = DEFAULT_PLANNER_SETTINGS
    particles: int = 32
    confidence: float = 0.5

    def __post_init__(self) -> None:
        self.max_time = self.settings.max_time

    def solve(self, state: LimbState, belief: LimbBelief, proposal: Any, stats: SeedStats,
              rng: np.random.Generator) -> tuple[LimbPlan | None, dict]:
        limits = ParticleLimits(belief.sample(rng, self.particles), self.confidence)
        task, retargeted = retarget(state.task, limits)
        sim, ctx = open_episode(task, state.theta, limits.quantile_theta(), state.seed, self.settings, limits)
        ctx.seed_config, ctx.seed_stats = self.seed_config, stats
        ctx.mpc_noise_seed, ctx.warm_start = state.seed, proposal
        info: dict = {"retargeted": retargeted}
        actions, reason = None, ""
        try:
            actions = self._actions(ctx, task)
        except PlanningTimeout as timeout:
            reason = str(timeout)
        if actions is None:
            info["failure"] = reason or (ctx.best_attempt.reason if ctx.best_attempt else "no grasp or approach")
            sim.close()
            return None, info
        return LimbPlan(sim, ctx, actions, task), info

    def _actions(self, ctx, task: LimbProblem):
        grasp = wrist_grasp(ctx, self.setup.offset, self.setup.roll)
        base = self.setup.base_pose
        found = approach(ctx, grasp, base, task.limb_init)
        if found is None:
            return None
        trajectory, coupled = found
        path = next(iter(plan_base_motion(ctx, ctx.start_base_pose, base)), (None,))[0]
        path = path or [ctx.start_base_pose, base]
        goal = LimbConf(tuple(task.limb_goal))
        solved = solve_limb_motion(ctx, coupled, goal)
        if solved is None:
            return None
        torques, reached = solved
        start = LimbConf(tuple(task.limb_init))
        return [
            ("move_base", (ctx.start_base_pose, path, base)),
            ("grasp", (ctx.limb_name, grasp, base, trajectory, coupled)),
            ("move_limb", (ctx.limb_name, grasp, base, coupled, start, goal, torques, reached)),
        ]
