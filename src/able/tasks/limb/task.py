"""The limb task: drive a person's limb to a goal with MPC from a fixed grasp and base."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from kinder.envs.dynamic3d.limb_utils import LIMB_JOINT_INFOS, joint_position_distance
from numpy.typing import NDArray

from able.api import Examples, Observation, PlanResult, Step
from able.planners.mpc import SeedConfig, SeedStats
from able.tasks.limb.belief import LimbBelief, LimbPrior, execution_evidence
from able.tasks.limb.episode import (
    DEFAULT_PLANNER_SETTINGS,
    EpisodeResult,
    ParticleLimits,
    PlannerSettings,
    PlanningTimeout,
    hard_deadline,
    joint_limits,
    make_sampling_context,
    open_episode,
    score_range_of_motion,
)
from able.tasks.limb.features import (
    _move_limb_segments,
    dense_rows,
    limb_segments,
    state_features,
)
from able.tasks.limb.human import DEFAULT_PRIOR, active_joints
from able.tasks.limb.planning.run import reset_to_start
from able.tasks.limb.planning.stream import (
    LimbConf,
    MPCConfig,
    TorqueTrajectory,
    check_human_joint_limits,
    check_human_torque_limits,
    check_robot_torque_limits,
    plan_base_motion,
    plan_limb_motion,
)
from able.tasks.limb.queries import PushToLimitGatherer
from able.tasks.limb.setup import FixedSetup, approach, load_setup, wrist_grasp
from able.tasks.limb.tasks import ReachableTaskSampler, Task
from able.tasks.limb.utils import RolloutLog, execute_plan
from able.tasks.limb.variant import select as select_variant

RETARGET_MARGIN = 0.05  # rad inside the believed edge
CHECKS = (check_human_joint_limits, check_human_torque_limits, check_robot_torque_limits)


@dataclass(frozen=True)
class LimbState:
    task: Task  # start, robot start and goal
    theta: NDArray[np.float64]  # the person (execution only)
    seed: int


@dataclass
class LimbPlan:
    sim: Any
    ctx: Any
    actions: list
    task: Task  # after retargeting


def retarget(task: Task, limits: ParticleLimits) -> tuple[Task, bool]:
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


@dataclass
class MPCPlanner:
    """MPC only: fixed grasp and base, then one seeded MPC run init -> goal."""

    setup: FixedSetup
    settings: PlannerSettings = DEFAULT_PLANNER_SETTINGS
    seed_config: SeedConfig = field(default_factory=SeedConfig)
    particles: int = 32
    confidence: float = 0.5
    effort_unit: str = "mpc_rollouts"

    def plan(self, state: LimbState, belief: LimbBelief, proposal=None, budget=None, rng=None) -> PlanResult:
        rng = rng if rng is not None else np.random.default_rng(state.seed)
        limits = ParticleLimits(belief.sample(rng, self.particles), self.confidence)
        task, retargeted = retarget(state.task, limits)
        sim, ctx = open_episode(task, state.theta, limits.quantile_theta(), state.seed, self.settings, limits)
        stats = SeedStats()
        ctx.seed_config, ctx.seed_stats = self.seed_config, stats
        ctx.mpc_noise_seed, ctx.warm_start = state.seed, proposal
        limit = budget.seconds if budget is not None and budget.seconds else self.settings.max_time
        started, actions, reason = time.time(), None, ""
        try:
            with hard_deadline(limit):
                actions = self._actions(ctx, task)
        except PlanningTimeout as timeout:
            reason = str(timeout)
        seconds = time.time() - started
        info = {"retargeted": retargeted, "seeded_steps": stats.seeded_steps, "full_steps": stats.full_steps,
                "model_seconds": stats.model_seconds}
        if actions is None:
            info["failure"] = reason or (ctx.best_attempt.reason if ctx.best_attempt else "no grasp or approach")
            sim.close()
            return PlanResult(None, seconds, stats.rollouts, info=info)
        return PlanResult(LimbPlan(sim, ctx, actions, task), seconds, stats.rollouts, info=info)

    def _actions(self, ctx, task: Task):
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


@dataclass
class LimbPsi:
    """x = limb/robot state, goal, belief, step (+ velocities, last torque); y = next horizon of torques."""

    mpc: MPCConfig = field(default_factory=MPCConfig)

    def examples(self, episode, belief: LimbBelief) -> Examples:
        rows = []
        for s in episode.steps:
            if s.step is not None and "segments" in s.step.info:
                rows += dense_rows(s.step.info["segments"], s.step.info["log"], belief.features(), self.mpc)
        if not rows:
            return Examples.empty(0, 0)
        x = np.array([r.features for r in rows])
        y = np.array([np.asarray(r.torques).ravel() for r in rows])
        return Examples(x, y, limb_segments(x).astype(np.int64))

    def proposal(self, learner, state, belief: LimbBelief):
        """`ctx.warm_start` hook: predicted horizon, plus learner draws as `samples`."""
        features, horizon = belief.features(), self.mpc.horizon
        draws = np.random.default_rng(getattr(state, "seed", None))  # reproducible learner draws

        def hook(s, goal, step=0.0):
            if step == 0:
                hook.previous = None
            x = state_features(s, goal, features, step, hook.previous).reshape(1, -1)
            return learner.predict(x)[0].reshape(horizon, -1)

        def samples(s, goal, step, k):
            x = state_features(s, goal, features, step, hook.previous).reshape(1, -1)
            return learner.sample(x, k, seed=int(draws.integers(2**31)))[0].reshape(k, horizon, -1)

        hook.previous, hook.samples = None, samples
        return hook


@dataclass
class LimbTask:
    """One variant per process (the variant is process-global)."""

    variant: str = "human-left-arm"
    settings: PlannerSettings = DEFAULT_PLANNER_SETTINGS
    setup: FixedSetup | None = None  # default: fixed_setup.json
    name: str = "limb"

    def __post_init__(self) -> None:
        select_variant(self.variant)
        self.setup = self.setup or load_setup(self.variant)
        self.prior = LimbPrior()
        self.gatherer = PushToLimitGatherer()
        self.psi = LimbPsi(self.settings.mpc)
        self._sampler = None

    def make_planner(self) -> MPCPlanner:
        return MPCPlanner(self.setup, self.settings)

    @property
    def sampler(self) -> ReachableTaskSampler:
        """Draws tasks reachable from the fixed grasp and base."""
        if self._sampler is None:
            self._sim, ctx = make_sampling_context(self.settings)
            reset_to_start(ctx)
            grasp = wrist_grasp(ctx, self.setup.offset, self.setup.roll)
            # One fixed base reaches few random tasks, so draw more (IK checks only).
            self._sampler = ReachableTaskSampler(ctx, max_goal_draws=60, max_init_draws=40,
                                                 grasp=grasp, base=self.setup.base_pose)
        return self._sampler

    def sample_problem(self, rng: np.random.Generator) -> Task:
        """Goals inside the prior-mean person's range; narrower beliefs retarget them."""
        return self.sampler.sample(rng, np.full(3, DEFAULT_PRIOR.mean))

    def reset(self, problem: Task, theta, rng: np.random.Generator) -> LimbState:
        return LimbState(problem, np.asarray(theta, np.float64), int(rng.integers(2**31)))

    def execute(self, state: LimbState, plan: LimbPlan, rng, frames: list | None = None) -> Step:
        del rng
        sim, ctx = plan.sim, plan.ctx
        try:
            result = EpisodeResult(solved=True)
            reset_to_start(ctx)
            log = RolloutLog()
            result.goal_reached = execute_plan(ctx, plan.actions, log=log, frames=frames)
            if not result.goal_reached:
                result.failure_reason = "the executed plan missed the goal"
            result.limb_path = [list(q) for q in log.limb_positions]
            score_range_of_motion(result, ctx)
        finally:
            sim.close()
        final = result.limb_path[-1] if result.limb_path else state.task.limb_init
        distance = joint_position_distance(LIMB_JOINT_INFOS, list(final), list(state.task.limb_goal))
        evidence = execution_evidence(result.limb_path, state.theta)
        segments = [(s1, goal, t) for s1, goal, t in _move_limb_segments(plan.actions)
                    if isinstance(t, TorqueTrajectory)]
        info = {"goal_distance": distance, "rom_violation_steps": result.rom_violation_steps,
                "failure": result.failure_reason, "segments": segments, "log": log,
                "limb_path": result.limb_path}
        observations = [Observation("execution", evidence)] if evidence else []
        return Step(state, observations, done=True, success=bool(result.goal_reached), info=info)
