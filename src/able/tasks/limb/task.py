"""The limb task: drive a person's limb to a goal with MPC from a fixed grasp and base."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from kinder.envs.dynamic3d.limb_utils import (
    LIMB_JOINT_INFOS,
    NUM_LIMB_JOINTS,
    NUM_ROBOT_JOINTS,
    joint_position_distance,
)
from numpy.typing import NDArray

from able.api import Observation, Step, Task
from able.tasks.limb.belief import LimbPrior, execution_evidence
from able.tasks.limb.distribution import LimbDistribution
from able.tasks.limb.env.execute import RolloutLog, execute_plan
from able.tasks.limb.env.sim import (
    DEFAULT_PLANNER_SETTINGS,
    EpisodeResult,
    PlannerSettings,
    score_range_of_motion,
)
from able.tasks.limb.env.variant import select as select_variant
from able.tasks.limb.planner import LimbPlan, LimbPlanner
from able.tasks.limb.planning.run import reset_to_start
from able.tasks.limb.planning.setup import FixedSetup, load_setup
from able.tasks.limb.planning.stream import TorqueTrajectory
from able.tasks.limb.psi import LimbPsi, _move_limb_segments
from able.tasks.limb.query import LimbGatherer


@dataclass(frozen=True)
class LimbProblem:
    """One `(q_init_human, q_init_robot, q_goal_human)`."""

    limb_init: tuple[float, ...]
    robot_init: tuple[float, ...]
    limb_goal: tuple[float, ...]
    # Base pose the reachability check accepted, if any.
    base_pose: object | None = None

    def __post_init__(self) -> None:
        assert len(self.limb_init) == NUM_LIMB_JOINTS
        assert len(self.limb_goal) == NUM_LIMB_JOINTS
        assert len(self.robot_init) == NUM_ROBOT_JOINTS

    @property
    def separation(self) -> float:
        """Limb travel distance in the environment's metric."""
        return joint_position_distance(
            LIMB_JOINT_INFOS, list(self.limb_init), list(self.limb_goal)
        )

    def as_features(self) -> NDArray[np.float64]:
        """The 19 task numbers the warm-start model reads."""
        return np.concatenate([self.limb_init, self.robot_init, self.limb_goal])


@dataclass(frozen=True)
class LimbState:
    task: LimbProblem  # start, robot start and goal
    theta: NDArray[np.float64]  # the person (execution only)
    seed: int


@dataclass
class LimbTask(Task):
    """One variant per process (the variant is process-global)."""

    variant: str = "human-left-arm"
    settings: PlannerSettings = DEFAULT_PLANNER_SETTINGS
    setup: FixedSetup | None = None  # default: planning/fixed_setup.json
    name: str = "limb"

    def __post_init__(self) -> None:
        select_variant(self.variant)
        self.setup = self.setup or load_setup(self.variant)
        self.prior = LimbPrior()
        self.gatherer = LimbGatherer()
        self.psi = LimbPsi(self.settings.mpc)
        self.distribution = LimbDistribution(self.settings, self.setup)

    def make_planner(self) -> LimbPlanner:
        return LimbPlanner(setup=self.setup, settings=self.settings)

    def reset(self, problem: LimbProblem, theta, rng: np.random.Generator) -> LimbState:
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
