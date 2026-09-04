"""A task: where the limb starts, where the robot starts, and where it should end up.

Drawn against a person, because the configurations must lie inside their range of motion,
but the same record is handed to planners that know different amounts about them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from kinder.envs.dynamic3d.limb_utils import (
    LIMB_JOINT_INFOS,
    NUM_LIMB_JOINTS,
    NUM_ROBOT_JOINTS,
    joint_position_distance,
)
from numpy.typing import NDArray

from limb_baseline.human import active_joints, theta_to_joint_limits
from limb_baseline.variant import scene_config


def inactive_joints() -> tuple[int, ...]:
    """The limb joints a task does not vary; they keep the variant's own values."""
    return tuple(j for j in range(NUM_LIMB_JOINTS) if j not in active_joints())


@dataclass(frozen=True)
class Task:
    """One `(q_init_human, q_init_robot, q_goal_human)`."""

    limb_init: tuple[float, ...]
    robot_init: tuple[float, ...]
    limb_goal: tuple[float, ...]
    # The base pose this task was accepted with, if it came from a reachability check.
    base_pose: object | None = None

    def __post_init__(self) -> None:
        assert len(self.limb_init) == NUM_LIMB_JOINTS
        assert len(self.limb_goal) == NUM_LIMB_JOINTS
        assert len(self.robot_init) == NUM_ROBOT_JOINTS

    @property
    def separation(self) -> float:
        """How far the limb has to travel, in the environment's own metric."""
        return joint_position_distance(
            LIMB_JOINT_INFOS, list(self.limb_init), list(self.limb_goal)
        )

    def as_features(self) -> NDArray[np.float64]:
        """The nineteen numbers of a task the warm-start model reads."""
        return np.concatenate([self.limb_init, self.robot_init, self.limb_goal])


@dataclass(frozen=True)
class TaskDistribution:
    """`P(task)` for the single variant, as a box to sample inside.

    Active joints are drawn from the middle `inner_fraction` of the person's range: a
    configuration on the limit leaves the MPC no room to overshoot into.
    """

    inner_fraction: float = 0.6
    min_separation: float = 0.5
    # Above this the MPC does not converge; every episode solved so far was under 1.6.
    max_separation: float = 1.5
    robot_jitter: float = 0.2
    max_attempts: int = 1000

    def sample(
        self,
        rng: np.random.Generator,
        theta: NDArray[np.float64],
        robot_limits: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    ) -> Task:
        """Draw one task for the person `theta`."""
        scene = scene_config().scene
        nominal_limb = np.asarray(scene.limb_init_joint_positions, dtype=np.float64)
        nominal_robot = np.asarray(
            scene_config().robot_initial_joints, dtype=np.float64
        )
        lower, upper = theta_to_joint_limits(theta)
        margin = 0.5 * (1.0 - self.inner_fraction) * (upper - lower)
        inner_low, inner_high = lower + margin, upper - margin

        for _ in range(self.max_attempts):
            limb_init = nominal_limb.copy()
            limb_goal = nominal_limb.copy()
            for j in active_joints():
                limb_init[j] = rng.uniform(inner_low[j], inner_high[j])
                limb_goal[j] = rng.uniform(inner_low[j], inner_high[j])
            separation = joint_position_distance(
                LIMB_JOINT_INFOS, list(limb_init), list(limb_goal)
            )
            if not self.min_separation <= separation <= self.max_separation:
                continue
            robot_init = nominal_robot + rng.uniform(
                -self.robot_jitter, self.robot_jitter, size=NUM_ROBOT_JOINTS
            )
            if robot_limits is not None:
                robot_init = np.clip(robot_init, robot_limits[0], robot_limits[1])
            return Task(
                limb_init=tuple(limb_init),
                robot_init=tuple(robot_init),
                limb_goal=tuple(limb_goal),
            )
        raise RuntimeError(
            f"could not draw a task with separation >= {self.min_separation} for "
            f"theta={theta} in {self.max_attempts} attempts"
        )

    def sample_many(
        self,
        rng: np.random.Generator,
        theta: NDArray[np.float64],
        count: int,
        robot_limits: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    ) -> Iterator[Task]:
        """Draw `count` tasks for one person."""
        for _ in range(count):
            yield self.sample(rng, theta, robot_limits)


DEFAULT_TASK_DISTRIBUTION = TaskDistribution()


class ReachableTaskSampler:
    """Draws tasks the robot can actually attempt.

    Conditions `P(task)` on reachability from one placement -- the same population it is
    evaluated on.
    """

    def __init__(
        self,
        ctx,
        distribution: TaskDistribution = None,  # type: ignore[assignment]
        max_goal_draws: int = 30,
        max_init_draws: int = 8,
    ) -> None:
        from kinder_pddlstream_planning.limbrepositioning3d.run import LIMB_NAME
        from kinder_pddlstream_planning.limbrepositioning3d.stream import sample_grasp

        self.ctx = ctx
        self.distribution = distribution or DEFAULT_TASK_DISTRIBUTION
        self.max_goal_draws = max_goal_draws
        self.max_init_draws = max_init_draws
        self.limb_name = LIMB_NAME
        # The nominal grasp only: the other thirteen recover nothing and cost 10x.
        self.grasp = next(sample_grasp(ctx, LIMB_NAME))[0]

    def _base_pose_for(self, limb_init):
        """The nearest base pose that can hold the grasp at `limb_init`, or None.

        Paid once per init draw, not once per goal draw.
        """
        from kinder_pddlstream_planning.limbrepositioning3d.utils import (
            _base_candidates,
            _grasp_ik,
            _saved_sim_state,
            release_grasp,
        )

        sim = self.ctx.sim
        with _saved_sim_state(sim):
            release_grasp(sim)
            for base_conf in _base_candidates(self.ctx):
                sim.robot.set_base(base_conf)
                if (
                    _grasp_ik(
                        self.ctx,
                        self.grasp,
                        tuple(limb_init),
                        self.ctx.retract_joints,
                    )
                    is not None
                ):
                    return base_conf
        return None

    def _reachable_from(self, base_conf, limb_init, limb_goal) -> bool:
        """Whether that one base pose can carry the limb from init to goal.

        `sample_base_pose`'s check against one placement, not the whole grid, so
        rejecting a goal costs a handful of IK solves rather than four hundred.
        """
        from kinder_pddlstream_planning.limbrepositioning3d.stream import (
            NUM_REACH_CHECKS,
        )
        from kinder_pddlstream_planning.limbrepositioning3d.utils import (
            _grasp_ik,
            _saved_sim_state,
            limb_is_controllable,
            release_grasp,
        )

        sim = self.ctx.sim
        waypoints = np.linspace(
            np.asarray(limb_init), np.asarray(limb_goal), NUM_REACH_CHECKS
        )
        with _saved_sim_state(sim):
            release_grasp(sim)
            sim.robot.set_base(base_conf)
            seed = self.ctx.retract_joints
            for waypoint in waypoints:
                solution = _grasp_ik(self.ctx, self.grasp, tuple(waypoint), seed)
                if solution is None or not limb_is_controllable(self.ctx):
                    return False
                seed = solution
        return True

    def _inner_range(
        self, theta: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The box the limb's active joints are drawn from, off the person's edge."""
        lower, upper = theta_to_joint_limits(theta)
        margin = 0.5 * (1.0 - self.distribution.inner_fraction) * (upper - lower)
        return lower + margin, upper - margin

    def _draw_limb_conf(
        self,
        rng: np.random.Generator,
        low: NDArray[np.float64],
        high: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """One limb configuration; the joints a task does not vary keep the default."""
        conf = np.asarray(
            scene_config().scene.limb_init_joint_positions, np.float64
        )
        for j in active_joints():
            conf[j] = rng.uniform(low[j], high[j])
        return conf

    def _draw_robot_init(self, rng: np.random.Generator) -> NDArray[np.float64]:
        """The robot's own start, jittered around the variant's."""
        nominal = np.asarray(scene_config().robot_initial_joints, np.float64)
        return nominal + rng.uniform(
            -self.distribution.robot_jitter,
            self.distribution.robot_jitter,
            size=NUM_ROBOT_JOINTS,
        )

    def sample_init(
        self, rng: np.random.Generator, theta: NDArray[np.float64]
    ) -> Task:
        """A start the robot can grasp from, with no goal: `limb_goal` repeats the init.

        For collection that has no goal to plan to, such as hindsight babbling.
        """
        low, high = self._inner_range(theta)
        for _ in range(self.max_init_draws):
            limb_init = self._draw_limb_conf(rng, low, high)
            base_conf = self._base_pose_for(limb_init)
            if base_conf is None:
                continue
            return Task(
                limb_init=tuple(limb_init),
                robot_init=tuple(self._draw_robot_init(rng)),
                limb_goal=tuple(limb_init),
                base_pose=base_conf,
            )
        raise RuntimeError(
            f"no graspable start found for theta={theta} in "
            f"{self.max_init_draws} draws"
        )

    def sample(self, rng: np.random.Generator, theta: NDArray[np.float64]) -> Task:
        """One task for this person that some base pose can carry out."""
        low, high = self._inner_range(theta)
        for _ in range(self.max_init_draws):
            limb_init = self._draw_limb_conf(rng, low, high)
            base_conf = self._base_pose_for(limb_init)
            if base_conf is None:
                continue
            for _ in range(self.max_goal_draws):
                limb_goal = self._draw_limb_conf(rng, low, high)
                separation = joint_position_distance(
                    LIMB_JOINT_INFOS, list(limb_init), list(limb_goal)
                )
                if not (
                    self.distribution.min_separation
                    <= separation
                    <= self.distribution.max_separation
                ):
                    continue
                if not self._reachable_from(base_conf, limb_init, limb_goal):
                    continue
                return Task(
                    limb_init=tuple(limb_init),
                    robot_init=tuple(self._draw_robot_init(rng)),
                    limb_goal=tuple(limb_goal),
                    base_pose=base_conf,
                )
        raise RuntimeError(
            f"no reachable task found for theta={theta} in "
            f"{self.max_init_draws} x {self.max_goal_draws} draws"
        )
