"""P(problem): limb start, robot start and limb goal, drawn against a person's range and reachability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
from kinder.envs.dynamic3d.limb_utils import (
    LIMB_JOINT_INFOS,
    NUM_LIMB_JOINTS,
    NUM_ROBOT_JOINTS,
    joint_position_distance,
)
from numpy.typing import NDArray

from able.api import TaskDistribution
from able.tasks.limb.env.human import (
    DEFAULT_PRIOR,
    active_joints,
    theta_to_joint_limits,
)
from able.tasks.limb.env.variant import scene_config

if TYPE_CHECKING:
    from able.tasks.limb.env.sim import PlannerSettings
    from able.tasks.limb.planning.setup import FixedSetup
    from able.tasks.limb.task import LimbProblem


def inactive_joints() -> tuple[int, ...]:
    """Limb joints a task does not vary (kept at the variant's values)."""
    return tuple(j for j in range(NUM_LIMB_JOINTS) if j not in active_joints())


@dataclass(frozen=True)
class TaskBox:
    """`P(task)` as a box: active joints drawn from the middle `inner_fraction` of the range."""

    inner_fraction: float = 1.0
    min_separation: float = 0.5
    # Above this the MPC does not converge.
    max_separation: float = 1.5
    robot_jitter: float = 0.2
    max_attempts: int = 1000

    def inner_range(
        self, theta: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Box the active joints are drawn from."""
        lower, upper = theta_to_joint_limits(theta)
        margin = 0.5 * (1.0 - self.inner_fraction) * (upper - lower)
        return lower + margin, upper - margin

    def sample(
        self,
        rng: np.random.Generator,
        theta: NDArray[np.float64],
        true_theta: NDArray[np.float64] | None = None,
        robot_limits: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    ) -> LimbProblem:
        """One task: goal drawn against `theta`, start inside `true_theta` (only the goal may lie outside)."""
        from able.tasks.limb.task import LimbProblem

        scene = scene_config().scene
        nominal_limb = np.asarray(scene.limb_init_joint_positions, dtype=np.float64)
        nominal_robot = np.asarray(
            scene_config().robot_initial_joints, dtype=np.float64
        )
        inner_low, inner_high = self.inner_range(theta)
        start_low, start_high = self.inner_range(
            theta if true_theta is None else true_theta
        )

        for _ in range(self.max_attempts):
            limb_init = nominal_limb.copy()
            limb_goal = nominal_limb.copy()
            for j in active_joints():
                limb_init[j] = rng.uniform(start_low[j], start_high[j])
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
            return LimbProblem(
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
    ) -> Iterator[LimbProblem]:
        """Draw `count` tasks for one person."""
        for _ in range(count):
            yield self.sample(rng, theta, robot_limits)


DEFAULT_TASK_BOX = TaskBox()


class ReachableTaskSampler:
    """Draws tasks from `P(task)` conditioned on reachability from one placement."""

    def __init__(
        self,
        ctx,
        box: TaskBox = None,  # type: ignore[assignment]
        max_goal_draws: int = 30,
        max_init_draws: int = 8,
        max_conf_draws: int = 200,
        grasp=None,
        base=None,
    ) -> None:
        from able.tasks.limb.planning.run import LIMB_NAME
        from able.tasks.limb.planning.stream import sample_grasp

        self.ctx = ctx
        self.box = box or DEFAULT_TASK_BOX
        self.max_goal_draws = max_goal_draws
        self.max_init_draws = max_init_draws
        self.max_conf_draws = max_conf_draws
        self.limb_name = LIMB_NAME
        self.grasp = grasp if grasp is not None else next(sample_grasp(ctx, LIMB_NAME))[0]
        self.base = base

    def _in_furniture(self, limb_conf) -> bool:
        """Reject furniture penetration beyond the numerical contact tolerance."""
        from able.tasks.limb.planning.utils import (
            CONTACT_PENETRATION_TOLERANCE,
            pair_clearance,
            saved_sim_state,
        )

        ctx, sim = self.ctx, self.ctx.sim
        with saved_sim_state(sim):
            sim.limb.set_joints(list(limb_conf))
            return any(
                pair_clearance(sim, sim.limb.robot_id, body)
                < -CONTACT_PENETRATION_TOLERANCE
                for body in ctx.scene_collision_ids
            )

    def _base_pose_for(self, limb_init):
        """Nearest base pose that can hold the grasp at `limb_init`, or None."""
        from able.tasks.limb.planning.utils import (
            base_candidates,
            grasp_ik,
            release_grasp,
            saved_sim_state,
        )

        sim = self.ctx.sim
        with saved_sim_state(sim):
            release_grasp(sim)
            bases = [self.base] if self.base is not None else base_candidates(self.ctx, self.grasp, tuple(limb_init))
            for base_conf in bases:
                sim.robot.set_base(base_conf)
                if (
                    grasp_ik(
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
        """Whether one base pose can carry the limb init -> goal (cheaper than the full grid)."""
        from able.tasks.limb.planning.stream import (
            NUM_REACH_CHECKS,
        )
        from able.tasks.limb.planning.utils import (
            grasp_ik,
            limb_is_controllable,
            release_grasp,
            saved_sim_state,
        )

        if self._in_furniture(limb_init) or self._in_furniture(limb_goal):
            return False
        sim = self.ctx.sim
        waypoints = np.linspace(
            np.asarray(limb_init), np.asarray(limb_goal), NUM_REACH_CHECKS
        )
        with saved_sim_state(sim):
            release_grasp(sim)
            sim.robot.set_base(base_conf)
            seed = self.ctx.retract_joints
            for waypoint in waypoints:
                solution = grasp_ik(self.ctx, self.grasp, tuple(waypoint), seed)
                if solution is None or not limb_is_controllable(self.ctx):
                    return False
                seed = solution
        return True

    def _draw_limb_conf(
        self,
        rng: np.random.Generator,
        low: NDArray[np.float64],
        high: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """One limb configuration clear of the furniture; inactive joints keep the default."""
        for _ in range(self.max_conf_draws):
            conf = np.asarray(
                scene_config().scene.limb_init_joint_positions, np.float64
            )
            for j in active_joints():
                conf[j] = rng.uniform(low[j], high[j])
            if not self._in_furniture(conf):
                return conf
        raise RuntimeError(
            f"no limb configuration clear of the furniture in {self.max_conf_draws} draws"
        )

    def _self_colliding(self, robot_init) -> bool:
        """Whether the arm self-collides (the arm planner could never leave such a start)."""
        from pybullet_helpers.motion_planning import check_collisions_with_held_object

        from able.tasks.limb.planning.utils import extend_with_fingers, saved_sim_state

        sim = self.ctx.sim
        with saved_sim_state(sim):
            return check_collisions_with_held_object(
                sim.robot.arm, [], sim.physics_client_id, None, None,
                extend_with_fingers(list(robot_init)),
            )

    def _draw_robot_init(self, rng: np.random.Generator) -> NDArray[np.float64]:
        """Robot start jittered around the variant's."""
        nominal = np.asarray(scene_config().robot_initial_joints, np.float64)
        for _ in range(self.max_conf_draws):
            robot_init = nominal + rng.uniform(
                -self.box.robot_jitter,
                self.box.robot_jitter,
                size=NUM_ROBOT_JOINTS,
            )
            if not self._self_colliding(robot_init):
                return robot_init
        raise RuntimeError(
            f"no self-collision-free robot start in {self.max_conf_draws} draws"
        )

    def sample_init(
        self, rng: np.random.Generator, theta: NDArray[np.float64]
    ) -> LimbProblem:
        """A graspable start with no goal (`limb_goal` = init), e.g. for hindsight babbling."""
        from able.tasks.limb.task import LimbProblem

        low, high = self.box.inner_range(theta)
        for _ in range(self.max_init_draws):
            limb_init = self._draw_limb_conf(rng, low, high)
            base_conf = self._base_pose_for(limb_init)
            if base_conf is None:
                continue
            return LimbProblem(
                limb_init=tuple(limb_init),
                robot_init=tuple(self._draw_robot_init(rng)),
                limb_goal=tuple(limb_init),
                base_pose=base_conf,
            )
        raise RuntimeError(
            f"no graspable start found for theta={theta} in "
            f"{self.max_init_draws} draws"
        )

    def sample(
        self,
        rng: np.random.Generator,
        theta: NDArray[np.float64],
        true_theta: NDArray[np.float64] | None = None,
    ) -> LimbProblem:
        """One task that some base pose can carry out; see `TaskBox.sample`."""
        from able.tasks.limb.task import LimbProblem

        low, high = self.box.inner_range(theta)
        start_low, start_high = self.box.inner_range(
            theta if true_theta is None else true_theta
        )
        for _ in range(self.max_init_draws):
            limb_init = self._draw_limb_conf(rng, start_low, start_high)
            base_conf = self._base_pose_for(limb_init)
            if base_conf is None:
                continue
            for _ in range(self.max_goal_draws):
                limb_goal = self._draw_limb_conf(rng, low, high)
                separation = joint_position_distance(
                    LIMB_JOINT_INFOS, list(limb_init), list(limb_goal)
                )
                if not (
                    self.box.min_separation
                    <= separation
                    <= self.box.max_separation
                ):
                    continue
                if not self._reachable_from(base_conf, limb_init, limb_goal):
                    continue
                return LimbProblem(
                    limb_init=tuple(limb_init),
                    robot_init=tuple(self._draw_robot_init(rng)),
                    limb_goal=tuple(limb_goal),
                    base_pose=base_conf,
                )
        raise RuntimeError(
            f"no reachable task found for theta={theta} in "
            f"{self.max_init_draws} x {self.max_goal_draws} draws"
        )


@dataclass
class LimbDistribution(TaskDistribution):
    """Problems reachable from the fixed grasp and base (IK checks only)."""

    settings: PlannerSettings
    setup: FixedSetup
    _sim: Any = field(default=None, init=False, repr=False)
    _sampler: ReachableTaskSampler | None = field(default=None, init=False, repr=False)

    @property
    def sampler(self) -> ReachableTaskSampler:
        """Built on first use: it needs a simulator."""
        from able.tasks.limb.env.sim import make_sampling_context
        from able.tasks.limb.planning.run import reset_to_start
        from able.tasks.limb.planning.setup import wrist_grasp

        if self._sampler is None:
            self._sim, ctx = make_sampling_context(self.settings)
            reset_to_start(ctx)
            grasp = wrist_grasp(ctx, self.setup.offset, self.setup.roll)
            # One fixed base reaches few random tasks, so draw more (IK checks only).
            self._sampler = ReachableTaskSampler(ctx, max_goal_draws=60, max_init_draws=40,
                                                 grasp=grasp, base=self.setup.base_pose)
        return self._sampler

    def sample(self, rng: np.random.Generator) -> LimbProblem:
        """Goals inside the prior-mean person's range; narrower beliefs retarget them."""
        return self.sampler.sample(rng, np.full(3, DEFAULT_PRIOR.mean))
