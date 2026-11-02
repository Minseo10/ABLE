"""The simulated person: they push one (ungrasped) joint toward a target; where they stop is the answer.

Effort is constant, not PD (PD would always reach the target), and below `stiffness * zone_width`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pybullet as p
from kinder.envs.dynamic3d.limb_utils import NUM_LIMB_JOINTS, PYBULLET_TIMESTEP
from kinder.envs.dynamic3d.limbs import create_human_limb
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose

from able.tasks.limb.env.human import (
    active_joints,
    active_signs,
    limb_name,
    theta_to_bounds,
)


@dataclass(frozen=True)
class QueryModel:
    """End-of-range resistance and push effort; placeholder constants, not fitted."""

    zone_width: float = 0.3
    stiffness: float = 40.0
    wall_stiffness: float = 400.0
    effort_mean: float = 4.0
    effort_stddev: float = 2.0

    # Overdamps the end-range spring; below about 8 the joint overshoots the bound.
    joint_damping: float = 15.0
    # Only bites if the solver blows up; a passive push tops out near effort/damping.
    max_velocity: float = 2.0
    moving_velocity: float = 1e-2
    settle_velocity: float = 1e-3
    settle_steps: int = 100
    # Must cover the slowest push that still counts as one.
    max_steps: int = 150000

    def __post_init__(self) -> None:
        if self.effort_mean >= self.stiffness * self.zone_width:
            raise ValueError(
                "effort_mean must stay below stiffness * zone_width, so the person "
                "stops short of their passive bound"
            )

    @property
    def short_fall(self) -> float:
        """Mean distance short of the passive bound where the person stops."""
        return self.zone_width - self.effort_mean / self.stiffness

    @property
    def tau(self) -> float:
        """Likelihood noise (rad): effort stddev over stiffness."""
        return self.effort_stddev / self.stiffness

    def draw_effort(self, rng: np.random.Generator) -> float:
        """One trial's effort, redrawn until in `(0, stiffness * zone_width)`."""
        ceiling = self.stiffness * self.zone_width
        while True:
            effort = float(rng.normal(self.effort_mean, self.effort_stddev))
            if 0.0 < effort < ceiling:
                return effort

    def restoring_torque(self, reach: float, bound: float) -> float:
        """End-of-range torque, in coordinates where the person pushes upward."""
        if reach <= bound - self.zone_width:
            return 0.0
        if reach < bound:
            return -self.stiffness * (reach - (bound - self.zone_width))
        return -self.stiffness * self.zone_width - self.wall_stiffness * (reach - bound)


DEFAULT_QUERY_MODEL = QueryModel()


@dataclass(frozen=True)
class QueryOutcome:
    """Where the person stopped, and whether that was their limit or the target."""

    motion_index: int
    target: float
    answer: float
    reached_target: bool
    effort: float
    settled: bool = True
    peak_velocity: float = 0.0


class QuerySimulator:
    """Standalone query limb, kept alive across a person's rounds."""

    def __init__(
        self, theta: NDArray[np.float64], model: QueryModel = DEFAULT_QUERY_MODEL
    ) -> None:
        self.theta = np.asarray(theta, dtype=np.float64)
        self.model = model
        self.bounds = theta_to_bounds(self.theta)
        self.physics_client_id = p.connect(p.DIRECT)
        p.setPhysicsEngineParameter(
            fixedTimeStep=PYBULLET_TIMESTEP, physicsClientId=self.physics_client_id
        )
        p.setGravity(0.0, 0.0, 0.0, physicsClientId=self.physics_client_id)
        self.limb = create_human_limb(
            limb_name(),
            Pose.identity(),
            [0.0] * NUM_LIMB_JOINTS,
            "none",
            "none",
            self.physics_client_id,
        )
        self._prepare()

    def _prepare(self) -> None:
        """Free the joints of their default motors so applied torques act directly."""
        joints = self.limb.arm_joints
        p.setJointMotorControlArray(
            self.limb.robot_id,
            joints,
            p.VELOCITY_CONTROL,
            forces=np.zeros(len(joints)),
            physicsClientId=self.physics_client_id,
        )
        for joint in joints:
            # One combined call silently drops the damping.
            p.changeDynamics(
                self.limb.robot_id,
                joint,
                maxJointVelocity=self.model.max_velocity,
                physicsClientId=self.physics_client_id,
            )
            p.changeDynamics(
                self.limb.robot_id,
                joint,
                jointDamping=self.model.joint_damping,
                physicsClientId=self.physics_client_id,
            )

    def ask(
        self,
        motion_index: int,
        target: float,
        rng: np.random.Generator,
        on_step: Callable[[int, float], None] | None = None,
    ) -> QueryOutcome:
        """Push one joint toward `target` (a magnitude); `on_step(step, reach)` is for recording."""
        model = self.model
        joint = self.limb.arm_joints[active_joints()[motion_index]]
        sign = active_signs()[motion_index]
        bound = abs(self.bounds[motion_index])
        effort = model.draw_effort(rng)

        self.limb.set_joints(
            [0.0] * NUM_LIMB_JOINTS, joint_velocities=[0.0] * NUM_LIMB_JOINTS
        )
        held = [j for j in self.limb.arm_joints if j != joint]
        p.setJointMotorControlArray(
            self.limb.robot_id,
            held,
            p.VELOCITY_CONTROL,
            targetVelocities=[0.0] * len(held),
            forces=[1e4] * len(held),
            physicsClientId=self.physics_client_id,
        )
        # A joint held during an earlier query keeps that motor until it is released.
        p.setJointMotorControl2(
            self.limb.robot_id,
            joint,
            p.VELOCITY_CONTROL,
            force=0.0,
            physicsClientId=self.physics_client_id,
        )

        # Settled only counts once it has started moving.
        settled = 0
        peak = 0.0
        moving = False
        reached_target = False
        for _step in range(model.max_steps):
            state = p.getJointState(
                self.limb.robot_id, joint, physicsClientId=self.physics_client_id
            )
            reach, velocity = sign * state[0], sign * state[1]
            if reach >= target:
                reached_target = True
                break
            peak = max(peak, abs(velocity))
            moving = moving or abs(velocity) > model.moving_velocity
            torque = effort + model.restoring_torque(reach, bound)
            p.setJointMotorControl2(
                self.limb.robot_id,
                joint,
                p.TORQUE_CONTROL,
                force=sign * torque,
                physicsClientId=self.physics_client_id,
            )
            p.stepSimulation(physicsClientId=self.physics_client_id)
            # Pinned, not motored: a motor cannot hold these at the gimbal singularity.
            for other in held:
                p.resetJointState(
                    self.limb.robot_id,
                    other,
                    0.0,
                    0.0,
                    physicsClientId=self.physics_client_id,
                )
            if on_step is not None:
                on_step(_step, reach)
            if moving and abs(velocity) < model.settle_velocity:
                settled += 1
            else:
                settled = 0
            if settled >= model.settle_steps:
                break

        final = p.getJointState(
            self.limb.robot_id, joint, physicsClientId=self.physics_client_id
        )[0]
        return QueryOutcome(
            motion_index=motion_index,
            target=target,
            answer=abs(final),
            reached_target=reached_target,
            effort=effort,
            settled=bool(moving and settled >= model.settle_steps),
            peak_velocity=peak,
        )

    def close(self) -> None:
        if p.isConnected(physicsClientId=self.physics_client_id):
            p.disconnect(physicsClientId=self.physics_client_id)

