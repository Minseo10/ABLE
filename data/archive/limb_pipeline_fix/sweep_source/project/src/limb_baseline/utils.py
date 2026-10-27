"""Planner helpers kinder-baselines `main` renamed or dropped, under the names used here.

Adds the per-step `RolloutLog`, an `execute_plan` that fills it, and thin wrappers over
the renamed base-pose and IK helpers, so the planner repo stays untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from kinder_pddlstream_planning.limbrepositioning3d.stream import (
    ArmTrajectory,
    LimbStreamContext,
    TorqueTrajectory,
)
from kinder_pddlstream_planning.limbrepositioning3d.utils import (
    HumanTorques,
    JointTorques,
    _limb_state,
    advance,
    arm_in_collision,
    base_in_collision,
    engage_grasp,
    exceeds_human_torque_limits,
    exceeds_robot_torque_limits,
    grasp_ik,
    hold_torque,
    human_in_collision,
    limb_out_of_limits,
    release_grasp,
    robot_motion_violation,
    saved_sim_state,
)
from kinder_pddlstream_planning.limbrepositioning3d.utils import (
    base_candidates as _base_candidates_for,
)
from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions, JointVelocities

__all__ = [
    "RolloutLog",
    "advance_logged",
    "base_candidates",
    "execute_plan",
    "grasp_ik",
    "saved_sim_state",
]


@dataclass
class RolloutLog:
    """Per-step measurements of an executed `move_limb`, filled by `advance_logged`.

    Records the state *after* each simulator step, which is what `dense_rows` expects.
    """

    correction_torques: list[JointTorques] = field(default_factory=list)
    commanded_torques: list[JointTorques] = field(default_factory=list)
    limb_positions: list[JointPositions] = field(default_factory=list)
    limb_velocities: list[JointVelocities] = field(default_factory=list)
    robot_positions: list[JointPositions] = field(default_factory=list)
    robot_velocities: list[JointVelocities] = field(default_factory=list)
    human_total: list[JointTorques] = field(default_factory=list)
    human_robot_induced: list[JointTorques] = field(default_factory=list)


def advance_logged(sim, correction, log: RolloutLog) -> HumanTorques:
    """`advance_corrected`, recording what the step did into `log`."""
    correction = np.asarray(correction, dtype=np.float64)
    commanded = correction + hold_torque(sim)
    human = advance(sim, commanded)
    positions, velocities = _limb_state(sim)
    log.correction_torques.append(list(correction))
    log.commanded_torques.append(list(commanded))
    log.limb_positions.append(list(positions))
    log.limb_velocities.append(list(velocities))
    log.robot_positions.append(list(sim.robot.arm.get_joint_positions()))
    log.robot_velocities.append(list(sim.robot.arm.get_joint_velocities()))
    log.human_total.append(list(human.total))
    log.human_robot_induced.append(list(human.robot))
    return human


def execute_plan(
    ctx: LimbStreamContext,
    plan: list[tuple[str, tuple]],
    log: RolloutLog | None = None,
) -> bool:
    """`run.execute_plan`, with the optional per-step log `main` no longer takes.

    Same action handling as the planner's own; only the `move_limb` steps are logged, so
    `log` indexes torque steps one to one.
    """
    sim = ctx.sim
    for name, args in plan:
        if name == "move_base":
            _, base_plan, _ = args
            release_grasp(sim)
            for target_base in base_plan[1:]:
                sim.robot.set_base(target_base)
                if ctx.check_base_collisions and base_in_collision(ctx):
                    return False
        elif name == "grasp":
            _, _, base_conf, arm_trajectory, _ = args
            assert isinstance(arm_trajectory, ArmTrajectory)
            sim.robot.set_base(base_conf)
            for waypoint in arm_trajectory.joint_plan:
                sim.robot.arm.set_joints(list(waypoint))
                if arm_in_collision(ctx) or human_in_collision(ctx):
                    return False
            engage_grasp(sim)
        elif name == "move_limb":
            _, _, _, _, _, _, torque_trajectory, _ = args
            assert isinstance(torque_trajectory, TorqueTrajectory)
            for torque in torque_trajectory.robot_torques:
                commanded = np.asarray(torque, dtype=np.float64) + hold_torque(sim)
                if exceeds_robot_torque_limits(ctx, commanded):
                    return False
                if log is None:
                    human = advance(sim, commanded)
                else:
                    human = advance_logged(sim, torque, log)
                if exceeds_human_torque_limits(ctx, human):
                    return False
                if robot_motion_violation(ctx) or limb_out_of_limits(ctx):
                    return False
                if arm_in_collision(ctx) or human_in_collision(ctx):
                    return False
        else:
            raise ValueError(f"Unknown action: {name}")
    return sim.goal_reached()


def base_candidates(
    ctx: LimbStreamContext, grasp: Pose, limb_positions: JointPositions
) -> Iterator:
    """Base poses from which the arm can hold `grasp` at `limb_positions`, nearest first.

    The old helper swept a grid around the variant's placement and took no grasp; `main`
    samples placements around the grasp target instead and already screens each with an
    IK solve, so callers keep only their own collision and controllability checks. It is
    a generator: each candidate costs an IK solve, so take as many as needed and stop.
    """
    yield from _base_candidates_for(ctx, grasp, list(limb_positions))
