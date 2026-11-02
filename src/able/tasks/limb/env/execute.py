"""Executing a limb plan on the simulator, with a per-step `RolloutLog` and optional RGB frames."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pybullet_helpers.joint import JointPositions, JointVelocities

from able.tasks.limb.planning.stream import (
    ArmTrajectory,
    LimbStreamContext,
    TorqueTrajectory,
)
from able.tasks.limb.planning.utils import (
    HumanTorques,
    JointTorques,
    _limb_state,
    advance,
    arm_in_collision,
    base_in_collision,
    engage_grasp,
    exceeds_human_torque_limits,
    exceeds_robot_torque_limits,
    hold_torque,
    human_in_collision,
    limb_out_of_limits,
    release_grasp,
    robot_motion_violation,
)

__all__ = ["RolloutLog", "advance_logged", "execute_action", "execute_plan"]


@dataclass
class RolloutLog:
    """Per-step `move_limb` measurements, recorded *after* each simulator step (as `dense_rows` expects)."""

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


def execute_action(
    ctx: LimbStreamContext, name: str, args: tuple, log: RolloutLog | None = None,
    frames: list | None = None, every: int = 15,
) -> bool:
    """Execute one plan action; False if it broke a constraint. Only `move_limb` steps are logged."""
    sim = ctx.sim
    if name == "move_base":
        _, base_plan, _ = args
        release_grasp(sim)
        for target_base in base_plan[1:]:
            sim.robot.set_base(target_base)
            _snap(sim, frames)
            if ctx.check_base_collisions and base_in_collision(ctx):
                return False
    elif name == "grasp":
        _, _, base_conf, arm_trajectory, _ = args
        assert isinstance(arm_trajectory, ArmTrajectory)
        sim.robot.set_base(base_conf)
        for waypoint in arm_trajectory.joint_plan:
            sim.robot.arm.set_joints(list(waypoint))
            _snap(sim, frames)
            if arm_in_collision(ctx) or human_in_collision(ctx):
                return False
        engage_grasp(sim)
    elif name == "move_limb":
        _, _, _, _, _, _, torque_trajectory, _ = args
        assert isinstance(torque_trajectory, TorqueTrajectory)
        for i, torque in enumerate(torque_trajectory.robot_torques):
            if i % every == 0:
                _snap(sim, frames)
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
    return True


def execute_plan(
    ctx: LimbStreamContext,
    plan: list[tuple[str, tuple]],
    log: RolloutLog | None = None,
    frames: list | None = None,
) -> bool:
    """Execute `plan`, with an optional per-step log and RGB frames."""
    try:
        for name, args in plan:
            if not execute_action(ctx, name, args, log, frames):
                return False
        return ctx.sim.goal_reached()
    finally:
        _snap(ctx.sim, frames)


def _snap(sim, frames: list | None) -> None:
    if frames is not None:
        frames.append(np.asarray(sim.render(), dtype=np.uint8)[..., :3])

