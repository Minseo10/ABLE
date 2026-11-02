"""Build the limb environment and the stream context the MPC runs in."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from kinder.envs.dynamic3d.limbrepositioning3d import (
    LimbRepositioning3DEnvConfig,
    ObjectCentricLimbRepositioning3DEnv,
    create_variant_config,
)
from kinder.envs.dynamic3d.limbs import DEFAULT_RANGE_OF_MOTION, MOTION_NAMES, BodyMass
from pybullet_helpers.geometry import SE2Pose

from able.tasks.limb.planning.stream import (
    LimbStreamContext,
    MPCConfig,
)
from able.tasks.limb.planning.utils import (
    DEFAULT_GRAVITY,
    DEFAULT_LIMB_JOINT_DAMPING,
    ROBOT_TORQUE_LIMITS,
    apply_limb_joint_damping,
    engage_grasp,
    extend_with_fingers,
    is_grasping,
    release_grasp,
)

# How far behind its grasp placement the base starts, in meters.
DEFAULT_START_STANDOFF = 1.0


LIMB_NAME = "limb"


# Muscle tone model: "none" (limp) or "spring" (spring-damper).
DEFAULT_MUSCLE_TONE = "spring"


# Joint limits model: "none" or "box".
DEFAULT_JOINT_LIMITS_MODEL = "box"


# Empty: unsafe placements are rejected, never exempted from collision checks.
_VARIANT_OVERRIDES: dict[str, dict[str, Any]] = {}


def variant_kwargs(variant: str, **kwargs: Any) -> dict[str, Any]:
    """Overlay `variant`'s overrides (which win) onto the sweep's shared settings."""
    return {**kwargs, **_VARIANT_OVERRIDES.get(variant, {})}


def start_base_pose(
    grasp_base_pose: SE2Pose, standoff: float = DEFAULT_START_STANDOFF
) -> SE2Pose:
    """A start pose `standoff` meters behind `grasp_base_pose`, same heading."""
    return SE2Pose(
        grasp_base_pose.x - standoff * np.cos(grasp_base_pose.rot),
        grasp_base_pose.y - standoff * np.sin(grasp_base_pose.rot),
        grasp_base_pose.rot,
    )


def create_env(
    variant: str,
    standoff: float = DEFAULT_START_STANDOFF,
    use_gui: bool = False,
    config: LimbRepositioning3DEnvConfig | None = None,
    robot_base_z: float | None = None,
    gravity: tuple[float, float, float] = DEFAULT_GRAVITY,
    muscle_tone: str = DEFAULT_MUSCLE_TONE,
    joint_limits_model: str = DEFAULT_JOINT_LIMITS_MODEL,
    range_of_motion_scale: float = 1.0,
    goal_scale: float = 1.0,
    body_mass: BodyMass | None = None,
    robot_torque_limits: tuple[float, ...] = ROBOT_TORQUE_LIMITS,
    limb_joint_damping: float = DEFAULT_LIMB_JOINT_DAMPING,
) -> ObjectCentricLimbRepositioning3DEnv:
    """Build the env with its base backed off to the start pose (variant's = goal).

    Gravity is on, so the shipped +-1 N*m action space is widened to Kinova limits.
    """
    if config is None:
        config = create_variant_config(variant)
    if robot_base_z is None:
        robot_base_z = config.robot_base_z
    goal = config.scene.limb_goal_joint_positions
    if goal_scale != 1.0:
        init = config.scene.limb_init_joint_positions
        goal = tuple(i + goal_scale * (g - i) for i, g in zip(init, goal))
    range_of_motion = config.scene.limb_range_of_motion
    if range_of_motion_scale != 1.0:
        range_of_motion = DEFAULT_RANGE_OF_MOTION.scaled(
            {name: range_of_motion_scale for name in MOTION_NAMES}
        )
    config = dataclasses.replace(
        config,
        robot_base_home_pose=start_base_pose(config.robot_base_home_pose, standoff),
        robot_base_z=robot_base_z,
        gravity=gravity,
        torque_lower_limits=tuple(-t for t in robot_torque_limits),
        torque_upper_limits=tuple(robot_torque_limits),
        scene=dataclasses.replace(
            config.scene,
            limb_muscle_tone_model_name=muscle_tone,
            limb_joint_limits_model_name=joint_limits_model,
            limb_range_of_motion=range_of_motion,
            limb_goal_joint_positions=goal,
            limb_body_mass=body_mass or config.scene.limb_body_mass,
        ),
    )
    sim = ObjectCentricLimbRepositioning3DEnv(
        variant=variant, config=config, use_gui=use_gui
    )
    apply_limb_joint_damping(sim, damping=limb_joint_damping)
    # RRT must have finite sampling bounds when a straight drive is obstructed.
    home = sim.robot.get_base()
    sim.robot.base.pose_lower_bound = SE2Pose(home.x - 3.0, home.y - 3.0, -np.pi)
    sim.robot.base.pose_upper_bound = SE2Pose(home.x + 3.0, home.y + 3.0, np.pi)
    return sim


def build_stream_context(
    sim: ObjectCentricLimbRepositioning3DEnv,
    standoff: float = DEFAULT_START_STANDOFF,
    motion_seed: int = 0,
    mpc: MPCConfig | None = None,
    check_base_collisions: bool = True,
    check_robot_collisions: bool = True,
    filter_saturated_bases: bool = True,
    human_torque_limit: float | None = None,
    robot_induced_torque_limit: float | None = None,
) -> LimbStreamContext:
    """Precompute the shared stream context; split out so tests can drive streams."""
    grasp_base_pose = create_variant_config(sim.variant).robot_base_home_pose
    return LimbStreamContext(
        sim=sim,
        start_base_pose=start_base_pose(grasp_base_pose, standoff),
        grasp_base_pose=grasp_base_pose,
        retract_joints=extend_with_fingers(list(sim.config.robot_initial_joints)),
        limb_name=LIMB_NAME,
        motion_seed=motion_seed,
        mpc=mpc or MPCConfig(),
        check_base_collisions=check_base_collisions,
        check_robot_collisions=check_robot_collisions,
        filter_saturated_bases=filter_saturated_bases,
        human_torque_limit=human_torque_limit,
        robot_induced_torque_limit=robot_induced_torque_limit,
    )


def reset_to_start(ctx: LimbStreamContext) -> None:
    """Restore the plan's start state; `reset()` restores neither base nor weld."""
    sim = ctx.sim
    if not is_grasping(sim):
        engage_grasp(sim)
    sim.reset()
    release_grasp(sim)
    sim.robot.set_base(ctx.start_base_pose)
    sim.robot.arm.set_joints(
        list(ctx.retract_joints), joint_velocities=[0.0] * len(ctx.retract_joints)
    )
    scene = sim.config.scene
    sim.limb.set_joints(
        list(scene.limb_init_joint_positions),
        joint_velocities=[0.0] * len(scene.limb_init_joint_positions),
    )
    ctx.refresh_collision_baseline()
