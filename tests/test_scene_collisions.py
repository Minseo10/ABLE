"""Regression checks for the penetrations observed in the variant-sweep GIFs."""

import numpy as np
import pybullet as p
import pytest
from kinder.envs.dynamic3d.limb_scenes import create_scene
from kinder.envs.dynamic3d.limbrepositioning3d import create_variant_config
from kinder_pddlstream_planning.limbrepositioning3d.run import (
    build_stream_context,
    create_env,
    reset_to_start,
    variant_kwargs,
)
from kinder_pddlstream_planning.limbrepositioning3d.utils import (
    CONTACT_PENETRATION_TOLERANCE,
    base_in_collision,
    grasp_ik,
    human_in_collision,
    pair_clearance,
    release_grasp,
)
from pybullet_helpers.geometry import SE2Pose

pytestmark = pytest.mark.slow
FURNISHED = [
    f"{scene}-{limb}"
    for scene in ("bed", "wheelchair")
    for limb in ("left-arm", "right-arm", "left-leg", "right-leg")
]


@pytest.mark.parametrize("variant", FURNISHED)
def test_furnished_postures_are_clear_and_resting_limbs_do_not_drift(variant):
    """MPC's scratch physics must not change limbs absent from its state snapshot."""
    client = p.connect(p.DIRECT)
    try:
        config = create_variant_config(variant)
        scene = create_scene(config.scene, client)
        resting = {
            name: limb
            for name, limb in scene.limbs.items()
            if limb is not scene.passive_limb
        }
        before = {name: limb.get_joint_positions() for name, limb in resting.items()}
        for limb in scene.limbs.values():
            points = p.getClosestPoints(
                limb.robot_id, scene.furniture_id, 0.5, physicsClientId=client
            )
            assert min(point[8] for point in points) >= 0.0
        scene.passive_limb.set_joints(list(config.scene.limb_goal_joint_positions))
        points = p.getClosestPoints(
            scene.passive_limb.robot_id, scene.furniture_id, 0.5, physicsClientId=client
        )
        assert min(point[8] for point in points) >= 0.0
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=client)
        for _ in range(3000):
            p.stepSimulation(physicsClientId=client)
        for name, limb in resting.items():
            np.testing.assert_array_equal(limb.get_joint_positions(), before[name])
            np.testing.assert_array_equal(limb.get_joint_velocities(), np.zeros(6))
    finally:
        p.disconnect(physicsClientId=client)


@pytest.mark.parametrize("variant", ["bed-left-arm", "bed-right-arm"])
def test_released_arm_rests_on_mattress_instead_of_falling_through(variant):
    sim = create_env(variant)
    try:
        ctx = build_stream_context(sim)
        reset_to_start(ctx)
        release_grasp(sim)
        worst = 0.0
        for _ in range(1000):
            p.stepSimulation(physicsClientId=sim.physics_client_id)
            gap = pair_clearance(sim, sim.limb.robot_id, sim.scene.furniture_id)
            worst = min(worst, gap)
        assert worst >= -CONTACT_PENETRATION_TOLERANCE
        assert abs(gap) < 0.0001
    finally:
        sim.close()


def test_isolated_leg_cannot_be_placed_inside_mobile_base():
    """An empty scene must still check the manipulated limb against the base."""
    sim = create_env("isolated-left-leg")
    try:
        ctx = build_stream_context(sim)
        reset_to_start(ctx)
        assert not human_in_collision(ctx)
        release_grasp(sim)
        sim.limb.set_joints(list(sim.config.scene.limb_goal_joint_positions))
        # Put the base's solid enclosure around the leg's grasp point.
        x, y, z = sim.limb.get_end_effector_pose().position
        sim.robot.base.z = z - 0.15
        sim.robot.set_base(SE2Pose(x, y, 0.0))
        assert pair_clearance(sim, sim.limb.robot_id, sim.robot.base.robot_id) < -0.01
        assert base_in_collision(ctx)
        assert human_in_collision(ctx)
        assert (
            grasp_ik(
                ctx,
                sim.scene.grasp_transform,
                sim.config.scene.limb_goal_joint_positions,
                ctx.retract_joints,
            )
            is None
        )
    finally:
        sim.close()


def test_furniture_overlap_cannot_be_whitelisted_by_refreshing_baseline():
    sim = create_env("bed-right-arm")
    try:
        ctx = build_stream_context(sim)
        reset_to_start(ctx)
        release_grasp(sim)
        # Shoulder extension swings the arm downward through the mattress.
        q = list(sim.config.scene.limb_init_joint_positions)
        q[0] = 0.5
        sim.limb.set_joints(q)
        assert pair_clearance(sim, sim.limb.robot_id, sim.scene.furniture_id) < -0.01
        ctx.refresh_collision_baseline()
        assert human_in_collision(ctx)
        assert sim.limb_penetration() < -0.01
    finally:
        sim.close()


@pytest.mark.parametrize("variant", FURNISHED)
def test_variant_sweep_keeps_base_collision_checks_enabled(variant):
    assert variant_kwargs(variant, check_base_collisions=True)["check_base_collisions"]
