"""Executing a bedside plan in the sim (BiRRT-refined arm paths, dense base paths) and capturing frames."""

from __future__ import annotations

import numpy as np
from kinder.envs.kinematic3d.utils import (
    Kinematic3DRobotActionSpace,
    remove_fingers_from_extended_joints,
)
from pybullet_helpers.geometry import SE2Pose
from pybullet_helpers.inverse_kinematics import InverseKinematicsError
from pybullet_helpers.joint import get_jointwise_difference
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_motion_planning,
    smoothly_follow_end_effector_path,
)

from able.tasks.bedside.env.bedside_stocking3d import ObjectCentricBedsideStocking3DEnv
from able.tasks.bedside.planning.stream import CARRY_JOINTS


def _clip(sim, action):
    space = sim.action_space
    assert isinstance(space, Kinematic3DRobotActionSpace)
    return np.clip(np.asarray(action, dtype=np.float32), space.low, space.high)


def _arm_path(sim, target, exclude: set[int], seed: int) -> list:
    """BiRRT from the current arm conf to `target`; straight line if that fails."""
    # pylint: disable=protected-access
    arm = sim.robot.arm
    base, cur = sim.robot.get_base(), list(arm.get_joint_positions())
    if np.abs(get_jointwise_difference(arm.get_arm_joint_infos()[:7], list(target[:7]), cur[:7])).max() < 1e-6:
        return [cur]
    finger = arm.get_finger_state()
    held = sim._grasped_object_id
    bodies = sim._get_collision_object_ids() - exclude - ({held} if held is not None else set())
    path = run_motion_planning(
        arm,
        cur,
        list(target),
        bodies,
        seed=seed,
        physics_client_id=sim.physics_client_id,
        held_object=held,
        base_link_to_held_obj=sim._grasped_object_transform,
    )
    sim._set_robot_and_held_object(base, remove_fingers_from_extended_joints(cur), finger)
    if path is None:
        path = [cur, list(target)]
    return remap_joint_position_plan_to_constant_distance(
        path, arm, max_distance=sim.config.max_action_mag / 2
    )


def _descent(sim, hover, grasp, target: int, seed: int) -> list:
    """Straight-line end-effector path from the hover conf down to the grasp conf."""
    # pylint: disable=protected-access
    arm = sim.robot.arm
    base, cur = sim.robot.get_base(), list(arm.get_joint_positions())
    finger = arm.get_finger_state()
    held = sim._grasped_object_id
    bodies = sim._get_collision_object_ids() - {target} - ({held} if held is not None else set())
    waypoints = [arm.forward_kinematics(list(hover)), arm.forward_kinematics(list(grasp))]
    try:
        path = smoothly_follow_end_effector_path(
            arm,
            waypoints,
            initial_joints=list(hover),
            collision_ids=bodies,
            joint_distance_fn=create_joint_distance_fn(arm),
            seed=seed,
            include_start=False,
            held_object=held,
            base_link_to_held_obj=sim._grasped_object_transform,
        )
    except InverseKinematicsError:
        path = [list(grasp)]
    sim._set_robot_and_held_object(base, remove_fingers_from_extended_joints(cur), finger)
    return remap_joint_position_plan_to_constant_distance(
        [list(hover)] + list(path), arm, max_distance=0.02
    )[1:]


BASE_RESOLUTION = 0.05


def _base_path(sim, goal: SE2Pose) -> list[SE2Pose]:
    """Dense collision-free base waypoints to `goal`; tries both turn directions (a ~pi turn's is float noise)."""
    # pylint: disable=protected-access
    start = sim.robot.get_base()
    joints = remove_fingers_from_extended_joints(sim.robot.arm.get_joint_positions())
    finger = sim.robot.arm.get_finger_state()
    d = goal - start
    turns = [d.rot] if abs(d.rot) < 1e-6 else [d.rot, d.rot - np.sign(d.rot) * 2 * np.pi]
    shift = np.array([d.x, d.y, 0.0])
    candidates = []
    for t in turns:
        turn = np.array([0.0, 0.0, t])
        candidates += [[shift + turn], [turn, shift], [shift, turn]]
    paths = []
    for segments in candidates:
        path, at = [], np.zeros(3)
        for seg in segments:
            n = max(1, int(np.ceil(np.abs(seg).max() / BASE_RESOLUTION)))
            path += [at + seg * (i / n) for i in range(1, n + 1)]
            at = at + seg
        paths.append([SE2Pose(start.x + x, start.y + y, (start.rot + r + np.pi) % (2 * np.pi) - np.pi)
                      for x, y, r in path])
    found = paths[0]
    for path in paths:
        ok = True
        for q in path:
            sim._set_robot_and_held_object(q, joints, finger)
            if sim._robot_or_held_object_collision_exists():
                ok = False
                break
        if ok:
            found = path
            break
    sim._set_robot_and_held_object(start, joints, finger)
    return found


def _caption(plan, i: int) -> str:
    """Label for action i: the next pick/place."""
    for name, args in plan[i:]:
        if name == "pick":
            return f"pick {args[0]}"
        if name == "place":
            return f"place {args[0]} on {args[1].surface}"
    return ""


def render_frame(sim: ObjectCentricBedsideStocking3DEnv, caption: str = "") -> np.ndarray:
    """RGB uint8 (H, W, 3) from the config camera, optionally captioned."""
    img = np.ascontiguousarray(np.asarray(sim.render(), dtype=np.uint8)[..., :3])
    if caption:
        from PIL import Image, ImageDraw

        im = Image.fromarray(img)
        ImageDraw.Draw(im).text((10, 10), caption, fill=(0, 0, 0))
        img = np.asarray(im)
    return img


def execute_plan(sim: ObjectCentricBedsideStocking3DEnv, plan, seed: int = 0,
                 frames: list | None = None, stride: int = 8) -> bool:
    """Execute a move_base/pick/place plan; arm paths are refined with BiRRT.

    If `frames` is a list, append a rendered frame every `stride` sim steps and after each gripper action."""
    caption, steps = "", 0

    def capture() -> None:
        if frames is not None:
            frames.append(render_frame(sim, caption))

    def step(action) -> None:
        nonlocal steps
        sim.step(action)
        steps += 1
        if frames is not None and steps % stride == 0:
            capture()

    def to_joints(target, tol=1e-3, attempts=40):
        for _ in range(attempts):
            cur = remove_fingers_from_extended_joints(sim.robot.arm.get_joint_positions())
            d = get_jointwise_difference(
                sim.robot.arm.get_arm_joint_infos()[:7], list(target[:7]), cur
            )
            step(_clip(sim, [0.0, 0.0, 0.0] + d + [0.0]))
            if max(abs(x) for x in d) < tol:
                return

    def to_base(target, attempts=200):
        for q in _base_path(sim, target):
            d = q - sim.robot.get_base()
            step(_clip(sim, [d.x, d.y, d.rot] + [0.0] * 8))
        for _ in range(attempts):
            d = target - sim.robot.get_base()
            if max(abs(d.x), abs(d.y), abs(d.rot)) < 1e-3:
                return
            step(_clip(sim, [d.x, d.y, d.rot] + [0.0] * 8))

    def gripper(close: bool):
        for _ in range(5):
            step(_clip(sim, [0.0] * 10 + [-1.0 if close else 1.0]))
            held = sim._grasped_object  # pylint: disable=protected-access
            if (held is not None) == close:
                return

    capture()
    for q in _arm_path(sim, CARRY_JOINTS, set(), seed)[1:]:
        to_joints(q)
    for i, (name, args) in enumerate(plan):
        caption = _caption(plan, i) if frames is not None else ""
        if name == "move_base":
            to_base(args[1].pose)
        elif name in ("pick", "place"):
            traj = args[4]
            target = sim._object_name_to_pybullet_id(args[0])  # pylint: disable=protected-access
            _, hover, grasp = traj.joint_plan
            approach = _arm_path(sim, hover, {target}, seed)
            for q in approach[1:] + _descent(sim, hover, grasp, target, seed):
                to_joints(q)
            gripper(close=name == "pick")
            capture()
            to_joints(hover)
            retreat = _arm_path(sim, traj.joint_plan[0], set(), seed)
            for q in retreat[1:]:
                to_joints(q)
        else:
            raise ValueError(f"Unknown action: {name}")
    capture()
    return sim.goal_reached()
