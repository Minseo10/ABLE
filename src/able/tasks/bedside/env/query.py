"""Reach queries in their own sim (an empty copy of the scene): the robot holds its gripper at points
along a direction from the shoulder and the person answers whether they can reach each one."""

from __future__ import annotations

import time

import numpy as np
from kinder.envs.kinematic3d.utils import remove_fingers_from_extended_joints
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, SE2Pose
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    check_body_collisions,
    inverse_kinematics,
)

from able.tasks.bedside.env.bedside_stocking3d import (
    BedsideStocking3DEnvConfig,
    ObjectCentricBedsideStocking3DEnv,
    reach_margin,
)
from able.tasks.bedside.env.execute import Driver
from able.tasks.bedside.planning.stream import (
    CARRY_JOINTS,
    GRASP_ORIENTATIONS,
    HOME_JOINTS,
    _base_candidates,
)
from able.video import FrameRecorder

PRESENT_TOL = 0.01  # m, the gripper must end this close to the planned point


def present_conf(sim: ObjectCentricBedsideStocking3DEnv, point: NDArray) -> tuple[SE2Pose, list] | None:
    """A collision-free base pose and arm conf with the gripper at `point`, else None. Restores the robot."""
    # pylint: disable=protected-access
    arm, static, cid = sim.robot.arm, sim.static_collision_ids, sim.physics_client_id
    base0, joints0 = sim.robot.get_base(), list(arm.get_joint_positions())
    finger = arm.get_finger_state()
    found = None
    for base in _base_candidates(point[0], point[1]):
        sim.robot.set_base(base)
        if any(check_body_collisions(sim.robot.base.robot_id, b, cid) for b in static):
            continue
        for quat in GRASP_ORIENTATIONS:
            arm.set_joints(HOME_JOINTS)
            try:
                q = inverse_kinematics(arm, Pose(tuple(point), quat), validate=True, set_joints=True)
            except InverseKinematicsError:
                continue
            if not any(check_body_collisions(arm.robot_id, b, cid) for b in static):
                found = (base, list(q))
                break
        if found:
            break
    sim._set_robot_and_held_object(base0, remove_fingers_from_extended_joints(joints0), finger)
    return found


class BedsideQuerySimulator:
    """Owns the query sim (built lazily); never the planner's or executor's sim."""

    def __init__(self, config: BedsideStocking3DEnvConfig, scale: float, noisy: bool = True,
                 stop_at_miss: bool = True, frame_stride: int = 8, seed: int = 0) -> None:
        self.config, self.scale, self.noisy = config, scale, noisy  # noisy: Bernoulli(sigmoid(m / scale)), else m >= 0
        self.stop_at_miss, self.frame_stride, self.seed = stop_at_miss, frame_stride, seed
        self.last_seconds = 0.0
        self._sim: ObjectCentricBedsideStocking3DEnv | None = None

    @property
    def sim(self) -> ObjectCentricBedsideStocking3DEnv:
        if self._sim is None:
            self._sim = ObjectCentricBedsideStocking3DEnv(0, 0, config=self.config, allow_state_access=True)
            self._sim.reset(seed=self.seed)
        return self._sim

    def can_present(self, point: NDArray) -> bool:
        """Static collision-free IK with the gripper at `point`."""
        return present_conf(self.sim, point) is not None

    def ask(self, theta: NDArray[np.float64], direction: NDArray, distances, rng: np.random.Generator,
            frames: list | None = None) -> list[tuple[tuple[float, ...], bool]]:
        """[(gripper point, reached), ...] at the points along `direction` the robot could present."""
        started = time.time()
        sim = self.sim
        sim.reset(seed=self.seed)
        sim.reach_theta = np.array(theta, float)
        recorder = FrameRecorder(sim, frames, every=self.frame_stride)
        drive = Driver(sim, recorder, self.seed)
        recorder.snap()
        outcomes, base = [], None
        for s in distances:
            target = sim.shoulder + s * np.asarray(direction, float)
            conf = present_conf(sim, target)
            if conf is None:
                continue
            recorder.text = f"query: reach {s:.2f} m?" if recorder.on else ""
            if conf[0] != base:
                drive.arm(CARRY_JOINTS)
                drive.base(conf[0])
                base = conf[0]
            drive.arm(conf[1])
            point = np.array(sim.robot.arm.get_end_effector_pose().position)
            if np.linalg.norm(point - target) > PRESENT_TOL:
                continue  # blocked on the way
            m = float(reach_margin(point, sim.reach_theta, sim.shoulder))
            reached = bool(rng.random() < 1 / (1 + np.exp(-m / self.scale))) if self.noisy else m >= 0
            outcomes.append((tuple(float(v) for v in point), reached))
            if recorder.on:
                recorder.text = f"query: reach {s:.2f} m? {'yes' if reached else 'no'}"
                recorder.snap()
            if self.stop_at_miss and not reached:
                break
        self.last_seconds = time.time() - started
        return outcomes
