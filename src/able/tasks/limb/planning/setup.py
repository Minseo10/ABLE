"""One fixed grasp (around the wrist/ankle) and base pose per variant, chosen by scripts/limb/select_setup.py."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pybullet as p
from kinder.envs.dynamic3d.limb_utils import NUM_LIMB_JOINTS
from pybullet_helpers.geometry import Pose, SE2Pose, matrix_from_quat, multiply_poses
from pybullet_helpers.motion_planning import run_motion_planning

from able.tasks.limb.planning.stream import ArmTrajectory, LimbStreamContext
from able.tasks.limb.planning.utils import (
    CoupledState,
    base_in_collision,
    grasp_ik,
    release_grasp,
    saved_sim_state,
)

SETUP_PATH = Path(__file__).with_name("fixed_setup.json")
# Grasp centres up the forearm/shank from the wrist/ankle joint (m), and gripper rolls.
GRASP_OFFSETS = (0.0, 0.03, 0.06)
GRASP_ROLLS = (0.0, np.pi)


@dataclass(frozen=True)
class FixedSetup:
    variant: str
    offset: float  # m above the wrist/ankle
    roll: float
    base: tuple[float, float, float]  # x, y, rotation
    stats: dict = field(default_factory=dict)  # selection scores

    @property
    def base_pose(self) -> SE2Pose:
        return SE2Pose(*self.base)


def _link_position(sim, name: str) -> np.ndarray:
    limb, client = sim.limb, sim.physics_client_id
    index = next(j for j in range(p.getNumJoints(limb.robot_id, physicsClientId=client))
                 if p.getJointInfo(limb.robot_id, j, physicsClientId=client)[12].decode() == name)
    return np.asarray(p.getLinkState(limb.robot_id, index, physicsClientId=client)[4])


def wrist_grasp(ctx: LimbStreamContext, offset: float, roll: float) -> Pose:
    """The scene's grasp moved to `offset` m up the forearm/shank from the wrist/ankle."""
    sim = ctx.sim
    arm = "arm" in sim.limb.get_name()
    wrist = _link_position(sim, "hand_x" if arm else "foot_x")
    elbow = _link_position(sim, "lower_arm" if arm else "lower_leg")
    target = wrist + offset * (elbow - wrist) / np.linalg.norm(elbow - wrist)
    ee = sim.limb.get_end_effector_pose()
    local = matrix_from_quat(ee.orientation).T @ (target - np.asarray(ee.position))
    spin = Pose.from_rpy((0.0, 0.0, 0.0), (0.0, 0.0, roll))
    return multiply_poses(Pose(tuple(local.tolist())), sim.scene.grasp_transform, spin)


def approach(ctx: LimbStreamContext, grasp: Pose, base: SE2Pose, limb_init):
    """(arm path onto the grasp, grasped state) from `base`, or None."""
    sim = ctx.sim
    with saved_sim_state(sim):
        release_grasp(sim)
        sim.robot.set_base(base)
        if ctx.check_base_collisions and base_in_collision(ctx):
            return None
        joints = grasp_ik(ctx, grasp, tuple(limb_init), ctx.retract_joints)
        if joints is None:
            return None
        path = run_motion_planning(sim.robot.arm, initial_positions=list(ctx.retract_joints),
                                   target_positions=joints, collision_bodies=ctx.obstacle_ids,
                                   seed=ctx.motion_seed, physics_client_id=sim.physics_client_id)
    if path is None:
        return None
    trajectory = ArmTrajectory(list(path))
    state = CoupledState(base_pose=base, robot_positions=list(path[-1]),
                         robot_velocities=[0.0] * len(path[-1]), limb_positions=list(limb_init),
                         limb_velocities=[0.0] * NUM_LIMB_JOINTS, approach=trajectory)
    return trajectory, state


def load_setups(path: Path = SETUP_PATH) -> dict[str, FixedSetup]:
    if not path.exists():
        return {}
    return {k: FixedSetup(**{**v, "base": tuple(v["base"])}) for k, v in json.loads(path.read_text()).items()}


def load_setup(variant: str, path: Path = SETUP_PATH) -> FixedSetup:
    setups = load_setups(path)
    if variant not in setups:
        raise KeyError(f"no fixed setup for {variant}; run scripts/limb/select_setup.py")
    return setups[variant]


def save_setup(setup: FixedSetup, path: Path = SETUP_PATH) -> None:
    """Add or replace one variant's entry."""
    data = json.loads(path.read_text()) if path.exists() else {}
    data[setup.variant] = asdict(setup)
    path.write_text(json.dumps(dict(sorted(data.items())), indent=1) + "\n")
