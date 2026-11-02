"""Physical reach queries: in its own sim, the robot holds its gripper at points along a direction from the
shoulder and the person answers whether they can reach each one."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from kinder.envs.kinematic3d.utils import remove_fingers_from_extended_joints
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, SE2Pose
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    check_body_collisions,
    inverse_kinematics,
)

from able.api import Belief, InfoGatherer, Observation
from able.tasks.bedside.belief import EXECUTION_SCALE, reach_distance
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
    NOMINAL_HALF_HEIGHT,
    _base_candidates,
)
from able.video import FrameRecorder

PRESENT_TOL = 0.01  # m, the gripper must end this close to the planned point


def bedside_directions(config: BedsideStocking3DEnvConfig, shoulder: NDArray) -> NDArray[np.float64]:
    """Unit vectors from the shoulder to the top of a nominal item on every bedside slot."""
    tops = np.concatenate([config.region_slots(r) for r in config.bedside_regions])
    d = tops + [0.0, 0.0, 2 * NOMINAL_HALF_HEIGHT] - np.asarray(shoulder)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


@dataclass(frozen=True, eq=False)
class BedsideQuery:
    """Present the gripper at shoulder + s * direction for each s in `distances` (outward)."""

    direction: NDArray[np.float64]  # unit vector from the shoulder
    distances: tuple[float, ...]  # m


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


@dataclass
class BedsideGatherer(InfoGatherer):
    """Asks along the direction where the belief disagrees most about the reach distance, at its quantiles.

    Queries run in the gatherer's own sim (an empty copy of the scene), never the planner's."""

    directions: NDArray[np.float64]  # (n, 3) unit vectors
    config: BedsideStocking3DEnvConfig = field(default_factory=BedsideStocking3DEnvConfig)
    num_points: int = 4  # distances per query, at belief quantiles
    noisy: bool = True  # answer ~ Bernoulli(sigmoid(m / scale)), the belief's own model; else m >= 0
    scale: float = EXECUTION_SCALE
    stop_at_miss: bool = True  # stop presenting after the first "no"
    grid_step: float = 0.025  # m, spacing of the candidate distances along each direction
    min_entropy: float = 0.1  # nats; ask nothing once no query's answers are this uncertain
    num_particles: int = 1000
    frame_stride: int = 8
    seed: int = 0
    last_seconds: float = field(default=0.0, init=False)
    _sim: ObjectCentricBedsideStocking3DEnv | None = field(default=None, init=False, repr=False, compare=False)
    _known: dict[tuple[int, int], bool] = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def sim(self) -> ObjectCentricBedsideStocking3DEnv:
        if self._sim is None:
            self._sim = ObjectCentricBedsideStocking3DEnv(0, 0, config=self.config, allow_state_access=True)
            self._sim.reset(seed=self.seed)
        return self._sim

    @property
    def distances(self) -> NDArray[np.float64]:
        """Candidate distances along every direction, spanning the prior's reach."""
        lb, ub = np.array(self.config.reach_theta_lb), np.array(self.config.reach_theta_ub)
        r = reach_distance(self.directions, np.stack([lb, ub])[:, None])
        return np.arange(r[0].min(), r[1].max() + 1e-9, self.grid_step)

    def presentable(self, k: int, j: int) -> bool:
        """Static collision-free IK for distance j along direction k (memoized)."""
        if (k, j) not in self._known:
            sim = self.sim
            self._known[k, j] = present_conf(sim, sim.shoulder + self.distances[j] * self.directions[k]) is not None
        return self._known[k, j]

    def choose(self, belief: Belief, rng: np.random.Generator) -> BedsideQuery | None:
        """Per direction, the presentable distances nearest the belief's reach quantiles; the direction whose
        answers are most uncertain (sum of binary entropies) wins. Unchecked distances count as presentable
        until the winner's are checked."""
        particles = getattr(belief, "particles", None)
        if particles is None:
            particles = belief.sample(rng, self.num_particles)
        dist = self.distances
        levels = np.arange(1, self.num_points + 1) / (self.num_points + 1)
        p = (reach_distance(self.directions[:, None], particles[None])[..., None] > dist).mean(1)  # P(reached)
        while True:
            best, score = None, self.min_entropy
            for k in range(len(self.directions)):
                idx = np.array([j for j in range(len(dist)) if self._known.get((k, j), True)], int)
                if not len(idx):
                    continue
                picked = idx[np.unique(np.abs(p[k, idx][:, None] - levels).argmin(0))]
                q = np.clip(p[k, picked], 1e-9, 1 - 1e-9)
                h = float(-(q * np.log(q) + (1 - q) * np.log(1 - q)).sum())
                if h > score:
                    best, score = (k, picked), h
            if best is None:
                return None
            k, picked = best
            unknown = [j for j in picked if (k, j) not in self._known]
            if not unknown:
                return BedsideQuery(self.directions[k].copy(), tuple(float(dist[j]) for j in picked))
            for j in unknown:
                self.presentable(k, j)

    def respond(self, theta: NDArray[np.float64], query: BedsideQuery, rng: np.random.Generator,
                frames: list | None = None) -> Observation:
        """Observation value: [(gripper point, reached), ...] at the points the robot could present."""
        started = time.time()
        sim = self.sim
        sim.reset(seed=self.seed)
        sim.reach_theta = np.array(theta, float)
        recorder = FrameRecorder(sim, frames, every=self.frame_stride)
        drive = Driver(sim, recorder, self.seed)
        recorder.snap()
        outcomes, base = [], None
        for s in query.distances:
            target = sim.shoulder + s * np.asarray(query.direction, float)
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
        return Observation("query", outcomes, np.asarray(query.direction, float))
