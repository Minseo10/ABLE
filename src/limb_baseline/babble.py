"""Collection method 2: apply random torque trajectories within the limits, then relabel
wherever the limb ended up as the goal that motion was aiming for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from kinder.envs.dynamic3d.limb_utils import (
    LIMB_JOINT_INFOS,
    NUM_ROBOT_JOINTS,
    joint_position_distance,
)
from kinder_pddlstream_planning.limbrepositioning3d.run import (
    LIMB_NAME,
    build_stream_context,
    reset_to_start,
)
from kinder_pddlstream_planning.limbrepositioning3d.stream import (
    LimbStreamContext,
    sample_grasp,
)
from kinder_pddlstream_planning.limbrepositioning3d.utils import (
    CoupledState,
    JointTorques,
    _base_candidates,
    _grasp_ik,
    advance,
    arm_in_collision,
    capture_state,
    commanded_robot_torque,
    engage_grasp,
    exceeds_human_torque_limits,
    exceeds_robot_torque_limits,
    human_in_collision,
    limb_is_controllable,
    limb_out_of_limits,
    restore_state,
)
from numpy.typing import NDArray
from pybullet_helpers.joint import JointPositions

from limb_baseline.belief import Belief
from limb_baseline.planner import (
    DEFAULT_PLANNER_SETTINGS,
    ENVELOPE,
    PlannerSettings,
    SegmentSample,
    build_episode_env,
    joint_limits,
)
from limb_baseline.tasks import DEFAULT_TASK_DISTRIBUTION, Task
from limb_baseline.warmstart import control_basis, segment_features


@dataclass(frozen=True)
class BabbleSettings:
    """How the random exploration is run and which of its segments become rows."""

    num_segments: int = 4
    control_steps: int = 60
    noise_scale: float = 0.3
    drift_scale: float = 0.1
    max_draws: int = 8
    noise_decay: float = 0.6
    min_separation: float = DEFAULT_TASK_DISTRIBUTION.min_separation
    max_separation: float = DEFAULT_TASK_DISTRIBUTION.max_separation


DEFAULT_BABBLE_SETTINGS = BabbleSettings()

# Base poses tried when the task's own does not hold the grasp in this environment.
MAX_FALLBACK_BASE_POSES = 30


@dataclass
class BabbleResult:
    """What one babbling episode produced."""

    grasped: bool = False
    num_attempted: int = 0
    num_redraws: int = 0
    num_control_steps: int = 0
    separations: list[float] = field(default_factory=list)
    failure_reason: str = ""
    segments: list[SegmentSample] = field(default_factory=list)

    @property
    def num_kept(self) -> int:
        """Segments whose relabeled goal is as far off as a real task's."""
        return len(self.segments)


def _engage_at(ctx: LimbStreamContext, task: Task) -> bool:
    """Weld the hand onto the limb at the task's start, from wherever it can be held.

    The task's own base pose is tried against every grasp first, then nearby placements
    against the nominal grasp alone -- sweeping all pairs costs more than the episode.
    """
    sim = ctx.sim
    grasps = [grasp for (grasp,) in sample_grasp(ctx, LIMB_NAME)]
    attempts = [(task.base_pose, grasps)] if task.base_pose is not None else []
    attempts += [
        (pose, grasps[:1])
        for pose in _base_candidates(ctx)[:MAX_FALLBACK_BASE_POSES]
        if pose is not task.base_pose
    ]
    for base_conf, offered in attempts:
        sim.robot.set_base(base_conf)
        for grasp in offered:
            joints = _grasp_ik(ctx, grasp, tuple(task.limb_init), ctx.retract_joints)
            if joints is None or not limb_is_controllable(ctx, joints):
                continue
            sim.robot.arm.set_joints(list(joints), joint_velocities=[0.0] * len(joints))
            engage_grasp(sim)
            return True
    return False


def _draw_horizon(
    ctx: LimbStreamContext,
    rng: np.random.Generator,
    scale: float,
    center: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Control points near `center`, and their expansion to one torque per step.

    Same basis the MPC expands its nominal on, so refitting recovers the drawn points.
    """
    cfg = ctx.mpc
    lower, upper = ctx.robot_torque_limits
    step = rng.normal(scale=scale, size=(cfg.num_control_points, NUM_ROBOT_JOINTS))
    points = np.clip(center + step, lower, upper)
    return points, control_basis(cfg) @ points


def _run_horizon(
    ctx: LimbStreamContext, horizon_torques: NDArray[np.float64]
) -> list[JointTorques] | None:
    """Apply one horizon, or None as soon as any limit would be broken.

    The simulator is left wherever the rejected horizon got to; the caller restores it.
    """
    sim = ctx.sim
    cfg = ctx.mpc
    applied: list[JointTorques] = []
    for torque in horizon_torques:
        for _ in range(cfg.action_repeat):
            # Read before the step, which clips a saturated command away.
            commanded = commanded_robot_torque(sim, torque)
            if exceeds_robot_torque_limits(ctx, commanded):
                return None
            human = advance(sim, commanded)
            applied.append(list(torque))
            if exceeds_human_torque_limits(ctx, human):
                return None
            if limb_out_of_limits(ctx):
                return None
            if not sim.robot.arm.check_joint_limits(
                list(sim.robot.arm.get_joint_positions())
            ):
                return None
        if arm_in_collision(ctx) or (
            ctx.resting_penetration and human_in_collision(ctx)
        ):
            return None
    return applied


@dataclass(frozen=True)
class Waypoint:
    """Where the limb and the robot stood at one horizon boundary."""

    limb: JointPositions
    robot: JointPositions


def _babble_segment(
    ctx: LimbStreamContext,
    rng: np.random.Generator,
    settings: BabbleSettings,
    result: BabbleResult,
    start_limb: JointPositions,
    target_separation: float,
) -> tuple[list[Waypoint], list[NDArray[np.float64]]]:
    """Random torques for whole horizons, rejecting the draws that break a limit.

    Each horizon is drawn around the previous accepted one, so the segment is one smooth
    push whose first horizon points where the whole thing goes. A rejected horizon is
    rolled back and redrawn with less noise; the segment ends when no draw survives,
    when the limb has travelled `target_separation`, or
    after `control_steps` control steps, whichever comes first.

    Returns one waypoint per horizon boundary and the control points applied between
    them, so `n` horizons give `n + 1` waypoints and `n` labels.
    """
    cfg = ctx.mpc
    waypoints: list[Waypoint] = []
    labels: list[NDArray[np.float64]] = []
    center = np.zeros((cfg.num_control_points, NUM_ROBOT_JOINTS))
    for horizon_index in range(max(settings.control_steps // cfg.horizon, 1)):
        start = capture_state(ctx.sim)
        # The first horizon starts the push; the rest only drift away from it.
        scale = settings.noise_scale if horizon_index == 0 else settings.drift_scale
        for _ in range(settings.max_draws):
            points, horizon = _draw_horizon(ctx, rng, scale, center)
            applied = _run_horizon(ctx, horizon)
            if applied is not None:
                waypoints.append(
                    Waypoint(list(start.limb_positions), list(start.robot_positions))
                )
                labels.append(points)
                center = points
                break
            result.num_redraws += 1
            restore_state(ctx.sim, start)
            scale *= settings.noise_decay
        else:
            break
        travelled = joint_position_distance(
            LIMB_JOINT_INFOS, list(start_limb), list(ctx.sim.limb.get_joint_positions())
        )
        if travelled >= target_separation:
            break
    if waypoints:
        landed = capture_state(ctx.sim)
        waypoints.append(
            Waypoint(list(landed.limb_positions), list(landed.robot_positions))
        )
    return waypoints, labels


def _rested(state: CoupledState) -> CoupledState:
    """The same configuration, standing still.

    A demonstration starts from rest, so chained segments must too or the features lie.
    """
    state.robot_velocities = [0.0] * len(state.robot_velocities)
    state.limb_velocities = [0.0] * len(state.limb_velocities)
    return state


def run_babble_episode(
    task: Task,
    theta: NDArray[np.float64],
    belief: Belief | None = None,
    believed_theta: NDArray[np.float64] | None = None,
    seed: int = 0,
    settings: PlannerSettings = DEFAULT_PLANNER_SETTINGS,
    babble: BabbleSettings = DEFAULT_BABBLE_SETTINGS,
) -> BabbleResult:
    """Grasp this person's limb and babble it around, relabeling where it lands."""
    result = BabbleResult()
    rng = np.random.default_rng(seed)
    belief_features = (
        belief.as_features() if belief is not None else np.zeros(2 * len(theta))
    )

    sim = build_episode_env(task, np.full(len(theta), ENVELOPE), settings)
    try:
        sim.reset(seed=seed)
        ctx = build_stream_context(
            sim, standoff=settings.standoff, motion_seed=seed, mpc=settings.mpc
        )
        ctx.believed_joint_limits = joint_limits(believed_theta)
        ctx.true_joint_limits = joint_limits(theta)
        reset_to_start(ctx)
        result.grasped = _engage_at(ctx, task)
        if not result.grasped:
            result.failure_reason = "no arm configuration holds the grasp at the start"
            return result

        for _ in range(babble.num_segments):
            start = _rested(capture_state(sim))
            restore_state(sim, start)
            result.num_attempted += 1
            waypoints, labels = _babble_segment(
                ctx,
                rng,
                babble,
                result,
                start.limb_positions,
                rng.uniform(babble.min_separation, babble.max_separation),
            )
            if not labels:
                continue
            result.num_control_steps += len(labels) * ctx.mpc.horizon
            result.separations.append(
                joint_position_distance(
                    LIMB_JOINT_INFOS, waypoints[0].limb, waypoints[-1].limb
                )
            )
            # Where the whole push landed is the goal its first horizon aimed at,
            # which is the one horizon the warm start has to guess.
            if (
                not babble.min_separation
                <= result.separations[-1]
                <= babble.max_separation
            ):
                continue
            result.segments.append(
                SegmentSample(
                    features=segment_features(
                        waypoints[0].limb,
                        waypoints[0].robot,
                        np.asarray(waypoints[-1].limb, dtype=np.float64),
                        belief_features,
                    ),
                    psi=labels[0],
                )
            )
        if not result.segments:
            result.failure_reason = (
                f"no segment landed between {babble.min_separation} "
                f"and {babble.max_separation} from where it started"
            )
        return result
    finally:
        sim.close()
