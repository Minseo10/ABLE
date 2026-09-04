"""Collection method 1: plan and execute one episode with the PDDLStream planner."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field

import numpy as np
from kinder.envs.dynamic3d.limbs import BodyMass
from kinder_pddlstream_planning.limbrepositioning3d.run import (
    build_stream_context,
    create_env,
    execute_plan,
    plan_limbrepositioning3d,
    reset_to_start,
)
from kinder_pddlstream_planning.limbrepositioning3d.stream import (
    MPCConfig,
    TorqueTrajectory,
)
from kinder_pddlstream_planning.limbrepositioning3d.utils import RolloutLog
from numpy.typing import NDArray

from limb_baseline.belief import Belief
from limb_baseline.human import theta_to_joint_limits, theta_to_range_of_motion
from limb_baseline.tasks import Task
from limb_baseline.variant import current as current_variant
from limb_baseline.variant import scene_config
from limb_baseline.warmstart import WarmStartModel, project_torques, segment_features


@dataclass(frozen=True)
class PlannerSettings:
    """The fixed environment, and how long the planner is given."""

    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    muscle_tone: str = "none"
    joint_limits_model: str = "box"
    limb_joint_damping: float = 0.5
    # Assistive gym's 50th percentile female; the env ships the 78.4 kg male.
    body_mass_kg: float = 62.5
    robot_base_z: float | None = None
    standoff: float = 1.0
    max_time: float = 600.0
    mpc: MPCConfig = field(default_factory=MPCConfig)


DEFAULT_PLANNER_SETTINGS = PlannerSettings()


@dataclass
class SegmentSample:
    """One `move_limb` segment as a training row."""

    features: NDArray[np.float64]
    psi: NDArray[np.float64]


@dataclass
class EpisodeResult:
    """What one `(task, person, belief, psi)` run produced."""

    solved: bool = False
    goal_reached: bool = False
    plan_time: float = 0.0
    execute_time: float = 0.0
    num_torque_steps: int = 0
    num_control_steps: int = 0
    mpc_simulator_steps: int = 0
    rom_violation_steps: int = 0
    worst_rom_violation: float = 0.0
    # Closest the execution came to the true limit; negative is a violation.
    rom_margin: float = float("nan")
    failure_reason: str = ""
    num_segments: int = 0
    segments: list["SegmentSample"] = field(default_factory=list)


def believed_limits_from(
    belief: Belief | None,
    theta: NDArray[np.float64],
    rng: np.random.Generator,
    oracle: bool,
) -> tuple[list[float], list[float]] | None:
    """What the planner is allowed to treat as the person's range of motion.

    `oracle` hands it the truth; otherwise one `theta` is drawn and held for the call.
    """
    if oracle:
        return None
    assert belief is not None
    lower, upper = theta_to_joint_limits(belief.sample(rng))
    return list(lower), list(upper)


def build_episode_env(task: Task, theta: NDArray[np.float64], settings: PlannerSettings):
    """The environment for one person and one task."""
    config = scene_config()
    config = dataclasses.replace(
        config,
        robot_initial_joints=tuple(task.robot_init),
        scene=dataclasses.replace(
            config.scene,
            limb_init_joint_positions=tuple(task.limb_init),
            limb_goal_joint_positions=tuple(task.limb_goal),
            limb_range_of_motion=theta_to_range_of_motion(theta),
        ),
    )
    return create_env(
        current_variant().name,
        config=config,
        standoff=settings.standoff,
        gravity=settings.gravity,
        muscle_tone=settings.muscle_tone,
        joint_limits_model=settings.joint_limits_model,
        limb_joint_damping=settings.limb_joint_damping,
        robot_base_z=settings.robot_base_z,
        body_mass=BodyMass(total=settings.body_mass_kg),
    )


def make_sampling_context(settings: PlannerSettings = DEFAULT_PLANNER_SETTINGS):
    """Return a `(sim, ctx)` pair used only to test whether a goal is reachable,
    with few IK attempts so that rejection is cheap.
    """
    config = scene_config()
    task = Task(
        limb_init=tuple(config.scene.limb_init_joint_positions),
        robot_init=tuple(config.robot_initial_joints),
        limb_goal=tuple(config.scene.limb_goal_joint_positions),
    )
    sim = build_episode_env(task, np.ones(3), settings)
    sim.reset(seed=0)
    ctx = build_stream_context(
        sim, standoff=settings.standoff, motion_seed=0, mpc=settings.mpc
    )
    reset_to_start(ctx)
    ctx.num_ik_attempts = 2
    return sim, ctx


def _move_limb_segments(plan) -> list[tuple]:
    """Retrieve torque trajectory from every `move_limb` in the plan, 
    as `(start state, goal conf, trajectory)`.
    """
    segments = []
    for name, args in plan:
        if name == "move_limb":
            _, _, _, s1, _, goal_conf, trajectory, _ = args
            assert isinstance(trajectory, TorqueTrajectory)
            segments.append((s1, goal_conf, trajectory))
    return segments


def run_episode(
    task: Task,
    theta: NDArray[np.float64],
    belief: Belief | None = None,
    warm_start: WarmStartModel | None = None,
    seed: int = 0,
    oracle: bool = False,
    settings: PlannerSettings = DEFAULT_PLANNER_SETTINGS,
) -> EpisodeResult:
    """Plan and execute one episode, returning what it cost and whether it worked."""
    result = EpisodeResult()
    rng = np.random.default_rng(seed)
    belief_features = (
        belief.as_features() if belief is not None else np.zeros(2 * len(theta))
    )

    sim = build_episode_env(task, theta, settings)
    try:
        sim.reset(seed=seed)
        ctx = build_stream_context(
            sim, standoff=settings.standoff, motion_seed=seed, mpc=settings.mpc
        )
        ctx.believed_joint_limits = believed_limits_from(belief, theta, rng, oracle)
        ctx.base_pose_hint = task.base_pose
        if warm_start is not None:
            ctx.warm_start = warm_start.as_hook(belief_features)
        reset_to_start(ctx)

        started = time.time()
        plan = plan_limbrepositioning3d(ctx, max_time=settings.max_time, verbose=False)
        result.plan_time = time.time() - started
        result.solved = plan is not None
        if plan is None:
            result.failure_reason = (
                ctx.best_attempt.reason
                if ctx.best_attempt is not None
                else "no limb motion was attempted (planner ran out of time)"
            )
            return result

        mpc = settings.mpc
        segments = _move_limb_segments(plan)
        result.num_segments = len(segments)
        for start_state, goal_conf, trajectory in segments:
            if not trajectory.robot_torques:
                continue
            result.num_torque_steps += len(trajectory.robot_torques)
            result.num_control_steps += (
                len(trajectory.robot_torques) // mpc.action_repeat
            )
            # Every control step rolls `num_rollouts` candidates over the horizon.
            result.mpc_simulator_steps += (
                len(trajectory.robot_torques)
                // mpc.action_repeat
                * mpc.num_rollouts
                * mpc.horizon
                * mpc.action_repeat
            )
            result.segments.append(
                SegmentSample(
                    features=segment_features(
                        start_state.limb_positions,
                        start_state.robot_positions,
                        np.asarray(goal_conf.positions),
                        belief_features,
                    ),
                    psi=project_torques(trajectory.robot_torques, mpc),
                )
            )

        reset_to_start(ctx)
        log = RolloutLog()
        started = time.time()
        result.goal_reached = execute_plan(ctx, plan, log=log)
        result.execute_time = time.time() - started
        if not result.goal_reached:
            result.failure_reason = "the executed plan missed the goal"

        lower, upper = np.asarray(ctx.true_joint_limits[0]), np.asarray(
            ctx.true_joint_limits[1]
        )
        margins = [
            float(np.min(np.minimum(np.asarray(q) - lower, upper - np.asarray(q))))
            for q in log.limb_positions
        ]
        if margins:
            result.rom_margin = min(margins)
        result.rom_violation_steps = sum(m < 0.0 for m in margins)
        result.worst_rom_violation = max((-m for m in margins if m < 0.0), default=0.0)
        return result
    finally:
        sim.close()
