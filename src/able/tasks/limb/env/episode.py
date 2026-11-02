"""Per-episode simulator setup and execution scoring."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from kinder.envs.dynamic3d.limbs import BodyMass
from numpy.typing import NDArray
from pybullet_helpers.geometry import SE2Pose

from able.tasks.limb.env.human import (
    DEFAULT_PRIOR,
    theta_to_joint_limits,
    theta_to_range_of_motion,
)
from able.tasks.limb.env.variant import current as current_variant
from able.tasks.limb.env.variant import scene_config
from able.tasks.limb.planning.run import (
    build_stream_context,
    create_env,
    reset_to_start,
)
from able.tasks.limb.planning.stream import (
    MPCConfig,
)

if TYPE_CHECKING:
    from able.tasks.limb.task import LimbProblem


@dataclass(frozen=True)
class PlannerSettings:
    """Fixed environment settings and planner time budget."""

    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    muscle_tone: str = "none"
    joint_limits_model: str = "box"
    limb_joint_damping: float = 0.5
    # Assistive gym's 50th percentile female; the env ships the 78.4 kg male.
    body_mass_kg: float = 62.5
    robot_base_z: float | None = None
    standoff: float = 1.0
    max_time: float = 600.0
    # Enforce `max_time` via SIGALRM: PDDLStream only checks its clock between iterations.
    hard_time_limit: bool = False
    mpc: MPCConfig = field(default_factory=MPCConfig)


DEFAULT_PLANNER_SETTINGS = PlannerSettings()


# Metres either side of its home pose the base motion planner may sample.
BASE_REACH = 3.0


@dataclass
class EpisodeResult:
    """Outcome of one `(task, person, belief, psi)` run."""

    solved: bool = False
    goal_reached: bool = False
    plan_time: float = 0.0
    # Process CPU time; unlike `plan_time`, comparable across machine loads.
    plan_cpu_time: float = 0.0
    execute_time: float = 0.0
    num_torque_steps: int = 0
    num_control_steps: int = 0
    mpc_simulator_steps: int = 0
    rom_violation_steps: int = 0
    worst_rom_violation: float = 0.0
    # Closest the execution came to the true limit; negative is a violation.
    rom_margin: float = float("nan")
    failure_reason: str = ""
    limb_path: list = field(default_factory=list)


def joint_limits(theta: NDArray[np.float64]) -> tuple[list[float], list[float]]:
    """Joint limits for `theta` as `(lower, upper)` lists."""
    lower, upper = theta_to_joint_limits(theta)
    return list(lower), list(upper)


# Widest possible range, so a goal posed against an over-wide belief still builds.
ENVELOPE = DEFAULT_PRIOR.high


@dataclass
class ParticleLimits:
    """Chance-constrained limit check: inside if at least `confidence` of the particles allow it."""

    particles: NDArray[np.float64]
    confidence: float = 0.5

    def __post_init__(self) -> None:
        self.particles = np.atleast_2d(np.asarray(self.particles, dtype=np.float64))
        assert 0.0 < self.confidence <= 1.0
        bounds = [theta_to_joint_limits(theta) for theta in self.particles]
        self.lower = np.stack([low for low, _ in bounds])
        self.upper = np.stack([high for _, high in bounds])

    def __call__(self, joint_positions) -> bool:
        """`LimbStreamContext.in_believed_limits`, over the particles."""
        q = np.asarray(joint_positions, dtype=np.float64)
        inside = np.all((self.lower <= q) & (q <= self.upper), axis=1)
        return bool(inside.mean() >= self.confidence)

    def quantile_theta(self) -> NDArray[np.float64]:
        """Per-motion theta that `confidence` of the particles reach."""
        return np.quantile(self.particles, 1.0 - self.confidence, axis=0)


def build_episode_env(task: LimbProblem, theta: NDArray[np.float64], settings: PlannerSettings):
    """Build the env for one person and task."""
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
    env = create_env(
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
    # Base ships with infinite bounds, which the base planner cannot sample; box it.
    base = env.robot.base
    home = env.robot.get_base()
    base.pose_lower_bound = SE2Pose(home.x - BASE_REACH, home.y - BASE_REACH, -np.pi)
    base.pose_upper_bound = SE2Pose(home.x + BASE_REACH, home.y + BASE_REACH, np.pi)
    return env


def make_sampling_context(settings: PlannerSettings = DEFAULT_PLANNER_SETTINGS):
    """A `(sim, ctx)` for cheap goal-reachability tests (few IK attempts)."""
    from able.tasks.limb.task import LimbProblem

    config = scene_config()
    task = LimbProblem(
        limb_init=tuple(config.scene.limb_init_joint_positions),
        robot_init=tuple(config.robot_initial_joints),
        limb_goal=tuple(config.scene.limb_goal_joint_positions),
    )
    sim = build_episode_env(task, np.full(3, ENVELOPE), settings)
    sim.reset(seed=0)
    ctx = build_stream_context(
        sim, standoff=settings.standoff, motion_seed=0, mpc=settings.mpc
    )
    reset_to_start(ctx)
    ctx.num_ik_attempts = 2
    return sim, ctx


def open_episode(
    task: LimbProblem,
    theta: NDArray[np.float64],
    believed_theta: NDArray[np.float64],
    seed: int,
    settings: PlannerSettings,
    particle_limits: ParticleLimits | None = None,
):
    """One episode's env and stream context at the task start; caller closes `sim`."""
    sim = build_episode_env(task, np.full(len(theta), ENVELOPE), settings)
    try:
        sim.reset(seed=seed)
        ctx = build_stream_context(
            sim, standoff=settings.standoff, motion_seed=seed, mpc=settings.mpc
        )
        ctx.believed_joint_limits = joint_limits(believed_theta)
        if particle_limits is not None:
            ctx.in_believed_limits = particle_limits
        ctx.true_joint_limits = joint_limits(theta)
        ctx.base_pose_hint = task.base_pose
        reset_to_start(ctx)
    except BaseException:
        sim.close()
        raise
    return sim, ctx


def score_range_of_motion(result: EpisodeResult, ctx) -> None:
    """Check `result.limb_path` against the true limits; a violation is a failure."""
    lower, upper = np.asarray(ctx.true_joint_limits[0]), np.asarray(
        ctx.true_joint_limits[1]
    )
    margins = [
        float(np.min(np.minimum(np.asarray(q) - lower, upper - np.asarray(q))))
        for q in result.limb_path
    ]
    if margins:
        result.rom_margin = min(margins)
    result.rom_violation_steps = sum(m < 0.0 for m in margins)
    result.worst_rom_violation = max((-m for m in margins if m < 0.0), default=0.0)
    if result.rom_violation_steps:
        result.goal_reached = False
        result.failure_reason = (
            f"drove the limb {result.worst_rom_violation:.3f} rad past the "
            "person's true anatomical limit"
        )
