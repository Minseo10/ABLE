"""Stages 1-3: query the person, update the belief, then plan and execute one episode.

`gather_information` probes the least certain motion each round; an answer that reaches
its target only raises that motion's floor. `run_episode` plans with PDDLStream.
"""

from __future__ import annotations

import dataclasses
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial

import numpy as np
from kinder.envs.dynamic3d.limbs import BodyMass
from numpy.typing import NDArray
from pybullet_helpers.geometry import SE2Pose

from able.tasks.limb.belief import Belief, pooled_tau, refit
from able.tasks.limb.human import (
    DEFAULT_PRIOR,
    RoMPrior,
    default_magnitudes,
    theta_to_joint_limits,
    theta_to_range_of_motion,
)
from able.tasks.limb.planning.run import (
    build_stream_context,
    create_env,
    plan_limbrepositioning3d,
    reset_to_start,
)
from able.tasks.limb.planning.stream import (
    MPCConfig,
    TorqueTrajectory,
)
from able.tasks.limb.queries import (
    DEFAULT_QUERY_MODEL,
    QueryModel,
    QueryOutcome,
    QuerySimulator,
)
from able.tasks.limb.tasks import Task
from able.tasks.limb.utils import RolloutLog, execute_plan
from able.tasks.limb.variant import current as current_variant
from able.tasks.limb.variant import scene_config
from able.tasks.limb.warmstart import (
    WarmStartModel,
    horizon_window,
    project_torques,
    segment_features,
)


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
    # Stop planning at exactly `max_time`, whatever stream is running. PDDLStream only
    # checks its clock between iterations, so one MPC call can otherwise run far past it.
    hard_time_limit: bool = False
    mpc: MPCConfig = field(default_factory=MPCConfig)


DEFAULT_PLANNER_SETTINGS = PlannerSettings()
# Metres either side of its home pose the base motion planner may sample.
BASE_REACH = 3.0


class PlanningTimeout(BaseException):
    """Raised by the alarm at the planning deadline.

    A `BaseException`, as `KeyboardInterrupt` is, so no stream's `except Exception`
    swallows it on the way out.
    """


@contextmanager
def hard_deadline(seconds: float | None):
    """Raise `PlanningTimeout` after `seconds` of wall-clock time, wherever the code is.

    A Unix interval timer on the main thread; the handler interrupts any Python code,
    and a C call (one PyBullet step) returns within milliseconds. None or <= 0 is no
    limit. The previous handler and timer are restored on the way out.
    """
    if seconds is None or seconds <= 0:
        yield
        return

    def fire(signum, frame):  # pylint: disable=unused-argument
        raise PlanningTimeout(f"hit the {seconds:g} s hard planning time limit")

    previous = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@dataclass
class SegmentSample:
    """One `move_limb` control step as a training row.

    `psi` is the horizon fitted to control points; `torques` the same horizon as the
    raw per-control-step torques it was fitted to, `(horizon, NUM_ROBOT_JOINTS)`.
    """

    features: NDArray[np.float64]
    psi: NDArray[np.float64]
    torques: NDArray[np.float64] | None = None


@dataclass
class EpisodeResult:
    """What one `(task, person, belief, psi)` run produced."""

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
    num_segments: int = 0
    segments: list["SegmentSample"] = field(default_factory=list)
    # The limb's joint positions after every executed simulator step.
    limb_path: list = field(default_factory=list)


def assumed_theta(
    belief: Belief | None,
    theta: NDArray[np.float64],
    rng: np.random.Generator,
    oracle: bool,
) -> NDArray[np.float64]:
    """The range of motion the robot acts on: the truth for an oracle, else one draw.

    One draw serves both the task and the planner's limits, so the goal it is handed
    always sits inside the range it believes.
    """
    if oracle:
        return np.asarray(theta, dtype=np.float64)
    assert belief is not None
    return belief.sample(rng)


def joint_limits(theta: NDArray[np.float64]) -> tuple[list[float], list[float]]:
    """One person's range of motion, as the context stores its bounds."""
    lower, upper = theta_to_joint_limits(theta)
    return list(lower), list(upper)


# The widest anyone can be: a goal posed against an over-wide belief still builds.
ENVELOPE = DEFAULT_PRIOR.high


@dataclass
class ParticleLimits:
    """The believed range as particles, the notebook's belief, as a chance constraint.

    The believed range only ever enters the planner as a box check on rolled-out limb
    positions (`LimbStreamContext.in_believed_limits`); the physics is the same under
    every hypothesis. So one rollout can be checked against every particle at once: a
    configuration is believed inside when at least `confidence` of the particles allow
    it. At 0.5 that is roughly the median person, near 1 the narrowest plausible one.
    """

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
        """The one theta whose range `confidence` of the particles reach, per motion.

        Goals posed against it are admissible under the check above, joint by joint;
        it also stands in for the belief wherever a single theta is needed.
        """
        return np.quantile(self.particles, 1.0 - self.confidence, axis=0)


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
    # The base ships with infinite bounds, which the base motion planner cannot sample
    # from once a straight drive is blocked (the furnished scenes); box it around home.
    base = env.robot.base
    home = env.robot.get_base()
    base.pose_lower_bound = SE2Pose(home.x - BASE_REACH, home.y - BASE_REACH, -np.pi)
    base.pose_upper_bound = SE2Pose(home.x + BASE_REACH, home.y + BASE_REACH, np.pi)
    return env


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
    sim = build_episode_env(task, np.full(3, ENVELOPE), settings)
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


def dense_rows(segments, log: RolloutLog, belief_features, mpc: MPCConfig) -> list:
    """One row per control step: where the limb stood, and the horizon flown from there.

    The log records the state *after* each simulator step, so a control step's own
    state is the entry before it, and the first falls back to the segment's start.
    """
    rows: list[SegmentSample] = []
    offset = 0
    for start_state, goal_conf, trajectory in segments:
        torques = trajectory.robot_torques
        goal = np.asarray(goal_conf.positions, dtype=np.float64)
        for step in range(len(torques) // mpc.action_repeat):
            index = offset + step * mpc.action_repeat - 1
            if index >= len(log.limb_positions):
                break
            if index < offset:
                limb, robot = start_state.limb_positions, start_state.robot_positions
                limb_velocity = start_state.limb_velocities
                robot_velocity = start_state.robot_velocities
            else:
                limb, robot = log.limb_positions[index], log.robot_positions[index]
                limb_velocity = log.limb_velocities[index]
                robot_velocity = log.robot_velocities[index]
            previous = torques[(step - 1) * mpc.action_repeat] if step else None
            rows.append(
                SegmentSample(
                    features=segment_features(
                        limb, robot, goal, belief_features, step,
                        limb_velocity, robot_velocity, previous,
                    ),
                    psi=project_torques(torques[step * mpc.action_repeat :], mpc),
                    torques=horizon_window(torques[step * mpc.action_repeat :], mpc),
                )
            )
        offset += len(torques)
    return rows


def open_episode(
    task: Task,
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
            # Every limit check the planner makes goes through this one method.
            ctx.in_believed_limits = particle_limits
        ctx.true_joint_limits = joint_limits(theta)
        ctx.base_pose_hint = task.base_pose
        reset_to_start(ctx)
    except BaseException:
        sim.close()
        raise
    return sim, ctx


def count_effort(result: EpisodeResult, segments: list[tuple], mpc: MPCConfig) -> None:
    """Add the planning effort of `segments` to `result`."""
    result.num_segments += len(segments)
    for _, _, trajectory in segments:
        if not trajectory.robot_torques:
            continue
        result.num_torque_steps += len(trajectory.robot_torques)
        result.num_control_steps += len(trajectory.robot_torques) // mpc.action_repeat
        # Every control step rolls `num_rollouts` candidates over the horizon.
        result.mpc_simulator_steps += (
            len(trajectory.robot_torques)
            // mpc.action_repeat
            * mpc.num_rollouts
            * mpc.horizon
            * mpc.action_repeat
        )


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


def run_episode(
    task: Task,
    theta: NDArray[np.float64],
    belief: Belief | None = None,
    believed_theta: NDArray[np.float64] | None = None,
    warm_start: WarmStartModel | None = None,
    seed: int = 0,
    settings: PlannerSettings = DEFAULT_PLANNER_SETTINGS,
    particle_limits: ParticleLimits | None = None,
    plan_fn=None,
) -> EpisodeResult:
    """Plan and execute one episode, returning what it cost and whether it worked.

    `believed_theta` is the one range the planner plans inside; `particle_limits`
    replaces that box with a chance constraint over belief particles when given.
    `plan_fn(ctx, max_time)` returns a plan or None (default PDDLStream).
    """
    result = EpisodeResult()
    belief_features = (
        belief.as_features() if belief is not None else np.zeros(2 * len(theta))
    )
    if plan_fn is None:
        plan_fn = partial(plan_limbrepositioning3d, verbose=False)

    sim, ctx = open_episode(task, theta, believed_theta, seed, settings, particle_limits)
    try:
        if warm_start is not None:
            ctx.warm_start = warm_start.as_hook(belief_features)

        started, cpu_started = time.time(), time.process_time()
        try:
            with hard_deadline(settings.max_time if settings.hard_time_limit else None):
                plan = plan_fn(ctx, max_time=settings.max_time)
        except PlanningTimeout as timeout:
            result.plan_time = time.time() - started
            result.plan_cpu_time = time.process_time() - cpu_started
            result.failure_reason = str(timeout)
            return result
        result.plan_time = time.time() - started
        result.plan_cpu_time = time.process_time() - cpu_started
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
        count_effort(result, segments, mpc)

        reset_to_start(ctx)
        log = RolloutLog()
        started = time.time()
        result.goal_reached = execute_plan(ctx, plan, log=log)
        result.execute_time = time.time() - started
        if not result.goal_reached:
            result.failure_reason = "the executed plan missed the goal"

        result.segments = dense_rows(segments, log, belief_features, mpc)
        result.limb_path = [list(q) for q in log.limb_positions]
        score_range_of_motion(result, ctx)
        return result
    finally:
        sim.close()


# ---------------------------------------------------------------- stages 1-2


@dataclass(frozen=True)
class QueryPolicy:
    """How many questions to ask, and where around the default range to ask."""

    # Six, not three: censored motions need a second round after the floor rises.
    num_queries: int = 6
    # How far either side of the boundary the target is drawn, in radians.
    probe_band: float = 0.3

    def choose(
        self, belief: Belief, rng: np.random.Generator, floors: NDArray[np.float64]
    ) -> tuple[int, float]:
        """The motion to ask about, and the angle to ask for.

        The angle is drawn around the *default* range of motion, since the person's own
        is what the query is after. `floors` raises it where a target was already
        reached, so a wider-ranged person is asked higher instead of answering trivially.
        """
        motion_index = int(np.argmax(belief.stddev))
        bound = max(default_magnitudes()[motion_index], float(floors[motion_index]))
        band = self.probe_band
        return motion_index, float(bound + rng.uniform(-band, band))


DEFAULT_QUERY_POLICY = QueryPolicy()


@dataclass
class InformationGathering:
    """What Stage 1 produced."""

    posterior: Belief
    outcomes: list[QueryOutcome]
    floors: NDArray[np.float64] | None = None
    tau: float = DEFAULT_QUERY_MODEL.tau

    @property
    def num_censored(self) -> int:
        """Answers that stopped at the target rather than at the person's limit."""
        return sum(outcome.reached_target for outcome in self.outcomes)

    @property
    def num_unsettled(self) -> int:
        """Answers cut off by the step cap, which locate nothing and are skipped."""
        return sum(
            not o.settled and not o.reached_target for o in self.outcomes
        )


def gather_information(
    theta: NDArray[np.float64],
    rng: np.random.Generator,
    policy: QueryPolicy = DEFAULT_QUERY_POLICY,
    prior: RoMPrior = DEFAULT_PRIOR,
    model: QueryModel = DEFAULT_QUERY_MODEL,
    simulator: QuerySimulator | None = None,
    start: Belief | None = None,
    floors: NDArray[np.float64] | None = None,
    estimate_tau: bool = False,
) -> InformationGathering:
    """Ask `policy.num_queries` questions, updating the belief after each."""
    owned = simulator is None
    simulator = simulator or QuerySimulator(theta, model)
    try:
        base = start if start is not None else Belief.from_prior(prior)
        belief = base
        outcomes: list[QueryOutcome] = []
        answers: list[list[float]] = [[] for _ in base.mean]
        tau = model.tau
        floors = np.zeros(len(belief.mean)) if floors is None else floors.copy()
        for _ in range(policy.num_queries):
            motion_index, target = policy.choose(belief, rng, floors)
            outcome = simulator.ask(motion_index, target, rng)
            outcomes.append(outcome)
            # A joint that saturated its speed cap was not making a passive push.
            if outcome.peak_velocity >= model.max_velocity:
                continue
            # A censored answer only raises the floor; it does not locate the bound.
            if outcome.reached_target:
                floors[motion_index] = outcome.answer + model.short_fall
            elif outcome.settled:
                answers[motion_index].append(outcome.answer)
                if estimate_tau:
                    tau = pooled_tau(answers, model.tau)
                    belief = refit(base, answers, tau, model.short_fall)
                else:
                    belief = belief.updated(
                        motion_index, outcome.answer, tau, model.short_fall
                    )
        return InformationGathering(belief, outcomes, floors, tau)
    finally:
        if owned:
            simulator.close()
