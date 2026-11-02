"""The limb-repositioning POMDP (domain.pddl's actions) that POUCT/POMCP search, for pomdp_py.

Only the person's `theta` is hidden; a `move_limb` stops at the first limit on its line, which
ends the episode and is observed. Simulator-free: planning and execution are in `pomdp_planner`.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pomdp_py
from kinder.envs.dynamic3d.limb_utils import NUM_LIMB_JOINTS
from numpy.typing import NDArray

from able.tasks.limb.belief import Belief
from able.tasks.limb.human import active_joints, active_signs, default_magnitudes


@dataclass(frozen=True)
class PomdpConfig:
    """Waypoints, rewards and search settings."""

    # Evenly spaced `move_limb` targets on the line; 1 is the goal alone, as in PDDLStream.
    num_waypoints: int = 1
    # Grasps sampled per refill, and base poses per grasp.
    num_grasps: int = 2
    bases_per_grasp: int = 2
    # For the whole line; a move earns its share on arrival.
    goal_reward: float = 10.0
    # PDDLStream's unit cost.
    action_cost: float = 1.0
    # Charged on top of the progress forfeited at a limit.
    limit_penalty: float = 0.0
    # Limit-reading bin width (scale units of `theta`) and noise (radians).
    resolution: float = 0.02
    tau: float = 0.05
    # "pouct": a new tree per plan step; "pomcp": one tree kept down the plan;
    # "pomcpow": continuous `move_limb` targets, a new tree per plan step.
    solver: str = "pouct"
    num_sims: int = 3000
    exploration: float = 5.0
    # POMCPOW: simulations, and progressive widening k * N^alpha for actions and
    # observations (tuned on `example_facts`; wider action widening underrates the root).
    pomcpow_sims: int = 10000
    action_widening: tuple[float, float] = (1.0, 0.5)
    observation_widening: tuple[float, float] = (4.0, 0.5)
    # POMCPOW: limb moves per plan, and how close to a failed target a new one may be.
    max_limb_moves: int = 3
    block_radius: float = 0.05
    # Draws of `theta` when the belief has no particles.
    belief_draws: int = 32


PLAN_ONCE = PomdpConfig()


# ------------------------------------------------- state, actions, observations


@dataclass(frozen=True)
class DomainState(pomdp_py.State):
    """Everything but `theta` is observed; `base`/`grasp` index facts (-1: start, empty hand)."""

    base: int
    grasp: int
    base_pose: tuple[float, ...]
    arm: tuple[float, ...]
    limb: tuple[float, ...]
    # Fraction of the line covered.
    along: float
    theta: tuple[float, ...]
    # Motion whose limit stopped the last move, or -1.
    hit: int = -1
    done: bool = False


@dataclass(frozen=True)
class MoveBase(pomdp_py.Action):
    """Drive the base to sampled pose `base`; once, with an empty hand."""

    base: int


@dataclass(frozen=True)
class Grasp(pomdp_py.Action):
    """Grasp the limb with sampled grasp `grasp` from the current base."""

    grasp: int


@dataclass(frozen=True)
class MoveLimb(pomdp_py.Action):
    """Move the limb to waypoint `waypoint`; the last is the goal."""

    waypoint: int


@dataclass(frozen=True)
class MoveLimbTo(pomdp_py.Action):
    """Move the limb to fraction `along` of the line (POMCPOW's continuous target)."""

    along: float


def is_limb_move(action) -> bool:
    return isinstance(action, (MoveLimb, MoveLimbTo))


def target_of(action, num_waypoints: int) -> float | None:
    """A limb move's target as a fraction of the line; None for other actions."""
    if isinstance(action, MoveLimbTo):
        return action.along
    if isinstance(action, MoveLimb):
        return action.waypoint / num_waypoints
    return None


@dataclass(frozen=True)
class Stop(pomdp_py.Action):
    """End the episode."""


@dataclass(frozen=True)
class Clear(pomdp_py.Observation):
    """No limit was met."""


@dataclass(frozen=True)
class Limit(pomdp_py.Observation):
    """A noisy reading of the met limit; `bin` counts `resolution` steps of `theta`'s scale."""

    motion: int
    bin: int


# ------------------------------------------------------------------ geometry


@dataclass(frozen=True)
class Path:
    """The limb's straight line to the goal, in each active motion's signed reach."""

    start: NDArray[np.float64]
    end: NDArray[np.float64]
    magnitudes: NDArray[np.float64]
    # Joint-space distance, radians.
    length: float
    # Full limb configurations at the ends.
    init: NDArray[np.float64]
    goal: NDArray[np.float64]

    @classmethod
    def between(cls, limb_init, limb_goal) -> "Path":
        init = np.asarray(limb_init, dtype=np.float64)
        goal = np.asarray(limb_goal, dtype=np.float64)
        joints, signs = list(active_joints()), np.asarray(active_signs())
        return cls(
            signs * init[joints],
            signs * goal[joints],
            default_magnitudes(),
            float(np.linalg.norm(goal - init)),
            init,
            goal,
        )

    def point(self, along: float) -> NDArray[np.float64]:
        """The limb configuration at `along`."""
        return self.init + along * (self.goal - self.init)

    @property
    def outward(self) -> list[int]:
        """Motions pushed toward their limit; only these can stop the path."""
        return [i for i in range(len(self.start)) if self.end[i] > self.start[i]]

    def scale(self, along: float) -> NDArray[np.float64]:
        """Each motion's reach at `along`, as a fraction of its default range."""
        return (self.start + along * (self.end - self.start)) / self.magnitudes

    def advance(self, along: float, target: float, theta) -> tuple[float, int]:
        """Where a move to `target` stops for `theta`, and the stopping motion (-1: none)."""
        reached = self.scale(target)
        stop, motion = target, -1
        for i in self.outward:
            if reached[i] <= theta[i]:
                continue
            slope = (self.end[i] - self.start[i]) / self.magnitudes[i]
            at = max(along, (theta[i] - self.start[i] / self.magnitudes[i]) / slope)
            if at < stop:
                stop, motion = at, i
        return stop, motion


def one_motion_task(start_scale: float, goal_scale: float, motion: int = 0):
    """Start and goal differing in one active motion, as scales of its range."""
    joint, sign = active_joints()[motion], active_signs()[motion]
    magnitude = default_magnitudes()[motion]
    limb_init, limb_goal = np.zeros(NUM_LIMB_JOINTS), np.zeros(NUM_LIMB_JOINTS)
    limb_init[joint] = sign * start_scale * magnitude
    limb_goal[joint] = sign * goal_scale * magnitude
    return limb_init, limb_goal


# ---------------------------------------------------------------------- facts


def pose_of(pose) -> tuple[float, float, float]:
    return (float(pose.x), float(pose.y), float(pose.rot))


@dataclass
class Facts:
    """The sampled poses and the actions they certify."""

    path: Path
    num_waypoints: int
    start_pose: tuple[float, ...]
    start_arm: tuple[float, ...]
    grasps: list = field(default_factory=list)
    bases: list = field(default_factory=list)
    # BaseMotion: start-to-pose base path, per pose.
    base_paths: list = field(default_factory=list)
    # GraspKin: (base, grasp) -> (approach path, grasped state).
    kin: dict = field(default_factory=dict)
    # Reachable: (base, grasp) -> arm joints per waypoint, None if unreachable.
    reach: dict = field(default_factory=dict)
    # Per waypoint: allowed by the believed limits.
    allowed: list = field(default_factory=list)
    # (base, grasp, waypoint) the MPC failed or a check rejected.
    blocked: set = field(default_factory=set)

    def target(self, waypoint: int) -> NDArray[np.float64]:
        return self.path.point(waypoint / self.num_waypoints)

    def open(self, base: int, grasp: int, along: float = 0.0) -> list[int]:
        """Waypoints ahead still open to `move_limb`."""
        reach = self.reach.get((base, grasp))
        if reach is None:
            return []
        n = self.num_waypoints
        return [
            k
            for k in range(1, n + 1)
            if k / n > along + 1e-9
            and reach[k - 1] is not None
            and self.allowed[k - 1]
            and (base, grasp, k) not in self.blocked
        ]

    def target_open(self, base: int, grasp: int, along: float, radius: float) -> bool:
        """Whether a continuous target is clear of the failed ones on this pair."""
        return not any(
            (b, g) == (base, grasp) and isinstance(a, float) and abs(a - along) < radius
            for b, g, a in self.blocked
        )

    def usable(self) -> bool:
        return any(self.open(base, grasp) for base, grasp in self.kin)

    def root(self, theta=None) -> DomainState:
        """The start state: start base pose, empty hand, limb at the line's start."""
        theta = tuple(0.0 for _ in self.path.magnitudes) if theta is None else tuple(theta)
        return DomainState(
            -1, -1, self.start_pose, self.start_arm, tuple(self.path.init), 0.0, theta
        )


def example_facts(
    goal_scale: float, num_waypoints: int = 4, pairs=((0, 0),), start_scale: float = 0.5
) -> Facts:
    """Simulator-free facts for tests and figures; every pair reaches and allows every waypoint."""
    arm = (0.1,) * 7
    path = Path.between(*one_motion_task(start_scale, goal_scale))
    facts = Facts(path, num_waypoints, (0.0, 0.0, 0.0), arm)
    num_bases = 1 + max(b for b, _ in pairs)
    facts.bases = [SimpleNamespace(x=1.0 + b, y=0.0, rot=0.0) for b in range(num_bases)]
    facts.base_paths = [f"base path {b}" for b in range(num_bases)]
    facts.grasps = [object() for _ in range(1 + max(g for _, g in pairs))]
    for pair in pairs:
        facts.kin[pair] = (None, SimpleNamespace(robot_positions=list(arm)))
        facts.reach[pair] = [list(arm) for _ in range(num_waypoints)]
    facts.allowed = [True] * num_waypoints
    return facts


# -------------------------------------------------------------------- models


class Transition(pomdp_py.TransitionModel):
    """Domain effects; a `move_limb` arrives or stops at the first limit."""

    def __init__(self, facts: Facts) -> None:
        self.facts = facts

    def sample(self, state: DomainState, action) -> DomainState:
        facts = self.facts
        if state.done or isinstance(action, Stop):
            return dataclasses.replace(state, hit=-1, done=True)
        if isinstance(action, MoveBase):
            return dataclasses.replace(
                state, base=action.base, base_pose=pose_of(facts.bases[action.base]), hit=-1
            )
        if isinstance(action, Grasp):
            _, grasped = facts.kin[(state.base, action.grasp)]
            return dataclasses.replace(
                state, grasp=action.grasp, arm=tuple(grasped.robot_positions), hit=-1
            )
        target = target_of(action, facts.num_waypoints)
        along, motion = facts.path.advance(state.along, target, state.theta)
        if motion >= 0:
            return dataclasses.replace(
                state, limb=tuple(facts.path.point(along)), along=along, hit=motion, done=True
            )
        # A continuous target's arm joints are unknown until IK at commit time.
        arm = (
            state.arm
            if isinstance(action, MoveLimbTo)
            else tuple(facts.reach[(state.base, state.grasp)][action.waypoint - 1])
        )
        return dataclasses.replace(
            state,
            arm=arm,
            limb=tuple(facts.path.point(target)),
            along=along,
            hit=-1,
            done=target >= 1.0 - 1e-9,
        )


class Observation(pomdp_py.ObservationModel):
    """A met limit gives a noisy, binned reading of its `theta`; else `Clear`."""

    def __init__(self, facts: Facts, config: PomdpConfig, rng: np.random.Generator) -> None:
        self.facts, self.config, self.rng = facts, config, rng

    def _noise(self, motion: int) -> float:
        return self.config.tau / self.facts.path.magnitudes[motion]

    def sample(self, next_state: DomainState, action):
        if not is_limb_move(action) or next_state.hit < 0:
            return Clear()
        motion = next_state.hit
        reading = next_state.theta[motion] + self.rng.normal(0.0, self._noise(motion))
        return Limit(motion, int(round(reading / self.config.resolution)))

    def probability(self, observation, next_state: DomainState, action) -> float:
        hit = next_state.hit if is_limb_move(action) else -1
        if isinstance(observation, Clear):
            return float(hit < 0)
        if observation.motion != hit:
            return 0.0
        # The Gaussian reading's mass in the observed bin.
        width, sigma = self.config.resolution, self._noise(hit) * math.sqrt(2.0)
        low = (observation.bin - 0.5) * width - next_state.theta[hit]
        return 0.5 * (math.erf((low + width) / sigma) - math.erf(low / sigma))


class Reward(pomdp_py.RewardModel):
    """Unit action cost; progress earned on arrival, forfeited at a limit."""

    def __init__(self, config: PomdpConfig) -> None:
        self.config = config

    def sample(self, state: DomainState, action, next_state: DomainState) -> float:
        config = self.config
        if state.done or isinstance(action, Stop):
            return 0.0
        if not is_limb_move(action):
            return -config.action_cost
        if next_state.hit >= 0:
            forfeited = config.goal_reward * state.along
            return -config.action_cost - forfeited - config.limit_penalty
        return config.goal_reward * (next_state.along - state.along) - config.action_cost


class Policy(pomdp_py.RolloutPolicy):
    """Actions whose preconditions hold, plus `Stop`."""

    def __init__(self, facts: Facts) -> None:
        self.facts = facts

    def get_all_actions(self, state: DomainState | None = None, history=None) -> list:
        facts = self.facts
        if state is None or state.done:
            return [Stop()]
        if state.grasp >= 0:
            moves = facts.open(state.base, state.grasp, state.along)
            return [MoveLimb(k) for k in moves] + [Stop()]
        if state.base >= 0:
            # CanMove is spent.
            grasps = [g for b, g in facts.kin if b == state.base and facts.open(b, g)]
            return [Grasp(g) for g in grasps] + [Stop()]
        bases = sorted({b for b, g in facts.kin if facts.open(b, g)})
        return [MoveBase(b) for b in bases] + [Stop()]

    def rollout(self, state: DomainState, history=None):
        # Stop is always legal and worth zero: a leaf's lower bound.
        return Stop()

    def random_action(self, state: DomainState, rng: np.random.Generator):
        """POMCPOW's rollout: a random legal step, or a random target then `Stop`."""
        if state.done:
            return Stop()
        if state.grasp < 0:
            actions = self.get_all_actions(state)
            moves = [a for a in actions if not isinstance(a, Stop)]
            return moves[rng.integers(len(moves))] if moves else Stop()
        if state.along >= 1.0 - 1e-9 or rng.random() < 0.5:
            return Stop()
        return MoveLimbTo(float(rng.uniform(state.along, 1.0)))

    def new_action(self, state: DomainState, tried, rng: np.random.Generator, config):
        """POMCPOW's next untried action: the finite ones first, then continuous targets.

        With the limb grasped: `Stop`, the goal if open, then targets drawn uniformly
        between the limb and the goal, away from failed ones. None when exhausted.
        """
        if state.done or state.grasp < 0:
            return next((a for a in self.get_all_actions(state) if a not in tried), None)
        facts, base, grasp = self.facts, state.base, state.grasp
        if Stop() not in tried:
            return Stop()
        if state.along >= 1.0 - 1e-9 or (base, grasp) not in facts.kin:
            return None
        goal = MoveLimbTo(1.0)
        if goal not in tried and facts.num_waypoints in facts.open(base, grasp, state.along):
            return goal
        for _ in range(10):
            along = round(float(rng.uniform(state.along, 1.0)), 3)
            action = MoveLimbTo(along)
            if (
                state.along < along < 1.0
                and action not in tried
                and facts.target_open(base, grasp, along, config.block_radius)
            ):
                return action
        return None


# --------------------------------------------------------------------- belief


def particle_belief(state: DomainState, thetas) -> pomdp_py.Particles:
    """The observed `state` once per `theta` particle."""
    return pomdp_py.Particles(
        [dataclasses.replace(state, theta=tuple(map(float, theta))) for theta in thetas]
    )


def surviving(facts: Facts, state: DomainState, action, thetas) -> list:
    """Particles under which `action` meets no limit: the belief after `Clear`."""
    target = target_of(action, facts.num_waypoints)
    if target is None:
        return list(thetas)
    return [theta for theta in thetas if facts.path.advance(state.along, target, theta)[1] < 0]


def belief_samples(
    belief: Belief | None,
    believed_theta,
    particle_limits,
    config: PomdpConfig,
    rng: np.random.Generator,
):
    """The `theta`s the tree draws from, preferring `particle_limits` (a `ParticleLimits`)."""
    if particle_limits is not None:
        return particle_limits.particles
    if belief is not None:
        return np.stack([belief.sample(rng) for _ in range(config.belief_draws)])
    return np.atleast_2d(np.asarray(believed_theta, dtype=np.float64))
