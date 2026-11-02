"""The friction task: push a puck through the threeroutes maze over floors of unknown friction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from able.api import Observation, Step
from able.planners.pomcpow import POMCPOWConfig
from able.tasks.friction.belief import FrictionBelief
from able.tasks.friction.env import puckpushing_utils as pu
from able.tasks.friction.levels import level_world, make_level_env
from able.tasks.friction.model import PuckPushingModel, PuckRewards
from able.tasks.friction.planner import POMCPOWPlanner
from able.tasks.friction.psi import FrictionPsi
from able.tasks.friction.query import ProbeGatherer
from able.tasks.friction.video import Recorder


@dataclass(frozen=True)
class FrictionPrior:
    """Shuffled friction prior (levels dealt to floors at random) as an able.api Prior."""

    world: pu.PuckWorld
    dist: pu.FrictionPrior | pu.BoxFrictionPrior = pu.DEFAULT_FRICTION_PRIOR
    num_particles: int = 500

    def sample(self, rng: np.random.Generator) -> NDArray[np.float64]:
        return self.dist.sample(rng)

    def belief(self, rng: np.random.Generator) -> FrictionBelief:
        low, high = self.dist.bounds
        return FrictionBelief(self.dist.sample(rng, self.num_particles), low, high,
                              seed=int(rng.integers(2**31)), world=self.world)


@dataclass(frozen=True)
class FrictionProblem:
    seed: int  # env reset seed


@dataclass(frozen=True)
class FrictionState:
    puck: NDArray[np.float64]
    robot: NDArray[np.float64]
    pushes: int = 0


@dataclass
class FrictionTask:
    name: str = "friction"
    env: str = "2d"  # or "3d"
    level: str = "L2"
    num_particles: int = 500
    max_pushes: int = 15
    stance_mode: str = "grid"
    config: POMCPOWConfig = field(default_factory=lambda: POMCPOWConfig(max_depth=8))
    rewards: PuckRewards = field(default_factory=PuckRewards)

    def __post_init__(self) -> None:
        self.world = level_world(self.level)
        self.model = PuckPushingModel(self.world, rewards=self.rewards, max_pushes=self.max_pushes,
                                      stance_mode=self.stance_mode)
        self.prior = FrictionPrior(self.world, self.model.friction_prior, self.num_particles)
        self.gatherer = ProbeGatherer(self.world.physics)
        self.psi = FrictionPsi(self.model)
        self._env: Any = None
        self._recorder: Any = None

    @property
    def environment(self) -> Any:
        """Simulator, built lazily."""
        if self._env is None:
            self._env = make_level_env(self.env, self.level)
        return self._env

    @property
    def recorder(self) -> Recorder:
        """Frame recorder for `environment`, built lazily."""
        if self._recorder is None:
            self._recorder = Recorder(self.environment, self.env, header=f"{self.env.upper()} {self.level}")
        return self._recorder

    def make_planner(self) -> POMCPOWPlanner:
        return POMCPOWPlanner(self.model, self.config)

    def sample_problem(self, rng: np.random.Generator) -> FrictionProblem:
        return FrictionProblem(int(rng.integers(2**31)))

    def reset(self, problem: FrictionProblem, theta: NDArray[np.float64], rng: np.random.Generator) -> FrictionState:
        del rng
        self.environment.reset(seed=problem.seed, options={"frictions": tuple(float(f) for f in theta)})
        start = self.world.puck_map
        return FrictionState(np.array(start.puck_start, np.float64), np.array(start.robot_start, np.float64))

    def execute(self, state: FrictionState, plan: pu.Push, rng: np.random.Generator,
                frames: list | None = None) -> Step:
        del rng
        action = np.array([plan.angle, plan.speed] + ([] if plan.stance is None else [plan.stance]))
        if frames is None:
            _, _, terminated, _, info = self.environment.step(action)
        else:
            _, _, terminated, _, info = self.recorder.record(frames, action, state.pushes + 1)
        stop = np.asarray(info["stop_position"], np.float64)
        feasible = bool(info["push_feasible"])
        record = {
            "puck": state.puck.tolist(), "angle": plan.angle, "speed": plan.speed, "stance": plan.stance,
            "feasible": feasible, "release_speed": float(info["release_speed"]), "stop": stop.tolist(),
            "fell_in_hole": bool(info["fell_in_hole"]), "goal_reached": bool(info["goal_reached"]),
            "drive_distance": float(info["drive_distance"]),
        }
        robot = state.robot
        if feasible:
            robot = self.world.push_station(state.puck, plan.angle, plan.stance) + (
                self.world.physics.accel_distance * plan.direction)
        pushes = state.pushes + 1
        # A noisy measurement can land the puck slightly inside a wall.
        new = FrictionState(self.world.nearest_free_puck_position(stop), robot, pushes)
        return Step(new, [Observation("execution", record)] if feasible else [],
                    done=bool(terminated) or pushes >= self.max_pushes, success=record["goal_reached"],
                    info={**record, "reward": self.reward(record)})

    def reward(self, record: dict) -> float:
        r, drive = self.rewards, record["drive_distance"]
        reward = -r.push_cost - r.drive_cost * (drive if math.isfinite(drive) else 0.0)  # as the model does
        if record["fell_in_hole"]:
            reward -= r.hole_penalty
        elif record["goal_reached"]:
            reward += r.goal_reward
        return reward
