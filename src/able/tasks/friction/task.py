"""The friction task: push a puck through the threeroutes maze over floors of unknown friction."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from able.api import Observation, Step, Task
from able.planners.pomcpow import POMCPOWConfig
from able.tasks.friction.belief import FrictionPrior
from able.tasks.friction.env import puckpushing_utils as pu
from able.tasks.friction.env.levels import level_world, make_level_env
from able.tasks.friction.env.push_recorder import Recorder
from able.tasks.friction.planner import FrictionPlanner
from able.tasks.friction.planning.model import PuckPushingModel, PuckRewards
from able.tasks.friction.psi import FrictionPsi
from able.tasks.friction.query import FrictionGatherer, push_record


@dataclass(frozen=True)
class FrictionProblem:
    """What FrictionDistribution samples."""

    seed: int  # env reset seed


@dataclass(frozen=True)
class FrictionState:
    """Puck and robot positions, and pushes so far."""

    puck: NDArray[np.float64]
    robot: NDArray[np.float64]
    pushes: int = 0


@dataclass
class FrictionTask(Task):
    """Push the puck to the goal; theta is the friction of each floor."""

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
        # Probes run in their own simulator, never `environment`.
        self.gatherer = FrictionGatherer(self.world, functools.partial(make_level_env, self.env, self.level),
                                         self.prior.dist.bounds[0], self.env, f"{self.env.upper()} {self.level}")
        self.psi = FrictionPsi(self.model)
        from able.tasks.friction.distribution import (
            FrictionDistribution,  # it imports FrictionProblem from here
        )

        self.distribution = FrictionDistribution()
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

    def make_planner(self) -> FrictionPlanner:
        return FrictionPlanner(self.config, model=self.model)

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
        record = push_record(info, state.puck, plan)
        stop, feasible = np.asarray(record["stop"]), record["feasible"]
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
