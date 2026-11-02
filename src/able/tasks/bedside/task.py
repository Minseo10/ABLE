"""Bedside stocking on the shared API: put requested items within a person's unknown reach."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from able.api import Observation, Step
from able.tasks.bedside.belief import ReachBelief
from able.tasks.bedside.env import (
    BedsideStocking3DEnvConfig,
    BedsideStocking3DObjectCentricState,
    ObjectCentricBedsideStocking3DEnv,
    reach_margin,
)
from able.tasks.bedside.planner.run import execute_plan
from able.tasks.bedside.psi import BedsidePsi
from able.tasks.bedside.query import ReachGatherer, bedside_directions
from able.tasks.bedside.solver import BedsidePlanner


@dataclass(frozen=True)
class BedsideProblem:
    seed: int  # scene reset seed


@dataclass
class BedsideState:
    obs: BedsideStocking3DObjectCentricState
    seed: int
    theta: NDArray[np.float64]  # true reach, used only to score execution


@dataclass(frozen=True)
class ReachPrior:
    """Uniform over the config's reach box."""

    low: NDArray[np.float64]
    high: NDArray[np.float64]
    shoulder: tuple[float, float, float]
    num_particles: int = 2000

    def sample(self, rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.uniform(self.low, self.high)

    def belief(self, rng: np.random.Generator) -> ReachBelief:
        particles = rng.uniform(self.low, self.high, (self.num_particles, len(self.low)))
        return ReachBelief(particles, self.low, self.high, shoulder=self.shoulder)


@dataclass
class BedsideTask:
    name: str = "bedside"
    num_requested: int = 2
    num_clutter: int = 2
    delta: float = 0.1
    num_particles: int = 2000
    max_time: float = 120.0
    frame_stride: int = 8  # sim steps per video frame
    config: BedsideStocking3DEnvConfig = field(default_factory=BedsideStocking3DEnvConfig)
    sim: ObjectCentricBedsideStocking3DEnv = field(init=False, repr=False)
    prior: ReachPrior = field(init=False)
    gatherer: ReachGatherer = field(init=False)
    psi: BedsidePsi = field(init=False)

    def __post_init__(self) -> None:
        cfg = self.config
        self.sim = ObjectCentricBedsideStocking3DEnv(self.num_requested, self.num_clutter, config=cfg,
                                                     allow_state_access=True)
        shoulder = tuple(float(v) for v in self.sim.shoulder)
        self.prior = ReachPrior(np.array(cfg.reach_theta_lb), np.array(cfg.reach_theta_ub), shoulder,
                                self.num_particles)
        self.gatherer = ReachGatherer(bedside_directions(cfg, self.sim.shoulder))
        self.psi = BedsidePsi(cfg)

    def make_planner(self) -> BedsidePlanner:
        return BedsidePlanner(max_time=self.max_time, sim=self.sim, delta=self.delta)

    def sample_problem(self, rng: np.random.Generator) -> BedsideProblem:
        return BedsideProblem(int(rng.integers(2**31 - 1)))

    def reset(self, problem: BedsideProblem, theta: NDArray[np.float64], rng: np.random.Generator) -> BedsideState:
        obs, _ = self.sim.reset(seed=problem.seed)
        theta = np.array(theta, float)
        self.sim.reach_theta = theta.copy()
        return BedsideState(obs, problem.seed, theta)

    def execute(self, state: BedsideState, plan: Any, rng: np.random.Generator, frames: list | None = None) -> Step:
        """Run the plan; observe whether each placed requested item is reached. Appends RGB frames if given."""
        sim = self.sim
        sim.reset(options={"init_state": state.obs})
        sim.reach_theta = state.theta.copy()
        success = bool(execute_plan(sim, plan, seed=int(rng.integers(2**31 - 1)), frames=frames,
                                    stride=self.frame_stride))
        observations, reachable = [], {}
        for o in sim.requested_objects:
            body = sim._object_name_to_pybullet_id(o)  # pylint: disable=protected-access
            if sim._grasped_object == o or not sim._get_surfaces_supporting_object(body):  # pylint: disable=protected-access
                continue
            x = sim.grasp_point(o)
            reached = bool(reach_margin(x, state.theta, sim.shoulder) >= 0)
            observations.append(Observation("execution", reached, tuple(float(v) for v in x)))
            reachable[o] = sim.item_reachable(o)
        new = BedsideState(sim._get_obs(), state.seed, state.theta)  # pylint: disable=protected-access
        return Step(new, observations, done=True, success=success, info={"reachable": reachable})
