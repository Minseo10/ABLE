"""Bedside stocking on the shared API: put requested items within a person's unknown reach."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from able.api import Observation, Step, Task
from able.tasks.bedside.belief import BedsidePrior
from able.tasks.bedside.env.bedside_stocking3d import (
    BedsideStocking3DEnvConfig,
    BedsideStocking3DObjectCentricState,
    ObjectCentricBedsideStocking3DEnv,
    reach_margin,
)
from able.tasks.bedside.env.execute import execute_plan
from able.tasks.bedside.planner import BedsidePlanner
from able.tasks.bedside.psi import BedsidePsi
from able.tasks.bedside.query import BedsideGatherer, bedside_directions

if TYPE_CHECKING:
    from able.tasks.bedside.distribution import BedsideDistribution


@dataclass(frozen=True)
class BedsideProblem:
    """A scene: which items are requested and where clutter starts."""

    seed: int  # scene reset seed


@dataclass
class BedsideState:
    """The sim state planned from."""

    obs: BedsideStocking3DObjectCentricState
    seed: int
    theta: NDArray[np.float64]  # true reach, used only to score execution


@dataclass
class BedsideTask(Task):
    name: str = "bedside"
    num_requested: int = 2
    num_clutter: int = 2
    delta: float = 0.1
    num_particles: int = 2000
    max_time: float = 120.0
    frame_stride: int = 8  # sim steps per video frame
    config: BedsideStocking3DEnvConfig = field(default_factory=BedsideStocking3DEnvConfig)
    sim: ObjectCentricBedsideStocking3DEnv = field(init=False, repr=False)
    prior: BedsidePrior = field(init=False)
    gatherer: BedsideGatherer = field(init=False)
    psi: BedsidePsi = field(init=False)
    distribution: BedsideDistribution = field(init=False)

    def __post_init__(self) -> None:
        from able.tasks.bedside.distribution import (
            BedsideDistribution,  # imports BedsideProblem from here
        )

        cfg = self.config
        self.sim = ObjectCentricBedsideStocking3DEnv(self.num_requested, self.num_clutter, config=cfg,
                                                     allow_state_access=True)
        shoulder = tuple(float(v) for v in self.sim.shoulder)
        self.prior = BedsidePrior(np.array(cfg.reach_theta_lb), np.array(cfg.reach_theta_ub), shoulder,
                                  self.num_particles)
        self.gatherer = BedsideGatherer(bedside_directions(cfg, self.sim.shoulder), cfg)  # owns its own query sim
        self.psi = BedsidePsi(cfg)
        self.distribution = BedsideDistribution()

    def make_planner(self) -> BedsidePlanner:
        return BedsidePlanner(max_time=self.max_time, sim=self.sim, delta=self.delta)

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
