"""PDDLStream for the bedside: usable poses need P_b(reachable) >= 1 - delta over belief particles."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from pddlstream.language.constants import PDDLProblem

from able.api import Belief
from able.planners.pddlstream import PDDLStreamPlanner
from able.tasks.bedside.env import ObjectCentricBedsideStocking3DEnv
from able.tasks.bedside.planner.run import build_stream_context, create_problem


@dataclass
class BedsidePlanner(PDDLStreamPlanner):
    sim: ObjectCentricBedsideStocking3DEnv | None = None
    delta: float = 0.1
    num_particles: int = 256

    def particles(self, belief: Belief, rng: np.random.Generator) -> np.ndarray:
        particles = getattr(belief, "particles", None)
        if particles is None:
            return belief.sample(rng, self.num_particles)
        if len(particles) <= self.num_particles:
            return np.asarray(particles)
        return particles[rng.choice(len(particles), self.num_particles, replace=False)]

    def problem(self, state: Any, belief: Belief, proposal: Any, rng: np.random.Generator
                ) -> tuple[PDDLProblem, Counter]:
        self.sim.reset(options={"init_state": state.obs})
        ctx = build_stream_context(self.sim, state.obs, theta_particles=self.particles(belief, rng),
                                   delta=self.delta, seed=int(rng.integers(2**31 - 1)), proposal=proposal)
        return create_problem(self.sim, state.obs, ctx=ctx), ctx.calls

    def trace(self, plan) -> list:
        """(item, place position, surface) of each place action."""
        return [(a[0], tuple(a[1].pose.position), a[1].surface) for n, a in plan or [] if n == "place"]
