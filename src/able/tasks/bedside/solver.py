"""PDDLStream planner under a particle belief over the reach ellipsoid."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from able.api import Belief, Budget, PlanResult
from able.tasks.bedside.env import ObjectCentricBedsideStocking3DEnv
from able.tasks.bedside.planner.run import build_stream_context, plan_bedside


@dataclass
class PDDLStreamPlanner:
    """Usable poses need P_b(reachable) >= 1 - delta over a subsample of the belief's particles."""

    sim: ObjectCentricBedsideStocking3DEnv
    delta: float = 0.1
    max_time: float = 120.0
    num_particles: int = 256
    effort_unit: str = "stream_calls"

    def particles(self, belief: Belief, rng: np.random.Generator) -> np.ndarray:
        particles = getattr(belief, "particles", None)
        if particles is None:
            return belief.sample(rng, self.num_particles)
        if len(particles) <= self.num_particles:
            return np.asarray(particles)
        return particles[rng.choice(len(particles), self.num_particles, replace=False)]

    def plan(self, state: Any, belief: Belief, proposal: Any = None, budget: Budget | None = None,
             rng: np.random.Generator | None = None) -> PlanResult:
        rng = rng if rng is not None else np.random.default_rng(state.seed)
        max_time = budget.seconds if budget is not None and budget.seconds is not None else self.max_time
        self.sim.reset(options={"init_state": state.obs})
        particles = self.particles(belief, rng)
        ctx = build_stream_context(self.sim, state.obs, theta_particles=particles, delta=self.delta,
                                   seed=int(rng.integers(2**31 - 1)), proposal=proposal)
        t0 = time.perf_counter()
        plan = plan_bedside(self.sim, state.obs, max_time=max_time, ctx=ctx)
        seconds = time.perf_counter() - t0
        trace = [(a[0], tuple(a[1].pose.position), a[1].surface) for n, a in plan or [] if n == "place"]
        info = {"calls": dict(ctx.calls), "plan_length": None if plan is None else len(plan),
                "num_particles": len(particles)}
        return PlanResult(plan, seconds, float(sum(ctx.calls.values())), trace, info)
