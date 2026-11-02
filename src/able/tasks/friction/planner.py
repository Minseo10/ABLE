"""POMCPOW on the puck-pushing POMDP, as an able.api Planner."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from able.api import Budget, PlanResult
from able.planners.pomcpow import POMCPOW, POMCPOWConfig
from able.tasks.friction.model import PuckPushingModel


@dataclass
class POMCPOWPlanner:
    """Plans one push from the belief's particles; psi's hook is POMCPOW's ActionProposer."""

    model: PuckPushingModel
    config: POMCPOWConfig = field(default_factory=POMCPOWConfig)
    effort_unit: str = "pomcpow_sims"

    def plan(self, state: Any, belief: Any, proposal: Any = None, budget: Budget | None = None,
             rng: np.random.Generator | None = None) -> PlanResult:
        config = self.config
        if budget is not None and budget.effort is not None:
            config = dataclasses.replace(config, num_sims=int(budget.effort))
        if budget is not None and budget.seconds is not None:
            config = dataclasses.replace(config, max_time=budget.seconds)
        model = self.model
        model.rollout_frictions = tuple(float(f) for f in belief.mean)
        particles = [model.initial_state(state.puck, state.robot, f, state.pushes) for f in belief.particles]
        solver = POMCPOW(model, config, proposer=proposal, rng=rng if rng is not None else np.random.default_rng(0))
        action, stats = solver.plan(particles)
        return PlanResult(action, stats.seconds, stats.num_sims, trace=stats.root_values,
                          info={"root_actions": stats.root_actions, "max_depth": stats.max_depth_reached})
