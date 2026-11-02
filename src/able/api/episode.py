"""The plan-execute-observe loop shared by every task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from able.api.belief import Belief
from able.api.task import Budget, Planner, PlanResult, Step, Task


@dataclass
class EpisodeStep:
    state: Any  # state planned from
    belief: Belief  # belief planned under
    result: PlanResult
    step: Step | None  # None if planning failed


@dataclass
class Episode:
    problem: Any
    steps: list[EpisodeStep] = field(default_factory=list)
    belief: Belief | None = None  # after the last observation

    @property
    def success(self) -> bool:
        return bool(self.steps) and self.steps[-1].step is not None and self.steps[-1].step.success

    @property
    def plan_seconds(self) -> float:
        return sum(s.result.seconds for s in self.steps)

    @property
    def effort(self) -> float:
        return sum(s.result.effort for s in self.steps)

    @property
    def observations(self) -> list:
        return [o for s in self.steps if s.step for o in s.step.observations]


def run_episode(
    task: Task,
    planner: Planner,
    problem: Any,
    theta: NDArray[np.float64],
    belief: Belief,
    rng: np.random.Generator,
    proposal_fn: Callable[[Any, Belief], Any] | None = None,
    budget: Budget | None = None,
    max_steps: int = 50,
    fallback: bool = True,
) -> Episode:
    """Plan, execute, update the belief until done; `fallback` replans plain if psi fails (both costs charged)."""
    episode = Episode(problem)
    state = task.reset(problem, theta, rng)
    for _ in range(max_steps):
        proposal = proposal_fn(state, belief) if proposal_fn else None
        result = planner.plan(state, belief, proposal=proposal, budget=budget, rng=rng)
        if not result.success and proposal is not None and fallback:
            plain = planner.plan(state, belief, proposal=None, budget=budget, rng=rng)
            plain.seconds += result.seconds
            plain.effort += result.effort
            plain.info = {**plain.info, "fallback": True}
            result = plain
        if not result.success:
            episode.steps.append(EpisodeStep(state, belief, result, None))
            break
        step = task.execute(state, result.plan, rng)
        episode.steps.append(EpisodeStep(state, belief, result, step))
        for obs in step.observations:
            belief = belief.update(obs)
        state = step.state
        if step.done:
            break
    episode.belief = belief
    return episode
