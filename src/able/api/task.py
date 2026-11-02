"""Tasks, info gathering, planners and psi: the per-task pieces the pipeline calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from able.api.belief import Belief, Observation, Prior


@runtime_checkable
class InfoGatherer(Protocol):
    """Asks the person something before planning."""

    def choose(self, belief: Belief, rng: np.random.Generator) -> Any:
        """The next query, or None to ask nothing."""
        ...

    def respond(self, theta: NDArray[np.float64], query: Any, rng: np.random.Generator) -> Observation:
        """The simulated person's answer."""
        ...


@dataclass(frozen=True)
class Budget:
    """Planner limits; None means the planner's default."""

    seconds: float | None = None
    effort: float | None = None


@dataclass
class PlanResult:
    plan: Any  # None if planning failed
    seconds: float
    effort: float  # planner-native: MPC rollouts, POMCPOW sims, PDDLStream stream calls
    trace: Any = None  # whatever psi needs for labels
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.plan is not None


@runtime_checkable
class Planner(Protocol):
    effort_unit: str

    def plan(self, state: Any, belief: Belief, proposal: Any = None, budget: Budget | None = None,
             rng: np.random.Generator | None = None) -> PlanResult:
        """Plan from `state` under `belief`; `proposal` is psi's hook (None = plain)."""
        ...


@dataclass
class Step:
    """One execution of a plan (all of it, or one action for closed-loop tasks)."""

    state: Any  # state after executing
    observations: list[Observation]
    done: bool
    success: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Examples:
    """Training rows for psi; `groups` keeps rows of one plan together in held-out splits."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    groups: NDArray[np.int64]

    @classmethod
    def empty(cls, x_dim: int, y_dim: int) -> Examples:
        return cls(np.zeros((0, x_dim)), np.zeros((0, y_dim)), np.zeros(0, np.int64))

    def __len__(self) -> int:
        return len(self.x)


@runtime_checkable
class PsiSpec(Protocol):
    """What psi is for a task: its features, labels and planner hook."""

    def examples(self, episode: Any, belief: Belief) -> Examples:
        """Rows from a solved episode, conditioned on `belief`."""
        ...

    def proposal(self, learner: Any, state: Any, belief: Belief) -> Any:
        """Planner hook built from a fitted learner."""
        ...


@runtime_checkable
class Task(Protocol):
    name: str
    prior: Prior
    gatherer: InfoGatherer
    psi: PsiSpec

    def make_planner(self) -> Planner: ...

    def sample_problem(self, rng: np.random.Generator) -> Any:
        """A goal and scene, independent of theta."""
        ...

    def reset(self, problem: Any, theta: NDArray[np.float64], rng: np.random.Generator) -> Any:
        """Start state for a person with `theta`."""
        ...

    def execute(self, state: Any, plan: Any, rng: np.random.Generator) -> Step: ...
