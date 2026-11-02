"""Tasks, info gathering, planners and psi: the per-task pieces the pipeline calls."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from able.api.belief import Belief, Observation, Prior


class InfoGatherer(ABC):
    """Asks the person something before planning."""

    @abstractmethod
    def choose(self, belief: Belief, rng: np.random.Generator) -> Any:
        """The next query, or None to ask nothing."""

    @abstractmethod
    def respond(self, theta: NDArray[np.float64], query: Any, rng: np.random.Generator) -> Observation:
        """The simulated person's answer."""


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


class Planner(ABC):
    effort_unit: str = ""  # what PlanResult.effort counts

    @abstractmethod
    def plan(self, state: Any, belief: Belief, proposal: Any = None, budget: Budget | None = None,
             rng: np.random.Generator | None = None) -> PlanResult:
        """Plan from `state` under `belief`; `proposal` is psi's hook (None = plain)."""


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


class PsiSpec(ABC):
    """What psi is for a task: its features, labels and planner hook."""

    @abstractmethod
    def examples(self, episode: Any, belief: Belief) -> Examples:
        """Rows from a solved episode, conditioned on `belief`."""

    @abstractmethod
    def proposal(self, learner: Any, state: Any, belief: Belief) -> Any:
        """Planner hook built from a fitted learner."""


class TaskDistribution(ABC):
    """P(problem): goals and scenes, independent of theta."""

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> Any: ...


class Task(ABC):
    """A task: who the person is (prior), what to ask, what to plan, and psi.

    Subclasses set `name`, `prior`, `gatherer`, `psi` and `distribution`.
    """

    name: str
    prior: Prior
    gatherer: InfoGatherer
    psi: PsiSpec
    distribution: TaskDistribution

    @abstractmethod
    def make_planner(self) -> Planner: ...

    def sample_problem(self, rng: np.random.Generator) -> Any:
        """A goal and scene from `distribution`."""
        return self.distribution.sample(rng)

    @abstractmethod
    def reset(self, problem: Any, theta: NDArray[np.float64], rng: np.random.Generator) -> Any:
        """Start state for a person with `theta`."""

    @abstractmethod
    def execute(self, state: Any, plan: Any, rng: np.random.Generator, frames: list | None = None) -> Step:
        """Run `plan` on the person; append RGB frames to `frames` if given."""
