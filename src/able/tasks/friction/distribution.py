"""The friction task's problem distribution: an env reset seed."""

from __future__ import annotations

import numpy as np

from able.api import TaskDistribution
from able.tasks.friction.task import FrictionProblem


class FrictionDistribution(TaskDistribution):
    """Problems differ only in the env reset seed; the maze, start and goal are fixed by the level."""

    def sample(self, rng: np.random.Generator) -> FrictionProblem:
        return FrictionProblem(int(rng.integers(2**31)))
