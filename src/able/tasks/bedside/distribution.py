"""Bedside scenes: each problem is a reset seed for the sim's item and clutter layout."""

from __future__ import annotations

import numpy as np

from able.api import TaskDistribution
from able.tasks.bedside.task import BedsideProblem


class BedsideDistribution(TaskDistribution):
    """Uniform over scene seeds."""

    def sample(self, rng: np.random.Generator) -> BedsideProblem:
        return BedsideProblem(int(rng.integers(2**31 - 1)))
