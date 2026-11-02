"""Particle belief over the reach ellipsoid theta = (r_x, r_y, r_z)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from able.api import Observation, ParticleBelief
from able.tasks.bedside.env import reach_margin

QUERY_NOISE = 0.02  # m, std of the reported reach distance
EXECUTION_SCALE = 0.03  # margin scale of P(reached | theta)


def reach_distance(direction: NDArray, theta: NDArray) -> NDArray[np.float64]:
    """r(d, theta) = 1 / ||d / theta||: distance to the ellipsoid along unit `direction`."""
    return 1.0 / np.linalg.norm(np.asarray(direction, float) / np.asarray(theta, float), axis=-1)


@dataclass(frozen=True)
class ReachBelief(ParticleBelief):
    """Queries report r(d, theta) + noise; executions report whether an item was reached."""

    shoulder: tuple[float, float, float] = (0.0, 0.0, 0.0)
    query_noise: float = QUERY_NOISE
    execution_scale: float = EXECUTION_SCALE

    def log_likelihood(self, observation: Observation, particles: NDArray[np.float64]) -> NDArray[np.float64]:
        if observation.kind == "query":
            r = reach_distance(observation.query, particles)
            return -0.5 * ((observation.value - r) / self.query_noise) ** 2
        if observation.kind == "execution":
            m = reach_margin(np.asarray(observation.query, float), particles, np.asarray(self.shoulder))
            z = m / self.execution_scale if observation.value else -m / self.execution_scale
            return -np.logaddexp(0.0, -z)
        raise ValueError(f"Unknown observation kind: {observation.kind}")
