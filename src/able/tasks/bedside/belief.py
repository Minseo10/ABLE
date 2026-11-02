"""Prior and particle belief over the reach ellipsoid theta = (r_x, r_y, r_z)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from able.api import Observation, ParticleBelief, Prior
from able.tasks.bedside.env.bedside_stocking3d import reach_margin

EXECUTION_SCALE = 0.03  # margin scale of P(reached | theta), shared by queries and deliveries


def reach_distance(direction: NDArray, theta: NDArray) -> NDArray[np.float64]:
    """r(d, theta) = 1 / ||d / theta||: distance to the ellipsoid along unit `direction`."""
    return 1.0 / np.linalg.norm(np.asarray(direction, float) / np.asarray(theta, float), axis=-1)


def reached_log_likelihood(points: NDArray, reached: NDArray, particles: NDArray, shoulder: NDArray,
                           scale: float = EXECUTION_SCALE) -> NDArray[np.float64]:
    """sum_i log sigmoid(+-m(x_i, theta) / scale) per particle; points (k, 3), reached (k,)."""
    m = reach_margin(np.asarray(points, float)[:, None], particles[None], np.asarray(shoulder, float))
    z = np.where(np.asarray(reached, bool)[:, None], m, -m) / scale
    return -np.logaddexp(0.0, -z).sum(0)


@dataclass(frozen=True)
class BedsideBelief(ParticleBelief):
    """One person model: P(reached x | theta) = sigmoid(m(x, theta) / scale), for queries and deliveries."""

    shoulder: tuple[float, float, float] = (0.0, 0.0, 0.0)
    execution_scale: float = EXECUTION_SCALE

    def log_likelihood(self, observation: Observation, particles: NDArray[np.float64]) -> NDArray[np.float64]:
        if observation.kind == "query":  # value: [(point, reached), ...]
            if not observation.value:
                return np.zeros(len(particles))
            points, reached = zip(*observation.value)
        elif observation.kind == "execution":
            points, reached = [observation.query], [observation.value]
        else:
            raise ValueError(f"Unknown observation kind: {observation.kind}")
        return reached_log_likelihood(points, reached, particles, self.shoulder, self.execution_scale)


@dataclass(frozen=True)
class BedsidePrior(Prior):
    """Uniform over the config's reach box."""

    low: NDArray[np.float64]
    high: NDArray[np.float64]
    shoulder: tuple[float, float, float]
    num_particles: int = 2000

    def sample(self, rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.uniform(self.low, self.high)

    def belief(self, rng: np.random.Generator) -> BedsideBelief:
        particles = rng.uniform(self.low, self.high, (self.num_particles, len(self.low)))
        return BedsideBelief(particles, self.low, self.high, shoulder=self.shoulder)
