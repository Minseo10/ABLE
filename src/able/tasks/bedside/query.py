"""Reach queries: the robot holds a target along direction d and the person reaches as far as they can."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from able.api import Belief, InfoGatherer, Observation
from able.tasks.bedside.belief import QUERY_NOISE, reach_distance
from able.tasks.bedside.env.bedside_stocking3d import BedsideStocking3DEnvConfig
from able.tasks.bedside.planning.stream import NOMINAL_HALF_HEIGHT


def bedside_directions(config: BedsideStocking3DEnvConfig, shoulder: NDArray) -> NDArray[np.float64]:
    """Unit vectors from the shoulder to the top of a nominal item on every bedside slot."""
    tops = np.concatenate([config.region_slots(r) for r in config.bedside_regions])
    d = tops + [0.0, 0.0, 2 * NOMINAL_HALF_HEIGHT] - np.asarray(shoulder)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


@dataclass(frozen=True, eq=False)
class BedsideQuery:
    """Hold a target along `direction`; the person reports how far they reach."""

    direction: NDArray[np.float64]  # unit vector from the shoulder


@dataclass
class BedsideGatherer(InfoGatherer):
    """Asks along the direction where the belief disagrees most about the reach distance."""

    directions: NDArray[np.float64]  # (n, 3) unit vectors
    noise: float = QUERY_NOISE
    min_std: float = 0.01  # ask nothing once every direction's reach std is below this (m)
    num_particles: int = 1000

    def choose(self, belief: Belief, rng: np.random.Generator) -> BedsideQuery | None:
        particles = getattr(belief, "particles", None)
        if particles is None:
            particles = belief.sample(rng, self.num_particles)
        std = reach_distance(self.directions[:, None], particles[None]).std(1)
        k = int(std.argmax())
        return None if std[k] < self.min_std else BedsideQuery(self.directions[k].copy())

    def respond(self, theta: NDArray[np.float64], query: BedsideQuery, rng: np.random.Generator) -> Observation:
        """Observation.query is the direction itself."""
        r = float(reach_distance(query.direction, theta)) + rng.normal(0.0, self.noise)
        return Observation("query", r, np.asarray(query.direction, float))
