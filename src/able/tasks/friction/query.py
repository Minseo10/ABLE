"""Info gathering: a probe push on a flat test patch of one floor colour."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from able.api import Observation
from able.tasks.friction.env.puckpushing_utils import GRAVITY, PuckPhysics

# Quadrature grid over standard-normal speed errors.
_NODES = np.linspace(-5.0, 5.0, 201)
_WEIGHTS = np.exp(-0.5 * _NODES**2) / np.exp(-0.5 * _NODES**2).sum()


@dataclass(frozen=True)
class Probe:
    """Push the puck at `speed` on a patch of floor `floor` and measure the slide."""

    floor: int
    speed: float


def probe_distance(speed, mu) -> NDArray[np.float64]:
    """Coulomb slide length of a puck released at `speed` (PuckWorld.slide on open floor)."""
    return speed**2 / (2.0 * np.asarray(mu) * GRAVITY)


def probe_log_likelihood(physics: PuckPhysics, probe: Probe, distance: float,
                         mu: NDArray[np.float64]) -> NDArray[np.float64]:
    """log p(distance | mu): speed noise integrated out, Gaussian position noise."""
    speeds = np.maximum(probe.speed * (1.0 + physics.speed_noise * _NODES), 0.0)
    pred = probe_distance(speeds[None, :], np.asarray(mu)[:, None])
    sigma = physics.position_obs_noise
    log_p = -0.5 * ((distance - pred) / sigma) ** 2 + np.log(_WEIGHTS)
    top = log_p.max(1)
    return top + np.log(np.exp(log_p - top[:, None]).sum(1)) - np.log(sigma * np.sqrt(2 * np.pi))


@dataclass(frozen=True)
class ProbeGatherer:
    """Probes the floor whose friction the belief is least sure of."""

    physics: PuckPhysics
    speed: float = 0.8
    min_std: float = 0.005  # about the resampling jitter

    def choose(self, belief, rng: np.random.Generator) -> Probe | None:
        del rng
        std = belief.std
        floor = int(np.argmax(std))
        return Probe(floor, self.speed) if std[floor] > self.min_std else None

    def respond(self, theta: NDArray[np.float64], query: Probe, rng: np.random.Generator) -> Observation:
        phys = self.physics
        speed = max(query.speed * (1.0 + rng.normal(0.0, phys.speed_noise)), 0.0)
        distance = float(probe_distance(speed, theta[query.floor])) + rng.normal(0.0, phys.position_obs_noise)
        return Observation("query", float(distance), query)
