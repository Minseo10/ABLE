"""Friction prior and particle belief over the floors' friction coefficients."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from able.api import Observation, ParticleBelief, Prior
from able.tasks.friction.env import puckpushing_utils as pu
from able.tasks.friction.env.puckpushing_utils import SURFACE_NAMES, PuckWorld
from able.tasks.friction.query import probe_log_likelihood

# Likelihood factor when a particle's predicted hole outcome disagrees.
_HOLE_MISMATCH = 0.05


@dataclass(frozen=True)
class FrictionPrior(Prior):
    """Shuffled friction prior (levels dealt to floors at random) as an able.api Prior."""

    world: PuckWorld
    dist: pu.FrictionPrior | pu.BoxFrictionPrior = pu.DEFAULT_FRICTION_PRIOR
    num_particles: int = 500

    def sample(self, rng: np.random.Generator) -> NDArray[np.float64]:
        return self.dist.sample(rng)

    def belief(self, rng: np.random.Generator) -> FrictionBelief:
        low, high = self.dist.bounds
        return FrictionBelief(self.dist.sample(rng, self.num_particles), low, high,
                              seed=int(rng.integers(2**31)), world=self.world)


@dataclass(frozen=True)
class FrictionBelief(ParticleBelief):
    """Particles of the friction per floor; explains probes and executed pushes."""

    world: PuckWorld | None = None

    def log_likelihood(self, observation: Observation, particles: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.world is not None
        if observation.kind == "query":
            probe = observation.query
            return probe_log_likelihood(self.world.physics, probe, observation.value, particles[:, probe.floor])
        assert observation.kind == "execution", observation.kind
        v = observation.value
        return push_log_likelihood(self.world, particles, np.asarray(v["puck"]), v["angle"], v["release_speed"],
                                   np.asarray(v["stop"]), v["fell_in_hole"], v["stance"])

    def summary(self) -> dict[str, Any]:
        return {name: [float(m), float(s)] for name, m, s in zip(SURFACE_NAMES, self.mean, self.std)}


def push_log_likelihood(world: PuckWorld, particles: NDArray[np.float64], puck: NDArray[np.float64], angle: float,
                        release_speed: float, stop: NDArray[np.float64], fell: bool,
                        stance: float | None = None) -> NDArray[np.float64]:
    """Per-particle slide at the measured speed; stop error Gaussian along/across the final heading."""
    phys = world.physics
    release = world.release_point(puck, angle)
    robot = world.strike_end(world.push_station(puck, angle, stance), puck, angle)
    speed = max(release_speed, 1e-3)
    out = np.empty(len(particles))
    for k, frictions in enumerate(particles):
        result = world.slide(release, release_speed, angle, frictions, robot=robot)
        length, heading = result.distance, angle
        if len(result.path) >= 2:
            last = result.path[-1] - result.path[-2]
            if np.linalg.norm(last) > 1e-9:
                heading = math.atan2(last[1], last[0])
        along_dir = np.array([math.cos(heading), math.sin(heading)])
        error = stop - result.stop
        along = float(error @ along_dir)
        across = float(error[0] * along_dir[1] - error[1] * along_dir[0])
        var_along = phys.position_obs_noise**2 + (2.0 * length * phys.speed_obs_noise / speed) ** 2
        var_across = phys.position_obs_noise**2 + (length * phys.angle_noise) ** 2
        out[k] = -0.5 * (along**2 / var_along + across**2 / var_across) - 0.5 * math.log(var_along * var_across)
        if (result.hole is not None) != fell:
            out[k] += math.log(_HOLE_MISMATCH)
    return out
