"""Info gathering: which direction and distances to ask about; the robot presents them in env/query.py."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from able.api import Belief, InfoGatherer, Observation
from able.tasks.bedside.belief import EXECUTION_SCALE, reach_distance
from able.tasks.bedside.env.bedside_stocking3d import (
    BedsideStocking3DEnvConfig,
    ObjectCentricBedsideStocking3DEnv,
)
from able.tasks.bedside.env.query import BedsideQuerySimulator
from able.tasks.bedside.planning.stream import (
    NOMINAL_HALF_HEIGHT,
)


def bedside_directions(config: BedsideStocking3DEnvConfig, shoulder: NDArray) -> NDArray[np.float64]:
    """Unit vectors from the shoulder to the top of a nominal item on every bedside slot."""
    tops = np.concatenate([config.region_slots(r) for r in config.bedside_regions])
    d = tops + [0.0, 0.0, 2 * NOMINAL_HALF_HEIGHT] - np.asarray(shoulder)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


@dataclass(frozen=True, eq=False)
class BedsideQuery:
    """Present the gripper at shoulder + s * direction for each s in `distances` (outward)."""

    direction: NDArray[np.float64]  # unit vector from the shoulder
    distances: tuple[float, ...]  # m


@dataclass
class BedsideGatherer(InfoGatherer):
    """Asks along the direction where the belief disagrees most about the reach distance, at its quantiles.

    Queries run in the gatherer's own sim (an empty copy of the scene), never the planner's."""

    directions: NDArray[np.float64]  # (n, 3) unit vectors
    config: BedsideStocking3DEnvConfig = field(default_factory=BedsideStocking3DEnvConfig)
    num_points: int = 4  # distances per query, at belief quantiles
    noisy: bool = True  # answer ~ Bernoulli(sigmoid(m / scale)), the belief's own model; else m >= 0
    scale: float = EXECUTION_SCALE
    stop_at_miss: bool = True  # stop presenting after the first "no"
    grid_step: float = 0.025  # m, spacing of the candidate distances along each direction
    min_entropy: float = 0.1  # nats; ask nothing once no query's answers are this uncertain
    num_particles: int = 1000
    frame_stride: int = 8
    seed: int = 0
    simulator: BedsideQuerySimulator = field(init=False, repr=False, compare=False)
    _known: dict[tuple[int, int], bool] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.simulator = BedsideQuerySimulator(self.config, self.scale, self.noisy, self.stop_at_miss,
                                               self.frame_stride, self.seed)

    @property
    def sim(self) -> ObjectCentricBedsideStocking3DEnv:
        """The query simulator's sim (never the planner's)."""
        return self.simulator.sim

    @property
    def last_seconds(self) -> float:
        return self.simulator.last_seconds

    @property
    def distances(self) -> NDArray[np.float64]:
        """Candidate distances along every direction, spanning the prior's reach."""
        lb, ub = np.array(self.config.reach_theta_lb), np.array(self.config.reach_theta_ub)
        r = reach_distance(self.directions, np.stack([lb, ub])[:, None])
        return np.arange(r[0].min(), r[1].max() + 1e-9, self.grid_step)

    def presentable(self, k: int, j: int) -> bool:
        """Static collision-free IK for distance j along direction k (memoized)."""
        if (k, j) not in self._known:
            point = self.sim.shoulder + self.distances[j] * self.directions[k]
            self._known[k, j] = self.simulator.can_present(point)
        return self._known[k, j]

    def choose(self, belief: Belief, rng: np.random.Generator) -> BedsideQuery | None:
        """Per direction, the presentable distances nearest the belief's reach quantiles; the direction whose
        answers are most uncertain (sum of binary entropies) wins. Unchecked distances count as presentable
        until the winner's are checked."""
        particles = getattr(belief, "particles", None)
        if particles is None:
            particles = belief.sample(rng, self.num_particles)
        dist = self.distances
        levels = np.arange(1, self.num_points + 1) / (self.num_points + 1)
        p = (reach_distance(self.directions[:, None], particles[None])[..., None] > dist).mean(1)  # P(reached)
        while True:
            best, score = None, self.min_entropy
            for k in range(len(self.directions)):
                idx = np.array([j for j in range(len(dist)) if self._known.get((k, j), True)], int)
                if not len(idx):
                    continue
                picked = idx[np.unique(np.abs(p[k, idx][:, None] - levels).argmin(0))]
                q = np.clip(p[k, picked], 1e-9, 1 - 1e-9)
                h = float(-(q * np.log(q) + (1 - q) * np.log(1 - q)).sum())
                if h > score:
                    best, score = (k, picked), h
            if best is None:
                return None
            k, picked = best
            unknown = [j for j in picked if (k, j) not in self._known]
            if not unknown:
                return BedsideQuery(self.directions[k].copy(), tuple(float(dist[j]) for j in picked))
            for j in unknown:
                self.presentable(k, j)

    def respond(self, theta: NDArray[np.float64], query: BedsideQuery, rng: np.random.Generator,
                frames: list | None = None) -> Observation:
        """Observation value: [(gripper point, reached), ...] at the points the robot could present."""
        outcomes = self.simulator.ask(theta, query.direction, query.distances, rng, frames)
        return Observation("query", outcomes, np.asarray(query.direction, float))
