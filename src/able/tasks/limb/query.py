"""Queries: push one motion toward a target near its limit; the person's stop angle is the answer."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from able.api import InfoGatherer, Observation
from able.tasks.limb.belief import LimbBelief
from able.tasks.limb.env.human import default_magnitudes
from able.tasks.limb.env.query import (
    DEFAULT_QUERY_MODEL,
    LimbQuerySimulator,
    QueryModel,
)


@dataclass(frozen=True)
class LimbQuery:
    """Push active motion `motion_index` toward `target` (a magnitude, rad)."""

    motion_index: int
    target: float


@dataclass
class LimbGatherer(InfoGatherer):
    """Query the least-certain motion with a target near its default limit (raised past any floor)."""

    probe_band: float = 0.3  # target spread around the limit, rad
    model: QueryModel = DEFAULT_QUERY_MODEL
    simulator: LimbQuerySimulator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.simulator = LimbQuerySimulator(self.model)

    def choose(self, belief: LimbBelief, rng: np.random.Generator) -> LimbQuery:
        g = belief.gaussian
        i = int(np.argmax(g.stddev))
        bound = default_magnitudes()[i] * max(1.0, float(g.lower[i]))
        return LimbQuery(i, float(bound + rng.uniform(-self.probe_band, self.probe_band)))

    def respond(self, theta, query: LimbQuery, rng: np.random.Generator,
                frames: list | None = None) -> Observation:
        return Observation("query", self.simulator.ask(theta, query.motion_index, query.target, rng, frames), query)

    def close(self) -> None:
        self.simulator.close()
