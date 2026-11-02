"""Queries: push one motion toward a target near its limit; the person's stop angle is the answer."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from able.api import InfoGatherer, Observation
from able.tasks.limb.belief import LimbBelief
from able.tasks.limb.env.human import default_magnitudes
from able.tasks.limb.env.query import DEFAULT_QUERY_MODEL, QueryModel, QuerySimulator


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
    _sims: dict = field(default_factory=dict, repr=False)

    def choose(self, belief: LimbBelief, rng: np.random.Generator) -> LimbQuery:
        g = belief.gaussian
        i = int(np.argmax(g.stddev))
        bound = default_magnitudes()[i] * max(1.0, float(g.lower[i]))
        return LimbQuery(i, float(bound + rng.uniform(-self.probe_band, self.probe_band)))

    def respond(self, theta, query: LimbQuery, rng: np.random.Generator) -> Observation:
        key = np.asarray(theta, np.float64).tobytes()
        if key not in self._sims:
            self.close()  # one person at a time
            self._sims[key] = QuerySimulator(theta, self.model)
        return Observation("query", self._sims[key].ask(query.motion_index, query.target, rng), query)

    def close(self) -> None:
        for sim in self._sims.values():
            sim.close()
        self._sims.clear()
