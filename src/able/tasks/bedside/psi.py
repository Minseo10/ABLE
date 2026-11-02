"""Psi for bedside stocking: where each requested item ends up, given the scene and belief."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from able.api import Belief, Examples
from able.tasks.bedside.env import BedsideStocking3DEnvConfig

MAX_REQUESTED = 8  # req0..req7, the largest registered variant


@dataclass(frozen=True)
class PlaceProposal:
    """Flow samples of each requested item's placement; ranks the stream's candidate slots."""

    samples: dict[str, NDArray[np.float64]]  # item -> (k, 3)

    def order(self, o: str, points: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.int64]:
        """Slots nearest a sample, most-sampled first, then the rest at random."""
        if o not in self.samples:
            return rng.permutation(len(points))
        near = np.linalg.norm(self.samples[o][:, None] - points[None], axis=-1).argmin(1)
        ids, first, counts = np.unique(near, return_index=True, return_counts=True)
        ranked = ids[np.lexsort((first, -counts))]
        rest = rng.permutation(np.setdiff1d(np.arange(len(points)), ranked))
        return np.concatenate([ranked, rest]).astype(np.int64)


@dataclass
class BedsidePsi:
    """One row per requested item: item one-hot (MAX_REQUESTED), clutter occupancy of every
    bedside region slot (config.region_slots, in bedside_regions order), belief mean and std.
    Label: the item's final placement position in the plan."""

    config: BedsideStocking3DEnvConfig = field(default_factory=BedsideStocking3DEnvConfig)
    k: int = 16

    @property
    def slots(self) -> NDArray[np.float64]:
        return np.concatenate([self.config.region_slots(r) for r in self.config.bedside_regions])

    def features(self, state: Any, belief: Belief, item: str) -> NDArray[np.float64]:
        onehot = np.zeros(MAX_REQUESTED)
        onehot[int(item.removeprefix("req"))] = 1.0
        clutter = np.array([state.obs.get_object_pose(c).position[:2] for c in state.obs.clutter_objects])
        occupied = np.zeros(len(self.slots))
        if len(clutter):
            gap = np.abs(self.slots[:, None, :2] - clutter[None]).max(-1)
            occupied = (gap < self.config.slot_spacing / 2).any(1).astype(float)
        return np.concatenate([onehot, occupied, belief.features()])

    def examples(self, episode: Any, belief: Belief) -> Examples:
        x, y, groups = [], [], []
        for i, s in enumerate(episode.steps):
            if not s.result.success:
                continue
            final = {item: xyz for item, xyz, _ in s.result.trace}
            for item in s.state.obs.requested_objects:
                if item in final:
                    x.append(self.features(s.state, belief, item))
                    y.append(final[item])
                    groups.append(i)
        if not x:
            return Examples.empty(MAX_REQUESTED + len(self.slots) + len(belief.features()), 3)
        return Examples(np.array(x), np.array(y, float), np.array(groups, np.int64))

    def proposal(self, learner: Any, state: Any, belief: Belief) -> PlaceProposal:
        items = state.obs.requested_objects
        x = np.stack([self.features(state, belief, o) for o in items])
        draws = learner.sample(x, k=self.k, seed=state.seed)
        return PlaceProposal(dict(zip(items, draws)))
