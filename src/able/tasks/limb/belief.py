"""The robot's belief about `theta`: one Gaussian per active motion.

The likelihood is linear-Gaussian, so the update is closed form. A query answer is where
the person stopped, which `short_fall` offsets back toward their passive bound. An
execution that stayed inside the person only says the limit lies beyond where the limb
went: that raises a floor that truncates the Gaussian from below.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.stats import truncnorm

from able.tasks.limb.human import (
    DEFAULT_PRIOR,
    RoMPrior,
    active_motions,
    default_magnitudes,
)


def pooled_tau(answers: Sequence[Sequence[float]], fallback: float) -> float:
    """The likelihood's noise, measured from repeat answers about the same motion.

    A person's bound does not move between rounds, so answers about one motion differ
    only by how hard they pushed. Pooled across motions; `fallback` until a repeat.
    """
    residuals = []
    degrees = 0
    for repeats in answers:
        if len(repeats) < 2:
            continue
        values = np.asarray(repeats, dtype=np.float64)
        residuals.append(values - values.mean())
        degrees += len(values) - 1
    if degrees == 0:
        return fallback
    estimate = float(np.sqrt(np.sum(np.concatenate(residuals) ** 2) / degrees))
    # Identical answers would divide by zero in the update.
    return max(estimate, 1e-4)


def refit(base: "Belief", answers, tau: float, short_fall: float) -> "Belief":
    """Fold every answer so far into `base`, so a revised `tau` applies to all of them."""
    belief = base
    for index, repeats in enumerate(answers):
        for answer in repeats:
            belief = belief.updated(index, answer, tau, short_fall)
    return belief


@dataclass(frozen=True)
class Belief:
    """A mean and a standard deviation per active motion.

    `low` and `high` are the population's bounds. The Gaussian is untruncated so the
    conjugate update stays closed form; `sample`, `planning_mean` and `width` read the
    truncation to `[lower, high]`, where `lower` is the population's bound raised by
    what the person has been seen to reach.
    """

    mean: NDArray[np.float64]
    stddev: NDArray[np.float64]
    low: float = DEFAULT_PRIOR.low
    high: float = DEFAULT_PRIOR.high
    # Per motion, a scale factor the person reached safely, so the limit lies beyond.
    floor: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        assert self.mean.shape == self.stddev.shape == (len(active_motions()),)
        assert np.all(self.stddev > 0.0)
        assert self.low < self.high
        assert self.floor is None or self.floor.shape == self.mean.shape

    @property
    def lower(self) -> NDArray[np.float64]:
        """The least each motion's factor can be: the population's bound, or the floor."""
        low = np.full(len(self.mean), self.low)
        return low if self.floor is None else np.maximum(low, self.floor)

    def _truncated(self):
        a = (self.lower - self.mean) / self.stddev
        b = (self.high - self.mean) / self.stddev
        return truncnorm(a, b, loc=self.mean, scale=self.stddev)

    @property
    def planning_mean(self) -> NDArray[np.float64]:
        """The one theta to plan under: the mean of the truncated belief."""
        return np.asarray(self._truncated().mean(), dtype=np.float64)

    @property
    def width(self) -> float:
        """How uncertain the belief still is, summed over motions, as the notebook's."""
        return float(np.sum(self._truncated().std()))

    def raised(self, motion_index: int, scale: float) -> "Belief":
        """Condition on the person having reached `scale` of a motion's default range."""
        floor = np.zeros(len(self.mean)) if self.floor is None else self.floor.copy()
        # A floor at the top would leave nothing to sample.
        floor[motion_index] = min(max(floor[motion_index], scale), self.high - 1e-3)
        return replace(self, floor=floor)

    @classmethod
    def from_prior(cls, prior: RoMPrior) -> "Belief":
        """The belief before any query, taken from the population prior."""
        n = len(active_motions())
        return cls(
            np.full(n, prior.mean), np.full(n, prior.stddev), prior.low, prior.high
        )

    def sample(self, rng: np.random.Generator) -> NDArray[np.float64]:
        """One `theta` drawn from the belief truncated to `[lower, high]`."""
        return np.asarray(self._truncated().rvs(random_state=rng), dtype=np.float64)

    def updated(
        self, motion_index: int, answer: float, tau: float, short_fall: float = 0.0
    ) -> "Belief":
        """Condition on one query answer, leaving the other motions untouched.

        `answer` is the magnitude of the joint angle the person stopped at.
        """
        magnitude = default_magnitudes()[motion_index]
        prior_mean = self.mean[motion_index]
        prior_var = self.stddev[motion_index] ** 2
        observation = abs(answer) + short_fall
        precision = 1.0 / prior_var + magnitude**2 / tau**2
        posterior_var = 1.0 / precision
        posterior_mean = posterior_var * (
            prior_mean / prior_var + magnitude * observation / tau**2
        )
        mean = self.mean.copy()
        stddev = self.stddev.copy()
        mean[motion_index] = posterior_mean
        stddev[motion_index] = np.sqrt(posterior_var)
        return replace(self, mean=mean, stddev=stddev)

    def as_features(self) -> NDArray[np.float64]:
        """The belief as the six numbers the warm-start model reads."""
        return np.concatenate([self.mean, self.stddev])

    def key_features(self) -> NDArray[np.float64]:
        """Everything a cache key should tell apart: the Gaussian and its floor."""
        return np.concatenate([self.mean, self.stddev, self.lower])
