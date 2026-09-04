"""The robot's belief about `theta`: one Gaussian per active motion.

The likelihood is linear-Gaussian, so the update is closed form. A query answer is where
the person stopped, which `short_fall` offsets back toward their passive bound.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from limb_baseline.human import (
    DEFAULT_PRIOR,
    RoMPrior,
    active_motions,
    default_magnitudes,
)


@dataclass(frozen=True)
class Belief:
    """A mean and a standard deviation per active motion.

    `low` and `high` are the population's bounds. The Gaussian is untruncated so the
    conjugate update stays closed form; only `sample` is bounded.
    """

    mean: NDArray[np.float64]
    stddev: NDArray[np.float64]
    low: float = DEFAULT_PRIOR.low
    high: float = DEFAULT_PRIOR.high

    def __post_init__(self) -> None:
        assert self.mean.shape == self.stddev.shape == (len(active_motions()),)
        assert np.all(self.stddev > 0.0)
        assert self.low < self.high

    @classmethod
    def from_prior(cls, prior: RoMPrior) -> "Belief":
        """The belief before any query, taken from the population prior."""
        n = len(active_motions())
        return cls(
            np.full(n, prior.mean), np.full(n, prior.stddev), prior.low, prior.high
        )

    def sample(self, rng: np.random.Generator) -> NDArray[np.float64]:
        """One `theta` drawn from the belief, clipped to the population's range."""
        return np.clip(rng.normal(self.mean, self.stddev), self.low, self.high)

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
