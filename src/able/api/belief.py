"""Beliefs over a person's hidden parameters theta."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Observation:
    """What the robot saw: a query answer or an execution outcome."""

    kind: str  # "query" or "execution"
    value: Any
    query: Any = None


class Belief(ABC):
    """Immutable: `update` returns a new belief."""

    @abstractmethod
    def sample(self, rng: np.random.Generator, n: int = 1) -> NDArray[np.float64]:
        """(n, dim) thetas."""

    @abstractmethod
    def update(self, observation: Observation) -> Belief: ...

    @abstractmethod
    def features(self) -> NDArray[np.float64]:
        """Fixed-length vector psi is conditioned on."""

    @abstractmethod
    def summary(self) -> dict[str, Any]:
        """Small JSON-able record for logs."""


class Prior(ABC):
    """Population prior over theta."""

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> NDArray[np.float64]: ...

    @abstractmethod
    def belief(self, rng: np.random.Generator) -> Belief:
        """The belief before any observation."""


@dataclass(frozen=True)
class ParticleBelief(Belief):
    """Equal-weight particles; subclasses define `log_likelihood`."""

    particles: NDArray[np.float64]  # (n, dim)
    low: NDArray[np.float64]
    high: NDArray[np.float64]
    jitter: float = 0.02  # resampling noise, fraction of (high - low)
    seed: int = 0

    def log_likelihood(self, observation: Observation, particles: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError

    def sample(self, rng: np.random.Generator, n: int = 1) -> NDArray[np.float64]:
        return self.particles[rng.integers(len(self.particles), size=n)]

    def update(self, observation: Observation) -> ParticleBelief:
        log_w = np.asarray(self.log_likelihood(observation, self.particles), np.float64)
        if not np.isfinite(log_w).any():
            return self  # nothing explains it; keep the belief
        w = np.exp(log_w - log_w[np.isfinite(log_w)].max())
        w = np.where(np.isfinite(w), w, 0.0)
        w /= w.sum()
        rng = np.random.default_rng(self.seed)
        n = len(self.particles)
        # Systematic resampling plus jitter.
        idx = np.searchsorted(np.cumsum(w), (rng.random() + np.arange(n)) / n).clip(max=n - 1)
        span = np.asarray(self.high) - np.asarray(self.low)
        new = self.particles[idx] + rng.normal(0.0, self.jitter, self.particles.shape) * span
        return replace(self, particles=np.clip(new, self.low, self.high), seed=self.seed + 1)

    @property
    def mean(self) -> NDArray[np.float64]:
        return self.particles.mean(0)

    @property
    def std(self) -> NDArray[np.float64]:
        return self.particles.std(0)

    def features(self) -> NDArray[np.float64]:
        return np.concatenate([self.mean, self.std])

    def summary(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}
