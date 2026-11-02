"""Belief over `theta`: one Gaussian per active motion, closed-form linear-Gaussian update.

Executions that stayed inside the person raise a floor that truncates the Gaussian from below.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.stats import truncnorm

from able.api.belief import Observation
from able.tasks.limb.human import (
    DEFAULT_PRIOR,
    RoMPrior,
    active_joints,
    active_motions,
    active_signs,
    default_magnitudes,
    theta_to_bounds,
)
from able.tasks.limb.queries import DEFAULT_QUERY_MODEL, QueryModel, QueryOutcome

# A floor must rise by this much (in scale) to count as new.
FLOOR_STEP = 1e-3


@dataclass(frozen=True)
class Belief:
    """Untruncated Gaussian per motion (closed-form update); reads truncate to `[lower, high]`."""

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
        """Per-motion lower bound: the population's `low` or the floor."""
        low = np.full(len(self.mean), self.low)
        return low if self.floor is None else np.maximum(low, self.floor)

    def _truncated(self):
        a = (self.lower - self.mean) / self.stddev
        b = (self.high - self.mean) / self.stddev
        return truncnorm(a, b, loc=self.mean, scale=self.stddev)

    @property
    def planning_mean(self) -> NDArray[np.float64]:
        """Mean of the truncated belief."""
        return np.asarray(self._truncated().mean(), dtype=np.float64)

    @property
    def width(self) -> float:
        """Truncated stddev summed over motions."""
        return float(np.sum(self._truncated().std()))

    def raised(self, motion_index: int, scale: float) -> "Belief":
        """Condition on the person having reached `scale` of a motion's default range."""
        floor = np.zeros(len(self.mean)) if self.floor is None else self.floor.copy()
        # A floor at the top would leave nothing to sample.
        floor[motion_index] = min(max(floor[motion_index], scale), self.high - 1e-3)
        return replace(self, floor=floor)

    @classmethod
    def from_prior(cls, prior: RoMPrior) -> "Belief":
        """Belief before any query."""
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
        """Condition one motion on a query answer (magnitude of the stop angle)."""
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
        """Mean and stddev, as the warm-start model reads them."""
        return np.concatenate([self.mean, self.stddev])

    def key_features(self) -> NDArray[np.float64]:
        """Cache-key features: the Gaussian and its floor."""
        return np.concatenate([self.mean, self.stddev, self.lower])


def execution_evidence(limb_path, person: NDArray[np.float64]) -> list[tuple[str, int, float]]:
    """Per active motion: ("crossed", i, angle) if the path passed the limit, else ("reached", i, scale)."""
    if not len(limb_path):
        return []
    path = np.asarray(limb_path, dtype=np.float64)
    bounds = theta_to_bounds(person)
    magnitudes = default_magnitudes()
    evidence = []
    for i, (joint, sign) in enumerate(zip(active_joints(), active_signs())):
        reach = sign * path[:, joint]
        crossed = np.flatnonzero(reach > sign * bounds[i])
        if len(crossed):
            evidence.append(("crossed", i, float(reach[crossed[0]])))
        else:
            evidence.append(("reached", i, float(reach.max()) / magnitudes[i]))
    return evidence


def apply_evidence(belief: Belief, evidence, tau: float) -> tuple[Belief, list[str]]:
    """Crossing angle = a limit measurement; farthest reach = a floor. Returns notes too."""
    magnitudes = default_magnitudes()
    notes = []
    for kind, i, value in evidence:
        if kind == "crossed":
            belief = belief.updated(i, value, tau, 0.0)
            notes.append(f"crossed:{i}@{value / magnitudes[i]:.3f}")
        elif value > belief.lower[i] + FLOOR_STEP:
            # Repeating a task reaches the same angle within numerical noise: not a revision.
            belief = belief.raised(i, value)
            notes.append(f"floor:{i}@{value:.3f}")
    return belief, notes


@dataclass(frozen=True)
class LimbBelief:
    """`Belief` behind the shared API: query answers and executions both update it."""

    gaussian: Belief
    query_model: QueryModel = DEFAULT_QUERY_MODEL

    def sample(self, rng: np.random.Generator, n: int = 1) -> NDArray[np.float64]:
        return np.stack([self.gaussian.sample(rng) for _ in range(n)])

    def update(self, observation: Observation) -> "LimbBelief":
        if observation.kind == "execution":
            return replace(self, gaussian=apply_evidence(
                self.gaussian, observation.value, self.query_model.tau)[0])
        outcome: QueryOutcome = observation.value
        model, i = self.query_model, outcome.motion_index
        # A joint at its speed cap was not a passive push; an unsettled one locates nothing.
        if outcome.peak_velocity >= model.max_velocity:
            return self
        if outcome.reached_target:
            # Censored: the limit lies beyond the target.
            scale = (outcome.answer + model.short_fall) / default_magnitudes()[i]
            return replace(self, gaussian=self.gaussian.raised(i, scale))
        if not outcome.settled:
            return self
        return replace(self, gaussian=self.gaussian.updated(i, outcome.answer, model.tau, model.short_fall))

    def features(self) -> NDArray[np.float64]:
        return self.gaussian.as_features()

    def summary(self) -> dict:
        g = self.gaussian
        return {"mean": g.mean.tolist(), "std": g.stddev.tolist(), "lower": g.lower.tolist()}


@dataclass(frozen=True)
class LimbPrior:
    """Population prior over the range-of-motion scales."""

    prior: RoMPrior = DEFAULT_PRIOR

    def sample(self, rng: np.random.Generator) -> NDArray[np.float64]:
        return self.prior.sample(rng)

    def belief(self, rng: np.random.Generator) -> LimbBelief:
        return LimbBelief(Belief.from_prior(self.prior))
