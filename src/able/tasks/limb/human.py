"""The person: `theta` scales each active motion of the AAOS normal adult range of motion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from kinder.envs.dynamic3d.limbs import (
    DEFAULT_RANGE_OF_MOTION,
    RangeOfMotion,
    get_human_joint_limits,
)
from numpy.typing import NDArray

from able.tasks.limb.variant import current as current_variant


def variant() -> str:
    """The scene and limb this process works on, e.g. "bed-right-leg"."""
    return current_variant().name


def limb_name() -> str:
    """The limb the environment calls by this name, e.g. "human-right-leg"."""
    return current_variant().limb_name


def active_motions() -> tuple[str, ...]:
    """The motions `theta` scales, one factor each."""
    return current_variant().active_motions


def active_joints() -> tuple[int, ...]:
    """Which limb joint each active motion bounds."""
    return current_variant().active_joints


def active_signs() -> tuple[int, ...]:
    """Which side each active motion bounds from: -1 below, +1 above."""
    return current_variant().active_signs


@dataclass(frozen=True)
class RoMPrior:
    """The population `theta` is drawn from, truncated to a plausible range."""

    mean: float = 0.9
    stddev: float = 0.15
    low: float = 0.7
    high: float = 1.2

    def sample(self, rng: np.random.Generator) -> NDArray[np.float64]:
        """One scale factor per active motion, redrawing anything out of range."""
        motions = active_motions()
        draws = np.empty(len(motions))
        for i in range(len(motions)):
            while True:
                value = rng.normal(self.mean, self.stddev)
                if self.low <= value <= self.high:
                    draws[i] = value
                    break
        return draws


DEFAULT_PRIOR = RoMPrior()


def default_magnitudes() -> NDArray[np.float64]:
    """Default magnitude of each active motion (rad); `R` in the likelihood."""
    defaults = DEFAULT_RANGE_OF_MOTION.as_dict()
    return np.deg2rad([defaults[motion] for motion in active_motions()])


def theta_to_range_of_motion(theta: NDArray[np.float64]) -> RangeOfMotion:
    """Full range of motion; inactive motions stay at default."""
    factors = dict(zip(active_motions(), map(float, theta), strict=True))
    return DEFAULT_RANGE_OF_MOTION.scaled(factors)


def theta_to_joint_limits(
    theta: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The limb's per-joint (lower, upper) bounds under `theta`, in radians."""
    return get_human_joint_limits(limb_name(), theta_to_range_of_motion(theta))


def theta_to_bounds(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """The signed joint angle each active motion reaches at its limit."""
    return np.asarray(theta, dtype=np.float64) * default_magnitudes() * active_signs()
