"""Stages 1-2: ask the person a few questions and update the belief.

Each round probes the motion the robot is least sure about, with a target near the
default range of motion rather than the person's own. An answer that reaches the target
only proves the bound is at least that far, which the Gaussian cannot represent, so it is
recorded, skipped, and the motion's floor is raised.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from limb_baseline.belief import Belief, pooled_tau, refit
from limb_baseline.human import DEFAULT_PRIOR, RoMPrior, default_magnitudes
from limb_baseline.queries import (
    DEFAULT_QUERY_MODEL,
    QueryModel,
    QueryOutcome,
    QuerySimulator,
)


@dataclass(frozen=True)
class QueryPolicy:
    """How many questions to ask, and where around the default range to ask."""

    # Six, not three: censored motions need a second round after the floor rises.
    num_queries: int = 6
    # How far either side of the boundary the target is drawn, in radians.
    probe_band: float = 0.3

    def choose(
        self, belief: Belief, rng: np.random.Generator, floors: NDArray[np.float64]
    ) -> tuple[int, float]:
        """The motion to ask about, and the angle to ask for.

        The angle is drawn around the *default* range of motion, since the person's own
        is what the query is after. `floors` raises it where a target was already
        reached, so a wider-ranged person is asked higher instead of answering trivially.
        """
        motion_index = int(np.argmax(belief.stddev))
        bound = max(default_magnitudes()[motion_index], float(floors[motion_index]))
        band = self.probe_band
        return motion_index, float(bound + rng.uniform(-band, band))


DEFAULT_QUERY_POLICY = QueryPolicy()


@dataclass
class InformationGathering:
    """What Stage 1 produced."""

    posterior: Belief
    outcomes: list[QueryOutcome]
    floors: NDArray[np.float64] | None = None
    tau: float = DEFAULT_QUERY_MODEL.tau

    @property
    def num_censored(self) -> int:
        """Answers that stopped at the target rather than at the person's limit."""
        return sum(outcome.reached_target for outcome in self.outcomes)

    @property
    def num_unsettled(self) -> int:
        """Answers cut off by the step cap, which locate nothing and are skipped."""
        return sum(
            not o.settled and not o.reached_target for o in self.outcomes
        )


def gather_information(
    theta: NDArray[np.float64],
    rng: np.random.Generator,
    policy: QueryPolicy = DEFAULT_QUERY_POLICY,
    prior: RoMPrior = DEFAULT_PRIOR,
    model: QueryModel = DEFAULT_QUERY_MODEL,
    simulator: QuerySimulator | None = None,
    start: Belief | None = None,
    floors: NDArray[np.float64] | None = None,
    estimate_tau: bool = False,
) -> InformationGathering:
    """Ask `policy.num_queries` questions, updating the belief after each."""
    owned = simulator is None
    simulator = simulator or QuerySimulator(theta, model)
    try:
        base = start if start is not None else Belief.from_prior(prior)
        belief = base
        outcomes: list[QueryOutcome] = []
        answers: list[list[float]] = [[] for _ in base.mean]
        tau = model.tau
        floors = np.zeros(len(belief.mean)) if floors is None else floors.copy()
        for _ in range(policy.num_queries):
            motion_index, target = policy.choose(belief, rng, floors)
            outcome = simulator.ask(motion_index, target, rng)
            outcomes.append(outcome)
            # A joint that saturated its speed cap was not making a passive push.
            if outcome.peak_velocity >= model.max_velocity:
                continue
            # A censored answer only raises the floor; it does not locate the bound.
            if outcome.reached_target:
                floors[motion_index] = outcome.answer + model.short_fall
            elif outcome.settled:
                answers[motion_index].append(outcome.answer)
                if estimate_tau:
                    tau = pooled_tau(answers, model.tau)
                    belief = refit(base, answers, tau, model.short_fall)
                else:
                    belief = belief.updated(
                        motion_index, outcome.answer, tau, model.short_fall
                    )
        return InformationGathering(belief, outcomes, floors, tau)
    finally:
        if owned:
            simulator.close()
