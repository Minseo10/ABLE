"""Limb prior, belief and queries behind the shared API."""

import numpy as np
import pytest

from able.api import Belief, InfoGatherer, Observation, Prior
from able.tasks.limb.belief import LimbPrior
from able.tasks.limb.human import default_magnitudes
from able.tasks.limb.queries import (
    DEFAULT_QUERY_MODEL,
    PushToLimitGatherer,
    QueryOutcome,
)


def test_limb_pieces_satisfy_the_protocols():
    prior = LimbPrior()
    belief = prior.belief(np.random.default_rng(0))
    assert isinstance(prior, Prior) and isinstance(belief, Belief)
    assert isinstance(PushToLimitGatherer(), InfoGatherer)
    assert belief.sample(np.random.default_rng(0), 5).shape == (5, 3)
    assert belief.features().shape == (6,)


def test_settled_answer_updates_and_censored_answer_raises_the_floor():
    belief = LimbPrior().belief(np.random.default_rng(0))
    mag = default_magnitudes()[0]
    settled = QueryOutcome(0, 2.0, 0.8 * mag, reached_target=False, effort=1.0)
    after = belief.update(Observation("query", settled))
    assert after.gaussian.stddev[0] < belief.gaussian.stddev[0]
    assert np.allclose(after.gaussian.stddev[1:], belief.gaussian.stddev[1:])
    censored = QueryOutcome(1, 1.0, 1.0 * default_magnitudes()[1], reached_target=True, effort=1.0)
    raised = belief.update(Observation("query", censored))
    expected = (censored.answer + DEFAULT_QUERY_MODEL.short_fall) / default_magnitudes()[1]
    assert raised.gaussian.lower[1] == pytest.approx(expected)
    fast = QueryOutcome(0, 2.0, 0.8 * mag, False, 1.0, peak_velocity=DEFAULT_QUERY_MODEL.max_velocity)
    assert belief.update(Observation("query", fast)) is belief


@pytest.mark.slow
def test_gatherer_narrows_the_belief_toward_the_person():
    theta = np.array([0.85, 1.05, 0.95])
    rng = np.random.default_rng(7)
    gatherer, belief = PushToLimitGatherer(), LimbPrior().belief(rng)
    start = belief.gaussian.stddev.copy()
    try:
        for _ in range(6):
            belief = belief.update(gatherer.respond(theta, gatherer.choose(belief, rng), rng))
    finally:
        gatherer.close()
    assert np.all(belief.gaussian.stddev <= start) and belief.gaussian.stddev.sum() < start.sum()
