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


def test_execution_evidence_matches_observe():
    from able.tasks.limb.belief import execution_evidence
    from able.tasks.limb.dream_pipeline import observe

    rng = np.random.default_rng(3)
    belief = LimbPrior().belief(rng)
    person = np.array([0.8, 1.0, 0.9])
    path = rng.uniform(-2.5, 2.5, (30, 7))
    expected, _ = observe(belief.gaussian, path, person)
    got = belief.update(Observation("execution", execution_evidence(path, person))).gaussian
    assert np.allclose(got.mean, expected.mean) and np.allclose(got.lower, expected.lower)


@pytest.mark.slow
def test_gatherer_reproduces_gather_information():
    from able.tasks.limb.planner import gather_information

    theta = np.array([0.85, 1.05, 0.95])
    reference = gather_information(theta, np.random.default_rng(7))
    rng = np.random.default_rng(7)
    gatherer = PushToLimitGatherer()
    belief = LimbPrior().belief(rng)
    try:
        for _ in range(len(reference.outcomes)):
            belief = belief.update(gatherer.respond(theta, gatherer.choose(belief, rng), rng))
    finally:
        gatherer.close()
    assert np.allclose(belief.gaussian.mean, reference.posterior.mean)
    assert np.allclose(belief.gaussian.stddev, reference.posterior.stddev)
