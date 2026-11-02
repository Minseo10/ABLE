"""Tests for the seeded MPC's lazy scoring and batch reuse, on a fake MPC: no simulator."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from able.tasks.limb import memo
from able.tasks.limb.memo import STATS, MPCConfig, _SeededMPC


def fake_mpc(costs: dict, num_rollouts: int = 10):
    """An MPC whose rollouts cost what `costs` says for each candidate's first value."""
    mpc = object.__new__(_SeededMPC)
    mpc._cfg = dataclasses.replace(
        MPCConfig(), num_rollouts=num_rollouts, feedback_gains=(4.0,)
    )
    mpc._ctx = SimpleNamespace(warm_start="hook")
    mpc._commit, mpc._step_index, mpc._scored = 1, 0, 0
    mpc._expand = lambda points: points
    mpc._feedback_candidates = lambda start: [np.full((2, 1), 1.0)]
    mpc._rollout_error = lambda: 1.0  # never within reach, so the cost decides
    fresh_sizes = []

    def sample(error, start):
        assert mpc._ctx.warm_start is None and mpc._cfg.feedback_gains == ()
        fresh_sizes.append(mpc._cfg.num_rollouts)
        return [np.full((2, 1), 100.0 + i) for i in range(mpc._cfg.num_rollouts)]

    def score(start, torques):
        mpc._scored += 1
        value = float(torques[0, 0])
        return costs.get(value, (0, value))

    mpc._sample_candidates = sample
    mpc._score = score
    return mpc, fresh_sizes


@pytest.fixture(autouse=True)
def fresh_stats():
    STATS.reset()
    yield
    STATS.reset()


def test_the_first_passing_draw_is_taken_and_the_rest_are_not_scored():
    # Feedback bar costs 1.0; the seed (2.0) misses it, the first draw (0.5) passes.
    mpc, fresh = fake_mpc({})
    batch = [np.full((2, 1), v) for v in (2.0, 0.5, 0.1)]
    committed = mpc._reusing_step(None, 1.0, 0.1, batch, False, 0.0, 0.0)
    assert committed == [[0.5]]
    assert STATS.seeded_steps == 1 and STATS.scored_rollouts == 3 and not fresh


def test_a_miss_keeps_what_was_scored_and_fills_the_batch_only_to_its_size():
    # Every model candidate misses the feedback bar: the step falls back.
    mpc, fresh = fake_mpc({})
    batch = [np.full((2, 1), v) for v in (5.0, 6.0)]
    committed = mpc._reusing_step(None, 1.0, 0.1, batch, False, 0.0, 0.0)
    # Three scored (the bar and two draws), so seven fresh fill the batch of ten.
    assert fresh == [7] and STATS.full_rollouts == 7 and STATS.scored_rollouts == 3
    # The feedback candidate scored while seeding is still the best of ten.
    assert committed == [[1.0]]
    assert STATS.seed_rejected == 1 and STATS.full_steps == 1
    assert mpc._ctx.warm_start == "hook" and mpc._cfg.feedback_gains == (4.0,)


def test_a_batch_with_every_candidate_rejected_raises():
    mpc, _ = fake_mpc({v: (1, 0.0) for v in [1.0, 5.0] + [100.0 + i for i in range(10)]})
    with pytest.raises(memo.NoFeasibleMotion):
        mpc._reusing_step(None, 1.0, 0.1, [np.full((2, 1), 5.0)], False, 0.0, 0.0)
