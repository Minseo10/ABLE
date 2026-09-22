"""Tests for the limb manipulation baseline."""

import os

import numpy as np
import pytest
from kinder.envs.dynamic3d.limbs import (
    DEFAULT_RANGE_OF_MOTION,
    get_human_joint_limits,
)
from kinder_pddlstream_planning.limbrepositioning3d.stream import MPCConfig

from limb_baseline.belief import Belief, pooled_tau, refit
from limb_baseline.dataset import (
    append_row,
    load_dataset,
    row_from_episode,
    train_test_split_by_person,
)
from limb_baseline.human import (
    active_joints,
    active_motions,
    active_signs,
    DEFAULT_PRIOR,
    default_magnitudes,
    theta_to_bounds,
    theta_to_joint_limits,
)
from limb_baseline.planner import QueryPolicy
from limb_baseline.queries import DEFAULT_QUERY_MODEL, QueryModel
from limb_baseline.tasks import DEFAULT_TASK_DISTRIBUTION, inactive_joints
from limb_baseline.variant import DEFAULT_VARIANT, ENV_VAR, VARIANTS
from limb_baseline.variant import _build as _build_variant
from limb_baseline.variant import select as select_variant
from limb_baseline.warmstart import (
    control_basis,
    make_scaled_mlp,
    project_torques,
    segment_features,
)


def test_theta_scales_the_bound_of_the_joint_each_motion_governs():
    """theta is a scale factor, so halving it should halve that joint's bound."""
    defaults = DEFAULT_RANGE_OF_MOTION.as_dict()
    theta = np.array([0.9, 0.8, 0.7])
    lower, upper = theta_to_joint_limits(theta)
    for i, motion in enumerate(active_motions()):
        joint, sign = active_joints()[i], active_signs()[i]
        expected = np.deg2rad(defaults[motion] * theta[i])
        got = upper[joint] if sign > 0 else -lower[joint]
        assert got == pytest.approx(expected)


def test_inactive_motions_keep_their_defaults():
    """Only the three active motions are scaled; the rest of the person is unchanged."""
    ones = theta_to_joint_limits(np.ones(3))
    scaled = theta_to_joint_limits(np.array([0.5, 0.5, 0.5]))
    for joint in inactive_joints():
        assert scaled[0][joint] == pytest.approx(ones[0][joint])
        assert scaled[1][joint] == pytest.approx(ones[1][joint])


def test_prior_draws_stay_inside_the_plausible_range():
    rng = np.random.default_rng(0)
    draws = np.array([DEFAULT_PRIOR.sample(rng) for _ in range(500)])
    assert draws.min() >= DEFAULT_PRIOR.low
    assert draws.max() <= DEFAULT_PRIOR.high


def test_belief_update_moves_toward_the_truth_and_narrows():
    magnitudes = default_magnitudes()
    truth = np.array([0.85, 1.05, 0.75])
    belief = Belief.from_prior(DEFAULT_PRIOR)
    model = DEFAULT_QUERY_MODEL
    for _ in range(4):
        for i in range(len(truth)):
            answer = truth[i] * magnitudes[i] - model.short_fall
            belief = belief.updated(i, answer, model.tau, model.short_fall)
    assert np.all(np.abs(belief.mean - truth) < 0.02)
    assert np.all(belief.stddev < DEFAULT_PRIOR.stddev)


def test_pooled_tau_recovers_the_spread_of_repeat_answers():
    rng = np.random.default_rng(0)
    truth, tau = np.array([2.9, 3.2, 2.5]), 0.05
    answers = [list(rng.normal(bound, tau, size=80)) for bound in truth]
    assert pooled_tau(answers, fallback=1.0) == pytest.approx(tau, rel=0.15)


def test_pooled_tau_falls_back_until_a_motion_is_asked_twice():
    assert pooled_tau([[2.9], [], [2.5]], fallback=0.07) == 0.07


def test_refit_matches_updating_answer_by_answer():
    model = DEFAULT_QUERY_MODEL
    base = Belief.from_prior(DEFAULT_PRIOR)
    answers = [[2.80, 2.83], [], [2.30]]
    one_by_one = base
    for i, repeats in enumerate(answers):
        for answer in repeats:
            one_by_one = one_by_one.updated(i, answer, model.tau, model.short_fall)
    folded = refit(base, answers, model.tau, model.short_fall)
    assert np.allclose(folded.mean, one_by_one.mean)
    assert np.allclose(folded.stddev, one_by_one.stddev)


def test_a_noiseless_answer_is_inverted_exactly_in_the_limit():
    """With a vanishing tau the posterior mean is the answer divided by the range."""
    magnitudes = default_magnitudes()
    belief = Belief.from_prior(DEFAULT_PRIOR)
    truth = 0.83
    answer = truth * magnitudes[0] - 0.2
    updated = belief.updated(0, answer, tau=1e-6, short_fall=0.2)
    assert updated.mean[0] == pytest.approx(truth, abs=1e-4)


def test_effort_stays_below_the_end_range_torque():
    """The person must stop short of the passive bound, not push through it."""
    model = DEFAULT_QUERY_MODEL
    ceiling = model.stiffness * model.zone_width
    rng = np.random.default_rng(0)
    draws = np.array([model.draw_effort(rng) for _ in range(2000)])
    assert draws.min() > 0.0
    assert draws.max() < ceiling
    assert model.short_fall > 0.0


def test_an_effort_model_that_could_break_through_is_rejected():
    with pytest.raises(ValueError):
        QueryModel(zone_width=0.1, stiffness=10.0, effort_mean=5.0)


def test_restoring_torque_is_zero_then_linear_then_stiff():
    model = DEFAULT_QUERY_MODEL
    bound = 2.0
    assert model.restoring_torque(bound - model.zone_width - 0.1, bound) == 0.0
    inside = model.restoring_torque(bound - model.zone_width / 2, bound)
    assert inside < 0.0
    past = model.restoring_torque(bound + 0.05, bound)
    assert past < inside


def test_sampled_tasks_respect_the_person_and_the_separation():
    rng = np.random.default_rng(0)
    for _ in range(50):
        theta = DEFAULT_PRIOR.sample(rng)
        task = DEFAULT_TASK_DISTRIBUTION.sample(rng, theta)
        lower, upper = theta_to_joint_limits(theta)
        for positions in (task.limb_init, task.limb_goal):
            assert np.all(np.asarray(positions) >= lower - 1e-9)
            assert np.all(np.asarray(positions) <= upper + 1e-9)
        assert task.separation >= DEFAULT_TASK_DISTRIBUTION.min_separation


def test_control_basis_matches_the_mpc_expansion():
    """The label must live in the basis the MPC actually interpolates with."""
    mpc = MPCConfig()
    basis = control_basis(mpc)
    points = np.arange(mpc.num_control_points * 7, dtype=float).reshape(-1, 7)
    source = np.linspace(0.0, 1.0, mpc.num_control_points)
    target = np.linspace(0.0, 1.0, mpc.horizon)
    expected = np.stack(
        [np.interp(target, source, points[:, j]) for j in range(7)], axis=1
    )
    assert np.allclose(basis @ points, expected)


def test_projection_recovers_control_points_it_was_built_from():
    mpc = MPCConfig()
    rng = np.random.default_rng(0)
    points = rng.normal(0.0, 0.1, size=(mpc.num_control_points, 7))
    expanded = control_basis(mpc) @ points
    per_simulator_step = np.repeat(expanded, mpc.action_repeat, axis=0)
    assert np.allclose(project_torques(per_simulator_step.tolist(), mpc), points)


def test_projection_reads_only_the_first_horizon():
    """psi seeds the MPC's first step, so later torques must not change the label."""
    mpc = MPCConfig()
    rng = np.random.default_rng(0)
    points = rng.normal(0.0, 0.1, size=(mpc.num_control_points, 7))
    head = np.repeat(control_basis(mpc) @ points, mpc.action_repeat, axis=0)
    tail = rng.normal(0.0, 5.0, size=(2000, 7))
    combined = np.vstack([head, tail])
    assert np.allclose(project_torques(combined.tolist(), mpc), points)


def test_segment_features_drop_the_gripper_joints():
    features = segment_features(np.zeros(6), np.arange(13.0), np.ones(6), np.zeros(6), 4)
    assert features.shape == (26,)
    assert np.allclose(features[6:13], np.arange(7.0))
    assert features[-1] == 4.0


def test_test_split_shares_no_person_with_training(tmp_path):
    path = tmp_path / "demos.csv"
    rng = np.random.default_rng(0)
    for person in range(20):
        for _ in range(4):
            append_row(
                path,
                row_from_episode(
                    rng.normal(size=25), rng.normal(size=(4, 7)), {"person": person}
                ),
            )
    features, labels, metadata = load_dataset(path)
    train_x, _, test_x, _ = train_test_split_by_person(features, labels, metadata)
    assert len(train_x) + len(test_x) == len(features)
    assert len(test_x) > 0
    people = metadata["person"].to_numpy()
    train_rows = {tuple(row) for row in train_x}
    for row, person in zip(features, people):
        if tuple(row) in train_rows:
            continue
        # A test person must contribute no row to training.
        assert not any(
            tuple(other) in train_rows
            for other, other_person in zip(features, people)
            if other_person == person
        )


@pytest.mark.slow
def test_a_query_stops_short_of_the_passive_bound():
    """The whole observation model: the person stops inside their own range."""
    from limb_baseline.queries import QuerySimulator

    theta = np.array([0.9, 1.0, 0.8])
    bounds = np.abs(theta_to_bounds(theta))
    simulator = QuerySimulator(theta)
    try:
        rng = np.random.default_rng(0)
        for i in range(3):
            outcome = simulator.ask(i, bounds[i] + 0.5, rng)
            shortfall = bounds[i] - outcome.answer
            assert not outcome.reached_target
            assert 0.0 < shortfall < 2 * DEFAULT_QUERY_MODEL.zone_width
    finally:
        simulator.close()


def test_the_target_is_drawn_around_the_default_range_of_motion():
    """The robot does not know the person's range, so it asks around the default one."""
    from limb_baseline.human import default_magnitudes

    policy = QueryPolicy()
    belief = Belief.from_prior(DEFAULT_PRIOR)
    floors = np.zeros(len(belief.mean))
    rng = np.random.default_rng(0)
    for _ in range(50):
        motion_index, target = policy.choose(belief, rng, floors)
        boundary = default_magnitudes()[motion_index]
        assert boundary - policy.probe_band <= target <= boundary + policy.probe_band


def test_a_reached_target_raises_the_floor_instead_of_moving_the_belief():
    """Reaching proves only that the bound is at least there, not where it is."""
    policy = QueryPolicy()
    belief = Belief.from_prior(DEFAULT_PRIOR)
    floors = np.zeros(len(belief.mean))
    rng = np.random.default_rng(0)
    motion_index, target = policy.choose(belief, rng, floors)
    floors[motion_index] = target + DEFAULT_QUERY_MODEL.short_fall
    _, raised = policy.choose(belief, rng, floors)
    assert raised > target


@pytest.mark.parametrize("name", VARIANTS)
def test_every_variant_bounds_the_joints_its_table_claims(name):
    """Scaling one active motion must move exactly the bound the table records."""
    limb = _build_variant(name)
    base_lower, base_upper = get_human_joint_limits(limb.limb_name)
    for motion, joint, sign in zip(
        limb.active_motions, limb.active_joints, limb.active_signs, strict=True
    ):
        rom = DEFAULT_RANGE_OF_MOTION.scaled({motion: 0.5})
        lower, upper = get_human_joint_limits(limb.limb_name, rom)
        assert np.isclose(lower[joint], base_lower[joint]) == (sign > 0)
        assert np.isclose(upper[joint], base_upper[joint]) == (sign < 0)
        others = [j for j in range(len(lower)) if j != joint]
        assert np.allclose(lower[others], base_lower[others])
        assert np.allclose(upper[others], base_upper[others])


def test_selecting_a_variant_changes_what_theta_means():
    """The active motions follow the limb, and the choice reaches spawned workers."""
    try:
        select_variant("bed-right-leg")
        assert active_motions() == ("hip_flexion", "hip_abduction", "knee_flexion")
        assert os.environ[ENV_VAR] == "bed-right-leg"
        select_variant("isolated-left-arm")
        assert active_motions()[0] == "shoulder_flexion"
        assert active_signs()[0] == +1
    finally:
        select_variant(DEFAULT_VARIANT)


def test_an_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown variant"):
        select_variant("sofa-third-arm")


def test_a_resumed_belief_keeps_sharpening_across_rounds():
    """Rounds of gathering compound: a resumed belief is tighter than a fresh one."""
    prior = Belief.from_prior(DEFAULT_PRIOR)
    first = prior.updated(0, 0.9 * default_magnitudes()[0], DEFAULT_QUERY_MODEL.tau)
    second = first.updated(0, 0.9 * default_magnitudes()[0], DEFAULT_QUERY_MODEL.tau)
    assert second.stddev[0] < first.stddev[0] < prior.stddev[0]
    assert np.allclose(second.stddev[1:], prior.stddev[1:])


def test_fine_tuning_continues_from_the_pretrained_weights():
    """A round of training keeps the weights it has and freezes the scaler."""
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(40, 6)), rng.normal(size=(40, 3))
    model = make_scaled_mlp(epochs=20, learning_rate=1e-3, hidden=(8,))
    model.train(x, y)
    weights, scaler_mean = model.net.coefs_[0].copy(), model.scaler.mean_.copy()

    # At a vanishing learning rate, continuing barely moves; reinitializing would not.
    model.net.set_params(max_iter=5, learning_rate_init=1e-12)
    model.train(x[:3] * 10.0, y[:3])
    assert np.abs(weights - model.net.coefs_[0]).max() < 1e-6
    assert np.allclose(scaler_mean, model.scaler.mean_)


def test_the_memo_learns_from_the_first_success_and_serves_it_after(monkeypatch):
    """Before its first success the memo defers to the MPC; after it, a naive memo
    answers from memory and never calls the MPC again."""
    from kinder_pddlstream_planning.limbrepositioning3d.stream import TorqueTrajectory

    from limb_baseline import memo as memo_module

    calls = []
    trajectories = {"goal": TorqueTrajectory([[0.0] * 7])}

    def fake_mpc(ctx, s1, q2):
        calls.append(q2)
        if q2 == "unreachable":
            return
        yield trajectories[q2], f"state-after-{q2}"

    monkeypatch.setattr(memo_module, "plan_limb_motion", fake_mpc)
    memo = memo_module.LimbMotionMemo(mode="naive")

    assert list(memo.stream(None, "s", "unreachable")) == []
    assert not memo.filled and memo.mpc_calls == 1 and memo.hits == 0

    assert list(memo.stream(None, "s", "goal")) == [
        (trajectories["goal"], "state-after-goal")
    ]
    assert memo.filled and memo.mpc_calls == 2

    memo.reset()
    assert list(memo.stream(None, "other-state", "other-goal")) == [
        (trajectories["goal"], "state-after-goal")
    ]
    assert (memo.hits, memo.mpc_calls, memo.calls) == (1, 0, 1)
    assert calls == ["unreachable", "goal"]
    assert memo.seconds_in_stream >= 0.0


def test_the_memo_routes_the_planner_stream_and_can_be_removed():
    from kinder_pddlstream_planning.limbrepositioning3d import run as planner_run

    from limb_baseline import memo as memo_module

    memo = memo_module.LimbMotionMemo()
    memo_module.install(memo)
    assert planner_run.plan_limb_motion == memo.stream
    memo_module.uninstall()
    assert planner_run.plan_limb_motion is memo_module.plan_limb_motion
