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
    BASE_FEATURES,
    control_basis,
    make_regressor,
    make_scaled_mlp,
    project_torques,
    segment_features,
    state_features,
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


def test_history_features_follow_the_base_features():
    base = segment_features(np.zeros(6), np.arange(13.0), np.ones(6), np.zeros(6), 4)
    full = segment_features(np.zeros(6), np.arange(13.0), np.ones(6), np.zeros(6), 4,
                            np.full(6, 2.0), np.arange(13.0) + 10, np.full(7, 3.0))
    assert full.shape == (BASE_FEATURES + 20,)
    assert np.allclose(full[:BASE_FEATURES], base)
    assert np.allclose(full[BASE_FEATURES + 6:BASE_FEATURES + 13], np.arange(7.0) + 10)
    assert np.allclose(full[-7:], 3.0)
    start = segment_features(np.zeros(6), np.zeros(13), np.ones(6), np.zeros(6), 0,
                             np.zeros(6), np.zeros(13))
    assert np.allclose(start[-7:], 0.0)


def test_only_hist_specs_read_the_history_columns():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, BASE_FEATURES + 20))
    y = np.column_stack([x[:, -1], x[:, 0]])
    plain = make_regressor("ridge").fit(x, y)
    assert np.allclose(plain.predict(x), make_regressor("ridge").fit(x[:, :BASE_FEATURES], y)
                       .predict(x[:, :BASE_FEATURES]))
    hist = make_regressor("ridge:hist").fit(x, y)
    assert np.abs(hist.predict(x)[:, 0] - y[:, 0]).mean() < 0.05
    assert np.abs(plain.predict(x)[:, 0] - y[:, 0]).mean() > 0.5


def test_dropout_mlp_fits_and_pickles():
    import pickle

    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, BASE_FEATURES + 20))
    x[:, BASE_FEATURES - 1] = np.tile(np.arange(20), 20)  # 20 segments of 20 steps
    y = np.column_stack([np.sin(x[:, 0]), x[:, -1]])
    model = make_regressor("net:32x32:d0.1:s300:hist").fit(x, y)
    again = pickle.loads(pickle.dumps(model))
    assert again.predict(x).shape == (400, 2)
    assert np.allclose(again.predict(x[:5]), model.predict(x[:5]))
    assert ((model.predict(x) - y) ** 2).mean() < y.var(0).mean()


def test_learned_hook_tracks_the_previous_torque_per_segment():
    from types import SimpleNamespace

    from limb_baseline.memo import OnlineWarmStart

    class Echo:
        def predict(self, features):
            return np.tile(features[:, -7:], (1, 12))

    learner = OnlineWarmStart(estimator=Echo(), mpc=MPCConfig())
    hook = learner.as_hook(np.zeros(6))
    state = SimpleNamespace(limb_positions=np.zeros(6), robot_positions=np.zeros(13),
                            limb_velocities=np.zeros(6), robot_velocities=np.zeros(13))
    assert np.allclose(hook(state, np.zeros(6), 0), 0.0)
    hook.previous = np.full(7, 2.0)
    assert np.allclose(hook(state, np.zeros(6), 1)[0], 2.0)
    assert np.allclose(hook(state, np.zeros(6), 0), 0.0)


def test_training_rows_match_what_the_hook_sees():
    from types import SimpleNamespace

    from limb_baseline.planner import dense_rows
    from limb_baseline.utils import RolloutLog

    mpc = MPCConfig()
    repeat, steps = mpc.action_repeat, 3
    rng = np.random.default_rng(0)
    torques = rng.normal(size=(steps * repeat, 7)).tolist()
    start = SimpleNamespace(limb_positions=rng.normal(size=6), robot_positions=rng.normal(size=13),
                            limb_velocities=rng.normal(size=6), robot_velocities=rng.normal(size=13))
    log = RolloutLog()
    for _ in range(steps * repeat):
        log.limb_positions.append(rng.normal(size=6).tolist())
        log.limb_velocities.append(rng.normal(size=6).tolist())
        log.robot_positions.append(rng.normal(size=13).tolist())
        log.robot_velocities.append(rng.normal(size=13).tolist())
    goal = SimpleNamespace(positions=np.ones(6))
    rows = dense_rows([(start, goal, SimpleNamespace(robot_torques=torques))], log,
                      np.zeros(6), mpc)
    assert len(rows) == steps
    for step, row in enumerate(rows):
        if step == 0:
            state, previous = start, None
        else:
            i = step * repeat - 1
            state = SimpleNamespace(
                limb_positions=log.limb_positions[i], robot_positions=log.robot_positions[i],
                limb_velocities=log.limb_velocities[i], robot_velocities=log.robot_velocities[i])
            previous = torques[(step - 1) * repeat]
        expected = state_features(state, np.ones(6), np.zeros(6), step, previous)
        assert np.allclose(row.features, expected)


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


def test_the_memo_serves_stored_plans_by_goal_and_seeds_the_mpc_on_a_hit(monkeypatch):
    """A goal seen before seeds the MPC from its stored plan; a new goal seeds nothing."""
    from types import SimpleNamespace

    from kinder_pddlstream_planning.limbrepositioning3d.stream import TorqueTrajectory

    from limb_baseline import memo as memo_module

    mpc = MPCConfig()
    # Zero torque for one control step, then a constant push.
    torques = [[0.0] * 7] * mpc.action_repeat + [[1.0] * 7] * (2 * mpc.action_repeat)
    trajectory = TorqueTrajectory(torques)

    def fake_mpc(ctx, s1, q2):
        if q2.positions[0] > 5.0:
            return
        yield trajectory, "state-after"

    monkeypatch.setattr(memo_module, "plan_limb_motion", fake_mpc)
    # Step mode, so a hit hands the seed to the MPC rather than rolling it out here.
    monkeypatch.setattr(memo_module.SEED, "mode", "step")
    memo = memo_module.LimbMotionMemo()
    stream = memo_module.SeededStream(memo)
    memo.stream = stream.stream
    ctx = SimpleNamespace(mpc=mpc, warm_start=None)
    goal = SimpleNamespace(positions=(0.5, 0.0, 0.0, 0.0, 0.0, 0.0))
    near = SimpleNamespace(positions=(0.52, 0.0, 0.0, 0.0, 0.0, 0.0))
    far = SimpleNamespace(positions=(0.8, 0.0, 0.0, 0.0, 0.0, 0.0))
    unreachable = SimpleNamespace(positions=(9.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    assert list(memo.stream(ctx, "s", unreachable)) == []
    assert memo.size == 0 and (memo.hits, memo.misses) == (0, 1)

    assert list(memo.stream(ctx, "s", goal)) == [(trajectory, "state-after")]
    assert memo.size == 1 and ctx.warm_start is None and memo.misses == 2

    # Within the key's resolution: a hit, and the seed comes from the stored torques.
    memo.reset()
    assert list(memo.stream(ctx, "s", near)) == [(trajectory, "state-after")]
    assert (memo.hits, memo.misses) == (1, 0) and ctx.warm_start is not None
    seeded = ctx.warm_start(None, None, 1)
    assert seeded.shape == (mpc.horizon, 7) and np.allclose(seeded, 1.0)
    # From step 0 the horizon opens with the zero-torque step, verbatim.
    assert np.allclose(ctx.warm_start(None, None, 0)[0], 0.0)
    assert np.allclose(ctx.warm_start(None, None, 0)[1:], 1.0)
    assert np.allclose(ctx.warm_start(None, None, 99), 1.0)
    # The same key now holds two plans, capped at the best few.
    assert memo.size == 2

    # A goal outside the resolution misses and plans from scratch.
    memo.reset()
    assert list(memo.stream(ctx, "s", far)) == [(trajectory, "state-after")]
    assert (memo.hits, memo.misses) == (0, 1) and ctx.warm_start is None


def test_the_memo_routes_the_planner_stream_and_can_be_removed():
    from kinder_pddlstream_planning.limbrepositioning3d import run as planner_run

    from limb_baseline import memo as memo_module

    stream = memo_module.SeededStream(memo_module.LimbMotionMemo())
    memo_module.install(stream)
    assert planner_run.plan_limb_motion == stream.stream
    memo_module.uninstall()
    assert planner_run.plan_limb_motion is memo_module.plan_limb_motion


def test_the_dream_loop_memorizes_only_imagined_successes_and_charges_only_real_planning(
    monkeypatch, tmp_path
):
    """Dreams fill the cache before the real episode; a failed dream leaves nothing."""
    from types import SimpleNamespace

    from kinder_pddlstream_planning.limbrepositioning3d.stream import TorqueTrajectory

    from limb_baseline import dream_pipeline as dream_module
    from limb_baseline.planner import EpisodeResult, PlannerSettings
    from limb_baseline.tasks import Task

    select_variant(DEFAULT_VARIANT)
    nominal = np.zeros(6)
    tasks = []
    for k in range(3):
        goal = nominal.copy()
        goal[active_joints()[0]] = 0.5 + 0.3 * k
        tasks.append(Task(tuple(nominal), tuple(np.zeros(7)), tuple(goal)))
    monkeypatch.setattr(dream_module, "isolate_worker", lambda: None)
    monkeypatch.setattr(
        dream_module, "make_task_set",
        lambda theta, rng, config: dream_module.TaskSet(tasks[0], tasks),
    )

    # The stand-in planner: a person narrower than 0.8 cannot reach the far goals, so
    # dreams on such hypotheses fail and must not leave a plan behind.
    calls = []

    def fake_run_episode(task, theta, belief, believed_theta, warm_start, seed, settings,
                         particle_limits=None):
        calls.append((tuple(task.limb_goal), float(theta[0])))
        goal = SimpleNamespace(positions=task.limb_goal)
        stream = dream_module.memo_module.planner_run.plan_limb_motion
        trajectory = TorqueTrajectory([[1.0] * 7] * settings.mpc.action_repeat)
        # A stock stream (plain) is the real planner; skip it, nothing to record.
        if stream is not dream_module.memo_module.plan_limb_motion:
            ctx = SimpleNamespace(mpc=settings.mpc, warm_start=None)
            monkeypatch.setattr(
                dream_module.memo_module, "plan_limb_motion",
                lambda ctx, s1, q2: iter([(trajectory, "s2")]),
            )
            monkeypatch.setattr(dream_module.SEED, "mode", "step")
            list(stream(ctx, "s1", goal))
        reached = task.limb_goal[active_joints()[0]] <= float(theta[0])
        return EpisodeResult(solved=True, goal_reached=reached, plan_time=1.0)

    monkeypatch.setattr(dream_module, "run_episode", fake_run_episode)
    config = dream_module.DreamConfig(n_dreams=2, num_episodes=2, num_goals=3)
    settings = PlannerSettings(max_time=1.0)
    out = tmp_path / "cache.csv"
    records = dream_module.run_sequence(
        "cache", config, person=22, task_seed=0, model_name="ridge", settings=settings,
        out=out, tag="t",
    )
    assert len(records) == 2 and all(r["dreams_planned"] == 2 for r in records)
    # Two dreams per episode came before each real episode: 2 * (2 + 1) planner calls.
    assert len(calls) == 6
    dreams = __import__("pandas").read_csv(out.with_name("cache_dreams.csv"))
    assert len(dreams) == 4
    # Only plans whose imagined execution reached the goal stay in the cache. The last
    # record counts the cache before its own episode was gated, hence the + 1.
    memo = dream_module.memo_module
    expected = int(dreams["goal_reached"].sum()) + int(records[0]["goal_reached"]) + 1
    assert records[-1]["memo_size"] == expected < 6
    assert records[-1]["dream_time"] >= 0.0 and records[0]["belief_revised"] is False

    # Plain ignores dreams entirely and writes no dream rows.
    calls.clear()
    plain_out = tmp_path / "plain.csv"
    plain = dream_module.run_sequence(
        "plain", config, person=22, task_seed=0, model_name="ridge", settings=settings,
        out=plain_out, tag="t",
    )
    assert len(calls) == 2 and plain[0]["n_dreams"] == 0
    assert not plain_out.with_name("plain_dreams.csv").exists()
    assert memo.planner_run.plan_limb_motion is memo.plan_limb_motion


def test_particle_limits_are_a_chance_constraint_over_the_belief():
    """A configuration is believed inside when enough particles allow it."""
    from limb_baseline.planner import ParticleLimits

    select_variant(DEFAULT_VARIANT)
    joint, sign = active_joints()[0], active_signs()[0]
    # Four people, evenly spread; every other motion at the population mean.
    scales = np.array([0.75, 0.85, 0.95, 1.05])
    particles = np.full((4, len(active_motions())), 0.9)
    particles[:, 0] = scales
    limits = ParticleLimits(particles, confidence=0.5)
    assert limits.lower.shape == limits.upper.shape == (4, 6)

    def at(scale):
        q = np.zeros(6)
        q[joint] = sign * scale * default_magnitudes()[0]
        return q

    # Just short of the third person's bound: two of four allow it, exactly half.
    assert limits(at(0.94)) and not limits(at(0.96))
    assert ParticleLimits(particles, confidence=0.75)(at(0.84))
    assert not ParticleLimits(particles, confidence=0.75)(at(0.86))
    assert ParticleLimits(particles, confidence=1.0)(at(0.74))
    # The quantile theta is the range goals are posed against: the 0.5 quantile
    # of the spread with confidence 0.5, the narrowest person at confidence 1.
    assert np.isclose(limits.quantile_theta()[0], np.quantile(scales, 0.5))
    assert np.isclose(ParticleLimits(particles, 1.0).quantile_theta()[0], 0.75)
    assert np.allclose(limits.quantile_theta()[1:], 0.9)


def test_a_floor_truncates_the_belief_and_narrows_it():
    """Reaching a motion safely rules out every narrower person."""
    belief = Belief.from_prior(DEFAULT_PRIOR)
    raised = belief.raised(1, 1.0)
    assert np.allclose(raised.lower, [DEFAULT_PRIOR.low, 1.0, DEFAULT_PRIOR.low])
    assert raised.width < belief.width
    assert raised.planning_mean[1] > belief.planning_mean[1] >= belief.mean[1] - 1e-9
    draws = np.array([raised.sample(np.random.default_rng(i)) for i in range(200)])
    assert draws[:, 1].min() >= 1.0 and draws.max() <= DEFAULT_PRIOR.high
    # A floor survives a query update, and never rises above the population's top.
    updated = raised.updated(0, 1.0, DEFAULT_QUERY_MODEL.tau)
    assert np.allclose(updated.lower, raised.lower)
    assert raised.raised(1, 5.0).lower[1] < DEFAULT_PRIOR.high
    assert len(raised.key_features()) == 3 * len(belief.mean)


def test_an_execution_updates_the_belief_as_the_notebook_does():
    """Inside the person: a floor. Past the limit: the crossing angle is the limit."""
    from limb_baseline.dream_pipeline import observe

    select_variant(DEFAULT_VARIANT)
    joint, sign = active_joints()[0], active_signs()[0]
    person = np.array([0.8, 0.9, 1.0])
    belief = Belief.from_prior(DEFAULT_PRIOR)

    def path(scales):
        q = np.zeros((len(scales), 6))
        q[:, joint] = sign * np.asarray(scales) * default_magnitudes()[0]
        return q.tolist()

    inside, notes = observe(belief, path([0.2, 0.5, 0.75, 0.6]), person)
    assert notes == ["floor:0@0.750"]
    assert inside.lower[0] == pytest.approx(0.75) and inside.mean[0] == belief.mean[0]

    crossed, notes = observe(belief, path([0.5, 0.79, 0.81, 0.9]), person, tau=1e-6)
    assert notes[0].startswith("crossed:0@0.81")
    assert crossed.mean[0] == pytest.approx(0.81, abs=1e-3) and crossed.floor is None
    assert crossed.stddev[0] < 1e-3

    same, notes = observe(inside, path([0.1]), person)
    assert notes == [] and same is inside
    same, notes = observe(inside, path([0.7505]), person)
    assert notes == [] and same is inside
    assert observe(belief, [], person) == (belief, [])


def test_a_goal_beyond_the_belief_is_retargeted_just_inside_it():
    """As the notebook's planner stops at the nearest point it believes reachable."""
    from limb_baseline.dream_pipeline import RETARGET_MARGIN, Believed, DreamConfig
    from limb_baseline.planner import EpisodeResult, joint_limits
    from limb_baseline.dream_pipeline import goal_distance
    from limb_baseline.tasks import Task

    select_variant(DEFAULT_VARIANT)
    belief = Belief.from_prior(DEFAULT_PRIOR)
    believed = Believed(belief)
    joint, sign = active_joints()[0], active_signs()[0]
    init = np.zeros(6)

    def task_to(scale):
        goal = np.zeros(6)
        goal[joint] = sign * scale * default_magnitudes()[0]
        return Task(tuple(init), tuple(np.zeros(7)), tuple(goal))

    inside = task_to(0.5)
    same, moved = believed.retarget(inside)
    assert same is inside and not moved

    beyond = task_to(1.1)
    target, moved = believed.retarget(beyond)
    assert moved
    lower, upper = joint_limits(believed.theta)
    edge = upper[joint] if sign > 0 else lower[joint]
    assert target.limb_goal[joint] == pytest.approx(edge - sign * RETARGET_MARGIN)
    assert np.allclose(np.delete(target.limb_goal, joint), np.delete(beyond.limb_goal, joint))
    assert target.limb_init == beyond.limb_init

    # With particles, the retargeted goal passes the particle vote.
    rng = np.random.default_rng(0)
    particled = Believed.build(belief, DreamConfig(particles=16, confidence=0.9), rng)
    target, moved = particled.retarget(beyond)
    assert moved and particled.limits(np.asarray(target.limb_goal))

    # Scored from where the limb ended, or from the start when nothing ran.
    # The environment's metric wraps angles, so compare against it directly.
    from limb_baseline.dream_pipeline import distance

    stopped = EpisodeResult(limb_path=[list(init), list(target.limb_goal)])
    assert goal_distance(beyond, stopped) == pytest.approx(
        distance(target.limb_goal, beyond.limb_goal))
    assert 0.0 < goal_distance(beyond, stopped) < goal_distance(beyond, EpisodeResult())
    assert goal_distance(beyond, EpisodeResult()) == pytest.approx(
        distance(init, beyond.limb_goal))


def test_the_hard_deadline_stops_any_code_at_the_limit_and_cleans_up():
    """Even a loop that swallows every Exception is stopped on time."""
    import signal
    import time

    from limb_baseline.planner import PlanningTimeout, hard_deadline

    before = signal.getsignal(signal.SIGALRM)
    started = time.perf_counter()
    with pytest.raises(PlanningTimeout, match="0.3 s hard planning time limit"):
        with hard_deadline(0.3):
            while True:
                try:
                    sum(range(10_000))
                except Exception:  # a stream's broad handler must not stop it
                    pass
    elapsed = time.perf_counter() - started
    assert 0.29 <= elapsed < 0.4
    assert signal.getsignal(signal.SIGALRM) == before
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)

    # Finishing early cancels the alarm; no limit means no alarm at all.
    with hard_deadline(0.2):
        pass
    time.sleep(0.3)
    with hard_deadline(None):
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_the_belief_memo_serves_the_nearest_belief_within_tolerance():
    """The notebook's nearest-neighbour cache: goal near-exact, belief by distance."""
    from kinder_pddlstream_planning.limbrepositioning3d.stream import TorqueTrajectory

    from limb_baseline.dream_pipeline import BeliefMemo

    def plan(steps):
        return TorqueTrajectory([[0.0] * 7] * steps)

    memo = BeliefMemo(tolerance=0.15)
    base = np.full(9, 0.9)
    memo.current = base
    memo.store((5, 0), plan(30))
    memo.store((5, 0), plan(20), belief=base + 0.1)
    memo.store((5, 0), plan(10), belief=base + 0.3)

    # Exact belief: served, and at distance 0 even though a shorter plan exists nearby.
    assert len(memo.lookup((5, 0)).robot_torques) == 30 and memo.last_distance == 0.0
    # Nearest within tolerance wins: 0.12 away from the 20-step plan's belief is 0.02.
    memo.current = base + 0.12
    assert len(memo.lookup((5, 0)).robot_torques) == 20
    assert memo.last_distance == pytest.approx(0.02)
    # Beyond tolerance on any one number: a miss, however close the others are.
    memo.current = base.copy()
    memo.current[4] += 0.5
    assert memo.lookup((5, 0)) is None
    # Another goal is another key; the size counts every filed plan.
    assert memo.lookup((6, 0)) is None and memo.size == 3
    # The same rounded belief keeps at most KEEP_PER_KEY plans, shortest first.
    memo.current = base
    for steps in (40, 50, 60, 5):
        memo.store((7, 0), plan(steps))
    assert sorted(len(e[1].robot_torques) for e in memo.plans[(7, 0)]) == [5, 40, 50, 60]
    memo.store((7, 0), plan(70))
    assert len(memo.plans[(7, 0)]) == 4 and memo.last_stored[0] == (7, 0)


def test_a_dream_refiles_its_plan_instead_of_searching_again():
    """Same aim point under the imagined belief: refile. A moved aim point: search."""
    from kinder_pddlstream_planning.limbrepositioning3d.stream import TorqueTrajectory

    from limb_baseline.dream_pipeline import BeliefMemo, Believed, Memory, refile
    from limb_baseline.memo import goal_key
    from limb_baseline.tasks import Task

    select_variant(DEFAULT_VARIANT)
    joint, sign = active_joints()[0], active_signs()[0]
    belief = Belief.from_prior(DEFAULT_PRIOR)
    raised = belief.raised(0, 1.0)

    def task_to(scale):
        goal = np.zeros(6)
        goal[joint] = sign * scale * default_magnitudes()[0]
        return Task(tuple(np.zeros(6)), tuple(np.zeros(7)), tuple(goal))

    memo = BeliefMemo()
    memory = Memory("cache", SimpleStream(), memo, None)
    memo.current = belief.key_features()
    inside = task_to(0.5)
    memo.store(goal_key(inside.limb_goal), TorqueTrajectory([[0.0] * 7] * 12))

    # A goal inside both beliefs aims at the same place: refiled, now served there.
    assert refile(inside, Believed(belief), Believed(raised), memory)
    memo.current = raised.key_features()
    assert memo.lookup(goal_key(inside.limb_goal)) is not None
    # A goal beyond the old belief is aimed further under the raised one: search.
    beyond = task_to(1.15)
    assert not refile(beyond, Believed(belief), Believed(raised), memory)


class SimpleStream:
    key_suffix = ()


def test_append_row_keeps_columns_aligned_when_rows_differ(tmp_path):
    """A row with fewer, extra or reordered keys lands under the right headers."""
    import pandas as pd

    path = tmp_path / "rows.csv"
    append_row(path, {"a": 1, "b": 2, "c": 3})
    append_row(path, {"c": 30, "a": 10})            # reordered, one missing
    append_row(path, {"a": 100, "b": 200, "c": 300, "d": 4})  # one extra
    frame = pd.read_csv(path)
    assert list(frame.columns) == ["a", "b", "c"]
    assert frame["a"].tolist() == [1, 10, 100]
    assert frame["c"].tolist() == [3, 30, 300]
    assert frame["b"].isna().tolist() == [False, True, False]
