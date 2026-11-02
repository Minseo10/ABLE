"""Limb pieces: theta, belief, queries, variants, task sampling, features, deadlines."""

import os

import numpy as np
import pytest
from kinder.envs.dynamic3d.limbs import (
    DEFAULT_RANGE_OF_MOTION,
    get_human_joint_limits,
)

from able.tasks.limb.belief import (
    LimbBelief,
    RoMGaussian,
    apply_evidence,
    execution_evidence,
)
from able.tasks.limb.distribution import DEFAULT_TASK_BOX, inactive_joints
from able.tasks.limb.env.human import (
    DEFAULT_PRIOR,
    active_joints,
    active_motions,
    active_signs,
    default_magnitudes,
    theta_to_bounds,
    theta_to_joint_limits,
)
from able.tasks.limb.env.query import DEFAULT_QUERY_MODEL, QueryModel
from able.tasks.limb.env.variant import DEFAULT_VARIANT, ENV_VAR, VARIANTS
from able.tasks.limb.env.variant import _build as _build_variant
from able.tasks.limb.env.variant import select as select_variant
from able.tasks.limb.planning.stream import MPCConfig
from able.tasks.limb.psi import (
    BASE_FEATURES,
    control_basis,
    project_torques,
    segment_features,
    state_features,
)
from able.tasks.limb.query import LimbGatherer


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
    belief = RoMGaussian.from_prior(DEFAULT_PRIOR)
    model = DEFAULT_QUERY_MODEL
    for _ in range(4):
        for i in range(len(truth)):
            answer = truth[i] * magnitudes[i] - model.short_fall
            belief = belief.updated(i, answer, model.tau, model.short_fall)
    assert np.all(np.abs(belief.mean - truth) < 0.02)
    assert np.all(belief.stddev < DEFAULT_PRIOR.stddev)


def test_a_noiseless_answer_is_inverted_exactly_in_the_limit():
    """With a vanishing tau the posterior mean is the answer divided by the range."""
    magnitudes = default_magnitudes()
    belief = RoMGaussian.from_prior(DEFAULT_PRIOR)
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
        task = DEFAULT_TASK_BOX.sample(rng, theta)
        lower, upper = theta_to_joint_limits(theta)
        for positions in (task.limb_init, task.limb_goal):
            assert np.all(np.asarray(positions) >= lower - 1e-9)
            assert np.all(np.asarray(positions) <= upper + 1e-9)
        assert task.separation >= DEFAULT_TASK_BOX.min_separation


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


def test_training_rows_match_what_the_hook_sees():
    from types import SimpleNamespace

    from able.tasks.limb.env.execute import RolloutLog
    from able.tasks.limb.psi import dense_rows

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


@pytest.mark.slow
def test_a_query_stops_short_of_the_passive_bound():
    """The whole observation model: the person stops inside their own range."""
    from able.tasks.limb.env.query import LimbQuerySimulator

    theta = np.array([0.9, 1.0, 0.8])
    bounds = np.abs(theta_to_bounds(theta))
    simulator = LimbQuerySimulator()
    try:
        rng = np.random.default_rng(0)
        for i in range(3):
            outcome = simulator.ask(theta, i, bounds[i] + 0.5, rng)
            shortfall = bounds[i] - outcome.answer
            assert not outcome.reached_target
            assert 0.0 < shortfall < 2 * DEFAULT_QUERY_MODEL.zone_width
    finally:
        simulator.close()


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


def test_particle_limits_are_a_chance_constraint_over_the_belief():
    """A configuration is believed inside when enough particles allow it."""
    from able.tasks.limb.env.sim import ParticleLimits

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
    belief = RoMGaussian.from_prior(DEFAULT_PRIOR)
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


def test_the_hard_deadline_stops_any_code_at_the_limit_and_cleans_up():
    """Even a loop that swallows every Exception is stopped on time."""
    import signal
    import time

    from able.planners.deadline import PlanningTimeout, hard_deadline

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


def test_the_target_is_drawn_around_the_default_range_of_motion():
    """The robot does not know the person's range, so it asks around the default one."""
    gatherer = LimbGatherer()
    belief = LimbBelief(RoMGaussian.from_prior(DEFAULT_PRIOR))
    rng = np.random.default_rng(0)
    for _ in range(50):
        query = gatherer.choose(belief, rng)
        target, boundary = query.target, default_magnitudes()[query.motion_index]
        assert boundary - gatherer.probe_band <= target <= boundary + gatherer.probe_band


def test_a_raised_floor_raises_the_next_target():
    """Reaching proves only that the bound is at least there, so the next ask goes higher."""
    gatherer = LimbGatherer(probe_band=0.0)
    belief = LimbBelief(RoMGaussian.from_prior(DEFAULT_PRIOR))
    rng = np.random.default_rng(0)
    query = gatherer.choose(belief, rng)
    raised = LimbBelief(belief.gaussian.raised(query.motion_index, 1.1))
    assert gatherer.choose(raised, rng).target > query.target


def test_an_execution_updates_the_belief_as_the_notebook_does():
    """Inside the person: a floor. Past the limit: the crossing angle is the limit."""
    select_variant(DEFAULT_VARIANT)
    joint, sign = active_joints()[0], active_signs()[0]
    person = np.array([0.8, 0.9, 1.0])
    belief = RoMGaussian.from_prior(DEFAULT_PRIOR)

    def observe(b, scales, tau=DEFAULT_QUERY_MODEL.tau):
        q = np.zeros((len(scales), 6))
        q[:, joint] = sign * np.asarray(scales) * default_magnitudes()[0]
        return apply_evidence(b, execution_evidence(q.tolist() if len(scales) else [], person), tau)

    inside, notes = observe(belief, [0.2, 0.5, 0.75, 0.6])
    assert notes[0] == "floor:0@0.750"
    assert inside.lower[0] == pytest.approx(0.75) and inside.mean[0] == belief.mean[0]
    crossed, notes = observe(belief, [0.5, 0.79, 0.81, 0.9], tau=1e-6)
    assert notes[0].startswith("crossed:0@0.81")
    assert crossed.mean[0] == pytest.approx(0.81, abs=1e-3) and crossed.stddev[0] < 1e-3
    assert observe(inside, [0.7505])[0].lower[0] == inside.lower[0]
    assert observe(belief, []) == (belief, [])


def test_a_goal_beyond_the_belief_is_retargeted_just_inside_it():
    """As the notebook's planner stops at the nearest point it believes reachable."""
    from able.tasks.limb.env.sim import ParticleLimits
    from able.tasks.limb.planner import retarget
    from able.tasks.limb.task import LimbProblem

    select_variant(DEFAULT_VARIANT)
    belief = LimbBelief(RoMGaussian.from_prior(DEFAULT_PRIOR))
    limits = ParticleLimits(belief.sample(np.random.default_rng(0), 16), 0.9)
    joint, sign = active_joints()[0], active_signs()[0]

    def task_to(scale):
        goal = np.zeros(6)
        goal[joint] = sign * scale * default_magnitudes()[0]
        return LimbProblem(tuple(np.zeros(6)), tuple(np.zeros(7)), tuple(goal))

    inside = task_to(0.5)
    assert retarget(inside, limits) == (inside, False)
    beyond = task_to(1.1)
    target, moved = retarget(beyond, limits)
    assert moved and limits(np.asarray(target.limb_goal))
    assert abs(target.limb_goal[joint]) < abs(beyond.limb_goal[joint])
    assert np.allclose(np.delete(target.limb_goal, joint), np.delete(beyond.limb_goal, joint))
