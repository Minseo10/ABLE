"""The friction task on the shared able.api."""

import math

import numpy as np
import pytest

from able.api import (
    Belief,
    Episode,
    InfoGatherer,
    Observation,
    ParticleBelief,
    Planner,
    Prior,
    PsiSpec,
    Task,
    TaskDistribution,
    make_learner,
    run_episode,
)
from able.planners.pomcpow import POMCPOWConfig, POMCPOWPlanner
from able.tasks.friction import (
    FrictionBelief,
    FrictionDistribution,
    FrictionGatherer,
    FrictionPlanner,
    FrictionPrior,
    FrictionProblem,
    FrictionPsi,
    FrictionQuery,
    FrictionTask,
)
from able.tasks.friction.env.puckpushing_utils import GRAVITY, Push
from able.tasks.friction.psi import X_DIM, Y_DIM, encode


@pytest.fixture(scope="module")
def task():
    return FrictionTask(num_particles=100, max_pushes=2, config=POMCPOWConfig(num_sims=150, max_depth=4))


@pytest.fixture(scope="module")
def episode(task):
    rng = np.random.default_rng(0)
    theta = np.array([0.12, 0.08, 0.03])
    return run_episode(task, task.make_planner(), task.sample_problem(rng), theta, task.prior.belief(rng), rng,
                       max_steps=task.max_pushes)


def test_components_satisfy_the_protocols(task):
    belief = task.prior.belief(np.random.default_rng(0))
    assert isinstance(task, Task)
    assert isinstance(task.prior, Prior)
    assert isinstance(task.gatherer, InfoGatherer)
    assert isinstance(task.psi, PsiSpec)
    assert isinstance(task.make_planner(), Planner)
    assert isinstance(belief, Belief)
    assert task.prior.sample(np.random.default_rng(0)).shape == (3,)
    assert set(belief.summary()) == {"blue", "tan", "green"}


def test_role_classes_inherit_the_bases(task):
    assert issubclass(FrictionTask, Task) and issubclass(FrictionDistribution, TaskDistribution)
    assert issubclass(FrictionPrior, Prior) and issubclass(FrictionBelief, ParticleBelief)
    assert issubclass(FrictionGatherer, InfoGatherer) and issubclass(FrictionPsi, PsiSpec)
    assert issubclass(FrictionPlanner, POMCPOWPlanner)
    assert isinstance(task.distribution, FrictionDistribution)
    assert isinstance(task.gatherer.choose(task.prior.belief(np.random.default_rng(0)), None), FrictionQuery)
    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
    assert task.sample_problem(rng_a) == FrictionProblem(int(rng_b.integers(2**31)))


def test_probe_poses_fit_their_floor(task):
    """Each floor's probe, at either extreme friction, stops on that floor and clear of holes."""
    world, (low, high) = task.world, task.prior.dist.bounds
    rng = np.random.default_rng(0)
    for floor, query in enumerate(task.gatherer.queries):
        assert query.floor == floor == world.surface_at(np.array(query.puck))
        assert world.push_feasible(np.array(query.puck), query.angle)
        for mu in (low[floor], high[floor]):
            theta = np.full(3, 0.08)
            theta[floor] = mu
            obs = task.gatherer.respond(theta, query, rng)
            v = obs.value
            assert obs.kind == "query" and obs.query == query
            assert v["feasible"] and not v["fell_in_hole"] and v["release_speed"] > 0
            assert world.surface_at(np.array(v["stop"])) == floor
            assert np.linalg.norm(np.array(v["stop"]) - query.puck) > world.physics.accel_distance


def test_three_probes_narrow_each_floor(task):
    rng = np.random.default_rng(1)
    theta = np.array([0.15, 0.03, 0.08])
    before = belief = task.prior.belief(rng)
    for query in task.gatherer.queries:
        belief = belief.update(task.gatherer.respond(theta, query, rng))
    assert np.all(belief.std < 0.3 * before.std)
    assert np.all(np.abs(belief.mean - theta) < np.abs(before.mean - theta))
    assert np.all(np.abs(belief.mean - theta) < 0.2 * theta)


def test_queries_run_in_their_own_env(task):
    """A probe leaves the task's simulator untouched; the episode starts at the problem's start."""
    rng = np.random.default_rng(2)
    theta = np.array([0.12, 0.08, 0.03])
    problem = task.sample_problem(rng)
    state = task.reset(problem, theta, rng)
    env = task.environment

    def snapshot():
        return (env._puck_position().tolist(), env._robot_pose.tolist(), env.frictions,
                str(env.np_random.bit_generator.state))

    held = snapshot()
    probe = task.gatherer.respond(np.array([0.05, 0.1, 0.15]), task.gatherer.queries[1], rng)
    assert task.gatherer.environment is not env and snapshot() == held
    assert np.allclose(state.puck, task.world.puck_map.puck_start)
    assert np.allclose(env._puck_position(), task.world.puck_map.puck_start)
    assert np.allclose(env._robot_pose[:2], task.world.puck_map.robot_start)
    step = task.execute(state, Push(0.0, 0.3), rng)
    assert set(step.observations[0].value) == set(probe.value)
    # The probe env returns to the map's start when reset without a placement.
    task.gatherer.environment.reset(seed=0)
    assert np.allclose(task.gatherer.environment._puck_position(), task.world.puck_map.puck_start)


def test_push_likelihood_matches_the_old_math(task):
    """A clear straight slide along the blue corridor, by hand."""
    phys = task.world.physics
    puck, angle, speed = np.array([1.0, 3.4]), 0.0, 0.5
    stop = np.array([1.45, 3.41])
    particles = np.array([[0.10, 0.08, 0.03], [0.06, 0.08, 0.03]])
    obs = Observation("execution", {"puck": puck.tolist(), "angle": angle, "stance": None, "release_speed": speed,
                                    "stop": stop.tolist(), "fell_in_hole": False})
    belief = task.prior.belief(np.random.default_rng(0))
    got = belief.log_likelihood(obs, particles)

    def by_hand(mu):
        length = speed**2 / (2 * mu * GRAVITY)
        along, across = stop[0] - (puck[0] + phys.accel_distance + length), stop[1] - puck[1]
        var_along = phys.position_obs_noise**2 + (2 * length * phys.speed_obs_noise / speed) ** 2
        var_across = phys.position_obs_noise**2 + (length * phys.angle_noise) ** 2
        return -0.5 * (along**2 / var_along + across**2 / var_across) - 0.5 * math.log(var_along * var_across)

    assert got == pytest.approx([by_hand(0.10), by_hand(0.06)], abs=1e-9)
    fell = Observation("execution", {**obs.value, "fell_in_hole": True})
    assert belief.log_likelihood(fell, particles) == pytest.approx(got + math.log(0.05))
    posterior = belief.update(obs)
    assert posterior.std[0] < belief.std[0]
    assert np.all(posterior.particles >= belief.low) and np.all(posterior.particles <= belief.high)


def test_short_episode_runs(task, episode):
    assert isinstance(episode, Episode)
    assert 1 <= len(episode.steps) <= task.max_pushes
    assert episode.effort > 0 and episode.steps[0].result.effort == 150
    first = episode.steps[0]
    assert isinstance(first.result.plan, Push) and first.result.trace
    assert first.step.state.pushes == 1
    if first.step.info["feasible"]:
        assert first.step.observations[0].kind == "execution"


def test_psi_examples_and_proposal(task, episode):
    rows = task.psi.examples(episode, episode.belief)
    assert rows.x.shape[1] == X_DIM == 13 and rows.y.shape[1] == Y_DIM == 4
    assert len(rows) >= len(episode.steps) and len(rows.groups) == len(rows)
    chosen = episode.steps[0].result.plan
    assert any(np.allclose(y, encode(chosen)) for y in rows.y[rows.groups == 0])

    rng = np.random.default_rng(0)
    model = task.model
    start = task.reset(task.sample_problem(rng), np.array([0.12, 0.08, 0.03]), rng)
    node = model.initial_state(start.puck, start.robot, (0.1, 0.1, 0.1))
    belief = task.prior.belief(rng)
    x = np.repeat(task.psi.features(start, belief)[None], 300, 0) + rng.normal(0, 0.01, (300, X_DIM))
    y = np.array([encode(model.propose(node, [], rng)) for _ in range(300)])
    learner = make_learner("flow", width=32, depth=2, steps=200, seed=0).fit(x, y)
    task.psi.mix = 1.0
    try:
        propose = task.psi.proposal(learner, start, belief)
        tried = []
        for _ in range(30):
            push = propose(node, [node], tried, rng)
            assert isinstance(push, Push) and push not in tried
            assert model.world.push_feasible(node.puck_array, push.angle, push.stance)
            assert np.isclose(model.stances, push.stance).any()
            tried.append(push)
    finally:
        task.psi.mix = 0.5


@pytest.mark.slow
def test_3d_episode_smoke():
    task = FrictionTask(env="3d", num_particles=50, max_pushes=1, config=POMCPOWConfig(num_sims=50, max_depth=3))
    rng = np.random.default_rng(0)
    episode = run_episode(task, task.make_planner(), task.sample_problem(rng), task.prior.sample(rng),
                          task.prior.belief(rng), rng, max_steps=1)
    assert len(episode.steps) == 1 and episode.effort == 50
    assert episode.steps[0].step is not None and episode.steps[0].step.done


@pytest.mark.slow
def test_3d_probe():
    task = FrictionTask(env="3d", num_particles=50)
    rng = np.random.default_rng(0)
    theta = np.array([0.03, 0.15, 0.08])
    for query in task.gatherer.queries:
        v = task.gatherer.respond(theta, query, rng).value
        assert v["feasible"] and not v["fell_in_hole"]
        assert task.world.surface_at(np.array(v["stop"])) == query.floor
    assert task._env is None  # the task's own simulator is never built by queries
