"""BedsideTask on the shared API."""

from types import SimpleNamespace

import numpy as np
import pytest

from able.api import (
    Belief,
    InfoGatherer,
    Observation,
    Planner,
    PlanResult,
    Prior,
    PsiSpec,
    Task,
    make_learner,
    run_episode,
)
from able.tasks.bedside import BedsideTask, PlaceProposal, reach_distance
from able.tasks.bedside.planner.run import _base_path, build_stream_context

THETA = np.array([0.7, 0.6, 0.65])


@pytest.fixture(scope="module")
def task():
    return BedsideTask()


def test_protocols(task):
    rng = np.random.default_rng(0)
    assert isinstance(task, Task)
    assert isinstance(task.prior, Prior)
    assert isinstance(task.gatherer, InfoGatherer)
    assert isinstance(task.psi, PsiSpec)
    assert isinstance(task.make_planner(), Planner)
    assert isinstance(task.prior.belief(rng), Belief)
    assert task.prior.sample(rng).shape == (3,)


def test_query_narrows_the_belief_along_its_direction(task):
    rng = np.random.default_rng(0)
    belief = task.prior.belief(rng)
    d = task.gatherer.choose(belief, rng)
    before = reach_distance(d, belief.particles).std()
    for _ in range(3):
        belief = belief.update(task.gatherer.respond(THETA, d, rng))
    r = reach_distance(d, belief.particles)
    assert r.std() < before / 3
    assert abs(r.mean() - reach_distance(d, THETA)) < 0.02


def test_execution_likelihood_sign(task):
    belief = task.prior.belief(np.random.default_rng(0))
    point = np.asarray(belief.shoulder) + [0.6, 0.0, 0.0]
    small, large = np.array([[0.5, 0.6, 0.6]]), np.array([[0.9, 0.6, 0.6]])
    for reached in (True, False):
        obs = Observation("execution", reached, tuple(point))
        gap = belief.log_likelihood(obs, large)[0] - belief.log_likelihood(obs, small)[0]
        assert gap > 0 if reached else gap < 0


def test_psi_rows(task):
    rng = np.random.default_rng(0)
    belief = task.prior.belief(rng)
    state = task.reset(task.sample_problem(rng), THETA, rng)
    x = task.psi.features(state, belief, "req1")
    assert x.shape == (8 + len(task.psi.slots) + 6,)
    assert x[1] == 1 and x[:8].sum() == 1
    assert x[8:8 + len(task.psi.slots)].sum() == task.num_clutter  # clutter starts on slot centers
    trace = [("req0", (0.7, 0.3, 0.6), "side_table"), ("clutter0", (2.2, 0.0, 0.6), "dresser"),
             ("req1", (0.7, 0.0, 0.7), "nightstand"), ("req0", (0.8, 0.3, 0.6), "side_table")]
    episode = SimpleNamespace(steps=[SimpleNamespace(state=state, result=PlanResult([], 0.0, 1.0, trace))])
    rows = task.psi.examples(episode, belief)
    assert rows.x.shape == (2, len(x)) and rows.y.shape == (2, 3)
    assert np.allclose(rows.y[0], (0.8, 0.3, 0.6))  # the final placement


def test_place_order_ranks_the_snapped_sample_first(task):
    rng = np.random.default_rng(0)
    state = task.reset(task.sample_problem(rng), THETA, rng)
    plain = build_stream_context(task.sim, state.obs)
    points = plain.slot_points("req0")
    assert sorted(plain.place_order("req0")) == list(range(len(points)))
    j = len(points) // 2
    samples = points[j] + rng.normal(0, 0.005, (16, 3))
    ctx = build_stream_context(task.sim, state.obs, proposal=PlaceProposal({"req0": samples}))
    order = ctx.place_order("req0")
    assert order[0] == j and sorted(order) == list(range(len(points)))
    assert sorted(ctx.place_order("req1")) == list(range(len(points)))  # no samples: random


def test_base_path_turns_without_collision(task):
    from pybullet_helpers.geometry import SE2Pose

    task.reset(task.sample_problem(np.random.default_rng(0)), THETA, np.random.default_rng(0))
    task.sim.robot.set_base(SE2Pose(1.75, 0.9, 0.0))
    path = _base_path(task.sim, SE2Pose(1.28, 0.21, -np.pi))
    assert np.allclose([path[-1].x, path[-1].y, abs(path[-1].rot)], [1.28, 0.21, np.pi])
    assert task.sim.robot.get_base().x == pytest.approx(1.75)


@pytest.mark.slow
def test_episode_under_belief_then_with_psi(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    task = BedsideTask(num_clutter=0, max_time=60.0)
    rng = np.random.default_rng(0)
    problem = task.sample_problem(rng)
    belief = task.prior.belief(rng)
    planner = task.make_planner()
    first = run_episode(task, planner, problem, THETA, belief, rng)
    assert first.success and first.effort > 0
    assert len(first.observations) == 2 and all(o.value for o in first.observations)
    rows = task.psi.examples(first, belief)
    learner = make_learner("flow", width=32, depth=2, steps=200, seed=0).fit(rows.x, rows.y, rows.groups)
    second = run_episode(task, planner, problem, THETA, belief, rng,
                         proposal_fn=lambda s, b: task.psi.proposal(learner, s, b))
    assert second.success
    assert not list(tmp_path.iterdir())  # pddlstream ran in its own temp dir
