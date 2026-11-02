"""BedsideTask on the shared API."""

from types import SimpleNamespace

import numpy as np
import pytest

from able.api import (
    Belief,
    InfoGatherer,
    Observation,
    ParticleBelief,
    Planner,
    PlanResult,
    Prior,
    PsiSpec,
    Task,
    TaskDistribution,
    make_learner,
    run_episode,
)
from able.planners.pddlstream import PDDLStreamPlanner
from able.tasks.bedside import (
    BedsideBelief,
    BedsideDistribution,
    BedsideGatherer,
    BedsidePlanner,
    BedsidePrior,
    BedsideProblem,
    BedsidePsi,
    BedsideQuery,
    BedsideTask,
    PlaceProposal,
    reach_distance,
)
from able.tasks.bedside.env.execute import _base_path
from able.tasks.bedside.planning.problem import build_stream_context
from able.tasks.bedside.query import PRESENT_TOL, present_conf
from able.video import render

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


def test_role_classes_inherit_the_api_bases(task):
    rng = np.random.default_rng(0)
    for cls, base in ((BedsideTask, Task), (BedsideDistribution, TaskDistribution), (BedsidePrior, Prior),
                      (BedsideBelief, ParticleBelief), (BedsideGatherer, InfoGatherer),
                      (BedsidePlanner, PDDLStreamPlanner), (BedsidePlanner, Planner), (BedsidePsi, PsiSpec)):
        assert issubclass(cls, base)
    assert isinstance(task.distribution, BedsideDistribution)
    assert isinstance(task.prior.belief(rng), BedsideBelief)
    assert isinstance(task.gatherer.choose(task.prior.belief(rng), rng), BedsideQuery)
    problem = task.sample_problem(np.random.default_rng(3))
    assert problem == BedsideProblem(int(np.random.default_rng(3).integers(2**31 - 1)))


@pytest.fixture(scope="module")
def exact(task):
    """A gatherer whose person answers m >= 0 at every presented point."""
    return BedsideGatherer(task.gatherer.directions, task.config, noisy=False, stop_at_miss=False)


def test_query_reached_near_not_far_and_skips_unreachable_points(exact):
    d = exact.directions[0]  # r(d, THETA) = 0.655; the robot cannot present 0.3 m (arm hits the person)
    obs = exact.respond(THETA, BedsideQuery(d, (0.3, 0.6, 0.8)), np.random.default_rng(0))
    sim = exact.sim
    assert obs.kind == "query" and [reached for _, reached in obs.value] == [True, False]
    for (point, _), s in zip(obs.value, (0.6, 0.8)):
        assert np.linalg.norm(np.array(point) - (sim.shoulder + s * d)) < PRESENT_TOL
        assert present_conf(sim, np.array(point)) is not None
    assert present_conf(sim, sim.shoulder + 0.3 * d) is None
    assert not sim._robot_or_held_object_collision_exists()  # pylint: disable=protected-access


def test_query_leaves_the_task_sim_and_episode_start_unchanged(task, exact):
    rng = np.random.default_rng(0)
    problem = task.sample_problem(rng)
    before = task.reset(problem, THETA, rng)
    base, joints = task.sim.robot.get_base(), task.sim.robot.arm.get_joint_positions()
    exact.respond(THETA, BedsideQuery(exact.directions[0], (0.6, 0.8)), rng)
    assert exact.sim is not task.sim and exact.sim.physics_client_id != task.sim.physics_client_id
    assert task.sim.robot.get_base() == base and task.sim.robot.arm.get_joint_positions() == joints
    after = task.reset(problem, THETA, rng)
    assert after.obs.allclose(before.obs, atol=1e-9)


def test_query_likelihood_matches_execution(task):
    belief = task.prior.belief(np.random.default_rng(0))
    points = [tuple(np.asarray(belief.shoulder) + [0.5, 0.0, 0.0]), tuple(np.asarray(belief.shoulder) + [0.8, 0.0, 0.0])]
    query = Observation("query", list(zip(points, (True, False))))
    executions = [Observation("execution", r, x) for x, r in zip(points, (True, False))]
    total = sum(belief.log_likelihood(o, belief.particles) for o in executions)
    assert np.allclose(belief.log_likelihood(query, belief.particles), total)
    assert np.allclose(belief.log_likelihood(Observation("query", []), belief.particles), 0.0)


@pytest.mark.slow
def test_queries_narrow_the_belief_toward_theta(task):
    theta = np.array([0.5, 0.75, 0.85])  # away from the prior mean
    rng = np.random.default_rng(0)
    belief = task.prior.belief(rng)
    dirs = task.gatherer.directions
    d = task.gatherer.choose(belief, rng).direction

    def reach_error(b):  # mean |r(d, particle) - r(d, theta)| over every direction
        return np.abs(reach_distance(dirs[:, None], b.particles[None]) - reach_distance(dirs, theta)[:, None]).mean()

    before, error, spread = reach_distance(d, belief.particles).std(), np.abs(belief.mean - theta), reach_error(belief)
    frames: list = []
    for i in range(5):
        query = task.gatherer.choose(belief, rng)
        assert len(query.distances) and np.all(np.diff(query.distances) > 0)
        obs = task.gatherer.respond(theta, query, rng, frames=frames if i == 0 else None)
        assert obs.value and (i == 0 or task.gatherer.last_seconds < 5.0)  # rendering dominates query 0
        belief = belief.update(obs)
    r = reach_distance(d, belief.particles)
    assert r.std() < before / 3 and abs(r.mean() - reach_distance(d, theta)) < 0.03
    assert reach_error(belief) < spread / 2
    assert (np.abs(belief.mean - theta)[:2] < error[:2] / 2).all()
    assert len(frames) > 5 and all(f.ndim == 3 and f.dtype == np.uint8 for f in frames)


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


def test_render(task):
    task.reset(task.sample_problem(np.random.default_rng(0)), THETA, np.random.default_rng(0))
    plain, captioned = render(task.sim), render(task.sim, "place req0 on nightstand")
    for f in (plain, captioned):
        assert f.ndim == 3 and f.shape[2] == 3 and f.dtype == np.uint8
    assert plain.shape == captioned.shape and (plain != captioned).any()


@pytest.mark.slow
def test_episode_frames(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    task = BedsideTask(num_clutter=0, max_time=60.0)
    rng = np.random.default_rng(0)
    frames: list = []
    episode = run_episode(task, task.make_planner(), task.sample_problem(rng), THETA, task.prior.belief(rng), rng,
                          frames=frames)
    assert episode.success and len(frames) > 10
    assert all(f.ndim == 3 and f.shape[2] == 3 and f.dtype == np.uint8 for f in frames)
    assert len({f.shape for f in frames}) == 1


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
