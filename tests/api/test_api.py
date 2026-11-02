"""The shared API on a toy task: theta is a 1-d reach; the goal is beyond or within it."""

from dataclasses import dataclass, field

import numpy as np
import pytest

from able.api import (
    Belief,
    Examples,
    InfoGatherer,
    Observation,
    ParticleBelief,
    Planner,
    PlanResult,
    Prior,
    PsiSpec,
    Step,
    Task,
    TaskDistribution,
    make_learner,
    run_episode,
)

NOISE = 0.02


@dataclass(frozen=True)
class ReachBelief(ParticleBelief):
    def log_likelihood(self, obs, particles):
        return -0.5 * ((obs.value - particles[:, 0]) / NOISE) ** 2


@dataclass(frozen=True)
class ReachPrior(Prior):
    def sample(self, rng):
        return rng.uniform(0.2, 1.0, 1)

    def belief(self, rng):
        return ReachBelief(rng.uniform(0.2, 1.0, (400, 1)), np.array([0.2]), np.array([1.0]))


class AskMax(InfoGatherer):
    def choose(self, belief, rng):
        return "max"

    def respond(self, theta, query, rng):
        return Observation("query", float(theta[0] + rng.normal(0, NOISE)), query)


class StepPlanner(Planner):
    """Moves at most 0.3 per step toward the goal (or psi's target)."""

    effort_unit = "steps"

    def plan(self, state, belief, proposal=None, budget=None, rng=None):
        target = state["goal"] if proposal is None else proposal
        move = float(np.clip(target - state["x"], -0.3, 0.3))
        return PlanResult(move, seconds=0.0, effort=1.0, trace=target)


class ReachPsi(PsiSpec):
    def examples(self, episode, belief):
        x = [np.concatenate([[s.state["goal"]], belief.features()]) for s in episode.steps]
        y = [[s.result.trace] for s in episode.steps]
        return Examples(np.array(x), np.array(y), np.zeros(len(x), np.int64))

    def proposal(self, learner, state, belief):
        x = np.concatenate([[state["goal"]], belief.features()])[None]
        return float(learner.predict(x)[0, 0])


class ReachDistribution(TaskDistribution):
    def sample(self, rng):
        return {"goal": float(rng.uniform(0.2, 1.0))}


@dataclass
class ReachTask(Task):
    name: str = "reach"
    prior: ReachPrior = field(default_factory=ReachPrior)
    gatherer: AskMax = field(default_factory=AskMax)
    psi: ReachPsi = field(default_factory=ReachPsi)
    distribution: ReachDistribution = field(default_factory=ReachDistribution)

    def make_planner(self):
        return StepPlanner()

    def reset(self, problem, theta, rng):
        return {"x": 0.0, "goal": problem["goal"], "theta": float(theta[0])}

    def execute(self, state, plan, rng, frames=None):
        x = min(state["x"] + plan, state["theta"])
        new = {**state, "x": x}
        done = abs(x - min(state["goal"], state["theta"])) < 0.05 or x >= state["theta"] - 1e-9
        obs = [Observation("execution", x)] if x >= state["theta"] - 1e-9 else []
        return Step(new, obs, done, abs(x - state["goal"]) < 0.05)


def test_toy_task_satisfies_the_protocols():
    task = ReachTask()
    belief = task.prior.belief(np.random.default_rng(0))
    assert isinstance(task, Task)
    assert isinstance(task.prior, Prior)
    assert isinstance(task.gatherer, InfoGatherer)
    assert isinstance(task.psi, PsiSpec)
    assert isinstance(task.make_planner(), Planner)
    assert isinstance(belief, Belief)


def test_query_narrows_the_particle_belief():
    rng = np.random.default_rng(0)
    task = ReachTask()
    belief = task.prior.belief(rng)
    theta = np.array([0.6])
    for _ in range(3):
        belief = belief.update(task.gatherer.respond(theta, task.gatherer.choose(belief, rng), rng))
    assert belief.std[0] < 0.05
    assert abs(belief.mean[0] - 0.6) < 0.05
    assert belief.features().shape == (2,)


def test_run_episode_loops_until_done():
    rng = np.random.default_rng(1)
    task = ReachTask()
    episode = run_episode(task, task.make_planner(), {"goal": 0.8}, np.array([1.0]),
                          task.prior.belief(rng), rng)
    assert episode.success
    assert len(episode.steps) == 3  # 0.3 + 0.3 + 0.2
    assert episode.effort == 3.0
    rows = task.psi.examples(episode, episode.belief)
    assert rows.x.shape == (3, 3) and rows.y.shape == (3, 1)


def test_failed_plan_ends_the_episode():
    class Fails(StepPlanner):
        def plan(self, *a, **k):
            return PlanResult(None, 0.0, 5.0)

    rng = np.random.default_rng(0)
    task = ReachTask()
    episode = run_episode(task, Fails(), {"goal": 0.5}, np.array([1.0]), task.prior.belief(rng), rng)
    assert not episode.success and episode.effort == 5.0 and episode.steps[0].step is None


@pytest.mark.parametrize("kind", ["flow", "mlp"])
def test_learner_fits_a_conditional_map(kind):
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (600, 2))
    y = np.stack([x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]], 1)
    model = make_learner(kind, width=64, depth=2, steps=600, seed=0).fit(x, y, groups=np.arange(600) // 10)
    pred = model.predict(x[:50])
    assert pred.shape == (50, 2)
    assert np.mean((pred - y[:50]) ** 2) < 0.05
    assert model.sample(x[:5], k=4, seed=0).shape == (5, 4, 2)
    assert np.isfinite(model.val_nmse)


def test_flow_keeps_both_modes():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (2000, 1))
    y = np.where(rng.random((2000, 1)) < 0.5, 1.0, -1.0) + 0.05 * rng.normal(size=(2000, 1))
    model = make_learner("flow", width=128, depth=3, steps=3000, seed=0).fit(x, y)
    draws = model.sample(np.zeros((1, 1)), k=200, seed=0)[0, :, 0]
    assert (draws > 0.5).mean() > 0.4 and (draws < -0.5).mean() > 0.4
    assert np.isfinite(model.log_prob(x[:3], y[:3])).all()
