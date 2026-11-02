"""One logged friction-task episode (probes, then POMCPOW pushes); scripts/friction CLIs run it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from able.api import run_episode
from able.tasks.friction.task import FrictionProblem, FrictionTask


@dataclass
class StepLog:
    angle: float
    speed: float
    stance: float | None  # base heading relative to the push (None: the robot picked it)
    feasible: bool
    release_speed: float
    stop: list[float]
    fell_in_hole: bool
    goal_reached: bool
    drive_distance: float
    reward: float
    plan_seconds: float
    plan_sims: int
    root_actions: int
    ess: float
    belief: dict[str, list[float]]


@dataclass
class EpisodeLog:
    env: str
    map_name: str
    seed: int
    true_frictions: list[float]
    prior_belief: dict[str, list[float]]
    probes: list[dict] = field(default_factory=list)
    outcome: str = "timeout"
    total_reward: float = 0.0
    total_plan_seconds: float = 0.0
    steps: list[StepLog] = field(default_factory=list)


def run(task: FrictionTask, seed: int, frictions: tuple[float, ...] | None = None, probes: int = 0,
        proposal_fn: Callable[[Any, Any], Any] | None = None, frames: list | None = None) -> EpisodeLog:
    """Probe up to `probes` floors, then push until the goal, a hole, or `max_pushes`; frames go to `frames`."""
    rng = np.random.default_rng(seed)
    theta = np.asarray(task.prior.sample(rng) if frictions is None else frictions, np.float64)
    belief = task.prior.belief(rng)
    log = EpisodeLog(task.env, task.world.puck_map.name, seed, theta.tolist(), belief.summary())
    for _ in range(probes):
        query = task.gatherer.choose(belief, rng)
        if query is None:
            break
        answer = task.gatherer.respond(theta, query, rng)
        belief = belief.update(answer)
        log.probes.append({"floor": query.floor, "speed": query.speed, "distance": answer.value,
                           "belief": belief.summary()})
    episode = run_episode(task, task.make_planner(), FrictionProblem(seed), theta, belief, rng,
                          proposal_fn=proposal_fn, max_steps=task.max_pushes, frames=frames)
    for k, s in enumerate(episode.steps):
        if s.step is None:
            log.outcome = "stuck"
            break
        info = s.step.info
        after = episode.steps[k + 1].belief if k + 1 < len(episode.steps) else episode.belief
        log.steps.append(StepLog(
            angle=info["angle"], speed=info["speed"], stance=info["stance"], feasible=info["feasible"],
            release_speed=info["release_speed"], stop=info["stop"], fell_in_hole=info["fell_in_hole"],
            goal_reached=info["goal_reached"], drive_distance=info["drive_distance"], reward=info["reward"],
            plan_seconds=s.result.seconds, plan_sims=int(s.result.effort),
            root_actions=s.result.info["root_actions"], ess=_ess(s.belief, s.step.observations),
            belief=after.summary()))
        log.total_reward += info["reward"]
        if info["goal_reached"] or info["fell_in_hole"]:
            log.outcome = "goal" if info["goal_reached"] else "hole"
    log.total_plan_seconds = episode.plan_seconds
    return log


def _ess(belief, observations) -> float:
    """Effective sample size of the push's weights (nan if unobserved, 0 if none explain it)."""
    if not observations:
        return float("nan")
    log_w = belief.log_likelihood(observations[0], belief.particles)
    if not np.isfinite(log_w).any():
        return 0.0
    w = np.exp(log_w - log_w[np.isfinite(log_w)].max())
    w = np.where(np.isfinite(w), w, 0.0)
    return float(w.sum() ** 2 / (w**2).sum())
