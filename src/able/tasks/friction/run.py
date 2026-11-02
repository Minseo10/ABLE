"""Run and log one friction-task episode with POMCPOW (2D PyMunk or 3D PyBullet env).

Example:
    python -m able.tasks.friction.run --env 2d --level L2 --seed 0 --sims 2000
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from able.api import run_episode
from able.planners.pomcpow import POMCPOWConfig
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=["2d", "3d"], default="2d")
    parser.add_argument("--level", default="L2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sims", type=int, default=1000)
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-pushes", type=int, default=15)
    parser.add_argument("--particles", type=int, default=500)
    parser.add_argument("--probes", type=int, default=0)
    parser.add_argument("--frictions", type=float, nargs="+", default=None)
    parser.add_argument("--stance", choices=["grid", "continuous", "auto"], default="grid")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    config = POMCPOWConfig(num_sims=args.sims, max_time=args.max_time, max_depth=args.max_depth)
    task = FrictionTask(env=args.env, level=args.level, num_particles=args.particles, max_pushes=args.max_pushes,
                        stance_mode=args.stance, config=config)
    start = time.perf_counter()
    log = run(task, args.seed, tuple(args.frictions) if args.frictions else None, probes=args.probes)
    for step in log.steps:
        means = {k: round(v[0], 3) for k, v in step.belief.items()}
        print(f"push angle={step.angle:+.2f} speed={step.speed:.2f} stop=({step.stop[0]:.2f},{step.stop[1]:.2f}) "
              f"reward={step.reward:.1f} plan={step.plan_seconds:.1f}s belief={means}")
    print(f"{log.outcome}: reward={log.total_reward:.1f} pushes={len(log.steps)} plan={log.total_plan_seconds:.1f}s "
          f"wall={time.perf_counter() - start:.1f}s true={np.round(log.true_frictions, 3).tolist()}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(asdict(log), indent=1))


if __name__ == "__main__":
    main()
