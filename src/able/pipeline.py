"""The ABLE loop: gather -> dream -> refit psi -> plan with psi (plain fallback) -> execute -> update.

Conditions: `plain` (no psi), `learned` (dreams + refits), `prior_only` (psi fit once at the prior, frozen).
"""

from __future__ import annotations

import csv
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from able.api import Budget, Episode, Examples, make_learner, run_episode

CONDITIONS = ("plain", "learned", "prior_only")


@dataclass
class PipelineConfig:
    condition: str = "learned"
    episodes: int = 20
    dreams: int = 4  # per episode (learned only)
    queries: int = 1  # per episode
    info_gathering: bool = True  # False: skip queries and the belief update from them (stages 1-2)
    belief_update: bool = True  # False: the belief stays at the prior (no human model learning at all)
    seconds: float | None = None  # planning budget per call; None = planner default
    model: str = "flow"
    model_options: dict = field(default_factory=dict)
    fallback: bool = True


@dataclass
class Memory:
    """psi's training rows and the learner fit on them."""

    x: list = field(default_factory=list)
    y: list = field(default_factory=list)
    groups: list = field(default_factory=list)
    learner: Any = None

    def add(self, rows: Examples) -> int:
        if not len(rows):
            return 0
        offset = (max(self.groups) + 1) if self.groups else 0
        self.x += list(rows.x)
        self.y += list(rows.y)
        self.groups += list(rows.groups - rows.groups.min() + offset)
        return len(rows)

    def refit(self, config: PipelineConfig) -> dict:
        if not self.x:
            return {}
        started = time.time()
        self.learner = make_learner(config.model, **config.model_options).fit(
            np.array(self.x), np.array(self.y), np.array(self.groups))
        return {"fit_seconds": time.time() - started, "rows": len(self.x),
                "val_nmse": float(getattr(self.learner, "val_nmse", float("nan")))}


def proposal_fn(task, learner):
    if learner is None:
        return None
    return lambda state, belief: task.psi.proposal(learner, state, belief)


def gather(task, belief, theta, rng, n: int):
    """Up to `n` queries; returns the belief and the queries asked."""
    asked = 0
    for _ in range(n):
        query = task.gatherer.choose(belief, rng)
        if query is None:
            break
        belief = belief.update(task.gatherer.respond(theta, query, rng))
        asked += 1
    return belief, asked


def dream(task, planner, belief, memory: Memory, config: PipelineConfig, rng) -> dict:
    """Imagined people from the belief; solved dreams' rows go under the imagined next belief."""
    started, solved, rows = time.time(), 0, 0
    for _ in range(config.dreams):
        theta = belief.sample(rng, 1)[0]
        episode = run_episode(task, planner, task.sample_problem(rng), theta, belief, rng,
                              proposal_fn(task, memory.learner), _budget(config), fallback=config.fallback,
                              update_belief=config.belief_update)
        if episode.success:
            solved += 1
            rows += memory.add(task.psi.examples(episode, episode.belief))
    return {"dream_seconds": time.time() - started, "dreams_solved": solved, "dream_rows": rows}


def _budget(config: PipelineConfig) -> Budget | None:
    return Budget(config.seconds) if config.seconds else None


def record(task, planner, config: PipelineConfig, person: int, index: int, theta, episode: Episode,
           extra: dict) -> dict:
    last = episode.steps[-1] if episode.steps else None
    info = {**(last.result.info if last else {}), **(last.step.info if last and last.step else {})}
    scalars = {f"info_{k}": v for k, v in info.items() if isinstance(v, (bool, int, float, str))}
    return {
        "task": task.name, "condition": config.condition, "person": person, "episode": index,
        "info_gathering": config.info_gathering, "belief_update": config.belief_update,
        "success": episode.success, "plan_seconds": episode.plan_seconds, "effort": episode.effort,
        "effort_unit": planner.effort_unit, "steps": len(episode.steps),
        "fallback": any(s.result.info.get("fallback") for s in episode.steps),
        "theta": json.dumps(np.asarray(theta).round(4).tolist()),
        "belief": json.dumps(episode.belief.summary()), **extra, **scalars,
    }


def run_person(task, config: PipelineConfig, person: int, out: Path, seed: int = 0,
               prior_model: Path | None = None, resume: bool = False) -> list[dict]:
    """One person's sequence of episodes; writes `<out>.csv` and a checkpoint per episode."""
    assert config.condition in CONDITIONS, config.condition
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = out.with_suffix(".pkl")
    planner = task.make_planner()
    if resume and ckpt.exists():
        state = pickle.loads(ckpt.read_bytes())
        rng, theta, belief, memory, records = (state[k] for k in ("rng", "theta", "belief", "memory", "records"))
    else:
        rng = np.random.default_rng([person, seed])
        theta = task.prior.sample(rng)
        belief, memory, records = task.prior.belief(rng), Memory(), []
        if config.condition == "prior_only":
            memory.learner = pickle.loads(Path(prior_model).read_bytes())
    for index in range(len(records), config.episodes):
        asked = 0
        if config.info_gathering and config.belief_update:
            belief, asked = gather(task, belief, theta, rng, config.queries)
        extra = {"queries": asked}
        if config.condition == "learned":
            extra.update(dream(task, planner, belief, memory, config, rng))
            if extra["dream_rows"]:
                extra.update(memory.refit(config))
        episode = run_episode(task, planner, task.sample_problem(rng), theta, belief, rng,
                              proposal_fn(task, memory.learner), _budget(config), fallback=config.fallback,
                              update_belief=config.belief_update)
        belief = episode.belief
        if config.condition == "learned" and episode.success:
            if memory.add(task.psi.examples(episode, belief)):
                extra.update(memory.refit(config))
        records.append(record(task, planner, config, person, index, theta, episode, extra))
        _write_csv(out.with_suffix(".csv"), records)
        ckpt.write_bytes(pickle.dumps({"rng": rng, "theta": theta, "belief": belief, "memory": memory,
                                       "records": records}))
    return records


def factory_rows(task, config: PipelineConfig, dreams: int, seed: int = 0) -> Memory:
    """One shard of prior_only's rows: dreams at the prior belief."""
    rng = np.random.default_rng([10**6, seed])
    memory = Memory()
    dream(task, task.make_planner(), task.prior.belief(rng), memory,
          PipelineConfig(**{**config.__dict__, "dreams": dreams}), rng)
    return memory


def fit_factory(shards: list[Memory], config: PipelineConfig):
    """One learner from every shard's rows."""
    memory = Memory()
    for shard in shards:
        if shard.x:
            memory.add(Examples(np.array(shard.x), np.array(shard.y), np.array(shard.groups)))
    if not memory.x:
        raise RuntimeError("no dream solved; nothing to learn")
    memory.refit(config)
    return memory.learner


def _write_csv(path: Path, records: list[dict]) -> None:
    columns = list(dict.fromkeys(k for r in records for k in r))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(records)
