"""Predictive-sampling MPC over control points, optionally seeded by psi.

Each step samples candidate control sequences around the shifted last plan, scores them by
rolling out, and commits the best. With a psi hook, the seed and the learner's draws are scored
first and committed if they beat a model-independent bar; otherwise the full batch runs, reusing
those rollouts. Subclasses supply the rollouts (see the abstract methods).
"""

from __future__ import annotations

import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from able.api import Belief, Budget, Planner, PlanResult
from able.planners.deadline import PlanningTimeout, hard_deadline


class NoFeasibleMotion(RuntimeError):
    """Every candidate broke a constraint."""


@dataclass
class SeedConfig:
    rollouts: int = 8  # seed + draws scored before the full batch
    tolerance: float = 0.0  # accept within this fraction of the bar's cost


@dataclass
class SeedStats:
    seeded_steps: int = 0
    full_steps: int = 0
    scored_rollouts: int = 0
    full_rollouts: int = 0
    model_seconds: float = 0.0
    seed_margin: list = field(default_factory=list)  # best seed cost / bar, per step

    @property
    def rollouts(self) -> int:
        return self.scored_rollouts + self.full_rollouts


def control_basis(horizon: int, num_control_points: int) -> NDArray[np.float64]:
    """(horizon, points) linear interpolation from control points to per-step controls."""
    source = np.linspace(0.0, 1.0, num_control_points)
    target = np.linspace(0.0, 1.0, horizon)
    return np.stack([np.interp(target, source, np.eye(num_control_points)[k])
                     for k in range(num_control_points)], axis=1)


class PredictiveSampling:
    """`cfg` needs num_rollouts, horizon, num_control_points, commit_steps, noise_scale,
    noise_reference_error, min_noise_ratio."""

    def __init__(self, cfg, dim: int, lower, upper, goal, rng: np.random.Generator, hook=None,
                 seed_config: SeedConfig | None = None, stats: SeedStats | None = None, start=None) -> None:
        self._cfg, self._dim, self._goal, self._rng = cfg, dim, goal, rng
        self._lower, self._upper = lower, upper
        self._hook = hook
        self._seed_cfg = seed_config or SeedConfig()
        self._stats = stats or SeedStats()
        self._nominal = np.zeros((cfg.num_control_points, dim))
        self._control_indices = np.round(np.linspace(0, cfg.horizon - 1, cfg.num_control_points)).astype(int)
        self._commit = int(np.clip(cfg.commit_steps, 1, cfg.horizon))
        self._step_index = 0
        self._scored = 0
        if hook is not None and start is not None:
            seeded = np.asarray(hook(start, goal, 0), dtype=np.float64)
            self._nominal = self._clip(self._as_points(seeded))

    # ---- subclass interface

    def _error(self) -> float:
        """Distance to the goal now."""
        raise NotImplementedError

    def _capture(self) -> Any:
        """The current state, to roll out from."""
        raise NotImplementedError

    def _rollout(self, start, controls: NDArray[np.float64]) -> tuple[int, float]:
        """Roll per-step `controls` out from `start`: (rejected flag, cost)."""
        raise NotImplementedError

    def _rollout_error(self) -> float:
        """Distance to the goal at the end of the last rollout."""
        raise NotImplementedError

    def _reach(self) -> float:
        """A rollout ending closer than this counts as reaching the goal."""
        raise NotImplementedError

    def _directed_candidates(self, scale: float) -> list[NDArray[np.float64]]:
        """Goal-directed control points (consumes rng); part of the bar."""
        return []

    def _feedback_candidates(self, start) -> list[NDArray[np.float64]]:
        """Feedback-controller control points; the bar when present."""
        return []

    # ---- shared

    def _clip(self, controls: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.clip(controls, self._lower, self._upper)

    def _expand(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Control points -> one control per step, linearly interpolated."""
        source = np.linspace(0.0, 1.0, len(points))
        target = np.linspace(0.0, 1.0, self._cfg.horizon)
        return np.stack([np.interp(target, source, points[:, j]) for j in range(self._dim)], axis=1)

    def _as_points(self, controls: NDArray[np.float64]) -> NDArray[np.float64]:
        """Per-step controls (horizon rows) projected onto control points; points pass through."""
        if controls.shape[0] != self._cfg.horizon:
            return controls
        basis = control_basis(self._cfg.horizon, self._cfg.num_control_points)
        return np.linalg.lstsq(basis, controls, rcond=None)[0]

    def _score(self, start, controls):
        self._scored += 1
        return self._rollout(start, controls)

    def _scale(self, error: float) -> float:
        cfg = self._cfg
        return cfg.noise_scale * float(np.clip(error / cfg.noise_reference_error, cfg.min_noise_ratio, 1.0))

    def _sample_candidates(self, error: float, state=None, num_rollouts: int | None = None,
                           use_hook: bool = True, use_feedback: bool = True) -> list[NDArray[np.float64]]:
        """Shifted nominal, zero, psi's seed, goal-directed, feedback, then noise around the nominal."""
        num_rollouts = self._cfg.num_rollouts if num_rollouts is None else num_rollouts
        scale = self._scale(error)
        expanded = self._expand(self._nominal)
        tail = self._rng.normal(expanded[-1], scale)
        ramp = np.linspace(expanded[-1], tail, self._commit + 1)[1:]
        nominal = np.vstack([expanded[self._commit:], ramp])[self._control_indices]
        candidates = [self._clip(nominal), np.zeros_like(nominal)]
        if use_hook and self._hook is not None and state is not None:
            candidates.append(self._clip(np.asarray(self._hook(state, self._goal, self._step_index), np.float64)))
        candidates.extend(self._directed_candidates(scale))
        if use_feedback and state is not None:
            candidates.extend(self._feedback_candidates(state))
        noise = self._rng.normal(scale=scale, size=(max(num_rollouts - len(candidates), 0),
                                                    self._cfg.num_control_points, self._dim))
        candidates.extend(self._clip(nominal + sample) for sample in noise)
        return candidates

    def _commit_best(self, nominal, controls) -> list:
        self._nominal = nominal
        self._step_index += self._commit
        return [list(c) for c in controls[: self._commit]]

    def step(self) -> list:
        """The controls to apply next (`commit_steps` of them)."""
        if self._hook is None or self._seed_cfg.rollouts <= 0:
            self._stats.full_steps += 1
            committed = self._full_step()
        else:
            committed = self._seeded_step()
        if self._hook is not None:
            self._hook.previous = committed[-1]
        return committed

    def _full_step(self) -> list:
        before = self._scored
        try:
            start = self._capture()
            candidates = self._sample_candidates(self._error(), start, self._cfg.num_rollouts,
                                                 use_hook=True, use_feedback=True)
            scores = [self._score(start, self._expand(cp)) for cp in candidates]
            index = min(range(len(scores)), key=scores.__getitem__)
            if scores[index][0]:
                raise NoFeasibleMotion("all MPC candidates violate constraints")
            return self._commit_best(candidates[index], self._expand(candidates[index]))
        finally:
            self._stats.full_rollouts += self._scored - before

    def _seeded_step(self) -> list:
        stats, k = self._stats, self._seed_cfg.rollouts - 1
        error = self._error()
        start = self._capture()
        started = time.process_time()
        seed = self._clip(np.asarray(self._hook(start, self._goal, self._step_index), float))
        scale = self._scale(error)
        sampler = getattr(self._hook, "samples", None)
        if sampler is not None and k > 0:
            batch = [seed] + [self._clip(np.asarray(p, float)) for p in sampler(start, self._goal, self._step_index, k)]
        else:
            batch = [seed] + [self._clip(seed + self._rng.normal(scale=scale, size=seed.shape)) for _ in range(k)]
        stats.model_seconds += time.process_time() - started
        exact = seed.shape[0] == self._cfg.horizon  # per-step controls, not control points

        scored = []  # (nominal, controls, (rejected, cost))
        # Model-independent bar, so a poor policy cannot vouch for itself.
        bar_candidates = self._feedback_candidates(start) or self._directed_candidates(scale)
        bar = float("inf")
        for points in bar_candidates:
            controls = self._expand(points)
            scored.append((points, controls, self._score(start, controls)))
            if not scored[-1][2][0]:
                bar = min(bar, scored[-1][2][1] * (1.0 + self._seed_cfg.tolerance))
        chosen, reach = None, self._reach()
        for points in batch:
            controls = points if exact else self._expand(points)
            nominal = self._clip(self._as_points(points)) if exact else points
            scored.append((nominal, controls, self._score(start, controls)))
            rejected, cost = scored[-1][2]
            if not rejected and (self._rollout_error() < reach or cost <= bar):
                chosen = scored[-1]
                break
        model_costs = [c for _, _, (r, c) in scored[len(bar_candidates):] if not r]
        stats.seed_margin.append((chosen[2][1] if chosen else min(model_costs, default=float("inf"))) / bar)
        stats.scored_rollouts += len(scored)
        if chosen is not None:
            stats.seeded_steps += 1
            return self._commit_best(chosen[0], chosen[1])
        stats.full_steps += 1
        return self._finish_batch(start, error, scored)

    def _finish_batch(self, start, error, scored) -> list:
        """The full batch, minus the rollouts seeding already scored."""
        fresh = self._sample_candidates(error, start, max(self._cfg.num_rollouts - len(scored), 0),
                                        use_hook=False, use_feedback=False)
        for points in fresh:
            controls = self._expand(points)
            scored.append((points, controls, self._score(start, controls)))
        self._stats.full_rollouts += len(fresh)
        nominal, controls, (rejected, _) = min(scored, key=lambda entry: entry[2])
        if rejected:
            raise NoFeasibleMotion("all MPC candidates violate constraints")
        return self._commit_best(nominal, controls)


@dataclass
class MPCPlanner(Planner):
    """Times one MPC plan under a hard deadline and charges its rollouts; subclasses `solve`."""

    max_time: float = 600.0
    seed_config: SeedConfig = field(default_factory=SeedConfig)
    effort_unit: str = "mpc_rollouts"

    @abstractmethod
    def solve(self, state: Any, belief: Belief, proposal: Any, stats: SeedStats,
              rng: np.random.Generator) -> tuple[Any, dict]:
        """(plan or None, info); MPCs it builds count into `stats`."""

    def plan(self, state: Any, belief: Belief, proposal: Any = None, budget: Budget | None = None,
             rng: np.random.Generator | None = None) -> PlanResult:
        rng = rng if rng is not None else np.random.default_rng(getattr(state, "seed", None))
        limit = budget.seconds if budget is not None and budget.seconds else self.max_time
        stats, started = SeedStats(), time.time()
        try:
            with hard_deadline(limit):
                plan, info = self.solve(state, belief, proposal, stats, rng)
        except PlanningTimeout as timeout:
            plan, info = None, {"failure": str(timeout)}
        info = {**info, "seeded_steps": stats.seeded_steps, "full_steps": stats.full_steps,
                "model_seconds": stats.model_seconds}
        return PlanResult(plan, time.time() - started, stats.rollouts, info=info)
