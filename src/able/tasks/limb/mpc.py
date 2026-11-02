"""MPC seeded by psi: the seed (and the learner's draws) are batch 0 of every MPC step."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field

import numpy as np

from able.tasks.limb.features import control_basis
from able.tasks.limb.planning.stream import NoFeasibleMotion, PredictiveSamplingMPC
from able.tasks.limb.planning.utils import capture_state, limb_error


@dataclass
class SeedConfig:
    rollouts: int = 8  # seed + draws scored before the full batch
    tolerance: float = 0.0  # accept within this fraction of the feedback candidates' cost


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


class SeededMPC(PredictiveSamplingMPC):
    """Reads `ctx.warm_start` (hook), `ctx.seed_config`, `ctx.seed_stats`, `ctx.mpc_noise_seed`."""

    def __init__(self, ctx, goal, start=None) -> None:
        # Stock constructor expects control points; a torque horizon is projected below.
        hook, ctx.warm_start = ctx.warm_start, None
        try:
            super().__init__(ctx, goal, start)
        finally:
            ctx.warm_start = hook
        self._seed_cfg = getattr(ctx, "seed_config", None) or SeedConfig()
        self._stats = getattr(ctx, "seed_stats", None) or SeedStats()
        if getattr(ctx, "mpc_noise_seed", None) is not None:
            self._rng = np.random.default_rng(ctx.mpc_noise_seed)
        if hook is not None and start is not None:
            seeded = np.asarray(hook(start, goal, 0), dtype=np.float64)
            if seeded.shape[0] == self._cfg.horizon:
                seeded, *_ = np.linalg.lstsq(control_basis(self._cfg), seeded, rcond=None)
            self._nominal = self._clip(seeded)

    def step(self) -> list:
        committed = self._seeded_step()
        hook = self._ctx.warm_start
        if hook is not None:
            hook.previous = committed[-1]
        return committed

    def _seeded_step(self) -> list:
        hook, cfg, stats = self._ctx.warm_start, self._cfg, self._stats
        if hook is None or self._seed_cfg.rollouts <= 0:
            stats.full_steps += 1
            return self._full_step()
        error = limb_error(self._ctx, self._sim.limb.get_joint_positions(), self._goal)
        start = capture_state(self._sim)
        started = time.process_time()
        seed = self._clip(np.asarray(hook(start, self._goal, self._step_index), float))
        scale = cfg.noise_scale * float(np.clip(error / cfg.noise_reference_error, cfg.min_noise_ratio, 1.0))
        k = self._seed_cfg.rollouts - 1
        sampler = getattr(hook, "samples", None)
        if sampler is not None and k > 0:
            batch = [seed] + [self._clip(np.asarray(p, float))
                              for p in sampler(start, self._goal, self._step_index, k)]
        else:
            batch = [seed] + [self._clip(seed + self._rng.normal(scale=scale, size=seed.shape))
                              for _ in range(k)]
        stats.model_seconds += time.process_time() - started
        exact = seed.shape[0] == cfg.horizon  # per-step torques, not control points
        reach = max(self._ctx.goal_atol - cfg.replay_slack, self._ctx.goal_atol / 2)
        return self._reusing_step(start, error, scale, batch, exact, reach)

    def _reusing_step(self, start, error, scale, batch, exact, reach) -> list:
        """Score the bar, then seed and draws until one passes; a miss reuses them in the full batch."""
        cfg, stats = self._cfg, self._stats

        def as_nominal(points):
            if not exact:
                return points
            solution, *_ = np.linalg.lstsq(control_basis(cfg), points, rcond=None)
            return self._clip(solution)

        scored = []  # (nominal, torques, (rejected, cost))

        def score(nominal, torques):
            scored.append((nominal, torques, self._score(start, torques)))
            return scored[-1][2]

        # Model-independent bar, so a poor policy cannot vouch for itself.
        bar_candidates = self._feedback_candidates(start) or self._directed_candidates(scale)
        bar = float("inf")
        for points in bar_candidates:
            rejected, cost = score(points, self._expand(points))
            if not rejected:
                bar = min(bar, cost * (1.0 + self._seed_cfg.tolerance))
        chosen = None
        for points in batch:
            torques = points if exact else self._expand(points)
            rejected, cost = score(as_nominal(points), torques)
            terminal = limb_error(self._rollout_ctx, self._rollout_sim.limb.get_joint_positions(), self._goal)
            if not rejected and (terminal < reach or cost <= bar):
                chosen = scored[-1]
                break
        model_costs = [c for _, _, (r, c) in scored[len(bar_candidates):] if not r]
        stats.seed_margin.append((chosen[2][1] if chosen else min(model_costs, default=float("inf"))) / bar)
        stats.scored_rollouts += len(scored)
        if chosen is not None:
            nominal, torques, _ = chosen
            stats.seeded_steps += 1
            self._nominal = nominal
            self._step_index += self._commit
            return [list(t) for t in torques[: self._commit]]
        stats.full_steps += 1
        return self._finish_batch(start, error, scored)

    def _finish_batch(self, start, error, scored) -> list:
        """The stock step, minus the rollouts seeding already scored."""
        cfg, hook = self._cfg, self._ctx.warm_start
        self._cfg = dataclasses.replace(cfg, feedback_gains=(), num_rollouts=max(cfg.num_rollouts - len(scored), 0))
        self._ctx.warm_start = None
        try:
            fresh = self._sample_candidates(error, start)
        finally:
            self._cfg, self._ctx.warm_start = cfg, hook
        for points in fresh:
            torques = self._expand(points)
            scored.append((points, torques, self._score(start, torques)))
        self._stats.full_rollouts += len(fresh)
        nominal, torques, (rejected, _) = min(scored, key=lambda entry: entry[2])
        if rejected:
            raise NoFeasibleMotion("all MPC candidates violate constraints")
        self._nominal = nominal
        self._step_index += self._commit
        return [list(t) for t in torques[: self._commit]]

    def _full_step(self) -> list:
        before = self._scored
        try:
            return super().step()
        finally:
            self._stats.full_rollouts += self._scored - before

    _scored = 0

    def _score(self, *args, **kwargs):
        self._scored += 1
        return super()._score(*args, **kwargs)
