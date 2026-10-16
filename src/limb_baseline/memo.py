"""Plan from a cache of solved plans or a learned model first, and search only if that fails.

After the notebook's cache: on a hit (or with a model) the seed is rolled out closed-loop
from the grasped state with no search, one torque per control step, and kept as the plan
if it reaches the goal within the limits. Otherwise it is discarded and the stock MPC
plans from the original state, as plain does. `--seed-mode step` instead seeds every
MPC step and falls back per step. Stages 1-2 skipped, true range of motion.

    python -m limb_baseline.memo --condition warm --variation 0 --num-tasks 20 \\
        --out data/memo/same_task.csv
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
from kinder.envs.dynamic3d.limb_utils import (
    LIMB_JOINT_INFOS,
    NUM_ROBOT_JOINTS,
    joint_position_distance,
)
from kinder_pddlstream_planning.limbrepositioning3d import run as planner_run
from kinder_pddlstream_planning.limbrepositioning3d import stream as planner_stream
from kinder_pddlstream_planning.limbrepositioning3d.stream import (
    LimbStreamContext,
    MPCConfig,
    NoFeasibleMotion,
    TorqueTrajectory,
    plan_limb_motion,
)
from kinder_pddlstream_planning.limbrepositioning3d.utils import (
    CoupledState,
    advance,
    capture_state,
    commanded_robot_torque,
    exceeds_human_torque_limits,
    exceeds_robot_torque_limits,
    human_in_collision,
    limb_error,
    limb_out_of_limits,
    restore_state,
    saved_sim_state,
)

from limb_baseline.dataset import append_row
from limb_baseline.human import DEFAULT_PRIOR, active_joints, theta_to_joint_limits
from limb_baseline.planner import PlannerSettings, run_episode
from limb_baseline.tasks import DEFAULT_TASK_DISTRIBUTION, Task
from limb_baseline.variant import DEFAULT_VARIANT, VARIANTS
from limb_baseline.variant import current as current_variant
from limb_baseline.variant import select as select_variant
from limb_baseline.warmstart import (
    BASE_FEATURES,
    control_basis,
    horizon_window,
    make_regressor,
    state_features,
)
from limb_baseline.worker import isolate_worker, task_sampler

CONDITIONS = ("plain", "warm", "learned")
SEED_MODES = ("plan", "step")
MODELS = ("mlp", "mlp_light", "ridge", "gradient_boosting")


# ------------------------------------------------------------------ seeded MPC


@dataclass
class SeedConfig:
    """How the seed is used: a whole policy rollout, or batch 0 of every MPC step."""

    mode: str = "plan"
    # Plan mode: give up the rollout after this many control steps without improvement,
    # or this many in total.
    patience: int = 60
    max_steps: int = 300
    # Rollouts spent on the seed and its noisy copies before the full batch is tried.
    rollouts: int = 1
    # Accept the seed when its cost is within this fraction of the best of the MPC's
    # goal-directed candidates, a bar the seed cannot influence; a worse seed sends the
    # step to the full batch.
    tolerance: float = 0.0
    # Step mode: score seed then draws until one beats the feedback candidates; a miss
    # reuses the scored rollouts in the full batch. False: score all, rerun batch on a miss.
    reuse: bool = True


@dataclass
class SeedStats:
    """Per-episode counts of what the seeded MPC did."""

    seeded_steps: int = 0
    full_steps: int = 0
    seed_rejected: int = 0
    policy_tried: int = 0
    policy_reached: int = 0
    policy_steps: int = 0
    policy_seconds: float = 0.0
    # CPU seconds (`time.process_time`, unaffected by load) per phase: model, seed
    # scoring (bar included), full batch. `scored_rollouts` counts all seeding rollouts.
    model_seconds: float = 0.0
    score_seconds: float = 0.0
    full_seconds: float = 0.0
    scored_rollouts: int = 0
    # Full-batch rollouts beyond those reused from seeding.
    full_rollouts: int = 0
    # Physics steps simulated; rollouts that break a limit stop early.
    scored_physics: int = 0
    full_physics: int = 0
    policy_failure: str = ""
    policy_final_error: float = float("nan")
    # Fraction of the remaining error each committed seed removed over its horizon.
    seed_progress: list = field(default_factory=list)
    # The same for the seed on steps that fell back to the full batch.
    rejected_progress: list = field(default_factory=list)
    # Per step: the best model candidate's cost over the bar (inf when all were rejected).
    seed_margin: list = field(default_factory=list)

    def reset(self) -> None:
        self.seeded_steps = self.full_steps = self.seed_rejected = 0
        self.policy_tried = self.policy_reached = self.policy_steps = 0
        self.policy_seconds = 0.0
        self.model_seconds = self.score_seconds = self.full_seconds = 0.0
        self.scored_rollouts = self.full_rollouts = 0
        self.scored_physics = self.full_physics = 0
        self.policy_failure, self.policy_final_error = "", float("nan")
        self.seed_progress, self.rejected_progress, self.seed_margin = [], [], []


SEED = SeedConfig()
STATS = SeedStats()
# The seed the MPC's noise draws from on the next construction, when set.
MPC_NOISE_SEED: int | None = None
_STOCK_MPC = planner_stream.PredictiveSamplingMPC
_STOCK_ADVANCE = planner_stream.advance
# Physics steps `advance` has taken in this process.
_ADVANCES = 0


def _counting_advance(*args, **kwargs):
    global _ADVANCES  # pylint: disable=global-statement
    _ADVANCES += 1
    return _STOCK_ADVANCE(*args, **kwargs)


class _SeededMPC(_STOCK_MPC):  # type: ignore[misc,valid-type]
    """The planner's MPC with a batch-0 seed from `ctx.warm_start`, and its own noise seed.

    `LimbStreamContext.motion_seed` seeds grasp, base, IK and MPC noise alike; repeating
    one task needs the first three fixed while the MPC still searches afresh.
    """

    def __init__(self, ctx, goal, start=None) -> None:
        # The stock constructor seeds its nominal from the hook and expects control
        # points; a horizon of torques is projected onto them here instead.
        hook, ctx.warm_start = ctx.warm_start, None
        try:
            super().__init__(ctx, goal, start)
        finally:
            ctx.warm_start = hook
        if MPC_NOISE_SEED is not None:
            self._rng = np.random.default_rng(MPC_NOISE_SEED)
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
        hook = self._ctx.warm_start
        if hook is None or SEED.rollouts <= 0:
            STATS.full_steps += 1
            return self._full_step()
        ctx, sim, cfg = self._ctx, self._sim, self._cfg
        error = limb_error(ctx, sim.limb.get_joint_positions(), self._goal)
        start = capture_state(sim)
        started = time.process_time()
        seed = self._clip(np.asarray(hook(start, self._goal, self._step_index), float))
        scale = cfg.noise_scale * float(
            np.clip(error / cfg.noise_reference_error, cfg.min_noise_ratio, 1.0)
        )
        sampler = getattr(hook, "samples", None)
        if sampler is not None and SEED.rollouts > 1:
            # A generative model's own draws replace the seed's noisy copies.
            drawn = sampler(start, self._goal, self._step_index, SEED.rollouts - 1)
            batch = [seed] + [self._clip(np.asarray(p, float)) for p in drawn]
        else:
            batch = [seed] + [
                self._clip(seed + self._rng.normal(scale=scale, size=seed.shape))
                for _ in range(SEED.rollouts - 1)
            ]
        scoring = time.process_time()
        STATS.model_seconds += scoring - started
        scored_before, physics_before = self._scored, self._physics
        # A seed given per control step is rolled out verbatim; control points are
        # interpolated as the MPC's own candidates are.
        exact = seed.shape[0] == cfg.horizon

        def expand(points):
            return points if exact else self._expand(points)

        def as_nominal(points):
            if not exact:
                return points
            solution, *_ = np.linalg.lstsq(control_basis(cfg), points, rcond=None)
            return self._clip(solution)
        reach = max(ctx.goal_atol - cfg.replay_slack, ctx.goal_atol / 2)
        if SEED.reuse:
            return self._reusing_step(start, error, scale, batch, exact, reach, scoring)

        # The bar: the MPC's own goal-directed candidates, proportional pushes toward
        # the goal at each gain with a noisy twin. Unlike the shifted nominal they owe
        # nothing to earlier seeds, so a mediocre policy cannot keep vouching for itself.
        bar = float("inf")
        for points in self._directed_candidates(scale):
            directed_rejected, directed_cost = self._score(start, self._expand(points))
            if not directed_rejected:
                bar = min(bar, directed_cost * (1.0 + SEED.tolerance))

        def passes(rejected_flag: int, cost: float, terminal: float) -> bool:
            return not rejected_flag and (terminal < reach or cost <= bar)

        # The seed itself first: following it exactly keeps the state on the path it
        # came from. Its noisy copies are only a backup when it falls short.
        chosen = None
        rejected, cost = self._score(start, expand(seed))
        seed_terminal = self._rollout_error()
        seed_cost = cost
        if sampler is not None and SEED.rollouts > 1:
            # Draws are alternatives, not backups: score them all and keep the cheapest.
            best = None if rejected else (cost, seed, seed_terminal)
            for points in batch[1:]:
                rejected_draw, cost = self._score(start, expand(points))
                terminal = self._rollout_error()
                if not rejected_draw and (best is None or cost < best[0]):
                    best = (cost, points, terminal)
            if best is not None and passes(0, best[0], best[2]):
                chosen = (best[1], best[2])
            STATS.seed_margin.append(best[0] / bar if best is not None else float("inf"))
        elif passes(rejected, cost, seed_terminal):
            STATS.seed_margin.append(cost / bar if not rejected else float("inf"))
            chosen = (seed, seed_terminal)
        else:
            best = None
            for points in batch[1:]:
                rejected_copy, cost = self._score(start, expand(points))
                terminal = self._rollout_error()
                if not rejected_copy and (best is None or cost < best[0]):
                    best = (cost, points, terminal)
            if best is not None and passes(0, best[0], best[2]):
                chosen = (best[1], best[2])
            options = [c for c in ([] if rejected else [seed_cost]) + ([best[0]] if best else [])]
            STATS.seed_margin.append(min(options) / bar if options else float("inf"))
        restore_state(sim, start, regrasp=False)
        STATS.score_seconds += time.process_time() - scoring
        STATS.scored_rollouts += self._scored - scored_before
        STATS.scored_physics += self._physics - physics_before
        if os.environ.get("MEMO_DEBUG"):
            print(
                f"seed step={self._step_index} error={error:.4f} "
                f"terminal={seed_terminal:.4f} rejected={rejected} cost={seed_cost:.1f} "
                f"bar={bar:.1f} accepted={chosen is not None} nearest={getattr(hook, 'last', None)}",
                flush=True,
            )
        if chosen is not None:
            STATS.seeded_steps += 1
            STATS.seed_progress.append((error - chosen[1]) / max(error, 1e-9))
            self._nominal = as_nominal(chosen[0])
            self._step_index += self._commit
            return [list(t) for t in expand(chosen[0])[: self._commit]]
        STATS.rejected_progress.append((error - seed_terminal) / max(error, 1e-9))
        STATS.seed_rejected += 1
        STATS.full_steps += 1
        return self._full_step()

    def _reusing_step(self, start, error, scale, batch, exact, reach, scoring) -> list:
        """Score seed and draws lazily; a miss reuses them in the full batch."""
        cfg, physics_before = self._cfg, self._physics

        def as_nominal(points):
            if not exact:
                return points
            solution, *_ = np.linalg.lstsq(control_basis(cfg), points, rcond=None)
            return self._clip(solution)

        # (nominal control points, torques, (rejected, cost)) per rollout.
        scored = []

        def score(nominal, torques):
            scored.append((nominal, torques, self._score(start, torques)))
            return scored[-1][2]

        # The bar is model-independent (the full batch scores these anyway), so a poor
        # policy cannot vouch for itself.
        bar_candidates = self._feedback_candidates(start) or self._directed_candidates(scale)
        bar = float("inf")
        for points in bar_candidates:
            rejected, cost = score(points, self._expand(points))
            if not rejected:
                bar = min(bar, cost * (1.0 + SEED.tolerance))
        chosen = None
        for points in batch:
            torques = points if exact else self._expand(points)
            rejected, cost = score(as_nominal(points), torques)
            terminal = self._rollout_error()
            if not rejected and (terminal < reach or cost <= bar):
                chosen = (scored[-1], terminal)
                break
        model_costs = [c for _, _, (r, c) in scored[len(bar_candidates):] if not r]
        STATS.seed_margin.append(
            (chosen[0][2][1] if chosen else min(model_costs, default=float("inf"))) / bar
        )
        STATS.score_seconds += time.process_time() - scoring
        STATS.scored_rollouts += len(scored)
        STATS.scored_physics += self._physics - physics_before
        if chosen is not None:
            (nominal, torques, _), terminal = chosen
            STATS.seeded_steps += 1
            STATS.seed_progress.append((error - terminal) / max(error, 1e-9))
            self._nominal = nominal
            self._step_index += self._commit
            return [list(t) for t in torques[: self._commit]]
        STATS.seed_rejected += 1
        STATS.full_steps += 1
        started = time.process_time()
        try:
            return self._finish_batch(start, error, scored)
        finally:
            STATS.full_seconds += time.process_time() - started

    def _finish_batch(self, start, error, scored) -> list:
        """The stock step, minus what the seeding phase already scored."""
        cfg, hook = self._cfg, self._ctx.warm_start
        # Sample only the rest of the batch, minus feedback candidates and seed.
        self._cfg = dataclasses.replace(
            cfg, feedback_gains=(), num_rollouts=max(cfg.num_rollouts - len(scored), 0)
        )
        self._ctx.warm_start = None
        try:
            fresh = self._sample_candidates(error, start)
        finally:
            self._cfg, self._ctx.warm_start = cfg, hook
        physics_before = self._physics
        for points in fresh:
            torques = self._expand(points)
            scored.append((points, torques, self._score(start, torques)))
        STATS.full_rollouts += len(fresh)
        STATS.full_physics += self._physics - physics_before
        nominal, torques, (rejected, _) = min(scored, key=lambda entry: entry[2])
        if rejected:
            raise NoFeasibleMotion("all MPC candidates violate constraints")
        self._nominal = nominal
        self._step_index += self._commit
        return [list(t) for t in torques[: self._commit]]

    def _rollout_error(self) -> float:
        """Limb-to-goal error at the end of the last scored rollout."""
        return limb_error(
            self._rollout_ctx, self._rollout_sim.limb.get_joint_positions(), self._goal
        )

    _scored = 0
    _physics = 0

    def _score(self, *args, **kwargs):
        self._scored += 1
        before = _ADVANCES
        try:
            return super()._score(*args, **kwargs)
        finally:
            self._physics += _ADVANCES - before

    def _full_step(self) -> list:
        started, scored, physics = time.process_time(), self._scored, self._physics
        try:
            return super().step()
        finally:
            STATS.full_seconds += time.process_time() - started
            STATS.full_rollouts += self._scored - scored
            STATS.full_physics += self._physics - physics


def install_mpc() -> None:
    """Have the planner build the seeded MPC from here on."""
    planner_stream.PredictiveSamplingMPC = _SeededMPC
    planner_stream.advance = _counting_advance


# --------------------------------------------------------------- plan mode


def policy_rollout(ctx: LimbStreamContext, s1: CoupledState, q2, hook):
    """Follow `hook` closed-loop from `s1`; the trajectory and end state if it reaches `q2`.

    The same per-step checks `plan_limb_motion` applies, and the same stopping
    tolerance, so a returned plan is one the planner would have certified. Nothing is
    searched: a rollout that violates a limit, stalls or runs long returns None.
    """
    sim, cfg = ctx.sim, ctx.mpc
    if not ctx.in_believed_limits(list(q2.positions)):
        return None
    goal = np.asarray(q2.positions, dtype=np.float64)
    lower, upper = ctx.robot_torque_limits
    threshold = max(ctx.goal_atol - cfg.replay_slack, ctx.goal_atol / 2)
    robot_torques, human_torques, robot_induced, commanded = [], [], [], []
    with saved_sim_state(sim):
        restore_state(sim, s1)
        limb_path = [list(sim.limb.get_joint_positions())]
        error = limb_error(ctx, sim.limb.get_joint_positions(), goal)
        best, since, reached = error, 0, error < threshold
        for step in range(SEED.max_steps):
            if reached or since >= SEED.patience:
                break
            state = capture_state(sim)
            block = np.asarray(hook(state, goal, step), dtype=np.float64)
            torque = np.clip(block[0], lower, upper)
            hook.previous = torque
            for _ in range(cfg.action_repeat):
                command = commanded_robot_torque(sim, torque)
                human = advance(sim, command)
                robot_torques.append(list(torque))
                commanded.append(list(command))
                human_torques.append(list(human.total))
                robot_induced.append(list(human.robot))
                limb_path.append(list(sim.limb.get_joint_positions()))
                failure = (
                    "robot torque limit" if exceeds_robot_torque_limits(ctx, command)
                    else "human torque limit" if exceeds_human_torque_limits(ctx, human)
                    else "limb out of limits" if limb_out_of_limits(ctx)
                    else "collision" if ctx.resting_penetration and human_in_collision(ctx)
                    else ""
                )
                error = limb_error(ctx, sim.limb.get_joint_positions(), goal)
                if failure:
                    STATS.policy_failure = f"{failure} at step {step}"
                    STATS.policy_final_error = error
                    STATS.policy_steps = step
                    return None
                if error < threshold:
                    reached = True
                    break
            since = 0 if best - error > cfg.divergence_tolerance else since + 1
            best = min(best, error)
        STATS.policy_steps = len(robot_torques) // cfg.action_repeat
        STATS.policy_final_error = error
        if not reached or not robot_torques:
            STATS.policy_failure = (
                f"stalled at {error:.3f} after {STATS.policy_steps} steps"
                if since >= SEED.patience else f"ran out of steps at {error:.3f}"
            )
            return None
        s2 = capture_state(sim)
        if limb_error(ctx, s2.limb_positions, goal) >= ctx.goal_atol:
            STATS.policy_failure = "outside tolerance at the end"
            return None
    return (
        TorqueTrajectory(
            robot_torques, human_torques, robot_induced, limb_path, commanded,
            [0.0] * len(robot_torques),
        ),
        s2,
    )


# ------------------------------------------------------------------------ memo


# The notebook keys its cache on the goal rounded to this resolution, in radians.
KEY_RESOLUTION = 0.1
# Plans kept per key, best first.
KEEP_PER_KEY = 4


def goal_key(positions) -> tuple[int, ...]:
    """The cache key of a goal: each joint rounded to `KEY_RESOLUTION`."""
    return tuple(int(v) for v in np.round(np.asarray(positions) / KEY_RESOLUTION))


@dataclass
class LimbMotionMemo:
    """Solved `plan-limb-motion` trajectories by goal, served as the MPC's seed.

    The belief would be part of the key too, but stages 1-2 are skipped here, so it is
    the same on every call and drops out.
    """

    plans: dict = field(default_factory=dict)
    calls: int = 0
    hits: int = 0
    misses: int = 0

    def reset(self) -> None:
        self.calls = self.hits = self.misses = 0

    @property
    def size(self) -> int:
        return sum(len(entries) for entries in self.plans.values())

    def store(self, key: tuple, trajectory: TorqueTrajectory) -> None:
        """Keep `trajectory` under `key`; shorter plans rank first."""
        entries = self.plans.setdefault(key, [])
        entries.append((len(trajectory.robot_torques), trajectory))
        entries.sort(key=lambda entry: entry[0])
        del entries[KEEP_PER_KEY:]

    def lookup(self, key: tuple) -> TorqueTrajectory | None:
        entries = self.plans.get(key)
        return entries[0][1] if entries else None

    @staticmethod
    def as_hook(trajectory: TorqueTrajectory, mpc: MPCConfig):
        """`ctx.warm_start`: the stored torques, verbatim, from the stored state nearest
        `state`, one row per control step of the horizon.

        Indexed by state rather than by step, so a search step that leaves the stored
        path picks the seed back up wherever the path passes closest.
        """
        torques = np.asarray(trajectory.robot_torques, dtype=np.float64)
        path = [list(q) for q in trajectory.limb_path[: len(torques)]]

        def hook(state: CoupledState, goal: np.ndarray, step: float = 0.0):
            del goal
            if path:
                here = list(state.limb_positions)
                first = min(
                    range(len(path)),
                    key=lambda i: joint_position_distance(LIMB_JOINT_INFOS, path[i], here),
                )
                hook.last = (first, round(joint_position_distance(LIMB_JOINT_INFOS, path[first], here), 5))
            else:
                first = min(int(step) * mpc.action_repeat, len(torques) - 1)
            return horizon_window(torques[first:], mpc)

        hook.last = None
        return hook



@dataclass
class SeededStream:
    """Drop-in for `plan_limb_motion` that plans from a cache or a model first.

    With `memo`, the seed is the cached plan for this goal, if any. Without it, the seed
    is whatever `run_episode` put in `ctx.warm_start`, the learned model's hook.
    """

    memo: LimbMotionMemo | None = None
    # Appended to every goal key: dream_pipeline.py puts the belief here, so a plan solved
    # under one belief is not served under another. Empty when the belief is fixed.
    key_suffix: tuple = ()

    def stream(
        self, ctx: LimbStreamContext, s1: CoupledState, q2
    ) -> Iterator[tuple[TorqueTrajectory, CoupledState]]:
        key = goal_key(q2.positions) + tuple(self.key_suffix)
        if self.memo is not None:
            self.memo.calls += 1
            stored = self.memo.lookup(key)
            if stored is not None:
                self.memo.hits += 1
                hook = LimbMotionMemo.as_hook(stored, ctx.mpc)
                print(f"plan-limb-motion (memo): hit, {len(stored.robot_torques)} steps stored.")
            else:
                self.memo.misses += 1
                hook = None
                print("plan-limb-motion (memo): miss.")
        else:
            hook = ctx.warm_start

        if hook is not None and SEED.mode == "plan":
            STATS.policy_tried += 1
            started = time.perf_counter()
            result = policy_rollout(ctx, s1, q2, hook)
            STATS.policy_seconds += time.perf_counter() - started
            if result is not None:
                STATS.policy_reached += 1
                print(f"plan-limb-motion (policy): reached the goal in "
                      f"{STATS.policy_steps} control steps, no search.")
                if self.memo is not None:
                    self.memo.store(key, result[0])
                yield result
                return
            print("plan-limb-motion (policy): rollout failed; planning from scratch.")
            hook = None
        ctx.warm_start = hook
        for trajectory, s2 in plan_limb_motion(ctx, s1, q2):
            if self.memo is not None:
                self.memo.store(key, trajectory)
            yield trajectory, s2


def install(stream: SeededStream) -> None:
    """Route every `plan-limb-motion` call this process makes through `stream`."""
    planner_run.plan_limb_motion = stream.stream


def uninstall() -> None:
    """Put the stock stream back."""
    planner_run.plan_limb_motion = plan_limb_motion


# --------------------------------------------------------------- learned seed


def _variance(labels: np.ndarray) -> np.ndarray:
    """Per-output label variance, floored so a constant output does not divide by zero."""
    return np.maximum(labels.var(0), 1e-12) if len(labels) > 1 else np.ones(labels.shape[1])


@dataclass
class OnlineWarmStart:
    """Rows from every solved episode so far, and the model refit on them.

    The label is the next horizon's torques verbatim, `(horizon, NUM_ROBOT_JOINTS)`
    flattened, so a model that fits its rows closely reproduces the planner's motion.
    """

    name: str = "mlp_light"
    features: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    estimator: object | None = None
    mpc: MPCConfig = field(default_factory=MPCConfig)
    # When set, each refit appends a JSON line: the error on the rows just added, before
    # fitting them, and the fit's own errors; `context` (episode, dream, ...) is copied in.
    log_path: str | None = None
    context: dict = field(default_factory=dict)

    @property
    def model(self):
        return self if self.estimator is not None else None

    def add(self, segments) -> None:
        new_x = [segment.features for segment in segments]
        new_y = [np.asarray(segment.torques).ravel() for segment in segments]
        self._pending = {"new_rows": len(new_x)}
        if getattr(self, "log_path", None) and new_x and self.estimator is not None:
            y, pred = np.asarray(new_y), self.estimator.predict(np.asarray(new_x))
            var = _variance(np.asarray(self.labels))
            first = NUM_ROBOT_JOINTS
            self._pending.update(
                new_nmse=float((((pred - y) ** 2) / var).mean()),
                new_nmse_first=float((((pred - y)[:, :first] ** 2) / var[:first]).mean()),
                new_mse=float(((pred - y) ** 2).mean()),
            )
        self.features.extend(new_x)
        self.labels.extend(new_y)

    def refit(self, mpc: MPCConfig) -> None:
        if not self.features:
            return
        self.mpc = mpc
        estimator = make_regressor(self.name)
        started = time.perf_counter()
        x, y = np.asarray(self.features), np.asarray(self.labels)
        estimator.fit(x, y)
        self.estimator = estimator
        if getattr(self, "log_path", None):
            self._log(x, y, estimator, time.perf_counter() - started)

    def _log(self, x, y, estimator, seconds: float) -> None:
        from sklearn.pipeline import Pipeline

        inner = estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator
        record = {
            **getattr(self, "context", {}),
            **getattr(self, "_pending", {}),
            "model": self.name,
            "rows": len(x),
            "segments": int((x[:, BASE_FEATURES - 1] == 0).sum()),
            "train_nmse": float((((estimator.predict(x) - y) ** 2) / _variance(y)).mean()),
            "fit_seconds": round(seconds, 3),
        }
        if hasattr(inner, "history"):
            record.update(val_nmse=inner.best_val, best_step=inner.best_step,
                          val_recent_nmse=getattr(inner, "best_val_recent", float("nan")),
                          curve=[list(h) for h in inner.history])
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self._pending = {}

    def as_hook(self, belief_features: np.ndarray):
        """The callable `LimbStreamContext.warm_start` expects."""
        assert self.estimator is not None
        horizon = self.mpc.horizon

        def hook(state, goal: np.ndarray, step: float = 0.0):
            if step == 0:
                hook.previous = None
            features = state_features(state, goal, belief_features, step, hook.previous)
            flat = self.estimator.predict(np.asarray(features).reshape(1, -1))[0]
            return flat.reshape(horizon, -1)

        # The torque applied at the last control step, set by whoever applies it.
        hook.previous = None
        if hasattr(self.estimator, "sample"):
            def samples(state, goal: np.ndarray, step: float, k: int):
                features = state_features(state, goal, belief_features, step, hook.previous)
                return self.estimator.sample(features.reshape(1, -1), k)[0].reshape(k, horizon, -1)

            hook.samples = samples
        return hook


# ----------------------------------------------------------------------- tasks


def nominal_task(theta: np.ndarray, seed: int) -> Task:
    """The one task both conditions repeat: reachable for this person, fixed by `seed`."""
    return task_sampler().sample(np.random.default_rng(seed), theta, theta)


def perturbed_task(
    nominal: Task,
    theta: np.ndarray,
    rng: np.random.Generator,
    variation: float,
    vary_init: bool,
    max_draws: int = 200,
) -> Task:
    """`nominal` with its goal (and optionally its start) nudged by up to `variation` rad.

    Rejected if the separation leaves the task distribution's band or the nominal base
    pose can no longer carry it, so every variant is as solvable as the original.
    """
    sampler = task_sampler()
    lower, upper = theta_to_joint_limits(theta)
    distribution = DEFAULT_TASK_DISTRIBUTION
    for _ in range(max_draws):
        goal = np.asarray(nominal.limb_goal, dtype=np.float64)
        init = np.asarray(nominal.limb_init, dtype=np.float64)
        for j in active_joints():
            goal[j] = np.clip(goal[j] + rng.uniform(-variation, variation), lower[j], upper[j])
            if vary_init:
                init[j] = np.clip(
                    init[j] + rng.uniform(-variation, variation), lower[j], upper[j]
                )
        separation = joint_position_distance(LIMB_JOINT_INFOS, list(init), list(goal))
        if not distribution.min_separation <= separation <= distribution.max_separation:
            continue
        if nominal.base_pose is not None and not sampler._reachable_from(  # noqa: SLF001
            nominal.base_pose, init, goal
        ):
            continue
        return dataclasses.replace(nominal, limb_init=tuple(init), limb_goal=tuple(goal))
    raise RuntimeError(
        f"no reachable perturbation of the nominal task within {variation} rad "
        f"in {max_draws} draws"
    )


def random_task(
    nominal: Task, theta: np.ndarray, rng: np.random.Generator, max_draws: int = 2000
) -> Task:
    """`nominal` with its goal drawn uniformly over the person's true range on each active
    joint, under the same two rejections as `perturbed_task`: the separation from the
    start, and reach from the nominal base pose. Start and base pose stay the nominal's."""
    sampler = task_sampler()
    lower, upper = theta_to_joint_limits(theta)
    distribution = DEFAULT_TASK_DISTRIBUTION
    init = np.asarray(nominal.limb_init, dtype=np.float64)
    for _ in range(max_draws):
        goal = np.asarray(nominal.limb_goal, dtype=np.float64)
        for j in active_joints():
            goal[j] = rng.uniform(lower[j], upper[j])
        separation = joint_position_distance(LIMB_JOINT_INFOS, list(init), list(goal))
        if not distribution.min_separation <= separation <= distribution.max_separation:
            continue
        if nominal.base_pose is not None and not sampler._reachable_from(  # noqa: SLF001
            nominal.base_pose, init, goal
        ):
            continue
        return dataclasses.replace(nominal, limb_goal=tuple(goal))
    raise RuntimeError(f"no reachable random goal in {max_draws} draws")


def task_shift(task: Task, nominal: Task) -> float:
    """How far `task`'s goal sits from the nominal goal, in the environment's metric."""
    return joint_position_distance(
        LIMB_JOINT_INFOS, list(task.limb_goal), list(nominal.limb_goal)
    )


# ------------------------------------------------------------------ experiment


def run_sequence(
    condition: str,
    num_tasks: int,
    variation: float,
    vary_init: bool,
    person: int,
    task_seed: int,
    model_name: str,
    settings: PlannerSettings,
    out: Path,
    tag: str,
) -> list[dict]:
    """Plan `num_tasks` episodes in order in this one process, appending a row each.

    The memo and the model live across iterations, so the sequence cannot be spread
    over workers the way `evaluate` spreads independent episodes.
    """
    global MPC_NOISE_SEED  # pylint: disable=global-statement
    isolate_worker()
    theta = DEFAULT_PRIOR.sample(np.random.default_rng(person))
    nominal = nominal_task(theta, task_seed)
    memo = LimbMotionMemo()
    learner = OnlineWarmStart(model_name)
    if condition == "warm":
        install(SeededStream(memo))
    elif condition == "learned":
        install(SeededStream(None))
    else:
        uninstall()
    install_mpc()

    records = []
    for iteration in range(num_tasks):
        rng = np.random.default_rng(task_seed * 1000 + iteration)
        task = (
            nominal
            if variation <= 0.0
            else perturbed_task(nominal, theta, rng, variation, vary_init)
        )
        memo.reset()
        STATS.reset()
        MPC_NOISE_SEED = task_seed * 1000 + iteration
        record = {
            "tag": tag,
            "condition": condition,
            "model": model_name if condition == "learned" else "",
            "seed_mode": SEED.mode,
            "seed_rollouts": SEED.rollouts,
            "seed_tolerance": SEED.tolerance,
            "variation": variation,
            "vary_init": vary_init,
            "variant": current_variant().name,
            "person": person,
            "task_seed": task_seed,
            "iteration": iteration,
            "mpc_seed": MPC_NOISE_SEED,
            "separation": round(task.separation, 4),
            "goal_shift": round(task_shift(task, nominal), 4),
            "num_train_rows": len(learner.features),
        }
        try:
            # One geometry seed for every iteration, so grasp, base pose and IK repeat;
            # only the MPC noise (above) is fresh, so plain shows its own spread.
            result = run_episode(
                task,
                theta,
                belief=None,
                believed_theta=theta,
                warm_start=learner.model if condition == "learned" else None,
                seed=task_seed,
                settings=settings,
            )
            record.update(
                {
                    "solved": result.solved,
                    "goal_reached": result.goal_reached,
                    "plan_time": round(result.plan_time, 2),
                    "execute_time": round(result.execute_time, 2),
                    "stream_calls": memo.calls,
                    "memo_hits": memo.hits,
                    "memo_misses": memo.misses,
                    "memo_size": memo.size,
                    "policy_tried": STATS.policy_tried,
                    "policy_reached": STATS.policy_reached,
                    "policy_steps": STATS.policy_steps,
                    "policy_seconds": round(STATS.policy_seconds, 2),
                    "policy_failure": STATS.policy_failure,
                    "policy_final_error": round(STATS.policy_final_error, 4),
                    "seeded_steps": STATS.seeded_steps,
                    "full_steps": STATS.full_steps,
                    "seed_rejected": STATS.seed_rejected,
                    "seed_progress_mean": round(float(np.mean(STATS.seed_progress)), 4)
                    if STATS.seed_progress else float("nan"),
                    "rejected_progress_mean": round(float(np.mean(STATS.rejected_progress)), 4)
                    if STATS.rejected_progress else float("nan"),
                    "num_control_steps": result.num_control_steps,
                    "rom_violation_steps": result.rom_violation_steps,
                    "failure_reason": result.failure_reason,
                    "fit_time": 0.0,
                }
            )
            if condition == "learned" and result.goal_reached:
                learner.add(result.segments)
                started = time.perf_counter()
                learner.refit(settings.mpc)
                record["fit_time"] = round(time.perf_counter() - started, 2)
        except Exception as exc:  # pylint: disable=broad-except
            record.update(
                {
                    "solved": False,
                    "goal_reached": False,
                    "plan_time": float("nan"),
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
            traceback.print_exc()
        append_row(out, record)
        records.append(record)
        print(
            f"[{tag}/{condition} {iteration + 1}/{num_tasks}] "
            f"plan={record.get('plan_time')}s reached={record.get('goal_reached')} "
            f"policy {record.get('policy_reached', 0)}/{record.get('policy_tried', 0)} "
            f"seeded/full steps={record.get('seeded_steps', 0)}/{record.get('full_steps', 0)} "
            f"memo hits={record.get('memo_hits', 0)} rows={len(learner.features)} "
            f"{record.get('failure_reason', '')}",
            flush=True,
        )
    uninstall()
    return records


def main() -> None:
    """CLI entry point: one condition, one task sequence, one CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument(
        "--variation", type=float, default=0.0,
        help="Radians each active goal joint may move from the nominal task; 0 repeats "
             "one identical task.",
    )
    parser.add_argument("--vary-init", action="store_true",
                        help="Also nudge the limb's start, not only its goal.")
    parser.add_argument("--model", type=str, default="mlp_light",
                        help="Regressor the learned condition refits each iteration: one "
                             f"of {', '.join(MODELS)}, or a spec such as mlp:128x128:a0.001 "
                             "(see warmstart.make_regressor).")
    parser.add_argument("--seed-mode", choices=SEED_MODES, default=SeedConfig.mode,
                        help="plan: roll the seed out whole, search only if it fails; "
                             "step: seed each MPC step and fall back per step.")
    parser.add_argument("--seed-rollouts", type=int, default=SeedConfig.rollouts,
                        help="Rollouts spent on the seed and its noisy copies; 0 disables.")
    parser.add_argument("--seed-tolerance", type=float, default=SeedConfig.tolerance,
                        help="How much worse than the MPC's best goal-directed candidate a "
                             "seed may score.")
    parser.add_argument("--variant", choices=VARIANTS, default=DEFAULT_VARIANT)
    parser.add_argument("--person", type=int, default=22,
                        help="Seed of the one person whose true range of motion is used.")
    parser.add_argument("--task-seed", type=int, default=0)
    parser.add_argument("--max-time", type=float, default=600.0)
    parser.add_argument("--num-rollouts", type=int, default=MPCConfig.num_rollouts)
    parser.add_argument("--max-control-steps", type=int, default=MPCConfig.max_control_steps)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, "1")

    make_regressor(args.model)  # fail on a bad spec before any planning
    select_variant(args.variant)
    SEED.mode = args.seed_mode
    SEED.rollouts, SEED.tolerance = args.seed_rollouts, args.seed_tolerance
    mpc = dataclasses.replace(
        MPCConfig(),
        num_rollouts=args.num_rollouts,
        max_control_steps=args.max_control_steps,
    )
    settings = PlannerSettings(max_time=args.max_time, mpc=mpc)
    out = args.out.resolve()
    tag = args.tag or ("same_task" if args.variation <= 0.0 else f"varied_{args.variation:g}")
    records = run_sequence(
        args.condition, args.num_tasks, args.variation, args.vary_init, args.person,
        args.task_seed, args.model, settings, out, tag,
    )
    times = [r["plan_time"] for r in records if r.get("solved")]
    print(
        f"\n{args.condition}: {sum(bool(r.get('goal_reached')) for r in records)}/"
        f"{len(records)} reached the goal; median plan time of solved episodes "
        f"{np.median(times) if times else float('nan'):.1f} s\nWritten to {out}"
    )


if __name__ == "__main__":
    main()
