"""The notebook's dream-then-act loop on the arm, with the person's range of motion unknown.

One episode: (1) information gathering, (2) human model learning, (3) `n_dreams`
dreams, (4) one planned and executed task. Stages 1-2 are commented out; the belief
starts at the population prior and learns only from execution, as in the notebook.
Environment, planner, belief and task sampler are personalize.py's; memory is memo.py's.

A dream: draw a hypothetical person from the belief and a goal from the task set, plan
under the current belief, imagine executing on that person, update a copy of the belief
from what that revealed, and memorize plans under that imagined next belief (`cache`:
stored under (goal, belief); `learned`: rows added and the model refit). A real episode
tries the memory first and searches only if the rollout fails. Only real planning is
charged; dream time is logged apart.

A goal beyond the belief is not refused: as the notebook's planner stops at the nearest
point it believes reachable, the plan is retargeted to just inside the believed edge.
Quality is the final distance to the original goal, as the notebook's error.

    python -m limb_baseline.dream_pipeline --condition cache --n-dreams 4 \\
        --num-episodes 20 --out data/dream/cache_4.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import pickle
import time
import traceback
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from kinder_pddlstream_planning.limbrepositioning3d.stream import MPCConfig

from limb_baseline import memo as memo_module
from limb_baseline.belief import Belief
from limb_baseline.dataset import append_row
from kinder.envs.dynamic3d.limb_utils import LIMB_JOINT_INFOS, joint_position_distance

from limb_baseline.human import (
    DEFAULT_PRIOR,
    active_joints,
    active_signs,
    default_magnitudes,
    theta_to_bounds,
)
from limb_baseline.memo import (
    KEY_RESOLUTION,
    KEEP_PER_KEY,
    goal_key,
    MODELS,
    SEED,
    STATS,
    LimbMotionMemo,
    OnlineWarmStart,
    SeedStats,
    SeededStream,
    install,
    install_mpc,
    perturbed_task,
    random_task,
    task_shift,
    uninstall,
)
from limb_baseline.planner import (
    EpisodeResult,
    ParticleLimits,
    PlannerSettings,
    joint_limits,
    run_episode,
)
from limb_baseline.queries import DEFAULT_QUERY_MODEL
from limb_baseline.warmstart import make_regressor
from limb_baseline.tasks import Task
from limb_baseline.variant import VARIANTS
from limb_baseline.variant import current as current_variant
from limb_baseline.variant import select as select_variant
from limb_baseline.worker import isolate_worker, task_sampler

# Stages 1-2, switched off; see `run_sequence`.
# from limb_baseline.planner import QueryPolicy, gather_information

CONDITIONS = ("plain", "cache", "learned")
# The least a floor must rise, in scale units, to count as new information.
FLOOR_STEP = 1e-3
# How far inside the believed edge a retargeted goal sits, in radians. Small, so reaching
# it raises the floor past what the belief already allowed and the edge moves toward the
# goal each episode; at 0.2 the floor landed below the edge and the belief never grew.
RETARGET_MARGIN = 0.05
# The limb starts inside this fraction of the person's true range, as the notebook's arm
# starts at zero: a start on the edge leaves the planner no room to move without
# touching the believed limit it has just learned from that start.
START_FRACTION = 0.8


@dataclass(frozen=True)
class DreamConfig:
    """How much to dream, and the task distribution dreamed about."""

    n_dreams: int = 0
    # Notebook step 3: solve every goal under the imagined belief, not just the one.
    all_goals: bool = False
    # Every real episode gets a goal of its own, never repeated; dreams draw from a
    # separate pool of `dream_pool` goals from the same distribution, so no dream ever
    # solves a real episode's goal in advance.
    fresh_goals: bool = False
    # Grasp, base pose and IK are sampled afresh for every plan (real or dreamed), seeded
    # by the plan's own noise seed, instead of fixed by the run's task seed.
    vary_geometry: bool = False
    # Planner for every episode: PDDLStream, or the POMDP planner via POUCT or POMCP.
    planner: str = "pddlstream"
    dream_pool: int = 200
    # With `fresh_goals`: goals drawn uniformly over the person's true range instead of
    # within `variation` of the nominal goal.
    random_goals: bool = False
    num_goals: int = 6
    # Radians each goal may sit from the nominal goal; 0 repeats one task.
    variation: float = 0.3
    num_episodes: int = 20
    # Belief particles the planner checks against; 0 plans under the belief's mean.
    particles: int = 32
    # Fraction of particles a configuration must lie inside to count as believed inside:
    # 0.5 is about the typical person. At 0.9 the edge sat at the narrowest plausible
    # person and almost every goal was cut short.
    confidence: float = 0.5
    # Planner budget per imagined episode, in seconds, enforced exactly.
    dream_max_time: float = 600.0
    # Stage 1, unused while it is off.
    num_queries: int = 6


# --------------------------------------------------------------------- belief


def belief_key(belief: Belief, resolution: float = KEY_RESOLUTION) -> tuple[int, ...]:
    """The cache key of a belief: its summary rounded to `resolution`."""
    return tuple(int(v) for v in np.round(belief.key_features() / resolution))


def observe(
    belief: Belief, limb_path, person: np.ndarray, tau: float = DEFAULT_QUERY_MODEL.tau
) -> tuple[Belief, list[str]]:
    """Update the belief from one execution on `person` (real or hypothetical).

    Per active motion: crossed the limit -> the crossing angle is the limit (a query
    answer); stayed inside -> the farthest angle reached is a floor. Returns the belief
    and a note per motion changed.
    """
    if not limb_path:
        return belief, []
    path = np.asarray(limb_path, dtype=np.float64)
    bounds = theta_to_bounds(person)
    magnitudes = default_magnitudes()
    notes = []
    for i, (joint, sign) in enumerate(zip(active_joints(), active_signs())):
        reach = sign * path[:, joint]
        crossed = np.flatnonzero(reach > sign * bounds[i])
        if len(crossed):
            angle = float(reach[crossed[0]])
            belief = belief.updated(i, angle, tau, 0.0)
            notes.append(f"crossed:{i}@{angle / magnitudes[i]:.3f}")
            continue
        scale = float(reach.max()) / magnitudes[i]
        # Repeating a task reaches the same angle within numerical noise: not a revision.
        if scale > belief.lower[i] + FLOOR_STEP:
            belief = belief.raised(i, scale)
            notes.append(f"floor:{i}@{scale:.3f}")
    return belief, notes


@dataclass
class Believed:
    """The belief as the planner takes it: one theta, or particles."""

    belief: Belief
    limits: ParticleLimits | None = None

    @classmethod
    def build(cls, belief: Belief, config: DreamConfig, rng: np.random.Generator) -> "Believed":
        if config.particles <= 0:
            return cls(belief)
        particles = np.stack([belief.sample(rng) for _ in range(config.particles)])
        return cls(belief, ParticleLimits(particles, config.confidence))

    def with_belief(self, belief: Belief, config: DreamConfig, rng: np.random.Generator) -> "Believed":
        """Move to `belief`; particles are redrawn only when its key changes."""
        if belief is self.belief:
            return self
        if belief_key(belief) != belief_key(self.belief):
            return Believed.build(belief, config, rng)
        return dataclasses.replace(self, belief=belief)

    @property
    def theta(self) -> np.ndarray:
        """The one range goals are posed against and the planner's fallback box."""
        if self.limits is None:
            return self.belief.planning_mean
        return self.limits.quantile_theta()

    def retarget(self, task: Task) -> tuple[Task, bool]:
        """`task`, its goal moved just inside the believed range if it lies beyond.

        The notebook's planner ends at the nearest point it believes reachable; this is
        that point, clipped per active joint to the believed box less `RETARGET_MARGIN`.
        """
        goal = np.asarray(task.limb_goal, dtype=np.float64)
        inside = (self.limits(goal) if self.limits is not None
                  else self._in_box(goal, 0.0))
        if inside:
            return task, False
        lower, upper = (np.asarray(b) for b in joint_limits(self.theta))
        # Only the joints beyond the box move; a joint inside stays, even on its edge.
        beyond = [j for j in active_joints() if not lower[j] <= goal[j] <= upper[j]]
        margin = RETARGET_MARGIN
        for _ in range(10):
            moved = goal.copy()
            for j in beyond or active_joints():
                low = min(lower[j] + margin, upper[j])
                high = max(upper[j] - margin, lower[j])
                if not low <= goal[j] <= high:
                    moved[j] = np.clip(goal[j], low, high)
            # Particles vote on the whole configuration, the box per joint: step inward
            # until enough particles allow it too.
            if self.limits is None or self.limits(moved):
                break
            margin += RETARGET_MARGIN
        return dataclasses.replace(task, limb_goal=tuple(moved)), True

    def _in_box(self, goal: np.ndarray, margin: float) -> bool:
        lower, upper = (np.asarray(b) for b in joint_limits(self.theta))
        return bool(np.all((lower + margin <= goal) & (goal <= upper - margin)))

    def hypothesis(self, rng: np.random.Generator) -> np.ndarray:
        """A person the belief allows: one particle, or one draw."""
        if self.limits is None:
            return self.belief.sample(rng)
        return self.limits.particles[int(rng.integers(len(self.limits.particles)))]


# ------------------------------------------------------------------ task set


@dataclass
class TaskSet:
    """The goals episodes and dreams draw from, around one nominal task."""

    nominal: Task
    tasks: list[Task]

    def __len__(self) -> int:
        return len(self.tasks)


def make_task_set(
    theta: np.ndarray, rng: np.random.Generator, config: DreamConfig
) -> TaskSet:
    """`num_goals` goals within `variation` rad of one nominal task, inside the true range.

    As in the notebook, every goal is reachable by the real person, and the start is
    well inside it (`START_FRACTION`). Start and base pose are the nominal's, so stored
    plans can be rolled out from the same grasped state.
    """
    nominal = task_sampler().sample(rng, theta, np.asarray(theta) * START_FRACTION)
    if config.variation <= 0.0:
        return TaskSet(nominal, [nominal])
    tasks = [
        perturbed_task(nominal, theta, rng, config.variation, vary_init=False)
        for _ in range(config.num_goals)
    ]
    return TaskSet(nominal, tasks)


# --------------------------------------------------------------------- memory

# How far, per belief number (mean, spread and floor of each motion, in scale units), a
# stored plan's belief may be from the current one and still be served. Exact matches
# of the rounded belief almost never happened: dreamed and real floors differed by
# about 0.1-0.18. A wrong reuse costs one failed rollout before the usual search.
BELIEF_TOLERANCE = 0.15


@dataclass
class BeliefMemo(LimbMotionMemo):
    """The plan cache by goal, each plan filed with the belief it was made under.

    The notebook's suggested nearest-neighbour cache: a lookup serves the plan whose
    belief is closest to `current`, if within `tolerance` on every number, the shorter
    plan on a tie. The goal part stays near-exact (rounded to `KEY_RESOLUTION`) because
    a stored plan is replayed, and it ends at its own goal.
    """

    tolerance: float = BELIEF_TOLERANCE
    current: np.ndarray | None = None
    # How far the served plan's belief was, and the last plan filed, for the dream.
    last_distance: float = float("nan")
    last_stored: tuple | None = None

    def reset(self) -> None:
        super().reset()
        self.last_distance = float("nan")
        self.last_stored = None

    @property
    def size(self) -> int:
        return sum(len(entries) for entries in self.plans.values())

    def store(self, key: tuple, trajectory, belief: np.ndarray | None = None) -> None:
        """File `trajectory` under goal `key` and `belief` (default: the current one);
        at most `KEEP_PER_KEY` plans per goal and rounded belief, shortest first."""
        vector = np.asarray(self.current if belief is None else belief, dtype=np.float64)
        entries = self.plans.setdefault(key, [])
        entries.append((len(trajectory.robot_torques), trajectory, vector))
        group = tuple(np.round(vector / KEY_RESOLUTION).astype(int))
        same = [e for e in entries
                if tuple(np.round(e[2] / KEY_RESOLUTION).astype(int)) == group]
        same.sort(key=lambda e: e[0])
        drop = {id(e) for e in same[KEEP_PER_KEY:]}
        self.plans[key] = [e for e in entries if id(e) not in drop]
        self.last_stored = (key, trajectory)

    def lookup(self, key: tuple):
        entries = self.plans.get(key)
        if not entries or self.current is None:
            return None
        scored = [(float(np.max(np.abs(e[2] - self.current))), e[0], e[1]) for e in entries]
        scored = [s for s in scored if s[0] <= self.tolerance]
        if not scored:
            return None
        distance, _, trajectory = min(scored, key=lambda s: (s[0], s[1]))
        self.last_distance = distance
        return trajectory


@dataclass
class Memory:
    """What one condition carries across dreams and episodes."""

    condition: str
    stream: SeededStream
    memo: LimbMotionMemo | None
    learner: OnlineWarmStart

    @classmethod
    def build(cls, condition: str, model_name: str) -> "Memory":
        memo = BeliefMemo() if condition == "cache" else None
        stream = SeededStream(memo)
        if condition == "plain":
            uninstall()
        else:
            install(stream)
        install_mpc()
        return cls(condition, stream, memo, OnlineWarmStart(model_name))

    @property
    def warm_start(self):
        return self.learner.model if self.condition == "learned" else None

    def snapshot(self) -> dict | None:
        """The cache before an episode, restored if the episode fails."""
        if self.memo is None:
            return None
        return {key: list(entries) for key, entries in self.memo.plans.items()}

    def memorize(self, result: EpisodeResult, snapshot, mpc: MPCConfig) -> float:
        """Keep a solved episode's plan or rows; forget a failed one's. Returns refit time."""
        if not result.goal_reached:
            if self.memo is not None and snapshot is not None:
                self.memo.plans = snapshot
            return 0.0
        if self.condition != "learned":
            return 0.0
        self.learner.add(result.segments)
        started = time.perf_counter()
        self.learner.refit(mpc)
        return time.perf_counter() - started


# -------------------------------------------------------------------- episode


def planning_effort(stats: SeedStats, mpc: MPCConfig) -> dict:
    """The notebook's "shooting samples": MPC rollouts, plus one per policy rollout.

    `rollouts` counts every simulated rollout once; `rollouts_kept` is the older count
    (`SEED.rollouts` per seeded step, bar and rejected seeds excluded).
    """
    searched = stats.full_rollouts + stats.scored_rollouts
    kept = stats.full_steps * mpc.num_rollouts + stats.seeded_steps * SEED.rollouts
    return {
        "mpc_steps": stats.full_steps + stats.seeded_steps,
        "rollouts": searched + stats.policy_tried,
        "rollouts_kept": kept + stats.policy_tried,
        "simulator_steps": searched * mpc.horizon * mpc.action_repeat
        + stats.policy_steps * mpc.action_repeat,
        # `simulator_steps` less early stops.
        "physics_steps": stats.scored_physics + stats.full_physics
        + stats.policy_steps * mpc.action_repeat,
    }


def plan_and_execute(
    task: Task,
    theta: np.ndarray,
    believed: Believed,
    memory: Memory,
    settings: PlannerSettings,
    geometry_seed: int,
    noise_seed: int,
    planner: str = "pddlstream",
) -> tuple[EpisodeResult, dict]:
    """Plan `task` under the belief and run it on person `theta`.

    `geometry_seed` fixes grasp, base pose and IK; only the MPC noise takes `noise_seed`.
    `planner` is PDDLStream, or "pouct", "pomcp" or "pomcpow" for the POMDP planner.
    """
    memo_module.MPC_NOISE_SEED = noise_seed
    STATS.reset()
    if memory.memo is not None:
        memory.memo.reset()
    # The cache matches beliefs by distance, so the belief is not part of the key.
    memory.stream.key_suffix = ()
    if memory.memo is not None:
        memory.memo.current = believed.belief.key_features()
    target, retargeted = believed.retarget(task)
    if planner == "pddlstream":
        plan_episode = run_episode
    else:
        # Lazy import: the POMDP planner is optional.
        from limb_baseline.pomdp_model import PLAN_ONCE
        from limb_baseline.pomdp_planner import run_episode_plan_once

        plan_episode = partial(
            run_episode_plan_once, config=dataclasses.replace(PLAN_ONCE, solver=planner)
        )
    result = plan_episode(
        target,
        theta,
        belief=believed.belief,
        believed_theta=believed.theta,
        warm_start=memory.warm_start,
        seed=geometry_seed,
        settings=settings,
        particle_limits=believed.limits,
    )
    memo = memory.memo
    stats = {
        "solved": result.solved,
        "goal_reached": result.goal_reached,
        "plan_time": round(result.plan_time, 2),
        "plan_cpu_time": round(result.plan_cpu_time, 2),
        "execute_time": round(result.execute_time, 2),
        **planning_effort(STATS, settings.mpc),
        "memo_hits": memo.hits if memo else 0,
        "memo_misses": memo.misses if memo else 0,
        "memo_size": memo.size if memo else 0,
        "memo_belief_distance": round(memo.last_distance, 4) if memo else float("nan"),
        "policy_tried": STATS.policy_tried,
        "policy_reached": STATS.policy_reached,
        "policy_steps": STATS.policy_steps,
        "policy_seconds": round(STATS.policy_seconds, 2),
        "model_seconds": round(STATS.model_seconds, 2),
        "score_seconds": round(STATS.score_seconds, 2),
        "full_seconds": round(STATS.full_seconds, 2),
        "scored_rollouts": STATS.scored_rollouts,
        "scored_physics": STATS.scored_physics,
        "full_physics": STATS.full_physics,
        "policy_failure": STATS.policy_failure,
        "seeded_steps": STATS.seeded_steps,
        "full_steps": STATS.full_steps,
        "seed_rejected": STATS.seed_rejected,
        "num_control_steps": result.num_control_steps,
        "rom_violation_steps": result.rom_violation_steps,
        "rom_margin": round(result.rom_margin, 4),
        "failure_reason": result.failure_reason,
        "goal_beyond_belief": retargeted,
        "retarget_shift": round(distance(target.limb_goal, task.limb_goal), 4),
        "goal_distance": round(goal_distance(task, result), 4),
    }
    return result, stats


def limits_record(task: Task, theta: np.ndarray, believed: Believed) -> dict:
    """Start, goal, and true and believed limits on the active joints, for plotting."""
    joints = list(active_joints())
    true_lo, true_hi = (np.asarray(b)[joints] for b in joint_limits(theta))
    bel_lo, bel_hi = (np.asarray(b)[joints] for b in joint_limits(believed.theta))

    def fmt(values) -> str:
        return " ".join(f"{v:.4f}" for v in values)

    return {
        "start_active": fmt(np.asarray(task.limb_init)[joints]),
        "goal_active": fmt(np.asarray(task.limb_goal)[joints]),
        "true_lower_active": fmt(true_lo),
        "true_upper_active": fmt(true_hi),
        "believed_lower_active": fmt(bel_lo),
        "believed_upper_active": fmt(bel_hi),
    }


def distance(a, b) -> float:
    """The environment's own metric between two limb configurations."""
    return joint_position_distance(LIMB_JOINT_INFOS, list(a), list(b))


def goal_distance(task: Task, result: EpisodeResult) -> float:
    """The notebook's error: where the limb ended, from the original goal.

    An episode with no plan never moved, so it is scored from the start.
    """
    final = result.limb_path[-1] if result.limb_path else task.limb_init
    return distance(final, task.limb_goal)


def dream_once(
    believed: Believed,
    task_set: TaskSet,
    memory: Memory,
    settings: PlannerSettings,
    rng: np.random.Generator,
    config: DreamConfig,
    geometry_seed: int,
    noise_seed: int,
) -> list[dict]:
    """One dream. Returns a record per imagined episode."""
    hypothesis = believed.hypothesis(rng)                # 1. pretend this is the person
    index = int(rng.integers(len(task_set)))
    records = []

    def imagine(task_index: int, seed: int, role: str, under: Believed) -> EpisodeResult:
        task = task_set.tasks[task_index]
        snapshot = memory.snapshot()
        result, stats = plan_and_execute(
            task, hypothesis, under, memory, settings,
            seed if config.vary_geometry else geometry_seed, seed, config.planner,
        )
        memory.learner.context.update(role=role, goal=task_index)
        fit_time = memory.memorize(result, snapshot, settings.mpc)
        records.append({
            "role": role,
            "goal_index": task_index,
            "goal_shift": round(task_shift(task, task_set.nominal), 4),
            "hypothesis": " ".join(f"{v:.4f}" for v in hypothesis),
            "belief_key": " ".join(map(str, belief_key(under.belief))),
            "fit_time": round(fit_time, 2),
            **stats,
            # Every row carries every column: the CSV header is the first row's keys.
            "observed": "",
            "refiled": False,
        })
        return result

    # 2. Plan as the agent would now, imagine executing, and update a copy of the belief.
    result = imagine(index, noise_seed, "chosen", believed)
    next_belief, notes = observe(believed.belief, result.limb_path, hypothesis)
    imagined = believed.with_belief(next_belief, config, rng)
    records[-1]["observed"] = " ".join(notes)

    # 3. Memorize plans under the imagined next belief. The chosen goal needs no second
    # search when its plan reached the goal and the imagined belief aims at the same
    # place: a safe execution only raises floors, so the plan still fits, and it is
    # refiled under the imagined belief instead.
    same_key = belief_key(imagined.belief) == belief_key(believed.belief)
    goals = range(len(task_set)) if config.all_goals else [index]
    for goal in goals:
        if goal == index:
            if same_key:
                continue
            if result.goal_reached and refile(task_set.tasks[goal], believed, imagined, memory):
                records[-1]["refiled"] = True
                continue
        imagine(goal, noise_seed * 10 + goal, "imagined", imagined)
    return records


def refile(task: Task, believed: Believed, imagined: Believed, memory: Memory) -> bool:
    """File the plan just made for `task` under the imagined belief, if it still aims
    at the same goal there. The learned condition needs nothing: its rows read the
    belief's mean and spread, which a floor does not move. False means search again."""
    before, _ = believed.retarget(task)
    after, _ = imagined.retarget(task)
    key = goal_key(before.limb_goal)
    if goal_key(after.limb_goal) != key:
        return False
    if memory.memo is not None:
        stored = memory.memo.last_stored
        if stored is None or stored[0] != key:
            return False
        memory.memo.store(key, stored[1], belief=imagined.belief.key_features())
    return True


# ----------------------------------------------------------------- checkpoint

# What must match for a checkpoint to be resumed; everything but the episode count.
IDENTITY = ("condition", "model", "n_dreams", "all_goals", "num_goals", "variation",
            "particles", "confidence", "dream_max_time", "seed_mode", "seed_rollouts",
            "variant", "person", "task_seed", "planner", "seed_reuse", "num_rollouts",
            "feedback")


def checkpoint_path(out: Path) -> Path:
    return out.with_name(out.stem + "_checkpoint.pkl")


def save_checkpoint(path: Path, state: dict) -> None:
    """Write `state` whole or not at all: a kill mid-write leaves the previous one."""
    partial = path.with_name(path.name + ".tmp")
    with open(partial, "wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(partial, path)


def truncate_rows(path: Path, episodes: int) -> None:
    """Drop rows of episode `episodes` on, written after the checkpoint by a killed run."""
    if not path.exists():
        return
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return
    column = rows[0].index("episode")
    kept = [rows[0]] + [r for r in rows[1:] if int(r[column]) < episodes]
    if len(kept) == len(rows):
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(kept)


# ------------------------------------------------------------------ experiment


def run_sequence(
    condition: str,
    config: DreamConfig,
    person: int,
    task_seed: int,
    model_name: str,
    settings: PlannerSettings,
    out: Path,
    tag: str,
    resume: bool = False,
) -> list[dict]:
    """Run the episodes in this process, one CSV row each; dreams go to `<out>_dreams.csv`.

    After every episode, what the next one depends on (belief, particles, generators,
    cache or training rows) is saved to `<out>_checkpoint.pkl`. With `resume` a run
    picks up there, so raising `num_episodes` only runs the new episodes: goals are
    drawn so that a longer run's first episodes are a shorter run's.
    """
    isolate_worker()
    theta = DEFAULT_PRIOR.sample(np.random.default_rng(person))
    belief = Belief.from_prior(DEFAULT_PRIOR)
    floors = None  # noqa: F841  (Stage 1 state)
    task_rng = np.random.default_rng(task_seed)
    dream_rng = np.random.default_rng(task_seed + 1)
    particle_rng = np.random.default_rng(task_seed + 2)
    # Fixed goals inside the true range, as in the notebook.
    if config.fresh_goals:
        task_set = make_task_set(
            theta, task_rng, dataclasses.replace(config, num_goals=config.num_episodes))
        pool_rng = np.random.default_rng(task_seed + 3)
        nominal = task_set.nominal

        def draw(rng):
            if config.random_goals:
                return random_task(nominal, theta, rng)
            return perturbed_task(nominal, theta, rng, config.variation, vary_init=False)

        if config.random_goals:
            task_set = TaskSet(nominal, [draw(task_rng) for _ in range(config.num_episodes)])
        dream_tasks = TaskSet(nominal, [draw(pool_rng) for _ in range(config.dream_pool)])
    else:
        task_set = make_task_set(theta, task_rng, config)
        dream_tasks = task_set
    # The limb is seen resting at its start, so the person reaches it: a floor, as a
    # safe execution gives. Without it a start beyond the belief makes every plan
    # leave the believed range at its first step.
    belief, start_notes = observe(belief, [task_set.nominal.limb_init], theta)
    believed = Believed.build(belief, config, particle_rng)
    goal_sequence = (np.arange(config.num_episodes) if config.fresh_goals
                     else task_rng.integers(len(task_set), size=config.num_episodes))
    n_dreams = config.n_dreams if condition != "plain" else 0
    memory = Memory.build(condition, model_name)
    dreams_out = out.with_name(out.stem + "_dreams.csv")
    dream_settings = dataclasses.replace(settings, max_time=config.dream_max_time)
    common = {
        "tag": tag,
        "condition": condition,
        "model": model_name if condition == "learned" else "",
        "n_dreams": n_dreams,
        "all_goals": config.all_goals,
        "fresh_goals": config.fresh_goals,
        "random_goals": config.random_goals,
        "vary_geometry": config.vary_geometry,
        "planner": config.planner,
        "num_goals": len(task_set),
        "variation": config.variation,
        "particles": config.particles,
        "confidence": config.confidence,
        "dream_max_time": config.dream_max_time,
        "seed_mode": SEED.mode,
        "seed_rollouts": SEED.rollouts,
        "seed_reuse": SEED.reuse,
        "num_rollouts": settings.mpc.num_rollouts,
        "feedback": bool(settings.mpc.feedback_gains),
        "variant": current_variant().name,
        "person": person,
        "task_seed": task_seed,
        "theta": " ".join(f"{v:.4f}" for v in theta),
        "start_observed": " ".join(start_notes),
    }
    previous_width = belief.width

    records = []
    start = 0
    checkpoint = checkpoint_path(out)
    if resume and checkpoint.exists():
        with open(checkpoint, "rb") as handle:
            state = pickle.load(handle)
        # Defaults for checkpoints that predate these fields.
        saved = {"planner": "pddlstream", "seed_reuse": False,
                 "num_rollouts": MPCConfig.num_rollouts, "feedback": True,
                 **state["common"]}
        mismatch = {k: (saved[k], common[k]) for k in IDENTITY if saved[k] != common[k]}
        if mismatch:
            raise ValueError(f"{checkpoint} is from another run: {mismatch}")
        start = state["episode"]
        belief, believed = state["belief"], state["believed"]
        previous_width = state["previous_width"]
        dream_rng.bit_generator.state = state["dream_rng"]
        particle_rng.bit_generator.state = state["particle_rng"]
        if memory.memo is not None:
            memory.memo.plans = state["plans"]
        memory.learner = state["learner"]
        records = state["records"]
        truncate_rows(out, start)
        truncate_rows(dreams_out, start)
        print(f"[{tag}] resumed from {checkpoint} at episode {start}", flush=True)
    elif resume:
        # Nothing to resume: start clean rather than append to a stale CSV.
        truncate_rows(out, 0)
        truncate_rows(dreams_out, 0)
    if condition == "learned":
        # One JSON line per refit: the learned model's learning curve.
        learning_log = out.with_name(out.stem + "_learning.jsonl")
        if start == 0:
            learning_log.unlink(missing_ok=True)
        memory.learner.log_path = str(learning_log)

    for episode in range(start, config.num_episodes):
        # Stage 1: information gathering (off).
        # gathered = gather_information(
        #     theta, np.random.default_rng(task_seed * 1000 + episode),
        #     QueryPolicy(num_queries=config.num_queries), start=belief, floors=floors,
        # )
        # Stage 2: human model learning from the answers (off).
        # belief, floors = gathered.posterior, gathered.floors
        width = belief.width
        revised = width < previous_width - 1e-9
        previous_width = width
        believed = believed.with_belief(belief, config, particle_rng)

        # Stage 3: dreaming, timed apart from real planning.
        dream_started = time.perf_counter()
        dream_rows = []
        for dream in range(n_dreams):
            noise_seed = task_seed * 100_000 + episode * 100 + dream + 1
            memory.learner.context = {"episode": episode, "dream": dream}
            try:
                rows = dream_once(
                    believed, dream_tasks, memory, dream_settings, dream_rng, config,
                    task_seed, noise_seed,
                )
            except Exception as exc:  # pylint: disable=broad-except
                rows = [{"role": "chosen", "solved": False, "goal_reached": False,
                         "failure_reason": f"{type(exc).__name__}: {exc}"}]
                traceback.print_exc()
            for row in rows:
                append_row(dreams_out, {**common, "episode": episode, "dream": dream, **row})
                print(
                    f"[{tag}/{condition} ep {episode} dream {dream} {row.get('role')}] "
                    f"goal={row.get('goal_index')} plan={row.get('plan_time')}s "
                    f"reached={row.get('goal_reached')} hit={row.get('memo_hits', 0)} "
                    f"{row.get('failure_reason', '')}",
                    flush=True,
                )
            dream_rows.extend(rows)
        dream_time = time.perf_counter() - dream_started

        # Stage 4: plan and execute one real task, then learn from the execution.
        task = task_set.tasks[int(goal_sequence[episode])]
        record = {
            **common,
            "episode": episode,
            "goal_index": int(goal_sequence[episode]),
            "mpc_seed": task_seed * 100_000 + episode * 100,
            "separation": round(task.separation, 4),
            "goal_shift": round(task_shift(task, task_set.nominal), 4),
            "belief_mean": " ".join(f"{v:.4f}" for v in belief.mean),
            "belief_stddev": " ".join(f"{v:.4f}" for v in belief.stddev),
            "belief_floor": " ".join(f"{v:.4f}" for v in belief.lower),
            "believed_theta": " ".join(f"{v:.4f}" for v in believed.theta),
            "belief_width": round(width, 6),
            "belief_revised": revised,
            **limits_record(task, theta, believed),
            "num_train_rows": len(memory.learner.features),
            "dream_time": round(dream_time, 2),
            "dreams_reached": sum(bool(r.get("goal_reached")) for r in dream_rows),
            "dreams_planned": len(dream_rows),
            "fit_time": 0.0,
            "observed": "",
        }
        memory.learner.context = {"episode": episode, "dream": -1, "role": "real",
                                  "goal": record["goal_index"]}
        try:
            snapshot = memory.snapshot()
            result, stats = plan_and_execute(
                task, theta, believed, memory, settings,
                record["mpc_seed"] if config.vary_geometry else task_seed, record["mpc_seed"],
                config.planner,
            )
            record.update(stats)
            record["fit_time"] = round(memory.memorize(result, snapshot, settings.mpc), 2)
            belief, notes = observe(belief, result.limb_path, theta)   # belief update
            record["observed"] = " ".join(notes)
        except Exception as exc:  # pylint: disable=broad-except
            record.update({"solved": False, "goal_reached": False,
                           "plan_time": float("nan"),
                           "failure_reason": f"{type(exc).__name__}: {exc}"})
            traceback.print_exc()
        append_row(out, record)
        records.append(record)
        print(
            f"[{tag}/{condition} ep {episode + 1}/{config.num_episodes}] "
            f"goal={record['goal_index']} plan={record.get('plan_time')}s "
            f"rollouts={record.get('rollouts')} reached={record.get('goal_reached')} "
            f"hits={record.get('memo_hits', 0)} rows={len(memory.learner.features)} "
            f"dreams={record['dreams_reached']}/{record['dreams_planned']} "
            f"in {dream_time:.0f}s width={width:.3f} observed=[{record['observed']}] "
            f"{record.get('failure_reason', '')}",
            flush=True,
        )
        save_checkpoint(checkpoint, {
            "episode": episode + 1,
            "common": common,
            "belief": belief,
            "believed": believed,
            "previous_width": previous_width,
            "dream_rng": dream_rng.bit_generator.state,
            "particle_rng": particle_rng.bit_generator.state,
            "plans": memory.memo.plans if memory.memo is not None else None,
            "learner": memory.learner,
            "records": records,
        })
    uninstall()
    return records


def main() -> None:
    """CLI: one condition, one sequence of episodes, one CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--n-dreams", type=int, default=DreamConfig.n_dreams,
                        help="Dreams before every episode; plain ignores them.")
    parser.add_argument("--all-goals", action="store_true",
                        help="Each dream also solves every goal of the task set.")
    parser.add_argument("--planner", choices=("pddlstream", "pouct", "pomcp", "pomcpow"),
                        default=DreamConfig.planner,
                        help="PDDLStream, or the POMDP planner solved by POUCT or POMCP")
    parser.add_argument("--vary-geometry", action="store_true",
                        help="Sample grasp, base pose and IK afresh for every plan.")
    parser.add_argument("--random-goals", action="store_true",
                        help="Fresh goals drawn uniformly over the person's true range.")
    parser.add_argument("--fresh-goals", action="store_true",
                        help="A new goal every real episode; dreams draw their own goals.")
    parser.add_argument("--num-goals", type=int, default=DreamConfig.num_goals)
    parser.add_argument("--variation", type=float, default=DreamConfig.variation,
                        help="Radians each goal may sit from the nominal goal.")
    parser.add_argument("--num-episodes", type=int, default=DreamConfig.num_episodes)
    parser.add_argument("--particles", type=int, default=DreamConfig.particles,
                        help="Belief particles the planner checks against; 0 uses the "
                             "belief's mean as one range.")
    parser.add_argument("--confidence", type=float, default=DreamConfig.confidence,
                        help="Fraction of particles a configuration must lie inside.")
    parser.add_argument("--dream-max-time", type=float, default=DreamConfig.dream_max_time,
                        help="Planner budget per imagined episode, in seconds.")
    parser.add_argument("--model", type=str, default="ridge",
                        help="Regressor the learned condition refits after each solve: one "
                             f"of {', '.join(MODELS)}, or a spec such as mlp:64x64:a0.01 "
                             "or ridge:a-1,5:poly2 (see warmstart.make_regressor).")
    parser.add_argument("--seed-mode", choices=("plan", "step"), default=SEED.mode,
                        help="plan: roll the seed out whole, search only if it fails; "
                             "step: offer the seed as batch 0 of every MPC step.")
    parser.add_argument("--seed-rollouts", type=int, default=SEED.rollouts)
    parser.add_argument("--no-seed-reuse", dest="seed_reuse", action="store_false",
                        help="Step mode: the earlier scoring, which scores every draw "
                             "against the directed candidates and runs a rejected step's "
                             "full batch from scratch.")
    # The arm every dream experiment and the tuning rows use; the package default is the
    # right arm, which a bare command would otherwise silently switch to.
    parser.add_argument("--variant", choices=VARIANTS, default="isolated-left-arm")
    parser.add_argument("--person", type=int, default=22,
                        help="Seed of the person whose true range of motion is used.")
    parser.add_argument("--task-seed", type=int, default=0)
    parser.add_argument("--max-time", type=float, default=600.0,
                        help="Planner budget per real episode, in seconds, enforced exactly "
                             "(the planner is stopped mid-stream at the deadline).")
    parser.add_argument("--num-rollouts", type=int, default=MPCConfig.num_rollouts)
    parser.add_argument("--no-feedback", dest="feedback", action="store_false",
                        help="Drop the MPC's feedback-controller candidates, leaving it "
                             "the sampled rollouts alone.")
    parser.add_argument("--max-control-steps", type=int, default=MPCConfig.max_control_steps)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from <out>_checkpoint.pkl if it exists, running "
                             "only the episodes past it; without one, start afresh.")
    args = parser.parse_args()
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, "1")

    make_regressor(args.model)  # fail on a bad spec before any planning
    select_variant(args.variant)
    SEED.mode = args.seed_mode
    SEED.rollouts = args.seed_rollouts
    SEED.reuse = args.seed_reuse
    mpc = dataclasses.replace(
        MPCConfig(),
        num_rollouts=args.num_rollouts,
        max_control_steps=args.max_control_steps,
    )
    if not args.feedback:
        mpc = dataclasses.replace(mpc, feedback_gains=())
    settings = PlannerSettings(max_time=args.max_time, mpc=mpc, hard_time_limit=True)
    config = DreamConfig(
        n_dreams=args.n_dreams,
        all_goals=args.all_goals,
        fresh_goals=args.fresh_goals or args.random_goals,
        random_goals=args.random_goals,
        vary_geometry=args.vary_geometry,
        planner=args.planner,
        num_goals=args.num_goals,
        variation=args.variation,
        num_episodes=args.num_episodes,
        particles=args.particles,
        confidence=args.confidence,
        dream_max_time=args.dream_max_time,
    )
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.condition}_{args.n_dreams}"
    records = run_sequence(
        args.condition, config, args.person, args.task_seed, args.model, settings,
        out, tag, resume=args.resume,
    )
    times = [r["plan_time"] for r in records if r.get("goal_reached")]
    print(
        f"\n{args.condition}, {config.n_dreams} dreams: "
        f"{sum(bool(r.get('goal_reached')) for r in records)}/{len(records)} reached "
        f"the goal; median plan time of those "
        f"{np.median(times) if times else float('nan'):.1f} s\nWritten to {out}"
    )


if __name__ == "__main__":
    main()
