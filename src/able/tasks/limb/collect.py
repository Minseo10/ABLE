"""Collect demonstrations in parallel; one work item is one person and one task.

The person is drawn from the prior and asked `K` questions, so a demonstration is what
the robot does knowing only what the queries told it. Every attempt is logged; only the
successful ones become training rows. `--method pddlstream` plans to a sampled goal;
`--method babble` relabels random torques.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from able.tasks.limb.babble import BabbleSettings, run_babble_episode
from able.tasks.limb.belief import Belief
from able.tasks.limb.dataset import append_row, guard_variant, row_from_episode
from able.tasks.limb.human import DEFAULT_PRIOR
from able.tasks.limb.planner import (
    PlannerSettings,
    QueryPolicy,
    assumed_theta,
    gather_information,
    run_episode,
)
from able.tasks.limb.variant import DEFAULT_VARIANT, VARIANTS
from able.tasks.limb.variant import current as current_variant
from able.tasks.limb.variant import select as select_variant
from able.tasks.limb.worker import isolate_worker, task_sampler

METHODS = ("pddlstream", "babble")


@dataclass(frozen=True)
class CollectConfig:
    """What one sweep collects and where it lands."""

    demos_path: Path
    attempts_path: Path
    method: str = "pddlstream"
    num_queries: int = 6
    max_time: float = 600.0
    oracle: bool = False
    use_prior_only: bool = False
    babble: BabbleSettings = dataclasses.field(default_factory=BabbleSettings)




def _plan_demonstration(
    task, theta, belief, assumed, seed: int, config: CollectConfig
) -> tuple[dict, list]:
    """Plan and execute one task, returning what it did and the rows it earned."""
    result = run_episode(
        task,
        theta,
        belief=belief,
        believed_theta=assumed,
        seed=seed,
        settings=PlannerSettings(max_time=config.max_time),
    )
    record = {
        "solved": result.solved,
        "goal_reached": result.goal_reached,
        "plan_time": round(result.plan_time, 2),
        "execute_time": round(result.execute_time, 2),
        "num_control_steps": result.num_control_steps,
        "mpc_simulator_steps": result.mpc_simulator_steps,
        "rom_violation_steps": result.rom_violation_steps,
        "worst_rom_violation": round(result.worst_rom_violation, 6),
        "rom_margin": round(result.rom_margin, 6),
        "separation": round(task.separation, 4),
        "failure_reason": result.failure_reason,
    }
    return record, result.segments if result.goal_reached else []


def _babble_demonstration(
    task, theta, belief, assumed, seed: int, config: CollectConfig
) -> tuple[dict, list]:
    """Babble the limb around from one start, returning what it did and its rows."""
    started = time.time()
    result = run_babble_episode(
        task,
        theta,
        belief=belief,
        believed_theta=assumed,
        seed=seed,
        settings=PlannerSettings(max_time=config.max_time),
        babble=config.babble,
    )
    separations = result.separations
    record = {
        # A babbled goal is reached by construction, so "solved" is whether the
        # episode produced any segment worth keeping.
        "solved": result.num_kept > 0,
        "goal_reached": result.num_kept > 0,
        "plan_time": 0.0,
        "execute_time": round(time.time() - started, 2),
        "num_control_steps": result.num_control_steps,
        "mpc_simulator_steps": 0,
        "rom_violation_steps": 0,
        "worst_rom_violation": 0.0,
        "separation": round(float(np.mean(separations)), 4) if separations else 0.0,
        "num_segments": result.num_kept,
        "num_attempted": result.num_attempted,
        "num_redraws": result.num_redraws,
        "failure_reason": result.failure_reason,
    }
    return record, result.segments


def _work_item(payload: tuple[int, int, CollectConfig]) -> dict:
    """Run one (person, task) and return what it did, recording it on the way."""
    person, task_index, config = payload
    isolate_worker()
    seed = person * 1000 + task_index
    rng = np.random.default_rng(seed)
    record: dict = {
        "variant": current_variant().name,
        "person": person,
        "task_index": task_index,
        "seed": seed,
    }
    try:
        # The true range of motion belongs to the person, not to one of their tasks.
        theta = DEFAULT_PRIOR.sample(np.random.default_rng(person))
        if config.use_prior_only:
            belief = Belief.from_prior(DEFAULT_PRIOR)
            num_censored = 0
        else:
            gathered = gather_information(
                theta, rng, QueryPolicy(num_queries=config.num_queries)
            )
            belief, num_censored = gathered.posterior, gathered.num_censored
        sampler = task_sampler()
        assumed = assumed_theta(belief, theta, rng, config.oracle)
        if config.method == "babble":
            task = sampler.sample_init(rng, theta)
            outcome, segments = _babble_demonstration(
                task, theta, belief, assumed, seed, config
            )
        else:
            task = sampler.sample(rng, assumed, theta)
            outcome, segments = _plan_demonstration(
                task, theta, belief, assumed, seed, config
            )
        # Last of the shared columns, as it was before there was a second method.
        reason = outcome.pop("failure_reason", "")
        record.update(outcome)
        record.update(
            {
                "num_censored": num_censored,
                "belief_error": round(float(np.abs(belief.mean - theta).max()), 6),
                "failure_reason": reason,
                # Recorded so a failed task can be replayed.
                **{f"limb_init_{i}": round(v, 6) for i, v in enumerate(task.limb_init)},
                **{f"limb_goal_{i}": round(v, 6) for i, v in enumerate(task.limb_goal)},
                **{
                    f"robot_init_{i}": round(v, 6)
                    for i, v in enumerate(task.robot_init)
                },
                **{f"theta_{i}": round(float(v), 6) for i, v in enumerate(theta)},
            }
        )
        append_row(config.attempts_path, record)
        # Only the demonstrations carry it, so the attempts table keeps its columns.
        record.setdefault("num_segments", len(segments))
        # One row per segment: each is its own warm start.
        for index, segment in enumerate(segments):
            append_row(
                config.demos_path,
                row_from_episode(
                    segment.features, segment.psi, {**record, "segment": index}
                ),
            )
    except Exception as exc:  # pylint: disable=broad-except
        record.update(
            {
                "solved": False,
                "goal_reached": False,
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }
        )
        traceback.print_exc()
        append_row(config.attempts_path, record)
    return record


def collect(
    num_people: int,
    tasks_per_person: int,
    workers: int,
    config: CollectConfig,
    task_offset: int = 0,
    person_offset: int = 0,
) -> list[dict]:
    """Run the sweep across a process pool, one PyBullet client per worker."""
    import multiprocessing as mp

    payloads = [
        (person, task_offset + task_index, config)
        for person in range(person_offset, person_offset + num_people)
        for task_index in range(tasks_per_person)
    ]
    records: list[dict] = []
    with mp.get_context("spawn").Pool(processes=workers) as pool:
        for done, record in enumerate(
            pool.imap_unordered(_work_item, payloads), start=1
        ):
            records.append(record)
            print(
                f"[{done}/{len(payloads)}] person={record['person']} "
                f"task={record['task_index']} "
                f"solved={record.get('solved')} reached={record.get('goal_reached')} "
                f"plan={record.get('plan_time')}s "
                f"steps={record.get('num_control_steps')} "
                f"sep={record.get('separation')} "
                f"{record.get('failure_reason', '')}",
                flush=True,
            )
    return records



def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", type=str, default=DEFAULT_VARIANT, choices=VARIANTS
    )
    parser.add_argument("--num-people", type=int, default=50)
    parser.add_argument("--person-offset", type=int, default=0,
                        help="First person id; theta is drawn from it.")
    parser.add_argument("--tasks-per-person", type=int, default=5)
    parser.add_argument("--task-offset", type=int, default=0,
                        help="First task index, so a top-up run draws fresh tasks.")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--num-queries", type=int, default=6)
    parser.add_argument("--max-time", type=float, default=600.0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--tag", type=str, default="demos")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--prior-only", action="store_true")
    parser.add_argument(
        "--method",
        choices=METHODS,
        default="pddlstream",
        help="How a demonstration is produced: plan for a goal, or babble and relabel.",
    )
    babble = parser.add_argument_group("babble")
    babble.add_argument(
        "--babble-segments", type=int, default=BabbleSettings.num_segments
    )
    babble.add_argument(
        "--babble-control-steps", type=int, default=BabbleSettings.control_steps
    )
    babble.add_argument(
        "--babble-noise-scale", type=float, default=BabbleSettings.noise_scale
    )
    babble.add_argument(
        "--babble-max-draws", type=int, default=BabbleSettings.max_draws
    )
    args = parser.parse_args()

    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, "1")

    select_variant(args.variant)
    # One run lives in data/<method>/<tag>/, so the two methods never share a table.
    run_dir = (args.data_dir / args.method / args.tag).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    for path in (run_dir / "demos.csv", run_dir / "attempts.csv"):
        guard_variant(path, args.variant)
    config = CollectConfig(
        demos_path=run_dir / "demos.csv",
        attempts_path=run_dir / "attempts.csv",
        num_queries=args.num_queries,
        max_time=args.max_time,
        oracle=args.oracle,
        use_prior_only=args.prior_only,
        method=args.method,
        babble=BabbleSettings(
            num_segments=args.babble_segments,
            control_steps=args.babble_control_steps,
            noise_scale=args.babble_noise_scale,
            max_draws=args.babble_max_draws,
        ),
    )
    records = collect(args.num_people, args.tasks_per_person, args.workers, config,
                      args.task_offset, args.person_offset)
    solved = sum(bool(r.get("goal_reached")) for r in records)
    print(f"\n=== {solved}/{len(records)} episodes reached the goal ===")
    print(f"demos    -> {config.demos_path.resolve()}")
    print(f"attempts -> {config.attempts_path.resolve()}")


if __name__ == "__main__":
    main()
