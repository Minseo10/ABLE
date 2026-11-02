"""Stage 4: plan for freshly drawn people the model has never met, and log the results."""

from __future__ import annotations

import argparse
import dataclasses
import os
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from able.tasks.limb.belief import Belief
from able.tasks.limb.dataset import append_row, guard_variant
from able.tasks.limb.human import DEFAULT_PRIOR
from able.tasks.limb.planner import (
    PlannerSettings,
    QueryPolicy,
    assumed_theta,
    gather_information,
    run_episode,
)
from able.tasks.limb.planning.stream import MPCConfig
from able.tasks.limb.variant import DEFAULT_VARIANT, VARIANTS
from able.tasks.limb.variant import current as current_variant
from able.tasks.limb.variant import select as select_variant
from able.tasks.limb.warmstart import WarmStartModel
from able.tasks.limb.worker import isolate_worker, task_sampler

# Person ids for evaluation start here, well clear of any collection sweep.
EVAL_PERSON_OFFSET = 1_000_000

# The oracle knows `theta`, so its belief is a spike on the truth.
ORACLE_STDDEV = 1e-6


@dataclass(frozen=True)
class BaselineSpec:
    """One row of the evaluation table."""

    name: str
    learn_theta: bool
    use_warm_start: bool
    oracle: bool
    note: str = ""


BASELINES: tuple[BaselineSpec, ...] = (
    BaselineSpec("pddlstream", False, False, False, "the planner as it stands today"),
    BaselineSpec("belief", True, False, False, "querying alone"),
    BaselineSpec("warm_start", False, True, False, "warm starting alone"),
    BaselineSpec("baseline", True, True, False, "both, with the simplest models"),
    BaselineSpec("oracle_cold", False, False, True, "ground truth theta, cold start"),
    BaselineSpec("oracle", False, True, True, "ground truth theta, the ceiling"),
)


def belief_for(
    spec: BaselineSpec,
    theta: np.ndarray,
    rng: np.random.Generator,
    num_queries: int,
) -> Belief:
    """What this row believes about the person before it plans."""
    if spec.oracle:
        return Belief(mean=np.asarray(theta, dtype=np.float64).copy(),
                      stddev=np.full(len(theta), ORACLE_STDDEV))
    if not spec.learn_theta:
        return Belief.from_prior(DEFAULT_PRIOR)
    return gather_information(theta, rng, QueryPolicy(num_queries=num_queries)).posterior




def _work_item(payload) -> dict:
    """Run one (row, person, task)."""
    spec, person, task_index, model, num_queries, settings, results_path = payload
    isolate_worker()
    seed = person * 1000 + task_index
    rng = np.random.default_rng(seed)
    record = {
        "row": spec.name,
        "variant": current_variant().name,
        "person": person,
        "task_index": task_index,
        "seed": seed,
    }
    try:
        # The true range of motion belongs to the person, not to one of their tasks.
        theta = DEFAULT_PRIOR.sample(np.random.default_rng(person))
        belief = belief_for(spec, theta, rng, num_queries)
        assumed = assumed_theta(belief, theta, rng, spec.oracle)
        task = task_sampler().sample(rng, assumed, theta)
        result = run_episode(
            task,
            theta,
            belief=belief,
            believed_theta=assumed,
            warm_start=model if spec.use_warm_start else None,
            seed=seed,
            settings=settings,
        )
        record.update(
            {
                "solved": result.solved,
                "goal_reached": result.goal_reached,
                "plan_time": round(result.plan_time, 2),
                "mpc_simulator_steps": result.mpc_simulator_steps,
                "num_control_steps": result.num_control_steps,
                "num_segments": result.num_segments,
                "rom_violation_steps": result.rom_violation_steps,
                "worst_rom_violation": round(result.worst_rom_violation, 6),
                "rom_margin": round(result.rom_margin, 6),
                "belief_error": round(float(np.abs(belief.mean - theta).max()), 6),
                "separation": round(task.separation, 4),
                "failure_reason": result.failure_reason,
            }
        )
    except Exception as exc:  # pylint: disable=broad-except
        record.update({"solved": False, "goal_reached": False,
                       "failure_reason": f"{type(exc).__name__}: {exc}"})
        traceback.print_exc()
    append_row(results_path, record)
    return record


def summarize(records: list[dict]) -> str:
    """The evaluation table: one line per row, the four metrics the doc asks for."""
    frame = pd.DataFrame(records)
    lines = [
        f"{'row':<12s} {'success':>8s} {'plan s':>8s} {'mpc steps':>11s} "
        f"{'RoM viol':>9s} {'ctrl steps':>11s}",
        "-" * 64,
    ]
    for spec in BASELINES:
        rows = frame[frame["row"] == spec.name]
        if rows.empty:
            continue
        reached = rows[rows["goal_reached"] == True]  # noqa: E712
        lines.append(
            f"{spec.name:<12s} "
            f"{len(reached) / len(rows):>7.0%} "
            f"{rows['plan_time'].mean():>8.0f} "
            f"{(reached['mpc_simulator_steps'].mean() if len(reached) else float('nan')):>11.0f} "
            f"{rows['rom_violation_steps'].sum():>9.0f} "
            f"{(reached['num_control_steps'].mean() if len(reached) else float('nan')):>11.0f}"
        )
    return "\n".join(lines)



def main() -> None:
    """CLI entry point."""
    import multiprocessing as mp
    import pickle

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", type=str, default=DEFAULT_VARIANT, choices=VARIANTS
    )
    parser.add_argument("--num-tasks", type=int, default=50)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--num-queries", type=int, default=6)
    parser.add_argument("--max-time", type=float, default=600.0)
    parser.add_argument("--mpc-rollouts", type=int, default=None,
                        help="Shrink the MPC's own search so the warm start matters.")
    parser.add_argument("--mpc-max-control-steps", type=int, default=None,
                        help="Cap the controller budget one segment may spend.")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="A collect run directory; eval.csv is written there and "
                             "warm_start.pkl read from it unless --model says otherwise.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Where rows are written (default <run-dir>/eval.csv).")
    parser.add_argument("--model", type=Path, default=None,
                        help="Pickled WarmStartModel; rows needing psi are skipped without one.")
    parser.add_argument("--rows", type=str, default="",
                        help="Comma-separated subset of row names; empty runs all.")
    parser.add_argument("--person", type=int, default=None,
                        help="Evaluate this one person on held-out tasks, for a run "
                             "collected from a single person; omit to draw fresh people.")
    parser.add_argument("--commit-steps", type=int, default=MPCConfig.commit_steps,
                        help="Control steps executed per MPC solve.")
    parser.add_argument("--num-rollouts", type=int, default=MPCConfig.num_rollouts,
                        help="Candidate horizons scored per MPC solve.")
    parser.add_argument("--directed-gains", type=str, default="",
                        help="Comma-separated goal-directed gains; empty keeps the default.")
    parser.add_argument("--task-offset", type=int, default=0,
                        help="First task index, so --person tasks clear the collected ones.")
    args = parser.parse_args()
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, "1")

    select_variant(args.variant)
    out = (args.out or args.run_dir / "eval.csv").resolve()
    guard_variant(out, args.variant)

    model_path = args.model or args.run_dir / "warm_start.pkl"
    model: WarmStartModel | None = None
    if model_path.exists():
        with open(model_path, "rb") as handle:
            model = pickle.load(handle)

    mpc = MPCConfig()
    overrides = {}
    if args.mpc_rollouts is not None:
        overrides["num_rollouts"] = args.mpc_rollouts
    if args.mpc_max_control_steps is not None:
        overrides["max_control_steps"] = args.mpc_max_control_steps
    if overrides:
        mpc = dataclasses.replace(mpc, **overrides)
    settings = PlannerSettings(max_time=args.max_time, mpc=mpc)

    mpc_kwargs = {"commit_steps": args.commit_steps,
                  "num_rollouts": args.num_rollouts}
    if args.directed_gains:
        mpc_kwargs["directed_gains"] = tuple(
            float(g) for g in args.directed_gains.split(",")
        )

    wanted = set(args.rows.split(",")) if args.rows else None
    specs = [s for s in BASELINES if wanted is None or s.name in wanted]
    specs = [s for s in specs if model is not None or not s.use_warm_start]
    if not specs:
        raise SystemExit("no rows to run (rows needing psi require --model)")

    if args.person is None:
        drawn = [(EVAL_PERSON_OFFSET + i, args.task_offset) for i in range(args.num_tasks)]
    else:
        drawn = [(args.person, args.task_offset + i) for i in range(args.num_tasks)]
    payloads = [
        (spec, person, task_index, model, args.num_queries, settings, out)
        for spec in specs
        for person, task_index in drawn
    ]
    records = []
    with mp.get_context("spawn").Pool(processes=args.workers) as pool:
        for done, record in enumerate(pool.imap_unordered(_work_item, payloads), 1):
            records.append(record)
            print(f"[{done}/{len(payloads)}] {record['row']} person={record['person']} "
                  f"reached={record.get('goal_reached')} {record.get('failure_reason','')}",
                  flush=True)
    print("\n" + summarize(records))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
