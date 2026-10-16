"""Compare PDDLStream with the POMDP planner, solved by POUCT and by POMCP.

All planners share person, goal, prior belief, particles and seed; one process per episode.

    python scripts/pomdp_compare.py --people 3 --goals 3 --jobs 9 \\
        --out data/pomdp/compare.csv
"""

import argparse
import contextlib
import csv
import dataclasses
import multiprocessing
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PLANNERS = ("pddlstream", "pouct", "pomcp")


def starting_belief(person: int, nominal_init):
    """The person, and the belief and particles every planner starts with."""
    from limb_baseline.belief import Belief
    from limb_baseline.dream_pipeline import Believed, DreamConfig, observe
    from limb_baseline.human import DEFAULT_PRIOR

    config = DreamConfig()
    theta = DEFAULT_PRIOR.sample(np.random.default_rng(person))
    belief, _ = observe(Belief.from_prior(DEFAULT_PRIOR), [nominal_init], theta)
    return theta, Believed.build(belief, config, np.random.default_rng(person + 2)), config


def draw_tasks(person: int, num_goals: int) -> tuple[list, tuple]:
    """One person's goals and nominal start, drawn in the parent so all planners share them.

    The sampler's reachability checks did not always agree across worker processes.
    """
    from limb_baseline.dream_pipeline import DreamConfig, make_task_set
    from limb_baseline.human import DEFAULT_PRIOR

    theta = DEFAULT_PRIOR.sample(np.random.default_rng(person))
    rng = np.random.default_rng(person)
    task_set = make_task_set(
        theta, rng, DreamConfig(fresh_goals=True, num_goals=num_goals)
    )
    return task_set.tasks, tuple(task_set.nominal.limb_init)


def run_one(job: dict, on_search=None) -> dict:
    """Run one planner's episode; return its CSV row."""
    from limb_baseline.variant import select

    select(job["variant"])
    # Select before imports that build variant-dependent defaults.
    from kinder_pddlstream_planning.limbrepositioning3d.stream import MPCConfig

    from limb_baseline.dream_pipeline import distance, goal_distance, observe
    from limb_baseline.planner import PlannerSettings, run_episode
    from limb_baseline.pomdp_model import PLAN_ONCE
    from limb_baseline.pomdp_planner import run_episode_plan_once
    from limb_baseline.worker import isolate_worker

    row = {k: job[k] for k in ("planner", "person", "goal", "variant")}
    log = Path(job["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as handle, contextlib.redirect_stdout(handle):
        try:
            isolate_worker()
            task = job["task"]
            theta, believed, _ = starting_belief(job["person"], job["nominal_init"])
            target, retargeted = believed.retarget(task)
            mpc = dataclasses.replace(MPCConfig(), num_rollouts=job["num_rollouts"])
            settings = PlannerSettings(
                max_time=job["max_time"], mpc=mpc, hard_time_limit=True
            )
            seed = job["person"] * 1000 + job["goal"]
            shared = (target, theta, believed.belief, believed.theta, None, seed, settings)
            if job["planner"] == "pddlstream":
                result = run_episode(*shared, believed.limits)
            else:
                config = dataclasses.replace(
                    PLAN_ONCE, solver=job["solver"], num_waypoints=job["num_waypoints"]
                )
                result = run_episode_plan_once(*shared, believed.limits, config, on_search)
            after, _ = observe(believed.belief, result.limb_path, theta)
            row.update(
                theta=" ".join(f"{v:.3f}" for v in theta),
                goal_beyond_belief=retargeted,
                retarget_shift=round(distance(target.limb_goal, task.limb_goal), 4),
                solved=result.solved,
                goal_reached=result.goal_reached,
                goal_distance=round(goal_distance(task, result), 4),
                plan_time=round(result.plan_time, 1),
                execute_time=round(result.execute_time, 1),
                moves=result.num_segments,
                control_steps=result.num_control_steps,
                mpc_rollouts=result.num_control_steps * mpc.num_rollouts,
                rom_violation_steps=result.rom_violation_steps,
                belief_width_before=round(believed.belief.width, 4),
                belief_width_after=round(after.width, 4),
                failure_reason=result.failure_reason,
            )
        except Exception:  # pylint: disable=broad-except
            traceback.print_exc()
            row["failure_reason"] = "crashed: " + traceback.format_exc().splitlines()[-1]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--variant", default="isolated-left-arm")
    parser.add_argument("--people", type=int, default=3)
    parser.add_argument("--goals", type=int, default=3, help="goals per person")
    parser.add_argument("--planners", nargs="+", default=list(PLANNERS), choices=PLANNERS)
    parser.add_argument("--num-waypoints", type=int, nargs="+", default=[1],
                        help="POMDP move_limb targets on the line to the goal; each "
                             "count is its own planner, e.g. pomcp_w4")
    parser.add_argument("--num-rollouts", type=int, default=64)
    parser.add_argument("--max-time", type=float, default=600.0)
    parser.add_argument("--jobs", type=int, default=9)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    # One thread per worker, to avoid oversubscription.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, "1")

    out = args.out.resolve()
    from limb_baseline.variant import select

    select(args.variant)
    configurations = [
        (planner, None, 1) if planner == "pddlstream" else (f"{planner}_w{n}", planner, n)
        for planner in args.planners
        for n in ([1] if planner == "pddlstream" else args.num_waypoints)
    ]
    jobs = []
    for person in range(args.people):
        tasks, nominal_init = draw_tasks(person, args.goals)
        for goal, task in enumerate(tasks):
            for planner, solver, waypoints in configurations:
                jobs.append({
                    "planner": planner,
                    "solver": solver,
                    "person": person,
                    "goal": goal,
                    "variant": args.variant,
                    "task": task,
                    "nominal_init": nominal_init,
                    "num_waypoints": waypoints,
                    "num_rollouts": args.num_rollouts,
                    "max_time": args.max_time,
                    "log": str(out.parent / "logs"
                               / f"{out.stem}_p{person}_g{goal}_{planner}.log"),
                })
    print(f"TOTAL {len(jobs)} episodes", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=context) as pool:
        futures = [pool.submit(run_one, job) for job in jobs]
        for done in as_completed(futures):
            rows.append(done.result())
            fields = list(dict.fromkeys(key for row in rows for key in row))
            with open(out, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(sorted(rows, key=lambda r: (r["person"], r["goal"], r["planner"])))
            print(f"[{len(rows)}/{len(jobs)}] {rows[-1]}", flush=True)


if __name__ == "__main__":
    main()
