"""Compare PDDLStream with the POMDP planners: POUCT, POMCP and POMCPOW.

All planners share person, goal, prior belief, particles and seed; one process per episode.
`--goal-mode risky` poses goals that pass the belief check but that some particles
place past the person's limit, where stopping short can pay.

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

PLANNERS = ("pddlstream", "pouct", "pomcp", "pomcpow")


def starting_belief(person: int, nominal_init):
    """The person, and the belief and particles every planner starts with."""
    from able.tasks.limb.belief import Belief
    from able.tasks.limb.dream_pipeline import Believed, DreamConfig, observe
    from able.tasks.limb.human import DEFAULT_PRIOR

    config = DreamConfig()
    theta = DEFAULT_PRIOR.sample(np.random.default_rng(person))
    belief, _ = observe(Belief.from_prior(DEFAULT_PRIOR), [nominal_init], theta)
    return theta, Believed.build(belief, config, np.random.default_rng(person + 2)), config


def draw_tasks(person: int, num_goals: int) -> tuple[list, tuple]:
    """One person's goals and nominal start, drawn in the parent so all planners share them.

    The sampler's reachability checks did not always agree across worker processes.
    """
    from able.tasks.limb.dream_pipeline import DreamConfig, make_task_set
    from able.tasks.limb.human import DEFAULT_PRIOR

    theta = DEFAULT_PRIOR.sample(np.random.default_rng(person))
    rng = np.random.default_rng(person)
    task_set = make_task_set(
        theta, rng, DreamConfig(fresh_goals=True, num_goals=num_goals)
    )
    return task_set.tasks, tuple(task_set.nominal.limb_init)


def particle_share(limits, goal) -> float:
    """The share of belief particles that allow `goal`."""
    goal = np.asarray(goal, dtype=np.float64)
    return float(np.mean(np.all((limits.lower <= goal) & (goal <= limits.upper), axis=1)))


def draw_risky_tasks(person: int, num_goals: int, band: tuple[float, float]) -> tuple[list, tuple]:
    """Goals from the nominal start whose particle share lies in `band`."""
    from able.tasks.limb.dream_pipeline import distance
    from able.tasks.limb.human import theta_to_joint_limits
    from able.tasks.limb.tasks import DEFAULT_TASK_DISTRIBUTION
    from able.tasks.limb.worker import task_sampler

    tasks, nominal_init = draw_tasks(person, 1)
    nominal = tasks[0]
    _, believed, _ = starting_belief(person, nominal_init)
    sampler = task_sampler()
    # Directions toward the widest plausible range.
    low, high = theta_to_joint_limits(np.quantile(believed.limits.particles, 0.95, axis=0))
    rng = np.random.default_rng(person + 1000)
    init = np.asarray(nominal.limb_init, dtype=np.float64)
    separations = np.linspace(DEFAULT_TASK_DISTRIBUTION.min_separation,
                              DEFAULT_TASK_DISTRIBUTION.max_separation, 41)
    found = []
    for _ in range(2000):
        # A direction, then a distance on it whose share is in `band`; particle boxes are
        # convex, so the share only falls along it.
        toward = sampler._draw_limb_conf(rng, low, high)  # pylint: disable=protected-access
        length = distance(init, toward)
        if length < 1e-6:
            continue
        candidates = [init + (toward - init) * d / length for d in separations]
        candidates = [g for g in candidates
                      if band[0] <= particle_share(believed.limits, g) <= band[1]]
        if not candidates:
            continue
        goal = candidates[int(rng.integers(len(candidates)))]
        if not sampler._reachable_from(nominal.base_pose, nominal.limb_init, goal):  # pylint: disable=protected-access
            continue
        found.append(dataclasses.replace(nominal, limb_goal=tuple(goal)))
        if len(found) == num_goals:
            return found, nominal_init
    raise RuntimeError(f"only {len(found)} risky goals for person {person}")


def draw_risky_fresh(variant: str, person: int, num_goals: int, band) -> tuple[list, tuple]:
    """`draw_risky_tasks` in a fresh process: sampling depends on the simulator's history."""
    from able.tasks.limb.variant import select

    select(variant)
    return draw_risky_tasks(person, num_goals, band)


def run_one(job: dict, on_search=None) -> dict:
    """Run one planner's episode; return its CSV row."""
    from able.tasks.limb.variant import select

    select(job["variant"])
    # Select before imports that build variant-dependent defaults.
    from able.tasks.limb.dream_pipeline import distance, goal_distance, observe
    from able.tasks.limb.human import theta_to_joint_limits
    from able.tasks.limb.planner import PlannerSettings, run_episode
    from able.tasks.limb.planning.stream import MPCConfig
    from able.tasks.limb.pomdp_model import PLAN_ONCE
    from able.tasks.limb.pomdp_planner import run_episode_plan_once
    from able.tasks.limb.worker import isolate_worker

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
            lower, upper = theta_to_joint_limits(theta)
            goal = np.asarray(target.limb_goal)
            reached = goal_distance(task, result)
            row.update(
                theta=" ".join(f"{v:.3f}" for v in theta),
                goal_beyond_belief=retargeted,
                retarget_shift=round(distance(target.limb_goal, task.limb_goal), 4),
                goal_particle_share=round(particle_share(believed.limits, goal), 3),
                goal_in_true_range=bool(np.all((lower <= goal) & (goal <= upper))),
                solved=result.solved,
                goal_reached=result.goal_reached,
                goal_distance=round(reached, 4),
                # Share of the way to the goal at the end.
                progress=round(1.0 - reached / distance(task.limb_init, task.limb_goal), 3),
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
    parser.add_argument("--person-ids", type=int, nargs="+", help="instead of range(--people)")
    parser.add_argument("--goals", type=int, default=3, help="goals per person")
    parser.add_argument("--planners", nargs="+", default=list(PLANNERS), choices=PLANNERS)
    parser.add_argument("--num-waypoints", type=int, nargs="+", default=[1],
                        help="POUCT/POMCP move_limb targets on the line to the goal; each "
                             "count is its own planner, e.g. pomcp_w4 (POMCPOW: always 1)")
    parser.add_argument("--goal-mode", choices=("fresh", "risky"), default="fresh")
    parser.add_argument("--risk-band", type=float, nargs=2, default=(0.5, 0.85),
                        help="risky goals: the share of particles allowing the goal")
    parser.add_argument("--num-rollouts", type=int, default=64)
    parser.add_argument("--max-time", type=float, default=600.0)
    parser.add_argument("--jobs", type=int, default=9)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    # One thread per worker, to avoid oversubscription.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, "1")

    out = args.out.resolve()
    from able.tasks.limb.variant import select

    select(args.variant)
    configurations = [
        (planner, None, 1) if planner == "pddlstream" else (f"{planner}_w{n}", planner, n)
        for planner in args.planners
        for n in ([1] if planner in ("pddlstream", "pomcpow") else args.num_waypoints)
    ]
    people = list(args.person_ids or range(args.people))
    context = multiprocessing.get_context("spawn")
    if args.goal_mode == "risky":
        with context.Pool(min(len(people), args.jobs), maxtasksperchild=1) as pool:
            drawn = pool.starmap(draw_risky_fresh, [
                (args.variant, person, args.goals, tuple(args.risk_band)) for person in people
            ])
    else:
        drawn = [draw_tasks(person, args.goals) for person in people]
    jobs = []
    for person, (tasks, nominal_init) in zip(people, drawn):
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
