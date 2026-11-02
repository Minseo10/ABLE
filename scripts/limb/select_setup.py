"""Pick a limb variant's fixed grasp and base: stage A ranks pairs by IK reach, stage B by plain MPC.

    python scripts/limb/select_setup.py --variant human-left-arm --out data/limb/setup
    python scripts/limb/select_setup.py --merge data/limb/setup   # -> src/able/tasks/limb/planning/fixed_setup.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np


def candidates(ctx, max_bases: int):
    """(offset, roll, grasp, base) tuples, up to `max_bases` approachable bases per grasp."""
    from able.tasks.limb.planning.setup import (
        GRASP_OFFSETS,
        GRASP_ROLLS,
        approach,
        wrist_grasp,
    )
    from able.tasks.limb.planning.utils import base_candidates

    init = list(ctx.sim.limb.get_joint_positions())
    out = []
    for offset, roll in itertools.product(GRASP_OFFSETS, GRASP_ROLLS):
        grasp = wrist_grasp(ctx, offset, roll)
        kept = 0
        for base in itertools.islice(base_candidates(ctx, grasp, init), 4 * max_bases):
            if approach(ctx, grasp, base, init) is not None:
                out.append((offset, roll, grasp, base))
                kept += 1
                if kept >= max_bases:
                    break
    return out


def draw_tasks(sampler, rng, n: int):
    """(init, goal) pairs for people drawn from the prior."""
    from kinder.envs.dynamic3d.limb_utils import (
        LIMB_JOINT_INFOS,
        joint_position_distance,
    )

    from able.tasks.limb.env.human import DEFAULT_PRIOR

    dist, tasks = sampler.box, []
    while len(tasks) < n:
        low, high = dist.inner_range(DEFAULT_PRIOR.sample(rng))
        init, goal = sampler._draw_limb_conf(rng, low, high), sampler._draw_limb_conf(rng, low, high)
        if dist.min_separation <= joint_position_distance(LIMB_JOINT_INFOS, list(init), list(goal)) <= dist.max_separation:
            tasks.append((init, goal))
    return tasks


def reach_score(sampler, grasp, base, tasks) -> float:
    sampler.grasp, sampler.base = grasp, base
    return float(np.mean([sampler._base_pose_for(i) is not None and sampler._reachable_from(base, i, g)
                          for i, g in tasks]))


def mpc_score(variant, offset, roll, base, n: int, max_time: float, seed: int) -> dict:
    """Plain MPC (prior belief) on `n` tasks reachable from the pair."""
    from able.api import Budget, run_episode
    from able.tasks.limb.planning.setup import FixedSetup
    from able.tasks.limb.task import LimbTask

    task = LimbTask(variant, setup=FixedSetup(variant, offset, roll, (base.x, base.y, base.rot)))
    planner, rng = task.make_planner(), np.random.default_rng(seed)
    runs = []
    for _ in range(n):
        try:
            problem = task.sample_problem(rng)
        except RuntimeError:
            runs.append({"success": False, "seconds": max_time, "failure": "no reachable task"})
            continue
        theta = task.prior.sample(rng)
        ep = run_episode(task, planner, problem, theta, task.prior.belief(rng), rng, budget=Budget(max_time))
        last = ep.steps[-1]
        runs.append({"success": ep.success, "seconds": ep.plan_seconds, "rollouts": ep.effort,
                     "failure": last.result.info.get("failure") or (last.step.info.get("failure") if last.step else "")})
    return {**mpc_stats(runs), "runs": runs}


def mpc_stats(runs: list[dict]) -> dict:
    """Success and time over runs that reached the MPC (not task-sampling misses)."""
    tried = [r for r in runs if r["failure"] != "no reachable task"] or [{"success": False, "seconds": float("inf")}]
    return {"success": float(np.mean([r["success"] for r in tried])),
            "seconds": float(np.mean([r["seconds"] for r in tried])), "tried": len(tried)}


def select(args) -> dict:
    from able.tasks.limb.distribution import ReachableTaskSampler
    from able.tasks.limb.env.sim import make_sampling_context
    from able.tasks.limb.env.variant import select as select_variant
    from able.tasks.limb.planning.run import reset_to_start

    select_variant(args.variant)
    os.chdir(tempfile.mkdtemp(prefix="select-setup-"))
    sim, ctx = make_sampling_context()
    reset_to_start(ctx)
    started = time.time()
    pairs = candidates(ctx, args.bases)
    sampler = ReachableTaskSampler(ctx)
    tasks = draw_tasks(sampler, np.random.default_rng(args.seed), args.tasks)
    scored = sorted(((reach_score(sampler, g, b, tasks), o, r, b) for o, r, g, b in pairs),
                    key=lambda e: -e[0])
    sim.close()
    report = {"variant": args.variant, "pairs": len(pairs), "stage_a": [
        {"reach": s, "offset": o, "roll": r, "base": [b.x, b.y, b.rot]} for s, o, r, b in scored]}
    best = None
    for reach, offset, roll, base in scored[: args.top]:
        if reach == 0.0:
            break
        mpc = mpc_score(args.variant, offset, roll, base, args.mpc_tasks, args.max_time, args.seed)
        entry = {"offset": offset, "roll": roll, "base": [base.x, base.y, base.rot], "reach": reach, **mpc}
        report.setdefault("stage_b", []).append(entry)
        key = (mpc["success"], reach, -mpc["seconds"])
        if best is None or key > best[0]:
            best = (key, entry)
    report["seconds"] = time.time() - started
    if best is not None:
        e = best[1]
        report["chosen"] = {"variant": args.variant, "offset": e["offset"], "roll": e["roll"], "base": e["base"],
                            "stats": {"reach": e["reach"], "mpc_success": e["success"], "mpc_seconds": e["seconds"],
                                      "mpc_tasks": len(e["runs"])}}
    return report


def merge(directory: Path) -> None:
    from able.tasks.limb.planning.setup import FixedSetup, save_setup

    for path in sorted(directory.glob("*.json")):
        report = json.loads(path.read_text())
        best = None
        for e in report.get("stage_b", []):
            e.update(mpc_stats(e["runs"]))  # rescore reports from before mpc_stats
            key = (e["success"], e["reach"], -e["seconds"])
            if best is None or key > best[0]:
                best = (key, e)
        chosen = None if best is None else {
            "variant": report["variant"], "offset": best[1]["offset"], "roll": best[1]["roll"],
            "base": best[1]["base"], "stats": {"reach": best[1]["reach"], "mpc_success": best[1]["success"],
                                               "mpc_seconds": best[1]["seconds"], "mpc_tried": best[1]["tried"]}}
        if chosen is None:
            print(f"{path.stem}: no usable pair")
            continue
        save_setup(FixedSetup(**{**chosen, "base": tuple(chosen["base"])}))
        print(f"{path.stem}: {chosen['stats']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant")
    parser.add_argument("--out", type=Path, default=Path("data/limb/setup"))
    parser.add_argument("--merge", type=Path)
    parser.add_argument("--bases", type=int, default=4, help="bases kept per grasp")
    parser.add_argument("--tasks", type=int, default=40, help="stage A tasks")
    parser.add_argument("--top", type=int, default=3, help="pairs given to stage B")
    parser.add_argument("--mpc-tasks", type=int, default=4)
    parser.add_argument("--max-time", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.merge:
        merge(args.merge)
        return
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = select(args)
    (out / f"{args.variant}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report.get("chosen"), indent=1))


if __name__ == "__main__":
    main()
