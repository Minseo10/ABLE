"""Solve BedsideStocking3D with PDDLStream under prior particles (or the true theta), optionally execute.

Usage: python scripts/bedside/solve.py --num-requested 4 --num-clutter 2 --seed 0 [--execute] [--json-out f.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from able.tasks.bedside.env.bedside_stocking3d import (
    BedsideStocking3DEnvConfig,
    ObjectCentricBedsideStocking3DEnv,
)
from able.tasks.bedside.env.execute import execute_plan
from able.tasks.bedside.planning.problem import build_stream_context, plan_bedside


def solve_and_execute(
    num_requested: int = 4,
    num_clutter: int = 2,
    seed: int = 0,
    max_time: float = 600.0,
    delta: float = 0.1,
    search_sample_ratio: float = 1.0,
    max_complexity: float = float("inf"),
    execute: bool = False,
    use_gui: bool = False,
    verbose: bool = False,
    true_theta: bool = False,
    num_particles: int = 256,
) -> dict:
    """Plan under prior particles (or the true theta), optionally execute."""
    sim = ObjectCentricBedsideStocking3DEnv(
        num_requested=num_requested,
        num_clutter=num_clutter,
        config=BedsideStocking3DEnvConfig(),
        use_gui=use_gui,
        allow_state_access=True,
    )
    state, _ = sim.reset(seed=seed)
    cfg = sim.config
    particles = None if true_theta else np.random.default_rng(seed).uniform(
        cfg.reach_theta_lb, cfg.reach_theta_ub, (num_particles, 3))
    ctx = build_stream_context(sim, state, theta_particles=particles, delta=delta, seed=seed)
    t0 = time.perf_counter()
    plan = plan_bedside(
        sim,
        state,
        max_time=max_time,
        search_sample_ratio=search_sample_ratio,
        max_complexity=max_complexity,
        verbose=verbose,
        ctx=ctx,
    )
    plan_time = time.perf_counter() - t0
    result = {
        "num_requested": num_requested,
        "num_clutter": num_clutter,
        "seed": seed,
        "search_sample_ratio": search_sample_ratio,
        "max_complexity": max_complexity,
        "plan_found": plan is not None,
        "true_theta": true_theta,
        "plan_time": plan_time,
        "stream_calls": sum(ctx.calls.values()),
        "plan_length": None if plan is None else len(plan),
        "num_moves": None if plan is None else sum(a[0] == "place" for a in plan),
        "executed": None,
    }
    if plan is not None and execute:
        sim.reset(seed=seed)
        result["executed"] = bool(execute_plan(sim, plan))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-requested", type=int, default=4)
    parser.add_argument("--num-clutter", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-time", type=float, default=600.0)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--search-sample-ratio", type=float, default=1.0)
    parser.add_argument("--max-complexity", type=float, default=float("inf"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--use-gui", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--true-theta", action="store_true", help="plan with the true theta")
    parser.add_argument("--num-particles", type=int, default=256, help="prior particles")
    args = parser.parse_args()
    result = solve_and_execute(
        num_requested=args.num_requested,
        num_clutter=args.num_clutter,
        seed=args.seed,
        max_time=args.max_time,
        delta=args.delta,
        search_sample_ratio=args.search_sample_ratio,
        max_complexity=args.max_complexity,
        execute=args.execute,
        use_gui=args.use_gui,
        verbose=args.verbose,
        true_theta=args.true_theta,
        num_particles=args.num_particles,
    )
    print(json.dumps(result))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
