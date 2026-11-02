"""Run and log one friction-task episode with POMCPOW (2D PyMunk or 3D PyBullet env).

Example:
    python scripts/friction/run_episode.py --env 2d --level L2 --seed 0 --sims 2000
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from able.planners.pomcpow import POMCPOWConfig
from able.tasks.friction import FrictionTask
from able.tasks.friction.env.episode import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=["2d", "3d"], default="2d")
    parser.add_argument("--level", default="L2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sims", type=int, default=1000)
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-pushes", type=int, default=15)
    parser.add_argument("--particles", type=int, default=500)
    parser.add_argument("--probes", type=int, default=0)
    parser.add_argument("--frictions", type=float, nargs="+", default=None)
    parser.add_argument("--stance", choices=["grid", "continuous", "auto"], default="grid")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    config = POMCPOWConfig(num_sims=args.sims, max_time=args.max_time, max_depth=args.max_depth)
    task = FrictionTask(env=args.env, level=args.level, num_particles=args.particles, max_pushes=args.max_pushes,
                        stance_mode=args.stance, config=config)
    start = time.perf_counter()
    log = run(task, args.seed, tuple(args.frictions) if args.frictions else None, probes=args.probes)
    for step in log.steps:
        means = {k: round(v[0], 3) for k, v in step.belief.items()}
        print(f"push angle={step.angle:+.2f} speed={step.speed:.2f} stop=({step.stop[0]:.2f},{step.stop[1]:.2f}) "
              f"reward={step.reward:.1f} plan={step.plan_seconds:.1f}s belief={means}")
    print(f"{log.outcome}: reward={log.total_reward:.1f} pushes={len(log.steps)} plan={log.total_plan_seconds:.1f}s "
          f"wall={time.perf_counter() - start:.1f}s true={np.round(log.true_frictions, 3).tolist()}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(asdict(log), indent=1))


if __name__ == "__main__":
    main()
