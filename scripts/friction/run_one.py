"""Run one POMCPOW episode on a complexity level and save its log as JSON."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from run_episode import run

from able.planners.pomcpow import POMCPOWConfig
from able.tasks.friction import FrictionTask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["2d", "3d"], required=True)
    parser.add_argument("--level", required=True)
    parser.add_argument("--sims", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--stance", choices=["grid", "continuous", "auto"], default="grid")
    parser.add_argument("--probes", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    task = FrictionTask(env=args.env, level=args.level, stance_mode=args.stance,
                        config=POMCPOWConfig(num_sims=args.sims, max_depth=args.max_depth))
    log = run(task, args.seed, probes=args.probes)
    record = asdict(log)
    record.update(level=args.level, sims=args.sims, stance=args.stance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1))
    print(f"{args.env} {args.level} sims={args.sims} seed={args.seed}: {log.outcome} "
          f"pushes={len(log.steps)} plan={log.total_plan_seconds:.1f}s")


if __name__ == "__main__":
    main()
