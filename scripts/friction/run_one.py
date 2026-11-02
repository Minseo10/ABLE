"""Run one POMCPOW episode on a complexity level and save its log as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from levels import make_level_env  # noqa: E402

from able.planners.pomcpow import POMCPOWConfig  # noqa: E402
from able.tasks.friction.run import run_episode  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["2d", "3d"], required=True)
    parser.add_argument("--level", required=True)
    parser.add_argument("--sims", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--stance", choices=["grid", "continuous", "auto"], default="grid")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    env = make_level_env(args.env, args.level)
    config = POMCPOWConfig(num_sims=args.sims, max_depth=args.max_depth)
    log = run_episode(env, args.env, args.seed, config, stance_mode=args.stance)
    record = asdict(log)
    record.update(level=args.level, sims=args.sims, stance=args.stance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1))
    print(f"{args.env} {args.level} sims={args.sims} seed={args.seed}: {log.outcome} "
          f"pushes={len(log.steps)} plan={log.total_plan_seconds:.1f}s")


if __name__ == "__main__":
    main()
