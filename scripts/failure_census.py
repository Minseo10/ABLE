"""Tally failed stream calls by whether the limb alone, or the robot, could have caught them.

Reads the `--log-path` JSONs of a sweep, one directory per run.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# First match wins; "replay" marks failures that only the open-loop replay found.
KINDS = [
    ("human", r"person's torque|anatomical limit|into the person or the furniture|limb's joint limits"),
    ("robot", r"robot for more torque|robot's torque|arm cannot reach|no arm path|singular|cannot hold|base collides"),
    ("search", r"diverged|control steps|outside the tolerance|already at the target"),
]


def kind(reason: str) -> str:
    """The side a failure reason belongs to."""
    for name, pattern in KINDS:
        if re.search(pattern, reason):
            return f"{name} (replay)" if reason.startswith("replayed") else name
    return f"other: {reason}"


def census(run: Path) -> dict:
    """Counts and seconds per failure kind for one run."""
    log = json.loads((run / "log.json").read_text())
    tally: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    rejected: dict[str, int] = defaultdict(int)
    for call in log["calls"]:
        for *_, reason in call.get("rejected", []):
            rejected[kind(reason)] += 1
        if "failure" in call:
            key = f"{call['stream']}: {kind(call['failure'])}"
        elif call["stream"] == "plan-limb-motion" and call["outputs"]:
            key = "plan-limb-motion: success"
        else:
            continue
        tally[key][0] += 1
        tally[key][1] += call["seconds"]
    logged = sum(call["seconds"] for call in log["calls"])
    tally["unlogged (interrupted call, search overhead)"] = [
        0,
        max(log["result"]["plan_time"] - logged, 0.0),
    ]
    return {"result": log["result"], "tally": tally, "rejected": rejected}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", type=Path)
    args = parser.parse_args()

    by_variant: dict[str, list[dict]] = defaultdict(list)
    for run in sorted(args.sweep.glob("*/")):
        if (run / "log.json").exists():
            by_variant[run.name.split("-g")[0]].append(census(run))
        else:
            print(f"no log: {run.name}")

    for variant, runs in by_variant.items():
        reached = sum(r["result"]["goal_reached"] for r in runs)
        plan_time = sum(r["result"]["plan_time"] for r in runs)
        print(f"\n== {variant}: {reached}/{len(runs)} reached, {plan_time:.0f} s planning")
        tally: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
        rejected: dict[str, int] = defaultdict(int)
        for r in runs:
            for key, (count, seconds) in r["tally"].items():
                tally[key][0] += count
                tally[key][1] += seconds
            for key, count in r["rejected"].items():
                rejected[key] += count
        for key, (count, seconds) in sorted(tally.items(), key=lambda kv: -kv[1][1]):
            print(f"  {key:60s} {count:4d} calls {seconds:7.0f} s {seconds / plan_time:5.1%}")
        print(f"  base candidates rejected: {dict(rejected)}")
        for r in runs:
            if not r["result"]["goal_reached"]:
                print(f"  run failure: {r['result']['failure_reason']}")


if __name__ == "__main__":
    main()
