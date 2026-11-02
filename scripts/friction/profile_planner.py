"""Profile POMCPOW on PuckPushing: where planning time goes, and how the tree grows.

For each complexity level, runs one episode under cProfile and records, per planning
call, the tree's shape (actions at the root, visits per depth) and how many sampled
pushes the proposer rejected. Saves a JSON summary.

    python scripts/friction/profile_planner.py --sims 1000 --seed 0
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
from pathlib import Path

from able.planners import pomcpow
from able.planners.pomcpow import POMCPOWConfig
from able.tasks.friction import FrictionTask
from able.tasks.friction import model as puck_model
from able.tasks.friction.levels import LEVELS
from able.tasks.friction.run import run

# Planner parts, by the function whose cumulative time they are measured with.
PARTS = {
    "propose (sample feasible push)": ("model.py", "propose"),
    "transition (slide + drive)": ("model.py", "step"),
    "leaf value (greedy rollout)": ("model.py", "value"),
    "observe + likelihood": [("model.py", "observe"), ("model.py", "likelihood")],
    "env step (simulate push)": ("dyn_puckpushing2d.py", "step"),
    "belief update": ("belief.py", "update"),
}
# Inner functions worth reporting on their own (cumulative, overlapping the parts).
INNER = {
    "PuckWorld.slide": ("puckpushing_utils.py", "slide"),
    "PuckWorld.push_pose": ("puckpushing_utils.py", "push_pose"),
    "PuckWorld.drive_distance": ("puckpushing_utils.py", "drive_distance"),
    "rollout: path_to_goal": ("model.py", "path_to_goal"),
    "rollout: speed_to_travel": ("model.py", "speed_to_travel"),
    "rollout: puck_path_clear": ("puckpushing_utils.py", "puck_path_clear"),
    "POMCPOW.plan": ("pomcpow.py", "plan"),
}


def cumulative(stats: pstats.Stats, spec) -> float:
    """Cumulative seconds of the functions matching (file suffix, name) specs."""
    specs = spec if isinstance(spec, list) else [spec]
    total = 0.0
    for (filename, _, name), (_, _, _, cum, _) in stats.stats.items():  # type: ignore
        for suffix, fn in specs:
            if filename.endswith(suffix) and name == fn:
                total += cum
    return total


def tree_stats(root) -> dict:
    """Root fan-out and how many simulations continued at each depth."""
    visits_by_depth: list[int] = []
    nodes_by_depth: list[int] = []
    frontier = [root]
    while frontier:
        visits_by_depth.append(sum(n.visits for n in frontier))
        nodes_by_depth.append(len(frontier))
        frontier = [
            child
            for node in frontier
            for action in node.actions.values()
            for child in action.children.values()
        ]
    root_visits = [a.visits for a in root.actions.values()]
    return {
        "root_actions": len(root.actions),
        "root_visits_mean": float(sum(root_visits) / max(len(root_visits), 1)),
        "root_visits_max": int(max(root_visits, default=0)),
        "obs_nodes_by_depth": nodes_by_depth,
        "decisions_by_depth": visits_by_depth,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", default="2d")
    parser.add_argument(
        "--out", type=Path, default=Path("data/friction/complexity/profile.json")
    )
    args = parser.parse_args()

    # Record the tree after every planning call, and proposal rejections.
    trees: list[dict] = []
    original_plan = pomcpow.POMCPOW.plan

    def plan(self, particles):
        result = original_plan(self, particles)
        trees.append(tree_stats(self.root))
        return result

    pomcpow.POMCPOW.plan = plan  # type: ignore[method-assign]
    counts = {"propose_calls": 0, "feasibility_checks": 0, "proposals_none": 0}
    original_propose = puck_model.PuckPushingModel.propose

    def propose(self, state, tried, rng):
        counts["propose_calls"] += 1
        check = self.world.push_feasible

        def counted(*args):
            counts["feasibility_checks"] += 1
            return check(*args)

        self.world.push_feasible = counted
        try:
            push = original_propose(self, state, tried, rng)
        finally:
            del self.world.push_feasible
        counts["proposals_none"] += push is None
        return push

    puck_model.PuckPushingModel.propose = propose  # type: ignore[method-assign]

    results = {}
    for level in LEVELS:
        trees.clear()
        for key in counts:
            counts[key] = 0
        task = FrictionTask(env=args.env, level=level, config=POMCPOWConfig(num_sims=args.sims))
        task.environment  # built outside the profile
        profiler = cProfile.Profile()
        profiler.enable()
        log = run(task, args.seed)
        profiler.disable()
        stats = pstats.Stats(profiler)
        total = sum(v[2] for v in stats.stats.values())  # type: ignore[attr-defined]
        plan_time = cumulative(stats, INNER["POMCPOW.plan"])
        parts = {name: cumulative(stats, spec) for name, spec in PARTS.items()}
        inside = sum(
            parts[k]
            for k in (
                "propose (sample feasible push)",
                "transition (slide + drive)",
                "leaf value (greedy rollout)",
                "observe + likelihood",
            )
        )
        parts["tree bookkeeping (POMCPOW itself)"] = max(plan_time - inside, 0.0)
        results[level] = {
            "outcome": log.outcome,
            "pushes": len(log.steps),
            "profiled_seconds": total,
            "plan_seconds": plan_time,
            "parts": parts,
            "inner": {name: cumulative(stats, spec) for name, spec in INNER.items()},
            "calls": {
                "slide": sum(
                    v[1]
                    for k, v in stats.stats.items()  # type: ignore[attr-defined]
                    if k[0].endswith("puckpushing_utils.py") and k[2] == "slide"
                ),
                **counts,
            },
            "trees": list(trees),
        }
        print(level, log.outcome, f"plan={plan_time:.1f}s", {k: round(v, 1) for k, v in parts.items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
