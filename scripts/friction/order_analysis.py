"""Slot shot vs detour in the PuckPushing sweep, split by whether the slot floor
(blue) drew the slippery friction level.

Reads data/friction/complexity/*_seed*.json and prints, per env and budget: success,
how often the planner sent the puck into the slot, how often that worked, and what the
belief said about the blue floor's friction when it did.

    python scripts/friction/order_analysis.py
"""

from __future__ import annotations

import collections
import glob
import json
import math

import numpy as np

DATA = "data/friction/complexity"
# The slot between the central walls, and the floor before its mouth.
SLOT_X = (2.0, 3.2)
SLOT_Y = (2.0, 2.4)
PUCK_RADIUS = 0.12


def in_slot(point) -> bool:
    """Whether the puck's center is inside the slot."""
    x, y = point
    return SLOT_X[0] - PUCK_RADIUS < x < SLOT_X[1] + PUCK_RADIUS and (
        SLOT_Y[0] < y < SLOT_Y[1]
    )


def shot_into_slot(start, angle: float, stop) -> bool:
    """Whether a push from `start` along `angle` sent the puck into the slot: it
    starts left of the slot and its heading line crosses the slot's middle."""
    if start[0] >= SLOT_X[0] or math.cos(angle) <= 0:
        return False
    mid = (SLOT_X[0] + SLOT_X[1]) / 2
    y = start[1] + math.tan(angle) * (mid - start[0])
    return SLOT_Y[0] < y < SLOT_Y[1] and stop[0] > SLOT_X[0] - PUCK_RADIUS


def _blue(belief: dict) -> list[float]:
    """The blue floor's (mean, std); logs from before the rename call it laminate."""
    return belief["blue"] if "blue" in belief else belief["laminate"]


def episode_summary(run: dict) -> dict:
    """Ordering, slot use and the belief at the slot shot for one episode."""
    frictions = run["true_frictions"]
    slippery_slot = int(np.argmin(frictions)) == 0  # blue is surface 0
    puck = [1.2, 2.2]  # puck start of the threeroutes map
    belief = _blue(run["prior_belief"])
    shot = None
    for k, step in enumerate(run["steps"]):
        if step["feasible"] and shot_into_slot(puck, step["angle"], step["stop"]):
            shot = {"push": k + 1, "blue_mean": belief[0], "blue_std": belief[1]}
            break
        belief = _blue(step["belief"])
        puck = step["stop"]
    final = run["steps"][-1]["stop"] if run["steps"] else puck
    return {
        "slippery_slot": slippery_slot,
        "shot": shot,
        "goal": run["outcome"] == "goal",
        "stuck_in_slot": run["outcome"] == "stuck" and in_slot(final),
        "pushes": len(run["steps"]),
        "outcome": run["outcome"],
    }


def main() -> None:
    groups: dict = collections.defaultdict(list)
    for path in glob.glob(f"{DATA}/*_seed*.json"):
        with open(path, encoding="utf-8") as f:
            run = json.load(f)
        groups[(run["env"], run["sims"], run["level"])].append(episode_summary(run))
    print(
        "slot = blue drew the slippery level (1/3 of episodes); "
        "shot = the planner pushed the puck into the slot"
    )
    for env in ("2d", "3d"):
        for sims in (300, 1000, 3000):
            rows = [r for (e, s, _), rs in groups.items() if e == env and s == sims for r in rs]
            if not rows:
                continue
            print(f"\n{env.upper()} {sims} sims ({len(rows)} episodes, all levels)")
            for slippery in (True, False):
                part = [r for r in rows if r["slippery_slot"] == slippery]
                if not part:
                    continue
                shots = [r for r in part if r["shot"]]
                good = [r for r in shots if r["goal"]]
                label = "slippery slot" if slippery else "sticky slot  "
                line = (
                    f"  {label}: n={len(part):3d} SR {np.mean([r['goal'] for r in part]):.2f}"
                    f" | shot {len(shots):2d} ({len(shots) / len(part):.0%}), "
                    f"reached goal after shot {len(good):2d}, "
                    f"stuck in slot {sum(r['stuck_in_slot'] for r in part):2d}"
                    f" | pushes (successes) "
                    f"{np.mean([r['pushes'] for r in part if r['goal']] or [np.nan]):.1f}"
                )
                if shots:
                    line += (
                        f" | at the shot: push {np.mean([r['shot']['push'] for r in shots]):.1f},"
                        f" blue belief {np.mean([r['shot']['blue_mean'] for r in shots]):.3f}"
                        f" +- {np.mean([r['shot']['blue_std'] for r in shots]):.3f}"
                    )
                print(line)
            outcomes = collections.Counter(r["outcome"] for r in rows)
            print(f"  outcomes: {dict(outcomes)}")


if __name__ == "__main__":
    main()
