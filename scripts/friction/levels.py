"""Complexity levels of the PuckPushing (friction) task, and env construction.

Levels change the holes and the goal size on the threeroutes maze:
    L1: no holes, goal radius 0.40 m
    L2: the map's 2 holes, goal radius 0.30 m (the default task)
    L3: 4 holes, goal radius 0.25 m
    L4: 4 holes, goal radius 0.20 m
"""

from __future__ import annotations

import dataclasses

from able.tasks.friction.env.puckpushing_utils import MAPS, THREE_ROUTES, PuckPhysics

# Two more holes: one in the middle of the blue corridor, one at the green
# corridor's exit toward the goal.
EXTRA_HOLES = ((2.6, 3.3), (4.2, 1.2))

LEVELS: dict[str, tuple[tuple[tuple[float, float], ...], float]] = {
    "L1": ((), 0.40),
    "L2": (THREE_ROUTES.holes, 0.30),
    "L3": (THREE_ROUTES.holes + EXTRA_HOLES, 0.25),
    "L4": (THREE_ROUTES.holes + EXTRA_HOLES, 0.20),
}


def register_level(level: str) -> tuple[str, PuckPhysics]:
    """Add the level's map to the env registry; return its name and physics."""
    holes, goal_radius = LEVELS[level]
    name = f"threeroutes_{level}"
    MAPS[name] = dataclasses.replace(THREE_ROUTES, name=name, holes=holes)
    return name, PuckPhysics(goal_radius=goal_radius)


def make_level_env(env_name: str, level: str):
    """The object-centric 2D or 3D env for a level."""
    map_name, physics = register_level(level)
    if env_name == "2d":
        from able.tasks.friction.env.dyn_puckpushing2d import (
            DynPuckPushing2DEnvConfig,
            ObjectCentricDynPuckPushing2DEnv,
        )

        return ObjectCentricDynPuckPushing2DEnv(
            DynPuckPushing2DEnvConfig(map_name=map_name, physics=physics)
        )
    from able.tasks.friction.env.puckpushing3d import (
        ObjectCentricPuckPushing3DEnv,
        PuckPushing3DEnvConfig,
    )

    return ObjectCentricPuckPushing3DEnv(
        PuckPushing3DEnvConfig(map_name=map_name, physics=physics)
    )
