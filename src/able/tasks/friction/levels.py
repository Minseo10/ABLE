"""Complexity levels of the friction task on the threeroutes maze.

    L1: no holes, goal radius 0.40 m
    L2: the map's 2 holes, goal radius 0.30 m (the default task)
    L3: 4 holes, goal radius 0.25 m
    L4: 4 holes, goal radius 0.20 m
"""

from __future__ import annotations

import dataclasses

from able.tasks.friction.env.puckpushing_utils import (
    MAPS,
    THREE_ROUTES,
    PuckPhysics,
    PuckWorld,
)

# Two more holes: mid blue corridor, and the green corridor's exit toward the goal.
EXTRA_HOLES = ((2.6, 3.3), (4.2, 1.2))

LEVELS: dict[str, tuple[tuple[tuple[float, float], ...], float]] = {
    "L1": ((), 0.40),
    "L2": (THREE_ROUTES.holes, 0.30),
    "L3": (THREE_ROUTES.holes + EXTRA_HOLES, 0.25),
    "L4": (THREE_ROUTES.holes + EXTRA_HOLES, 0.20),
}


def level_world(level: str) -> PuckWorld:
    """The level's map and physics."""
    holes, goal_radius = LEVELS[level]
    puck_map = dataclasses.replace(THREE_ROUTES, name=f"threeroutes_{level}", holes=holes)
    return PuckWorld(puck_map, PuckPhysics(goal_radius=goal_radius))


def register_level(level: str) -> tuple[str, PuckPhysics]:
    """Add the level's map to the env registry; return its name and physics."""
    world = level_world(level)
    MAPS[world.puck_map.name] = world.puck_map
    return world.puck_map.name, world.physics


def make_level_env(env_name: str, level: str):
    """The object-centric 2D or 3D env for a level."""
    map_name, physics = register_level(level)
    if env_name == "2d":
        from able.tasks.friction.env.dyn_puckpushing2d import (
            DynPuckPushing2DEnvConfig,
            ObjectCentricDynPuckPushing2DEnv,
        )

        return ObjectCentricDynPuckPushing2DEnv(DynPuckPushing2DEnvConfig(map_name=map_name, physics=physics))
    if env_name == "3d":
        from able.tasks.friction.env.puckpushing3d import (
            ObjectCentricPuckPushing3DEnv,
            PuckPushing3DEnvConfig,
        )

        return ObjectCentricPuckPushing3DEnv(PuckPushing3DEnvConfig(map_name=map_name, physics=physics))
    raise ValueError(f"Unknown env {env_name!r}; expected '2d' or '3d'")
