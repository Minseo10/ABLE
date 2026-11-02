"""Gymnasium ids for the envs that live in this repo."""

import gymnasium

_ENVS = {
    "able/DynPuckPushing2D-threeroutes-v0": (
        "able.tasks.friction.env.dyn_puckpushing2d:DynPuckPushing2DEnv", {"map_name": "threeroutes"}),
    "able/PuckPushing3D-threeroutes-v0": (
        "able.tasks.friction.env.puckpushing3d:PuckPushing3DEnv", {"map_name": "threeroutes"}),
    **{f"able/BedsideStocking3D-r{r}-c{c}-v0": (
        "able.tasks.bedside.env:BedsideStocking3DEnv", {"num_requested": r, "num_clutter": c})
       for r in (2, 4, 6, 8) for c in (0, 2, 4, 6)},
}


def register_all() -> None:
    """Register every env id above once."""
    for env_id, (entry_point, kwargs) in _ENVS.items():
        if env_id not in gymnasium.registry:
            gymnasium.register(env_id, entry_point=entry_point, kwargs=kwargs)
