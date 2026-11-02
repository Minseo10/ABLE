"""Limb task: MPC drives a person's limb to a goal within their unknown range of motion."""

from importlib import import_module

# Lazy, so importing the package does not load PyBullet.
_EXPORTS = {
    "LimbBelief": "belief", "LimbPrior": "belief", "RoMGaussian": "belief",
    "LimbDistribution": "distribution",
    "LimbPlan": "planner", "LimbPlanner": "planner",
    "LimbPsi": "psi",
    "LimbGatherer": "query", "LimbQuery": "query",
    "LimbProblem": "task", "LimbState": "task", "LimbTask": "task",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
