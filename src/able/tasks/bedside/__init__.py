"""Bedside task: PDDLStream stocks a bedside within a person's unknown reach."""

from importlib import import_module

# Lazy, so importing the package does not load PyBullet and PDDLStream.
_EXPORTS = {
    "BedsideBelief": "belief", "BedsidePrior": "belief", "reach_distance": "belief",
    "BedsideDistribution": "distribution",
    "BedsidePlanner": "planner",
    "BedsidePsi": "psi", "PlaceProposal": "psi",
    "BedsideGatherer": "query", "BedsideQuery": "query", "bedside_directions": "query",
    "BedsideProblem": "task", "BedsideState": "task", "BedsideTask": "task",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
