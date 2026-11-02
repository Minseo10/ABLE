"""Bedside task: PDDLStream stocks a bedside within a person's unknown reach."""

from importlib import import_module

# Lazy, so `python -m able.tasks.bedside.planner.run` does not import run.py twice.
_EXPORTS = {
    "ReachBelief": "belief", "reach_distance": "belief",
    "BedsidePsi": "psi", "PlaceProposal": "psi",
    "ReachGatherer": "query", "bedside_directions": "query",
    "PDDLStreamPlanner": "solver",
    "BedsideProblem": "task", "BedsideState": "task", "BedsideTask": "task", "ReachPrior": "task",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
