"""PDDLStream as an able.api Planner: private working dir, time budget, stream calls as effort."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from pddlstream.algorithms.meta import solve
from pddlstream.language.constants import PDDLProblem, print_solution
from pddlstream.language.generator import from_gen_fn, from_test

from able.api import Belief, Budget, PlanResult


@contextmanager
def private_cwd():
    """Run in a fresh temp dir: Fast Downward writes ./temp in the cwd, so parallel runs would collide."""
    old, tmp = os.getcwd(), tempfile.mkdtemp(prefix="pddlstream_")
    os.chdir(tmp)
    try:
        yield
    finally:
        os.chdir(old)
        shutil.rmtree(tmp, ignore_errors=True)


def counted(fn: Callable, calls: Counter, name: str, generator: bool) -> Callable:
    """`fn` with each evaluation of stream `name` counted in `calls`."""

    def test(*args):
        calls[name] += 1
        return fn(*args)

    def gen(*args):
        outputs = fn(*args)
        while True:
            calls[name] += 1
            try:
                out = next(outputs)
            except StopIteration:
                return
            yield out

    return gen if generator else test


def stream_map(gens: dict[str, Callable], tests: dict[str, Callable], calls: Counter) -> dict:
    """PDDLStream stream map with every evaluation counted."""
    out = {n: from_gen_fn(counted(f, calls, n, True)) for n, f in gens.items()}
    return out | {n: from_test(counted(f, calls, n, False)) for n, f in tests.items()}


def solve_problem(problem: PDDLProblem, max_time: float, algorithm: str = "adaptive",
                  search_sample_ratio: float = 1.0, max_complexity: float = float("inf"),
                  verbose: bool = False):
    """The plan (list of actions) or None."""
    # pddlstream walks its binding tree recursively; long plans exceed the default limit.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    with private_cwd():
        solution = solve(problem, algorithm=algorithm, unit_costs=True, max_time=max_time,
                         search_sample_ratio=search_sample_ratio, max_complexity=max_complexity,
                         verbose=verbose)
    if verbose:
        print_solution(solution)
    return solution[0]


@dataclass
class PDDLStreamPlanner:
    """Subclasses build the problem from (state, belief, psi); this solves it and counts stream calls."""

    max_time: float = 120.0
    algorithm: str = "adaptive"
    effort_unit: str = "stream_calls"

    def problem(self, state: Any, belief: Belief, proposal: Any, rng: np.random.Generator
                ) -> tuple[PDDLProblem, Counter]:
        """(problem, the Counter its streams count into)."""
        raise NotImplementedError

    def trace(self, plan) -> Any:
        """What psi learns from a solved plan."""
        return None

    def plan(self, state: Any, belief: Belief, proposal: Any = None, budget: Budget | None = None,
             rng: np.random.Generator | None = None) -> PlanResult:
        rng = rng if rng is not None else np.random.default_rng(getattr(state, "seed", None))
        max_time = budget.seconds if budget is not None and budget.seconds else self.max_time
        t0 = time.perf_counter()  # problem building (samplers' setup) counts as planning
        problem, calls = self.problem(state, belief, proposal, rng)
        plan = solve_problem(problem, max_time, self.algorithm)
        info = {"calls": dict(calls), "plan_length": None if plan is None else len(plan)}
        return PlanResult(plan, time.perf_counter() - t0, float(sum(calls.values())), self.trace(plan), info)
