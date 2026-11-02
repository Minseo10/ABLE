"""A hard wall-clock limit on planning, whatever code is running."""

from __future__ import annotations

import signal
from contextlib import contextmanager


class PlanningTimeout(BaseException):
    """Planning deadline hit; a `BaseException` so streams' `except Exception` can't swallow it."""


@contextmanager
def hard_deadline(seconds: float | None):
    """Raise `PlanningTimeout` after `seconds` wall-clock (SIGALRM, main thread); None/<=0 = no limit."""
    if seconds is None or seconds <= 0:
        yield
        return

    def fire(signum, frame):  # pylint: disable=unused-argument
        raise PlanningTimeout(f"hit the {seconds:g} s hard planning time limit")

    previous = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
