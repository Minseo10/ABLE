"""Per-process state a parallel sweep needs: a working directory and a task sampler."""

from __future__ import annotations

import os
import tempfile

from able.tasks.limb.tasks import ReachableTaskSampler

_WORKER_DIRECTORY: str | None = None
_TASK_SAMPLER: ReachableTaskSampler | None = None
_SAMPLING_SIM = None


def isolate_worker() -> None:
    """Give this process its own working directory.

    PDDLStream writes `temp/` and `statistics/` relative to the working directory, and
    workers sharing one race on those and lose episodes to `FileExistsError`.
    """
    global _WORKER_DIRECTORY  # pylint: disable=global-statement
    if _WORKER_DIRECTORY is None:
        _WORKER_DIRECTORY = tempfile.mkdtemp(prefix=f"limb-worker-{os.getpid()}-")
        os.chdir(_WORKER_DIRECTORY)


def task_sampler() -> ReachableTaskSampler:
    """One reachability-testing environment per worker, built on first use."""
    global _TASK_SAMPLER, _SAMPLING_SIM  # pylint: disable=global-statement
    if _TASK_SAMPLER is None:
        from able.tasks.limb.planner import make_sampling_context

        _SAMPLING_SIM, ctx = make_sampling_context()
        _TASK_SAMPLER = ReachableTaskSampler(ctx)
    return _TASK_SAMPLER
