"""Frame capture in FrictionTask.execute."""

import numpy as np
import pytest

from able.api import Step
from able.tasks.friction import FrictionTask


def _push(env: str, frames: list | None, fps: float = 2.0) -> Step:
    """One proposed push from the start; frames at `fps` if `frames` is a list."""
    task = FrictionTask(env=env, num_particles=20, max_pushes=2)
    task.recorder.fps = fps
    rng = np.random.default_rng(0)
    state = task.reset(task.sample_problem(rng), np.array([0.12, 0.08, 0.03]), rng)
    push = task.model.propose(task.model.initial_state(state.puck, state.robot, (0.1, 0.1, 0.1)), [], rng)
    return task.execute(state, push, rng, frames=frames)


def _check(env: str) -> None:
    frames: list = []
    recorded, plain = _push(env, frames), _push(env, None)
    assert len(frames) >= 3
    assert all(f.dtype == np.uint8 and f.ndim == 3 and f.shape[2] == 3 for f in frames)
    assert len({f.shape for f in frames}) == 1
    assert isinstance(plain, Step) and type(recorded.state) is type(plain.state)
    assert recorded.info.keys() == plain.info.keys()
    assert recorded.info["stop"] == pytest.approx(plain.info["stop"])  # capture does not perturb the sim


def test_2d_execute_records_frames():
    _check("2d")


@pytest.mark.slow
def test_3d_execute_records_frames():
    _check("3d")
