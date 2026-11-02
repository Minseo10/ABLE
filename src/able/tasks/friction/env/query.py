"""Probe pushes in their own simulator: same env class, map and physics as the task, never the task's env."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from able.tasks.friction.env.execute import execute_push, push_record
from able.tasks.friction.env.puckpushing_utils import PuckWorld, Push
from able.tasks.friction.env.push_recorder import Recorder

BACKOFF = 0.05  # robot start behind its push station (m)


class FrictionQuerySimulator:
    """Pushes the puck from a given start in a probe env, built lazily."""

    def __init__(self, world: PuckWorld, make_env: Callable[[], Any], env_name: str = "2d",
                 header: str = "") -> None:
        self.world, self.make_env, self.env_name, self.header = world, make_env, env_name, header
        self._env: Any = None
        self._recorder: Recorder | None = None

    @property
    def environment(self) -> Any:
        if self._env is None:
            self._env = self.make_env()
        return self._env

    def ask(self, theta: NDArray[np.float64], puck, angle: float, speed: float, rng: np.random.Generator,
            frames: list | None = None) -> dict:
        """The push record (release speed, stop, hole, ...) of a probe at the true frictions `theta`."""
        env, start, push = self.environment, np.array(puck, np.float64), Push(angle, speed)
        pose = self.world.push_pose(start, angle)
        assert pose is not None, (puck, angle)
        base, heading = pose
        robot = base - BACKOFF * push.direction
        env.reset(seed=int(rng.integers(2**31)), options={
            "frictions": tuple(float(f) for f in theta), "puck": tuple(puck),
            "robot": (float(robot[0]), float(robot[1]), float(heading))})
        if frames is not None and self._recorder is None:
            self._recorder = Recorder(env, self.env_name, header=f"{self.header} probe".strip())
        _, info = execute_push(env, np.array([angle, speed]), self._recorder, frames, 0)
        return push_record(info, start, push)
