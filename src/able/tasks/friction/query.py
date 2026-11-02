"""Info gathering: a real probe push on one floor colour, in the gatherer's own simulator.

Probe poses are searched per map (`probe_queries`): a puck start and heading whose straight run
(3-sigma heading cone, clear of walls, holes by 0.1 m and other floors) fits the paddle's
acceleration plus the slowest-friction slide at a 3-sigma fast strike; where no run fits, the
longest one is used at a lower speed. Threeroutes L2 (lowest friction 0.024):
    blue:  puck (0.95, 2.85), heading  20 deg, 0.80 m/s
    tan:   puck (3.45, 1.65), heading  30 deg, 0.70 m/s (longest run 1.76 m)
    green: puck (2.85, 0.65), heading  10 deg, 0.80 m/s (right of the (2.6, 1.0) hole)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from able.api import InfoGatherer, Observation
from able.tasks.friction.env.puckpushing_utils import (
    GRAVITY,
    PuckMap,
    PuckPhysics,
    PuckWorld,
    Push,
)
from able.tasks.friction.env.push_recorder import Recorder

_SIGMAS = 3.0  # strike noise a probe's run must cover
_BACKOFF = 0.05  # robot start behind its push station (m)


@dataclass(frozen=True)
class FrictionQuery:
    """Probe push on floor `floor`: puck placed at `puck`, pushed along `angle` at `speed` (auto stance)."""

    floor: int
    puck: tuple[float, float]
    angle: float
    speed: float

    @property
    def push(self) -> Push:
        return Push(self.angle, self.speed)


def push_record(info: dict, puck, push: Push) -> dict:
    """An env step's info as the record the belief explains."""
    return {
        "puck": [float(v) for v in puck], "angle": push.angle, "speed": push.speed, "stance": push.stance,
        "feasible": bool(info["push_feasible"]), "release_speed": float(info["release_speed"]),
        "stop": np.asarray(info["stop_position"], np.float64).tolist(), "fell_in_hole": bool(info["fell_in_hole"]),
        "goal_reached": bool(info["goal_reached"]), "drive_distance": float(info["drive_distance"]),
    }


def probe_queries(world: PuckWorld, low, speed: float = 0.8, margin: float = 0.1) -> tuple[FrictionQuery, ...]:
    """One probe per floor; `low` is the lowest possible friction per floor."""
    return _probe_queries(world.puck_map, world.physics, tuple(float(v) for v in low), speed, margin)


@lru_cache(maxsize=16)
def _probe_queries(puck_map: PuckMap, physics: PuckPhysics, low: tuple[float, ...], speed: float,
                   margin: float) -> tuple[FrictionQuery, ...]:
    world = PuckWorld(puck_map, physics)
    queries = []
    fast = 1.0 + _SIGMAS * physics.speed_noise
    for floor, mu in enumerate(low):
        # Worst case: fastest strike on the slipperiest floor, after the paddle's acceleration.
        need = physics.accel_distance + (speed * fast) ** 2 / (2.0 * mu * GRAVITY) + margin
        puck, angle, run = _longest_run(world, floor, need)
        room = run - physics.accel_distance - margin
        fit = math.sqrt(max(room, 0.0) * 2.0 * mu * GRAVITY) / fast
        queries.append(FrictionQuery(floor, puck, angle, round(max(min(speed, fit), physics.min_speed), 4)))
    return tuple(queries)


def _longest_run(world: PuckWorld, floor: int, need: float, grid: float = 0.1, step: float = 0.04,
                 max_run: float = 4.0) -> tuple[tuple[float, float], float, float]:
    """Feasible probe (puck, angle) on `floor` with the longest clear straight run up to `need`, and that run.

    Ties go to the start farthest from holes.
    """
    phys, pm = world.physics, world.puck_map
    xs = np.arange(grid / 2, pm.width, grid)
    ys = np.arange(grid / 2, pm.height, grid)
    pucks = np.array([(x, y) for x in xs for y in ys])
    pucks = pucks[[world.surface_at(p) == floor and world.puck_position_free(p) for p in pucks]]
    angles = np.radians(np.arange(-180.0, 180.0, 10.0))
    s = np.arange(step, max_run + 1e-9, step)
    run = np.full((len(pucks), len(angles)), max_run)
    spread = _SIGMAS * phys.angle_noise
    for offset in (-spread, 0.0, spread):
        dirs = np.stack([np.cos(angles + offset), np.sin(angles + offset)], -1)
        pts = pucks[:, None, None] + s[None, None, :, None] * dirs[None, :, None]
        row = np.clip((pts[..., 1] // pm.cell_size).astype(int), 0, pm.num_rows - 1)
        col = np.clip((pts[..., 0] // pm.cell_size).astype(int), 0, pm.num_cols - 1)
        ok = world.surfaces[row, col] == floor
        for x0, y0, x1, y1 in world.walls:
            dx = np.maximum(np.maximum(x0 - pts[..., 0], pts[..., 0] - x1), 0.0)
            dy = np.maximum(np.maximum(y0 - pts[..., 1], pts[..., 1] - y1), 0.0)
            ok &= dx**2 + dy**2 > phys.puck_radius**2
        for hole in world.holes:
            ok &= np.linalg.norm(pts - hole, axis=-1) > phys.hole_radius + phys.puck_radius + 0.1
        first_bad = np.where(ok.all(-1), len(s), np.argmin(ok, -1))
        run = np.minimum(run, first_bad * step)  # clear up to the last good sample
    gap = np.min([np.linalg.norm(pucks - hole, axis=-1) for hole in world.holes] + [np.full(len(pucks), 1.0)], 0)
    score = np.minimum(run, need) + 1e-3 * gap[:, None]
    for k in np.argsort(-score, axis=None, kind="stable"):
        i, j = np.unravel_index(k, run.shape)
        if world.push_feasible(pucks[i], angles[j]):
            puck = (round(float(pucks[i][0]), 4), round(float(pucks[i][1]), 4))
            return puck, round(float(angles[j]), 4), float(run[i, j])
    raise ValueError(f"No feasible probe on floor {floor} of {pm.name}")


class FrictionGatherer(InfoGatherer):
    """Probes the floor the belief is least sure of, with a real push in its own simulator."""

    def __init__(self, world: PuckWorld, make_env: Callable[[], Any], low, env_name: str = "2d",
                 header: str = "", speed: float = 0.8, min_std: float = 0.005) -> None:
        self.world = world
        self.make_env = make_env  # builds the probe's own env, never the task's
        self.env_name = env_name
        self.header = header
        self.min_std = min_std  # about the resampling jitter
        self.queries = probe_queries(world, low, speed)
        self._env: Any = None
        self._recorder: Any = None

    @property
    def environment(self) -> Any:
        """Probe simulator, built lazily."""
        if self._env is None:
            self._env = self.make_env()
        return self._env

    def choose(self, belief, rng: np.random.Generator) -> FrictionQuery | None:
        del rng
        std = belief.std
        floor = int(np.argmax(std))
        return self.queries[floor] if std[floor] > self.min_std else None

    def respond(self, theta: NDArray[np.float64], query: FrictionQuery, rng: np.random.Generator,
                frames: list | None = None) -> Observation:
        env, puck = self.environment, np.array(query.puck, np.float64)
        pose = self.world.push_pose(puck, query.angle)
        assert pose is not None, query
        base, heading = pose
        robot = base - _BACKOFF * query.push.direction
        env.reset(seed=int(rng.integers(2**31)), options={
            "frictions": tuple(float(f) for f in theta), "puck": query.puck,
            "robot": (float(robot[0]), float(robot[1]), float(heading))})
        action = np.array([query.angle, query.speed])
        if frames is None:
            info = env.step(action)[4]
        else:
            if self._recorder is None:
                self._recorder = Recorder(env, self.env_name, header=f"{self.header} probe".strip())
            info = self._recorder.record(frames, action, 0)[4]
        return Observation("query", push_record(info, puck, query.push), query)
