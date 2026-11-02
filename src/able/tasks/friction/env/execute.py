"""Executing one push in the env (task episodes and probe queries alike), optionally recording frames."""

from __future__ import annotations

from typing import Any

import numpy as np

from able.tasks.friction.env.puckpushing_utils import Push


def execute_push(env: Any, action: np.ndarray, recorder: Any = None, frames: list | None = None,
                 push: int = 0) -> tuple[bool, dict]:
    """`env.step(action)`; with `frames`, the recorder captures the drive, strike and slide. (terminated, info)."""
    if frames is None:
        _, _, terminated, _, info = env.step(action)
    else:
        _, _, terminated, _, info = recorder.record(frames, action, push)
    return bool(terminated), info


def push_record(info: dict, puck, push: Push) -> dict:
    """An env step's info as the record the belief explains."""
    return {
        "puck": [float(v) for v in puck], "angle": push.angle, "speed": push.speed, "stance": push.stance,
        "feasible": bool(info["push_feasible"]), "release_speed": float(info["release_speed"]),
        "stop": np.asarray(info["stop_position"], np.float64).tolist(), "fell_in_hole": bool(info["fell_in_hole"]),
        "goal_reached": bool(info["goal_reached"]), "drive_distance": float(info["drive_distance"]),
    }
