"""Live W&B logging for one person's run: per-episode metrics and, optionally, videos (not kept locally)."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT = "learning-forceful-manipulation"
METRICS = ("plan_seconds", "effort", "success", "steps", "queries", "dream_seconds", "dreams_solved",
           "rows", "fit_seconds", "val_nmse", "info_goal_distance")


def run_name(task: str, experiment: str, stem: str) -> tuple[str, str]:
    """(group, name), shared with scripts/log_to_wandb.py so reruns never duplicate."""
    group = f"{task}/{experiment}"
    return group, f"{group}/{stem}"


def episode_log(record: dict) -> dict:
    log = {"episode": int(record["episode"])}
    for k in METRICS:
        v = record.get(k)
        if isinstance(v, (bool, int, float, np.floating)) and v == v:
            log[f"episode/{k}"] = float(v)
    return log


class Logger:
    """One W&B run; media go through a temp dir that is deleted on finish."""

    def __init__(self, task: str, experiment: str, out: Path, config) -> None:
        import wandb

        self.wandb, self.dir = wandb, tempfile.mkdtemp(prefix="able-wandb-")
        group, name = run_name(task, experiment, Path(out).stem)
        self.run = wandb.init(project=PROJECT, group=group, name=name, job_type=config.condition,
                              id=hashlib.md5(name.encode()).hexdigest()[:16], resume="allow",
                              config=asdict(config), tags=[task], dir=self.dir, reinit="finish_previous")
        self.run.define_metric("episode")
        self.run.define_metric("episode/*", step_metric="episode")

    def log(self, record: dict, frames: list | None, fps: int) -> None:
        log = episode_log(record)
        if frames:
            video = np.stack(frames).transpose(0, 3, 1, 2)  # (T, C, H, W)
            log["episode/video"] = self.wandb.Video(video, fps=fps, format="mp4")
        self.run.log(log)

    def finish(self) -> None:
        self.run.finish()
        shutil.rmtree(self.dir, ignore_errors=True)
