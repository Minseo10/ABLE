"""Upload one experiment to W&B: one run per <condition>_p<person>.csv, grouped by task/experiment.

    python scripts/log_to_wandb.py data/friction_full [--figures figures/friction_full] [--force]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import wandb

PROJECT = "learning-forceful-manipulation"
METRICS = ("plan_seconds", "effort", "success", "steps", "queries", "dream_seconds", "dreams_solved",
           "rows", "fit_seconds", "val_nmse", "info_goal_distance")


def existing(group: str) -> set[str]:
    api = wandb.Api()
    return {r.name for r in api.runs(f"{api.default_entity}/{PROJECT}", filters={"group": group})}


def upload(path: Path, group: str) -> None:
    frame = pd.read_csv(path)
    first = frame.iloc[0]
    config = {k: (first[k].item() if hasattr(first[k], "item") else first[k])
              for k in ("task", "condition", "person", "effort_unit") if k in frame}
    config["theta"] = json.loads(first["theta"])
    run = wandb.init(project=PROJECT, name=f"{group}/{path.stem}", group=group,
                     job_type=str(first["condition"]), config=config, tags=[str(first["task"])],
                     reinit="finish_previous")
    run.define_metric("episode")
    run.define_metric("episode/*", step_metric="episode")
    for _, row in frame.iterrows():
        log = {"episode": int(row.episode)}
        for k in METRICS:
            if k in row and pd.notna(row[k]):
                log[f"episode/{k}"] = float(str(row[k]).lower() == "true") if k == "success" else float(row[k])
        run.log(log)
    run.summary.update({"success": float(frame.success.astype(str).str.lower().eq("true").mean()),
                        "plan_seconds_mean": float(frame.plan_seconds.mean()),
                        "effort_mean": float(frame.effort.mean()), "episodes": len(frame)})
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    runs = sorted(args.data.glob("*_p*.csv"))
    if not runs:
        raise SystemExit(f"no runs in {args.data}")
    group = f"{pd.read_csv(runs[0]).task.iloc[0]}/{args.data.name}"
    done = set() if args.force else existing(group)
    for path in runs:
        if f"{group}/{path.stem}" in done:
            print(f"skip {path.stem}")
            continue
        upload(path, group)
    if args.figures and args.figures.exists():
        run = wandb.init(project=PROJECT, name=f"{group}/figures", group=group, job_type="figures",
                         reinit="finish_previous")
        run.log({p.stem: wandb.Image(str(p)) for p in sorted(args.figures.glob("*.png"))})
        run.finish()


if __name__ == "__main__":
    main()
