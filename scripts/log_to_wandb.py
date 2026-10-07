"""Upload the warm-start study's results to Weights & Biases, one project, a group per study.

Groups: `ml-sweep` (a run per phase), `flow-policy` (a run per config, with its loss
curves), `offline-model-compare`, `belief-features`, `dream/<tag>` (a run per pipeline run,
per-episode metrics and, for learned models, every refit), and `figures`. Uploaded run names
are kept in `data/wandb_manifest.json`, so rerunning only adds what is new.

    python scripts/log_to_wandb.py                      # everything not yet uploaded
    python scripts/log_to_wandb.py --only dream --dream-dirs data/dream/fresh_goals
    WANDB_MODE=offline python scripts/log_to_wandb.py   # write locally, send nothing
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

PROJECT = "learning-forceful-manipulation"
MANIFEST = Path("data/wandb_manifest.json")
DREAM_DIRS = ("data/dream/warmstart_compare", "data/dream/fresh_goals", "data/dream/generative_models")
FIGURE_DIRS = ("figures/dream/ml_search", "figures/archive/dream/flow",
               "figures/archive/dream/warmstart_models", "figures/dream/warmstart_compare",
               "figures/dream/fresh_goals", "figures/dream/generative_models")
CONFIG_COLUMNS = ("tag", "condition", "model", "n_dreams", "all_goals", "fresh_goals", "num_goals",
                  "variation", "particles", "confidence", "dream_max_time", "seed_mode",
                  "seed_rollouts", "variant", "person", "task_seed", "theta")
EPISODE_COLUMNS = ("plan_time", "rollouts", "goal_reached", "memo_hits", "dream_time", "fit_time",
                   "num_train_rows", "belief_width", "goal_distance", "seeded_steps", "full_steps",
                   "dreams_reached", "dreams_planned")


class Uploader:
    def __init__(self, force: bool):
        self.force = force
        self.done = set(json.loads(MANIFEST.read_text())) if MANIFEST.exists() else set()

    def run(self, name: str, group: str, **kwargs):
        if name in self.done and not self.force:
            print(f"skip {name}")
            return None
        print(f"upload {name}")
        run_id = hashlib.md5(name.encode()).hexdigest()[:16]
        if self.force and os.environ.get("WANDB_MODE") != "offline":
            # Resuming a run appends to its history, so a forced upload replaces the run.
            api = wandb.Api()
            for old in api.runs(f"{api.default_entity}/{PROJECT}", filters={"display_name": name}):
                old.delete()
            run_id = hashlib.md5(f"{name}:{time.time()}".encode()).hexdigest()[:16]
        return wandb.init(project=PROJECT, name=name, group=group, id=run_id, resume="allow",
                          dir=os.environ.get("WANDB_DIR"), reinit="finish_previous", **kwargs)

    def finish(self, run, name: str):
        run.finish()
        if os.environ.get("WANDB_MODE") != "offline":
            self.done.add(name)
            MANIFEST.write_text(json.dumps(sorted(self.done), indent=1))


def table(frame: pd.DataFrame) -> wandb.Table:
    frame = frame.copy()
    for column in frame:
        if frame[column].map(lambda v: isinstance(v, (dict, list))).any():
            frame[column] = frame[column].map(json.dumps)
    return wandb.Table(dataframe=frame)


def ml_sweep(up: Uploader):
    for path in sorted(glob.glob("data/dream/ml_sweep/*.jsonl")):
        phase = Path(path).stem
        name = f"ml-sweep/{phase}"
        run = up.run(name, "ml-sweep", job_type="offline-sweep",
                     config={"phase": phase, "metric": "nMSE = 1 - mean R^2, split by segment"})
        if run is None:
            continue
        rows = [json.loads(line) for line in open(path)]
        frame = pd.DataFrame([{k: v for k, v in r.items() if k not in ("curve", "error")}
                              for r in rows if "error" not in r])
        frame["cfg"] = frame["cfg"].map(lambda c: json.dumps(c, sort_keys=True))
        run.log({"fits": table(frame)})
        best = frame.loc[frame["val"].idxmin()]
        run.summary.update({"fits": len(frame), "best_val": best["val"], "best_cfg": best["cfg"],
                            "best_ds": best["ds"]})
        up.finish(run, name)


def flow_policy(up: Uploader):
    search = json.load(open("data/archive/dream/flow/search.json"))
    for row in search["flow"]:
        cfg = row["config"]
        name = "flow/{width}x{depth}-d{dropout}-e{epochs}".format(**cfg)
        run = up.run(name, "flow-policy", job_type="flow", config={**cfg, "split": "70/15/15 segments"})
        if run is None:
            continue
        for h in row["history"]:
            run.log({"epoch": h["epoch"], "loss/train": h["train"], "loss/val": h["val"],
                     "loss/test": h["test"]})
        run.summary.update({"best_epoch": row["best_epoch"],
                            **{f"val/{k}": v for k, v in row["val"].items() if k != "best_of_k"},
                            **{f"val/best_of_{k}": v for k, v in row["val"]["best_of_k"].items()}})
        up.finish(run, name)
    reg = search["regression"]
    name = "flow/regression-baseline"
    run = up.run(name, "flow-policy", job_type="regression",
                 config={"model": "MLP 256x2 dropout 0.3", "split": "70/15/15 segments"})
    if run is not None:
        for h in reg["history"]:
            run.log({"epoch": h["epoch"], "loss/train": h["train"], "loss/val": h["val"],
                     "loss/test": h["test"]})
        run.summary.update({"best_epoch": reg["best_epoch"], **{f"val/{k}": v for k, v in reg["val"].items()}})
        up.finish(run, name)
    name = "flow/scaling"
    run = up.run(name, "flow-policy", job_type="scaling")
    if run is not None:
        scaling = json.load(open("data/archive/dream/flow/scaling.json"))
        flat = [{"fraction": r["fraction"], "seed": r["seed"], "segments": r["segments"],
                 "model": m, "test_nmse": r[m]["test"]["nmse_det"], "test_energy": r[m]["test"]["energy"],
                 "train_nmse": r[m]["train_nmse_det"],
                 "test_best_of_8": r[m]["test"].get("best_of_k", {}).get("8")}
                for r in scaling for m in ("flow", "regression")]
        run.log({"scaling": table(pd.DataFrame(flat))})
        up.finish(run, name)


def offline_tables(up: Uploader):
    sources = {
        "offline-model-compare": ("data/archive/dream/warmstart_models/results.json", "offline-model-compare"),
        "belief-features": ("data/archive/dream/belief_features/results.csv", "belief-features"),
    }
    for name, (path, group) in sources.items():
        if not Path(path).exists():
            continue
        run = up.run(name, group, job_type="offline")
        if run is None:
            continue
        frame = pd.read_csv(path) if path.endswith(".csv") else pd.DataFrame(json.load(open(path)))
        run.log({"fits": table(frame)})
        summary = Path(path).with_name("summary.json")
        if summary.exists():
            run.summary.update(json.load(open(summary)))
        up.finish(run, name)


def dream_runs(up: Uploader, dirs, min_episodes: int):
    for directory in dirs:
        experiment = Path(directory).name
        for path in sorted(glob.glob(f"{directory}/*_p??.csv")):
            frame = pd.read_csv(path)
            if frame.episode.nunique() < min_episodes:
                print(f"skip {path}: {frame.episode.nunique()} episodes, still running?")
                continue
            stem = Path(path).stem
            name = f"{experiment}/{stem}"
            first = frame.iloc[0]
            config = {c: (first[c].item() if hasattr(first[c], "item") else first[c])
                      for c in CONFIG_COLUMNS if c in frame}
            config.update(experiment=experiment, episodes=int(frame.episode.nunique()))
            run = up.run(name, f"dream/{first['tag']}", job_type=str(first["condition"]),
                         config=config, tags=[experiment, str(first["seed_mode"]),
                                              f"dreams={first['n_dreams']}"])
            if run is None:
                continue
            run.define_metric("episode")
            run.define_metric("episode/*", step_metric="episode")
            run.define_metric("refit/index")
            run.define_metric("refit/*", step_metric="refit/index")
            for _, row in frame.iterrows():
                log = {"episode": int(row.episode)}
                for c in EPISODE_COLUMNS:
                    if c in row and pd.notna(row[c]):
                        log[f"episode/{c}"] = float(row[c])
                steps = row.get("seeded_steps", 0) + row.get("full_steps", 0)
                if steps:
                    log["episode/seed_accepted"] = float(row["seeded_steps"]) / steps
                log["episode/timeout"] = float(row["plan_time"] >= 599)
                run.log(log)
            learning = Path(path).with_name(stem + "_learning.jsonl")
            if learning.exists():
                refits = [json.loads(line) for line in open(learning)]
                for i, r in enumerate(refits):
                    log = {"refit/index": i}
                    for k in ("new_nmse", "new_nmse_first", "train_nmse", "val_nmse", "rows",
                              "fit_seconds", "episode"):
                        if k in r and r[k] is not None and np.isfinite(r[k]):
                            log[f"refit/{k}"] = r[k]
                    run.log(log)
                run.log({"refits": table(pd.DataFrame(refits).drop(columns="curve", errors="ignore"))})
            done = frame[frame.episode > 0]
            run.summary.update({
                "reached": float(frame.goal_reached.mean()),
                "timeouts": int((frame.plan_time >= 599).sum()),
                "plan_time_mean": float(frame.plan_time.mean()),
                "rollouts_mean": float(frame.rollouts.mean()),
                "rollouts_mean_after_first": float(done.rollouts.mean()),
                "dream_time_mean": float(frame.dream_time.mean()),
            })
            up.finish(run, name)


def figures(up: Uploader):
    for directory in FIGURE_DIRS:
        paths = sorted(glob.glob(f"{directory}/*.png"))
        if not paths:
            continue
        name = f"figures/{Path(directory).name}"
        run = up.run(name, "figures", job_type="figures")
        if run is None:
            continue
        run.log({Path(p).stem: wandb.Image(p) for p in paths})
        up.finish(run, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*",
                        choices=("ml-sweep", "flow", "offline", "dream", "figures"))
    parser.add_argument("--dream-dirs", nargs="*", default=DREAM_DIRS)
    parser.add_argument("--force", action="store_true",
                        help="Replace runs already uploaded (deletes them on W&B first).")
    parser.add_argument("--min-episodes", type=int, default=20,
                        help="Skip pipeline runs with fewer episodes (unfinished).")
    args = parser.parse_args()
    up = Uploader(args.force)
    parts = args.only or ("ml-sweep", "flow", "offline", "dream", "figures")
    if "ml-sweep" in parts:
        ml_sweep(up)
    if "flow" in parts:
        flow_policy(up)
    if "offline" in parts:
        offline_tables(up)
    if "dream" in parts:
        dream_runs(up, args.dream_dirs, args.min_episodes)
    if "figures" in parts:
        figures(up)


if __name__ == "__main__":
    main()
