"""Pick the learned condition's regressor for the dream pipeline, offline.

`collect` plans every goal of each person's task set once with the plain planner, in
parallel, and keeps the rows of the solved episodes: the features the learned model
reads and the next horizon of torques it predicts. `compare` fits each candidate on a
person's other goals and scores it on the held-out goal, which mimics the online
learner (a few episodes of one person). The rollout uses only the first predicted
torque at every step, so that error ranks first.

    python scripts/tune_warm_start.py collect --out data/dream/tuning/rows.pkl
    python scripts/tune_warm_start.py compare --rows data/dream/tuning/rows.pkl
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np

# Tuning people, clear of the experiment's (22-26).
PEOPLE = (101, 102, 103, 104, 105)
RIDGE = ("ridge", "ridge:a-3,3", "ridge:a-1,5:poly2", "ridge:a-3,3:poly2")
MLP = tuple(
    f"mlp:{h}:a{a:g}"
    for h in ("32x32", "64x64", "128x128", "64x64x64")
    for a in (1e-3, 1e-2, 1e-1, 1.0)
)


def _collect_one(payload) -> dict:
    person, goal_index, variant, max_time = payload
    from limb_baseline.variant import select as select_variant

    select_variant(variant)
    from limb_baseline import dream_pipeline as dp
    from limb_baseline.belief import Belief
    from limb_baseline.human import DEFAULT_PRIOR
    from limb_baseline.planner import PlannerSettings
    from limb_baseline.worker import isolate_worker

    isolate_worker()
    config = dp.DreamConfig()
    task_seed = 0
    theta = DEFAULT_PRIOR.sample(np.random.default_rng(person))
    task_set = dp.make_task_set(theta, np.random.default_rng(task_seed), config)
    belief, _ = dp.observe(Belief.from_prior(DEFAULT_PRIOR),
                           [task_set.nominal.limb_init], theta)
    believed = dp.Believed.build(belief, config, np.random.default_rng(task_seed + 2))
    memory = dp.Memory.build("plain", "ridge")
    record = {"person": person, "goal": goal_index, "features": [], "labels": []}
    try:
        result, stats = dp.plan_and_execute(
            task_set.tasks[goal_index], theta, believed, memory,
            PlannerSettings(max_time=max_time), task_seed, 7_000 + goal_index,
        )
        record.update(stats)
        if result.goal_reached:
            for segment in result.segments:
                record["features"].append(segment.features)
                record["labels"].append(np.asarray(segment.torques).ravel())
    except Exception as exc:  # pylint: disable=broad-except
        record["failure_reason"] = f"{type(exc).__name__}: {exc}"
    return record


def collect(out: Path, variant: str, num_goals: int, max_time: float, workers: int) -> None:
    items = [(p, g, variant, max_time) for p in PEOPLE for g in range(num_goals)]
    records = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for record in pool.imap_unordered(_collect_one, items):
            records.append(record)
            print(f"[{len(records)}/{len(items)}] person={record['person']} "
                  f"goal={record['goal']} reached={record.get('goal_reached')} "
                  f"plan={record.get('plan_time')}s rows={len(record['features'])} "
                  f"{record.get('failure_reason', '')}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as handle:
        pickle.dump(records, handle)
    print(f"rows -> {out}")


def _score_one(payload) -> dict:
    spec, person, held, rows = payload
    from sklearn.exceptions import ConvergenceWarning

    from limb_baseline.warmstart import make_regressor

    mine = [r for r in rows if r["person"] == person and r["features"]]
    if held is None:  # seen goals: fit every goal, score the fit on them
        train = test = mine
    else:
        train = [r for r in mine if r["goal"] != held]
        test = [r for r in mine if r["goal"] == held]
    if not train or not test:
        return {}
    x = np.vstack([r["features"] for r in train]); y = np.vstack([r["labels"] for r in train])
    xt = np.vstack([r["features"] for r in test]); yt = np.vstack([r["labels"] for r in test])
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = make_regressor(spec).fit(x, y)
    fit_time = time.perf_counter() - started
    predicted = model.predict(xt)
    joints = 7
    first = np.mean((predicted[:, :joints] - yt[:, :joints]) ** 2) / np.var(yt[:, :joints])
    whole = np.mean((predicted - yt) ** 2) / np.var(yt)
    return {"spec": spec, "person": person, "held": held, "train_rows": len(x),
            "first_nmse": first, "horizon_nmse": whole, "fit_time": fit_time}


def compare(rows_path: Path, workers: int, out: Path, specs=RIDGE + MLP, seen: bool = False) -> None:
    import pandas as pd

    with open(rows_path, "rb") as handle:
        rows = pickle.load(handle)
    solved = [r for r in rows if r["features"]]
    print(f"{len(solved)}/{len(rows)} episodes solved, "
          f"{sum(len(r['features']) for r in solved)} rows")
    people = sorted({r["person"] for r in solved})
    if seen:
        items = [(spec, p, None, solved) for spec in specs for p in people]
    else:
        items = [(spec, p, g, solved) for spec in specs for p in people
                 for g in sorted({r["goal"] for r in solved if r["person"] == p})]
    with mp.get_context("spawn").Pool(workers) as pool:
        scores = [s for s in pool.imap_unordered(_score_one, items) if s]
    frame = pd.DataFrame(scores)
    frame.to_csv(out, index=False)
    per_person = frame.groupby(["spec", "person"])[["first_nmse", "horizon_nmse", "fit_time"]].mean()
    summary = per_person.groupby("spec").agg(
        first_nmse=("first_nmse", "mean"), first_se=("first_nmse", "sem"),
        horizon_nmse=("horizon_nmse", "mean"), fit_time=("fit_time", "mean"),
    ).sort_values("first_nmse")
    summary["family"] = ["ridge" if s.startswith("ridge") else "mlp" for s in summary.index]
    print(summary.round(4).to_string())
    print(f"\nscores -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", type=Path, default=Path("data/dream/tuning/rows.pkl"))
    c.add_argument("--variant", default="isolated-left-arm")
    c.add_argument("--num-goals", type=int, default=6)
    c.add_argument("--max-time", type=float, default=600.0)
    c.add_argument("--workers", type=int, default=30)
    m = sub.add_parser("compare")
    m.add_argument("--rows", type=Path, default=Path("data/dream/tuning/rows.pkl"))
    m.add_argument("--out", type=Path, default=Path("data/dream/tuning/scores.csv"))
    m.add_argument("--workers", type=int, default=30)
    m.add_argument("--seen", action="store_true",
                   help="Score the fit on the goals trained on, the regime a dreamed goal "
                        "is in, instead of a held-out goal.")
    m.add_argument("--specs", nargs="+", default=None,
                   help="Candidate specs to compare instead of the default grid.")
    args = parser.parse_args()
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    if args.command == "collect":
        collect(args.out, args.variant, args.num_goals, args.max_time, args.workers)
    else:
        compare(args.rows, args.workers, args.out, tuple(args.specs or RIDGE + MLP), args.seen)


if __name__ == "__main__":
    main()
