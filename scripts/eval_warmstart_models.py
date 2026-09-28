"""Offline check of the dream pipeline's learned-condition models, on the recorded runs.

Each learned run's rows are refit with `make_regressor`, as `OnlineWarmStart` does. The
recorded rows predate the history features, so velocities are stand-ins (position
differences per control step); the previous torque is exact. Two protocols per run:

- `heldout`: fit the earlier 80% of the segments, score the rest;
- `online`: fit the first n segments, score the next `NEXT`, for growing n.

    python scripts/eval_warmstart_models.py
    python scripts/make_warmstart_figures.py
"""

from __future__ import annotations

import glob
import json
import multiprocessing as mp
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np

OUT = Path("data/dream/warmstart_models")
SPECS = {
    "deployed MLP 128×128": "mlp:128x128:a0.001",
    "deployed ridge": "ridge",
    "new MLP, old inputs": "net:256x256:d0.3",
    "new MLP + history": "net:256x256:d0.3:hist",
}
SEEN = (3, 5, 10, 20, 30, 40, 50, 60)
NEXT = 5
J = 7


def load_runs(pattern: str = "data/dream/ep15/*_4*_checkpoint.pkl") -> dict:
    from limb_baseline import dream_pipeline

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            return super().find_class(dream_pipeline.__name__ if module == "__main__" else module, name)

    runs = {}
    for path in sorted(glob.glob(pattern)):
        if "cache" in path or "plain" in path:
            continue
        with open(path, "rb") as handle:
            learner = Unpickler(handle).load()["learner"]
        x, y = np.asarray(learner.features), np.asarray(learner.labels)
        seg = np.cumsum(x[:, 25] == 0) - 1
        first = np.r_[True, seg[1:] != seg[:-1]]
        pos = x[:, :13]
        velocity = pos - np.vstack([pos[:1], pos[:-1]])
        velocity[first] = 0.0
        previous = np.vstack([y[:1, :J], y[:-1, :J]])
        previous[first] = 0.0
        name = Path(path).name.replace("_checkpoint.pkl", "")
        runs[name] = (np.hstack([x, velocity, previous]), y, seg)
    return runs


def nmse(pred, y, var):
    return float((((pred - y) ** 2) / var).mean())


def score(payload):
    run, label, spec, protocol, n = payload
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    from limb_baseline.warmstart import make_regressor

    x, y, seg = RUNS[run]
    if protocol == "heldout":
        cut = int(0.8 * (seg.max() + 1))
        train, test = seg < cut, seg >= cut
    else:
        train, test = seg < n, (seg >= n) & (seg < n + NEXT)
    var = y[train].var(0)
    var[var < 1e-12] = 1.0
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = make_regressor(spec).fit(x[train], y[train])
    fit_time = time.perf_counter() - started
    return {"run": run, "model": label, "protocol": protocol, "seen": int(n or cut),
            "train": nmse(model.predict(x[train]), y[train], var),
            "test": nmse(model.predict(x[test]), y[test], var),
            "test_first": nmse(model.predict(x[test])[:, :J], y[test][:, :J], var[:J]),
            "rows": int(train.sum()), "fit_seconds": fit_time}


RUNS = load_runs()


def main():
    jobs = []
    for run, (_, _, seg) in RUNS.items():
        for label, spec in SPECS.items():
            jobs.append((run, label, spec, "heldout", 0))
            jobs += [(run, label, spec, "online", n) for n in SEEN if n + NEXT <= seg.max() + 1]
    jobs.sort(key=lambda j: -j[4] - ("net" in j[2]) * 100)
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    with mp.get_context("fork").Pool(12) as pool:
        for i, row in enumerate(pool.imap_unordered(score, jobs)):
            results.append(row)
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                print(f"[{i + 1}/{len(jobs)}]", flush=True)
                json.dump(results, open(OUT / "results.json", "w"))
    print(f"{len(RUNS)} runs -> {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
