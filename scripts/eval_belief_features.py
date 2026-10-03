"""Does the belief's floor (or its cache key) help the learned model? Offline, on recorded runs.

The model reads the belief's mean and stddev, which barely move in these runs, but not its
floor, which is what changes and what decides the planner's particles. Each learned 4-dream
step run's rows are matched to the refit that added them (`_learning.jsonl`, in order) and
to the belief they were planned under (the run and dreams CSVs), then the new model is fit
on episodes before `CUT` and scored on the rest, per row role, with these extra inputs:

- `base`: the 46 inputs the model reads today;
- `floor`: plus the floor per motion (raw; the imagined belief's from its key);
- `key`: plus the rounded (mean, stddev, floor) cache key, which picks the particles;
- `floor shuffled`: `floor` with floors permuted across episodes, a control for width.

    python scripts/eval_belief_features.py
"""

from __future__ import annotations

import glob
import json
import multiprocessing as mp
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/dream/warmstart_compare")
OUT = Path("data/dream/belief_features")
CUT = 14
SPEC = "net:256x256:d0.3:hist"
SETS = ("base", "floor", "key", "floor shuffled")
SEEDS = 3


def numbers(text) -> np.ndarray:
    return np.array([float(v) for v in str(text).split()])


def load_run(stem: str):
    from limb_baseline import dream_pipeline

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            return super().find_class(dream_pipeline.__name__ if module == "__main__" else module, name)

    with open(DATA / f"{stem}_checkpoint.pkl", "rb") as handle:
        learner = Unpickler(handle).load()["learner"]
    x, y = np.asarray(learner.features), np.asarray(learner.labels)
    real = pd.read_csv(DATA / f"{stem}.csv").set_index("episode")
    dreams = pd.read_csv(DATA / f"{stem}_dreams.csv")
    floor_of = {e: numbers(real.loc[e, "belief_floor"]) for e in real.index}
    key_of = {e: np.round(np.concatenate([numbers(real.loc[e, k]) for k in
                                          ("belief_mean", "belief_stddev", "belief_floor")]) / 0.1)
              for e in real.index}
    meta, floors, keys = [], [], []
    for line in open(DATA / f"{stem}_learning.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["role"] == "imagined":
            match = dreams[(dreams.episode == r["episode"]) & (dreams.dream == r["dream"])
                           & (dreams.role == "imagined")]
            key = numbers(match.iloc[-1].belief_key)
            floor = key[6:] * 0.1
        else:
            key, floor = key_of[r["episode"]], floor_of[r["episode"]]
        meta += [(r["episode"], r["role"])] * r["new_rows"]
        floors += [floor] * r["new_rows"]
        keys += [key] * r["new_rows"]
    assert len(meta) == len(x), (stem, len(meta), len(x))
    meta = pd.DataFrame(meta, columns=["episode", "role"])
    return x, y, meta, np.asarray(floors), np.asarray(keys)


def score(job):
    stem, features, seed = job
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    from limb_baseline.warmstart import make_regressor

    x, y, meta, floors, keys = load_run(stem)
    if features == "floor":
        x = np.hstack([x, floors])
    elif features == "key":
        x = np.hstack([x, keys])
    elif features == "floor shuffled":
        episodes = meta.episode.to_numpy()
        order = np.random.default_rng(seed).permutation(np.unique(episodes))
        lookup = {e: floors[episodes == o][0] for e, o in zip(np.unique(episodes), order)}
        x = np.hstack([x, np.stack([lookup[e] for e in episodes])])
    train = (meta.episode < CUT).to_numpy()
    model = make_regressor(SPEC)
    model.seed = seed
    model.fit(x[train], y[train])
    var = np.maximum(y[train].var(0), 1e-12)
    err = (((model.predict(x) - y) ** 2) / var).mean(1)
    out = []
    for role in ("all", "real", "chosen", "imagined"):
        mask = ~train & ((meta.role == role).to_numpy() if role != "all" else True)
        if mask.any():
            out.append({"run": stem, "features": features, "seed": seed, "role": role,
                        "rows": int(mask.sum()), "nmse": float(err[mask].mean())})
    out.append({"run": stem, "features": features, "seed": seed, "role": "train",
                "rows": int(train.sum()), "nmse": float(err[train].mean())})
    return out


def main():
    stems = sorted(Path(p).name[: -len("_learning.jsonl")]
                   for p in glob.glob(str(DATA / "*_4_step_p*_learning.jsonl")))
    jobs = [(s, f, seed) for s in stems for f in SETS for seed in range(SEEDS)]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with mp.get_context("fork").Pool(12) as pool:
        for i, result in enumerate(pool.imap_unordered(score, jobs)):
            rows += result
            if (i + 1) % 20 == 0:
                print(f"[{i + 1}/{len(jobs)}]", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "results.csv", index=False)
    per_run = frame.groupby(["features", "role", "run"]).nmse.mean().reset_index()
    print(per_run.groupby(["role", "features"]).nmse.median().unstack(0).round(3).to_string())


if __name__ == "__main__":
    main()
