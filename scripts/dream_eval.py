"""Offline studies of the warm-start models on recorded dream-pipeline runs.

Each learned run's rows come from its checkpoint, matched to the refit that added them
(`_learning.jsonl`, in order). Most studies fit on episodes before `CUT` and score the rest.

    python scripts/dream_eval.py belief-features
    python scripts/dream_eval.py backfill data/dream/fresh_goals data/dream/warmstart_compare
    python scripts/dream_eval.py screen data/dream/generative_models [--figure]
    python scripts/dream_eval.py sampling-steps data/dream/generative_models
    python scripts/dream_eval.py planner data/dream/generative_models [--from-episode 10]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

CUT = 14
K = 8
SPEC = "net:256x256:d0.3:hist"
FEATURE_SETS = ("base", "floor", "key", "floor shuffled")
SCREEN_COLUMNS = ("single", "best_of_8", "energy", "nll", "fit_seconds")
SCREEN_MODELS = {
    "regression": SPEC,
    "regression + noise": SPEC,
    "mixture density": "gen:mdn:hist",
    "flow matching": "gen:flow:hist",
    "diffusion": "gen:ddpm:hist",
    "conditional VAE": "gen:cvae:hist",
    "energy-based": "gen:ebm:hist",
}
# The models with a density; the regression's is a Gaussian at its train residual scale.
DENSITY_MODELS = ("mixture density", "flow matching", "diffusion")
STEPS = (1, 2, 5, 10, 25, 50, 100)
# label: (model kind, the attribute holding its number of sampling steps)
SAMPLERS = {"flow matching": ("flow", "integration_steps"), "diffusion": ("ddpm", "sample_steps")}


def numbers(text) -> np.ndarray:
    return np.array([float(v) for v in str(text).split()])


def load_run(directory, stem: str):
    """A learned run's rows: inputs, labels, (episode, role) per row, belief floors and keys."""
    from able.tasks.limb import dream_pipeline

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            return super().find_class(dream_pipeline.__name__ if module == "__main__" else module, name)

    data = Path(directory)
    with open(data / f"{stem}_checkpoint.pkl", "rb") as handle:
        learner = Unpickler(handle).load()["learner"]
    x, y = np.asarray(learner.features), np.asarray(learner.labels)
    real = pd.read_csv(data / f"{stem}.csv").set_index("episode")
    dreams = pd.read_csv(data / f"{stem}_dreams.csv")
    floor_of = {e: numbers(real.loc[e, "belief_floor"]) for e in real.index}
    key_of = {e: np.round(np.concatenate([numbers(real.loc[e, k]) for k in
                                          ("belief_mean", "belief_stddev", "belief_floor")]) / 0.1)
              for e in real.index}
    meta, floors, keys = [], [], []
    for line in open(data / f"{stem}_learning.jsonl", encoding="utf-8"):
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


def stems(directory, pattern: str) -> list[str]:
    return sorted(p.name[: -len("_learning.jsonl")] for p in Path(directory).glob(pattern))


def one_thread():
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)


def split(directory, stem):
    """Train rows (episodes before `CUT`), test rows, and the train labels' variance and std."""
    x, y, meta, _, _ = load_run(directory, stem)
    train = (meta.episode < CUT).to_numpy()
    var = np.maximum(y[train].var(0), 1e-12)
    return x[train], y[train], x[~train], y[~train], var, np.sqrt(var)


def nmse(pred, y, var):
    return (((pred - y) ** 2) / var).mean(-1)


def energy_score(samples, y, std):
    s, t = samples / std, y[:, None] / std
    fit = np.linalg.norm(s - t, axis=-1).mean(1)
    spread = np.linalg.norm(s[:, :, None] - s[:, None], axis=-1).mean((1, 2)) / 2
    return float((fit - spread).mean() / np.sqrt(y.shape[1]))


def gaussian_nll(mean, scale, y, std):
    """Per-dimension negative log-likelihood, in nats, of labels standardized by `std`."""
    z = (y - mean) / scale
    return float((0.5 * z ** 2 + np.log(scale / std) + 0.5 * np.log(2 * np.pi)).mean())


# --- belief-features ---------------------------------------------------------------------


def belief_job(job):
    directory, stem, features, seed = job
    one_thread()
    from able.tasks.limb.warmstart import make_regressor

    x, y, meta, floors, keys = load_run(directory, stem)
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


def belief_features(args):
    """Does the belief's floor (or its cache key) help the learned model?

    The model reads the belief's mean and stddev, which barely move in these runs, but not
    its floor, which is what changes and what decides the planner's particles. Each learned
    4-dream step run's rows are matched to the belief they were planned under (the run and
    dreams CSVs), then the new model is scored per row role with these extra inputs:

    - `base`: the 46 inputs the model reads today;
    - `floor`: plus the floor per motion (raw; the imagined belief's from its key);
    - `key`: plus the rounded (mean, stddev, floor) cache key, which picks the particles;
    - `floor shuffled`: `floor` with floors permuted across episodes, a control for width.
    """
    jobs = [(args.data, s, f, seed) for s in stems(args.data, "*_4_step_p*_learning.jsonl")
            for f in FEATURE_SETS for seed in range(args.seeds)]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with mp.get_context("fork").Pool(12) as pool:
        for i, result in enumerate(pool.imap_unordered(belief_job, jobs)):
            rows += result
            if (i + 1) % 20 == 0:
                print(f"[{i + 1}/{len(jobs)}]", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "results.csv", index=False)
    per_run = frame.groupby(["features", "role", "run"]).nmse.mean().reset_index()
    print(per_run.groupby(["role", "features"]).nmse.median().unstack(0).round(3).to_string())


# --- backfill ----------------------------------------------------------------------------


def backfill_job(job):
    directory, stem, i = job
    one_thread()
    from able.tasks.limb.warmstart import DropoutMLP

    x, y, _, _, _ = load_run(directory, stem)
    log = [json.loads(line) for line in open(Path(directory) / f"{stem}_learning.jsonl")]
    n = int(np.cumsum([r["new_rows"] for r in log])[i])
    model = DropoutMLP(hidden=(256, 256), dropout=0.3).fit(x[:n], y[:n])
    logged = log[i].get("val_nmse")
    match = logged is None or not np.isfinite(logged) or abs(model.best_val - logged) < 1e-6
    return directory, stem, i, {"val_nmse": model.best_val, "val_recent_nmse": model.best_val_recent,
                     "curve": [list(h) for h in model.history], "reproduced": bool(match)}


def backfill(args):
    """Recompute the new model's refits for runs logged before `val_recent_nmse` existed.

    Each refit is fit again on the same rows with the same seed (it reproduces the logged
    validation error exactly) and its recent validation error and training curve are written
    to `<run>_learning_backfill.jsonl`, one line per refit; `dream_figures.py curves` merges
    them.
    """
    jobs = []
    for directory in args.directories:
        for stem in stems(directory, "net_*_learning.jsonl"):
            refits = sum(1 for _ in open(Path(directory) / f"{stem}_learning.jsonl"))
            jobs += [(directory, stem, i) for i in range(refits)]
    results = {}
    with mp.get_context("fork").Pool(12) as pool:
        for k, (directory, stem, i, row) in enumerate(pool.imap_unordered(backfill_job, jobs)):
            results.setdefault((directory, stem), {})[i] = row
            if (k + 1) % 100 == 0:
                print(f"[{k + 1}/{len(jobs)}]", flush=True)
    for (directory, stem), rows in sorted(results.items()):
        with open(Path(directory) / f"{stem}_learning_backfill.jsonl", "w") as handle:
            for i in sorted(rows):
                handle.write(json.dumps({"refit": i, **rows[i]}) + "\n")
        bad = sum(not r["reproduced"] for r in rows.values())
        print(f"{directory}/{stem}: {len(rows)} refits, {bad} not reproduced")


# --- screen ------------------------------------------------------------------------------


def screen_job(job):
    directory, stem, label = job
    one_thread()
    from able.tasks.limb.warmstart import make_regressor

    xt, yt, xs, ys, var, std = split(directory, stem)
    started = time.perf_counter()
    model = make_regressor(SCREEN_MODELS[label]).fit(xt, yt)
    fit_seconds = time.perf_counter() - started
    best = model.predict(xs)
    nll = float("nan")
    if label == "regression + noise":
        scale = (yt - model.predict(xt)).std(0)
        rng = np.random.default_rng(0)
        draws = best[:, None] + rng.normal(size=(len(xs), K - 1, yt.shape[1])) * scale
        nll = gaussian_nll(best, scale, ys, std)
    elif label == "regression":
        draws = best[:, None].repeat(K - 1, axis=1)
    else:
        draws = model.sample(xs, K - 1, seed=0)
        if label in DENSITY_MODELS:
            nll = float(-(model.log_prob(xs, ys) + np.log(std).sum()).mean() / ys.shape[1])
    candidates = np.concatenate([best[:, None], draws], 1)
    return {"run": stem, "model": label, "rows": len(xs),
            "single": float(nmse(best, ys, var).mean()),
            "best_of_8": float(nmse(candidates, ys[:, None], var).min(1).mean()),
            "energy": energy_score(candidates, ys, std) if label != "regression" else float("nan"),
            "nll": nll, "fit_seconds": fit_seconds}


def screen_figure(frame: pd.DataFrame, out: Path, data: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = list(SCREEN_MODELS)
    med = frame.groupby("model")[list(SCREEN_COLUMNS)].median().loc[order]
    fig, axes = plt.subplots(1, 5, figsize=(18.5, 3.8), sharey=True)
    titles = {"single": "Most likely plan (nMSE)", "best_of_8": "Best of 8 candidates (nMSE)",
              "energy": "Energy score of 8 (lower better)",
              "nll": "Held-out NLL (nats/dim, lower better)", "fit_seconds": "Fit time (s, one CPU)"}
    colors = ["#8C8C8C", "#b0b0b0", "#2a78d6", "#1baf7a", "#eb6834", "#e87ba4", "#4a3aa7"]
    for ax, column in zip(axes, titles):
        ax.barh(range(len(order)), med[column], color=colors)
        # A diffusion NLL tens of nats per dimension would flatten the others: cut its bar.
        limit = 1.3 * med[column][med[column] < 5].max() if column == "nll" else np.inf
        for i, v in enumerate(med[column]):
            if np.isfinite(v):
                ax.text(min(v, limit), i, f" {v:.2f}" if column != "fit_seconds" else f" {v:.0f}",
                        va="center", ha="left" if v <= limit else "right", fontsize=8)
        if np.isfinite(limit):
            ax.set_xlim(0, limit)
        ax.set_title(titles[column], loc="left", fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    shown = {"regression": "MLP 256×256 + vel/torque",
             "regression + noise": "MLP 256×256 + vel/torque, noisy copies"}
    axes[0].set_yticks(range(len(order)), [shown.get(m, m) for m in order])
    axes[0].invert_yaxis()
    fig.suptitle(f"Offline screen of generative models on {data}: median over runs "
                 "(train episodes 0-13, test 14-19)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)


def screen(args):
    """Offline screen of generative warm-start models.

    Per learned run: the single most likely plan, the best of 8 candidates (the most likely
    plan and 7 draws, as the step-mode MPC picks), the energy score of 8 draws, and the
    held-out negative log-likelihood per plan dimension (labels standardized by the train
    std), which differs from the KL divergence to the true plan distribution by a constant
    shared by all models. Flow matching and diffusion give their continuous ODE's density.
    `regression + noise` is today's seeding: the regression plan and 7 copies with Gaussian
    noise at the regression's train residual scale. `--figure` redraws from the saved CSV.
    """
    results = Path(args.directory) / "generative_screen.csv"
    if not args.figure:
        jobs = [(args.directory, s, m) for s in stems(args.directory, "net_*_learning.jsonl")
                for m in SCREEN_MODELS]
        jobs.sort(key=lambda j: j[2] != "energy-based")
        with mp.get_context("fork").Pool(12) as pool:
            rows = pool.map(screen_job, jobs)
        pd.DataFrame(rows).to_csv(results, index=False)
    frame = pd.read_csv(results)
    print(frame.groupby("model")[list(SCREEN_COLUMNS)].median()
          .loc[list(SCREEN_MODELS)].round(3).to_string())
    name = Path(args.directory).name
    screen_figure(frame, Path("figures/dream") / name / "generative_screen.png", name)


# --- sampling-steps ----------------------------------------------------------------------


def steps_job(job):
    directory, stem, label, seed = job
    one_thread()
    from able.tasks.limb.generative import KINDS, limb_segments

    xt, yt, xs, ys, var, std = split(directory, stem)
    kind, attribute = SAMPLERS[label]
    model = KINDS[kind](seed=seed, group_fn=limb_segments).fit(xt, yt)
    rows = []
    for steps in STEPS:
        setattr(model, attribute, steps)
        best = model.predict(xs)
        candidates = np.concatenate([best[:, None], model.sample(xs, K - 1, seed=seed)], 1)
        started = time.perf_counter()
        for _ in range(20):
            model.sample(xs[:1], K - 1)
        seconds = (time.perf_counter() - started) / 20
        rows.append({"run": stem, "model": label, "seed": seed, "steps": steps,
                     "single": float(nmse(best, ys, var).mean()),
                     "best_of_8": float(nmse(candidates, ys[:, None], var).min(1).mean()),
                     "energy": energy_score(candidates, ys, std), "ms_per_draw": 1000 * seconds})
    return rows


def steps_figure(frame, reference, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"flow matching": "#4a3aa7", "diffusion": "#e87ba4"}
    panels = [("single", "Most likely plan (nMSE)"), ("best_of_8", "Best of 8 candidates (nMSE)"),
              ("energy", "Energy score of 8 (lower better)"), ("ms_per_draw", "Time to draw 8 plans (ms)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (column, title) in zip(axes, panels):
        for label, color in colors.items():
            per_run = frame[frame.model == label].groupby(["steps", "run"])[column].mean().unstack()
            m, se = per_run.median(1), per_run.sem(1)
            ax.plot(m.index, m, color=color, marker="o", ms=4, label=label)
            ax.fill_between(m.index, m - se, m + se, color=color, alpha=0.15, lw=0)
        if column in reference:
            ax.axhline(reference[column], color="#8C8C8C", ls="--", lw=1.2,
                       label="regression + noise (today)")
        ax.axvline(10, color="#c3c2b7", lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_xticks(STEPS, [str(s) for s in STEPS])
        ax.set_xlabel("sampling steps")
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        if column == "ms_per_draw":
            ax.set_yscale("log")
    axes[1].legend(fontsize=8)
    fig.suptitle("Sampling steps, varied geometry (median over 5 runs x 2 seeds; dotted line: 10 used so far)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)


def sampling_steps(args):
    """How many sampling steps do flow matching and diffusion need?

    Each model is trained once per run and seed, then sampled with each step count on the
    later episodes: the most likely plan (zero noise), the best of 8 candidates (it and 7
    draws), the energy score, and the time to draw 8 plans for one row. Needs `screen`'s CSV
    for the regression reference.
    """
    directory = Path(args.directory)
    jobs = [(args.directory, s, m, seed) for s in stems(directory, "net_4_step_p*_learning.jsonl")
            for m in SAMPLERS for seed in (0, 1)]
    with mp.get_context("fork").Pool(min(12, len(jobs))) as pool:
        frame = pd.DataFrame([r for rows in pool.map(steps_job, jobs) for r in rows])
    frame.to_csv(directory / "sampling_steps.csv", index=False)
    screened = pd.read_csv(directory / "generative_screen.csv")
    reference = screened[screened.model == "regression + noise"][["single", "best_of_8", "energy"]].median().to_dict()
    per_run = frame.groupby(["model", "steps", "run"])[["single", "best_of_8", "energy", "ms_per_draw"]].mean()
    print(per_run.groupby(["model", "steps"]).median().round(3).to_string())
    print("reference (regression + noise):", {k: round(v, 3) for k, v in reference.items()})
    steps_figure(frame, reference, Path("figures/dream") / directory.name / "sampling_steps.png")


# --- planner -----------------------------------------------------------------------------


def planner(args):
    """What the planner got from each line's seeds, over every person and real episode.

    `seed_accept` is the share of offered seeds the MPC step took over a full search, and
    `seeded_share` the share of all MPC steps that ran on a seed.
    """
    data = Path(args.directory)
    frame = pd.concat(pd.read_csv(p) for p in sorted(data.glob("*_p??.csv")))
    frame = frame[frame.episode >= args.from_episode]
    frame["reached"] = frame.goal_reached.astype(str) == "True"
    g = frame.groupby("tag")
    seeded, rejected = g.seeded_steps.sum(), g.seed_rejected.sum()
    table = pd.DataFrame({
        "reached": g.reached.mean(), "rollouts_mean": g.rollouts.mean(),
        "rollouts_median": g.rollouts.median(), "plan_time_mean": g.plan_time.mean(),
        "plan_time_median": g.plan_time.median(), "seed_accept": seeded / (seeded + rejected),
        "seeded_share": seeded / g.mpc_steps.sum(), "goal_distance_mean": g.goal_distance.mean(),
    })
    print(table.round(3).to_string())
    suffix = f"_from{args.from_episode}" if args.from_episode else ""
    table.to_csv(data / f"planner_metrics{suffix}.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name, function):
        sub = commands.add_parser(name, description=function.__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
        sub.set_defaults(function=function)
        return sub

    sub = add("belief-features", belief_features)
    sub.add_argument("--data", default="data/dream/warmstart_compare")
    sub.add_argument("--out", type=Path, default=Path("data/dream/belief_features"))
    sub.add_argument("--seeds", type=int, default=3)
    add("backfill", backfill).add_argument("directories", nargs="+")
    sub = add("screen", screen)
    sub.add_argument("directory")
    sub.add_argument("--figure", action="store_true")
    add("sampling-steps", sampling_steps).add_argument("directory")
    sub = add("planner", planner)
    sub.add_argument("directory")
    sub.add_argument("--from-episode", type=int, default=0)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
