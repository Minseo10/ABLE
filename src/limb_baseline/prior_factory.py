"""The prior-only baseline's factory: learn the warm-start model once, from the prior.

Before deployment the robot knows only the prior over users, b0. It imagines homes: each
home is a hypothetical person drawn from b0 with tasks of its own (a nominal start and
base, and goals around it inside that person's range, as `run_sequence` builds a real
person's fresh goals). In each home it dreams `dreams_per_home` times from b0 (plus the
start floor that home's resting limb shows), with the same `dream_once` ABLE dreams with,
and keeps the rows of every plan that reached its goal. No refit happens while dreaming,
so every factory plan is the slow planner's.

`fit` then trains one model on the rows of the first homes up to a dream budget, and
pickles it; `dream_pipeline --condition prior --prior-model <pickle>` ships it, frozen, to
every real person, whose belief still updates from execution.

    python -m limb_baseline.prior_factory dream --homes 100 --dreams-per-home 4 \\
        --shard 0 --num-shards 16 --out-dir data/dream/prior_only/factory
    python -m limb_baseline.prior_factory fit --factory data/dream/prior_only/factory \\
        --dreams 400 --model gen:flow:hist --out data/dream/prior_only/factory/g_400.pkl
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import pickle
import time
import traceback
from pathlib import Path

import numpy as np
from kinder_pddlstream_planning.limbrepositioning3d.stream import MPCConfig

from limb_baseline.belief import Belief
from limb_baseline.dataset import append_row
from limb_baseline.dream_pipeline import (
    START_FRACTION,
    Believed,
    DreamConfig,
    Memory,
    TaskSet,
    dream_once,
    observe,
    save_checkpoint,
)
from limb_baseline.human import DEFAULT_PRIOR
from limb_baseline.memo import SEED, OnlineWarmStart, perturbed_task, uninstall
from limb_baseline.planner import PlannerSettings
from limb_baseline.variant import VARIANTS
from limb_baseline.variant import select as select_variant
from limb_baseline.warmstart import make_regressor
from limb_baseline.worker import isolate_worker, task_sampler

# Hypothetical homes are people 10000+; the test people are 23-29 and fillers 100+.
HOME_OFFSET = 10_000


def make_home(home: int, config: DreamConfig, pool: int) -> tuple[np.ndarray, TaskSet]:
    """Home `home`'s person, drawn from b0, and its pool of `pool` goals."""
    seed = HOME_OFFSET + home
    theta = DEFAULT_PRIOR.sample(np.random.default_rng(seed))
    task_rng = np.random.default_rng(seed)
    nominal = task_sampler().sample(task_rng, theta, np.asarray(theta) * START_FRACTION)
    pool_rng = np.random.default_rng(seed + 3)
    tasks = [perturbed_task(nominal, theta, pool_rng, config.variation, vary_init=False)
             for _ in range(pool)]
    return theta, TaskSet(nominal, tasks)


def shard_path(out_dir: Path, shard: int) -> Path:
    return out_dir / f"shard_{shard:02d}.pkl"


def dream(args) -> None:
    """Dream in this shard's homes (every `num_shards`-th, from `shard`)."""
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    isolate_worker()
    select_variant(args.variant)
    SEED.mode = args.seed_mode
    SEED.rollouts = args.seed_rollouts
    mpc = dataclasses.replace(MPCConfig(), num_rollouts=args.num_rollouts)
    if not args.feedback:
        mpc = dataclasses.replace(mpc, feedback_gains=())
    settings = PlannerSettings(max_time=args.dream_max_time, mpc=mpc, hard_time_limit=True)
    config = DreamConfig(
        n_dreams=args.dreams_per_home, fresh_goals=True, vary_geometry=True,
        planner=args.planner, variation=args.variation, particles=args.particles,
        confidence=args.confidence, dream_max_time=args.dream_max_time,
    )
    memory = Memory.build("learned", args.model)
    memory.training = "collect"
    path = shard_path(out_dir, args.shard)
    dreams_out = out_dir / f"shard_{args.shard:02d}_dreams.csv"
    state = {"homes": [], "home": [], "dream": [], "features": [], "labels": [],
             "dreams": 0, "args": vars(args)}
    if path.exists():
        with open(path, "rb") as handle:
            state = pickle.load(handle)
        print(f"[shard {args.shard}] resumed: {len(state['homes'])} homes done", flush=True)

    for home in range(args.shard, args.homes, args.num_shards):
        if home in state["homes"]:
            continue
        theta, pool = make_home(home, config, args.pool)
        belief, _ = observe(Belief.from_prior(DEFAULT_PRIOR), [pool.nominal.limb_init], theta)
        dream_rng = np.random.default_rng(HOME_OFFSET + home + 1)
        particle_rng = np.random.default_rng(HOME_OFFSET + home + 2)
        believed = Believed.build(belief, config, particle_rng)
        memory.learner.features, memory.learner.labels = [], []
        tags = {"home": [], "dream": []}
        for d in range(args.dreams_per_home):
            noise_seed = 10_000_000 + home * 100 + d + 1
            memory.learner.context = {"home": home, "dream": d}
            before = len(memory.learner.features)
            started = time.perf_counter()
            try:
                rows = dream_once(believed, pool, memory, settings, dream_rng, config,
                                  0, noise_seed)
            except Exception as exc:  # pylint: disable=broad-except
                rows = [{"role": "chosen", "solved": False, "goal_reached": False,
                         "failure_reason": f"{type(exc).__name__}: {exc}"}]
                traceback.print_exc()
            added = len(memory.learner.features) - before
            tags["home"] += [home] * added
            tags["dream"] += [d] * added
            for row in rows:
                append_row(dreams_out, {
                    "home": home, "dream": d, "planner": config.planner,
                    "theta": " ".join(f"{v:.4f}" for v in theta),
                    "belief_floor": " ".join(f"{v:.4f}" for v in belief.lower), **row,
                })
            print(f"[shard {args.shard} home {home} dream {d}] "
                  f"{sum(bool(r.get('goal_reached')) for r in rows)}/{len(rows)} reached, "
                  f"{added} rows in {time.perf_counter() - started:.0f}s "
                  f"{rows[-1].get('failure_reason', '')}", flush=True)
        # A home is kept whole or not at all, so a resumed shard redoes a killed home.
        state["features"] += memory.learner.features
        state["labels"] += memory.learner.labels
        state["home"] += tags["home"]
        state["dream"] += tags["dream"]
        state["homes"].append(home)
        state["dreams"] += args.dreams_per_home
        save_checkpoint(path, state)
    uninstall()
    print(f"[shard {args.shard}] done: {len(state['homes'])} homes, "
          f"{len(state['features'])} rows -> {path}", flush=True)


def fit(args) -> None:
    """Train one model on the first homes' rows, up to `dreams` dreams, and pickle it."""
    shards = []
    for path in sorted(args.factory.glob("shard_*.pkl")):
        with open(path, "rb") as handle:
            shards.append(pickle.load(handle))
    per_home = {s["args"]["dreams_per_home"] for s in shards}
    assert len(per_home) == 1, f"shards dreamed different amounts per home: {per_home}"
    per_home = per_home.pop()
    homes = sorted(h for s in shards for h in s["homes"])
    wanted = math.ceil(args.dreams / per_home)
    if len(homes) < wanted:
        raise SystemExit(f"{len(homes)} homes dreamed, {wanted} needed for {args.dreams} dreams")
    keep = set(homes[:wanted])
    learner = OnlineWarmStart(args.model)
    for s in shards:
        for home, x, y in zip(s["home"], s["features"], s["labels"]):
            if home in keep:
                learner.features.append(x)
                learner.labels.append(y)
    learner.prior_dreams = wanted * per_home
    learner.prior_homes = wanted
    mpc = dataclasses.replace(MPCConfig(), num_rollouts=args.num_rollouts)
    if not args.feedback:
        mpc = dataclasses.replace(mpc, feedback_gains=())
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".jsonl")
    log.unlink(missing_ok=True)
    learner.log_path = str(log)
    learner.context = {"homes": wanted, "dreams": learner.prior_dreams}
    started = time.perf_counter()
    learner.refit(mpc)
    learner.log_path = None
    with open(out, "wb") as handle:
        pickle.dump(learner, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{args.model} on {len(learner.features)} rows from {wanted} homes "
          f"({learner.prior_dreams} dreams) in {time.perf_counter() - started:.0f}s -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    # Defaults are the pddlstream_full runs' settings.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--num-rollouts", type=int, default=MPCConfig.num_rollouts)
    common.add_argument("--feedback", action="store_true",
                        help="Keep the MPC's feedback gains (pddlstream_full ran without).")

    d = commands.add_parser("dream", parents=[common])
    d.add_argument("--homes", type=int, required=True)
    d.add_argument("--dreams-per-home", type=int, default=4)
    d.add_argument("--shard", type=int, default=0)
    d.add_argument("--num-shards", type=int, default=1)
    d.add_argument("--pool", type=int, default=20, help="Goals per home.")
    d.add_argument("--planner", choices=("pddlstream", "pouct", "pomcp", "pomcpow"),
                   default="pddlstream")
    d.add_argument("--model", default="gen:flow:hist",
                   help="Only builds the memory; the rows do not depend on it.")
    d.add_argument("--seed-mode", choices=("plan", "step"), default="step")
    d.add_argument("--seed-rollouts", type=int, default=8)
    d.add_argument("--variation", type=float, default=DreamConfig.variation)
    d.add_argument("--particles", type=int, default=DreamConfig.particles)
    d.add_argument("--confidence", type=float, default=DreamConfig.confidence)
    d.add_argument("--dream-max-time", type=float, default=DreamConfig.dream_max_time)
    d.add_argument("--variant", choices=VARIANTS, default="isolated-left-arm")
    d.add_argument("--out-dir", type=Path, required=True)

    f = commands.add_parser("fit", parents=[common])
    f.add_argument("--factory", type=Path, required=True)
    f.add_argument("--dreams", type=int, required=True,
                   help="Dream budget: the first homes up to this many dreams.")
    f.add_argument("--model", default="gen:flow:hist")
    f.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    make_regressor(args.model)  # fail on a bad spec before any planning
    {"dream": dream, "fit": fit}[args.command](args)


if __name__ == "__main__":
    main()
