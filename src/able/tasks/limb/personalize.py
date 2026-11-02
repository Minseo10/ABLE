"""The two personalization pipelines, run round by round so they can be compared.

Both loop the same four steps: ask queries, update the belief, train `psi` on this
person's demonstrations, then plan and execute held-out tasks with what was learned.

- Version 1 starts every person from a factory model pretrained on many people.
- Version 2 starts from random weights, and is given more demonstrations per round to
  make up for it.

Nothing else differs, so the gap between them is what pretraining is worth.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from able.tasks.limb.belief import Belief
from able.tasks.limb.dataset import append_row, guard_variant, load_dataset
from able.tasks.limb.human import DEFAULT_PRIOR
from able.tasks.limb.planner import (
    PlannerSettings,
    QueryPolicy,
    gather_information,
    run_episode,
)
from able.tasks.limb.planning.stream import MPCConfig
from able.tasks.limb.variant import DEFAULT_VARIANT, VARIANTS
from able.tasks.limb.variant import current as current_variant
from able.tasks.limb.variant import select as select_variant
from able.tasks.limb.warmstart import ScaledMLP, WarmStartModel, make_scaled_mlp
from able.tasks.limb.worker import isolate_worker, task_sampler

# Person ids for personalization, clear of the collection sweep and of evaluate.py.
PERSON_OFFSET = 2_000_000


@dataclass(frozen=True)
class LoopConfig:
    """One arm of the comparison: how long the loop runs and how it trains."""

    rounds: int = 5
    tasks_per_round: int = 5
    eval_tasks: int = 5
    num_queries: int = 6
    max_time: float = 600.0
    epochs: int = 50
    learning_rate: float = 1e-4
    # Factory rows mixed into each fine-tune, as a multiple of the person's own rows.
    replay: float = 0.0


def person_theta(person: int) -> NDArray[np.float64]:
    """The true range of motion of one person, fixed across rounds and versions."""
    return DEFAULT_PRIOR.sample(np.random.default_rng(person))


def _query_item(payload) -> tuple:
    """One person's round of queries, run in a worker because the push is simulated."""
    person, theta, belief, floors, num_queries, seed = payload
    gathered = gather_information(
        theta,
        np.random.default_rng(seed),
        QueryPolicy(num_queries=num_queries),
        start=belief,
        floors=floors,
    )
    return person, gathered.posterior, gathered.floors, gathered.num_censored


def _episode_item(payload) -> dict:
    """One planned and executed episode, for a demonstration or for evaluation."""
    person, theta, belief, seed, model, max_time = payload
    isolate_worker()
    rng = np.random.default_rng(seed)
    record = {"person": person, "seed": seed}
    try:
        task = task_sampler().sample(rng, theta)
        result = run_episode(
            task,
            theta,
            belief=belief,
            warm_start=model,
            seed=seed,
            settings=PlannerSettings(max_time=max_time),
        )
        record.update(
            {
                "solved": result.solved,
                "goal_reached": result.goal_reached,
                "plan_time": round(result.plan_time, 2),
                "mpc_simulator_steps": result.mpc_simulator_steps,
                "num_control_steps": result.num_control_steps,
                "num_segments": result.num_segments,
                "rom_violation_steps": result.rom_violation_steps,
                "rom_margin": round(result.rom_margin, 6),
                "separation": round(task.separation, 4),
                "failure_reason": result.failure_reason,
                "segments": result.segments if result.goal_reached else [],
            }
        )
    except Exception as exc:  # pylint: disable=broad-except
        record.update({"solved": False, "goal_reached": False, "segments": [],
                       "failure_reason": f"{type(exc).__name__}: {exc}"})
        traceback.print_exc()
    return record


@dataclass
class PersonState:
    """One person's belief, training rows and model, carried across rounds."""

    person: int
    theta: NDArray[np.float64]
    model: ScaledMLP
    # A pretrained model can already predict; a fresh one cannot until it is trained.
    fitted: bool = False
    belief: Belief = field(default=None)
    floors: NDArray[np.float64] | None = None
    features: list = field(default_factory=list)
    labels: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.belief is None:
            self.belief = Belief.from_prior(DEFAULT_PRIOR)

    @property
    def belief_error(self) -> float:
        return float(np.abs(self.belief.mean - self.theta).max())

    def train(self, factory: tuple | None, replay: float) -> None:
        """Fine-tune on everything collected for this person so far."""
        if not self.features:
            return
        x, y = np.asarray(self.features), np.asarray(self.labels)
        if factory is not None and replay > 0.0:
            fx, fy = factory
            take = min(len(fx), round(replay * len(x)))
            pick = np.random.default_rng(self.person).choice(len(fx), take, replace=False)
            x, y = np.vstack([x, fx[pick]]), np.vstack([y, fy[pick]])
        self.model.train(x, y)
        self.fitted = True

    def warm_start(self, mpc: MPCConfig) -> WarmStartModel | None:
        """The model as the planner takes it, or nothing before the first fit."""
        if not self.fitted:
            return None
        return WarmStartModel(name="mlp", estimator=self.model, mpc=mpc)


def _run_pool(pool, items, label: str) -> list:
    """Map `items` across the pool, printing progress as they land."""
    done = []
    for record in pool.imap_unordered(_episode_item, items):
        done.append(record)
        print(f"  [{len(done)}/{len(items)}] {label} person={record['person']} "
              f"reached={record.get('goal_reached')} plan={record.get('plan_time')}s "
              f"{record.get('failure_reason', '')}", flush=True)
    return done


def run(
    people: list[int],
    make_model,
    config: LoopConfig,
    out_dir: Path,
    version: str,
    workers: int,
    pretrained: bool = False,
    factory: tuple | None = None,
    mpc: MPCConfig | None = None,
) -> None:
    """Run the loop for every person, in lock-step so the pool stays busy."""
    mpc = mpc or MPCConfig()
    states = {p: PersonState(p, person_theta(p), make_model(p), fitted=pretrained)
              for p in people}
    results_path = out_dir / "rounds.csv"
    guard_variant(results_path, current_variant().name)
    with mp.get_context("spawn").Pool(processes=workers) as pool:
        for round_index in range(config.rounds):
            print(f"\n=== {version} round {round_index} ===", flush=True)

            # Stage 1-2: more queries, folded into the belief each person already has.
            queries = [
                (s.person, s.theta, s.belief, s.floors, config.num_queries,
                 s.person * 10_000 + round_index)
                for s in states.values()
            ]
            for person, posterior, floors, _ in pool.imap_unordered(_query_item, queries):
                states[person].belief = posterior
                states[person].floors = floors

            # Demonstrations for this round, planned without a warm start: the label has
            # to be what the slow planner does, not what the model already predicts.
            demos = [
                (s.person, s.theta, s.belief,
                 s.person * 10_000 + round_index * 100 + i, None, config.max_time)
                for s in states.values()
                for i in range(config.tasks_per_round)
            ]
            for record in _run_pool(pool, demos, "demo"):
                state = states[record["person"]]
                for segment in record["segments"]:
                    state.features.append(segment.features)
                    # Flattened, the same shape the warm-start model predicts.
                    state.labels.append(segment.psi.ravel())

            # Stage 3: fine-tune (v1) or keep training from random weights (v2).
            for state in states.values():
                state.train(factory, config.replay)

            # Stage 4: the same held-out tasks every round, so the curve is comparable.
            evals = [
                (s.person, s.theta, s.belief, s.person * 10_000 + 9_000 + i,
                 s.warm_start(mpc), config.max_time)
                for s in states.values()
                for i in range(config.eval_tasks)
            ]
            for record in _run_pool(pool, evals, "eval"):
                state = states[record["person"]]
                record.pop("segments")
                append_row(results_path, {
                    "version": version,
                    "variant": current_variant().name,
                    "round": round_index,
                    "tasks_per_round": config.tasks_per_round,
                    "num_train_rows": len(state.features),
                    "belief_error": round(state.belief_error, 6),
                    **record,
                })
            print(summarize(results_path, round_index), flush=True)

    for state in states.values():
        name = f"model_{version}_person_{state.person}.pkl"
        with open(out_dir / name, "wb") as handle:
            pickle.dump(state.model, handle)
    print(f"\nrounds -> {results_path.resolve()}")


def summarize(results_path: Path, round_index: int | None = None) -> str:
    """One line per (version, round): the metrics the comparison turns on."""
    import pandas as pd

    frame = pd.read_csv(results_path)
    if round_index is not None:
        frame = frame[frame["round"] == round_index]
    header = (f"{'version':<10s} {'round':>5s} {'rows':>5s} {'success':>8s} "
              f"{'plan s':>8s} {'mpc steps':>11s} {'b err':>7s}")
    lines = [header]
    for (version, index), rows in frame.groupby(["version", "round"]):
        reached = rows[rows["goal_reached"] == True]  # noqa: E712
        lines.append(
            f"{version:<10s} {index:>5d} {rows['num_train_rows'].max():>5.0f} "
            f"{len(reached) / len(rows):>7.0%} {rows['plan_time'].mean():>8.0f} "
            f"{(reached['mpc_simulator_steps'].mean() if len(reached) else float('nan')):>11.0f} "
            f"{rows['belief_error'].mean():>7.3f}"
        )
    return "\n".join(lines)


def pretrain(demos_path: Path, out_path: Path, epochs: int, learning_rate: float) -> None:
    """Factory learning: one MLP over every person and task in a collect run."""
    features, labels, metadata = load_dataset(demos_path)
    variant = str(metadata["variant"].iloc[0])
    model = make_scaled_mlp(epochs, learning_rate, variant=variant)
    model.train(features, labels)
    predicted = model.predict(features)
    print(f"pretrained on {len(features)} rows of {variant}, "
          f"train mse {float(np.mean((predicted - labels) ** 2)):.5f}")
    with open(out_path, "wb") as handle:
        pickle.dump(model, handle)
    print(f"model -> {out_path.resolve()}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("pretrain", help="Factory learning, before any person is met.")
    pre.add_argument("--run-dir", type=Path, required=True,
                     help="A collect run directory; demos.csv is read from it.")
    pre.add_argument("--out", type=Path, default=None,
                     help="Where to write the model (default <run-dir>/pretrained.pkl).")
    pre.add_argument("--epochs", type=int, default=2000)
    pre.add_argument("--learning-rate", type=float, default=1e-3)

    loop = sub.add_parser("run", help="Run one arm of the comparison.")
    loop.add_argument("--variant", type=str, default=DEFAULT_VARIANT, choices=VARIANTS)
    loop.add_argument("--version", type=str, required=True,
                      help="Name for this arm in rounds.csv, e.g. v1 or v2_matched.")
    loop.add_argument("--pretrained", type=Path, default=None,
                      help="Factory model to personalize; omit for version 2.")
    loop.add_argument("--factory-demos", type=Path, default=None,
                      help="demos.csv the factory model was fit on; only used by --replay.")
    loop.add_argument("--out-dir", type=Path, default=Path("data/personalize/run"))
    loop.add_argument("--num-people", type=int, default=5)
    loop.add_argument("--rounds", type=int, default=LoopConfig.rounds)
    loop.add_argument("--tasks-per-round", type=int, default=LoopConfig.tasks_per_round)
    loop.add_argument("--eval-tasks", type=int, default=LoopConfig.eval_tasks)
    loop.add_argument("--num-queries", type=int, default=LoopConfig.num_queries)
    loop.add_argument("--max-time", type=float, default=LoopConfig.max_time)
    loop.add_argument("--epochs", type=int, default=LoopConfig.epochs)
    loop.add_argument("--learning-rate", type=float, default=LoopConfig.learning_rate)
    loop.add_argument("--replay", type=float, default=LoopConfig.replay)
    loop.add_argument("--workers", type=int, default=12)

    args = parser.parse_args()
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, "1")

    if args.command == "pretrain":
        pretrain(args.run_dir / "demos.csv",
                 args.out or args.run_dir / "pretrained.pkl",
                 args.epochs, args.learning_rate)
        return

    select_variant(args.variant)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = LoopConfig(
        rounds=args.rounds,
        tasks_per_round=args.tasks_per_round,
        eval_tasks=args.eval_tasks,
        num_queries=args.num_queries,
        max_time=args.max_time,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        replay=args.replay,
    )

    if args.pretrained is not None:
        with open(args.pretrained, "rb") as handle:
            pretrained: ScaledMLP = pickle.load(handle)
        if pretrained.variant and pretrained.variant != args.variant:
            raise SystemExit(
                f"{args.pretrained} was fit on {pretrained.variant!r}, not "
                f"{args.variant!r}; theta means a different limb."
            )
        # Fine-tuning takes smaller steps than pretraining did, on far fewer rows.
        pretrained.net.set_params(max_iter=config.epochs,
                                  learning_rate_init=config.learning_rate)
        # A copy per person: one person's fine-tuning must not touch another's.
        def make_model(person: int) -> ScaledMLP:  # pylint: disable=unused-argument
            return pickle.loads(pickle.dumps(pretrained))
    else:
        def make_model(person: int) -> ScaledMLP:
            return make_scaled_mlp(
                config.epochs, config.learning_rate, seed=person, variant=args.variant
            )

    factory = None
    if args.factory_demos is not None:
        features, labels, _ = load_dataset(args.factory_demos)
        factory = (features, labels)

    people = [PERSON_OFFSET + i for i in range(args.num_people)]
    run(people, make_model, config, args.out_dir, args.version, args.workers,
        pretrained=args.pretrained is not None, factory=factory)
    print("\n" + summarize(args.out_dir / "rounds.csv"))


if __name__ == "__main__":
    main()
