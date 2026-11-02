"""Command line for every task.

    python -m able.run person  --config configs/friction/full.yaml --condition learned --person 3 --out data/friction/full/learned_p3
    python -m able.run factory --config configs/friction/full.yaml --dreams 25 --seed 0 --out data/friction/full/factory/rows_0.pkl
    python -m able.run fit --config configs/friction/full.yaml --rows data/friction/full/factory/rows_*.pkl --out data/friction/full/factory.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import yaml

from able.pipeline import (
    CONDITIONS,
    PipelineConfig,
    factory_rows,
    fit_factory,
    run_person,
)


def make_task(name: str, **options):
    """Build a task from config options."""
    if name == "limb":
        from able.tasks.limb.episode import PlannerSettings
        from able.tasks.limb.task import LimbTask

        max_time = options.pop("max_time", None)
        settings = PlannerSettings(max_time=max_time) if max_time else PlannerSettings()
        return LimbTask(settings=settings, **options)
    if name == "friction":
        from able.planners.pomcpow import POMCPOWConfig
        from able.tasks.friction.task import FrictionTask

        search = {k: options.pop(k) for k in ("num_sims", "max_depth", "max_time") if k in options}
        return FrictionTask(config=POMCPOWConfig(**{"max_depth": 8, **search}), **options)
    if name == "bedside":
        from able.tasks.bedside.task import BedsideTask

        return BedsideTask(**options)
    raise ValueError(f"unknown task {name!r}")


def load_config(path: Path) -> tuple[dict, dict]:
    """(task section, pipeline section) of a YAML config."""
    data = yaml.safe_load(Path(path).read_text())
    return {"name": data["task"], **data.get("options", {})}, data.get("pipeline", {})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("person", "factory", "fit"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, default="learned")
    parser.add_argument("--person", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--dreams", type=int, help="per episode (person) or total (factory)")
    parser.add_argument("--queries", type=int)
    parser.add_argument("--prior-model", type=Path, help="factory pickle, for prior_only")
    parser.add_argument("--rows", type=Path, nargs="*", help="factory shards, for fit")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    task_options, pipeline = load_config(args.config)
    for key in ("episodes", "queries"):
        if getattr(args, key) is not None:
            pipeline[key] = getattr(args, key)
    if args.dreams is not None and args.command == "person":
        pipeline["dreams"] = args.dreams
    config = PipelineConfig(**{**pipeline, "condition": args.condition})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "fit":
        shards = [pickle.loads(p.read_bytes()) for p in args.rows]
        args.out.write_bytes(pickle.dumps(fit_factory(shards, config)))
        return
    task = make_task(task_options.pop("name"), **task_options)
    if args.command == "factory":
        args.out.write_bytes(pickle.dumps(factory_rows(task, config, args.dreams or 25, args.seed)))
        return
    if args.condition == "prior_only" and args.prior_model is None:
        parser.error("prior_only needs --prior-model")
    run_person(task, config, args.person, args.out, args.seed, args.prior_model, args.resume)


if __name__ == "__main__":
    main()
