"""Stage A: fixed-seed fingerprints of a task's components vs tests/baselines/<host>/<task>.json.

    PYTHONPATH=src python scripts/check/baseline.py --task limb [--update] [--report out.json]

Ints, strings, bools and hashes must match exactly; floats within rtol 1e-6, atol 1e-9.
"""

from __future__ import annotations

import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":  # fixed set/dict-of-str order
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")  # before numpy/torch load
os.environ.setdefault("WANDB_MODE", "disabled")

import argparse
import hashlib
import json
import math
import random
import socket
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RTOL, ATOL = 1e-6, 1e-9


def seed_all() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)


def host() -> str:
    return socket.gethostname().split(".")[0]


def jsonable(v):
    """Plain JSON types (numpy -> python, tuples -> lists)."""
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return jsonable(v.tolist())
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def md5(a) -> str:
    return hashlib.md5(np.asarray(a, np.float64).round(10).tobytes()).hexdigest()


# ---- limb


def limb():
    from able.api import Budget, make_learner, run_episode
    from able.tasks.limb.task import LimbTask

    task = LimbTask(variant="human-left-arm")
    rng = np.random.default_rng(0)
    belief = task.prior.belief(rng)
    query = task.gatherer.choose(belief, rng)
    obs = task.gatherer.respond(np.array([0.85, 1.05, 0.95]), query, rng)
    o = obs.value
    yield "query", {"motion": query.motion_index, "target": query.target, "answer": o.answer,
                    "reached_target": o.reached_target, "settled": o.settled,
                    "posterior": belief.update(obs).summary()}

    planner = task.make_planner()
    rng = np.random.default_rng(3)
    problem, theta = task.sample_problem(rng), np.ones(3)
    belief = task.prior.belief(rng)

    def episode_fp(ep):
        last = ep.steps[-1]
        segs = last.step.info.get("segments") if last.step else None
        torques = segs[0][2].robot_torques if segs else []
        return {"success": ep.success, "effort": ep.effort, "steps": len(ep.steps),
                "seeded_steps": last.result.info.get("seeded_steps"),
                "full_steps": last.result.info.get("full_steps"),
                "torques": len(torques), "torque_md5": md5(torques),
                "belief_after": ep.belief.summary()}

    plain = run_episode(task, planner, problem, theta, belief, np.random.default_rng(11), budget=Budget(400))
    yield "planning", {"problem": problem.as_features(), **episode_fp(plain)}

    rows = task.psi.examples(plain, belief)
    yield "psi", {"rows": len(rows), "sum_x": rows.x.sum(), "sum_y": rows.y.sum(),
                  "groups": len(np.unique(rows.groups))}

    if not len(rows):
        raise RuntimeError("plain episode gave no psi rows to fit")
    learner = make_learner("flow", steps=300).fit(rows.x, rows.y, rows.groups)
    fit = float(learner.predict(rows.x[:4]).sum())
    torch.manual_seed(0)
    seeded = run_episode(task, planner, problem, theta, belief, np.random.default_rng(11),
                         lambda s, b: task.psi.proposal(learner, s, b), Budget(400))
    yield "seeded", {"fit_predict_sum": fit, **episode_fp(seeded)}


# ---- friction


def friction():
    from able.api import run_episode
    from able.planners.pomcpow import POMCPOWConfig
    from able.tasks.friction.task import FrictionTask

    task = FrictionTask(env="2d", level="L2", num_particles=200, max_pushes=4,
                        config=POMCPOWConfig(num_sims=200, max_depth=8))
    theta = np.array([0.08, 0.15, 0.03])
    rng = np.random.default_rng(0)
    belief = task.prior.belief(rng)
    query = task.gatherer.choose(belief, rng)
    obs = task.gatherer.respond(theta, query, rng)
    yield "query", {"prior": belief.summary(), "floor": query.floor, "puck": query.puck, "angle": query.angle,
                    "speed": query.speed, "record": obs.value, "posterior": belief.update(obs).summary()}

    rng = np.random.default_rng(5)
    problem = task.sample_problem(rng)
    belief = task.prior.belief(rng)
    ep = run_episode(task, task.make_planner(), problem, theta, belief, rng)
    steps = []
    for s in ep.steps:
        push = s.result.plan
        steps.append({"action": None if push is None else [push.angle, push.speed, push.stance],
                      "sims": s.result.effort, "belief": s.belief.summary(),
                      "stop": s.step.info["stop"] if s.step else None,
                      "goal_reached": s.step.info["goal_reached"] if s.step else None})
    yield "episode", {"problem": problem.seed, "steps": steps, "effort": ep.effort, "success": ep.success,
                      "belief_after": ep.belief.summary()}

    rows = task.psi.examples(ep, ep.belief)
    yield "psi", {"rows": len(rows), "sum_x": rows.x.sum(), "sum_y": rows.y.sum()}


# ---- bedside


def bedside():
    from able.tasks.bedside.task import BedsideTask

    task = BedsideTask(num_requested=2, num_clutter=0)
    rng = np.random.default_rng(3)
    problem = task.sample_problem(rng)
    belief = task.prior.belief(rng)
    yield "problem", {"seed": problem.seed, "prior": belief.summary()}

    theta = np.array([0.5, 0.75, 0.85])
    rng = np.random.default_rng(0)
    query = task.gatherer.choose(belief, rng)
    k = int(np.argmin(np.linalg.norm(task.gatherer.directions - query.direction, axis=1)))
    obs = task.gatherer.respond(theta, query, rng)
    outcomes = [[[round(v, 6) for v in point], reached] for point, reached in obs.value]
    yield "query", {"direction": k, "distances": list(query.distances), "outcomes": outcomes,
                    "posterior": belief.update(obs).summary()}

    # PDDLStream is timing-dependent: only whether it solves and executes.
    rng = np.random.default_rng(11)
    state = task.reset(problem, theta, rng)
    result = task.make_planner().plan(state, belief, rng=rng)
    executed = bool(result.success and task.execute(state, result.plan, rng).success)
    yield "planning", {"solved": result.success, "executed": executed}


TASKS = {"limb": limb, "friction": friction, "bedside": bedside}
# Sections that must also hold regardless of the baseline.
REQUIRED = {("bedside", "planning"): lambda fp: all(fp.values()) or f"not solved and executed: {fp}"}


# ---- comparison


def flatten(v, prefix="") -> dict:
    if isinstance(v, dict):
        out = {}
        for k in sorted(v):
            out |= flatten(v[k], f"{prefix}.{k}" if prefix else k)
        return out
    if isinstance(v, list):
        out = {f"{prefix}#len": len(v)}
        for i, x in enumerate(v):
            out |= flatten(x, f"{prefix}[{i}]")
        return out
    return {prefix: v}


def same(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return type(a) is type(b) and a == b
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)


def diff(expected, got) -> list[tuple[str, object, object]]:
    e, g = flatten(expected), flatten(got)
    return [(k, e.get(k, "<missing>"), g.get(k, "<missing>")) for k in sorted(set(e) | set(g))
            if k not in e or k not in g or not same(e[k], g[k])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--update", action="store_true", help="write this machine's baseline")
    parser.add_argument("--report", type=Path, help="JSON list of check results")
    parser.add_argument("--baselines", type=Path, default=ROOT / "tests" / "baselines")
    args = parser.parse_args()

    path = args.baselines / host() / f"{args.task}.json"
    baseline = json.loads(path.read_text()) if path.exists() else None
    seed_all()
    results, fingerprint, sections = [], {}, TASKS[args.task]()
    while True:
        started = time.time()
        try:
            name, fp = next(sections)
        except StopIteration:
            break
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            results.append({"check": "error", "status": "FAIL", "seconds": time.time() - started,
                            "note": traceback.format_exc().strip().splitlines()[-1][:120]})
            break
        fp = fingerprint[name] = jsonable(fp)
        status, note = "PASS", ""
        required = REQUIRED.get((args.task, name), lambda _: True)(fp)
        if required is not True:
            status, note = "FAIL", required
        elif args.update:
            note = "baseline written"
        elif baseline is None or name not in baseline:
            status, note = "SKIP", f"no baseline for {host()}: run scripts/check.sh --save-baseline"
        else:
            bad = diff(baseline[name], fp)
            if bad:
                status, note = "FAIL", f"{len(bad)} keys differ: {', '.join(k for k, _, _ in bad[:3])}"
                print(f"[{args.task}.{name}] {len(bad)} differing keys (expected -> got):")
                for k, e, g in bad[:40]:
                    print(f"  {k}: {e!r} -> {g!r}")
        results.append({"check": name, "status": status, "seconds": round(time.time() - started, 1), "note": note})
        print(f"[{args.task}.{name}] {status} {note}".rstrip(), flush=True)

    if args.update and all(r["status"] == "PASS" for r in results):
        if baseline is not None:
            changed = {s: len(diff(baseline.get(s, {}), fp)) for s, fp in fingerprint.items()}
            print("vs old baseline, differing keys per section:", changed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fingerprint, sort_keys=True, indent=1) + "\n")
        print(f"wrote {path}")
    elif args.update:
        for r in results:
            if r["status"] == "PASS":
                r["note"] = "not written: a section failed"
    if args.report:
        args.report.write_text(json.dumps(results))
    sys.exit(any(r["status"] == "FAIL" for r in results))


if __name__ == "__main__":
    main()
