"""Stage B: property checks on one task's end-to-end runs (layout from scripts/check/run.py).

    PYTHONPATH=src python scripts/check/e2e_validate.py --task limb --dir D --config configs/limb/check.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = ("task", "condition", "person", "episode", "success", "plan_seconds", "effort", "queries", "belief")
MAIN = ("plain", "learned", "prior_only")


def runs(episodes: int) -> dict[str, tuple[str, str, int, int]]:
    """name -> (csv, condition, person, episodes)."""
    return {"plain": ("main/plain_p0.csv", "plain", 0, episodes),
            "learned": ("main/learned_p0.csv", "learned", 0, episodes),
            "prior_only": ("main/prior_only_p0.csv", "prior_only", 0, episodes),
            "no_info": ("no_info/plain_p1.csv", "plain", 1, 1),
            "no_belief": ("no_belief/plain_p2.csv", "plain", 2, 1),
            "resume": ("resume/plain_p3.csv", "plain", 3, 2)}


def read(path: Path) -> list[dict]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


class Checks:
    def __init__(self, task: str, d: Path, config: Path) -> None:
        from able.pipeline import PipelineConfig
        from able.run import load_config, make_task

        self.name, self.d = task, d
        options, pipeline = load_config(config)
        self.config = PipelineConfig(**pipeline)
        self.task = make_task(options.pop("name"), **options)
        self.runs = runs(self.config.episodes)
        self.data = {k: read(d / v[0]) if (d / v[0]).exists() else None for k, v in self.runs.items()}
        self.results: list[dict] = []

    def prior(self, person: int, seed: int = 0):
        """The belief run_person starts from."""
        rng = np.random.default_rng([person, seed])
        self.task.prior.sample(rng)
        return self.task.prior.belief(rng)

    def spread(self, summary: dict) -> np.ndarray:
        """Per-dimension uncertainty of a belief summary (limb: the truncated Gaussian's std)."""
        if self.name == "limb":
            from scipy.stats import truncnorm

            mean, std, lower = (np.asarray(summary[k]) for k in ("mean", "std", "lower"))
            high = self.task.prior.prior.high
            return truncnorm((lower - mean) / std, (high - mean) / std, loc=mean, scale=std).std()
        if self.name == "friction":
            return np.array([v[1] for _, v in sorted(summary.items())])
        return np.asarray(summary["std"])

    def check(self, name: str, fn) -> None:
        started = time.time()
        try:
            out = fn()
        except Exception as error:  # noqa: BLE001
            out = ("FAIL", f"{type(error).__name__}: {error}")
        status, note = out if isinstance(out, tuple) else (("PASS", "") if out is True else ("FAIL", str(out)))
        self.results.append({"check": name, "status": status, "seconds": round(time.time() - started, 1),
                             "note": note[:160]})
        print(f"{name}: {status} {note}".rstrip(), flush=True)

    def rows(self, run: str) -> list[dict]:
        rows = self.data[run]
        if rows is None:
            raise FileNotFoundError(f"no {self.runs[run][0]}")
        return rows

    # ---- checks

    def csvs(self):
        problems = []
        for run, (path, condition, person, n) in self.runs.items():
            rows = self.data[run]
            if rows is None:
                problems.append(f"{path} missing")
                continue
            missing = [c for c in COLUMNS if c not in rows[0]] if rows else list(COLUMNS)
            if len(rows) != n or missing:
                problems.append(f"{path}: {len(rows)}/{n} rows, missing {missing}")
                continue
            for i, r in enumerate(rows):
                want = (self.name, condition, str(person), str(i))
                got = (r["task"], r["condition"], r["person"], r["episode"])
                if got != want:
                    problems.append(f"{path} row {i}: {got} != {want}")
                if not float(r["effort"]) > 0 or not float(r["plan_seconds"]) > 0:
                    problems.append(f"{path} row {i}: effort {r['effort']}, plan_seconds {r['plan_seconds']}")
        return not problems or "; ".join(problems)

    def success(self):
        wins = {run: sum(r["success"] == "True" for r in self.data[run] or []) for run in MAIN}
        if sum(wins.values()):
            return "PASS", f"successes {wins}"
        return "FAIL", f"no success in {MAIN}"

    def learned(self):
        rows = self.rows("learned")
        if any(r.get("dreams_solved", "") == "" or not float(r.get("dream_seconds") or 0) > 0 for r in rows):
            return "dreams_solved/dream_seconds missing on some row"
        fitted = [int(r["rows"]) for r in rows if r.get("rows")]
        if not fitted or max(fitted) <= 0:
            return f"no psi rows added or fit (dreams_solved {[r['dreams_solved'] for r in rows]})"
        return "PASS", f"dreams_solved {[r['dreams_solved'] for r in rows]}, rows {fitted}"

    def prior_only(self):
        import torch

        self.rows("prior_only")
        factory = pickle.loads((self.d / "factory.pkl").read_bytes())
        memory = pickle.loads((self.d / "main" / "prior_only_p0.pkl").read_bytes())["memory"]
        if memory.learner is None or memory.x:
            return f"learner {type(memory.learner).__name__}, {len(memory.x)} own rows (want factory's, 0)"
        for a, b in zip(factory.modules(), memory.learner.modules(), strict=True):
            sa, sb = a.state_dict(), b.state_dict()
            if sa.keys() != sb.keys() or not all(torch.equal(sa[k], sb[k]) for k in sa):
                return "prior_only's learner is not the factory model"
        return True

    def info_gathering(self):
        prior = self.spread(self.prior(0).summary())
        notes = []
        for run in MAIN:
            rows = self.rows(run)
            queries = [int(r["queries"]) for r in rows]
            if any(q != self.config.queries for q in queries):
                return f"{run}: queries {queries}, want {self.config.queries} each"
            after = self.spread(json.loads(rows[0]["belief"]))
            if not np.any(after < prior - 1e-9):
                return f"{run}: belief after episode 0 not more certain (std {after.round(4)} vs {prior.round(4)})"
            notes.append(f"{run} {np.round(after / prior, 2).tolist()}")
        return "PASS", "std/prior: " + "; ".join(notes)

    def no_info(self):
        queries = [int(r["queries"]) for r in self.rows("no_info")]
        return all(q == 0 for q in queries) or f"queries {queries}"

    def no_belief(self):
        rows, prior = self.rows("no_belief"), self.prior(2).summary()
        queries = [int(r["queries"]) for r in rows]
        if any(queries):
            return f"queries {queries}"
        for i, r in enumerate(rows):
            if json.loads(r["belief"]) != json.loads(json.dumps(prior)):
                return f"row {i}: belief {r['belief'][:60]} != prior {json.dumps(prior)[:60]}"
        return True

    def resume(self):
        rows, first = self.rows("resume"), read(self.d / "resume" / "first.csv")
        if len(first) != 1 or len(rows) != 2:
            return f"{len(first)} then {len(rows)} rows, want 1 then 2"
        bad = [k for k, v in first[0].items() if "seconds" not in k and rows[0].get(k) != v]
        return not bad or f"row 0 changed on resume: {bad}"

    def video(self):
        rows = self.rows("learned")
        return rows[0].get("video") == "True" or f"learned row 0 video={rows[0].get('video')!r}"

    def figures(self):
        out = self.d / "figures"
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "figures.py"), str(self.d / "main"),
                               "--out", str(out)], capture_output=True, text=True, check=False)
        if proc.returncode or not (out / "curves.png").exists():
            return f"figures.py exit {proc.returncode}: {(proc.stderr.strip().splitlines() or [''])[-1]}"
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    c = Checks(args.task, args.dir, args.config)
    for name, fn in (("csv rows/columns", c.csvs), ("some success", c.success), ("learned dreams+fit", c.learned),
                     ("prior_only uses factory g", c.prior_only), ("info gathering", c.info_gathering),
                     ("--no-info-gathering", c.no_info), ("--no-belief-update", c.no_belief),
                     ("resume", c.resume), ("video", c.video), ("figures.py", c.figures)):
        c.check(name, fn)
    if args.report:
        args.report.write_text(json.dumps(c.results))
    sys.exit(any(r["status"] == "FAIL" for r in c.results))


if __name__ == "__main__":
    main()
