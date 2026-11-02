"""Post-merge check: Stage A (baseline fingerprints) and Stage B (end-to-end runs), in parallel.

Run it through scripts/check.sh (sets PYTHONPATH, threads and W&B off); see its header.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / "scripts" / "check"
TASKS = ("limb", "friction", "bedside")
# Measured seconds per job (32 cores, 12 jobs); longest chains start first.
COST = {
    "limb": {"baseline": 220, "factory": 100, "fit": 20, "plain": 245, "learned": 355, "prior_only": 200,
             "no_info": 75, "no_belief": 80, "resume_1": 70, "resume_2": 50, "validate": 15},
    "friction": {"baseline": 10, "factory": 20, "fit": 2, "plain": 45, "learned": 95, "prior_only": 45,
                 "no_info": 22, "no_belief": 22, "resume_1": 12, "resume_2": 23, "validate": 10},
    "bedside": {"baseline": 40, "factory": 35, "fit": 2, "plain": 70, "learned": 140, "prior_only": 70,
                "no_info": 35, "no_belief": 35, "resume_1": 55, "resume_2": 40, "validate": 15},
}


@dataclass
class Job:
    name: str
    stage: str
    task: str
    check: str
    cmd: list[str]
    cost: float
    deps: tuple[str, ...] = ()  # must pass first
    waits: tuple[str, ...] = ()  # must finish first, pass or not
    report: Path | None = None  # JSON list of sub-checks the job writes
    then: Callable[[], None] | None = None  # after a clean exit
    timeout: float = 900.0
    proc: subprocess.Popen | None = None
    started: float = 0.0
    rows: list[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.rows) and all(r["status"] in ("PASS", "WARN") for r in self.rows)


def stage_a(task: str, root: Path, py: str, update: bool) -> list[Job]:
    report = root / "reports" / f"baseline_{task}.json"
    cmd = [py, str(CHECK / "baseline.py"), "--task", task, "--report", str(report)] + (["--update"] if update else [])
    return [Job(f"baseline_{task}", "A", task, "baseline", cmd, COST[task]["baseline"], report=report)]


def stage_b(task: str, root: Path, py: str) -> list[Job]:
    d, cost = root / "B" / task, COST[task]
    config = str(ROOT / "configs" / task / "check.yaml")
    run = [py, "-m", "able.run"]

    def person(cond: str, p: int, sub: str, *extra: str) -> list[str]:
        return run + ["person", "--config", config, "--condition", cond, "--person", str(p),
                      "--out", str(d / sub / f"{cond}_p{p}"), *extra]

    def snapshot() -> None:
        shutil.copy(d / "resume" / "plain_p3.csv", d / "resume" / "first.csv")

    n = f"{task}_"
    shards = [Job(f"{n}factory_{k}", "B", task, f"run factory {k}",
                  run + ["factory", "--config", config, "--dreams", "1", "--seed", str(k),
                         "--out", str(d / "factory" / f"rows_{k}.pkl")], cost["factory"]) for k in range(3)]
    jobs = shards + [
        Job(f"{n}fit", "B", task, "run fit", run + ["fit", "--config", config, "--rows",
            *[str(d / "factory" / f"rows_{k}.pkl") for k in range(3)], "--out", str(d / "factory.pkl")],
            cost["fit"], deps=tuple(j.name for j in shards)),
        Job(f"{n}plain", "B", task, "run plain p0", person("plain", 0, "main"), cost["plain"]),
        Job(f"{n}learned", "B", task, "run learned p0 (+video)",
            person("learned", 0, "main", "--wandb", "check", "--video-every", "2"), cost["learned"]),
        Job(f"{n}prior_only", "B", task, "run prior_only p0",
            person("prior_only", 0, "main", "--prior-model", str(d / "factory.pkl")), cost["prior_only"],
            deps=(f"{n}fit",)),
        Job(f"{n}no_info", "B", task, "run --no-info-gathering p1",
            person("plain", 1, "no_info", "--episodes", "1", "--no-info-gathering"), cost["no_info"]),
        Job(f"{n}no_belief", "B", task, "run --no-belief-update p2",
            person("plain", 2, "no_belief", "--episodes", "1", "--no-belief-update"), cost["no_belief"]),
        Job(f"{n}resume_1", "B", task, "run resume p3 (1 ep)", person("plain", 3, "resume", "--episodes", "1"),
            cost["resume_1"], then=snapshot),
        Job(f"{n}resume_2", "B", task, "run resume p3 (+1 ep)",
            person("plain", 3, "resume", "--episodes", "2", "--resume"), cost["resume_2"], deps=(f"{n}resume_1",)),
    ]
    report = root / "reports" / f"e2e_{task}.json"
    jobs.append(Job(f"{n}validate", "B", task, "validate",
                    [py, str(CHECK / "e2e_validate.py"), "--task", task, "--dir", str(d), "--config", config,
                     "--report", str(report)], cost["validate"], waits=tuple(j.name for j in jobs), report=report))
    return jobs


def chains(jobs: dict[str, Job]) -> dict[str, float]:
    """Longest cost path from each job to the end."""
    after: dict[str, list[str]] = {name: [] for name in jobs}
    for j in jobs.values():
        for d in j.deps + j.waits:
            after[d].append(j.name)
    memo: dict[str, float] = {}

    def length(name: str) -> float:
        if name not in memo:
            memo[name] = jobs[name].cost + max((length(a) for a in after[name]), default=0.0)
        return memo[name]

    return {name: length(name) for name in jobs}


def failure_note(log: Path, code: int | None) -> str:
    text = log.read_text(errors="replace") if log.exists() else ""
    if "no dream solved" in text:
        return "no factory dream solved: nothing to fit"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and "b3Warning" not in ln]
    errors = [ln for ln in lines if "Error" in ln or "Exception" in ln or ln.startswith("failed")]
    last = (errors or lines or [""])[-1]
    return f"{'timed out' if code is None else f'exit {code}'}: {last[:90]}"


def finish(job: Job, code: int | None, log: Path) -> None:
    seconds = round(time.time() - job.started, 1)
    if code == 0 and job.then is not None:
        try:
            job.then()
        except Exception as error:  # noqa: BLE001
            code = 1
            log.open("a").write(f"\npost-step failed: {error!r}\n")
    sub = []
    if job.report is not None and job.report.exists():
        sub = json.loads(job.report.read_text())
    if sub:
        job.rows = [{"check": r["check"], "status": r["status"], "seconds": r.get("seconds", 0.0),
                     "note": r.get("note", "")} for r in sub]
        if code != 0 and all(r["status"] != "FAIL" for r in job.rows):
            job.rows.append({"check": job.check, "status": "FAIL", "seconds": seconds,
                             "note": failure_note(log, code)})
    else:
        job.rows = [{"check": job.check, "status": "PASS" if code == 0 else "FAIL", "seconds": seconds,
                     "note": "" if code == 0 else failure_note(log, code)}]


def schedule(jobs: dict[str, Job], limit: int, root: Path, env: dict) -> None:
    priority = chains(jobs)
    pending, running, done = dict(jobs), {}, {}
    try:
        while pending or running:
            for name, job in list(running.items()):
                code = job.proc.poll()
                if code is None and time.time() - job.started > job.timeout:
                    os.killpg(job.proc.pid, signal.SIGKILL)
                    job.proc.wait()
                elif code is None:
                    continue
                finish(job, code, root / "logs" / f"{name}.log")
                done[name] = running.pop(name)
            for name, job in list(pending.items()):
                if not all(d in done for d in job.deps + job.waits):
                    continue
                blocked = [d for d in job.deps if not done[d].passed]
                if blocked:
                    job.rows = [{"check": job.check, "status": "SKIP", "seconds": 0.0,
                                 "note": f"needs {', '.join(blocked)}"}]
                    done[name] = pending.pop(name)
            ready = sorted((j for j in pending.values() if all(d in done for d in j.deps + j.waits)),
                           key=lambda j: -priority[j.name])
            for job in ready[: max(limit - len(running), 0)]:
                work = root / "work" / job.name
                work.mkdir(parents=True)
                job.started = time.time()
                with open(root / "logs" / f"{job.name}.log", "w") as log:  # the child keeps its own copy
                    job.proc = subprocess.Popen(job.cmd, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                start_new_session=True)
                running[job.name] = pending.pop(job.name)
            time.sleep(0.5)
    except BaseException:
        for job in running.values():
            os.killpg(job.proc.pid, signal.SIGKILL)
        raise


def table(jobs: list[Job]) -> str:
    rows = [(j.stage, j.task, r["check"], r["status"], f"{r['seconds']:.1f}", r["note"]) for j in jobs for r in j.rows]
    head = ("stage", "task", "check", "status", "seconds", "note")
    width = [max(len(str(r[i])) for r in rows + [head]) for i in range(5)]
    fmt = "  ".join(f"{{:{'>' if i == 4 else '<'}{w}}}" for i, w in enumerate(width)) + "  {}"
    return "\n".join(fmt.format(*r).rstrip() for r in [head] + rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=" ".join(TASKS))
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    tasks = args.tasks.split()
    if unknown := set(tasks) - set(TASKS):
        parser.error(f"unknown tasks {sorted(unknown)}")
    limit = int(os.environ.get("JOBS", "12"))
    root = Path(tempfile.mkdtemp(prefix="able-check-"))
    for sub in ("logs", "work", "reports", "tmp"):
        (root / sub).mkdir()
    env = {**os.environ, "WANDB_MODE": "disabled", "PYTHONHASHSEED": "0", "TMPDIR": str(root / "tmp"),
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "MPLCONFIGDIR": str(root / "tmp")}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT / "src"), env.get("PYTHONPATH")) if p)
    py = sys.executable
    order = [j for t in tasks for j in stage_a(t, root, py, args.save_baseline)]
    order += [j for t in tasks for j in stage_b(t, root, py)]
    jobs = {j.name: j for j in order}
    print(f"check: {len(jobs)} jobs, at most {limit} at once; logs in {root / 'logs'}", flush=True)
    started = time.time()
    schedule(jobs, limit, root, env)
    wall = time.time() - started
    print(table(order))
    for job in order:  # baseline diffs (expected -> got)
        if job.stage == "A" and any(r["status"] == "FAIL" for r in job.rows):
            text = (root / "logs" / f"{job.name}.log").read_text(errors="replace").splitlines()
            print("\n".join(ln for ln in text if "differing keys" in ln or ln.startswith("  ") and " -> " in ln))
    failed = any(r["status"] == "FAIL" for j in order for r in j.rows)
    print(f"\nwall: {wall / 60:.1f} min; {'FAIL' if failed else 'all passed'}")
    if failed or args.keep:
        print(f"logs: {root / 'logs'} (kept)")
    else:
        shutil.rmtree(root, ignore_errors=True)
        print(f"logs: {root / 'logs'} (deleted; --keep keeps them)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
