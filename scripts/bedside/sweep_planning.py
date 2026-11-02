"""Planning-time sweep for BedsideStocking3D: requested items x clutter items x seeds.

Each run is a subprocess of able.tasks.bedside.planner.run that writes
one JSON file. Runs whose JSON already exists are skipped, so the sweep can be resumed.
A run is killed if its memory exceeds --max-rss-gb (recorded as killed="memory").

Usage:
  python scripts/bedside/sweep_planning.py --out data/bedside/planning_sweep
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil

SRC = str(Path(__file__).resolve().parents[2] / "src")
PYTHONPATH = ":".join(p for p in (SRC, os.environ.get("PYTHONPATH")) if p)


def _rss(ps: psutil.Process) -> int:
    total = 0
    for p in [ps] + ps.children(recursive=True):
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total


def run_one(r: int, c: int, seed: int, args) -> dict:
    out = args.out / f"r{r}_c{c}_s{seed}.json"
    if out.exists():
        return json.loads(out.read_text())
    cmd = [
        sys.executable,
        "-m",
        "able.tasks.bedside.planner.run",
        "--num-requested", str(r),
        "--num-clutter", str(c),
        "--seed", str(seed),
        "--max-time", str(args.max_time),
        "--json-out", str(out),
    ]
    if args.execute:
        cmd.append("--execute")
    if args.true_theta:
        cmd.append("--true-theta")
    env = {**os.environ, "PYTHONPATH": PYTHONPATH, "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    log = args.out / "logs" / f"r{r}_c{c}_s{seed}.log"
    # pddlstream writes FastDownward files to ./temp, so each run needs its own cwd.
    work = args.out / "work" / out.stem
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    killed, peak = None, 0
    with open(log, "w") as f:
        proc = subprocess.Popen(cmd, env=env, cwd=work, stdout=f, stderr=subprocess.STDOUT)
        ps = psutil.Process(proc.pid)
        while proc.poll() is None:
            time.sleep(2)
            rss = _rss(ps)
            peak = max(peak, rss)
            if rss > args.max_rss_gb * 2**30:
                killed = "memory"
            elif time.time() - t0 > args.max_time + 300:
                killed = "timeout"
            if killed:
                for p in ps.children(recursive=True) + [ps]:
                    try:
                        p.kill()
                    except psutil.Error:
                        pass
                proc.wait()
                break
    if out.exists() and killed is None:
        result = json.loads(out.read_text())
    else:
        result = {"num_requested": r, "num_clutter": c, "seed": seed, "plan_found": False,
                  "plan_time": time.time() - t0, "plan_length": None, "num_moves": None,
                  "executed": None, "crashed": killed is None, "killed": killed}
    result["peak_rss_gb"] = round(peak / 2**30, 2)
    out.write_text(json.dumps(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/bedside/planning_sweep"))
    parser.add_argument("--requested", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--clutter", type=int, nargs="+", default=[0, 2, 4, 6])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-time", type=float, default=600.0)
    parser.add_argument("--max-jobs", type=int, default=5)
    parser.add_argument("--max-rss-gb", type=float, default=4.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--true-theta", action="store_true", help="plan with the true theta, not the prior")
    args = parser.parse_args()
    args.out = args.out.resolve()
    (args.out / "logs").mkdir(parents=True, exist_ok=True)
    jobs = [(r, c, s) for r in args.requested for c in args.clutter for s in range(args.seeds)]
    jobs.sort(key=lambda j: -(j[0] + j[1]))  # slowest first
    with ThreadPoolExecutor(args.max_jobs) as pool:
        for res in pool.map(lambda j: run_one(*j, args), jobs):
            print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
