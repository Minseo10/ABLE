"""Videos of complexity-sweep episodes, replayed deterministically from (env, level, sims, seed).

    # One episode:
    python scripts/friction/videos.py one --env 2d --level L2 --sims 1000 --seed 0
    # Pick a success and a failure per (env, level) from the sweep and list them:
    python scripts/friction/videos.py select
    # Add the floor-friction legend to videos rendered before it existed:
    python scripts/friction/videos.py legend
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from able.planners.pomcpow import POMCPOWConfig
from able.tasks.friction import FrictionTask
from able.tasks.friction.env.episode import run
from able.tasks.friction.env.video import friction_legend

DATA = Path("data/friction/complexity")
OUT = Path("videos/friction/complexity")
FPS = 20


def record(env_name: str, level: str, sims: int, seed: int, out: Path) -> str:
    """Replay one episode while recording; returns its outcome."""
    task = FrictionTask(env=env_name, level=level, config=POMCPOWConfig(num_sims=sims, max_depth=8))
    recorder = task.recorder
    recorder.header = f"{env_name.upper()} {level} | POMCPOW {sims} sims | seed {seed}"
    recorder.fps = FPS
    frames: list[np.ndarray] = []
    log = run(task, seed, frames=frames)
    recorder.status = f"- {log.outcome.upper()} after {len(log.steps)} pushes"
    recorder.snap(frames)
    frames += [frames[-1]] * FPS * 2
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, frames, fps=FPS, macro_block_size=8)
    return log.outcome


def select() -> list[tuple[str, str, int, int, str]]:
    """One success and one failure per (env, level), preferring 1000 sims."""
    runs = collections.defaultdict(list)
    for path in glob.glob(str(DATA / "*_seed*.json")):
        with open(path, encoding="utf-8") as f:
            run = json.load(f)
        runs[(run["env"], run["level"])].append(run)
    chosen = []
    order = {1000: 0, 300: 1, 3000: 2}
    for (env_name, level), items in sorted(runs.items()):
        items.sort(key=lambda r: (order[r["sims"]], r["seed"]))
        for kind, test in (("success", lambda r: r["outcome"] == "goal"),
                           ("failure", lambda r: r["outcome"] != "goal")):
            match = next((r for r in items if test(r)), None)
            if match is not None:
                chosen.append((env_name, level, match["sims"], match["seed"], kind))
    return chosen


def add_legends() -> None:
    """Draw the friction legend onto existing videos in OUT from their logs."""
    for path in sorted(OUT.glob("*.mp4")):
        env_name, level, _, sims, seed = path.stem.split("_")
        log = DATA / f"{env_name}_{level}_{sims}_{seed}.json"
        with open(log, encoding="utf-8") as f:
            frictions = json.load(f)["true_frictions"]
        frames = [friction_legend(frame, frictions) for frame in imageio.mimread(path, memtest=False)]
        imageio.mimwrite(path, frames, fps=FPS, macro_block_size=8)
        print(f"{path}: {np.round(frictions, 3).tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("one")
    one.add_argument("--env", required=True)
    one.add_argument("--level", required=True)
    one.add_argument("--sims", type=int, required=True)
    one.add_argument("--seed", type=int, required=True)
    one.add_argument("--kind", default="episode")
    sub.add_parser("select")
    sub.add_parser("legend")
    args = parser.parse_args()
    if args.cmd == "select":
        for item in select():
            print(*item)
        return
    if args.cmd == "legend":
        add_legends()
        return
    out = OUT / f"{args.env}_{args.level}_{args.kind}_s{args.sims}_seed{args.seed}.mp4"
    outcome = record(args.env, args.level, args.sims, args.seed, out)
    print(f"{out}: {outcome}")


if __name__ == "__main__":
    main()
