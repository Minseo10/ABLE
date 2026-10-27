"""The kinder-baselines PDDLStream planner on every scene/limb variant, at its own start and goal.

    python scripts/variant_sweep.py sweep --seeds 0 1 2 --jobs 15   # one process per (variant, seed)
    python scripts/variant_sweep.py plot                            # figures from the rows

Gravity on, muscle tone `none`, as the dream experiments run. Rows go to
data/variant_sweep/results.jsonl, logs beside them, GIFs for every run to
videos/variant_sweep and figures to figures/variant_sweep.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES = ("isolated", "human", "bed", "wheelchair")
LIMBS = ("left-arm", "right-arm", "left-leg", "right-leg")
VARIANTS = tuple(f"{scene}-{limb}" for scene in SCENES for limb in LIMBS)


class PlanningDeadline(BaseException):
    """Interrupt a stream without being swallowed by its exception handlers."""


def plan_with_deadline(plan, ctx, *, max_time, verbose):
    """Return normally on timeout so the runner can still save its replay GIF."""
    def expire(_signum, _frame):
        raise PlanningDeadline()

    previous = signal.signal(signal.SIGALRM, expire)
    old_timer = signal.setitimer(signal.ITIMER_REAL, max_time)
    try:
        return plan(ctx, max_time=max_time, verbose=verbose), False
    except PlanningDeadline:
        print(f"Planning stopped at the {max_time:g} s deadline; saving a recording.",
              flush=True)
        return None, True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        signal.setitimer(signal.ITIMER_REAL, *old_timer)


def one(args) -> None:
    """Plan and execute one variant with one seed; print its row as JSON."""
    from kinder_pddlstream_planning.limbrepositioning3d import run as runner
    from kinder_pddlstream_planning.limbrepositioning3d.stream import PredictiveSamplingMPC
    from kinder_pddlstream_planning.rendering import render_frame

    result = runner.RunResult(variant=args.variant, seed=args.seed)
    original_plan = runner.plan_limbrepositioning3d
    original_step = PredictiveSamplingMPC.step
    original_save = runner.save_gif
    timed_out = False
    progress_frames = []
    progress_frame_duration = 100
    control_steps = 0
    recording_kind = ""
    num_frames = 0

    def bounded_plan(ctx, *, max_time, verbose):
        nonlocal timed_out
        plan, timed_out = plan_with_deadline(
            original_plan, ctx, max_time=max_time, verbose=verbose
        )
        return plan

    def recorded_step(mpc):
        nonlocal control_steps, progress_frame_duration
        # Capture committed states before the MPC scratches through candidate rollouts.
        if control_steps % 20 == 0 and len(progress_frames) < 256:
            progress_frames.append(render_frame(mpc._sim))
            progress_frame_duration = round(
                1000 * 20 * mpc._commit * mpc._cfg.action_repeat * mpc._sim.config.dt
            )
        control_steps += 1
        return original_step(mpc)

    def labeled_save(path, frames, duration=100):
        nonlocal recording_kind, num_frames
        import numpy as np
        from PIL import Image, ImageDraw

        if result.plan_found:
            recording_kind = "goal reached" if result.goal_reached else "execution failed"
        elif len(frames) == 1 and progress_frames:
            frames = progress_frames
            duration = progress_frame_duration
            recording_kind = "planning progress; no accepted plan"
        else:
            recording_kind = "rejected attempt" if len(frames) > 1 else "initial state only"
        label = recording_kind + ("; planning timeout" if timed_out else "")
        labeled = []
        for frame in frames:
            frame = Image.fromarray(frame).convert("RGB")
            canvas = Image.new("RGB", (frame.width, frame.height + 36), "white")
            canvas.paste(frame, (0, 36))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 3), f"{args.variant} | seed {args.seed}", fill="black")
            draw.text((8, 19), label, fill="black")
            labeled.append(np.asarray(canvas))
        num_frames = len(labeled)
        original_save(path, labeled, duration=duration)

    runner.plan_limbrepositioning3d = bounded_plan
    PredictiveSamplingMPC.step = recorded_step
    runner.save_gif = labeled_save
    start = time.time()
    try:
        runner.solve_and_execute(
            variant=args.variant,
            result=result,
            gif_path=args.gif,
            verbose=True,
            **runner.variant_kwargs(
                args.variant,
                seed=args.seed,
                max_time=args.max_time,
                gravity=(0.0, 0.0, -9.81),
                muscle_tone="none",
            ),
        )
    except Exception as exc:  # pylint: disable=broad-except
        result.failure_reason = result.failure_reason or f"{type(exc).__name__}: {exc}"
    finally:
        runner.plan_limbrepositioning3d = original_plan
        PredictiveSamplingMPC.step = original_step
        runner.save_gif = original_save
    if timed_out:
        result.failure_reason = f"planning exceeded {args.max_time:g} s"
    result.total_time = time.time() - start
    row = dataclasses.asdict(result)
    row.update(video=str(args.gif) if args.gif else None, recording_kind=recording_kind,
               num_frames=num_frames, timed_out=timed_out)
    print("ROW " + json.dumps(row), flush=True)


def sweep(args) -> None:
    """Every (variant, seed) as its own process, each in its own working directory
    (PDDLStream writes its scratch files to the current one)."""
    args.out.mkdir(parents=True, exist_ok=True)
    args.videos.mkdir(parents=True, exist_ok=True)
    rows = args.out / "results.jsonl"

    def run(job: tuple[str, int]) -> dict:
        variant, seed = job
        work = args.out / "work" / f"{variant}_s{seed}"
        work.mkdir(parents=True, exist_ok=True)
        log = args.out / "logs" / f"{variant}_s{seed}.log"
        log.parent.mkdir(exist_ok=True)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "one",
                   "--variant", variant, "--seed", str(seed), "--max-time", str(args.max_time)]
        video_name = f"{variant}.gif" if len(args.seeds) == 1 else f"{variant}_s{seed}.gif"
        command += ["--gif", str((args.videos / video_name).resolve())]
        row = {"variant": variant, "seed": seed, "plan_found": False, "goal_reached": False,
               "plan_time": float("nan"), "total_time": float("nan"), "failure_reason": ""}
        start = time.time()
        try:
            with log.open("w") as handle:
                subprocess.run(command, cwd=work, stdout=handle, stderr=subprocess.STDOUT,
                               timeout=args.max_time + args.grace, check=False)
            lines = [line for line in log.read_text(errors="replace").splitlines()
                     if line.startswith("ROW ")]
            if lines:
                row = json.loads(lines[-1][4:])
            else:
                row["failure_reason"] = "the process ended without a result"
        except subprocess.TimeoutExpired:
            row["failure_reason"] = f"killed {args.grace:.0f} s past the planning budget"
            row["plan_time"] = row["total_time"] = time.time() - start
        with rows.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"{variant} seed {seed}: reached={row['goal_reached']} "
              f"plan={row['plan_time']:.0f}s {row['failure_reason']}", flush=True)
        return row

    jobs = [(variant, seed) for seed in args.seeds for variant in args.variants]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(run, jobs))


# --- figures -----------------------------------------------------------------------------

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e4e3dc", "#fcfcfb"
REACHED = "#2a78d6"
# One hue, light to dark, for the fraction of seeds that reached the goal.
RAMP = ("#f0efec", "#b7d3f6", "#5598e7", "#1c5cab")


def load(path: Path):
    """The sweep's rows, the last one kept if a (variant, seed) was rerun."""
    import pandas as pd

    frame = pd.DataFrame([json.loads(line) for line in path.read_text().splitlines() if line])
    return frame.drop_duplicates(["variant", "seed"], keep="last")


def summarize(frame):
    """Per variant: seeds, how many reached the goal, and planning time."""
    import pandas as pd

    rows = []
    for variant in VARIANTS:
        runs = frame[frame["variant"] == variant]
        if runs.empty:
            continue
        solved = runs[runs["goal_reached"]]
        reasons = runs.loc[~runs["goal_reached"], "failure_reason"].value_counts()
        rows.append({
            "variant": variant, "seeds": len(runs), "reached": len(solved),
            "success_rate": len(solved) / len(runs),
            "plan_time_median_reached": solved["plan_time"].median(),
            "plan_time_mean_all": runs["plan_time"].mean(),
            "failure_reasons": "; ".join(f"{count}x {reason}" for reason, count in reasons.items()),
        })
    return pd.DataFrame(rows)


def plot(args) -> None:
    """Success per scene and limb, and every run's planning time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    frame = load(args.rows)
    table = summarize(frame)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.rows.with_name("summary.csv"), index=False)
    print(table.to_string(index=False))

    plt.rcParams.update({
        "font.size": 10, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False,
    })
    seeds = sorted(frame["seed"].unique())
    budget = args.max_time
    fig, (grid, strip) = plt.subplots(
        1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1, 1.7], "wspace": 0.42})
    fig.suptitle(
        f"PDDLStream on the 16 variants: gravity on, muscle tone none, {len(seeds)} "
        f"{'seed' if len(seeds) == 1 else 'seeds'}, "
        f"{budget:.0f} s budget", x=0.015, ha="left", fontsize=12.5, fontweight="bold")

    # Left: seeds that reached the goal, scene by limb.
    rate = table.set_index("variant")
    cmap = LinearSegmentedColormap.from_list("reached", RAMP)
    for row, scene in enumerate(SCENES):
        for column, limb in enumerate(LIMBS):
            name = f"{scene}-{limb}"
            if name not in rate.index:
                continue
            value = rate.loc[name, "success_rate"]
            grid.add_patch(plt.Rectangle((column + 0.03, row + 0.03), 0.94, 0.94,
                                         color=cmap(value), linewidth=0))
            grid.text(column + 0.5, row + 0.5,
                      f"{rate.loc[name, 'reached']}/{rate.loc[name, 'seeds']}",
                      ha="center", va="center", fontsize=14, fontweight="bold",
                      color="white" if value > 0.5 else INK)
    grid.set_xlim(0, len(LIMBS))
    grid.set_ylim(len(SCENES), 0)
    grid.set_xticks(np.arange(len(LIMBS)) + 0.5, [limb.replace("-", " ") for limb in LIMBS])
    grid.set_yticks(np.arange(len(SCENES)) + 0.5, SCENES)
    grid.xaxis.tick_top()
    grid.tick_params(length=0, labelsize=10.5)
    grid.set_aspect("equal")
    for side in grid.spines.values():
        side.set_visible(False)
    grid.set_title("Seeds that reached the goal", pad=28)

    # Right: one mark per run, grouped by scene.
    positions, labels = {}, []
    for index, variant in enumerate(VARIANTS):
        positions[variant] = index + index // len(LIMBS) * 0.8
        labels.append(variant.split("-", 1)[1].replace("-", " "))
    offsets = np.linspace(-0.22, 0.22, len(seeds)) if len(seeds) > 1 else [0.0]
    for variant, y in positions.items():
        strip.axhline(y, color=GRID, linewidth=0.6, zorder=0)
        for offset, seed in zip(offsets, seeds):
            run = frame[(frame["variant"] == variant) & (frame["seed"] == seed)]
            if run.empty or np.isnan(run["plan_time"].iloc[0]):
                continue
            time_s = min(float(run["plan_time"].iloc[0]), budget * 1.12)
            if bool(run["goal_reached"].iloc[0]):
                strip.scatter(time_s, y + offset, s=58, color=REACHED, edgecolor=SURFACE,
                              linewidth=1.2, zorder=3)
            else:
                strip.scatter(time_s, y + offset, s=46, marker="X", facecolor=SURFACE,
                              edgecolor=INK2, linewidth=1.1, zorder=3)
    strip.axvline(budget, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    strip.text(budget, -0.95, "planning budget", ha="center", va="bottom", color=INK2, fontsize=9)
    for row, scene in enumerate(SCENES):
        first = positions[f"{scene}-{LIMBS[0]}"]
        strip.text(-0.17, first + (len(LIMBS) - 1) / 2, scene, transform=strip.get_yaxis_transform(),
                   ha="right", va="center", fontsize=10.5, fontweight="bold", color=INK)
    strip.set_yticks(list(positions.values()), labels)
    strip.set_ylim(max(positions.values()) + 0.8, -0.8)
    strip.set_xlim(0, budget * 1.16)
    strip.set_xlabel("planning time (s)")
    strip.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        strip.spines[side].set_visible(False)
    strip.set_title("Planning time of every run", pad=28)
    strip.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", markersize=8, color=REACHED,
                   markeredgecolor=SURFACE, label="reached the goal"),
            Line2D([], [], marker="X", linestyle="", markersize=8, markerfacecolor=SURFACE,
                   markeredgecolor=INK2, label="did not reach it"),
        ],
        loc="lower left", bbox_to_anchor=(0.0, 1.005), ncols=2, handletextpad=0.3,
        columnspacing=1.4, fontsize=9.5, borderaxespad=0)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.83, bottom=0.09)
    for suffix in ("png", "svg"):
        fig.savefig(args.out / f"summary.{suffix}", dpi=170)
    print(f"Wrote {args.out / 'summary.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("one")
    sub.add_argument("--variant", required=True, choices=VARIANTS)
    sub.add_argument("--seed", type=int, default=0)
    sub.add_argument("--max-time", type=float, default=600.0)
    sub.add_argument("--gif", type=Path, default=None)
    sub.set_defaults(run=one)
    sub = commands.add_parser("sweep")
    sub.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    sub.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    sub.add_argument("--jobs", type=int, default=15)
    sub.add_argument("--max-time", type=float, default=600.0)
    sub.add_argument("--grace", type=float, default=300.0,
                     help="Seconds past --max-time before a run is killed.")
    sub.add_argument("--out", type=Path, default=ROOT / "data/variant_sweep")
    sub.add_argument("--videos", type=Path, default=ROOT / "videos/variant_sweep")
    sub.set_defaults(run=sweep)
    sub = commands.add_parser("plot")
    sub.add_argument("--rows", type=Path, default=ROOT / "data/variant_sweep/results.jsonl")
    sub.add_argument("--out", type=Path, default=ROOT / "figures/variant_sweep")
    sub.add_argument("--max-time", type=float, default=600.0)
    sub.set_defaults(run=plot)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
