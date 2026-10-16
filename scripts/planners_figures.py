"""Planner-comparison figures: plain, no dreaming and ABLE, for each planner.

`--against DIR` also compares with an earlier run's seed scoring on shared people/episodes.

    python scripts/planners_figures.py data/dream/planners figures/dream/planners
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LINES = ["plain", "no-dream flow", "no-dream ddpm", "ABLE flow", "ABLE ddpm"]
STYLE = {
    "plain": ("0.45", "-"),
    "no-dream flow": ("#9e9ac8", "--"),
    "no-dream ddpm": ("#fa9fb5", "--"),
    "ABLE flow": ("#54278f", "-"),
    "ABLE ddpm": ("#c51b8a", "-"),
}
LABEL = {
    "plain": "Plain planner",
    "no-dream flow": "No dreaming, flow matching",
    "no-dream ddpm": "No dreaming, diffusion",
    "ABLE flow": "ABLE, flow matching",
    "ABLE ddpm": "ABLE, diffusion",
}
PLANNERS = {"pddlstream": "PDDLStream", "pouct": "POUCT"}
LATE = 5  # episodes from here on are summarised


def load(directory: Path) -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in sorted(directory.glob("*.csv")) if "dreams" not in f.name]
    df = pd.concat(frames, ignore_index=True)
    kind = np.where(df.n_dreams > 0, "ABLE ", "no-dream ")
    model = df.model.fillna("").str.split(":").str[1].fillna("")
    df["line"] = np.where(df.condition == "plain", "plain", kind + model)
    df["other_seconds"] = (
        df.plan_cpu_time - df.score_seconds - df.full_seconds - df.model_seconds
    )
    return df


def per_episode(df, value, smooth=3):
    """Mean over people per episode, then a centred running mean."""
    curve = df.groupby("episode")[value].mean()
    return curve.rolling(smooth, center=True, min_periods=1).mean()


def acceptance(df):
    sums = df.groupby("episode")[["seeded_steps", "seed_rejected"]].sum()
    return per_episode(
        (sums.seeded_steps / (sums.seeded_steps + sums.seed_rejected)).rename("a")
        .reset_index(), "a")


def compare(new: pd.DataFrame, old: pd.DataFrame, out: Path) -> None:
    """Each line's plan time and acceptance, old scoring dashed, new solid."""
    keys = ["planner", "line", "person", "episode"]
    common = new[keys].merge(old[keys])
    new, old = new.merge(common), old.merge(common)
    lines = [l for l in LINES if l in set(new.line)]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), constrained_layout=True)
    for row, (planner, name) in enumerate(PLANNERS.items()):
        ax_time, ax_acc, ax_bar = axes[row]
        for line in lines:
            color, _ = STYLE[line]
            for df, dash, tag in ((old, "--", "earlier scoring"), (new, "-", "reuse")):
                sub = df[(df.planner == planner) & (df.line == line)]
                ax_time.plot(per_episode(sub, "plan_time"), dash, color=color, lw=2,
                             label=f"{LABEL[line]} ({tag})")
                if line != "plain":
                    ax_acc.plot(acceptance(sub), dash, color=color, lw=2)
        ax_time.set(title=f"{name}: planning time per episode", xlabel="episode",
                    ylabel="seconds (people mean, smoothed)", ylim=(0, None))
        ax_acc.set(title=f"{name}: seed acceptance per MPC step", xlabel="episode",
                   ylabel="accepted / (accepted + rejected)", ylim=(0, 1))
        late = {tag: df[(df.planner == planner) & (df.episode >= LATE)]
                .groupby("line").plan_time.mean().reindex(lines)
                for df, tag in ((old, "earlier scoring"), (new, "reuse"))}
        x = np.arange(len(lines))
        for offset, (tag, values) in zip((-0.2, 0.2), late.items()):
            bars = ax_bar.bar(x + offset, values, 0.4, label=tag,
                              color=[STYLE[l][0] for l in lines],
                              alpha=0.45 if tag == "earlier scoring" else 1.0)
            ax_bar.bar_label(bars, fmt="%.0f", fontsize=8)
        ax_bar.set_xticks(x, [LABEL[l].replace(", ", "\n") for l in lines], fontsize=8)
        ax_bar.set(title=f"{name}: mean planning time, episodes {LATE}+",
                   ylabel="seconds per episode")
        for ax in axes[row]:
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    axes[0, 2].legend(fontsize=8, title="pale: earlier scoring", title_fontsize=8)
    people = ", ".join(str(p) for p in sorted(set(new.person)))
    fig.suptitle(f"Seed scoring with reuse (solid) vs the earlier scoring (dashed); "
                 f"people {people}, same tasks")
    fig.savefig(out / "against_earlier.png", dpi=150)
    print(f"Wrote {out / 'against_earlier.png'}")


def main() -> None:
    directory, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    df = load(directory)
    lines = [l for l in LINES if l in set(df.line)]
    last = int(df.episode.max())
    if "--against" in sys.argv:
        compare(df, load(Path(sys.argv[sys.argv.index("--against") + 1])), out)
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), constrained_layout=True)
    for row, (planner, name) in enumerate(PLANNERS.items()):
        runs = df[df.planner == planner]
        ax_time, ax_acc, ax_bar = axes[row]
        for line in lines:
            color, dash = STYLE[line]
            sub = runs[runs.line == line]
            ax_time.plot(per_episode(sub, "plan_time"), dash, color=color, lw=2,
                         label=LABEL[line])
            if line != "plain":
                ax_acc.plot(acceptance(sub), dash, color=color, lw=2, label=LABEL[line])
        ax_time.set(title=f"{name}: planning time per episode", xlabel="episode",
                    ylabel="seconds (people mean, smoothed)", ylim=(0, None))
        ax_acc.set(title=f"{name}: seed acceptance per MPC step",
                   xlabel="episode", ylabel="accepted / (accepted + rejected)", ylim=(0, 1))
        late = runs[runs.episode >= LATE].groupby("line").mean(numeric_only=True).reindex(lines)
        parts = [("full_seconds", "full MPC steps", "#4d4d4d"),
                 ("score_seconds", "scoring the seed", "#e08214"),
                 ("model_seconds", "model inference", "#2166ac"),
                 ("other_seconds", "planner and checks", "#bababa")]
        bottom = np.zeros(len(lines))
        for column, label, color in parts:
            ax_bar.bar(range(len(lines)), late[column], bottom=bottom, color=color, label=label)
            bottom += late[column].to_numpy()
        ax_bar.set_xticks(range(len(lines)), [LABEL[l].replace(", ", "\n") for l in lines],
                          fontsize=8)
        ax_bar.set(title=f"{name}: CPU time, episodes {LATE}-{last}",
                   ylabel="CPU seconds per episode")
        for ax in axes[row]:
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    axes[0, 2].legend(fontsize=8)
    fig.suptitle(f"{df.variant.iloc[0]}, {df.person.nunique()} people x {last + 1} episodes, "
                 f"{int(df.n_dreams.max())} dreams per episode for ABLE")
    fig.savefig(out / "comparison.png", dpi=150)
    print(f"Wrote {out / 'comparison.png'}")


if __name__ == "__main__":
    main()
