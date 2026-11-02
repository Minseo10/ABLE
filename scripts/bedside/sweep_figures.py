"""Summarize and plot the BedsideStocking3D planning-time sweep.

Usage: python scripts/bedside/sweep_figures.py data/bedside/planning_sweep figures/bedside
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/bedside/planning_sweep")
dst = Path(sys.argv[2] if len(sys.argv) > 2 else "figures/bedside")
dst.mkdir(parents=True, exist_ok=True)
rows = [json.loads(f.read_text()) for f in sorted(src.glob("r*_c*_s*.json"))]
rs = sorted({r["num_requested"] for r in rows})
cs = sorted({r["num_clutter"] for r in rows})

print(f"{'r':>3}{'c':>3}{'n':>4}{'solved':>8}{'median t (s)':>14}{'mean t':>9}{'moves':>7}")
for r in rs:
    for c in cs:
        sel = [x for x in rows if x["num_requested"] == r and x["num_clutter"] == c]
        if not sel:
            continue
        t = np.array([x["plan_time"] for x in sel])
        ok = [x for x in sel if x["plan_found"]]
        moves = np.mean([x["num_moves"] for x in ok]) if ok else float("nan")
        print(f"{r:>3}{c:>3}{len(sel):>4}{len(ok):>5}/{len(sel):<2}{np.median(t):>14.1f}"
              f"{t.mean():>9.1f}{moves:>7.1f}")

fig, ax = plt.subplots(figsize=(5.5, 3.6))
for c in cs:
    med, lo, hi = [], [], []
    for r in rs:
        t = np.array([x["plan_time"] for x in rows
                      if x["num_requested"] == r and x["num_clutter"] == c])
        med.append(np.median(t)); lo.append(np.percentile(t, 25)); hi.append(np.percentile(t, 75))
    ax.plot(rs, med, marker="o", label=f"{c} clutter")
    ax.fill_between(rs, lo, hi, alpha=0.15)
ax.axhline(100, color="grey", ls="--", lw=0.8)
ax.set_yscale("log")
ax.set_xlabel("requested items")
ax.set_ylabel("planning time (s, median, IQR)")
ax.set_xticks(rs)
ax.legend(frameon=False, fontsize=8)
ax.set_title("BedsideStocking3D: PDDLStream (adaptive) planning time")
fig.tight_layout()
fig.savefig(dst / "planning_sweep.png", dpi=200)
print("saved", dst / "planning_sweep.png")
