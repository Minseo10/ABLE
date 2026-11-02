"""Table image of SR and planning time per level and budget, from summary.csv.

For each env and level, the best-SR budget is bold (ties go to the faster one).

    python scripts/friction/table_figure.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("figures/friction/complexity")
ENVS = ["2d", "3d"]

rows = list(csv.DictReader(open(OUT / "summary.csv")))
cell = {(r["env"], r["level"], int(r["sims"])): r for r in rows}
levels = sorted({r["level"] for r in rows})
budgets = sorted({int(r["sims"]) for r in rows})

best = {}
for env in ENVS:
    for lv in levels:
        best[env, lv] = min(
            budgets,
            key=lambda s: (-float(cell[env, lv, s]["success_rate"]),
                           float(cell[env, lv, s]["plan_s_per_episode_mean"])),
        )

header = ["Level", "Sims"] + [f"{e.upper()} {c}" for e in ENVS
                              for c in ("SR", "s/push", "s/episode")]
body, bold = [], []
for lv in levels:
    for s in budgets:
        row, b = [lv, str(s)], [False, False]
        for env in ENVS:
            r = cell[env, lv, s]
            row += [f"{float(r['success_rate']):.1f}",
                    f"{float(r['plan_s_per_push_mean']):.2f}",
                    f"{float(r['plan_s_per_episode_mean']):.1f}"]
            b += [best[env, lv] == s] * 3
        body.append(row)
        bold.append(b)

fig, ax = plt.subplots(figsize=(10, 4.6))
ax.axis("off")
tab = ax.table(cellText=body, colLabels=header, loc="center", cellLoc="center")
tab.auto_set_font_size(False)
tab.set_fontsize(10)
tab.scale(1, 1.35)
tab.auto_set_column_width(list(range(len(header))))
for j in range(len(header)):
    tab[0, j].set_text_props(weight="bold")
    tab[0, j].set_facecolor("#e8e8e8")
for i, (b, row) in enumerate(zip(bold, body), start=1):
    shade = "#f7f7f7" if levels.index(row[0]) % 2 else "white"
    for j in range(len(header)):
        tab[i, j].set_facecolor(shade)
        if b[j]:
            tab[i, j].set_text_props(weight="bold")
ax.set_title("PuckPushing POMCPOW: success rate and planning time (10 seeds per cell)\n"
             "Bold: best SR per env and level (ties go to the lower planning time)",
             fontsize=11)
fig.savefig(OUT / "sr_time_table.png", dpi=200, bbox_inches="tight")
print(OUT / "sr_time_table.png")
