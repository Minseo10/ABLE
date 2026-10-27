"""Render the operator guide's architecture image without external assets."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
fig, ax = plt.subplots(figsize=(16, 11), dpi=150)
ax.set(xlim=(0, 16), ylim=(0, 11))
ax.axis("off")
fig.patch.set_facecolor("#fbfcfe")


def box(x, y, w, h, title, body, color, edge):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
        facecolor=color, edgecolor=edge, linewidth=1.5,
    ))
    ax.text(x + 0.2, y + h - 0.3, title, fontsize=15, weight="bold", va="top", color=edge)
    ax.text(x + 0.2, y + h - 0.78, body, fontsize=12, va="top", linespacing=1.55, color="#273448")


def arrow(start, end, label, *, dashed=False, color="#56647a", xy=None):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=18, linewidth=1.7,
        linestyle="--" if dashed else "-", color=color,
    ))
    if xy:
        ax.text(*xy, label, fontsize=10.5, color=color, ha="center", va="center",
                bbox={"facecolor": "#fbfcfe", "edgecolor": "none", "pad": 3})


ax.text(0.35, 10.65, "Sprout left forearm • predefined lift and set down", fontsize=23, weight="bold", color="#17263a")
ax.text(0.35, 10.18, "Implemented: camera patch tracking + MuJoCo preview + approval gate + compliant adapter    |    Live commissioning pending", fontsize=11, color="#7a4913")

box(0.4, 7.1, 4.2, 2.5, "YOUR LAPTOP", "SSH: authorized workstation account\nView / download preview.mp4\nEnter or yes: approval gate\nCtrl-C: real software-stop request", "#e9f2ff", "#245685")
box(5.5, 7.1, 10.0, 2.5, "WORKSTATION • purple box", "learning-forceful-manipulation  |  branch: real-world\nForearm + whole-body boxes → base route + arm approach\nMuJoCo: measured TidyBot + keep-out → MP4 + base/joint commands\nKinematic review; no contact-force validation; reject failed plans", "#eeebff", "#5743a9")
arrow((4.7, 8.35), (5.4, 8.35), "SSH", xy=(5.05, 8.65))

box(0.4, 3.6, 4.2, 2.3, "PERCEPTION MACHINE", "Ceiling: two calibrated views\nTape target + body/feet envelope\nFrame timestamps + calibration hash\nFresh chassis marker map pose", "#f0eee7", "#766846")
box(5.5, 3.1, 10.0, 2.8, "ROBOT NUC • live adapter", "live_runtime → guarded base transit → ceiling servo + compliant arm\nReviewed model / recipe / limits; sampled geometry checks\nFresh feedback; measured arrival; heartbeat / stop latch\nOperator confirms grasp before lift, support before release\nPhysical calibration + force limits + stop verification pending", "#e2f4ed", "#20705b")
arrow((10.5, 7.0), (10.5, 6.0), "reviewed artifacts via laptop; approval + fresh-state checks", dashed=True, xy=(10.5, 6.5))
arrow((4.7, 4.65), (5.4, 4.65), "frames", dashed=True, xy=(5.05, 4.98))

box(5.5, 0.95, 10.0, 1.4, "TIDYBOT + SPROUT", "Base stops before grasp • Kinova + gripper • Sprout supine\nBody model optional in sim and real; externally movable arm assumption", "#f0f1f4", "#4a5364")
arrow((10.5, 3.0), (10.5, 2.45), "bounded compliant commands", dashed=True, xy=(12.8, 2.7))
ax.text(0.5, 2.6, "Code paths implemented\nHardware commissioning pending\n\nCtrl-C is a network software stop.\nIt is not a physical emergency stop.", fontsize=11.5, color="#526070", va="top", linespacing=1.5)
ax.text(0.5, 0.36, "Cycle: base transit → stop → pregrasp ↓ align ↓ grasp → close → confirm → lift ≤15 cm ↑ → lower ↓ → confirm support → open → retreat ↑", fontsize=13, color="#24334a")
fig.savefig(OUT / "sprout-stack.png", bbox_inches="tight", facecolor=fig.get_facecolor())
fig.savefig(OUT / "sprout-stack.svg", bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
