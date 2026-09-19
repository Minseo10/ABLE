"""Draw the limb planner's graphs in two views, for the current and the separator design.

Stream view: streams as factors, arrows from inputs to outputs, as PDDLStream runs them.
Constraint view: certified facts as factors, undirected, on the variables they truly touch.

Variables keep the same position in all four figures, so the views can be compared.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "figures"

GIVEN, SAMPLED = "#d9d9d9", "#ffffff"
ROBOT, HUMAN, PATH, REALIZED = "#cfe2f3", "#fce5cd", "#d9ead3", "#fff2cc"

LEGEND = [
    (GIVEN, "round", "given in the problem"),
    (SAMPLED, "round", "sampled value"),
    (ROBOT, "square", "robot side"),
    (HUMAN, "square", "human side, on the reference"),
    (REALIZED, "square", "certifies the realized motion"),
    (PATH, "square", "limb path"),
]

VARIABLES = {
    "?l": ((6.4, 7.4), "?l\nLimb", GIVEN),
    "?qL1": ((0.8, 4.2), "?qL1\nInitConf", GIVEN),
    "?qL2": ((0.8, 1.6), "?qL2\nGoalConf", GIVEN),
    "?q1": ((8.2, 9.0), "?q1\nstart BConf", GIVEN),
    "?g": ((4.4, 6.0), "?g\nGrasp", SAMPLED),
    "?q": ((8.2, 4.8), "?q\nBConf", SAMPLED),
    "?bt": ((12.2, 9.0), "?bt\nbase path", SAMPLED),
    "?at": ((12.2, 7.0), "?at\narm path", SAMPLED),
    "?s1": ((12.2, 4.8), "?s1\nState", SAMPLED),
    "?tt": ((15.4, 3.7), "?tt\ntorque traj", SAMPLED),
    "?s2": ((15.4, 2.0), "?s2\nState", SAMPLED),
}
PATH_VARIABLE = {"?xi": ((6.6, 0.2), "?xi\nLimbPath", SAMPLED)}

# Extra curvature for edges that would otherwise run through a node.
BENDS = {
    ("?l", "sample-grasp"): 0.25,
    ("?l", "Grasp"): 0.25,
    ("?l", "sample-limb-path"): 0.3,
    ("?l", "LimbPath"): 0.3,
    ("?qL1", "plan-grasp-motion"): -0.15,
    ("?qL1", "GraspKin"): -0.15,
    ("?g", "test-path-reach"): 0.22,
    ("?g", "ReachablePath"): 0.22,
}

# name: (position, sublabel, color, inputs, outputs); a test stream has no outputs.
CURRENT_STREAMS = {
    "sample-grasp": ((2.4, 6.0), "Grasp", ROBOT, ["?l"], ["?g"]),
    "sample-base-pose": ((6.4, 4.8), "Reachable", ROBOT, ["?l", "?g", "?qL1", "?qL2"], ["?q"]),
    "plan-base-motion": ((10.2, 9.0), "BaseMotion", ROBOT, ["?q1", "?q"], ["?bt"]),
    "plan-grasp-motion": ((10.2, 6.0), "GraspKin", ROBOT, ["?l", "?g", "?q", "?qL1"], ["?at", "?s1"]),
    "plan-limb-motion": ((13.2, 2.8), "LimbMotion", ROBOT, ["?s1", "?qL2"], ["?tt", "?s2"]),
    "check-human-joint-limits": ((18.4, 4.8), "SafeHumanJoints", HUMAN, ["?s1", "?qL2", "?tt", "?s2"], []),
    "check-human-torque-limits": ((18.4, 2.9), "SafeHumanTorques", HUMAN, ["?s1", "?qL2", "?tt", "?s2"], []),
    "check-robot-torque-limits": ((18.4, 1.0), "SafeRobotTorques", ROBOT, ["?s1", "?qL2", "?tt", "?s2"], []),
}

# name: (position, sublabel, color, variables it touches)
CURRENT_CONSTRAINTS = {
    "Grasp": ((2.4, 6.0), "sample-grasp", ROBOT, ["?l", "?g"]),
    "Reachable": ((6.7, 5.9), "at ?qL1", ROBOT, ["?l", "?g", "?q", "?qL1"]),
    "Reachable ": ((5.2, 2.9), "at ?qL2", ROBOT, ["?l", "?g", "?q", "?qL2"]),
    "BaseMotion": ((10.2, 9.0), "plan-base-motion", ROBOT, ["?q1", "?q", "?bt"]),
    "GraspKin": ((10.2, 6.0), "plan-grasp-motion", ROBOT, ["?l", "?g", "?q", "?at", "?s1"]),
    "LimbMotion": ((13.2, 2.8), "plan-limb-motion", ROBOT, ["?s1", "?qL2", "?tt", "?s2"]),
    "SafeHumanJoints": ((18.4, 4.8), "check-human-joint-limits", HUMAN, ["?tt"]),
    "SafeHumanTorques": ((18.4, 2.9), "check-human-torque-limits", HUMAN, ["?tt"]),
    "SafeRobotTorques": ((18.4, 1.0), "check-robot-torque-limits", ROBOT, ["?tt"]),
}

SEPARATOR_STREAMS = {
    "sample-limb-path": ((3.2, 0.2), "LimbPath", PATH, ["?l", "?qL1", "?qL2"], ["?xi"]),
    "check-limb-path-joints": ((1.6, -1.8), "SafeLimbPathJoints", HUMAN, ["?l", "?xi"], []),
    "check-limb-path-torque": ((5.8, -1.8), "SafeLimbPathTorques", HUMAN, ["?l", "?xi"], []),
    "check-limb-path-contact": ((10.0, -1.8), "SafeLimbPathContact", HUMAN, ["?l", "?xi"], []),
    "sample-grasp": ((2.4, 6.0), "Grasp", ROBOT, ["?l"], ["?g"]),
    "sample-base-pose": ((6.4, 4.8), "Reachable", ROBOT, ["?l", "?g", "?qL1", "?qL2"], ["?q"]),
    "test-path-reach": ((11.2, 1.2), "ReachablePath", ROBOT, ["?l", "?g", "?q", "?xi"], []),
    "plan-base-motion": ((10.2, 9.0), "BaseMotion", ROBOT, ["?q1", "?q"], ["?bt"]),
    "plan-grasp-motion": ((10.2, 6.0), "GraspKin", ROBOT, ["?l", "?g", "?q", "?qL1"], ["?at", "?s1"]),
    "plan-limb-motion": ((13.6, 3.2), "LimbMotion", ROBOT, ["?l", "?s1", "?xi", "?qL1", "?qL2"], ["?tt", "?s2"]),
    "check-human-joint-limits": ((18.4, 4.8), "SafeHumanJoints", REALIZED, ["?s1", "?qL2", "?tt", "?s2"], []),
    "check-human-torque-limits": ((18.4, 2.9), "SafeHumanTorques", REALIZED, ["?s1", "?qL2", "?tt", "?s2"], []),
    "check-robot-torque-limits": ((18.4, 1.0), "SafeRobotTorques", ROBOT, ["?s1", "?qL2", "?tt", "?s2"], []),
}

SEPARATOR_CONSTRAINTS = {
    "LimbPath": ((3.2, 0.2), "sample-limb-path", PATH, ["?l", "?xi", "?qL1", "?qL2"]),
    "SafeLimbPathJoints": ((1.6, -1.8), "range of motion on ?xi", HUMAN, ["?xi"]),
    "SafeLimbPathTorques": ((5.8, -1.8), "limb inverse dynamics on ?xi", HUMAN, ["?xi"]),
    "SafeLimbPathContact": ((10.0, -1.8), "contact along ?xi", HUMAN, ["?xi"]),
    "Grasp": ((2.4, 6.0), "sample-grasp", ROBOT, ["?l", "?g"]),
    "ReachablePath": ((11.2, 1.2), "test-path-reach", ROBOT, ["?l", "?g", "?q", "?xi"]),
    "BaseMotion": ((10.2, 9.0), "plan-base-motion", ROBOT, ["?q1", "?q", "?bt"]),
    "GraspKin": ((10.2, 6.0), "plan-grasp-motion", ROBOT, ["?l", "?g", "?q", "?at", "?s1"]),
    "LimbMotion": ((13.6, 3.2), "plan-limb-motion", ROBOT, ["?s1", "?xi", "?tt", "?s2"]),
    "SafeHumanJoints": ((18.4, 4.8), "tracking error on ?tt", REALIZED, ["?tt"]),
    "SafeHumanTorques": ((18.4, 2.9), "tracking error on ?tt", REALIZED, ["?tt"]),
    "SafeRobotTorques": ((18.4, 1.0), "check-robot-torque-limits", ROBOT, ["?tt"]),
}


def draw(ax, variables: dict, factors: dict, title: str, stream_view: bool) -> None:
    """Variables as rounded nodes, factors as boxes; arrows mark stream outputs."""
    nodes = {}
    for name, (xy, label, color) in variables.items():
        nodes[name] = ax.text(
            *xy, label, ha="center", va="center", fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", fc=color, ec="#444444", lw=1.2),
        )
    for name, entry in factors.items():
        nodes[name] = ax.text(
            *entry[0], f"{name.strip()}\n({entry[1]})", ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="square,pad=0.5", fc=entry[2], ec="#222222", lw=1.4),
        )
    for name, entry in factors.items():
        outputs = entry[4] if stream_view else []
        for source in entry[3]:
            _edge(ax, nodes[source], nodes[name], "-", BENDS.get((source, name.strip()), 0.06))
        for target in outputs:
            _edge(ax, nodes[name], nodes[target], "-|>", BENDS.get((name, target), 0.06))
    ax.set_title(title, fontsize=14, loc="left", pad=12)
    ax.set_xlim(-0.2, 20.6)
    ax.set_ylim(-2.8 if "?xi" in variables else -0.4, 9.6)
    ax.axis("off")


def _edge(ax, a, b, style: str, rad: float) -> None:
    ax.annotate(
        "", xy=(0, 0), xytext=(0, 0),
        arrowprops=dict(
            arrowstyle=style, color="#555555", lw=1.0, alpha=0.8,
            patchA=a.get_bbox_patch(), patchB=b.get_bbox_patch(),
            connectionstyle=f"arc3,rad={rad}", mutation_scale=14,
        ),
        xycoords=b, textcoords=a,
    )


def legend(fig, colors: set) -> None:
    """One chip per node kind the figure uses."""
    for i, (color, shape, label) in enumerate(i for i in LEGEND if i[0] in colors):
        fig.text(
            0.04 + 0.16 * i, 0.02, label, fontsize=9.5, va="center",
            bbox=dict(boxstyle=f"{shape},pad=0.4", fc=color, ec="#444444"),
        )


def main() -> None:
    """Write one image per design and view."""
    OUT.mkdir(parents=True, exist_ok=True)
    separator = dict(VARIABLES, **PATH_VARIABLE)
    figures = [
        ("stream_graph_current", VARIABLES, CURRENT_STREAMS, True,
         "1a. Current planner, stream view: streams as factors, inputs to outputs"),
        ("constraint_graph_current", VARIABLES, CURRENT_CONSTRAINTS, False,
         "1b. Current planner, constraint view: each fact on the variables it truly touches"),
        ("stream_graph_separator", separator, SEPARATOR_STREAMS, True,
         "2a. Separator design, stream view: a limb path stream, path checks, and a reach test"),
        ("constraint_graph_separator", separator, SEPARATOR_CONSTRAINTS, False,
         "2b. Separator design, constraint view: ?xi cuts the human side from the robot side"),
    ]
    for name, variables, factors, stream_view, title in figures:
        fig, ax = plt.subplots(figsize=(19, 9.5 if "?xi" not in variables else 11.5))
        draw(ax, variables, factors, title, stream_view)
        colors = {c for _, _, c in variables.values()} | {e[2] for e in factors.values()}
        legend(fig, colors)
        fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {OUT / f'{name}.png'}")


if __name__ == "__main__":
    main()
