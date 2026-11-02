"""Draw the belief trees the POMDP planner's solver (POUCT or POMCP) grows.

One search on hand-made facts by default; --episode replays a `pomdp_compare.py` episode,
one panel per search. Blue: action nodes; orange: belief nodes; red: the preferred path.

    python scripts/pomdp_tree.py --episode 0 1 --solver pomcp --num-waypoints 4 \\
        --out figures/pomdp/trees/p0_g1_pomcp_w4.png
"""

import argparse
import dataclasses
import sys
from pathlib import Path as FilePath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from able.tasks.limb.human import DEFAULT_PRIOR
from able.tasks.limb.pomdp_model import (
    Clear,
    Facts,
    Grasp,
    Limit,
    MoveBase,
    MoveLimb,
    PomdpConfig,
    Stop,
    Transition,
    example_facts,
)
from able.tasks.limb.pomdp_planner import make_agent, make_solver


def make_facts(goal_scale: float, num_waypoints: int, thetas: np.ndarray) -> Facts:
    facts = example_facts(goal_scale, num_waypoints, pairs=((0, 0), (1, 1)))
    # Base 1's arm cannot hold the limb at the goal.
    facts.reach[(1, 1)][-1] = None
    # As PDDLStream's particle limits: allowed when half the particles allow it.
    path = facts.path
    facts.allowed = [
        bool(np.mean(np.all(path.scale(k / num_waypoints)[path.outward]
                            <= thetas[:, path.outward], axis=1)) >= 0.5)
        for k in range(1, num_waypoints + 1)
    ]
    return facts


def label(action) -> str:
    if isinstance(action, MoveBase):
        return f"move_base({action.base})"
    if isinstance(action, Grasp):
        return f"grasp({action.grasp})"
    if isinstance(action, MoveLimb):
        return f"move_limb(w{action.waypoint})"
    return "stop"


def consistent(ctx, history) -> int:
    """How many particles could have produced `history` from the root."""
    facts, config = ctx["facts"], ctx["config"]
    transition, count = Transition(facts), 0
    for theta in ctx["thetas"]:
        state, keep = dataclasses.replace(ctx["root"], theta=tuple(theta)), True
        for action, observation in history:
            state = transition.sample(state, action)
            if isinstance(observation, Limit):
                width = 3 * config.tau / facts.path.magnitudes[observation.motion]
                reading = observation.bin * config.resolution
                keep = state.hit == observation.motion and abs(theta[state.hit] - reading) <= width
            elif state.hit >= 0:
                keep = False
            if not keep:
                break
        count += keep
    return count


def build(vnode, history, min_visits, depth, ctx) -> dict:
    """A nested-dict snapshot of the pruned tree below `vnode` (POMCP keeps growing it)."""
    node = {"kind": "belief", "visits": vnode.num_visits, "alive": consistent(ctx, history),
            "children": []}
    belief = getattr(vnode, "belief", None)
    if belief is not None:
        node["stored"] = len(belief)
    if depth == 0:
        return node
    best = max(vnode.children, key=lambda a: vnode[a].value, default=None)
    for action in sorted(vnode.children, key=lambda a: -vnode[a].num_visits):
        qnode = vnode[action]
        if qnode.num_visits < min_visits:
            continue
        child = {"kind": "action", "label": label(action), "visits": qnode.num_visits,
                 "value": qnode.value, "best": action == best, "children": []}
        node["children"].append(child)
        if isinstance(action, Stop):
            continue  # the episode ends
        observations = sorted(qnode.children, key=lambda o: -qnode[o].num_visits)
        for observation in [o for o in observations if isinstance(o, Clear)]:
            below = qnode[observation]
            if below.num_visits < min_visits:
                continue
            sub = build(below, history + [(action, observation)], min_visits, depth - 1, ctx)
            sub["label"] = "clear"
            sub["best"] = child["best"]
            child["children"].append(sub)
        # A met limit ends the episode: merge its readings into one leaf.
        limits = [o for o in observations if isinstance(o, Limit)]
        visits = sum(qnode[o].num_visits for o in limits)
        if visits >= min_visits:
            readings = [o.bin * ctx["config"].resolution for o in limits]
            child["children"].append({
                "kind": "limit", "visits": visits, "children": [], "best": False,
                "label": f"limit hit ({len(limits)} readings,\n"
                         f"θ≈{min(readings):.2f}–{max(readings):.2f})\nN={visits}",
            })
    return node


def snapshot(agent, facts, state, thetas, config, min_frac, depth) -> dict:
    """The drawable tree of `agent` after a search from `state`."""
    ctx = {"thetas": np.asarray(thetas), "facts": facts, "config": config, "root": state}
    min_visits = max(1, int(min_frac * agent.tree.num_visits))
    tree = build(agent.tree, [], min_visits, depth, ctx)
    tree.update(label="root", total=len(thetas), min_visits=min_visits)
    return tree


def layout(node, depth=0, next_row=None):
    """Leaves get consecutive rows; parents sit at their children's mean row."""
    next_row = next_row if next_row is not None else [0]
    node["x"] = depth
    if not node["children"]:
        node["y"] = next_row[0]
        next_row[0] += 1
    else:
        for child in node["children"]:
            layout(child, depth + 1, next_row)
        node["y"] = float(np.mean([c["y"] for c in node["children"]]))
    return next_row[0]


def draw(ax, node, root_visits, total):
    for child in node["children"]:
        width = 0.5 + 6 * child["visits"] / root_visits
        color = "tab:red" if child.get("best") and node.get("on_path", True) else "0.6"
        child["on_path"] = color == "tab:red"
        ax.plot([node["x"], child["x"]], [node["y"], child["y"]], color=color, lw=width,
                zorder=1, solid_capstyle="round")
        draw(ax, child, root_visits, total)
    if node["kind"] == "action":
        text = f"{node['label']}\nN={node['visits']}  Q={node['value']:.1f}"
        box = dict(boxstyle="round,pad=0.3", fc="#dbe9f6", ec="tab:blue")
    elif node["kind"] == "belief":
        text = f"{node.get('label', 'root')}\nN={node['visits']}  {node['alive']}/{total} θ"
        if node.get("stored"):  # POMCP stores particles only one level below the root
            text += f"\n{node['stored']} particles"
        box = dict(boxstyle="round4,pad=0.3", fc="#fde9d4", ec="tab:orange")
    else:
        text = node["label"]
        box = dict(boxstyle="round4,pad=0.3", fc="#f6d5d5", ec="tab:red", ls="--")
    ax.text(node["x"], node["y"], text, ha="center", va="center", fontsize=6.5, bbox=box,
            zorder=2)


def render(panels: list[tuple[dict, str]], heading: str, depth: int, out: FilePath) -> None:
    """One panel per tree, stacked in order."""
    rows = [layout(tree) for tree, _ in panels]
    heights = [0.5 * r + 1.0 for r in rows]
    fig, axes = plt.subplots(len(panels), 1, squeeze=False,
                             figsize=(2.3 * (depth * 2 + 1), sum(heights) + 0.8),
                             gridspec_kw={"height_ratios": heights})
    for ax, (tree, title), n in zip(axes[:, 0], panels, rows, strict=True):
        draw(ax, tree, tree["visits"], tree["total"])
        ax.set_xlim(-0.6, 2 * depth + 0.6)
        ax.set_ylim(n - 0.4, -0.6)
        ax.axis("off")
        ax.set_title(f"{title}; branches under {tree['min_visits']} visits pruned",
                     fontsize=8, loc="left")
    fig.suptitle(heading, fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def waypoint_note(facts: Facts) -> str:
    return ", ".join(
        f"w{k}" + ("" if facts.allowed[k - 1] else " (not allowed)")
        for k in range(1, facts.num_waypoints + 1)
    )


def hand_made(args) -> None:
    rng = np.random.default_rng(args.seed)
    thetas = np.array([DEFAULT_PRIOR.sample(rng) for _ in range(32)])
    config = dataclasses.replace(PomdpConfig(), num_waypoints=args.num_waypoints,
                                 num_sims=args.num_sims, solver=args.solver)
    facts = make_facts(args.goal_scale, args.num_waypoints, thetas)
    root = facts.root(thetas[0])
    agent = make_agent(root, facts, thetas, config, rng)
    chosen = make_solver(facts, config).plan(agent)
    tree = snapshot(agent, facts, root, thetas, config, args.min_frac, args.depth)
    scales = ", ".join(f"w{k}={facts.path.scale(k / facts.num_waypoints)[0]:.2f}"
                       for k in range(1, facts.num_waypoints + 1))
    title = (f"{args.solver.upper()}, {config.num_sims} simulations; chosen: {label(chosen)}; "
             f"waypoint scales {scales}; base 1 cannot reach w{facts.num_waypoints}")
    render([(tree, title)], "hand-made facts", args.depth, args.out)
    print(f"chosen {label(chosen)}; saved {args.out} and .svg")


def episode(args) -> None:
    """Replay a `pomdp_compare.py` episode, drawing every search."""
    sys.path.insert(0, str(FilePath(__file__).parent))
    import pomdp_compare  # pylint: disable=import-outside-toplevel

    from able.tasks.limb.variant import (
        select,  # pylint: disable=import-outside-toplevel
    )

    select(args.variant)
    person, goal = args.episode
    tasks, nominal_init = pomdp_compare.draw_tasks(person, goal + 1)
    config = dataclasses.replace(PomdpConfig(), num_waypoints=args.num_waypoints,
                                 solver=args.solver)
    panels, attempt = [], [0]

    def on_search(agent, facts, state, action, alive, outcome):
        if state.base < 0:
            attempt[0] += 1  # a search from the start begins a new attempt
        tree = snapshot(agent, facts, state, alive, config, args.min_frac, args.depth)
        title = (f"attempt {attempt[0]}, search {len(panels) + 1}: chose {label(action)} "
                 f"→ {outcome}; {len(facts.bases)} base poses, {len(facts.grasps)} grasps; "
                 f"waypoints {waypoint_note(facts)}")
        panels.append((tree, title))

    planner = f"{args.solver}_w{args.num_waypoints}"
    log = args.out.with_suffix(".log")
    job = {
        "planner": planner, "solver": args.solver, "person": person, "goal": goal,
        "variant": args.variant, "task": tasks[goal], "nominal_init": nominal_init,
        "num_waypoints": args.num_waypoints, "num_rollouts": args.num_rollouts,
        "max_time": args.max_time, "log": str(log),
    }
    row = pomdp_compare.run_one(job, on_search)
    heading = (f"person {person}, goal {goal}, {planner}: "
               f"{'goal reached' if row.get('goal_reached') else 'goal not reached'}, "
               f"distance left {row.get('goal_distance')}, plan time {row.get('plan_time')} s"
               + (f" ({row['failure_reason']})" if row.get("failure_reason") else ""))
    if not panels:
        print(f"no search was made: {row.get('failure_reason')}")
        return
    render(panels, heading, args.depth, args.out)
    print(heading)
    print(f"{len(panels)} searches; saved {args.out} and .svg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--episode", type=int, nargs=2, metavar=("PERSON", "GOAL"),
                        help="replay this episode of pomdp_compare.py instead")
    parser.add_argument("--solver", choices=("pouct", "pomcp"), default="pouct")
    parser.add_argument("--num-waypoints", type=int, default=4)
    parser.add_argument("--goal-scale", type=float, default=0.95, help="hand-made only")
    parser.add_argument("--num-sims", type=int, default=3000, help="hand-made only")
    parser.add_argument("--seed", type=int, default=0, help="hand-made only")
    parser.add_argument("--variant", default="isolated-left-arm")
    parser.add_argument("--num-rollouts", type=int, default=64)
    parser.add_argument("--max-time", type=float, default=600.0)
    parser.add_argument("--min-frac", type=float, default=0.01,
                        help="prune branches with fewer than this fraction of root visits")
    parser.add_argument("--depth", type=int, default=5, help="action levels drawn")
    parser.add_argument("--out", type=FilePath, default=FilePath("figures/pomdp/tree.png"))
    args = parser.parse_args()
    # Absolute: episodes run in their own working directory.
    args.out = args.out.resolve()
    if args.episode is None:
        hand_made(args)
    else:
        episode(args)


if __name__ == "__main__":
    main()
