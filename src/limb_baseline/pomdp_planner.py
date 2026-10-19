"""Planning with the `pomdp_model` POMDP: POUCT/POMCP over PDDLStream's domain and streams.

The MPC runs only for the `move_limb` the tree picks; a move it cannot solve, or that fails
PDDLStream's checks, is ruled out and the tree searched again. Plans execute open loop.
"""

from __future__ import annotations

import dataclasses
import random
import time
from dataclasses import dataclass

import numpy as np
import pomdp_py
from kinder_pddlstream_planning.limbrepositioning3d import run as planner_run
from kinder_pddlstream_planning.limbrepositioning3d.stream import (
    LimbConf,
    LimbStreamContext,
    check_human_joint_limits,
    check_human_torque_limits,
    check_robot_torque_limits,
    plan_base_motion,
    plan_grasp_motion,
    sample_grasp,
)
from kinder_pddlstream_planning.limbrepositioning3d.utils import (
    grasp_ik,
    release_grasp,
    saved_sim_state,
)
from numpy.typing import NDArray

from limb_baseline.belief import Belief
from limb_baseline.planner import (
    DEFAULT_PLANNER_SETTINGS,
    EpisodeResult,
    ParticleLimits,
    PlannerSettings,
    run_episode,
)
from limb_baseline.pomdp_model import (
    PLAN_ONCE,
    Clear,
    DomainState,
    Facts,
    Grasp,
    MoveBase,
    MoveLimb,
    Observation,
    Path,
    Policy,
    PomdpConfig,
    Reward,
    Stop,
    Transition,
    belief_samples,
    particle_belief,
    pose_of,
    surviving,
    target_of,
)
from limb_baseline.pomcpow import POMCPOW
from limb_baseline.tasks import Task
from limb_baseline.warmstart import WarmStartModel

# The checks domain.pddl requires of a `move_limb` trajectory.
TRAJECTORY_CHECKS = (
    check_human_joint_limits,
    check_human_torque_limits,
    check_robot_torque_limits,
)


# ---------------------------------------------------------------------- facts


def reach_along(ctx: LimbStreamContext, grasp, base, points, seed) -> list:
    """Arm joints holding `grasp` from `base` at each of `points`, None if unreachable.

    Each IK is seeded from the last so the solutions stay on one branch.
    """
    sim = ctx.sim
    solutions = []
    with saved_sim_state(sim):
        release_grasp(sim)
        sim.robot.set_base(base)
        for point in points:
            solution = grasp_ik(ctx, grasp, tuple(point), seed)
            solutions.append(solution)
            if solution is not None:
                seed = solution
    return solutions


def ground(ctx: LimbStreamContext, facts: Facts, grasps, config: PomdpConfig) -> int:
    """Sample the next few grasps and base poses into `facts`; return the pairs added.

    `grasps` is the caller's `sample_grasp` iterator, so later calls resume it.
    """
    init_conf = LimbConf(tuple(facts.path.init))
    goal_conf = LimbConf(tuple(facts.path.goal))
    points = [facts.target(k) for k in range(1, facts.num_waypoints + 1)]
    added = 0
    for _ in range(config.num_grasps):
        found = next(grasps, None)
        if found is None:
            break
        (grasp,) = found
        facts.grasps.append(grasp)
        grasp_index = len(facts.grasps) - 1
        outputs = plan_grasp_motion(ctx, ctx.limb_name, grasp, init_conf, goal_conf)
        # Close before the next grasp: each generator restores the simulator on exit.
        candidates = []
        for output in outputs:
            candidates.append(output)
            if len(candidates) >= config.bases_per_grasp:
                break
        outputs.close()
        for base, approach, state in candidates:
            known = [i for i, other in enumerate(facts.bases) if pose_of(other) == pose_of(base)]
            if known:
                base_index = known[0]
            else:
                base_path = next(iter(plan_base_motion(ctx, ctx.start_base_pose, base)), None)
                if base_path is None:
                    continue
                facts.bases.append(base)
                facts.base_paths.append(base_path[0])
                base_index = len(facts.bases) - 1
            key = (base_index, grasp_index)
            facts.kin[key] = (approach, state)
            facts.reach[key] = reach_along(ctx, grasp, base, points, state.robot_positions)
            added += 1
    return added


# ------------------------------------------------------------------ planning


def make_agent(state: DomainState, facts: Facts, thetas, config: PomdpConfig, rng):
    """A pomdp_py agent at `state`, believing `theta` is one of `thetas`."""
    return pomdp_py.Agent(
        particle_belief(state, thetas),
        Policy(facts),
        Transition(facts),
        Observation(facts, config, rng),
        Reward(config),
    )


def make_solver(facts: Facts, config: PomdpConfig, rng: np.random.Generator | None = None):
    """POUCT, POMCP or POMCPOW, per `config.solver`."""
    if config.solver == "pomcpow":
        # Base, grasp, the limb moves and stop.
        return POMCPOW(config, max_depth=config.max_limb_moves + 3, rng=rng)
    solvers = {"pouct": pomdp_py.POUCT, "pomcp": pomdp_py.POMCP}
    if config.solver not in solvers:
        raise ValueError(f"Unknown solver {config.solver!r}; expected one of {list(solvers)}")
    return solvers[config.solver](
        max_depth=facts.num_waypoints + 3,
        discount_factor=1.0,
        num_sims=config.num_sims,
        exploration_const=config.exploration,
        rollout_policy=Policy(facts),
    )


def search(
    state: DomainState, facts: Facts, thetas, config: PomdpConfig, rng: np.random.Generator
):
    """The solver's preferred action from `state` (its `theta` is ignored)."""
    return make_solver(facts, config, rng).plan(make_agent(state, facts, thetas, config, rng))


def solve_limb_motion(ctx: LimbStreamContext, state, target: LimbConf):
    """One MPC attempt, or None if it fails or PDDLStream's checks reject it."""
    outputs = planner_run.plan_limb_motion(ctx, state, target)
    try:
        solved = next(iter(outputs), None)
    finally:
        if hasattr(outputs, "close"):
            outputs.close()
    if solved is None:
        return None
    trajectory, reached = solved
    if not all(check(ctx, state, target, trajectory, reached) for check in TRAJECTORY_CHECKS):
        return None
    return solved


@dataclass
class Session:
    """One episode's planner and the base and grasp it committed to."""

    ctx: LimbStreamContext
    config: PomdpConfig
    rng: np.random.Generator
    # Committed by the last plan.
    base: object = None
    grasp: object = None
    # Called after every search: `on_search(agent, facts, state, action, alive, outcome)`.
    on_search: object = None

    def _allowed(self, facts: Facts) -> list[bool]:
        return [
            bool(self.ctx.in_believed_limits(list(facts.target(k))))
            for k in range(1, facts.num_waypoints + 1)
        ]

    def _plan(self, facts: Facts, root: DomainState, thetas, deadline: float, refill=None):
        """The tree's actions from `root` in PDDLStream's plan format, or None.

        Moves are assumed to arrive, so later searches keep only the surviving particles.
        A `move_limb` the MPC cannot solve is blocked and planning restarts from `root`.
        `refill` samples more facts when none are usable; it returns False when out.
        """
        ctx, name = self.ctx, self.ctx.limb_name
        transition = Transition(facts)
        while time.time() < deadline:
            if not facts.usable():
                if refill is None or not refill():
                    return None
                continue
            state, plan, failed = root, [], False
            coupled = facts.kin[(root.base, root.grasp)][1] if root.grasp >= 0 else None
            limb = LimbConf(tuple(facts.path.init))
            alive, agent, solver = list(thetas), None, None
            while not state.done:
                if agent is None:
                    agent = make_agent(state, facts, alive, self.config, self.rng)
                    solver = make_solver(facts, self.config, self.rng)
                action = solver.plan(agent)
                outcome = "planned"
                if isinstance(action, MoveBase):
                    base = facts.bases[action.base]
                    plan.append(
                        ("move_base", (ctx.start_base_pose, facts.base_paths[action.base], base))
                    )
                elif isinstance(action, Grasp):
                    approach, coupled = facts.kin[(state.base, action.grasp)]
                    grasp, base = facts.grasps[action.grasp], facts.bases[state.base]
                    plan.append(("grasp", (name, grasp, base, approach, coupled)))
                elif (along := target_of(action, facts.num_waypoints)) is not None:
                    target = LimbConf(tuple(facts.path.point(along)))
                    solved = solve_limb_motion(ctx, coupled, target)
                    if solved is None:
                        # Waypoints (the goal included) by index, other targets by value.
                        if isinstance(action, MoveLimb):
                            step = action.waypoint
                        else:
                            step = facts.num_waypoints if along >= 1.0 - 1e-9 else along
                        facts.blocked.add((state.base, state.grasp, step))
                        failed, outcome = True, "MPC failed or a check rejected it"
                    else:
                        trajectory, reached = solved
                        grasp, base = facts.grasps[state.grasp], facts.bases[state.base]
                        plan.append(
                            (
                                "move_limb",
                                (name, grasp, base, coupled, limb, target, trajectory, reached),
                            )
                        )
                        coupled, limb = reached, target
                else:
                    outcome = "the plan ends here"
                if self.on_search is not None:
                    self.on_search(agent, facts, state, action, alive, outcome)
                if failed or isinstance(action, Stop):
                    break
                alive = surviving(facts, state, action, alive)
                if not alive:
                    break  # every particle meets a limit: plan no further
                agent = self._advance(solver, agent, action)
                state = transition.sample(state, action)
            if failed:
                continue
            if not any(step == "move_limb" for step, _ in plan):
                return None
            self.base, self.grasp = facts.bases[state.base], facts.grasps[state.grasp]
            return plan
        return None

    def _advance(self, solver, agent, action):
        """The agent for the next search after `action` arrived, or None for a new tree.

        POMCP reuses the (`action`, `Clear`) subtree unless it holds no particles.
        """
        if self.config.solver != "pomcp":
            return None
        try:
            solver.update(agent, action, Clear())
        except Exception:  # pylint: disable=broad-except
            return None  # outcome never simulated
        if len(agent.belief) == 0:
            return None
        return agent

    def _root(self, facts: Facts, **observed) -> DomainState:
        # Plans assume every move arrives: no limits.
        unlimited = tuple(np.inf for _ in facts.path.magnitudes)
        return dataclasses.replace(facts.root(unlimited), **observed)

    def plan_start(self, limb_init, limb_goal, thetas, deadline: float):
        """The tree's plan from the start toward `limb_goal`, in PDDLStream's format, or None."""
        ctx, config = self.ctx, self.config
        facts = Facts(
            Path.between(limb_init, limb_goal),
            config.num_waypoints,
            pose_of(ctx.start_base_pose),
            tuple(ctx.retract_joints),
        )
        facts.allowed = self._allowed(facts)
        if not any(facts.allowed):
            return None
        grasps = iter(sample_grasp(ctx, ctx.limb_name))

        def refill() -> bool:
            sampled = len(facts.grasps)
            ground(ctx, facts, grasps, config)
            return len(facts.grasps) > sampled

        return self._plan(facts, self._root(facts), thetas, deadline, refill)


# ------------------------------------------------------------------- episodes


def run_episode_plan_once(
    task: Task,
    theta: NDArray[np.float64],
    belief: Belief | None = None,
    believed_theta: NDArray[np.float64] | None = None,
    warm_start: WarmStartModel | None = None,
    seed: int = 0,
    settings: PlannerSettings = DEFAULT_PLANNER_SETTINGS,
    particle_limits: ParticleLimits | None = None,
    config: PomdpConfig = PLAN_ONCE,
    on_search=None,
) -> EpisodeResult:
    """`planner.run_episode`, planned by POUCT or POMCP instead of PDDLStream."""
    rng = np.random.default_rng(seed)
    random.seed(seed)  # pomdp_py draws particles with `random`
    thetas = belief_samples(belief, believed_theta, particle_limits, config, rng)

    def plan_fn(ctx: LimbStreamContext, max_time: float):
        scene = ctx.sim.config.scene
        return Session(ctx, config, rng, on_search=on_search).plan_start(
            scene.limb_init_joint_positions,
            scene.limb_goal_joint_positions,
            thetas,
            time.time() + max_time,
        )

    return run_episode(
        task, theta, belief, believed_theta, warm_start, seed, settings, particle_limits, plan_fn
    )
