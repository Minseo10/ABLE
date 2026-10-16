"""Tests for the POUCT search on hand-made facts: no simulator, no MPC."""

import time
from types import SimpleNamespace

import numpy as np

from limb_baseline import pomdp_planner
from limb_baseline.human import active_motions
from limb_baseline.pomdp_model import (
    Clear,
    Grasp,
    MoveBase,
    MoveLimb,
    PomdpConfig,
    Stop,
    Transition,
    example_facts,
    surviving,
)
from limb_baseline.pomdp_planner import Session, make_agent, make_solver, search


def people(scales) -> np.ndarray:
    return np.array([[scale] * len(active_motions()) for scale in scales])


def test_the_tree_plans_move_base_grasp_and_one_move_to_a_safe_goal():
    config = PomdpConfig(num_waypoints=1)
    facts = example_facts(0.7, num_waypoints=1)
    transition, rng = Transition(facts), np.random.default_rng(0)
    thetas = people([0.8, 0.9, 1.0, 1.1])
    state, plan = facts.root(thetas[0]), []
    while not state.done:
        plan.append(search(state, facts, thetas, config, rng))
        state = transition.sample(state, plan[-1])
    assert plan == [MoveBase(0), Grasp(0), MoveLimb(1)]


def test_the_tree_stops_short_of_a_goal_most_people_cannot_reach():
    config = PomdpConfig(num_waypoints=4)
    facts = example_facts(0.9)
    transition, rng = Transition(facts), np.random.default_rng(0)
    # Waypoints at 0.6, 0.7, 0.8, 0.9: all reach the second, one in four the last.
    thetas = people([0.75, 0.78, 0.82, 1.0])
    grasped = transition.sample(
        transition.sample(facts.root(thetas[0]), MoveBase(0)), Grasp(0)
    )
    assert search(grasped, facts, thetas, config, rng) == MoveLimb(2)


def test_with_nothing_usable_the_tree_does_not_move():
    config = PomdpConfig(num_waypoints=1)
    facts = example_facts(0.9, num_waypoints=1)
    facts.blocked.add((0, 0, 1))
    assert search(facts.root(), facts, people([1.0]), config, np.random.default_rng(0)) == Stop()


def test_the_plan_is_the_sequence_the_tree_picks_and_skips_an_unsolvable_move(monkeypatch):
    facts = example_facts(0.7, num_waypoints=1, pairs=((0, 0), (1, 1)))
    attempts = []

    def fake_mpc(ctx, state, target):
        attempts.append(target)
        return None if len(attempts) == 1 else ("torques", "reached")

    monkeypatch.setattr(pomdp_planner, "solve_limb_motion", fake_mpc)
    ctx = SimpleNamespace(limb_name="limb", start_base_pose="start")
    session = Session(ctx, PomdpConfig(num_waypoints=1), np.random.default_rng(0))
    thetas = people([0.8, 0.9, 1.0, 1.1])
    plan = session._plan(facts, session._root(facts), thetas, time.time() + 60)

    assert [step for step, _ in plan] == ["move_base", "grasp", "move_limb"]
    # The first chosen pair failed the MPC; the plan uses the other.
    assert len(facts.blocked) == 1 and len(attempts) == 2
    (blocked_base, _, _), = facts.blocked
    assert plan[0][1][2] is facts.bases[1 - blocked_base]
    assert plan[2][1][-2:] == ("torques", "reached")


def test_both_solvers_plan_the_same_safe_goal():
    facts = example_facts(0.7, num_waypoints=1)
    thetas = people([0.8, 0.9, 1.0, 1.1])
    for solver in ("pouct", "pomcp"):
        config = PomdpConfig(num_waypoints=1, solver=solver)
        transition, rng = Transition(facts), np.random.default_rng(0)
        state, plan = facts.root(thetas[0]), []
        while not state.done:
            plan.append(search(state, facts, thetas, config, rng))
            state = transition.sample(state, plan[-1])
        assert plan == [MoveBase(0), Grasp(0), MoveLimb(1)], solver


def test_after_a_move_only_the_particles_it_clears_are_believed():
    facts = example_facts(0.9)
    thetas = people([0.75, 0.78, 0.82, 1.0])
    start = facts.root()
    # Waypoint 3 is at scale 0.8: the two narrowest people meet their limit.
    assert np.allclose(surviving(facts, start, MoveLimb(3), thetas), thetas[2:])
    assert len(surviving(facts, start, MoveBase(0), thetas)) == 4

    config = PomdpConfig(num_waypoints=4, solver="pomcp")
    transition, rng = Transition(facts), np.random.default_rng(0)
    grasped = transition.sample(transition.sample(start, MoveBase(0)), Grasp(0))
    agent, solver = make_agent(grasped, facts, thetas, config, rng), make_solver(facts, config)
    solver.plan(agent)
    solver.update(agent, MoveLimb(3), Clear())
    # POMCP's kept subtree holds only states the search reached it with.
    assert {p.theta[0] for p in agent.belief.particles} <= {0.82, 1.0}


def test_pomcp_keeps_one_tree_down_the_plan(monkeypatch):
    facts = example_facts(0.7, num_waypoints=2)
    monkeypatch.setattr(pomdp_planner, "solve_limb_motion", lambda *_: ("torques", "reached"))
    ctx = SimpleNamespace(limb_name="limb", start_base_pose="start")
    built = []
    real = pomdp_planner.make_solver
    monkeypatch.setattr(
        pomdp_planner, "make_solver", lambda *a: built.append(1) or real(*a)
    )
    thetas = people([0.8, 0.9, 1.0, 1.1])
    for solver, trees in (("pomcp", 1), ("pouct", None)):
        built.clear()
        config = PomdpConfig(num_waypoints=2, solver=solver)
        session = Session(ctx, config, np.random.default_rng(0))
        plan = session._plan(facts, session._root(facts), thetas, time.time() + 60)
        assert [step for step, _ in plan][:2] == ["move_base", "grasp"]
        if trees is not None:
            assert len(built) == trees
        else:
            assert len(built) == len(plan)
