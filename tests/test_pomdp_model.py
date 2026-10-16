"""Tests for the POMDP model on hand-made facts: no simulator, no MPC, no search."""

import numpy as np

from limb_baseline.human import active_motions
from limb_baseline.pomdp_model import (
    Clear,
    Grasp,
    Limit,
    MoveBase,
    MoveLimb,
    Observation,
    Policy,
    PomdpConfig,
    Reward,
    Stop,
    Transition,
    example_facts,
)


def person(scale: float) -> tuple:
    return (scale,) * len(active_motions())


def test_a_move_stops_at_the_first_limit_on_its_way():
    facts = example_facts(1.0)
    arrived, motion = facts.path.advance(0.0, 1.0, person(1.1))
    assert (arrived, motion) == (1.0, -1)
    stopped, motion = facts.path.advance(0.0, 1.0, person(0.8))
    assert motion == 0 and np.isclose(facts.path.scale(stopped)[0], 0.8)


def test_actions_follow_the_domains_preconditions():
    facts = example_facts(0.9, pairs=((0, 0), (1, 1)))
    policy, transition = Policy(facts), Transition(facts)
    state = facts.root(person(1.0))
    # Empty hand at the start: only the base may move.
    assert policy.get_all_actions(state) == [MoveBase(0), MoveBase(1), Stop()]

    state = transition.sample(state, MoveBase(1))
    assert state.base_pose == (2.0, 0.0, 0.0)
    # The base moves once; only this pose's grasps are offered.
    assert policy.get_all_actions(state) == [Grasp(1), Stop()]

    state = transition.sample(state, Grasp(1))
    assert policy.get_all_actions(state) == [MoveLimb(k) for k in (1, 2, 3, 4)] + [Stop()]

    state = transition.sample(state, MoveLimb(2))
    assert np.allclose(state.limb, facts.target(2)) and not state.done
    assert policy.get_all_actions(state) == [MoveLimb(3), MoveLimb(4), Stop()]


def test_a_move_is_withheld_where_it_is_unreachable_disallowed_or_unsolved():
    facts = example_facts(0.9)
    facts.reach[(0, 0)][0] = None
    facts.allowed[1] = False
    facts.blocked.add((0, 0, 3))
    assert facts.open(0, 0) == [4]
    facts.blocked.add((0, 0, 4))
    assert not facts.usable()
    assert Policy(facts).get_all_actions(facts.root(person(1.0))) == [Stop()]


def test_meeting_a_limit_ends_the_episode_and_forfeits_the_progress():
    config = PomdpConfig(num_waypoints=4, limit_penalty=2.0)
    facts = example_facts(0.9)
    transition, reward = Transition(facts), Reward(config)
    grasped = transition.sample(
        transition.sample(facts.root(person(0.75)), MoveBase(0)), Grasp(0)
    )

    halfway = transition.sample(grasped, MoveLimb(2))
    assert reward.sample(grasped, MoveLimb(2), halfway) == 0.5 * config.goal_reward - 1.0

    stopped = transition.sample(halfway, MoveLimb(4))
    assert stopped.hit == 0 and stopped.done
    assert np.isclose(facts.path.scale(stopped.along)[0], 0.75)
    assert reward.sample(halfway, MoveLimb(4), stopped) == -1.0 - 0.5 * config.goal_reward - 2.0

    observation = Observation(facts, config, np.random.default_rng(0))
    assert observation.sample(halfway, MoveLimb(2)) == Clear()
    reading = observation.sample(stopped, MoveLimb(4))
    assert isinstance(reading, Limit) and reading.motion == 0
    assert abs(reading.bin * config.resolution - 0.75) < 0.05
