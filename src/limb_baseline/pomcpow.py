"""POMCPOW (Sunberg & Kochenderfer, 2018): online POMDP search with continuous actions
and observations, by progressive widening and weighted particles per observation node.

Takes a pomdp_py `Agent` like POUCT; the policy model must offer
`new_action(state, tried, rng, config)` and `random_action(state, rng)`
(see `pomdp_model.Policy`).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class _ObservationNode:
    visits: int = 0
    actions: dict = field(default_factory=dict)
    # The states this observation was reached with, and their running weight total.
    states: list = field(default_factory=list)
    cumulative: list = field(default_factory=list)
    exhausted: bool = False


@dataclass
class _ActionNode:
    visits: int = 0
    value: float = 0.0
    children: dict = field(default_factory=dict)
    # Times each child observation was generated.
    counts: dict = field(default_factory=dict)


class POMCPOW:
    """`plan(agent)` returns the best root action after `config.pomcpow_sims` simulations."""

    def __init__(self, config, max_depth: int, rng: np.random.Generator | None = None,
                 discount: float = 1.0) -> None:
        self.config, self.max_depth, self.discount = config, max_depth, discount
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.root: _ObservationNode | None = None

    def plan(self, agent):
        self._agent = agent
        particles = list(agent.belief.particles)
        self.root = _ObservationNode(
            states=particles, cumulative=list(range(1, len(particles) + 1))
        )
        for _ in range(self.config.pomcpow_sims):
            state = particles[self.rng.integers(len(particles))]
            self._simulate(state, self.root, self.max_depth)
        # The most visited action: values of rarely tried continuous ones are noisy.
        return max(self.root.actions.items(), key=lambda kv: (kv[1].visits, kv[1].value))[0]

    def _widened(self, visits: int, size: int, widening) -> bool:
        k, alpha = widening
        return size <= k * visits**alpha

    def _select_action(self, state, node: _ObservationNode):
        if not node.exhausted and self._widened(
            node.visits, len(node.actions), self.config.action_widening
        ):
            action = self._agent.policy_model.new_action(
                state, node.actions, self.rng, self.config
            )
            if action is None:
                node.exhausted = True
            else:
                node.actions[action] = _ActionNode()
        log_visits = math.log(max(node.visits, 1))

        def ucb(item):
            child = item[1]
            if child.visits == 0:
                return math.inf
            return child.value + self.config.exploration * math.sqrt(log_visits / child.visits)

        return max(node.actions.items(), key=ucb)

    def _rollout(self, state, depth: int) -> float:
        agent, total, scale = self._agent, 0.0, 1.0
        while depth > 0 and not state.done:
            action = agent.policy_model.random_action(state, self.rng)
            next_state = agent.transition_model.sample(state, action)
            total += scale * agent.reward_model.sample(state, action, next_state)
            state, scale, depth = next_state, scale * self.discount, depth - 1
        return total

    def _simulate(self, state, node: _ObservationNode, depth: int) -> float:
        if depth == 0 or state.done:
            return 0.0
        agent = self._agent
        action, child = self._select_action(state, node)
        next_state = agent.transition_model.sample(state, action)
        observation = agent.observation_model.sample(next_state, action)
        reward = agent.reward_model.sample(state, action, next_state)
        if self._widened(child.visits, len(child.children), self.config.observation_widening):
            child.counts[observation] = child.counts.get(observation, 0) + 1
        else:
            # Reuse an observation, picked by how often it was generated.
            seen = list(child.counts)
            counts = np.array([child.counts[o] for o in seen], dtype=float)
            observation = seen[self.rng.choice(len(seen), p=counts / counts.sum())]
        new = observation not in child.children
        if new:
            child.children[observation] = _ObservationNode()
        below = child.children[observation]
        weight = agent.observation_model.probability(observation, next_state, action)
        below.states.append(next_state)
        below.cumulative.append((below.cumulative[-1] if below.cumulative else 0.0) + weight)
        if new:
            total = reward + self.discount * self._rollout(next_state, depth - 1)
        else:
            total_weight = below.cumulative[-1]
            if total_weight > 0:
                pick = bisect.bisect_right(below.cumulative, self.rng.random() * total_weight)
            else:
                pick = int(self.rng.integers(len(below.states)))
            next_state = below.states[min(pick, len(below.states) - 1)]
            reward = agent.reward_model.sample(state, action, next_state)
            total = reward + self.discount * self._simulate(next_state, below, depth - 1)
        node.visits += 1
        child.visits += 1
        child.value += (total - child.value) / child.visits
        return total
