"""psi for the friction task: a learned proposal of root pushes for POMCPOW."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from able.api import Examples, PsiSpec
from able.tasks.friction.env.puckpushing_utils import Push
from able.tasks.friction.planning.model import PuckPushingModel

X_DIM, Y_DIM = 13, 4


@dataclass
class FrictionPsi(PsiSpec):
    """x = puck, goal, floor one-hot, belief mean and std; y = (cos, sin, speed, stance) of a root push."""

    model: PuckPushingModel
    min_visit_fraction: float = 0.2  # root actions kept, relative to the most visited
    mix: float = 0.5  # chance a proposal comes from the learner
    k: int = 16  # learner draws cached per puck cell
    cell: float = 0.05  # cache resolution (m)

    def features(self, state, belief) -> NDArray[np.float64]:
        return self._features(np.asarray(state.puck, np.float64), belief)

    def _features(self, puck, belief) -> NDArray[np.float64]:
        world = self.model.world
        floor = np.zeros(3)
        surface = world.surface_at(puck)
        if surface >= 0:
            floor[surface] = 1.0
        return np.concatenate([puck, world.goal, floor, belief.mean, belief.std])

    def examples(self, episode, belief) -> Examples:
        x, y, groups = [], [], []
        for index, s in enumerate(episode.steps):
            if not s.result.success or not s.result.trace:
                continue
            top = max(visits for _, visits, _ in s.result.trace)
            feats = self.features(s.state, belief)
            for action, visits, _ in s.result.trace:
                if visits >= self.min_visit_fraction * top or action == s.result.plan:
                    x.append(feats)
                    y.append(encode(action))
                    groups.append(index)
        if not x:
            return Examples.empty(X_DIM, Y_DIM)
        return Examples(np.array(x), np.array(y), np.array(groups, np.int64))

    def proposal(self, learner, state, belief):
        """ActionProposer: learner draw w.p. `mix`, else (or if all tried) model.propose."""
        del state
        model, cache = self.model, {}

        def propose(node_state, states, tried, rng):
            del states
            if rng.random() < self.mix:
                puck = node_state.puck_array
                key = tuple(np.round(puck / self.cell).astype(int))
                if key not in cache:
                    draws = learner.sample(self._features(puck, belief)[None], k=self.k,
                                           seed=int(rng.integers(2**31)))[0]
                    pushes = (decode(model, row) for row in draws)
                    cache[key] = list(dict.fromkeys(p for p in pushes
                                                    if model.world.push_feasible(puck, p.angle, p.stance)))
                for push in cache[key]:
                    if push not in tried:
                        return push
            return model.propose(node_state, tried, rng)

        return propose


def encode(push: Push) -> list[float]:
    return [math.cos(push.angle), math.sin(push.angle), push.speed, push.stance or 0.0]


def decode(model: PuckPushingModel, row) -> Push:
    """Label to push, rounded and limited as model.propose does."""
    phys = model.world.physics
    angle = round(math.atan2(row[1], row[0]), 4)
    speed = round(float(np.clip(row[2], phys.min_speed, phys.max_speed)), 4)
    if model.stance_mode == "auto":
        return Push(angle, speed)
    if model.stance_mode == "grid":
        stance = float(model.stances[np.argmin(np.abs(model.stances - row[3]))])
    else:
        stance = float(row[3])
    limit = phys.max_stance
    return Push(angle, speed, min(max(round(stance, 4), -limit), limit))
