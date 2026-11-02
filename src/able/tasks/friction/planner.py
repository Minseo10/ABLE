"""POMCPOW on the puck-pushing POMDP, as an able.api Planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from able.planners.pomcpow import POMCPOWPlanner
from able.tasks.friction.belief import FrictionBelief
from able.tasks.friction.planning.model import PuckPushingModel, PuckState

if TYPE_CHECKING:
    from able.tasks.friction.task import FrictionState


@dataclass
class FrictionPlanner(POMCPOWPlanner):
    """Plans one push from the belief's particles; rollouts slide on the belief mean."""

    model: PuckPushingModel = field(kw_only=True)

    def pomdp(self, state: FrictionState, belief: FrictionBelief) -> PuckPushingModel:
        self.model.rollout_frictions = tuple(float(f) for f in belief.mean)
        return self.model

    def particles(self, state: FrictionState, belief: FrictionBelief) -> list[PuckState]:
        return [self.model.initial_state(state.puck, state.robot, f, state.pushes) for f in belief.particles]
