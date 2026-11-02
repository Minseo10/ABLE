"""psi's features and labels for the limb: one row per MPC control step."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from kinder.envs.dynamic3d.limb_utils import NUM_ROBOT_JOINTS
from numpy.typing import NDArray

from able.planners.mpc import control_basis as _control_basis
from able.tasks.limb.planning.stream import (
    MPCConfig,
    TorqueTrajectory,
)
from able.tasks.limb.utils import RolloutLog

# Limb (6) and robot (7) positions, goal (6), belief (6) and step: what every model reads.
BASE_FEATURES = 26


def control_basis(mpc: MPCConfig) -> NDArray[np.float64]:
    """The `(horizon, num_control_points)` map the MPC's `_expand` applies."""
    return _control_basis(mpc.horizon, mpc.num_control_points)


def horizon_window(
    robot_torques: list[list[float]], mpc: MPCConfig
) -> NDArray[np.float64]:
    """First horizon of per-sim-step torques, decimated to control steps, padded with the last."""
    torques = np.asarray(robot_torques, dtype=np.float64)
    assert torques.ndim == 2 and torques.shape[1] == NUM_ROBOT_JOINTS
    window = torques[:: mpc.action_repeat][: mpc.horizon]
    if len(window) < mpc.horizon:
        window = np.vstack([window, np.repeat(window[-1:], mpc.horizon - len(window), axis=0)])
    return window


def project_torques(
    robot_torques: list[list[float]], mpc: MPCConfig
) -> NDArray[np.float64]:
    """Least-squares control points for the first horizon of per-sim-step torques."""
    torques = np.asarray(robot_torques, dtype=np.float64)
    assert torques.ndim == 2 and torques.shape[1] == NUM_ROBOT_JOINTS
    control_steps = torques[:: mpc.action_repeat]
    window = control_steps[: mpc.horizon]
    if len(window) < mpc.horizon:
        padding = np.repeat(window[-1:], mpc.horizon - len(window), axis=0)
        window = np.vstack([window, padding])
    basis = control_basis(mpc)
    solution, *_ = np.linalg.lstsq(basis, window, rcond=None)
    return solution


def segment_features(
    limb_positions,
    robot_positions,
    goal: NDArray[np.float64],
    belief_features: NDArray[np.float64],
    step: float = 0.0,
    limb_velocities=None,
    robot_velocities=None,
    previous_torque=None,
) -> NDArray[np.float64]:
    """Features for one `move_limb` control step; velocities and previous torque are for `:hist`."""
    parts = [
        np.asarray(limb_positions, dtype=np.float64),
        np.asarray(robot_positions, dtype=np.float64)[:NUM_ROBOT_JOINTS],
        np.asarray(goal, dtype=np.float64),
        np.asarray(belief_features, dtype=np.float64),
        [float(step)],
    ]
    if limb_velocities is not None:
        previous = np.zeros(NUM_ROBOT_JOINTS) if previous_torque is None else previous_torque
        parts += [
            np.asarray(limb_velocities, dtype=np.float64),
            np.asarray(robot_velocities, dtype=np.float64)[:NUM_ROBOT_JOINTS],
            np.asarray(previous, dtype=np.float64)[:NUM_ROBOT_JOINTS],
        ]
    return np.concatenate(parts)


def state_features(state, goal, belief_features, step, previous_torque=None):
    """`segment_features` for a captured simulator state, as a warm-start hook sees it."""
    return segment_features(
        state.limb_positions, state.robot_positions, goal, belief_features, step,
        state.limb_velocities, state.robot_velocities, previous_torque,
    )


@dataclass
class SegmentSample:
    """One control-step training row: `psi` = fitted control points, `torques` = raw horizon."""

    features: NDArray[np.float64]
    psi: NDArray[np.float64]
    torques: NDArray[np.float64] | None = None


def _move_limb_segments(plan) -> list[tuple]:
    """`(start state, goal conf, trajectory)` for every `move_limb` in the plan."""
    segments = []
    for name, args in plan:
        if name == "move_limb":
            _, _, _, s1, _, goal_conf, trajectory, _ = args
            assert isinstance(trajectory, TorqueTrajectory)
            segments.append((s1, goal_conf, trajectory))
    return segments


def dense_rows(segments, log: RolloutLog, belief_features, mpc: MPCConfig) -> list:
    """One row per control step; the log holds post-step states, so a step reads the prior entry."""
    rows: list[SegmentSample] = []
    offset = 0
    for start_state, goal_conf, trajectory in segments:
        torques = trajectory.robot_torques
        goal = np.asarray(goal_conf.positions, dtype=np.float64)
        for step in range(len(torques) // mpc.action_repeat):
            index = offset + step * mpc.action_repeat - 1
            if index >= len(log.limb_positions):
                break
            if index < offset:
                limb, robot = start_state.limb_positions, start_state.robot_positions
                limb_velocity = start_state.limb_velocities
                robot_velocity = start_state.robot_velocities
            else:
                limb, robot = log.limb_positions[index], log.robot_positions[index]
                limb_velocity = log.limb_velocities[index]
                robot_velocity = log.robot_velocities[index]
            previous = torques[(step - 1) * mpc.action_repeat] if step else None
            rows.append(
                SegmentSample(
                    features=segment_features(
                        limb, robot, goal, belief_features, step,
                        limb_velocity, robot_velocity, previous,
                    ),
                    psi=project_torques(torques[step * mpc.action_repeat :], mpc),
                    torques=horizon_window(torques[step * mpc.action_repeat :], mpc),
                )
            )
        offset += len(torques)
    return rows


def limb_segments(features) -> np.ndarray:
    """One group per planned segment: a new one starts at step 0."""
    return np.cumsum(np.asarray(features)[:, BASE_FEATURES - 1] == 0)
