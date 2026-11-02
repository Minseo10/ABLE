"""Smoke-test run_episode and particle constraints on a variant's nominal task.

Run each variant in a separate working directory because PDDLStream writes scratch
files. This uses the project's body mass and execution logger, unlike variant_sweep.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from able.tasks.limb.variant import scene_config, select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    select(args.variant)

    # Selection precedes imports that construct variant-dependent defaults.
    from able.tasks.limb import planner
    from able.tasks.limb.human import default_magnitudes
    from able.tasks.limb.planner import (
        ParticleLimits,
        PlannerSettings,
        run_episode,
    )
    from able.tasks.limb.tasks import Task

    config = scene_config()
    task = Task(
        tuple(config.scene.limb_init_joint_positions),
        tuple(config.robot_initial_joints),
        tuple(config.scene.limb_goal_joint_positions),
    )
    # Theta contains dimensionless RoM scale factors, not angles in radians.
    theta = np.ones_like(default_magnitudes())
    contexts = []
    original_build = planner.build_stream_context

    def capture_context(*args, **kwargs):
        ctx = original_build(*args, **kwargs)
        contexts.append(ctx)
        return ctx

    planner.build_stream_context = capture_context
    try:
        result = run_episode(
            task,
            theta,
            believed_theta=theta,
            particle_limits=ParticleLimits(np.tile(theta, (4, 1)), confidence=1.0),
            seed=0,
            settings=PlannerSettings(hard_time_limit=True),
        )
    finally:
        planner.build_stream_context = original_build
    row = dataclasses.asdict(result)
    row.pop("segments")
    row.pop("limb_path")
    row.update(variant=args.variant, particle_constraints=True, body_mass_kg=62.5)
    if contexts:
        row.update(
            motion_outcomes=dict(contexts[0].motion_outcomes),
            replay_max_joint_error=contexts[0].replay_max_joint_error,
            robot_peak_velocity=contexts[0].robot_peak_velocity,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=2) + "\n")
    print("ROW " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
