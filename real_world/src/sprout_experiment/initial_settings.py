"""Initial low-speed settings, without inventing physical verification.

Driver percentages are not measured contact forces. Model-derived dimensions are
nominal geometry. Existing non-null settings and all verification flags survive.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def fill_initial_settings(
    config: dict[str, Any], model_path: Path | None = None
) -> list[str]:
    """Fill missing control parameters and optional model geometry in place.

    Torque thresholds and obstacle/held-limb bounds require additional evidence.
    They are deliberately not inferred from actuator ratings or a tape patch.
    """
    if config.get("schema") != "sprout_live_v1":
        return []
    defaults = {
        "geometry": {
            "minimum_clearance_m": 0.01,
            "downward_error_budget_m": 0.01,
        },
        "hardware": {
            "velocity_rad_s": [math.radians(10)] * 7,
            "acceleration_rad_s2": [math.radians(20)] * 7,
            "gripper_force_percent": 10.0,
            "gripper_speed_percent": 10.0,
            "command_timeout_s": 0.5,
            "feedback_timeout_s": 0.1,
            "startup_timeout_s": 5.0,
        },
        "motion": {
            "period_s": 0.05,
            "timeout_s": 120.0,
            "feedback_max_age_s": 0.1,
            "translation_speed_m_s": 0.02,
            "max_command_lead_m": 0.01,
            "position_tolerance_m": 0.005,
            "yaw_tolerance_rad": 0.05,
            "settled_speed_rad_s": 0.02,
            "stable_samples": 3,
            "open_position": 0.0,
            "close_position": 1.0,
            "gripper_position_rate_s": 0.1,
            "gripper_tolerance": 0.02,
        },
        "servo": {
            "tolerance_m": 0.005,
            "max_step_m": 0.005,
            "max_travel_m": 0.03,
            "timeout_s": 30.0,
            "stable_frames": 3,
            "max_updates": 20,
            "camera_max_age_s": 0.25,
            "base_tolerance_m": 0.01,
            "base_tolerance_rad": 0.02,
            "path_resolution_rad": 0.02,
            "max_joint_step_rad": 0.1,
            "orientation_speed_rad_s": 0.1,
            "orientation_tolerance_rad": 0.05,
            "confirmation_timeout_s": 120.0,
            "gripper_stall_samples": 5,
            "gripper_stall_epsilon": 0.002,
            "grasp_reading_min": 0.1,
            "grasp_reading_max": 0.9,
            "collision_margin_m": 0.005,
            "startup_timeout_s": 5.0,
        },
        "revalidation_tolerances": {
            "grasp_xyz": 0.01,
            "base_xyyaw": 0.01,
            "arm_joints_rad": 0.03,
        },
    }
    if model_path is not None:
        import mujoco
        from prpl_tidybot.sprout.pad_geometry import box_grasp_geometry

        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        ranges = []
        for i in range(1, 8):
            joint = model.joint(f"joint_{i}").id
            bounds = (
                model.jnt_range[joint]
                if model.jnt_limited[joint]
                else (-math.pi, math.pi)
            )
            low, high = max(float(bounds[0]), -math.pi), min(float(bounds[1]), math.pi)
            if not math.isfinite(low + high) or low >= high:
                raise ValueError(f"Invalid wrapped model bounds for joint {i}")
            ranges.append((low, high))
        defaults["hardware"]["joint_min_rad"] = [pair[0] for pair in ranges]
        defaults["hardware"]["joint_max_rad"] = [pair[1] for pair in ranges]
        # The public model's arm base is distinct from its first joint anchor.
        arm_base = model.body("gen3/base_link").id
        defaults["geometry"]["floor_z_arm_m"] = -float(data.xpos[arm_base, 2])
        defaults["geometry"]["pad_drop_m"] = box_grasp_geometry(model_path, 0.08)[
            "pad_drop_m"
        ]
    changed = []
    for section, values in defaults.items():
        target = config.setdefault(section, {})
        for key, value in values.items():
            if target.get(key) is None:
                target[key] = value
                changed.append(f"{section}.{key}")
    return changed
