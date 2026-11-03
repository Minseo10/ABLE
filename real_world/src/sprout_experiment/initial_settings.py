"""Initial low-speed settings for preview-approved execution.

Driver percentages are not measured contact forces. Model-derived dimensions are
nominal geometry. Kinova torque defaults are overload thresholds, not contact-force
limits. Slower custom settings and other non-null values are preserved.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def fill_initial_settings(
    config: dict[str, Any], model_path: Path | None = None
) -> list[str]:
    """Fill missing controls/model geometry and enforce first-trial speed caps.

    Joint torque defaults use the Gen3 7 DoF joint soft limits (User Guide R07,
    Table 44). They include gravity and do not establish a contact-force bound.
    Manual obstacle/held-limb boxes are optional and are not fabricated here.
    """
    if config.get("schema") != "sprout_live_v1":
        return []
    defaults = {
        "geometry": {
            "minimum_clearance_m": 0.01,
            "downward_error_budget_m": 0.01,
        },
        "hardware": {
            "velocity_rad_s": [math.radians(5)] * 7,
            "acceleration_rad_s2": [math.radians(10)] * 7,
            # Kinova Gen3 User Guide R07, Table 44 and the PRPL Gen3 URDF:
            # https://www.kinovarobotics.com/uploads/User-Guide-Gen3-R07.pdf#page=104
            # Absolute measured torque (gravity included), not an external-force limit.
            "measured_torque_limit_nm": [39.0] * 4 + [9.0] * 3,
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
            "translation_speed_m_s": 0.01,
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
            "orientation_speed_rad_s": 0.05,
            "orientation_tolerance_rad": 0.05,
            "confirmation_timeout_s": 120.0,
            "gripper_stall_samples": 5,
            "gripper_stall_epsilon": 0.002,
            "grasp_reading_min": 0.1,
            "grasp_reading_max": 0.9,
            "collision_margin_m": 0.005,
            "startup_timeout_s": 5.0,
        },
        "navigation": {
            "speed_m_s": 0.02,
            "yaw_speed_rad_s": 0.05,
            "acceleration_m_s2": 0.04,
            "yaw_acceleration_rad_s2": 0.075,
            "position_tolerance_m": 0.005,
            "yaw_tolerance_rad": 0.01,
            "path_tolerance_m": 0.015,
            "timeout_s": 300.0,
            "period_s": 0.02,
            "arm_tolerance_rad": 0.01,
            "camera_max_age_s": 0.25,
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
    # First-trial speed caps also lower older, faster defaults. Slower custom
    # settings remain intact. Invalid values are left for runtime validation.
    for section, keys in {
        "hardware": ("velocity_rad_s", "acceleration_rad_s2"),
        "motion": ("translation_speed_m_s",),
        "servo": ("orientation_speed_rad_s",),
        "navigation": (
            "speed_m_s",
            "yaw_speed_rad_s",
            "acceleration_m_s2",
            "yaw_acceleration_rad_s2",
        ),
    }.items():
        for key in keys:
            current, cap = config[section][key], defaults[section][key]
            if isinstance(cap, list):
                if not isinstance(current, (list, tuple)) or len(current) != len(cap):
                    continue
                if not all(
                    isinstance(v, (int, float))
                    and not isinstance(v, bool)
                    and math.isfinite(v)
                    and v > 0
                    for v in current
                ):
                    continue
                limited = [min(value, bound) for value, bound in zip(current, cap)]
            elif isinstance(current, (int, float)) and not isinstance(current, bool):
                limited = min(current, cap)
            else:
                continue
            if limited != current:
                config[section][key] = limited
                name = f"{section}.{key}"
                if name not in changed:
                    changed.append(name)
    # Old 120s route timeouts can expire after slowing a long base transit.
    if config["navigation"]["timeout_s"] == 120.0:
        config["navigation"]["timeout_s"] = defaults["navigation"]["timeout_s"]
        changed.append("navigation.timeout_s")
    return changed
