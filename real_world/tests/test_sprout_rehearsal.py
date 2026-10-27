"""Camera ray projection and preview input semantics; no hardware imports."""

import json
import math
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from sprout_experiment.rehearsal import project


def calibration():
    return {
        "camera_matrix": [[100, 0, 50], [0, 100, 50], [0, 0, 1]],
        "rvec": [math.pi, 0, 0],
        "tvec": [0, 0, 2],
    }


def test_project_uses_target_height_not_floor_homography():
    point = project(np.array([60, 50]), calibration(), 0.2)
    assert point == pytest.approx([0.18, 0, 0.2])
    assert project(np.array([60, 50]), calibration(), 0) == pytest.approx([0.2, 0, 0])


def test_reject_plane_behind_camera():
    with pytest.raises(ValueError, match="behind-camera"):
        project(np.array([50, 50]), calibration(), 3)


def test_preview_initializes_model_from_measured_state(tmp_path, monkeypatch):
    from sprout_experiment.rehearsal import preview

    joints = [0.1 * i for i in range(7)]
    base = [1.2, -0.4, 0.7]
    state = {
        "schema": "sprout_robot_snapshot_v1",
        "arm": {"joints_rad": joints, "received_at_unix": time.time()},
        "base": {"frame": "map", "xyyaw": base, "received_at_unix": time.time()},
    }
    config = {
        "synthetic_demo": False,
        "tidybot_model": "unused",
        "sprout_model": "unused",
        "scene": {"tidybot_base_xyyaw": None, "tidybot_arm_joints_rad": None},
    }
    config_path, state_path = tmp_path / "scene.json", tmp_path / "state.json"
    config_path.write_text(json.dumps(config))
    state_path.write_text(json.dumps(state))

    class ReachedModel(Exception):
        pass

    def model(tidy, sprout, scene):
        assert scene["tidybot_base_xyyaw"] == base
        assert scene["tidybot_arm_joints_rad"] == joints
        assert scene["tidybot_state_source"] == state
        raise ReachedModel

    monkeypatch.setitem(
        sys.modules, "prpl_tidybot.sprout.rehearsal", SimpleNamespace(Rehearsal=model)
    )
    with pytest.raises(ReachedModel):
        preview(
            SimpleNamespace(
                config=config_path, observation=None, robot_state=state_path
            )
        )


def test_automatic_detection_fills_grasp_without_body_values(tmp_path, monkeypatch):
    from pathlib import Path

    import cv2
    import kinder
    from prpl_tidybot.visual_servo import forearm_template
    from sprout_experiment.rehearsal import detect

    image = np.full((256, 128, 3), 90, np.uint8)
    cv2.fillConvexPoly(
        image, cv2.boxPoints(((64, 64), (20, 32), 25)).astype(int), (180, 180, 180)
    )
    path = tmp_path / "image.jpg"
    cv2.imwrite(str(path), image)

    class Tracker:
        def __init__(self, *args, **kwargs):
            pass

        def detect(self, *args):
            return SimpleNamespace(center_uv=(64, 64), quality=1)

    monkeypatch.setattr(forearm_template, "ForearmPatchTracker", Tracker)
    model = (
        Path(kinder.__file__).parent
        / "envs/dynamic3d/models/stanford_tidybot/tidybot.xml"
    )
    config = {
        "reference_image": str(path),
        "camera_index": 0,
        "patch_xywh": [48, 50, 31, 27],
        "search_xywh": [29, 29, 70, 70],
        "minimum_quality": 0.8,
        "minimum_margin": 0.1,
        "auto_box": {"enabled": True, "tidybot_model": str(model)},
        "projection": {
            "camera_matrix": [[300, 0, 64], [0, 300, 64], [0, 0, 1]],
            "rvec": [math.pi, 0, 0],
            "tvec": [0, 0, 1],
            "patch_height_m": 0.08,
            "grasp_height_m": None,
            "grasp_yaw_rad": None,
            "patch_to_grasp_xy_m": None,
        },
    }
    result = detect(config, path, tmp_path / "output")
    assert result["grasp"]["z"] == pytest.approx(0.05063539)
    assert result["grasp"]["yaw"] == pytest.approx(-math.radians(25), abs=0.1)
    assert all(
        config["projection"][key] is not None
        for key in ("grasp_height_m", "grasp_yaw_rad", "patch_to_grasp_xy_m")
    )
    assert result["forearm_box"]["width_m"] < 0.085
    assert "whole-body model omitted" in result["body_pose_source"]
    assert (tmp_path / "output/target_closeup.png").exists()
