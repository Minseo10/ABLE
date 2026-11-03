"""Live-camera evidence and preview authorization, with no hardware connections."""

import hashlib
import json
import time
from pathlib import Path

import pytest
from sprout_experiment import live_bundle as bundle
from sprout_experiment.initial_settings import fill_initial_settings


def test_snapshot_metadata_bound_to_image_and_calibration(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"synthetic image bytes")
    projection = {
        "camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "rvec": [0, 0, 0],
        "tvec": [0, 0, 1],
    }
    cfg = {
        "camera_index": 0,
        "camera_calibration": {"serial": "test", "image_height": 10, "image_width": 10},
        "projection": projection,
    }
    meta = {
        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "cameras": [
            {
                "serial": "test",
                "captured_at_unix": time.time(),
                "calibration_sha256": bundle.calibration_digest(projection),
            },
            {},
        ],
        "image_shape": [20, 10, 3],
    }
    image.with_suffix(".json").write_text(json.dumps(meta))
    assert bundle.camera_evidence(image, cfg)["serial"] == "test"
    cfg["projection"]["tvec"][0] = 0.1
    with pytest.raises(ValueError, match="calibration differs"):
        bundle.camera_evidence(image, cfg)
    image.write_bytes(b"replaced image")
    with pytest.raises(ValueError, match="differs from its timestamp"):
        bundle.camera_evidence(image, cfg)


def test_bundle_readiness_is_conditional_and_hashes_live_inputs(tmp_path, monkeypatch):
    reference = tmp_path / "ref.jpg"
    reference.write_bytes(b"test reference")
    (tmp_path / "detection.json").write_text(
        json.dumps({"reference_image": str(reference)})
    )
    config = json.loads(
        (Path(__file__).parents[1] / "config/sprout_live.example.json").read_text()
    )
    config.update(
        execution_enabled=False,
        verified={"physical_stop": False},
        obstacles=[{"center": None, "half_size": None}],
        held_forearm={"center": None, "half_size": None},
    )
    config["hardware"]["measured_torque_limit_nm"] = None
    fill_initial_settings(config)
    config["servo"]["camera_max_age_s"] = 1
    live = tmp_path / "source-live.json"
    live.write_text(json.dumps(config))
    output = tmp_path / "preview"
    output.mkdir()
    for name in ("preview.mp4", "scene.json", "trajectory.json"):
        (output / name).write_text("[]")
    report = {
        "completed": True,
        "observed": {"base_xyyaw": [0, 0, 0], "arm_joints_rad": [0] * 7},
        "plan": {},
    }
    monkeypatch.setattr(bundle, "model_digest", lambda _: "test model digest")
    checked = []

    class Kinematics:
        def __init__(self, *args):
            pass

        def validate_trajectory(self, *args):
            checked.append(True)

    monkeypatch.setattr(bundle, "LiveKinematics", Kinematics)
    state = {
        "base": {
            "acquisition_timestamp_available": True,
            "captured_at_unix": time.time(),
        }
    }
    bundle.prepare_live_bundle(
        report,
        {"tidybot_model": "unused"},
        {"source": "live_camera"},
        state,
        live,
        output,
    )
    assert report["execution_ready"] and checked
    assert "maximum_observation_age_s" not in report
    reviewed = json.loads((output / "live.json").read_text())
    assert reviewed["servo"]["camera_max_age_s"] == 1.0
    assert reviewed["obstacles"] == [] and reviewed["held_forearm"] is None
    assert len(report["execution_limitations"]) == 2
    assert set(report["artifact_sha256"]) == {
        "preview.mp4",
        "trajectory.json",
        "scene.json",
        "live.json",
        "reference.jpg",
    }
    state["base"]["acquisition_timestamp_available"] = False
    bundle.prepare_live_bundle(
        report,
        {"tidybot_model": "unused"},
        {"source": "live_camera"},
        state,
        live,
        output,
    )
    assert not report["execution_ready"]
    assert "marker server" in report["blockers"][0]


def test_missing_live_config_cannot_enable_even_successful_preview(tmp_path):
    for name in ("preview.mp4", "scene.json", "trajectory.json"):
        (tmp_path / name).write_text("[]")
    report = {"completed": True}
    bundle.prepare_live_bundle(report, {}, None, None, None, tmp_path)
    assert not report["execution_ready"] and report["blockers"]
