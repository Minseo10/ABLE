"""Initial settings preserve user calibration and do not grant motion approval."""

import copy
import math
import xml.etree.ElementTree as ET

import pytest
from sprout_experiment.initial_settings import fill_initial_settings


def test_defaults_preserve_zero_custom_limits_and_verification():
    config = {
        "schema": "sprout_live_v1",
        "execution_enabled": False,
        "verified": {"physical_stop": False},
        "navigation": {"verified": False, "speed_m_s": 0.015},
        "hardware": {"gripper_force_percent": 4, "measured_torque_limit_nm": None},
        "motion": {"open_position": 0.0, "translation_speed_m_s": 0.005},
        "obstacles": [{"center": None, "half_size": None}],
    }
    original = copy.deepcopy(config)
    assert fill_initial_settings(config)
    for section in ("verified", "obstacles", "execution_enabled"):
        assert config[section] == original[section]
    for key, value in original["navigation"].items():
        assert config["navigation"][key] == value
    assert config["hardware"]["gripper_force_percent"] == 4
    assert config["hardware"]["measured_torque_limit_nm"] == [39.0] * 4 + [9.0] * 3
    assert config["motion"]["open_position"] == 0.0
    assert config["motion"]["translation_speed_m_s"] == 0.005
    assert fill_initial_settings(config) == []


def test_model_bounds_and_mount_are_derived_not_joint_anchor(tmp_path):
    model = ET.Element("mujoco")
    ET.SubElement(model, "compiler", angle="radian")
    world = ET.SubElement(model, "worldbody")
    body = ET.SubElement(world, "body", name="gen3/base_link", pos="0 0 0.4")
    for i in range(1, 8):
        body = ET.SubElement(body, "body", name=f"link_{i}", pos="0 0 0.15")
        attributes = {"name": f"joint_{i}", "type": "hinge", "limited": "false"}
        if i == 4:
            attributes.update(limited="true", range="-1.2 1.2")
        ET.SubElement(body, "joint", **attributes)
        ET.SubElement(body, "geom", type="sphere", size="0.01", mass="0.1")
    ET.SubElement(body, "site", name="pinch_site")
    for name, x in (("left_pad", 0.05), ("right_pad", -0.05)):
        pad = ET.SubElement(body, "body", name=name, pos=f"{x} 0 0")
        ET.SubElement(pad, "geom", type="box", size="0.005 0.01 0.02")
    path = tmp_path / "model.xml"
    ET.ElementTree(model).write(path)
    config = {"schema": "sprout_live_v1", "execution_enabled": False}
    fill_initial_settings(config, path)
    assert config["geometry"]["floor_z_arm_m"] == pytest.approx(-0.4)
    assert config["geometry"]["pad_drop_m"] == pytest.approx(0.02)
    assert config["hardware"]["joint_min_rad"][3] == pytest.approx(-1.2)
    assert config["hardware"]["joint_max_rad"][3] == pytest.approx(1.2)
    assert config["hardware"]["joint_min_rad"][0] == -math.pi
    assert config["hardware"]["joint_max_rad"][0] == math.pi
    assert not config["execution_enabled"]
    assert "verified" not in config
    assert config["hardware"]["measured_torque_limit_nm"] == [39.0] * 4 + [9.0] * 3


def test_invalid_model_does_not_partially_mutate_config(tmp_path):
    path = tmp_path / "invalid.xml"
    path.write_text("<mujoco><worldbody/></mujoco>")
    config = {"schema": "sprout_live_v1", "hardware": {"gripper_force_percent": 4}}
    original = copy.deepcopy(config)
    with pytest.raises(KeyError):
        fill_initial_settings(config, path)
    assert config == original


def test_configured_torque_limits_survive_setup():
    custom = [12.0, 12.0, 12.0, 12.0, 3.0, 3.0, 3.0]
    config = {
        "schema": "sprout_live_v1",
        "hardware": {"measured_torque_limit_nm": custom.copy()},
    }
    changed = fill_initial_settings(config)
    assert "hardware.measured_torque_limit_nm" not in changed
    assert config["hardware"]["measured_torque_limit_nm"] == custom


def test_old_fast_settings_are_capped_and_missing_navigation_is_filled():
    config = {
        "schema": "sprout_live_v1",
        "hardware": {
            "velocity_rad_s": [math.radians(10)] * 7,
            "acceleration_rad_s2": [math.radians(20)] * 7,
        },
        "motion": {"translation_speed_m_s": 0.02},
        "navigation": {"speed_m_s": 0.04, "timeout_s": 120.0},
    }
    changed = fill_initial_settings(config)
    assert config["motion"]["translation_speed_m_s"] == 0.01
    assert config["navigation"]["speed_m_s"] == 0.02
    assert config["navigation"]["timeout_s"] == 300.0
    assert config["hardware"]["velocity_rad_s"] == [math.radians(5)] * 7
    assert config["hardware"]["acceleration_rad_s2"] == [math.radians(10)] * 7
    assert config["hardware"]["measured_torque_limit_nm"] == [39.0] * 4 + [9.0] * 3
    assert "motion.translation_speed_m_s" in changed
    assert fill_initial_settings(config) == []
