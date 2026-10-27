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
        "navigation": {"verified": False, "speed_m_s": 0.025},
        "hardware": {"gripper_force_percent": 4, "measured_torque_limit_nm": None},
        "motion": {"open_position": 0.0, "translation_speed_m_s": 0.005},
        "obstacles": [{"center": None, "half_size": None}],
    }
    original = copy.deepcopy(config)
    assert fill_initial_settings(config)
    for section in ("verified", "navigation", "obstacles", "execution_enabled"):
        assert config[section] == original[section]
    assert config["hardware"]["gripper_force_percent"] == 4
    assert config["hardware"]["measured_torque_limit_nm"] is None
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
    assert "measured_torque_limit_nm" not in config["hardware"]


def test_invalid_model_does_not_partially_mutate_config(tmp_path):
    path = tmp_path / "invalid.xml"
    path.write_text("<mujoco><worldbody/></mujoco>")
    config = {"schema": "sprout_live_v1", "hardware": {"gripper_force_percent": 4}}
    original = copy.deepcopy(config)
    with pytest.raises(KeyError):
        fill_initial_settings(config, path)
    assert config == original
