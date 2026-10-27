"""Inspect a local, Git-ignored Sprout model without commanding hardware.

No proprietary model data is bundled with this module. Load assets at runtime
from the explicitly supplied directory; never copy them into package data.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def inspect_model(directory: Path, load_mujoco: bool = False) -> dict[str, Any]:
    """Validate URDF mesh paths and optionally compile the supplied MJCF scene.

    This checks asset integrity and model loading, not hardware force safety or
    feasibility of any grasp. MuJoCo's default pose is not a measured ground pose.
    """
    directory = directory.resolve(strict=True)
    robot = ET.parse(directory / "sprout.urdf").getroot()
    meshes = {m.attrib["filename"] for m in robot.findall(".//mesh")}
    for filename in meshes:
        mesh = (directory / filename).resolve()
        if not mesh.is_relative_to(directory) or not mesh.is_file():
            raise ValueError(f"Missing or external URDF mesh: {filename}")
    forearm = robot.find("link[@name='left_radius_link']")
    if forearm is None or not forearm.findall("collision"):
        raise ValueError("Left forearm link/collision geometry not found")
    result: dict[str, Any] = {
        "model_directory": str(directory),
        "robot": robot.get("name"),
        "links": len(robot.findall("link")),
        "joints": len(robot.findall("joint")),
        "verified_mesh_files": len(meshes),
        "forearm_link": "left_radius_link",
        "forearm_collision_elements": len(forearm.findall("collision")),
        "left_arm_joints": [
            joint.get("name")
            for joint in robot.findall("joint")
            if joint.get("name", "").startswith(
                ("left_shoulder_", "left_elbow_", "left_wrist_")
            )
        ],
        "hardware_readiness": "not established by model inspection",
    }
    if load_mujoco:
        # Optional workstation dependency; the installed wheel has no type marker.
        import mujoco  # type: ignore[import-untyped]

        model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        forearm_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "left_radius_link"
        )
        if forearm_id < 0:
            raise ValueError("MuJoCo left forearm body not found")
        result["mujoco"] = {
            "version": mujoco.__version__,
            "bodies": model.nbody,
            "joints": model.njnt,
            "actuators": model.nu,
            "forearm_body_id": forearm_id,
            "pose_source": "model default, not measured ground configuration",
            "note": "Compiled and ran FK only; position actuators are not passive mode",
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "private_sprout/v0.1.4",
    )
    parser.add_argument("--mujoco", action="store_true")
    args = parser.parse_args()
    print(json.dumps(inspect_model(args.model_dir, args.mujoco), indent=2))


if __name__ == "__main__":
    main()
