"""Bind a completed camera rehearsal to concrete live-runtime inputs."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
from prpl_tidybot.sprout.live_config import (
    calibration_digest,
    collision_boxes,
    model_digest,
    validate_config,
)
from prpl_tidybot.sprout.live_kinematics import LiveKinematics
from prpl_tidybot.sprout.review import INITIAL_OBSERVATION_MAX_AGE_S, digest


def camera_evidence(image: Path, detection: dict) -> dict:
    """A saved JPEG is live evidence only when paired with its acquired frame."""
    sidecar = image.with_suffix(".json")
    if not sidecar.exists():
        raise ValueError("Capture again with timestamped perception servers")
    metadata = json.loads(sidecar.read_text())
    if metadata.get("image_sha256") != digest(image):
        raise ValueError("Camera snapshot differs from its timestamp metadata")
    index = detection["camera_index"]
    cameras = metadata.get("cameras", [])
    if len(cameras) != 2:
        raise ValueError("Update/restart perception servers: camera timestamps missing")
    calibration = detection["camera_calibration"]
    if metadata.get("image_shape") != [
        2 * calibration["image_height"],
        calibration["image_width"],
        3,
    ]:
        raise ValueError("Capture dimensions differ from calibrated two-view composite")
    camera = cameras[index]
    stamp = camera.get("captured_at_unix")
    if stamp is None or not np.isfinite(stamp) or not 0 <= time.time() - stamp <= 120:
        raise ValueError(
            "Camera acquisition is stale/missing; capture again and check clocks"
        )
    if camera.get("serial") != detection.get("camera_calibration", {}).get("serial"):
        raise ValueError("Camera serial differs from detection calibration")
    if camera.get("calibration_sha256") != calibration_digest(detection["projection"]):
        raise ValueError("Perception-PC camera calibration differs from workstation")
    if detection.get("body_obstacle", {}).get("enabled"):
        for view in detection["body_obstacle"]["views"]:
            item = cameras[view["camera_index"]]
            stamp = item.get("captured_at_unix")
            if (
                stamp is None
                or not np.isfinite(stamp)
                or not 0 <= time.time() - stamp <= 120
            ):
                raise ValueError("Body camera acquisition stale/missing")
            if item.get("serial") != view["serial"] or item.get(
                "calibration_sha256"
            ) != calibration_digest(view["projection"]):
                raise ValueError("Body camera calibration differs from workstation")
    return camera


def prepare_live_bundle(
    report: dict,
    scene: dict,
    observation: dict | None,
    state: dict | None,
    live_path: Path | None,
    output: Path,
) -> None:
    """Readiness follows actual validation; never require manually editing a flag."""
    report["execution_ready"] = False
    report["blockers"] = []
    try:
        if live_path is None or not live_path.exists():
            raise ValueError("Fill .local/sprout/live.json commissioning inputs")
        live = json.loads(live_path.read_text())
        detection_path = live_path.parent / "detection.json"
        live["detection"] = json.loads(detection_path.read_text())
        reference = Path(live["detection"]["reference_image"])
        shutil.copyfile(reference, output / "reference.jpg")
        # Only the explicitly copied reference is used on the NUC.
        live["detection"]["reference_image"] = "reference.jpg"
        model = Path(scene["tidybot_model"])
        live["model_sha256"] = model_digest(model)
        if report.get("collision_model_sha256") is not None:
            from prpl_tidybot.sprout.urdf_collision import urdf_digest

            if report["collision_model_sha256"] != urdf_digest():
                raise ValueError("Preview/runtime URDF models differ")
            live["collision_model_sha256"] = report["collision_model_sha256"]
        live["base_plan"] = report.get("base_plan")
        arm_boxes, held_box = collision_boxes(live)
        # Normalize legacy null placeholders in the reviewed bundle only.
        live["obstacles"], live["held_forearm"] = arm_boxes, held_box
        body_config = live["detection"].get("body_obstacle", {})
        if observation is not None and observation.get("body_obstacle") is not None:
            body_config["reviewed_box"] = observation["body_obstacle"]
            live["detection"]["body_obstacle"] = body_config
        report.setdefault("collision_coverage", {}).update(
            {
                "arm_obstacles": arm_boxes,
                "held_forearm": held_box,
                "whole_body_box_scope": "base keep-out only",
            }
        )
        report["execution_limitations"] = []
        if not arm_boxes:
            report["execution_limitations"].append(
                "No manual arm-obstacle boxes: arm-to-Sprout body collisions "
                "are not checked. Whole-body camera box protects the base."
            )
        if held_box is None:
            report["execution_limitations"].append(
                "No held-forearm box: attached-limb collision geometry is not checked."
            )
        (output / "live.json").write_text(json.dumps(live, indent=2) + "\n")
        if not report.get("completed"):
            raise ValueError(
                "Simulation did not complete: "
                + str(report.get("failure", "unknown failure"))
            )
        validate_config(live)
        if observation is None or observation.get("source") != "live_camera":
            raise ValueError("Timestamped calibrated live-camera observation required")
        if not state or not state["base"].get("acquisition_timestamp_available"):
            raise ValueError(
                "Update/restart marker server and capture TidyBot state again"
            )
        stamp = state["base"].get("captured_at_unix")
        if (
            stamp is None
            or not np.isfinite(stamp)
            or not 0 <= time.time() - stamp <= 120
        ):
            raise ValueError("Base camera acquisition is stale; recapture")
        base_plan = report.get("base_plan")
        final_base = report["observed"]["base_xyyaw"]
        if base_plan and base_plan.get("transit_geometry"):
            from prpl_tidybot.sprout.transit_geometry import validate_model_geometry

            validate_model_geometry(
                base_plan["transit_geometry"],
                model,
                report["observed"]["arm_joints_rad"],
                live.get("navigation", {}),
            )
        if base_plan and base_plan["required"]:
            from prpl_tidybot.sprout.base_motion import validate_navigation
            from prpl_tidybot.sprout.base_planning import validate_base_plan

            validate_base_plan(base_plan)
            validate_navigation(live.get("navigation", {}))
            final_base = base_plan["path_xyyaw"][-1]
        kinematics = LiveKinematics(model, final_base, live)
        trajectory = json.loads((output / "trajectory.json").read_text())
        kinematics.validate_trajectory(trajectory, report["observed"]["arm_joints_rad"])
        for pose in report["plan"].values():
            from prpl_tidybot.sprout.approach import TopDownGeometry
            from prpl_tidybot.sprout.sequence import GraspPose

            TopDownGeometry(**live["geometry"]).validate_pose(GraspPose(**pose))
        # Initial stationary-scene approval allows processing/session overhead.
        # Per-frame live motion guards retain the configured camera age bound.
        report["maximum_observation_age_s"] = INITIAL_OBSERVATION_MAX_AGE_S
        report["revalidation_tolerances"] = live["revalidation_tolerances"]
        report["execution_ready"] = True
    except (ValueError, RuntimeError, KeyError, TypeError, OSError) as error:
        report["blockers"].append(str(error))
    names = (
        "preview.mp4",
        "trajectory.json",
        "scene.json",
        "live.json",
        "reference.jpg",
    )
    report["artifact_sha256"] = {
        name: digest(output / name) for name in names if (output / name).exists()
    }
    for limitation in report.get("execution_limitations", []):
        print("Preview coverage: " + limitation, flush=True)
    print(
        "Real execution: "
        + (
            "ready for terminal approval and fresh-state recheck"
            if report["execution_ready"]
            else "blocked: " + "; ".join(report["blockers"])
        ),
        flush=True,
    )
