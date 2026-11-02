"""Workstation camera -> MuJoCo preview -> explicit approval workflow.

No ABLE or private model copying. The live runtime uses reviewed calibration.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def project(uv: np.ndarray, calibration: dict[str, Any], height: float) -> np.ndarray:
    """Intersect an undistorted camera-local pixel ray with a known map Z plane."""
    k = np.asarray(calibration["camera_matrix"], dtype=float)
    rvec = np.asarray(calibration["rvec"], dtype=float)
    tvec = np.asarray(calibration["tvec"], dtype=float).reshape(3)
    if k.shape != (3, 3) or not np.all(
        np.isfinite([*k.flat, *rvec.flat, *tvec, height])
    ):
        raise ValueError("Invalid projection calibration")
    rotation, _ = cv2.Rodrigues(rvec)
    origin = -rotation.T @ tvec
    ray = rotation.T @ np.linalg.solve(k, np.r_[uv, 1])
    if abs(ray[2]) < 1e-8:
        raise ValueError("Camera ray parallel to grasp plane")
    distance = (height - origin[2]) / ray[2]
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("Invalid/behind-camera plane intersection")
    return origin + distance * ray


def detect(
    config: dict[str, Any], image_path: Path, output: Path, require_body: bool = False
) -> dict[str, Any]:
    from prpl_tidybot.visual_servo.forearm_template import (
        ForearmPatchTracker,
        ceiling_view,
    )

    if require_body:
        require_tracking_config(config, require_body=True)
    reference = cv2.imread(config["reference_image"])
    current = cv2.imread(str(image_path))
    if reference is None or current is None:
        raise ValueError("Cannot read reference/current camera image")
    reference_composite = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
    current_composite = cv2.cvtColor(current, cv2.COLOR_BGR2RGB)
    reference = ceiling_view(
        cv2.cvtColor(reference, cv2.COLOR_BGR2RGB), config["camera_index"]
    )
    current = ceiling_view(
        cv2.cvtColor(current, cv2.COLOR_BGR2RGB), config["camera_index"]
    )
    tracker = ForearmPatchTracker(
        reference,
        tuple(config["patch_xywh"]),
        minimum_quality=config["minimum_quality"],
        minimum_margin=config["minimum_margin"],
    )
    match = tracker.detect(current, tuple(config["search_xywh"]))
    if match is None:
        raise RuntimeError("Selected forearm patch lost or ambiguous")
    uv = np.array(match.center_uv)
    box = None
    if config.get("auto_box", {}).get("enabled"):
        from prpl_tidybot.sprout.pad_geometry import box_grasp_geometry
        from prpl_tidybot.visual_servo.forearm_box import box_in_map, fit_box

        p = config["projection"]
        box = box_in_map(
            fit_box(current, uv, config["patch_xywh"][2:], config["search_xywh"]), p
        )
        geometry = box_grasp_geometry(
            Path(config["auto_box"]["tidybot_model"]), p["patch_height_m"]
        )
        if box["width_m"] >= geometry["maximum_open_width_m"]:
            raise ValueError("Detected forearm box is wider than the open gripper")
        tape = project(uv, p, p["patch_height_m"])
        p["patch_to_grasp_xy_m"] = (
            np.asarray(box["center_xyz"][:2]) - tape[:2]
        ).tolist()
        p["grasp_yaw_rad"] = box["short_axis_yaw_rad"]
        p["grasp_height_m"] = geometry["grasp_height_m"]
        config["auto_box"]["last_box"] = box
        config["auto_box"]["pad_geometry"] = geometry
    result: dict[str, Any] = {
        "source": "recorded_camera",
        "image": str(image_path.resolve()),
        "camera_index": config["camera_index"],
        "pixel_uv": uv.tolist(),
        "quality": match.quality,
        "body_pose_source": "manual_model_alignment",
        "grasp": None,
    }
    if box is not None:
        result["forearm_box"] = box
        result["pad_geometry"] = geometry
        result["body_pose_source"] = "forearm_box_from_camera; whole-body model omitted"
    if config.get("body_obstacle", {}).get("enabled"):
        from prpl_tidybot.sprout.body_obstacle import detect_body, draw_body

        body = detect_body(
            reference_composite, current_composite, config["body_obstacle"]
        )
        result["body_obstacle"] = body
        config["body_obstacle"]["reviewed_box"] = body
        output.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(output / "body_obstacle.png"),
            cv2.cvtColor(draw_body(current_composite, body), cv2.COLOR_RGB2BGR),
        )
    from sprout_experiment.live_bundle import camera_evidence

    try:
        result["camera_evidence"] = camera_evidence(image_path, config)
        result["source"] = "live_camera"
    except (ValueError, KeyError, TypeError) as error:
        result["live_evidence_missing"] = str(error)
    calibration = config.get("projection")
    if calibration is not None:
        try:
            require_tracking_config(config)
        except ValueError as error:
            result["calibration_missing"] = str(error)
    if calibration is not None and all(
        calibration.get(key) is not None
        for key in ("camera_matrix", "rvec", "tvec", "patch_height_m")
    ):
        # Inputs are for this camera's UNDISTORTED, camera-local pixels.
        xyz = project(uv, calibration, float(calibration["patch_height_m"]))
        result["white_tape_xyz"] = xyz.tolist()
        result["tape_height_source"] = config.get("tape_height_source", "configured")
        if "calibration_missing" not in result:
            xyz[:2] += np.asarray(calibration["patch_to_grasp_xy_m"])
            result["grasp"] = {
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(calibration["grasp_height_m"]),
                "yaw": float(calibration["grasp_yaw_rad"]),
            }
    # Optional whole-body reference tracking gives a 2D displacement only. It
    # does not infer hidden joints, body depth, or a new 3D articulated pose.
    if require_body:
        body = ForearmPatchTracker(
            reference,
            tuple(config["body_patch_xywh"]),
            minimum_quality=config["minimum_quality"],
            minimum_margin=config["minimum_margin"],
        )
        found = body.detect(current, tuple(config["body_search_xywh"]))
        result["body_match"] = (
            None
            if found is None
            else {
                "pixel_uv": list(found.center_uv),
                "quality": found.quality,
                "estimate": "2D template localization only",
            }
        )
        if (
            found is not None
            and calibration is not None
            and config.get("body_projection") is not None
        ):
            body_cal = config["body_projection"]
            center = project(
                np.array(found.center_uv), calibration, body_cal["height_m"]
            )
            result["sprout_root_xy"] = (
                center[:2] + body_cal["center_to_root_xy_m"]
            ).tolist()
            result["body_pose_source"] = (
                "camera template root XY; manual orientation and joint pose"
            )
    if require_body and result.get("sprout_root_xy") is None:
        raise RuntimeError(
            "Sprout body patch lost or ambiguous; reselect/check the body ROI"
        )
    cv2.circle(
        current,
        tuple(uv.astype(int)),
        2 if box is not None else 16,
        (255, 255, 255) if box is not None else (255, 40, 30),
        2,
    )
    if box is not None:
        cv2.polylines(
            current, [np.asarray(box["corners_uv"], np.int32)], True, (30, 255, 90), 2
        )
        center = np.asarray(box["center_uv"])
        axis = np.array(
            [
                math.cos(box["short_axis_image_rad"]),
                math.sin(box["short_axis_image_rad"]),
            ]
        )
        cv2.arrowedLine(
            current,
            tuple((center - 20 * axis).astype(int)),
            tuple((center + 20 * axis).astype(int)),
            (30, 210, 255),
            2,
        )
        cv2.putText(
            current,
            "FOREARM BOX / closing axis",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (30, 255, 90),
            1,
        )
    output.mkdir(parents=True, exist_ok=True)
    if box is not None:
        x, y, w, h = config["search_xywh"]
        crop = cv2.resize(
            current[y : y + h, x : x + w],
            None,
            fx=5,
            fy=5,
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imwrite(
            str(output / "target_closeup.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        )
    cv2.imwrite(str(output / "detection.png"), cv2.cvtColor(current, cv2.COLOR_RGB2BGR))
    write_json(output / "observation.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def require_tracking_config(config: dict[str, Any], require_body: bool = False) -> None:
    """Require forearm projection; body tracking is an explicit opt-in."""
    required = ["projection"]
    if require_body:
        required += ["body_patch_xywh", "body_search_xywh", "body_projection"]
    missing = [key for key in required if config.get(key) is None]
    if config.get("projection") is not None:
        missing += [
            "projection." + key
            for key in (
                "camera_matrix",
                "rvec",
                "tvec",
                "patch_height_m",
                "patch_to_grasp_xy_m",
                "grasp_height_m",
                "grasp_yaw_rad",
            )
            if config["projection"].get(key) is None
        ]
    if missing:
        raise ValueError("Fill .local/sprout/detection.json: " + ", ".join(missing))


def import_camera_projection(config: dict[str, Any]) -> bool:
    """Import repo camera geometry, never invent tape/grasp measurements."""
    if config.get("projection") is not None or "camera_index" not in config:
        return False
    import os

    import prpl_tidybot
    import yaml
    from prpl_tidybot.marker_detector.extrinsics import load_extrinsics
    from prpl_tidybot.marker_detector.utils import get_camera_params

    repo = Path(prpl_tidybot.__file__).resolve().parents[2]
    lab = os.environ.get("PRPL_LAB", "prpl")
    lab_config = yaml.safe_load((repo / "conf" / "lab" / f"{lab}.yaml").read_text())
    index = config["camera_index"]
    if type(index) is not int or index not in (0, 1):
        raise ValueError("Expected ceiling camera index 0 or 1")
    serial = lab_config["camera_serials"][index]
    width, height, matrix, _ = get_camera_params(serial)
    rvec, tvec = load_extrinsics(serial)
    config["projection"] = {
        "camera_matrix": matrix.tolist(),
        "rvec": rvec.tolist(),
        "tvec": tvec.tolist(),
        "patch_height_m": 0.08 if config.get("auto_box", {}).get("enabled") else None,
        "patch_to_grasp_xy_m": None,
        "grasp_height_m": None,
        "grasp_yaw_rad": None,
    }
    config["camera_calibration"] = {
        "lab": lab,
        "serial": serial,
        "image_width": width,
        "image_height": height,
        "source": "local prpl-tidybot camera_params; verify perception PC uses the same calibration",
        "live_server_verified": False,
    }
    return True


def setup(check: bool = False) -> None:
    """Create worksheets once; preserve existing calibration on every rerun."""
    import shutil

    from prpl_tidybot.sprout.planner_check import check as check_planner

    check_planner(build_ikfast=True)
    directory = Path(".local/sprout")
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("detection", "scene", "live"):
        target = directory / f"{name}.json"
        if not target.exists():
            shutil.copyfile(
                Path(f"real_world/config/sprout_{name}.example.json"), target
            )
            print(f"Created {target}; fill calibration before preview", flush=True)
    from sprout_experiment.initial_settings import fill_initial_settings

    live_path = directory / "live.json"
    live = read_json(live_path)
    scene_path = directory / "scene.json"
    scene_config = read_json(scene_path)
    model_name = scene_config.get("tidybot_model")
    if "tidybot_model" in scene_config:
        from sprout_experiment.setup_paths import resolve_tidybot_model

        model_name = str(resolve_tidybot_model(model_name, Path.cwd()))
        if scene_config["tidybot_model"] != model_name:
            scene_config["tidybot_model"] = model_name
            write_json(scene_path, scene_config)
    changed = fill_initial_settings(live, Path(model_name) if model_name else None)
    if changed:
        write_json(live_path, live)
        print(
            f"Filled {len(changed)} initial control/model settings. "
            "Existing values preserved. Hardware verification is unchanged.",
            flush=True,
        )
    detection_path = directory / "detection.json"
    detection = read_json(detection_path)
    if detection.get("auto_box", {}).get("enabled") and not detection["auto_box"].get(
        "tidybot_model"
    ):
        detection["auto_box"]["tidybot_model"] = model_name
        write_json(detection_path, detection)
    if import_camera_projection(detection):
        write_json(detection_path, detection)
        print(
            "Imported PRPL camera geometry; tape/grasp measurements remain unset",
            flush=True,
        )
    if check:
        detection = read_json(directory / "detection.json")
        if (
            not detection.get("reference_image")
            or not Path(detection["reference_image"]).is_file()
            or detection.get("patch_xywh") is None
        ):
            raise ValueError(
                "First run: python3 ~/sprout.py init to select the taped forearm"
            )
        if detection.get("auto_box", {}).get("enabled"):
            p = detection.get("projection") or {}
            missing = [
                key
                for key in ("camera_matrix", "rvec", "tvec", "patch_height_m")
                if p.get(key) is None
            ]
            if missing:
                raise ValueError(
                    "Automatic box detection needs camera calibration and top height: "
                    + ", ".join(missing)
                )
        else:
            require_tracking_config(detection)
        config = read_json(directory / "scene.json")
        if config.get("synthetic_demo", False):
            raise ValueError("Default workflow requires a calibrated scene, not a demo")
        required = (
            ()
            if config["scene"].get("representation") == "forearm_box"
            else ("sprout_root_xyz_wxyz", "forearm_point_local_m")
        )
        missing = [key for key in required if config["scene"].get(key) is None]
        missing += [key for key, value in config["recipe"].items() if value is None]
        if missing:
            raise ValueError("Fill .local/sprout/scene.json: " + ", ".join(missing))
        print(
            "Calibration fields present; numerical/IK checks run during preview",
            flush=True,
        )


def preview(args: argparse.Namespace) -> None:
    from prpl_tidybot.sprout.rehearsal import Rehearsal
    from prpl_tidybot.sprout.review import digest
    from prpl_tidybot.sprout.sequence import GraspPose, Recipe

    config = read_json(args.config)
    observation = read_json(args.observation) if args.observation else None
    scene = config["scene"]
    if observation is not None:
        if observation.get("white_tape_xyz") is None:
            raise ValueError(
                "Run detect again: calibrated white-tape position is missing"
            )
        scene["body_obstacle"] = observation.get("body_obstacle")
        scene["white_tape_xyz"] = observation["white_tape_xyz"]
        scene["white_tape_source"] = "camera measurement at configured tape height"
        if scene.get("representation") == "forearm_box":
            if observation.get("forearm_box") is None:
                raise ValueError("Run automatic forearm box detection first")
            scene["forearm_box"] = observation["forearm_box"]
            scene["grasp"] = observation["grasp"]
            scene["body_pose_source"] = "forearm box only; no Sprout body model"
    robot_state_path = getattr(args, "robot_state", None)
    state = None
    if robot_state_path:
        from prpl_tidybot.sprout.read_robot_state import validate_snapshot

        if config.get("synthetic_demo", False):
            raise ValueError(
                "Use a calibrated scene, not the synthetic demo, with robot feedback"
            )
        state = read_json(robot_state_path)
        validate_snapshot(
            state, max_age_s=86400 if getattr(args, "recorded", False) else 120
        )
        scene["tidybot_base_xyyaw"] = state["base"]["xyyaw"]
        scene["tidybot_arm_joints_rad"] = state["arm"]["joints_rad"]
        scene["tidybot_state_source"] = state
        print(
            "Initializing simulated TidyBot from recorded arm feedback and map pose",
            flush=True,
        )
    if observation is not None and observation.get("sprout_root_xy") is not None:
        scene["sprout_root_xyz_wxyz"][:2] = observation["sprout_root_xy"]
        scene["body_pose_source"] = observation["body_pose_source"]
    sprout_model = (
        None
        if scene.get("representation") == "forearm_box"
        else Path(config["sprout_model"])
    )
    live_path = getattr(args, "live_config", None)
    if live_path and live_path.exists():
        configured = read_json(live_path).get("obstacles", [])
        scene["arm_obstacles"] = [
            box
            for box in configured
            if box.get("center") is not None and box.get("half_size") is not None
        ]
        scene["unconfigured_arm_obstacles"] = [
            box.get("name", "unnamed")
            for box in configured
            if box.get("center") is None or box.get("half_size") is None
        ]
    sim = Rehearsal(Path(config["tidybot_model"]), sprout_model, scene)
    if observation is None:
        if not config.get("synthetic_demo", False):
            raise ValueError("Supply a calibrated camera observation")
        goal = sim.data.site_xpos[sim.forearm].copy()
        grasp = GraspPose(float(goal[0]), float(goal[1]), float(goal[2]), yaw=0.0)
    else:
        if observation["grasp"] is None:
            raise ValueError(
                "Pixel detection succeeded, but camera projection/grasp-height calibration is missing"
            )
        grasp = GraspPose(**observation["grasp"])
    recipe = Recipe(**config["recipe"])
    base_error = None
    scene.pop("base_plan", None)
    if config.get("base_planning", {}).get("enabled"):
        from prpl_tidybot.sprout.base_planning import plan_base

        try:
            if scene.get("body_obstacle") is None:
                raise ValueError("Whole-body envelope required for base planning")
            obstacles = [
                scene["body_obstacle"],
                *config["base_planning"].get("obstacles", []),
            ]
            scene["base_plan"] = plan_base(
                sim,
                Path(config["tidybot_model"]),
                grasp,
                recipe,
                obstacles,
                config["base_planning"],
                navigation=(
                    read_json(args.live_config).get("navigation", {})
                    if getattr(args, "live_config", None) and args.live_config.exists()
                    else None
                ),
            )
            print("SIM BASE: " + json.dumps(scene["base_plan"]), flush=True)
        except ValueError as error:
            base_error = str(error)
            print("SIM BASE BLOCKED: " + base_error, flush=True)
    report = sim.render_plan(
        grasp,
        recipe,
        args.output,
        fps=args.fps,
        steps=args.steps,
        blocked_reason=("base planning: " + base_error) if base_error else None,
    )
    if base_error:
        report["completed"], report["failure"] = False, "base planning: " + base_error
    from prpl_tidybot.sprout.urdf_collision import urdf_digest

    report["planner_backend"] = "prpl_urdf"
    report["collision_coverage"] = {
        "robot": "PRPL TidyBotKinova URDF collision meshes/primitives",
        "body_box": "whole robot during base transit, base only during manipulation",
        "arm_obstacles": scene.get("arm_obstacles", []),
        "unconfigured_arm_obstacles": scene.get("unconfigured_arm_obstacles", []),
        "contact_dynamics_validated": False,
    }
    report["collision_model_sha256"] = urdf_digest()
    report["base_plan"] = scene.get("base_plan")
    trajectory = read_json(args.output / "trajectory.json")
    pregrasp_frames = [frame for frame in trajectory if frame["phase"] == "pregrasp"]
    if pregrasp_frames:
        report["pregrasp_path_rad"] = [
            frame["arm_joints_rad"] for frame in pregrasp_frames
        ]
        report["pregrasp_joints_rad"] = pregrasp_frames[-1]["arm_joints_rad"]
    write_json(args.output / "scene.json", config)
    if observation is not None:
        write_json(args.output / "observation.json", observation)
    report.update(
        {
            "source": (
                "synthetic_demo" if observation is None else observation["source"]
            ),
            "execution_ready": False,
            "created_at_unix": time.time(),
            "observed": {
                "grasp_xyz": [grasp.x, grasp.y, grasp.z],
                "base_xyyaw": scene["tidybot_base_xyyaw"],
                "arm_joints_rad": scene["tidybot_arm_joints_rad"],
            },
            "recipe": config["recipe"],
            "blockers": [
                "Calibrated live camera/body pose and joint snapshot required",
                "Commissioned compliant scene adapter, gripper limits, and physical stop required",
                "Kinematic preview does not validate contact forces or collision clearance",
            ],
            "artifact_sha256": {
                name: digest(args.output / name)
                for name in ("preview.mp4", "trajectory.json", "scene.json")
            },
        }
    )
    from sprout_experiment.live_bundle import prepare_live_bundle

    prepare_live_bundle(
        report,
        config,
        observation,
        state,
        getattr(args, "live_config", None),
        args.output,
    )
    if getattr(args, "recorded", False):
        report["source"] = "recorded_camera"
        report["execution_ready"] = False
        report["blockers"].append(
            "Recorded-state diagnostic preview cannot authorize hardware"
        )
    write_json(args.output / "review.json", report)
    print(
        f"Saved {args.output / 'preview.mp4'}\nSaved {args.output / 'review.json'}",
        flush=True,
    )


def demo(args: argparse.Namespace) -> None:
    from prpl_tidybot.sprout.rehearsal import Rehearsal

    scene = {
        "body_pose_source": "synthetic_supine_pose_NOT_camera_measurement",
        "sprout_root_xyz_wxyz": [
            1.0,
            0.0,
            0.12,
            math.sqrt(0.5),
            0.0,
            -math.sqrt(0.5),
            0.0,
        ],
        "sprout_joints_rad": {
            "left_shoulder_roll_joint": -math.pi / 4,
            "left_elbow_joint": -1.0,
        },
        "forearm_point_local_m": [0.0, 0.05, 0.027],
        "tidybot_base_xyyaw": [0, 0, 0],
        "tidybot_arm_joints_rad": [0, -0.35, math.pi, -2.55, 0, -0.87, math.pi / 2],
    }
    sim = Rehearsal(args.tidybot_model, args.sprout_model, scene)
    goal = sim.data.site_xpos[sim.forearm].copy()
    scene["white_tape_xyz"] = (goal + np.array([0.02, 0.0, 0.01])).tolist()
    scene["white_tape_source"] = "synthetic demo marker, not a camera measurement"
    scene["tidybot_base_xyyaw"] = [float(goal[0] - 0.6), float(goal[1]), 0.0]
    config = {
        "synthetic_demo": True,
        "tidybot_model": str(args.tidybot_model.resolve()),
        "sprout_model": str(args.sprout_model.resolve()),
        "scene": scene,
        "recipe": {
            "pregrasp_height_m": 0.10,
            "retreat_height_m": 0.10,
            "lift_height_m": args.lift_height,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "demo-scene.json", config)
    args.config, args.observation = args.output / "demo-scene.json", None
    preview(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "preview", "detect"):
        sub = commands.add_parser(name)
        sub.add_argument("--output", type=Path, required=True)
        if name != "demo":
            sub.add_argument("--config", type=Path, required=True)
        if name == "detect":
            sub.add_argument("--image", type=Path, required=True)
            sub.add_argument("--require-body", action="store_true")
        else:
            sub.add_argument("--fps", type=int, default=12)
            sub.add_argument("--steps", type=int, default=20)
        if name == "preview":
            sub.add_argument(
                "--recorded",
                action="store_true",
                help="Diagnostic only: allow a saved state up to one day old; disables hardware authorization",
            )
            sub.add_argument("--live-config", type=Path)
            sub.add_argument("--observation", type=Path)
            sub.add_argument(
                "--robot-state",
                type=Path,
                help="NUC telemetry snapshot, at most 120 seconds old",
            )
        if name == "demo":
            sub.add_argument("--lift-height", type=float, default=0.15)
            sub.add_argument("--tidybot-model", type=Path, required=True)
            sub.add_argument("--sprout-model", type=Path, required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--host", required=True)
    capture.add_argument("--port", type=int, default=6003)
    capture.add_argument("--output", type=Path, required=True)
    initialize = commands.add_parser("initialize")
    initialize.add_argument("--selection", type=Path, required=True)
    commands.add_parser("setup")
    commands.add_parser("check")
    execute = commands.add_parser("execute-reviewed")
    execute.add_argument("--review", type=Path, required=True)
    execute.add_argument(
        "--runtime",
        help="Commissioned module:factory with observe() and execute(review)",
    )
    args = parser.parse_args()
    if args.command in ("setup", "check"):
        setup(check=args.command == "check")
    elif args.command == "initialize":
        from sprout_experiment.reference_setup import initialize_reference

        initialize_reference(args.selection)
    elif args.command == "capture":
        from prpl_tidybot.marker_detector.ceiling_image_client import CeilingImageClient

        client = CeilingImageClient(host=args.host, port=args.port, poll_timeout_s=5)
        try:
            frame = client.get_image()
            if frame is None:
                raise RuntimeError("No ceiling frame received")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.output), frame):
                raise RuntimeError("Could not save ceiling frame")
            write_json(
                args.output.with_suffix(".json"),
                {
                    "received_at_unix": time.time(),
                    "cameras": getattr(client, "metadata", {}).get("cameras", []),
                    "image_shape": list(frame.shape),
                    "image_sha256": __import__("hashlib")
                    .sha256(args.output.read_bytes())
                    .hexdigest(),
                },
            )
            print(f"Saved ceiling snapshot: {args.output}")
        finally:
            client.close()
    elif args.command == "detect":
        config = read_json(args.config)
        detect(config, args.image, args.output, args.require_body)
        if config.get("auto_box", {}).get("enabled"):
            write_json(args.config, config)
            print(
                f"Automatically filled grasp offset/yaw/height in {args.config}",
                flush=True,
            )
    elif args.command == "demo":
        demo(args)
    elif args.command == "preview":
        preview(args)
    else:
        from prpl_tidybot.sprout.review import approve_and_execute

        report = read_json(args.review)
        if not report.get("execution_ready") or not args.runtime:
            parser.error(
                "Hardware execution remains disabled: commission the live runtime and fresh-scene checks first; offline Enter cannot enable it"
            )
        module, factory = args.runtime.split(":", 1)
        # Factory must be read-only; hardware activation belongs to execute().
        runtime = getattr(importlib.import_module(module), factory)(args.review)
        try:
            approve_and_execute(args.review, runtime.observe, runtime.execute)
        finally:
            runtime.close()


if __name__ == "__main__":
    main()
