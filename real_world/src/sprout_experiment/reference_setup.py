"""Initialize template tracking from an operator-selected current camera image."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from prpl_tidybot.sprout.live_config import calibration_digest
from prpl_tidybot.visual_servo.forearm_template import ForearmPatchTracker, ceiling_view


def rectangle(value: list[int], width: int, height: int) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(type(v) is not int for v in value)
    ):
        raise ValueError("Select an image rectangle")
    x, y, w, h = value
    if min(w, h) < 8 or x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError(
            "Rectangle must fit one camera view and be at least 8 pixels wide/high"
        )
    return value


def search_region(patch: list[int], width: int, height: int) -> list[int]:
    x, y, w, h = patch
    # Search remains in one camera and outside the masked seam.
    left, top = max(0, x - 40), max(8, y - 40)
    right, bottom = min(width, x + w + 40), min(height - 8, y + h + 40)
    return [left, top, right - left, bottom - top]


def body_patch(
    image: np.ndarray, envelope: list[int], margin: float
) -> tuple[list[int], list[int]]:
    """Choose a textured, unambiguous patch inside the selected body envelope."""
    height, width = image.shape[:2]
    x, y, w, h = rectangle(envelope, width, height)
    y0, y1 = max(y, 12), min(y + h, height - 12)
    w0, h0 = min(w, 128), min(y1 - y0, 128)
    if min(w0, h0) < 16:
        raise ValueError(
            "Select a larger visible body region away from the camera seam"
        )
    candidates = []
    for left in np.linspace(x, x + w - w0, 4).astype(int):
        for top in np.linspace(y0, y1 - h0, 4).astype(int):
            patch = list(map(int, (left, top, w0, h0)))
            search = search_region(patch, width, height)
            try:
                tracker = ForearmPatchTracker(
                    image, tuple(patch), minimum_quality=0.8, minimum_margin=margin
                )
                match = tracker.detect(image, tuple(search))
            except ValueError:
                continue
            if match is not None:
                candidates.append(
                    (match.quality - match.runner_up_quality, patch, search)
                )
    if not candidates:
        raise ValueError(
            "Body region has no distinct tracking patch. Include visible Sprout detail."
        )
    _, patch, search = max(candidates, key=lambda item: item[0])
    return patch, search


def build_reference_config(
    config: dict, selection: dict, image: np.ndarray, calibrations: list[dict]
) -> dict:
    """Build fresh tracking regions without changing live-execution permissions."""
    result = copy.deepcopy(config)
    camera = selection.get("camera_index")
    if type(camera) is not int or camera not in (0, 1):
        raise ValueError("Select the camera containing the taped left forearm")
    top_height = selection.get("top_height_m")
    if (
        isinstance(top_height, bool)
        or not isinstance(top_height, (int, float))
        or not math.isfinite(top_height)
        or not 0 < top_height < 0.3
    ):
        raise ValueError("Enter the forearm floor-to-top height in centimetres (0–30)")
    view = ceiling_view(image, camera)
    height, width = view.shape[:2]
    patch = rectangle(selection.get("patch_xywh"), width, height)
    search = search_region(patch, width, height)
    tracker = ForearmPatchTracker(
        view,
        tuple(patch),
        minimum_quality=result["minimum_quality"],
        minimum_margin=result["minimum_margin"],
    )
    if tracker.detect(view, tuple(search)) is None:
        raise ValueError(
            "Tape patch is ambiguous or touches an edge. Include the tape and nearby forearm detail."
        )
    result.update(camera_index=camera, patch_xywh=patch, search_xywh=search)
    result["projection"] = copy.deepcopy(calibrations[camera]["projection"])
    result["projection"].update(
        patch_height_m=float(top_height),
        patch_to_grasp_xy_m=None,
        grasp_height_m=None,
        grasp_yaw_rad=None,
    )
    result["camera_calibration"] = copy.deepcopy(
        calibrations[camera]["camera_calibration"]
    )
    result["tape_height_source"] = (
        "operator-specified floor-to-top height at reference initialization"
    )
    result["body_patch_xywh"] = result["body_search_xywh"] = result[
        "body_projection"
    ] = None
    for key in ("last_box", "pad_geometry"):
        result.get("auto_box", {}).pop(key, None)
    body = result["body_obstacle"]
    body.pop("reviewed_box", None)
    envelopes = selection.get("body_envelopes_xywh")
    if not isinstance(envelopes, list) or len(envelopes) != 2:
        raise ValueError("Select the visible Sprout body envelope in each camera")
    body["enabled"], body["views"] = True, []
    for index, envelope in enumerate(envelopes):
        rectangle(envelope, width, height)
        if index == camera:
            x, y, w, h = envelope
            px, py, pw, ph = patch
            if not (x <= px and y <= py and px + pw <= x + w and py + ph <= y + h):
                raise ValueError("The body envelope must include the taped forearm")
        tile, search = body_patch(
            ceiling_view(image, index), envelope, body["minimum_margin"]
        )
        cal = calibrations[index]
        body["views"].append(
            {
                "camera_index": index,
                "serial": cal["camera_calibration"]["serial"],
                "patch_xywh": tile,
                "search_xywh": search,
                "envelope_xywh": envelope,
                "projection": {
                    key: cal["projection"][key]
                    for key in ("camera_matrix", "rvec", "tvec")
                },
            }
        )
    result["note"] = (
        "Operator-selected current scene. Reinitialize after pose/scale changes. Body boxes are not articulated pose estimation."
    )
    return result


def initialize_reference(
    selection_path: Path, directory: Path = Path(".local/sprout")
) -> None:
    from sprout_experiment.rehearsal import (
        detect,
        import_camera_projection,
        read_json,
        write_json,
    )

    selection = read_json(selection_path)
    capture = directory / "current.jpg"
    raw = capture.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if selection.get("image_sha256") != checksum:
        raise ValueError("Camera image changed during selection. Run init again.")
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode camera capture")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    metadata = read_json(capture.with_suffix(".json"))
    if metadata.get("image_sha256") != checksum or metadata.get("image_shape") != list(
        image.shape
    ):
        raise ValueError("Capture metadata does not match the selected image")
    cameras = metadata.get("cameras", [])
    if len(cameras) != 2:
        raise ValueError("Capture must include both cameras' calibration metadata")
    calibrations = []
    for index in (0, 1):
        calibration = {"camera_index": index, "auto_box": {"enabled": True}}
        import_camera_projection(calibration)
        cal = calibration["camera_calibration"]
        if image.shape != (2 * cal["image_height"], cal["image_width"], 3):
            raise ValueError("Capture size differs from camera calibration")
        if cameras[index].get("serial") != cal["serial"] or cameras[index].get(
            "calibration_sha256"
        ) != calibration_digest(calibration["projection"]):
            raise ValueError("Perception PC and workstation camera calibrations differ")
        calibrations.append(calibration)
    path = directory / "detection.json"
    original = path.read_text()
    config = build_reference_config(
        json.loads(original), selection, image, calibrations
    )
    # This image is a reference, not authorization to execute. Fresh captures and
    # state are still required in every run. Never re-label its acquisition time.
    reference = directory / f"reference-{checksum}.jpg"
    reference.write_bytes(raw)
    config["reference_image"] = str(reference)
    detect(config, reference, directory / "reference-check")
    # Commit calibration only after self-matching, forearm fit and body checks pass.
    if path.read_text() != original:
        raise ValueError(
            "Detection settings changed during init. Retry without another session."
        )
    (directory / f"detection-before-init-{time.time_ns()}.json").write_text(original)
    staged = directory / "detection.pending.json"
    write_json(staged, config)
    staged.replace(path)
    print(
        "Reference initialized from this capture. Run python3 ~/sprout.py for a fresh preview. No hardware motion.",
        flush=True,
    )
