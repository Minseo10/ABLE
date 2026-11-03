"""Project map coordinates and the planned route onto measured ceiling images."""

from pathlib import Path

import cv2
import numpy as np
import prpl_tidybot
import yaml
from prpl_tidybot.marker_detector.extrinsics import load_extrinsics
from prpl_tidybot.marker_detector.utils import get_camera_params


def draw_map_overlay(image: Path, detection: dict, report: dict, output: Path) -> None:
    """Diagnostic visualization only. Never change measured map coordinates."""
    frame = cv2.imread(str(image))
    if frame is None or frame.shape[0] % 2:
        raise ValueError("A two-camera ceiling image is required")
    height, width = frame.shape[0] // 2, frame.shape[1]
    lab = detection["camera_calibration"].get("lab", "prpl")
    lab_config = (
        Path(prpl_tidybot.__file__).resolve().parents[2] / "conf/lab" / (lab + ".yaml")
    )
    serials = list(yaml.safe_load(lab_config.read_text())["camera_serials"])
    serials[detection["camera_index"]] = detection["camera_calibration"]["serial"]
    for view in detection.get("body_obstacle", {}).get("views", []):
        serials[view["camera_index"]] = view["serial"]
    for index, serial in enumerate(serials):
        w, h, camera, _ = get_camera_params(serial)
        if (w, h) != (width, height):
            raise ValueError("Calibration/image dimensions differ")
        rvec, tvec = load_extrinsics(serial)
        rotation, _ = cv2.Rodrigues(rvec)
        view = frame[index * height : (index + 1) * height]

        def project(points):
            points = np.asarray(points, dtype=float).reshape(-1, 3)
            if np.any((points @ rotation.T + tvec)[:, 2] <= 0.05):
                return None
            uv, _ = cv2.projectPoints(points, rvec, tvec, camera, np.zeros(5))
            return np.clip(uv.reshape(-1, 2), -100000, 100000).astype(np.int32)

        def line(points, color, closed=False, thickness=1):
            uv = project(points)
            if uv is not None:
                cv2.polylines(view, [uv], closed, color, thickness)

        for value in np.arange(-2, 2.01, 0.5):
            line([[value, -2, 0], [value, 2, 0]], (155, 155, 155))
            line([[-2, value, 0], [2, value, 0]], (155, 155, 155))
        path = (
            report.get("base_plan", {}).get("path_xyyaw", [])
            if report.get("base_plan")
            else []
        )
        if path:
            line([[x, y, 0] for x, y, _ in path], (255, 230, 0), thickness=3)
        markers = [("map (0,0)", [0, 0, 0], (255, 0, 255))]
        observed = report.get("observed", {})
        if observed.get("grasp_xyz") is not None:
            markers.append(("grasp", observed["grasp_xyz"], (0, 255, 0)))
        if observed.get("base_xyyaw") is not None:
            markers.append(
                (
                    "robot top center",
                    [*observed["base_xyyaw"][:2], 0.378],
                    (0, 255, 255),
                )
            )
        for label, point, color in markers:
            uv = project([point])
            if uv is not None and 0 <= uv[0, 0] < width and 0 <= uv[0, 1] < height:
                position = tuple(uv[0])
                cv2.circle(view, position, 7, color, 2)
                cv2.putText(
                    view,
                    label,
                    (position[0] + 10, position[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
        cv2.rectangle(view, (0, 0), (width, 58), (25, 30, 40), -1)
        cv2.putText(
            view,
            "Camera/map overlay | gray grid 0.5m | target and planned base path",
            (12, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            view,
            "No virtual cabinets or walls. Check the real route is clear.",
            (12, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frame):
        raise OSError("Could not save map overlay")
