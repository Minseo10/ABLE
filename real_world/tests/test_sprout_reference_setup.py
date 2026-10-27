"""Fresh-account initialization, reference identity, and local selection transport."""

import copy
import http.client
import importlib.util
import json
import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import numpy as np
import pytest
from sprout_experiment import reference_setup, setup_paths


@pytest.fixture
def selection_config():
    return {
        "minimum_quality": 0.8,
        "minimum_margin": 0.1,
        "auto_box": {"enabled": True, "last_box": {"old": True}},
        "body_obstacle": {
            "enabled": True,
            "minimum_margin": 0.04,
            "maximum_height_m": 0.35,
            "margin_m": 0.06,
            "reviewed_box": {"old": True},
        },
    }


@pytest.fixture
def reference_input():
    image = np.random.default_rng(11).integers(0, 256, (320, 180, 3), dtype=np.uint8)
    calibration = [
        {
            "camera_calibration": {"serial": f"camera-{i}"},
            "projection": {
                "camera_matrix": np.eye(3).tolist(),
                "rvec": [0, 0, 0],
                "tvec": [0, 0, 1],
            },
        }
        for i in range(2)
    ]
    selection = {
        "camera_index": 0,
        "patch_xywh": [30, 40, 20, 20],
        "body_envelopes_xywh": [[10, 12, 100, 100], [15, 20, 110, 100]],
        "top_height_m": 0.08,
    }
    return selection, image, calibration


def test_reference_regions_are_built_from_this_capture(
    selection_config, reference_input
):
    selection, image, calibration = reference_input
    original = copy.deepcopy(selection_config)
    result = reference_setup.build_reference_config(
        selection_config, selection, image, calibration
    )
    assert selection_config == original
    assert result["patch_xywh"] == selection["patch_xywh"]
    assert result["projection"]["patch_height_m"] == 0.08
    assert result["projection"]["grasp_yaw_rad"] is None
    assert "last_box" not in result["auto_box"]
    assert "reviewed_box" not in result["body_obstacle"]
    assert [v["serial"] for v in result["body_obstacle"]["views"]] == [
        "camera-0",
        "camera-1",
    ]
    assert "execution_enabled" not in result and "verified" not in result


@pytest.mark.parametrize(
    "field,value",
    [
        ("camera_index", True),
        ("top_height_m", float("nan")),
        ("patch_xywh", [0, 0, 8, 8]),
        ("body_envelopes_xywh", [[90, 90, 30, 30], [15, 20, 110, 100]]),
    ],
)
def test_invalid_selections_do_not_modify_calibration(
    selection_config, reference_input, field, value
):
    selection, image, calibration = reference_input
    selection[field] = value
    original = copy.deepcopy(selection_config)
    with pytest.raises(ValueError):
        reference_setup.build_reference_config(
            selection_config, selection, image, calibration
        )
    assert selection_config == original


def test_changed_capture_rejected_before_any_calibration_write(tmp_path):
    (tmp_path / "current.jpg").write_bytes(b"new image")
    (tmp_path / "detection.json").write_text('{"existing": true}')
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"image_sha256": "old image hash"}))
    with pytest.raises(ValueError, match="changed during selection"):
        reference_setup.initialize_reference(selection, tmp_path)
    assert (tmp_path / "detection.json").read_text() == '{"existing": true}'


def test_model_discovery_uses_installed_package_and_rejects_bad_explicit_path(
    tmp_path, monkeypatch
):
    model = tmp_path / "kinder/envs/dynamic3d/models/stanford_tidybot/tidybot.xml"
    model.parent.mkdir(parents=True)
    model.touch()
    monkeypatch.delenv("SPROUT_TIDYBOT_MODEL", raising=False)
    monkeypatch.setattr(
        setup_paths.importlib.util,
        "find_spec",
        lambda _: SimpleNamespace(
            submodule_search_locations=[str(tmp_path / "kinder")]
        ),
    )
    assert setup_paths.resolve_tidybot_model(None, tmp_path) == model
    with pytest.raises(ValueError, match="not found"):
        setup_paths.resolve_tidybot_model(str(tmp_path / "old-account.xml"), tmp_path)


def test_shell_python_selection_is_account_local(tmp_path):
    selector = Path(__file__).parents[1] / "scripts/sprout_python.sh"
    repo = tmp_path / "experiment"
    repo.mkdir()
    sibling = tmp_path / "prpl-tidybot/.venv/bin/python"
    sibling.parent.mkdir(parents=True)
    sibling.touch(mode=0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SPROUT_")}

    def selected():
        return subprocess.check_output(
            [
                "bash",
                "-c",
                'repo="$1"; source "$2"; echo "$SPROUT_PYTHON"',
                "bash",
                str(repo),
                str(selector),
            ],
            env=env,
            text=True,
        ).strip()

    assert Path(selected()).resolve() == sibling.resolve()
    own = repo / ".venv/bin/python"
    own.parent.mkdir(parents=True)
    own.touch(mode=0o755)
    assert selected() == str(own)
    env["SPROUT_PYTHON"] = "/explicit/python"
    assert selected() == "/explicit/python"
    del env["SPROUT_PYTHON"]
    own.unlink()
    sibling.unlink()
    assert selected() == "python3"


def test_selector_only_accepts_its_own_browser_origin():
    path = Path(__file__).parents[1] / "scripts/sprout_select_reference.py"
    spec = importlib.util.spec_from_file_location("selector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    server, url, state = module.selection_server(b"test-jpeg")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    parts = urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=2)
    try:
        conn.request("GET", parts.path)
        response = conn.getresponse()
        assert response.status == 200 and b"Select" in response.read()
        payload = json.dumps({"camera_index": 0})
        for origin, status in (
            ("http://unrelated.invalid", 403),
            (f"http://{parts.netloc}", 200),
        ):
            conn.request(
                "POST",
                parts.path + "selection",
                body=payload,
                headers={"Content-Type": "application/json", "Origin": origin},
            )
            response = conn.getresponse()
            assert response.status == status
            response.read()
            if status == 403:
                assert not state
        assert state["selection"] == {"camera_index": 0}
    finally:
        conn.close()
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
