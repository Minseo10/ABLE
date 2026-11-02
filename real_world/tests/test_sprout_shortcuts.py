"""Default workflow and calibration guards; no network or robot connections."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sprout_experiment import rehearsal


@pytest.fixture
def laptop():
    path = Path(__file__).parents[1] / "scripts/sprout_laptop.py"
    spec = importlib.util.spec_from_file_location("sprout_laptop", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_arguments_selects_full_workflow(laptop, monkeypatch):
    called = []
    monkeypatch.setattr(sys, "argv", ["sprout.py"])
    monkeypatch.setattr(laptop, "workflow", lambda args: called.append(args.action))
    laptop.main()
    assert called == ["run"]


@pytest.mark.parametrize("override", [False, True])
def test_laptop_connection_settings_from_environment(laptop, monkeypatch, override):
    settings = {
        "workstation": "student@workstation",
        "repo": "/home/student/experiment",
        "nuc": "robot@nuc",
        "perception": "camera-host",
        "perception_ssh": "student@camera-host",
    }
    for name, value in settings.items():
        monkeypatch.setenv("SPROUT_" + name.upper(), value)
    argv = ["sprout.py"]
    expected = settings.copy()
    if override:
        for name in settings:
            expected[name] = "explicit-" + name
            argv.extend(["--" + name.replace("_", "-"), expected[name]])
    monkeypatch.setattr(sys, "argv", argv)
    called = []
    monkeypatch.setattr(laptop, "workflow", called.append)
    laptop.main()
    assert len(called) == 1
    assert {name: getattr(called[0], name) for name in settings} == expected


@pytest.mark.parametrize("failure", [None, "check", "capture", "state", "preview"])
def test_workflow_order_and_fail_closed(laptop, monkeypatch, failure):
    called = []

    def step(name):
        called.append(name)
        if name == failure:
            raise subprocess.CalledProcessError(1, name)

    monkeypatch.setattr(laptop, "remote", lambda args, action, **kwargs: step(action))
    monkeypatch.setattr(laptop, "state", lambda args: step("state"))
    monkeypatch.setattr(laptop, "video", lambda args: step("video"))
    args = SimpleNamespace(action="run")
    order = ["check", "capture", "state", "preview", "video"]
    if failure:
        with pytest.raises(subprocess.CalledProcessError):
            laptop.workflow(args)
        assert called == order[: order.index(failure) + 1]
    else:
        laptop.workflow(args)
        assert called == order


def test_setup_preserves_calibration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    examples = tmp_path / "real_world/config"
    examples.mkdir(parents=True)
    for name in ("detection", "scene", "live"):
        (examples / f"sprout_{name}.example.json").write_text("{}")
    rehearsal.setup()
    target = tmp_path / ".local/sprout/detection.json"
    target.write_text('{"calibrated": true}')
    rehearsal.setup()
    assert json.loads(target.read_text()) == {"calibrated": True}


def test_required_body_fails_before_image_read(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="body_patch_xywh"):
        rehearsal.detect({}, tmp_path / "missing.jpg", tmp_path, require_body=True)


@pytest.mark.parametrize("require_body", [False, True])
def test_body_loss_only_blocks_when_opted_in(monkeypatch, tmp_path, require_body):
    from prpl_tidybot.visual_servo import forearm_template

    class Tracker:
        def __init__(self, reference, patch, **kwargs):
            self.body = patch[0] == 20

        def detect(self, current, search):
            return None if self.body else SimpleNamespace(center_uv=(10, 10), quality=1)

    monkeypatch.setattr(forearm_template, "ForearmPatchTracker", Tracker)
    monkeypatch.setattr(forearm_template, "ceiling_view", lambda image, index: image)
    monkeypatch.setattr(
        rehearsal.cv2, "imread", lambda path: np.zeros((50, 50, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(rehearsal, "project", lambda *args: np.array([0.1, 0.2, 0.3]))
    config = {
        "reference_image": "unused",
        "camera_index": 0,
        "patch_xywh": [0, 0, 10, 10],
        "search_xywh": [0, 0, 50, 50],
        "body_patch_xywh": [20, 0, 10, 10],
        "body_search_xywh": [0, 0, 50, 50],
        "minimum_quality": 0.8,
        "minimum_margin": 0.1,
        "body_projection": {"height_m": 0.1, "center_to_root_xy_m": [0, 0]},
        "projection": {
            "camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "rvec": [0, 0, 0],
            "tvec": [0, 0, 1],
            "patch_height_m": 0.1,
            "patch_to_grasp_xy_m": [0, 0],
            "grasp_height_m": 0.1,
            "grasp_yaw_rad": 0,
        },
    }
    if require_body:
        with pytest.raises(RuntimeError, match="body patch lost"):
            rehearsal.detect(
                config, tmp_path / "image.jpg", tmp_path, require_body=True
            )
        assert not (tmp_path / "observation.json").exists()
    else:
        result = rehearsal.detect(config, tmp_path / "image.jpg", tmp_path)
        assert result["grasp"] is not None
        assert result["white_tape_xyz"] == [0.1, 0.2, 0.3]
        assert "body_match" not in result
        assert (tmp_path / "observation.json").exists()


def test_default_config_requires_forearm_projection_only():
    rehearsal.require_tracking_config(
        {
            "projection": {
                key: "present"
                for key in (
                    "camera_matrix",
                    "rvec",
                    "tvec",
                    "patch_height_m",
                    "patch_to_grasp_xy_m",
                    "grasp_height_m",
                    "grasp_yaw_rad",
                )
            }
        }
    )
    with pytest.raises(ValueError, match="projection"):
        rehearsal.require_tracking_config({})


@pytest.mark.parametrize("answer,accepted", [("", True), ("yes", True), ("no", False)])
def test_laptop_approval(laptop, monkeypatch, answer, accepted):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    assert laptop.approval(lambda _: answer) == accepted


def test_piped_approval_rejected(laptop, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(ValueError, match="interactive"):
        laptop.approval(lambda _: "yes")


def test_partial_projection_names_missing_measurements():
    with pytest.raises(ValueError, match="projection.patch_height_m"):
        rehearsal.require_tracking_config(
            {"projection": {"camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}}
        )


def test_uncommissioned_review_never_prompts(laptop, tmp_path):
    (tmp_path / "review.json").write_text('{"execution_ready": false}')
    with pytest.raises(ValueError, match="unavailable"):
        laptop.validate_execution_bundle(tmp_path, tmp_path / "nonexistent-video")


def test_review_bound_to_viewed_video(laptop, tmp_path):
    artifacts = (
        "preview.mp4",
        "trajectory.json",
        "scene.json",
        "live.json",
        "reference.jpg",
    )
    hashes = {}
    for name in artifacts:
        (tmp_path / name).write_text("test fixture")
        hashes[name] = hashlib.sha256(b"test fixture").hexdigest()
    report = {
        "execution_ready": True,
        "completed": True,
        "source": "live_camera",
        "artifact_sha256": hashes,
    }
    (tmp_path / "review.json").write_text(json.dumps(report))
    laptop.validate_execution_bundle(tmp_path, tmp_path / "preview.mp4")
    (tmp_path / "preview.mp4").write_text("different video")
    with pytest.raises(ValueError, match="differs"):
        laptop.validate_execution_bundle(tmp_path, tmp_path / "preview.mp4")


def test_execution_requested_only_after_video(laptop, monkeypatch):
    events = []
    monkeypatch.setattr(laptop, "remote", lambda args, name, **kw: events.append(name))
    monkeypatch.setattr(laptop, "state", lambda args: events.append("state"))
    monkeypatch.setattr(laptop, "video", lambda args: events.append("video"))
    monkeypatch.setattr(
        laptop, "execute_reviewed", lambda args: events.append("approval_gate")
    )
    laptop.workflow(SimpleNamespace(action="run", execute=True))
    assert events == ["check", "capture", "state", "preview", "video", "approval_gate"]


def test_laptop_heartbeat_transports_both_operator_confirmations(laptop, monkeypatch):
    """Real subprocess IO/reader/heartbeat threads; simulated NUC only."""
    script = """import json,sys
approved=False
while not approved:
    msg=json.loads(sys.stdin.readline())
    approved=msg.get('kind')=='approve' and msg.get('review_sha256')=='test'
for phase in ('grasp','support'):
    print(json.dumps({'event':'operator_confirmation','phase':phase,'token':phase+'-nonce'}),flush=True)
    while True:
        msg=json.loads(sys.stdin.readline())
        if msg.get('kind')=='heartbeat': continue
        assert msg=={'kind':'confirm','phase':phase,'token':phase+'-nonce'},msg
        break
"""
    questions = []
    monkeypatch.setattr(
        "builtins.input", lambda question: questions.append(question) or "yes"
    )
    laptop.heartbeat_session([sys.executable, "-u", "-c", script], "test")
    assert len(questions) == 2
    assert "securely grasped" in questions[0] and "ready to release" in questions[1]


def test_shutdown_requires_non_tmux_terminal(laptop, monkeypatch):
    monkeypatch.setenv("TMUX", "test-session")
    with pytest.raises(ValueError, match="outside tmux"):
        laptop.shutdown(SimpleNamespace())


def test_shutdown_cancel_has_no_remote_calls(laptop, monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "no")
    calls = []
    monkeypatch.setattr(laptop, "run", lambda *args, **kwargs: calls.append(args))
    laptop.shutdown(SimpleNamespace())
    assert not calls


def test_shutdown_failure_never_powers_off(laptop, monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "shutdown")
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        return "/tmp/sprout-shutdown.test" if kwargs.get("capture") else ""

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "stop_servers.sh")

    monkeypatch.setattr(laptop, "run", run)
    monkeypatch.setattr(laptop.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        laptop.shutdown(
            SimpleNamespace(
                nuc="nuc",
                perception_ssh="perception",
                workstation="ws",
                repo="/repo/real-world",
            )
        )
    assert not any("poweroff" in str(c) or "shutdown -h" in str(c) for c in calls)


def test_reference_initialization_never_reads_or_moves_robot(laptop, monkeypatch):
    events = []
    monkeypatch.setattr(
        laptop, "remote", lambda args, action, **kwargs: events.append(action)
    )
    monkeypatch.setattr(laptop, "run", lambda *args, **kwargs: events.append(args[0]))
    monkeypatch.setattr(
        laptop, "state", lambda *_: pytest.fail("init must not connect to robot")
    )
    args = SimpleNamespace(
        action="init", workstation="student@ws", repo="/home/student/repo"
    )
    laptop.workflow(args)
    assert events == ["capture", "scp", "scp", sys.executable, "scp", "init"]


@pytest.mark.parametrize("failure", [None, "retract", "servers", "poweroff"])
def test_shutdown_retracts_before_servers_and_poweroff(laptop, monkeypatch, failure):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    answers = iter(["shutdown", "poweroff"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    events = []

    def run(*args, **kwargs):
        command = args[-1]
        if "hardware_tests/test_arm_retract.py --shutdown" in command:
            events.append("retract")
            assert args[:2] == ("ssh", "-tt")
            assert "sprout-live" in command
            if failure == "retract":
                raise subprocess.CalledProcessError(1, args)
        elif command == "sudo poweroff":
            events.append("poweroff")
            if failure == "poweroff":
                raise subprocess.CalledProcessError(1, args)
        return ""

    def stop(args, **kwargs):
        assert args[0] == "bash" and args[1].endswith("stop_servers.sh")
        assert kwargs["env"]["PRPL_NUC_HOST"] == "nuc"
        assert kwargs["env"]["PRPL_PERCEPTION_HOST"] == "perception"
        events.append("servers")
        if failure == "servers":
            raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(laptop, "run", run)
    monkeypatch.setattr(laptop.subprocess, "run", stop)
    args = SimpleNamespace(
        nuc="nuc",
        perception_ssh="perception",
        workstation="ws",
        repo="/repo/real-world",
    )
    if failure:
        with pytest.raises(subprocess.CalledProcessError):
            laptop.shutdown(args)
    else:
        laptop.shutdown(args)
    expected = ["retract", "servers", "poweroff"]
    if failure:
        expected = expected[: expected.index(failure) + 1]
    assert events == expected


def test_shutdown_disconnection_does_not_claim_poweroff_success(
    laptop, monkeypatch, capsys
):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    answers = iter(["shutdown", "poweroff"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    def run(*args, **kwargs):
        if args[-1] == "sudo poweroff":
            raise subprocess.CalledProcessError(255, args)
        return ""

    monkeypatch.setattr(laptop, "run", run)
    monkeypatch.setattr(laptop.subprocess, "run", lambda *args, **kwargs: None)
    laptop.shutdown(
        SimpleNamespace(
            nuc="nuc",
            perception_ssh="perception",
            workstation="ws",
            repo="/repo/real-world",
        )
    )
    assert "Shutdown is unconfirmed" in capsys.readouterr().out
