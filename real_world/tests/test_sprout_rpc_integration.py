"""Actual local RPC and OS SIGINT tests, with a synthetic backend only.

Set SPROUT_TIDYBOT_REPO to the companion checkout. Never use a robot address.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from sprout_experiment.client import SproutClient


@pytest.fixture
def server(tmp_path):
    tidy = os.environ.get("SPROUT_TIDYBOT_REPO")
    if not tidy:
        pytest.skip("Set SPROUT_TIDYBOT_REPO for the offline RPC integration tests")
    auth = tmp_path / "authkey"
    auth.write_bytes(os.urandom(32))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(tidy) / "src"), str(Path(__file__).resolve().parents[1] / "src")]
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "prpl_tidybot.sprout.offline_server",
            "--port",
            "0",
            "--authkey-file",
            str(auth),
            "--phase-delay",
            "0.3",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # Process startup is local and produces one flushed JSON readiness line.
        import select

        assert select.select([process.stdout], [], [], 5)[0], "Server startup timed out"
        ready = json.loads(process.stdout.readline())
        assert ready["mode"] == "offline_fake"
        yield SproutClient("127.0.0.1", ready["port"], auth.read_bytes()), auth, env
    finally:
        process.terminate()
        process.communicate(timeout=5)


def test_full_cycle_over_rpc(server):
    client, _, _ = server
    assert client.cycle()["completed"][-1] == "retreat"
    assert client.status()["gripper_closed"] is False


def test_ctrl_c_during_rpc_preserves_closed_gripper(server):
    client, auth, env = server
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sprout_experiment.client",
            "cycle",
            "--port",
            str(client.address[1]),
            "--authkey-file",
            str(auth),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if client.status()["gripper_closed"]:
                break
            time.sleep(0.01)
        else:
            pytest.fail("Synthetic gripper did not close")
        process.send_signal(signal.SIGINT)
        process.communicate(timeout=5)
        assert process.returncode == 130
        state = client.status()
        assert state["stopped"] is True
        assert state["gripper_closed"] is True
        assert "open" not in state["events"] and "retreat" not in state["events"]
        with pytest.raises(RuntimeError):
            client.cycle()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
