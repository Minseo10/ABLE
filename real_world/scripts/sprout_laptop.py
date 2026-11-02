"""Laptop helper: fresh camera + robot state + forearm tracking + preview by default.

Standard library only; never copies private Sprout models. Default is preview only.
--execute additionally requests one commissioned real cycle after terminal
approval. It refuses incomplete/uncommissioned previews. SSH may prompt normally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shlex
import subprocess
import sys
import tempfile
import threading
from pathlib import Path, PurePosixPath

LIVE_PREFIX = 'cd ~/sprout-live && export PRPL_LAB=prpl PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" SPROUT_TIDYBOT_MODEL="$PWD/tidybot-model/stanford_tidybot/tidybot.xml" && '


def deploy(args: argparse.Namespace) -> None:
    """Install in fresh minseo branches; preserve original repositories/servers."""
    remote(args, "package")
    with tempfile.TemporaryDirectory(prefix="sprout-deploy-") as local:
        archive = str(Path(local) / "deploy.tar.gz")
        run(
            "scp",
            args.workstation + ":" + args.repo + "/.local/sprout/deploy.tar.gz",
            archive,
        )
        for host in (args.nuc, args.perception_ssh):
            temporary = run(
                "ssh", host, "mktemp -d /tmp/sprout-deploy.XXXXXX", capture=True
            )
            if (
                not temporary.startswith("/tmp/sprout-deploy.")
                or "/" in temporary[len("/tmp/") :]
                or "\n" in temporary
            ):
                raise ValueError("Unexpected deployment temporary path")
            run("scp", archive, host + ":" + temporary + "/deploy.tar.gz")
            install = """set -eu
test ! -e "$HOME/sprout-live" || test -L "$HOME/sprout-live"
build=$(mktemp -d "$HOME/sprout-build.XXXXXX")
rmdir "$build"
git -C "$HOME/prpl-tidybot" worktree add -b "minseo/$(basename "$build")" "$build" HEAD
tar -xzf "$1/deploy.tar.gz" -C "$build"
# Preserve the machine's working calibration, including uncommitted lab updates.
cp -a "$HOME/prpl-tidybot/src/prpl_tidybot/marker_detector/camera_params/." "$build/src/prpl_tidybot/marker_detector/camera_params/"
cp "$HOME/prpl-tidybot/conf/lab/prpl.yaml" "$build/conf/lab/prpl.yaml"
ln -s "$HOME/prpl-tidybot/.venv" "$build/.venv"
cd "$build"
PYTHONPATH="$build/src" .venv/bin/python -c 'import mujoco, cv2, numpy; import prpl_tidybot.sprout.live_runtime; print("Live runtime imports OK")'
PYTHONPATH="$build/src" .venv/bin/python -m prpl_tidybot.sprout.planner_check
ln -sfn "$build" "$HOME/sprout-live"
rm "$1/deploy.tar.gz"
rmdir "$1"
"""
            print(
                f"Installing isolated code on {host}; existing servers stay unchanged",
                flush=True,
            )
            subprocess.run(
                ["ssh", host, "bash -s -- " + shlex.quote(temporary)],
                input=install,
                text=True,
                check=True,
            )
    print(
        "Installed. Stop the old perception process, then run: python3 ~/sprout.py perception",
        flush=True,
    )


def perception(args: argparse.Namespace) -> None:
    run(
        "ssh",
        "-tt",
        args.perception_ssh,
        LIVE_PREFIX
        + ".venv/bin/python -u -m prpl_tidybot.marker_detector.server --lab prpl --host 0.0.0.0",
    )


def shutdown(args: argparse.Namespace) -> None:
    """Verify retraction, stop servers outside tmux, then request NUC poweroff."""
    if os.environ.get("TMUX"):
        raise ValueError(
            "Run shutdown from a separate laptop terminal outside tmux, as Josh's guide requires"
        )
    if not sys.stdin.isatty():
        raise ValueError("Shutdown requires an interactive laptop terminal")
    print(
        "Shutdown will RETRACT THE ARM, stop servers, then power off the NUC.\n"
        "Finish/cancel the experiment first; Sprout must be supported and released.\n"
        "The complete retract sweep must be clear. Physical switches remain manual.",
        flush=True,
    )
    if input("Type shutdown to continue: ").strip() != "shutdown":
        print("Shutdown cancelled; no changes requested.", flush=True)
        return
    # Refuse to kill a supervised skill that might still be holding Sprout.
    guard = "if pgrep -f '[p]rpl_tidybot.sprout.remote_execute' >/dev/null; then echo 'Active Sprout session: stop/finish it in its original terminal first' >&2; exit 1; fi"
    run("ssh", "-o", "ConnectTimeout=5", args.nuc, guard)
    with tempfile.TemporaryDirectory(prefix="sprout-shutdown-") as temporary:
        tidy_repo = str(PurePosixPath(args.repo).parent / "prpl-tidybot")
        stop_script = str(Path(temporary) / "stop_servers.sh")
        run(
            "scp",
            args.workstation + ":" + tidy_repo + "/scripts/stop_servers.sh",
            stop_script,
        )
        print("1/3 Retract and verify completion on the NUC", flush=True)
        run(
            "ssh",
            "-tt",
            args.nuc,
            LIVE_PREFIX
            + "test -f hardware_tests/test_arm_retract.py && "
            + ".venv/bin/python -u hardware_tests/test_arm_retract.py --shutdown",
        )
        # The laptop owns this process, outside the tmux session being removed.
        # A failed/aborted retract raises above and never reaches server teardown.
        print("2/3 Stop arm/base/perception servers from the laptop", flush=True)
        env = dict(
            os.environ, PRPL_NUC_HOST=args.nuc, PRPL_PERCEPTION_HOST=args.perception_ssh
        )
        subprocess.run(["bash", stop_script], env=env, check=True)
    if (
        input(
            "Verify the arm is stably retracted. Type poweroff to shut down the NUC: "
        ).strip()
        != "poweroff"
    ):
        print("NUC left on; servers are stopped.", flush=True)
        return
    print("3/3 Request NUC poweroff (sudo may ask for its password)", flush=True)
    try:
        run("ssh", "-tt", args.nuc, "sudo poweroff")
    except subprocess.CalledProcessError as error:
        if error.returncode != 255:
            raise
        print(
            "SSH disconnected during poweroff. Shutdown is unconfirmed. "
            "Check that the NUC is off before switching batteries off.",
            flush=True,
        )
        return
    print(
        "NUC poweroff requested. Wait until it is off.\n"
        "Then: press arm power button → camping battery AC to DC → camping battery off → SLA switch off.\n"
        "These physical steps cannot be performed by this script.",
        flush=True,
    )


def run(*args: str, capture: bool = False) -> str:
    print("+ " + shlex.join(args), flush=True)
    result = subprocess.run(
        args, check=True, text=True, stdout=subprocess.PIPE if capture else None
    )
    return result.stdout.strip() if capture else ""


def remote(args: argparse.Namespace, action: str, tunnel: bool = False) -> None:
    options = ["-o", "ConnectTimeout=5"]
    if tunnel:
        options += [
            "-o",
            "ExitOnForwardFailure=yes",
            "-R",
            f"127.0.0.1:16003:{args.perception}:6003",
        ]
    run(
        "ssh",
        *options,
        args.workstation,
        shlex.join(["bash", args.repo + "/real_world/scripts/sprout_sim.sh", action]),
    )


def initialize(args: argparse.Namespace) -> None:
    """Capture and select a reference locally; never read or move arm/base."""
    remote(args, "capture", tunnel=True)
    with tempfile.TemporaryDirectory(prefix="sprout-reference-") as directory:
        root = Path(directory)
        image, selector, selection = (
            root / "current.jpg",
            root / "select.py",
            root / "selection.json",
        )
        run(
            "scp",
            args.workstation + ":" + args.repo + "/.local/sprout/current.jpg",
            str(image),
        )
        run(
            "scp",
            args.workstation
            + ":"
            + args.repo
            + "/real_world/scripts/sprout_select_reference.py",
            str(selector),
        )
        run(
            sys.executable,
            str(selector),
            "--image",
            str(image),
            "--output",
            str(selection),
        )
        run(
            "scp",
            str(selection),
            args.workstation
            + ":"
            + args.repo
            + "/.local/sprout/reference-selection.json",
        )
        remote(args, "init")


def state(args: argparse.Namespace) -> None:
    reader = str(
        PurePosixPath(args.repo).parent
        / "prpl-tidybot/src/prpl_tidybot/sprout/read_robot_state.py"
    )
    with tempfile.TemporaryDirectory(prefix="sprout-") as directory:
        script = str(Path(directory) / "read_robot_state.py")
        snapshot = str(Path(directory) / "robot-state.json")
        run("scp", args.workstation + ":" + reader, script)
        temporary = run(
            "ssh",
            "-o",
            "ConnectTimeout=5",
            args.nuc,
            "mktemp -d /tmp/sprout-state.XXXXXX",
            capture=True,
        )
        if (
            not temporary.startswith("/tmp/sprout-state.")
            or "/" in temporary[len("/tmp/") :]
            or "\n" in temporary
        ):
            raise ValueError("Unexpected NUC temporary path")
        try:
            run("scp", script, args.nuc + ":" + temporary + "/reader.py")
            command = "cd ~/prpl-tidybot && " + shlex.join(
                [
                    "timeout",
                    "30s",
                    ".venv/bin/python",
                    "-u",
                    temporary + "/reader.py",
                    "--marker-host",
                    args.perception,
                    "--robot-id",
                    str(args.robot_id),
                    "--output",
                    temporary + "/state.json",
                ]
            )
            run("ssh", "-tt", "-o", "ConnectTimeout=5", args.nuc, command)
            run("scp", args.nuc + ":" + temporary + "/state.json", snapshot)
            run(
                "ssh",
                args.workstation,
                shlex.join(["mkdir", "-p", args.repo + "/.local/sprout"]),
            )
            run(
                "scp",
                snapshot,
                args.workstation + ":" + args.repo + "/.local/sprout/robot-state.json",
            )
        finally:
            cleanup = shlex.join(
                ["rm", "-f", temporary + "/reader.py", temporary + "/state.json"]
            )
            cleanup += " && " + shlex.join(["rmdir", temporary])
            # Cleanup must not mask a failed telemetry capture.
            subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", args.nuc, cleanup], check=False
            )


def video(args: argparse.Namespace, demo: bool = False) -> None:
    args.video.parent.mkdir(parents=True, exist_ok=True)
    folder = "rehearsal" if demo else "camera-preview"
    run(
        "scp",
        f"{args.workstation}:{args.repo}/.local/sprout/{folder}/preview.mp4",
        str(args.video),
    )
    if sys.platform == "darwin" and not args.no_open:
        run("open", str(args.video))


def validate_execution_bundle(directory: Path, viewed_video: Path) -> dict:
    """Never execute a failed/uncommissioned preview or a different video."""
    review = json.loads((directory / "review.json").read_text())
    if review.get("execution_ready") is not True:
        blockers = review.get(
            "blockers", ["Live scene and stop integration not commissioned"]
        )
        raise ValueError("Real execution unavailable: " + "; ".join(blockers))
    if review.get("completed") is not True or review.get("source") != "live_camera":
        raise ValueError(
            "Real execution requires a completed, calibrated live-camera preview"
        )
    hashes = review.get("artifact_sha256", {})
    if set(hashes) != {
        "preview.mp4",
        "trajectory.json",
        "scene.json",
        "live.json",
        "reference.jpg",
    }:
        raise ValueError("Unexpected or missing execution artifacts")
    for name, expected in hashes.items():
        path = viewed_video if name == "preview.mp4" else directory / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"{name} differs from reviewed artifact; preview again")
    return review


def approval(prompt=input) -> bool:
    """Only a deliberate interactive answer authorizes the real-cycle request."""
    if not sys.stdin.isatty():
        raise ValueError("Real execution requires an interactive laptop terminal")
    try:
        answer = prompt(
            "Watch the video. Enter or yes requests ONE REAL cycle; anything else cancels: "
        )
    except (EOFError, KeyboardInterrupt):
        return False
    return answer.strip().lower() in ("", "yes")


def heartbeat_session(command: list[str], review_hash: str) -> None:
    """Keep approval alive over SSH; Ctrl-C sends stop and closes the channel."""
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
    )
    finished = threading.Event()
    send_lock = threading.Lock()
    events = queue.Queue()

    def output() -> None:
        for line in process.stdout:
            print(line, end="", flush=True)
            try:
                event = json.loads(line)
                if (
                    isinstance(event, dict)
                    and event.get("event") == "operator_confirmation"
                ):
                    events.put(event)
            except ValueError:
                pass

    threading.Thread(target=output, daemon=True).start()

    def send(message: dict) -> None:
        with send_lock:
            if process.stdin is None or process.stdin.closed:
                return
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()

    def heartbeat() -> None:
        try:
            send({"kind": "approve", "review_sha256": review_hash})
            while not finished.wait(0.5):
                send({"kind": "heartbeat"})
        except (BrokenPipeError, OSError):
            pass

    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        while process.poll() is None:
            try:
                event = events.get(timeout=0.1)
            except queue.Empty:
                continue
            phase = event.get("phase")
            question = {
                "grasp": "Is the forearm securely grasped at the tape, without pinching other parts?",
                "support": "Is the forearm resting on the floor/support and ready to release?",
            }.get(phase)
            if question is None:
                send({"kind": "stop"})
                raise ValueError("Unknown operator confirmation")
            try:
                answer = input(
                    question + " Type yes to continue; anything else stops: "
                )
            except EOFError:
                answer = "no"
            if answer.strip().lower() != "yes":
                send({"kind": "stop"})
                raise ValueError("Operator stopped the cycle")
            send({"kind": "confirm", "phase": phase, "token": event["token"]})
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    except KeyboardInterrupt:
        print(
            "Stop requested. NUC stop latch/watchdog should halt the cycle; verify the robot physically.",
            flush=True,
        )
        try:
            send({"kind": "stop"})
        except (BrokenPipeError, OSError):
            pass
        raise
    finally:
        finished.set()
        worker.join(timeout=1)
        if process.stdin:
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()


def execute_reviewed(args: argparse.Namespace) -> None:
    """Transfer only reviewed artifacts; approved NUC runtime owns all motion."""
    with tempfile.TemporaryDirectory(prefix="sprout-review-") as local:
        bundle = Path(local)
        source = args.workstation + ":" + args.repo + "/.local/sprout/camera-preview/"
        run("scp", source + "review.json", str(bundle / "review.json"))
        review = json.loads((bundle / "review.json").read_text())
        if review.get("execution_ready") is not True:
            raise ValueError(
                "Real execution unavailable: " + "; ".join(review.get("blockers", []))
            )
        for name in ("trajectory.json", "scene.json", "live.json", "reference.jpg"):
            run("scp", source + name, str(bundle / name))
        validate_execution_bundle(bundle, args.video)
        if not approval():
            print("Real execution cancelled; no motion requested.", flush=True)
            return
        validate_execution_bundle(bundle, args.video)
        review_hash = hashlib.sha256((bundle / "review.json").read_bytes()).hexdigest()
        directory = run(
            "ssh",
            "-o",
            "ConnectTimeout=5",
            args.nuc,
            "mktemp -d /tmp/sprout-review.XXXXXX",
            capture=True,
        )
        if (
            not directory.startswith("/tmp/sprout-review.")
            or "/" in directory[len("/tmp/") :]
            or "\n" in directory
        ):
            raise ValueError("Unexpected NUC review directory")
        for name in (
            "review.json",
            "trajectory.json",
            "scene.json",
            "live.json",
            "reference.jpg",
        ):
            run("scp", str(bundle / name), args.nuc + ":" + directory + "/" + name)
        run("scp", str(args.video), args.nuc + ":" + directory + "/preview.mp4")
        command = LIVE_PREFIX + shlex.join(
            [
                ".venv/bin/python",
                "-u",
                "-m",
                "prpl_tidybot.sprout.remote_execute",
                "--review",
                directory + "/review.json",
                "--runtime",
                args.runtime,
            ]
        )
        print(
            "Starting approved NUC session. Ctrl-C requests a software stop.",
            flush=True,
        )
        heartbeat_session(
            [
                "ssh",
                "-T",
                "-o",
                "ConnectTimeout=5",
                "-o",
                "ServerAliveInterval=1",
                "-o",
                "ServerAliveCountMax=3",
                args.nuc,
                command,
            ],
            review_hash,
        )


def workflow(args: argparse.Namespace) -> None:
    if args.action == "run":
        remote(args, "check")
        remote(args, "capture", tunnel=True)
        state(args)
        remote(args, "preview")
        video(args)
        if getattr(args, "execute", False):
            execute_reviewed(args)
    elif args.action == "init":
        initialize(args)
    elif args.action == "state":
        state(args)
    elif args.action == "deploy":
        deploy(args)
    elif args.action == "shutdown":
        shutdown(args)
    elif args.action == "perception":
        perception(args)
    else:
        remote(args, args.action, tunnel=args.action == "capture")
        if args.action in ("preview", "demo"):
            video(args, demo=args.action == "demo")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        default="run",
        choices=[
            "run",
            "setup",
            "init",
            "capture",
            "state",
            "preview",
            "demo",
            "deploy",
            "perception",
            "shutdown",
        ],
    )
    parser.add_argument(
        "--workstation",
        default=os.environ.get("SPROUT_WORKSTATION", "minseo@100.83.23.79"),
    )
    parser.add_argument(
        "--nuc", default=os.environ.get("SPROUT_NUC", "tidybot@192.168.0.76")
    )
    parser.add_argument(
        "--perception", default=os.environ.get("SPROUT_PERCEPTION", "192.168.0.131")
    )
    parser.add_argument(
        "--perception-ssh",
        default=os.environ.get("SPROUT_PERCEPTION_SSH", "yixuan@192.168.0.131"),
    )
    parser.add_argument("--robot-id", type=int, default=0)
    parser.add_argument(
        "--repo",
        default=os.environ.get(
            "SPROUT_REPO",
            "/home/minseo/research/learning-forceful-manipulation/tmp/real-world",
        ),
    )
    parser.add_argument(
        "--video", type=Path, default=Path.home() / "Downloads/sprout-preview.mp4"
    )
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="After preview, request one commissioned real cycle with terminal approval",
    )
    parser.add_argument(
        "--runtime",
        default="prpl_tidybot.sprout.live_runtime:create_runtime",
        help="Installed NUC module:factory(review_path); concrete adapter is the default",
    )
    args = parser.parse_args()
    if args.execute and args.action != "run":
        parser.error("--execute is only allowed with the full run workflow")
    try:
        workflow(args)
    except (subprocess.CalledProcessError, ValueError) as error:
        parser.exit(1, f"Stopped: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Cancelled. If a real session started, verify it stopped.\n")


if __name__ == "__main__":
    main()
