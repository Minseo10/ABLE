"""Workstation recipe client and laptop stop control for the offline Sprout stack.

Adapts Josh's TossingRobotClient interruption pattern (prpl-tidybot
fe53983330b2dd032c8ad1fb12e561ff701c68e4): stop uses a fresh connection,
because the skill RPC connection may be blocked. No automatic retry of motion.
Only the offline protocol is currently accepted. Authenticated manager RPC uses
pickle; keep it on loopback or an SSH tunnel with a trusted private authkey.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from multiprocessing import AuthenticationError
from multiprocessing.managers import BaseManager, RemoteError
from pathlib import Path
from typing import Any


class SproutManager(BaseManager):
    """Separate protocol name prevents accidentally invoking Josh's cube skills."""


SproutManager.register("Sprout")


class SproutClient:
    def __init__(self, host: str, port: int, authkey: bytes) -> None:
        self.address = (host, port)
        self.authkey = authkey

    def _connect(self) -> Any:
        manager = SproutManager(address=self.address, authkey=self.authkey)
        manager.connect()
        return manager.Sprout()  # type: ignore[attr-defined]

    def stop(self) -> dict[str, Any]:
        """Fresh connection on every call; an acknowledgement is required."""
        # BaseProxy caches a connection per address AND thread. Merely making a
        # new proxy in the interrupted thread can reuse its still-busy socket.
        # A new thread also bounds how long the operator waits for an ACK.
        reply: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def request() -> None:
            try:
                reply.put((True, dict(self._connect().stop())))
            except (
                OSError,
                EOFError,
                RuntimeError,
                AuthenticationError,
                RemoteError,
            ) as exc:
                reply.put((False, exc))

        threading.Thread(target=request, daemon=True).start()
        try:
            succeeded, result = reply.get(timeout=3.0)
        except queue.Empty as exc:
            raise RuntimeError("STOP NOT CONFIRMED within 3 seconds") from exc
        if not succeeded:
            raise RuntimeError("STOP NOT CONFIRMED: connection failed") from result
        if result.get("stopped") is not True:
            raise RuntimeError("STOP NOT CONFIRMED by the server")
        return result

    def status(self) -> dict[str, Any]:
        return dict(self._connect().status())

    def cycle(self) -> dict[str, Any]:
        try:
            proxy = self._connect()
            if dict(proxy.status()).get("mode") != "offline_fake":
                raise RuntimeError("Real execution is not implemented or commissioned")
            return dict(proxy.cycle())
        except KeyboardInterrupt:
            self.stop()
            raise

    def watch_stop(self) -> None:
        """Foreground laptop terminal: Ctrl-C requests a latched software stop."""
        self.status()
        print("Stop control connected. Press Ctrl-C here to request STOP.", flush=True)
        try:
            while True:
                self.status()
                time.sleep(0.5)
        except KeyboardInterrupt:
            print(json.dumps(self.stop(), indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cycle", "status", "stop", "watch-stop"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50012)
    parser.add_argument("--authkey-file", type=Path, required=True)
    args = parser.parse_args()
    client = SproutClient(args.host, args.port, args.authkey_file.read_bytes())
    try:
        if args.command == "watch-stop":
            client.watch_stop()
        else:
            print(json.dumps(getattr(client, args.command)(), indent=2))
    except KeyboardInterrupt:
        return 130
    except (OSError, EOFError, RuntimeError, AuthenticationError, RemoteError) as exc:
        print(f"Request failed; do not assume motion stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
