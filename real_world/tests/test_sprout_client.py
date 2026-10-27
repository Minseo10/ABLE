"""Client interruption tests with no sockets or hardware."""

import threading

import pytest
from sprout_experiment.client import SproutClient


def test_interrupt_uses_different_thread_for_stop():
    calling_thread = threading.get_ident()
    stop_threads = []

    class Proxy:
        def status(self):
            return {"mode": "offline_fake"}

        def cycle(self):
            raise KeyboardInterrupt

        def stop(self):
            stop_threads.append(threading.get_ident())
            return {"stopped": True}

    client = SproutClient("unused", 1, b"unused")
    client._connect = Proxy
    with pytest.raises(KeyboardInterrupt):
        client.cycle()
    assert len(stop_threads) == 1 and stop_threads[0] != calling_thread


def test_negative_stop_ack_is_failure():
    class Proxy:
        def stop(self):
            return {"stopped": False}

    client = SproutClient("unused", 1, b"unused")
    client._connect = Proxy
    with pytest.raises(RuntimeError, match="NOT CONFIRMED"):
        client.stop()


def test_real_server_is_refused():
    class Proxy:
        def status(self):
            return {"mode": "real"}

        def cycle(self):
            pytest.fail("No real motion may be requested")

    client = SproutClient("unused", 1, b"unused")
    client._connect = Proxy
    with pytest.raises(RuntimeError, match="not implemented"):
        client.cycle()
