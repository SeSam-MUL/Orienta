"""Closing the desktop launcher must leave no backend behind.

These exercise the real machinery against a real process tree: a parent that
spawns a child which holds a port, exactly the shape of npm -> concurrently ->
electron -> python that made a backend outlive the closed window.
"""

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import start_desktop  # noqa: E402

psutil = pytest.importorskip("psutil")

PORT = 8137          # not the app's port: this test must never touch a real backend


def _spawn_tree(port: int):
    """A parent process whose CHILD holds `port`. Returns the parent Popen."""
    child_src = textwrap.dedent(f"""
        import socket, time
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", {port}))
        s.listen(1)
        while True:
            time.sleep(0.2)
    """)
    parent_src = textwrap.dedent(f"""
        import subprocess, sys, time
        subprocess.Popen([sys.executable, "-c", {child_src!r}])
        while True:
            time.sleep(0.2)
    """)
    return subprocess.Popen([sys.executable, "-c", parent_src])


def _wait_for_owner(port, timeout=25):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        owner = start_desktop._port_owner(port)
        if owner:
            return owner
        time.sleep(0.3)
    return None


def _kill_quietly(proc):
    try:
        start_desktop._kill_tree(proc.pid)
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001 — teardown must not fail a passing test
        pass


def test_killing_the_tree_takes_the_port_holder_with_it():
    # This is the whole point: `proc.terminate()` on Windows ends the shim and
    # leaves the grandchild holding the port. `_kill_tree` must not.
    parent = _spawn_tree(PORT)
    try:
        owner = _wait_for_owner(PORT)
        assert owner is not None, "test child never took the port"
        assert owner[0] != parent.pid, "the CHILD should hold the port, not the parent"

        start_desktop._kill_tree(parent.pid)

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and start_desktop._port_owner(PORT):
            time.sleep(0.3)
        assert start_desktop._port_owner(PORT) is None, "port still held after the tree kill"
    finally:
        _kill_quietly(parent)


def test_the_port_owner_is_identified_by_pid_and_start_time():
    parent = _spawn_tree(PORT)
    try:
        owner = _wait_for_owner(PORT)
        assert owner is not None
        pid, created = owner
        assert isinstance(pid, int) and created > 0
        # Reading it twice gives the same identity — no drift from sampling.
        assert start_desktop._port_owner(PORT) == owner
    finally:
        _kill_quietly(parent)


def test_a_stranger_on_the_port_is_left_alone():
    """PIDs get recycled. A process we did not start must survive cleanup."""
    parent = _spawn_tree(PORT)
    try:
        owner = _wait_for_owner(PORT)
        assert owner is not None

        # Same pid, different start time: someone else's process now.
        stale_identity = (owner[0], owner[1] - 5000.0)
        killed = start_desktop._kill_backend_if_ours(stale_identity, PORT)

        assert killed is False
        assert start_desktop._port_owner(PORT) == owner, "we killed a process that was not ours"
    finally:
        _kill_quietly(parent)


def test_our_own_backend_is_killed():
    parent = _spawn_tree(PORT)
    try:
        owner = _wait_for_owner(PORT)
        assert owner is not None

        assert start_desktop._kill_backend_if_ours(owner, PORT) is True

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and start_desktop._port_owner(PORT):
            time.sleep(0.3)
        assert start_desktop._port_owner(PORT) is None
    finally:
        _kill_quietly(parent)


def test_nothing_to_kill_is_not_an_error():
    assert start_desktop._kill_backend_if_ours(None, PORT) is False
    assert start_desktop._port_owner(PORT) is None


def test_the_backend_is_told_to_watch_the_launcher():
    """The hard-kill path: no handler runs, so the backend must watch us.

    Measured before this wiring: killing the launcher outright left the backend
    holding port 8000 indefinitely. With the variable set it exits ~2.5 s later.
    """
    import os
    import start_app

    env = start_app._backend_env(headless=False, base={})
    assert env["KIKUCHIPY_PARENT_PID"] == str(os.getpid())
    assert env["KIKUCHIPY_WATCHDOG"] == "1"

    # Headless keeps the backend alive when the BROWSER closes — that is not
    # the same as this launcher exiting, which must still take it down.
    head = start_app._backend_env(headless=True, base={})
    assert head["KIKUCHIPY_WATCHDOG"] == "0"
    assert head["KIKUCHIPY_PARENT_PID"] == str(os.getpid())


def test_the_launcher_env_does_not_drop_the_rest_of_the_environment():
    import start_app

    env = start_app._backend_env(headless=False, base={"PATH": "/somewhere", "FOO": "bar"})
    assert env["PATH"] == "/somewhere" and env["FOO"] == "bar"
