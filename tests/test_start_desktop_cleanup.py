"""Closing the desktop launcher must leave no backend behind.

These exercise the real machinery against a real process tree: a parent that
spawns a child which holds a port, exactly the shape of npm -> concurrently ->
electron -> python that made a backend outlive the closed window.
"""

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import start_desktop  # noqa: E402

psutil = pytest.importorskip("psutil")

def _free_port() -> int:
    """A port nobody holds right now.

    This file used the fixed port 8137. When something else listened there (a
    verification backend another session had started on it), the spawned child
    could not bind, `_wait_for_owner` returned that stranger as the test's own
    child, and `test_our_own_backend_is_killed` killed it -- the very thing the
    launcher's identity check exists to prevent. A free port plus an owner that
    must belong to the spawned tree (`_wait_for_owner`) rule that out.

    The band stays near 8137 on purpose: below the OS's ephemeral range, so
    the port is not one the OS is about to hand to an outgoing connection
    between the bind check and the child's bind. (It used to be blamed on the
    owner lookup, which wanted LISTEN while most listeners on Windows 11 build
    26300 read NONE; since commit 2d2d22fc the lookup no longer reads the
    status at all.)"""
    import socket
    for port in range(8137, 8237):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free port in 8137-8236")


PORT = _free_port()  # not the app's port, and nobody else's


@pytest.fixture(autouse=True)
def _a_port_of_its_own():
    """Each test asks again: the child of the test before may still be dying."""
    global PORT
    PORT = _free_port()


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
    # The same isolation the launcher gives its own child (start_desktop.py:
    # `start_new_session=True` on POSIX, CREATE_NEW_PROCESS_GROUP on Windows).
    #
    # Without it this tree inherits PYTEST's process group, and `_kill_tree`
    # on POSIX is `killpg(getpgid(pid), SIGTERM)` -- so the test signals the
    # whole group: pytest, the shell, everything. Measured on the Linux
    # runner (run 35989161362): the suite died mid-test at 94 % with exit 143,
    # SIGTERM, no verdict line. Windows never showed it because its branch is
    # `taskkill /PID /T`, which is scoped to a pid rather than a group.
    #
    # So this is not merely "the test used to kill its runner". The test was
    # exercising a process shape the application never creates, and on the one
    # platform where that difference is observable it took the session down
    # with it.
    isolate = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
               if sys.platform == "win32" else {"start_new_session": True})
    return subprocess.Popen([sys.executable, "-c", parent_src], **isolate)


def _in_tree(pid: int, parent) -> bool:
    """True when ``pid`` is the spawned ``parent`` or one of its descendants."""
    try:
        proc = psutil.Process(pid)
        return pid == parent.pid or any(p.pid == parent.pid for p in proc.parents())
    except psutil.Error:
        return False


def _wait_for_owner(port, parent, timeout=25):
    """The identity of the process holding ``port`` -- only if it belongs to the tree
    this test spawned. A stranger on the port is never handed to the kill code."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        owner = start_desktop._port_owner(port)
        if owner and _in_tree(owner[0], parent):
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
        owner = _wait_for_owner(PORT, parent)
        assert owner is not None, "test child never took the port"
        assert owner[0] != parent.pid, "the CHILD should hold the port, not the parent"

        start_desktop._kill_tree(parent.pid)

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and start_desktop._port_owner(PORT):
            time.sleep(0.3)
        assert start_desktop._port_owner(PORT) is None, "port still held after the tree kill"
    finally:
        _kill_quietly(parent)


def test_lsof_is_used_where_psutil_is_refused_the_scan(monkeypatch):
    """macOS gives psutil.net_connections to root only.

    Measured on the runner (36000286001): four tests in this file fail with
    "None is not None" on macOS and all seven pass on Linux, because
    net_connections raises AccessDenied, _port_owner returned None, and the
    launcher's last-resort cleanup could therefore never identify a backend it
    had started. A Mac left with an orphaned process on port 8000 is exactly
    the failure the identity check was written for.

    Driven with a stub rather than a real lsof, so the test says the same
    thing on every platform.
    """
    import start_desktop

    def refuse(*_args, **_kwargs):
        raise psutil.AccessDenied(pid=None)

    monkeypatch.setattr(psutil, "net_connections", refuse)

    me = os.getpid()
    calls = []

    def fake_lsof(cmd, **_kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=str(me) + chr(10), stderr="")

    monkeypatch.setattr(start_desktop.subprocess, "run", fake_lsof)

    owner = start_desktop._port_owner(PORT)
    assert owner is not None, "the fallback did not run"
    assert owner[0] == me
    assert owner[1] == psutil.Process(me).create_time()
    assert calls and calls[0][0] == "lsof" and f"-iTCP:{PORT}" in calls[0]


def test_the_lsof_fallback_survives_a_machine_without_lsof(monkeypatch):
    """No lsof, or a hung one, must be "cannot tell" -- never a crash.

    _port_owner is called while the launcher is shutting down. An exception
    there would replace an orderly cleanup with a traceback.
    """
    import start_desktop

    for boom in (FileNotFoundError("lsof"), subprocess.TimeoutExpired("lsof", 5)):
        def raiser(*_a, **_k):
            raise boom
        assert start_desktop._port_owner_via_lsof(PORT, run=raiser) is None


def test_the_port_owner_is_identified_by_pid_and_start_time():
    parent = _spawn_tree(PORT)
    try:
        owner = _wait_for_owner(PORT, parent)
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
        owner = _wait_for_owner(PORT, parent)
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
        owner = _wait_for_owner(PORT, parent)
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


_THIS_PROCESS = object()


def _fake_conn(port, status, raddr=(), pid=_THIS_PROCESS):
    """A psutil connection as `_port_owner` reads it. Default pid: this process,
    so psutil.Process(pid).create_time() works. `pid=None` is a socket no
    process owns (psutil reports that for sockets it cannot attribute)."""
    from collections import namedtuple
    Addr = namedtuple("Addr", "ip port")
    Conn = namedtuple("Conn", "laddr raddr status pid")
    return Conn(Addr("127.0.0.1", port), raddr, status,
                os.getpid() if pid is _THIS_PROCESS else pid)


@pytest.mark.parametrize("status", ["LISTEN", "NONE", "CLOSE", "FIN_WAIT1"])
def test_a_socket_without_a_peer_is_the_listener_whatever_its_status_says(monkeypatch, status):
    # Windows 11 build 26300, psutil 7.2.2: one listener of ours read LISTEN, NONE
    # or FIN_WAIT1 depending on the moment and the port, so the status is not read.
    port = 45678
    monkeypatch.setattr(
        psutil, "net_connections",
        lambda kind="inet": [_fake_conn(port, status)])
    assert start_desktop._port_owner(port) == (os.getpid(), psutil.Process(os.getpid()).create_time())


@pytest.mark.parametrize("status", ["NONE", "LISTEN", "ESTABLISHED"])
def test_a_socket_with_a_peer_is_not_the_listener(monkeypatch, status):
    port = 45678
    monkeypatch.setattr(
        psutil, "net_connections",
        lambda kind="inet": [_fake_conn(port, status, ("10.0.0.5", 50000))])
    assert start_desktop._port_owner(port) is None


def test_a_socket_whose_pid_is_gone_is_not_an_owner(monkeypatch):
    dead = 4_000_000
    while psutil.pid_exists(dead):
        dead += 1
    port = 45678
    monkeypatch.setattr(
        psutil, "net_connections",
        lambda kind="inet": [_fake_conn(port, "NONE", pid=dead)])
    assert start_desktop._port_owner(port) is None


@pytest.mark.parametrize("pid", [None, 0])
def test_a_socket_without_a_pid_is_not_an_owner(monkeypatch, pid):
    """No peer, right port, but nobody owns it: there is no process to kill."""
    port = 45678
    monkeypatch.setattr(
        psutil, "net_connections",
        lambda kind="inet": [_fake_conn(port, "NONE", pid=pid)])
    assert start_desktop._port_owner(port) is None


def test_a_listener_on_another_port_is_not_the_owner(monkeypatch):
    monkeypatch.setattr(
        psutil, "net_connections",
        lambda kind="inet": [_fake_conn(45679, "NONE")])
    assert start_desktop._port_owner(45678) is None
