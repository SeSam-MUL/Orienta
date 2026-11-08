"""Every end of the backend process leaves its reason in the log (backend/lifecycle.py).

The ends are run for real in a child interpreter: an exit that skips ``atexit``
(``os._exit``), a plain exit, a native crash, and a state file that outlives a
hard kill. A test that only called the helpers in this process could not tell
whether the line is written before the process is gone.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _child(tmp_path, body, env_extra=None):
    """Run ``body`` in a fresh interpreter whose root logger writes to tmp_path/orienta.log."""
    script = textwrap.dedent(f"""
        import logging, sys
        sys.path.insert(0, {str(ROOT)!r})
        logging.basicConfig(filename={str(tmp_path / 'orienta.log')!r}, level=logging.INFO,
                            format="%(levelname)s %(name)s: %(message)s")
        from backend import lifecycle
        lifecycle.install({str(tmp_path)!r})
    """) + textwrap.dedent(body)
    env = dict(os.environ)
    env.pop("ORIENTA_BACKEND_STATE_FILE", None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          timeout=60, env=env, cwd=str(ROOT))


def _log(tmp_path):
    p = tmp_path / "orienta.log"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def test_exit_now_writes_the_reason_before_the_process_is_gone(tmp_path):
    r = _child(tmp_path, """
        t = lifecycle.request_started("POST", "/api/indexing/start")
        lifecycle.exit_now(7, "parent watchdog: the launching process (PID 1) is gone (exited)")
        print("not reached")
    """)
    assert r.returncode == 7
    assert "not reached" not in r.stdout
    log = _log(tmp_path)
    assert "Backend process ends" in log
    assert "parent watchdog: the launching process (PID 1) is gone (exited)" in log
    # what it was doing: the request that never finished
    assert "still running: POST /api/indexing/start" in log
    assert "memory commit" in log


def test_a_plain_exit_still_says_that_it_ended(tmp_path):
    r = _child(tmp_path, """
        t = lifecycle.request_started("GET", "/api/settings")
        lifecycle.request_finished(t)
        sys.exit(0)
    """)
    assert r.returncode == 0
    log = _log(tmp_path)
    assert "Backend process ends" in log
    assert "no reason was stated" in log
    assert "last request GET /api/settings" in log
    assert "still running: none" in log


def test_the_first_stated_reason_wins(tmp_path):
    r = _child(tmp_path, """
        lifecycle.note_reason("orderly server shutdown")
        lifecycle.note_reason("something later")
    """)
    assert r.returncode == 0
    log = _log(tmp_path)
    assert "orderly server shutdown" in log
    assert "something later" not in log


def test_a_native_crash_leaves_the_python_stack_in_the_fault_log(tmp_path):
    r = _child(tmp_path, """
        import faulthandler
        def deep_inside_a_c_extension():
            faulthandler._sigsegv()
        deep_inside_a_c_extension()
    """)
    assert r.returncode != 0
    fault = (tmp_path / "backend-fault.log").read_text(encoding="utf-8")
    assert "===== backend pid" in fault
    assert "Fatal Python error" in fault
    assert "deep_inside_a_c_extension" in fault


def test_the_state_file_names_what_was_running_when_the_process_was_killed(tmp_path):
    """A hard kill runs nothing in the process; the state file written before it is what is left."""
    state_file = tmp_path / "state" / "backend-state.json"
    r = _child(tmp_path, """
        lifecycle.request_started("POST", "/api/phase-map/apply-to-grain")
        lifecycle.write_state()
        import os
        os.kill(os.getpid(), 9)       # TerminateProcess on Windows, SIGKILL elsewhere
    """, env_extra={"ORIENTA_BACKEND_STATE_FILE": str(state_file)})
    assert r.returncode != 0
    s = json.loads(state_file.read_text(encoding="utf-8"))
    assert s["pid"] > 0
    assert s["in_flight"][0]["path"] == "/api/phase-map/apply-to-grain"
    assert "commit" in s["memory_mib"]
    assert "ended" not in s           # nothing ran after the kill
    assert "Backend process ends" not in _log(tmp_path)


def test_an_orderly_end_marks_the_state_file(tmp_path):
    state_file = tmp_path / "backend-state.json"
    r = _child(tmp_path, """
        lifecycle.note_reason("orderly server shutdown")
    """, env_extra={"ORIENTA_BACKEND_STATE_FILE": str(state_file)})
    assert r.returncode == 0
    assert json.loads(state_file.read_text(encoding="utf-8"))["ended"] == "orderly server shutdown"


@pytest.mark.parametrize("name, needle", [
    ("_frontend_watchdog", "frontend watchdog"),
    ("_parent_watchdog", "parent watchdog"),
])
def test_the_watchdogs_leave_through_exit_now(name, needle):
    """``os._exit`` would skip the end line; every watchdog exit states its reason."""
    import inspect
    from backend.api import main as api_main
    src = inspect.getsource(getattr(api_main, name))
    assert "os._exit(" not in src
    assert "lifecycle.exit_now(0" in src
    assert needle in src


def test_the_shutdown_endpoint_leaves_through_exit_now():
    import inspect
    from backend.api import main as api_main
    src = inspect.getsource(api_main.shutdown)
    assert "lifecycle.exit_now(" in src


def test_requests_are_noted_and_released_by_the_middleware():
    from fastapi.testclient import TestClient
    from backend import lifecycle
    from backend.api import main as api_main
    TestClient(api_main.app).get("/api/system/version")
    s = lifecycle.state()
    assert s["last_request"]["path"] == "/api/system/version"
    assert all(r["path"] != "/api/system/version" for r in s["in_flight"])


def test_exit_now_ends_the_process_even_when_logging_hangs(tmp_path):
    """A handler that never returns (a stalled pipe) must not keep a watchdog from exiting."""
    r = _child(tmp_path, """
        import threading, time
        class Stuck(logging.Handler):
            def emit(self, record):
                time.sleep(3600)
        logging.getLogger().addHandler(Stuck())
        t0 = time.monotonic()
        lifecycle.exit_now(5, "parent watchdog: test")
    """)
    assert r.returncode == 5


def test_the_loop_does_not_overwrite_the_end_in_the_state_file(tmp_path):
    state_file = tmp_path / "backend-state.json"
    r = _child(tmp_path, """
        lifecycle.note_reason("orderly server shutdown")
        lifecycle._log_end()
        lifecycle.write_state()          # the 5-s loop's next tick, after the end
    """, env_extra={"ORIENTA_BACKEND_STATE_FILE": str(state_file)})
    assert r.returncode == 0
    assert json.loads(state_file.read_text(encoding="utf-8"))["ended"] == "orderly server shutdown"
