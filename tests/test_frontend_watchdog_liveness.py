"""The frontend watchdog must not kill a backend that is being used.

Measured, from the M5 tester's diagnostics bundle
(tasks/mac-test-m5-2026-09-25/, logs/orienta.log):

    09:33:26  Frontend watchdog enabled, grace period = 600s
    09:33:26  Parent watchdog enabled - exiting when PID 3350 goes away
    09:33:27  WebSocket /ws [accepted]      (twice)
              ... the log never records either connection closing ...
    09:43:56  WARNING  No frontend connected - waiting 600s before shutdown
    09:48:58  CIF parsing warnings          <- the user is working
    09:49:18  a simulation starts           <- the user is working
    09:53:04  spherical wigner tensors      <- the user is working
    09:54:00  No frontend reconnected after 600s - shutting down backend.

The window was open the whole time, PID 3350 was alive across all four backend
restarts that morning, and the backend served requests *during* its own
countdown.

What emptied `active_connections` is an INFERENCE, not a measurement: uvicorn
logged "connection open" for every session and "connection closed" for none, and
the /ws handler's own cleanup path logs "WebSocket endpoint raised", which never
appears — so the silent prune in `ConnectionManager.broadcast` (it discards a
connection whose `send` raises) is the only remaining route to an empty set.
That prune now logs its reason, so the next bundle will say rather than imply.
Either way an empty count is not evidence that the app is gone, which is all
these tests depend on.

The shutdown itself is `os._exit(0)`, so these tests exercise the two
predicates that now gate it rather than running the loop.
"""

from __future__ import annotations

import os
import time

import pytest

from backend.api import main as api_main


VARS = ("KIKUCHIPY_PARENT_PID", "KIKUCHIPY_UI_PARENT")


@pytest.fixture(autouse=True)
def _restore():
    saved = {k: os.environ.get(k) for k in VARS}
    saved_clock = api_main._last_http_activity
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        api_main._last_http_activity = saved_clock


def _declare_ui_parent(pid=None):
    os.environ["KIKUCHIPY_UI_PARENT"] = "1"
    os.environ["KIKUCHIPY_PARENT_PID"] = str(os.getpid() if pid is None else pid)


class TestOnlyAWindowOwningParentCounts:
    def test_electron_declares_itself_and_is_believed(self):
        _declare_ui_parent()
        assert api_main._ui_parent_is_alive() is True

    def test_start_app_passes_a_pid_but_is_NOT_believed(self):
        """The regression a review caught, and the reason for the extra flag.

        start_app.py also exports KIKUCHIPY_PARENT_PID (start_app.py:77) but then
        blocks in backend_proc.wait() (:236), so its liveness is implied by the
        backend's own. Trusting it deadlocks the shutdown: launcher waits for
        backend, parent watchdog waits for launcher, frontend watchdog defers to
        the parent watchdog. Close the browser tab and the backend keeps port
        8000, the loaded dataset and its HDF5 handles forever — the 19-hour
        backend of 2026-08-05, and the opposite of what INSTALL.md promises.
        """
        os.environ.pop("KIKUCHIPY_UI_PARENT", None)
        os.environ["KIKUCHIPY_PARENT_PID"] = str(os.getpid())
        assert api_main._ui_parent_is_alive() is False

    def test_a_pid_that_does_not_exist_is_not_alive(self):
        # 2**22 is above every default pid_max on Linux and macOS.
        _declare_ui_parent(2 ** 22)
        assert api_main._ui_parent_is_alive() is False

    def test_no_declared_parent_means_nobody_vouches_for_us(self):
        for k in VARS:
            os.environ.pop(k, None)
        assert api_main._ui_parent_is_alive() is False

    def test_rubbish_does_not_raise(self):
        os.environ["KIKUCHIPY_UI_PARENT"] = "1"
        for bad in ("", "   ", "not-a-pid", "12.5", "-1"):
            os.environ["KIKUCHIPY_PARENT_PID"] = bad
            assert api_main._ui_parent_is_alive() is False

    def test_the_flag_must_be_exactly_one(self):
        os.environ["KIKUCHIPY_PARENT_PID"] = str(os.getpid())
        for bad in ("0", "", "true", "yes", "2"):
            os.environ["KIKUCHIPY_UI_PARENT"] = bad
            assert api_main._ui_parent_is_alive() is False

    def test_electron_actually_sets_the_flag(self):
        """Both halves of the contract, in one place.

        The backend refuses to trust a parent without this flag, so if Electron
        stopped sending it the watchdog would fall back to the HTTP clock — and
        a hidden-but-open window would be killed again.
        """
        from pathlib import Path
        main_js = (Path(__file__).resolve().parents[1]
                   / "electron" / "main.js").read_text(encoding="utf-8")
        assert "KIKUCHIPY_UI_PARENT: '1'" in main_js
        assert "KIKUCHIPY_PARENT_PID: String(process.pid)" in main_js


class TestRequestsCountAsAFrontend:
    def test_no_request_ever_reads_as_infinitely_idle(self):
        api_main._last_http_activity = 0.0
        assert api_main.seconds_since_http_activity() == float("inf")

    def test_a_request_resets_the_clock(self):
        api_main._last_http_activity = 0.0
        api_main.note_http_activity()
        assert api_main.seconds_since_http_activity() < 1.0

    def test_the_clock_is_monotonic_not_wall_time(self):
        """A backwards system-clock correction must not look like activity."""
        api_main.note_http_activity()
        assert api_main._last_http_activity <= time.monotonic()

    def test_an_old_request_does_not_keep_us_alive(self):
        api_main._last_http_activity = time.monotonic() - 3600
        assert api_main.seconds_since_http_activity() > 600


class TestTheMiddlewareRecordsEveryRequest:
    def test_health_is_recorded_even_though_it_is_not_logged(self):
        """The one request a live frontend makes every 30 s.

        `/api/health` sits in the timing middleware's excluded prefixes, so it
        produces no dev-panel entry. If the activity clock were updated after
        that check, the poll of a healthy frontend would count for nothing.
        (Counterfactual, not a cause: no activity clock existed when the 09:54
        shutdown was measured.)
        """
        import inspect

        src = inspect.getsource(api_main.HTTPTimingMiddleware.dispatch)
        before, _, after = src.partition("_EXCLUDED_PREFIXES")
        assert "note_http_activity()" in before, (
            "the activity clock must be updated BEFORE the exclusions:\n" + src)
        assert "note_http_activity()" not in after

    def test_health_really_is_excluded_from_the_log(self):
        # If this ever changes, the reasoning above changes with it.
        assert "/api/health" in api_main.HTTPTimingMiddleware._EXCLUDED_PREFIXES


class TestTheWatchdogAsksBothBeforeExiting:
    """The gate itself. os._exit(0) cannot be run in a test, so this reads the
    source of the loop and pins the order of its checks."""

    def test_both_predicates_gate_the_exit(self):
        import inspect

        src = inspect.getsource(api_main._frontend_watchdog)
        tail = src[src.index("Wait the full grace period"):]
        assert "_ui_parent_is_alive()" in tail
        assert "seconds_since_http_activity()" in tail
        # and the exit comes after both of them
        assert tail.index("_ui_parent_is_alive()") < tail.rindex("os._exit(0)")
        assert tail.index("seconds_since_http_activity()") < tail.rindex("os._exit(0)")

    def test_it_says_why_it_is_not_shutting_down(self):
        """A watchdog that silently declines is as hard to debug as one that
        silently fires; the tester's log is the only reason this was findable."""
        import inspect

        src = inspect.getsource(api_main._frontend_watchdog)
        assert "window-owning parent is alive" in src
        assert "HTTP request arrived" in src
