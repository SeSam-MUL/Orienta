"""The dev-panel log handler must not be able to feed itself.

Measured on GitHub's Linux runner (run 35982374292): the Python suite died
after 18 minutes on a 7.9 GB machine, 280,752 identical warnings in a 62 MB
log. Every link was production code:

    run_coroutine_threadsafe(manager.broadcast(payload), loop)

builds the coroutine as an ARGUMENT, so it exists before the call. On a closed
loop the call raises, `except RuntimeError: pass` swallowed it, and the
coroutine was dropped un-awaited. Python warns; logging.captureWarnings turns
the warning into a record on `py.warnings`; the record reaches the root logger
and therefore this same handler, which builds another coroutine against the
same dead loop. Nothing stopped it but the kernel.

Reproduced on Windows before the fix with ONE log record: 3001 records at the
cap, all of them `py.warnings`. It is not a Linux problem and not a CI problem
— Windows has more memory, so it reads as "the tests are slow".

A cycle needs more than one brake, so these tests pin all three independently:
the coroutine is closed, `py.warnings` cannot re-enter the handler, and the
handler does not outlive its loop.
"""

import asyncio
import gc
import logging
import warnings

import pytest

from backend.api import log_broadcast
from backend.api.log_broadcast import (
    WebSocketLogHandler,
    install_ws_log_handler,
    timed_step,
    uninstall_ws_log_handler,
)


class FakeWSManager:
    """Stands in for ConnectionManager; broadcast is a real coroutine."""

    def __init__(self):
        self.sent = []

    async def broadcast(self, payload):
        self.sent.append(payload)


@pytest.fixture
def clean_root():
    """Give each test the root logger back exactly as it was.

    These tests attach handlers, change the level and toggle
    captureWarnings — all global state shared with every other test in the
    process.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_globals = (log_broadcast._ws_manager, log_broadcast._loop)
    try:
        yield root
    finally:
        # False is this suite's resting state, not a restore of a saved value
        # — same as tests/test_file_log.py does. conftest opts the file log out.
        logging.captureWarnings(False)
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
        log_broadcast._ws_manager, log_broadcast._loop = saved_globals


class Counter(logging.Handler):
    """Counts records and breaks any runaway so the test can still report."""

    def __init__(self, cap=3000):
        super().__init__()
        self.cap = cap
        self.total = 0
        self.by_logger = {}

    def emit(self, record):
        self.total += 1
        self.by_logger[record.name] = self.by_logger.get(record.name, 0) + 1
        if self.total >= self.cap:
            root = logging.getLogger()
            for h in list(root.handlers):
                if isinstance(h, WebSocketLogHandler):
                    root.removeHandler(h)


def _install_against_a_closed_loop(manager):
    """Install the handler, then close the loop it points at.

    This is what the lifespan used to leave behind: the loop of a finished app
    is gone, the handler is still on the root logger.
    """
    # Deliberately NOT asyncio.set_event_loop(): run_until_complete does not
    # need the loop to be current and install_ws_log_handler uses
    # get_running_loop(). Setting it left the thread without a current loop
    # afterwards, which a later test calling asyncio.get_event_loop() outside
    # a running loop would have paid for (indexing.py, batch_v2.py do).
    loop = asyncio.new_event_loop()
    loop.run_until_complete(_install(manager))
    loop.close()
    return loop


async def _install(manager):
    install_ws_log_handler(manager)


class TestTheCycleIsBroken:
    def test_a_thousand_records_stay_a_thousand(self, clean_root):
        """The whole point: no amplification.

        Before the fix a single record produced thousands. The assertion is
        equality, not a threshold, because any amplification at all is the
        bug — the growth rate only decides how long the machine survives.
        """
        _install_against_a_closed_loop(FakeWSManager())
        logging.captureWarnings(True)          # file_log.py does this
        counter = Counter()
        clean_root.addHandler(counter)
        clean_root.setLevel(logging.DEBUG)

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            log = logging.getLogger("demo.runaway")
            for i in range(1000):
                log.info("record %d", i)
            gc.collect()                        # coroutines warn when collected

        assert counter.total == 1000, (
            f"{counter.total} records reached the root logger for 1000 emitted; "
            f"by logger: {counter.by_logger}"
        )
        assert "py.warnings" not in counter.by_logger

    def test_a_dropped_broadcast_never_warns(self, clean_root):
        """The coroutine must be closed, not abandoned."""
        _install_against_a_closed_loop(FakeWSManager())
        clean_root.setLevel(logging.DEBUG)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            logging.getLogger("demo.warn").info("this cannot be delivered")
            gc.collect()

        never_awaited = [w for w in caught if "never awaited" in str(w.message)]
        assert not never_awaited, [str(w.message) for w in never_awaited]

    def test_the_timing_path_does_not_warn_either(self, clean_root):
        """timed_step has the same construct; it was not in the report."""
        _install_against_a_closed_loop(FakeWSManager())
        clean_root.setLevel(logging.DEBUG)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with timed_step("a step nobody will see"):
                pass
            gc.collect()

        assert not [w for w in caught if "never awaited" in str(w.message)]

    def test_the_globals_are_read_once(self, clean_root):
        """Shutdown clears `_loop` from the loop thread while other threads are
        inside _schedule (prewarm, reaper, thread excepthook, sim heartbeat).

        Reading the global again at call time meant a thread could pass the
        `is None` guard and then call run_coroutine_threadsafe(coro, None),
        which raises AttributeError — NOT RuntimeError — so the old handler
        missed it and the coroutine escaped un-awaited. Measured.
        """
        class ClearsMidFlight:
            """Disappears exactly the way uninstall makes it disappear."""

            def __init__(self):
                self.calls = 0

            async def broadcast(self, payload):
                pass

            def __call__(self):  # pragma: no cover - not used
                pass

        manager = ClearsMidFlight()

        real_broadcast = manager.broadcast

        def clearing_broadcast(payload):
            log_broadcast._loop = None      # the race, deterministically
            log_broadcast._ws_manager = None
            return real_broadcast(payload)

        manager.broadcast = clearing_broadcast
        _install_against_a_closed_loop(manager)
        clean_root.setLevel(logging.DEBUG)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            logging.getLogger("demo.race").info("cleared while in flight")
            gc.collect()

        assert not [w for w in caught if "never awaited" in str(w.message)], (
            [str(w.message) for w in caught]
        )

    def test_emit_never_raises_into_the_caller(self, clean_root):
        """logging.Handler.handle does not guard emit.

        Without a try/except here, a bad format string comes out of an
        unrelated logger.info() as a TypeError. Every other root handler
        survives the same record via handleError.
        """
        _install_against_a_closed_loop(FakeWSManager())
        clean_root.setLevel(logging.DEBUG)

        logging.raiseExceptions = False      # keep handleError quiet in tests
        try:
            logging.getLogger("demo.badfmt").info("a %d", "not-an-int")
        finally:
            logging.raiseExceptions = True

    def test_py_warnings_cannot_re_enter_the_handler(self, clean_root):
        """Second brake, independent of the first."""
        assert "py.warnings" in log_broadcast._SUPPRESSED_LOGGERS

        manager = FakeWSManager()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_install(manager))
            record = logging.LogRecord(
                "py.warnings", logging.WARNING, __file__, 1,
                "RuntimeWarning: coroutine ... was never awaited", (), None,
            )
            handler = next(h for h in clean_root.handlers
                           if isinstance(h, WebSocketLogHandler))
            handler.emit(record)
            assert manager.sent == []
        finally:
            loop.close()

    def test_the_file_log_still_records_warnings(self):
        """The suppression belongs to the dev panel, not to the log file.

        Warnings are exactly what a bug report needs; dropping them from the
        file would trade one silent failure for another.
        """
        from backend.api import file_log

        noisy = getattr(file_log, "_NOISY_LOGGERS", None) or getattr(
            file_log, "_SUPPRESSED_LOGGERS", frozenset())
        assert "py.warnings" not in noisy


class TestTheHandlerDoesNotOutliveItsLoop:
    def _ours(self, root):
        return [h for h in root.handlers if isinstance(h, WebSocketLogHandler)]

    def test_installing_twice_leaves_one_handler(self, clean_root):
        """A second app in one process must not leave the first one attached."""
        first = _install_against_a_closed_loop(FakeWSManager())
        assert len(self._ours(clean_root)) == 1

        second = FakeWSManager()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_install(second))
            assert len(self._ours(clean_root)) == 1
            assert log_broadcast._ws_manager is second
            assert log_broadcast._loop is loop
            assert log_broadcast._loop is not first
        finally:
            loop.close()

    def test_uninstall_detaches_and_forgets_the_loop(self, clean_root):
        _install_against_a_closed_loop(FakeWSManager())
        assert self._ours(clean_root)

        removed = uninstall_ws_log_handler()

        assert removed == 1
        assert self._ours(clean_root) == []
        assert log_broadcast._loop is None
        assert log_broadcast._ws_manager is None

    def test_uninstall_is_safe_to_call_twice(self, clean_root):
        _install_against_a_closed_loop(FakeWSManager())
        uninstall_ws_log_handler()
        assert uninstall_ws_log_handler() == 0

    def test_emit_is_inert_once_uninstalled(self, clean_root):
        """Second brake for anything still holding a handler reference."""
        manager = FakeWSManager()
        _install_against_a_closed_loop(manager)
        handler = self._ours(clean_root)[0]
        uninstall_ws_log_handler()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            handler.emit(logging.LogRecord(
                "demo", logging.INFO, __file__, 1, "after shutdown", (), None))
            gc.collect()

        assert manager.sent == []
        assert not [w for w in caught if "never awaited" in str(w.message)]

    def test_the_lifespan_removes_it(self):
        """The shutdown half of main.py must call this, not just the startup."""
        import inspect
        from backend.api import main

        source = inspect.getsource(main.lifespan)
        head, _, tail = source.partition("yield")
        assert "install_ws_log_handler(ws_manager)" in head
        assert "uninstall_ws_log_handler()" in tail, (
            "the handler is installed in the lifespan but never removed; it "
            "then outlives the loop it points at"
        )


class TestDeliveryStillWorks:
    def test_a_live_loop_still_receives_the_broadcast(self, clean_root):
        """None of the brakes may cost the feature its actual job."""
        manager = FakeWSManager()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_install(manager))
            logging.getLogger("demo.live").warning("hello panel")
            # run_coroutine_threadsafe queues onto the loop; let it turn once.
            loop.run_until_complete(asyncio.sleep(0.01))
            assert any(p.get("message") == "hello panel" for p in manager.sent), (
                f"nothing delivered; got {manager.sent}"
            )

            # The timing payload is a second, structured channel and had no
            # delivery test at all — only a "does not warn" one.
            manager.sent.clear()
            with timed_step("a step somebody will see"):
                pass
            loop.run_until_complete(asyncio.sleep(0.01))
            timing = [p for p in manager.sent if p.get("category") == "timing"]
            assert len(timing) == 1, f"timing payloads: {manager.sent}"
            assert timing[0]["step"] == "a step somebody will see"
            assert "duration_ms" in timing[0]
        finally:
            uninstall_ws_log_handler()
            loop.close()
