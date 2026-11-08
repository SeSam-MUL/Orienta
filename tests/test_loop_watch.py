"""A held event loop is logged with the place it was held (backend/loop_watch.py)."""
import asyncio
import logging
import time

from backend.loop_watch import LoopWatch


def _blocking_section_of_a_route():
    time.sleep(1.2)          # synchronous work inside an async route


def test_a_held_loop_is_logged_once_with_its_stack(caplog):
    async def main():
        watch = LoopWatch(asyncio.get_running_loop(), threshold_s=0.3, interval_s=0.05).start()
        await asyncio.sleep(0.2)
        _blocking_section_of_a_route()
        await asyncio.sleep(1.3)           # let the watch see the answer and log
        watch.stop()
        return watch
    with caplog.at_level(logging.WARNING, logger="backend.loop_watch"):
        watch = asyncio.run(main())
    assert len(watch.stalls) == 1
    held, snapshot = watch.stalls[0]
    assert 0.9 < held < 2.5
    assert "_blocking_section_of_a_route" in snapshot
    msgs = [r.getMessage() for r in caplog.records if r.name == "backend.loop_watch"]
    assert len(msgs) == 1 and "did not answer for" in msgs[0]


def test_a_healthy_loop_logs_nothing(caplog):
    async def main():
        watch = LoopWatch(asyncio.get_running_loop(), threshold_s=0.3, interval_s=0.05).start()
        for _ in range(20):
            await asyncio.sleep(0.02)
        watch.stop()
        return watch
    with caplog.at_level(logging.WARNING, logger="backend.loop_watch"):
        watch = asyncio.run(main())
    assert watch.stalls == []
    assert not [r for r in caplog.records if r.name == "backend.loop_watch"]


def test_the_backend_watches_its_loop():
    import inspect
    from backend.api import main as api_main
    src = inspect.getsource(api_main.lifespan)
    assert "start_loop_watch()" in src
