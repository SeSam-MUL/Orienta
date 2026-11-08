"""Many sphere requests at once must cost the memory of ONE.

Measured on the CPU backend (see the comment in sht_pattern_renderer): six
simultaneous High requests took the process from 3 GB to 13.7 GB, three Max
requests to 28.5 GB. The user sees no error, the machine just stops. These
tests pin the queue that prevents it, with the expensive step replaced by a
stand-in that counts how many run at the same time.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import sht_sphere  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_lock(monkeypatch):
    """A broken lock (the thing some of these tests provoke on purpose) must not
    leak into the next test and turn one failure into a hang."""
    monkeypatch.setattr(sht_sphere, "_COMPUTE_LOCK", threading.Lock())
    # The same for the table of running reconstructions: a flight that one test
    # leaves behind (the very defect some of them provoke on purpose) would make
    # the next test that asks for the same file wait on it for ever.
    monkeypatch.setattr(sht_sphere, "_INFLIGHT", {})


def _run(coro, timeout=30):
    """Run a coroutine to completion, but fail the test instead of hanging it."""
    return asyncio.run(asyncio.wait_for(coro, timeout))


class _FakeRequest:
    """Stands in for starlette's Request: ``receive()`` delivers the disconnect
    after ``after`` seconds (never, by default). The route listens on
    ``receive()``; polling ``is_disconnected()`` does not work behind the app's
    middleware (see TestRealServer)."""

    def __init__(self, after=None):
        self._after = after

    async def receive(self):
        if self._after is None:
            await asyncio.Event().wait()                 # nothing ever arrives
        await asyncio.sleep(self._after)
        return {"type": "http.disconnect"}


class _Probe:
    """A stand-in for ``_build_payload`` that records its own concurrency."""

    def __init__(self, hold=0.15, fail_for=()):
        self.hold = hold
        self.fail_for = set(fail_for)
        self.active = 0
        self.max_active = 0
        self.calls = []
        self._lock = threading.Lock()

    def __call__(self, resolved_path, max_bandwidth, target_size):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append((resolved_path, max_bandwidth, target_size))
        try:
            time.sleep(self.hold)
            if resolved_path in self.fail_for:
                raise RuntimeError("boom")
            return {"filename": resolved_path, "bw": max_bandwidth, "tgt": target_size}
        finally:
            with self._lock:
                self.active -= 1


@pytest.fixture
def probe(monkeypatch, tmp_path):
    """Each file name resolves to itself; the build is the probe."""
    sht_sphere.clear_cache()
    monkeypatch.setattr(sht_sphere, "_resolve_sht", lambda name: Path(name))
    monkeypatch.setattr(sht_sphere, "_cheap_path", lambda name: Path(name))
    p = _Probe()
    monkeypatch.setattr(sht_sphere, "_build_payload", p)
    yield p
    sht_sphere.clear_cache()


def _run_all(fn_args):
    """Start one thread per call, return (results, errors) by index."""
    results, errors = {}, {}

    def go(i, args):
        try:
            results[i] = sht_sphere.sphere_from_sht(*args[0], **args[1])
        except BaseException as e:  # noqa: BLE001
            errors[i] = e

    threads = [threading.Thread(target=go, args=(i, a), daemon=True) for i, a in enumerate(fn_args)]
    [t.start() for t in threads]
    [t.join(timeout=30) for t in threads]
    return results, errors


class TestSerialised:
    def test_six_different_requests_never_run_together(self, probe):
        calls = [((f"file{i}.sht",), {"max_bandwidth": 256, "target_size": 192}) for i in range(6)]
        results, errors = _run_all(calls)
        assert not errors
        assert len(results) == 6
        assert probe.max_active == 1, f"{probe.max_active} reconstructions ran at the same time"
        assert len(probe.calls) == 6

    def test_a_failing_job_does_not_keep_the_queue_closed(self, monkeypatch):
        sht_sphere.clear_cache()
        monkeypatch.setattr(sht_sphere, "_resolve_sht", lambda name: Path(name))
        monkeypatch.setattr(sht_sphere, "_cheap_path", lambda name: Path(name))
        p = _Probe(hold=0.05, fail_for={"bad.sht"})
        monkeypatch.setattr(sht_sphere, "_build_payload", p)
        with pytest.raises(RuntimeError, match="boom"):
            sht_sphere.sphere_from_sht("bad.sht", max_bandwidth=256, target_size=192)
        # The lock was released: the next job runs instead of waiting forever.
        done = []
        t = threading.Thread(
            target=lambda: done.append(sht_sphere.sphere_from_sht("good.sht", max_bandwidth=256, target_size=192)),
            daemon=True)      # a deadlock must fail the test, not hang the interpreter at exit
        t.start()
        t.join(timeout=5)
        assert not t.is_alive(), "the queue stayed closed after a failed job"
        assert done and done[0]["filename"] == "good.sht"

    def test_a_cached_sphere_does_not_wait_for_a_running_job(self, monkeypatch):
        sht_sphere.clear_cache()
        monkeypatch.setattr(sht_sphere, "_resolve_sht", lambda name: Path(name))
        monkeypatch.setattr(sht_sphere, "_cheap_path", lambda name: Path(name))
        fast = _Probe(hold=0.0)
        monkeypatch.setattr(sht_sphere, "_build_payload", fast)
        sht_sphere.sphere_from_sht("cached.sht", max_bandwidth=128, target_size=128)

        slow = _Probe(hold=1.5)
        monkeypatch.setattr(sht_sphere, "_build_payload", slow)
        runner = threading.Thread(
            target=lambda: sht_sphere.sphere_from_sht("slow.sht", max_bandwidth=384, target_size=256),
            daemon=True)
        runner.start()
        time.sleep(0.2)                                  # the slow job holds the lock
        t0 = time.time()
        hit = sht_sphere.sphere_from_sht("cached.sht", max_bandwidth=128, target_size=128)
        waited = time.time() - t0
        runner.join(timeout=10)
        assert hit["filename"] == "cached.sht"
        assert waited < 0.5, f"a cache hit waited {waited:.2f}s behind a running job"

    def test_the_queue_does_not_block_the_event_loop(self, probe, monkeypatch):
        """The route runs the queue in a worker thread: while a job runs and
        others wait, the loop keeps serving (health polls, other routes)."""
        from backend.api.routes import database

        probe.hold = 0.6
        gaps = []

        async def ticker(stop):
            last = time.perf_counter()
            while not stop.is_set():
                await asyncio.sleep(0.02)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        async def main():
            stop = asyncio.Event()
            tick = asyncio.create_task(ticker(stop))
            jobs = [database.sphere_file(_FakeRequest(), f"f{i}.sht", 256, 192) for i in range(3)]
            await asyncio.gather(*jobs)
            stop.set()
            await tick

        _run(main())
        assert max(gaps) < 0.25, f"the event loop stalled for {max(gaps):.2f}s"
        assert probe.max_active == 1


class TestCoalesced:
    def test_identical_requests_share_one_reconstruction(self, probe, monkeypatch):
        probe.hold = 0.4
        same = ((("same.sht",), {"max_bandwidth": 256, "target_size": 192}),) * 5
        looks = []
        real_get = sht_sphere._cache_get
        monkeypatch.setattr(sht_sphere, "_cache_get", lambda k: (looks.append(1), real_get(k))[1])
        results, errors = _run_all(list(same))
        assert not errors
        assert len(probe.calls) == 1, f"{len(probe.calls)} reconstructions for one sphere"
        assert len({id(r) for r in results.values()}) == 1      # literally the one result
        # Waiters sleep on the leader's event; a polling loop would look at the
        # cache thousands of times in 0.4 s.
        assert len(looks) < 50, f"{len(looks)} cache lookups: the waiters are spinning"

    def test_a_different_resolution_of_the_same_file_is_a_different_job(self, probe):
        calls = [((("a.sht",), {"max_bandwidth": 256, "target_size": 192})),
                 ((("a.sht",), {"max_bandwidth": 384, "target_size": 256}))]
        results, errors = _run_all(calls)
        assert not errors and len(probe.calls) == 2

    def test_only_the_target_size_differs(self, probe):
        """Two requests for the same file and bandwidth that differ ONLY in
        target_size are two spheres. A key without it handed one client the
        other's grid (in a stress run, 7 of 12 clients got the wrong one)."""
        probe.hold = 0.4
        results, errors = _run_all([
            (("a.sht",), {"max_bandwidth": 256, "target_size": 192}),
            (("a.sht",), {"max_bandwidth": 256, "target_size": 256}),
        ])
        assert not errors
        assert len(probe.calls) == 2, probe.calls
        assert (results[0]["tgt"], results[1]["tgt"]) == (192, 256)
        # and the cache keeps them apart as well
        # Asked twice in a row: a lookup that matched on the other fields only
        # would get the first answer right by luck of the ordering and the second
        # one wrong, because every hit moves its entry to the end.
        for want in (192, 192, 256, 256):
            assert sht_sphere.sphere_from_sht("a.sht", max_bandwidth=256, target_size=want)["tgt"] == want
        assert len(probe.calls) == 2

    def test_only_the_bandwidth_differs(self, probe):
        probe.hold = 0.4
        results, errors = _run_all([
            (("a.sht",), {"max_bandwidth": 256, "target_size": 192}),
            (("a.sht",), {"max_bandwidth": 384, "target_size": 192}),
        ])
        assert not errors
        assert len(probe.calls) == 2, probe.calls
        assert (results[0]["bw"], results[1]["bw"]) == (256, 384)
        for want in (256, 256, 384, 384):                         # twice in a row, see above
            assert sht_sphere.sphere_from_sht("a.sht", max_bandwidth=want, target_size=192)["bw"] == want
        assert len(probe.calls) == 2

    def test_only_the_file_differs(self, probe):
        probe.hold = 0.4
        results, errors = _run_all([
            (("a.sht",), {"max_bandwidth": 256, "target_size": 192}),
            (("b.sht",), {"max_bandwidth": 256, "target_size": 192}),
        ])
        assert not errors and len(probe.calls) == 2
        assert (results[0]["filename"], results[1]["filename"]) == ("a.sht", "b.sht")

    def test_a_failure_reaches_every_waiter_and_is_not_cached(self, monkeypatch):
        sht_sphere.clear_cache()
        monkeypatch.setattr(sht_sphere, "_resolve_sht", lambda name: Path(name))
        monkeypatch.setattr(sht_sphere, "_cheap_path", lambda name: Path(name))
        p = _Probe(hold=0.3, fail_for={"bad.sht"})
        monkeypatch.setattr(sht_sphere, "_build_payload", p)
        calls = [(("bad.sht",), {"max_bandwidth": 256, "target_size": 192})] * 4
        results, errors = _run_all(calls)
        assert not results and len(errors) == 4
        assert all("boom" in str(e) for e in errors.values())
        assert len(p.calls) == 1
        # nothing wrong was remembered: the next request tries again
        with pytest.raises(RuntimeError, match="boom"):
            sht_sphere.sphere_from_sht("bad.sht", max_bandwidth=256, target_size=192)
        assert len(p.calls) == 2

    def test_the_flight_is_forgotten_afterwards(self, probe):
        sht_sphere.sphere_from_sht("x.sht", max_bandwidth=256, target_size=192)
        assert sht_sphere._INFLIGHT == {}


class TestDroppedWhenTheClientIsGone:
    def _start(self, key, cancelled, out):
        def go():
            try:
                out[key] = sht_sphere.sphere_from_sht(
                    key, max_bandwidth=256, target_size=192, cancelled=cancelled)
            except BaseException as e:  # noqa: BLE001
                out[key] = e
        t = threading.Thread(target=go, daemon=True)
        t.start()
        return t

    def test_a_queued_request_whose_client_left_never_starts(self, probe):
        probe.hold = 3.0
        out = {}
        running = self._start("running.sht", lambda: False, out)
        time.sleep(0.2)                                   # it holds the queue now
        left = threading.Event()
        queued = self._start("queued.sht", left.is_set, out)
        time.sleep(0.2)
        left.set()                                        # the user picked something else
        queued.join(timeout=1.0)
        # It stops waiting at once -- it does not sit in the queue (holding a
        # worker thread) until the running job is done.
        assert not queued.is_alive(), "a cancelled request stayed queued behind the running job"
        assert isinstance(out["queued.sht"], sht_sphere.SphereCancelled)
        running.join(timeout=6)
        assert [c[0] for c in probe.calls] == ["running.sht"], probe.calls

    def test_a_request_cancelled_before_it_starts_does_not_even_take_a_free_queue(self, probe):
        with pytest.raises(sht_sphere.SphereCancelled):
            sht_sphere.sphere_from_sht("x.sht", max_bandwidth=256, target_size=192, cancelled=lambda: True)
        assert probe.calls == []

    def test_a_leader_that_left_does_not_take_a_waiting_client_down_with_it(self, probe):
        probe.hold = 1.5
        out = {}
        blocker = self._start("blocker.sht", lambda: False, out)
        time.sleep(0.2)
        left = threading.Event()
        leader = self._start("same.sht", left.is_set, out)        # queued behind the blocker
        time.sleep(0.2)
        follower_out = {}
        def follower():
            try:
                follower_out["r"] = sht_sphere.sphere_from_sht(
                    "same.sht", max_bandwidth=256, target_size=192, cancelled=lambda: False)
            except BaseException as e:  # noqa: BLE001
                follower_out["r"] = e
        f = threading.Thread(target=follower, daemon=True)
        f.start()
        time.sleep(0.2)
        left.set()                                        # only the leader's client is gone
        f.join(timeout=10)
        leader.join(timeout=10)
        blocker.join(timeout=10)
        assert isinstance(out["same.sht"], sht_sphere.SphereCancelled)
        assert follower_out["r"]["filename"] == "same.sht", follower_out
        assert [c[0] for c in probe.calls].count("same.sht") == 1

    def test_a_waiter_that_left_stops_waiting(self, probe):
        probe.hold = 1.0
        out = {}
        lead = self._start("same.sht", lambda: False, out)
        time.sleep(0.2)
        left = threading.Event()
        waiter = self._start("same.sht#2", left.is_set, out)     # a different key: queued
        time.sleep(0.2)
        left.set()
        waiter.join(timeout=3)
        assert not waiter.is_alive(), "a cancelled waiter kept waiting for the queue"
        lead.join(timeout=5)

    def test_an_identical_request_whose_client_left_stops_waiting_for_the_leader(self, probe):
        probe.hold = 1.5
        out = {}
        lead = self._start("same.sht", lambda: False, out)
        time.sleep(0.2)
        left = threading.Event()
        t0 = time.time()
        follower = self._start("same.sht", left.is_set, {})       # same key: it waits on the leader
        time.sleep(0.2)
        left.set()
        follower.join(timeout=1.0)
        assert not follower.is_alive(), "a departed follower kept waiting for the leader"
        assert time.time() - t0 < 1.2
        lead.join(timeout=5)
        assert out["same.sht"]["filename"] == "same.sht"            # the leader was not disturbed

    def test_the_watcher_ends_with_the_request(self, probe):
        """The task that listens for the disconnect is cancelled when the route
        returns; left running it would wait on receive() for as long as the
        connection lives, one leaked task per request."""
        from backend.api.routes import database

        async def main():
            await database.sphere_file(_FakeRequest(), "x.sht", 256, 192)   # the fake never disconnects
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            return [t.get_coro().__qualname__ for t in asyncio.all_tasks()
                    if t is not asyncio.current_task() and not t.done()]

        leftovers = _run(main())
        assert not [n for n in leftovers if "_watch_client" in n], leftovers

    def test_the_route_answers_499_and_drops_the_job_when_the_client_disconnects(self, probe):
        from backend.api.routes import database

        probe.hold = 1.0

        async def main():
            running = asyncio.create_task(database.sphere_file(_FakeRequest(), "running.sht", 256, 192))
            await asyncio.sleep(0.2)
            queued = asyncio.create_task(database.sphere_file(_FakeRequest(after=0.3), "queued.sht", 256, 192))
            return await running, await queued

        first, second = _run(main())
        assert first["filename"] == "running.sht"
        assert second.status_code == 499
        assert [c[0] for c in probe.calls] == ["running.sht"], probe.calls


CRLF = chr(13) + chr(10)      # the wire format of HTTP/1.1 lines


class TestRealServer:
    """The disconnect must be seen behind the REAL middleware chain.

    The unit tests above feed the route a fake request, so they cannot tell
    whether the real one ever reports the disconnect. It did not:
    ``request.is_disconnected()`` stays False behind ``HTTPTimingMiddleware``
    (a ``BaseHTTPMiddleware``) in Starlette 1.0, and a client that had closed its
    socket was still served. Here the real app runs under real uvicorn and a
    raw socket really closes.
    """

    @staticmethod
    def _free_port():
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    @pytest.fixture
    def server(self):
        import uvicorn

        from backend.api.main import app

        port = self._free_port()
        srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
        t = threading.Thread(target=srv.run, daemon=True)
        t.start()
        for _ in range(100):
            if srv.started:
                break
            time.sleep(0.1)
        assert srv.started, "uvicorn did not start"
        yield port
        srv.should_exit = True
        t.join(timeout=10)

    @staticmethod
    def _get(port, name):
        import socket
        c = socket.create_connection(("127.0.0.1", port), timeout=30)
        request = (
            f"GET /api/database/sphere/{name}?max_bandwidth=256&target_size=192 HTTP/1.1" + CRLF
            + f"Host: 127.0.0.1:{port}" + CRLF + "Connection: close" + CRLF + CRLF
        )
        c.sendall(request.encode())
        return c

    def test_a_client_that_closes_its_socket_is_dropped_by_the_real_app(self, probe, server, monkeypatch):
        probe.hold = 2.0
        seen = []
        real = sht_sphere.sphere_from_sht

        def spy(filename, **kw):
            seen.append(Path(filename).name)
            return real(filename, **kw)

        monkeypatch.setattr(sht_sphere, "sphere_from_sht", spy)

        first = self._get(server, "running.sht")
        time.sleep(0.4)                                   # it holds the queue
        second = self._get(server, "queued.sht")
        time.sleep(0.6)
        assert seen == ["running.sht", "queued.sht"], f"the second request never reached the route: {seen}"
        second.close()                                    # the user moved on

        body = b""
        first.settimeout(30)
        while True:
            chunk = first.recv(65536)
            if not chunk:
                break
            body += chunk
        first.close()
        time.sleep(1.0)                                   # time for a wrongly kept job to start
        assert body.startswith(b"HTTP/1.1 200"), body[:80]
        assert [c[0] for c in probe.calls] == ["running.sht"], (
            f"the abandoned request was computed anyway: {[c[0] for c in probe.calls]}")

    def test_the_real_app_answers_a_cache_hit_at_once_while_the_queue_is_full(self, probe, server):
        """Measured on the real app before the peek: a warm hit came back after
        4.0 s, behind four sphere jobs, instead of after 30 ms."""
        sht_sphere.sphere_from_sht("warm.sht", max_bandwidth=256, target_size=192)   # cached
        probe.hold = 1.5
        busy = [self._get(server, f"busy{i}.sht") for i in range(6)]
        time.sleep(0.5)                                   # four threads taken, the rest queued
        t0 = time.time()
        c = self._get(server, "warm.sht")
        c.settimeout(10)
        body = b""
        while True:
            chunk = c.recv(65536)
            if not chunk:
                break
            body += chunk
        waited = time.time() - t0
        c.close()
        for b in busy:
            b.close()                                     # drop the busy jobs, nobody needs them
        assert body.startswith(b"HTTP/1.1 200"), body[:80]
        assert b"warm.sht" in body
        assert waited < 0.5, f"the real app made a cache hit wait {waited:.2f}s behind the queue"


class TestOwnExecutor:
    """Waiting sphere requests must not take the threads other routes use."""

    def test_a_queue_of_sphere_jobs_does_not_starve_the_default_executor(self, probe):
        from concurrent.futures import ThreadPoolExecutor

        from backend.api.routes import database

        probe.hold = 0.7

        async def main():
            # A small default executor makes the effect deterministic: with the
            # sphere on it, six waiting requests would hold every thread.
            asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=2))
            jobs = [asyncio.create_task(database.sphere_file(_FakeRequest(), f"s{i}.sht", 256, 192))
                    for i in range(6)]
            await asyncio.sleep(0.2)                   # all six are in, one builds, the rest wait
            t0 = time.perf_counter()
            await asyncio.to_thread(lambda: None)      # what any other route does
            other_route_latency = time.perf_counter() - t0
            await asyncio.gather(*jobs)
            return other_route_latency

        latency = _run(main())
        assert latency < 0.3, f"an unrelated route waited {latency:.2f}s behind the sphere queue"

    def test_the_sphere_never_uses_more_threads_than_its_executor_has(self, probe, monkeypatch):
        from backend.api.routes import database

        inside = {"now": 0, "max": 0}
        guard = threading.Lock()
        real = sht_sphere.sphere_from_sht

        def counting(filename, **kw):
            with guard:
                inside["now"] += 1
                inside["max"] = max(inside["max"], inside["now"])
            try:
                return real(filename, **kw)
            finally:
                with guard:
                    inside["now"] -= 1

        monkeypatch.setattr(sht_sphere, "sphere_from_sht", counting)
        probe.hold = 0.2

        async def main():
            await asyncio.gather(*[database.sphere_file(_FakeRequest(), f"t{i}.sht", 256, 192) for i in range(10)])

        _run(main())
        assert inside["max"] <= sht_sphere.SPHERE_EXECUTOR._max_workers == 4, inside
        assert inside["max"] >= 2, "the bound was never approached, the test proves nothing"


class TestHitsSkipTheQueue:
    def test_a_finished_sphere_is_answered_while_the_queue_is_full(self, probe, server=None):
        """Through the route, with the queue busy: the hit comes back at once."""
        from backend.api.routes import database

        sht_sphere.sphere_from_sht("warm.sht", max_bandwidth=256, target_size=192)   # now cached
        probe.hold = 1.0

        async def main():
            busy = [asyncio.create_task(database.sphere_file(_FakeRequest(), f"busy{i}.sht", 256, 192))
                    for i in range(6)]
            await asyncio.sleep(0.3)                      # four threads taken, queue held
            t0 = time.perf_counter()
            hit = await database.sphere_file(_FakeRequest(), "warm.sht", 256, 192)
            waited = time.perf_counter() - t0
            await asyncio.gather(*busy)
            return hit, waited

        hit, waited = _run(main())
        assert hit["filename"] == "warm.sht"
        assert waited < 0.3, f"a cache hit waited {waited:.2f}s behind the queue"

    def test_cached_sphere_never_computes(self, probe):
        assert sht_sphere.cached_sphere("never.sht", 256, 192) is None
        assert probe.calls == []

    def test_cached_sphere_respects_the_resolution(self, probe):
        sht_sphere.sphere_from_sht("a.sht", max_bandwidth=256, target_size=192)
        assert sht_sphere.cached_sphere("a.sht", 256, 192) is not None
        assert sht_sphere.cached_sphere("a.sht", 384, 192) is None
        assert sht_sphere.cached_sphere("a.sht", 256, 256) is None

    def test_a_deleted_file_is_not_served_from_the_cache(self, monkeypatch, tmp_path):
        sht_sphere.clear_cache()
        f = tmp_path / "gone.sht"
        f.write_bytes(b"")
        monkeypatch.setattr(sht_sphere, "_SHT_DB", tmp_path)
        monkeypatch.setattr(sht_sphere, "_RESOLVED", {})
        monkeypatch.setattr(sht_sphere, "_build_payload", _Probe(hold=0.0))
        sht_sphere.sphere_from_sht("gone.sht", max_bandwidth=256, target_size=192)
        assert sht_sphere.cached_sphere("gone.sht", 256, 192) is not None
        f.unlink()
        assert sht_sphere.cached_sphere("gone.sht", 256, 192) is None
        sht_sphere.clear_cache()
