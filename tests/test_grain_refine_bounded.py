"""apply-to-grain with refine: bounded in memory, off the event loop, and safe against a
concurrent edit.

Measured on a real file: the refine of an 8x8 grain
held the event loop for 45 s, ran out of GPU memory and left the backend at 32-42 GiB. The
cause: ``newton_refine`` ran at bandwidth 128, where autograd keeps about
245 B * L**4 (62 GiB) for the Hessian.
"""
import asyncio
import time

import numpy as np
import pytest
from fastapi import HTTPException

from backend.api.routes import indexing as route
from tests.test_grain_flip import _fake_nests_result

GiB = 2**30


class _Cuda:
    def __init__(self, free):
        self.free = free

    def is_available(self):
        return True

    def mem_get_info(self):
        return (self.free, 12 * GiB)


@pytest.mark.parametrize("free_gib, band", [(10.8, (64, 80)), (4.0, (48, 64)), (0.6, (28, 40))])
def test_the_newton_bandwidth_follows_the_free_gpu_memory(monkeypatch, free_gib, band):
    import torch
    monkeypatch.setattr(torch, "cuda", _Cuda(int(free_gib * GiB)))
    L, free, device = route._newton_bandwidth(128)
    assert device == "GPU" and band[0] <= L <= band[1]
    # the largest that fits the share of the free memory it may plan for
    budget = route.NEWTON_MEMORY_SHARE * free
    assert route.NEWTON_BYTES_PER_L4 * L ** 4 <= budget < route.NEWTON_BYTES_PER_L4 * (L + 1) ** 4


def test_the_requested_bandwidth_is_an_upper_bound(monkeypatch):
    import torch
    monkeypatch.setattr(torch, "cuda", _Cuda(200 * GiB))
    assert route._newton_bandwidth(68)[0] == 68


def test_a_newton_step_at_bandwidth_128_does_not_fit_a_12_gib_card():
    """The arithmetic of the old default: why every refine on a real file ran out of memory."""
    assert route.NEWTON_BYTES_PER_L4 * 128 ** 4 > 12 * GiB


@pytest.fixture
def nests(monkeypatch):
    result, q_target, q_wrong, nests = _fake_nests_result()
    monkeypatch.setattr(route, "_get_result", lambda *a, **k: result)
    monkeypatch.setattr(route, "_free_interactive_gpu_caches", lambda: None)
    return result, [float(v) for v in q_target]


def test_the_refine_does_not_hold_the_event_loop(monkeypatch, nests):
    result, quat = nests

    def slow_refine(*a, **k):
        time.sleep(1.0)
        return {}, {"status": "rejected", "reason": "guards"}
    monkeypatch.setattr(route, "_grain_newton_refine", slow_refine)

    async def scenario():
        task = asyncio.ensure_future(route.apply_variant_to_grain(
            route.GrainApplyRequest(row=0, col=0, quat=quat, refine=True)))
        ticks = []
        t0 = time.monotonic()
        while not task.done():
            await asyncio.sleep(0.05)
            ticks.append(time.monotonic() - t0)
        return await task, ticks

    out, ticks = asyncio.run(scenario())
    assert out["n_changed"] == 4
    gaps = np.diff([0.0] + ticks)
    assert gaps.max() < 0.5, f"the loop was held for {gaps.max():.2f} s during the refine"


def test_an_edit_during_the_refine_is_not_overwritten(monkeypatch, nests):
    """Apply A refines; apply B (no refine) changes another grain meanwhile. A must answer 409
    and write nothing, so B's edit stays."""
    result, quat = nests
    entered = asyncio.Event()
    go = None

    def refine_waiting_for_b(*a, **k):
        loop.call_soon_threadsafe(entered.set)
        go.wait(5)
        return {}, {"status": "rejected", "reason": "guards"}
    monkeypatch.setattr(route, "_grain_newton_refine", refine_waiting_for_b)

    async def scenario():
        nonlocal go
        import threading
        go = threading.Event()
        a = asyncio.ensure_future(route.apply_variant_to_grain(
            route.GrainApplyRequest(row=0, col=0, quat=quat, refine=True)))
        await asyncio.wait_for(entered.wait(), 5)
        out_b = await route.apply_variant_to_grain(
            route.GrainApplyRequest(row=6, col=0, quat=quat, refine=False))
        go.set()
        with pytest.raises(HTTPException) as err:
            await a
        return out_b, err.value

    loop = asyncio.new_event_loop()
    try:
        out_b, exc = loop.run_until_complete(scenario())
    finally:
        loop.close()
    assert out_b["n_changed"] == 8
    assert exc.status_code == 409 and "changed while the grain was being refined" in exc.detail
    # B's undo is the one stored: A wrote nothing
    assert sorted(result.metadata["grain_flip_undo"]["xmap_indices"]) != [0, 1, 8, 9]


def test_a_second_refine_while_one_runs_is_refused_at_once(monkeypatch, nests):
    """Each refine plans from the same free GPU memory; two at once would book it twice."""
    import threading
    result, quat = nests
    started, go = threading.Event(), threading.Event()

    def refine(*a, **k):
        started.set()
        go.wait(5)
        return {}, {"status": "rejected", "reason": "guards"}
    monkeypatch.setattr(route, "_grain_newton_refine", refine)

    async def scenario():
        a = asyncio.ensure_future(route.apply_variant_to_grain(
            route.GrainApplyRequest(row=0, col=0, quat=quat, refine=True)))
        while not started.is_set():
            await asyncio.sleep(0.01)
        t = time.monotonic()
        with pytest.raises(HTTPException) as err:
            await route.apply_variant_to_grain(
                route.GrainApplyRequest(row=6, col=0, quat=quat, refine=True))
        refused_after = time.monotonic() - t
        go.set()
        return await a, err.value, refused_after

    out_a, exc, refused_after = asyncio.run(scenario())
    assert exc.status_code == 409 and "Another grain is being refined" in exc.detail
    assert refused_after < 0.5
    assert out_a["n_changed"] == 4
    assert not route._refine_lock.locked()
