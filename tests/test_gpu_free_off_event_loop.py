"""The interactive GPU pre-free must not stall the event loop.

``_free_interactive_gpu_caches`` ends the idle compute worker and waits for it
to exit (measured 5 s, limit up to 15 s). Called straight from an ``async def``
route, that wait froze the whole backend: progress and cancel requests
included. The routes now run it in a worker thread.

Each test below replaces ``release_idle_worker`` by something that waits on a
``threading.Event``. The event can only be set by a second coroutine on the SAME
event loop, so if the release ran on the loop the coroutine would never get to
run and the wait would time out; the tests read that time-out as the failure.

The three phase routes also take ``_phase_check_busy``. With an ``await`` in
front of the work, the flag must be taken BEFORE the first ``await`` (else a
second request slips in while the release waits) and the ``try``/``finally``
that clears it must cover the release (else a failing release leaves the flag
stuck and locks the feature out until a restart).
"""
from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest
from fastapi import HTTPException

import backend.api.routes.indexing as route

#: Seconds the fake release waits for the other coroutine. Only reached when the
#: loop is blocked, i.e. when the test is about to fail.
WAIT_S = 3.0


class _Release:
    """A stand-in for ``release_idle_worker`` that blocks until told to go."""

    def __init__(self):
        self.entered = threading.Event()
        self.go = threading.Event()
        self.outcomes: list[bool] = []
        #: "release" when the release starts; the tests add "read" where the
        #: route first touches the result, to read the ORDER of the two off it.
        self.log: list[str] = []

    def __call__(self, timeout=15.0):
        self.log.append("release")
        self.entered.set()
        self.outcomes.append(self.go.wait(WAIT_S))
        return True


@pytest.fixture
def release(monkeypatch):
    import backend.api.services.sht_pattern_renderer as renderer
    import backend.worker.manager as mm
    import torch

    rel = _Release()
    monkeypatch.setattr(mm, "release_idle_worker", rel)
    # The rest of the pre-free is not under test and must not touch a GPU.
    monkeypatch.setattr(renderer, "release_gpu_caches", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(route, "_phase_check_busy", False)
    return rel


class _Stop(RuntimeError):
    """Raised by the faked work, so a route ends right after the release."""


def _island_setup(monkeypatch):
    monkeypatch.setattr(route, "_get_result", lambda *a, **k: object())
    monkeypatch.setattr(route, "_phase_check_ctx", lambda *a, **k: {})

    def stop(*a, **k):
        raise _Stop("island work")

    monkeypatch.setattr(route, "_island_work", stop)
    return lambda: route.island_check(route.IslandCheckRequest())


def _phase_check_setup(monkeypatch):
    import backend.spherical_gpu.pipeline.phase_reassignment as pr
    monkeypatch.setattr(route, "_get_result", lambda *a, **k: object())
    monkeypatch.setattr(route, "_phase_check_ctx", lambda *a, **k: {
        "full_q": None, "phase_full": None, "n_rows": 0, "n_cols": 0,
        "phases": None, "score_fns": None, "hough_quats_fn": None, "topology": None})

    def stop(*a, **k):
        raise _Stop("check_map")

    monkeypatch.setattr(pr, "check_map", stop)
    return lambda: route.phase_check(route.PhaseCheckRequest())


def _assign_setup(monkeypatch):
    monkeypatch.setattr(route, "_get_result", lambda *a, **k: object())

    def stop(*a, **k):
        raise _Stop("assign work")

    monkeypatch.setattr(route, "_grain_phase_assign_blocking", stop)
    return lambda: route.grain_phase_assign(route.GrainPhaseAssignRequest())


FLAG_ROUTES = pytest.mark.parametrize(
    "setup", [_island_setup, _phase_check_setup, _assign_setup],
    ids=["island-check", "phase-check", "grain-phase-assign"])


@FLAG_ROUTES
def test_a_second_run_during_the_release_is_refused_and_the_flag_is_freed(
        monkeypatch, release, setup):
    call = setup(monkeypatch)
    seen = {}

    async def scenario():
        first = asyncio.ensure_future(call())
        for _ in range(300):                     # let the route reach the release
            if release.entered.is_set():
                break
            await asyncio.sleep(0.01)
        # The loop is alive (we are running) and the first run is mid-release.
        seen["entered"] = release.entered.is_set()
        seen["busy_during_release"] = route._phase_check_busy
        try:
            await call()
        except HTTPException as exc:
            seen["second"] = exc.status_code
        release.go.set()
        with pytest.raises(_Stop):
            await first

    asyncio.run(scenario())

    assert seen["entered"], "the release was never reached"
    assert release.outcomes == [True], (
        "the release timed out waiting for another coroutine: it ran on the "
        "event loop and blocked it")
    assert seen["busy_during_release"] is True, (
        "the flag must be held before the first await")
    assert seen["second"] == 409
    assert route._phase_check_busy is False, "freed after the run"


@FLAG_ROUTES
def test_a_failing_release_still_frees_the_flag(monkeypatch, release, setup):
    call = setup(monkeypatch)

    def boom():
        raise RuntimeError("release failed")

    monkeypatch.setattr(route, "_free_interactive_gpu_caches", boom)

    with pytest.raises(RuntimeError, match="release failed"):
        asyncio.run(call())

    assert route._phase_check_busy is False


def test_the_propagation_pre_free_does_not_block_the_event_loop(
        monkeypatch, release):
    """``apply-to-grain`` with ``propagate_similar`` frees the worker at the top
    of the route, in a thread. ``_propagate_to_similar_grains`` is a plain
    function again and releases nothing itself."""
    import backend.spherical_gpu.pipeline.variant_unification as vu
    from tests.test_grain_flip import _fake_nests_result

    result, q_target, _q_wrong, _nests = _fake_nests_result()
    monkeypatch.setattr(route, "_get_result", lambda *a, **k: result)
    monkeypatch.setattr(
        vu, "build_render_score_fn",
        lambda sht_path, det, get_pattern, max_bandwidth=128:
            (lambda flats, quats: np.zeros(np.asarray(flats).reshape(-1).size)))
    req = route.GrainApplyRequest(
        row=0, col=0, quat=[float(v) for v in q_target],
        threshold_deg=5.0, propagate_similar=True)

    async def scenario():
        task = asyncio.ensure_future(route.apply_variant_to_grain(req))
        for _ in range(300):
            if release.entered.is_set():
                break
            await asyncio.sleep(0.01)
        release.go.set()                          # only a live loop gets here
        return await task

    out = asyncio.run(scenario())

    assert release.outcomes == [True], "the release blocked the event loop"
    assert out["propagate"]["n_candidates"] == 2


def test_the_propagation_pre_free_also_runs_when_no_sibling_qualifies(
        monkeypatch, release):
    """The release now sits at the top of the route, before the result is read
    (see the race test below), so it no longer waits for the candidate search:
    a click with nothing to propagate frees the worker too. That is the price
    of a route without an await between its read and its write."""
    import backend.spherical_gpu.pipeline.variant_unification as vu
    from tests.test_grain_flip import _fake_nests_result

    result, q_target, _q_wrong, nests = _fake_nests_result()
    # Make the sibling and the real nest already equal to the target: no
    # candidate is left once the clicked nest is the only wrong one.
    qdata = np.asarray(result.xmap.rotations.data).copy()
    for key in ("sibling", "real"):
        for (r, c) in nests[key]:
            qdata[r * 8 + c] = q_target
    from orix.quaternion import Rotation
    result.xmap._rotations = Rotation(qdata)
    monkeypatch.setattr(route, "_get_result", lambda *a, **k: result)
    monkeypatch.setattr(
        vu, "build_render_score_fn",
        lambda *a, **k: (lambda flats, quats: np.zeros(1)))
    release.go.set()
    req = route.GrainApplyRequest(
        row=0, col=0, quat=[float(v) for v in q_target],
        threshold_deg=5.0, propagate_similar=True)

    out = asyncio.run(route.apply_variant_to_grain(req))

    assert out["propagate"]["n_candidates"] == 0
    assert release.outcomes == [True], "released once, before the result is read"


# ---------------------------------------------------------------------------
# The release of apply-to-grain / unify / phase-reassign: off the loop, and
# BEFORE the result is read.
# ---------------------------------------------------------------------------

def _first_read(release):
    """A stand-in for the route's first touch of the result: note it, stop."""
    def read(*a, **k):
        release.log.append("read")
        raise _Stop("read")
    return read


def _apply_refine_setup(monkeypatch, release):
    monkeypatch.setattr(route, "_get_result", _first_read(release))
    req = route.GrainApplyRequest(row=0, col=0, quat=[1.0, 0, 0, 0], refine=True)
    return lambda: route.apply_variant_to_grain(req)


def _apply_propagate_setup(monkeypatch, release):
    monkeypatch.setattr(route, "_get_result", _first_read(release))
    req = route.GrainApplyRequest(row=0, col=0, quat=[1.0, 0, 0, 0],
                                  propagate_similar=True)
    return lambda: route.apply_variant_to_grain(req)


def _unify_setup(monkeypatch, release):
    class _Result:
        metadata = {"indexing_method": "spherical",
                    "sht_paths_by_phase": {0: "dummy.sht"},
                    "detector_geometry": {"pat_width": 8}}

        @property
        def xmap(self):                      # the first read of the orientations
            release.log.append("read")
            raise _Stop("read")

    monkeypatch.setattr(route, "_get_result", lambda *a, **k: _Result())
    return lambda: route.unify_pseudosym_variants(route.UnifyVariantsRequest())


def _reassign_setup(monkeypatch, release):
    from types import SimpleNamespace
    monkeypatch.setattr(
        route, "_get_result",
        lambda *a, **k: SimpleNamespace(metadata={"phase_check": {"report": {}}}))

    # `_phase_check_ctx` builds the orientation arrays.
    monkeypatch.setattr(route, "_phase_check_ctx", _first_read(release))
    return lambda: route.phase_reassign(route.PhaseReassignRequest())


@pytest.mark.parametrize(
    "setup", [_apply_refine_setup, _apply_propagate_setup, _unify_setup,
              _reassign_setup],
    ids=["apply-refine", "apply-propagate", "unify", "phase-reassign"])
def test_the_release_runs_off_the_loop_and_before_the_result_is_read(
        monkeypatch, release, setup):
    call = setup(monkeypatch, release)

    async def scenario():
        task = asyncio.ensure_future(call())
        for _ in range(300):
            if release.entered.is_set():
                break
            await asyncio.sleep(0.01)
        release.go.set()                          # only a live loop gets here
        with pytest.raises(_Stop):
            await task

    asyncio.run(scenario())

    assert release.outcomes == [True], (
        "the release timed out waiting for another coroutine: it ran on the "
        "event loop and blocked it (or never ran)")
    assert release.log == ["release", "read"], (
        "the release must finish before the route reads the result, else the "
        "wait for the worker widens the window to a concurrent edit")


def test_a_second_apply_during_the_release_is_not_overwritten(
        monkeypatch, release):
    """Two panels: apply A (refine, so it releases the worker) is still waiting
    for the worker to exit when apply B fixes a different grain. A must not
    write back the orientations it read before B's edit.

    Map from ``_fake_nests_result``: 8x8, qT everywhere except three 2x2 nests at
    qW, and a stripe of another phase in row 4. A clicks the nest at rows 0-1 /
    cols 0-1 (flat indices 0, 1, 8, 9): above the stripe it is alone, so A changes
    4 pixels. B clicks the nest at rows 6-7 / cols 0-1: below the stripe the
    same-phase region (rows 5-7) holds that nest AND the one at rows 6-7 / cols
    6-7, and the snap fill walks to both, so B changes 4 + 4 = 8 pixels."""
    from backend.spherical_gpu.pseudosym import same_orientation_angle_deg
    from tests.test_grain_flip import _fake_nests_result

    result, q_target, q_wrong, nests = _fake_nests_result()
    monkeypatch.setattr(route, "_get_result", lambda *a, **k: result)
    monkeypatch.setattr(route, "_grain_newton_refine",
                        lambda *a, **k: ({}, {"status": "skipped"}))
    quat = [float(v) for v in q_target]
    req_a = route.GrainApplyRequest(row=0, col=0, quat=quat, refine=True)
    req_b = route.GrainApplyRequest(row=6, col=0, quat=quat, refine=False)

    async def scenario():
        a = asyncio.ensure_future(route.apply_variant_to_grain(req_a))
        for _ in range(300):
            if release.entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert release.entered.is_set(), "A never reached the release"
        out_b = await route.apply_variant_to_grain(req_b)   # during A's release
        release.go.set()
        return await a, out_b

    out_a, out_b = asyncio.run(scenario())

    assert out_a["n_changed"] == 4 and out_b["n_changed"] == 8
    q = np.asarray(result.xmap.rotations.data).reshape(-1, 4)
    for name in ("clicked", "sibling", "real"):   # all 12 nest pixels at qT
        for (r, c) in nests[name]:
            ang = float(same_orientation_angle_deg(
                q[r * 8 + c][None, :], q_target, "m-3")[0])
            assert ang < 2.0, f"{name} nest pixel ({r},{c}) was reverted"
    # B ran first, A second: the single undo slot holds A's edit (the last
    # one), and it restores A's nest to the wrong orientation without
    # touching B's.
    undo = result.metadata["grain_flip_undo"]
    assert sorted(undo["xmap_indices"]) == [0, 1, 8, 9]
    for old in undo["old_quats"]:
        assert float(same_orientation_angle_deg(
            np.asarray(old)[None, :], q_wrong, "m-3")[0]) < 1e-3


@pytest.mark.parametrize("change", ["replaced", "dropped"])
def test_phase_reassign_is_refused_when_the_phase_check_changes_during_the_release(
        monkeypatch, release, change):
    """The route checks that a phase check exists, then waits for the worker.
    An apply-to-grain in that window drops the check (a new check replaces it);
    the orientations read after the wait no longer match its grain ids, so the
    reassign must answer 409 and leave the result alone."""
    from types import SimpleNamespace

    xmap = object()
    result = SimpleNamespace(
        xmap=xmap, metadata={"phase_check": {"report": {}}})
    ctx_calls = []
    monkeypatch.setattr(route, "_get_result", lambda *a, **k: result)
    monkeypatch.setattr(route, "_phase_check_ctx",
                        lambda *a, **k: ctx_calls.append(1))

    async def scenario():
        task = asyncio.ensure_future(
            route.phase_reassign(route.PhaseReassignRequest()))
        for _ in range(300):
            if release.entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert release.entered.is_set(), "the release was never reached"
        if change == "replaced":
            result.metadata["phase_check"] = {"report": {}}
        else:
            result.metadata.pop("phase_check")
        release.go.set()
        with pytest.raises(HTTPException) as err:
            await task
        return err.value

    exc = asyncio.run(scenario())

    assert exc.status_code == 409
    assert "out of date" in exc.detail
    assert ctx_calls == [], "the orientations were read for a stale check"
    assert "phase_reassign_undo" not in result.metadata
    assert result.xmap is xmap


def test_apply_stays_on_the_result_it_was_started_on_when_another_is_activated(
        monkeypatch, release):
    """The client sends no ``result_id``, so the route works on the ACTIVE
    result. The user activates result B in the gallery while apply (started on
    A) still waits for the worker to exit: the correction and its undo belong
    to A, B must stay as it was. Uses the real registry and the real
    ``activate_result``.

    Both results are the 8x8 map of ``_fake_nests_result``: qT everywhere except
    three 2x2 nests at qW, and a stripe of another phase in row 4. The click
    targets the nest at rows 0-1 / cols 0-1 (flat 0, 1, 8, 9), which the stripe
    keeps apart from the other two, so a correct apply changes exactly 4 pixels
    of A and none of B."""
    from collections import OrderedDict

    from backend.spherical_gpu.pseudosym import same_orientation_angle_deg
    from tests.test_grain_flip import _fake_nests_result

    res_a, q_target, q_wrong, nests = _fake_nests_result()
    res_b, _qt, _qw, _nests = _fake_nests_result()
    b_before = np.asarray(res_b.xmap.rotations.data).copy()
    monkeypatch.setattr(route, "_result_registry",
                        OrderedDict([("A", res_a), ("B", res_b)]))
    monkeypatch.setattr(route, "_active_result_id", "A")
    monkeypatch.setattr(route, "_grain_newton_refine",
                        lambda *a, **k: ({}, {"status": "skipped"}))
    req = route.GrainApplyRequest(
        row=0, col=0, quat=[float(v) for v in q_target], refine=True)

    async def scenario():
        task = asyncio.ensure_future(route.apply_variant_to_grain(req))
        for _ in range(300):
            if release.entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert release.entered.is_set(), "apply never reached the release"
        await route.activate_result("B")          # during the release
        release.go.set()
        return await task

    out = asyncio.run(scenario())

    assert route._active_result_id == "B"
    assert out["n_changed"] == 4
    qa = np.asarray(res_a.xmap.rotations.data).reshape(-1, 4)
    for (r, c) in nests["clicked"]:
        assert float(same_orientation_angle_deg(
            qa[r * 8 + c][None, :], q_target, "m-3")[0]) < 2.0
    np.testing.assert_array_equal(
        np.asarray(res_b.xmap.rotations.data), b_before)
    assert sorted(res_a.metadata["grain_flip_undo"]["xmap_indices"]) == [0, 1, 8, 9]
    assert "grain_flip_undo" not in res_b.metadata


def test_apply_to_grain_awaits_only_the_release_and_the_guarded_refine():
    """The race test above runs one interleaving. This pins the property behind
    it: between the first read of the result and the write-back the route must
    not be able to suspend unguarded. Two awaits are allowed: the release before
    the first read, and the refine (45 s on the loop otherwise, 2026-11-01),
    which must be followed by the check that the result is unchanged
    (tests/test_grain_refine_bounded.py runs that interleaving). The propagation
    helper must stay a plain function."""
    import ast
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(route.apply_variant_to_grain))
    tree = ast.parse(src)
    awaits = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Await)]
    assert len(awaits) == 2, awaits
    assert any("_free_interactive_gpu_caches" in a for a in awaits)
    refine = [a for a in awaits if "_grain_newton_refine" in a]
    assert len(refine) == 1
    after = src[src.index("_grain_newton_refine, result"):]
    guard = after.index("_orientation_fingerprint(result) != before")
    assert guard < after.index("new_q_map.update(refined)")
    assert not inspect.iscoroutinefunction(route._propagate_to_similar_grains)
