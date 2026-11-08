"""The grain-tool routes give the grain code the topology of the scan stamped on the result."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.routes import indexing as ix  # noqa: E402
from backend.api.services.scan_geometry import ScanGeometry  # noqa: E402
from backend.api.services.scan_neighbours import Topology  # noqa: E402
from backend.spherical_gpu.pipeline import island_check as ic_mod  # noqa: E402
from backend.spherical_gpu.pipeline import phase_reassignment as pr  # noqa: E402
from backend.spherical_gpu.pipeline import variant_unification as vu  # noqa: E402
from backend.spherical_gpu import pseudosym as ps  # noqa: E402
from tests.test_grain_flip import (  # noqa: E402
    _fake_nests_result, _fake_spherical_result, _fake_two_grain_result, _patch_variant_render_seams, _q_axis_angle)
from tests.test_phase_reassignment import (  # noqa: E402
    _fake_result_with_island, _fake_two_phase_result, _patch_endpoint_seams, _run)

STAMPS = {
    "hex": lambda n_rows, n_cols: ScanGeometry.hex(n_rows, n_cols, 0.5).to_dict(),
    "square": lambda n_rows, n_cols: ScanGeometry.square(n_rows, n_cols, 0.5).to_dict(),
    "unknown": lambda n_rows, n_cols: None,
}
KINDS = pytest.mark.parametrize("kind", ["hex", "square", "unknown"])


def _stamp(result, kind, shape=None):
    n_rows, n_cols = shape or result.original_shape
    stamp = STAMPS[kind](n_rows, n_cols)
    if stamp is None:
        result.metadata.pop("scan_geometry", None)
    else:
        result.metadata["scan_geometry"] = stamp
    return result


def _expect(seen_topology, kind, shape):
    """The topology a tool was handed is the hexagonal one of the stamp, or None."""
    if kind == "hex":
        assert (seen_topology.kind, seen_topology.n_rows, seen_topology.n_cols, seen_topology.odd_row_shift) == (
            "hex", shape[0], shape[1], 0.5)
    else:
        assert seen_topology is None


@pytest.fixture
def seen(monkeypatch):
    """Records the topology each grain tool is handed; the real function still runs."""
    got = {}

    def spy(module, name):
        original = getattr(module, name)
        key = f"{module.__name__.rsplit('.', 1)[-1]}.{name}"

        def wrapper(*args, **kwargs):
            got.setdefault(key, []).append(kwargs.get("topology"))
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, wrapper)

    for module, name in ((ic_mod, "check_islands"), (pr, "check_map"), (pr, "apply_reassignment"),
                         (pr, "segment_supergroup_grains"), (vu, "segment_supergroup_grains"),
                         (ps, "grain_snap_floodfill"), (ix, "_neighbour_grain_candidates"), (ix, "_grain_contacts"),
                         (ix, "_propagate_to_similar_grains")):
        spy(module, name)
    return got


# ------------------------------------------------------------------------------ the helpers

def test_the_topology_of_a_result_is_the_one_its_stamp_describes():
    result = SimpleNamespace(metadata={"scan_geometry": STAMPS["hex"](4, 5)})
    topology = ix._result_topology(result, (4, 5))
    assert (topology.kind, topology.n_rows, topology.n_cols, topology.odd_row_shift) == ("hex", 4, 5, 0.5)
    for metadata in ({}, {"scan_geometry": STAMPS["square"](4, 5)}, None):
        assert ix._result_topology(SimpleNamespace(metadata=metadata), (4, 5)) is None


def test_a_stamp_of_another_grid_is_a_409_with_both_shapes():
    result = SimpleNamespace(metadata={"scan_geometry": STAMPS["hex"](4, 5)})
    with pytest.raises(HTTPException) as caught:
        ix._result_topology(result, (5, 4))
    assert caught.value.status_code == 409 and "4 x 5" in caught.value.detail and "5 x 4" in caught.value.detail


@pytest.mark.parametrize("shift", [0.5, -0.5])
def test_the_contacts_of_a_grain_are_counted_on_the_neighbours_of_the_lattice(shift):
    n_rows = n_cols = 6
    g = ScanGeometry.hex(n_rows, n_cols, 1.0, odd_row_shift=shift)
    x, y = g.positions()
    pts = np.stack([x.ravel(), y.ravel()], axis=1)
    dist = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    labels = np.full(n_rows * n_cols, 1)
    labels[[8, 14, 15, 20]] = 0                                  # a small grain in the middle
    labels[[0, 1, 6]] = -1                                       # and some unindexed points
    inside = np.flatnonzero(labels == 0)
    # by hand, from the positions: the pairs one step apart whose far end is in grain 1
    expected = sum(1 for p in inside for q in range(n_rows * n_cols)
                   if abs(dist[p, q] - 1.0) < 1e-6 and labels[q] == 1)
    assert expected > 0
    hexagonal = ix._grain_contacts(labels, 0, n_rows, n_cols, Topology("hex", n_rows, n_cols, shift))
    assert hexagonal == {1: expected}
    # the four neighbours of the index grid count a different number of pairs for this grain
    assert ix._grain_contacts(labels, 0, n_rows, n_cols) != hexagonal


def _inline_loop(labels, g0, n_rows, n_cols):
    """The loop that was in _neighbour_grain_candidates, copied literally."""
    old: dict[int, int] = {}
    for f in np.flatnonzero(labels == g0):
        r, c = divmod(int(f), n_cols)
        for nb in ((f - n_cols) if r > 0 else -1, (f + n_cols) if r < n_rows - 1 else -1,
                   (f - 1) if c > 0 else -1, (f + 1) if c < n_cols - 1 else -1):
            if nb >= 0 and labels[nb] >= 0 and labels[nb] != g0:
                old[int(labels[nb])] = old.get(int(labels[nb]), 0) + 1
    return old


def test_the_contacts_without_a_topology_are_the_ones_the_inline_loop_counted():
    rng = np.random.default_rng(2)
    n_rows, n_cols = 7, 9
    labels = rng.integers(-1, 4, size=n_rows * n_cols)
    for g0 in range(4):
        old = _inline_loop(labels, g0, n_rows, n_cols)
        assert ix._grain_contacts(labels, g0, n_rows, n_cols) == old
        assert ix._grain_contacts(labels, g0, n_rows, n_cols, Topology("square", n_rows, n_cols)) == old


@pytest.mark.parametrize("seed", [2, 3, 4, 5, 6])
def test_the_contacts_come_in_the_order_the_inline_loop_met_them(seed, monkeypatch):
    """The caller sorts by count; a tie is settled by this order, so it must not change. A block of 5 pixels
    makes the grain span several blocks."""
    monkeypatch.setattr(ix, "_CONTACT_BLOCK", 5)
    rng = np.random.default_rng(seed)
    n_rows, n_cols = 11, 13
    labels = rng.integers(-1, 7, size=n_rows * n_cols)
    for g0 in range(7):
        old = _inline_loop(labels, g0, n_rows, n_cols)
        new = ix._grain_contacts(labels, g0, n_rows, n_cols)
        assert list(new.items()) == list(old.items())


def test_the_contacts_refuse_a_topology_of_another_grid():
    labels = np.zeros(12, dtype=int)
    with pytest.raises(ValueError, match="3 x 5"):
        ix._grain_contacts(labels, 0, 3, 4, Topology("hex", 3, 5, 0.5))


# ------------------------------------------------------------------ the phase check and its tools

@KINDS
def test_the_phase_check_and_the_reassignment_are_given_the_topology_of_the_stamp(monkeypatch, seen, kind):
    result = _stamp(_fake_two_phase_result(), kind)
    ix_ = _patch_endpoint_seams(monkeypatch, result)
    _run(ix_.phase_check(ix_.PhaseCheckRequest()))
    shape = result.original_shape
    assert len(seen["phase_reassignment.check_map"]) == 1
    _expect(seen["phase_reassignment.check_map"][0], kind, shape)
    assert seen["phase_reassignment.segment_supergroup_grains"]                  # inside check_map
    for topology in seen["phase_reassignment.segment_supergroup_grains"]:
        _expect(topology, kind, shape)
    seen.clear()
    _run(ix_.phase_reassign(ix_.PhaseReassignRequest()))
    assert len(seen["phase_reassignment.apply_reassignment"]) == 1
    _expect(seen["phase_reassignment.apply_reassignment"][0], kind, shape)
    for topology in seen["phase_reassignment.segment_supergroup_grains"]:         # re-derived with the same one
        _expect(topology, kind, shape)


@KINDS
def test_the_island_check_is_given_the_topology_of_the_stamp(monkeypatch, seen, kind):
    result = _stamp(_fake_result_with_island(row=6, col=6), kind)
    ix_ = _patch_endpoint_seams(monkeypatch, result)
    _run(ix_.island_check(ix_.IslandCheckRequest(max_variants=0)))
    assert len(seen["island_check.check_islands"]) == 1
    _expect(seen["island_check.check_islands"][0], kind, result.original_shape)


@KINDS
def test_assign_phase_segments_and_applies_on_the_topology_of_the_stamp(monkeypatch, seen, kind):
    result = _stamp(_fake_two_phase_result(), kind)
    ix_ = _patch_endpoint_seams(monkeypatch, result)
    _run(ix_.assign_phase_to_grain(ix_.AssignPhaseRequest(row=0, col=0, target_phase_id=1)))
    shape = result.original_shape
    assert len(seen["variant_unification.segment_supergroup_grains"]) == 1
    _expect(seen["variant_unification.segment_supergroup_grains"][0], kind, shape)
    assert len(seen["phase_reassignment.apply_reassignment"]) == 1
    _expect(seen["phase_reassignment.apply_reassignment"][0], kind, shape)


@pytest.mark.parametrize("route", ["check", "island", "assign"])
def test_a_stamp_of_another_grid_stops_the_phase_tools_before_they_run(monkeypatch, seen, route):
    result = _fake_two_phase_result()
    result.metadata["scan_geometry"] = STAMPS["hex"](3, 3)                       # the map is 8 x 8
    ix_ = _patch_endpoint_seams(monkeypatch, result)
    calls = {
        "check": lambda: ix_.phase_check(ix_.PhaseCheckRequest()),
        "island": lambda: ix_.island_check(ix_.IslandCheckRequest(max_variants=0)),
        "assign": lambda: ix_.assign_phase_to_grain(ix_.AssignPhaseRequest(row=0, col=0, target_phase_id=1)),
    }
    with pytest.raises(HTTPException) as caught:
        _run(calls[route]())
    assert caught.value.status_code == 409 and "3 x 3" in caught.value.detail
    assert seen == {}


def test_the_reassignment_refuses_a_stamp_of_another_grid_and_leaves_the_result_as_it_was(monkeypatch, seen):
    """The check ran on a good stamp; the stamp then no longer describes the map. Nothing is written."""
    result = _stamp(_fake_two_phase_result(), "hex")
    ix_ = _patch_endpoint_seams(monkeypatch, result)
    _run(ix_.phase_check(ix_.PhaseCheckRequest()))
    result.metadata["scan_geometry"] = STAMPS["hex"](3, 3)
    before_q = np.array(result.xmap.rotations.data, copy=True)
    before_pid = np.array(result.xmap.phase_id, copy=True)
    seen.clear()
    with pytest.raises(HTTPException) as caught:
        _run(ix_.phase_reassign(ix_.PhaseReassignRequest()))
    assert caught.value.status_code == 409
    assert seen == {}
    assert np.array_equal(result.xmap.rotations.data, before_q)
    assert np.array_equal(result.xmap.phase_id, before_pid)
    assert "phase_reassign_undo" not in result.metadata


BROKEN_STAMPS = [
    {"kind": "hex"},
    {"kind": "hex", "n_rows": "x", "n_cols": 4},
    {"kind": "hex", "n_rows": 4, "n_cols": 4},                                    # no row shift
    {"kind": "hex", "n_rows": 4, "n_cols": 4, "odd_row_shift": "abc"},
    {"kind": "hex", "n_rows": float("inf"), "n_cols": 4, "odd_row_shift": 0.5},
    {"kind": "hex", "n_rows": 4.7, "n_cols": 4, "odd_row_shift": 0.5},
    # the routes that draw a map refuse a stamp with a valid mask (a 400): the grain tools refuse it too
    {**ScanGeometry.hex(4, 4, 0.5).to_dict(), "has_valid_mask": True},
]
BROKEN_IDS = ["no fields", "n_rows text", "no shift", "shift text", "n_rows inf", "n_rows fractional", "valid mask"]
ALL_SIX_ROUTES = ["variants", "apply-to-grain", "assign-phase", "island-check", "phase-check", "phase-reassign"]


@pytest.mark.parametrize("broken", BROKEN_STAMPS, ids=BROKEN_IDS)
@pytest.mark.parametrize("route", ALL_SIX_ROUTES)
def test_a_hexagonal_stamp_that_cannot_be_read_is_a_409_with_a_sentence_not_a_500(monkeypatch, seen, route, broken):
    """A missing, non-numeric, infinite or fractional field, or no row shift: on every route that reads the
    stamp it is a 409 that says the stamp cannot be read, before anything runs (never a 500)."""
    if route in ("variants", "apply-to-grain"):
        result, q_a, _q_b = _fake_spherical_result()
        monkeypatch.setattr(ix, "_get_result", lambda *_a, **_k: result)
        ix_ = ix
        call = {
            "variants": lambda: ix.get_pattern_match_variants(row=0, col=0),
            "apply-to-grain": lambda: ix.apply_variant_to_grain(ix.GrainApplyRequest(
                row=0, col=0, quat=[float(v) for v in q_a], threshold_deg=5.0, max_total_deg=15.0)),
        }[route]
    else:
        result = _fake_result_with_island(row=6, col=6) if route == "island-check" else _fake_two_phase_result()
        ix_ = _patch_endpoint_seams(monkeypatch, result)
        call = {
            "assign-phase": lambda: ix_.assign_phase_to_grain(
                ix_.AssignPhaseRequest(row=0, col=0, target_phase_id=1)),
            "island-check": lambda: ix_.island_check(ix_.IslandCheckRequest(max_variants=0)),
            "phase-check": lambda: ix_.phase_check(ix_.PhaseCheckRequest()),
            "phase-reassign": lambda: ix_.phase_reassign(ix_.PhaseReassignRequest()),
        }[route]
    if route == "phase-reassign":
        # The reassignment needs a check on file first (as in the test of the stamp of another grid).
        _stamp(result, "hex")
        _run(ix_.phase_check(ix_.PhaseCheckRequest()))
        seen.clear()
    result.metadata["scan_geometry"] = broken
    with pytest.raises(HTTPException) as caught:
        _run(call())
    assert caught.value.status_code == 409
    assert "The scan geometry stamped on this result cannot be read" in caught.value.detail
    assert seen == {}


def test_the_lookup_of_a_results_phases_does_not_need_its_stamp_to_be_right(monkeypatch):
    result = _fake_two_phase_result()
    result.metadata["scan_geometry"] = STAMPS["hex"](3, 3)
    ix_ = _patch_endpoint_seams(monkeypatch, result)
    out = _run(ix_.phase_check_phases(result_id=None))
    assert [e["phase_id"] for e in out["phases"]] == [1, 2]                         # it answers; no 409


def test_the_context_of_a_render_verification_does_not_need_its_stamp_to_be_right(monkeypatch):
    result = _fake_two_phase_result()
    result.metadata["scan_geometry"] = STAMPS["hex"](3, 3)
    _patch_endpoint_seams(monkeypatch, result)
    ctx = ix._phase_check_ctx(result, build_scorers=True, with_topology=False)
    assert ctx["topology"] is None
    with pytest.raises(HTTPException):
        ix._phase_check_ctx(result, build_scorers=True)


# ------------------------------------------------------------- the pseudo-symmetry tools

@KINDS
def test_the_variant_gallery_looks_for_neighbouring_grains_on_the_topology_of_the_stamp(monkeypatch, seen, kind):
    ix_ = _patch_variant_render_seams(monkeypatch)
    result, _q_a, _q_b = _fake_two_grain_result()
    _stamp(result, kind)
    monkeypatch.setattr(ix_, "_get_result", lambda *_a, **_k: result)
    _run(ix_.get_pattern_match_variants(row=0, col=0))
    assert len(seen["indexing._neighbour_grain_candidates"]) == 1
    _expect(seen["indexing._neighbour_grain_candidates"][0], kind, result.original_shape)
    assert seen["indexing._grain_contacts"]                                       # the census of the grains around
    for topology in seen["indexing._grain_contacts"]:
        _expect(topology, kind, result.original_shape)
    for topology in seen["variant_unification.segment_supergroup_grains"]:
        _expect(topology, kind, result.original_shape)


@KINDS
def test_apply_to_grain_fills_and_propagates_on_the_topology_of_the_stamp(monkeypatch, seen, kind):
    result, q_t, _q_w, _nests = _fake_nests_result()
    _stamp(result, kind)
    monkeypatch.setattr(ix, "_get_result", lambda *_a, **_k: result)
    monkeypatch.setattr(ix, "_free_interactive_gpu_caches", lambda: None)
    monkeypatch.setattr(vu, "build_render_score_fn", lambda *a, **k: (lambda flats, quats: np.full(
        np.asarray(flats).reshape(-1).size, 0.5)))
    req = ix.GrainApplyRequest(row=0, col=0, quat=[float(v) for v in q_t], threshold_deg=5.0, propagate_similar=True)
    _run(ix.apply_variant_to_grain(req))
    shape = result.original_shape
    assert seen["pseudosym.grain_snap_floodfill"]                                 # the clicked grain, then siblings
    for topology in seen["pseudosym.grain_snap_floodfill"]:
        _expect(topology, kind, shape)
    assert len(seen["indexing._propagate_to_similar_grains"]) == 1
    _expect(seen["indexing._propagate_to_similar_grains"][0], kind, shape)
    for topology in seen["variant_unification.segment_supergroup_grains"]:
        _expect(topology, kind, shape)


def test_apply_to_grain_and_the_gallery_refuse_a_stamp_of_another_grid(monkeypatch, seen):
    result, q_a, _q_b = _fake_spherical_result()
    result.metadata["scan_geometry"] = STAMPS["hex"](3, 3)                       # the map is 4 x 4
    monkeypatch.setattr(ix, "_get_result", lambda *_a, **_k: result)
    before = np.array(result.xmap.rotations.data, copy=True)
    req = ix.GrainApplyRequest(row=0, col=0, quat=[float(v) for v in q_a], threshold_deg=5.0, max_total_deg=15.0)
    with pytest.raises(HTTPException) as caught:
        _run(ix.apply_variant_to_grain(req))
    assert caught.value.status_code == 409
    assert seen == {}
    assert np.array_equal(result.xmap.rotations.data, before) and "grain_flip_undo" not in result.metadata
    with pytest.raises(HTTPException) as caught:                                  # the gallery, before any render
        _run(ix.get_pattern_match_variants(row=0, col=0))
    assert caught.value.status_code == 409 and "3 x 3" in caught.value.detail
    assert seen == {}


def _chain_result(n, chain, q_ok, q_wrong):
    """An n x n map of phase "Al" with one chain of phase-"alpha" cells at a wrong variant (everything else is
    the other phase, so the flood fill can only follow the chain)."""
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    quats = np.tile(q_ok, (n * n, 1)).astype(float)
    pid = np.ones(n * n, dtype=int)
    for r, c in chain:
        quats[r * n + c] = q_wrong
        pid[r * n + c] = 0
    x, y = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    xmap = CrystalMap(rotations=Rotation(quats), phase_id=pid, x=x.ravel(), y=y.ravel(),
                      phase_list=PhaseList(names=["alpha", "Al"], point_groups=["m-3", "m-3m"], ids=[0, 1]))
    return SimpleNamespace(
        xmap=xmap, original_shape=(n, n), selection_mask=np.ones((n, n), bool),
        metadata={"indexing_method": "spherical", "sht_paths_by_phase": {0: "dummy.sht"},
                  "detector_geometry": {"pat_width": 8, "pat_height": 8, "pc_x": 0.5, "pc_y": 0.5, "pc_z": 0.5}})


@pytest.mark.parametrize("shift", [0.5, -0.5])
def test_a_hexagonal_correction_crosses_a_diagonal_link_that_the_index_grid_does_not_have(monkeypatch, shift):
    """The whole route, not a spy: a chain of wrongly indexed cells that touch only diagonally."""
    n = 6
    g = ScanGeometry.hex(n, n, 1.0, odd_row_shift=shift)
    x, y = g.positions()
    pts = np.stack([x.ravel(), y.ravel()], axis=1)
    dist = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    chain = [(0, 3)]
    for r in range(1, n):                                         # one cell per row, one step from the one above
        pr_, pc = chain[-1]
        chain.append(next((r, c) for c in (pc - 1, pc + 1)
                          if 0 <= c < n and abs(dist[pr_ * n + pc, r * n + c] - 1.0) < 1e-6))
    q_ok, q_wrong = _q_axis_angle([0, 0, 1], 0.0), _q_axis_angle([0, 0, 1], 90.0)
    req = ix.GrainApplyRequest(row=chain[0][0], col=chain[0][1], quat=[float(v) for v in q_ok],
                               threshold_deg=5.0, max_total_deg=15.0)
    sizes = {}
    for kind in ("hex", "square"):
        result = _chain_result(n, chain, q_ok, q_wrong)
        result.metadata["scan_geometry"] = (ScanGeometry.hex(n, n, 1.0, odd_row_shift=shift).to_dict()
                                            if kind == "hex" else STAMPS["square"](n, n))
        monkeypatch.setattr(ix, "_get_result", lambda *_a, _r=result, **_k: _r)
        sizes[kind] = _run(ix.apply_variant_to_grain(req))["grain_size"]
    assert sizes == {"hex": n, "square": 1}      # on the index grid the six cells are six strangers
