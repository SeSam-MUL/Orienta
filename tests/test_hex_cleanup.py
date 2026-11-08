"""Cleanup of a phase map (min cluster, modal filter, fill) and the phase adjacency on a hexagonal result."""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.routes import phase_map  # noqa: E402
from backend.api.services import hex_raster  # noqa: E402
from backend.api.services.checkpoint_writer import CheckpointWriter  # noqa: E402
from backend.api.services.scan_geometry import ScanGeometry  # noqa: E402
from backend.api.services.scan_neighbours import Topology, topology_of_stamp  # noqa: E402
from tests.test_phase_map_layer_endpoint import _make_fake_result  # noqa: E402

N_ROWS, N_COLS, STEP = 6, 5, 0.5
clean = CheckpointWriter.clean_phase_assignment


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app, raise_server_exceptions=False)


def _arm(monkeypatch, result=None, dataset=None):
    # the routes read the two getters under three names: module-level in phase_map (`/layer`, `/phase-adjacency`),
    # and imported at call time from `indexing` and `analysis` (`_render_to_figure`, hence `/render` and `/export`)
    monkeypatch.setattr("backend.api.routes.phase_map.get_last_indexing_result", lambda: result)
    monkeypatch.setattr("backend.api.routes.phase_map.get_analysis_dataset", lambda: dataset)
    monkeypatch.setattr("backend.api.routes.indexing.get_last_indexing_result", lambda: result)
    monkeypatch.setattr("backend.api.routes.analysis.get_analysis_dataset", lambda: dataset)


def _result(phase_ids, stamp, n_rows=N_ROWS, n_cols=N_COLS):
    fake = _make_fake_result(n_rows=n_rows, n_cols=n_cols)
    fake.xmap.phase_id = np.asarray(phase_ids, dtype=np.int16).ravel()
    fake.metadata = {"scan_geometry": stamp} if stamp is not None else {}
    return fake


def _field(*points):
    """A field of phase 0 with the given (row, col) points set to phase 1."""
    ids = np.zeros((N_ROWS, N_COLS), dtype=np.int16)
    for r, c in points:
        ids[r, c] = 1
    return ids


HEX = ScanGeometry.hex(N_ROWS, N_COLS, STEP).to_dict()
SQUARE = ScanGeometry.square(N_ROWS, N_COLS, STEP).to_dict()
PAIR = ((1, 1), (2, 2))          # two points of an offset lattice that touch, but are not 4-neighbours on the index grid


# ------------------------------------------------------------ the pure function

def test_a_diagonal_pair_survives_a_minimum_cluster_size_of_two_on_a_hexagonal_scan():
    ids = _field(*PAIR)
    ci = np.ones(ids.shape, dtype=np.float32)
    out = clean(ids, ci, np.zeros_like(ci), min_cluster_size=2, topology=Topology("hex", N_ROWS, N_COLS, 0.5))
    assert out[1, 1] == out[2, 2] == 1
    assert int((out == 1).sum()) == 2


def test_the_same_pair_is_two_specks_on_the_index_grid():
    ids = _field(*PAIR)
    ci = np.ones(ids.shape, dtype=np.float32)
    assert int((clean(ids, ci, np.zeros_like(ci), min_cluster_size=2) == 1).sum()) == 0


def test_a_single_speck_is_removed_on_a_hexagonal_scan():
    ids = _field((2, 2))
    ci = np.ones(ids.shape, dtype=np.float32)
    out = clean(ids, ci, np.zeros_like(ci), min_cluster_size=2, topology=Topology("hex", N_ROWS, N_COLS, 0.5))
    assert int((out == 1).sum()) == 0


def test_the_modal_filter_removes_a_speck_on_a_hexagonal_scan():
    ids = _field((2, 2))
    ci = np.ones(ids.shape, dtype=np.float32)
    out = clean(ids, ci, np.zeros_like(ci), modal_filter_size=3, topology=Topology("hex", N_ROWS, N_COLS, 0.5))
    assert int((out == 1).sum()) == 0


def test_a_hole_is_filled_from_the_nearest_indexed_point_on_the_sample():
    # (0, 0) is unindexed; the indexed points are (2, 0) [phase 2, 1.732 away] and (0, 2) [phase 3, 2 away]
    ids = np.full((5, 5), -1, dtype=np.int16)
    ids[2, 0], ids[0, 2] = 2, 3
    ci = np.ones(ids.shape, dtype=np.float32)
    out = clean(ids, ci, np.zeros_like(ci), fill_unindexed=True, topology=Topology("hex", 5, 5, 0.5))
    assert out[0, 0] == 2


@pytest.mark.parametrize("seed", range(4))
def test_a_square_topology_removes_isolated_specks_like_no_topology(seed):
    rng = np.random.default_rng(seed)
    ids = np.zeros((9, 8), dtype=np.int16)                       # a uniform field with isolated specks of phase 1
    for r, c in rng.integers(0, [9, 8], size=(5, 2)):
        ids[r, c] = 1
    ci = np.ones(ids.shape, dtype=np.float32)
    square = Topology("square", 9, 8)
    for kwargs in ({"min_cluster_size": 2}, {"modal_filter_size": 3}):
        np.testing.assert_array_equal(
            clean(ids, ci, np.zeros_like(ci), **kwargs),
            clean(ids, ci, np.zeros_like(ci), topology=square, **kwargs))


# ------------------------------------------------------------------ the routes

def _png(b64):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))


def _pixel_of(row, col):
    g = ScanGeometry.hex(N_ROWS, N_COLS, STEP)
    spec = hex_raster.raster_spec(g)
    return hex_raster.pixel_of_cell(g, spec, row, col)


def test_the_layer_keeps_a_diagonal_pair_on_a_hexagonal_result(client, monkeypatch):
    _arm(monkeypatch, _result(_field(*PAIR), HEX))
    index_rgba = phase_map._compute_layer_rgba(kind="phase")
    body = client.get("/api/phasemap/layer", params={"kind": "phase", "min_cluster_size": 2}).json()
    img = _png(body["image"])
    i, j = _pixel_of(1, 1)
    np.testing.assert_array_equal(img[i, j], index_rgba[1, 1])               # still the colour of phase 1
    assert not np.array_equal(index_rgba[1, 1], index_rgba[0, 0])           # (so that the check can fail)


def test_the_layer_removes_the_same_pair_on_a_square_result(client, monkeypatch):
    _arm(monkeypatch, _result(_field(*PAIR), SQUARE))
    index_rgba = phase_map._compute_layer_rgba(kind="phase")
    body = client.get("/api/phasemap/layer", params={"kind": "phase", "min_cluster_size": 2}).json()
    img = _png(body["image"])
    np.testing.assert_array_equal(img[1, 1], index_rgba[0, 0])               # the colour of phase 0


def test_apply_cleanup_to_the_result_uses_the_hexagonal_neighbours(client, monkeypatch):
    fake = _result(_field(*PAIR), HEX)
    _arm(monkeypatch, fake)
    r = client.post("/api/phasemap/apply-cleanup", json={"target": "result", "min_cluster_size": 2})
    assert r.status_code == 200, r.text
    cleaned = fake.metadata["cleaned_phase_id"]
    assert cleaned[1, 1] == cleaned[2, 2] == 1


def test_a_stamp_that_does_not_describe_the_map_is_refused_not_guessed(client, monkeypatch):
    other = ScanGeometry.hex(4, 4, STEP).to_dict()                           # a 4 x 4 stamp on a 6 x 5 map
    _arm(monkeypatch, _result(_field(*PAIR), other))
    r = client.post("/api/phasemap/apply-cleanup", json={"target": "result", "min_cluster_size": 2})
    assert r.status_code == 400
    assert "scan geometry" in r.json()["detail"]


def test_the_adjacency_counts_the_six_neighbour_pairs_by_hand(client, monkeypatch):
    # 4 x 4, phase 0 in columns 0-1 and phase 1 in columns 2-3. Rows are offset +0.5 in odd rows.
    #   horizontal: one pair per row = 4
    #   rows 0-1: (0,2)-(1,1); rows 1-2: (1,1)-(2,2); rows 2-3: (2,2)-(3,1): 3 diagonal pairs
    ids = np.zeros((4, 4), dtype=np.int16)
    ids[:, 2:] = 1
    _arm(monkeypatch, _result(ids, ScanGeometry.hex(4, 4, STEP).to_dict(), 4, 4))
    body = client.get("/api/phasemap/phase-adjacency").json()
    assert body["adjacency"] == [[0, 7], [7, 0]] and body["total_borders"] == 7


def test_the_adjacency_of_a_square_result_counts_four_neighbours_as_before(client, monkeypatch):
    ids = np.zeros((4, 4), dtype=np.int16)
    ids[:, 2:] = 1
    _arm(monkeypatch, _result(ids, ScanGeometry.square(4, 4, STEP).to_dict(), 4, 4))
    body = client.get("/api/phasemap/phase-adjacency").json()
    assert body["adjacency"] == [[0, 4], [4, 0]] and body["total_borders"] == 4


def _spy_topology(monkeypatch):
    seen = []
    original = phase_map._compute_effective_phase_ids_2d

    def spy(*args, **kwargs):
        seen.append(kwargs.get("topology"))
        return original(*args, **kwargs)

    monkeypatch.setattr(phase_map, "_compute_effective_phase_ids_2d", spy)
    return seen


@pytest.mark.parametrize("stamp, hexagonal", [(HEX, True), (SQUARE, False), (None, False)],
                         ids=["hex", "square", "unknown"])
def test_render_and_export_hand_the_scans_topology_to_the_cleanup(client, monkeypatch, tmp_path, stamp, hexagonal):
    _arm(monkeypatch, _result(_field(*PAIR), stamp))
    seen = _spy_topology(monkeypatch)
    client.get("/api/phasemap/render", params={"min_cluster_size": 2})
    client.post("/api/phasemap/export", json={"output_path": str(tmp_path / "m.png"), "min_cluster_size": 2})
    assert len(seen) == 2
    for topology in seen:
        if hexagonal:
            assert (topology.kind, topology.n_rows, topology.n_cols, topology.odd_row_shift) == ("hex", N_ROWS, N_COLS, 0.5)
        else:
            assert topology is None


def _positions(shift, n_rows, n_cols):
    g = ScanGeometry.hex(n_rows, n_cols, 1.0, odd_row_shift=shift)
    x, y = g.positions()
    pts = np.stack([x.ravel(), y.ravel()], axis=1)
    return np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])


@pytest.mark.parametrize("shift, seed", [(0.5, 0), (-0.5, 1)])
def test_the_modal_filter_votes_in_the_hexagonal_disc_not_in_the_square_box(shift, seed):
    ids = np.random.default_rng(seed).integers(-1, 3, size=(8, 7)).astype(np.int16)
    ci = np.ones(ids.shape, dtype=np.float32)
    dist = _positions(shift, 8, 7)
    flat = ids.ravel()
    want = np.full(flat.size, -1, dtype=np.int16)
    for p in range(flat.size):
        window = flat[dist[p] <= 1 + 1e-6]                    # the point and its six neighbours
        window = window[window >= 0]
        if window.size:
            values, counts = np.unique(window, return_counts=True)
            want[p] = values[np.argmax(counts)]
    got = clean(ids, ci, np.zeros_like(ci), modal_filter_size=3, topology=Topology("hex", 8, 7, shift))
    np.testing.assert_array_equal(got.ravel(), want)
    assert (clean(ids, ci, np.zeros_like(ci), modal_filter_size=3) != got).any()      # (so that the check can fail)


@pytest.mark.parametrize("shift, seed", [(0.5, 0), (-0.5, 1)])
def test_the_fill_takes_the_phase_of_a_nearest_indexed_point_on_the_sample(shift, seed):
    rng = np.random.default_rng(seed)
    ids = np.full((8, 7), -1, dtype=np.int16)
    where = rng.choice(56, size=5, replace=False)
    ids.ravel()[where] = rng.integers(0, 3, size=5)
    ci = np.ones(ids.shape, dtype=np.float32)
    dist = _positions(shift, 8, 7)
    indexed = np.flatnonzero(ids.ravel() >= 0)

    def allowed(p):
        d = dist[p, indexed]
        return set(ids.ravel()[indexed[np.isclose(d, d.min(), atol=1e-9)]].tolist())

    got = clean(ids, ci, np.zeros_like(ci), fill_unindexed=True, topology=Topology("hex", 8, 7, shift)).ravel()
    assert all(got[p] in allowed(p) for p in range(56))
    plain = clean(ids, ci, np.zeros_like(ci), fill_unindexed=True).ravel()
    assert any(plain[p] not in allowed(p) for p in range(56))       # the index-grid fill gets some of them wrong


# ------------- the topology comes from the source the route draws (the gate used to look at that source too)

def _dataset(stamp, fake, shape=(N_ROWS, N_COLS)):
    """The analysis dataset the routes draw when there is no result that holds a crystal map."""
    return SimpleNamespace(xmap=fake.xmap, shape=shape, scan_geometry=stamp, per_phase_data={})


@pytest.mark.parametrize("stamp, hexagonal", [(HEX, True), (SQUARE, False), (None, False)],
                         ids=["hex", "square", "unknown"])
def test_without_a_result_the_analysis_datasets_stamp_decides_the_topology(client, monkeypatch, tmp_path, stamp,
                                                                           hexagonal):
    # No result: `/layer`, `/render` and `/export` draw the dataset's crystal map, so the dataset's stamp decides.
    # A hexagonal dataset was once refused by the gate; it must not now run on the four neighbours.
    _arm(monkeypatch, None, _dataset(stamp, _result(_field(*PAIR), None)))
    seen = _spy_topology(monkeypatch)
    client.get("/api/phasemap/layer", params={"kind": "phase", "min_cluster_size": 2})
    client.get("/api/phasemap/render", params={"min_cluster_size": 2})
    client.post("/api/phasemap/export", json={"output_path": str(tmp_path / "m.png"), "min_cluster_size": 2})
    assert len(seen) == 3
    for topology in seen:
        if hexagonal:
            assert (topology.kind, topology.n_rows, topology.n_cols, topology.odd_row_shift) == ("hex", N_ROWS, N_COLS, 0.5)
        else:
            assert topology is None


def test_a_square_result_next_to_a_hexagonal_dataset_keeps_the_square_code(client, monkeypatch):
    fake = _result(_field(*PAIR), SQUARE)                     # the result has a crystal map: the routes draw it
    _arm(monkeypatch, fake, _dataset(HEX, fake))
    seen = _spy_topology(monkeypatch)
    client.get("/api/phasemap/layer", params={"kind": "phase", "min_cluster_size": 2})
    assert seen == [None]


def test_a_hexagonal_result_next_to_a_square_dataset_gets_the_hexagonal_code(client, monkeypatch):
    fake = _result(_field(*PAIR), HEX)
    _arm(monkeypatch, fake, _dataset(SQUARE, fake))
    seen = _spy_topology(monkeypatch)
    client.get("/api/phasemap/layer", params={"kind": "phase", "min_cluster_size": 2})
    assert len(seen) == 1 and seen[0].kind == "hex"


def test_the_adjacency_of_a_hexagonal_dataset_counts_six_neighbours(client, monkeypatch):
    # the by-hand case of the result above (7 pairs on a hexagonal scan, 4 on a square one), now with no result
    ids = np.zeros((4, 4), dtype=np.int16)
    ids[:, 2:] = 1
    fake = _result(ids, None, 4, 4)
    _arm(monkeypatch, None, _dataset(ScanGeometry.hex(4, 4, STEP).to_dict(), fake, (4, 4)))
    assert client.get("/api/phasemap/phase-adjacency").json()["adjacency"] == [[0, 7], [7, 0]]
    _arm(monkeypatch, None, _dataset(ScanGeometry.square(4, 4, STEP).to_dict(), fake, (4, 4)))
    assert client.get("/api/phasemap/phase-adjacency").json()["adjacency"] == [[0, 4], [4, 0]]


def test_apply_cleanup_takes_no_topology_from_the_analysis_dataset(client, monkeypatch):
    # `apply-cleanup` cleans the active result and never the dataset (`_require_xmap(active, None)`): a hexagonal
    # dataset next to a square result does not make the cleanup hexagonal.
    fake = _result(_field(*PAIR), SQUARE)
    _arm(monkeypatch, fake, _dataset(HEX, fake))
    seen = []
    original = CheckpointWriter.clean_phase_assignment

    def spy(*args, **kwargs):
        seen.append(kwargs.get("topology"))
        return original(*args, **kwargs)

    monkeypatch.setattr(CheckpointWriter, "clean_phase_assignment", staticmethod(spy))
    r = client.post("/api/phasemap/apply-cleanup", json={"target": "result", "min_cluster_size": 2})
    assert r.status_code == 200, r.text
    assert seen == [None]


def test_apply_cleanup_asks_for_the_topology_of_the_result_alone(client, monkeypatch):
    # A result that holds a crystal map never lets the dataset decide, so the answer above cannot tell a call that
    # also hands over the dataset from one that does not. The call itself is held here: the route reads the result
    # only (`_require_xmap(active, None)`), so it names no dataset.
    fake = _result(_field(*PAIR), HEX)
    _arm(monkeypatch, fake, _dataset(SQUARE, fake))
    asked = []
    original = phase_map._hex_topology_of

    def spy(result, dataset, shape):
        asked.append((result, dataset))
        return original(result, dataset, shape)

    monkeypatch.setattr(phase_map, "_hex_topology_of", spy)
    r = client.post("/api/phasemap/apply-cleanup", json={"target": "result", "min_cluster_size": 2})
    assert r.status_code == 200, r.text
    assert asked == [(fake, None)]


# ---------------- the nearest-point lookup is asked only for the points the cleanup overwrites (speed on 1000 x 1200)

def _spy_nearest(monkeypatch):
    from backend.api.services import scan_filters
    calls = []
    original = scan_filters.nearest_valid

    def spy(valid, topology, targets=None):
        calls.append(None if targets is None else np.argwhere(targets).tolist())
        return original(valid, topology, targets)

    monkeypatch.setattr(scan_filters, "nearest_valid", spy)
    return calls


def test_the_small_cluster_removal_looks_up_only_the_points_it_relabels(monkeypatch):
    calls = _spy_nearest(monkeypatch)
    ids = _field((2, 2))
    ci = np.ones(ids.shape, dtype=np.float32)
    out = clean(ids, ci, np.zeros_like(ci), min_cluster_size=2, topology=Topology("hex", N_ROWS, N_COLS, 0.5))
    assert calls == [[[2, 2]]]
    assert out[2, 2] == 0


def test_the_fill_looks_up_only_the_unindexed_points(monkeypatch):
    calls = _spy_nearest(monkeypatch)
    ids = _field()
    ids[0, 0] = ids[3, 4] = -1
    ci = np.ones(ids.shape, dtype=np.float32)
    out = clean(ids, ci, np.zeros_like(ci), fill_unindexed=True, topology=Topology("hex", N_ROWS, N_COLS, 0.5))
    assert calls == [[[0, 0], [3, 4]]]
    assert out[0, 0] == out[3, 4] == 0


# ---------------- a hexagonal stamp that cannot be turned into a geometry is a 400 naming the stamp, not a 500

BAD_STAMPS = {
    "valid_mask": {**HEX, "has_valid_mask": True},
    "no_row_shift": {**HEX, "odd_row_shift": 0.0},
    "missing_field": {k: v for k, v in HEX.items() if k != "step_x"},
    "infinite_rows": {**HEX, "n_rows": float("inf")},     # int(inf) is an OverflowError, not a ValueError
}


def _route(client, name, **params):
    if name == "apply-cleanup":
        return client.post("/api/phasemap/apply-cleanup", json={"target": "result", "min_cluster_size": 2, **params})
    if name == "export":
        return client.post("/api/phasemap/export", json={"output_path": params.pop("path"), **params})
    if name == "layer":
        return client.get("/api/phasemap/layer", params={"kind": "phase", **params})
    return client.get(f"/api/phasemap/{name}", params=params)


@pytest.mark.parametrize("route", ["apply-cleanup", "phase-adjacency", "render", "layer", "export"])
@pytest.mark.parametrize("stamp", list(BAD_STAMPS), ids=list(BAD_STAMPS))
def test_an_unusable_hexagonal_stamp_is_answered_with_400_naming_the_stamp(client, monkeypatch, tmp_path, route, stamp):
    _arm(monkeypatch, _result(_field(*PAIR), BAD_STAMPS[stamp]))
    r = _route(client, route, path=str(tmp_path / "m.png"))
    assert r.status_code == 400, r.text
    assert "scan geometry stamped on this map" in r.json()["detail"]


# ---------------- the modal window is limited on a hexagonal scan only

@pytest.mark.parametrize("route", ["layer", "apply-cleanup"])
def test_a_modal_window_of_seven_is_accepted_and_nine_is_refused_on_a_hexagonal_scan(client, monkeypatch, route):
    _arm(monkeypatch, _result(_field(*PAIR), HEX))
    assert _route(client, route, modal_filter_size=7).status_code == 200
    r = _route(client, route, modal_filter_size=9)
    assert r.status_code == 400, r.text
    assert "modal filter size 9" in r.json()["detail"] and "7" in r.json()["detail"]


def test_an_oversize_modal_window_is_refused_by_render_and_export_too_not_drawn_raw(client, monkeypatch, tmp_path):
    _arm(monkeypatch, _result(_field(*PAIR), HEX))
    assert _route(client, "render", modal_filter_size=9).status_code == 400
    assert _route(client, "export", modal_filter_size=9, path=str(tmp_path / "m.png")).status_code == 400
    assert not (tmp_path / "m.png").exists()


@pytest.mark.parametrize("route", ["layer", "apply-cleanup"])
def test_a_modal_window_of_nine_on_a_square_scan_is_as_before(client, monkeypatch, route):
    _arm(monkeypatch, _result(_field(*PAIR), SQUARE))
    assert _route(client, route, modal_filter_size=9).status_code == 200


# ---------------- apply-cleanup does not run a hexagonal result on a checkpoint's four neighbours

def _checkpoint(tmp_path, stem="scan", geometry=None, phase_ids=None):
    """A checkpoint of the 6 x 5 grid. ``geometry`` is stamped on it (``None``: a checkpoint without ``grid_type``,
    which reads as square)."""
    import h5py
    path = tmp_path / f"{stem}_multiphase.h5"
    if geometry is not None:
        CheckpointWriter(str(tmp_path / f"{stem}.h5oina")).init_metadata(
            (N_ROWS, N_COLS), "batch", "hough", geometry=geometry)
    with h5py.File(path, "a") as f:
        if "metadata" not in f:
            f.create_group("metadata").attrs["grid_shape"] = [N_ROWS, N_COLS]
        aa = f.create_group("auto_assignment")
        aa.create_dataset("best_phase_id", data=_field(*PAIR) if phase_ids is None else phase_ids)
        aa.create_dataset("best_ci", data=np.ones((N_ROWS, N_COLS), dtype=np.float32))
        aa.create_dataset("uncertainty", data=np.zeros((N_ROWS, N_COLS), dtype=np.float32))
    return path


def _cleaned_in(path):
    import h5py
    with h5py.File(path, "r") as f:
        aa = f["auto_assignment"]
        return np.array(aa["cleaned_phase_id"]) if "cleaned_phase_id" in aa else None


def _apply(client, tmp_path, target):
    return client.post("/api/phasemap/apply-cleanup", json={
        "file_stem": "scan", "search_dir": str(tmp_path), "target": target, "min_cluster_size": 2})


HEX_SCAN = ScanGeometry.hex(N_ROWS, N_COLS, STEP)             # odd rows +0.5, like HEX


@pytest.mark.parametrize("target, active", [("checkpoint", HEX), ("checkpoint", SQUARE), ("auto", SQUARE)],
                         ids=["checkpoint, hex result active", "checkpoint, square result active",
                              "auto, square result active"])
def test_apply_cleanup_route_hex_checkpoint_six_neighbours(client, monkeypatch, tmp_path, target, active):
    """The checkpoint is cleaned on the lattice IT was written for. PAIR = (1, 1) and (2, 2): on the 6 x 5 lattice with
    the odd rows shifted +0.5 the neighbours of (2, 2) are (2, 1), (2, 3), (1, 1), (1, 2), (3, 1), (3, 2) (even row 2:
    x = col; odd rows 1 and 3: x = col + 0.5, so x = 2 has the columns 1 and 2 half a step either side), which
    includes (1, 1): the two form one cluster of two and survive a minimum cluster size of 2. With four neighbours
    they are two clusters of one and would be removed."""
    _arm(monkeypatch, _result(_field(*PAIR), active))
    path = _checkpoint(tmp_path, geometry=HEX_SCAN)
    r = _apply(client, tmp_path, target)
    assert r.status_code == 200, r.text
    assert r.json()["target"] == "checkpoint"
    cleaned = _cleaned_in(path)
    assert cleaned[1, 1] == cleaned[2, 2] == 1


def test_auto_cleans_a_hexagonal_result_in_memory_even_when_a_checkpoint_exists(client, monkeypatch, tmp_path):
    """Nothing ties a checkpoint to the active result (the page sends the stem of its last batch): with a hexagonal
    result on screen `auto` cleans that result, in memory, and leaves the file alone."""
    fake = _result(_field(*PAIR), HEX)
    _arm(monkeypatch, fake)
    path = _checkpoint(tmp_path)
    assert "cleaned_phase_id" not in fake.metadata                                                    # premise
    r = _apply(client, tmp_path, "auto")
    assert r.status_code == 200, r.text
    assert r.json()["target"] == "result"
    assert fake.metadata["cleaned_phase_id"][1, 1] == fake.metadata["cleaned_phase_id"][2, 2] == 1   # six neighbours
    assert _cleaned_in(path) is None                                                                  # file untouched


def test_an_explicit_checkpoint_takes_the_file_by_its_own_stamp_while_a_hexagonal_result_is_active(
        client, monkeypatch, tmp_path):
    """A checkpoint without a stamp is a square scan: four neighbours, the pair is removed, although the active
    result is hexagonal. The in-memory result is not touched."""
    fake = _result(_field(*PAIR), HEX)
    _arm(monkeypatch, fake)
    path = _checkpoint(tmp_path)
    r = _apply(client, tmp_path, "checkpoint")
    assert r.status_code == 200 and r.json()["target"] == "checkpoint", r.text
    assert _cleaned_in(path)[1, 1] == _cleaned_in(path)[2, 2] == 0
    assert "cleaned_phase_id" not in fake.metadata


def test_a_hexagonal_checkpoints_neighbours_are_the_literal_six_of_the_brief():
    """3 x 5 lattice, odd rows shifted +0.5, read from the stamp the way the cleanup reads it. Row 1 is odd: x = col +
    0.5, so the point (1, 2) at x = 2.5 has in row 0 (x = col) the columns 2 and 3 at x = 2 and 3, in row 2 the same,
    and in its own row the columns 1 and 3."""
    topology = topology_of_stamp(ScanGeometry.hex(3, 5, STEP).to_dict(), (3, 5))
    got = {divmod(int(i), 5) for i in topology.neighbours(1 * 5 + 2)}
    assert got == {(0, 2), (0, 3), (1, 1), (1, 3), (2, 2), (2, 3)}


def test_the_second_pass_of_a_hexagonal_checkpoint_votes_in_the_hexagonal_window(client, monkeypatch, tmp_path):
    """Modal filter of size 3 at (2, 2), by hand. Hexagonal window: the point and (2, 1), (2, 3), (1, 1), (1, 2),
    (3, 1), (3, 2). Phase 2 sits at (1, 1) and (3, 1), phase 1 at (1, 3) and (3, 3), which the hexagonal window does
    not hold but the square 3 x 3 box does: hexagonal votes 2 : 0 for phase 2, the box 2 : 2 and then the lower id,
    phase 1."""
    _arm(monkeypatch, None)
    ids = np.full((N_ROWS, N_COLS), -1, dtype=np.int16)
    ids[1, 1] = ids[3, 1] = 2
    ids[1, 3] = ids[3, 3] = 1
    path = _checkpoint(tmp_path, geometry=HEX_SCAN, phase_ids=ids)
    r = client.post("/api/phasemap/apply-cleanup", json={
        "file_stem": "scan", "search_dir": str(tmp_path), "target": "checkpoint", "modal_filter_size": 3})
    assert r.status_code == 200, r.text
    assert _cleaned_in(path)[2, 2] == 2


def test_the_second_pass_of_a_hexagonal_checkpoint_refuses_an_oversize_modal_window(client, monkeypatch, tmp_path):
    _arm(monkeypatch, None)
    path = _checkpoint(tmp_path, geometry=HEX_SCAN)
    r = client.post("/api/phasemap/apply-cleanup", json={
        "file_stem": "scan", "search_dir": str(tmp_path), "target": "checkpoint", "modal_filter_size": 9})
    assert r.status_code == 400, r.text
    assert "modal" in r.json()["detail"]
    assert _cleaned_in(path) is None


def test_a_checkpoint_stamp_that_cannot_be_read_is_answered_with_400_and_writes_nothing(client, monkeypatch, tmp_path):
    import h5py
    _arm(monkeypatch, None)
    path = _checkpoint(tmp_path, geometry=HEX_SCAN)
    with h5py.File(path, "a") as f:
        del f["metadata"].attrs["odd_row_shift"]
    r = _apply(client, tmp_path, "checkpoint")
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert "odd_row_shift" in detail
    assert "checkpoint" in detail and "was not changed" in detail and "load the file again" not in detail
    assert _cleaned_in(path) is None


def test_the_writer_reads_the_stamp_itself(tmp_path):
    """No argument carries the topology: ``apply_cleanup`` of a hexagonal checkpoint keeps the diagonal pair."""
    path = _checkpoint(tmp_path, geometry=HEX_SCAN)
    cw = CheckpointWriter(str(tmp_path / "scan.h5oina"))
    assert cw.checkpoint_path == str(path)
    cw.apply_cleanup(min_cluster_size=2)
    assert _cleaned_in(path)[1, 1] == _cleaned_in(path)[2, 2] == 1


@pytest.mark.parametrize("target", ["auto", "checkpoint"])
@pytest.mark.parametrize("stamp", [SQUARE, None], ids=["square", "unknown"])
def test_a_square_or_unknown_result_still_cleans_the_checkpoint(client, monkeypatch, tmp_path, target, stamp):
    _arm(monkeypatch, _result(_field(*PAIR), stamp))
    path = _checkpoint(tmp_path)
    r = _apply(client, tmp_path, target)
    assert r.status_code == 200, r.text
    assert r.json()["target"] == "checkpoint"
    cleaned = _cleaned_in(path)
    assert cleaned[1, 1] == cleaned[2, 2] == 0                    # two 4-neighbour specks of one point each: removed
