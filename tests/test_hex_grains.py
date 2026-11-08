"""Grains of a hexagonal scan: segmentation on the six neighbours, and the grain-stabilised IPF colouring."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.routes import phase_map  # noqa: E402
from backend.api.services.scan_geometry import ScanGeometry  # noqa: E402
from backend.api.services.scan_neighbours import Topology  # noqa: E402
from backend.spherical_gpu.pipeline import variant_unification as vu  # noqa: E402
from backend.spherical_gpu.pipeline.variant_unification import segment_supergroup_grains  # noqa: E402
from tests.test_phase_map_layer_endpoint import _make_fake_result  # noqa: E402

SIGNS = (0.5, -0.5)


def _qz(deg):
    a = np.radians(deg) / 2.0
    return np.array([np.cos(a), 0.0, 0.0, np.sin(a)])


def _field(n_rows, n_cols, rotated):
    """Quaternions of a field of one orientation with the cells in ``rotated`` turned 30 degrees about z."""
    q = np.tile(_qz(0.0), (n_rows * n_cols, 1))
    for r, c in rotated:
        q[r * n_cols + c] = _qz(30.0)
    return q


def _lattice(shift, n_rows, n_cols):
    g = ScanGeometry.hex(n_rows, n_cols, 1.0, odd_row_shift=shift)
    x, y = g.positions()
    pts = np.stack([x.ravel(), y.ravel()], axis=1)
    return np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])


# 4 x 4, offset rows: with +0.5 the point (1, 1) touches (2, 1) and (2, 2); with -0.5 it touches (2, 0) and (2, 1)
@pytest.mark.parametrize("shift, pair", [(0.5, ((1, 1), (2, 2))), (-0.5, ((1, 1), (2, 0)))])
def test_a_diagonal_pair_is_one_grain_on_a_hexagonal_scan_and_two_on_the_index_grid(shift, pair):
    q = _field(4, 4, pair)
    phase = np.zeros(16, dtype=np.int64)
    a, b = (r * 4 + c for r, c in pair)
    hexagonal = segment_supergroup_grains(q, phase, 4, 4, 0, "m-3m", 5.0, topology=Topology("hex", 4, 4, shift))
    legacy = segment_supergroup_grains(q, phase, 4, 4, 0, "m-3m", 5.0)
    assert hexagonal[a] == hexagonal[b] != hexagonal[0]
    assert legacy[a] != legacy[b]


def _partition(labels):
    """The same grouping with the labels renamed in order of first appearance."""
    seen, out = {}, []
    for v in labels.tolist():
        out.append(seen.setdefault(v, len(seen)) if v >= 0 else -1)
    return out


def _oracle_partition(classes, dist):
    """Components of equal class one step apart on the sample, by breadth first search."""
    n = classes.size
    label = np.full(n, -1)
    next_label = 0
    for start in range(n):
        if label[start] >= 0:
            continue
        label[start] = next_label
        frontier = [start]
        while frontier:
            p = frontier.pop()
            for q in np.flatnonzero((label < 0) & (classes == classes[p]) & (dist[p] > 1e-9) & (dist[p] <= 1 + 1e-6)):
                label[q] = next_label
                frontier.append(q)
        next_label += 1
    return _partition(label)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("shift", SIGNS)
def test_the_grains_are_the_components_of_points_one_step_apart(shift, seed):
    n_rows, n_cols = 7, 6
    classes = np.random.default_rng(seed).integers(0, 2, size=n_rows * n_cols)
    q = np.stack([_qz(30.0 * k) for k in classes])
    labels = segment_supergroup_grains(
        q, np.zeros(q.shape[0], dtype=np.int64), n_rows, n_cols, 0, "m-3m", 5.0,
        topology=Topology("hex", n_rows, n_cols, shift))
    assert _partition(labels) == _oracle_partition(classes, _lattice(shift, n_rows, n_cols))


def test_a_square_topology_segments_like_the_index_grid():
    classes = np.random.default_rng(3).integers(0, 2, size=30)
    q = np.stack([_qz(30.0 * k) for k in classes])
    phase = np.zeros(30, dtype=np.int64)
    np.testing.assert_array_equal(
        segment_supergroup_grains(q, phase, 5, 6, 0, "m-3m", 5.0, topology=Topology("square", 5, 6)),
        segment_supergroup_grains(q, phase, 5, 6, 0, "m-3m", 5.0))


def test_a_topology_of_another_grid_is_refused():
    q = _field(4, 4, [])
    with pytest.raises(ValueError, match="grid"):
        segment_supergroup_grains(q, np.zeros(16, dtype=np.int64), 4, 4, 0, "m-3m", 5.0,
                                  topology=Topology("hex", 3, 4, 0.5))


def test_a_topology_of_the_transposed_grid_is_refused():
    # 4 x 6 and 6 x 4 hold the same number of points but not the same neighbours
    q = _field(6, 4, [])
    with pytest.raises(ValueError, match="grid"):
        segment_supergroup_grains(q, np.zeros(24, dtype=np.int64), 6, 4, 0, "m-3m", 5.0,
                                  topology=Topology("hex", 4, 6, 0.5))


@pytest.mark.parametrize("block", [1, 5, 1000])
@pytest.mark.parametrize("shift", SIGNS)
def test_the_pairs_are_reduced_in_blocks_and_the_grains_stay_the_same(monkeypatch, shift, block):
    n_rows, n_cols = 7, 6
    classes = np.random.default_rng(2).integers(0, 2, size=n_rows * n_cols)
    q = np.stack([_qz(30.0 * k) for k in classes])
    topology = Topology("hex", n_rows, n_cols, shift)
    sizes = []
    original = vu.pair_orientation_angle_deg

    def spy(qa, qb, sym):
        sizes.append(len(qa))
        return original(qa, qb, sym)

    monkeypatch.setattr(vu, "_PAIR_BLOCK", block)
    monkeypatch.setattr(vu, "pair_orientation_angle_deg", spy)
    labels = segment_supergroup_grains(q, np.zeros(q.shape[0], dtype=np.int64), n_rows, n_cols, 0, "m-3m", 5.0,
                                       topology=topology)
    assert _partition(labels) == _oracle_partition(classes, _lattice(shift, n_rows, n_cols))
    assert max(sizes) <= block
    assert sum(sizes) == topology.pairs()[0].size       # every pair is reduced once (all points are indexed)
    if block == 1:
        assert len(sizes) == topology.pairs()[0].size


# ------------------------------------------------------ the grain-stabilised IPF colouring


def _xmap(quats, nr, nc, pg="m-3"):
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    x, y = np.meshgrid(np.arange(nc, dtype=float), np.arange(nr, dtype=float))
    return CrystalMap(rotations=Rotation(quats), phase_id=np.zeros(nr * nc, int),
                      x=x.ravel(), y=y.ravel(),
                      phase_list=PhaseList(Phase(name="a", point_group=pg)))


def test_the_colouring_segments_on_the_topology_it_is_given(monkeypatch):
    from tools.phase_map_generator import compute_ipf_colors_grain_consistent
    seen = []
    original = vu.segment_supergroup_grains

    def spy(*args, **kwargs):
        seen.append(kwargs.get("topology"))
        return original(*args, **kwargs)

    monkeypatch.setattr(vu, "segment_supergroup_grains", spy)
    topology = Topology("hex", 4, 4, 0.5)
    xm = _xmap(_field(4, 4, []), 4, 4)
    compute_ipf_colors_grain_consistent(xm, "Z", n_rows=4, n_cols=4, mask=None, topology=topology)
    compute_ipf_colors_grain_consistent(xm, "Z", n_rows=4, n_cols=4, mask=None)
    assert seen == [topology, None]


def test_the_colouring_refuses_a_topology_of_another_grid_instead_of_keeping_the_baseline():
    from tools.phase_map_generator import compute_ipf_colors_grain_consistent
    xm = _xmap(_field(4, 4, []), 4, 4)
    with pytest.raises(ValueError, match="grid"):
        compute_ipf_colors_grain_consistent(xm, "Z", n_rows=4, n_cols=4, mask=None,
                                            topology=Topology("hex", 3, 4, 0.5))


def test_the_colouring_refuses_a_topology_of_the_transposed_grid():
    from tools.phase_map_generator import compute_ipf_colors_grain_consistent
    xm = _xmap(_field(6, 4, []), 6, 4)
    with pytest.raises(ValueError, match="grid"):
        compute_ipf_colors_grain_consistent(xm, "Z", n_rows=6, n_cols=4, mask=None,
                                            topology=Topology("hex", 4, 6, 0.5))


# A hexagonal scan of 30 x 24 points in which every lattice line along one diagonal is one grain: the orientation
# of line k is a 7.5 degree turn about z per step of k, applied to a direction near the edge of the colour key's
# sector, so a grain split by the key shows as a jump of colour. The points of a line touch only along the
# diagonal of the lattice, which the four neighbours of the index grid do not see.
def _lines_scan(shift, roi):
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    from scipy.spatial.transform import Rotation as SciRotation

    n_rows, n_cols = 30, 24
    x, y = ScanGeometry.hex(n_rows, n_cols, 1.0, odd_row_shift=shift).positions()
    line = np.round(np.asarray(x).ravel() - np.sign(shift) * np.asarray(y).ravel() / np.sqrt(3.0)).astype(int)
    edge = np.array([0.5, 1.0, 1.0]) / np.linalg.norm([0.5, 1.0, 1.0])
    to_edge = SciRotation.align_vectors([edge], [[0.0, 0.0, 1.0]])[0]
    quats = []
    for k in line % 12:
        w_last = (to_edge * SciRotation.from_rotvec([0.0, 0.0, np.radians(7.5 * k)])).as_quat()   # x, y, z, w
        quats.append(np.roll(w_last, 1))
    quats = np.array(quats)
    rng = np.random.default_rng(0)
    for i in range(quats.shape[0]):                       # about one degree of noise on every point
        axis = rng.normal(size=3)
        half = np.radians(rng.normal(scale=1.0)) / 2.0
        noise = np.array([np.cos(half), *(np.sin(half) * axis / np.linalg.norm(axis))])
        w1, v1, w2, v2 = noise[0], noise[1:], quats[i, 0], quats[i, 1:]
        quats[i] = np.array([w1 * w2 - v1 @ v2, *(w1 * v2 + w2 * v1 + np.cross(v1, v2))])
        quats[i] /= np.linalg.norm(quats[i])
    mask = None
    keep = np.ones(n_rows * n_cols, dtype=bool)
    if roi:                                               # the last four columns lie outside the region
        mask = np.ones((n_rows, n_cols), dtype=bool)
        mask[:, 20:] = False
        keep = mask.ravel()
    xm = CrystalMap(rotations=Rotation(quats[keep]), phase_id=np.zeros(int(keep.sum()), int),
                    x=np.tile(np.arange(n_cols, dtype=float), n_rows)[keep],
                    y=np.repeat(np.arange(n_rows, dtype=float), n_cols)[keep],
                    phase_list=PhaseList(Phase(name="a", point_group="m-3")))
    return xm, Topology("hex", n_rows, n_cols, shift), line, keep, mask


def _spread_per_line(rgb, line, keep):
    """Per lattice line, the largest distance of a point's colour from the median colour of the line."""
    flat = rgb.reshape(-1, 3)
    out = []
    for value in np.unique(line):
        pick = (line == value) & keep
        if pick.sum() >= 2:
            out.append(np.abs(flat[pick] - np.median(flat[pick], axis=0)).max())
    return np.array(out)


@pytest.mark.parametrize("roi", [False, True], ids=["full", "roi"])
@pytest.mark.parametrize("shift", SIGNS)
def test_the_stabilised_colouring_gives_every_line_of_a_hexagonal_scan_one_colour(shift, roi):
    from tools.phase_map_generator import compute_ipf_colors_grain_consistent
    xm, topology, line, keep, mask = _lines_scan(shift, roi)
    hexagonal = compute_ipf_colors_grain_consistent(xm, "Z", n_rows=30, n_cols=24, mask=mask, topology=topology)
    four_neighbours = compute_ipf_colors_grain_consistent(xm, "Z", n_rows=30, n_cols=24, mask=mask)
    spread_hex = _spread_per_line(hexagonal, line, keep)
    spread_four = _spread_per_line(four_neighbours, line, keep)
    # the set-up can fail: on the four neighbours of the index grid most lines come out in two colours
    assert (spread_four > 0.2).sum() > spread_four.size // 2
    # on the six neighbours of the scan none does
    assert spread_hex.max() < 0.2


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("stamp, hexagonal", [
    (ScanGeometry.hex(6, 5, 0.5).to_dict(), True),
    (ScanGeometry.square(6, 5, 0.5).to_dict(), False),
    (None, False),
], ids=["hex", "square", "unknown"])
def test_the_layer_route_hands_the_scans_topology_to_the_stabilised_colouring(client, monkeypatch, stamp, hexagonal):
    fake = _make_fake_result(n_rows=6, n_cols=5)
    fake.metadata = {"scan_geometry": stamp} if stamp is not None else {}
    monkeypatch.setattr("backend.api.routes.phase_map.get_last_indexing_result", lambda: fake)
    monkeypatch.setattr("backend.api.routes.phase_map.get_analysis_dataset", lambda: None)
    seen = []

    def stub(xmap, direction, **kwargs):
        seen.append(kwargs.get("topology"))
        return np.zeros((6, 5, 3))

    monkeypatch.setattr("tools.phase_map_generator.compute_ipf_colors_grain_consistent", stub)
    r = client.get("/api/phasemap/layer", params={"kind": "ipf-z", "grain_stabilized": "true"})
    assert r.status_code == 200, r.text
    assert len(seen) == 1
    if hexagonal:
        assert (seen[0].kind, seen[0].n_rows, seen[0].n_cols, seen[0].odd_row_shift) == ("hex", 6, 5, 0.5)
    else:
        assert seen[0] is None


# The route draws the analysis dataset when no result holds a crystal map, and the gate used to look there too:
# the colouring must take its topology from the same source (`phase_map._hex_topology_of(result, dataset, ...)`).
def _stub_colouring(monkeypatch):
    seen = []

    def stub(xmap, direction, **kwargs):
        seen.append(kwargs.get("topology"))
        return np.zeros((6, 5, 3))

    monkeypatch.setattr("tools.phase_map_generator.compute_ipf_colors_grain_consistent", stub)
    return seen


def _dataset(stamp, fake):
    from types import SimpleNamespace
    return SimpleNamespace(xmap=fake.xmap, shape=(6, 5), scan_geometry=stamp, per_phase_data={})


@pytest.mark.parametrize("dataset_stamp, result_stamp, hexagonal", [
    (ScanGeometry.hex(6, 5, 0.5).to_dict(), "no result", True),            # no result: the dataset is drawn
    (ScanGeometry.square(6, 5, 0.5).to_dict(), "no result", False),
    (ScanGeometry.hex(6, 5, 0.5).to_dict(), ScanGeometry.square(6, 5, 0.5).to_dict(), False),   # the result wins
    (ScanGeometry.square(6, 5, 0.5).to_dict(), ScanGeometry.hex(6, 5, 0.5).to_dict(), True),
], ids=["dataset-hex", "dataset-square", "square-result-over-hex-dataset", "hex-result-over-square-dataset"])
def test_the_layer_route_takes_the_topology_from_the_source_it_draws(client, monkeypatch, dataset_stamp,
                                                                     result_stamp, hexagonal):
    fake = _make_fake_result(n_rows=6, n_cols=5)
    if result_stamp == "no result":
        result = None
    else:
        fake.metadata = {"scan_geometry": result_stamp}
        result = fake
    monkeypatch.setattr("backend.api.routes.phase_map.get_last_indexing_result", lambda: result)
    monkeypatch.setattr("backend.api.routes.phase_map.get_analysis_dataset", lambda: _dataset(dataset_stamp, fake))
    seen = _stub_colouring(monkeypatch)
    r = client.get("/api/phasemap/layer", params={"kind": "ipf-z", "grain_stabilized": "true"})
    assert r.status_code == 200, r.text
    assert len(seen) == 1
    assert (seen[0] is not None) is hexagonal


@pytest.mark.parametrize("stamp", [ScanGeometry.hex(6, 5, 0.5).to_dict(), None], ids=["hex", "square"])
def test_the_layer_route_answers_an_error_when_the_stabilised_colouring_runs_out_of_memory(client, monkeypatch, stamp):
    fake = _make_fake_result(n_rows=6, n_cols=5)
    fake.metadata = {"scan_geometry": stamp} if stamp is not None else {}
    monkeypatch.setattr("backend.api.routes.phase_map.get_last_indexing_result", lambda: fake)
    monkeypatch.setattr("backend.api.routes.phase_map.get_analysis_dataset", lambda: None)

    def colouring(*args, **kwargs):
        raise MemoryError("Unable to allocate 1.8 GiB")

    monkeypatch.setattr("tools.phase_map_generator.compute_ipf_colors_grain_consistent", colouring)
    r = client.get("/api/phasemap/layer", params={"kind": "ipf-z", "grain_stabilized": "true"})
    assert r.status_code == 500
