"""The pseudo-symmetry variant unification on the six neighbours of a hexagonal scan."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services.scan_geometry import ScanGeometry  # noqa: E402
from backend.api.services.scan_neighbours import Topology  # noqa: E402
from backend.spherical_gpu.pipeline import variant_unification as vu  # noqa: E402
from backend.spherical_gpu.pipeline.variant_unification import (  # noqa: E402
    arbitrate_resolver_by_render, proper_coset_class_reps, unify_after_hough_resolve, unify_map,
)
from backend.spherical_gpu.pseudosym import _qmul, same_orientation_angle_deg  # noqa: E402

SIGNS = (0.5, -0.5)


def _qaa(axis, deg):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    a = np.radians(deg) / 2.0
    return np.array([np.cos(a), *(np.sin(a) * axis)])


def _ang(qa, qb, pg="m-3"):
    return float(same_orientation_angle_deg(np.atleast_2d(qa), np.asarray(qb), pg)[0])


def _speckled(nr, nc, seed=7):
    """One grain with a slight bend, pixels randomly in one of the two m-3 variant classes."""
    rng = np.random.default_rng(seed)
    reps = proper_coset_class_reps("m-3")
    truth = np.stack([_qmul(_qaa([1, 0, 0], 0.05 * (r + c))[None, :], _qaa([0, 0, 1], 5.0)[None, :])[0]
                      for r in range(nr) for c in range(nc)])
    q = truth.copy()
    flip = rng.integers(0, 2, nr * nc) == 1
    q[flip] = _qmul(reps[1][None, :], truth[flip])
    return q, truth


def _truth_scorer(truth, good=0.5, bad=0.2):
    def score(flat_idx, quats):
        return np.asarray([good if _ang(qq, truth[int(f)]) < 5.0 else bad
                           for f, qq in zip(flat_idx, np.atleast_2d(quats))])
    return score


# ----------------------------------------------- the grains, the contact census and the rescue use six neighbours

@pytest.mark.parametrize("shift", SIGNS)
def test_an_orphan_joins_the_grain_most_of_its_six_neighbours_belong_to(shift):
    # One point X between two big grains, A to its left and B to its right on the sample, and the point below it
    # in the index grid unindexed. Counting the four neighbours of the index grid and counting all six give
    # different winners (the premise is computed below from the positions, for either sign of the offset).
    nr = nc = 8
    g = ScanGeometry.hex(nr, nc, 1.0, odd_row_shift=shift)
    x, y = g.positions()
    xr, xc = 3, 3
    qa = _qaa([0, 0, 1], 5.0)
    qb = _qmul(_qaa([2, 1, 3], 35.0)[None, :], qa[None, :])[0]
    belongs = np.where(x < x[xr, xc] - 1e-9, "A", "B")
    belongs[(np.abs(x - x[xr, xc]) < 1e-9) & (np.arange(nr)[:, None] < xr)] = "A"
    belongs[xr + 1, xc] = "-"                                           # unindexed
    q = np.where(belongs[..., None] == "A", qa, qb).reshape(nr * nc, 4).astype(float)
    q[xr * nc + xc] = _qaa([1, 1, 1], 50.0)                              # X itself: an orphan of its own
    q[(xr + 1) * nc + xc] = np.nan
    phase = np.where(belongs == "-", -1, 0).reshape(-1)

    def winner(cells):
        votes = {"A": 0, "B": 0}
        for r, c in cells:
            if 0 <= r < nr and 0 <= c < nc and (r, c) != (xr, xc) and belongs[r, c] in votes:
                votes[belongs[r, c]] += 1
        assert votes["A"] != votes["B"]
        return max(votes, key=votes.get)

    six = [(r, c) for r in range(nr) for c in range(nc) if (r, c) != (xr, xc)
           and abs(np.hypot(x[r, c] - x[xr, xc], y[r, c] - y[xr, xc]) - 1.0) < 1e-6]
    four = [(xr, xc - 1), (xr, xc + 1), (xr - 1, xc), (xr + 1, xc)]
    assert winner(six) != winner(four)                                   # the premise
    flat = lambda i, qq: np.full(len(np.atleast_1d(i)), 0.3)             # no score decides anything
    hexagonal, rep_hex = unify_map(q, phase, nr, nc, 0, "m-3", flat, topology=Topology("hex", nr, nc, shift))
    legacy, rep_old = unify_map(q, phase, nr, nc, 0, "m-3", flat)
    assert rep_hex["n_rescued"] == rep_old["n_rescued"] == 1
    donor = {"A": qa, "B": qb}
    assert _ang(hexagonal[xr * nc + xc], donor[winner(six)]) < 1e-6
    assert _ang(legacy[xr * nc + xc], donor[winner(four)]) < 1e-6


@pytest.mark.parametrize("shift", SIGNS)
@pytest.mark.parametrize("seed", range(4))
def test_a_fifty_fifty_mix_of_the_two_classes_is_still_speckle_on_a_hexagonal_scan(shift, seed):
    # On the six-neighbour lattice a random 50/50 mix percolates (site threshold 0.5, against 0.593 for four
    # neighbours): its clusters, counted with six neighbours, would make a big grain look like coherent twin
    # domains (measured on 20 x 20 grains: 98 of 100 against 2 of 100). The speckle-or-twin policy was set on the
    # four neighbours of the index grid, which are true neighbours on a hexagonal scan too, and keeps them.
    nr, nc = 12, 12
    q, truth = _speckled(nr, nc, seed=seed)
    new_q, report = unify_map(q, np.zeros(nr * nc, int), nr, nc, 0, "m-3", _truth_scorer(truth),
                              topology=Topology("hex", nr, nc, shift))
    assert [g["mode"] for g in report["grains"]] == ["speckle"]
    assert max(_ang(new_q[i], truth[i]) for i in range(nr * nc)) < 0.2


@pytest.mark.parametrize("shift", SIGNS)
def test_a_speckled_hexagonal_grain_is_unified_and_keeps_its_texture(shift):
    nr, nc = 6, 8
    q, truth = _speckled(nr, nc)
    new_q, report = unify_map(q, np.zeros(nr * nc, int), nr, nc, 0, "m-3", _truth_scorer(truth),
                              topology=Topology("hex", nr, nc, shift))
    assert new_q is not None and report["n_grains"] == 1 and report["n_flipped_units"] == 1
    assert max(_ang(new_q[i], truth[i]) for i in range(nr * nc)) < 0.2


@pytest.mark.parametrize("shift", SIGNS)
def test_a_blob_of_a_second_basin_is_adopted_on_a_hexagonal_scan(shift):
    nr, nc = 8, 8
    qa = _qaa([0, 0, 1], 5.0)
    bad = _qaa([2, 1, 3], 35.0)
    q = np.tile(qa, (nr * nc, 1))
    blob = np.asarray([r * nc + c for r, c in ((3, 3), (3, 4), (4, 3), (4, 4), (2, 3))])
    q[blob] = _qmul(bad[None, :], qa[None, :])[0]
    new_q, report = unify_map(q, np.zeros(nr * nc, int), nr, nc, 0, "m-3", _truth_scorer(np.tile(qa, (nr * nc, 1))),
                              topology=Topology("hex", nr, nc, shift))
    assert report["n_adopted"] == 1
    assert max(_ang(new_q[f], qa) for f in blob) < 0.5


# A small grain on the physical boundary of two big ones: grain A on the left, grain B on the right of a vertical
# line at x = XB. The score prefers A, so the grain that is chosen as the donor decides whether the small grain is
# adopted (donor A) or kept (donor B). Counting the four neighbours of the index grid and counting the six neighbours
# on the sample give different donors in these cases (the premise is computed below from the positions).
#   (shift, XB, the small grain as (row, column) of the index grid)
_BOUNDARY_BLOBS = [
    (0.5, 6.310, [(4, 6), (3, 5), (2, 6), (2, 5)]),
    (0.5, 6.753, [(4, 7), (3, 6), (5, 6), (3, 7), (5, 7)]),
    (-0.5, 6.273, [(3, 6), (4, 5), (2, 6), (2, 7), (4, 6)]),
    (-0.5, 6.070, [(7, 7), (8, 6), (6, 6), (8, 7)]),
]


@pytest.mark.parametrize("shift, xb, blob", _BOUNDARY_BLOBS, ids=[f"{s:+}-{len(b)}pts" for s, _, b in _BOUNDARY_BLOBS])
def test_a_small_grain_on_a_boundary_is_adopted_through_its_six_neighbours_not_the_four_of_the_index_grid(
        shift, xb, blob):
    nr = nc = 14
    g = ScanGeometry.hex(nr, nc, 1.0, odd_row_shift=shift)
    x, y = g.positions()
    qa = _qaa([0, 0, 1], 5.0)
    qb = _qmul(_qaa([1, 0, 0], 60.0)[None, :], qa[None, :])[0]
    qs = _qmul(_qaa([2, 1, 3], 35.0)[None, :], qa[None, :])[0]
    big = np.where(x < xb, 0, 1)                                          # 0: grain A, 1: grain B
    q = np.where(big[..., None] == 0, qa, qb).astype(float)
    for r, c in blob:
        q[r, c] = qs

    def census(cells_of):
        votes = [0, 0]
        for r, c in blob:
            for rr, cc in cells_of(r, c):
                if 0 <= rr < nr and 0 <= cc < nc and (rr, cc) not in blob:
                    votes[big[rr, cc]] += 1
        assert votes[0] != votes[1]
        return int(votes[1] > votes[0])                                    # the donor: 0 = A, 1 = B

    six = lambda r, c: [(a, b) for a in range(nr) for b in range(nc) if (a, b) != (r, c)
                        and abs(np.hypot(x[a, b] - x[r, c], y[a, b] - y[r, c]) - 1.0) < 1e-6]
    four = lambda r, c: [(r, c - 1), (r, c + 1), (r - 1, c), (r + 1, c)]
    assert census(six) != census(four)                                     # the premise
    donor_hex = census(six)

    def score(flat, quats):                                                # the truth is A's orientation
        return np.asarray([0.5 if _ang(qq, qa) < 5.0 else 0.2 for qq in np.atleast_2d(quats)])

    new_q, report = unify_map(q.reshape(-1, 4), np.zeros(nr * nc, int), nr, nc, 0, "m-3", score,
                              topology=Topology("hex", nr, nc, shift))
    flat = [r * nc + c for r, c in blob]
    if donor_hex == 0:                                                     # donor A renders better: adopted
        assert report["n_adopted"] == 1
        assert max(_ang(new_q[f], qa) for f in flat) < 1.0
    else:                                                                  # donor B renders worse: kept as it was
        assert report["n_adopted"] == 0
        weighed = [e["decision"] for e in report["grains"] if e["mode"] == "adoption"]            # it was weighed
        assert len(weighed) == 1 and weighed[0].startswith("kept")
        assert max(_ang(new_q[f], qs) for f in flat) < 1.0


@pytest.mark.parametrize("shift", SIGNS)
def test_a_tiny_orphan_takes_the_orientation_of_a_neighbouring_grain_on_a_hexagonal_scan(shift):
    nr, nc = 6, 6
    qa = _qaa([0, 0, 1], 5.0)
    bad = _qmul(_qaa([2, 1, 3], 35.0)[None, :], qa[None, :])[0]
    q = np.tile(qa, (nr * nc, 1))
    q[2 * nc + 2] = bad                                         # one point, surrounded by the grain
    new_q, report = unify_map(q, np.zeros(nr * nc, int), nr, nc, 0, "m-3", _truth_scorer(np.tile(qa, (nr * nc, 1))),
                              topology=Topology("hex", nr, nc, shift))
    assert report["n_rescued"] == 1 and _ang(new_q[2 * nc + 2], qa) < 0.5


# ----------------------------------------------- a square topology is the code that was

def _same(a, b):
    if a is None or b is None:
        return a is b
    return np.allclose(a, b, atol=0, equal_nan=True)


def test_a_square_topology_unifies_exactly_like_the_index_grid():
    nr, nc = 6, 8
    q, truth = _speckled(nr, nc)
    for scorer in (_truth_scorer(truth), lambda i, qq: np.full(len(np.atleast_1d(i)), 0.3)):
        old = unify_map(q, np.zeros(nr * nc, int), nr, nc, 0, "m-3", scorer)
        new = unify_map(q, np.zeros(nr * nc, int), nr, nc, 0, "m-3", scorer, topology=Topology("square", nr, nc))
        assert _same(old[0], new[0]) and old[1] == new[1]


def test_a_square_topology_adopts_and_rescues_exactly_like_the_index_grid():
    nr, nc = 8, 8
    qa = _qaa([0, 0, 1], 5.0)
    bad = _qmul(_qaa([2, 1, 3], 35.0)[None, :], qa[None, :])[0]
    q = np.tile(qa, (nr * nc, 1))
    for r, c in ((3, 3), (3, 4), (4, 3), (4, 4), (2, 3), (6, 6)):
        q[r * nc + c] = bad
    scorer = _truth_scorer(np.tile(qa, (nr * nc, 1)))
    old = unify_map(q, np.zeros(nr * nc, int), nr, nc, 0, "m-3", scorer)
    new = unify_map(q, np.zeros(nr * nc, int), nr, nc, 0, "m-3", scorer, topology=Topology("square", nr, nc))
    assert _same(old[0], new[0]) and old[1] == new[1]
    assert old[1]["n_adopted"] == 1 and old[1]["n_rescued"] == 1        # (so that the comparison says something)


# ----------------------------------------------- the topology is handed on

@pytest.fixture
def seen(monkeypatch):
    got = []
    original = vu.segment_supergroup_grains

    def spy(*args, **kwargs):
        got.append(kwargs.get("topology"))
        return original(*args, **kwargs)

    monkeypatch.setattr(vu, "segment_supergroup_grains", spy)
    return got


def test_the_arbitration_segments_on_the_given_topology(seen):
    nr, nc = 4, 4
    q = np.tile(_qaa([0, 0, 1], 5.0), (nr * nc, 1))
    topology = Topology("hex", nr, nc, 0.5)
    arbitrate_resolver_by_render(q, q.copy(), np.zeros(nr * nc, int), nr, nc, 0, "m-3",
                                 lambda i, qq: np.zeros(len(np.atleast_1d(i))), topology=topology)
    assert seen == [topology]


def test_the_unification_segments_on_the_given_topology(seen):
    nr, nc = 4, 4
    q = np.tile(_qaa([0, 0, 1], 5.0), (nr * nc, 1))
    topology = Topology("hex", nr, nc, 0.5)
    unify_map(q, np.zeros(nr * nc, int), nr, nc, 0, "m-3", lambda i, qq: np.zeros(len(np.atleast_1d(i))),
              topology=topology)
    assert seen == [topology]


def _entry_arguments(n_rows=4, n_cols=4):
    n = n_rows * n_cols
    return dict(
        eulers=np.zeros((n, 3)), phase_id=np.ones(n, dtype=int), patterns=np.zeros((n, 2, 2)),
        sht_paths=["x.sht"], masters_meta=[{"point_group": "m-3"}],
        det_params={"n_rows": n_rows, "n_cols": n_cols}, selection_mask=None, roi_mode=False,
        resolved_phase_ids={1})


def test_the_pipeline_entry_hands_the_topology_to_both_stages(monkeypatch):
    calls = []
    monkeypatch.setattr(vu, "build_render_score_fn", lambda *a, **k: (lambda i, q: np.zeros(len(np.atleast_1d(i)))))
    monkeypatch.setattr(vu, "arbitrate_resolver_by_render",
                        lambda *a, **k: (calls.append(("arbitrate", k.get("topology"))), (None, {}))[1])
    monkeypatch.setattr(vu, "unify_map", lambda *a, **k: (calls.append(("unify", k.get("topology"))), (None, {}))[1])
    topology = Topology("hex", 4, 4, 0.5)
    unify_after_hough_resolve(**_entry_arguments(), raw_eulers=np.zeros((16, 3)), topology=topology)
    assert calls == [("arbitrate", topology), ("unify", topology)]


def test_a_topology_of_another_grid_is_refused_by_the_pipeline_entry():
    with pytest.raises(ValueError, match="grid"):
        unify_after_hough_resolve(**_entry_arguments(), topology=Topology("hex", 3, 4, 0.5))


def test_a_transposed_topology_is_refused_by_the_pipeline_entry():
    # 4 x 6 and 6 x 4 hold the same number of points: the product alone would let it through.
    with pytest.raises(ValueError, match="4 x 6"):
        unify_after_hough_resolve(**_entry_arguments(6, 4), topology=Topology("hex", 4, 6, 0.5))


# ----------------------------------------------- the post-processing route

def _stored_result(stamp, n_rows=4, n_cols=4):
    """A spherical result with just what POST /pseudosym/unify reads, in the registry under the id "unify-r"."""
    from types import SimpleNamespace
    from orix.crystal_map import Phase
    from backend.api.routes import indexing
    n = n_rows * n_cols
    quats = np.tile(_qaa([0, 0, 1], 5.0), (n, 1))
    phase = Phase(point_group="m-3")
    result = SimpleNamespace(
        xmap=SimpleNamespace(rotations=SimpleNamespace(data=quats, size=n), phase_id=np.zeros(n, dtype=int),
                             phases={0: phase}),
        selection_mask=np.ones((n_rows, n_cols), dtype=bool), original_shape=(n_rows, n_cols),
        metadata={"indexing_method": "spherical", "sht_paths_by_phase": {0: "x.sht"}, "detector_geometry": {"a": 1},
                  **({"scan_geometry": stamp} if stamp is not None else {})})
    indexing._result_registry["unify-r"] = result
    return result


@pytest.fixture
def unify_route(monkeypatch):
    from fastapi.testclient import TestClient
    from backend.api.main import app
    from backend.api.routes import indexing
    seen = []
    monkeypatch.setattr(vu, "build_render_score_fn", lambda *a, **k: (lambda i, q: np.zeros(len(np.atleast_1d(i)))))
    monkeypatch.setattr(vu, "unify_map", lambda *a, **k: (seen.append(k.get("topology")), (None, {"n_grains": 1}))[1])
    monkeypatch.setattr(indexing, "_free_interactive_gpu_caches", lambda: None)
    yield TestClient(app, raise_server_exceptions=False), seen
    indexing._result_registry.pop("unify-r", None)


@pytest.mark.parametrize("stamp, hexagonal", [
    (ScanGeometry.hex(4, 4, 0.5).to_dict(), True),
    (ScanGeometry.square(4, 4, 0.5).to_dict(), False),
    (None, False),
], ids=["hex", "square", "unknown"])
def test_the_unify_route_uses_the_topology_of_the_results_stamp(unify_route, stamp, hexagonal):
    client, seen = unify_route
    _stored_result(stamp)
    r = client.post("/api/indexing/pseudosym/unify", json={"result_id": "unify-r"})
    assert r.status_code == 200, r.text
    assert len(seen) == 1
    if hexagonal:
        assert (seen[0].kind, seen[0].n_rows, seen[0].n_cols, seen[0].odd_row_shift) == ("hex", 4, 4, 0.5)
    else:
        assert seen[0] is None


def test_the_unify_route_refuses_a_stamp_of_another_grid(unify_route):
    client, seen = unify_route
    _stored_result(ScanGeometry.hex(3, 4, 0.5).to_dict())
    r = client.post("/api/indexing/pseudosym/unify", json={"result_id": "unify-r"})
    assert r.status_code == 409 and "3 x 4" in r.json()["detail"] and seen == []


@pytest.mark.parametrize("stamp", [
    {"kind": "hex", "n_cols": 4, "odd_row_shift": 0.5},
    {"kind": "hex", "n_rows": None, "n_cols": 4, "odd_row_shift": 0.5},
], ids=["no n_rows", "n_rows None"])
def test_the_unify_route_refuses_a_hexagonal_stamp_it_cannot_read(unify_route, stamp):
    client, seen = unify_route
    _stored_result(stamp)
    r = client.post("/api/indexing/pseudosym/unify", json={"result_id": "unify-r"})
    assert r.status_code == 409 and seen == []


def test_the_unify_route_names_the_grids_as_rows_x_columns(unify_route):
    client, seen = unify_route
    _stored_result(ScanGeometry.hex(4, 6, 0.5).to_dict(), n_rows=6, n_cols=4)
    r = client.post("/api/indexing/pseudosym/unify", json={"result_id": "unify-r"})
    assert r.status_code == 409 and "4 x 6" in r.json()["detail"] and "6 x 4" in r.json()["detail"]
