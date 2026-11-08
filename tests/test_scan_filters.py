"""The three neighbourhood filters, held against the physical positions of the lattice (not against themselves)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services.scan_filters import cluster_sizes, modal_vote, nearest_valid  # noqa: E402
from backend.api.services.scan_geometry import ScanGeometry  # noqa: E402
from backend.api.services.scan_neighbours import Topology  # noqa: E402

SIGNS = (0.5, -0.5)
N_ROWS, N_COLS = 8, 7


def _lattice(shift):
    g = ScanGeometry.hex(N_ROWS, N_COLS, 1.0, odd_row_shift=shift)
    x, y = g.positions()
    pts = np.stack([x.ravel(), y.ravel()], axis=1)
    dist = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    return Topology("hex", N_ROWS, N_COLS, shift), dist


def _oracle_cluster_sizes(mask, dist):
    """Components of True points that are one step apart on the sample, by breadth first search."""
    flat = mask.ravel()
    seen = np.zeros(flat.size, dtype=bool)
    sizes = np.zeros(flat.size, dtype=int)
    for start in np.flatnonzero(flat):
        if seen[start]:
            continue
        members, frontier = [start], [start]
        seen[start] = True
        while frontier:
            p = frontier.pop()
            for q in np.flatnonzero(flat & ~seen & (dist[p] > 1e-9) & (dist[p] <= 1 + 1e-6)):
                seen[q] = True
                members.append(q)
                frontier.append(q)
        sizes[members] = len(members)
    return sizes.reshape(mask.shape)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("shift", SIGNS)
def test_cluster_sizes_are_the_components_of_points_one_step_apart(shift, seed):
    topology, dist = _lattice(shift)
    mask = np.random.default_rng(seed).random((N_ROWS, N_COLS)) < 0.5
    np.testing.assert_array_equal(cluster_sizes(mask, topology), _oracle_cluster_sizes(mask, dist))


@pytest.mark.parametrize("shift, pair", [(0.5, ((1, 1), (2, 2))), (-0.5, ((1, 1), (2, 0)))])
def test_a_diagonal_pair_is_one_cluster_by_hand(shift, pair):
    # row 1 is an offset row: with +0.5 its point 1 sits at x = 1.5 and touches (2, 1) and (2, 2); with -0.5 at
    # x = 0.5 and touches (2, 0) and (2, 1). The pair is two points of the index grid that no 4-neighbour rule joins.
    mask = np.zeros((4, 4), dtype=bool)
    for r, c in pair:
        mask[r, c] = True
    sizes = cluster_sizes(mask, Topology("hex", 4, 4, shift))
    assert sizes[pair[0]] == sizes[pair[1]] == 2
    assert sizes.sum() == 4


@pytest.mark.parametrize("seed", range(4))
def test_the_square_topology_labels_like_ndi(seed):
    mask = np.random.default_rng(seed).random((9, 8)) < 0.5
    labeled, n = ndi.label(mask)
    want = np.zeros(mask.shape, dtype=int)
    for k in range(1, n + 1):
        want[labeled == k] = int((labeled == k).sum())
    np.testing.assert_array_equal(cluster_sizes(mask, Topology("square", 9, 8)), want)


def _oracle_modal(ids, dist, radius):
    flat = ids.ravel()
    out = np.full(flat.size, -1, dtype=np.int16)
    for p in range(flat.size):
        window = flat[dist[p] <= radius + 1e-6]
        window = window[window >= 0]
        if window.size:
            values, counts = np.unique(window, return_counts=True)
            out[p] = values[np.argmax(counts)]            # np.unique is ascending: a tie goes to the lowest id
    return out.reshape(ids.shape)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("size, radius", [(3, 1.0), (5, 2.0)])
@pytest.mark.parametrize("shift", SIGNS)
def test_the_vote_is_the_majority_of_the_points_within_the_disc(shift, size, radius, seed):
    topology, dist = _lattice(shift)
    ids = np.random.default_rng(seed).integers(-1, 3, size=(N_ROWS, N_COLS)).astype(np.int16)
    np.testing.assert_array_equal(modal_vote(ids, topology, size), _oracle_modal(ids, dist, radius))


def test_a_disc_of_five_is_nineteen_points_on_a_hexagonal_scan():
    # a point deep inside a field of one phase, with exactly the 19 points of its 2-ring set to another one
    topology, dist = _lattice(0.5)
    centre = 3 * N_COLS + 3
    ids = np.zeros(N_ROWS * N_COLS, dtype=np.int16)
    ids[dist[centre] <= 2 + 1e-6] = 1
    assert int((ids == 1).sum()) == 19
    out = modal_vote(ids.reshape(N_ROWS, N_COLS), topology, 5).ravel()
    assert out[centre] == 1


def test_a_point_without_an_indexed_neighbour_stays_unindexed():
    ids = np.full((6, 6), -1, dtype=np.int16)
    ids[0, 0] = 2
    out = modal_vote(ids, Topology("hex", 6, 6, 0.5), 3)
    assert out[5, 5] == -1 and out[0, 0] == 2


@pytest.mark.parametrize("seed", range(4))
def test_the_square_topology_votes_like_the_box_filter(seed):
    ids = np.random.default_rng(seed).integers(-1, 3, size=(9, 8)).astype(np.int16)
    from backend.api.services.checkpoint_writer import CheckpointWriter
    want = CheckpointWriter.clean_phase_assignment(ids, np.ones(ids.shape), np.zeros(ids.shape), modal_filter_size=3)
    np.testing.assert_array_equal(modal_vote(ids, Topology("square", 9, 8), 3), want)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("shift", SIGNS)
def test_the_nearest_valid_point_is_nearest_on_the_sample(shift, seed):
    topology, dist = _lattice(shift)
    valid = np.random.default_rng(seed).random((N_ROWS, N_COLS)) < 0.15
    valid[0, 0] = True                                           # never empty
    got = nearest_valid(valid, topology).ravel()
    where = np.flatnonzero(valid.ravel())
    for p in range(N_ROWS * N_COLS):
        assert valid.ravel()[got[p]]
        assert dist[p, got[p]] == pytest.approx(dist[p, where].min(), abs=1e-9)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("shift", SIGNS)
def test_nearest_valid_for_targets_answers_only_those_points_in_row_major_order(shift, seed):
    topology, dist = _lattice(shift)
    rng = np.random.default_rng(seed)
    valid = rng.random((N_ROWS, N_COLS)) < 0.15
    valid[0, 0] = True
    targets = rng.random((N_ROWS, N_COLS)) < 0.3
    targets[3, 3] = True
    got = nearest_valid(valid, topology, targets)
    asked = np.flatnonzero(targets.ravel())
    where = np.flatnonzero(valid.ravel())
    assert got.shape == asked.shape                              # one answer per target, no more
    for p, found in zip(asked, got):
        assert valid.ravel()[found]
        assert dist[p, found] == pytest.approx(dist[p, where].min(), abs=1e-9)


def test_nearest_valid_for_targets_by_hand():
    # Valid points: (2, 0) at x = 0, y = 1.732 and (0, 2) at x = 2, y = 0. Asked: (0, 0) at the origin, whose
    # distances are 1.732 and 2, so (2, 0), flat index 10; and (4, 4) at x = 4, y = 3.464, whose distances are
    # hypot(4, 1.732) = 4.36 to (2, 0) and hypot(2, 3.464) = 4.0 to (0, 2), so (0, 2), flat index 2.
    valid = np.zeros((5, 5), dtype=bool)
    valid[2, 0] = valid[0, 2] = True
    targets = np.zeros((5, 5), dtype=bool)
    targets[0, 0] = targets[4, 4] = True
    got = nearest_valid(valid, Topology("hex", 5, 5, 0.5), targets)
    np.testing.assert_array_equal(got, [10, 2])


def test_nearest_on_the_sample_is_not_nearest_in_index_space_by_hand():
    # (0, 0) is at (0, 0). Valid: (2, 0) at (0, 1.732) and (0, 2) at (2, 0). Both are two index steps away; the
    # first is nearer on the sample (1.732 against 2), and a distance transform on the index grid cannot tell.
    valid = np.zeros((5, 5), dtype=bool)
    valid[2, 0] = valid[0, 2] = True
    got = nearest_valid(valid, Topology("hex", 5, 5, 0.5))
    assert got[0, 0] == 2 * 5 + 0


def test_a_valid_point_is_its_own_nearest():
    valid = np.random.default_rng(0).random((6, 6)) < 0.4
    valid[1, 1] = True
    got = nearest_valid(valid, Topology("hex", 6, 6, 0.5))
    idx = np.flatnonzero(valid.ravel())
    np.testing.assert_array_equal(got.ravel()[idx], idx)
