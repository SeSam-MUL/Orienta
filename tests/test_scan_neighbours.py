"""One definition of a neighbour: the hexagonal table is held against the lattice positions."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services.scan_geometry import ScanGeometry  # noqa: E402
from backend.api.services.scan_neighbours import Topology, hex_topology, topology_for  # noqa: E402

SIGNS = (0.5, -0.5)


def _oracle(g):
    """For every point the set of other points within one step, from the physical positions alone."""
    x, y = g.positions()
    pts = np.stack([x.ravel(), y.ravel()], axis=1)
    dist = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    near = (dist > 1e-9) & (dist <= g.step_x * (1 + 1e-6))
    return [set(np.flatnonzero(row).tolist()) for row in near]


def _hops(graph, start, k):
    """Points within k hops of start in the neighbour graph (breadth first)."""
    seen, frontier = {start}, {start}
    for _ in range(k):
        frontier = {q for p in frontier for q in graph[p]} - seen
        seen |= frontier
    return seen


@pytest.mark.parametrize("shift", SIGNS)
def test_the_table_is_the_set_of_points_one_step_away(shift):
    g = ScanGeometry.hex(9, 8, 1.0, odd_row_shift=shift)
    table = Topology("hex", 9, 8, shift).table()
    want = _oracle(g)
    for p in range(g.n_points):
        assert {int(q) for q in table[p] if q >= 0} == want[p], p


@pytest.mark.parametrize("shift", SIGNS)
def test_an_interior_point_has_six_neighbours(shift):
    t = Topology("hex", 9, 8, shift)
    assert t.table().shape == (72, 6)
    assert len(t.neighbours(4 * 8 + 3)) == 6 and len(t.neighbours(3 * 8 + 3)) == 6     # an even and an odd row


@pytest.mark.parametrize("shift", SIGNS)
def test_neighbours_of_every_point_is_the_set_of_points_one_step_away(shift):
    # the per-point list (odd and even rows, edges) against the lattice, not only its length
    g = ScanGeometry.hex(6, 5, 1.0, odd_row_shift=shift)
    t = Topology("hex", 6, 5, shift)
    want = _oracle(g)
    for p in range(g.n_points):
        got = t.neighbours(p)
        assert len(got) == len(set(got)) and set(got) == want[p], p


@pytest.mark.parametrize("shift", SIGNS)
def test_topology_for_keeps_the_sign_of_a_geometry_and_of_its_dict(shift):
    g = ScanGeometry.hex(4, 5, 1.0, odd_row_shift=shift)
    assert topology_for(g) == Topology("hex", 4, 5, shift)
    assert topology_for(g.to_dict()) == Topology("hex", 4, 5, shift)


def test_a_corner_has_two_or_three_neighbours_by_hand():
    # (0, 0) at x = 0. Odd rows to the right (+0.5): the row below starts at x = 0.5, so (1, 0) is a
    # neighbour and (1, -1) does not exist: right, below -> 2. Odd rows to the left (-0.5): the row below
    # starts at x = -0.5, so both (1, 0) and (1, 1) are neighbours -> 3.
    assert sorted(Topology("hex", 4, 4, 0.5).neighbours(0)) == [1, 4]
    assert sorted(Topology("hex", 4, 4, -0.5).neighbours(0)) == [1, 4, 5]


@pytest.mark.parametrize("shift", SIGNS)
def test_a_crop_on_an_odd_row_is_followed_without_a_special_case(shift):
    parent = ScanGeometry.hex(8, 7, 1.0, odd_row_shift=shift)
    crop = parent.cropped(1, 0, 5, 7)                    # starts on an odd row: the sign reverses
    assert crop.odd_row_shift == -shift
    table = topology_for(crop).table()
    want = _oracle(crop)
    for p in range(crop.n_points):
        assert {int(q) for q in table[p] if q >= 0} == want[p]


@pytest.mark.parametrize("shift", SIGNS)
def test_the_pairs_are_every_neighbouring_pair_once(shift):
    g = ScanGeometry.hex(7, 6, 1.0, odd_row_shift=shift)
    a, b = Topology("hex", 7, 6, shift).pairs()
    assert np.all(a < b) and a.dtype == np.int64
    got = {(int(i), int(j)) for i, j in zip(a, b)}
    assert len(got) == a.size                                # none twice
    want = {(p, q) for p, qs in enumerate(_oracle(g)) for q in qs if p < q}
    assert got == want


@pytest.mark.parametrize("rings", [1, 2, 3])
@pytest.mark.parametrize("shift", SIGNS)
def test_a_disc_is_the_points_within_rings_hops(shift, rings):
    n_rows, n_cols = 15, 15
    g = ScanGeometry.hex(n_rows, n_cols, 1.0, odd_row_shift=shift)
    graph = _oracle(g)
    table = Topology("hex", n_rows, n_cols, shift).table(rings=rings, include_self=True)
    counts = {1: 7, 2: 19, 3: 37}
    for row in (7, 8):                                       # an even and an odd interior row
        p = row * n_cols + 7
        got = {int(q) for q in table[p] if q >= 0}
        assert got == _hops(graph, p, rings) and len(got) == counts[rings]
        assert table[p][0] == p                              # the centre comes first


def test_a_disc_table_without_the_centre_omits_it():
    table = Topology("hex", 9, 8, 0.5).table(rings=2)
    assert table.shape == (72, 18)
    assert all(p not in set(table[p].tolist()) for p in range(72))


def test_the_square_topology_is_the_four_neighbours_the_existing_code_uses():
    n_rows, n_cols = 5, 6
    t = Topology("square", n_rows, n_cols)
    for f in range(n_rows * n_cols):
        r, c = divmod(f, n_cols)
        legacy = [f - n_cols if r > 0 else -1, f + n_cols if r < n_rows - 1 else -1,
                  f - 1 if c > 0 else -1, f + 1 if c < n_cols - 1 else -1]       # variant_unification.py:240
        assert t.neighbours(f) == [q for q in legacy if q >= 0]
    a, b = t.pairs()
    assert a.size == n_rows * (n_cols - 1) + (n_rows - 1) * n_cols       # right and down
    assert t.disc_offsets(0, 1)[0] == (0, 0) and len(t.disc_offsets(0, 2)) == 25


def test_the_tables_are_read_only_and_cached():
    t = Topology("hex", 6, 6, 0.5)
    assert t.table() is t.table() and not t.table().flags.writeable
    assert t.pairs()[0] is t.pairs()[0] and not t.pairs()[0].flags.writeable


def test_rings_must_be_at_least_one():
    with pytest.raises(ValueError):
        Topology("hex", 4, 4, 0.5).disc_offsets(0, 0)
    with pytest.raises(ValueError):
        Topology("hex", 4, 4, 0.5).table(rings=0)


def test_topology_for_reads_a_geometry_a_dict_and_nothing():
    g = ScanGeometry.hex(4, 5, 1.0)
    assert topology_for(g) == Topology("hex", 4, 5, 0.5)
    assert topology_for(g.to_dict()) == Topology("hex", 4, 5, 0.5)
    assert topology_for(ScanGeometry.square(3, 3, 1.0)) == Topology("square", 3, 3, 0.0)
    assert topology_for(None) is None
    assert topology_for({"kind": "triangle", "n_rows": 2, "n_cols": 2}) is None


def test_an_unknown_kind_is_refused_not_read_as_hex():
    with pytest.raises(ValueError, match="scan kind"):
        Topology("triangle", 4, 4, 0.5)


def test_a_hex_lattice_without_a_half_step_shift_is_refused():
    with pytest.raises(ValueError, match=r"\+-0\.5"):
        Topology("hex", 4, 4, 0.0)
    with pytest.raises(ValueError, match=r"\+-0\.5"):
        Topology("hex", 4, 4, 0.25)


def test_a_hex_dict_without_the_shift_key_is_refused():
    with pytest.raises(ValueError, match=r"\+-0\.5"):
        topology_for({"kind": "hex", "n_rows": 4, "n_cols": 4})
    with pytest.raises(ValueError, match=r"\+-0\.5"):
        hex_topology({"kind": "hex", "n_rows": 4, "n_cols": 4, "odd_row_shift": 0.0})


def test_hex_topology_is_the_legacy_path_switch():
    assert hex_topology(ScanGeometry.square(3, 3, 1.0)) is None
    assert hex_topology(None) is None
    assert hex_topology(ScanGeometry.hex(3, 3, 1.0)).kind == "hex"


@pytest.mark.parametrize("shift", SIGNS)
def test_xy_is_the_lattice_position_in_steps(shift):
    g = ScanGeometry.hex(6, 5, 2.0, odd_row_shift=shift)
    x, y = Topology("hex", 6, 5, shift).xy()
    gx, gy = g.positions()
    np.testing.assert_allclose(x, gx / 2.0, atol=1e-12)
    np.testing.assert_allclose(y, gy / 2.0, atol=1e-12)


def test_xy_of_a_square_scan_is_the_index():
    x, y = Topology("square", 3, 4).xy()
    np.testing.assert_array_equal(x, np.tile(np.arange(4.0), (3, 1)))
    np.testing.assert_array_equal(y, np.tile(np.arange(3.0)[:, None], (1, 4)))


def test_a_stamp_gives_a_topology_only_when_hexagonal_and_of_the_right_grid():
    from backend.api.services.scan_neighbours import topology_of_stamp
    hex_stamp = ScanGeometry.hex(4, 5, 1.0).to_dict()
    assert topology_of_stamp(hex_stamp, (4, 5)).kind == "hex"
    assert topology_of_stamp(hex_stamp).odd_row_shift == 0.5
    assert topology_of_stamp(ScanGeometry.square(4, 5, 1.0).to_dict(), (4, 5)) is None
    assert topology_of_stamp(None, (4, 5)) is None
    with pytest.raises(ValueError, match="4 x 5"):
        topology_of_stamp(hex_stamp, (6, 5))


@pytest.mark.parametrize("broken", [
    {"kind": "hex", "n_cols": 5, "odd_row_shift": 0.5},                       # no n_rows
    {"kind": "hex", "n_rows": None, "n_cols": 5, "odd_row_shift": 0.5},
    {"kind": "hex", "n_rows": 4, "odd_row_shift": 0.5},                       # no n_cols
    {"kind": "hex", "n_rows": 4, "n_cols": 5},                                # no shift: not a lattice
], ids=["no n_rows", "n_rows None", "no n_cols", "no odd_row_shift"])
def test_a_hexagonal_stamp_that_cannot_be_read_is_a_value_error(broken):
    # Every caller catches ValueError for "this stamp does not describe the map"; a TypeError from int(None)
    # escaped it (a 500 on the route, a run thrown away after the GPU work).
    from backend.api.services.scan_neighbours import topology_of_stamp
    with pytest.raises(ValueError):
        topology_of_stamp(broken, (4, 5))
    with pytest.raises(ValueError):
        topology_of_stamp(broken)


@pytest.mark.parametrize("broken", [
    {"kind": "hex", "n_rows": "x", "n_cols": 5, "odd_row_shift": 0.5},
    {"kind": "hex", "n_rows": 4, "n_cols": 5, "odd_row_shift": "abc"},
    {"kind": "hex", "n_rows": float("inf"), "n_cols": 5, "odd_row_shift": 0.5},
    {"kind": "hex", "n_rows": float("nan"), "n_cols": 5, "odd_row_shift": 0.5},
    {"kind": "hex", "n_rows": 4.7, "n_cols": 5, "odd_row_shift": 0.5},
    {"kind": "hex", "n_rows": 4, "n_cols": 5.5, "odd_row_shift": 0.5},
    {"kind": "hex", "n_rows": 4, "n_cols": 5, "odd_row_shift": 0.25},          # not a hex lattice
], ids=["n_rows text", "shift text", "n_rows inf", "n_rows nan", "n_rows fractional", "n_cols fractional",
        "shift 0.25"])
def test_a_hexagonal_stamp_with_a_bad_value_says_the_stamp_cannot_be_read(broken):
    # The raw int()/float() messages ("invalid literal for int() ...") named no stamp, and inf was a 500.
    from backend.api.services.scan_neighbours import topology_of_stamp
    for shape in ((4, 5), None):
        with pytest.raises(ValueError, match="The scan geometry stamped on this result cannot be read") as excinfo:
            topology_of_stamp(broken, shape)
        assert len(str(excinfo.value)) < 200


def test_a_whole_number_stamp_given_as_text_or_float_is_still_read():
    from backend.api.services.scan_neighbours import topology_of_stamp
    t = topology_of_stamp({"kind": "hex", "n_rows": "4", "n_cols": 5.0, "odd_row_shift": "0.5"}, (4, 5))
    assert (t.n_rows, t.n_cols, t.odd_row_shift) == (4, 5, 0.5)


@pytest.mark.parametrize("stamp", [
    {"kind": "square"},                                                        # no n_rows, no n_cols
    {"kind": "square", "n_cols": 5},                                           # no n_rows
    {"kind": "square", "n_rows": None, "n_cols": 5},
    {"kind": "square", "n_rows": "x", "n_cols": 5, "odd_row_shift": "?"},      # not numeric
    {"kind": "square", "n_rows": 9, "n_cols": 9},                              # a grid other than the map's
], ids=["bare", "no n_rows", "n_rows None", "not numeric", "other grid"])
def test_a_square_stamp_is_none_complete_or_not(stamp):
    # On a square scan only the index grid counts: a damaged square stamp must cost a square run nothing.
    from backend.api.services.scan_neighbours import topology_of_stamp
    assert topology_of_stamp(stamp, (4, 5)) is None
    assert topology_of_stamp(stamp) is None


def test_a_stamp_of_unknown_kind_is_none_and_a_bare_hex_stamp_is_still_a_value_error():
    from backend.api.services.scan_neighbours import topology_of_stamp
    assert topology_of_stamp({"n_rows": 4, "n_cols": 5}, (4, 5)) is None          # unknown kind: as before
    with pytest.raises(ValueError):
        topology_of_stamp({"kind": "hex"}, (4, 5))


def test_the_grid_of_a_stamp_of_another_map_is_named_as_rows_x_columns():
    from backend.api.services.scan_neighbours import topology_of_stamp
    with pytest.raises(ValueError) as excinfo:
        topology_of_stamp(ScanGeometry.hex(4, 6, 1.0).to_dict(), (6, 4))
    assert "4 x 6" in str(excinfo.value) and "6 x 4" in str(excinfo.value) and "(" not in str(excinfo.value)
