"""Hex <-> physical position, the registry lifecycle and the crop parity of ScanGeometry."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import scan_geometry as sg  # noqa: E402
from backend.api.services.scan_geometry import ScanGeometry  # noqa: E402

SQRT3 = math.sqrt(3.0)


@pytest.fixture(autouse=True)
def clean_registry():
    sg.clear_all()
    yield
    sg.clear_all()


# ------------------------------------------------------------------ positions

def test_hex_positions_follow_the_row_rule_by_hand():
    # step_x 2 um, rows sqrt(3) um apart (2 * sqrt(3)/2), odd rows half a step right.
    g = ScanGeometry.hex(3, 4, 2.0)
    x, y = g.positions()
    assert x[0].tolist() == [0.0, 2.0, 4.0, 6.0]
    assert x[1].tolist() == [1.0, 3.0, 5.0, 7.0]          # +0.5 * 2.0
    assert x[2].tolist() == [0.0, 2.0, 4.0, 6.0]
    assert y[:, 0].tolist() == pytest.approx([0.0, SQRT3, 2 * SQRT3])


def test_the_oim_step_pair_of_a_measured_scan_is_a_regular_lattice():
    # Step X 0.15 / Step Y 0.1299038 is the pair of an OIM hexagonal scan.
    g = ScanGeometry.hex(10, 5, 0.15, 0.1299038)
    assert g.positions()[1][5, 0] == pytest.approx(0.649519, abs=1e-6)   # 5 rows * 0.1299038


def test_a_negative_offset_moves_the_odd_rows_left():
    g = ScanGeometry.hex(2, 3, 2.0, odd_row_shift=-0.5)
    assert g.positions()[0][1].tolist() == [-1.0, 1.0, 3.0]


def test_square_positions_are_a_plain_raster():
    x, y = ScanGeometry.square(2, 3, 0.5, 0.25).positions()
    assert x.tolist() == [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]
    assert y.tolist() == [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]


@pytest.mark.parametrize("shift", [0.5, -0.5])
def test_the_four_index_neighbours_of_a_hex_scan_are_true_neighbours_one_step_away(shift):
    # Why the variant unification may run on a hexagonal scan: it joins a pixel
    # with left, right, and the same column one row up and down. By hand the vertical pair is
    # sqrt(0.5**2 + (sqrt(3)/2)**2) = 1 step apart for either sign of the offset, like the horizontal
    # pair; the two diagonal neighbours are the ones a four-neighbour union-find does not see.
    g = ScanGeometry.hex(4, 5, 2.0, odd_row_shift=shift)
    x, y = g.positions()
    for r in (1, 2):
        for c in (1, 2, 3):
            for dr, dc in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                d = math.hypot(x[r + dr, c + dc] - x[r, c], y[r + dr, c + dc] - y[r, c])
                assert d == pytest.approx(2.0)


@pytest.mark.parametrize("kwargs", [
    dict(kind="hex", n_rows=3, n_cols=3, step_x=1.0, step_y=1.0, odd_row_shift=0.5),      # not sqrt(3)/2
    dict(kind="hex", n_rows=3, n_cols=3, step_x=1.0, step_y=0.8660254, odd_row_shift=0.0),
    dict(kind="hex", n_rows=3, n_cols=3, step_x=1.0, step_y=0.8660254, odd_row_shift=0.25),
    dict(kind="square", n_rows=3, n_cols=3, step_x=1.0, step_y=1.0, odd_row_shift=0.5),
    dict(kind="triangle", n_rows=3, n_cols=3, step_x=1.0, step_y=1.0),
    dict(kind="square", n_rows=0, n_cols=3, step_x=1.0, step_y=1.0),
    dict(kind="square", n_rows=3, n_cols=3, step_x=0.0, step_y=1.0),
])
def test_a_geometry_that_cannot_exist_is_refused(kwargs):
    with pytest.raises(ValueError):
        ScanGeometry(**kwargs)


# -------------------------------------------------------------------- nearest

def test_nearest_finds_the_point_a_position_belongs_to():
    g = ScanGeometry.hex(4, 4, 2.0)
    row, col = g.nearest(1.0, SQRT3)               # exactly row 1, col 0 (x = 0*2 + 1)
    assert (int(row), int(col)) == (1, 0)
    row, col = g.nearest(2.9, 1.6)                 # row 1 grid x = 1, 3, 5: closest 3 -> col 1
    assert (int(row), int(col)) == (1, 1)


def test_an_exact_tie_goes_to_the_lower_row_and_column():
    # The OIM format fixes the COLUMN tie (OIM takes the lower column); the row tie below is
    # OUR rule, chosen to match it, not something a file or a vendor document states.
    g = ScanGeometry.hex(3, 3, 2.0)
    row, col = g.nearest(1.0, 0.0)                 # halfway between (0,0) and (0,1)
    assert (int(row), int(col)) == (0, 0)


def test_an_exact_tie_between_rows_goes_to_the_lower_row():
    """Our own rule (not stated by any file): the lower row wins a tie between two rows."""
    g = ScanGeometry.square(3, 3, 2.0)
    row, col = g.nearest(0.0, 1.0)                 # y = 1 is halfway between row 0 (y = 0) and row 1 (y = 2)
    assert (int(row), int(col)) == (0, 0)


def test_nearest_inverts_positions_for_every_point():
    g = ScanGeometry.hex(7, 6, 0.15)
    x, y = g.positions()
    row, col = g.nearest(x, y)
    rr, cc = np.indices((7, 6))
    assert np.array_equal(row, rr) and np.array_equal(col, cc)


def test_a_position_outside_the_scan_gives_the_edge_point():
    g = ScanGeometry.hex(3, 3, 2.0)
    row, col = g.nearest(-50.0, 99.0)
    assert (int(row), int(col)) == (2, 0)


def test_the_corner_reproducers_find_the_row_beside_the_edge_row():
    # By hand: hex(7, 6, 2.0), rows sqrt(3) = 1.732 apart, odd rows at x = 1, 3, ..., 11. Query
    # (13.75, 11.18) is past the last row (y = 6*1.732 = 10.39). Row 6 ends at x = 10: distance
    # sqrt(3.75**2 + 0.79**2) = 3.83. Row 5 (y = 8.66) ends at x = 11: sqrt(2.75**2 + 2.52**2) = 3.73.
    # So row 5 is the closer one.
    row, col = ScanGeometry.hex(7, 6, 2.0).nearest(13.75, 11.18)
    assert (int(row), int(col)) == (5, 5)
    # Mirror: odd rows shifted left (x = -1, 1, ...); query (-4.4, 11.5). Row 6 starts at x = 0:
    # sqrt(4.4**2 + 1.11**2) = 4.54. Row 5 (y = 8.66) starts at x = -1: sqrt(3.4**2 + 2.84**2) = 4.43.
    # Row 5 wins.
    row, col = ScanGeometry.hex(7, 6, 2.0, odd_row_shift=-0.5).nearest(-4.4, 11.5)
    assert (int(row), int(col)) == (5, 0)


def _brute_force_nearest(geometry, qx, qy):
    """The nearest point by trying every point, positions built by hand (not from positions())."""
    sx, sy, shift = geometry.step_x, geometry.step_y, geometry.odd_row_shift
    best, best_d, second_d = None, np.inf, np.inf
    for r in range(geometry.n_rows):
        for c in range(geometry.n_cols):
            x = c * sx + (shift * sx if r % 2 == 1 else 0.0)
            y = r * sy
            d = (qx - x) ** 2 + (qy - y) ** 2
            if d < best_d:
                best, second_d, best_d = (r, c), best_d, d
            elif d < second_d:
                second_d = d
    return best, second_d - best_d


def _nearest_cases():
    return [
        ("hex +0.5", ScanGeometry.hex(7, 6, 2.0)),
        ("hex -0.5", ScanGeometry.hex(7, 6, 2.0, odd_row_shift=-0.5)),
        ("hex cropped from an odd row", ScanGeometry.hex(9, 7, 2.0).cropped(1, 1, 7, 5)),
        ("square", ScanGeometry.square(5, 6, 2.0, 1.5)),
    ]


@pytest.mark.parametrize("label,geometry", _nearest_cases(), ids=[c[0] for c in _nearest_cases()])
def test_nearest_is_the_brute_force_nearest_inside_and_outside_the_scan(label, geometry):
    # Query grid: from 7 um before the scan to 7 um past it on both axes, so all four corner regions
    # and all four sides are covered; the strides are irrational-ish so no query sits on a tie.
    width = (geometry.n_cols - 1) * geometry.step_x + geometry.step_x
    height = (geometry.n_rows - 1) * geometry.step_y
    qx, qy = np.meshgrid(np.linspace(-7.0 + 0.0123, width + 7.0 - 0.0171, 47),
                         np.linspace(-7.0 + 0.0311, height + 7.0 - 0.0219, 43))
    row, col = geometry.nearest(qx, qy)
    wrong = []
    for i in np.ndindex(qx.shape):
        expected, gap = _brute_force_nearest(geometry, qx[i], qy[i])
        assert gap > 1e-9, "the query grid hit a tie; choose other strides"
        if (int(row[i]), int(col[i])) != expected:
            wrong.append((float(qx[i]), float(qy[i]), (int(row[i]), int(col[i])), expected))
    assert wrong == [], f"{len(wrong)} of {qx.size} queries differ, first: {wrong[:3]}"


# ------------------------------------------------------------------------ crop

def test_a_crop_from_an_even_row_keeps_the_offset():
    child = ScanGeometry.hex(6, 5, 2.0).cropped(2, 1, 3, 3)
    assert child.odd_row_shift == 0.5


def test_a_crop_starting_on_an_odd_row_reverses_the_offset():
    parent = ScanGeometry.hex(6, 5, 2.0)
    child = parent.cropped(1, 1, 3, 3)
    assert child.odd_row_shift == -0.5
    # By hand: parent row 1 is shifted, so the window's row 0 is; the window's row 1 is
    # parent row 2, one full 0.5 step LEFT of it. Measured from window point (0, 0).
    x, _ = child.positions()
    assert x[1, 0] == -1.0                                  # -0.5 * 2.0
    assert x[0].tolist() == [0.0, 2.0, 4.0]
    # and it is the same lattice as the parent's, cut out and re-anchored
    px, py = parent.positions()
    cx, cy = child.positions()
    assert np.allclose(cx, px[1:4, 1:4] - px[1, 1])
    assert np.allclose(cy, py[1:4, 1:4] - py[1, 1])


def test_two_odd_crops_are_one_even_crop():
    parent = ScanGeometry.hex(8, 6, 2.0)
    twice = parent.cropped(1, 0, 6, 6).cropped(1, 0, 4, 6)
    once = parent.cropped(2, 0, 4, 6)
    assert twice.odd_row_shift == once.odd_row_shift == 0.5


def test_a_square_crop_never_gets_an_offset():
    child = ScanGeometry.square(6, 5, 1.0).cropped(1, 1, 3, 3)
    assert child.odd_row_shift == 0.0
    assert math.copysign(1.0, child.to_dict()["odd_row_shift"]) == 1.0      # not -0.0


@pytest.mark.parametrize("window", [(-1, 0, 2, 2), (0, 0, 7, 2), (0, 4, 2, 2), (0, 0, 0, 2)])
def test_a_window_that_does_not_fit_is_refused(window):
    with pytest.raises(ValueError):
        ScanGeometry.hex(6, 5, 2.0).cropped(*window)


def test_the_mask_is_cut_with_the_window():
    mask = np.ones((4, 4), dtype=bool)
    mask[1, 1] = False
    g = ScanGeometry("hex", 4, 4, 1.0, 1.0 * sg.HEX_ROW_RATIO, 0.5, False, True, mask)
    cut = g.cropped(1, 1, 2, 2)
    assert cut.valid_mask.tolist() == [[False, True], [True, True]]
    assert cut.n_valid == 3
    assert not cut.valid_mask.flags.writeable


# ------------------------------------------------------------------------ wire

def test_the_dict_round_trips_and_says_whether_the_direction_is_known():
    g = ScanGeometry.hex(37, 41, 0.15)
    d = g.to_dict()
    assert d["kind"] == "hex" and d["grid_shape"] == [37, 41]
    assert d["shift_confirmed"] is False and d["has_valid_mask"] is False
    assert ScanGeometry.from_dict(d) == g


def test_a_geometry_with_a_valid_mask_is_not_silently_unmasked_by_the_dict_form():
    mask = np.ones((3, 3), dtype=bool)
    mask[0, 0] = False
    d = ScanGeometry("square", 3, 3, 1.0, 1.0, 0.0, True, True, mask).to_dict()
    assert d["has_valid_mask"] is True
    with pytest.raises(ValueError, match="mask"):
        ScanGeometry.from_dict(d)


# -------------------------------------------------------------------- signals

class _Axis(SimpleNamespace):
    pass


class _Signal:
    def __init__(self, nav, meta=None):
        self.axes_manager = SimpleNamespace(navigation_axes=nav)
        self.metadata = SimpleNamespace(get_item=lambda key, default=None: (meta or {}).get(key, default))


def test_a_signal_without_a_mirror_is_a_square_scan_read_off_its_axes():
    sig = _Signal([_Axis(size=7, scale=0.5, units="um"), _Axis(size=3, scale=0.25, units="um")])
    g = sg.geometry_of_signal(sig)
    assert (g.kind, g.n_rows, g.n_cols, g.step_x, g.step_y) == ("square", 3, 7, 0.5, 0.25)
    assert g.steps_known


def test_undefined_units_make_the_step_a_placeholder_not_a_measurement():
    sig = _Signal([_Axis(size=5, scale=1.0, units="<undefined>"), _Axis(size=2, scale=1.0, units="<undefined>")])
    g = sg.geometry_of_signal(sig)
    assert (g.n_rows, g.n_cols) == (2, 5) and not g.steps_known


def test_a_one_dimensional_navigation_is_one_row():
    g = sg.geometry_of_signal(_Signal([_Axis(size=5, scale=1.0, units="um")]))
    assert g.grid_shape == (1, 5)


def test_the_loaders_mirror_wins_over_the_axes():
    mirror = ScanGeometry.hex(3, 4, 0.15).to_dict()
    sig = _Signal([_Axis(size=4, scale=0.15, units="um"), _Axis(size=3, scale=0.15, units="um")],
                  {"Signal.scan_geometry": mirror})
    assert sg.geometry_of_signal(sig).kind == "hex"


def test_a_mirror_that_disagrees_with_the_axes_is_refused_loudly():
    mirror = ScanGeometry.hex(3, 4, 0.15).to_dict()
    sig = _Signal([_Axis(size=5, scale=0.15, units="um"), _Axis(size=3, scale=0.15, units="um")],
                  {"Signal.scan_geometry": mirror})
    with pytest.raises(sg.GeometryMismatchError):
        sg.geometry_of_signal(sig)
    with pytest.raises(sg.GeometryMismatchError):
        sg.register_from_signal("ds", sig)
    assert sg.geometry_for("ds") is None


def test_registering_takes_the_loaders_copy_off_the_signal_so_a_crop_cannot_inherit_it():
    """The copy is a transport. A crop (inav) or a deepcopy of a signal that still carried it would
    describe the whole scan and make the next geometry_of_signal raise."""
    import hyperspy.api as hs
    sig = hs.signals.Signal2D(np.zeros((3, 5, 4, 4)))                  # navigation (5, 3), a 3 x 5 scan
    sig.metadata.set_item("Signal.scan_geometry", ScanGeometry.hex(3, 5, 0.5).to_dict())
    assert sg.register_from_signal("raw", sig).kind == "hex"
    assert not sig.metadata.has_item("Signal.scan_geometry")
    crop = sig.inav[0:3, 1:3]                                          # 2 rows x 3 columns
    assert sg.geometry_of_signal(crop).grid_shape == (2, 3)            # square, read off its own axes
    assert sg.geometry_of_signal(sig.deepcopy()).grid_shape == (3, 5)
    assert sg.geometry_for("raw").kind == "hex"                        # the registry still knows


def test_a_crop_made_before_the_mirror_was_dropped_would_have_raised():
    """The failure the drop prevents, shown on a signal that kept its copy."""
    import hyperspy.api as hs
    sig = hs.signals.Signal2D(np.zeros((3, 5, 4, 4)))
    sig.metadata.set_item("Signal.scan_geometry", ScanGeometry.hex(3, 5, 0.5).to_dict())
    with pytest.raises(sg.GeometryMismatchError):
        sg.geometry_of_signal(sig.inav[0:3, 1:3])


def test_dropping_the_copy_from_a_stand_in_or_a_signal_without_one_does_nothing():
    sg.drop_signal_mirror(object())
    import hyperspy.api as hs
    sg.drop_signal_mirror(hs.signals.Signal2D(np.zeros((2, 2, 4, 4))))


def test_a_signal_that_cannot_be_read_registers_nothing_and_does_not_raise():
    assert sg.register_from_signal("ds", object()) is None
    assert sg.geometry_for("ds") is None


def test_hyperspy_metadata_trees_come_back_as_plain_dicts():
    class Tree:
        def as_dictionary(self):
            return {"a": 1}
    assert sg.plain(Tree()) == {"a": 1}
    assert sg.plain({"b": 2}) == {"b": 2} and sg.plain(None) is None


# -------------------------------------------------------------------- registry

def test_a_deepcopy_sits_where_its_parent_does():
    parent = ScanGeometry.hex(6, 5, 2.0)
    sg.set_geometry("raw", parent)
    assert sg.derive("raw_copy", "raw") == parent
    assert sg.geometry_for("raw_copy") == parent


def test_a_crop_is_composed_not_copied():
    sg.set_geometry("raw", ScanGeometry.hex(6, 5, 2.0))
    window = SimpleNamespace(row0=1, col0=1, rows=3, cols=3)
    child = sg.derive_cropped("crop", "raw", window)
    assert child.odd_row_shift == -0.5
    assert sg.geometry_for("raw").odd_row_shift == 0.5             # the parent did not move


def test_a_crop_of_a_crop_is_cut_from_the_crop():
    sg.set_geometry("raw", ScanGeometry.hex(8, 6, 2.0))
    sg.derive_cropped("c1", "raw", SimpleNamespace(row0=1, col0=0, rows=6, cols=6))
    c2 = sg.derive_cropped("c2", "c1", SimpleNamespace(row0=1, col0=0, rows=4, cols=6))
    assert c2.odd_row_shift == 0.5 and c2.grid_shape == (4, 6)


def test_an_unknown_parent_registers_nothing():
    assert sg.derive("x", "nobody") is None
    assert sg.derive_cropped("x", "nobody", SimpleNamespace(row0=0, col0=0, rows=1, cols=1)) is None
    assert sg.geometry_for("x") is None


def test_clear_drops_one_and_clear_all_drops_everything():
    sg.set_geometry("a", ScanGeometry.square(2, 2, 1.0))
    sg.set_geometry("b", ScanGeometry.square(2, 2, 1.0))
    sg.clear("a")
    assert sg.geometry_for("a") is None and sg.geometry_for("b") is not None
    sg.clear_all()
    assert sg.geometry_for("b") is None


def test_the_stash_round_trips_and_a_live_geometry_wins():
    sg.set_geometry("raw", ScanGeometry.hex(6, 5, 2.0))
    sg.derive("copy", "raw")
    snap = sg.snapshot(["raw", "copy", "ghost"])
    assert set(snap) == {"raw", "copy"}
    sg.clear_all()
    live = ScanGeometry.square(2, 2, 1.0)
    sg.set_geometry("raw", live)                           # the freshly loaded raw dataset
    sg.restore(snap, skip="raw")
    assert sg.geometry_for("raw") == live                  # not overwritten
    assert sg.geometry_for("copy").kind == "hex"           # the derived dataset is back


def test_restore_never_overwrites_a_dataset_that_is_live_without_being_the_skipped_one():
    sg.set_geometry("copy", ScanGeometry.hex(6, 5, 2.0))
    snap = sg.snapshot(["copy"])
    live = ScanGeometry.square(2, 2, 1.0)
    sg.set_geometry("copy", live)                          # live again, and not the one named by skip
    sg.restore(snap, skip="raw")
    assert sg.geometry_for("copy") == live


def test_restore_never_brings_back_the_skipped_name_even_when_it_is_not_live():
    sg.set_geometry("raw", ScanGeometry.hex(6, 5, 2.0))
    snap = sg.snapshot(["raw"])
    sg.clear_all()                                         # "raw" is NOT live now
    sg.restore(snap, skip="raw")
    assert sg.geometry_for("raw") is None


# ------------------------------------------- the one place that holds the assumed row offset

def test_the_assumed_offset_is_one_constant_and_the_hex_default():
    # By hand: OIM's own hex files offset the odd rows to the right, by half a step.
    assert sg.OIM_ODD_ROW_SHIFT == 0.5
    assert ScanGeometry.hex(2, 2, 1.0).odd_row_shift == sg.OIM_ODD_ROW_SHIFT
    assert not ScanGeometry.hex(2, 2, 1.0).shift_confirmed


def test_no_source_file_outside_scan_geometry_writes_the_offset_as_a_literal():
    """The assumption lives in scan_geometry.OIM_ODD_ROW_SHIFT and nowhere else: when a reference
    file settles the direction, one line changes. (-0.5 is a legitimate test input and tests are not
    scanned; the loader and every other source file take the constant.)"""
    import re
    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r"odd_row_shift\s*=\s*[+-]?0?\.5\b")
    sources = [p for p in list(root.glob("*.py")) + list((root / "backend").rglob("*.py"))
               if p.name != "scan_geometry.py"]
    assert sources, "found no source files to scan"
    offenders = [str(p.relative_to(root)) for p in sources
                 if pattern.search(p.read_text(encoding="utf-8", errors="replace"))]
    assert offenders == []
    # the positive anchor: the APEX loader really does take the constant
    assert "OIM_ODD_ROW_SHIFT" in (root / "edax_apex.py").read_text(encoding="utf-8")
