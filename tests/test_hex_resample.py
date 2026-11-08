"""Resampling a hexagonal scan to a square grid by OIM's own nearest-neighbour rule.

The hand cases use step_x = 1 and the hexagonal row distance step_y = sqrt(3)/2 = 0.8660254, with
the odd rows shifted by +0.5 or -0.5 steps. Every expected index below is worked out on paper from
those positions (the arithmetic sits next to the assertion), never read from the function. The last
group runs the function against OIM's own hexagonal scan and the square scan OIM made from it; it
needs ``ORIENTA_HEX_REFERENCE`` and skips visibly without it.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import hex_resample  # noqa: E402
from backend.api.services.scan_geometry import ScanGeometry  # noqa: E402

ENV = "ORIENTA_HEX_REFERENCE"
SQRT3_2 = math.sqrt(3.0) / 2.0


def _hex(n_rows, n_cols, shift, mask=None, step_x=1.0):
    geometry = ScanGeometry.hex(n_rows, n_cols, step_x, odd_row_shift=shift)
    if mask is None:
        return geometry
    return ScanGeometry(
        kind="hex", n_rows=n_rows, n_cols=n_cols, step_x=geometry.step_x, step_y=geometry.step_y,
        odd_row_shift=shift, shift_confirmed=False, steps_known=True, valid_mask=mask)


# ------------------------------------------------------------------ the shape of the square grid

def test_five_rows_by_four_columns_give_four_by_four():
    # y_max = 4 * 0.8660254 = 3.4641; round(3.4641) = 3, so 3 + 1 = 4 rows. x_max = 3 * 1 = 3,
    # so 4 columns. The 20 hex points map to a grid of 16 cells, so at least 4 points are taken by no cell.
    r = hex_resample.square_resampling(_hex(5, 4, +0.5))
    assert r.shape == (4, 4)
    assert (r.n_rows, r.n_cols) == (4, 4)
    assert r.n_out == 16
    assert r.n_source == 20
    assert r.step == 1.0
    assert r.point_index.shape == (16,)


@pytest.mark.parametrize("shift", [+0.5, -0.5])
@pytest.mark.parametrize("n_cols", [4, 5])
def test_the_square_grid_keeps_the_columns_of_the_hex_rows(n_cols, shift):
    # The shifted rows reach half a step past the last column. That half step gets no column:
    # x_max = (n_cols - 1) * step_x, so n_cols columns for an even and an odd count alike (rounding
    # n_cols - 0.5 would give 4 for 4 and 6 for 5, the half going to the even neighbour).
    r = hex_resample.square_resampling(_hex(5, n_cols, shift))
    assert r.n_cols == n_cols
    assert r.n_rows == 4


# ------------------------------------------------------------------ which point each cell takes

def test_shift_plus_half_hand_case():
    # 3 rows x 4 columns, odd rows at x = 0.5 1.5 2.5 3.5 (y 0.866), even rows at 0 1 2 3
    # (y 0 and 1.732). Point numbers: row 0 = 0..3, row 1 = 4..7, row 2 = 8..11.
    # Square grid: 3 x 4 (round(2 * 0.866) = round(1.732) = 2, +1), cells at x 0..3, y 0 1 2.
    #  y = 0: the points of row 0 themselves: 0 1 2 3.
    #  y = 1: row 1 (y 0.866, 0.134 away) beats rows 0 and 2 (1 and 0.732 away). x = 0 -> 0.5 (pt 4);
    #         x = 1 is an exact tie between 0.5 and 1.5, the lower column wins (pt 4); x = 2 ties
    #         between 1.5 and 2.5 (pt 5); x = 3 ties between 2.5 and 3.5 (pt 6).
    #  y = 2: the points of row 2 themselves: 8 9 10 11. Point 7 (x 3.5) is nobody's nearest.
    r = hex_resample.square_resampling(_hex(3, 4, +0.5))
    assert r.shape == (3, 4)
    assert r.point_index.tolist() == [0, 1, 2, 3, 4, 4, 5, 6, 8, 9, 10, 11]
    assert (r.n_source, r.n_out, r.n_unused) == (12, 12, 1)


def test_shift_minus_half_hand_case():
    # The odd rows now sit half a step to the LEFT of the even rows. Every x gets +0.5 so the
    # smallest row start is 0: odd rows (row 1) at 0 1 2 3, even rows at 0.5 1.5 2.5 3.5.
    #  y = 0: x = 0 -> 0.5 (pt 0); x = 1 is a tie between 0.5 and 1.5 (pt 0); x = 2 ties 1.5 / 2.5
    #         (pt 1); x = 3 ties 2.5 / 3.5 (pt 2).
    #  y = 1: row 1 (4..7) at x 0 1 2 3: 4 5 6 7.
    #  y = 2: row 2 like row 0: x = 0 -> pt 8, x = 1 -> pt 8, x = 2 -> pt 9, x = 3 -> pt 10.
    r = hex_resample.square_resampling(_hex(3, 4, -0.5))
    assert r.shape == (3, 4)
    assert r.point_index.tolist() == [0, 0, 1, 2, 4, 5, 6, 7, 8, 8, 9, 10]
    # unused: pt 3 and pt 11 (x 3.5 of the even rows)
    assert (r.n_source, r.n_out, r.n_unused) == (12, 12, 2)


def test_a_coarser_output_step():
    # 3 rows x 5 columns, +0.5, out_step 2: x_max = 4 -> round(4 / 2) + 1 = 3 columns at 0 2 4;
    # y_max = 1.732 -> round(1.732 / 2) + 1 = 2 rows at y 0 and 2. Row 0 = pts 0..4, row 2 = pts 10..14
    # (y 1.732, 0.268 from y = 2; row 1 is 1.134 away). Cells take the even-row points at x 0 2 4.
    r = hex_resample.square_resampling(_hex(3, 5, +0.5), out_step=2.0)
    assert r.shape == (2, 3)
    assert r.step == 2.0
    assert r.point_index.tolist() == [0, 2, 4, 10, 12, 14]
    np.testing.assert_allclose(r.axes[0], [0.0, 2.0, 4.0])
    np.testing.assert_allclose(r.axes[1], [0.0, 2.0])


def test_the_axes_are_the_ones_the_oim_map_uses():
    r = hex_resample.square_resampling(_hex(3, 4, +0.5))
    np.testing.assert_array_equal(r.axes[0], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(r.axes[1], [0.0, 1.0, 2.0])


# ------------------------------------------------------------------ the step

DECIMAL_STEP = 0.15
WIDENED_STEP = 0.15000000596046448       # float32(0.15) written out as a float64, by hand


def test_a_float32_widened_step_gives_the_decimal_steps_choices():
    # OIM's ties depend on the decimal step the instrument states. Written by hand: the 3 x 4 case
    # has ties in square row 1 (cells 4 to 7); with the widened step two cells of it take another
    # point unless the step is normalised first. This test states the invariance only: which point the
    # float32 ties go to at 0.15 is decided by OIM's float32 arithmetic (the unit-step list does not
    # carry over), and the reference test below holds that against OIM's own file.
    decimal = hex_resample.square_resampling(_hex(3, 4, +0.5, step_x=DECIMAL_STEP))
    widened = hex_resample.square_resampling(_hex(3, 4, +0.5, step_x=WIDENED_STEP))
    assert widened.point_index.tolist() == decimal.point_index.tolist()
    assert widened.step == decimal.step == DECIMAL_STEP


def test_a_step_that_is_already_decimal_is_left_alone():
    for step in (1.0, 0.15, 2.5):
        g = _hex(3, 4, +0.5, step_x=step)
        assert hex_resample.oim_step_geometry(g) is g


def test_the_geometry_itself_keeps_its_step():
    g = _hex(3, 4, +0.5, step_x=WIDENED_STEP)
    hex_resample.square_resampling(g)
    assert g.step_x == WIDENED_STEP
    assert g.positions()[0][0, 1] == WIDENED_STEP


# ------------------------------------------------------------------ padded slots

def test_a_padded_slot_is_never_a_candidate():
    # 2 rows x 3 columns, +0.5. Row 0 sits at x 0 1 2, row 1 at 0.5 1.5 2.5. Row 0's last slot
    # (flat index 2, x = 2) is padding. Square grid 2 x 3 (round(0.866) = 1, +1): cells x 0 1 2,
    # y 0 1. Flat slots: row 0 = 0 1 2(pad), row 1 = 3 4 5.
    mask = np.array([[True, True, False], [True, True, True]])
    r = hex_resample.square_resampling(_hex(2, 3, +0.5, mask=mask))
    assert r.shape == (2, 3)
    assert 2 not in r.point_index.tolist()
    # cell (y 0, x 2) sits exactly on the padded slot; without the mask it takes it
    free = hex_resample.square_resampling(_hex(2, 3, +0.5))
    assert free.point_index[2] == 2
    # with the mask, the next best. The cell is at (x 2, y 0). Row 1 sits at x 0.5 1.5 2.5 (y 0.866):
    # pt 3 = (0.5, 0.866), pt 4 = (1.5, 0.866), pt 5 = (2.5, 0.866). Distances from (2, 0):
    # pt 1 at (1, 0) is 1 away; pt 4 and pt 5 are sqrt(0.25 + 0.75) = 1 away; pt 3 is
    # sqrt(2.25 + 0.75) = 1.73 away. Pts 1, 4 and 5 are all exactly 1 away, so any of the three is
    # correct; the padded slot is not
    assert r.point_index[2] in (1, 4, 5)
    # the cells that never touched the padded slot are the same: y = 1 takes row 1 (pts 3 3 4)
    assert r.point_index.tolist()[:2] == [0, 1]
    assert r.point_index.tolist()[3:] == [3, 3, 4]
    assert (r.n_source, r.n_out) == (5, 6)


def test_a_mask_that_changes_nothing_counts_the_valid_points():
    # Point 7 of the 3 x 4 case (x 3.5, last slot of row 1) is padding: the mapping is the one of
    # the unmasked case, and now nothing is unused.
    mask = np.ones((3, 4), dtype=bool)
    mask[1, 3] = False
    r = hex_resample.square_resampling(_hex(3, 4, +0.5, mask=mask))
    assert r.point_index.tolist() == [0, 1, 2, 3, 4, 4, 5, 6, 8, 9, 10, 11]
    assert (r.n_source, r.n_out, r.n_unused) == (11, 12, 0)


def test_a_mask_that_is_not_a_row_prefix_raises():
    mask = np.ones((3, 4), dtype=bool)
    mask[1, 1] = False                      # a hole inside a row, not padding at its end
    with pytest.raises(ValueError, match="(?i)mask"):
        hex_resample.square_resampling(_hex(3, 4, +0.5, mask=mask))


def test_a_row_without_a_point_raises():
    mask = np.ones((3, 4), dtype=bool)
    mask[1, :] = False
    with pytest.raises(ValueError, match="(?i)row"):
        hex_resample.square_resampling(_hex(3, 4, +0.5, mask=mask))


def test_a_square_geometry_is_refused():
    with pytest.raises(ValueError, match="(?i)hex"):
        hex_resample.square_resampling(ScanGeometry.square(3, 4, 1.0))


# ------------------------------------------------------------------ gather

def test_gather_takes_the_rows_of_the_chosen_points():
    r = hex_resample.square_resampling(_hex(3, 4, +0.5))
    arr = np.arange(12 * 4, dtype=np.float64).reshape(12, 4)       # row i = 4i .. 4i + 3
    out = r.gather(arr)
    assert out.shape == (12, 4)
    # point_index is [0,1,2,3, 4,4,5,6, 8,9,10,11] (hand case above); row i of arr starts at 4i
    expected_first_col = [0, 4, 8, 12, 16, 16, 20, 24, 32, 36, 40, 44]
    assert out[:, 0].tolist() == expected_first_col
    assert out[5].tolist() == [16, 17, 18, 19]


def test_gather_of_a_flat_channel():
    r = hex_resample.square_resampling(_hex(3, 4, -0.5))
    ci = np.arange(12) * 0.1
    out = r.gather(ci)
    # point_index [0,0,1,2, 4,5,6,7, 8,8,9,10]
    np.testing.assert_allclose(out, np.array([0, 0, 1, 2, 4, 5, 6, 7, 8, 8, 9, 10]) * 0.1)


@pytest.mark.parametrize("shape", [(11,), (13, 4), (4, 3), ()])
def test_gather_refuses_a_first_axis_that_is_not_the_point_count(shape):
    r = hex_resample.square_resampling(_hex(3, 4, +0.5))
    with pytest.raises(ValueError, match="(?i)points"):
        r.gather(np.zeros(shape))


def test_gather_does_not_return_none():
    # remap_point_values answers None for a size mismatch; the writers must not mistake that
    # for "no data", so gather raises
    r = hex_resample.square_resampling(_hex(3, 4, +0.5))
    with pytest.raises(ValueError):
        r.gather(np.zeros(5))


# ------------------------------------------------------------------ positions

def test_float32_positions_do_not_change_the_shared_ones():
    g = _hex(3, 4, -0.5)
    x_before, y_before = g.positions()
    hex_resample.float32_positions(g)
    x_after, y_after = g.positions()
    np.testing.assert_array_equal(x_before, x_after)
    np.testing.assert_array_equal(y_before, y_after)
    assert x_after.dtype == np.float64


def test_float32_positions_of_the_minus_half_scan_start_at_zero():
    # -0.5: the odd rows were at -0.5, +0.5 makes them 0; the even rows are at 0.5
    x, y = hex_resample.float32_positions(_hex(3, 4, -0.5))
    assert x.dtype == y.dtype == np.float32
    assert x[0].tolist() == [0.5, 1.5, 2.5, 3.5]
    assert x[1].tolist() == [0.0, 1.0, 2.0, 3.0]


# ------------------------------------------------------------------ the resampled map

POINT_INDEX_PLUS = [0, 1, 2, 3, 4, 4, 5, 6, 8, 9, 10, 11]       # the 3 x 4, +0.5 hand case above


def _crystal_map(n=12, k=None, phases=None, phase_id=None, prop=None):
    """A map on the index-space rectangle of a 3 x 4 scan: point i has phi1 = 10 i degrees."""
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    phi1 = np.deg2rad(10.0 * np.arange(n))
    if k is None:
        rot = Rotation.from_euler(np.stack([phi1, np.full(n, 0.3), np.full(n, 0.2)], axis=1))
    else:
        # match j of point i has phi2 = 0.1 * (j + 1)
        eul = np.zeros((n, k, 3))
        eul[:, :, 0] = phi1[:, None]
        eul[:, :, 1] = 0.3
        eul[:, :, 2] = 0.1 * (np.arange(k)[None, :] + 1)
        rot = Rotation.from_euler(eul)
    if phases is None:
        phases = PhaseList([Phase(name="Al", point_group="m-3m")])
    return CrystalMap(
        rotations=rot, phase_id=np.zeros(n, dtype=int) if phase_id is None else phase_id,
        x=np.tile(np.arange(4.0), 3)[:n], y=np.repeat(np.arange(3.0), 4)[:n],
        phase_list=phases, prop={} if prop is None else prop, scan_unit="um")


def test_the_resampled_map_takes_the_rows_of_the_chosen_points():
    r = hex_resample.square_resampling(_hex(3, 4, +0.5))
    xmap = _crystal_map(prop={"ci": np.arange(12) * 0.1, "fit": np.arange(12) * 1.0})
    out = hex_resample.resample_xmap(xmap, r)
    assert out.size == 12
    # phi1 of cell i is 10 * POINT_INDEX_PLUS[i] degrees (hand list from the case above)
    np.testing.assert_allclose(
        np.rad2deg(out.rotations.to_euler()[:, 0]), 10.0 * np.array(POINT_INDEX_PLUS), atol=1e-6)
    np.testing.assert_allclose(out.prop["ci"], 0.1 * np.array(POINT_INDEX_PLUS))
    np.testing.assert_allclose(out.prop["fit"], np.array(POINT_INDEX_PLUS, dtype=float))


def test_the_resampled_map_has_the_coordinates_of_the_axes():
    r = hex_resample.square_resampling(_hex(3, 4, +0.5))
    source = _crystal_map()
    source.scan_unit = "px"             # the unit of the index-space map is not the unit of the axes
    out = hex_resample.resample_xmap(source, r)
    # 3 rows x 4 columns at step 1, row-major
    np.testing.assert_array_equal(out.x, [0, 1, 2, 3] * 3)
    np.testing.assert_array_equal(out.y, [0] * 4 + [1] * 4 + [2] * 4)
    assert out.shape == (3, 4)
    assert out.scan_unit == "um"


def test_the_coordinates_use_the_decimal_step_not_the_widened_one():
    # A float32 header step arrives as 0.15000000596...; the file states 0.15. The axes are OIM's
    # float32 positions (index times the float32 step), so the coordinates are 0.15 i to within a
    # float32 ulp (about 1e-8 here), and the widened and the decimal step give the same ones.
    widened = hex_resample.resample_xmap(
        _crystal_map(), hex_resample.square_resampling(_hex(3, 4, +0.5, step_x=WIDENED_STEP)))
    decimal = hex_resample.resample_xmap(
        _crystal_map(), hex_resample.square_resampling(_hex(3, 4, +0.5, step_x=DECIMAL_STEP)))
    # three rows of four columns: the float32 axes are 4 distinct x and 3 distinct y values
    assert widened.shape == (3, 4) and decimal.shape == (3, 4)
    np.testing.assert_array_equal(widened.x, decimal.x)
    np.testing.assert_array_equal(widened.y, decimal.y)
    np.testing.assert_allclose(widened.x, 0.15 * np.tile(np.arange(4.0), 3), rtol=0, atol=1e-7)
    np.testing.assert_allclose(widened.y, 0.15 * np.repeat(np.arange(3.0), 4), rtol=0, atol=1e-7)
    assert widened.dx == pytest.approx(0.15, abs=1e-7)


def test_an_unindexed_point_stays_unindexed():
    # point 3 and point 4 are not indexed; cell 3 takes point 3 and cells 4 and 5 take point 4
    # (POINT_INDEX_PLUS), so cells 3, 4 and 5 are unindexed and nothing else is
    phase_id = np.zeros(12, dtype=int)
    phase_id[[3, 4]] = -1
    out = hex_resample.resample_xmap(
        _crystal_map(phase_id=phase_id), hex_resample.square_resampling(_hex(3, 4, +0.5)))
    assert out.phase_id.tolist() == [0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0]
    assert (~out.is_indexed).tolist() == [False] * 3 + [True] * 3 + [False] * 6


def test_the_phase_list_survives_with_its_frame_phase_and_space_group():
    from diffpy.structure import Lattice, Structure
    from orix.crystal_map import Phase, PhaseList
    from crystal_symmetry import FramePhase, frame_symmetry
    # Space group 162 (P-31m) has its two-fold axes along Y: the marked group "-31m" (its name is
    # "-3m", only the object tells the two settings apart). A plain orix Phase of the same space
    # group answers with the unmarked -3m.
    lattice = Lattice(4.8, 4.8, 4.5, 90, 90, 120)
    phase = FramePhase(name="Trig", space_group=162, structure=Structure(lattice=lattice))
    assert phase.point_group is frame_symmetry("-31m")
    assert Phase(name="x", space_group=162, structure=Structure(lattice=lattice)).point_group \
        is not frame_symmetry("-31m")
    out = hex_resample.resample_xmap(
        _crystal_map(phases=PhaseList([phase])), hex_resample.square_resampling(_hex(3, 4, +0.5)))
    got = out.phases[0]
    assert isinstance(got, FramePhase)
    assert got.name == "Trig"
    assert got.space_group.number == 162
    assert got.point_group is frame_symmetry("-31m")
    assert got.structure.lattice.a == pytest.approx(4.8)


def test_two_phases_keep_their_ids_and_names():
    from orix.crystal_map import Phase, PhaseList
    phases = PhaseList([Phase(name="A", point_group="m-3m"), Phase(name="B", point_group="6/mmm")])
    phase_id = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1])
    out = hex_resample.resample_xmap(
        _crystal_map(phases=phases, phase_id=phase_id), hex_resample.square_resampling(_hex(3, 4, +0.5)))
    # cells take points 0 1 2 3 4 4 5 6 8 9 10 11
    assert out.phase_id.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1]
    assert [p.name for _, p in out.phases] == ["A", "B"]


def test_a_phase_no_cell_takes_drops_out_and_the_others_keep_their_ids():
    # Phase B is at point 7 only, which no cell takes (the cells take 0 1 2 3 4 4 5 6 8 9 10 11).
    # orix leaves it out of the resampled map's phase list and does not renumber A (0) and C (2):
    # whoever writes a table of phases from this map must number its rows from the map itself.
    from orix.crystal_map import Phase, PhaseList
    phases = PhaseList([Phase(name="A", point_group="m-3m"), Phase(name="B", point_group="6/mmm"),
                        Phase(name="C", point_group="m-3m")])
    phase_id = np.array([0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2])
    source = _crystal_map(phases=phases, phase_id=phase_id)
    assert list(source.phases.ids) == [0, 1, 2]
    out = hex_resample.resample_xmap(source, hex_resample.square_resampling(_hex(3, 4, +0.5)))
    assert list(out.phases.ids) == [0, 2]
    assert [p.name for _, p in out.phases] == ["A", "C"]
    assert out.phase_id.tolist() == [0] * 8 + [2] * 4


def test_several_matches_per_point_are_gathered_by_rows():
    r = hex_resample.square_resampling(_hex(3, 4, +0.5))
    xmap = _crystal_map(k=3, prop={"scores": np.arange(36, dtype=float).reshape(12, 3)})
    assert xmap.rotations.shape == (12, 3)
    out = hex_resample.resample_xmap(xmap, r)
    assert out.size == 12
    assert out.rotations.shape == (12, 3)
    eul = out.rotations.to_euler()
    # cell 5 takes point 4: phi1 40 degrees, phi2 0.1 0.2 0.3 for the three matches
    assert np.rad2deg(eul[5, :, 0]).tolist() == pytest.approx([40.0] * 3, abs=1e-6)
    np.testing.assert_allclose(eul[5, :, 2], [0.1, 0.2, 0.3], atol=1e-9)
    # cell 8 takes point 8: phi1 80 degrees
    assert np.rad2deg(eul[8, 0, 0]) == pytest.approx(80.0, abs=1e-6)
    # scores row of point p is 3p, 3p + 1, 3p + 2: cell 5 -> point 4 -> 12 13 14
    assert out.prop["scores"][5].tolist() == [12.0, 13.0, 14.0]
    assert out.prop["scores"].shape == (12, 3)


def test_a_property_that_is_not_one_row_per_point_raises():
    r = hex_resample.square_resampling(_hex(3, 4, +0.5))
    for bad in (np.arange(7.0), np.arange(24.0)):
        with pytest.raises(ValueError, match="(?i)odd"):
            hex_resample.resample_xmap(_crystal_map(prop={"odd": bad}), r)


def test_a_map_with_another_point_count_than_the_scan_raises():
    r = hex_resample.square_resampling(_hex(3, 4, +0.5))
    with pytest.raises(ValueError, match="(?i)points"):
        hex_resample.resample_xmap(_crystal_map(n=8), r)


def test_the_source_map_is_left_alone():
    r = hex_resample.square_resampling(_hex(3, 4, +0.5))
    xmap = _crystal_map(prop={"ci": np.arange(12) * 0.1})
    before = xmap.rotations.data.copy()
    hex_resample.resample_xmap(xmap, r)
    assert xmap.size == 12
    np.testing.assert_array_equal(xmap.rotations.data, before)
    np.testing.assert_allclose(xmap.prop["ci"], np.arange(12) * 0.1)


# ------------------------------------------------------------------ the note

NOTE_CONFIRMED = (
    "Resampled from a hexagonal scan to a square grid by nearest neighbour at step 1 um "
    "(source points 12, points written 12, source points unused 1); nothing was interpolated.")


def _confirmed(shift=+0.5):
    return ScanGeometry.hex(3, 4, 1.0, odd_row_shift=shift, shift_confirmed=True)


def test_the_note_gives_the_numbers():
    from backend.api.services import ang_export
    g = _confirmed()
    r = hex_resample.square_resampling(g)
    # 3 x 4 hand case: 12 source points, 12 cells, point 7 unused
    assert ang_export.resampling_note(r, g) == NOTE_CONFIRMED


def test_the_note_gives_the_decimal_step():
    from backend.api.services import ang_export
    g = ScanGeometry.hex(3, 4, WIDENED_STEP, odd_row_shift=+0.5, shift_confirmed=True)
    note = ang_export.resampling_note(hex_resample.square_resampling(g), g)
    assert "at step 0.15 um" in note
    assert "0.1500000" not in note


def test_the_note_names_the_assumption_exactly_when_the_direction_is_unconfirmed():
    from backend.api.services import ang_export
    for shift, side in ((+0.5, "right"), (-0.5, "left")):
        unconfirmed = ScanGeometry.hex(3, 4, 1.0, odd_row_shift=shift, shift_confirmed=False)
        note = ang_export.resampling_note(hex_resample.square_resampling(unconfirmed), unconfirmed)
        assert "assumption" in note
        assert f"half a step to the {side} of the even rows" in note
        # (the unused count differs with the direction: 1 for +0.5, 2 for -0.5, see above)
        assert note.startswith(NOTE_CONFIRMED[:NOTE_CONFIRMED.index("source points unused")])
    confirmed = _confirmed()
    assert "assum" not in ang_export.resampling_note(hex_resample.square_resampling(confirmed), confirmed)


def test_the_note_is_one_ascii_line_without_a_tab():
    from backend.api.services import ang_export
    for confirmed in (True, False):
        g = ScanGeometry.hex(3, 4, 1.0, odd_row_shift=+0.5, shift_confirmed=confirmed)
        note = ang_export.resampling_note(hex_resample.square_resampling(g), g)
        assert note.isascii()
        assert note == " ".join(note.split()), "no tab, no newline, no double space"
        assert not note.endswith(" ")


def test_the_header_writes_the_notes_as_orienta_note_lines():
    from backend.api.services import ang_export
    lines = ang_export.header_comments(
        method=None, vendor="edax", provenance={"text": "x"}, step_um=1.0, notes=[NOTE_CONFIRMED])
    assert f"# ORIENTA_NOTE: {NOTE_CONFIRMED}" in lines


def test_a_hexagonal_file_keeps_its_own_notes_and_the_given_ones():
    from backend.api.services import ang_export
    g = ScanGeometry.hex(3, 4, 1.0, odd_row_shift=+0.5, shift_confirmed=False)
    lines = ang_export.header_comments(
        method=None, vendor="edax", provenance={"text": "x"}, step_um=1.0, grid=g, notes=["extra note"])
    assert "# ORIENTA_NOTE: extra note" in lines
    assert any("The direction of the row offset is an assumption" in ln for ln in lines)


SCAN = {"scan_row_offset": 2, "scan_col_offset": 3, "scan_shape": (30, 40)}


def test_a_resampled_file_states_its_origin_in_micrometres_only():
    from backend.api.services import ang_export
    g = _confirmed()
    lines = ang_export.header_comments(
        method=None, vendor="edax", provenance={"text": "x"}, scan=SCAN, step_um=1.0, resampled_from=g)
    origin = next(ln for ln in lines if ln.startswith("# ORIENTA_SCAN_ORIGIN"))
    # column 3 at step_x 1 is x 3.000 um; row 2 at step_y sqrt(3)/2 is y 1.732 um
    assert "x 3.000 um, y 1.732 um" in origin
    assert "row 2" not in origin and "column 3" not in origin
    # the same crop of a square scan keeps its sentence
    square = ang_export.header_comments(
        method=None, vendor="edax", provenance={"text": "x"}, scan=SCAN, step_um=1.0)
    assert "row 2, column 3" in next(ln for ln in square if ln.startswith("# ORIENTA_SCAN_ORIGIN"))


def test_a_resampled_file_that_is_not_a_crop_says_so_without_row_and_column():
    from backend.api.services import ang_export
    scan = {"scan_row_offset": 0, "scan_col_offset": 0, "scan_shape": (3, 4)}
    lines = ang_export.header_comments(
        method=None, vendor="edax", provenance={"text": "x"}, scan=scan, step_um=1.0,
        resampled_from=_confirmed())
    origin = next(ln for ln in lines if ln.startswith("# ORIENTA_SCAN_ORIGIN"))
    assert "not a crop" in origin
    assert "row 0, column 0" not in origin


def test_a_resampled_file_cannot_also_be_a_hexagonal_one():
    from backend.api.services import ang_export
    with pytest.raises(ValueError, match="(?i)resampled"):
        ang_export.header_comments(
            method=None, vendor="edax", provenance={"text": "x"}, step_um=1.0,
            grid=_confirmed(), resampled_from=_confirmed())


def _prj(**kw):
    from backend.api.services import ang_export
    return ang_export.ctf_project_note(vendor="edax", provenance={"text": "assigned"}, scan=None,
                                       step_um=1.0, **kw)


def test_without_notes_the_ctf_project_note_is_what_it_was():
    from backend.api.services import ang_export
    expected = ("Euler angles: " + ang_export.ctf_frame_text("edax")
                + " Scan origin: not recorded by this export. Phase assignment: assigned")
    assert _prj() == expected
    assert _prj(notes=()) == expected


def test_the_notes_are_appended_to_the_ctf_project_note():
    prj = _prj(notes=[NOTE_CONFIRMED, "second"])
    assert prj.endswith(f" Note: {NOTE_CONFIRMED} Note: second")
    assert "\t" not in prj and "\n" not in prj


def test_a_resampled_ctf_states_its_origin_in_micrometres_only():
    from backend.api.services import ang_export
    prj = ang_export.ctf_project_note(
        vendor="edax", provenance={"text": "assigned"}, scan=SCAN, step_um=1.0,
        resampled_from=_confirmed())
    assert "x 3.000 um, y 1.732 um" in prj
    assert "row 2" not in prj and "column 3" not in prj


# ------------------------------------------------------------------ OIM's own files

def _reference_file(name: str) -> Path:
    folder = os.environ.get(ENV)
    if not folder:
        pytest.skip(f"{ENV} is not set: OIM's hexagonal reference files are not available")
    path = Path(folder) / name
    if not path.is_file():
        pytest.skip(f"{ENV} is set, but {name} is not in that folder")
    return path


def _read_hex(path: Path):
    """``(geometry, x, y, channels)`` of OIM's scan: the geometry is built from the header
    alone, with the mask taken from the sentinel positions of the file."""
    import h5py
    with h5py.File(path, "r") as f:
        scan = [k for k in f if isinstance(f[k], h5py.Group) and "EBSD" in f[k]][0]
        ebsd = f[scan]["EBSD"]
        header = ebsd["Header"]
        assert header["Grid Type"][0] in (b"HexGrid", "HexGrid")
        n_rows, n_cols = int(header["nRows"][0]), int(header["nColumns"][0])
        # The header stores the step as float32. The decimal the file holds is the shortest
        # one that reads back as the same float32 (0.15 for 0.1500000059604...).
        step_x = float(str(np.float32(header["Step X"][0])))
        data = ebsd["Data"]
        x = data["X Position"][()].reshape(n_rows, n_cols)
        y = data["Y Position"][()].reshape(n_rows, n_cols)
        channels = {k: data[k][()] for k in ("CI", "IQ", "Fit", "Phase", "Phi1", "Phi", "Phi2")}
    mask = x > -1e5
    assert np.array_equal(mask, y > -1e5)
    geometry = ScanGeometry(
        kind="hex", n_rows=n_rows, n_cols=n_cols, step_x=step_x, step_y=step_x * SQRT3_2,
        odd_row_shift=0.5, shift_confirmed=False, steps_known=True, valid_mask=mask)
    return geometry, x, y, channels


def _read_square(path: Path):
    import h5py
    with h5py.File(path, "r") as f:
        scan = [k for k in f if isinstance(f[k], h5py.Group) and "EBSD" in f[k]][0]
        data = f[scan]["EBSD"]["Data"]
        return {k: data[k][()] for k in ("CI", "IQ", "Fit", "Phase", "Phi1", "Phi", "Phi2")}


def test_the_float32_positions_are_the_ones_oim_stored():
    geometry, x, y, _ = _read_hex(_reference_file("prim_phase.h5"))
    px, py = hex_resample.float32_positions(geometry)
    valid = geometry.valid_mask
    assert px.dtype == py.dtype == np.float32
    assert int((px[valid] != x[valid]).sum()) == 0, "x positions differ from the stored float32"
    assert int((py[valid] != y[valid]).sum()) == 0, "y positions differ from the stored float32"


@pytest.mark.parametrize("step_form", ["decimal", "float32 widened"])
def test_the_resampling_reproduces_oims_square_scan(step_form):
    geometry, _, _, hexch = _read_hex(_reference_file("prim_phase.h5"))
    if step_form == "float32 widened":
        # the step as a float32 header value becomes when it is read as a float64
        step = float(np.float32(geometry.step_x))
        assert step != geometry.step_x
        geometry = ScanGeometry(
            kind="hex", n_rows=geometry.n_rows, n_cols=geometry.n_cols, step_x=step,
            step_y=step * SQRT3_2, odd_row_shift=0.5, shift_confirmed=False, steps_known=True,
            valid_mask=geometry.valid_mask)
    square = _read_square(_reference_file("prim_phase_square.h5"))
    r = hex_resample.square_resampling(geometry)

    # OIM's padded rectangle: 397 x 118 slots, 46 648 real points; its square scan 344 x 118
    assert geometry.n_points == 46846
    assert r.n_source == 46648
    assert r.shape == (344, 118)
    assert r.n_out == 40592 == square["CI"].shape[0]

    compared = 0
    for name in ("CI", "IQ", "Fit", "Phase"):
        ours, theirs = r.gather(hexch[name]), square[name]
        assert ours.shape == theirs.shape
        deviations = int((ours != theirs).sum())
        compared += theirs.size
        assert deviations == 0, f"{name}: {deviations} of {theirs.size} values differ from OIM"
    for name in ("Phi1", "Phi", "Phi2"):
        ours, theirs = r.gather(hexch[name]), square[name]
        compared += theirs.size
        # OIM rewrites the Euler angles through its own arithmetic: the last float32 bit can differ
        assert np.allclose(ours, theirs, rtol=0, atol=1e-7), f"{name} differs by more than float32 rounding"
    assert compared == 7 * 40592


def test_the_note_of_the_reference_scan_gives_oims_counts():
    from backend.api.services import ang_export
    geometry, _, _, _ = _read_hex(_reference_file("prim_phase.h5"))
    r = hex_resample.square_resampling(geometry)
    note = ang_export.resampling_note(r, geometry)
    # 46 648 real points, 40 592 cells written (the numbers OIM's two files hold)
    assert "source points 46648," in note
    assert "points written 40592," in note
    assert f"source points unused {46648 - len(set(r.point_index.tolist()))})" in note
    assert "at step 0.15 um" in note
