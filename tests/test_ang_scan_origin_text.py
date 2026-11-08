"""The scan-origin sentence of a hexagonal crop states where its first pixel really is.

A crop that starts on an odd row of a hexagonal scan has its first pixel half a step to the side of
``column * step`` (the odd rows of the original are shifted), and a file whose odd rows lie to the left
(offset -0.5) has its x raised by half a step, so "coordinates start at 0" was false for it.

Hand values: step_x 1.0 um, step_y = sqrt(3)/2 = 0.866 um. In the original scan (offset +0.5) the point at row 1,
column 2 is at x = 2 * 1.0 + 0.5 = 2.5, y = 0.866; at row 1, column 0 of an original with offset -0.5 it is at
x = -0.5. The positions are read off the original's own geometry, never from the sentence.
"""
from __future__ import annotations

import pytest

from backend.api.services import ang_export
from backend.api.services.scan_geometry import ScanGeometry

SX = 1.0
SY = SX * 3 ** 0.5 / 2
SHAPE = (10, 10)


def _crop(original_shift, r0, c0):
    original = ScanGeometry.hex(*SHAPE, SX, odd_row_shift=original_shift, shift_confirmed=True)
    return original, original.cropped(r0, c0, 3, 4)


def _text(original_shift, r0, c0, **kw):
    original, crop = _crop(original_shift, r0, c0)
    return ang_export.scan_origin_text(r0, c0, SHAPE, SX, step_y_um=SY, geometry=crop, **kw)


def test_the_hand_positions_of_the_original():
    original, crop = _crop(0.5, 1, 2)
    x, y = original.positions()
    assert (x[1, 2], round(float(y[1, 2]), 3)) == (2.5, 0.866)           # 2 * 1.0 + 0.5
    assert crop.odd_row_shift == -0.5                                      # a crop at an odd row reverses it
    original, _ = _crop(-0.5, 1, 0)
    assert original.positions()[0][1, 0] == -0.5


def test_a_crop_at_an_odd_row_of_a_plus_half_scan_states_the_half_step_and_the_lift():
    text = _text(0.5, 1, 2)
    assert "row 1, column 2 of the original 10 x 10 scan (0-based)" in text
    assert "lies at x 2.500 um, y 0.866 um of the original scan" in text       # not 2.000
    assert "raised by half a step (0.500 um)" in text
    assert "first pixel is at x 0.500 um, y 0.000 um" in text
    assert "origin of its coordinates lies at x 2.000 um, y 0.866 um of the original scan" in text
    assert "start at 0" not in text                                             # it was false


def test_a_crop_at_an_odd_row_of_a_minus_half_scan_starts_at_zero_half_a_step_left():
    text = _text(-0.5, 1, 0)
    # original x of row 1, column 0 = -0.5; the crop's offset is +0.5, so nothing is lifted
    assert "Coordinates in this file start at 0" in text
    assert "origin lies at x -0.500 um, y 0.866 um of the original scan" in text


def test_a_crop_at_an_even_row_of_a_plus_half_scan_is_as_before():
    text = _text(0.5, 2, 1)
    assert "Coordinates in this file start at 0" in text
    assert "origin lies at x 1.000 um, y 1.732 um of the original scan" in text
    assert "raised" not in text


def test_a_crop_at_an_even_row_of_a_minus_half_scan_is_lifted():
    text = _text(-0.5, 2, 1)
    assert "lies at x 1.000 um, y 1.732 um of the original scan" in text
    assert "raised by half a step (0.500 um)" in text
    assert "origin of its coordinates lies at x 0.500 um, y 1.732 um of the original scan" in text


@pytest.mark.parametrize("original_shift", [0.5, -0.5])
@pytest.mark.parametrize("r0, c0", [(0, 2), (1, 0), (1, 3), (2, 1), (3, 2)])
def test_the_stated_first_pixel_is_the_position_in_the_original(original_shift, r0, c0):
    """For every start: the sentence's first-pixel position is the original geometry's position of that
    point, and the origin of the file is that position minus the first pixel's x in the file."""
    original, crop = _crop(original_shift, r0, c0)
    ox, oy = original.positions()
    x_first, y_first = float(ox[r0, c0]), float(oy[r0, c0])
    lift = 0.5 * SX if crop.odd_row_shift < 0 else 0.0                       # hand rule of the .ang
    text = ang_export.scan_origin_text(r0, c0, SHAPE, SX, step_y_um=SY, geometry=crop)
    if lift:
        assert f"lies at x {x_first:.3f} um, y {y_first:.3f} um of the original scan" in text
        assert f"origin of its coordinates lies at x {x_first - lift:.3f} um, y {y_first:.3f} um" in text
    else:
        assert f"origin lies at x {x_first:.3f} um, y {y_first:.3f} um of the original scan" in text


@pytest.mark.parametrize("original_shift", [0.5, -0.5])
def test_a_resampled_file_states_the_same_first_pixel_position(original_shift):
    """The resampled square file's cell 0 is the crop's first pixel; its coordinates start at 0 there. The
    position in the original is the same as the .ang states (x 2.500 for the +0.5 original at row 1,
    column 2; x 1.500 for the -0.5 one)."""
    original, crop = _crop(original_shift, 1, 2)
    x_first = float(original.positions()[0][1, 2])
    assert x_first == (2.5 if original_shift > 0 else 1.5)
    text = _text(original_shift, 1, 2, resampled=True)
    assert f"lies at x {x_first:.3f} um, y 0.866 um of the original scan" in text
    assert "Coordinates in this file start at 0, at that pixel" in text
    hex_text = _text(original_shift, 1, 2)
    assert f"x {x_first:.3f} um, y 0.866 um" in hex_text                        # the same numbers


def test_without_a_geometry_the_sentence_is_the_old_one():
    text = ang_export.scan_origin_text(3, 2, (90, 120), 2.0, step_y_um=3 ** 0.5)
    assert "Coordinates in this file start at 0; the origin lies at x 4.000 um, y 5.196 um" in text


def test_the_resampled_files_state_the_position_through_their_header_and_prj_lines():
    """The square .ang (header comment) and the .ctf (Prj note) of a hexagonal crop: the crop at row 1,
    column 2 of the +0.5 original (a -0.5 geometry) lies at x 2.500 um, y 0.866 um of it (not 2.000)."""
    _, crop = _crop(0.5, 1, 2)
    scan = {"scan_row_offset": 1, "scan_col_offset": 2, "scan_shape": [10, 10]}
    lines = ang_export.header_comments(
        method=None, vendor="edax", provenance={"text": "t"}, scan=scan, step_um=SX, resampled_from=crop)
    origin = next(ln for ln in lines if ln.startswith("# ORIENTA_SCAN_ORIGIN"))
    assert "lies at x 2.500 um, y 0.866 um of the original scan" in origin
    prj = ang_export.ctf_project_note(
        vendor="edax", provenance={"text": "t"}, scan=scan, step_um=SX, resampled_from=crop)
    assert "lies at x 2.500 um, y 0.866 um of the original scan" in prj


def test_the_header_line_of_a_hexagonal_crop_says_it(tmp_path):
    """Through ``header_comments``: a -0.5 crop at row 3, column 2 of a scan with step 2.0 (the original's
    odd rows at +0.5: x = 2 * 2.0 + 0.5 * 2.0 = 5.0), lifted by 0.5 * 2.0 = 1.0."""
    crop = ScanGeometry.hex(3, 4, 2.0, odd_row_shift=-0.5, shift_confirmed=True)
    lines = ang_export.header_comments(
        method=None, vendor="edax", provenance={"text": "t"},
        scan={"scan_row_offset": 3, "scan_col_offset": 2, "scan_shape": [90, 120]},
        step_um=2.0, grid=crop)
    origin = next(ln for ln in lines if ln.startswith("# ORIENTA_SCAN_ORIGIN"))
    assert "lies at x 5.000 um, y 5.196 um of the original scan" in origin
    assert "first pixel is at x 1.000 um, y 0.000 um" in origin
    assert "origin of its coordinates lies at x 4.000 um, y 5.196 um" in origin
    unrecorded = ang_export.header_comments(
        method=None, vendor="edax", provenance={"text": "t"}, scan=None, step_um=2.0, grid=crop)
    text = next(ln for ln in unrecorded if ln.startswith("# ORIENTA_SCAN_ORIGIN"))
    assert "raised by half a step (1.000 um)" in text and "start at 0" not in text
