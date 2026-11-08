"""The PC in an .ang header is the EDAX TSL one, not the Bruker numbers.

(Verification of a fix that shipped in 0.4.7: the export wrote the Bruker PCy as
y-star, so OIM and MTEX read 0.327 where 0.673 was meant.) The numbers are
The Oxford reference scan's: Bruker PC (0.5026, 0.3272, 0.8458) on a 128 x 156 detector. By hand:
x* = PCx, y* = 1 - PCy, z* = PCz * rows / min(rows, cols), and rows = min here.
"""
import pytest

from backend.api.services import ang_export

BRUKER = (0.5026, 0.3272, 0.8458)


def test_tsl_pc_of_a_landscape_detector():
    x, y, z = ang_export.pc_tsl_from_bruker(BRUKER, (128, 156))
    assert (x, y, z) == pytest.approx((0.5026, 1 - 0.3272, 0.8458))


def test_tsl_pc_of_a_portrait_detector_scales_z_by_the_aspect():
    x, y, z = ang_export.pc_tsl_from_bruker(BRUKER, (156, 128))
    assert z == pytest.approx(0.8458 * 156 / 128)


def test_the_header_lines_carry_the_tsl_values():
    tsl = ang_export.pc_tsl_from_bruker(BRUKER, (128, 156))
    rep = ang_export.header_replacements(pc_tsl=tsl)
    assert rep["# x-star"].split()[-1] == "0.502600"
    assert rep["# y-star"].split()[-1] == "0.672800"        # not the Bruker 0.327200
    assert rep["# z-star"].split()[-1] == "0.845800"


def test_without_the_detector_shape_no_pc_is_invented():
    assert ang_export.pc_tsl_from_bruker(BRUKER, None) is None
