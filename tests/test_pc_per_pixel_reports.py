"""What the PC Refinement routes report as "the PC" once a PC map is active.

After a grid calibration the controller's detector holds one PC per scan pixel.
``/detector/info`` and ``/status`` used to flatten it: the first three numbers
(pixel (0, 0)) stood in for "the PC" and the rest, rows x cols x 3 floats (179,760
for a 300 x 200 scan), went over the wire. Now they report three numbers, say
whether those are the map mean or a pixel's PC, describe the map in a few
numbers, and never send it.

The test map is a plane with hand-set slopes, so every expected value below is
arithmetic on the slopes and not a call into the code under test.
"""
import asyncio
import json

import numpy as np
import pytest
from kikuchipy.detectors import EBSDDetector

from pc_controller import PCController

NR, NC = 300, 200
# PC(row, col) = (0.50 + 1e-4 col, 0.40 + 2e-4 row, 0.60 + 1e-4 (row + col))
SLOPE_X, SLOPE_Y, SLOPE_Z = 1e-4, 2e-4, 1e-4


def _plane_detector():
    rows, cols = np.mgrid[0:NR, 0:NC]
    pcs = np.stack([0.50 + SLOPE_X * cols, 0.40 + SLOPE_Y * rows,
                    0.60 + SLOPE_Z * (rows + cols)], axis=-1)
    return EBSDDetector(shape=(60, 60), pc=pcs, sample_tilt=70.0)


def _expected_at(row, col):
    return [0.50 + SLOPE_X * col, 0.40 + SLOPE_Y * row, 0.60 + SLOPE_Z * (row + col)]


def _expected_mean():
    # the mean of a plane is its value at the middle of the scan
    return _expected_at((NR - 1) / 2, (NC - 1) / 2)


@pytest.fixture
def ctrl(monkeypatch):
    from backend.api.routes import pcrefinement as R
    c = PCController()
    c.apply_pixelwise_pc(_plane_detector())
    monkeypatch.setattr(R, "_get_controller", lambda: c)
    return c


def _info(**kw):
    from backend.api.routes import pcrefinement as R
    return asyncio.run(R.detector_info(**kw))


def _status(**kw):
    from backend.api.routes import pcrefinement as R
    return asyncio.run(R.pc_status(**kw))


def test_detector_info_reports_the_map_mean_not_pixel_zero(ctrl):
    info = _info()
    assert info["pc_per_pixel"] is True
    assert info["pc_basis"] == "mean"
    np.testing.assert_allclose(info["pc"], _expected_mean(), atol=1e-9)
    # pixel (0, 0) is what flatten()[:3] gave; the mean must not be it
    assert abs(info["pc"][1] - _expected_at(0, 0)[1]) > 0.01


def test_detector_info_does_not_ship_the_map(ctrl):
    info = _info()
    lists = [v for v in info.values() if isinstance(v, list)]
    assert lists and max(len(v) for v in lists) <= 3
    assert len(json.dumps(info)) < 3000          # 180000 floats were ~3 MB
    assert info["pc_map_shape"] == [NR, NC]
    span = [SLOPE_X * (NC - 1), SLOPE_Y * (NR - 1), SLOPE_Z * (NR - 1 + NC - 1)]
    np.testing.assert_allclose(info["pc_span"], span, atol=1e-9)
    for key in ("pc_tsl", "pc_bruker", "pc_oxford"):
        assert len(info[key]) == 3


def test_detector_info_at_a_pixel_reports_that_pixel(ctrl):
    info = _info(row=100, col=50)
    assert info["pc_basis"] == "pixel"
    np.testing.assert_allclose(info["pc"], _expected_at(100, 50), atol=1e-9)
    # off the map: the mean, not an error and not the first pixel
    off = _info(row=NR + 5, col=3)
    assert off["pc_basis"] == "mean"
    np.testing.assert_allclose(off["pc"], _expected_mean(), atol=1e-9)


def test_status_reports_three_numbers_and_the_basis(ctrl):
    ctrl.add_pattern((10, 20), np.zeros((60, 60), np.uint8))
    st = _status()
    assert st["pc_per_pixel"] is True and st["pc_basis"] == "mean"
    np.testing.assert_allclose(st["pc"], _expected_mean(), atol=1e-9)   # length 3, so
    assert len(st["pc"]) == 3                                           # the page uses it
    assert len(json.dumps(st)) < 3000
    one = _status(pattern=0)
    assert one["pc_basis"] == "pixel"
    np.testing.assert_allclose(one["pc"], _expected_at(10, 20), atol=1e-9)


def test_a_single_pc_detector_is_reported_as_it_always_was():
    from backend.api.routes import pcrefinement as R
    c = PCController()
    c.attach_detector(EBSDDetector(shape=(60, 60), pc=(0.51, 0.32, 0.84), sample_tilt=70.0))
    old = R._get_controller
    R._get_controller = lambda: c
    try:
        info = asyncio.run(R.detector_info())
    finally:
        R._get_controller = old
    assert info["pc_basis"] == "single" and info["pc_per_pixel"] is False
    assert info["pc_map_shape"] is None
    np.testing.assert_allclose(info["pc"], (0.51, 0.32, 0.84), atol=1e-12)
