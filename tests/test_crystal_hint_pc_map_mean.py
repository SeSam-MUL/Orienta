"""Crystal Hint's detector geometry takes the MEAN of a per-pixel PC map.

A grid calibration leaves a PC map (rows x cols x 3) on the detector. The lattice
estimate needs one PC; the old code flattened the map and took the first triplet,
the PC of pixel (0, 0) - a corner of the scan, where the PC differs most from the
centre. Numbers below are chosen so the mean, the corner and the centre differ.
"""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from backend.api.routes.crystal_hint import _gather_detector_geom

# a 2 x 3 map: PCx runs 0.40 .. 0.60 along the columns, PCy 0.30 / 0.50 down the rows
MAP = np.array([
    [[0.40, 0.30, 0.70], [0.50, 0.30, 0.74], [0.60, 0.30, 0.78]],
    [[0.40, 0.50, 0.70], [0.50, 0.50, 0.74], [0.60, 0.50, 0.78]],
])


def _geom(pc):
    det = SimpleNamespace(pc=pc)
    signal = SimpleNamespace(axes_manager=SimpleNamespace(signal_shape=(118, 118)), detector=det)
    with patch("backend.api.routes.ebsd_viewer._get_active_signal", return_value=signal), \
         patch("backend.api.routes.ebsd_viewer._active_dataset", "ds"), \
         patch("backend.api.routes.ebsd_viewer._ebsd_file_path", None), \
         patch("backend.api.services.calibration_store.calibration_store.get_detector",
               return_value=det):
        return _gather_detector_geom()


def test_a_pc_map_gives_its_mean_not_the_corner_pixel():
    g = _geom(MAP)
    np.testing.assert_allclose((g["pc_x"], g["pc_y"], g["pc_z"]), (0.5, 0.4, 0.74), atol=1e-12)   # by hand: 3.0/6, 2.4/6, 4.44/6
    assert g["pc_x"] != MAP[0, 0, 0] and g["pc_y"] != MAP[0, 0, 1]


def test_a_single_pc_is_unchanged():
    g = _geom(np.array([0.51, 0.32, 0.81]))
    assert (g["pc_x"], g["pc_y"], g["pc_z"]) == (0.51, 0.32, 0.81)


def test_a_malformed_pc_gives_no_geometry():
    assert _geom(np.array([0.5, 0.5])) is None
