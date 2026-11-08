"""A grid calibration keeps the detector's tilts; the fit's estimate is only reported.

kikuchipy's ``fit_pc`` re-estimates the sample tilt from the plane it fits and puts
it on the detector it returns. That detector became the PC controller's, so one
grid run silently changed the sample tilt of the page (on synthetic PCs 70 degrees
became 39.7 degrees) while the calibration store kept 70: the page and indexing
disagreed, and a noisy fit changed the geometry. The tilts the user set stay; the
fitted sample tilt is reported next to the result for the user to compare.
"""
import numpy as np
import pytest
from kikuchipy.detectors import EBSDDetector

from pc_controller import PCController

N_ROWS, N_COLS, STEP = 40, 50, 20


@pytest.fixture
def setup():
    c = PCController()
    c.attach_detector(EBSDDetector(shape=(60, 60), pc=(0.5, 0.5, 0.5), sample_tilt=70.0,
                                   tilt=7.5, azimuthal=1.25))
    grid = c.generate_calibration_grid(N_ROWS, N_COLS, step=STEP)
    refined = np.array([[0.45 + 0.002 * cc, 0.55 - 0.001 * r, 0.50 + 0.0005 * (r + cc)]
                        for (r, cc) in grid])
    return c, grid, refined


def test_the_fitted_detector_keeps_the_tilts_of_the_page(setup):
    c, grid, refined = setup
    det = c.fit_pc_from_grid(grid, refined, N_ROWS, N_COLS)
    assert (det.sample_tilt, det.tilt, det.azimuthal) == (70.0, 7.5, 1.25)


def test_the_estimate_of_the_fit_is_kept_for_the_report(setup):
    c, grid, refined = setup
    c.fit_pc_from_grid(grid, refined, N_ROWS, N_COLS)
    # what kikuchipy's own fit makes of the same PCs, called directly
    ref = EBSDDetector(shape=(60, 60), px_size=1.0, binning=1, tilt=7.5, azimuthal=1.25,
                       sample_tilt=70.0, pc=refined)
    rows, cols = np.mgrid[0:N_ROWS, 0:N_COLS]
    direct = ref.fit_pc(pc_indices=np.array(grid).T, map_indices=np.array([rows, cols]),
                        transformation="affine", plot=False)
    assert abs(direct.sample_tilt - 70.0) > 1.0                  # the estimate differs from the set tilt
    assert c.last_fitted_sample_tilt == pytest.approx(direct.sample_tilt, abs=1e-9)


def test_the_extrapolated_detector_keeps_the_tilts_too(setup):
    c, grid, refined = setup
    det = c.extrapolate_pc_from_points(grid, refined, N_ROWS, N_COLS, step_sizes=(1.0, 1.0))
    assert (det.sample_tilt, det.tilt, det.azimuthal) == (70.0, 7.5, 1.25)
