"""The camera elevation must reach PyEBSDIndex, or every Hough orientation is
rigidly wrong by exactly that angle -- and nothing in Hough's own output says so.

PyEBSDIndex builds its detector-to-sample rotation from
``tiltang = -(90 - sampleTilt + camElev)``
(``pyebsdindex/_ebsd_index_single.py``, ``BandIndexer._detector2refframe``), and
kikuchipy feeds it ``sampleTilt=detector.sample_tilt``, ``camElev=detector.tilt``
(``kikuchipy/indexing/_hough_indexing.py::_get_indexer_from_detector``). That is
the SAME combination the spherical route uses for its ``alpha`` angle
(``backend/spherical_gpu/pipeline/indexer.py``), so the two methods agree only as
long as both are handed the same elevation.

Losing it is a silent failure, and that is the point of these tests:

* The elevation only ROTATES the answer. It does not change band detection, so
  the Hough confidence index is bit-identical with and without it -- measured on
  ``Test_data/LoGainNi.h5`` (EDAX, camElev 10 deg): median CI 0.8020 at
  camElev = 10.0, 0.0 and 5.3 alike. A quality metric cannot catch this.
* The error is exactly the elevation, rigidly, on every pixel and both vendors.
  Measured 2026-09-12: Oxford SampleB (camElev 4.3226 deg) 4.323 deg median;
  EDAX LoGainNi (camElev 10 deg) 10.000 deg median with an IQR of 0.000.
* What does catch it is the forward render. On 12 high-CI Al pixels of the
  ``crop1`` Oxford scan (camElev 4.3107 deg) the render-NCC of the Hough answer
  is 0.833 with the elevation and **-0.247** without it (the verified spherical
  route scores 0.834 on the same pixels) -- the render-NCC peak is only a couple
  of degrees wide, so 4.3 deg off is already anti-correlated.

History: a measurement harness written for the ICAA20 talk built its Hough
reference detector without the elevation and produced a 4.20 / 3.74 / 4.31 deg
"Hough vs spherical" disagreement per phase that looked like a product bug. It
was not -- ``hough_index_patterns`` threads the detector through untouched. These
tests pin that so the next refactor cannot quietly reintroduce it.
"""
from __future__ import annotations

import numpy as np
import pytest

# A deliberately odd angle: it can only show up on the indexer if it was read
# off the detector rather than defaulted.
CAM_ELEV = 4.3226
SAMPLE_TILT = 70.0031


def _al_phase_list(cif_path):
    from orix.crystal_map import Phase, PhaseList

    from ebsd_utils import sanitize_cif

    phase = Phase.from_cif(sanitize_cif(str(cif_path)))
    phase.name = "Al"
    return PhaseList(phase)


def test_create_indexer_passes_camera_tilt_to_pyebsdindex(al_cif_path):
    """``create_indexer`` must hand the detector's elevation on as ``camElev``."""
    pytest.importorskip("pyebsdindex")
    from kikuchipy.detectors import EBSDDetector

    from ebsd_utils import create_indexer, prepare_reflectors

    detector = EBSDDetector(
        shape=(128, 156),
        pc=(0.5026, 0.3273, 0.8458),
        sample_tilt=SAMPLE_TILT,
        tilt=CAM_ELEV,
        convention="bruker",
    )
    phase_list = _al_phase_list(al_cif_path)
    indexer = create_indexer(detector, phase_list, prepare_reflectors(phase_list))

    assert indexer.camElev == pytest.approx(CAM_ELEV, abs=1e-6), (
        "the detector's camera elevation did not reach PyEBSDIndex; every Hough "
        f"orientation would be rigidly wrong by {CAM_ELEV} deg"
    )
    assert indexer.sampleTilt == pytest.approx(SAMPLE_TILT, abs=1e-6)


def test_zero_camera_tilt_is_not_silently_substituted(al_cif_path):
    """A genuine 0 deg elevation must survive too -- the guard above must not be
    satisfied by a default that happens to match."""
    pytest.importorskip("pyebsdindex")
    from kikuchipy.detectors import EBSDDetector

    from ebsd_utils import create_indexer, prepare_reflectors

    detector = EBSDDetector(
        shape=(128, 156),
        pc=(0.5026, 0.3273, 0.8458),
        sample_tilt=SAMPLE_TILT,
        tilt=0.0,
        convention="bruker",
    )
    phase_list = _al_phase_list(al_cif_path)
    indexer = create_indexer(detector, phase_list, prepare_reflectors(phase_list))

    assert indexer.camElev == pytest.approx(0.0, abs=1e-6)


@pytest.mark.integration
def test_hough_orientations_move_by_exactly_the_camera_tilt(h5oina_path, al_cif_path):
    """End-to-end on real Oxford patterns: dropping the elevation rotates every
    Hough orientation by exactly that angle.

    This is the regression the harness bug produced. It needs no GPU and no
    master-pattern library -- only the h5oina that ships with the repo.
    """
    pytest.importorskip("pyebsdindex")
    from orix.quaternion import Orientation, Rotation
    from orix.quaternion.symmetry import Oh

    from indexing_controller import (
        IndexingConfig,
        IndexingMethod,
        hough_index_patterns,
    )
    from safe_loader import load_ebsd_safe

    signal = load_ebsd_safe(str(h5oina_path), verbose=False)
    detector = signal.detector
    assert detector.tilt > 0.5, (
        "this file is supposed to carry a non-zero 'Detector Orientation Euler'; "
        f"got tilt={detector.tilt}"
    )

    # A 12x12 window inside the single Al grain at rows 15..35, cols 0..20
    # (Aztec's own solution spreads 0.12 deg across it).
    n_rows, n_cols = signal.data.shape[:2]
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    mask[18:30, 2:14] = True

    phase_list = _al_phase_list(al_cif_path)

    def _index(tilt_deg):
        det = detector.deepcopy()
        det.tilt = float(tilt_deg)
        result = hough_index_patterns(
            signal, phase_list, det,
            IndexingConfig(method=IndexingMethod.HOUGH), mask.copy(),
        )
        return np.asarray(result.xmap.rotations.data).reshape(-1, 4)

    q_with = _index(detector.tilt)
    q_without = _index(0.0)

    # Symmetry on BOTH sides -- orix `angle_with` on Orientation does that;
    # a one-sided reduction reports nonsense for cubic pairs.
    angles = np.asarray(np.rad2deg(
        Orientation(Rotation(q_with), Oh).angle_with(
            Orientation(Rotation(q_without), Oh))
    )).ravel()

    assert np.median(angles) == pytest.approx(detector.tilt, abs=0.05), (
        "dropping the camera elevation should rotate the Hough answer by exactly "
        f"{detector.tilt:.4f} deg; measured {np.median(angles):.4f} deg"
    )
    # Rigid, not noisy: that is what makes it invisible in quality metrics.
    iqr = np.percentile(angles, 75) - np.percentile(angles, 25)
    assert iqr < 0.5, f"expected a rigid offset, got IQR {iqr:.3f} deg"
