"""A Hough batch that never ran must not be reported as rejected data.

`_hough_quats_for_phase_batch` is fail-soft: on any failure it returns all-NaN
so the caller keeps the stored phase. Downstream, `apply_reassignments` reads
"every row is NaN" as **"hough failed on every pixel"** — a statement about the
patterns. On 2026-09-03 a user pressed 'Assign "Al7FeCu2" to this grain' and got
exactly that sentence, while the real cause was in the log and nowhere near the
data:

    batch hough for phase failed (...Al7FeCu2.cif): Context failed: OUT_OF_HOST_MEMORY

The machine was at 85.3 GiB of an 87.5 GiB Windows commit limit, and building
the band-triplet library for that CIF — written as `P 1`, no symmetry — needs
48.6 GiB on its own (measured; an `mmm` phase on the same detector needs 5.6).

These tests are deliberately allocation-free: they never build an indexer, so
they can run on a machine that has no room to build one.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.routes.indexing import (  # noqa: E402
    _describe_hough_failure,
    _hough_quats_for_phase_batch,
)

LIB = Path(__file__).resolve().parents[1] / "Database" / "CIF_Library"
P1_CIF = LIB / "Al7FeCu2.cif"


class _CLError(Exception):
    """Stands in for pyopencl's LogicError, which we do not need to import."""


def test_out_of_memory_names_the_phase_and_the_cause():
    msg = _describe_hough_failure(_CLError("Context failed: OUT_OF_HOST_MEMORY"),
                                  LIB / "SomePhase.cif")
    assert "SomePhase" in msg
    assert "memory" in msg.lower()
    # It must not read as a verdict on the patterns.
    assert "every pixel" not in msg.lower()


@pytest.mark.skipif(not P1_CIF.exists(), reason="phase library CIF not present")
def test_a_symmetry_less_cif_is_called_out_by_name():
    """The one fact that turns this from bad luck into a fixable problem."""
    msg = _describe_hough_failure(_CLError("Context failed: OUT_OF_HOST_MEMORY"), P1_CIF)
    assert "point group 1" in msg
    assert "P 1" in msg
    assert "space group" in msg.lower()


def test_other_failures_keep_their_own_words():
    msg = _describe_hough_failure(ValueError("detector shape mismatch"),
                                  LIB / "SomePhase.cif")
    assert "detector shape mismatch" in msg
    assert "memory" not in msg.lower()


def test_batch_reports_the_reason_and_still_fails_soft():
    """All-NaN as before — plus the reason, so the caller can tell them apart."""
    det = {"pat_height": 8, "pat_width": 8, "sample_tilt": 70.0, "tilt": 0.0,
           "pc_x": 0.5, "pc_y": 0.5, "pc_z": 0.5, "binning": 1}
    pats = np.zeros((2, 8, 8), dtype=np.float32)
    errs = []
    out = _hough_quats_for_phase_batch(pats, str(LIB / "does-not-exist.cif"), det,
                                       err_out=errs)
    assert out.shape == (2, 4)
    assert np.isnan(out).all()          # fail-soft contract unchanged
    assert errs and errs[0]             # ...but no longer silent

    # And the old call shape still works for callers that do not want it.
    out2 = _hough_quats_for_phase_batch(pats, str(LIB / "does-not-exist.cif"), det)
    assert np.isnan(out2).all()
