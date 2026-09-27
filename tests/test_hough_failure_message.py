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

That real CIF has since been repaired (P4/mnc, 2026-09-13) and the P 1 phase
these tests need is synthesised — see ``tests/synthetic_cif`` for why reading it
out of the crystal library was wrong in the first place.
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
from tests.synthetic_cif import (  # noqa: E402
    forget_reflector_limit_for,
    write_synthetic_p1_cif,
)

LIB = Path(__file__).resolve().parents[1] / "Database" / "CIF_Library"


class _CLError(Exception):
    """Stands in for pyopencl's LogicError, which we do not need to import."""


def test_out_of_memory_names_the_phase_and_the_cause():
    msg = _describe_hough_failure(_CLError("Context failed: OUT_OF_HOST_MEMORY"),
                                  LIB / "SomePhase.cif")
    assert "SomePhase" in msg
    assert "memory" in msg.lower()
    # It must not read as a verdict on the patterns.
    assert "every pixel" not in msg.lower()


def _pin_the_machine(monkeypatch, cif_path, free_bytes):
    """Fix BOTH machine-state inputs to the wording, and nothing else.

    (1) Free memory. The message branches on "does the request reach half of
    what is free", so on a big enough machine even a 44 GiB ask is not plausible
    and on a small one everything is. Without the pin this file passes on the
    dev box by 44.73 against 19.0 GiB — a margin under 2x that a busy machine
    erases — and fails outright on anything with ~90 GiB spare.

    (2) The per-phase reflector limit. ``_hough_memory_evidence`` takes the first
    ladder step at or BELOW the phase's registered limit, so a remembered 32
    reports 0.21 GiB instead of 44.73 and the symmetry sentence disappears. That
    registry is persisted and keyed on the file stem, and
    ``test_phase_reflector_registry.py`` registers 32 for this very stem — so it
    really can decide this file's answer. Reproduced: with the store enabled, a
    leftover ``syntheticp1: 32`` fails
    ``test_a_symmetry_less_cif_is_called_out_by_name``.

    What is NOT pinned is everything the tests are actually about: the symmetry
    comes from the real CIF through the real reader, and the size from the real
    predictor.
    """
    import ebsd_utils
    monkeypatch.setattr(ebsd_utils, "_available_memory_bytes",
                        lambda: int(free_bytes), raising=True)
    forget_reflector_limit_for(cif_path)


def test_a_symmetry_less_cif_is_called_out_by_name(tmp_path, monkeypatch):
    """The one fact that turns this from bad luck into a fixable problem.

    The P 1 phase is synthesised (``tests/synthetic_cif``) rather than taken
    from the crystal library: this test used to read
    ``Database/CIF_Library/Al7FeCu2.cif`` because that file was stored without
    symmetry, and it went red the day the file was repaired to P4/mnc.
    """
    p1_cif = write_synthetic_p1_cif(tmp_path)
    _pin_the_machine(monkeypatch, p1_cif, 8 << 30)   # 8 GiB free, ask is 44.73
    msg = _describe_hough_failure(_CLError("Context failed: OUT_OF_HOST_MEMORY"), p1_cif)
    assert "point group 1" in msg
    assert "P 1" in msg
    assert "space group" in msg.lower()
    assert "SyntheticP1" in msg                 # and which phase it was


def test_the_same_phase_is_not_blamed_on_memory_when_there_is_plenty(tmp_path,
                                                                    monkeypatch):
    """Symmetry alone must not buy the memory story.

    This is the other half of the 2026-09-07 fix, and the pairing is the point:
    one phase, one error, two messages, decided by the numbers and nothing else.
    """
    p1_cif = write_synthetic_p1_cif(tmp_path)
    _pin_the_machine(monkeypatch, p1_cif, 512 << 30)  # no shortage anywhere
    msg = _describe_hough_failure(_CLError("Context failed: OUT_OF_HOST_MEMORY"), p1_cif)
    assert "NOT a memory shortage" in msg
    assert "Nothing is wrong with this phase" in msg


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
