"""Two repairs to the manual per-grain phase assignment (2026-09-07).

The user clicked a pixel, saw the Compare-phases panel score Al7FeCu2 at
R = 0.393 against R = 0.216 for the stored Al, pressed "Assign Al7FeCu2 to this
grain", and got:

    Could not assign phase: hough failed on every pixel

Two things were wrong with that, and they are independent.

1. The button threw away an orientation it had already computed and shown, and
   asked Hough for the same thing. When Hough cannot run, the assignment the
   evidence supports is refused for a reason unrelated to the evidence.

2. The failure was reported as memory. It was not: the driver returns
   `OUT_OF_HOST_MEMORY` when it cannot create the OpenCL context, and in that
   session it hit `Al.cif` -- m-3m, 50 families, a library this code labels
   "negligible" -- while 6.7 GiB was free. The user read the message, checked
   their memory, and asked what on earth was going on. Fair.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.spherical_gpu.pipeline.phase_reassignment import (  # noqa: E402
    rigid_grain_quats,
)

IDENT = np.array([1.0, 0.0, 0.0, 0.0])
# 90 deg about z, and 90 deg about x -- distinct, easy to reason about.
QZ = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
QX = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])


def _angle_deg(a, b):
    """Rotation angle between two unit quaternions, ignoring the sign ambiguity.

    Tolerances below are in DEGREES and no tighter than 1e-4: arccos is badly
    conditioned near 1, so two bit-identical quaternions still come back as
    ~2e-6 deg apart through this formula. Asserting 1e-9 would be testing the
    arithmetic of the assertion, not the code.
    """
    d = abs(float(np.dot(np.asarray(a).ravel(), np.asarray(b).ravel())))
    return float(np.degrees(2.0 * np.arccos(min(1.0, d))))


class TestRigidGrainQuats:
    def test_the_clicked_pixel_lands_exactly_on_the_seed(self):
        """The pixel the user judged must get the orientation they judged.

        Anything else means the R they read on screen was not the R of what
        was written.
        """
        stored = np.stack([QZ, QX, IDENT])
        out = rigid_grain_quats(stored, q_click=QZ, q_seed=QX)
        # Compared as |q1.q2| = 1 rather than as an angle: arccos is so badly
        # conditioned at 1 that two IDENTICAL quaternions read as 1.6e-4 deg
        # apart, which would make the tolerance a statement about arccos.
        assert abs(float(np.dot(out[0], QX))) == pytest.approx(1.0, abs=1e-12)

    def test_the_grain_keeps_its_internal_misorientation(self):
        """A constant orientation would erase the deformation the sample has.

        The relative rotation between any two pixels must survive the change of
        phase label -- that is a rotation in SAMPLE space, and the sample did
        not move.
        """
        rng = np.random.default_rng(0)
        stored = rng.normal(size=(6, 4))
        stored /= np.linalg.norm(stored, axis=1, keepdims=True)
        out = rigid_grain_quats(stored, q_click=stored[0], q_seed=QX)
        for i in range(1, len(stored)):
            before = _angle_deg(stored[0], stored[i])
            after = _angle_deg(out[0], out[i])
            assert after == pytest.approx(before, abs=1e-6), (
                f"pixel {i}: misorientation changed {before:.4f} -> {after:.4f}"
            )

    def test_a_seed_equal_to_the_click_changes_nothing(self):
        stored = np.stack([QZ, QX])
        out = rigid_grain_quats(stored, q_click=QZ, q_seed=QZ)
        assert np.allclose(np.abs(np.sum(out * stored, axis=1)), 1.0, atol=1e-9)

    def test_the_output_is_normalised(self):
        stored = np.stack([QZ * 3.0, QX * 0.1])
        out = rigid_grain_quats(stored, q_click=QZ, q_seed=QX)
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-9)

    def test_a_non_finite_or_zero_anchor_is_refused(self):
        """A NaN click orientation would poison the whole grain silently."""
        stored = np.stack([QZ])
        with pytest.raises(ValueError):
            rigid_grain_quats(stored, q_click=np.full(4, np.nan), q_seed=QX)
        with pytest.raises(ValueError):
            rigid_grain_quats(stored, q_click=QZ, q_seed=np.zeros(4))


class TestHoughFailureMessage:
    def _describe(self, monkeypatch, need, have):
        import backend.api.routes.indexing as ix
        monkeypatch.setattr(ix, "_hough_memory_evidence", lambda _p: (need, have))
        return ix._describe_hough_failure(
            RuntimeError("Context failed: OUT_OF_HOST_MEMORY"), "Al.cif")

    def test_it_does_not_blame_memory_when_the_numbers_refuse_to(self, monkeypatch):
        """The exact reported case: a negligible library, plenty free."""
        msg = self._describe(monkeypatch, need=1 * 2**20, have=7 * 2**30)
        assert "NOT a memory shortage" in msg
        assert "OpenCL" in msg
        # It must not send the user off to close programs for nothing.
        assert "close other programs" not in msg

    def test_it_prints_both_numbers_so_the_claim_is_checkable(self, monkeypatch):
        msg = self._describe(monkeypatch, need=1 * 2**20, have=7 * 2**30)
        assert "1 MiB" in msg and "7.00 GiB" in msg

    def test_a_small_library_is_not_printed_as_0_00_GiB(self, monkeypatch):
        """The one number that has to be believable."""
        msg = self._describe(monkeypatch, need=3 * 2**20, have=7 * 2**30)
        assert "0.00 GiB" not in msg

    def test_it_DOES_blame_memory_when_the_request_really_is_huge(self, monkeypatch):
        """The original diagnosis must survive: a 44 GiB P1 library on a 7 GiB
        machine is a shortage, and the advice to free memory is right there."""
        msg = self._describe(monkeypatch, need=44 * 2**30, have=7 * 2**30)
        assert "Not enough memory" in msg
        assert "44.00 GiB" in msg and "7.00 GiB" in msg
        assert "NOT a memory shortage" not in msg

    def test_unmeasurable_falls_back_to_the_cautious_wording(self, monkeypatch):
        """Never assert "this is not a shortage" without the evidence for it."""
        msg = self._describe(monkeypatch, need=None, have=None)
        assert "NOT a memory shortage" not in msg

    def test_a_failure_that_is_not_about_memory_is_passed_through(self, monkeypatch):
        import backend.api.routes.indexing as ix
        msg = ix._describe_hough_failure(ValueError("bad reflectors"), "Al.cif")
        assert "bad reflectors" in msg and "OpenCL" not in msg


class TestAssignEndpointFallback:
    """The route level: Hough fails, the assignment still happens, and the
    response says which orientation source it used."""

    def _setup(self, monkeypatch, hough_ok):
        import tests.test_phase_reassignment as H
        result = H._fake_two_phase_result()
        ix = H._patch_endpoint_seams(monkeypatch, result)
        if not hough_ok:
            # Exactly the observed failure: nothing comes back for any pixel.
            monkeypatch.setattr(
                ix, "_hough_quats_for_phase_batch",
                lambda pats, cif, det, **_kw: np.full(
                    (np.asarray(pats).shape[0], 4), np.nan))
        return ix, result, H

    def test_hough_still_wins_when_it_works(self, monkeypatch):
        """The seed must not quietly displace the stronger source."""
        ix, result, H = self._setup(monkeypatch, hough_ok=True)
        out = H._run(ix.assign_phase_to_grain(ix.AssignPhaseRequest(
            row=0, col=0, target_phase_id=1, seed_quat=[0.0, 1.0, 0.0, 0.0])))
        assert out["orientation_source"] == "hough"

    def test_the_seed_rescues_the_assignment_when_hough_produces_nothing(
            self, monkeypatch):
        """The reported bug, at the route."""
        ix, result, H = self._setup(monkeypatch, hough_ok=False)
        out = H._run(ix.assign_phase_to_grain(ix.AssignPhaseRequest(
            row=0, col=0, target_phase_id=1, seed_quat=[0.0, 1.0, 0.0, 0.0])))
        assert out["orientation_source"] == "seed"
        assert out["n_pixels_changed"] > 0
        assert out["undo_available"] is True

    def test_without_a_seed_it_still_refuses(self, monkeypatch):
        """No silent invention: given no orientation, it must not make one up."""
        from fastapi import HTTPException
        ix, result, H = self._setup(monkeypatch, hough_ok=False)
        with pytest.raises(HTTPException) as e:
            H._run(ix.assign_phase_to_grain(ix.AssignPhaseRequest(
                row=0, col=0, target_phase_id=1)))
        assert e.value.status_code == 409
