"""The grain decides the candidate; the PIXEL decides whether it is written.

A reassign on the deformed 7050 map moved 531 px in one press (2026-09-07).
Measured per pixel at the orientation actually written: the winner beat the
stored phase on 76 % of the Al->Al7FeCu2 pixels -- the 301 that passed the
floor+margin had band contrast 110 (particle), the 176 that failed had 83
(matrix) -- and on only 16 % of the 44 Al->MgCuAl2 pixels, whose written
orientation rendered at 0.09, worse than the phase it replaced. A 16-sample
grain median cannot separate those; per-pixel evidence can. These pin that
gate for the grain stage and the island stage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_phase_reassignment import IDENT, NC, NR, PHASES, _grid  # noqa: E402
from backend.spherical_gpu.pipeline.phase_reassignment import (  # noqa: E402
    SCORE_FLOOR, apply_reassignment, check_map,
)


def _report_for_first_phase2_grain(hough):
    """A check whose phase-2 grains (c<3 / c>=5) lose to phase 1 everywhere."""
    def alpha(flats, quats):
        return np.full(np.asarray(flats).reshape(-1).size, 0.10)

    def al(flats, quats):
        return np.full(np.asarray(flats).reshape(-1).size, 0.45)

    full_q, pf = _grid()
    rep, _ = check_map(full_q, pf, NR, NC, PHASES, {1: al, 2: alpha}, hough)
    assert rep["n_reassign"] >= 1
    return full_q, pf, rep


def _hough_ident(pid, flats):
    return np.tile(IDENT, (np.asarray(flats).reshape(-1).size, 1))


class TestGrainStageGate:
    def test_only_pixels_with_their_own_evidence_are_written(self):
        full_q, pf, rep = _report_for_first_phase2_grain(_hough_ident)
        # At APPLY time the winner (phase 1) renders well only in even rows.
        def al_apply(flats, quats):
            f = np.asarray(flats).reshape(-1)
            return np.where((f // NC) % 2 == 0, 0.45, 0.10)

        def alpha_apply(flats, quats):
            return np.full(np.asarray(flats).reshape(-1).size, 0.10)

        pf2, q2, applied, skipped = apply_reassignment(
            full_q, pf, NR, NC, rep, PHASES, _hough_ident,
            score_fns={1: al_apply, 2: alpha_apply})
        changed = np.flatnonzero(pf2 != pf)
        assert changed.size > 0
        assert np.all((changed // NC) % 2 == 0), "an odd-row pixel was written without evidence"
        # Odd rows of the same grains stayed phase 2.
        for a in applied:
            assert a["n_px_refused"] > 0 and a["n_px_applied"] > 0

    def test_a_winner_below_the_floor_is_refused_even_if_it_beats_the_stored_phase(self):
        """'Less bad' is not evidence: 0.20 vs 0.05 beats the margin but not the floor."""
        full_q, pf, rep = _report_for_first_phase2_grain(_hough_ident)
        fns = {1: lambda f, q: np.full(np.asarray(f).reshape(-1).size, 0.20),
               2: lambda f, q: np.full(np.asarray(f).reshape(-1).size, 0.05)}
        assert 0.20 < SCORE_FLOOR
        pf2, q2, applied, skipped = apply_reassignment(
            full_q, pf, NR, NC, rep, PHASES, _hough_ident, score_fns=fns)
        assert applied == []
        assert np.array_equal(pf2, pf)
        assert all("enough evidence" in s_["skip_reason"] for s_ in skipped)

    def test_pixels_without_their_own_hough_orientation_are_not_filled_from_a_neighbour(self):
        full_q, pf, rep = _report_for_first_phase2_grain(_hough_ident)
        # Hough fails on every pixel of column 0.
        def hough(pid, flats):
            f = np.asarray(flats).reshape(-1)
            out = np.tile(IDENT, (f.size, 1)).astype(float)
            out[f % NC == 0] = np.nan
            return out

        fns = {1: lambda f, q: np.full(np.asarray(f).reshape(-1).size, 0.45),
               2: lambda f, q: np.full(np.asarray(f).reshape(-1).size, 0.10)}
        pf2, q2, applied, skipped = apply_reassignment(
            full_q, pf, NR, NC, rep, PHASES, hough, score_fns=fns)
        changed = np.flatnonzero(pf2 != pf)
        assert changed.size > 0
        assert not np.any(changed % NC == 0), "a pixel with no Hough orientation was written"
        assert all(a["n_hough_filled"] == 0 for a in applied)
        assert any(a["n_px_no_hough"] > 0 for a in applied)

    def test_without_score_fns_the_old_whole_grain_behaviour_remains(self):
        full_q, pf, rep = _report_for_first_phase2_grain(_hough_ident)
        pf2, q2, applied, skipped = apply_reassignment(
            full_q, pf, NR, NC, rep, PHASES, _hough_ident)
        assert applied and all(a["n_px_refused"] == 0 for a in applied)
        for a in applied:
            assert a["n_px_applied"] == a["pixels"]


class TestIslandStageGate:
    """The island stage carries the same two rules: the enclosing phase must
    itself reach the floor, and only pixels that pass on their own are
    written (`apply_pixels`)."""

    NRI = NCI = 9
    FAIR_Q = np.array([0.0, 1.0, 0.0, 0.0])

    def _island(self, pixels):
        pf = np.full(self.NRI * self.NCI, 1, dtype=np.int64)
        pf[list(pixels)] = 2
        return np.tile(IDENT, (self.NRI * self.NCI, 1)).astype(float), pf

    def _fair(self, pid, flats):
        return np.tile(self.FAIR_Q, (np.asarray(flats).size, 1))

    def test_an_enclosing_phase_below_the_floor_does_not_win(self):
        from backend.spherical_gpu.pipeline.island_check import check_islands
        q, pf = self._island([4 * self.NCI + 4])
        fns = {2: lambda f, qq: np.full(np.asarray(f).size, 0.05),      # stored: terrible
               1: lambda f, qq: np.full(np.asarray(f).size, 0.20)}      # enclosing: bad, but "better"
        rep = check_islands(q, pf, self.NRI, self.NCI, {1: "m-3m", 2: "mmm"}, fns,
                            fair_orientation_fn=self._fair)
        assert rep["islands"][0]["decision"] != "phase"
        assert rep["n_phase_swap"] == 0

    def test_only_passing_pixels_of_an_island_are_applied(self):
        from backend.spherical_gpu.pipeline.island_check import (
            apply_island_findings, check_islands,
        )
        a, b = 4 * self.NCI + 4, 4 * self.NCI + 5          # a 2-px island
        q, pf = self._island([a, b])

        def enclosing(flats, qq):
            f = np.asarray(flats).reshape(-1)
            return np.where(f == a, 0.50, 0.10)             # only pixel a is convincing

        fns = {2: lambda f, qq: np.full(np.asarray(f).size, 0.05), 1: enclosing}
        rep = check_islands(q, pf, self.NRI, self.NCI, {1: "m-3m", 2: "mmm"}, fns,
                            fair_orientation_fn=self._fair)
        e = rep["islands"][0]
        assert e["decision"] == "phase"
        assert e["apply_pixels"] == [a] and e["n_px_refused"] == 1
        pf2, q2, applied = apply_island_findings(q, pf, self.NRI, self.NCI, rep)
        assert pf2[a] == 1 and pf2[b] == 2
        assert applied[0]["pixels"] == [a]
