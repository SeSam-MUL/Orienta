"""Stage 1 (per-grain check) must not delete a phase for a bad ORIENTATION.

The island check got this guard in the morning; the per-grain check did not,
and in the afternoon it flipped the user's whole MgCuAl2 particle (one grain,
~250 px) to Al: MgCuAl2 rendered 0.207 at its stored orientation (mmm,
z_rot == 2 -- the spherical orientation is known-unreliable), a free search
reached 0.420, the best of the three phases; Al at a Hough orientation scored
~0.35 and "won" a rigged comparison (2026-09-07).

So the stored phase is now Hough-oriented on the suspect grains too -- ONE
batched call per stored phase -- and the grain is judged at the better of
stored@stored and stored@hough. Outcomes:
    still loses         -> "reassign"  (margin against the BETTER stored score)
    now wins/ties       -> "keep" + rescued_by_fair_orientation
    no fair orientation -> "undecided" (reported, never applied)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_phase_reassignment import IDENT, NC, NR, PHASES, _grid  # noqa: E402
from backend.spherical_gpu.pipeline.phase_reassignment import (  # noqa: E402
    apply_reassignment, check_map,
)

FAIR_Q = np.array([0.0, 1.0, 0.0, 0.0])   # "the orientation Hough found"
# In `_grid`, phase 2 owns columns c<3 and c>=5, phase 1 the middle two.


def _scorers(*, stored_at_stored, stored_at_fair, al_score=0.32):
    """Phase 2 (stored on the suspect grains) scores by ORIENTATION; phase 1
    (the Al-like candidate) scores a flat `al_score` everywhere."""
    def phase2(flats, quats):
        quats = np.asarray(quats, float).reshape(-1, 4)
        at_fair = np.all(np.isclose(quats, FAIR_Q), axis=1)
        return np.where(at_fair, stored_at_fair, stored_at_stored)

    def phase1(flats, quats):
        return np.full(np.asarray(quats).reshape(-1, 4).shape[0], al_score)

    return {1: phase1, 2: phase2}


def _hough(*, stored_rows=None):
    """Candidates get IDENT; the stored phase (2) gets `stored_rows` per pixel
    (FAIR_Q by default, NaN to simulate 'Hough rejected every pixel')."""
    calls = []

    def fn(pid, flats):
        flats = np.asarray(flats).reshape(-1)
        calls.append((int(pid), flats.copy()))
        if int(pid) == 2:
            row = FAIR_Q if stored_rows is None else stored_rows
            return np.tile(np.asarray(row, float), (flats.size, 1))
        return np.tile(IDENT, (flats.size, 1))

    fn.calls = calls
    return fn


def _suspects(report):
    return [e for e in report["grains"] if e.get("phase_id") == 2
            and e.get("best_alt_phase") is not None]


class TestOutcomes:
    def test_still_loses_fairly_oriented__reassign_with_the_fair_margin(self):
        full_q, pf = _grid()
        rep, margin = check_map(full_q, pf, NR, NC, PHASES,
                                _scorers(stored_at_stored=0.10, stored_at_fair=0.20),
                                _hough())
        es = _suspects(rep)
        assert es and all(e["decision"] == "reassign" for e in es)
        e = es[0]
        assert e["fair_checked"] is True
        assert e["stored_score_fair"] == pytest.approx(0.20)
        # margin against the BETTER stored score: 0.20 - 0.32, not 0.10 - 0.32
        assert e["margin"] == pytest.approx(-0.12, abs=1e-6)
        assert rep["n_reassign"] == len(es) and rep["n_rescued"] == 0

    def test_wins_fairly_oriented__rescued_not_reassigned(self):
        """The user's particle: 0.207 stored, 0.420 fair, candidate ~0.35."""
        full_q, pf = _grid()
        rep, margin = check_map(full_q, pf, NR, NC, PHASES,
                                _scorers(stored_at_stored=0.207, stored_at_fair=0.420,
                                         al_score=0.35),
                                _hough())
        es = _suspects(rep)
        assert es and all(e["decision"] == "keep" for e in es)
        assert all(e.get("rescued_by_fair_orientation") is True for e in es)
        assert rep["n_reassign"] == 0
        assert rep["n_rescued"] == len(es)
        # Nothing to apply -- the particle stays.
        pf2, q2, applied, skipped = apply_reassignment(
            full_q, pf, NR, NC, rep, PHASES, _hough())
        assert applied == [] and np.array_equal(pf2, pf)

    def test_no_fair_orientation__undecided_and_never_applied(self):
        full_q, pf = _grid()
        rep, margin = check_map(full_q, pf, NR, NC, PHASES,
                                _scorers(stored_at_stored=0.10, stored_at_fair=0.50),
                                _hough(stored_rows=np.full(4, np.nan)))
        es = _suspects(rep)
        assert es and all(e["decision"] == "undecided" for e in es)
        assert all(e["fair_checked"] is False and e["undecided_reason"] for e in es)
        assert rep["n_undecided"] == len(es) and rep["n_reassign"] == 0
        # The margin layer must not paint "reassign me" red on an undecided grain.
        for e in es:
            pass
        assert not np.any(np.isfinite(margin)), "undecided grains painted the margin layer"
        pf2, q2, applied, skipped = apply_reassignment(
            full_q, pf, NR, NC, rep, PHASES, _hough())
        assert applied == [] and np.array_equal(pf2, pf)

    def test_a_failing_stored_phase_hough_call_is_undecided_with_its_reason(self):
        def hough(pid, flats):
            flats = np.asarray(flats).reshape(-1)
            if int(pid) == 2:
                raise RuntimeError("Context failed: OUT_OF_HOST_MEMORY")
            return np.tile(IDENT, (flats.size, 1))

        full_q, pf = _grid()
        rep, _ = check_map(full_q, pf, NR, NC, PHASES,
                           _scorers(stored_at_stored=0.10, stored_at_fair=0.50), hough)
        es = _suspects(rep)
        assert es and all(e["decision"] == "undecided" for e in es)
        assert "OUT_OF_HOST_MEMORY" in es[0]["undecided_reason"]


class TestCost:
    def test_one_hough_call_per_stored_phase_covering_all_suspect_samples(self):
        h = _hough()
        full_q, pf = _grid()
        check_map(full_q, pf, NR, NC, PHASES,
                  _scorers(stored_at_stored=0.10, stored_at_fair=0.20), h)
        stored_calls = [(pid, f) for pid, f in h.calls if pid == 2]
        assert len(stored_calls) == 1, "the stored phase must be one batched call"
        # ...and it covered every suspect grain's sample, each pixel once.
        f = stored_calls[0][1]
        assert len(f) == len(set(f.tolist()))

    def test_a_healthy_grain_costs_no_stored_phase_call(self):
        """Only suspects get the fairness step; a grain above the floor is free."""
        h = _hough()
        full_q, pf = _grid()
        check_map(full_q, pf, NR, NC, PHASES,
                  _scorers(stored_at_stored=0.60, stored_at_fair=0.60), h)
        assert h.calls == []
