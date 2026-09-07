"""The island check must not mistake a bad ORIENTATION for a bad PHASE.

Stage 1 judges the stored phase at its stored orientation. On a small island
that orientation is unreliable almost by definition -- these are the pixels
where indexing already went wrong -- so a loss there says nothing about the
phase. Measured on the real 7050 map (2026-09-07): 8 of 24 stage-1 findings
were artefacts; MgCuAl2 rendered 0.207 at its stored orientation and 0.420 at a
fair one, the best of the three phases. A random orientation search cannot
repair this (256 samples still miss 5 of 8), so the stored phase is re-oriented
by a directed method -- `fair_orientation_fn`, Hough in production -- before a
loss is believed.

These pin the four outcomes and the batching, with renderers whose answer is
baked in so the DECISION is what is under test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.spherical_gpu.pipeline.island_check import (  # noqa: E402
    apply_island_findings, check_islands,
)
from backend.spherical_gpu.pipeline.phase_reassignment import MIN_GRAIN_PX  # noqa: E402

NR = NC = 9
PHASES = {1: "m-3m", 2: "mmm"}
IDENT = np.array([1.0, 0.0, 0.0, 0.0])
FAIR_Q = np.array([0.0, 1.0, 0.0, 0.0])      # "the orientation Hough found"


def _q_all(q=IDENT):
    return np.tile(q, (NR * NC, 1)).astype(float)


def _map_with_island(flats):
    pf = np.full(NR * NC, 1, dtype=np.int64)     # everything phase 1 ...
    pf[list(flats)] = 2                          # ... except the island, phase 2
    return pf


def _scorers(*, stored_at_stored, stored_at_fair, enclosing):
    """Phase 2 (the island) scores `stored_at_stored` at IDENT and
    `stored_at_fair` at FAIR_Q; phase 1 (enclosing) scores `enclosing`."""
    def island_phase(flats, quats):
        quats = np.asarray(quats, float).reshape(-1, 4)
        at_fair = np.all(np.isclose(quats, FAIR_Q), axis=1)
        return np.where(at_fair, stored_at_fair, stored_at_stored)

    def enclosing_phase(flats, quats):
        return np.full(np.asarray(quats).reshape(-1, 4).shape[0], enclosing)

    return {2: island_phase, 1: enclosing_phase}


def _fair_returning(q):
    def fn(pid, flats):
        return np.tile(q, (np.asarray(flats).size, 1))
    return fn


CENTRE = 4 * NC + 4


class TestTheFourOutcomes:
    def test_still_loses_after_a_fair_orientation__phase(self):
        """The real finding: even fairly oriented, the stored phase is worse."""
        rep = check_islands(
            _q_all(), _map_with_island([CENTRE]), NR, NC, PHASES,
            _scorers(stored_at_stored=0.10, stored_at_fair=0.20, enclosing=0.45),
            fair_orientation_fn=_fair_returning(FAIR_Q))
        e = rep["islands"][0]
        assert e["decision"] == "phase"
        assert e["fair_checked"] is True
        assert e["stored_score_fair"] == pytest.approx(0.20)
        # The margin is against the BETTER of the two stored scores.
        assert e["margin"] == pytest.approx(0.25, abs=1e-9)
        assert rep["n_phase_swap"] == 1 and rep["n_rescued"] == 0

    def test_wins_once_fairly_oriented__rescued(self):
        """The artefact: a bad orientation, not a bad phase. Must NOT be offered."""
        rep = check_islands(
            _q_all(), _map_with_island([CENTRE]), NR, NC, PHASES,
            _scorers(stored_at_stored=0.10, stored_at_fair=0.50, enclosing=0.45),
            fair_orientation_fn=_fair_returning(FAIR_Q))
        e = rep["islands"][0]
        assert e["decision"] == "keep"
        assert e.get("rescued_by_fair_orientation") is True
        assert "winner_phase" not in e and "new_quats" not in e
        assert rep["n_phase_swap"] == 0 and rep["n_rescued"] == 1

    def test_no_fair_orientation_available__undecided(self):
        """Hough rejected every pixel: wrong phase and wrong orientation are
        indistinguishable, and the check must say so instead of guessing."""
        rep = check_islands(
            _q_all(), _map_with_island([CENTRE]), NR, NC, PHASES,
            _scorers(stored_at_stored=0.10, stored_at_fair=0.50, enclosing=0.45),
            fair_orientation_fn=_fair_returning(np.full(4, np.nan)))
        e = rep["islands"][0]
        assert e["decision"] == "undecided"
        assert e["undecided_reason"]
        assert "new_quats" not in e
        assert rep["n_undecided"] == 1 and rep["n_phase_swap"] == 0

    def test_no_fairness_step_at_all__suspect_and_never_applied(self):
        """A caller that cannot re-orient gets a report-only verdict."""
        rep = check_islands(
            _q_all(), _map_with_island([CENTRE]), NR, NC, PHASES,
            _scorers(stored_at_stored=0.10, stored_at_fair=0.50, enclosing=0.45))
        e = rep["islands"][0]
        assert e["decision"] == "suspect"
        assert e["fair_checked"] is False
        assert rep["n_suspect"] == 1 and rep["n_phase_swap"] == 0
        # ... and the apply step will not touch it.
        pf, q, applied = apply_island_findings(_q_all(), _map_with_island([CENTRE]),
                                               NR, NC, rep)
        assert applied == []
        assert pf[CENTRE] == 2


class TestBatchingAndRobustness:
    def test_one_fair_call_per_stored_phase_not_per_island(self):
        """Hough leaks ~0.24 GiB per call; the cost must be per phase."""
        islands = [1 * NC + 1, 4 * NC + 4, 7 * NC + 7]      # three separate islands
        calls = []

        def fair(pid, flats):
            calls.append((pid, np.asarray(flats).size))
            return np.tile(FAIR_Q, (np.asarray(flats).size, 1))

        rep = check_islands(
            _q_all(), _map_with_island(islands), NR, NC, PHASES,
            _scorers(stored_at_stored=0.10, stored_at_fair=0.20, enclosing=0.45),
            fair_orientation_fn=fair)
        assert rep["n_islands"] == 3 and rep["n_phase_swap"] == 3
        assert calls == [(2, 3)], calls      # ONE call, all three pixels

    def test_a_failing_fair_call_marks_its_islands_undecided_not_phase(self):
        def fair(pid, flats):
            raise RuntimeError("Context failed: OUT_OF_HOST_MEMORY")

        rep = check_islands(
            _q_all(), _map_with_island([CENTRE]), NR, NC, PHASES,
            _scorers(stored_at_stored=0.10, stored_at_fair=0.50, enclosing=0.45),
            fair_orientation_fn=fair)
        e = rep["islands"][0]
        assert e["decision"] == "undecided"
        assert "OUT_OF_HOST_MEMORY" in e["undecided_reason"]

    def test_a_wrong_sized_fair_answer_is_refused_not_zipped(self):
        """Mis-aligned rows would put one island's orientation on another."""
        def fair(pid, flats):
            return np.tile(FAIR_Q, (np.asarray(flats).size + 1, 1))

        rep = check_islands(
            _q_all(), _map_with_island([CENTRE]), NR, NC, PHASES,
            _scorers(stored_at_stored=0.10, stored_at_fair=0.20, enclosing=0.45),
            fair_orientation_fn=fair)
        assert rep["islands"][0]["decision"] == "undecided"

    def test_a_stage_one_winner_never_costs_a_fair_call(self):
        """The fairness step is for losers only; a healthy island is free."""
        calls = []

        def fair(pid, flats):
            calls.append(pid)
            return np.tile(FAIR_Q, (np.asarray(flats).size, 1))

        rep = check_islands(
            _q_all(), _map_with_island([CENTRE]), NR, NC, PHASES,
            _scorers(stored_at_stored=0.60, stored_at_fair=0.60, enclosing=0.45),
            fair_orientation_fn=fair)
        assert rep["islands"][0]["decision"] == "keep"
        assert calls == []


def test_island_size_is_derived_from_the_grain_size():
    """9 px is a grain (the user's definition), so 8 px is the largest island
    -- and nothing can fall between the two checks."""
    from backend.spherical_gpu.pipeline.island_check import MAX_ISLAND_PX
    assert MIN_GRAIN_PX == 9
    assert MAX_ISLAND_PX == MIN_GRAIN_PX - 1
