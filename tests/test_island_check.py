"""Wrong pixels INSIDE a grain — found by render, not by neighbourhood.

Reported on a 7050 sample (2026-09-05): in a heavily deformed region, matrix
pixels sit in the middle of an intermetallic particle. The per-grain phase check
cannot see them — it skips anything below MIN_GRAIN_PX (5), so a one-pixel
island is never even counted.

The point of this module is that it does NOT decide by majority vote of the
neighbours (that is what modal smoothing does, and it is a guess). It renders
the candidates and compares: the enclosing phase at the neighbour's orientation,
and the stored phase's own pseudo-symmetric variants. Only the numbers decide.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.spherical_gpu.pipeline.island_check import (  # noqa: E402
    check_islands,
    find_enclosed_islands,
)

NR = NC = 9
IDENT = np.array([1.0, 0.0, 0.0, 0.0])
PHASES = {1: "m-3m", 2: "4/mmm"}


def _map_with_island(island_flats, background=2, island_phase=1):
    """A `background` field with `island_flats` set to `island_phase`."""
    pf = np.full(NR * NC, background, dtype=np.int64)
    for f in island_flats:
        pf[f] = island_phase
    return pf


def _q_all_identity():
    return np.tile(IDENT, (NR * NC, 1)).astype(float)


# ---------------------------------------------------------------------------
# find_enclosed_islands
# ---------------------------------------------------------------------------

def test_finds_a_single_enclosed_pixel():
    centre = 4 * NC + 4
    isls = find_enclosed_islands(_map_with_island([centre]), NR, NC)
    assert len(isls) == 1
    assert isls[0]["pixels"].tolist() == [centre]
    assert isls[0]["stored_pid"] == 1
    assert isls[0]["enclosing_pid"] == 2
    # Every island pixel knows the neighbour whose orientation it will borrow.
    assert set(isls[0]["neighbour_of"]) == {centre}


def test_finds_a_small_multi_pixel_island():
    a, b = 4 * NC + 4, 4 * NC + 5
    isls = find_enclosed_islands(_map_with_island([a, b]), NR, NC)
    assert len(isls) == 1
    assert isls[0]["pixels"].tolist() == [a, b]
    assert set(isls[0]["neighbour_of"]) == {a, b}


def test_a_grain_is_not_an_island():
    """Anything the per-grain check already handles must be left alone."""
    big = [r * NC + c for r in range(2, 7) for c in range(2, 7)]   # 25 px
    assert find_enclosed_islands(_map_with_island(big), NR, NC) == []


def test_touching_two_phases_is_not_enclosed():
    """Without ONE surrounding phase there is no free candidate orientation."""
    pf = _map_with_island([4 * NC + 4])
    pf[4 * NC + 3] = 3            # a third phase on one side
    assert find_enclosed_islands(pf, NR, NC) == []


def test_unindexed_neighbours_do_not_shield_a_wrong_pixel():
    """A hole in the map must not make a wrong pixel unquestionable."""
    centre = 4 * NC + 4
    pf = _map_with_island([centre])
    pf[4 * NC + 3] = -1
    isls = find_enclosed_islands(pf, NR, NC)
    assert len(isls) == 1
    assert isls[0]["enclosing_pid"] == 2


def test_an_island_with_no_indexed_neighbour_is_skipped():
    centre = 4 * NC + 4
    pf = _map_with_island([centre])
    for nb in (centre - 1, centre + 1, centre - NC, centre + NC):
        pf[nb] = -1
    assert find_enclosed_islands(pf, NR, NC) == []


def test_the_map_edge_does_not_invent_islands():
    """A corner pixel of the background is not an island of itself."""
    pf = np.full(NR * NC, 2, dtype=np.int64)
    assert find_enclosed_islands(pf, NR, NC) == []


# ---------------------------------------------------------------------------
# check_islands
# ---------------------------------------------------------------------------

def _scorers(stored_score, enclosing_score, variant_score=None, variant_q=None):
    """Renderers with the answer baked in, so the DECISION is what is tested."""
    def phase1(flats, quats):
        quats = np.asarray(quats, dtype=float).reshape(-1, 4)
        if variant_q is not None:
            hit = np.all(np.isclose(quats, np.asarray(variant_q)), axis=1)
            return np.where(hit, variant_score, stored_score)
        return np.full(quats.shape[0], stored_score)

    def phase2(flats, quats):
        return np.full(np.asarray(quats).reshape(-1, 4).shape[0], enclosing_score)

    return {1: phase1, 2: phase2}


def _fair_agrees_with_stored(pid, flats):
    """A fairness step whose Hough answer IS the stored orientation.

    The fairness step (test_island_fairness.py) re-orients the stored phase
    before believing a loss, because on a real map a third of stage-1 losses
    were a bad orientation, not a bad phase. These older tests are about the
    geometry and the scoring rule, so they hand it an orientation that changes
    nothing -- with it, a loss is a loss.
    """
    return np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (np.asarray(flats).size, 1))


def test_the_enclosing_phase_wins_when_it_renders_better():
    centre = 4 * NC + 4
    rep = check_islands(_q_all_identity(), _map_with_island([centre]), NR, NC,
                        PHASES, _scorers(stored_score=0.15, enclosing_score=0.45),
                        fair_orientation_fn=_fair_agrees_with_stored)
    assert rep["n_islands"] == 1
    e = rep["islands"][0]
    assert e["decision"] == "phase"
    assert e["winner_phase"] == 2
    assert e["margin"] == pytest.approx(0.30, abs=1e-9)
    assert rep["n_phase_swap"] == 1


def test_a_near_tie_changes_nothing():
    """These are single pixels; a hair's difference is not evidence."""
    centre = 4 * NC + 4
    rep = check_islands(_q_all_identity(), _map_with_island([centre]), NR, NC,
                        PHASES, _scorers(stored_score=0.40, enclosing_score=0.43))
    assert rep["islands"][0]["decision"] == "keep"
    assert rep["n_phase_swap"] == 0


def test_a_pseudo_symmetric_variant_can_win_instead_of_the_phase():
    """The pixel is the right PHASE in the wrong ORIENTATION — a different fix."""
    centre = 4 * NC + 4
    var = np.array([0.0, 1.0, 0.0, 0.0])
    rep = check_islands(
        _q_all_identity(), _map_with_island([centre]), NR, NC, PHASES,
        _scorers(stored_score=0.15, enclosing_score=0.17,
                 variant_score=0.55, variant_q=var),
        variants_fn=lambda q, pg: np.stack([var]),
    )
    e = rep["islands"][0]
    assert e["decision"] == "variant"
    assert e["winner_phase"] == 1
    assert e["variant"] == pytest.approx(list(var))
    assert rep["n_variant_flip"] == 1
    assert rep["n_phase_swap"] == 0


def test_variants_are_not_even_generated_when_the_phase_already_won():
    """Stage 2 costs a render per variant; it must not run for nothing."""
    centre = 4 * NC + 4
    calls = []

    def variants_fn(q, pg):
        calls.append(q)
        return np.stack([np.array([0.0, 1.0, 0.0, 0.0])])

    check_islands(_q_all_identity(), _map_with_island([centre]), NR, NC, PHASES,
                  _scorers(stored_score=0.15, enclosing_score=0.45),
                  variants_fn=variants_fn)
    assert calls == []


def test_each_pixel_borrows_its_OWN_neighbour_orientation():
    """Deformed grains drift — one averaged orientation would be wrong exactly
    at the pixels this check exists for."""
    a, b = 4 * NC + 4, 4 * NC + 5
    q = _q_all_identity()
    # Give the two islands' neighbours clearly different orientations.
    q[a - NC] = [0.0, 1.0, 0.0, 0.0]
    q[b - NC] = [0.0, 0.0, 1.0, 0.0]
    seen = {}

    def phase2(flats, quats):
        for f, qq in zip(np.asarray(flats).reshape(-1), np.asarray(quats).reshape(-1, 4)):
            seen[int(f)] = qq.copy()
        return np.full(np.asarray(flats).reshape(-1).size, 0.1)

    fns = _scorers(stored_score=0.5, enclosing_score=0.1)
    fns[2] = phase2
    check_islands(q, _map_with_island([a, b]), NR, NC, PHASES, fns)
    assert not np.allclose(seen[a], seen[b]), (
        "both island pixels were scored with the same orientation — the "
        "neighbour orientation was averaged instead of taken per pixel"
    )


def test_an_island_without_a_renderer_is_reported_as_unjudged():
    """Not as 'keep': 'nothing was wrong' and 'nothing could be compared' are
    different answers and must never share a number."""
    centre = 4 * NC + 4
    rep = check_islands(_q_all_identity(), _map_with_island([centre]), NR, NC,
                        PHASES, {2: lambda f, q: np.full(len(f), 0.4)})
    assert rep["islands"][0]["decision"] == "no-scorer"
    assert rep["n_unresolved"] == 1
    assert rep["n_phase_swap"] == 0


def test_nothing_is_modified():
    centre = 4 * NC + 4
    q = _q_all_identity()
    pf = _map_with_island([centre])
    q_before, pf_before = q.copy(), pf.copy()
    check_islands(q, pf, NR, NC, PHASES, _scorers(0.1, 0.9))
    assert np.array_equal(q, q_before)
    assert np.array_equal(pf, pf_before)


def test_every_island_pixel_gets_a_neighbour_even_beside_unindexed_holes():
    """Found on a real 7050 map: it crashed with KeyError.

    A pixel whose own outside neighbours are ALL unindexed had no orientation
    to borrow. It must fall back to the nearest ring pixel rather than take the
    run down — an unindexed hole is common and is not a reason to stop.
    """
    a, b = 4 * NC + 4, 4 * NC + 5
    pf = _map_with_island([a, b])
    # Wall `a` off with unindexed pixels; only `b` keeps real neighbours.
    for nb in (a - 1, a - NC, a + NC):
        pf[nb] = -1
    isls = find_enclosed_islands(pf, NR, NC)
    assert len(isls) == 1
    assert set(isls[0]["neighbour_of"]) == {a, b}, "a pixel was left without one"

    # And the whole check runs on it instead of raising.
    rep = check_islands(_q_all_identity(), pf, NR, NC, PHASES,
                        _scorers(stored_score=0.15, enclosing_score=0.45),
                        fair_orientation_fn=_fair_agrees_with_stored)
    assert rep["n_islands"] == 1
    assert rep["islands"][0]["decision"] == "phase"
