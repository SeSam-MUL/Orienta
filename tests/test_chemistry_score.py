"""The vectorised phase-map scorer must agree with the indexing prior's
scalar `chemistry_fit`, and its relative veto must only ever tighten.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.services.chemistry_score import (
    has_chemistry, score_phase_vectorised,
)
from backend.api.services.crystal_hint_phase_fit import chemistry_fit

PHASE_AL = {"Al": 100.0}
PHASE_SI = {"Si": 100.0}
PHASE_INTERMETALLIC = {"Al": 75.8, "Fe": 11.6, "Si": 9.9, "Mn": 2.7}

# Real SampleB pixels, measured 2026-08-19.
PIXEL_CLEAN_MATRIX = {"Al": 98.0, "Si": 0.9, "Mn": 0.5, "Zn": 0.4, "Fe": 0.0,
                      "C": 0.8, "O": 1.0}
PIXEL_PARTICLE = {"Al": 72.1, "Fe": 11.1, "Si": 9.6, "Mn": 3.1, "Cu": 2.3,
                  "O": 1.0, "C": 0.8}
PIXEL_MEDIAN_MATRIX = {"Al": 84.2, "Fe": 2.8, "Si": 6.0, "Mn": 1.0,
                       "C": 0.5, "O": 0.9}
PIXEL_SI_PARTICLE = {"Al": 50.0, "Si": 48.9, "Fe": 0.14}

ALL_ELS = ["Al", "Fe", "Si", "Mn", "Cu", "Zn", "C", "O"]


def _as_maps(pixels, elements=ALL_ELS):
    return {el: np.array([p.get(el, 0.0) for p in pixels], dtype=np.float64)
            for el in elements}


def _scalar_ref(pixels, phase):
    return np.array([chemistry_fit(p, phase) for p in pixels], dtype=np.float32)


def test_matches_scalar_chemistry_fit_when_rel_req_zero():
    """rel_req=0 must reproduce the indexing prior's scorer exactly."""
    pixels = [PIXEL_CLEAN_MATRIX, PIXEL_PARTICLE, PIXEL_MEDIAN_MATRIX,
              PIXEL_SI_PARTICLE]
    maps = _as_maps(pixels)
    for phase in (PHASE_AL, PHASE_SI, PHASE_INTERMETALLIC):
        got = score_phase_vectorised(maps, phase, rel_req=0.0)
        want = _scalar_ref(pixels, phase)
        np.testing.assert_allclose(got, want, atol=1e-6)


def test_matches_scalar_on_random_compositions():
    """Fuzz the agreement so the two cannot drift apart unnoticed."""
    rng = np.random.default_rng(1234)
    els = ["Al", "Fe", "Si", "Mn", "Cu", "O", "C"]
    pixels = [{el: float(v) for el, v in zip(els, rng.uniform(0, 60, len(els)))}
              for _ in range(200)]
    maps = _as_maps(pixels, els)
    for phase in (PHASE_AL, PHASE_INTERMETALLIC, {"Al": 85.7, "Mn": 14.3}):
        np.testing.assert_allclose(
            score_phase_vectorised(maps, phase, rel_req=0.0),
            _scalar_ref(pixels, phase),
            atol=1e-6,
        )


def test_relative_veto_rejects_matrix_pixel_lacking_the_defining_element():
    """A matrix pixel with 2.8 at% Fe must not win a phase needing 11.6 at%."""
    maps = _as_maps([PIXEL_MEDIAN_MATRIX])
    without = score_phase_vectorised(maps, PHASE_INTERMETALLIC, rel_req=0.0)
    with_veto = score_phase_vectorised(maps, PHASE_INTERMETALLIC, rel_req=0.5)
    assert without[0] > 0.4, "baseline: the old behaviour accepts this pixel"
    assert with_veto[0] <= 0.05, "relative veto must reject it"


def test_relative_veto_keeps_a_real_particle_pixel():
    """A genuine particle pixel (Fe 11.1 vs required 11.6) must survive."""
    maps = _as_maps([PIXEL_PARTICLE])
    s = score_phase_vectorised(maps, PHASE_INTERMETALLIC, rel_req=0.5)
    assert s[0] > 0.7, f"real particle must keep a high score, got {s[0]}"


def test_relative_veto_never_raises_a_score():
    """Pure tightening: rel_req may only ever lower scores."""
    rng = np.random.default_rng(0)
    n = 500
    maps = {el: rng.uniform(0, 100, n) for el in ("Al", "Fe", "Si", "Mn")}
    base = score_phase_vectorised(maps, PHASE_INTERMETALLIC, rel_req=0.0)
    for rr in (0.2, 0.5, 0.8):
        tight = score_phase_vectorised(maps, PHASE_INTERMETALLIC, rel_req=rr)
        assert np.all(tight <= base + 1e-6)


def test_empty_phase_composition_is_neutral():
    maps = _as_maps([PIXEL_CLEAN_MATRIX])
    assert score_phase_vectorised(maps, {}, rel_req=0.5)[0] == pytest.approx(1.0)


def test_pixel_without_chemistry_is_neutral():
    maps = {"Al": np.array([0.0]), "Fe": np.array([0.0])}
    assert score_phase_vectorised(maps, PHASE_AL, rel_req=0.5)[0] == pytest.approx(1.0)


def test_c_and_o_are_ignored():
    """C/O are acquisition artifacts and must not change the score."""
    clean = _as_maps([{"Al": 75.8, "Fe": 11.6, "Si": 9.9, "Mn": 2.7}],
                     ["Al", "Fe", "Si", "Mn"])
    dirty = _as_maps([{"Al": 75.8, "Fe": 11.6, "Si": 9.9, "Mn": 2.7,
                       "C": 20.0, "O": 15.0}],
                     ["Al", "Fe", "Si", "Mn", "C", "O"])
    np.testing.assert_allclose(
        score_phase_vectorised(clean, PHASE_INTERMETALLIC, rel_req=0.5),
        score_phase_vectorised(dirty, PHASE_INTERMETALLIC, rel_req=0.5),
        atol=1e-6,
    )


def test_empty_input_returns_empty():
    assert score_phase_vectorised({}, PHASE_AL).shape == (0,)


# --- no-data handling -------------------------------------------------------
# chemistry_fit returns a NEUTRAL 1.0 for a pixel with no chemistry, which is
# right for the indexing prior (a multiplier) and catastrophic for a
# classifier (argmax hands the pixel to row 1 of the library at a perfect
# score). Measured: 63.7 % of one real map fabricated this way.

def test_no_data_pixel_scores_zero_for_classifiers():
    maps = {"Al": np.array([0.0, 90.0]), "Fe": np.array([0.0, 10.0])}
    s = score_phase_vectorised(maps, PHASE_AL, rel_req=0.3, no_data_score=0.0)
    assert s[0] == 0.0, "unmeasured pixel must not score"
    assert s[1] > 0.0, "the measured pixel is unaffected"


def test_no_data_default_stays_neutral_for_the_indexing_prior():
    maps = {"Al": np.array([0.0]), "Fe": np.array([0.0])}
    assert score_phase_vectorised(maps, PHASE_AL)[0] == pytest.approx(1.0)


def test_every_phase_ties_at_zero_on_a_dead_pixel():
    """The failure mode: all candidates tie at the MAXIMUM, so argmax picks
    library row 1 and reports a perfect match."""
    maps = {"Al": np.array([0.0]), "Fe": np.array([0.0]), "Si": np.array([0.0])}
    scores = [score_phase_vectorised(maps, p, no_data_score=0.0)[0]
              for p in (PHASE_AL, PHASE_SI, PHASE_INTERMETALLIC)]
    assert set(scores) == {0.0}


def test_nan_counts_are_treated_as_no_data():
    """np.maximum propagates NaN where Python's max(0.0, nan) returns 0.0,
    so the two scorers would otherwise disagree by the full range."""
    maps = {"Al": np.array([np.nan]), "Fe": np.array([11.0]),
            "Si": np.array([10.0]), "Mn": np.array([3.0])}
    s = score_phase_vectorised(maps, PHASE_INTERMETALLIC, rel_req=0.3,
                               no_data_score=0.0)
    assert np.isfinite(s[0]) and s[0] == 0.0


def test_has_chemistry_mask():
    maps = {"Al": np.array([0.0, 90.0, np.nan]), "Fe": np.array([0.0, 10.0, 0.0])}
    m = has_chemistry(maps)
    assert m.tolist() == [False, True, False]
    assert has_chemistry({}).shape == (0,)
