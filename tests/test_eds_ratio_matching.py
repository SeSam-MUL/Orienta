"""Ratio matching and the enrichment gate.

Spec: docs/superpowers/specs/2026-08-20-eds-ratio-matching-design.md

The two things this replaces, and why:
  - the relative missing-major veto (measured >= a fraction of NOMINAL)
    could not be calibrated across datasets and made Al7FeCu2 score the
    veto floor on every pixel of both Cu-bearing files;
  - the map and the Phase Suggestion panel used two different scorers and
    disagreed on the same pixel.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.services.chemistry_score import (
    background_levels, infer_matrix_element, score_phase_ratio,
)
from backend.api.services.crystal_hint_phase_fit import chemistry_fit

AL7FECU2 = {"Al": 70.0, "Fe": 10.0, "Cu": 20.0}
MGCU = {"Mg": 80.0, "Cu": 20.0}
AL = {"Al": 100.0}


def _maps(pixels, elements):
    return {el: np.array([p.get(el, 0.0) for p in pixels], dtype=np.float64)
            for el in elements}


# --- matrix inference -------------------------------------------------------

def test_infers_the_solvent_as_matrix():
    maps = _maps([{"Al": 95, "Cu": 5}, {"Al": 80, "Cu": 20}], ["Al", "Cu"])
    assert infer_matrix_element(maps) == "Al"


def test_no_matrix_when_nothing_dominates():
    """A 50/50 mixture has no solvent; scoring it as if it did would
    exempt the wrong element from the enrichment gate."""
    maps = _maps([{"Al": 34, "Cu": 33, "Mg": 33}], ["Al", "Cu", "Mg"])
    assert infer_matrix_element(maps) is None


# --- the ratio is what survives dilution -----------------------------------

def test_a_diluted_particle_still_matches_on_its_ratio():
    """The case the old veto could not express.

    A phase nominally Mg 80 / Cu 20 measured through a large interaction
    volume reads only a few at% of each — but Mg/Cu is still 4.
    """
    strong = {"Al": 20.0, "Mg": 64.0, "Cu": 16.0}     # barely diluted
    weak = {"Al": 90.0, "Mg": 8.0, "Cu": 2.0}         # heavily diluted, same ratio
    maps = _maps([strong, weak], ["Al", "Mg", "Cu"])
    # Backgrounds are RENORMALISED FRACTIONS, like the pixel side. Both
    # must sit below the diluted pixel (Mg 0.08, Cu 0.02) or the enrichment
    # gate correctly rejects it before the ratio is ever consulted.
    bg = {"Al": 0.95, "Mg": 0.010, "Cu": 0.004}
    s = score_phase_ratio(maps, MGCU, matrix_element="Al", background=bg)
    assert s[0] > 0.8, "the undiluted particle must match"
    assert s[1] > 0.8, f"the diluted particle must match too, got {s[1]:.3f}"


def test_absolute_composition_alone_would_have_lost_the_diluted_one():
    """Guards the premise: this is why ratios were introduced at all."""
    weak = {"Al": 90.0, "Mg": 8.0, "Cu": 2.0}
    assert chemistry_fit(weak, MGCU) < 0.3


# --- the enrichment gate ----------------------------------------------------

def test_enrichment_gate_rejects_background_level_signal():
    """An element merely at the map's background level is not a phase.

    This is what stops an aluminium matrix carrying a few at% of stray Fe
    from being labelled an Fe intermetallic — the failure the relative veto
    was invented for, now expressed against the map's own statistics
    instead of against a stoichiometry the quantification never reaches.
    """
    n = 200
    rng = np.random.default_rng(0)
    maps = {
        "Al": np.full(n, 90.0),
        "Mg": rng.normal(8.0, 0.2, n),
        "Cu": rng.normal(2.0, 0.05, n),    # flat: nothing is enriched
    }
    bg = background_levels(maps)
    s = score_phase_ratio(maps, MGCU, matrix_element="Al", background=bg)
    assert s.max() <= 0.05, "flat background must not read as a phase"


def test_enrichment_gate_passes_a_real_local_enrichment():
    n = 200
    maps = {
        "Al": np.full(n, 90.0),
        "Mg": np.full(n, 8.0),
        "Cu": np.full(n, 0.4),
    }
    maps["Cu"][:5] = 6.0          # a small, strongly enriched population
    maps["Mg"][:5] = 24.0
    bg = background_levels(maps)
    s = score_phase_ratio(maps, MGCU, matrix_element="Al", background=bg)
    assert s[:5].min() > 0.5, "the enriched population must match"
    assert s[5:].max() <= 0.05, "the background must not"


def test_matrix_element_is_exempt_from_the_enrichment_gate():
    """The matrix IS the background, so it can never be enriched over
    itself. Gating on it would veto the matrix phase across the map."""
    n = 50
    maps = {"Al": np.full(n, 99.0), "Cu": np.full(n, 1.0)}
    bg = background_levels(maps)
    s = score_phase_ratio(maps, AL, matrix_element="Al", background=bg)
    assert s.min() > 0.8


def test_single_sample_caller_skips_the_gate_instead_of_vetoing_everything():
    """Enrichment needs a population. A caller scoring one pixel without
    supplying a background must not have every phase vetoed by comparing
    a value against 1.3x itself."""
    one = {"Al": np.array([20.0]), "Mg": np.array([64.0]), "Cu": np.array([16.0])}
    s = score_phase_ratio(one, MGCU, matrix_element="Al")
    assert s[0] > 0.8


# --- the fallback path ------------------------------------------------------

def test_phase_without_two_non_matrix_elements_falls_back_to_composition():
    """Al.cif has no non-matrix element and Al6Fe has one — neither has a
    ratio. That is not a corner case: it covers the matrix phase, which is
    most of a typical map."""
    maps = _maps([{"Al": 99.0, "Fe": 1.0}], ["Al", "Fe"])
    assert score_phase_ratio(maps, AL, matrix_element="Al")[0] > 0.8


def test_dilution_does_not_rescue_a_wrong_ratio():
    """Ratio matching must not become a licence to match anything: a
    genuinely different Mg:Cu proportion still has to lose."""
    right = {"Al": 90.0, "Mg": 8.0, "Cu": 2.0}     # Mg/Cu = 4, nominal 4
    wrong = {"Al": 90.0, "Mg": 2.0, "Cu": 8.0}     # Mg/Cu = 0.25
    maps = _maps([right, wrong], ["Al", "Mg", "Cu"])
    # Low enough that BOTH pixels clear the enrichment gate, so the ratio
    # is what decides between them — which is the point of the test.
    bg = {"Al": 0.95, "Mg": 0.010, "Cu": 0.004}
    s = score_phase_ratio(maps, MGCU, matrix_element="Al", background=bg)
    assert s[0] > s[1] + 0.3, f"got {s[0]:.3f} vs {s[1]:.3f}"


def test_unmeasured_pixel_still_scores_zero_for_classifiers():
    maps = {"Al": np.array([0.0, 90.0]), "Cu": np.array([0.0, 10.0])}
    s = score_phase_ratio(maps, AL, matrix_element="Al", no_data_score=0.0)
    assert s[0] == 0.0
