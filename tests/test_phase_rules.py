"""User-authored rules: what they mean, and what they refuse to guess.

A rule decides ONE thing — whether a phase may compete for a region. Not a
weight, not a preference. These tests pin that meaning, the units it is
evaluated in, and every case where the honest answer is "I cannot tell", which
must never come out as "yes".
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.services.phase_rules import (
    ElementRange, EnrichmentRange, PhaseRule, RatioRange, RuleSet,
    evaluate_rule, gate_scores, renormalised_at_pct, rule_set_from_dict,
    rule_set_to_dict,
)


def maps(**cols):
    """Element maps from equal-length lists."""
    return {el: np.asarray(v, dtype=float) for el, v in cols.items()}


# --- units ------------------------------------------------------------------

def test_ranges_are_evaluated_on_renormalised_at_pct():
    """Raw at% totals drift with yield alone.

    Measured on SampleB the raw total ranges 91.30 to 100.00, so an identical
    composition would pass a raw-at% band in one region and fail it in another
    for a reason that is not chemistry. Both pixels below are the same
    composition; only the total differs.
    """
    at = maps(Al=[80.0, 40.0], Si=[20.0, 10.0])   # 100 total, then 50
    _els, pct = renormalised_at_pct(at)
    assert pct[0] == pytest.approx([80.0, 80.0])   # Al, both pixels
    assert pct[1] == pytest.approx([20.0, 20.0])   # Si, both pixels


def test_carbon_and_oxygen_take_no_part():
    """They are excluded from the clustering, so a rule must not see them
    either — otherwise a rule and the grouping would disagree about what the
    composition of a region even is."""
    at = maps(Al=[50.0], Si=[50.0], C=[100.0], O=[100.0])
    els, pct = renormalised_at_pct(at)
    assert els == ["Al", "Si"]
    assert pct[:, 0] == pytest.approx([50.0, 50.0])


# --- element content --------------------------------------------------------

def test_a_content_range_admits_only_what_is_inside_it():
    at = maps(Al=[97.0, 60.0, 43.0], Si=[3.0, 40.0, 57.0])
    r = PhaseRule("Si.cif", elements=(ElementRange("Si", 40.0, None),))
    assert list(evaluate_rule(r, at).allowed) == [False, True, True]


def test_both_bounds_apply():
    at = maps(Al=[90.0, 70.0, 40.0], Si=[10.0, 30.0, 60.0])
    r = PhaseRule("X", elements=(ElementRange("Si", 20.0, 40.0),))
    assert list(evaluate_rule(r, at).allowed) == [False, True, False]


def test_a_rule_with_no_clauses_admits_everything():
    at = maps(Al=[90.0], Si=[10.0])
    assert evaluate_rule(PhaseRule("X"), at).allowed.all()


def test_the_reason_names_the_clause_and_the_measurement():
    """The diagnostic is the point of hard semantics: a soft weight could not
    produce it."""
    at = maps(Al=[75.0], Si=[25.0])
    r = PhaseRule("Si.cif", elements=(ElementRange("Si", 40.0, None),))
    out = evaluate_rule(r, at)
    assert not out.allowed.any()
    assert "Si" in out.reason and "40" in out.reason and "25" in out.reason


# --- ratios -----------------------------------------------------------------

def test_a_ratio_range_uses_the_renormalised_vector():
    at = maps(Mg=[30.0, 10.0], Si=[10.0, 30.0], Al=[60.0, 60.0])
    r = PhaseRule("X", ratios=(RatioRange("Mg", "Si", 0.5, 3.0),))
    # 30/10 = 3.0 passes at the boundary; 10/30 = 0.33 does not.
    assert list(evaluate_rule(r, at).allowed) == [True, False]


def test_a_ratio_below_the_detection_floor_does_not_pass():
    """Measured on SampleB: where either leg is below the floor the ratio
    spans 0 to 2.5e10, and a naive band check passes 1.7 % of pure noise. A
    ratio we cannot measure is not a ratio that qualifies."""
    at = maps(Mg=[0.001], Si=[0.001], Al=[99.998])
    r = PhaseRule("X", ratios=(RatioRange("Mg", "Si", 0.5, 3.0),))
    out = evaluate_rule(r, at)
    assert not out.allowed.any()
    assert out.undecidable


def test_a_ratio_needs_both_legs_present_not_merely_nonzero():
    # 0.5 at% clears a bare "> 0" test but not the absence threshold.
    at = maps(Mg=[0.5], Si=[0.5], Al=[99.0])
    r = PhaseRule("X", ratios=(RatioRange("Mg", "Si", 0.5, 3.0),))
    assert not evaluate_rule(r, at).allowed.any()


def test_a_zero_denominator_does_not_blow_up():
    at = maps(Mg=[50.0], Si=[0.0], Al=[50.0])
    out = evaluate_rule(PhaseRule("X", ratios=(RatioRange("Mg", "Si", 0.5, 3.0),)), at)
    assert not out.allowed.any()
    assert np.isfinite(out.allowed).all()


# --- enrichment -------------------------------------------------------------

def test_enrichment_is_measured_against_the_map_background():
    """The rule type that keeps 96.3 % of a real particle where a content band
    keeps 46.3 %. The bar is the map's own median, so it needs no k-factor."""
    # Nine matrix pixels at Si 2 at%, one particle pixel at Si 50.
    at = maps(Si=[2.0] * 9 + [50.0], Al=[98.0] * 9 + [50.0])
    r = PhaseRule("Si.cif", enrichment=(EnrichmentRange("Si", 2.0, None),))
    out = evaluate_rule(r, at)
    assert list(out.allowed) == [False] * 9 + [True]


def test_enrichment_without_a_background_blocks_rather_than_passes():
    at = maps(Al=[100.0], Si=[0.0])
    r = PhaseRule("X", enrichment=(EnrichmentRange("Si", 2.0, None),))
    out = evaluate_rule(r, at)
    assert not out.allowed.any()
    assert out.undecidable


# --- what it refuses to guess ----------------------------------------------

def test_a_rule_on_an_unmeasured_element_blocks_and_says_so():
    """Silently passing would show a phase the user believes was checked."""
    at = maps(Al=[90.0], Si=[10.0])
    out = evaluate_rule(PhaseRule("X", elements=(ElementRange("W", 1.0, None),)), at)
    assert not out.allowed.any()
    assert out.undecidable and "W" in out.reason


def test_every_clause_must_hold():
    at = maps(Mg=[30.0], Si=[10.0], Al=[60.0])
    r = PhaseRule("X",
                  elements=(ElementRange("Mg", 20.0, None),),      # passes
                  ratios=(RatioRange("Mg", "Si", 0.1, 1.0),))      # 3.0, fails
    assert not evaluate_rule(r, at).allowed.any()


# --- the gate ---------------------------------------------------------------

def test_a_blocked_phase_scores_zero_so_it_cannot_win():
    """"May not compete" has to beat a good score, or it means nothing."""
    at = maps(Al=[10.0], Si=[90.0])
    scores = np.array([0.95])
    r = PhaseRule("X", elements=(ElementRange("Si", 0.0, 50.0),))
    gated, out = gate_scores(scores, r, at)
    assert gated[0] == 0.0
    assert out is not None and not out.allowed[0]


def test_no_rule_leaves_the_scores_untouched():
    at = maps(Al=[50.0], Si=[50.0])
    scores = np.array([0.42])
    gated, out = gate_scores(scores, None, at)
    assert gated is scores and out is None


def test_a_shape_mismatch_is_refused_rather_than_gating_the_wrong_pixels():
    at = maps(Al=[50.0, 50.0], Si=[50.0, 50.0])
    r = PhaseRule("X", elements=(ElementRange("Si", 10.0, None),))
    with pytest.raises(ValueError):
        gate_scores(np.array([0.5]), r, at)


# --- the rule set -----------------------------------------------------------

def test_rules_are_keyed_on_the_entry_key_not_a_position():
    """Phase ids are positions in a candidate list that changes with which
    phases are ticked; the store already carries hand-set pixels by key for
    this reason."""
    rs = RuleSet(rules=(PhaseRule("beta-AlFeSi.cif"),))
    assert rs.rule_for("beta-AlFeSi.cif") is not None
    assert rs.rule_for("Al.cif") is None


def test_an_empty_rule_set_is_recognised_as_empty():
    assert RuleSet().is_empty
    assert not RuleSet(rules=(PhaseRule("x", elements=(ElementRange("Si", 1),)),)).is_empty
    assert not RuleSet(phase_keys=("Al.cif",)).is_empty


def test_json_round_trip_keeps_every_clause():
    rs = RuleSet(
        name="Al-Fe-Si",
        phase_keys=("Al.cif",),
        matrix_elements=("Al",),
        rules=(PhaseRule(
            "beta-AlFeSi.cif", "Al5FeSi",
            elements=(ElementRange("Fe", 3.0, 14.0),),
            ratios=(RatioRange("Fe", "Si", 0.6, 1.6),),
            enrichment=(EnrichmentRange("Fe", 1.5, None),),
        ),),
    )
    back = rule_set_from_dict(rule_set_to_dict(rs))
    assert back == rs


def test_malformed_json_is_dropped_rather_than_crashing():
    rs = rule_set_from_dict({
        "rules": [
            {},                                        # no key
            {"phase_key": "a", "elements": [{}]},      # no element
            {"phase_key": "b", "elements": [{"element": "Si", "min_at_pct": "abc"}]},
        ],
    })
    assert [r.phase_key for r in rs.rules] == ["a", "b"]
    assert rs.rule_for("a").elements == ()
    assert rs.rule_for("b").elements[0].min_at_pct is None


def test_no_payload_means_no_rules():
    assert rule_set_from_dict(None) is None
    assert rule_set_from_dict({}) is None
