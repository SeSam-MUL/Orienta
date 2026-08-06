"""EDS pre-flight check, grid guard, and the missing-major-element veto.

The veto had NO test at all before 2026-08-05 — the whole block could be
deleted and the suite stayed green, while it is precisely what decides whether
a chemically impossible phase can win a pixel. These tests pin the behaviour
that was measured on ProbeB.
"""
import numpy as np
import pytest

from backend.api.services.crystal_hint_phase_fit import chemistry_fit, phase_nominal_at_pct
from backend.api.services.eds_preflight import (
    DEFINING_FRACTION, PRESENT_AT_PCT, defining_elements, run_preflight,
)
from backend.api.services import eds_indexing_prior as prior

ALPHA = "Mn0.5Fe0.5Al5Si0.68"          # Al 74.9 / Si 10.2 / Fe 7.5 / Mn 7.5 at%
# Two real ProbeB pixels from opposite ends of the Al<->Si mixing line. Both are
# "rim" pixels; which phase SHOULD win differs, and alpha must lose at both.
RIM_AL_SIDE = {"Al": 84.0, "Si": 10.0, "Mg": 2.0, "Fe": 0.2, "Mn": 0.3}
RIM_SI_SIDE = {"Si": 63.2, "Al": 31.2, "Mg": 1.7, "Fe": 0.2, "Mn": 0.3}
# measured inside the real 1607-px Fe/Mn particle of ProbeB
REAL_ALPHA = {"Al": 75.7, "Si": 9.6, "Fe": 4.2, "Mn": 2.2, "Mg": 2.5}


# --------------------------------------------------------------------------
# the veto itself
# --------------------------------------------------------------------------

def test_veto_fires_when_defining_minor_elements_are_absent():
    """alpha needs Fe 7.5 + Mn 7.5 at%; a rim pixel has ~0.2. It must be capped.

    Regression for the shipped bug: with _MAJOR_REQ = 0.15 this returned 0.596,
    i.e. chemistry preferred alpha over Si (0.235) on Al/Si mixing pixels, and
    the alpha rim survived even a fully symmetric prior.
    """
    for name, pixel in (("Al side", RIM_AL_SIDE), ("Si side", RIM_SI_SIDE)):
        fit = chemistry_fit(pixel, phase_nominal_at_pct(ALPHA))
        assert fit <= 0.05 + 1e-9, (
            f"alpha scored {fit:.3f} on the {name} of the mixing line "
            f"(Fe 0.2 / Mn 0.3 at%) — the missing-major veto did not fire "
            f"(is _MAJOR_REQ above 0.075?)"
        )


def test_veto_does_not_fire_on_the_real_alpha_particle():
    """The other side of the fix: real alpha must stay detectable.

    Heavy elements read ~0.42x nominal with standardless Cliff-Lorimer, so a
    7.5 at% nominal Fe measures ~4 at%. That must stay above the absence floor.
    """
    fit = chemistry_fit(REAL_ALPHA, phase_nominal_at_pct(ALPHA))
    assert fit > 0.5, f"real alpha core scored only {fit:.3f} — over-vetoed"
    assert fit > chemistry_fit(REAL_ALPHA, phase_nominal_at_pct("Al"))
    assert fit > chemistry_fit(REAL_ALPHA, phase_nominal_at_pct("Si"))


def test_the_endmember_wins_at_both_ends_of_the_mixing_line():
    """The user-visible symptom, as assertions.

    alpha's nominal Al/Si ratio sits ON the Al<->Si mixing line, so before the
    fix it beat BOTH endmembers on boundary pixels. Now the endmember that the
    pixel is actually made of must win at each end.
    """
    a_alpha = chemistry_fit(RIM_AL_SIDE, phase_nominal_at_pct(ALPHA))
    a_al = chemistry_fit(RIM_AL_SIDE, phase_nominal_at_pct("Al"))
    assert a_al > a_alpha, f"Al {a_al:.3f} should beat alpha {a_alpha:.3f}"

    s_alpha = chemistry_fit(RIM_SI_SIDE, phase_nominal_at_pct(ALPHA))
    s_si = chemistry_fit(RIM_SI_SIDE, phase_nominal_at_pct("Si"))
    assert s_si > s_alpha, f"Si {s_si:.3f} should beat alpha {s_alpha:.3f}"


# --------------------------------------------------------------------------
# defining elements — the preflight must use the same rule as the veto
# --------------------------------------------------------------------------

def test_defining_elements_include_alpha_minor_elements():
    d = defining_elements(ALPHA)
    assert "Fe" in d and "Mn" in d, (
        "Fe/Mn at 7.5 at% nominal must count as defining, otherwise the "
        "preflight and the veto disagree")


def test_defining_rule_matches_veto_threshold():
    """A phase reported at 0 % by the preflight must be vetoed by chemistry_fit."""
    formula = "Al9Fe1"          # Fe = 10 at% nominal -> defining
    assert "Fe" in defining_elements(formula)
    pixel_without_fe = {"Al": 98.0, "Si": 2.0}
    assert chemistry_fit(pixel_without_fe, phase_nominal_at_pct(formula)) <= 0.05 + 1e-9


def test_defining_fraction_constant_is_below_alpha_minor_fraction():
    nominal = phase_nominal_at_pct(ALPHA)
    frac_fe = nominal["Fe"] / sum(nominal.values())
    assert DEFINING_FRACTION <= frac_fe, (
        f"DEFINING_FRACTION={DEFINING_FRACTION} is above alpha's Fe fraction "
        f"{frac_fe:.3f} — the phase that caused the bug would slip through")


# --------------------------------------------------------------------------
# grid guard
# --------------------------------------------------------------------------

def _fake_maps(n_rows, n_cols):
    n = n_rows * n_cols
    return ({"Al": np.full(n, 90.0), "Si": np.full(n, 10.0)}, n_rows, n_cols, "f.h5")


def test_grid_mismatch_raises(monkeypatch):
    monkeypatch.setattr(prior, "_build_at_pct_maps_for_loaded_file",
                        lambda: _fake_maps(10, 10))
    with pytest.raises(prior.EdsGridMismatch):
        prior.measured_atpct_per_pixel(expected_shape=(12, 10))


def test_grid_mismatch_detected_via_selection_mask(monkeypatch):
    monkeypatch.setattr(prior, "_build_at_pct_maps_for_loaded_file",
                        lambda: _fake_maps(10, 10))
    with pytest.raises(prior.EdsGridMismatch):
        prior.measured_atpct_per_pixel(np.ones((5, 20), dtype=bool))


def test_matching_grid_passes(monkeypatch):
    monkeypatch.setattr(prior, "_build_at_pct_maps_for_loaded_file",
                        lambda: _fake_maps(4, 5))
    out = prior.measured_atpct_per_pixel(expected_shape=(4, 5))
    assert out is not None and len(out) == 20
    assert out[0]["Al"] == pytest.approx(90.0)


# --------------------------------------------------------------------------
# preflight report
# --------------------------------------------------------------------------

def test_preflight_flags_phase_with_no_chemical_support(monkeypatch):
    """A phase whose defining elements are absent everywhere must be flagged.

    This is the check that would have prevented the reported bug: alpha needs
    Fe+Mn, the map has none, so its area is bounded at 0 %.
    """
    import backend.api.services.eds_preflight as pf
    n_rows, n_cols = 6, 7
    n = n_rows * n_cols
    maps = {"Al": np.full(n, 92.0), "Si": np.full(n, 8.0)}   # no Fe, no Mn
    monkeypatch.setattr(
        "backend.api.routes.eds._build_at_pct_maps_for_loaded_file",
        lambda: (maps, n_rows, n_cols, "f.h5"), raising=False)
    monkeypatch.setattr(pf, "_ebsd_nav_shape", lambda: (n_rows, n_cols))

    rep = pf.run_preflight(["alpha.sht", "al.sht"], phase_formulas=[ALPHA, "Al"])
    assert rep["can_index"] is True          # missing chemistry only warns
    assert rep["checks"]["grid"]["ok"] is True
    by_name = {p["formula"]: p for p in rep["phases"]}
    assert by_name[ALPHA]["severity"] == "warn"
    assert by_name[ALPHA]["missing_elements"] == ["Fe", "Mn"] or \
           set(by_name[ALPHA]["missing_elements"]) == {"Fe", "Mn"}
    assert by_name["Al"]["severity"] == "ok"
    assert by_name["Al"]["max_area_pct"] == pytest.approx(100.0)


def test_preflight_blocks_on_grid_mismatch(monkeypatch):
    import backend.api.services.eds_preflight as pf
    n = 6 * 7
    monkeypatch.setattr(
        "backend.api.routes.eds._build_at_pct_maps_for_loaded_file",
        lambda: ({"Al": np.full(n, 100.0)}, 6, 7, "f.h5"), raising=False)
    monkeypatch.setattr(pf, "_ebsd_nav_shape", lambda: (7, 6))   # transposed

    rep = pf.run_preflight(["al.sht"], phase_formulas=["Al"])
    assert rep["can_index"] is False
    assert "grid" in rep["blocking"]
    assert rep["checks"]["grid"]["severity"] == "block"


def test_preflight_upper_bound_semantics(monkeypatch):
    """max_area_pct is an upper bound, never an estimate."""
    import backend.api.services.eds_preflight as pf
    n_rows, n_cols = 10, 10
    n = n_rows * n_cols
    fe = np.zeros(n); fe[:5] = 6.0            # Fe only on 5 % of the map
    maps = {"Al": np.full(n, 80.0), "Fe": fe, "Si": np.full(n, 14.0)}
    monkeypatch.setattr(
        "backend.api.routes.eds._build_at_pct_maps_for_loaded_file",
        lambda: (maps, n_rows, n_cols, "f.h5"), raising=False)
    monkeypatch.setattr(pf, "_ebsd_nav_shape", lambda: (n_rows, n_cols))

    rep = pf.run_preflight(["x.sht"], phase_formulas=["Al6Fe"])
    p = rep["phases"][0]
    assert p["max_area_pct"] == pytest.approx(5.0)
    assert p["is_upper_bound"] is True
    assert p["threshold_at_pct"] == PRESENT_AT_PCT
