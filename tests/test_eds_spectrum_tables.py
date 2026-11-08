"""Vendored X-ray tables for the EDS spectrum path."""
import os

import numpy as np
import pytest

from backend.eds_spectrum.tables import (
    SOURCES,
    edge_kev,
    family_lines,
    fluor_yield_k,
    main_line,
    mu_rho,
    mu_rho_mix,
    mu_steps,
)


def test_fe_k_family():
    lines = family_lines("Fe", "K")
    assert main_line("Fe", "K") == pytest.approx((6.4039, 1.0), abs=1e-3)
    kb = [w for e, w in lines if abs(e - 7.058) < 0.01]
    assert kb and kb[0] == pytest.approx(0.127, abs=0.01)


def test_mu_rho_al_below_and_above_its_k_edge():
    e = np.array([1.555, 1.565])  # Al K edge 1.5596
    mu = mu_rho("Al", e)
    assert mu[1] / mu[0] > 8  # jump ratio ~10


@pytest.mark.parametrize("el", ["Si", "Mn", "Fe", "Cu", "Zn"])
def test_mu_rho_jumps_at_k_edge_of_every_alloy_element(el):
    k = edge_kev(el, "K")
    mu = mu_rho(el, np.array([k - 0.01, k + 0.01]))
    assert mu[1] / mu[0] > 5


def test_mu_rho_al_at_si_ka_matches_xraylib():
    # xraylib 4.3.0 CS_Total(Al, 1.740 keV) = 3111.63 cm^2/g.
    assert mu_rho("Al", np.array([1.740]))[0] == pytest.approx(3111.6, rel=0.01)


def test_tables_match_eds_utils_line_energies():
    import eds_utils as eu

    for el in ("Al", "Si", "Mn", "Fe", "Cu", "Zn"):
        assert main_line(el, "K")[0] == pytest.approx(eu._XRAY_LINES["K"][el][1], abs=2e-3)


def test_sources_cited():
    assert {"lines", "mu_rho", "edges", "fluor_yield"} <= set(SOURCES)


def test_edges_and_fluorescence_yield():
    assert edge_kev("Al", "K") == pytest.approx(1.5596, abs=1e-3)
    assert edge_kev("Fe", "K") == pytest.approx(7.112, abs=2e-3)
    assert edge_kev("Fe", "L") is not None and edge_kev("Fe", "L") < 1.0
    assert edge_kev("Fe", "M") is None or edge_kev("Fe", "M") < 0.1
    assert fluor_yield_k("Fe") == pytest.approx(0.3546, abs=1e-3)


def test_empty_when_family_absent():
    assert family_lines("Xx", "K") == []
    assert family_lines("Al", "M") == []


def test_mu_rho_mix_is_mass_weighted_sum():
    e = np.array([1.0, 5.0, 10.0])
    mix = mu_rho_mix({"Al": 0.25, "Fe": 0.75}, e)
    assert mix == pytest.approx(0.25 * mu_rho("Al", e) + 0.75 * mu_rho("Fe", e))


def test_mu_rho_is_monotone_between_edges_and_positive():
    e = np.geomspace(0.2, 20.0, 50)
    mu = mu_rho("Cu", e)
    assert np.all(mu > 0) and np.all(np.isfinite(mu))
    # well above the Cu K edge (8.98 keV) mu/rho falls with energy
    hi = mu_rho("Cu", np.array([10.0, 20.0]))
    assert hi[1] < hi[0]


def test_unknown_element_raises():
    with pytest.raises(KeyError):
        mu_rho("Xx", np.array([1.0]))


@pytest.mark.parametrize("el,nominal", [("N", 0.4016), ("O", 0.5320), ("F", 0.6854)])
def test_low_z_k_edge_step_is_in_the_table(el, nominal):
    # xraylib puts these steps 8-11 eV above the nominal edge energy.
    steps = [(lo, hi) for lo, hi in mu_steps(el) if nominal - 0.01 < lo < nominal + 0.03]
    assert len(steps) == 1
    lo, hi = steps[0]
    mu = mu_rho(el, np.array([lo - 2e-4, hi + 2e-4]))
    assert mu[1] / mu[0] > 3


def test_cr_l3_and_l2_are_separate_steps():
    steps = [lo for lo, _ in mu_steps("Cr") if 0.55 < lo < 0.61]
    assert len(steps) == 2
    assert steps[0] == pytest.approx(0.5741, abs=1e-3)  # L3
    assert steps[1] == pytest.approx(0.5838, abs=1e-3)  # L2
    # between the two steps mu/rho sits on the L3-above plateau, not on L2's
    mid = mu_rho("Cr", np.array([0.5790]))[0]
    below, above = mu_rho("Cr", np.array([steps[1] - 2e-4, steps[1] + 2e-4]))
    assert above / below > 1.2
    assert below == pytest.approx(mid, rel=0.15)


def test_steps_are_ordered_and_bracket_a_rise():
    for el in ("Al", "Fe", "Cr", "O"):
        steps = mu_steps(el)
        assert steps == sorted(steps)
        for lo, hi in steps:
            assert 0 < hi - lo < 1e-5


def test_mu_rho_matches_xraylib_away_from_steps(monkeypatch):
    """Dev-only: needs xraylib via ORIENTA_EXSPY_PATH, skipped otherwise."""
    raw = os.environ.get("ORIENTA_EXSPY_PATH")
    if not raw:
        pytest.skip("ORIENTA_EXSPY_PATH not set")
    for part in raw.split(os.pathsep):
        if part:
            monkeypatch.syspath_prepend(part)
    xraylib = pytest.importorskip("xraylib")

    energies = np.geomspace(0.15, 20.0, 4000)
    worst = (0.0, None, None)
    for z in range(5, 43):
        sym = xraylib.AtomicNumberToSymbol(z)
        steps = mu_steps(sym)
        near = np.zeros(energies.shape, dtype=bool)
        for lo, hi in steps:
            near |= (energies > lo - 2e-3) & (energies < hi + 2e-3)
        e = energies[~near]
        ref = np.array([xraylib.CS_Total(z, float(x)) for x in e])
        err = np.abs(mu_rho(sym, e) / ref - 1.0)
        i = int(np.argmax(err))
        if err[i] > worst[0]:
            worst = (float(err[i]), sym, float(e[i]))
    assert worst[0] < 0.02, worst
