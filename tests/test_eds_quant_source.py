"""Source-aware quantification: vendor window integrals vs spectrum-fit nets.

The window path (raw AZtec window integrals, background included) and the
spectrum path (net line intensities from ``backend/eds_spectrum``) need their
own instrument-response curves: each curve absorbs the pedestal of the kind of
input it was measured on. These tests pin that

* the window path is bit-for-bit what it was before the source existed,
* the spectrum path refuses to run without its own net-intensity standard
  (never falls back to the window curve),
* with a standard it reproduces that standard,
* negative per-pixel net intensities are clipped at use, and counted.
"""
from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

import eds_utils
from eds_utils import (
    QUANT_SOURCES,
    CountsByElement,
    UnknownLineError,
    build_counts_by_element,
    build_response_curve,
    counts_to_weight_pct,
    weight_pct_to_atomic_pct,
)

#: Fe wt% of ``{"Al Ka1": 1e5, "Fe Ka1": 1e4}`` from the window path, which
#: the source parameter must not change (value printed with repr()).
OLD_FE_WT = 27.08499585983596


@pytest.fixture
def fake_standard():
    """A self-consistent net-intensity standard (Cu enters on BOTH lines)."""
    return {
        "label": "fake net-intensity standard",
        "matrix": "synthetic",
        "beam_kv": 20.0,
        "n_px": 100,
        "detector": "synthetic",
        "measured_on": "2026-10-25",
        "counts": {
            "Al K": 90000.0, "Si K": 9000.0, "Fe K": 6000.0,
            "Cu K": 1500.0, "Cu L": 4000.0,
        },
        "reference_at_pct": {"Al": 70.0, "Si": 10.0, "Fe": 12.0, "Cu": 8.0},
        "excluded": {},
    }


#: What the fit would deliver for the standard region, K lines only.
FAKE_ALPHA_NETS = {
    "Al Kα1": np.array([90000.0]),
    "Si Kα1": np.array([9000.0]),
    "Fe Kα1": np.array([6000.0]),
    "Cu Kα1": np.array([1500.0]),
}


def _at(wt):
    return weight_pct_to_atomic_pct(wt)


def test_sources_constant():
    assert QUANT_SOURCES == ("window", "spectrum")


def test_window_path_unchanged():
    counts = build_counts_by_element(
        {"Al Kα1": np.array([1e5]), "Fe Kα1": np.array([1e4])})
    assert isinstance(counts, CountsByElement)
    assert counts.source == "window"
    wt = counts_to_weight_pct(counts)
    assert wt["Fe"][0] == pytest.approx(OLD_FE_WT, rel=1e-12)
    assert "clipped_negative" not in wt.provenance
    assert wt.provenance["source_data"].startswith("vendor Window Integral")


def test_plain_dict_is_a_window_input():
    q_dict = counts_to_weight_pct(
        {eds_utils.parse_element_name("Al Kα1"): np.array([1e5]),
         eds_utils.parse_element_name("Fe Kα1"): np.array([1e4])})
    assert q_dict["Fe"][0] == pytest.approx(OLD_FE_WT, rel=1e-12)


def test_window_curve_is_the_generic_builder_on_the_window_standard():
    e, r = build_response_curve(eds_utils._CALIBRATION_STANDARD)
    assert np.array_equal(e, eds_utils._RESPONSE_ENERGY_KEV)
    assert np.array_equal(r, eds_utils._RESPONSE_VALUE)
    # the old private entry point still works and agrees
    e2, r2 = eds_utils._build_response_curve()
    assert np.array_equal(e2, e) and np.array_equal(r2, r)


def test_window_standard_numbers_untouched():
    std = eds_utils._CALIBRATION_STANDARD
    assert std["counts"]["Cu"] == 24989.6
    assert set(std["counts"]) == {"Al", "Si", "Fe", "Mn", "Cu", "Zn"}


def test_spectrum_source_without_standard_fails_loud(monkeypatch):
    monkeypatch.setattr(eds_utils, "_CALIBRATION_STANDARD_SPECTRUM", None)
    with pytest.raises(UnknownLineError, match="spectrum") as ei:
        counts_to_weight_pct(
            build_counts_by_element({"Al Kα1": np.array([1.0])},
                                    source="spectrum"))
    assert "net-intensity calibration" in str(ei.value)


def test_spectrum_without_standard_never_uses_the_window_curve(monkeypatch):
    monkeypatch.setattr(eds_utils, "_CALIBRATION_STANDARD_SPECTRUM", None)
    with pytest.raises(UnknownLineError, match="spectrum"):
        eds_utils.k_factor("Fe", "K", 20.0, source="spectrum")
    with pytest.raises(UnknownLineError, match="spectrum"):
        eds_utils.instrument_response(6.4, source="spectrum")
    with pytest.raises(UnknownLineError, match="spectrum"):
        eds_utils.quantification_provenance(20.0, source="spectrum")
    # explicit override of source on a window-tagged dict also fails loud
    with pytest.raises(UnknownLineError, match="spectrum"):
        counts_to_weight_pct(
            build_counts_by_element({"Al Kα1": np.array([1.0])}),
            source="spectrum")


def test_spectrum_with_caller_k_still_needs_the_standard(monkeypatch):
    # the provenance names the calibration; a missing one is not papered over
    monkeypatch.setattr(eds_utils, "_CALIBRATION_STANDARD_SPECTRUM", None)
    c = build_counts_by_element({"Al Kα1": np.array([1.0])}, source="spectrum")
    with pytest.raises(UnknownLineError, match="spectrum"):
        counts_to_weight_pct(c, k_factors={next(iter(c)): 1.0})


def test_spectrum_source_uses_its_own_curve(monkeypatch, fake_standard):
    monkeypatch.setattr(eds_utils, "_CALIBRATION_STANDARD_SPECTRUM",
                        fake_standard)
    counts = build_counts_by_element(FAKE_ALPHA_NETS, source="spectrum")
    assert counts.source == "spectrum"
    q = counts_to_weight_pct(counts)
    at = _at(q)
    for el, ref in fake_standard["reference_at_pct"].items():
        assert at[el][0] == pytest.approx(ref, abs=1e-6)
    assert q.provenance["source_data"].startswith("spectrum fit")
    assert q.provenance["spectral_overlaps"] == []
    assert q.provenance["calibration"]["standard"] == "fake net-intensity standard"
    # and it is NOT what the window curve would have said
    window = counts_to_weight_pct(
        build_counts_by_element(FAKE_ALPHA_NETS, source="window"))
    assert not np.allclose(window["Fe"], q["Fe"], rtol=1e-3)


def test_cu_l_alone_is_priced_by_the_cu_l_anchor(monkeypatch, fake_standard):
    monkeypatch.setattr(eds_utils, "_CALIBRATION_STANDARD_SPECTRUM",
                        fake_standard)
    nets = dict(FAKE_ALPHA_NETS)
    del nets["Cu Kα1"]
    nets["Cu Lα1,2"] = np.array([4000.0])
    q = counts_to_weight_pct(build_counts_by_element(nets, source="spectrum"))
    at = _at(q)
    for el, ref in fake_standard["reference_at_pct"].items():
        assert at[el][0] == pytest.approx(ref, abs=1e-6)
    assert q.provenance["lines"]["Cu"]["line"] == "L"


def test_both_lines_of_cu_are_anchors(fake_standard):
    e, r = build_response_curve(fake_standard)
    assert len(e) == 5                      # Al K Si K Fe K Cu K Cu L
    assert np.all(np.diff(e) > 0)
    assert r.max() == pytest.approx(1.0)


def test_spectrum_overlap_check_is_skipped(monkeypatch, fake_standard):
    monkeypatch.setattr(eds_utils, "_CALIBRATION_STANDARD_SPECTRUM",
                        fake_standard)
    std = copy.deepcopy(fake_standard)
    std["counts"]["Zn L"] = 500.0
    std["reference_at_pct"] = {**fake_standard["reference_at_pct"], "Zn": 1.0}
    monkeypatch.setattr(eds_utils, "_CALIBRATION_STANDARD_SPECTRUM", std)
    q = counts_to_weight_pct(build_counts_by_element(
        {"Cu Lα1,2": np.array([4000.0]), "Zn Lα1,2": np.array([500.0]),
         "Al Kα1": np.array([9e4])}, source="spectrum"))
    assert q.provenance["spectral_overlaps"] == []
    assert not any("overlap" in w for w in q.provenance["warnings"])
    # the window path still reports it
    w = counts_to_weight_pct(build_counts_by_element(
        {"Cu Lα1,2": np.array([4000.0]), "Zn Lα1,2": np.array([500.0]),
         "Al Kα1": np.array([9e4])}))
    assert len(w.provenance["spectral_overlaps"]) == 1


def test_negative_nets_are_clipped_at_use_and_counted(monkeypatch,
                                                      fake_standard):
    monkeypatch.setattr(eds_utils, "_CALIBRATION_STANDARD_SPECTRUM",
                        fake_standard)
    nets = {
        "Al Kα1": np.array([9e4, 9e4, 9e4]),
        "Fe Kα1": np.array([6000.0, -50.0, -1.0]),
        "Si Kα1": np.array([9000.0, 9000.0, 9000.0]),
    }
    q = counts_to_weight_pct(build_counts_by_element(nets, source="spectrum"))
    assert q["Fe"][0] > 0
    assert q["Fe"][1] == 0.0 and q["Fe"][2] == 0.0
    assert all(np.all(v >= 0) for v in q.values())
    assert np.allclose(sum(q.values()), 100.0)
    assert q.provenance["clipped_negative"] == {"Al": 0, "Fe": 2, "Si": 0}
    # the caller's array is not modified
    assert nets["Fe Kα1"][1] == -50.0


def test_window_path_does_not_clip():
    c = build_counts_by_element(
        {"Al Kα1": np.array([1e5, 1e5]), "Fe Kα1": np.array([1e4, -1e3])})
    q = counts_to_weight_pct(c)
    assert q["Fe"][1] < 0


def test_explicit_source_overrides_the_attribute(monkeypatch, fake_standard):
    monkeypatch.setattr(eds_utils, "_CALIBRATION_STANDARD_SPECTRUM",
                        fake_standard)
    c = build_counts_by_element(FAKE_ALPHA_NETS)      # tagged window
    q = counts_to_weight_pct(c, source="spectrum")
    assert q.provenance["source_data"].startswith("spectrum fit")


def test_unknown_source_is_refused():
    with pytest.raises(ValueError, match="source"):
        build_counts_by_element({"Al Kα1": np.array([1.0])}, source="pdf")
    with pytest.raises(ValueError, match="source"):
        counts_to_weight_pct({"Al": np.array([1.0])}, source="pdf")


def test_cu_line_when_both_lines_offered_depends_on_the_source():
    # The window source keeps K (unchanged);
    # the spectrum source takes L (_SPECTRUM_LINE_PREFERENCE, Al7Cu2Fe bench).
    raw = {"Cu Kα1": np.array([1.0]), "Cu Lα1,2": np.array([2.0])}
    assert next(iter(build_counts_by_element(raw))).line == "K"
    c = build_counts_by_element(raw, source="spectrum")
    assert next(iter(c)).line == "L"
    assert c.source == "spectrum"


def test_source_survives_copy_and_pickle():
    c = build_counts_by_element({"Al Kα1": np.array([1.0])}, source="spectrum")
    assert c.copy().source == "spectrum"
    assert copy.copy(c).source == "spectrum"
    assert copy.deepcopy(c).source == "spectrum"
    assert pickle.loads(pickle.dumps(c)).source == "spectrum"


def test_k_factor_and_response_window_behaviour_unchanged():
    # identical numbers with and without the explicit default
    assert eds_utils.k_factor("Fe", "K", 20.0) == eds_utils.k_factor(
        "Fe", "K", 20.0, source="window")
    assert eds_utils.instrument_response(6.4) == eds_utils.instrument_response(
        6.4, source="window")


def test_spectrum_k_factor_is_relative_to_si_k(monkeypatch, fake_standard):
    monkeypatch.setattr(eds_utils, "_CALIBRATION_STANDARD_SPECTRUM",
                        fake_standard)
    # at the standard's own anchors the response equals the curve value
    e, r = build_response_curve(fake_standard)
    fe_kev = eds_utils._XRAY_LINES["K"]["Fe"][1]
    idx = int(np.argmin(np.abs(e - fe_kev)))
    assert eds_utils.instrument_response(fe_kev, source="spectrum") == pytest.approx(
        r[idx], rel=1e-12)
    assert eds_utils.k_factor("Si", "K", 20.0, source="spectrum") > 0


def test_spectrum_standard_cache_follows_monkeypatch(monkeypatch,
                                                     fake_standard):
    monkeypatch.setattr(eds_utils, "_CALIBRATION_STANDARD_SPECTRUM",
                        fake_standard)
    a = eds_utils.instrument_response(1.5, source="spectrum")
    other = copy.deepcopy(fake_standard)
    other["counts"]["Al K"] = 45000.0       # halves Al's measured k -> new curve
    monkeypatch.setattr(eds_utils, "_CALIBRATION_STANDARD_SPECTRUM", other)
    b = eds_utils.instrument_response(1.5, source="spectrum")
    assert a != b


# --- line choice per source -------------------------------------------------
def _chosen_line(raw, source):
    counts = build_counts_by_element(raw, source=source)
    (key,) = [k for k in counts if str(k) == "Cu"]
    return key.line, float(np.ravel(counts[key])[0])


def test_spectrum_source_quantifies_cu_through_l_when_both_are_offered():
    raw = {"Cu Kα1": np.array([111.0]), "Cu Lα1,2": np.array([999.0])}
    assert _chosen_line(raw, "spectrum") == ("L", 999.0)
    # order of offering does not matter
    assert _chosen_line(dict(reversed(list(raw.items()))), "spectrum") == ("L", 999.0)


def test_window_source_still_prefers_k_for_cu():
    raw = {"Cu Kα1": np.array([111.0]), "Cu Lα1,2": np.array([999.0])}
    assert _chosen_line(raw, "window") == ("K", 111.0)
    assert eds_utils._SPECTRUM_LINE_PREFERENCE == {"Cu": "L"}


def test_spectrum_source_with_only_cu_k_offered_uses_it():
    assert _chosen_line({"Cu Kα1": np.array([111.0])}, "spectrum") == ("K", 111.0)


def test_spectrum_source_keeps_the_overvoltage_default_for_other_elements():
    raw = {"Zn Kα1": np.array([5.0]), "Zn Lα1,2": np.array([50.0]),
           "Fe Kα1": np.array([7.0]), "Fe Lα1,2": np.array([70.0])}
    counts = build_counts_by_element(raw, source="spectrum")
    assert {str(k): k.line for k in counts} == {"Zn": "K", "Fe": "K"}
