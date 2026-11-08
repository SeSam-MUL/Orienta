"""phi(rho z) matrix correction of the EDS spectrum source (backend/eds_spectrum/matrix.py).

The EPQ numbers below were produced by NIST EPQ (public domain) XPP1991 / XPP1989Ext through
JPype on 2026-11-06 (a development oracle, not shipped); they pin our transcription.
"""
import copy
import math
import pickle

import numpy as np
import pytest

import eds_utils as E
from backend.eds_spectrum import matrix as M
from backend.eds_spectrum.model import SpectrumMeta

ALPHA = {"Al": 0.55, "Si": 0.08, "Fe": 0.24, "Mn": 0.06, "Cu": 0.07}
DEPTHS = (0.0, 2e-4, 5e-4, 1e-3, 2e-3)          # g/cm^2
# EPQ XPP1991 (normal incidence), computeCurve at DEPTHS
EPQ_NORMAL = [
    ({"Al": 1.0}, "Al", "K", [1.333398, 2.632983, 1.318055, 0.126199, 0.000358]),
    ({"Cu": 1.0}, "Cu", "L", [1.789897, 3.303299, 1.520077, 0.140017, 0.000412]),
    (ALPHA, "Fe", "K", [1.384045, 1.684344, 0.636095, 0.049735, 0.000125]),
    (ALPHA, "Al", "K", [1.473002, 2.791783, 1.331797, 0.123079, 0.000341]),
    ({"Al": 0.4, "Cu": 0.45, "Fe": 0.15}, "Cu", "K", [1.412674, 1.469008, 0.504502, 0.03851, 0.00011]),
]
# EPQ XPP1989Ext at 70 deg tilt: phi(0) (the deeper values are ~0 there)
EPQ_TILT70_PHI0 = [
    ({"Al": 1.0}, "Al", "K", 1.53125),
    ({"Cu": 1.0}, "Cu", "L", 1.671875),
    (ALPHA, "Fe", "K", 1.453125),
    (ALPHA, "Al", "K", 1.5625),
    ({"Al": 0.4, "Cu": 0.45, "Fe": 0.15}, "Cu", "K", 1.433594),
]


@pytest.mark.parametrize("comp,el,fam,ref", EPQ_NORMAL)
def test_xpp_normal_incidence_matches_epq(comp, el, fam, ref):
    p = M.phi_rho_z(comp, el, fam, 20.0, 0.0)
    got = p.curve(np.array(DEPTHS))
    # Cu L edge 0.9327 keV here vs 0.9311 in EPQ: the only source of the residual
    np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-6)


@pytest.mark.parametrize("comp,el,fam,ref", EPQ_TILT70_PHI0)
def test_xpp_tilt_extension_matches_epq(comp, el, fam, ref):
    p = M.phi_rho_z(comp, el, fam, 20.0, math.radians(70.0), tilt_model="xpp_ext")
    # EPQ rounds through a huge A/B cancellation at 70 deg; 0.2 % is its own precision there
    assert float(p.curve(np.array([0.0]))[0]) == pytest.approx(ref, rel=2e-3)


@pytest.mark.parametrize("tilt", [0.0, 40.0, 70.0])
@pytest.mark.parametrize("model", M.TILT_MODELS)
def test_emitted_without_absorption_is_the_area(tilt, model):
    p = M.phi_rho_z(ALPHA, "Si", "K", 20.0, math.radians(tilt), tilt_model=model)
    rz = np.linspace(0.0, 5e-3, 200001)
    num = np.trapezoid(p.curve(rz), rz)
    assert float(p.emitted(0.0)[0]) == pytest.approx(float(p.area[0]), rel=1e-9)
    assert num == pytest.approx(float(p.area[0]), rel=1e-3)


def test_emitted_matches_numerical_absorption_integral():
    p = M.phi_rho_z(ALPHA, "Al", "K", 20.0, math.radians(70.0))
    chi = 4000.0
    rz = np.linspace(0.0, 5e-3, 400001)
    num = np.trapezoid(p.curve(rz) * np.exp(-chi * rz), rz)
    assert float(p.emitted(chi)[0]) == pytest.approx(num, rel=1e-3)


def test_scaled_tilt_compresses_the_normal_curve_by_sqrt_cos_tilt():
    tilt = math.radians(70.0)
    p0 = M.phi_rho_z({"Al": 1.0}, "Al", "K", 20.0, 0.0)
    pt = M.phi_rho_z({"Al": 1.0}, "Al", "K", 20.0, tilt)
    ext = M.phi_rho_z({"Al": 1.0}, "Al", "K", 20.0, tilt, tilt_model="xpp_ext")
    rz = np.linspace(0, 3e-3, 30001)
    peak0 = rz[np.argmax(p0.curve(rz))]
    peakt = rz[np.argmax(pt.curve(rz))]
    assert peakt == pytest.approx(math.cos(tilt) ** M.DEPTH_SCALE_EXPONENT * peak0, rel=2e-3)
    assert float(pt.area[0]) == pytest.approx(float(ext.area[0]), rel=1e-12)


def test_vectorised_rows_equal_scalar_calls():
    comps = [ALPHA, {"Al": 0.9, "Si": 0.1, "Fe": 0.0, "Mn": 0.0, "Cu": 0.0},
             {"Al": 0.4, "Si": 0.0, "Fe": 0.15, "Mn": 0.0, "Cu": 0.45}]
    basis = M.Basis(list(ALPHA))
    C = np.array([[c[e] for e in basis.elements] for c in comps])
    geo = M.Geometry.from_degrees(20.0, 30.0, 70.0)
    rows = M.emitted_rows(basis, C, "Al", "K", geo)
    for i, c in enumerate(comps):
        assert rows[i] == pytest.approx(M.emitted_per_mass(c, "Al", "K", geo), rel=1e-12)


@pytest.mark.parametrize("psi", [9.54, 30.0, 81.0])
def test_quantify_inverts_the_forward_model(psi):
    geo = M.Geometry.from_degrees(20.0, psi, 70.0)
    resp = M.stored_response()
    lines = [("Al", "K"), ("Si", "K"), ("Mn", "K"), ("Fe", "K"), ("Cu", "K"), ("Zn", "K")]
    truth = M.atomic_to_mass({"Al": 66.0, "Si": 10.0, "Mn": 3.7, "Fe": 14.0, "Cu": 5.7, "Zn": 0.6})
    r = {l: resp(*l) for l in lines}
    inten = M.predicted_intensities(truth, lines, geo, r)
    got, it = M.quantify(inten, geo, r)
    assert it < 50
    for el, c in truth.items():
        assert got[el] == pytest.approx(c, abs=1e-6)


def test_quantify_map_handles_empty_and_negative_pixels():
    geo = M.Geometry.from_degrees(20.0, 30.0, 70.0)
    r = {("Al", "K"): 1.0, ("Si", "K"): 1.0}
    comp, _ = M.quantify_map({"Al": ("K", np.array([100.0, 0.0, -5.0])),
                              "Si": ("K", np.array([10.0, 0.0, 3.0]))}, geo, r)
    assert comp["Al"][1] == 0.0 and comp["Si"][1] == 0.0          # nothing measured
    assert comp["Al"][2] == 0.0 and comp["Si"][2] == pytest.approx(1.0)
    assert comp["Al"][0] + comp["Si"][0] == pytest.approx(1.0)


def test_fluorescence_only_for_k_secondaries():
    # Si K fluoresces Al K; nothing is added to an L secondary (Cu L by Al K)
    assert M.fluorescence_factor({"Al": 0.5, "Si": 0.5}, "Al", "K", 20.0, math.radians(30)) > 1.01
    assert M.fluorescence_factor({"Al": 0.9, "Cu": 0.1}, "Cu", "L", 20.0, math.radians(30)) == 1.0


def test_take_off_angle_readings():
    assert math.degrees(M.take_off_angle(29.0, 0.0, 70.0, "axis")) == pytest.approx(9.545, abs=1e-3)
    assert math.degrees(M.take_off_angle(29.0, 0.0, 70.0, "toward")) == pytest.approx(81.0, abs=1e-3)
    assert math.degrees(M.take_off_angle(35.0, 0.0, 0.0, "axis")) == pytest.approx(35.0, abs=1e-9)


def test_geometry_from_meta_needs_tilt_and_elevation():
    base = dict(start_kev=0.0, width_kev=0.01, n_channels=2048, beam_kv=20.0)
    assert M.geometry_from_meta(SpectrumMeta(**base)) is None
    assert M.geometry_from_meta(SpectrumMeta(**base, tilt_deg=70.0)) is None
    g = M.geometry_from_meta(SpectrumMeta(**base, tilt_deg=70.0, elevation_deg=29.0,
                                          azimuth_deg=0.0, tilt_axis_deg=0.0))
    assert g is not None and g.tilt_deg == pytest.approx(70.0)


def test_stored_calibration_is_reproducible_from_its_references():
    cal = M.stored_calibration()
    refs = []
    for r in cal["references"]:
        h = r["header"]
        psi = M.take_off_angle(h["elevation_deg"], h["azimuth_deg"], h["tilt_deg"],
                               cal["geometry_convention"], h["tilt_axis_deg"])
        refs.append(dict(nets=r["nets"], ref_at_pct=r["ref_at_pct"],
                         geometry=M.Geometry(h["beam_kv"], psi, math.radians(h["tilt_deg"]))))
    anchors = [tuple(a.split()) for a in cal["anchors"]]
    resp, chi2, dof = M.fit_response(refs, anchors, cal["prior_sigma"])
    stored = M.stored_response()
    for f, v in stored.family_log.items():
        assert resp.family_log[f] == pytest.approx(v, abs=1e-9)
    for l, v in stored.delta.items():
        assert resp.delta[l] == pytest.approx(v, abs=1e-9)
    assert chi2 == pytest.approx(cal["chi2"], rel=1e-9)


# ---------------------------------------------------------------- eds_utils plumbing
GEO = M.Geometry.from_degrees(20.0, 30.0, 70.0)
NETS = {"Al Kα1": 186625.7, "Si Kα1": 22689.6, "Mn Kα1": 10040.3, "Fe Kα1": 19037.1,
        "Cu Kα1": 1768.3, "Cu Lα1,2": 4336.2, "Zn Kα1": 328.5, "C Kα1,2": 1112.0}


def _counts(geometry):
    return E.build_counts_by_element(((k, np.array([v])) for k, v in NETS.items()), 20.0,
                                     "spectrum", geometry=geometry)


def test_counts_keep_their_geometry_through_copy_and_pickle():
    c = _counts(GEO)
    assert c.copy().geometry == GEO
    assert copy.copy(c).geometry == GEO
    assert pickle.loads(pickle.dumps(c)).geometry == GEO
    assert pickle.loads(pickle.dumps(c)).source == "spectrum"


def test_geometry_only_on_the_spectrum_source():
    with pytest.raises(ValueError):
        E.CountsByElement(source="window", geometry=GEO)


def test_corrected_path_takes_cu_k_uncorrected_cu_l():
    corrected = {str(k): k.line for k in _counts(GEO)}
    uncorrected = {str(k): k.line for k in _counts(None)}
    assert corrected["Cu"] == "K"
    assert uncorrected["Cu"] == "L"


def test_corrected_path_names_its_method_and_leaves_carbon_out():
    wt = E.counts_to_weight_pct(_counts(GEO))
    prov = wt.provenance
    assert "phi(rho z)" in prov["summary"]
    assert prov["matrix_correction"]["exit_angle_deg"] == pytest.approx(30.0)
    assert "C" not in {str(k) for k in wt}
    assert any(x["element"] == "C" for x in prov["excluded_windows"])
    assert sum(float(v[0]) for v in wt.values()) == pytest.approx(100.0)


def test_k_factors_refused_with_the_matrix_correction():
    with pytest.raises(ValueError):
        E.counts_to_weight_pct(_counts(GEO), k_factors={"Al": 1.0})


def test_uncorrected_spectrum_path_unchanged_without_geometry():
    wt = E.counts_to_weight_pct(_counts(None))
    assert "NO ZAF" in wt.provenance["summary"]
    assert "C" in {str(k) for k in wt}


def test_large_maps_are_cached_and_handed_out_as_copies(monkeypatch):
    monkeypatch.setattr(E, "_MATRIX_CACHE", None)
    n = E._MATRIX_CACHE_MIN_VALUES
    rng = np.random.default_rng(1)
    raw = [("Al Kα1", rng.uniform(1e5, 2e5, n)), ("Si Kα1", rng.uniform(1e3, 3e4, n))]
    counts = E.build_counts_by_element(raw, 20.0, "spectrum", geometry=GEO)
    calls = []
    real = E._matrix_corrected_weight_pct_uncached
    monkeypatch.setattr(E, "_matrix_corrected_weight_pct_uncached",
                        lambda *a: calls.append(1) or real(*a))
    first = E.counts_to_weight_pct(counts)
    first["Al"][:] = -1.0                                   # a caller scribbles on its copy
    second = E.counts_to_weight_pct(counts)
    assert len(calls) == 1
    assert float(second["Al"].min()) > 0.0
    other = E.build_counts_by_element([(k, v * 1.01) for k, v in raw], 20.0, "spectrum",
                                      geometry=GEO)
    E.counts_to_weight_pct(other)
    assert len(calls) == 2                                  # different bytes, new result


# ---------------------------------------------------------------- fluorescence against NIST EPQ
# F - 1 from NIST EPQ Fluorescence.Reed at 20 kV, exit angle 81 deg, mass fractions;
# computed through JPype during development (largest deviation of ours 9.1 %).
EPQ_FLUOR = [
    ({"Al": 0.5, "Si": 0.5}, "Al", 0.02105),
    (ALPHA, "Al", 0.00254),
    (ALPHA, "Mn", 0.01916),
    (ALPHA, "Fe", 0.01278),
    ({"Al": 0.4, "Cu": 0.45, "Fe": 0.15}, "Fe", 0.10307),
    ({"Al": 0.4, "Cu": 0.45, "Fe": 0.15}, "Al", 0.00013),
    ({"Fe": 0.7, "Cr": 0.2, "Ni": 0.1}, "Cr", 0.17824),
    ({"Al": 0.9, "Zr": 0.1}, "Al", 0.00157),
]


@pytest.mark.parametrize("comp,el,ref", EPQ_FLUOR)
def test_fluorescence_matches_epq_reed(comp, el, ref):
    got = M.fluorescence_factor(comp, el, "K", 20.0, math.radians(81.0)) - 1.0
    assert got == pytest.approx(ref, rel=0.12, abs=2e-5)


def _meta(**kw):
    return SpectrumMeta(start_kev=0.0, width_kev=0.01, n_channels=2048, beam_kv=20.0, **kw)


def test_geometry_from_meta_reads_the_sampleb_header_as_81_deg():
    g = M.geometry_from_meta(_meta(tilt_deg=70.0, elevation_deg=29.0, azimuth_deg=0.0,
                                   tilt_axis_deg=0.0))
    assert g.psi_deg == pytest.approx(81.0, abs=0.01)
    assert g.note == ""


# -10 deg: the old shadow check (exit angle <= 1 deg) would accept it (19 deg)
@pytest.mark.parametrize("tilt", [91.0, -70.0, -10.0, 89.9])
def test_geometry_from_meta_refuses_an_implausible_tilt(tilt):
    assert M.geometry_from_meta(_meta(tilt_deg=tilt, elevation_deg=29.0)) is None


def test_unverified_azimuth_reaches_the_warnings():
    g = M.geometry_from_meta(_meta(tilt_deg=70.0, elevation_deg=29.0, azimuth_deg=45.0))
    assert g is not None and "azimuth 45" in g.note
    wt = E.counts_to_weight_pct(_counts(g))
    assert g.note in wt.provenance["warnings"]


@pytest.mark.parametrize("kw", [dict(beam_kv=20.0, psi_rad=0.5, tilt_rad=math.radians(95)),
                                dict(beam_kv=20.0, psi_rad=0.0, tilt_rad=1.0),
                                dict(beam_kv=0.0, psi_rad=0.5, tilt_rad=1.0)])
def test_geometry_validates_its_values(kw):
    with pytest.raises(ValueError):
        M.Geometry(**kw)


def test_l_line_alone_goes_through_the_corrected_path():
    raw = [("Al Kα1", np.array([186625.7])), ("Cu Lα1,2", np.array([4336.2]))]
    wt = E.counts_to_weight_pct(E.build_counts_by_element(raw, 20.0, "spectrum", geometry=GEO))
    assert wt.provenance["lines"]["Cu"]["line"] == "L"
    assert 0.0 < float(wt["Cu"][0]) < 100.0
    assert sum(float(v[0]) for v in wt.values()) == pytest.approx(100.0)


def test_quantify_map_reports_non_convergence_and_non_finite_values():
    geo = M.Geometry.from_degrees(20.0, 30.0, 70.0)
    r = {("Al", "K"): 1.0, ("Fe", "K"): 1.0}
    stats = {}
    comp, it = M.quantify_map({"Al": ("K", np.array([100.0, np.nan])),
                               "Fe": ("K", np.array([30.0, 5.0]))}, geo, r,
                              max_iter=1, stats=stats)
    assert stats["converged"] is False and it == 1
    assert stats["non_finite"] == 1
    assert np.all(np.isfinite(comp["Al"]))


def test_non_convergence_and_unanchored_lines_are_warned(monkeypatch):
    real = M.quantify_map
    monkeypatch.setattr(M, "quantify_map", lambda *a, **k: real(*a, **{**k, "max_iter": 1}))
    raw = [("Al Kα1", np.array([186625.7])), ("Mg Kα1", np.array([5000.0])),
           ("Fe Kα1", np.array([19037.1]))]
    wt = E.counts_to_weight_pct(E.build_counts_by_element(raw, 20.0, "spectrum", geometry=GEO))
    warnings = " | ".join(wt.provenance["warnings"])
    assert "did not converge" in warnings
    assert wt.provenance["matrix_correction"]["converged"] is False
    assert "Mg K: no reference measurement" in warnings


def test_spectrum_path_without_geometry_says_why_it_is_uncorrected():
    wt = E.counts_to_weight_pct(_counts(None))
    assert any("no usable acquisition geometry" in w for w in wt.provenance["warnings"])


def test_pixel_composition_keeps_the_matrix_correction():
    raw = [(k, np.array([v, v * 0.5])) for k, v in NETS.items()]
    counts = E.build_counts_by_element(raw, 20.0, "spectrum", geometry=GEO)
    px = E.get_pixel_composition(counts, 0, 0, 2)
    assert px.provenance["matrix_correction"] is not None
    assert "C" not in px.weight_pct and "C" in px.counts


def test_header_geometry_failures_are_not_remembered(monkeypatch, tmp_path):
    from backend.api.services import eds_spectrum_service as S
    f = tmp_path / "x.h5oina"
    f.write_bytes(b"0")
    calls = []

    def probe(path):
        calls.append(path)
        raise OSError("share away")

    monkeypatch.setattr(S, "_probe", probe)
    S._header_geometry_cached.cache_clear()
    assert S._header_geometry(str(f)) is None
    assert S._header_geometry(str(f)) is None
    assert len(calls) == 2


def test_quantify_map_refuses_a_line_below_the_overvoltage_gate():
    geo = M.Geometry.from_degrees(10.0, 30.0, 70.0)            # Cu K edge 8.98 keV: U 1.11
    with pytest.raises(ValueError, match="overvoltage"):
        M.quantify_map({"Al": ("K", np.array([100.0])), "Cu": ("K", np.array([5.0]))},
                       geo, {("Al", "K"): 1.0, ("Cu", "K"): 1.0})


def test_overvoltage_gates_of_matrix_and_eds_utils_agree():
    assert M.MIN_OVERVOLTAGE == E.MIN_OVERVOLTAGE


def test_geometry_convention_comes_from_the_calibration(monkeypatch):
    cal = dict(M.stored_calibration(), geometry_convention="axis")
    monkeypatch.setattr(M, "stored_calibration", lambda: cal)
    g = M.geometry_from_meta(_meta(tilt_deg=70.0, elevation_deg=29.0))
    assert g.psi_deg == pytest.approx(9.545, abs=1e-3)


def test_tilt_axis_alone_gives_a_note():
    g = M.geometry_from_meta(_meta(tilt_deg=70.0, elevation_deg=29.0, tilt_axis_deg=90.0))
    assert g is not None and "tilt axis 90" in g.note


def test_soft_line_gets_one_warning_not_two():
    raw = [("Al Kα1", np.array([186625.7])), ("O Kα1", np.array([900.0])),
           ("Fe Kα1", np.array([19037.1]))]
    wt = E.counts_to_weight_pct(E.build_counts_by_element(raw, 20.0, "spectrum", geometry=GEO))
    o = [w for w in wt.provenance["warnings"] if w.startswith("O K")]
    assert len(o) == 1 and "qualitative" in o[0]


def test_cache_key_tells_sub_lines_apart():
    n = E._MATRIX_CACHE_MIN_VALUES
    arr = np.full(n, 1000.0)
    a = E.CountsByElement({E.ElementLine("Fe", "K", "alpha1"): arr}, source="spectrum", geometry=GEO)
    b = E.CountsByElement({E.ElementLine("Fe", "K", "beta1"): arr}, source="spectrum", geometry=GEO)
    assert E._matrix_cache_key(a, GEO, 20.0) != E._matrix_cache_key(b, GEO, 20.0)


def test_header_geometry_is_read_again_when_the_file_changes(monkeypatch, tmp_path):
    import os
    from backend.api.services import eds_spectrum_service as S
    f = tmp_path / "x.h5oina"
    f.write_bytes(b"0")
    calls = []
    monkeypatch.setattr(S, "_probe", lambda path: calls.append(path) or (False, None))
    S._header_geometry_cached.cache_clear()
    S._header_geometry(str(f))
    S._header_geometry(str(f))
    assert len(calls) == 1                                  # same file: cached
    st = os.stat(f)
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    S._header_geometry(str(f))
    assert len(calls) == 2                                  # touched: read again
