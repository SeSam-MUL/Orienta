"""API tests for crystal_hint route.

Uses TestClient + minimal mocks (no real EBSD file required for most tests).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.services import user_config_manager as uc

# Needs the maintainer's crystal library / measurement data, which a clone
# does not have — skip with the missing path named, never fail.
from tests.data_deps import CIF_LIBRARY, requires


@pytest.fixture(autouse=True)
def isolate_user_config(tmp_path, monkeypatch):
    """Each test gets its own per-machine config — prevents picking up the
    developer's real Materials Project API key from %APPDATA% which would
    make "network failure" / "no external matches" tests flaky."""
    uc.set_config_path_for_test(tmp_path / "config.json")
    monkeypatch.setattr(uc, "_legacy_server_path", lambda: tmp_path / "absent1.json")
    monkeypatch.setattr(uc, "_legacy_manual_paths_path", lambda: tmp_path / "absent2.json")
    yield
    uc.set_config_path_for_test(None)
    uc.reset_cache_for_test()


def _client():
    from backend.api.main import app
    return TestClient(app)


def test_list_presets_returns_8():
    client = _client()
    r = client.get("/api/crystal-hint/presets")
    assert r.status_code == 200
    data = r.json()
    assert "AA226" in data
    assert "AA6061" in data
    assert "Custom" in data
    assert len(data) >= 8
    # Schema check
    aa226 = data["AA226"]
    assert "label" in aa226
    assert "elements" in aa226
    assert "Al" in aa226["elements"]
    assert "Si" in aa226["elements"]
    assert "Cu" in aa226["elements"]
    assert any("Mg2Si" in p["name"] for p in aa226["expected_phases"])


def test_get_one_preset_known():
    client = _client()
    r = client.get("/api/crystal-hint/presets/AA226")
    assert r.status_code == 200
    data = r.json()
    assert data["key"] == "AA226"
    assert "Al" in data["elements"]


def test_get_one_preset_unknown_returns_404():
    client = _client()
    r = client.get("/api/crystal-hint/presets/DoesNotExist")
    assert r.status_code == 404


@requires(CIF_LIBRARY)
def test_library_endpoint_returns_22_entries():
    client = _client()
    r = client.get("/api/crystal-hint/library")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) >= 20

    # At least one entry with full metadata
    al = next((e for e in data if e["key"] == "Al"), None)
    assert al is not None
    assert al["crystal_system"] == "cubic"
    assert al["has_sht"] is True
    assert al["lattice_a_A"] is not None
    assert 3.9 < al["lattice_a_A"] < 4.2


@requires(CIF_LIBRARY)
def test_library_includes_alpha_alfesi_cubic():
    """Verifies the cubic alpha-Al(Fe,Mn)Si phases are correctly recognized
    as cubic in the library endpoint (key bug from FINDINGS_v2 investigation)."""
    # Without simulated masters on disk the has_sht comparison below finds
    # nothing on either side and passes while verifying nothing. The SHT
    # library is gitignored, so that is a fresh clone. Skip out loud instead
    # — same rule as the data guards merged in d368bee9.
    sht_dir = Path(__file__).resolve().parents[1] / "Database" / "EBSD_SHT_Database"
    if not sht_dir.is_dir() or not any(sht_dir.rglob("*.sht")):
        pytest.skip(f"needs simulated masters in {sht_dir.name} — "
                    "run the Simulation tab, or this proves nothing")
    client = _client()
    r = client.get("/api/crystal-hint/library")
    data = r.json()
    for sd_key in ("sd_0302719", "sd_1401510"):
        entry = next((e for e in data if e["key"] == sd_key), None)
        assert entry is not None, f"{sd_key} missing from library"
        assert entry["crystal_system"] == "cubic"
        # has_sht must mirror the disk, not this machine's simulation queue:
        # the SHT library is gitignored and rebuilt by the Simulation tab, and
        # sd_1401510's master is not in this working copy (its folder holds
        # only the provenance sidecar). Both directions are asserted, so a
        # lost or mis-matched file is still caught.
        sht_dir = Path(__file__).resolve().parents[1] / "Database" / "EBSD_SHT_Database"
        on_disk = any(sd_key in p.stem for p in sht_dir.rglob("*.sht")) \
            if sht_dir.is_dir() else False
        assert entry["has_sht"] is on_disk, (
            f"{sd_key}: endpoint says has_sht={entry['has_sht']}, disk says "
            f"{on_disk}")


def test_analyze_pixel_without_open_file_returns_400():
    """Should fail loud with 400 if no EBSD file is open."""
    client = _client()
    with patch("backend.api.services.h5_session.is_open", return_value=False):
        r = client.post(
            "/api/crystal-hint/analyze-pixel",
            json={"pixel_index": 0, "elements": ["Al"]},
        )
    assert r.status_code == 400
    assert "no ebsd" in r.json()["detail"].lower()


def test_analyze_pixel_with_mocked_pattern_returns_report():
    """End-to-end with a mocked pattern: should return symmetry + lattice + candidates."""
    rng = np.random.RandomState(0)
    # Create a pattern with clear 4-fold symmetry (similar to test_crystal_hint_symmetry)
    size = 118
    img = rng.rand(size, size).astype(np.float32) * 0.05
    cy, cx = (size - 1) / 2, (size - 1) / 2
    y, x = np.indices((size, size))
    yy = y - cy; xx = x - cx
    r2 = np.sqrt(yy * yy + xx * xx)
    theta = np.arctan2(yy, xx)
    radial_falloff = np.exp(-r2 / (size * 0.20))
    band_intensity = np.zeros_like(img)
    for k in range(4):  # 4-fold
        angle = 2 * np.pi * k / 4
        d_angle = np.abs(np.angle(np.exp(1j * (theta - angle))))
        band_intensity += np.exp(-(d_angle / (np.pi / 12)) ** 2)
    img += (band_intensity * radial_falloff) * 2.0
    img += np.exp(-r2 ** 2 / (2 * (size * 0.025) ** 2)) * 8.0

    client = _client()
    with patch("backend.api.services.h5_session.is_open", return_value=True), \
         patch("backend.api.services.h5_session.get_cached_pattern", return_value=img):
        r = client.post(
            "/api/crystal-hint/analyze-pixel",
            json={
                "pixel_index": 42,
                "elements": ["Al", "Si", "Cu", "Fe", "Mg"],
                "preset_key": "AA226",
            },
        )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["pixel_index"] == 42
    assert "symmetry" in data
    assert "lattice" in data
    assert "local_matches" in data
    # Symmetry should detect 4-fold or 2-fold
    sym = data["symmetry"]
    assert sym["detected_n_fold"] in (2, 4) or sym["confidence"] == "none"
    # Local matches should be filtered to Al-alloy chemistry (no Ni)
    matches = data["local_matches"]
    assert all("Ni" not in m["elements"] for m in matches), "Ni should be filtered out"


def test_analyze_pixel_with_avg_radius_works_end_to_end():
    """avg_radius=1 should trigger 3×3 neighborhood averaging and return a
    SymmetryReport with method='spherical_avgN'."""
    rng = np.random.RandomState(0)
    size = 80
    img = rng.rand(size, size).astype(np.float32)

    fake_extractor = MagicMock()
    fake_extractor.get_metadata = MagicMock(return_value={
        "n_rows": 8, "n_cols": 8,
        "pattern_width": size, "pattern_height": size,
        "voltage_kv": 20.0,
        "pattern_center": {"x": 0.5, "y": 0.5, "z": 0.6},
    })

    # Mock signal + detector so _gather_detector_geom returns a valid dict
    fake_det = MagicMock()
    fake_det.pc = np.array([0.5, 0.5, 0.6])
    fake_signal = MagicMock()
    fake_signal.axes_manager.signal_shape = (size, size)
    fake_signal.detector = fake_det

    client = _client()
    with patch("backend.api.services.h5_session.is_open", return_value=True), \
         patch("backend.api.services.h5_session.get_extractor", return_value=fake_extractor), \
         patch("backend.api.services.h5_session.get_cached_pattern", return_value=img), \
         patch("backend.api.routes.ebsd_viewer._get_active_signal", return_value=fake_signal), \
         patch("backend.api.routes.ebsd_viewer._active_dataset", "test-dataset"), \
         patch("backend.api.routes.ebsd_viewer._ebsd_file_path", None), \
         patch("backend.api.services.calibration_store.calibration_store.get_detector",
               return_value=fake_det):
        r = client.post(
            "/api/crystal-hint/analyze-pixel",
            json={
                "pixel_index": 27,           # middle of an 8×8 grid
                "elements": ["Al"],
                "avg_radius": 1,
            },
        )
    assert r.status_code == 200, r.text
    data = r.json()
    sym = data["symmetry"]
    # Method should be spherical_avgN where N is up to 9 (3×3 neighborhood)
    assert sym["method"].startswith("spherical"), f"got method={sym['method']!r}"
    assert sym["n_patterns_averaged"] >= 1


def test_analyze_pixel_avg_radius_0_uses_single_pattern():
    """avg_radius=0 keeps the legacy single-pattern behavior."""
    rng = np.random.RandomState(0)
    size = 80
    img = rng.rand(size, size).astype(np.float32)
    fake_det = MagicMock()
    fake_det.pc = np.array([0.5, 0.5, 0.6])
    fake_signal = MagicMock()
    fake_signal.axes_manager.signal_shape = (size, size)
    fake_signal.detector = fake_det
    client = _client()
    with patch("backend.api.services.h5_session.is_open", return_value=True), \
         patch("backend.api.services.h5_session.get_cached_pattern", return_value=img), \
         patch("backend.api.routes.ebsd_viewer._get_active_signal", return_value=fake_signal), \
         patch("backend.api.routes.ebsd_viewer._active_dataset", "test-dataset"), \
         patch("backend.api.routes.ebsd_viewer._ebsd_file_path", None), \
         patch("backend.api.services.calibration_store.calibration_store.get_detector",
               return_value=fake_det):
        r = client.post(
            "/api/crystal-hint/analyze-pixel",
            json={"pixel_index": 0, "avg_radius": 0},
        )
    assert r.status_code == 200, r.text
    sym = r.json()["symmetry"]
    assert sym["method"] == "spherical"
    assert sym["n_patterns_averaged"] == 1


def test_analyze_pixel_method_field_is_image_when_no_geom():
    """Without detector geometry, falls back to Method A (image-space)."""
    rng = np.random.RandomState(0)
    size = 80
    img = rng.rand(size, size).astype(np.float32)
    client = _client()
    with patch("backend.api.services.h5_session.is_open", return_value=True), \
         patch("backend.api.services.h5_session.get_cached_pattern", return_value=img), \
         patch("backend.api.routes.ebsd_viewer._get_active_signal", return_value=None):
        r = client.post(
            "/api/crystal-hint/analyze-pixel",
            json={"pixel_index": 0},
        )
    assert r.status_code == 200, r.text
    sym = r.json()["symmetry"]
    assert sym["method"] == "image"


def test_analyze_pixel_bad_index_returns_404():
    client = _client()
    with patch("backend.api.services.h5_session.is_open", return_value=True), \
         patch("backend.api.services.h5_session.get_cached_pattern", return_value=None):
        r = client.post(
            "/api/crystal-hint/analyze-pixel",
            json={"pixel_index": 999999, "elements": ["Al"]},
        )
    assert r.status_code == 404


def test_analyze_region_without_file_returns_400():
    client = _client()
    with patch("backend.api.services.h5_session.is_open", return_value=False):
        r = client.post(
            "/api/crystal-hint/analyze-region",
            json={"elements": ["Al"], "roi_type": "whole", "max_pixels": 16},
        )
    assert r.status_code == 400


def test_analyze_region_with_mocked_session_returns_histograms():
    """End-to-end Mode 2 with mocked h5_session + fake patterns."""
    rng = np.random.RandomState(0)
    size = 64

    def make_4fold():
        img = rng.rand(size, size).astype(np.float32) * 0.05
        cy, cx = (size - 1) / 2, (size - 1) / 2
        y, x = np.indices((size, size))
        yy = y - cy; xx = x - cx
        r2 = np.sqrt(yy * yy + xx * xx)
        theta = np.arctan2(yy, xx)
        falloff = np.exp(-r2 / (size * 0.20))
        bands = np.zeros_like(img)
        for k in range(4):
            angle = 2 * np.pi * k / 4
            d_angle = np.abs(np.angle(np.exp(1j * (theta - angle))))
            bands += np.exp(-(d_angle / (np.pi / 12)) ** 2)
        img += (bands * falloff) * 2.0
        img += np.exp(-r2 ** 2 / (2 * (size * 0.025) ** 2)) * 8.0
        return img

    pat = make_4fold()

    # Mock extractor with metadata + get_pattern
    fake_extractor = MagicMock()
    fake_extractor.get_metadata = MagicMock(return_value={
        "n_rows": 4, "n_cols": 4,
        "pattern_width": size, "pattern_height": size,
        "voltage_kv": 20.0,
        "pattern_center": {"x": 0.5, "y": 0.5, "z": 0.6},
    })

    client = _client()
    with patch("backend.api.services.h5_session.is_open", return_value=True), \
         patch("backend.api.services.h5_session.get_extractor", return_value=fake_extractor), \
         patch("backend.api.services.h5_session.get_cached_pattern", return_value=pat):
        r = client.post(
            "/api/crystal-hint/analyze-region",
            json={"elements": ["Al", "Si"], "roi_type": "whole", "max_pixels": 16},
        )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["n_pixels_analyzed"] > 0
    assert "n_fold_histogram" in data
    assert "library_match_rate" in data


def test_analyze_region_validates_payload():
    """max_pixels out of bounds should be 422."""
    client = _client()
    r = client.post(
        "/api/crystal-hint/analyze-region",
        json={"elements": [], "roi_type": "whole", "max_pixels": 100000},
    )
    assert r.status_code == 422


def test_external_search_returns_empty_on_network_failure():
    """When COD is unreachable, the endpoint should return [] not error out."""
    client = _client()
    # Use an obviously-invalid URL so requests fails fast
    with patch("backend.api.services.crystal_hint_cod.COD_BASE_URL",
               "http://invalid.invalid.invalid/result"):
        r = client.post(
            "/api/crystal-hint/external-search",
            json={"elements": ["Al", "Si"], "crystal_system": "cubic"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data == []


def test_external_search_with_mocked_response():
    """Mock COD response, verify external-search returns parsed matches."""
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.text = (
        "file\tformula\tsg\ta\n"
        "1010976\tMg2Si\tFm-3m\t6.39\n"
    )
    fake_response.raise_for_status = MagicMock()

    from backend.api.services.crystal_hint_cod import clear_cache
    clear_cache()

    client = _client()
    with patch("requests.get", return_value=fake_response):
        r = client.post(
            "/api/crystal-hint/external-search",
            json={
                "elements": ["Mg", "Si"],
                "crystal_system": "cubic",
                "a_low_A": 5.0, "a_high_A": 7.0,
                "max_results": 5,
            },
        )
    assert r.status_code == 200, r.text
    data = r.json()
    assert isinstance(data, list)
    if len(data) > 0:
        first = data[0]
        assert first["source"] == "COD"
        assert "structure_id" in first
        assert first["formula"] == "Mg2Si"


def test_analyze_pixel_validates_payload():
    client = _client()
    # missing pixel_index
    r = client.post("/api/crystal-hint/analyze-pixel", json={"elements": []})
    assert r.status_code == 422
    # negative pixel_index
    r = client.post(
        "/api/crystal-hint/analyze-pixel",
        json={"pixel_index": -1, "elements": []},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# 2026-05-28 (afternoon): bug fixes + UX features
# ---------------------------------------------------------------------------

def test_analyze_pixel_rejects_invalid_pattern_type():
    """pattern_type must be 'raw' or 'processed' — Pydantic Literal."""
    rng = np.random.RandomState(0)
    img = rng.rand(64, 64).astype(np.float32)
    client = _client()
    with patch("backend.api.services.h5_session.is_open", return_value=True), \
         patch("backend.api.services.h5_session.get_cached_pattern", return_value=img):
        r = client.post(
            "/api/crystal-hint/analyze-pixel",
            json={"pixel_index": 0, "pattern_type": "background"},
        )
    assert r.status_code == 422
    body = r.json()
    # `detail` is the SENTENCE the user sees — one line per problem, each
    # naming its field. It was a list of dicts until 2026-09-12; every error
    # panel in this app interpolates `err.response.data.detail` into a string,
    # so a list arrived on screen as "[object Object]". Iterating it here
    # iterated the characters of the sentence, which is why this test failed
    # rather than the app.
    detail = body["detail"]
    assert isinstance(detail, str), f"detail must be prose, got {detail!r}"
    assert "pattern_type" in detail, detail
    assert "raw" in detail.lower() and "processed" in detail.lower(), detail

    # The structured form is still there for a caller that wants the field.
    errors = body["errors"]
    assert any("pattern_type" in e["loc"]
               and ("raw" in e["msg"].lower() or "processed" in e["msg"].lower())
               for e in errors), errors


@requires(CIF_LIBRARY)
def test_local_library_endpoint_returns_display_formula():
    """The /library response should include display_formula for prettified
    decimal-heavy CIF formulas."""
    client = _client()
    r = client.get("/api/crystal-hint/library")
    assert r.status_code == 200
    entries = r.json()
    # At least one entry has display_formula different from formula
    decimal_heavy = [e for e in entries if "." in e.get("formula", "")]
    assert len(decimal_heavy) >= 1
    from phase_metadata import format_formula_subscripts
    for e in decimal_heavy:
        df = e.get("display_formula")
        assert df, f"missing display_formula for {e['key']}"
        # display_formula is the Crystal-Database composition: the real formula
        # with subscripts — NOT an element-list abbreviation like "(Al,Fe,Si)".
        assert not df.startswith("("), f"should be a real formula, got '{df}'"
        assert df == format_formula_subscripts(e["formula"])


def test_external_search_sorted_by_lattice_distance():
    """External matches with a_low/a_high should sort by lattice distance
    to the target (midpoint), MP before COD on ties."""
    # Use the MP-mocked test pattern from earlier in this file: stub MP to
    # return three results with different lattice params spread around 6 Å.
    fake_mp_results = [
        # (mp_id, a_A) — distances from 6.0: 2.0, 0.5, 1.0
        MagicMock(mp_id="mp-A", formula="X", space_group="Fm-3m",
                  space_group_number=225, crystal_system="cubic",
                  a_A=8.0, b_A=8.0, c_A=8.0,
                  cif_url="", title="E_above_hull = 0.000 eV/atom",
                  n_sites=1, energy_above_hull_eV=0.0),
        MagicMock(mp_id="mp-B", formula="Y", space_group="Fm-3m",
                  space_group_number=225, crystal_system="cubic",
                  a_A=6.5, b_A=6.5, c_A=6.5,
                  cif_url="", title="E_above_hull = 0.100 eV/atom",
                  n_sites=1, energy_above_hull_eV=0.1),
        MagicMock(mp_id="mp-C", formula="Z", space_group="Fm-3m",
                  space_group_number=225, crystal_system="cubic",
                  a_A=7.0, b_A=7.0, c_A=7.0,
                  cif_url="", title="E_above_hull = 0.000 eV/atom",
                  n_sites=1, energy_above_hull_eV=0.0),
    ]
    client = _client()
    with patch("backend.api.routes.crystal_hint.search_mp_async",
               return_value=fake_mp_results), \
         patch("backend.api.routes.crystal_hint.search_cod_async",
               return_value=[]):
        r = client.post(
            "/api/crystal-hint/external-search",
            json={"elements": ["X"], "crystal_system": "cubic",
                  "a_low_A": 5.5, "a_high_A": 6.5, "max_results": 5},
        )
    assert r.status_code == 200
    results = r.json()
    # Target midpoint = 6.0. Distances: mp-A=2.0, mp-B=0.5, mp-C=1.0
    # Expected order: mp-B (closest), mp-C, mp-A
    assert results[0]["structure_id"] == "mp-B"
    assert results[1]["structure_id"] == "mp-C"
    assert results[2]["structure_id"] == "mp-A"


def test_external_search_chemistry_free_requires_constraint():
    """Chemistry-free (no elements) with no crystal_system and no lattice
    bound must 422 rather than trigger an unbounded full-DB query."""
    client = _client()
    r = client.post("/api/crystal-hint/external-search",
                    json={"elements": [], "max_results": 10})
    assert r.status_code == 422
    assert "chemistry-free" in r.json()["detail"].lower()


def test_external_search_chemistry_free_with_system_allowed():
    """Chemistry-free WITH a crystal_system is allowed (does not 422).
    Mock both DB clients so no network is hit; assert the response shape."""
    client = _client()
    with patch("backend.api.routes.crystal_hint.search_cod_async",
               return_value=[]), \
         patch("backend.api.routes.crystal_hint.search_mp_async",
               return_value=[]):
        r = client.post("/api/crystal-hint/external-search",
                        json={"elements": [], "crystal_system": "monoclinic",
                              "max_results": 10})
    assert r.status_code == 200
    assert isinstance(r.json(), list)


@requires(CIF_LIBRARY)
def test_external_search_demotes_in_library_phases():
    """External search ranks NOVEL phases above ones already in the local
    library (same space group + lattice). Al (Fm-3m a≈4.05) IS in the
    library → flagged in_local_library and sorted below a novel phase."""
    from backend.api.services.crystal_hint_mp import MpResult
    client = _client()
    novel = MpResult(mp_id="mp-novel", formula="FeNovelX",
                     space_group="C2/m", space_group_number=12,
                     crystal_system="monoclinic", a_A=9.99, b_A=4.4, c_A=8.1,
                     energy_above_hull_eV=0.0)
    al = MpResult(mp_id="mp-134", formula="Al",
                  space_group="Fm-3m", space_group_number=225,
                  crystal_system="cubic", a_A=4.05, b_A=4.05, c_A=4.05,
                  energy_above_hull_eV=0.0)
    with patch("backend.api.routes.crystal_hint.search_cod_async",
               return_value=[]), \
         patch("backend.api.routes.crystal_hint.search_mp_async",
               return_value=[al, novel]):
        r = client.post("/api/crystal-hint/external-search",
                        json={"elements": ["Al", "Fe"], "max_results": 10})
    assert r.status_code == 200
    out = r.json()
    by_formula = {m["formula"]: m for m in out}
    assert by_formula["Al"]["in_local_library"] is True
    assert by_formula["FeNovelX"]["in_local_library"] is False
    # novel phase ranks above the in-library Al
    order = [m["formula"] for m in out]
    assert order.index("FeNovelX") < order.index("Al")
