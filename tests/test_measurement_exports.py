"""The exports state the measurement a result was indexed on: quality fields, EDS payload,
acquisition header, the .ctf, the EDS export, the light .h5, the Aztec comparison.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from backend.api.services.load_choices import LoadChoiceError, LoadVariant
from tests.two_measurements import *  # noqa: F401,F403  (constants and fixtures)
from tests.two_measurements import _grid, _group, _xmap  # noqa: F401


@pytest.mark.parametrize("variant,g", BOTH, ids=BOTH_IDS)
def test_exported_band_contrast_and_bands_are_the_measurements(two, variant, g):
    from backend.api.services.result_exporter import _read_h5oina_quality
    got = _read_h5oina_quality(two, N, variant)
    np.testing.assert_array_equal(got["bc"], BC[g])
    np.testing.assert_array_equal(got["bands"], np.full(N, BANDS[g]))


@pytest.mark.parametrize("variant,g", BOTH, ids=BOTH_IDS)
def test_exported_acquisition_header_is_the_measurements(two, variant, g):
    from backend.api.services.result_exporter import _read_h5oina_acquisition
    got = _read_h5oina_acquisition(two, variant)
    assert got["kv"] == KV[g] and got["wd"] == WD[g] and got["mag"] == MAG[g]
    assert got["tilt"] == pytest.approx(np.radians(TILT_DEG[g]))


@pytest.mark.parametrize("variant,g", BOTH, ids=BOTH_IDS)
def test_the_light_exports_quality_fields_are_the_measurements(two, tmp_path, variant, g):
    from backend.api.services.result_exporter import _copy_quality_fields
    with h5py.File(tmp_path / "light.h5", "w") as out:
        grp = out.create_group("Assignment")
        copied = _copy_quality_fields(two, grp, (NY, NX), variant)
        assert sorted(copied) == ["band_contrast", "bands", "dd", "pc_x", "pc_y"]
        np.testing.assert_array_equal(grp["band_contrast"][...], _grid(BC[g], np.uint8))
        np.testing.assert_array_equal(grp["bands"][...], np.full((NY, NX), BANDS[g]))
        np.testing.assert_allclose(grp["pc_x"][...], PC[g][0])
        np.testing.assert_allclose(grp["pc_y"][...], PC[g][1])
        np.testing.assert_allclose(grp["dd"][...], PC[g][2])


@pytest.mark.parametrize("variant,g", BOTH, ids=BOTH_IDS)
def test_the_light_exports_eds_payload_is_the_measurements(two, tmp_path, variant, g):
    from backend.api.services.result_exporter import _copy_eds_full
    with h5py.File(tmp_path / "light.h5", "w") as out:
        assert _copy_eds_full(two, out, variant) is True
        np.testing.assert_array_equal(out["EDS/Data/Window Integral/Fe/Counts"][...], FE[g])
        assert out["EDS/Header/Beam Voltage"][0] == KV[g]


def test_a_missing_measurement_stops_the_exports_readers_with_the_code(two, tmp_path):
    from backend.api.services.result_exporter import (
        _copy_eds_full, _copy_quality_fields, _read_h5oina_acquisition, _read_h5oina_quality)
    with h5py.File(tmp_path / "light.h5", "w") as out:
        for call in (lambda: _read_h5oina_quality(two, N, V7),
                     lambda: _read_h5oina_acquisition(two, V7),
                     lambda: _copy_quality_fields(two, out.create_group("A"), (NY, NX), V7),
                     lambda: _copy_eds_full(two, out, V7)):
            with pytest.raises(LoadChoiceError) as e:
                call()
            assert e.value.code == "variantNotSupported"


@pytest.mark.parametrize("variant,g", BOTH, ids=BOTH_IDS)
def test_the_ctf_carries_the_measurements_band_contrast_and_header(two, tmp_path, variant, g):
    from backend.api.services.result_exporter import _write_ctf
    out = tmp_path / "out.ctf"
    _write_ctf(str(out), _xmap(), ["Al"], 0.5, source_h5_path=two, variant=variant)
    lines = out.read_text(encoding="utf-8").splitlines()
    head = next(i for i, ln in enumerate(lines) if ln.startswith("Phase\tX\tY"))
    cols = lines[head].split("\t")
    rows = [ln.split("\t") for ln in lines[head + 1:] if ln.strip()]
    assert [int(r[cols.index("BC")]) for r in rows] == BC[g]
    assert {int(r[cols.index("Bands")]) for r in rows} == {BANDS[g]}
    tokens = next(ln for ln in lines if "\tKV\t" in ln).split("\t")
    assert float(tokens[tokens.index("KV") + 1]) == KV[g]
    assert float(tokens[tokens.index("Mag") + 1]) == MAG[g]


@pytest.mark.parametrize("variant,g", BOTH, ids=BOTH_IDS)
def test_the_source_geometry_is_the_measurements_header(two, variant, g):
    from backend.api.services.ang_export import read_source_geometry
    got = read_source_geometry(two, variant)
    assert got["header_path"] == f"/{g}/EBSD/Header"
    assert got["beam_voltage_kv"] == KV[g]
    assert got["sample_tilt_deg"] == pytest.approx(TILT_DEG[g])
    assert got["working_distance_mm"] == WD[g]


@pytest.mark.parametrize("variant,g", BOTH, ids=BOTH_IDS)
def test_the_eds_acquisition_header_is_the_measurements(two, variant, g):
    from backend.api.services.eds_export import read_acquisition_header
    got = read_acquisition_header(two, variant)
    assert float(np.ravel(got["Beam Voltage"])[0]) == KV[g]
    assert float(np.ravel(got["Working Distance"])[0]) == WD[g]


def test_the_eds_export_states_the_viewers_measurements_acquisition(viewer, two):
    from backend.api.routes import eds_export as route
    from backend.api.services.eds_export import ExportOptions, _acquisition_block
    viewer(V2)
    state = SimpleNamespace(n_rows=NY, n_cols=NX)
    route._step_um = lambda: (0.25, 0.25)
    geometry = route._geometry(state, two)
    assert geometry.source_variant == V2
    block = _acquisition_block(geometry, ExportOptions(), [])
    assert block["beam_voltage_kv"] == KV[2]
    viewer(None)
    assert route._geometry(state, two).source_variant is None
    assert _acquisition_block(route._geometry(state, two), ExportOptions(), [])["beam_voltage_kv"] == KV[1]


def test_the_exports_context_states_the_results_measurements_header(two):
    from backend.api.routes.indexing import _export_context
    for meta, g in (({}, 1), ({"load_variant": V2.to_dict()}, 2)):
        active = SimpleNamespace(metadata=meta, steps=[])
        ctx = _export_context(active, two, "oxford")
        assert ctx["geometry"]["beam_voltage_kv"] == KV[g], g
        assert ctx["geometry"]["header_path"] == f"/{g}/EBSD/Header"


def _light_file(tmp_path, source, variant):
    """A light result file that points at ``source`` and says which measurement it was indexed on."""
    path = tmp_path / "light.h5"
    with h5py.File(path, "w") as f:
        f.create_group("SourceReference").attrs["source_file_path"] = str(source)
        idx = f.create_group("Indexing")
        if variant is not None:
            idx.attrs["measurement"] = variant.measurement
            idx.attrs["pattern_source"] = variant.pattern_source
    return str(path)


@pytest.mark.parametrize("variant,g,name,agree", [(None, 1, "Al", 6), (V2, 2, "Si", 0)],
                         ids=["default", "measurement-2"])
def test_the_aztec_comparison_reads_the_measurement_the_light_file_was_indexed_on(
        two, tmp_path, variant, g, name, agree):
    import backend.api.routes.analysis as ar
    saved = ar._dataset
    try:
        ar._dataset = SimpleNamespace(source=_light_file(tmp_path, two, variant), shape=(NY, NX), xmap=_xmap())
        got = ar.aztec_comparison()
    finally:
        ar._dataset = saved
    # our only phase is "Al": it meets measurement 1's Aztec phase "Al" and not measurement 2's "Si"
    assert got["phase_mapping"] == {str(g): {"aztec_name": name, "our_id": 0 if name == "Al" else None}}
    assert got["n_agree_phase"] == agree


def test_the_light_export_states_the_results_measurement_and_refuses_one_the_source_lacks(
        two, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.api.main import app
    from tests.test_indexing_export_light import restore_indexing_registry  # noqa: F401
    import backend.api.routes.indexing as ix
    from backend.api.routes import ebsd_viewer as ev
    from indexing_controller import IndexingMethod, IndexingResult
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation

    def result(stamp):
        xmap = CrystalMap(rotations=Rotation.identity(N), phase_id=np.zeros(N, int),
                          x=np.tile(np.arange(NX, dtype=np.float32), NY),
                          y=np.repeat(np.arange(NY, dtype=np.float32), NX),
                          phase_list=PhaseList(Phase(name="Al", space_group=225, point_group="m-3m")))
        return IndexingResult(
            xmap=xmap, selection_mask=np.ones(N, bool), original_shape=(NY, NX),
            method=IndexingMethod.SPHERICAL, confidence_scores=np.full(N, 0.8, np.float32),
            metadata={"step_size_um": 0.5, "source_vendor": "oxford", "load_variant": stamp})

    monkeypatch.setattr(ev, "_ebsd_file_path", two, raising=False)
    saved, saved_active = dict(ix._result_registry), ix._active_result_id
    try:
        for stamp, kv in ((None, KV[1]), (V2.to_dict(), KV[2])):
            ix._result_registry["light-measurement"] = result(stamp)
            ix._active_result_id = "light-measurement"
            out = tmp_path / f"light-{kv}.h5"
            resp = TestClient(app).post("/api/indexing/export", json={"format": "h5_light", "filename": str(out)})
            assert resp.status_code == 200, resp.text
            with h5py.File(out, "r") as f:
                assert f["Acquisition"].attrs["beam_voltage_kv"] == kv
        # a source that does not hold the measurement: refused up front, nothing written
        ix._result_registry["light-measurement"] = result(V7.to_dict())
        out = tmp_path / "light-7.h5"
        resp = TestClient(app).post("/api/indexing/export", json={"format": "h5_light", "filename": str(out)})
        assert resp.status_code == 400
        assert resp.headers["x-orienta-code"] == "variantNotSupported"
        assert not out.exists()
    finally:
        ix._result_registry.clear()
        ix._result_registry.update(saved)
        ix._active_result_id = saved_active
