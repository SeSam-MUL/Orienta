"""The surfaces that reach the measurement-bound readers hand them the measurement.

``test_measurement_quality`` / ``_session`` / ``_exports`` call the readers; here the code ABOVE them
is driven: the overview and metadata routes, the quality fallback of the Phase Map layer, Crystal
Hint's route and neighbourhood, the light export's Assignment fields, and the checkpoint-based
batch exporters. A reader that is right but is not given the variant shows up here.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from backend.api.services import variant_registry
from backend.api.services.load_choices import LoadChoiceError, LoadVariant
from tests.two_measurements import *  # noqa: F401,F403  (constants and fixtures)
from tests.two_measurements import _grid, _group, _xmap  # noqa: F401

V2_RAW = LoadVariant("2", "unprocessed")


# ============================================================================ overview and metadata

def _signal():
    kp = pytest.importorskip("kikuchipy")
    return kp.signals.EBSD(np.zeros((NY, NX, SY, SX), np.uint8))


@pytest.mark.parametrize("variant,g", [(None, 1), (V2, 2)], ids=["default", "measurement-2"])
def test_the_overview_uses_the_stored_map_of_the_loaded_measurement(two, monkeypatch, variant, g):
    """The fast path (the dataset is the file's unedited load) reads Aztec's Band Contrast."""
    from backend.api.routes import ebsd_viewer as ev
    stem = ev.display_stem(two)
    sig = _signal()
    monkeypatch.setattr(ev, "_ebsd_file_path", two)
    monkeypatch.setattr(ev, "_active_dataset", stem)
    monkeypatch.setattr(ev, "_get_active_signal", lambda *a, **k: sig)
    ev._overview_cache.clear()
    if variant is not None:
        variant_registry.register(stem, variant)
    try:
        got = asyncio.run(ev.overview("bc"))
    finally:
        variant_registry.drop(stem)
        ev._overview_cache.clear()
    assert got["source"] == "native" and got["metric"] == "band_contrast"
    assert (got["min"], got["max"]) == (min(BC[g]), max(BC[g]))


@pytest.mark.parametrize("variant,g", [(None, 1), (V2, 2)], ids=["default", "measurement-2"])
def test_the_metadata_states_the_loaded_measurements_beam_energy_and_step(viewer, variant, g):
    from backend.api.routes import ebsd_viewer as ev
    viewer(variant)
    with patch.object(ev, "_get_active_signal", return_value=_signal()):
        got = asyncio.run(ev.get_metadata())
    assert got["beam_energy"] == KV[g]
    assert got["step_size"]["x"] == STEP[g]
    assert got["eds_elements"] == ["Fe"]


# ====================================================================== Phase Map quality fallback

def test_the_phase_map_falls_back_to_the_image_quality_of_the_results_measurement(two, tmp_path):
    """Measurement 2 has no Band Contrast, only IQ: the layer shows ITS image quality, labelled as such,
    and never measurement 1's Band Contrast."""
    from backend.api.routes.phase_map import _compute_layer_rgba
    from tests.test_phase_map_layer_endpoint import _make_fake_result
    path = tmp_path / "iq.h5oina"
    import shutil
    shutil.copy2(two, path)
    with h5py.File(path, "a") as f:
        del f["2/EBSD/Data/Band Contrast"]
    fake = _make_fake_result(n_rows=NY, n_cols=NX, with_bc=False)
    fake.metadata = {"source_file": str(path), "load_variant": V2.to_dict()}
    scale = {}
    with patch("backend.api.routes.phase_map.get_last_indexing_result", return_value=fake), \
         patch("backend.api.routes.phase_map.get_analysis_dataset", return_value=None):
        _compute_layer_rgba(kind="bc", scale_out=scale)
    assert scale["label"] == "Image Quality (native)"
    assert (scale["min"], scale["max"]) == pytest.approx((min(IQ[2]), max(IQ[2])))


# ========================================================================================= Crystal Hint

def test_crystal_hints_route_analyses_the_loaded_sources_pattern(viewer, monkeypatch):
    from backend.api.routes import crystal_hint
    from backend.api.services import h5_session
    viewer(V2_RAW)
    seen = []
    real = h5_session.get_loaded_pattern

    def spy(idx, pattern_type="processed"):
        seen.append(real(idx, pattern_type))
        return seen[-1]

    class Stop(Exception):
        pass

    def stop():
        raise Stop

    monkeypatch.setattr(crystal_hint, "_fetch_active_pattern", lambda idx: None)
    monkeypatch.setattr(h5_session, "get_loaded_pattern", spy)
    monkeypatch.setattr(crystal_hint, "_gather_detector_geom", stop)
    with pytest.raises(Stop):
        crystal_hint.analyze_pixel(crystal_hint.AnalyzePixelRequest(pixel_index=1, elements=["Al"]))
    assert int(seen[0][0, 0]) == 1000 + 300 + 3 * SY * SX      # measurement 2's unprocessed pattern 1


def test_crystal_hints_neighbourhood_reads_the_loaded_measurement(viewer, monkeypatch):
    from backend.api.routes import crystal_hint
    viewer(V2)
    monkeypatch.setattr(crystal_hint, "_fetch_active_pattern", lambda idx: None)
    patterns = crystal_hint._collect_neighbour_patterns(0, 1, "processed")
    assert len(patterns) == 4                                   # pixel (0, 0) and its three neighbours
    assert {int(p[0, 0]) for p in patterns} == {20}             # measurement 2 processed: all 20


# ============================================================================== the light export's fields

def test_the_light_export_copies_the_results_measurements_quality_fields(two, tmp_path, monkeypatch):
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
            metadata={"step_size_um": 0.5, "source_vendor": "oxford", "load_variant": stamp,
                      "per_phase_data": {"Al": {"ci": np.full(N, 0.8, np.float32), "ci_mean": 0.8}}})

    monkeypatch.setattr(ev, "_ebsd_file_path", two, raising=False)
    saved, saved_active = dict(ix._result_registry), ix._active_result_id
    try:
        for stamp, g in ((None, 1), (V2.to_dict(), 2)):
            ix._result_registry["light-assignment"] = result(stamp)
            ix._active_result_id = "light-assignment"
            out = tmp_path / f"light-{g}.h5"
            resp = TestClient(app).post("/api/indexing/export", json={"format": "h5_light", "filename": str(out)})
            assert resp.status_code == 200, resp.text
            with h5py.File(out, "r") as f:
                assg = f["Indexing/Assignment"]
                np.testing.assert_array_equal(assg["band_contrast"][...], _grid(BC[g], np.uint8))
                np.testing.assert_allclose(assg["pc_x"][...], PC[g][0])
                assert f["Acquisition"].attrs["beam_voltage_kv"] == KV[g]
    finally:
        ix._result_registry.clear()
        ix._result_registry.update(saved)
        ix._active_result_id = saved_active


# ============================================================================== the batch exporters

def _checkpoint(tmp_path, source):
    from backend.api.services.checkpoint_writer import CheckpointWriter
    cw = CheckpointWriter(str(source))
    cw.init_metadata(grid_shape=(NY, NX), batch_id="m", method="spherical", step_size_um=0.5)
    meta = {"phase_file": "Al.cif", "ci_mean": 0.7, "duration_sec": 1.0, "space_group": 225,
            "point_group": "m-3m", "lattice_constants": [4.0495, 4.0495, 4.0495, 90, 90, 90]}
    rng = np.random.default_rng(1)
    cw.write_phase_result("Al", np.full((NY, NX), 0.7, np.float32),
                          rng.uniform(0, 3, (NY, NX, 3)).astype(np.float32), meta)
    cw.compute_auto_assignment(confidence_threshold=0.0)
    return cw


@pytest.mark.parametrize("variant,g", [(None, 1), (V2, 2)], ids=["default", "measurement-2"])
def test_the_checkpoint_ang_and_ctf_carry_the_measurements_band_contrast_and_header(two, tmp_path, variant, g):
    from backend.api.services.result_exporter import export_ang_ctf
    cw = _checkpoint(tmp_path, two)
    ang, ctf = export_ang_ctf(cw.checkpoint_path, str(tmp_path / f"o{g}"), source_h5_path=two, variant=variant)
    lines = Path(ctf).read_text(encoding="utf-8").splitlines()
    head = next(i for i, ln in enumerate(lines) if ln.startswith("Phase\tX\tY"))
    cols = lines[head].split("\t")
    assert [int(ln.split("\t")[cols.index("BC")]) for ln in lines[head + 1:] if ln.strip()] == BC[g]
    tokens = next(ln for ln in lines if "\tKV\t" in ln).split("\t")
    assert float(tokens[tokens.index("KV") + 1]) == KV[g]
    # the .ang: the IQ column is the measurement's Band Contrast, and the header states its geometry
    text = Path(ang).read_text(encoding="utf-8")
    rows = [ln.split() for ln in text.splitlines() if ln and not ln.startswith("#")]
    assert [int(float(r[5])) for r in rows] == BC[g]                      # IQ column
    assert f"beam_voltage_kv={KV[g]:g}" in text.replace(" ", "") or f"{KV[g]:g}" in text


@pytest.mark.parametrize("variant,g", [(None, 1), (V2, 2)], ids=["default", "measurement-2"])
def test_the_checkpoint_light_h5_carries_the_measurements_fields(two, tmp_path, variant, g):
    from backend.api.services.result_exporter import export_result_h5_light
    cw = _checkpoint(tmp_path, two)
    out = export_result_h5_light(two, cw.checkpoint_path, str(tmp_path / f"l{g}"), variant=variant)
    with h5py.File(out, "r") as f:
        np.testing.assert_array_equal(f["Indexing/Assignment/band_contrast"][...], _grid(BC[g], np.uint8))
        np.testing.assert_array_equal(f["EDS/Data/Window Integral/Fe/Counts"][...], FE[g])
        assert f["Acquisition"].attrs["beam_voltage_kv"] == KV[g]


# ============================== a measurement without Band Contrast: the image quality of ITS patterns

def _patterns_without_band_contrast(tmp_path):
    """Measurement 1: smooth patterns (every pixel the same quality). Measurement 2: noise (a different
    quality at every pixel). Neither has a Band Contrast channel, so an export has to compute one."""
    kp = pytest.importorskip("kikuchipy")
    rng = np.random.default_rng(0)
    sy = sx = 8
    stacks = {1: np.tile(np.linspace(10, 200, sx), (N, sy, 1)).astype(np.uint8),
              2: rng.integers(0, 255, (N, sy, sx)).astype(np.uint8)}
    path = tmp_path / "noband.h5oina"
    with h5py.File(path, "w") as f:
        f.create_dataset("Manufacturer", data=np.bytes_("Oxford Instruments"))
        f.create_dataset("Format Version", data=np.bytes_("5.0"))
        for g, stack in stacks.items():
            hdr = f.create_group(f"{g}/EBSD/Header")
            for key, value in (("X Cells", NX), ("Y Cells", NY), ("X Step", 0.5), ("Y Step", 0.5),
                               ("Pattern Height", sy), ("Pattern Width", sx)):
                hdr.create_dataset(key, data=value)
            f.create_dataset(f"{g}/EBSD/Data/Processed Patterns", data=stack)
    expected = {}
    for g, stack in stacks.items():
        iq = np.asarray(kp.signals.EBSD(stack.reshape(NY, NX, sy, sx)).get_image_quality(), float).ravel()
        expected[g] = np.clip(iq / iq.max() * 255.0, 0, 255)
    assert not np.allclose(expected[1], expected[2])
    return str(path), expected


@pytest.mark.parametrize("variant,g", [(None, 1), (V2, 2)], ids=["default", "measurement-2"])
def test_the_ctf_computes_the_missing_band_contrast_from_the_measurements_patterns(tmp_path, variant, g):
    from backend.api.services.result_exporter import _write_ctf
    path, expected = _patterns_without_band_contrast(tmp_path)
    out = tmp_path / "o.ctf"
    _write_ctf(str(out), _xmap(), ["Al"], 0.5, source_h5_path=path, variant=variant)
    lines = out.read_text(encoding="utf-8").splitlines()
    head = next(i for i, ln in enumerate(lines) if ln.startswith("Phase\tX\tY"))
    cols = lines[head].split("\t")
    got = [int(ln.split("\t")[cols.index("BC")]) for ln in lines[head + 1:] if ln.strip()]
    assert got == expected[g].astype(int).tolist()


@pytest.mark.parametrize("variant,g", [(None, 1), (V2, 2)], ids=["default", "measurement-2"])
def test_the_checkpoint_ang_computes_the_missing_iq_from_the_measurements_patterns(tmp_path, variant, g):
    from backend.api.services.result_exporter import export_ang_ctf
    path, expected = _patterns_without_band_contrast(tmp_path)
    cw = _checkpoint(tmp_path, path)
    ang, _ctf = export_ang_ctf(cw.checkpoint_path, str(tmp_path / f"o{g}"), source_h5_path=path,
                               variant=variant, write_ctf=False)
    text = Path(ang).read_text(encoding="utf-8")
    rows = [ln.split() for ln in text.splitlines() if ln and not ln.startswith("#")]
    np.testing.assert_allclose([float(r[5]) for r in rows], expected[g].astype(np.float32), atol=0.01)
