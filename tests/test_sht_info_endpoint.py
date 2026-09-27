import json
from pathlib import Path

import h5py
import pytest

from backend.api.services.sht_provenance import (
    build_sht_info,
    read_xtal_reference,
    _engine_from_software_version,
)

_PROJECT_ROOT = Path(__file__).parent.parent
_NI_EMSOFT_SHT = (
    _PROJECT_ROOT / "Database" / "EBSD_SHT_Database" / "Ni" / "Ni (Ni) [cF4] {20kV}.sht"
)


def test_info_prefers_sidecar(tmp_path):
    sht = tmp_path / "Ni (Ni) [cF4] {20kV}.sht"
    sht.write_bytes(b"")  # binary not parsed when sidecar present
    sidecar = sht.with_suffix(".sht.provenance.json")
    sidecar.write_text(json.dumps({
        "schema": 1,
        "source_xtal": {"name": "Ni.xtal", "path": "x", "found": True},
        "source_cif": {"name": "Ni.cif", "path": "c", "found": True},
        "reference": "ref",
        "parameters": {"dmin": 0.05, "npx": 500, "voltage_kV": 20.0},
    }), encoding="utf-8")

    info = build_sht_info(sht, xtal_dir=tmp_path, cif_dir=tmp_path)
    assert info["provenance"]["origin"] == "sidecar"
    assert info["provenance"]["source_cif"]["name"] == "Ni.cif"
    assert info["parameters"]["npx"] == 500


def test_info_sidecar_tolerates_missing_bethe(tmp_path):
    """Task 5's sidecar parameters may NOT contain 'bethe' — must not crash."""
    sht = tmp_path / "Ni (Ni) [cF4] {20kV}.sht"
    sht.write_bytes(b"")
    sidecar = sht.with_suffix(".sht.provenance.json")
    sidecar.write_text(json.dumps({
        "schema": 1,
        "source_xtal": {"name": "Ni.xtal", "path": "x", "found": True},
        "source_cif": {"name": "Ni.cif", "path": "c", "found": True},
        "reference": "ref",
        "parameters": {"dmin": 0.05, "npx": 500, "voltage_kV": 20.0},  # no 'bethe'
    }), encoding="utf-8")

    info = build_sht_info(sht, xtal_dir=tmp_path, cif_dir=tmp_path)
    assert info["parameters"]["source"] == "sidecar"
    assert "bethe" not in info["parameters"]


def test_info_unknown_when_nothing_available(tmp_path):
    sht = tmp_path / "Mystery (Mystery) {20kV}.sht"
    sht.write_bytes(b"not a real sht")
    info = build_sht_info(sht, xtal_dir=tmp_path, cif_dir=tmp_path)
    assert info["provenance"]["source_xtal"]["found"] is False
    assert info["provenance"]["source_cif"]["found"] is False


def test_recovered_reference_reads_xtal_use_reference(tmp_path):
    """Controller decision #1: recovered reference = .xtal /CrystalData/useReference."""
    sht = tmp_path / "Ni (Ni) [cF4] {20kV}.sht"
    sht.write_bytes(b"not a real sht")  # no sidecar -> recovery path
    xtal = tmp_path / "Ni.xtal"
    with h5py.File(str(xtal), "w") as f:
        grp = f.create_group("CrystalData")
        grp.create_dataset("useReference", data="10.17188/1316752")

    info = build_sht_info(sht, xtal_dir=tmp_path, cif_dir=tmp_path)
    assert info["provenance"]["source_xtal"]["found"] is True
    assert info["provenance"]["reference"] == "10.17188/1316752"


def test_read_xtal_reference_helper(tmp_path):
    xtal = tmp_path / "Ni.xtal"
    with h5py.File(str(xtal), "w") as f:
        grp = f.create_group("CrystalData")
        grp.create_dataset("useReference", data="10.17188/1316752")
    assert read_xtal_reference(xtal) == "10.17188/1316752"


def test_read_xtal_reference_missing_returns_empty(tmp_path):
    missing = tmp_path / "does_not_exist.xtal"
    assert read_xtal_reference(missing) == ""


# ---------------------------------------------------------------------------
# Engine detection + binary-DOI recovery (rename-proof traceability)
# ---------------------------------------------------------------------------

def test_engine_from_software_version():
    assert _engine_from_software_version(b"fwd_sim0") == "ours"
    assert _engine_from_software_version("fwd_sim0") == "ours"
    assert _engine_from_software_version(b"vcd75885") == "emsoft"
    assert _engine_from_software_version(b"") == "unknown"
    assert _engine_from_software_version(b"\x00\x00\x00") == "unknown"
    assert _engine_from_software_version(None) == "unknown"


def test_info_reports_engine_unknown_for_unparseable(tmp_path):
    sht = tmp_path / "Mystery (Mystery) {20kV}.sht"
    sht.write_bytes(b"not a real sht")
    info = build_sht_info(sht, xtal_dir=tmp_path, cif_dir=tmp_path)
    assert info["provenance"]["engine"] == "unknown"


def test_info_sidecar_surfaces_engine(tmp_path):
    sht = tmp_path / "Ni (Ni) [cF4] {20kV}.sht"
    sht.write_bytes(b"")
    sht.with_suffix(".sht.provenance.json").write_text(json.dumps({
        "schema": 1,
        "source_xtal": {"name": "Ni.xtal", "path": "x", "found": True},
        "source_cif": {"name": "Ni.cif", "path": "c", "found": True},
        "reference": "ref",
        "parameters": {"dmin": 0.05, "npx": 500, "engine": "ours"},
    }), encoding="utf-8")
    info = build_sht_info(sht, xtal_dir=tmp_path, cif_dir=tmp_path)
    assert info["provenance"]["engine"] == "ours"


def _fake_sht(software_version, doi):
    from types import SimpleNamespace
    return SimpleNamespace(
        formula="Ni", space_group="Fm-3m", point_group="m-3m",
        voltage_kv=20.0, primary_tilt_deg=70.0, bandwidth=384,
        crystal_lattice=(0.35, 0.35, 0.35, 90.0, 90.0, 90.0), sim_dmin=None,
        software_version=software_version,
        raw_header=SimpleNamespace(doi=doi, software_version=software_version),
    )


def test_recovery_filters_software_paper_doi(tmp_path, monkeypatch):
    """An 'Ours' file whose binary doi is the EMsoft software paper must NOT
    surface that as the crystal reference (it is not a structure citation)."""
    import backend.spherical_gpu.pipeline.sht_io as sht_io
    fake = _fake_sht(b"fwd_sim0", "https://doi.org/10.1016/j.ultramic.2019.112841")
    monkeypatch.setattr(sht_io, "read_sht_master", lambda *a, **k: fake)
    sht = tmp_path / "Ni (Ni) [cF4] {20kV}.sht"; sht.write_bytes(b"x")
    info = build_sht_info(sht, xtal_dir=tmp_path, cif_dir=tmp_path)
    assert info["provenance"]["engine"] == "ours"
    assert info["provenance"]["reference"] == ""  # filtered


def test_recovery_uses_real_binary_doi(tmp_path, monkeypatch):
    """A real structure citation in the binary doi IS surfaced when the .xtal is gone."""
    import backend.spherical_gpu.pipeline.sht_io as sht_io
    fake = _fake_sht(b"vcd75885", "10.1107/realcitation")
    monkeypatch.setattr(sht_io, "read_sht_master", lambda *a, **k: fake)
    sht = tmp_path / "Ni (Ni) [cF4] {20kV}.sht"; sht.write_bytes(b"x")
    info = build_sht_info(sht, xtal_dir=tmp_path, cif_dir=tmp_path)
    assert info["provenance"]["engine"] == "emsoft"
    assert info["provenance"]["reference"] == "10.1107/realcitation"


@pytest.mark.skipif(not _NI_EMSOFT_SHT.exists(), reason="real EMsoft Ni .sht not present")
def test_recovered_doi_from_binary_when_xtal_missing(tmp_path):
    """A real EMsoft .sht with NO linkable .xtal must still surface its embedded
    literature DOI (rescued from the binary header) and engine='emsoft'.

    The .sht is COPIED into tmp_path first, deliberately. This test was red for
    seven weeks because a sidecar sat beside the real one in the user's library
    (written by another test, claiming a source .xtal that never existed):
    build_sht_info reads the sidecar in preference to the binary, so the empty
    xtal_dir below had no effect. Reading a file out of a directory the app
    writes to means the test asserts today's state of that directory — and it
    would go red again, correctly this time, the day someone simulates Ni
    through the app and a genuine sidecar appears.
    """
    import shutil
    sht = tmp_path / _NI_EMSOFT_SHT.name
    shutil.copyfile(_NI_EMSOFT_SHT, sht)
    # xtal_dir/cif_dir intentionally empty -> the .xtal link is "broken"
    info = build_sht_info(sht, xtal_dir=tmp_path, cif_dir=tmp_path)
    assert info["provenance"]["source_xtal"]["found"] is False
    assert info["provenance"]["engine"] == "emsoft"
    assert info["provenance"]["reference"]  # non-empty, recovered from binary doi
