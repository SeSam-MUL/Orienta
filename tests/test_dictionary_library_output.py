"""A generated dictionary must land where the Indexing page looks for it.

Before this, ``POST /api/dictionary-gpu/generate`` always wrote to
``tasks/dict_gpu_<timestamp>.h5`` while ``discover_files_for_method`` only
scans ``Database/Dictionary_Library`` and attributes a file to a phase by
matching its filename against the CIF library. A file called
``dict_gpu_20260813_143000.h5`` therefore could never appear under a phase
card, no matter how often the page refreshed discovery — the user's
"Kein Dictionary generiert" never went away.

These tests pin the two halves of the fix:
  1. ONE canonical naming/location rule, shared by the legacy CPU writer
     (``save_dictionary``) and the GPU route.
  2. The route honours ``save_to_library`` and reports the resulting path.
"""
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from simulation.dictionary_generator import (
    DictionaryMetadata,
    dictionary_library_paths,
)


# ---------------------------------------------------------------------------
# Canonical path helper
# ---------------------------------------------------------------------------

def test_library_paths_match_the_existing_library_convention(tmp_path):
    """Name must be <master_stem>_dict_<E>kV_<HxW>_pc<x_y_z>_<res>deg.

    This is the shape already present in Database/Dictionary_Library and the
    one `find_linked_cif` can attribute to a phase (leading token "Si").
    """
    h5, js = dictionary_library_paths(
        tmp_path,
        master_path="C:/db/EBSD_H5_Cache/Si/Si_master_E20kV_npx500.h5",
        energy_kv=20.0,
        detector_shape=(128, 156),
        pc=(0.547, 0.465, 0.609),
        resolution_deg=2.0,
    )
    stem = "Si_master_E20kV_npx500_dict_20kV_128x156_pc547_465_609_2.0deg"
    assert h5 == tmp_path / "Si" / f"{stem}.h5"
    assert js == tmp_path / "Si" / f"{stem}.json"
    assert js == h5.with_suffix(".json"), "sidecar must sit NEXT TO the .h5"


def test_library_paths_group_by_short_material():
    """Subfolder is the leading filename token, as Dictionary_Library/Al/ is."""
    h5, _ = dictionary_library_paths(
        Path("/lib"),
        master_path="/db/Al_master_E20kV_npx500.h5",
        energy_kv=20.0,
        detector_shape=(60, 60),
        pc=(0.5, 0.5, 0.5),
        resolution_deg=5.0,
    )
    assert h5.parent.name == "Al"


def test_different_pc_does_not_overwrite():
    """Two runs that differ ONLY in PC must produce different files.

    The user's whole problem is that the library dictionaries were built for
    a different projection centre. Regenerating for the current PC must add a
    file, not silently replace the old one.
    """
    common = dict(
        master_path="/db/Si_master_E20kV_npx500.h5",
        energy_kv=20.0,
        detector_shape=(128, 156),
        resolution_deg=2.0,
    )
    a, _ = dictionary_library_paths(Path("/lib"), pc=(0.5, 0.5, 0.5), **common)
    b, _ = dictionary_library_paths(Path("/lib"), pc=(0.547, 0.465, 0.609), **common)
    assert a != b


def test_save_dictionary_uses_the_same_helper(tmp_path):
    """The legacy CPU writer and the GPU route must not drift apart."""
    saved = {}

    class _FakeSignal:
        def save(self, path, overwrite=False):
            saved["path"] = Path(path)

    meta = DictionaryMetadata(
        master_path="/db/Ni_master_E20kV_npx500.h5",
        material="Ni",
        phase_name="Nickel",
        energy_kv=20.0,
        detector_shape=(60, 60),
        pc=(0.5, 0.5, 0.5),
        resolution_deg=5.0,
    )
    from simulation.dictionary_generator import save_dictionary

    out = save_dictionary(_FakeSignal(), meta, tmp_path)

    expected_h5, expected_json = dictionary_library_paths(
        tmp_path,
        master_path=meta.master_path,
        energy_kv=meta.energy_kv,
        detector_shape=meta.detector_shape,
        pc=meta.pc,
        resolution_deg=meta.resolution_deg,
    )
    assert out == expected_h5 == saved["path"]
    assert expected_json.is_file()


# ---------------------------------------------------------------------------
# Route wiring
# ---------------------------------------------------------------------------

def _master(tmp_path: Path, name: str = "Si_master_E20kV_npx500.h5") -> Path:
    p = tmp_path / name
    p.write_bytes(b"\x89HDF\r\n\x1a\n")  # route only checks is_file()
    return p


def test_generate_with_save_to_library_targets_the_library(tmp_path, monkeypatch):
    """save_to_library=True must redirect the output into Dictionary_Library."""
    import backend.api.routes.dictionary_gpu as route_mod

    lib = tmp_path / "Dictionary_Library"
    monkeypatch.setattr(route_mod, "_dictionary_library_dir", lambda: lib)

    master = _master(tmp_path)
    payload = {
        "master_path": str(master),
        "detector_shape": [128, 156],
        "pc": [0.547, 0.465, 0.609],
        "sample_tilt": 70.0,
        "energy_kv": 20.0,
        "resolution_deg": 2.0,
        "normalize": False,
        "save_to_library": True,
    }

    client = TestClient(app)
    with patch.object(route_mod.threading, "Thread") as mock_thread:
        r = client.post("/api/dictionary-gpu/generate", json=payload)

    assert r.status_code == 200, r.text
    body = r.json()
    expected = lib / "Si" / (
        "Si_master_E20kV_npx500_dict_20kV_128x156_pc547_465_609_2.0deg.h5"
    )
    # Reported back so the UI can say where it went…
    assert Path(body["output_path"]) == expected
    # …and actually handed to the pipeline.
    job_payload = mock_thread.call_args.kwargs["args"][1]
    assert Path(job_payload.output_path) == expected


def test_generate_without_save_to_library_is_unchanged(tmp_path):
    """Default stays the old behaviour — no surprise writes into Database/."""
    import backend.api.routes.dictionary_gpu as route_mod

    master = _master(tmp_path)
    payload = {
        "master_path": str(master),
        "detector_shape": [60, 60],
        "pc": [0.5, 0.5, 0.5],
        "resolution_deg": 5.0,
    }
    client = TestClient(app)
    with patch.object(route_mod.threading, "Thread") as mock_thread:
        r = client.post("/api/dictionary-gpu/generate", json=payload)

    assert r.status_code == 200, r.text
    assert mock_thread.call_args.kwargs["args"][1].output_path is None


def test_save_to_library_without_energy_fails_loud(tmp_path):
    """energy_kv is part of the filename — guessing it would mislabel the file."""
    master = _master(tmp_path)
    payload = {
        "master_path": str(master),
        "detector_shape": [128, 156],
        "pc": [0.5, 0.5, 0.5],
        "resolution_deg": 2.0,
        "save_to_library": True,
    }
    client = TestClient(app)
    r = client.post("/api/dictionary-gpu/generate", json=payload)
    assert r.status_code == 400
    assert "energy" in r.json()["detail"].lower()


def test_generated_name_is_attributable_to_its_phase(tmp_path):
    """The Indexing page maps a dictionary to a phase card via its filename.

    DictPhaseCard filters on `formula`, which discovery fills from
    `find_linked_cif` -> filename tokens. A name whose leading token is the
    phase ("Si_…") resolves; the old "dict_gpu_<timestamp>" never could.
    """
    h5, _ = dictionary_library_paths(
        tmp_path,
        master_path="/db/Si_master_E20kV_npx500.h5",
        energy_kv=20.0,
        detector_shape=(128, 156),
        pc=(0.547, 0.465, 0.609),
        resolution_deg=2.0,
    )
    assert h5.name.split("_")[0] == "Si"
    assert "_dict_" in h5.name, "discovery classifies dictionaries on '_dict_'"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
