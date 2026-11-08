"""Loaders that re-open a loaded file reuse the metadata pairing the user made for it.

A measurement of an EDAX APEX file can be paired by hand with a metadata file in another folder
(``/load`` with ``apex``), and the backend remembers the pairing on the loaded-files entry. A loader
that re-opens the pattern file by path alone does not find that metadata file: it reads the header's
grid, or refuses the file as a hexagonal scan. The cross-file Pattern Match (``_get_source_signal``)
and the image-quality fallback (``image_quality_from_file``) used to be such loaders.

The measurement is a made-up hexagonal scan of 6 rows x 5 columns; its metadata file and its pattern
file are in different folders.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.apex_helpers import (GUID_A, make_apex_h5, make_up, measurement_spec,  # noqa: E402
                                wipe_backend)

ROWS, COLS = 6, 5


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app, client=("127.0.0.1", 5000))


@pytest.fixture(autouse=True)
def fresh_backend():
    from tools import pattern_comparison
    wipe_backend()
    pattern_comparison._source_signal_cache.clear()
    yield
    wipe_backend()
    pattern_comparison._source_signal_cache.clear()


@pytest.fixture
def paired(client, tmp_path):
    """A hexagonal pattern file loaded with a metadata file from another folder."""
    meta_dir, up_dir = tmp_path / "metadata", tmp_path / "patterns"
    meta_dir.mkdir()
    up_dir.mkdir()
    h5 = make_apex_h5(meta_dir / "s.h5", [measurement_spec(n_rows=ROWS, n_cols=COLS, step_x=0.5)])
    up = make_up(up_dir / "scan_hex.up1", n=ROWS * COLS, nx=COLS, ny=ROWS)
    r = client.post("/api/ebsd/load", json={
        "path": str(up), "apex": {"h5_path": str(h5), "measurement_id": GUID_A}})
    assert r.status_code == 200, r.text
    assert r.json()["scan_geometry"]["kind"] == "hex"          # the premise: the pairing works
    return up, h5


def _apex_of(signal):
    return signal.metadata.get_item("Signal.apex", None)


def test_the_cross_file_pattern_match_reads_the_file_with_its_pairing(paired):
    from tools import pattern_comparison
    up, h5 = paired
    signal = pattern_comparison._get_source_signal(str(up))
    assert signal is not None, "the hexagonal pattern file was refused: its pairing was not used"
    assert signal.axes_manager.navigation_shape == (COLS, ROWS)
    apex = _apex_of(signal)
    assert apex["measurement_id"] == GUID_A
    assert Path(apex["h5_path"]).name == "s.h5"


def test_the_image_quality_fallback_reads_the_file_with_its_pairing(paired, monkeypatch):
    from backend.api.services import pattern_quality
    up, h5 = paired
    seen = []

    def record(signal):
        seen.append(signal)
        return np.zeros(ROWS * COLS)

    monkeypatch.setattr(pattern_quality, "compute_image_quality", record)
    iq = pattern_quality.image_quality_from_file(str(up), ROWS, COLS)
    assert iq is not None and iq.shape == (ROWS, COLS), "the file could not be re-read with its pairing"
    assert _apex_of(seen[0])["measurement_id"] == GUID_A


def test_a_file_that_was_never_paired_is_loaded_as_before(tmp_path, monkeypatch):
    """No loaded-files entry, no pairing: the loader is called exactly as it was, without ``apex``."""
    import safe_loader
    from tools import pattern_comparison
    calls = []

    def fake(path, *args, **kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(safe_loader, "load_ebsd_safe", fake)
    plain = tmp_path / "plain.up1"
    plain.write_bytes(b"x")
    assert pattern_comparison._get_source_signal(str(plain)) is not None
    assert calls == [{"verbose": False}]


def test_a_remembered_pairing_that_is_gone_is_dropped_not_fatal(paired, tmp_path):
    """The metadata file was moved away: the file is loaded without the pairing, as a switch does."""
    from backend.api.routes import ebsd_viewer
    up, h5 = paired
    h5.unlink()
    assert ebsd_viewer.remembered_apex_pairing(str(up)) is None
