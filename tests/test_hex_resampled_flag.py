"""Every answer that states the scan geometry also says whether the map was resampled from a hexagonal scan.

An OIM-form hexagonal ``.h5`` is turned into a SQUARE map by nearest-point resampling on load
(``edax_hex``), so its ``scan_geometry`` is honestly ``square``; the loader tags the signal
(``Signal.hex_resampled``) and the answers carry that as ``hex_resampled``. The flag has to follow
the dataset the way the geometry does: /load, /info, /metadata, /datasets, /deepcopy, /crop.

The real resampling is exercised by ``test_edax_hex.py`` on a real file (environment variables);
here the loader's tag is put on a synthetic square scan, as ``test_scan_geometry_lifecycle.py`` does.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.apex_helpers import make_up, wipe_backend  # noqa: E402


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def fresh_backend():
    wipe_backend()
    yield
    wipe_backend()


def _square_scan(tmp_path):
    folder = tmp_path / "sq"
    folder.mkdir()
    # 3 columns x 4 rows, so that a 2x2 crop fits
    return make_up(folder / "scan_sq.up1", guid_bytes=None, version=3, hex_flag=0, n=12, nx=3, ny=4)


@pytest.fixture
def tag_resampled(monkeypatch):
    """The loader's own tag, put on the next loaded signal."""
    import safe_loader
    real = safe_loader.load_ebsd_safe

    def tagged(path, *args, **kwargs):
        sig = real(path, *args, **kwargs)
        sig.metadata.set_item("Signal.hex_resampled", True)
        return sig

    monkeypatch.setattr(safe_loader, "load_ebsd_safe", tagged)


def _load(client, up):
    r = client.post("/api/ebsd/load", json={"path": str(up)})
    assert r.status_code == 200, r.text
    return r.json()


def _datasets(client):
    return {d["name"]: d for d in client.get("/api/ebsd/datasets").json()["datasets"]}


def _crop(client, row0, col0, rows, cols):
    r = client.post("/api/ebsd/crop", json={
        "shape": "rect", "row0": row0, "col0": col0, "rows": rows, "cols": cols, "name": ""})
    assert r.status_code == 200, r.text
    return r.json()


def test_a_resampled_map_says_so_in_every_answer_that_states_the_geometry(client, tmp_path, tag_resampled):
    body = _load(client, _square_scan(tmp_path))
    assert body["scan_geometry"]["kind"] == "square"
    assert body["hex_resampled"] is True
    assert client.get("/api/ebsd/info").json()["hex_resampled"] is True
    assert client.get("/api/ebsd/metadata").json()["hex_resampled"] is True
    assert _datasets(client)["scan_sq"]["hex_resampled"] is True


def test_a_copy_and_a_crop_of_a_resampled_map_are_still_resampled(client, tmp_path, tag_resampled):
    _load(client, _square_scan(tmp_path))
    copy = client.post("/api/ebsd/deepcopy", json={"name": "twin"}).json()
    assert copy["hex_resampled"] is True
    crop = _crop(client, 1, 1, 2, 2)
    assert crop["hex_resampled"] is True
    ds = _datasets(client)
    assert ds["twin"]["hex_resampled"] is True
    assert ds[crop["name"]]["hex_resampled"] is True
    # the crop is the active dataset now: /info and /metadata read its own signal
    assert client.get("/api/ebsd/info").json()["hex_resampled"] is True
    assert client.get("/api/ebsd/metadata").json()["hex_resampled"] is True


def test_a_square_file_says_false_everywhere(client, tmp_path):
    body = _load(client, _square_scan(tmp_path))
    assert body["hex_resampled"] is False
    assert client.get("/api/ebsd/info").json()["hex_resampled"] is False
    assert client.get("/api/ebsd/metadata").json()["hex_resampled"] is False
    copy = client.post("/api/ebsd/deepcopy", json={"name": "twin"}).json()
    assert copy["hex_resampled"] is False
    crop = _crop(client, 1, 1, 2, 2)
    assert crop["hex_resampled"] is False
    ds = _datasets(client)
    assert {name: d["hex_resampled"] for name, d in ds.items()} == {
        "scan_sq": False, "twin": False, crop["name"]: False}


def test_only_the_tagged_dataset_of_a_mixed_list_is_resampled(client, tmp_path, tag_resampled):
    """The flag is read off each dataset's own signal, not off the active one."""
    from backend.api.routes import ebsd_viewer
    _load(client, _square_scan(tmp_path))
    client.post("/api/ebsd/deepcopy", json={"name": "twin"})
    ebsd_viewer._raw_signals["twin"].metadata.set_item("Signal.hex_resampled", False)
    ds = _datasets(client)
    assert ds["scan_sq"]["hex_resampled"] is True
    assert ds["twin"]["hex_resampled"] is False


def test_an_eds_only_answer_has_no_signal_to_ask_and_says_null(client, tmp_path):
    from backend.api.routes import ebsd_viewer
    answer = ebsd_viewer._load_eds_only_blocking(
        str(tmp_path / "eds_only.h5oina"), {"eds_only": True}, None, 0.0)
    assert answer["content_mode"] == "eds_only"
    assert "hex_resampled" in answer and answer["hex_resampled"] is None
