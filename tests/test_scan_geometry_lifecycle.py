"""The scan geometry follows a dataset through load, copy, crop, delete and file switches."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import scan_geometry  # noqa: E402
from tests.apex_helpers import make_apex_h5, make_up, measurement_spec, wipe_backend  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N_ROWS, N_COLS = 6, 5


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def fresh_backend():
    wipe_backend()
    yield
    wipe_backend()


def _hex_scan(tmp_path):
    folder = tmp_path / "hex"
    folder.mkdir()
    make_apex_h5(folder / "s.h5", [measurement_spec(n_rows=N_ROWS, n_cols=N_COLS, step_x=0.5)])
    return make_up(folder / "scan_hex.up1", n=N_ROWS * N_COLS, nx=N_COLS, ny=N_ROWS)


def _square_scan(tmp_path):
    folder = tmp_path / "sq"
    folder.mkdir()
    return make_up(folder / "scan_sq.up1", guid_bytes=None, version=3, hex_flag=0, n=6, nx=3, ny=2)


def _load(client, up):
    r = client.post("/api/ebsd/load", json={"path": str(up)})
    assert r.status_code == 200, r.text
    return r.json()


def _datasets(client):
    return {d["name"]: d for d in client.get("/api/ebsd/datasets").json()["datasets"]}


def _crop(client, row0, col0, rows, cols, name=""):
    r = client.post("/api/ebsd/crop", json={
        "shape": "rect", "row0": row0, "col0": col0, "rows": rows, "cols": cols, "name": name})
    assert r.status_code == 200, r.text
    return r.json()


def test_a_hex_load_reports_its_geometry_everywhere(client, tmp_path):
    body = _load(client, _hex_scan(tmp_path))
    g = body["scan_geometry"]
    assert (g["kind"], g["grid_shape"], g["odd_row_shift"], g["shift_confirmed"]) == (
        "hex", [N_ROWS, N_COLS], 0.5, True)
    assert body["pc_origin"] == "apex" and body["grid_shape"] == [N_ROWS, N_COLS]
    assert client.get("/api/ebsd/info").json()["scan_geometry"] == g
    assert client.get("/api/ebsd/metadata").json()["scan_geometry"] == g
    assert _datasets(client)["scan_hex"]["scan_geometry"] == g


def test_a_square_load_reports_a_square_geometry(client, tmp_path):
    g = _load(client, _square_scan(tmp_path))["scan_geometry"]
    assert (g["kind"], g["grid_shape"], g["odd_row_shift"]) == ("square", [2, 3], 0.0)


def test_the_load_answer_says_when_a_hexagonal_file_was_resampled_to_a_square_map(client, tmp_path, monkeypatch):
    """An OIM-form hexagonal .h5 is still read through the resampler (its geometry is "square", which
    is what is held); the answer must still say it is not what was measured. The real resampling is
    exercised by test_edax_hex.py on a real file; here the loader's tag is put on a square scan."""
    import safe_loader
    real = safe_loader.load_ebsd_safe

    def tagged(path, *args, **kwargs):
        sig = real(path, *args, **kwargs)
        sig.metadata.set_item("Signal.hex_resampled", True)
        return sig

    up = _square_scan(tmp_path)
    plain = _load(client, up)
    assert plain["hex_resampled"] is False
    monkeypatch.setattr(safe_loader, "load_ebsd_safe", tagged)
    wipe_backend()
    resampled = _load(client, up)
    assert resampled["hex_resampled"] is True
    assert resampled["scan_geometry"]["kind"] == "square"


def test_the_loaders_copy_is_gone_from_the_signal_after_the_load_so_no_crop_inherits_it(client, tmp_path):
    from backend.api.routes import ebsd_viewer
    _load(client, _hex_scan(tmp_path))
    assert not ebsd_viewer._raw_signals["scan_hex"].metadata.has_item("Signal.scan_geometry")
    crop = _crop(client, 1, 1, 3, 3)
    client.post("/api/ebsd/select-dataset", json={"name": crop["name"]})
    crop_signal = ebsd_viewer._get_active_signal()
    assert crop_signal is not ebsd_viewer._raw_signals["scan_hex"]
    assert not crop_signal.metadata.has_item("Signal.scan_geometry")
    # a later registration from the crop reads its own axes, it does not trip over the parent's copy
    assert scan_geometry.register_from_signal("probe", crop_signal).grid_shape == (3, 3)


def test_a_crop_on_an_odd_row_reports_the_reversed_offset(client, tmp_path):
    _load(client, _hex_scan(tmp_path))
    odd = _crop(client, 1, 1, 3, 3)
    assert odd["scan_geometry"]["odd_row_shift"] == -0.5
    assert odd["scan_geometry"]["grid_shape"] == [3, 3]
    ds = _datasets(client)
    assert ds["scan_hex"]["scan_geometry"]["odd_row_shift"] == 0.5          # the parent did not move
    assert ds[odd["name"]]["scan_geometry"]["odd_row_shift"] == -0.5


def test_a_crop_on_an_even_row_keeps_the_offset(client, tmp_path):
    _load(client, _hex_scan(tmp_path))
    even = _crop(client, 2, 1, 3, 3)
    assert even["scan_geometry"]["odd_row_shift"] == 0.5


def test_a_crop_of_a_crop_is_cut_from_the_crop(client, tmp_path):
    _load(client, _hex_scan(tmp_path))
    first = _crop(client, 1, 0, 4, 5, name="first")                  # row0 1: reversed
    second = _crop(client, 1, 0, 2, 5, name="second")                # row0 1 of the crop: reversed back
    assert first["scan_geometry"]["odd_row_shift"] == -0.5
    assert second["scan_geometry"]["odd_row_shift"] == 0.5           # = a crop at row 2 of the scan
    assert second["window"]["row0"] == 2


def test_a_deepcopy_sits_where_its_parent_does(client, tmp_path):
    _load(client, _hex_scan(tmp_path))
    crop = _crop(client, 1, 1, 3, 3)
    client.post("/api/ebsd/select-dataset", json={"name": crop["name"]})
    copy = client.post("/api/ebsd/deepcopy", json={"name": "twin"}).json()
    assert copy["scan_geometry"]["odd_row_shift"] == -0.5
    assert _datasets(client)["twin"]["scan_geometry"]["grid_shape"] == [3, 3]


def test_the_stash_brings_a_crop_back_with_its_geometry(client, tmp_path):
    hex_up = _hex_scan(tmp_path)
    _load(client, hex_up)
    crop = _crop(client, 1, 1, 3, 3)
    _load(client, _square_scan(tmp_path))                              # a new file: the registry is cleared
    assert scan_geometry.geometry_for(crop["name"]) is None
    assert scan_geometry.geometry_for("scan_hex") is None
    r = client.post("/api/ebsd/switch-file", json={"path": str(hex_up)})
    assert r.status_code == 200, r.text
    ds = _datasets(client)
    assert ds["scan_hex"]["scan_geometry"]["odd_row_shift"] == 0.5
    assert ds[crop["name"]]["scan_geometry"]["odd_row_shift"] == -0.5
    assert ds[crop["name"]]["scan_geometry"]["grid_shape"] == [3, 3]


def test_deleting_a_dataset_drops_its_geometry(client, tmp_path):
    _load(client, _hex_scan(tmp_path))
    crop = _crop(client, 1, 1, 3, 3)
    assert scan_geometry.geometry_for(crop["name"]) is not None
    assert client.delete(f"/api/ebsd/dataset/{crop['name']}").status_code == 200
    assert scan_geometry.geometry_for(crop["name"]) is None
    assert scan_geometry.geometry_for("scan_hex") is not None


def test_a_new_file_starts_with_nothing_of_the_old_one(client, tmp_path):
    _load(client, _hex_scan(tmp_path))
    crop = _crop(client, 1, 1, 3, 3)
    _load(client, _square_scan(tmp_path))
    assert scan_geometry.geometry_for(crop["name"]) is None
    assert scan_geometry.geometry_for("scan_sq").kind == "square"


def test_a_crop_that_does_not_fit_the_registered_geometry_registers_nothing(client, tmp_path):
    """The geometry is cut before anything is registered: a refusal leaves no orphan behind."""
    from backend.api.routes import ebsd_viewer
    from backend.api.services.calibration_store import calibration_store
    _load(client, _hex_scan(tmp_path))
    # the registry is made to disagree with the 6 x 5 signal: a 2 x 2 geometry cannot be cut at 1,1,3,3
    scan_geometry.set_geometry("scan_hex", scan_geometry.ScanGeometry.hex(2, 2, 0.5))
    r = client.post("/api/ebsd/crop", json={
        "shape": "rect", "row0": 1, "col0": 1, "rows": 3, "cols": 3, "name": "refused"})
    assert r.status_code == 400, r.text
    assert "does not fit" in r.json()["detail"]
    assert "refused" not in ebsd_viewer._raw_signals
    assert calibration_store.get_entry("refused") is None
    assert scan_geometry.geometry_for("refused") is None
    assert ebsd_viewer._active_dataset == "scan_hex"


def test_a_crop_refused_by_the_pc_map_leaves_no_geometry_behind(client, tmp_path):
    """register_cropped is the second call that can refuse; the geometry cut before it is dropped."""
    import numpy as np
    from backend.api.routes import ebsd_viewer
    from backend.api.services.calibration_store import calibration_store
    _load(client, _hex_scan(tmp_path))
    entry = calibration_store.get_entry("scan_hex")
    entry.pc_map = np.zeros((2, 2, 3))                  # not on the 6 x 5 grid the window is cut from
    r = client.post("/api/ebsd/crop", json={
        "shape": "rect", "row0": 1, "col0": 1, "rows": 3, "cols": 3, "name": "refused"})
    assert r.status_code == 400, r.text
    assert "refused" not in ebsd_viewer._raw_signals
    assert calibration_store.get_entry("refused") is None
    assert scan_geometry.geometry_for("refused") is None


def test_the_batch_loader_registers_its_file_and_forgets_the_previous_one(client, tmp_path):
    """load_ebsd_file is the synchronous helper batch indexing uses: same registry, same lifecycle."""
    from backend.api.routes import ebsd_viewer
    _load(client, _square_scan(tmp_path))
    assert scan_geometry.geometry_for("scan_sq") is not None
    assert ebsd_viewer.load_ebsd_file(str(_hex_scan(tmp_path))) is True
    g = scan_geometry.geometry_for("scan_hex")
    assert (g.kind, g.grid_shape, g.odd_row_shift) == ("hex", (N_ROWS, N_COLS), 0.5)
    assert scan_geometry.geometry_for("scan_sq") is None


def test_an_eds_only_file_has_no_scan_geometry_and_the_previous_one_is_forgotten(client, tmp_path):
    """The EDS-only path holds no EBSD signal: the answer says null and nothing of the old file stays."""
    from backend.api.routes import ebsd_viewer
    _load(client, _hex_scan(tmp_path))
    assert scan_geometry.geometry_for("scan_hex") is not None
    answer = ebsd_viewer._load_eds_only_blocking(
        str(tmp_path / "eds_only.h5oina"), {"eds_only": True}, None, 0.0)
    assert answer["content_mode"] == "eds_only"
    assert "scan_geometry" in answer and answer["scan_geometry"] is None
    assert scan_geometry.geometry_for("scan_hex") is None


# --------------------------------------------------------- structural guards

def _source(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_every_place_that_clears_the_crop_registry_clears_the_geometry_registry():
    """A fourth load path added later must not forget the geometry: the crop hook is the cue."""
    src = _source("backend/api/routes/ebsd_viewer.py")
    assert src.count("crop_window_service.clear_all()") == src.count("scan_geometry_service.clear_all()") == 3


def test_every_lifecycle_hook_is_called_where_the_crop_hooks_are():
    src = _source("backend/api/routes/ebsd_viewer.py")
    assert src.count("scan_geometry_service.register_from_signal(") == 2     # both load paths
    assert src.count("scan_geometry_service.derive(") == 1                   # deepcopy
    assert src.count("scan_geometry_service.derive_cropped(") == 1           # crop
    assert src.count("scan_geometry_service.clear(") == 2                    # delete, and a crop refused after its geometry was cut
    assert src.count("scan_geometry_service.snapshot(") == 2                 # both stash dicts
    assert src.count("scan_geometry_service.restore(") == 1


def test_only_one_function_reads_the_signals_copy_of_the_geometry():
    """The mirror on the signal is a transport from the loader to the registry: read once."""
    readers = []
    for path in list(ROOT.glob("*.py")) + list((ROOT / "backend").rglob("*.py")) \
            + list((ROOT / "analysis").rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"get_item\(\s*[\"']Signal\.scan_geometry[\"']", text):
            readers.append(path.relative_to(ROOT).as_posix())
    assert readers == ["backend/api/services/scan_geometry.py"]
