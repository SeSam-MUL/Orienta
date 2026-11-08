"""A hexagonal scan is refused, by name, wherever a square one is assumed."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest
from fastapi import APIRouter, Depends, FastAPI, HTTPException, WebSocket
from fastapi.testclient import TestClient
from starlette.requests import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.problem import CODE_HEADER  # noqa: E402
from backend.api.services import hex_guard, scan_geometry  # noqa: E402
from backend.api.services.hex_guard import GUARDED, hex_gate  # noqa: E402
from backend.api.services.scan_geometry import ScanGeometry  # noqa: E402
from tests.apex_helpers import wipe_backend  # noqa: E402
from tests.test_crop_provenance import _store_dummy_result  # noqa: E402

HEX = ScanGeometry.hex(4, 4, 0.5).to_dict()
SQUARE = ScanGeometry.square(4, 4, 0.5).to_dict()
IDS = [f"{g.method} {g.path}" for g in GUARDED]

#: The crystal hint's three routes that average a (2r+1) x (2r+1) window of patterns on the index grid.
CRYSTAL_HINT_PATHS = ("/api/crystal-hint/analyze-pixel", "/api/crystal-hint/analyze-region",
                      "/api/crystal-hint/quality-check")


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def fresh_backend():
    from backend.api.routes import analysis, indexing
    wipe_backend()
    indexing._result_registry.clear()
    indexing._active_result_id = None
    analysis._dataset = None
    yield
    wipe_backend()
    indexing._result_registry.clear()
    indexing._active_result_id = None
    analysis._dataset = None


def _arm(basis, geometry):
    """Make ``geometry`` (a dict, or None = unknown) the geometry the entry's basis looks at."""
    if basis == "dataset":
        from backend.api.routes import ebsd_viewer
        if geometry is not None:
            scan_geometry.set_geometry("ds", ScanGeometry.from_dict(geometry))
        ebsd_viewer._active_dataset = "ds"
    elif basis == "analysis":
        from backend.api.routes import analysis
        analysis._dataset = SimpleNamespace(scan_geometry=geometry, grains=None, shape=(4, 4))
    else:
        _store_dummy_result({"scan_geometry": geometry})


def _trigger(entry):
    """(url, query, body) that make the entry's predicate true."""
    url = entry.path.replace("{map_type}", "kam")
    query, body = {}, None
    if entry.path in CRYSTAL_HINT_PATHS:
        body = {"avg_radius": 1}
    elif entry.method == "POST":
        body = {}
    return url, query, body


def _gate(entry, url, query, body):
    """Run the gate itself on a request, without going through the route behind it."""
    payload = json.dumps(body).encode() if body is not None else b""

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    scope = {
        "type": "http", "method": entry.method, "path": url,
        "query_string": urlencode(query).encode(),
        "headers": [(b"content-type", b"application/json")],
        "route": SimpleNamespace(path=entry.path),
        "path_params": {"map_type": "kam"} if "{map_type}" in entry.path else {},
    }
    return asyncio.run(hex_gate(Request(scope, receive)))


# ------------------------------------------------------------ the table is real

def test_every_entry_names_a_real_route_that_has_the_gate_installed():
    from backend.api.main import app
    for entry in GUARDED:
        routes = [r for r in app.routes
                  if getattr(r, "path", None) == entry.path and entry.method in getattr(r, "methods", ())]
        assert len(routes) == 1, f"{entry.method} {entry.path} is not a route of the app"
        assert any(d.call is hex_gate for d in routes[0].dependant.dependencies), (
            f"{entry.method} {entry.path} is in GUARDED but its router does not run hex_gate")


def test_every_entry_has_a_message_for_its_code():
    for entry in GUARDED:
        assert entry.code in hex_guard.MESSAGES and "{what}" in hex_guard.MESSAGES[entry.code]


# ------------------------------------------------------------------- hex: 400

@pytest.mark.parametrize("entry", GUARDED, ids=IDS)
def test_every_guarded_endpoint_refuses_a_hex_scan(client, entry):
    _arm(entry.basis, HEX)
    url, query, body = _trigger(entry)
    resp = client.request(entry.method, url, params=query, json=body)
    assert resp.status_code == 400, resp.text
    assert resp.headers[CODE_HEADER.lower()] == entry.code
    assert entry.what in resp.json()["detail"]


# --------------------------------------------------- square and unknown: untouched

@pytest.mark.parametrize("entry", GUARDED, ids=IDS)
def test_a_square_scan_is_not_refused(entry):
    _arm(entry.basis, SQUARE)
    assert _gate(entry, *_trigger(entry)) is None


@pytest.mark.parametrize("entry", GUARDED, ids=IDS)
def test_an_unknown_geometry_is_not_refused(entry):
    _arm(entry.basis, None)
    assert _gate(entry, *_trigger(entry)) is None


def test_with_nothing_loaded_the_gate_lets_the_route_report_it():
    for entry in GUARDED:
        assert _gate(entry, *_trigger(entry)) is None


# --------------------------------------------------------------- the predicates

def _entry(path):
    return next(g for g in GUARDED if g.path == path)


@pytest.mark.parametrize("map_type", ["kam", "gos", "ecd", "rx", "grain-boundaries", "bc", "ipf_z", "texture_component"])
def test_every_analysis_map_is_refused_for_a_hexagonal_dataset(map_type):
    # The maps that do not compare neighbours (bc, ipf_*) are index-grid pictures all the same, which
    # a hexagonal dataset must not be drawn on; the whole page is refused until analysis/ is reworked.
    entry = _entry("/api/analysis/map/{map_type}")
    _arm("analysis", HEX)
    request = Request({
        "type": "http", "method": "GET", "path": entry.path.replace("{map_type}", map_type), "query_string": b"",
        "headers": [], "route": SimpleNamespace(path=entry.path), "path_params": {"map_type": map_type}},
        lambda: None)
    with pytest.raises(HTTPException) as caught:
        asyncio.run(hex_gate(request))
    assert caught.value.headers[CODE_HEADER] == hex_guard.ANALYSIS_PAGE
    # bc and ipf_* compare no neighbours, so the sentence must not say the page does
    assert "neighbour" not in caught.value.detail
    assert "cannot be drawn for a hexagonal scan yet" in caught.value.detail


@pytest.mark.parametrize("path", CRYSTAL_HINT_PATHS)
@pytest.mark.parametrize("radius, refused", [
    (0, False), ("0", False), (None, False),                     # one pattern, nothing averaged
    (1, True), (2, True), (3, True), ("1", True),                # a window of 9, 25, 49 index points
])
def test_the_crystal_hints_neighbourhood_average_is_refused_on_a_hex_scan_from_radius_one(path, radius, refused):
    """The window is (2r+1)^2 index points, up to 1.8 steps away on the hexagonal index grid, and
    misses the true neighbours; the crystal hint does not average on the neighbour table. Radius 0 is one pattern."""
    entry = _entry(path)
    body = {"pixel_index": 0} if radius is None else {"pixel_index": 0, "avg_radius": radius}
    _arm("dataset", HEX)
    if refused:
        with pytest.raises(HTTPException) as ei:
            _gate(entry, entry.path, {}, body)
        assert ei.value.headers[CODE_HEADER] == hex_guard.NEIGHBOURHOOD
    else:
        assert _gate(entry, entry.path, {}, body) is None
    _arm("dataset", SQUARE)
    assert _gate(entry, entry.path, {}, body) is None             # a square scan: as before


def test_the_crystal_hint_averaging_reaches_the_gate_on_the_real_routes(client):
    _arm("dataset", HEX)
    averaged = client.post("/api/crystal-hint/analyze-pixel",
                           json={"pixel_index": 0, "elements": [], "avg_radius": 1})
    assert averaged.status_code == 400, averaged.text
    assert averaged.headers[CODE_HEADER.lower()] == hex_guard.NEIGHBOURHOOD
    assert "neighbourhood averaging" in averaged.json()["detail"]
    single = client.post("/api/crystal-hint/analyze-pixel",
                         json={"pixel_index": 0, "elements": [], "avg_radius": 0})
    assert single.headers.get(CODE_HEADER.lower()) != hex_guard.NEIGHBOURHOOD   # the route's own business


def test_the_result_the_request_names_decides_not_the_active_one(client):
    from backend.api.routes import indexing
    square_id, _ = _store_dummy_result({"scan_geometry": SQUARE})
    hex_id, _ = _store_dummy_result({"scan_geometry": HEX})            # the active one now
    entry = _entry("/api/refinement/compute")
    ok = _gate(entry, entry.path, {}, {"result_id": square_id})
    assert ok is None
    with pytest.raises(HTTPException):
        _gate(entry, entry.path, {}, {"result_id": hex_id})
    assert indexing._active_result_id == hex_id
    resp = client.post(entry.path, json={"result_id": hex_id})
    assert resp.status_code == 400 and resp.headers[CODE_HEADER.lower()] == hex_guard.NEIGHBOURHOOD


def test_a_malformed_result_id_is_the_routes_422_not_a_500_in_the_gate(client):
    """A list is unhashable: looked up in the registry it would raise ``TypeError``."""
    for bad in ([1], {"a": 1}, 7):
        resp = client.post("/api/refinement/compute", json={"result_id": bad})
        assert resp.status_code == 422, (bad, resp.status_code, resp.text)


def test_the_gate_leaves_a_websocket_route_of_a_gated_router_alone():
    router = APIRouter(dependencies=[Depends(hex_gate)])

    @router.websocket("/ws-probe")
    async def probe(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_text("hello " + await websocket.receive_text())
        await websocket.close()

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as ws_client, ws_client.websocket_connect("/ws-probe") as ws:
        ws.send_text("gate")
        assert ws.receive_text() == "hello gate"


def test_the_messages_say_what_is_missing_and_what_to_do():
    assert "six neighbours" in hex_guard.MESSAGES[hex_guard.NEIGHBOURHOOD]
    assert "global pattern centre" in hex_guard.MESSAGES[hex_guard.PC_GRID]
    assert "index grid" in hex_guard.MESSAGES[hex_guard.ANALYSIS_PAGE]
    assert "neighbour" not in hex_guard.MESSAGES[hex_guard.ANALYSIS_PAGE]


def test_saving_a_hexagonal_result_is_not_refused_by_the_gate(client, tmp_path):
    """Saving writes the hexagonal grid as measured (HexGrid .ang, Orienta .h5), so the gate does not stand
    in front of ``POST /api/indexing/export`` any more: not in the table, not at the door."""
    assert not any(g.path == "/api/indexing/export" for g in GUARDED)
    _arm("result", HEX)
    entry = SimpleNamespace(method="POST", path="/api/indexing/export")
    assert _gate(entry, entry.path, {}, {"format": "ang"}) is None            # the gate lets it through
    resp = client.post(entry.path, json={"format": "nonsense", "filename": str(tmp_path / "x.out")})
    # the route answers for itself (an unknown format), not with a code of the gate
    assert resp.headers.get(CODE_HEADER.lower()) not in {g.code for g in GUARDED}
    assert "hex_export_pending" not in resp.text


def test_every_code_has_a_text_in_the_four_locales_of_the_interface():
    """The interface shows ``errors.<code>`` of the ``common`` namespace, not the English prose."""
    import json
    from pathlib import Path
    locales = Path(__file__).resolve().parents[1] / "frontend" / "src" / "locales"
    codes = set(hex_guard.MESSAGES) | {g.code for g in GUARDED}
    assert hex_guard.ANALYSIS_PAGE in codes
    for lang in ("en", "de", "ja", "zh"):
        errors = json.loads((locales / lang / "common.json").read_text(encoding="utf-8"))["errors"]
        missing = sorted(c for c in codes if not isinstance(errors.get(c), str) or len(errors[c]) < 30)
        assert missing == [], f"{lang}/common.json lacks errors.<code> for {missing}"


# ---------------------------------------------------------------- loading by path (the batch)

def _fake_signal(geometry):
    """A stand-in with what ``geometry_of_signal`` reads: the metadata mirror and the axes."""
    axes = [SimpleNamespace(size=4, scale=0.5, units="um") for _ in range(2)]
    metadata = SimpleNamespace(get_item=lambda key, default=None: geometry if geometry else default)
    return SimpleNamespace(metadata=metadata, axes_manager=SimpleNamespace(navigation_axes=axes))


def test_the_batch_no_longer_refuses_a_hex_scan_at_load(monkeypatch):
    """The load hands the signal on as it is; the run reads its geometry (tests/test_batch_hex_run.py)."""
    import safe_loader
    from backend.api.services import batch_manager
    signal = _fake_signal(HEX)
    monkeypatch.setattr(safe_loader, "load_ebsd_safe", lambda path: signal)
    assert batch_manager._load_signal("scan.up1") is signal


def test_the_batch_loads_a_square_scan_as_before(monkeypatch):
    import safe_loader
    from backend.api.services import batch_manager
    signal = _fake_signal(None)
    monkeypatch.setattr(safe_loader, "load_ebsd_safe", lambda path: signal)
    assert batch_manager._load_signal("scan.h5oina") is signal


# ------------------------------------------- the indexing batch: auto-export of a hex scan

def _batch_over_a_synthetic_up1(client, monkeypatch, tmp_path, *, hex_flag, during_index=None):
    """Drive the real ``POST /api/indexing/batch/start`` over a synthetic UP1 file (hexagonal or
    square by ``hex_flag``) with auto-export on and a fresh export folder; the indexer is a stub
    that first calls ``during_index()`` (if given), i.e. disturbs the live state mid-item.
    Returns (the batch's result entry, the export folder, the number of stored results, the log)."""
    import numpy as np
    import indexing_controller
    from indexing_controller import IndexingMethod, IndexingResult
    from backend.api.routes import indexing as I
    from tests.apex_helpers import make_apex_h5, make_up, measurement_spec

    n_rows, n_cols = 6, 5
    folder = tmp_path / ("hexbatch" if hex_flag else "squarebatch")
    folder.mkdir()
    make_apex_h5(folder / "s.h5", [measurement_spec(
        grid="HexGrid" if hex_flag else "SqrGrid", n_rows=n_rows, n_cols=n_cols, step_x=0.5)])
    up = make_up(folder / "scan.up1", n=n_rows * n_cols, nx=n_cols, ny=n_rows,
                 hex_flag=1 if hex_flag else 0)
    export_dir = tmp_path / ("export_hex" if hex_flag else "export_square")
    export_dir.mkdir()

    def stub_spherical(h5_path, config, detector_params, selection_mask, **kwargs):
        if during_index is not None:
            during_index()
        mask = np.asarray(selection_mask, dtype=bool)
        return IndexingResult(
            xmap=None, selection_mask=mask, original_shape=mask.shape,
            method=IndexingMethod.SPHERICAL,
            confidence_scores=np.full(mask.size, 0.5, dtype=np.float32))

    for name in ("spherical_index_patterns", "spherical_gpu_index_patterns"):
        monkeypatch.setattr(indexing_controller, name, stub_spherical)
    monkeypatch.setattr(I, "_attach_indexing_metadata", lambda *a, **k: None)
    monkeypatch.setattr(I, "_submit_run_guarded", lambda fn, kind="": fn())
    saved_batch, saved_results = dict(I._batch_state), dict(I._result_registry)
    try:
        resp = client.post("/api/indexing/batch/start", json={
            "datasets": [{"file_path": str(up), "method": "spherical", "selection_mode": "full"}],
            "auto_export": True, "export_dir": str(export_dir), "cleanup_after_export": False})
        assert resp.status_code == 200, resp.text
        entry = dict(I._batch_state["results"][-1])
        log = list(I._batch_state.get("log", []))
        return entry, export_dir, len(I._result_registry), log
    finally:
        I._batch_state.clear()
        I._batch_state.update(saved_batch)
        I._result_registry.clear()
        I._result_registry.update(saved_results)


def _index_stamp(path):
    """``(grid_type, grid_shape)`` the file's /Indexing states, or ``(None, shape)`` for a square scan."""
    import h5py
    with h5py.File(path, "r") as f:
        attrs = f["Indexing"].attrs
        grid_type = attrs.get("grid_type")
        grid_type = grid_type.decode() if isinstance(grid_type, bytes) else grid_type
        return grid_type, list(attrs["grid_shape"])


def test_the_indexing_batch_writes_the_file_of_a_hex_scan(client, monkeypatch, tmp_path):
    entry, export_dir, n_results, _log = _batch_over_a_synthetic_up1(
        client, monkeypatch, tmp_path, hex_flag=True)
    assert entry["error"] is None, entry
    assert entry["export_error"] is None
    assert entry["export_path"] is not None and Path(entry["export_path"]).parent == export_dir
    assert _index_stamp(entry["export_path"]) == ("HexGrid", [6, 5])    # the 6 x 5 scan the file states
    assert n_results == 1


def test_the_indexing_batch_still_exports_a_square_scan(client, monkeypatch, tmp_path):
    entry, export_dir, n_results, _log = _batch_over_a_synthetic_up1(
        client, monkeypatch, tmp_path, hex_flag=False)
    assert entry["error"] is None, entry
    assert entry["export_error"] is None
    assert entry["export_path"] is not None and Path(entry["export_path"]).is_file()
    assert Path(entry["export_path"]).parent == export_dir
    assert _index_stamp(entry["export_path"]) == (None, [6, 5])
    assert n_results == 1


def test_a_batch_export_that_writes_no_file_says_so_in_the_entry(client, monkeypatch, tmp_path):
    """A writer that fails leaves no file; the batch entry carries the reason, not only the log."""
    from backend.api.routes import indexing as I

    def refuse(*a, **k):
        raise OSError("the disk is full")

    monkeypatch.setattr(I, "_write_batch_indexed_h5", refuse)
    entry, export_dir, _n, _log = _batch_over_a_synthetic_up1(client, monkeypatch, tmp_path, hex_flag=False)
    assert entry["error"] is None and entry["export_path"] is None
    assert "the disk is full" in entry["export_error"]


def _activate_other_dataset(geometry):
    """Mid-item disturbance: another dataset with ``geometry`` becomes the active one."""
    from backend.api.routes import ebsd_viewer

    def disturb():
        scan_geometry.set_geometry("other_dataset", ScanGeometry.from_dict(geometry))
        ebsd_viewer._active_dataset = "other_dataset"
    return disturb


def test_the_export_follows_the_items_stamp_not_a_dataset_activated_mid_item(
        client, monkeypatch, tmp_path):
    """The item is hexagonal; while it runs a SQUARE dataset becomes the active one. A live read at
    the export would see square and write a square file for a hexagonal scan."""
    entry, export_dir, n_results, _log = _batch_over_a_synthetic_up1(
        client, monkeypatch, tmp_path, hex_flag=True, during_index=_activate_other_dataset(SQUARE))
    assert entry["error"] is None, entry
    assert entry["export_error"] is None
    assert _index_stamp(entry["export_path"]) == ("HexGrid", [6, 5])
    assert n_results == 1


def test_a_square_items_export_stays_square_although_a_hex_dataset_was_activated_mid_item(
        client, monkeypatch, tmp_path):
    """The reverse: the item is square, a HEXAGONAL dataset becomes active mid-item; the export
    follows the item's own stamp and carries no hexagonal block."""
    entry, export_dir, n_results, _log = _batch_over_a_synthetic_up1(
        client, monkeypatch, tmp_path, hex_flag=False, during_index=_activate_other_dataset(HEX))
    assert entry["error"] is None, entry
    assert entry["export_error"] is None
    assert _index_stamp(entry["export_path"]) == (None, [6, 5])
    assert n_results == 1


# ------------------------------------------- a crop needs the geometry of what it cuts

def test_a_crop_of_a_dataset_without_a_recorded_geometry_is_refused_by_name(
        client, tmp_path, monkeypatch):
    """Every load path registers a geometry, so a missing one is a broken contract: the cut-out's
    geometry would be unknown, and an unknown geometry passes the hexagonal gate. Nothing may be
    published for the refused crop, and the (possibly large) cut is not even started."""
    from backend.api.routes import ebsd_viewer
    from backend.api.services import crop_window
    from backend.api.services.calibration_store import calibration_store
    from tests.apex_helpers import make_apex_h5, make_up, measurement_spec

    folder = tmp_path / "nogeom"
    folder.mkdir()
    make_apex_h5(folder / "s.h5", [measurement_spec(n_rows=6, n_cols=5, step_x=0.5)])
    up = make_up(folder / "scan.up1", n=30, nx=5, ny=6)
    assert client.post("/api/ebsd/load", json={"path": str(up)}).status_code == 200
    parent = ebsd_viewer._active_dataset
    assert scan_geometry.geometry_for(parent) is not None        # premise: the load registered one
    scan_geometry.clear(parent)                                  # the broken contract

    raw_before = dict(ebsd_viewer._raw_signals)
    windows_before = dict(crop_window._windows)
    calibration_before = dict(calibration_store._entries)
    geometries_before = dict(scan_geometry._geometries)
    assert geometries_before == {}                               # the cut would have no parent to read

    cuts = []                  # the crop runs its cut (``_cut``) through ``asyncio.to_thread``
    real_to_thread = ebsd_viewer.asyncio.to_thread

    async def recording_to_thread(func, *args, **kwargs):
        cuts.append(getattr(func, "__name__", repr(func)))
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(ebsd_viewer.asyncio, "to_thread", recording_to_thread)

    resp = client.post("/api/ebsd/crop", json={
        "shape": "rect", "row0": 1, "col0": 1, "rows": 3, "cols": 3})
    assert cuts == [], "the cut ran before the refusal"
    assert resp.status_code == 400, resp.text
    assert resp.headers[CODE_HEADER.lower()] == "crop_geometry_unknown"
    assert parent in resp.json()["detail"] and "no recorded scan geometry" in resp.json()["detail"]

    assert ebsd_viewer._active_dataset == parent
    assert ebsd_viewer._raw_signals == raw_before
    assert crop_window._windows == windows_before
    assert calibration_store._entries == calibration_before
    assert scan_geometry._geometries == geometries_before


# ------------------------------------------------ NLPAR on a hexagonal scan of any source

NLPAR_URL = "/api/ebsd/nlpar"


def _register_scan(geometry):
    """A real noisy 12 x 14 scan as the viewer's active dataset, with ``geometry`` (a dict, or None
    = unregistered) as its recorded scan geometry. Returns the signal and a copy of its patterns."""
    import kikuchipy as kp
    from backend.api.routes import ebsd_viewer as ev
    from tests.test_nlpar_service import make_scan

    raw = make_scan()
    signal = kp.signals.EBSD(raw.copy())
    ev._raw_signals["ds"] = signal
    ev._positions["ds"] = (0, 0)
    ev._active_dataset = "ds"
    ev._ebsd_signal = signal
    ev._ebsd_file_path = None                    # not an OIM hexagonal .h5: only the registry knows
    if geometry is not None:
        scan_geometry.set_geometry("ds", ScanGeometry.from_dict(geometry))
    return signal, raw


def test_nlpar_is_in_the_table_before_indexing_with_the_neighbourhood_code():
    entry = next(g for g in GUARDED if g.path == NLPAR_URL)
    assert (entry.method, entry.basis, entry.code, entry.stage, entry.when) == (
        "POST", "dataset", hex_guard.NEIGHBOURHOOD, "before_indexing", None)


def test_nlpar_on_a_registered_hex_dataset_is_refused_and_the_patterns_stay_as_they_were(client):
    import numpy as np
    pytest.importorskip("pyebsdindex")
    pytest.importorskip("numba")
    signal, raw = _register_scan(HEX)
    resp = client.post(NLPAR_URL, json={"radius": 3})
    assert resp.status_code == 400, resp.text
    assert resp.headers[CODE_HEADER.lower()] == hex_guard.NEIGHBOURHOOD
    assert "NLPAR denoising" in resp.json()["detail"]
    assert np.array_equal(np.asarray(signal.data), raw)              # nothing was averaged
    assert signal.metadata.get_item("Orienta.nlpar", None) is None   # and nothing was marked


def test_nlpar_on_a_square_dataset_gets_past_the_gate_to_its_own_validation(client):
    """A square dataset is the route's: a radius it refuses proves the gate let the request in
    (and nothing is denoised)."""
    pytest.importorskip("pyebsdindex")
    pytest.importorskip("numba")
    _register_scan(SQUARE)
    resp = client.post(NLPAR_URL, json={"radius": 99})
    assert resp.status_code == 400 and CODE_HEADER.lower() not in resp.headers
    assert "Search radius" in resp.json()["detail"]


def test_nlpar_advice_says_hex_for_a_registered_hex_dataset_of_any_source(client):
    pytest.importorskip("pyebsdindex")
    pytest.importorskip("numba")
    _register_scan(HEX)
    out = client.get("/api/ebsd/nlpar/advice").json()
    assert (out["recommendation"], out["reason"]) == ("unavailable", "hex")


def test_nlpar_advice_does_not_say_hex_for_a_square_dataset(client):
    pytest.importorskip("pyebsdindex")
    pytest.importorskip("numba")
    _register_scan(SQUARE)
    out = client.get("/api/ebsd/nlpar/advice").json()
    assert out.get("reason") != "hex" and out["recommendation"] in ("recommended", "not_recommended")


GRAIN_TOOLS = (
    ("GET", "/api/indexing/pattern-match/variants"),
    ("POST", "/api/indexing/pattern-match/apply-to-grain"),
    ("POST", "/api/indexing/pattern-match/assign-phase"),
    ("POST", "/api/indexing/island-check"),
    ("POST", "/api/indexing/phase-check"),
    ("POST", "/api/indexing/phase-reassign"),
)


@pytest.mark.parametrize("method, path", GRAIN_TOOLS, ids=[p for _, p in GRAIN_TOOLS])
def test_a_hexagonal_result_gets_past_the_gate_on_the_grain_tools(client, method, path):
    """These six work on the six neighbours of a hexagonal scan: whatever the route answers for this
    stand-in result, it is not the gate's refusal."""
    result_id, _ = _store_dummy_result({"scan_geometry": HEX})
    if method == "GET":
        resp = client.get(path, params={"row": 0, "col": 0, "result_id": result_id})
    else:
        resp = client.post(path, json={"result_id": result_id, "row": 0, "col": 0, "quat": [1, 0, 0, 0],
                                       "target_phase_id": 0})
    assert CODE_HEADER.lower() not in resp.headers
    assert path not in {g.path for g in GUARDED}
