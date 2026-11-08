"""The batch writes the files of a hexagonal scan as it writes those of a square one.

The export used to be skipped for a hexagonal scan (the older indexing batch, by name) or the scan was
refused at load (the batch page). Now both batches write: the page's batch hands the checkpoint to
``export_all``, whose writers read the grid from it, and says in its export record which grid each file
was, which APEX measurement it was paired with, and what the resampled files did to it.

The run goes through ``BatchManager._run_batch_sync_impl`` with a real ``_load_signal`` (only the file
reader is replaced by a signal built here), a recording stand-in for the indexer and the real exporter.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from backend.api.services import batch_manager as M
from backend.api.services.scan_geometry import ScanGeometry
from tests.apex_helpers import GUID_A
from tests.test_batch_hex_run import _Indexer, _signal

N_ROWS, N_COLS, STEP = 3, 5, 0.5


def _source(path: Path) -> None:
    """The shape the exporters need of a source: a header, and a vendor."""
    with h5py.File(path, "w") as f:
        f.create_dataset("Manufacturer", data=np.array([b"EDAX"]))
        f.create_group("1/EBSD/Header").create_dataset("Beam Voltage", data=np.float32(20.0))
        f.create_group("1/EBSD/Data")


@pytest.fixture
def mixed(monkeypatch, tmp_path):
    """``mixed(formats, signals)`` -> (manager, batch id, output folder). ``signals``: file name -> signal."""
    import safe_loader
    monkeypatch.setattr(M, "_batch_names", {})
    monkeypatch.setattr(M, "_register_signal_calibration", lambda signal, file_path: None)
    monkeypatch.setattr(M, "run_single_indexing_job", _Indexer())

    def go(formats, signals):
        files = {}
        for name, signal in signals.items():
            _source(tmp_path / name)
            files[str(tmp_path / name)] = signal
        monkeypatch.setattr(safe_loader, "load_ebsd_safe", lambda path: files[str(path)])
        mgr = M.BatchManager(db_path=str(tmp_path / "jobs.db"))
        cif = tmp_path / "Ni.cif"
        cif.write_text("data_Ni\n", encoding="utf-8")
        out = tmp_path / "out"
        config = {"auto_export": True, "export_dir": str(out), "include_eds": False}
        if formats is not None:
            config["export_formats"] = formats
        batch_id = mgr.queue.create_batch(list(files), [{"name": "Ni", "path": str(cif), "method": "hough"}], config)
        mgr._run_batch_sync_impl(batch_id)
        return mgr, batch_id, out

    return go


def _scans():
    data = np.zeros((N_ROWS, N_COLS, 1, 3), dtype=np.uint8)
    square = _signal(data, ScanGeometry.square(N_ROWS, N_COLS, STEP), mirror=False)
    hexagonal = _signal(data, ScanGeometry.hex(N_ROWS, N_COLS, STEP, odd_row_shift=0.5))
    return {"sq.h5oina": square, "hx.h5oina": hexagonal}


def _header(path):
    return [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.startswith("#")]


def _rows(path):
    rows = [ln.split() for ln in Path(path).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]
    return np.array(rows, dtype=float)


def _record(mgr, batch_id, tmp_path, name):
    return mgr.exports[batch_id][str(tmp_path / name)]


FORMATS = ["h5_light", "ang", "ctf", "ang_square"]


def test_batch_mixed_writes_both_grids(mixed, tmp_path):
    """One square and one hexagonal 3 x 5 file at step 0.5 in one batch, every format asked for.

    Square .ang: x runs 0 .. 4 * 0.5 = 2.0 and y 0 .. 2 * 0.5 = 1.0 (the last point is column 4 of row 2).
    Hexagonal .ang: five points in every row (NCOLS_ODD = NCOLS_EVEN = 5); the odd row 1 sits half a step
    to the right, so its last point is at x = (4 + 0.5) * 0.5 = 2.25; row 2 is at y = 2 * 0.5 * sqrt(3) / 2
    = 0.8660254. The square .ang of the hexagonal scan and its .ctf are on a square grid; the square file
    gets no resampled .ang (it is square already)."""
    mgr, batch_id, out = mixed(FORMATS, _scans())
    assert sorted(j["status"] for j in mgr.get_jobs(batch_id)) == ["done", "done"]

    sq_ang = out / "result_sq.ang"
    assert "# GRID: SqrGrid" in _header(sq_ang)
    last = _rows(sq_ang)[-1]
    assert (last[3], last[4]) == pytest.approx((2.0, 1.0), abs=1e-5)

    hx_ang = out / "result_hx.ang"
    header = _header(hx_ang)
    assert "# GRID: HexGrid" in header
    assert any(ln.replace(":", " ").split()[:2] == ["#", "NCOLS_ODD"] and ln.split()[-1] == "5" for ln in header)
    assert any(ln.replace(":", " ").split()[:2] == ["#", "NCOLS_EVEN"] and ln.split()[-1] == "5" for ln in header)
    rows = _rows(hx_ang)
    assert rows.shape[0] == N_ROWS * N_COLS
    assert (rows[9][3], rows[9][4]) == pytest.approx((2.25, 0.4330127), abs=1e-5)     # row 1, column 4
    assert (rows[14][3], rows[14][4]) == pytest.approx((2.0, 0.8660254), abs=1e-5)    # row 2, column 4

    assert (out / "result_hx_square.ang").is_file() and (out / "result_hx.ctf").is_file()
    assert (out / "result_sq.ctf").is_file() and not (out / "result_sq_square.ang").exists()
    assert (out / "result_sq_light.h5").is_file() and (out / "result_hx_light.h5").is_file()
    assert "# GRID: SqrGrid" in _header(out / "result_hx_square.ang")

    sq, hx = _record(mgr, batch_id, tmp_path, "sq.h5oina"), _record(mgr, batch_id, tmp_path, "hx.h5oina")
    assert sq["errors"] is None and hx["errors"] is None
    assert (sq["grid_kind"], hx["grid_kind"]) == ("square", "hex")
    assert sq["outputs"]["ang_square"] is None and hx["outputs"]["ang_square"] is not None


def test_the_export_record_carries_the_resampling_note_of_a_hexagonal_scan(mixed, tmp_path):
    """The note is the sentence the resampled files carry: nearest neighbour, the step of the square grid, the
    counts, and that nothing was interpolated. A square file in the same batch has no resampling to state."""
    mgr, batch_id, _out = mixed(FORMATS, _scans())
    hx, sq = _record(mgr, batch_id, tmp_path, "hx.h5oina"), _record(mgr, batch_id, tmp_path, "sq.h5oina")
    (note,) = [n for n in hx["notes"] if n.startswith("Resampled from a hexagonal scan")]
    assert "by nearest neighbour" in note and "nothing was interpolated" in note
    assert not any(n.startswith("Resampled") for n in sq["notes"])
    # a square scan cannot make the resampled .ang: the record says why it is missing
    assert any("square" in n and ".ang" in n for n in sq["notes"]), sq["notes"]


def test_the_ctf_of_a_hexagonal_scan_alone_still_carries_the_note(mixed, tmp_path):
    mgr, batch_id, _out = mixed(["ang", "ctf"], {"hx.h5oina": _scans()["hx.h5oina"]})
    hx = _record(mgr, batch_id, tmp_path, "hx.h5oina")
    assert any(n.startswith("Resampled from a hexagonal scan") for n in hx["notes"])


def test_a_hexagonal_scan_in_the_default_formats_gets_its_ctf_and_its_note(mixed, tmp_path):
    mgr, batch_id, out = mixed(None, {"hx.h5oina": _scans()["hx.h5oina"]})
    assert (out / "result_hx.ctf").is_file() and (out / "result_hx.ang").is_file()
    assert any(n.startswith("Resampled") for n in _record(mgr, batch_id, tmp_path, "hx.h5oina")["notes"])


def test_the_record_names_the_pairing_the_checkpoint_was_written_for(mixed, tmp_path):
    scans = _scans()
    scans["hx.h5oina"].metadata.set_item("Signal.apex", {
        "h5_path": str(tmp_path / "session" / "session.h5"), "measurement_id": GUID_A,
        "measurement_name": "Area 1 / OIM Map 1", "pairing": "guid"})
    mgr, batch_id, _out = mixed(["ang"], scans)
    hx, sq = _record(mgr, batch_id, tmp_path, "hx.h5oina"), _record(mgr, batch_id, tmp_path, "sq.h5oina")
    # the file's name only, not the folder of the machine that ran the batch
    assert hx["pairing"] == {"h5_name": "session.h5", "measurement_id": GUID_A,
                             "measurement_name": "Area 1 / OIM Map 1"}
    assert sq["pairing"] is None


def test_an_export_that_wrote_no_file_says_so_in_the_record(mixed, tmp_path, monkeypatch):
    """``export_ang_ctf`` turns a failed .ang into ``None`` and a log line; the record must still show
    the user that the file is missing."""
    from backend.api.services import ang_export

    def refuse(*a, **k):
        raise RuntimeError("the writer refused this grid")

    monkeypatch.setattr(ang_export, "save_hex_ang", refuse)
    mgr, batch_id, out = mixed(["ang", "ctf"], {"hx.h5oina": _scans()["hx.h5oina"]})
    hx = _record(mgr, batch_id, tmp_path, "hx.h5oina")
    assert hx["outputs"]["ang"] is None and not (out / "result_hx.ang").exists()
    assert hx["outputs"]["ctf"] is not None
    assert set(hx["errors"]) == {"ang"}
    assert "not written" in hx["errors"]["ang"]


def test_a_failed_export_still_leaves_a_record(mixed, tmp_path, monkeypatch):
    from backend.api.services import result_exporter

    def explode(**kw):
        raise OSError("the disk is full")

    monkeypatch.setattr(result_exporter, "export_all", explode)
    mgr, batch_id, _out = mixed(["ang"], {"sq.h5oina": _scans()["sq.h5oina"]})
    sq = _record(mgr, batch_id, tmp_path, "sq.h5oina")
    assert sq["outputs"] == {} and "the disk is full" in sq["errors"]["export"]


# --------------------------------------------------------------- the record the page translates
#
# The page is localised, the notes and reasons are fixed English sentences. Beside the sentence every note
# and every reason carries a ``code`` and the ``params`` (the numbers) the page words it with. The files on
# disk keep the English sentence; ``notes`` and ``errors`` stay the strings they were.


def _plan(monkeypatch, geometry, formats):
    from backend.api.services import checkpoint_writer
    monkeypatch.setattr(checkpoint_writer, "read_checkpoint_geometry", lambda path: geometry)
    return M._export_plan(formats, "unused.h5")


def test_the_resampling_note_carries_its_numbers_as_params(monkeypatch):
    """Hand case of ``test_hex_resample``: 3 x 4 hexagonal points at step 1, odd rows +0.5 and confirmed. 12 measured
    points, 12 cells written, one measured point (the 8th) no cell takes."""
    geometry = ScanGeometry.hex(3, 4, 1.0, odd_row_shift=+0.5, shift_confirmed=True)
    _formats, grid_kind, notes = _plan(monkeypatch, geometry, ["ctf"])
    assert grid_kind == "hex"
    (note,) = notes
    assert note.code == "resampledFromHex"
    assert note.params == {"step": 1.0, "source_points": 12, "written": 12, "unused": 1,
                           "direction_assumed": False, "side": None}
    assert str(note).startswith("Resampled from a hexagonal scan")       # the sentence on disk is unchanged


def test_the_assumed_direction_is_a_param_exactly_when_the_sentence_says_so(monkeypatch):
    """The sentence names the assumption when no file confirmed the direction, and the side it placed the odd
    rows on. +0.5 puts them to the right (one measured point unused), -0.5 to the left (two unused)."""
    for shift, side, unused in ((+0.5, "right", 1), (-0.5, "left", 2)):
        geometry = ScanGeometry.hex(3, 4, 1.0, odd_row_shift=shift, shift_confirmed=False)
        (note,) = _plan(monkeypatch, geometry, ["ang_square"])[2]
        assert "is an assumption" in str(note)
        assert note.params["direction_assumed"] is True and note.params["side"] == side
        assert note.params["unused"] == unused
    confirmed = ScanGeometry.hex(3, 4, 1.0, odd_row_shift=+0.5, shift_confirmed=True)
    (note,) = _plan(monkeypatch, confirmed, ["ang_square"])[2]
    assert "assum" not in str(note) and note.params["direction_assumed"] is False


def test_the_note_of_a_square_scan_asked_for_a_resampled_file_has_a_code(monkeypatch):
    formats, grid_kind, notes = _plan(monkeypatch, ScanGeometry.square(3, 4, 1.0), ["ang", "ang_square"])
    assert grid_kind == "square" and formats == ["ang"]
    (note,) = notes
    assert (note.code, note.params) == ("noSquareAngForSquareScan", {})
    assert str(note) == "No resampled square .ang was written: this scan is already on a square grid."


def test_the_record_hands_the_page_notes_with_code_and_params(mixed, tmp_path):
    import json
    mgr, batch_id, _out = mixed(FORMATS, _scans())
    hx, sq = _record(mgr, batch_id, tmp_path, "hx.h5oina"), _record(mgr, batch_id, tmp_path, "sq.h5oina")
    (item,) = hx["note_items"]
    assert item["code"] == "resampledFromHex" and item["text"] == hx["notes"][0]
    # 3 rows x 5 columns of equal length, no point missing: 15 measured points; the hexagonal test
    # scan states no direction, and its odd rows are shifted by +0.5 (to the right)
    assert item["params"]["source_points"] == 15
    assert item["params"]["direction_assumed"] is True and item["params"]["side"] == "right"
    assert [i["code"] for i in sq["note_items"]] == ["noSquareAngForSquareScan"]
    assert sq["note_items"][0]["text"] == sq["notes"][0]
    json.dumps([hx, sq])                                              # the route sends it as JSON


def test_a_format_that_was_not_written_has_a_coded_reason(mixed, tmp_path, monkeypatch):
    from backend.api.services import ang_export

    def refuse(*a, **k):
        raise RuntimeError("the writer refused this grid")

    monkeypatch.setattr(ang_export, "save_hex_ang", refuse)
    mgr, batch_id, _out = mixed(["ang", "ctf"], {"hx.h5oina": _scans()["hx.h5oina"]})
    hx = _record(mgr, batch_id, tmp_path, "hx.h5oina")
    assert hx["error_items"] == {"ang": {"code": "fileNotWritten", "params": {}, "text": hx["errors"]["ang"]}}


def test_a_real_exception_stays_as_technical_detail(mixed, tmp_path, monkeypatch):
    from backend.api.services import result_exporter

    def explode(*a, **k):
        raise OSError("Permission denied: 'result_sq_light.h5'")

    monkeypatch.setattr(result_exporter, "export_result_h5_light", explode)
    mgr, batch_id, _out = mixed(["h5_light"], {"sq.h5oina": _scans()["sq.h5oina"]})
    sq = _record(mgr, batch_id, tmp_path, "sq.h5oina")
    assert sq["errors"] == {"h5_light": "Permission denied: 'result_sq_light.h5'"}
    assert sq["error_items"] == {"h5_light": {
        "code": "writerFailed", "params": {"detail": "Permission denied: 'result_sq_light.h5'"},
        "text": "Permission denied: 'result_sq_light.h5'"}}


def test_a_failed_export_has_a_coded_reason_with_its_detail(mixed, tmp_path, monkeypatch):
    from backend.api.services import result_exporter

    def explode(**kw):
        raise OSError("the disk is full")

    monkeypatch.setattr(result_exporter, "export_all", explode)
    mgr, batch_id, _out = mixed(["ang"], {"sq.h5oina": _scans()["sq.h5oina"]})
    sq = _record(mgr, batch_id, tmp_path, "sq.h5oina")
    assert sq["error_items"] == {"export": {"code": "exportFailed", "params": {"detail": "the disk is full"},
                                            "text": "the disk is full"}}
    assert sq["note_items"] == [] and sq["notes"] == []


def test_a_record_without_trouble_has_no_error_items(mixed, tmp_path):
    mgr, batch_id, _out = mixed(["ang"], {"sq.h5oina": _scans()["sq.h5oina"]})
    sq = _record(mgr, batch_id, tmp_path, "sq.h5oina")
    assert sq["errors"] is None and sq["error_items"] is None


def test_the_exports_route_sends_the_items(mixed, tmp_path, monkeypatch):
    """What the page reads: the route adds nothing and drops nothing."""
    from fastapi.testclient import TestClient
    from backend.api.main import app
    from backend.api.routes import batch_v2
    mgr, batch_id, _out = mixed(FORMATS, _scans())
    monkeypatch.setattr(batch_v2, "_get_manager", lambda: mgr)
    body = TestClient(app).get(f"/api/batch-v2/{batch_id}/exports").json()
    by_name = {Path(r["file_path"]).name: r for r in body["exports"]}
    assert by_name["hx.h5oina"]["note_items"][0]["code"] == "resampledFromHex"
    assert by_name["sq.h5oina"]["note_items"][0]["code"] == "noSquareAngForSquareScan"
