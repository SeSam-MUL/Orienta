"""Workbook name reservations: a leftover of a crashed run is not an existing workbook; a half-written file
after a failed save is kept under another name and named in the message."""
import pytest

from analysis import excel_exporter as ee
from analysis.excel_exporter import free_workbook_path, release_workbook_reservation


@pytest.fixture(autouse=True)
def no_live_reservations():
    ee._LIVE_RESERVATIONS.clear()
    yield
    ee._LIVE_RESERVATIONS.clear()


def test_an_empty_leftover_that_this_process_does_not_hold_is_reused(tmp_path, caplog):
    leftover = tmp_path / "doc.xlsx"
    leftover.write_bytes(b"")                             # a backend killed mid-batch left this
    with caplog.at_level("WARNING", logger=ee.logger.name):
        got = free_workbook_path(leftover)
    assert got == leftover                                # not "doc_2.xlsx": it was never a workbook
    assert "leftover" in caplog.text and "doc.xlsx" in caplog.text
    assert free_workbook_path(leftover).name == "doc_2.xlsx"        # now it is held by this process


def test_a_reservation_held_by_this_process_is_never_taken_over(tmp_path):
    held = free_workbook_path(tmp_path / "doc.xlsx")
    assert free_workbook_path(tmp_path / "doc.xlsx").name == "doc_2.xlsx"
    release_workbook_reservation(held)
    assert not held.exists()
    assert free_workbook_path(tmp_path / "doc.xlsx") == held        # released: the name is free again


def test_a_workbook_with_content_is_never_reused(tmp_path):
    (tmp_path / "doc.xlsx").write_bytes(b"PK earlier run")
    assert free_workbook_path(tmp_path / "doc.xlsx").name == "doc_2.xlsx"
    assert (tmp_path / "doc.xlsx").read_bytes() == b"PK earlier run"


def test_two_runs_never_take_over_the_same_leftover(tmp_path):
    import threading
    (tmp_path / "doc.xlsx").write_bytes(b"")
    n, got, lock = 8, [], threading.Lock()
    barrier = threading.Barrier(n)

    def pick():
        barrier.wait(10)
        p = free_workbook_path(tmp_path / "doc.xlsx")
        with lock:
            got.append(p)

    threads = [threading.Thread(target=pick) for _ in range(n)]
    [t.start() for t in threads]
    [t.join(30) for t in threads]
    assert len(got) == n and len(set(got)) == n


def test_release_of_nothing_is_no_error():
    release_workbook_reservation(None)


def test_a_finished_workbook_is_forgotten_by_the_reservations(tmp_path):
    path = free_workbook_path(tmp_path / "doc.xlsx")
    path.write_bytes(b"PK done")
    release_workbook_reservation(path)                    # the run is over with this name; content is never touched
    assert path.read_bytes() == b"PK done"
    path.write_bytes(b"")                                 # a later crash leaves an empty file under the same name
    assert free_workbook_path(tmp_path / "doc.xlsx") == path


def test_free_workbook_path_makes_the_parent_folder():
    import inspect
    assert "folder" in inspect.getdoc(free_workbook_path)


def test_a_partial_file_is_set_aside_under_the_first_free_incomplete_name(tmp_path):
    f = tmp_path / "doc.xlsx"
    f.write_bytes(b"PK half")
    assert ee.set_aside_incomplete_workbook(f) == tmp_path / "doc.incomplete.xlsx"
    assert not f.exists() and (tmp_path / "doc.incomplete.xlsx").read_bytes() == b"PK half"
    f.write_bytes(b"PK half again")
    assert ee.set_aside_incomplete_workbook(f) == tmp_path / "doc.incomplete_2.xlsx"
    assert (tmp_path / "doc.incomplete.xlsx").read_bytes() == b"PK half"          # the earlier one is untouched


def test_nothing_to_set_aside_for_an_empty_or_missing_file(tmp_path):
    (tmp_path / "e.xlsx").write_bytes(b"")
    assert ee.set_aside_incomplete_workbook(tmp_path / "e.xlsx") is None
    assert ee.set_aside_incomplete_workbook(tmp_path / "missing.xlsx") is None
    assert ee.set_aside_incomplete_workbook(None) is None


def test_the_failure_note_says_where_the_partial_file_went(tmp_path):
    f = tmp_path / "doc.xlsx"
    f.write_bytes(b"PK half")
    assert ee.incomplete_workbook_note(f) == " The incomplete file was kept as doc.incomplete.xlsx."
    assert ee.incomplete_workbook_note(tmp_path / "missing.xlsx") == ""
    assert ee.incomplete_workbook_note(None) == ""


def test_the_failure_note_says_so_when_the_partial_file_could_not_be_renamed(tmp_path, monkeypatch):
    from pathlib import Path
    f = tmp_path / "doc.xlsx"
    f.write_bytes(b"PK half")

    def locked(self, target):
        raise PermissionError("the file is open in another program")
    monkeypatch.setattr(Path, "rename", locked)
    assert ee.set_aside_incomplete_workbook(f) is None
    note = ee.incomplete_workbook_note(f)
    assert "could not be renamed" in note and "still has the workbook name doc.xlsx" in note
    assert f.read_bytes() == b"PK half"                      # nothing was lost


# ---------------------------------------------------------------------------------------------
# batch and route
# ---------------------------------------------------------------------------------------------
from tests.test_analysis_page_sync import _mixed_folder, client_map  # noqa: E402,F401


def _batch(client, tmp_path, out):
    r = client.post("/api/analysis/batch", json={"folder_path": str(tmp_path), "output_path": str(out)})
    return client.get(f"/api/analysis/task/{r.json()['task_id']}").json()


def test_an_empty_leftover_is_reused_by_the_batch_and_not_reported_as_an_existing_workbook(
        client_map, monkeypatch, tmp_path):
    client, _ = client_map
    _mixed_folder(tmp_path, monkeypatch)
    out = tmp_path / "doc.xlsx"
    out.write_bytes(b"")                                  # a backend killed mid-batch left this
    task = _batch(client, tmp_path, out)
    assert task["status"] == "completed", task
    assert task["result"]["workbook_path"] == str(out) and out.stat().st_size > 0
    assert "already existed" not in task["message"] and not (tmp_path / "doc_2.xlsx").exists()


def test_a_name_held_by_a_running_export_is_not_called_an_existing_workbook(client_map, monkeypatch, tmp_path):
    client, _ = client_map
    _mixed_folder(tmp_path, monkeypatch)
    held = free_workbook_path(tmp_path / "doc.xlsx")      # another run holds the name right now
    task = _batch(client, tmp_path, held)
    assert task["status"] == "completed", task
    assert task["result"]["workbook_path"] == str(tmp_path / "doc_2.xlsx")
    assert "already existed" not in task["message"] and "in use by another run" in task["message"]
    assert held.exists() and held.stat().st_size == 0     # the other run's reservation is untouched


def test_a_batch_whose_save_leaves_a_partial_file_keeps_it_under_another_name(client_map, monkeypatch, tmp_path):
    client, _ = client_map
    _mixed_folder(tmp_path, monkeypatch)

    def half_save(self):
        Path(self.file_path).write_bytes(b"PK half a workbook")
        raise PermissionError("disk full")

    monkeypatch.setattr(ee.ExcelExporter, "save", half_save)
    out = tmp_path / "doc.xlsx"
    task = _batch(client, tmp_path, out)
    assert task["status"] == "failed", task
    assert not out.exists()                               # the half-written file does not pass for the workbook
    assert (tmp_path / "doc.incomplete.xlsx").read_bytes() == b"PK half a workbook"
    assert "doc.incomplete.xlsx" in task["error"] and "disk full" in task["error"]


def test_a_failed_export_keeps_a_partial_file_under_another_name_and_says_so(client_map, monkeypatch, tmp_path):
    client, ar = client_map
    monkeypatch.setattr(ar, "_collect_export_results", lambda warnings: {})

    def half_export(path, name, results):
        Path(path).write_bytes(b"PK half")
        raise OSError("disk full")

    monkeypatch.setattr(ee, "export_to_excel", half_export)
    out = tmp_path / "o.xlsx"
    r = client.post("/api/analysis/export/excel", json={"output_path": str(out), "dataset_name": "d"})
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "failed", task
    assert not out.exists() and (tmp_path / "o.incomplete.xlsx").read_bytes() == b"PK half"
    assert "disk full" in task["error"] and "o.incomplete.xlsx" in task["error"]


def test_a_failed_export_without_a_partial_file_leaves_nothing_behind(client_map, monkeypatch, tmp_path):
    client, ar = client_map
    monkeypatch.setattr(ar, "_collect_export_results", lambda warnings: {})
    monkeypatch.setattr(ee, "export_to_excel", lambda path, name, results: (_ for _ in ()).throw(OSError("no")))
    out = tmp_path / "o.xlsx"
    r = client.post("/api/analysis/export/excel", json={"output_path": str(out), "dataset_name": "d"})
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "failed" and task["error"] == "no"
    assert list(tmp_path.iterdir()) == []


from pathlib import Path  # noqa: E402
