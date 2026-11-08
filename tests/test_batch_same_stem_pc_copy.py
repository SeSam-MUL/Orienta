"""Copy-PC and quick-load keep two files with one stem apart; an entry without a file is no file's.

Live finding: the viewer holds A/Scan5 with a typed PC. ``quick-load B/Scan5`` answered with A's
PC as B's (it kept the viewer's user-made entry), and ``copy-pc`` to "Scan5" overwrote A's
calibration - B never got the PC. Quick-load now gives each file its own name in the store
(the stem, or stem + path hash when an entry of that stem is another file's), answers with that
name, and copy-pc writes into the entry registered for the file whose PATH it is given.
An entry registered without a file (import-h5 used to) belongs to no file we can name: it is
treated as another file's. And a job whose entry describes another detector than the file's
patterns dies with a code instead of an index error.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from kikuchipy.detectors import EBSDDetector

from backend.api.main import app
from tests.test_batch_same_stem import (  # noqa: F401  (fixtures and helpers of the sibling tests)
    NI_CIF, OSC_A, OSC_B, SIDE, _fake_job, _run_batch, _write_osc, _write_up1, world)

TYPED = [0.48, 0.52, 0.60]


def _bruker(osc):
    return [osc[0], 1 - osc[1], osc[2]]            # TSL -> Bruker for a square pattern, by hand


def test_copy_pc_to_a_same_stem_file_reaches_that_file_and_leaves_the_viewers_entry_alone(world, monkeypatch):
    store, mgr, tmp = world
    from backend.api.services import batch_manager as M
    a = _write_up1(tmp / "A" / "Scan5.up1", seed=1)                    # the viewer's file
    b = _write_up1(tmp / "B" / "Scan5.up1", seed=2)
    (tmp / "S").mkdir()
    src = _write_up1(tmp / "S" / "Other.up1", seed=3)                  # the file the PC is copied from
    _write_osc(tmp / "B" / "Scan5.osc", OSC_B)
    _write_osc(tmp / "S" / "Other.osc", OSC_A)
    client = TestClient(app)
    assert client.post("/api/ebsd/load", json={"path": a}).status_code == 200      # A is the viewer's file
    store.update_pc("Scan5", TYPED, source="manual")

    ql_src = client.post("/api/batch-v2/quick-load", json={"file_path": src}).json()
    ql_b = client.post("/api/batch-v2/quick-load", json={"file_path": b}).json()
    # B is shown with ITS PC and its own name - not the viewer's typed PC
    np.testing.assert_allclose(ql_b["pc"], _bruker(OSC_B), atol=1e-5)
    assert ql_b["pc_source"] == "header" and ql_b["dataset_name"].startswith("Scan5__")

    resp = client.post("/api/batch-v2/copy-pc", json={
        "source_dataset": ql_src["dataset_name"], "target_datasets": ["Scan5.up1"],
        "source_file_path": src, "target_file_paths": [b]})
    assert resp.status_code == 200 and resp.json()["results"][0]["success"] is True
    # B's entry took the PC; the viewer's A entry is exactly as the user left it
    np.testing.assert_allclose(store.get_pc(ql_b["dataset_name"]), _bruker(OSC_A), atol=1e-5)
    assert store.get_entry(ql_b["dataset_name"]).pc_source == "inherited"
    np.testing.assert_allclose(store.get_pc("Scan5"), TYPED, atol=1e-12)
    assert store.get_entry("Scan5").pc_source == "manual"
    # and the batch indexes B at the copied PC (not at its header, not at A's)
    used = {}
    monkeypatch.setattr(M, "run_single_indexing_job", _fake_job(used))
    batch_id, _ = _run_batch(client, mgr, [b], tmp, ["ang"])
    assert [j["status"] for j in mgr.get_jobs(batch_id)] == ["done"]
    np.testing.assert_allclose(used[b], _bruker(OSC_A), atol=1e-5)


def test_copy_pc_to_a_file_that_is_not_in_the_store_is_refused_not_redirected_to_its_stem(world):
    store, mgr, tmp = world
    a = _write_up1(tmp / "A" / "Scan5.up1", seed=1)
    b = _write_up1(tmp / "B" / "Scan5.up1", seed=2)                    # never quick-loaded
    (tmp / "S").mkdir()
    src = _write_up1(tmp / "S" / "Other.up1", seed=3)
    _write_osc(tmp / "S" / "Other.osc", OSC_A)
    client = TestClient(app)
    client.post("/api/ebsd/load", json={"path": a})
    store.update_pc("Scan5", TYPED, source="manual")
    client.post("/api/batch-v2/quick-load", json={"file_path": src})
    resp = client.post("/api/batch-v2/copy-pc", json={
        "source_dataset": "Other.up1", "target_datasets": ["Scan5.up1"],
        "source_file_path": src, "target_file_paths": [b]})
    assert resp.status_code == 200
    (r,) = resp.json()["results"]
    assert r["success"] is False and "quick-load" in r["error"]
    np.testing.assert_allclose(store.get_pc("Scan5"), TYPED, atol=1e-12)           # A untouched


def test_an_entry_registered_without_a_file_is_not_taken_for_a_files_calibration(world, monkeypatch):
    """What import-h5 used to leave: an entry named like the file, with no file key. A batch over
    a same-stem file must neither use its (foreign) geometry/PC nor trust its 'ok'."""
    store, mgr, tmp = world
    from backend.api.services import batch_manager as M
    foreign = SimpleNamespace(detector=EBSDDetector(shape=(90, 120), pc=(0.5026, 0.3273, 0.8458), sample_tilt=70.0))
    store.register("Scan5", foreign)                                    # no file_key: an Oxford 90 x 120 entry
    a = _write_up1(tmp / "A" / "Scan5.up1", seed=1)                    # a UP1 without a PC, 24 x 24 patterns
    used = {}
    monkeypatch.setattr(M, "run_single_indexing_job", _fake_job(used))
    batch_id, _ = _run_batch(TestClient(app), mgr, [a], tmp, ["ang"])
    (job,) = mgr.get_jobs(batch_id)
    assert job["status"] == "failed" and job["error_code"] == "pc_missing"          # decided from A's own header
    assert used == {}
    assert M.dataset_name_for(a).startswith("Scan5__")                             # A did not take the foreign name


def test_a_job_whose_entry_is_for_another_detector_dies_with_a_code(world, monkeypatch):
    store, mgr, tmp = world
    from backend.api.services import batch_manager as M
    a = _write_up1(tmp / "A" / "Scan5.up1", seed=1)
    _write_osc(tmp / "A" / "Scan5.osc", OSC_A)
    # an entry of THIS file (right key) from an earlier, different version of the file: 90 x 120
    store.register("Scan5", SimpleNamespace(detector=EBSDDetector(shape=(90, 120), pc=(0.5, 0.4, 0.7))),
                   file_key=M._canonical_file(a))
    monkeypatch.setattr(M, "run_single_indexing_job", _fake_job({}))
    batch_id, _ = _run_batch(TestClient(app), mgr, [a], tmp, ["ang"])
    (job,) = mgr.get_jobs(batch_id)
    assert job["status"] == "failed" and job["error_code"] == "detector_shape_mismatch"
    assert "belongs to another file" in job["error_msg"]


def test_copy_from_a_same_stem_file_takes_that_files_pc_not_the_viewers(world):
    """The SOURCE of a copy is picked by path too: 'Scan5.up1' names two files."""
    store, mgr, tmp = world
    a = _write_up1(tmp / "A" / "Scan5.up1", seed=1)
    b = _write_up1(tmp / "B" / "Scan5.up1", seed=2)
    (tmp / "T").mkdir()
    t = _write_up1(tmp / "T" / "Target.up1", seed=3)
    _write_osc(tmp / "B" / "Scan5.osc", OSC_B)
    _write_osc(tmp / "T" / "Target.osc", OSC_A)
    client = TestClient(app)
    client.post("/api/ebsd/load", json={"path": a})
    store.update_pc("Scan5", TYPED, source="manual")                   # the viewer's file
    ql_b = client.post("/api/batch-v2/quick-load", json={"file_path": b}).json()
    ql_t = client.post("/api/batch-v2/quick-load", json={"file_path": t}).json()
    resp = client.post("/api/batch-v2/copy-pc", json={
        "source_dataset": "Scan5.up1", "target_datasets": ["Target.up1"],
        "source_file_path": b, "target_file_paths": [t]})
    assert resp.status_code == 200 and resp.json()["results"][0]["success"] is True
    np.testing.assert_allclose(store.get_pc(ql_t["dataset_name"]), _bruker(OSC_B), atol=1e-5)   # B's, not A's typed


@pytest.mark.parametrize("extra", [
    {},                                                                     # the old call: names only
    {"source_file_path": "S"},                                              # targets without paths
    {"target_file_paths": ["T"]},                                           # source without a path
    {"source_file_path": "S", "target_file_paths": []},                     # fewer paths than targets
    {"source_file_path": "S", "target_file_paths": ["T", "T2"]},            # more paths than targets
    {"source_file_path": "S", "target_file_paths": [""]},                   # an empty path
])
def test_copy_pc_without_the_files_is_refused_and_the_store_stays_as_it_was(world, extra):
    """Live finding: copy-pc by NAME ('Scan5') overwrote the viewer's typed entry of another
    file with the same stem, answering 200. Names are not addresses; the request is refused."""
    store, mgr, tmp = world
    a = _write_up1(tmp / "A" / "Scan5.up1", seed=1)                    # the viewer's file
    (tmp / "S").mkdir()
    src = _write_up1(tmp / "S" / "Other.up1", seed=3)
    _write_osc(tmp / "S" / "Other.osc", OSC_A)
    client = TestClient(app)
    assert client.post("/api/ebsd/load", json={"path": a}).status_code == 200
    store.update_pc("Scan5", TYPED, source="manual")
    ql = client.post("/api/batch-v2/quick-load", json={"file_path": src}).json()
    before = {n: (store.get_entry(n).pc_source, list(store.get_pc(n))) for n in ("Scan5", ql["dataset_name"])}

    body = {"source_dataset": ql["dataset_name"], "target_datasets": ["Scan5"]}
    for k, v in extra.items():
        body[k] = src if v == "S" else ([a if x == "T" else (a if x == "T2" else x) for x in v]
                                        if isinstance(v, list) else v)
    resp = client.post("/api/batch-v2/copy-pc", json=body)
    assert resp.status_code == 400
    assert resp.headers["X-Orienta-Code"] == "copy_pc_needs_files"
    after = {n: (store.get_entry(n).pc_source, list(store.get_pc(n))) for n in before}
    assert after == before                                              # nothing written, nowhere
    np.testing.assert_allclose(store.get_pc("Scan5"), TYPED, atol=1e-12)
    assert store.get_entry("Scan5").pc_source == "manual"
