"""Batch and the variant of a file (measurement x pattern source).

Batch v1 loads each file of its list itself (``load_ebsd_file``) and indexes what that load gave: the
variant THIS file was last loaded as, not the one the viewer happens to hold. Its results carry
``load_variant`` like a single run. Batch v2 always reads the file's default variant (the processed
patterns); the name it gives the file in the calibration store has to say the same, whatever variant
the viewer holds of that file.

Real routes, real loader, small synthetic files (``tests/test_load_variant.py``); the only stub is
the indexing engine at the edge of the run.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.routes import batch_v2 as B  # noqa: E402
from backend.api.routes import ebsd_viewer as ev  # noqa: E402
from backend.api.routes import indexing as I  # noqa: E402
from backend.api.services import batch_manager  # noqa: E402
from backend.api.services.calibration_store import calibration_store  # noqa: E402
from backend.api.services.load_choices import LoadVariant  # noqa: E402
from indexing_controller import IndexingMethod, IndexingResult  # noqa: E402
from tests.test_load_variant import NX, NY, SX, SY, _file, _processed, _unprocessed  # noqa: E402
from tests.test_load_variant_routes import _reset, _v  # noqa: E402

TYPED = [0.48, 0.52, 0.60]


@pytest.fixture
def client():
    from backend.api.main import app
    _reset()
    batch_manager._batch_names.clear()
    B._quick_loaded.clear()
    yield TestClient(app)
    _reset()
    batch_manager._batch_names.clear()
    B._quick_loaded.clear()


@pytest.fixture(autouse=True)
def restore_registry():
    saved, saved_active = dict(I._result_registry), I._active_result_id
    I._result_registry.clear()
    I._active_result_id = None
    try:
        yield
    finally:
        I._result_registry.clear()
        I._result_registry.update(saved)
        I._active_result_id = saved_active


@pytest.fixture
def multi(tmp_path):
    """Measurements 1 and 2, each with processed and unprocessed patterns."""
    return _file(tmp_path, "multi.h5oina", [(1, {}), (2, {})])


@pytest.fixture
def other(tmp_path):
    return _file(tmp_path, "other.h5oina", [(1, {})])


def _load(client, path, variant=None):
    body = {"path": path}
    if variant is not None:
        body["variant"] = variant
    r = client.post("/api/ebsd/load", json=body)
    assert r.status_code == 200, r.text
    return r


def _canon(path) -> str:
    return os.path.normcase(os.path.abspath(path))


# ------------------------------------------------------------------------------------------ batch v1


@pytest.mark.parametrize("order", ["variant_file_first", "variant_file_last"])
def test_batch_v1_uses_the_stored_variant(client, multi, other, monkeypatch, order):
    """``multi`` was loaded as measurement 2, unprocessed; ``other`` has one variant. The viewer holds
    ``other`` when the batch starts. Every item indexes the data of its OWN file's variant and its result
    carries that variant as ``load_variant``: measurement 2 / int16 for ``multi``, the default (None) /
    uint8 for ``other``, whichever file the batch visited before it."""
    import indexing_controller

    _load(client, multi, _v("2", "unprocessed"))
    _load(client, other)                                  # now the viewer holds another file
    I._result_registry.clear()
    seen = {}

    def spherical(h5_path, config, detector_params, selection_mask, **kwargs):
        signal = ev._get_active_signal()
        seen[_canon(h5_path)] = {"variant": kwargs.get("variant"), "data": np.asarray(signal.data)}
        # while the item runs, another variant becomes the loaded one: the stamp is the item's own
        monkeypatch.setattr(ev, "_ebsd_variant_key", ("9", "unprocessed"))
        mask = np.asarray(selection_mask, dtype=bool)
        return IndexingResult(
            xmap=None, selection_mask=mask, original_shape=mask.shape, method=IndexingMethod.SPHERICAL,
            confidence_scores=np.full(mask.size, 0.5, dtype=np.float32))

    for name in ("spherical_index_patterns", "spherical_gpu_index_patterns"):
        monkeypatch.setattr(indexing_controller, name, spherical)
    monkeypatch.setattr(I, "_attach_indexing_metadata", lambda *a, **k: None)
    monkeypatch.setattr(I, "_submit_run_guarded", lambda fn, kind="": fn())
    saved_batch = dict(I._batch_state)
    files = [multi, other] if order == "variant_file_first" else [other, multi]
    try:
        resp = client.post("/api/indexing/batch/start", json={
            "auto_export": False,
            "datasets": [{"file_path": f, "method": "spherical", "selection_mode": "full"} for f in files]})
        assert resp.status_code == 200, resp.text
        assert [r["error"] for r in I._batch_state["results"]] == [None, None], I._batch_state["results"]
    finally:
        I._batch_state.clear()
        I._batch_state.update(saved_batch)

    # the data of each item, expected by hand from the file builder (not from the loader)
    got = seen[_canon(multi)]
    assert got["data"].dtype == np.int16
    assert np.array_equal(got["data"], _unprocessed(2).reshape(NY, NX, SY, SX))
    assert got["variant"] == LoadVariant("2", "unprocessed")
    got = seen[_canon(other)]
    assert got["data"].dtype == np.uint8
    assert np.array_equal(got["data"], _processed(1).reshape(NY, NX, SY, SX))
    assert got["variant"] is None

    # ... and the stamp on each stored result
    stamps = {_canon(r.metadata["source_file"]): r.metadata["load_variant"]
              for r in I._result_registry.values()}
    assert stamps == {
        _canon(multi): {"measurement": "2", "pattern_source": "unprocessed"},
        _canon(other): None,
    }


# ------------------------------------------------------------------------------------------ batch v2


def test_batch_v2_loads_processed_and_says_so(client, multi, monkeypatch):
    """The viewer holds ``multi`` as measurement 2, unprocessed. Batch v2 reads the default variant of the
    file, the processed patterns of measurement 1, and the name it gives the file says that: the plain
    stem, not the viewer's ``multi . unprocessed`` and no ``__hash`` suffix. The viewer's own calibration
    entry is left alone."""
    _load(client, multi, _v("2", "unprocessed"))
    viewer_name = ev._active_dataset
    assert viewer_name != "multi"                                   # premise: the viewer's name carries the variant
    viewer_entry = calibration_store.get_entry(viewer_name)
    assert viewer_entry is not None

    answer = client.post("/api/batch-v2/quick-load", json={"file_path": multi})
    assert answer.status_code == 200, answer.text
    assert answer.json()["dataset_name"] == "multi"
    kept = B._quick_loaded["multi"]["signal"]
    assert np.asarray(kept.data).dtype == np.uint8
    assert np.array_equal(np.asarray(kept.data), _processed(1).reshape(NY, NX, SY, SX))

    assert batch_manager.store_name_for_file(Path(multi)) == "multi"
    assert batch_manager.assign_batch_dataset_names([multi]) == {_canon(multi): ("multi", "")}
    assert calibration_store.get_entry(viewer_name) is viewer_entry          # not overwritten

    run_signal = batch_manager._load_signal(multi)                  # what the run reads
    assert np.asarray(run_signal.data).dtype == np.uint8
    assert np.array_equal(np.asarray(run_signal.data), _processed(1).reshape(NY, NX, SY, SX))


def test_batch_v2_copy_pc_addresses_the_entry_of_its_own_data(client, multi, other):
    """copy-pc addresses a file by path. A file the viewer holds as another variant has two entries in the
    store; the one batch v2 copies from and to is the one of the processed patterns."""
    _load(client, multi, _v("2", "unprocessed"))
    for path in (multi, other):
        assert client.post("/api/batch-v2/quick-load", json={"file_path": path}).status_code == 200
    r = client.post("/api/batch-v2/copy-pc", json={
        "source_dataset": "other", "target_datasets": ["multi"],
        "source_file_path": other, "target_file_paths": [multi]})
    assert r.status_code == 200, r.text
    assert r.json()["results"][0]["key"] == "multi"
    assert calibration_store.get_entry("multi").pc_source == "inherited"
    viewer_entry = calibration_store.get_entry(ev._active_dataset)
    assert viewer_entry.pc_source != "inherited"


def test_batch_v2_copy_pc_from_a_file_the_viewer_holds_as_a_variant(client, multi, other):
    """The same, the variant file being the SOURCE: the PC copied is the one of its processed patterns."""
    _load(client, multi, _v("2", "unprocessed"))
    calibration_store.update_pc(ev._active_dataset, TYPED, source="manual")   # the viewer's entry only
    for path in (multi, other):
        assert client.post("/api/batch-v2/quick-load", json={"file_path": path}).status_code == 200
    calibration_store.update_pc("multi", [0.41, 0.42, 0.43], source="manual")  # the batch entry's own PC
    r = client.post("/api/batch-v2/copy-pc", json={
        "source_dataset": "multi", "target_datasets": ["other"],
        "source_file_path": multi, "target_file_paths": [other]})
    assert r.status_code == 200, r.text
    assert r.json()["source_key"] == "multi"
    assert r.json()["pc"] == pytest.approx([0.41, 0.42, 0.43], abs=1e-12)


def test_batch_v2_keeps_the_header_pc_when_the_user_made_none(client, multi):
    """Nothing is handed over from the viewer's entry unless the user made that PC."""
    _load(client, multi, _v("2", "unprocessed"))
    assert calibration_store.get_entry(ev._active_dataset).pc_source not in calibration_store.USER_MADE_SOURCES
    answer = client.post("/api/batch-v2/quick-load", json={"file_path": multi}).json()
    assert answer["pc_source"] != "inherited"
    assert calibration_store.get_entry("multi").parent_name is None


def test_batch_v2_takes_over_the_pc_the_user_made_for_the_viewers_variant(client, multi):
    """A PC the user typed or refined on the unprocessed patterns of the measurement batch v2 reads
    describes the same acquisition: quick-load hands it to the entry of the processed patterns (source
    ``inherited``) instead of showing the header PC under the same file."""
    _load(client, multi, _v("1", "unprocessed"))
    calibration_store.update_pc(ev._active_dataset, TYPED, source="manual")
    answer = client.post("/api/batch-v2/quick-load", json={"file_path": multi}).json()
    assert answer["dataset_name"] == "multi"
    assert answer["pc"] == pytest.approx(TYPED, abs=1e-12)
    assert answer["pc_source"] == "inherited"
    assert calibration_store.get_entry("multi").parent_name == ev._active_dataset


def test_a_pc_typed_on_another_measurement_stays_there(client, multi):
    """Measurement 2 has a PC of its own. Batch v2 reads measurement 1, and that one keeps its header
    PC; nothing ties its entry to the viewer's."""
    _load(client, multi, _v("2", "unprocessed"))
    calibration_store.update_pc(ev._active_dataset, TYPED, source="manual")
    answer = client.post("/api/batch-v2/quick-load", json={"file_path": multi}).json()
    assert answer["pc_source"] != "inherited"
    assert answer["pc"] != pytest.approx(TYPED, abs=1e-12)
    assert calibration_store.get_entry("multi").parent_name is None


def test_batch_v2_shares_the_entry_of_a_viewer_default_load(client, multi):
    """The viewer holds the default variant (also when it was named explicitly): one entry for the file,
    as before, and a PC the user made is kept by quick-load."""
    _load(client, multi, _v("1", "processed"))
    assert ev._active_dataset == "multi"
    calibration_store.update_pc("multi", TYPED, source="manual")
    answer = client.post("/api/batch-v2/quick-load", json={"file_path": multi}).json()
    assert answer["dataset_name"] == "multi"
    assert answer["pc_source"] == "manual"
    assert list(calibration_store.get_all()) == ["multi"]
