"""ML Hub: a trained model survives a backend restart, and the learning rate
the user types reaches the optimiser.

The routes run for real (TestClient -> /api/ml/train -> torch training ->
checkpoints on disk -> /api/ml/predict) on the 9 real Ni patterns that ship
with kikuchipy. Only the *source* of the samples is replaced: the registry
entry "last indexing result" is a small stand-in holding those patterns and
labels, because producing a real indexing result needs a full indexing run.
All ML code (store, trainer, checkpoints, predictor, routes) is the product code.

A backend restart is simulated by resetting the module-level predictor of
``ml_controller`` to None, which is exactly what a fresh process has.
"""
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("torch")
kikuchipy = pytest.importorskip("kikuchipy")

import ml_controller  # noqa: E402
from backend.api.main import app  # noqa: E402

PHASE_NAMES = ["Ni", "Fe"]
LR_FIRST = 0.0123
LR_SECOND = 0.0045


class _PhaseList:
    """Just enough of orix.PhaseList for ``_extract_phase_labels``."""

    def __init__(self, names):
        self.ids = list(range(len(names)))
        self._names = names

    def __getitem__(self, pid):
        return SimpleNamespace(name=self._names[pid])


def _fake_indexing_result(signal, phase_ids, phase_names=PHASE_NAMES):
    n = signal.data.shape[0] * signal.data.shape[1]
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    return SimpleNamespace(
        metadata={"signal": signal},
        xmap=SimpleNamespace(
            phase_id=np.asarray(phase_ids),
            phases=_PhaseList(phase_names),
            rotations=SimpleNamespace(data=quats),
        ),
        selection_mask=None,
        confidence_scores=np.ones(n),
    )


@pytest.fixture
def isolated_ml(monkeypatch, tmp_path):
    """Switch on, store in tmp_path, predictor empty (= freshly started backend)."""
    monkeypatch.setenv("ORIENTA_EXPERIMENTAL_ML", "1")
    monkeypatch.setattr(ml_controller, "_STORE_DIR", tmp_path / "ml_training_data")
    monkeypatch.setattr(ml_controller, "_store", None)
    monkeypatch.setattr(ml_controller, "_predictor", None)
    monkeypatch.setattr(ml_controller, "_loaded_run_id", None)
    monkeypatch.setattr(ml_controller, "_unloadable_runs", set())
    return TestClient(app)


def _simulate_restart():
    ml_controller._predictor = None
    ml_controller._loaded_run_id = None
    ml_controller._store = None
    ml_controller._unloadable_runs = set()


def _train_and_wait(client, learning_rate, **extra):
    r = client.post("/api/ml/train", json={
        "epochs": 2, "batch_size": 4, "learning_rate": learning_rate, "ci_threshold": 0.0,
        **extra,
    })
    assert r.status_code == 200, r.text
    task_id = r.json()["task_id"]
    for _ in range(300):
        st = client.get(f"/api/ml/train/{task_id}").json()
        if st["status"] != "running":
            break
        time.sleep(0.1)
    assert st["status"] == "completed", st
    return st


@pytest.fixture(scope="module")
def _trained_store(tmp_path_factory):
    """Train twice (different learning rates) on 9 real Ni patterns, once per module.

    Returns (store_dir, first_run_id, second_run_id). Each test gets its own
    copy of the store (see ``two_runs``), so tests may corrupt or extend it.
    """
    mp = pytest.MonkeyPatch()
    store_dir = tmp_path_factory.mktemp("trained") / "ml_training_data"
    try:
        mp.setenv("ORIENTA_EXPERIMENTAL_ML", "1")
        mp.setattr(ml_controller, "_STORE_DIR", store_dir)
        mp.setattr(ml_controller, "_store", None)
        mp.setattr(ml_controller, "_predictor", None)
        mp.setattr(ml_controller, "_loaded_run_id", None)
        signal = kikuchipy.data.nickel_ebsd_small()
        assert signal.data.shape[:2] == (3, 3)
        phase_ids = [0, 0, 0, 0, 0, 1, 1, 1, 1]  # any label split will do
        result = _fake_indexing_result(signal, phase_ids)
        import backend.api.routes.ebsd_viewer as ev
        import backend.api.routes.indexing as idx
        mp.setattr(idx, "get_last_indexing_result", lambda: result)
        mp.setattr(ev, "_get_active_signal", lambda: signal)
        mp.setattr(ev, "_ebsd_file_path", "somewhere/Ni_small.h5oina")

        client = TestClient(app)
        first = _train_and_wait(client, LR_FIRST)
        time.sleep(1.1)  # run ids are second-resolution timestamps
        second = _train_and_wait(client, LR_SECOND)
        yield store_dir, first["result"]["run_id"], second["result"]["run_id"]
    finally:
        mp.undo()


@pytest.fixture
def two_runs(_trained_store, isolated_ml, tmp_path, monkeypatch):
    """A private copy of the two trained runs, with an empty (restarted) predictor."""
    import shutil

    store_dir, first_id, second_id = _trained_store
    mine = tmp_path / "ml_training_data"  # same path isolated_ml pointed at
    shutil.copytree(store_dir, mine)
    return isolated_ml, first_id, second_id


def test_status_without_any_run_reports_no_model(isolated_ml):
    body = isolated_ml.get("/api/ml/status").json()
    assert body["available"] is True
    assert body["model_loaded"] is False
    assert body["loaded_run_id"] is None
    assert isolated_ml.get("/api/ml/models").json()["models"] == []


def test_models_lists_runs_newest_first_with_metadata(two_runs):
    client, first_id, second_id = two_runs
    assert first_id < second_id
    models = client.get("/api/ml/models").json()["models"]
    assert [m["run_id"] for m in models] == [second_id, first_id]
    newest = models[0]
    assert sorted(newest["phase_names"]) == sorted(PHASE_NAMES)
    # The store accumulates (18 samples after two rounds), but each run trains
    # only on the samples its own request added.
    assert newest["n_samples"] == 9
    assert models[1]["n_samples"] == 9
    assert newest["sample_sources"]["this_request"]["n_samples"] == 9
    assert newest["sample_sources"]["existing_store"] == {
        "available": 9, "included": False, "n_samples": 0,
    }
    assert newest["source_file"] == "Ni_small.h5oina"
    assert newest["epochs_completed"] >= 1
    assert isinstance(newest["final_val_loss"], float)
    assert newest["created"]


def test_learning_rate_from_the_request_reaches_training(two_runs):
    client, first_id, second_id = two_runs
    by_id = {m["run_id"]: m for m in client.get("/api/ml/models").json()["models"]}
    assert by_id[first_id]["learning_rate"] == pytest.approx(LR_FIRST)
    assert by_id[second_id]["learning_rate"] == pytest.approx(LR_SECOND)


def test_checkpoints_live_in_one_folder_per_run(two_runs, tmp_path):
    _client, first_id, second_id = two_runs
    root = tmp_path / "ml_training_data" / "checkpoints"
    for run_id in (first_id, second_id):
        assert (root / run_id / "best.pt").is_file()
        assert (root / run_id / "last.pt").is_file()
        assert (root / run_id / "meta.json").is_file()
    assert not (root / "best.pt").exists()  # nothing written to the flat legacy place


def test_freshly_trained_model_is_the_loaded_one(two_runs):
    client, _first_id, second_id = two_runs
    body = client.get("/api/ml/status").json()
    assert body["model_loaded"] is True
    assert body["loaded_run_id"] == second_id
    models = client.get("/api/ml/models").json()["models"]
    assert [m["loaded"] for m in models] == [True, False]


def test_status_after_restart_lazily_loads_newest_run(two_runs):
    client, _first_id, second_id = two_runs
    _simulate_restart()
    assert ml_controller.get_predictor() is None
    body = client.get("/api/ml/status").json()
    assert body["model_loaded"] is True
    assert body["loaded_run_id"] == second_id
    assert sorted(body["phase_names"]) == sorted(PHASE_NAMES)


def test_predict_after_restart_lazily_loads_and_predicts(two_runs):
    client, _first_id, _second_id = two_runs
    _simulate_restart()
    r = client.post("/api/ml/predict", json={"pattern_indices": [0, 4, 8, 99]})
    assert r.status_code == 200, r.text
    preds = r.json()["predictions"]
    assert [p["index"] for p in preds] == [0, 4, 8, 99]
    for p in preds[:3]:
        assert p["phase_name"] in PHASE_NAMES
        assert 0.0 <= p["confidence"] <= 1.0
        assert {k["name"] for k in p["top_k"]} <= set(PHASE_NAMES)
    assert "out of range" in preds[3]["error"]


def test_load_route_switches_the_live_model(two_runs):
    client, first_id, second_id = two_runs
    r = client.post("/api/ml/models/load", json={"run_id": first_id})
    assert r.status_code == 200, r.text
    assert r.json()["loaded_run_id"] == first_id
    assert sorted(r.json()["phase_names"]) == sorted(PHASE_NAMES)
    assert client.get("/api/ml/status").json()["loaded_run_id"] == first_id
    # and a restart afterwards goes back to the newest, not to the last clicked
    _simulate_restart()
    assert client.get("/api/ml/status").json()["loaded_run_id"] == second_id


@pytest.mark.parametrize("run_id", ["no-such-run", "../checkpoints", "..", "", "legacy"])
def test_load_route_404_for_unknown_run(two_runs, run_id):
    client, _first_id, second_id = two_runs
    r = client.post("/api/ml/models/load", json={"run_id": run_id})
    assert r.status_code == 404, (run_id, r.text)
    assert client.get("/api/ml/status").json()["loaded_run_id"] == second_id  # untouched


def test_flat_legacy_checkpoint_is_listed_last_and_loadable(two_runs, tmp_path):
    client, first_id, second_id = two_runs
    root = tmp_path / "ml_training_data" / "checkpoints"
    (root / "best.pt").write_bytes((root / second_id / "best.pt").read_bytes())
    models = client.get("/api/ml/models").json()["models"]
    assert [m["run_id"] for m in models] == [second_id, first_id, "legacy"]
    r = client.post("/api/ml/models/load", json={"run_id": "legacy"})
    assert r.status_code == 200, r.text
    assert client.get("/api/ml/status").json()["loaded_run_id"] == "legacy"


def test_corrupt_newest_run_falls_back_to_the_older_one(two_runs, tmp_path):
    client, first_id, second_id = two_runs
    root = tmp_path / "ml_training_data" / "checkpoints"
    (root / second_id / "best.pt").write_bytes(b"not a checkpoint")
    _simulate_restart()
    body = client.get("/api/ml/status").json()
    assert body["model_loaded"] is True
    assert body["loaded_run_id"] == first_id


def test_a_run_that_failed_to_load_is_not_retried_on_every_poll(two_runs, tmp_path, monkeypatch):
    client, first_id, second_id = two_runs
    root = tmp_path / "ml_training_data" / "checkpoints"
    (root / second_id / "best.pt").write_bytes(b"not a checkpoint")
    (root / first_id / "best.pt").write_bytes(b"not a checkpoint either")
    _simulate_restart()
    attempts = []
    real_load_run = ml_controller.load_run

    def counting_load_run(run_id):
        attempts.append(run_id)
        return real_load_run(run_id)

    monkeypatch.setattr(ml_controller, "load_run", counting_load_run)
    for _ in range(3):
        assert client.get("/api/ml/status").json()["model_loaded"] is False
    assert sorted(attempts) == sorted([first_id, second_id])


def test_run_folder_without_checkpoints_is_not_listed(isolated_ml, tmp_path):
    (tmp_path / "ml_training_data" / "checkpoints" / "20260101T000000Z").mkdir(parents=True)
    assert isolated_ml.get("/api/ml/models").json()["models"] == []
    assert isolated_ml.get("/api/ml/status").json()["model_loaded"] is False


def test_train_uses_only_this_requests_samples_unless_asked(isolated_ml, monkeypatch):
    """Two indexing results with different phases, trained one after the other.

    The store keeps both rounds (it did in the B4 real-data run too, where it
    had to be cleared by hand). The second run must not inherit the first
    run's samples or phases unless the request says include_existing_store.
    """
    import backend.api.routes.ebsd_viewer as ev
    import backend.api.routes.indexing as idx

    signal = kikuchipy.data.nickel_ebsd_small()
    monkeypatch.setattr(ev, "_get_active_signal", lambda: signal)
    ids = [0, 0, 0, 0, 0, 1, 1, 1, 1]

    monkeypatch.setattr(ev, "_ebsd_file_path", "one/A.h5oina")
    monkeypatch.setattr(idx, "get_last_indexing_result",
                        lambda: _fake_indexing_result(signal, ids, ["Ni", "Fe"]))
    first = _train_and_wait(isolated_ml, LR_FIRST)["result"]

    time.sleep(1.1)  # run ids are second-resolution timestamps
    monkeypatch.setattr(ev, "_ebsd_file_path", "two/B.h5oina")
    monkeypatch.setattr(idx, "get_last_indexing_result",
                        lambda: _fake_indexing_result(signal, ids, ["Cu", "Zn"]))
    second = _train_and_wait(isolated_ml, LR_SECOND)["result"]

    time.sleep(1.1)
    third = _train_and_wait(isolated_ml, LR_SECOND, include_existing_store=True)["result"]

    by_id = {m["run_id"]: m for m in isolated_ml.get("/api/ml/models").json()["models"]}
    assert len(by_id) == 3

    assert sorted(first["phase_names"]) == ["Fe", "Ni"]
    assert by_id[first["run_id"]]["n_samples"] == 9

    # Default: only the samples of this request, only their phases.
    assert sorted(second["phase_names"]) == ["Cu", "Zn"]
    m2 = by_id[second["run_id"]]
    assert m2["n_samples"] == 9
    assert m2["source_file"] == "B.h5oina"
    assert m2["sample_sources"] == {
        "this_request": {"n_samples": 9, "source_file": "B.h5oina"},
        "existing_store": {"available": 9, "included": False, "n_samples": 0},
    }

    # Explicitly asked for: everything in the store, phases of both results.
    assert sorted(third["phase_names"]) == ["Cu", "Fe", "Ni", "Zn"]
    m3 = by_id[third["run_id"]]
    assert m3["n_samples"] == 27
    assert m3["sample_sources"] == {
        "this_request": {"n_samples": 9, "source_file": "B.h5oina"},
        "existing_store": {"available": 18, "included": True, "n_samples": 18},
    }

    # The store itself still accumulates, and /store/clear still empties it.
    assert isolated_ml.get("/api/ml/status").json()["store_stats"]["total_samples"] == 27
    r = isolated_ml.post("/api/ml/store/clear")
    assert r.status_code == 200, r.text
    assert r.json() == {"cleared": True, "samples_removed": 27}


def test_predict_without_any_model_says_so(isolated_ml, monkeypatch):
    import backend.api.routes.ebsd_viewer as ev
    signal = kikuchipy.data.nickel_ebsd_small()
    monkeypatch.setattr(ev, "_get_active_signal", lambda: signal)
    r = isolated_ml.post("/api/ml/predict", json={"pattern_indices": [0]})
    assert r.status_code == 200
    assert "No trained model" in r.json()["predictions"][0]["error"]
