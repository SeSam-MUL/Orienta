"""The reconstruction task payload carries the grain statistics under `result`.

The Analysis page reads these keys (frontend/src/components/Analysis/grainStats.js);
this test pins the producer side so the two cannot drift apart again.
"""
import pytest
from fastapi.testclient import TestClient

from analysis.ebsd_dataset import EBSDDataset
from tests.synthetic_grain_map import make_map, to_xmap

STAT_KEYS = ("grain_count", "mean_ecd", "median_ecd", "std_ecd", "mean_aspect_ratio")


@pytest.fixture
def client_with_map():
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    m = make_map()
    saved = ar._dataset
    ar._dataset = EBSDDataset(to_xmap(m), step_size=m["step"])
    try:
        yield TestClient(app)
    finally:
        ar._dataset = saved


def _task(client, body):
    r = client.post("/api/analysis/grains/reconstruct", json=body)
    assert r.status_code == 200
    return client.get(f"/api/analysis/task/{r.json()['task_id']}").json()


def test_completed_task_carries_the_five_statistics_under_result(client_with_map):
    task = _task(client_with_map, {"misorientation_threshold": 5.0, "min_grain_size": 1})
    assert task["status"] == "completed"
    result = task["result"]
    for key in STAT_KEYS:
        assert key in result, key
        assert isinstance(result[key], (int, float)), key
    assert result["grain_count"] > 0
    assert result["mean_ecd"] > 0 and result["mean_aspect_ratio"] >= 1.0


def test_failed_task_reports_failed_with_the_error(client_with_map, monkeypatch):
    import analysis.grain_analysis as ga

    def boom(dataset, **kw):
        raise RuntimeError("no grains today")

    monkeypatch.setattr(ga, "reconstruct_grains", boom)
    task = _task(client_with_map, {"misorientation_threshold": 5.0, "min_grain_size": 1})
    assert task["status"] == "failed"
    assert task["error"] == "no grains today"
    assert "result" not in task
