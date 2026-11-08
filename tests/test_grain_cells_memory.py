"""The Qhull path of build_cells refuses a map it cannot hold in memory."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from analysis import grain_cells
from analysis.ebsd_dataset import EBSDDataset
from analysis.grain_cells import build_cells
from tests.synthetic_grain_map import add_defects, make_map, to_xmap

GB = 1e9


def _map_with_holes(ny=30, nx=40):
    gen = np.ones((ny, nx), bool)
    gen[10:12, 5:20] = False
    gen[3, 3] = False
    return gen


def test_estimate_above_available_memory_raises_with_the_numbers(monkeypatch):
    monkeypatch.setattr(grain_cells, "available_memory_bytes", lambda: int(0.5 * 1024 ** 2))
    gen = _map_with_holes()
    with pytest.raises(MemoryError) as e:
        build_cells(gen)
    msg = str(e.value)
    n = int(gen.sum())
    assert "30 x 40" in msg
    assert f"{n:,}" in msg                               # generators
    est_gb = n * grain_cells.QHULL_BYTES_PER_GENERATOR / GB
    assert f"{est_gb:.2f} GB" in msg                     # estimated need
    assert "0.00 GB available" in msg                    # what the machine has (0.5 MB)
    assert "crop" in msg.lower() and "more memory" in msg.lower()


def test_full_grid_never_refuses(monkeypatch):
    monkeypatch.setattr(grain_cells, "available_memory_bytes", lambda: 0)
    cells = build_cells(np.ones((20, 25), bool))
    assert cells.shape == (20, 25) and cells.generator.all()


def test_estimate_below_available_memory_builds(monkeypatch):
    monkeypatch.setattr(grain_cells, "available_memory_bytes", lambda: int(64 * GB))
    gen = _map_with_holes()
    cells = build_cells(gen)
    assert cells.shape == gen.shape


def test_the_estimate_is_2_2_gb_per_million_generators():
    assert grain_cells.QHULL_BYTES_PER_GENERATOR * 1_000_000 == pytest.approx(2.2e9)


def test_the_default_lookup_reports_real_memory():
    assert grain_cells.available_memory_bytes() > 0


def test_the_route_reports_a_refused_map_as_a_failed_task(monkeypatch):
    import backend.api.routes.analysis as ar
    from backend.api.main import app

    m = add_defects(make_map())
    saved = ar._dataset
    ar._dataset = EBSDDataset(to_xmap(m), step_size=m["step"])
    monkeypatch.setattr(grain_cells, "available_memory_bytes", lambda: 1024)
    try:
        client = TestClient(app)
        r = client.post("/api/analysis/grains/reconstruct",
                        json={"misorientation_threshold": 5.0, "min_grain_size": 1})
        assert r.status_code == 200
        task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    finally:
        ar._dataset = saved
    assert task["status"] == "failed"
    assert "60 x 80" in task["error"] and "crop" in task["error"].lower()
    assert "result" not in task
