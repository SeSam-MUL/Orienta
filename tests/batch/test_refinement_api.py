"""Tests for refinement REST endpoints."""
import os
import numpy as np
import h5py
import pytest
from fastapi.testclient import TestClient
from backend.api.services.checkpoint_writer import CheckpointWriter


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app)


@pytest.fixture
def sample_multiphase(tmp_path):
    """Create a sample _multiphase.h5 for refinement tests."""
    source = str(tmp_path / "Sample_A.h5oina")
    cw = CheckpointWriter(source)
    grid = (5, 5)
    cw.init_metadata(grid_shape=grid, batch_id="test", method="spherical")
    cw.write_phase_result("Al", np.random.rand(*grid).astype(np.float32),
                          np.random.rand(*grid, 3).astype(np.float32),
                          {"phase_file": "/p", "ci_mean": 0.8, "duration_sec": 1.0})
    cw.write_phase_result("Fe", np.random.rand(*grid).astype(np.float32) * 0.5,
                          np.random.rand(*grid, 3).astype(np.float32),
                          {"phase_file": "/p", "ci_mean": 0.4, "duration_sec": 2.0})
    cw.compute_auto_assignment()
    return cw.checkpoint_path


class TestRefinementAPI:

    def test_summary(self, client, sample_multiphase):
        stem = "Sample_A"
        resp = client.get(f"/api/refinement/{stem}/summary",
                          params={"search_dir": str(os.path.dirname(sample_multiphase))})
        assert resp.status_code == 200
        data = resp.json()
        assert "phases" in data
        assert len(data["phases"]) == 2

    def test_pixel_info(self, client, sample_multiphase):
        stem = "Sample_A"
        resp = client.get(f"/api/refinement/{stem}/pixel/2/3",
                          params={"search_dir": str(os.path.dirname(sample_multiphase))})
        assert resp.status_code == 200
        data = resp.json()
        assert "phases" in data
        assert len(data["phases"]) == 2

    def test_override_pixel(self, client, sample_multiphase):
        stem = "Sample_A"
        resp = client.post(f"/api/refinement/{stem}/override",
                           json={"pixels": [[2, 3, 1]], "source": "click"},
                           params={"search_dir": str(os.path.dirname(sample_multiphase))})
        assert resp.status_code == 200
