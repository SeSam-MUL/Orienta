"""Tests for batch v2 REST endpoints."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app)


class TestBatchV2API:

    def test_create_batch_requires_files(self, client):
        resp = client.post("/api/batch-v2/create", json={"files": [], "phases": [], "config": {}})
        assert resp.status_code == 400

    def test_list_batches_empty(self, client):
        resp = client.get("/api/batch-v2/list")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_status_not_found(self, client):
        resp = client.get("/api/batch-v2/nonexistent-id/status")
        assert resp.status_code == 404
