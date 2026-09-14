"""Plumbing tests — verify compute_mode flows through schema → config → dispatcher."""
import pytest


def test_indexing_config_has_compute_mode_field():
    from indexing_controller import IndexingConfig
    cfg = IndexingConfig()
    assert hasattr(cfg, "compute_mode")
    assert cfg.compute_mode == "auto"


def test_indexing_config_compute_mode_accepts_three_values():
    from indexing_controller import IndexingConfig
    for v in ("auto", "gpu", "cpu"):
        cfg = IndexingConfig(compute_mode=v)
        assert cfg.compute_mode == v


def test_dictionary_request_schema_has_compute_mode():
    """The schema used by the dictionary indexing route must expose
    compute_mode. The current backend uses a shared IndexingStartRequest
    for hough/dictionary/spherical — that's where compute_mode lives.
    The field is harmless on non-dict methods (they ignore it)."""
    from backend.api.routes import indexing as routes
    # Try Dict-named models first (per-method schema), else fall back to
    # any BaseModel that contains the dictionary metric/keep_n fields,
    # which uniquely identifies the dictionary-indexing request shape.
    candidates = [
        getattr(routes, n) for n in dir(routes)
        if "Dict" in n and hasattr(getattr(routes, n), "model_fields")
    ]
    if not candidates:
        # Fall back: find the request model used for dictionary indexing
        # — recognised by having both `metric` and `keep_n` fields.
        candidates = [
            getattr(routes, n) for n in dir(routes)
            if hasattr(getattr(routes, n), "model_fields")
            and "metric" in getattr(routes, n).model_fields
            and "keep_n" in getattr(routes, n).model_fields
        ]
    assert candidates, "no dictionary request model found"
    found = False
    for m in candidates:
        if "compute_mode" in m.model_fields:
            found = True
            break
    assert found, "no dictionary request model exposes compute_mode"


def test_dispatcher_imports_gpu_path():
    """After Task 16, the controller can import the GPU path without errors."""
    import indexing_controller
    assert indexing_controller is not None
    # Verify that the dispatcher's symbol is reachable
    from backend.dict_gpu.api import gpu_dictionary_index_patterns
    assert callable(gpu_dictionary_index_patterns)


def test_gpu_status_endpoint_returns_expected_keys():
    """GET /api/system/gpu returns the GpuStatus shape as JSON."""
    from fastapi.testclient import TestClient
    from backend.api.main import app
    with TestClient(app) as client:
        r = client.get("/api/system/gpu")
        assert r.status_code == 200, r.text
        d = r.json()
        for key in ("available", "name", "vram_total_gb", "vram_free_gb"):
            assert key in d, f"missing key {key!r} in response: {d}"
        assert isinstance(d["available"], bool)
        assert isinstance(d["name"], str)
        assert isinstance(d["vram_total_gb"], (int, float))
        assert isinstance(d["vram_free_gb"], (int, float))
