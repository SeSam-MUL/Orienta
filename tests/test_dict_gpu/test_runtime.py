import pytest
from backend.dict_gpu.runtime import detect_gpu, GpuStatus

def test_detect_gpu_returns_gpu_status_dataclass():
    status = detect_gpu()
    assert isinstance(status, GpuStatus)
    assert isinstance(status.available, bool)

def test_when_cuda_present_status_includes_name_and_vram(has_cuda):
    if not has_cuda:
        pytest.skip("no CUDA")
    s = detect_gpu()
    assert s.available is True
    assert s.name and len(s.name) > 0
    assert s.vram_total_gb > 0
    assert 0 <= s.vram_free_gb <= s.vram_total_gb

def test_when_no_cuda_status_is_unavailable_and_no_torch_required(monkeypatch):
    # Simulate a torch-less environment by intercepting the import
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)
    # detect_gpu must not crash even when torch is unavailable
    import importlib, backend.dict_gpu.runtime as rt
    importlib.reload(rt)
    s = rt.detect_gpu()
    assert s.available is False
    assert s.name == ""
