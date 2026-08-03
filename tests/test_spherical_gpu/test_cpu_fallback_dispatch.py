"""Spherical indexing must not crash when there is no NVIDIA GPU.

A user on a GPU-less box hit:

    ValueError: fused_max_only requires CUDA tensors

``_fused_max_only_dispatch`` gated only on ``is_cupy_kernel_available()``,
which is a LIBRARY check — cupy's pip wheel ships the CUDA runtime, so it
answers True even with no NVIDIA device present. CPU tensors were then handed
to a CUDA-only RawKernel. The dispatch must key on the DEVICE OF THE DATA.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.spherical_gpu.pipeline import indexer as ix  # noqa: E402


def _reference(cc, rDen):
    nc = cc * rDen
    m = nc.reshape(nc.shape[0], -1).max(dim=1)
    return m.values, m.indices


@pytest.fixture
def cpu_inputs():
    torch.manual_seed(0)
    S = 7
    cc = torch.rand(3, S, S, S, dtype=torch.float32)
    rDen = torch.rand(1, S, S, S, dtype=torch.float32)
    return cc, rDen


def test_max_only_dispatch_runs_on_cpu_tensors(cpu_inputs):
    cc, rDen = cpu_inputs
    vals, idx = ix._fused_max_only_dispatch(cc, rDen)

    exp_vals, exp_idx = _reference(cc, rDen)
    torch.testing.assert_close(vals, exp_vals)
    assert torch.equal(idx, exp_idx)


def test_mul_max_dispatch_runs_on_cpu_tensors(cpu_inputs):
    cc, rDen = cpu_inputs
    nc, vals, idx = ix._fused_mul_max_dispatch(cc, rDen)

    exp_vals, exp_idx = _reference(cc, rDen)
    torch.testing.assert_close(nc, cc * rDen)
    torch.testing.assert_close(vals, exp_vals)
    assert torch.equal(idx, exp_idx)


def test_cpu_dispatch_ignores_a_true_cupy_probe(cpu_inputs, monkeypatch):
    """The exact user situation: cupy reports available, data is on the CPU.

    Pin the probe to True so the test reproduces it on any machine, and make
    the cupy kernel explode if it is ever reached.
    """
    cc, rDen = cpu_inputs
    monkeypatch.setattr(ix, "_CUPY_FUSION_AVAILABLE", True)

    def _boom(*_a, **_k):
        raise AssertionError("CPU tensors must never reach the cupy kernel")

    monkeypatch.setattr(ix, "_fused_max_only_cupy", _boom)
    monkeypatch.setattr(ix, "_fused_mul_max_cupy", _boom)

    vals, idx = ix._fused_max_only_dispatch(cc, rDen)
    exp_vals, exp_idx = _reference(cc, rDen)
    torch.testing.assert_close(vals, exp_vals)
    assert torch.equal(idx, exp_idx)

    ix._fused_mul_max_dispatch(cc, rDen)  # must not raise either


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_and_cpu_dispatch_agree(cpu_inputs):
    """The CPU fallback is a real alternative, not a degraded one."""
    cc, rDen = cpu_inputs
    cpu_vals, cpu_idx = ix._fused_max_only_dispatch(cc, rDen)
    cuda_vals, cuda_idx = ix._fused_max_only_dispatch(cc.cuda(), rDen.cuda())

    torch.testing.assert_close(cpu_vals, cuda_vals.cpu(), rtol=1e-6, atol=1e-6)
    assert torch.equal(cpu_idx, cuda_idx.cpu())
