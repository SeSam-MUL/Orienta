"""CUDA detection and VRAM probe.

This module imports torch *lazily* so that environments without torch
(or without a CUDA build of torch) don't crash on import. The frontend
GPU-status endpoint depends on this not crashing.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuStatus:
    available: bool
    name: str
    vram_total_gb: float
    vram_free_gb: float


def detect_gpu() -> GpuStatus:
    try:
        import torch
        if torch is None or not torch.cuda.is_available():
            return GpuStatus(False, "", 0.0, 0.0)
    except Exception:
        return GpuStatus(False, "", 0.0, 0.0)

    idx = 0
    name = torch.cuda.get_device_name(idx)
    free, total = torch.cuda.mem_get_info(idx)
    return GpuStatus(
        available=True,
        name=name,
        vram_total_gb=total / 1e9,
        vram_free_gb=free / 1e9,
    )


def vram_budget_bytes(reserve_gb: float = 2.0) -> int:
    """Default VRAM budget = (free - reserve_gb) clamped to >= 1 GB."""
    s = detect_gpu()
    if not s.available:
        return 0
    budget_gb = max(1.0, s.vram_free_gb - reserve_gb)
    return int(budget_gb * 1e9)
