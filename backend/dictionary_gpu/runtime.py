"""GPU runtime utilities: CUDA detection, batch sizing, OOM-retry."""
from __future__ import annotations

import logging
from typing import Callable, Tuple

import torch

logger = logging.getLogger(__name__)


def has_cuda() -> bool:
    return bool(torch.cuda.is_available())


def available_vram_bytes(device_index: int = 0) -> int:
    """Free VRAM on the given device. Returns 0 if no CUDA."""
    if not has_cuda():
        return 0
    free, _ = torch.cuda.mem_get_info(device_index)
    return int(free)


def estimate_batch_size(
    detector_shape: Tuple[int, int],
    available_bytes: int,
    fp32: bool = True,
    overhead_factor: float = 4.0,
    max_fraction: float = 0.25,
) -> int:
    """Pick a batch size that fits in `max_fraction` of available bytes.

    The overhead factor accounts for intermediate tensors (rotated directions
    `(N, P, 3)` is 3x the output, plus grid_sample temporary buffers).
    """
    H, W = detector_shape
    bytes_per_pat = H * W * (4 if fp32 else 2)
    budget = max(1, int(available_bytes * max_fraction))
    n = int(budget // (bytes_per_pat * overhead_factor))
    return max(64, n)


def run_with_oom_retry(
    op: Callable[[int], "torch.Tensor"],
    initial_batch: int,
    min_batch: int = 32,
    halvings: int = 4,
):
    """Call op(batch_size); on CUDA OOM halve and retry up to `halvings` times."""
    batch = initial_batch
    for _ in range(halvings + 1):
        try:
            return op(batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            new_batch = batch // 2
            if new_batch < min_batch:
                logger.error("OOM with batch=%d, below min_batch=%d", batch, min_batch)
                raise
            logger.warning("OOM at batch=%d, retrying with batch=%d", batch, new_batch)
            batch = new_batch
    raise RuntimeError("run_with_oom_retry: exhausted retries")
