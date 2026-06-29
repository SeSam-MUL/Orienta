"""VRAM-aware tiling helpers for the GPU dictionary indexer.

These are pure CPU functions - they decide *how* to tile, the indexer
does the actual GPU work in slices.
"""
from __future__ import annotations
from typing import Iterator


def compute_tile_size(
    n_dict: int,
    pattern_dim: int,
    dtype_bytes: int,
    vram_budget_bytes: int,
    *,
    headroom_fraction: float = 0.30,
) -> int:
    """Largest tile that fits in (1 - headroom) of the VRAM budget.

    The dictionary tile occupies `tile * pattern_dim * dtype_bytes` bytes.
    We constrain that to `(1 - headroom) * vram_budget_bytes` so the GEMM
    output, normalisation buffers, and PyTorch's allocator overhead have
    breathing room.
    """
    if n_dict <= 0:
        raise ValueError(f"n_dict must be > 0, got {n_dict}")
    if pattern_dim <= 0 or dtype_bytes <= 0 or vram_budget_bytes <= 0:
        raise ValueError("pattern_dim, dtype_bytes, vram_budget_bytes must all be > 0")
    if not (0 <= headroom_fraction < 1):
        raise ValueError(f"headroom_fraction must be in [0, 1), got {headroom_fraction}")

    available = vram_budget_bytes * (1.0 - headroom_fraction)
    bytes_per_entry = pattern_dim * dtype_bytes
    tile = int(available // bytes_per_entry)
    tile = max(1, min(tile, n_dict))
    return tile


def iter_tiles(n_total: int, tile: int) -> Iterator[slice]:
    """Yield slices [0:tile, tile:2*tile, ...] covering [0, n_total)."""
    if tile <= 0:
        raise ValueError(f"tile must be > 0, got {tile}")
    if n_total <= 0:
        return
    start = 0
    while start < n_total:
        end = min(start + tile, n_total)
        yield slice(start, end)
        start = end
