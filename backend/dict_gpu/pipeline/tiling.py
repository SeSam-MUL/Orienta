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


# How much VRAM the resident path peaks at, as a multiple of the fp32
# dictionary. Measured through the real ``run_single_phase_method`` on this
# project's RTX 4070 (tasks/multiphase_vram_audit/m4_*.py): 3.76x at both
# 40,000 and 80,000 entries. It is the dictionary tensor, its flattened copy,
# the normalised copy, and the transient inside the normalisation, plus the
# experimental and top-k buffers.
RESIDENT_PEAK_FACTOR = 3.76

#: Below this many entries per tile the streaming loop is not worth starting —
#: the per-tile launch overhead dominates and the message the user needs is
#: "this will not fit", not a run that crawls.
MIN_STREAM_TILE = 16


def stream_bytes_per_entry(pattern_dim: int, feat_dim: int, n_query: int) -> int:
    """Peak device bytes one dictionary entry costs inside the tile loop.

    Counting what is alive at the same moment, per entry of the tile:

    * ``pattern_dim * 4`` — the fp32 block as it lands on the device;
    * ``feat_dim * 4`` — the masked copy, only when a detector mask drops
      columns (without a mask the block itself is normalised in place);
    * ``feat_dim * 2`` — the fp16 copy ``gemm_topk_ncc`` makes for the GEMM;
    * ``n_query * 6`` — this entry's column of the score matrix, which that
      kernel materialises once as fp16 and once as fp32.

    The score term is the one that is easy to forget and the one that bites on
    a full map: at 28,086 selected pixels it is 168 KB per dictionary entry,
    more than the 120 KB the entry itself costs at a 128x156 detector.
    """
    if pattern_dim <= 0 or feat_dim <= 0 or n_query <= 0:
        raise ValueError("pattern_dim, feat_dim, n_query must all be > 0")
    masked_copy = feat_dim * 4 if feat_dim < pattern_dim else 0
    return pattern_dim * 4 + masked_copy + feat_dim * 2 + n_query * 6


def compute_stream_tile_size(
    n_dict: int,
    pattern_dim: int,
    feat_dim: int,
    n_query: int,
    vram_budget_bytes: int,
    *,
    headroom_fraction: float = 0.40,
    chunk_entries: int = 1,
) -> int:
    """Largest streamed tile that fits in (1 - headroom) of the budget.

    ``chunk_entries`` rounds the tile down to a whole number of HDF5 chunks
    when the source is a file, so a block never asks the library for a chunk
    it will only use a slice of.

    Returns 0 when not even :data:`MIN_STREAM_TILE` entries fit — the caller
    turns that into a message with the numbers in it rather than an OOM.
    """
    if n_dict <= 0:
        raise ValueError(f"n_dict must be > 0, got {n_dict}")
    if vram_budget_bytes <= 0:
        raise ValueError("vram_budget_bytes must be > 0")
    if not (0 <= headroom_fraction < 1):
        raise ValueError(f"headroom_fraction must be in [0, 1), got {headroom_fraction}")

    per_entry = stream_bytes_per_entry(pattern_dim, feat_dim, n_query)
    available = vram_budget_bytes * (1.0 - headroom_fraction)
    tile = int(available // per_entry)
    if tile < MIN_STREAM_TILE:
        return 0
    tile = min(tile, n_dict)
    if chunk_entries > 1 and tile > chunk_entries:
        tile = (tile // chunk_entries) * chunk_entries
    return max(1, tile)


def resident_peak_bytes(n_dict: int, pattern_dim: int, feat_dim: int) -> int:
    """What the whole-dictionary-on-the-GPU path is expected to peak at.

    A detector mask adds one further full copy (``dict_flat[:, keep_cols]``
    copies), which is why the masked case is counted on top of the measured
    factor rather than inside it.
    """
    fp32 = n_dict * pattern_dim * 4
    extra = n_dict * feat_dim * 4 if feat_dim < pattern_dim else 0
    return int(fp32 * RESIDENT_PEAK_FACTOR) + extra


def should_stream(
    n_dict: int,
    pattern_dim: int,
    feat_dim: int,
    vram_budget_bytes: int,
    *,
    safety_fraction: float = 0.80,
) -> bool:
    """The one rule that decides between resident and streamed.

    Streaming when the resident path would not fit, and only then, so the
    small case keeps the path it was validated on.
    """
    if vram_budget_bytes <= 0:
        return True
    return resident_peak_bytes(n_dict, pattern_dim, feat_dim) > \
        safety_fraction * vram_budget_bytes
