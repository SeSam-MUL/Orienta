"""Custom CUDA kernels for the spherical-gpu indexer hot path.

perf Q0.4 (DONE iter 19, 2026-05-09): a `cupy.RawKernel` that fuses
``nc_vol = cc * rDen ; (max_vals, max_idx) = nc_vol.flat.max()`` into a single
GPU kernel. Eliminates the 21 MB DRAM round-trip between the multiply and
the reduction at L=88 (nc_vol shape (B, 175, 175, 175) FP32 = 21.4 MB / pattern).

Why a RawKernel and not torch.compile / cupy.fuse:
- ``torch.compile(mode="default")`` with PyTorch 2.11 + Triton-Windows 3.6 added
  ~91 ms/batch overhead even after warmup (Q0.2 reject, commit 76f883e). The
  Inductor dispatch path on Windows is too heavy for our 38 ms / batch hot path.
- ``cupy.fuse`` only fuses elementwise ops, NOT reductions, so mul+max stays
  two kernels (verified iter 19 bench: 9.49 ms, slower than torch eager).
- A custom CUDA kernel via ``cupy.RawKernel`` is the only path that gets us
  below torch eager's 5.96 ms baseline.

Standalone prototype bench (B=32, 175^3, FP32, RTX 4070):
    torch eager mul+max:                  5.96 ms median
    cupy separate mul+max:                9.44 ms
    cupy.fuse mul + separate max:         9.49 ms
    THIS RawKernel (mul+max+argmax fused): 4.73 ms (-21% vs torch eager)

Falls back to torch eager if cupy is not importable, so the indexer remains
runnable on systems without cupy installed.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


# Lazy-imported cupy (so import-time of this module is cheap and doesn't
# explode if cupy/CUDA libs are missing).
_CUPY = None
_CUPY_AVAILABLE: Optional[bool] = None


def _try_import_cupy():
    global _CUPY, _CUPY_AVAILABLE
    if _CUPY_AVAILABLE is None:
        try:
            import cupy as cp  # noqa
            # Touch a runtime API to make sure CUDA libs actually load.
            cp.cuda.runtime.runtimeGetVersion()
            _CUPY = cp
            _CUPY_AVAILABLE = True
        except Exception:
            _CUPY = None
            _CUPY_AVAILABLE = False
    return _CUPY


# ---------------------------------------------------------------------------
# Kernel source
# ---------------------------------------------------------------------------
# Each block handles ONE batch element. Threads cooperatively scan the
# (size^3 = 5,359,375) bins of one pattern's nc_vol in stripes of blockDim.x,
# accumulate a per-thread (max_value, max_index), then do a shared-memory
# reduction down to one (max_value, max_index) per block.
#
# rDen is shape (size, size, size) — broadcast over the batch axis.
# nc_out is the materialized (B, size, size, size) volume — needed by
# triquadratic_subbin downstream, so we cannot skip writing it.
_KERNEL_SOURCE = r"""
extern "C" __global__
void fused_mul_max_argmax(
    const float* __restrict__ cc,
    const float* __restrict__ rDen,
    float* __restrict__ nc_out,
    float* __restrict__ max_vals,
    long long* __restrict__ max_idx,
    const int B,
    const int VOL
) {
    const int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ char smem[];
    float* sval = (float*)smem;
    int*   sidx = (int*)(sval + blockDim.x);

    float local_max = -1e30f;
    int   local_idx = 0;

    const int tid    = threadIdx.x;
    const int stride = blockDim.x;
    const int b_off  = b * VOL;

    for (int i = tid; i < VOL; i += stride) {
        const float v = cc[b_off + i] * rDen[i];
        nc_out[b_off + i] = v;
        if (v > local_max) { local_max = v; local_idx = i; }
    }

    sval[tid] = local_max;
    sidx[tid] = local_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sval[tid + s] > sval[tid]) {
                sval[tid] = sval[tid + s];
                sidx[tid] = sidx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        max_vals[b] = sval[0];
        max_idx[b]  = (long long)sidx[0];
    }
}
"""


# max-only variant: identical to fused_mul_max_argmax but DOES NOT write
# nc_out. Saves the 21.4 MB/pattern DRAM round-trip (654 MB at B=32, L=88).
# Sub-bin refinement reads only 27 (cc * rDen) values per pattern via
# gather (see backend.spherical_gpu.pipeline.refiner.gather_n27_from_cc_rden),
# so materialising the full volume only to read 27 elements is wasted I/O.
#
# Microbench (RTX 4070, B=32, S=175, FP32):
#   fused_mul_max_argmax (with nc_out write):  5.06 ms median
#   fused_max_only       (no    nc_out write): 1.99 ms median   (2.55x)
# Correctness: bit-identical max_vals + argmax (32/32 same idx, diff = 0).
_KERNEL_SOURCE_MAX_ONLY = r"""
extern "C" __global__
void fused_max_only(
    const float* __restrict__ cc,
    const float* __restrict__ rDen,
    float* __restrict__ max_vals,
    long long* __restrict__ max_idx,
    const int B,
    const int VOL
) {
    const int b = blockIdx.x;
    if (b >= B) return;
    extern __shared__ char smem[];
    float* sval = (float*)smem;
    int*   sidx = (int*)(sval + blockDim.x);
    float local_max = -1e30f;
    int   local_idx = 0;
    const int tid    = threadIdx.x;
    const int stride = blockDim.x;
    const int b_off  = b * VOL;
    for (int i = tid; i < VOL; i += stride) {
        const float v = cc[b_off + i] * rDen[i];
        if (v > local_max) { local_max = v; local_idx = i; }
    }
    sval[tid] = local_max;
    sidx[tid] = local_idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sval[tid + s] > sval[tid]) {
                sval[tid] = sval[tid + s];
                sidx[tid] = sidx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        max_vals[b] = sval[0];
        max_idx[b]  = (long long)sidx[0];
    }
}
"""


_KERNEL = None
_KERNEL_MAX_ONLY = None


def _get_kernel():
    """Lazy-compile the RawKernel on first use."""
    global _KERNEL
    if _KERNEL is None:
        cp = _try_import_cupy()
        if cp is None:
            raise RuntimeError("cupy not available")
        _KERNEL = cp.RawKernel(_KERNEL_SOURCE, "fused_mul_max_argmax")
    return _KERNEL


def _get_kernel_max_only():
    """Lazy-compile the max-only RawKernel on first use."""
    global _KERNEL_MAX_ONLY
    if _KERNEL_MAX_ONLY is None:
        cp = _try_import_cupy()
        if cp is None:
            raise RuntimeError("cupy not available")
        _KERNEL_MAX_ONLY = cp.RawKernel(_KERNEL_SOURCE_MAX_ONLY, "fused_max_only")
    return _KERNEL_MAX_ONLY


_THREADS_PER_BLOCK = 256
_SHARED_MEM_BYTES = _THREADS_PER_BLOCK * (4 + 4)  # float + int per thread


def fused_mul_max(
    cc_real_f32: torch.Tensor,
    rDen: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (cc * rDen, max_values, max_indices) in one fused kernel.

    Inputs:
        cc_real_f32: (B, S, S, S) float32 CUDA tensor — real part of cc volume.
        rDen:        (1, S, S, S) or (S, S, S) float32 CUDA tensor — NC denominator.

    Returns:
        nc_vol:    (B, S, S, S) float32 — the materialized cc * rDen volume.
        max_vals:  (B,) float32 — per-batch maximum.
        max_idx:   (B,) int64 — per-batch argmax (flat index into S^3 bins).

    On systems without cupy, falls back to torch eager (mul + flat.max).
    """
    if not cc_real_f32.is_cuda:
        raise ValueError("fused_mul_max requires CUDA tensors")
    cp = _try_import_cupy()
    if cp is None:
        # Eager torch fallback — same math, two kernels.
        nc_vol = cc_real_f32 * rDen
        B = nc_vol.shape[0]
        flat = nc_vol.reshape(B, -1)
        m = flat.max(dim=1)
        return nc_vol, m.values, m.indices

    # cupy path — zero-copy via __cuda_array_interface__.
    cc_real_f32 = cc_real_f32.contiguous()
    rDen_flat = rDen.reshape(-1).contiguous()
    B = cc_real_f32.shape[0]
    S = cc_real_f32.shape[-1]
    VOL = S * S * S
    if rDen_flat.numel() != VOL:
        raise ValueError(
            f"rDen flat size {rDen_flat.numel()} != VOL {VOL} (S={S})"
        )

    cp_cc = cp.asarray(cc_real_f32)
    cp_rDen = cp.asarray(rDen_flat)
    cp_nc = cp.empty((B, S, S, S), dtype=cp.float32)
    cp_vals = cp.empty(B, dtype=cp.float32)
    cp_idx = cp.empty(B, dtype=cp.int64)

    kernel = _get_kernel()
    kernel(
        (B,), (_THREADS_PER_BLOCK,),
        (cp_cc, cp_rDen, cp_nc, cp_vals, cp_idx,
         np.int32(B), np.int32(VOL)),
        shared_mem=_SHARED_MEM_BYTES,
    )

    # Bridge back to torch via DLPack (zero-copy).
    nc_vol = torch.from_dlpack(cp_nc)
    max_vals = torch.from_dlpack(cp_vals)
    max_idx = torch.from_dlpack(cp_idx)
    return nc_vol, max_vals, max_idx


def fused_max_only(
    cc_real_f32: torch.Tensor,
    rDen: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute (max_values, max_indices) of cc * rDen WITHOUT materialising
    the full nc_vol = cc * rDen volume.

    Pairs with ``refiner.gather_n27_from_cc_rden`` for sub-bin refinement:
    skips the 21.4 MB/pattern DRAM round-trip in exchange for re-reading
    27 elements of cc and rDen at the peak. Net saving on RTX 4070 at
    L=88, B=32: 5.06 ms -> 1.99 ms median (2.55x) on this kernel; plus
    one 654 MB allocation avoided per batch.

    Falls back to torch eager (mul + flat.max) if cupy is missing —
    in that fallback the nc_vol IS materialised (eager has no way to
    fuse without it), so callers should be prepared for either path.

    Inputs:
        cc_real_f32: (B, S, S, S) float32 CUDA tensor.
        rDen:        (1, S, S, S) or (S, S, S) float32 CUDA tensor.

    Returns:
        max_vals: (B,) float32 — per-batch maximum of cc * rDen.
        max_idx:  (B,) int64   — per-batch argmax (flat index into S^3).
    """
    if not cc_real_f32.is_cuda:
        raise ValueError("fused_max_only requires CUDA tensors")
    cp = _try_import_cupy()
    if cp is None:
        nc_vol = cc_real_f32 * rDen
        B = nc_vol.shape[0]
        flat = nc_vol.reshape(B, -1)
        m = flat.max(dim=1)
        return m.values, m.indices
    cc_real_f32 = cc_real_f32.contiguous()
    rDen_flat = rDen.reshape(-1).contiguous()
    B = cc_real_f32.shape[0]
    S = cc_real_f32.shape[-1]
    VOL = S * S * S
    if rDen_flat.numel() != VOL:
        raise ValueError(f"rDen flat size {rDen_flat.numel()} != VOL {VOL} (S={S})")
    cp_cc = cp.asarray(cc_real_f32)
    cp_rDen = cp.asarray(rDen_flat)
    cp_vals = cp.empty(B, dtype=cp.float32)
    cp_idx = cp.empty(B, dtype=cp.int64)
    kernel = _get_kernel_max_only()
    kernel(
        (B,), (_THREADS_PER_BLOCK,),
        (cp_cc, cp_rDen, cp_vals, cp_idx,
         np.int32(B), np.int32(VOL)),
        shared_mem=_SHARED_MEM_BYTES,
    )
    return torch.from_dlpack(cp_vals), torch.from_dlpack(cp_idx)


def is_cupy_kernel_available() -> bool:
    """True iff cupy + CUDA libs are importable AND the RawKernel compiles."""
    cp = _try_import_cupy()
    if cp is None:
        return False
    try:
        _get_kernel()
        return True
    except Exception:
        return False
