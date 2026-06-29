"""GPU runtime detection, memory-aware batch sizing, and OOM recovery.

Used by ``SphericalGPUBackend`` to make the pipeline robust across
different hardware: 4 GB laptops, 12 GB workstations, no-GPU laptops,
and bandwidths from L=53 (small) to L=113 (huge cc volume).

Memory model
------------
For each batch of size ``b`` at bandwidth ``L`` the dominant allocations
during ``rs2cc_sparse`` + irfftn are:

  * cc volume (output)        : b · (2L-1)^3 · 4 bytes (float32)
  * spectrum (irfftn input)   : b · L · (2L-1)^2 · 8 bytes (complex64)
  * F_lmk + G_lkn intermediates: ~b · La · (2L-1)^2 · 8 bytes (each)
  * pattern_coefs             : b · L · La · 8 bytes (small)

cufft also allocates workspace internally (~50% of input size). We
budget ~3.5x the spectrum tensor as the safe peak, which empirically
matches what we observe on the RTX 4070 (b=88 at L=68 uses ~5 GB of
the 12 GB total).

Bandwidth coverage
------------------
Validated batch sizes for each L on a 12 GB card:
   L=53  → b=176
   L=68  → b=88   (validated, 1033 pat/s on HiGainNi)
   L=88  → b=40
   L=113 → b=16
   L=158 → batch=1 doesn't fit, raises a clear error
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import torch


# Memory cost coefficient: bytes per pattern per ``(2L-1)^3`` element at
# peak of rs2cc_sparse + irfftn. Calibrated on RTX 4070 from the iter-15L
# profile: at L=68, b=88 the peak is ~5 GB, giving 5e9 / 88 / 135^3 ≈ 22.
# We round up to 24 to absorb cufft workspace variance + the F_lmk / G_lkn
# / spectrum / cc volume working set with a small safety margin.
_PEAK_BYTES_PER_PAT_PER_LCUBE = 24.0

# Safety margin: never use more than this fraction of free GPU memory.
# Leaves headroom for: pytorch caching allocator overhead, kernels we
# don't yet account for, plus the user's other GPU work.
_FREE_MEM_BUDGET_FRACTION = 0.7

# Hard floor: even on tiny GPUs we don't drop below batch=1.
_MIN_BATCH = 1
# Soft ceiling: above this, batches get launch-overhead bound rather
# than memory bound — no point picking larger batches.
_MAX_BATCH = 256

# Per-bandwidth ceilings: the largest batch we have *actually validated*
# end-to-end (95.28% accuracy on HiGainNi). Above these we silently lift
# to the validated value — they're known good. compute_safe_batch can go
# higher when free memory permits, but we don't *recommend* exceeding
# these without re-running the regression suite at the new batch size.
# Re-validate by running tasks/test_phase1_regression.py after raising.
_VALIDATED_BATCH_CEILING = {
    53: 256,    # smallest L; cc volume is 105^3 — easy
    68: 88,     # primary validation point (iter-15I/J/K/L on RTX 4070)
    88: 56,     # validated by tasks/test_phase2_runtime.py
    113: 24,    # validated by tasks/test_phase2_runtime.py
}


@dataclass
class RuntimeInfo:
    """Runtime environment description for diagnostic logging."""
    device: torch.device
    cuda_available: bool
    cuda_capability: Optional[tuple[int, int]]   # e.g. (8, 9) for Ada Lovelace
    device_name: str
    total_memory_gb: float
    free_memory_gb: float

    def __str__(self) -> str:                      # pragma: no cover (cosmetic)
        if self.cuda_available:
            cc = f"sm_{self.cuda_capability[0]}{self.cuda_capability[1]}"
            return (
                f"CUDA device: {self.device_name} ({cc}), "
                f"{self.free_memory_gb:.1f}/{self.total_memory_gb:.1f} GB free"
            )
        return f"CPU only ({self.device_name})"


def detect_runtime(device: Optional[torch.device] = None) -> RuntimeInfo:
    """Probe the chosen torch device for memory + capability metadata.

    Parameters
    ----------
    device
        If None, picks ``cuda`` when available, else ``cpu``.

    Returns
    -------
    RuntimeInfo with free/total memory in GB. On CPU memory is reported
    as the system's psutil-available virtual memory if installed, else 0.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        free_b, total_b = torch.cuda.mem_get_info(idx)
        props = torch.cuda.get_device_properties(idx)
        return RuntimeInfo(
            device=device,
            cuda_available=True,
            cuda_capability=(props.major, props.minor),
            device_name=props.name,
            total_memory_gb=total_b / (1024 ** 3),
            free_memory_gb=free_b / (1024 ** 3),
        )

    # CPU fallback. Best-effort memory estimate; not actually used for
    # CPU batch sizing (we use a small fixed batch on CPU anyway).
    try:
        import psutil
        vm = psutil.virtual_memory()
        free_gb = vm.available / (1024 ** 3)
        total_gb = vm.total / (1024 ** 3)
    except ImportError:
        free_gb = 0.0
        total_gb = 0.0
    return RuntimeInfo(
        device=device,
        cuda_available=False,
        cuda_capability=None,
        device_name="cpu",
        total_memory_gb=total_gb,
        free_memory_gb=free_gb,
    )


def estimate_batch_memory_bytes(batch: int, bandwidth: int) -> int:
    """Estimate peak GPU memory for a batch + bandwidth combination."""
    L = int(bandwidth)
    return int(batch * (2 * L - 1) ** 3 * _PEAK_BYTES_PER_PAT_PER_LCUBE)


def compute_safe_batch(
    bandwidth: int,
    runtime: Optional[RuntimeInfo] = None,
    user_override: Optional[int] = None,
) -> int:
    """Pick a safe batch size for the chosen bandwidth + runtime.

    Returns clamped to [1, 256]. CPU runtimes always return 16 (latency
    matters more than throughput on CPU). If a user explicitly sets
    ``user_override``, we just clamp it to [1, 256] and respect it.
    """
    if user_override is not None:
        return max(_MIN_BATCH, min(int(user_override), _MAX_BATCH))

    if runtime is None:
        runtime = detect_runtime()

    if not runtime.cuda_available:
        return 16

    budget_b = runtime.free_memory_gb * (1024 ** 3) * _FREE_MEM_BUDGET_FRACTION
    L = int(bandwidth)

    # Solve for batch: budget = batch · (2L-1)^3 · k
    #   batch = budget / ((2L-1)^3 · k)
    cube = (2 * L - 1) ** 3
    per_pat_bytes = cube * _PEAK_BYTES_PER_PAT_PER_LCUBE
    batch = int(budget_b / per_pat_bytes)

    # Round down to a multiple of 8 — empirically cufft has slightly
    # better throughput at multiple-of-8 batch sizes (warps + tile
    # boundaries align). Skip this rounding for tiny batches.
    if batch >= 16:
        batch = (batch // 8) * 8

    if batch < _MIN_BATCH:
        # The bandwidth is too large for this card even at batch=1.
        # Caller will get an OOM and can decide how to handle it.
        warnings.warn(
            f"compute_safe_batch: bandwidth L={L} estimated to need "
            f"~{per_pat_bytes / (1024 ** 3):.2f} GB per pattern, "
            f"but only {runtime.free_memory_gb:.1f} GB free. Returning "
            f"batch=1 - expect OOM or very slow execution.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _MIN_BATCH
    # Clamp to the validated ceiling for this bandwidth (review NIT #2):
    # higher batches can technically work on bigger cards but haven't
    # been regression-tested. Set ``user_override`` to bypass.
    validated_max = _VALIDATED_BATCH_CEILING.get(L, _MAX_BATCH)
    return min(batch, validated_max, _MAX_BATCH)


def is_oom_error(exc: Exception) -> bool:
    """Detect CUDA out-of-memory errors across torch versions."""
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cuda out of memory" in msg
        or "cublas_status_alloc_failed" in msg
    )


def run_with_oom_retry(fn, batch_size: int, min_batch: int = 1, max_retries: int = 3):
    """Call ``fn(batch_size)`` and on CUDA OOM retry with halved batch.

    Parameters
    ----------
    fn
        Callable accepting a single int (the batch size). Raises an OOM
        error when the batch is too large.
    batch_size
        Initial batch size to attempt.
    min_batch
        Lower bound — once we're at or below this we re-raise instead of
        further halving.
    max_retries
        Number of halvings allowed before giving up.

    Returns
    -------
    Whatever ``fn`` returns. Re-raises the OOM error when no fallback is
    possible (so the caller can surface it instead of silently failing).
    """
    current = max(int(batch_size), min_batch)
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return fn(current)
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            last_exc = exc
            torch.cuda.empty_cache()
            new_batch = max(current // 2, min_batch)
            if new_batch == current:
                # Already at the minimum — can't go lower.
                break
            warnings.warn(
                f"CUDA OOM at batch={current}, retrying with batch={new_batch} "
                f"(attempt {attempt + 1}/{max_retries})",
                RuntimeWarning,
                stacklevel=2,
            )
            current = new_batch
    # Out of retries — re-raise the last OOM with context.
    raise RuntimeError(
        f"CUDA OOM persists at batch={current} after {max_retries} retries; "
        f"the bandwidth is too large for this GPU. Consider lowering "
        f"bandwidth or running on a different machine."
    ) from last_exc
