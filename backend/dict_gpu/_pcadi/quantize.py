"""Dynamic per-row INT8 quantize/dequantize helpers.

Inspired by ZacharyVarley/pcadi LinearLayer (utils.py ~4650, utils_knn.py ~8)
but trimmed to the quantize/dequantize round-trip. We do not need the
quantized linear-forward path because our usage is: store the dictionary
in INT8 to save VRAM, then dequantize on-the-fly during the GEMM.

Source repo: https://github.com/ZacharyVarley/pcadi  (commit 88f676e)
License: MIT (see ../__SOURCE.md)
"""
from __future__ import annotations
import torch


_INT8_MIN = -128
_INT8_MAX = 127


def quantize_int8(
    X: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantise a (n, d) FP tensor to INT8 with per-row scale + zero-point.

    Returns
    -------
    q : (n, d) int8
    scale : (n,) float32
    zero_point : (n,) float32

    Round-trip: ``X_hat = (q.float() - zero_point[:, None]) * scale[:, None]``
    is bounded to within one INT8 level of X (~1% of the per-row dynamic
    range for non-constant rows; exact for constant rows).
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {tuple(X.shape)}")

    Xf = X.to(torch.float32)
    row_min = Xf.min(dim=1).values        # (n,)
    row_max = Xf.max(dim=1).values        # (n,)
    row_range = row_max - row_min          # (n,) — zero on constant rows

    # Avoid divide-by-zero on constant rows: use scale=1 and zero_point=row_value
    # so the round-trip yields the constant exactly.
    is_const = row_range == 0
    scale = torch.where(
        is_const,
        torch.ones_like(row_range),
        row_range / (_INT8_MAX - _INT8_MIN),
    )
    zero_point = torch.where(
        is_const,
        row_min,                                       # constant row value
        torch.full_like(row_min, float(_INT8_MIN)) - row_min / scale,
    )

    # Quantise: q = round(X / scale + zero_point), clamp to [-128, 127]
    q_float = Xf / scale[:, None] + zero_point[:, None]
    q_clamped = q_float.clamp(_INT8_MIN, _INT8_MAX).round()
    q = q_clamped.to(torch.int8)
    return q, scale, zero_point


def dequantize_int8(
    q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """Inverse of quantize_int8. Returns FP32."""
    if q.dtype != torch.int8:
        raise ValueError(f"q must be int8, got {q.dtype}")
    return (q.to(torch.float32) - zero_point[:, None]) * scale[:, None]
