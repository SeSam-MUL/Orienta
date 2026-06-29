"""Wigner-d matrices — unified facade over the two vendored implementations.

Two numerical strategies live in ebsdtorch:
- log-space arithmetic (`_wigner_logspace`) — stable up to about j=32
- extended-precision (`_wigner_xnum`) — needed beyond j=32 (mantissa overflow)

This wrapper picks the right one based on bandwidth, so callers don't need to
know about the dual implementation.

Reference: ebsdtorch (vendored) — see `__SOURCE.md`.
"""
from __future__ import annotations

from typing import Optional

import torch

from . import _wigner_logspace as _logspace  # noqa: F401 (re-exported)
from . import _wigner_xnum as _xnum  # noqa: F401 (re-exported)


# Threshold above which extended-precision is required for numerical stability.
# Below: log-space is faster and accurate.
_XNUM_THRESHOLD_J = 32


def wigner_d(
    j: int | torch.Tensor,
    beta: torch.Tensor,
    method: str = "auto",
) -> torch.Tensor:
    """Compute the (2j+1)x(2j+1) Wigner-d matrix at angle beta.

    Parameters
    ----------
    j : int or 0-d tensor
        Total angular momentum quantum number (degree). Must be non-negative.
    beta : torch.Tensor
        Angle(s) in radians, any shape.
    method : {"auto", "logspace", "xnum"}
        Numerical strategy. "auto" picks based on j.

    Returns
    -------
    torch.Tensor
        Wigner-d values; output shape is `(*beta.shape, 2j+1, 2j+1)`.
    """
    if isinstance(j, torch.Tensor):
        j_int = int(j.item())
    else:
        j_int = int(j)

    if method == "auto":
        method = "logspace" if j_int <= _XNUM_THRESHOLD_J else "xnum"

    if method == "logspace":
        # The log-space module exposes wigner-d via its half-pi table builders.
        # The convention used downstream by `sht_cc` is to call
        # `read_lmn_wigner_d_half_pi_table` for full SHT consumption. For ad-hoc
        # use we expose a minimal wrapper here; production callers go through
        # the SHT/CC pipeline directly.
        return _logspace.read_mn_wigner_d_half_pi_table(j_int, beta)
    elif method == "xnum":
        # xnum returns the full table for all (l, m, n) up to the requested j.
        return _xnum.wigner_d_xnum(j_int, beta)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'auto', 'logspace', or 'xnum'.")


__all__ = ["wigner_d"]
