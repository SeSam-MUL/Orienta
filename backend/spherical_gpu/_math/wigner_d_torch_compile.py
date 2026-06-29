"""torch.compile-compatible port of `wigner_d_lt_half_pi`.

The original `_wigner_logspace.wigner_d_lt_half_pi` is `@torch.jit.script` and
takes `beta: float`, using Python `math.sin(beta/2)`. This makes it impossible
to fuse via `torch.compile` — `.item()` graph-breaks on every call.

This file provides a port:
- `wigner_d_lt_half_pi_compile(beta: Tensor, order_max: int) -> Tensor`
- Takes `beta` as a 0-d (or 1-d batched) Tensor, uses `torch.sin/cos/log`.
- NOT decorated with `@torch.jit.script` — torch.compile-friendly.
- Optionally wrapped in `torch.compile` at module level (lazy).

This is M2.3a-port (multi-day milestone) for GPU spherical-indexing perf.
The intent is to enable Inductor-level fusion of the ~200-400 small tensor
ops in the recurrence into a single Triton kernel per launch.

Reference upstream: `_wigner_logspace.wigner_d_lt_half_pi` (vendored from
ebsdtorch, Zachary Varley, MIT).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from ._wigner_logspace import index_from_coords, signedlogsumexp


def _log_powers_trig_half_beta_torch(
    beta: Tensor, n: int, dtype: torch.dtype, device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """Powers of log(sin(beta/2)) and log(cos(beta/2)) up to n.

    Args:
        beta: 0-d Tensor (single beta) or (...,) Tensor (batched).
        n: number of powers.

    Returns:
        (cos_powers, sin_powers) tensors of shape (..., n).
    """
    powers = torch.arange(0, n, device=device, dtype=dtype)  # (n,)
    log_cos_half = torch.log(torch.cos(beta / 2.0))  # (...,)
    log_sin_half = torch.log(torch.sin(beta / 2.0))  # (...,)
    cos_powers = log_cos_half.unsqueeze(-1) * powers  # (..., n)
    sin_powers = log_sin_half.unsqueeze(-1) * powers
    return cos_powers, sin_powers


def wigner_d_lt_half_pi_eager(
    beta: Tensor,
    order_max: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Single-beta Wigner-d table for beta in [0, pi/2). Eager (uncompiled).

    Args:
        beta: 0-d Tensor, single beta in radians.
        order_max: maximum degree (L-1).
        dtype: float dtype.
        device: torch device.

    Returns:
        (table_size,) tensor of Wigner-d values d^l_{m,n}(beta) for
        l >= m >= n >= 0, packed in the same layout as upstream.
    """
    if device is None:
        device = beta.device

    # initialize the output array
    table_size = (
        ((order_max * (order_max + 1)) // 2) * (order_max + 2) // 3
        + order_max * (order_max + 1) // 2
        + order_max
        + 1
    )
    wigner_d_logmag = torch.empty(table_size, dtype=dtype, device=device)
    wigner_d_sign = torch.ones_like(wigner_d_logmag, dtype=torch.bool, device=device)

    # initialize the powers of sin and cos of beta/2
    powers_of_cos_logscale, powers_of_sin_logscale = _log_powers_trig_half_beta_torch(
        beta, 2 * (order_max + 1), dtype, device
    )

    # tc = 1 - cos(beta) = 2 sin(beta/2)^2
    tc = 2.0 * torch.sin(beta / 2.0) ** 2  # 0-d tensor

    # get all coordinates of the form (m, m, n) for m, n in [0, order_max] and m >= n
    m = torch.arange(order_max + 1, device=device, dtype=torch.int32)
    n = torch.arange(order_max + 1, device=device, dtype=torch.int32)
    mm, nn = torch.meshgrid(m, n, indexing="ij")
    jj = mm
    coords = torch.stack((jj, mm, nn), dim=-1).view(-1, 3)
    mask = coords[:, 1] >= coords[:, 2]
    mmn_coords = coords[mask]
    first_seed_indices = index_from_coords(mmn_coords)

    mmn_coords_fp = mmn_coords.to(dtype)

    # d_m_mn = c_{m+n} s_{m-n} e_mn, where e_mn = sqrt((2m)! / ((m+n)! (m-n)!))
    wigner_d_logmag[first_seed_indices] = (
        0.5 * (
            torch.lgamma(2 * mmn_coords_fp[:, 0] + 1)
            - torch.lgamma(mmn_coords_fp[:, 0] + mmn_coords_fp[:, 2] + 1)
            - torch.lgamma(mmn_coords_fp[:, 0] - mmn_coords_fp[:, 2] + 1)
        )
        + powers_of_cos_logscale[mmn_coords[:, 1] + mmn_coords[:, 2]]
        + powers_of_sin_logscale[mmn_coords[:, 1] - mmn_coords[:, 2]]
    )
    wigner_d_sign[first_seed_indices] = True

    # second seed: d_m+1_mn from d_m_mn
    coords2 = torch.stack((mm, mm - 1, nn - 1), dim=-1).view(-1, 3)
    mask2 = (coords2[:, 1] >= coords2[:, 2]) & (coords2[:, 1] >= 0) & (coords2[:, 2] >= 0)
    mp1_mn_coords = coords2[mask2]
    mp1_mn_coords_fp = mp1_mn_coords.to(dtype)
    second_seed_indices = index_from_coords(mp1_mn_coords)

    first_seed_indices_trunc = first_seed_indices[: len(second_seed_indices)]
    u_mn = (
        2 * mp1_mn_coords_fp[:, 1] - 2 * mp1_mn_coords_fp[:, 2] + 2
    ) - (2 * mp1_mn_coords_fp[:, 1] + 2) * tc
    wigner_d_logmag[second_seed_indices] = (
        wigner_d_logmag[first_seed_indices_trunc]
        + 0.5 * (
            torch.log(2 * mp1_mn_coords_fp[:, 1] + 1)
            - torch.log(2 * (mp1_mn_coords_fp[:, 1] + mp1_mn_coords_fp[:, 2]) + 2)
            - torch.log(2 * (mp1_mn_coords_fp[:, 1] - mp1_mn_coords_fp[:, 2]) + 2)
        )
        + torch.log(torch.abs(u_mn))
    )
    wigner_d_sign[second_seed_indices] = ~torch.logical_xor(
        wigner_d_sign[first_seed_indices_trunc], u_mn >= 0
    )

    # main recursion loop
    curr_coords = mmn_coords[: -(2 * order_max + 1)]
    curr_coords_fp = curr_coords.to(dtype)
    curr_coords[:, 0] += 2
    curr_coords_fp[:, 0] += 2.0
    curr_indices = first_seed_indices[: -(2 * order_max + 1)] + (curr_coords[:, 0]) ** 2

    for step in range(0, order_max - 1):
        w_log = -1.0 * torch.log(2 * curr_coords_fp[:, 0] - 2) - 0.5 * (
            torch.log(2 * curr_coords_fp[:, 0] + 2 * curr_coords_fp[:, 1])
            + torch.log(2 * curr_coords_fp[:, 0] - 2 * curr_coords_fp[:, 1])
            + torch.log(2 * curr_coords_fp[:, 0] + 2 * curr_coords_fp[:, 2])
            + torch.log(2 * curr_coords_fp[:, 0] - 2 * curr_coords_fp[:, 2])
        )
        v_log = torch.log(2 * curr_coords_fp[:, 0]) + 0.5 * (
            torch.log(2 * curr_coords_fp[:, 0] + 2 * curr_coords_fp[:, 1] - 2)
            + torch.log(2 * curr_coords_fp[:, 0] - 2 * curr_coords_fp[:, 1] - 2)
            + torch.log(2 * curr_coords_fp[:, 0] + 2 * curr_coords_fp[:, 2] - 2)
            + torch.log(2 * curr_coords_fp[:, 0] - 2 * curr_coords_fp[:, 2] - 2)
        )
        b_log = w_log + v_log

        u = (
            (2 * curr_coords_fp[:, 0] * (2 * curr_coords_fp[:, 0] - 2))
            - (4 * curr_coords_fp[:, 1] * curr_coords_fp[:, 2])
            - (2 * curr_coords_fp[:, 0] * (2 * curr_coords_fp[:, 0] - 2) * tc)
        )
        u_log = torch.log(torch.abs(u))
        u_sign = u >= 0

        a_log = torch.log(4 * curr_coords_fp[:, 0] - 2) + u_log + w_log

        term1_logmag = (
            a_log
            + wigner_d_logmag[
                curr_indices - ((curr_coords[:, 0]) * (curr_coords[:, 0] + 1) // 2)
            ]
        )
        term1_sign = ~torch.logical_xor(
            wigner_d_sign[
                curr_indices - ((curr_coords[:, 0]) * (curr_coords[:, 0] + 1) // 2)
            ],
            u_sign,
        )

        term2_logmag = (
            b_log
            + wigner_d_logmag[curr_indices - (curr_coords[:, 0] * curr_coords[:, 0])]
        )
        term2_sign = ~wigner_d_sign[
            curr_indices - (curr_coords[:, 0] * curr_coords[:, 0])
        ]

        wigner_d_logmag[curr_indices], wigner_d_sign[curr_indices] = signedlogsumexp(
            term1_logmag, term2_logmag, term1_sign, term2_sign
        )

        curr_coords = curr_coords[: -(order_max - 1 - step), :]
        curr_coords_fp = curr_coords_fp[: -(order_max - 1 - step), :]
        curr_indices = curr_indices[: -(order_max - 1 - step)]
        curr_coords[:, 0] += 1
        curr_coords_fp[:, 0] += 1.0
        curr_indices = curr_indices + (
            (curr_coords[:, 0] * (curr_coords[:, 0] + 1)) // 2
        )

    wigner_d = torch.exp(wigner_d_logmag) * (
        2 * wigner_d_sign.to(wigner_d_logmag.dtype) - 1
    )
    return wigner_d


_COMPILED: Optional[callable] = None


def wigner_d_lt_half_pi_compiled(
    beta: Tensor, order_max: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> Tensor:
    """torch.compile()-wrapped variant. First call compiles."""
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = torch.compile(
            wigner_d_lt_half_pi_eager, mode="default", dynamic=False,
        )
    return _COMPILED(beta, order_max, dtype, device)


# ---------------------------------------------------------------------------
# Batched version — beta is now a (K,) tensor, output is (K, table_size).
# This is the M2.3a-batched key primitive that enables fusion at K=30k scale.
# ---------------------------------------------------------------------------

def _log_powers_trig_half_beta_batched(
    betas: Tensor, n: int, dtype: torch.dtype, device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """Batched powers of log(sin(β/2)) and log(cos(β/2)) up to n.

    Args:
        betas: (K,) tensor of β values in [0, π/2).
        n: number of powers (output tensor's last-dim size).

    Returns:
        (cos_powers, sin_powers) each (K, n) tensors. cos_powers[k, i] is
        i * log(cos(beta_k / 2)).
    """
    powers = torch.arange(0, n, device=device, dtype=dtype)  # (n,)
    log_cos_half = torch.log(torch.cos(betas / 2.0))  # (K,)
    log_sin_half = torch.log(torch.sin(betas / 2.0))  # (K,)
    cos_powers = log_cos_half.unsqueeze(-1) * powers  # (K, n)
    sin_powers = log_sin_half.unsqueeze(-1) * powers
    return cos_powers, sin_powers


def wigner_d_lt_half_pi_batched_eager(
    betas: Tensor,
    order_max: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Batched-over-betas Wigner-d table (eager mode, before compile).

    Args:
        betas: (K,) tensor of β values in [0, π/2).
        order_max: maximum degree (L-1).
        dtype: float dtype.
        device: torch device.

    Returns:
        (K, table_size) tensor of Wigner-d values, packed in the same
        (l ≥ m ≥ n ≥ 0) layout as the single-beta version.
    """
    if device is None:
        device = betas.device
    K = betas.shape[0]

    table_size = (
        ((order_max * (order_max + 1)) // 2) * (order_max + 2) // 3
        + order_max * (order_max + 1) // 2
        + order_max
        + 1
    )
    wigner_d_logmag = torch.empty((K, table_size), dtype=dtype, device=device)
    wigner_d_sign = torch.ones_like(wigner_d_logmag, dtype=torch.bool, device=device)

    powers_of_cos_logscale, powers_of_sin_logscale = _log_powers_trig_half_beta_batched(
        betas, 2 * (order_max + 1), dtype, device
    )  # (K, 2*(order_max+1))

    # tc = 2 sin(β/2)^2  — now (K,)
    tc = 2.0 * torch.sin(betas / 2.0) ** 2  # (K,)

    # Coords (shared across all betas):
    m = torch.arange(order_max + 1, device=device, dtype=torch.int32)
    n = torch.arange(order_max + 1, device=device, dtype=torch.int32)
    mm, nn = torch.meshgrid(m, n, indexing="ij")
    coords = torch.stack((mm, mm, nn), dim=-1).view(-1, 3)
    mask = coords[:, 1] >= coords[:, 2]
    mmn_coords = coords[mask]
    first_seed_indices = index_from_coords(mmn_coords)
    mmn_coords_fp = mmn_coords.to(dtype)

    # First seed: (K, n_first_seed) — broadcast lgamma + powers across K.
    # The lgamma part is K-independent → (n_first_seed,)
    # The powers parts are (K, ...) indexed by m+n and m-n → (K, n_first_seed)
    lgamma_part = 0.5 * (
        torch.lgamma(2 * mmn_coords_fp[:, 0] + 1)
        - torch.lgamma(mmn_coords_fp[:, 0] + mmn_coords_fp[:, 2] + 1)
        - torch.lgamma(mmn_coords_fp[:, 0] - mmn_coords_fp[:, 2] + 1)
    )  # (n_first_seed,)
    cos_part = powers_of_cos_logscale[:, mmn_coords[:, 1] + mmn_coords[:, 2]]  # (K, n_first_seed)
    sin_part = powers_of_sin_logscale[:, mmn_coords[:, 1] - mmn_coords[:, 2]]  # (K, n_first_seed)
    wigner_d_logmag[:, first_seed_indices] = lgamma_part + cos_part + sin_part
    wigner_d_sign[:, first_seed_indices] = True

    # Second seed:
    coords2 = torch.stack((mm, mm - 1, nn - 1), dim=-1).view(-1, 3)
    mask2 = (
        (coords2[:, 1] >= coords2[:, 2])
        & (coords2[:, 1] >= 0)
        & (coords2[:, 2] >= 0)
    )
    mp1_mn_coords = coords2[mask2]
    mp1_mn_coords_fp = mp1_mn_coords.to(dtype)
    second_seed_indices = index_from_coords(mp1_mn_coords)

    first_seed_indices_trunc = first_seed_indices[: len(second_seed_indices)]
    # u_mn now (K, n_second) because tc is (K,)
    coef_a = (
        2 * mp1_mn_coords_fp[:, 1] - 2 * mp1_mn_coords_fp[:, 2] + 2
    )  # (n_second,)
    coef_b = 2 * mp1_mn_coords_fp[:, 1] + 2  # (n_second,)
    u_mn = coef_a.unsqueeze(0) - coef_b.unsqueeze(0) * tc.unsqueeze(-1)  # (K, n_second)

    log_term = 0.5 * (
        torch.log(2 * mp1_mn_coords_fp[:, 1] + 1)
        - torch.log(2 * (mp1_mn_coords_fp[:, 1] + mp1_mn_coords_fp[:, 2]) + 2)
        - torch.log(2 * (mp1_mn_coords_fp[:, 1] - mp1_mn_coords_fp[:, 2]) + 2)
    )  # (n_second,)

    wigner_d_logmag[:, second_seed_indices] = (
        wigner_d_logmag[:, first_seed_indices_trunc]
        + log_term.unsqueeze(0)
        + torch.log(torch.abs(u_mn))
    )
    wigner_d_sign[:, second_seed_indices] = ~torch.logical_xor(
        wigner_d_sign[:, first_seed_indices_trunc], u_mn >= 0
    )

    # Main recursion loop. Indices and (l, m, n)-derived terms are K-independent;
    # only `tc` and the wigner_d_logmag/sign tensors are batched.
    curr_coords = mmn_coords[: -(2 * order_max + 1)]
    curr_coords_fp = curr_coords.to(dtype)
    curr_coords[:, 0] += 2
    curr_coords_fp[:, 0] += 2.0
    curr_indices = first_seed_indices[: -(2 * order_max + 1)] + (curr_coords[:, 0]) ** 2

    for step in range(0, order_max - 1):
        # K-independent terms:
        w_log = -1.0 * torch.log(2 * curr_coords_fp[:, 0] - 2) - 0.5 * (
            torch.log(2 * curr_coords_fp[:, 0] + 2 * curr_coords_fp[:, 1])
            + torch.log(2 * curr_coords_fp[:, 0] - 2 * curr_coords_fp[:, 1])
            + torch.log(2 * curr_coords_fp[:, 0] + 2 * curr_coords_fp[:, 2])
            + torch.log(2 * curr_coords_fp[:, 0] - 2 * curr_coords_fp[:, 2])
        )  # (n_curr,)
        v_log = torch.log(2 * curr_coords_fp[:, 0]) + 0.5 * (
            torch.log(2 * curr_coords_fp[:, 0] + 2 * curr_coords_fp[:, 1] - 2)
            + torch.log(2 * curr_coords_fp[:, 0] - 2 * curr_coords_fp[:, 1] - 2)
            + torch.log(2 * curr_coords_fp[:, 0] + 2 * curr_coords_fp[:, 2] - 2)
            + torch.log(2 * curr_coords_fp[:, 0] - 2 * curr_coords_fp[:, 2] - 2)
        )
        b_log = w_log + v_log

        # u is now (K, n_curr) because tc is (K,)
        const_part = (
            2 * curr_coords_fp[:, 0] * (2 * curr_coords_fp[:, 0] - 2)
        ) - (4 * curr_coords_fp[:, 1] * curr_coords_fp[:, 2])  # (n_curr,)
        tc_coef = (
            2 * curr_coords_fp[:, 0] * (2 * curr_coords_fp[:, 0] - 2)
        )  # (n_curr,)
        u = const_part.unsqueeze(0) - tc_coef.unsqueeze(0) * tc.unsqueeze(-1)  # (K, n_curr)

        u_log = torch.log(torch.abs(u))  # (K, n_curr)
        u_sign = u >= 0  # (K, n_curr)

        a_log = (
            torch.log(4 * curr_coords_fp[:, 0] - 2) + u_log + w_log.unsqueeze(0)
        )  # (K, n_curr)

        # term1 = a * d[l-1, m, n], (K, n_curr)
        prev_idx_a = curr_indices - ((curr_coords[:, 0]) * (curr_coords[:, 0] + 1) // 2)
        term1_logmag = a_log + wigner_d_logmag[:, prev_idx_a]  # (K, n_curr)
        term1_sign = ~torch.logical_xor(wigner_d_sign[:, prev_idx_a], u_sign)

        # term2 = -b * d[l-2, m, n], (K, n_curr)
        prev_idx_b = curr_indices - (curr_coords[:, 0] * curr_coords[:, 0])
        term2_logmag = b_log.unsqueeze(0) + wigner_d_logmag[:, prev_idx_b]
        term2_sign = ~wigner_d_sign[:, prev_idx_b]

        new_logmag, new_sign = signedlogsumexp(
            term1_logmag, term2_logmag, term1_sign, term2_sign
        )
        wigner_d_logmag[:, curr_indices] = new_logmag
        wigner_d_sign[:, curr_indices] = new_sign

        curr_coords = curr_coords[: -(order_max - 1 - step), :]
        curr_coords_fp = curr_coords_fp[: -(order_max - 1 - step), :]
        curr_indices = curr_indices[: -(order_max - 1 - step)]
        curr_coords[:, 0] += 1
        curr_coords_fp[:, 0] += 1.0
        curr_indices = curr_indices + (
            (curr_coords[:, 0] * (curr_coords[:, 0] + 1)) // 2
        )

    wigner_d = torch.exp(wigner_d_logmag) * (
        2 * wigner_d_sign.to(wigner_d_logmag.dtype) - 1
    )
    return wigner_d  # (K, table_size)


def wigner_d_gt_half_pi_batched_eager(
    betas: Tensor,
    order_max: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Batched Wigner-d for β ∈ (π/2, π] (analog of lt_half_pi).

    Internally uses `t = cos(β)` (scalar dependence on β) instead of `tc`.
    Recursion structure identical to the lt version. Output is the same
    packed (l ≥ m ≥ n ≥ 0) layout — for β > π/2 the values still satisfy
    the same packed-symmetry rules.

    Args:
        betas: (K,) tensor of β values in (π/2, π].
        order_max: max degree.

    Returns:
        (K, table_size) Wigner-d values in packed format.
    """
    if device is None:
        device = betas.device
    K = betas.shape[0]

    table_size = (
        ((order_max * (order_max + 1)) // 2) * (order_max + 2) // 3
        + order_max * (order_max + 1) // 2
        + order_max
        + 1
    )
    wigner_d_logmag = torch.empty((K, table_size), dtype=dtype, device=device)
    wigner_d_sign = torch.ones_like(wigner_d_logmag, dtype=torch.bool, device=device)

    powers_of_cos_logscale, powers_of_sin_logscale = _log_powers_trig_half_beta_batched(
        betas, 2 * (order_max + 1), dtype, device
    )

    # t = cos(β) — now (K,)
    t = torch.cos(betas)  # (K,)

    m = torch.arange(order_max + 1, device=device, dtype=torch.int32)
    n = torch.arange(order_max + 1, device=device, dtype=torch.int32)
    mm, nn = torch.meshgrid(m, n, indexing="ij")
    coords = torch.stack((mm, mm, nn), dim=-1).view(-1, 3)
    mask = coords[:, 1] >= coords[:, 2]
    mmn_coords = coords[mask]
    first_seed_indices = index_from_coords(mmn_coords)
    mmn_coords_fp = mmn_coords.to(dtype)

    lgamma_part = 0.5 * (
        torch.lgamma(2 * mmn_coords_fp[:, 0] + 1)
        - torch.lgamma(mmn_coords_fp[:, 0] + mmn_coords_fp[:, 2] + 1)
        - torch.lgamma(mmn_coords_fp[:, 0] - mmn_coords_fp[:, 2] + 1)
    )  # (n_first_seed,)
    cos_part = powers_of_cos_logscale[:, mmn_coords[:, 1] + mmn_coords[:, 2]]
    sin_part = powers_of_sin_logscale[:, mmn_coords[:, 1] - mmn_coords[:, 2]]
    wigner_d_logmag[:, first_seed_indices] = lgamma_part + cos_part + sin_part
    wigner_d_sign[:, first_seed_indices] = True

    # Second seed (gt version uses different u_mn formula):
    coords2 = torch.stack((mm, mm - 1, nn - 1), dim=-1).view(-1, 3)
    mask2 = (
        (coords2[:, 1] >= coords2[:, 2])
        & (coords2[:, 1] >= 0)
        & (coords2[:, 2] >= 0)
    )
    mp1_mn_coords = coords2[mask2]
    mp1_mn_coords_fp = mp1_mn_coords.to(dtype)
    second_seed_indices = index_from_coords(mp1_mn_coords)

    first_seed_indices_trunc = first_seed_indices[: len(second_seed_indices)]
    # u_mn for β in (π/2, π]: (2m + 2)*t − 2n
    coef_a = 2 * mp1_mn_coords_fp[:, 1] + 2  # (n_second,)
    coef_b = 2 * mp1_mn_coords_fp[:, 2]  # (n_second,)
    u_mn = coef_a.unsqueeze(0) * t.unsqueeze(-1) - coef_b.unsqueeze(0)  # (K, n_second)

    log_term = 0.5 * (
        torch.log(2 * mp1_mn_coords_fp[:, 1] + 1)
        - torch.log(2 * (mp1_mn_coords_fp[:, 1] + mp1_mn_coords_fp[:, 2]) + 2)
        - torch.log(2 * (mp1_mn_coords_fp[:, 1] - mp1_mn_coords_fp[:, 2]) + 2)
    )
    wigner_d_logmag[:, second_seed_indices] = (
        wigner_d_logmag[:, first_seed_indices_trunc]
        + log_term.unsqueeze(0)
        + torch.log(torch.abs(u_mn))
    )
    wigner_d_sign[:, second_seed_indices] = ~torch.logical_xor(
        wigner_d_sign[:, first_seed_indices_trunc], u_mn >= 0
    )

    # Main recursion (gt version uses different u formula):
    curr_coords = mmn_coords[: -(2 * order_max + 1)]
    curr_coords_fp = curr_coords.to(dtype)
    curr_coords[:, 0] += 2
    curr_coords_fp[:, 0] += 2.0
    curr_indices = first_seed_indices[: -(2 * order_max + 1)] + (curr_coords[:, 0]) ** 2

    for step in range(0, order_max - 1):
        w_log = -1.0 * torch.log(2 * curr_coords_fp[:, 0] - 2) - 0.5 * (
            torch.log(2 * curr_coords_fp[:, 0] + 2 * curr_coords_fp[:, 1])
            + torch.log(2 * curr_coords_fp[:, 0] - 2 * curr_coords_fp[:, 1])
            + torch.log(2 * curr_coords_fp[:, 0] + 2 * curr_coords_fp[:, 2])
            + torch.log(2 * curr_coords_fp[:, 0] - 2 * curr_coords_fp[:, 2])
        )
        v_log = torch.log(2 * curr_coords_fp[:, 0]) + 0.5 * (
            torch.log(2 * curr_coords_fp[:, 0] + 2 * curr_coords_fp[:, 1] - 2)
            + torch.log(2 * curr_coords_fp[:, 0] - 2 * curr_coords_fp[:, 1] - 2)
            + torch.log(2 * curr_coords_fp[:, 0] + 2 * curr_coords_fp[:, 2] - 2)
            + torch.log(2 * curr_coords_fp[:, 0] - 2 * curr_coords_fp[:, 2] - 2)
        )
        b_log = w_log + v_log

        # u for gt: 2l(2l-2)t - (2m)(2n) = 2l(2l-2)*t - 4mn
        const_part = -(4 * curr_coords_fp[:, 1] * curr_coords_fp[:, 2])  # (n_curr,)
        t_coef = 2 * curr_coords_fp[:, 0] * (2 * curr_coords_fp[:, 0] - 2)  # (n_curr,)
        u = t_coef.unsqueeze(0) * t.unsqueeze(-1) + const_part.unsqueeze(0)  # (K, n_curr)

        u_log = torch.log(torch.abs(u))
        u_sign = u >= 0

        a_log = (
            torch.log(4 * curr_coords_fp[:, 0] - 2) + u_log + w_log.unsqueeze(0)
        )

        prev_idx_a = curr_indices - ((curr_coords[:, 0]) * (curr_coords[:, 0] + 1) // 2)
        term1_logmag = a_log + wigner_d_logmag[:, prev_idx_a]
        term1_sign = ~torch.logical_xor(wigner_d_sign[:, prev_idx_a], u_sign)

        prev_idx_b = curr_indices - (curr_coords[:, 0] * curr_coords[:, 0])
        term2_logmag = b_log.unsqueeze(0) + wigner_d_logmag[:, prev_idx_b]
        term2_sign = ~wigner_d_sign[:, prev_idx_b]

        new_logmag, new_sign = signedlogsumexp(
            term1_logmag, term2_logmag, term1_sign, term2_sign
        )
        wigner_d_logmag[:, curr_indices] = new_logmag
        wigner_d_sign[:, curr_indices] = new_sign

        curr_coords = curr_coords[: -(order_max - 1 - step), :]
        curr_coords_fp = curr_coords_fp[: -(order_max - 1 - step), :]
        curr_indices = curr_indices[: -(order_max - 1 - step)]
        curr_coords[:, 0] += 1
        curr_coords_fp[:, 0] += 1.0
        curr_indices = curr_indices + (
            (curr_coords[:, 0] * (curr_coords[:, 0] + 1)) // 2
        )

    wigner_d = torch.exp(wigner_d_logmag) * (
        2 * wigner_d_sign.to(wigner_d_logmag.dtype) - 1
    )
    return wigner_d


_COMPILED_BATCHED: Optional[callable] = None


def wigner_d_lt_half_pi_batched_compiled(
    betas: Tensor, order_max: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> Tensor:
    """torch.compile()-wrapped batched variant. First call compiles."""
    global _COMPILED_BATCHED
    if _COMPILED_BATCHED is None:
        _COMPILED_BATCHED = torch.compile(
            wigner_d_lt_half_pi_batched_eager, mode="default", dynamic=False,
        )
    return _COMPILED_BATCHED(betas, order_max, dtype, device)


__all__ = [
    "wigner_d_lt_half_pi_eager",
    "wigner_d_lt_half_pi_compiled",
    "wigner_d_lt_half_pi_batched_eager",
    "wigner_d_lt_half_pi_batched_compiled",
]
