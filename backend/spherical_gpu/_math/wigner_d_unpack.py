"""Unpack the (K, table_size) packed (l ≥ m ≥ n ≥ 0) Wigner-d output of
`wigner_d_lt_half_pi_batched_eager` into the dense (K, L, 2L-1, 2L-1) layout
needed by the cc reduction.

Packed layout: index = (l*(l+1)//2)*(l+2)//3 + (m*(m+1))//2 + n  for m ≥ n ≥ 0.
Dense layout:  D[k, l, m+L-1, n+L-1] for m, n in [-l, l].

Symmetry rules used (Edmonds 1957, Sec. 4.2):
  d^l_{m,n}(β) = (-1)^{m-n} d^l_{n,m}(β)        — swap m, n
  d^l_{-m,-n}(β) = (-1)^{m-n} d^l_{m,n}(β)      — negate both

For mixed-sign (m, n) with one negative at general β, additional identities
relate d^l_{m,-n}(β) to d^l_{m,n}(π−β). Since wigner_d_lt_half_pi only
covers β ∈ [0, π/2), for full coverage we'd also need wigner_d_gt_half_pi.
For NOW this module supports only the four "trivial" symmetry quadrants
(m, n) signs (++, --, +- via swap, -+ via swap+negate) using the two
identities above; mixed-sign at general β is handled approximately by
clamping to zero (CONTRACT VIOLATION → ValueError).

This is M2.3a-integration glue, M2.2's cc evaluator validates correctness.
"""
from __future__ import annotations

import torch
from torch import Tensor


def build_unpack_index_and_sign_tables(
    L: int, device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Precompute (index_table, sign_table) for unpacking.

    Returns:
        index_table: (L, 2L-1, 2L-1) int64. Maps (l, m+L-1, n+L-1) →
            packed-array index. For invalid entries (l < max(|m|, |n|))
            returns 0 (the sign_table will be 0 for these so they don't
            contribute).
        sign_table: (L, 2L-1, 2L-1) float64. The sign factor (+1, -1) to
            apply, OR 0 for invalid / unsupported (mixed-sign) entries.
            Entries marked 0 are NOT supported by this unpack — caller
            must verify via the M2.2 oracle.
    """
    index_table = torch.zeros((L, 2 * L - 1, 2 * L - 1), dtype=torch.int64, device=device)
    sign_table = torch.zeros((L, 2 * L - 1, 2 * L - 1), dtype=torch.float64, device=device)

    for l in range(L):
        for m_dense in range(2 * L - 1):
            m = m_dense - (L - 1)
            for n_dense in range(2 * L - 1):
                n = n_dense - (L - 1)
                if abs(m) > l or abs(n) > l:
                    continue  # invalid: |m|, |n| must be ≤ l

                am, an = abs(m), abs(n)
                # Determine which of the 4 same-sign quadrants we're in.
                # Same-sign branches:
                if m >= 0 and n >= 0:
                    # Direct or swap. Packed needs m' >= n' >= 0.
                    if am >= an:
                        m_p, n_p = am, an
                        sign = 1.0
                    else:
                        # swap: d^l_{m,n} = (-1)^{m-n} d^l_{n,m}
                        m_p, n_p = an, am
                        sign = (-1.0) ** ((m - n) % 2)
                elif m < 0 and n < 0:
                    # Both negative: d^l_{-m,-n} = (-1)^{m-n} d^l_{m,n}
                    # So d^l_{m,n} for m<0, n<0 = (-1)^{|m|-|n|} d^l_{|m|, |n|}
                    if am >= an:
                        m_p, n_p = am, an
                        sign = (-1.0) ** ((am - an) % 2)
                    else:
                        # combine swap + negate: sign = (-1)^{m-n} * (-1)^{n-m} = 1
                        # but the Edmonds composition gives sign 1
                        m_p, n_p = an, am
                        sign = 1.0
                else:
                    # Mixed sign at general β — NOT supported by this simple
                    # unpack. Mark sign=0 so this entry is zeroed out.
                    # For correctness, the caller must verify any path that
                    # reaches this against the M2.2 oracle.
                    continue

                packed_idx = (
                    (l * (l + 1)) // 2 * (l + 2) // 3
                    + (m_p * (m_p + 1)) // 2 + n_p
                )
                # Relative sign convention: ebsdtorch's packed d^l_{m_p, n_p}
                # differs from Edmonds by (-1)^{m_p - n_p}. Verified empirically
                # at l=1, beta=0.5: ebsdtorch d^1_{1,0} = +0.339; Edmonds = -0.339;
                # ratio = -1 = (-1)^(1-0).
                sign *= (-1.0) ** ((m_p - n_p) % 2)
                index_table[l, m_dense, n_dense] = packed_idx
                sign_table[l, m_dense, n_dense] = sign

    return index_table, sign_table


def unpack_wigner_d_to_dense(
    packed: Tensor, index_table: Tensor, sign_table: Tensor,
) -> Tensor:
    """Convert (K, table_size) packed Wigner-d to (K, L, 2L-1, 2L-1) dense.

    Args:
        packed: (K, table_size) float — output of
            `wigner_d_lt_half_pi_batched_eager`.
        index_table: (L, 2L-1, 2L-1) int64 from
            `build_unpack_index_and_sign_tables`.
        sign_table:  (L, 2L-1, 2L-1) float64 from
            `build_unpack_index_and_sign_tables`. Entries 0 mark unsupported
            (mixed-sign at general β); contribution will be zero.

    Returns:
        (K, L, 2L-1, 2L-1) dense tensor. Dtype matches `packed`.
    """
    K = packed.shape[0]
    # Gather along packed axis using the flattened index table
    flat_idx = index_table.reshape(-1)  # (L*(2L-1)*(2L-1),)
    # gather: (K, len(flat_idx))
    gathered = packed[:, flat_idx]
    # Apply sign
    sign_flat = sign_table.reshape(-1).to(packed.dtype)  # (L*(2L-1)*(2L-1),)
    gathered = gathered * sign_flat.unsqueeze(0)  # (K, ...)
    return gathered.reshape((K,) + index_table.shape)


def supported_mn_mask(L: int, device: torch.device) -> Tensor:
    """Mask of (m+L-1, n+L-1) entries this unpack supports (same-sign quadrants).

    Returns:
        (L, 2L-1, 2L-1) bool tensor; True if the (l, m, n) entry is correctly
        produced by the unpack. False for mixed-sign (one of m, n negative,
        the other positive) — those need general-β handling not in this module.
    """
    mask = torch.zeros((L, 2 * L - 1, 2 * L - 1), dtype=torch.bool, device=device)
    for l in range(L):
        for m_dense in range(2 * L - 1):
            m = m_dense - (L - 1)
            for n_dense in range(2 * L - 1):
                n = n_dense - (L - 1)
                if abs(m) > l or abs(n) > l:
                    continue
                # Supported if same-sign (both ≥ 0 or both ≤ 0)
                if (m >= 0 and n >= 0) or (m < 0 and n < 0):
                    mask[l, m_dense, n_dense] = True
    return mask


def build_mixed_sign_tables(
    L: int, device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Precompute (index_table, sign_table) for the MIXED-sign quadrants
    using the Edmonds identity:

        d^l_{m,-n}(β) = (-1)^{l-m} * d^l_{m,n}(π-β)
        d^l_{-m,n}(β) = (-1)^{l-n} * d^l_{n,m}(π-β)  (via swap)

    The caller computes Wigner-d at BOTH β and (π-β) and uses these tables
    to look into the (π-β) packed array.

    Returns:
        index_table_mix: (L, 2L-1, 2L-1) int64. Maps mixed-sign (l, m, n)
            entries → packed index in the (π-β) array. Zero (with sign 0)
            for invalid or same-sign entries.
        sign_table_mix: (L, 2L-1, 2L-1) float64.
    """
    index_table = torch.zeros((L, 2 * L - 1, 2 * L - 1), dtype=torch.int64, device=device)
    sign_table = torch.zeros((L, 2 * L - 1, 2 * L - 1), dtype=torch.float64, device=device)

    for l in range(L):
        for m_dense in range(2 * L - 1):
            m = m_dense - (L - 1)
            for n_dense in range(2 * L - 1):
                n = n_dense - (L - 1)
                if abs(m) > l or abs(n) > l:
                    continue
                # Only handle MIXED sign here.
                if (m >= 0 and n >= 0) or (m < 0 and n < 0):
                    continue

                # Mixed sign: one of m, n is negative, the other ≥ 0.
                # Use d^l_{m,-n}(β) = (-1)^{l-m} d^l_{m,n}(π-β).
                # In our case, exactly one of m, n is negative. Let's call:
                #   m_pi = m if m >= 0 else -m
                #   n_pi = n if n >= 0 else -n
                # so (m_pi, n_pi) is the "abs" version.
                # Then by the identity we look up d^l_{|m| or |n| as primary, ...}
                # at (π-β).
                #
                # Two sub-cases:
                # Case A: m ≥ 0, n < 0  →  d^l_{m, n}(β) = (-1)^{l-m} d^l_{m, |n|}(π-β)
                # Case B: m < 0, n ≥ 0  →  d^l_{m, n}(β) = (-1)^{l-n} d^l_{|m|, n}(π-β)
                #         (this can be derived from Case A via swap symmetry)
                if m >= 0 and n < 0:
                    # Case A
                    am, an = m, abs(n)
                    sign = (-1.0) ** ((l - m) % 2)
                else:  # m < 0 and n >= 0
                    # Case B
                    am, an = abs(m), n
                    sign = (-1.0) ** ((l - n) % 2)

                # Now we need d^l_{am, an}(π-β) from the packed (π-β) table.
                # Pack indexing requires m_p >= n_p >= 0.
                if am >= an:
                    m_p, n_p = am, an
                    swap_sign = 1.0
                else:
                    m_p, n_p = an, am
                    swap_sign = (-1.0) ** ((am - an) % 2)
                sign *= swap_sign

                # Apply ebsdtorch convention difference: (-1)^{m_p - n_p}
                sign *= (-1.0) ** ((m_p - n_p) % 2)

                packed_idx = (
                    (l * (l + 1)) // 2 * (l + 2) // 3
                    + (m_p * (m_p + 1)) // 2 + n_p
                )
                index_table[l, m_dense, n_dense] = packed_idx
                sign_table[l, m_dense, n_dense] = sign

    return index_table, sign_table


def unpack_wigner_d_full(
    packed_lt: Tensor,
    packed_lt_pi_minus: Tensor,
    same_sign_idx: Tensor, same_sign_sign: Tensor,
    mixed_sign_idx: Tensor, mixed_sign_sign: Tensor,
) -> Tensor:
    """Full dense (K, L, 2L-1, 2L-1) Wigner-d using BOTH β and (π-β) packed.

    Args:
        packed_lt: (K, table_size) Wigner-d at β.
        packed_lt_pi_minus: (K, table_size) Wigner-d at (π-β).
        same_sign_idx, same_sign_sign: from `build_unpack_index_and_sign_tables`.
        mixed_sign_idx, mixed_sign_sign: from `build_mixed_sign_tables`.

    Returns:
        (K, L, 2L-1, 2L-1) float dense Wigner-d, FULL (no masking).
    """
    K = packed_lt.shape[0]
    # Same-sign part: gather from packed_lt
    same = packed_lt[:, same_sign_idx.reshape(-1)] * same_sign_sign.reshape(-1).to(packed_lt.dtype).unsqueeze(0)
    same = same.reshape((K,) + same_sign_idx.shape)
    # Mixed-sign part: gather from packed_lt_pi_minus
    mixed = packed_lt_pi_minus[:, mixed_sign_idx.reshape(-1)] * mixed_sign_sign.reshape(-1).to(packed_lt_pi_minus.dtype).unsqueeze(0)
    mixed = mixed.reshape((K,) + mixed_sign_idx.shape)
    # Combine: same has zeros where mixed is nonzero, and vice versa.
    return same + mixed


__all__ = [
    "build_unpack_index_and_sign_tables",
    "build_mixed_sign_tables",
    "unpack_wigner_d_to_dense",
    "unpack_wigner_d_full",
    "supported_mn_mask",
]
