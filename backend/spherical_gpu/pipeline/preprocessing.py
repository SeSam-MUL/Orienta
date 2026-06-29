"""GPU-batched pattern preprocessing for spherical indexing.

Pipeline order (matches EMSphInx ``include/modality/ebsd/imprc.hpp``):

1. ``circmask`` — zero pixels outside an inscribed circle (cuts phosphor edges).
2. ``gausbckg`` — subtract a fitted 2-D Gaussian background.
3. ``nregions`` — adaptive histogram equalization on N×N tiles (boost local contrast).

Implementation note: ``nregions`` here uses a per-tile rank transform, which
approximates the histogram equalization in EMSphInx's ``imprc.hpp:histeq``.
The approximation is exact for arbitrary monotone histogram targets and
matches CDF-based AHE up to interpolation at tile boundaries. If the per-pixel
correlation in ``test_04`` shows a systematic bias on patterns with strong
illumination gradients, a research-agent should re-extract the EMSphInx
formula and we patch the implementation.
"""
from __future__ import annotations

import torch


def circmask(patterns: torch.Tensor) -> torch.Tensor:
    """Zero the pixels outside an inscribed circle.

    Parameters
    ----------
    patterns : (B, H, W) or (H, W) float tensor

    Returns
    -------
    Same shape as input, with corners zeroed.
    """
    if patterns.dim() == 2:
        patterns = patterns.unsqueeze(0)
    _, H, W = patterns.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=patterns.device, dtype=patterns.dtype),
        torch.arange(W, device=patterns.device, dtype=patterns.dtype),
        indexing="ij",
    )
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    radius = min(H, W) / 2.0
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2
    return patterns * mask.to(patterns.dtype)


def gausbckg(patterns: torch.Tensor) -> torch.Tensor:
    """Subtract a fitted 2-D Gaussian background from each pattern.

    EMSphInx fits a Gaussian to the smoothed pattern and subtracts it. We
    approximate the same effect with a low-pass: a separable Gaussian blur
    with sigma proportional to the pattern size, then subtract from the
    original.

    Parameters
    ----------
    patterns : (B, H, W) float tensor

    Returns
    -------
    Same shape, mean-subtracted.
    """
    if patterns.dim() == 2:
        patterns = patterns.unsqueeze(0)
    B, H, W = patterns.shape

    # EMSphInx default: sigma proportional to pattern size, ~min(H,W)/4
    sigma = max(min(H, W) / 4.0, 1.0)
    half = max(int(3 * sigma), 1)
    kernel_size = 2 * half + 1
    if kernel_size > min(H, W):
        # Pattern too small for the default kernel — shrink to fit
        half = max((min(H, W) - 1) // 2, 1)
        kernel_size = 2 * half + 1

    coords = torch.arange(-half, half + 1, dtype=patterns.dtype, device=patterns.device)
    kernel_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    pat = patterns.unsqueeze(1)  # (B, 1, H, W)
    kx = kernel_1d.view(1, 1, 1, -1)
    ky = kernel_1d.view(1, 1, -1, 1)
    blurred = torch.nn.functional.conv2d(pat, kx, padding=(0, half))
    blurred = torch.nn.functional.conv2d(blurred, ky, padding=(half, 0))
    blurred = blurred.squeeze(1)

    return patterns - blurred


def nregions(patterns: torch.Tensor, n: int = 10) -> torch.Tensor:
    """Adaptive histogram equalization on ``n × n`` tiles via per-tile rank.

    Parameters
    ----------
    patterns : (B, H, W) float tensor
    n : int
        Tiles per axis. ``n=10`` matches EMSphInx default.

    Returns
    -------
    Same shape, output values in [0, 1] approximating a uniform distribution
    within each tile.

    iter-15H: vectorized — fold all n*n tiles into a single batch dimension
    so that the rank transform is one argsort call instead of n^2.
    Falls back to the original tile-by-tile loop only when H or W is not
    evenly divisible by n (rare in practice).
    """
    if patterns.dim() == 2:
        patterns = patterns.unsqueeze(0)
    B, H, W = patterns.shape

    if n > 0 and H % n == 0 and W % n == 0:
        th = H // n
        tw = W // n
        # (B, H, W) -> (B, n, th, n, tw) -> (B, n, n, th, tw)
        tiles = (
            patterns.reshape(B, n, th, n, tw)
            .permute(0, 1, 3, 2, 4)              # (B, n_y, n_x, th, tw)
            .contiguous()
            .reshape(B * n * n, th * tw)         # (B*n*n, th*tw)
        )
        denom = max(th * tw - 1, 1)
        ranks = torch.argsort(torch.argsort(tiles, dim=1), dim=1).to(patterns.dtype) / denom
        # Reshape back: (B*n*n, th*tw) -> (B, n_y, n_x, th, tw) -> (B, H, W)
        return (
            ranks.reshape(B, n, n, th, tw)
            .permute(0, 1, 3, 2, 4)
            .reshape(B, H, W)
        )

    # Non-divisible pattern dims: V2 group-uniform path. Same tile layout
    # as the original 100-iteration loop (V0):  9x9 interior + 9 right col +
    # 9 bottom row + 1 corner. Each of the 4 groups has uniform tile size
    # within the group, so it runs as a single batched argsort. Result is
    # bit-identical to the loop fallback (verified via
    # tasks/bench_nregions_variants.py: max_abs_diff = 0.000e+00) while
    # collapsing 100 sort kernel launches into 4.
    # Bench (RTX 4070, B=32, 128x156, n=10): V0 loop 29.3 ms -> V2 1.33 ms = 22x.
    th = max(H // n, 1)
    tw = max(W // n, 1)
    H_last = H - (n - 1) * th
    W_last = W - (n - 1) * tw
    out = torch.empty_like(patterns)

    def _rank_norm(t: torch.Tensor) -> torch.Tensor:
        N = t.shape[-1]
        denom = max(N - 1, 1)
        return torch.argsort(torch.argsort(t, dim=-1), dim=-1).to(t.dtype) / denom

    # G0: interior (n-1) x (n-1) tiles of (th × tw)
    if (n - 1) * th > 0 and (n - 1) * tw > 0:
        interior = patterns[:, : (n - 1) * th, : (n - 1) * tw]
        tiles_g0 = (
            interior.reshape(B, n - 1, th, n - 1, tw)
            .permute(0, 1, 3, 2, 4).contiguous()
            .reshape(B * (n - 1) * (n - 1), th * tw)
        )
        ranks_g0 = _rank_norm(tiles_g0).reshape(B, n - 1, n - 1, th, tw).permute(0, 1, 3, 2, 4)
        out[:, : (n - 1) * th, : (n - 1) * tw] = ranks_g0.reshape(B, (n - 1) * th, (n - 1) * tw)

    # G1: right column (n-1) x 1 tiles of (th × W_last)
    if (n - 1) * th > 0 and W_last > 0:
        right_col = patterns[:, : (n - 1) * th, (n - 1) * tw:]
        tiles_g1 = (
            right_col.reshape(B, n - 1, th, W_last)
            .reshape(B * (n - 1), th * W_last)
        )
        ranks_g1 = _rank_norm(tiles_g1).reshape(B, n - 1, th, W_last)
        out[:, : (n - 1) * th, (n - 1) * tw:] = ranks_g1.reshape(B, (n - 1) * th, W_last)

    # G2: bottom row 1 x (n-1) tiles of (H_last × tw)
    if H_last > 0 and (n - 1) * tw > 0:
        bot_row = patterns[:, (n - 1) * th:, : (n - 1) * tw]
        tiles_g2 = (
            bot_row.reshape(B, H_last, n - 1, tw)
            .permute(0, 2, 1, 3).contiguous()
            .reshape(B * (n - 1), H_last * tw)
        )
        ranks_g2 = _rank_norm(tiles_g2).reshape(B, n - 1, H_last, tw).permute(0, 2, 1, 3)
        out[:, (n - 1) * th:, : (n - 1) * tw] = ranks_g2.reshape(B, H_last, (n - 1) * tw)

    # G3: bottom-right corner (H_last × W_last)
    if H_last > 0 and W_last > 0:
        corner = patterns[:, (n - 1) * th:, (n - 1) * tw:]
        tile_g3 = corner.reshape(B, H_last * W_last)
        ranks_g3 = _rank_norm(tile_g3).reshape(B, H_last, W_last)
        out[:, (n - 1) * th:, (n - 1) * tw:] = ranks_g3
    return out


def preprocess_batch(
    patterns: torch.Tensor,
    circmask_on: bool = True,
    gausbckg_on: bool = True,
    nregions_n: int = 10,
) -> torch.Tensor:
    """Compose the three preprocessing stages in EMSphInx order.

    Parameters
    ----------
    patterns : (B, H, W)
    circmask_on : bool — apply circular mask
    gausbckg_on : bool — subtract Gaussian background
    nregions_n : int — number of tiles per axis for AHE; pass 0 to skip

    Returns
    -------
    Preprocessed patterns, same shape as input.
    """
    out = patterns
    if circmask_on:
        out = circmask(out)
    if gausbckg_on:
        out = gausbckg(out)
    if nregions_n > 0:
        out = nregions(out, n=nregions_n)
    return out
