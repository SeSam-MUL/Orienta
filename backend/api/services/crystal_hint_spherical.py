"""Method B — spherical-projection symmetry detection for EBSD patterns.

Image-space rotation NCC (Method A) underperforms on raw EBSD patterns
because the pattern center is NOT at the image centre and the detector is
tilted, so 3D crystal symmetry doesn't appear as 2D image symmetry around
any fixed point. Method B fixes this by projecting the detector intensity
onto the unit sphere (each pixel → direction), then rotating the sphere
intensity field around candidate 3D axes.

Geometric convention (consistent with `backend/spherical_gpu/pipeline/detector.py`):

- Sample at origin (0, 0, 0)
- Detector face at z = L, with face normal pointing back toward sample
- Pattern center is the perpendicular projection of sample onto detector:
  PC at (0, 0, L) on the detector face
- For pixel (i_col, j_row) on a W × H detector (Bruker PC convention):
    x_p = (i_col - PC_x · W) · pixel_size
    y_p = -(j_row - PC_y · H) · pixel_size      # flip so "up" is +y
    z_p = L  (= PC_z · W · pixel_size)
- Direction from sample to pixel: (x_p, y_p, z_p), normalised
- The sample tilt + detector tilt only RELABEL the coordinate axes; for
  RELATIVE symmetry detection we sweep over candidate axes anyway, so we
  work in the detector frame directly (saves a coordinate rotation that
  would not change the result).

The NCC compares the original pattern with a "rotated" pattern where each
pixel's intensity is sampled from the location that, under the rotation,
maps to that pixel. Mathematically: I_rot(p) = I(R^{-1} · n(p)), where
n(p) is the 3D direction of pixel p and R is the candidate symmetry rotation.

See also: `backend/api/services/crystal_hint_symmetry.py` for Method A
(image-space rotation, fast but inaccurate on raw patterns).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from backend.api.services.crystal_hint_symmetry import (
    CANDIDATE_FOLDS,
    SymmetryReport,
    _preprocess_pattern,
    compatible_systems_for_fold,
)


# Confidence thresholds — calibrated against REAL EBSD patterns
# (HiGainNi cubic Ni, 60x60 px, off-center PC at PCy=0.26):
#   Method A median NCC ≈ 0.01,  best ≈ 0.13
#   Method B median NCC ≈ 0.12,  best ≈ 0.26
# Method B gives a ~10-15x improvement on real off-axis pixels but still
# rarely clears 0.5 on small/tilted detectors. Synthetic clean patterns
# (test suite) saturate at 0.85+. Thresholds split the empirical range:
#   HIGH    >= 0.55 — clean zone-axis pattern, both fold and axis trustworthy
#   MEDIUM  >= 0.30 — meaningful detection, axis broadly correct
#   LOW     >= 0.15 — weak hint; pattern noisy or off-zone-axis
#   none    <  0.15 — inconclusive
CONF_HIGH = 0.55
CONF_MEDIUM = 0.30
CONF_LOW = 0.15


def fibonacci_sphere(n: int) -> np.ndarray:
    """Return n approximately-uniform unit vectors on the upper sphere.

    Uses the Fibonacci spiral — gives near-uniform area coverage with
    no clustering at the poles. Returns shape (n, 3).

    We restrict to z >= 0 because the detector only observes the upper
    hemisphere — directions with z < 0 would point away from the detector.
    """
    # Generate 2n full-sphere points, keep upper hemisphere
    indices = np.arange(2 * n) + 0.5
    phi = np.arccos(1 - 2 * indices / (2 * n))   # polar angle [0, π]
    theta = np.pi * (1 + 5 ** 0.5) * indices     # azimuth, golden angle
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    xyz = np.stack([x, y, z], axis=1)
    upper = xyz[xyz[:, 2] >= 0.0]
    if upper.shape[0] < n:
        # Pad with reflected lower points if needed (rare for n >= ~32)
        lower = xyz[xyz[:, 2] < 0.0]
        lower[:, 2] = -lower[:, 2]
        upper = np.concatenate([upper, lower], axis=0)
    return upper[:n]


def candidate_axes(n_fibonacci: int = 24) -> np.ndarray:
    """Generate candidate rotation axes for the symmetry sweep.

    Mixes:
      - Fibonacci-spaced upper-hemisphere directions (broad coverage)
      - Detector-frame poles: +z (detector normal), ±x, ±y, and the
        cubic high-symmetry diagonals [1,1,1], [1,0,1], [1,1,0].
    Returns shape (n, 3), unit-norm.
    """
    fibo = fibonacci_sphere(n_fibonacci)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    inv_sqrt3 = 1.0 / np.sqrt(3.0)
    explicit = np.array([
        [0.0, 0.0, 1.0],                            # detector normal
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [inv_sqrt2, 0.0, inv_sqrt2],
        [0.0, inv_sqrt2, inv_sqrt2],
        [inv_sqrt2, inv_sqrt2, 0.0],
        [inv_sqrt3, inv_sqrt3, inv_sqrt3],
        [-inv_sqrt3, inv_sqrt3, inv_sqrt3],
        [inv_sqrt3, -inv_sqrt3, inv_sqrt3],
    ], dtype=np.float64)
    return np.concatenate([explicit, fibo], axis=0)


def _rodrigues_rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Build a 3x3 rotation matrix for rotating by `angle_rad` around `axis`.

    `axis` must be a unit vector. Uses Rodrigues' formula:
        R = I + sin(θ)·K + (1-cos(θ))·K²
    where K is the skew-symmetric cross-product matrix.
    """
    a = axis / np.linalg.norm(axis)
    K = np.array([
        [0.0,    -a[2],  a[1]],
        [a[2],    0.0,  -a[0]],
        [-a[1],   a[0],  0.0],
    ], dtype=np.float64)
    return np.eye(3) + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * (K @ K)


def pixel_grid_directions(pat_h: int, pat_w: int, pc: tuple[float, float, float]) -> np.ndarray:
    """Return a (pat_h, pat_w, 3) array of unit vectors — each entry is the
    direction the corresponding detector pixel observes.

    PC convention: Bruker (PC_x, PC_y in [0,1], PC_z as detector-distance/width).
    Working pixel_size = 1 (only relative geometry matters for symmetry).
    """
    PCx, PCy, PCz = pc
    L = PCz * pat_w  # detector distance in pixel units
    i_grid, j_grid = np.meshgrid(np.arange(pat_w), np.arange(pat_h), indexing="xy")
    # i_grid: column index 0..W-1 (broadcasts along rows)
    # j_grid: row index 0..H-1 (broadcasts along cols)
    x_p = (i_grid - PCx * pat_w).astype(np.float64)
    y_p = -(j_grid - PCy * pat_h).astype(np.float64)
    z_p = np.full_like(x_p, L, dtype=np.float64)
    vecs = np.stack([x_p, y_p, z_p], axis=-1)        # (H, W, 3)
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return vecs / norms


def _direction_to_pixel(
    directions: np.ndarray,   # (..., 3)
    pat_h: int, pat_w: int,
    pc: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse projection: given unit-vector directions, find the detector
    pixel coords (i_col, j_row) that observe each direction. Returns
    (i, j) arrays of float; pixels outside the detector or with z<=0 get
    coordinates outside [0, W-1] × [0, H-1] (caller filters via bilinear-
    sample's nan-padding).
    """
    PCx, PCy, PCz = pc
    L = PCz * pat_w
    z = directions[..., 2]
    # Avoid division by zero: directions with z<=0 don't hit the detector
    safe_z = np.where(np.abs(z) < 1e-12, 1e-12, z)
    t = L / safe_z
    x_p = directions[..., 0] * t
    y_p = directions[..., 1] * t
    i_col = x_p + PCx * pat_w
    j_row = -y_p + PCy * pat_h
    # Mark backward-facing directions (would land "behind" the detector)
    # by pushing pixel coords way out of range:
    bad = z <= 0
    if np.any(bad):
        i_col = np.where(bad, -1e6, i_col)
        j_row = np.where(bad, -1e6, j_row)
    return i_col, j_row


def _bilinear_sample(
    image: np.ndarray,
    i_col: np.ndarray,
    j_row: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation on a 2-D image. Returns NaN for out-of-range
    coordinates."""
    H, W = image.shape
    out = np.full_like(i_col, np.nan, dtype=np.float64)
    valid = (i_col >= 0) & (i_col <= W - 1) & (j_row >= 0) & (j_row <= H - 1)
    if not np.any(valid):
        return out
    iv = i_col[valid]; jv = j_row[valid]
    i0 = np.floor(iv).astype(np.int64); i1 = np.clip(i0 + 1, 0, W - 1)
    j0 = np.floor(jv).astype(np.int64); j1 = np.clip(j0 + 1, 0, H - 1)
    di = iv - i0; dj = jv - j0
    f00 = image[j0, i0]; f01 = image[j1, i0]
    f10 = image[j0, i1]; f11 = image[j1, i1]
    sampled = (
        (1 - di) * (1 - dj) * f00
        + (1 - di) * dj * f01
        + di * (1 - dj) * f10
        + di * dj * f11
    )
    out[valid] = sampled
    return out


def _rotated_pattern(
    pattern: np.ndarray,
    pixel_directions: np.ndarray,   # (H, W, 3)
    R: np.ndarray,                  # 3x3 rotation matrix
    pc: tuple[float, float, float],
) -> np.ndarray:
    """Build the rotated pattern: for each detector pixel p, sample the
    original pattern at the location that observes direction R^-1 · n(p).

    Returns array with same shape as `pattern`, NaN where the rotated
    direction lands off the detector.
    """
    H, W = pattern.shape
    R_inv = R.T  # rotation matrices are orthogonal — inverse is transpose
    # Apply R_inv to each pixel direction → (H, W, 3)
    rotated_dirs = pixel_directions @ R_inv.T
    # Project rotated dirs back onto the detector
    i_col, j_row = _direction_to_pixel(rotated_dirs, H, W, pc)
    return _bilinear_sample(pattern, i_col, j_row)


def _ncc_masked(
    a: np.ndarray, b: np.ndarray,
) -> tuple[float, int]:
    """NCC over the overlap where both arrays are finite. Returns
    (ncc, n_valid_pixels)."""
    valid = np.isfinite(a) & np.isfinite(b)
    n = int(valid.sum())
    if n < 50:
        return (float("nan"), n)
    av = a[valid].astype(np.float64)
    bv = b[valid].astype(np.float64)
    av -= av.mean(); bv -= bv.mean()
    denom = np.sqrt((av * av).sum() * (bv * bv).sum())
    if denom < 1e-12:
        return (float("nan"), n)
    return (float((av * bv).sum() / denom), n)


def detect_symmetry_spherical_averaged(
    patterns: list[np.ndarray] | np.ndarray,
    det_geom: dict,
    n_fibonacci_axes: int = 24,
) -> SymmetryReport:
    """Method B variant that averages a neighborhood of patterns before
    running symmetry detection.

    Real raw EBSD patterns are noisy; averaging a 3×3 (or larger) neighborhood
    of patterns substantially raises the NCC by suppressing pixel-level noise
    while preserving the band structure (which is shared across neighbouring
    pixels due to the limited probe size relative to grain size).

    Args:
        patterns: list of 2-D arrays OR a 3-D (N, H, W) stack — all the
            patterns to average. Must share the same H, W shape.
        det_geom: detector geometry (same for all patterns).
        n_fibonacci_axes: Fibonacci-spaced candidate axes count.

    Returns SymmetryReport with method='spherical_avg3x3' (or with the actual
    averaging size encoded). Falls back to single-pattern if list has one
    element.
    """
    if isinstance(patterns, np.ndarray):
        if patterns.ndim == 2:
            stack = patterns[np.newaxis, ...]
        elif patterns.ndim == 3:
            stack = patterns
        else:
            raise ValueError(
                f"patterns array must be 2D or 3D (got shape {patterns.shape})"
            )
    else:
        if not patterns:
            raise ValueError("patterns list is empty")
        first = np.asarray(patterns[0], dtype=np.float32)
        stack = np.stack([np.asarray(p, dtype=np.float32) for p in patterns])
    if stack.shape[0] == 0:
        raise ValueError("patterns is empty")
    if stack.shape[0] == 1:
        report = detect_symmetry_spherical(stack[0], det_geom, n_fibonacci_axes)
        # Method stays "spherical" — no averaging happened.
        return report

    # Average. Patterns can have different scales (Aztec gain compensation,
    # etc.); normalize each to unit-mean before averaging so a single
    # bright pattern doesn't dominate the mean.
    n = stack.shape[0]
    normed = np.empty_like(stack, dtype=np.float32)
    for i in range(n):
        p = stack[i].astype(np.float32)
        m = float(p.mean())
        normed[i] = p / m if m > 1e-6 else p
    avg_pat = normed.mean(axis=0).astype(np.float32)

    report = detect_symmetry_spherical(avg_pat, det_geom, n_fibonacci_axes)
    report.method = f"spherical_avg{n}"
    report.n_patterns_averaged = n
    return report


def detect_symmetry_spherical(
    pattern: np.ndarray,
    det_geom: dict,
    n_fibonacci_axes: int = 24,
) -> SymmetryReport:
    """Detect rotational symmetry of a single EBSD pattern using the
    spherical-projection method.

    Args:
        pattern: 2D experimental pattern (H, W), float-convertible.
        det_geom: dict with pc_x, pc_y, pc_z (Bruker convention) +
            pat_width, pat_height (we re-derive from pattern.shape).
        n_fibonacci_axes: how many Fibonacci-spaced candidate axes to
            sweep (in addition to the 9 explicit poles/diagonals).

    Returns SymmetryReport. confidence='none' = inconclusive.
    """
    if pattern.ndim != 2:
        raise ValueError(f"pattern must be 2D (got shape {pattern.shape})")
    H, W = pattern.shape
    if H < 32 or W < 32:
        raise ValueError(f"pattern too small for symmetry detection ({H}x{W})")

    pc = (
        float(det_geom["pc_x"]),
        float(det_geom["pc_y"]),
        float(det_geom["pc_z"]),
    )

    # Preprocess (background subtract + CLAHE) — same pipeline as Method A
    pat = _preprocess_pattern(pattern.astype(np.float32))

    # Pre-compute unit vectors per pixel (reused across all rotations).
    pixel_dirs = pixel_grid_directions(H, W, pc)        # (H, W, 3)

    # Candidate axes + folds
    axes = candidate_axes(n_fibonacci=n_fibonacci_axes)  # (n_axes, 3)

    report = SymmetryReport(zone_axis_yx=None, method="spherical")
    # For each fold, track the best NCC and the axis that produced it
    fold_best_ncc: dict[int, float] = {n: -1.0 for n in CANDIDATE_FOLDS}
    fold_best_axis: dict[int, np.ndarray] = {}

    for axis in axes:
        # Skip axes pointing away from detector (we already filtered for
        # upper hemisphere but explicit poles include +-x +-y which have
        # z = 0; those still work but produce identical rotation classes)
        for n in CANDIDATE_FOLDS:
            angle = 2 * np.pi / n
            R = _rodrigues_rotation_matrix(axis, angle)
            rotated = _rotated_pattern(pat, pixel_dirs, R, pc)
            score, n_valid = _ncc_masked(pat, rotated)
            if np.isnan(score):
                continue
            if score > fold_best_ncc[n]:
                fold_best_ncc[n] = score
                fold_best_axis[n] = axis

    report.n_fold_scores = {n: max(0.0, s) for n, s in fold_best_ncc.items()}

    # Pick highest-order fold within tie tolerance (so a perfect 6-fold
    # doesn't get reported as 2-fold). Match the Method A convention.
    if not report.n_fold_scores or max(report.n_fold_scores.values()) <= 0.0:
        report.confidence = "none"
        return report
    max_score = max(report.n_fold_scores.values())
    tie_tolerance = 0.08
    best_n: Optional[int] = None
    for n in sorted(report.n_fold_scores.keys(), reverse=True):
        if report.n_fold_scores[n] >= max_score - tie_tolerance:
            best_n = n
            break
    if best_n is None:
        best_n = max(report.n_fold_scores, key=report.n_fold_scores.get)
    best_score = report.n_fold_scores[best_n]

    # Zone-axis-on-detector projection (where the chosen axis hits the
    # detector — useful for the UI overlay).
    best_axis = fold_best_axis.get(best_n)
    if best_axis is not None and best_axis[2] > 0.05:
        # Project best_axis onto detector and report as (row, col)
        i_proj = best_axis[0] / best_axis[2] * pc[2] * W + pc[0] * W
        j_proj = -best_axis[1] / best_axis[2] * pc[2] * W + pc[1] * H
        if 0 <= i_proj < W and 0 <= j_proj < H:
            report.zone_axis_yx = (int(round(j_proj)), int(round(i_proj)))

    report.detected_n_fold = best_n
    report.best_score = best_score
    if best_score >= CONF_HIGH:
        report.confidence = "high"
    elif best_score >= CONF_MEDIUM:
        report.confidence = "medium"
    elif best_score >= CONF_LOW:
        report.confidence = "low"
    else:
        report.confidence = "none"
        report.detected_n_fold = None

    if report.detected_n_fold is not None:
        # Use the UNION of compatible systems across ALL folds that scored
        # at or above the LOW threshold — not just the single best fold.
        # On real EBSD patterns it's normal for multiple folds to be in the
        # 0.3-0.5 NCC range (the zone axis sits near several symmetry axes
        # simultaneously). Using only the best fold over-constrains:
        # e.g. a pixel with 3-fold=0.44, 4-fold=0.35 was flagged "cubic only"
        # → tetragonal phases like Al7FeCu2 got marked "off-system" and
        # penalised, even though the 4-fold NCC clearly supports them.
        compat = set()
        for fold, score in report.n_fold_scores.items():
            if score >= CONF_LOW:
                compat.update(compatible_systems_for_fold(fold))
        # Always include the best fold's systems even if just below LOW,
        # so the user gets some hint.
        compat.update(compatible_systems_for_fold(report.detected_n_fold))
        # Stable order: cubic first, then tetragonal/hexagonal/trigonal,
        # then everything else alphabetical. Matches user expectation in
        # Al-alloy workflow (cubic is the matrix).
        preferred = ["cubic", "tetragonal", "hexagonal", "trigonal",
                     "orthorhombic", "monoclinic", "triclinic"]
        report.compatible_systems = [s for s in preferred if s in compat]
    if report.best_score < 0.40 and report.confidence != "none":
        report.warnings.append(
            "Spherical-projection symmetry NCC is moderate — pattern may be "
            "noisy, off-zone-axis, or have low-symmetry parts dominating."
        )

    return report
