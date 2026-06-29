"""Region / Whole-Scan analysis (Mode 2) for Crystal Hint.

Aggregates symmetry detection + lattice estimation across many pixels.
Reports:
  - Histogram of detected n-folds
  - Histogram of estimated lattice 'a'
  - Top 'missing phases' (phases that would fit a significant fraction of
    pixels but aren't in the local SHT library)
  - Pixel count per detected crystal system

Sampling strategy: cap at `max_pixels` (default 256) to keep total time
under ~30s on a typical scan. ROI shapes: rectangle, polygon, whole.

See docs/superpowers/specs/2026-05-26-crystal-hint-feature-design.md Section 6.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from backend.api.services.crystal_hint_lattice import (
    LatticeReport,
    estimate_lattice,
)
from backend.api.services.crystal_hint_local_library import search as library_search
from backend.api.services.crystal_hint_symmetry import (
    SymmetryReport,
    detect_symmetry,
)
from backend.api.services.crystal_hint_spherical import (
    detect_symmetry_spherical,
)

logger = logging.getLogger(__name__)


@dataclass
class RegionAnalysisResult:
    n_pixels_analyzed: int
    n_pixels_in_roi: int
    n_pixels_no_symmetry: int   # confidence == "none"
    n_fold_histogram: dict[int, int] = field(default_factory=dict)
    crystal_system_histogram: dict[str, int] = field(default_factory=dict)
    lattice_a_histogram: dict[str, int] = field(default_factory=dict)
    # Each bin keyed by "lo-hi Å" string, e.g. "3.5-5.0"
    library_match_rate: float = 0.0  # fraction of pixels that found a library match with score > 0.7
    # Top missing phases (signatures present in N pixels but no library match)
    missing_phase_clusters: list[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_pixels_analyzed": self.n_pixels_analyzed,
            "n_pixels_in_roi": self.n_pixels_in_roi,
            "n_pixels_no_symmetry": self.n_pixels_no_symmetry,
            "n_fold_histogram": {str(k): int(v) for k, v in self.n_fold_histogram.items()},
            "crystal_system_histogram": dict(self.crystal_system_histogram),
            "lattice_a_histogram": dict(self.lattice_a_histogram),
            "library_match_rate": float(self.library_match_rate),
            "missing_phase_clusters": self.missing_phase_clusters,
            "elapsed_seconds": float(self.elapsed_seconds),
            "warnings": list(self.warnings),
        }


def _bin_lattice(a_A: Optional[float]) -> Optional[str]:
    """Bin a lattice parameter into a coarse band label."""
    if a_A is None or not np.isfinite(a_A):
        return None
    bins = [
        (0, 3.5, "<3.5"),
        (3.5, 5.0, "3.5-5.0"),
        (5.0, 7.0, "5.0-7.0"),
        (7.0, 10.0, "7.0-10.0"),
        (10.0, 15.0, "10.0-15.0"),
        (15.0, 100.0, ">15"),
    ]
    for lo, hi, label in bins:
        if lo <= a_A < hi:
            return label
    return ">15"


def _sample_indices(
    n_rows: int,
    n_cols: int,
    roi_mask: Optional[np.ndarray],
    max_pixels: int,
) -> list[tuple[int, int]]:
    """Pick a representative subset of pixels.

    If the ROI fits in max_pixels: return all of it.
    Otherwise: uniformly sub-sample on a grid that covers the ROI.
    """
    if roi_mask is None:
        rows = np.arange(n_rows)
        cols = np.arange(n_cols)
        # Full-scan: subsample on a regular grid
        total = n_rows * n_cols
        if total <= max_pixels:
            return [(r, c) for r in rows for c in cols]
        step = max(1, int(np.ceil(np.sqrt(total / max_pixels))))
        return [(r, c) for r in range(0, n_rows, step) for c in range(0, n_cols, step)]
    # ROI present: collect pixel coords, subsample uniformly. Cast to plain
    # ints since downstream callers may type-check or use them as keys.
    coords = [(int(r), int(c)) for r, c in zip(*np.where(roi_mask))]
    if len(coords) <= max_pixels:
        return coords
    rng = np.random.default_rng(seed=0)
    idx = rng.choice(len(coords), size=max_pixels, replace=False)
    return [coords[i] for i in idx]


def analyze_region(
    get_pattern: callable,
    n_rows: int,
    n_cols: int,
    roi_mask: Optional[np.ndarray] = None,
    detector_geom: Optional[dict] = None,
    elements: Optional[list[str]] = None,
    max_pixels: int = 256,
    avg_radius: int = 0,
) -> RegionAnalysisResult:
    """Analyze a region of an EBSD scan.

    Args:
        get_pattern: callable (row, col) -> np.ndarray pattern (or None).
            Caller supplies this so we don't bake in a particular signal API.
        n_rows, n_cols: scan dimensions.
        roi_mask: bool array (n_rows, n_cols) or None for full scan.
        detector_geom: dict for lattice estimator (pc_x/y/z, voltage_kv, ...).
        elements: sample chemistry for library_search.
        max_pixels: cap on number of pixels analyzed (subsampled).

    Returns RegionAnalysisResult.
    """
    import time
    t0 = time.perf_counter()
    elements = elements or []

    sample_pixels = _sample_indices(n_rows, n_cols, roi_mask, max_pixels)
    n_in_roi = int(roi_mask.sum()) if roi_mask is not None else (n_rows * n_cols)
    result = RegionAnalysisResult(
        n_pixels_analyzed=0,
        n_pixels_in_roi=n_in_roi,
        n_pixels_no_symmetry=0,
    )

    n_fold_counter: Counter[int] = Counter()
    system_counter: Counter[str] = Counter()
    lattice_bin_counter: Counter[str] = Counter()
    missing_signatures: list[tuple] = []
    n_library_match = 0

    for (r, c) in sample_pixels:
        try:
            pat = get_pattern(r, c)
        except Exception as exc:
            logger.debug("Pattern fetch failed at (%d,%d): %s", r, c, exc)
            continue
        if pat is None:
            continue
        result.n_pixels_analyzed += 1
        pat_arr = np.asarray(pat, dtype=np.float32)
        if pat_arr.ndim != 2:
            continue
        # Optional neighborhood averaging (matches Mode 1 behavior). Pulls
        # (2*avg_radius+1)^2 neighbours, mean-normalises, averages. Edges
        # of the scan / ROI return fewer neighbours, which is fine.
        if avg_radius > 0:
            stack = [pat_arr]
            for dr in range(-avg_radius, avg_radius + 1):
                for dc in range(-avg_radius, avg_radius + 1):
                    if dr == 0 and dc == 0:
                        continue
                    rn, cn = r + dr, c + dc
                    if 0 <= rn < n_rows and 0 <= cn < n_cols:
                        try:
                            pn = get_pattern(rn, cn)
                        except Exception:
                            continue
                        if pn is None:
                            continue
                        arr_n = np.asarray(pn, dtype=np.float32)
                        if arr_n.ndim == 2:
                            stack.append(arr_n)
            if len(stack) > 1:
                normed = []
                for p in stack:
                    m = float(p.mean())
                    normed.append(p / m if m > 1e-6 else p)
                pat_arr = np.mean(normed, axis=0).astype(np.float32)
        # Prefer Method B (spherical) when detector geometry is available —
        # gives meaningful NCC on real (off-center PC, tilted) patterns.
        if detector_geom is not None:
            try:
                sym = detect_symmetry_spherical(pat_arr, detector_geom)
            except Exception as exc:
                logger.debug("Spherical symmetry failed at (%d,%d): %s — "
                             "falling back to image-space", r, c, exc)
                sym = detect_symmetry(pat_arr)
        else:
            sym = detect_symmetry(pat_arr)
        if sym.confidence == "none" or sym.detected_n_fold is None:
            result.n_pixels_no_symmetry += 1
            continue
        n_fold_counter[sym.detected_n_fold] += 1
        # Compatible systems → take first as proxy for system histogram
        if sym.compatible_systems:
            system_counter[sym.compatible_systems[0]] += 1
        # Lattice estimation
        if detector_geom is not None:
            try:
                # Pick a system hint from symmetry compatible_systems
                sys_hint = "cubic" if "cubic" in sym.compatible_systems else \
                    (sym.compatible_systems[0] if sym.compatible_systems else None)
                lat = estimate_lattice(pat_arr, detector_geom, crystal_system_hint=sys_hint)
                bin_label = _bin_lattice(lat.a_estimate_A)
                if bin_label:
                    lattice_bin_counter[bin_label] += 1
                # Library match — does any local entry score > 0.7?
                a_range = lat.a_range_A
                matches = library_search(
                    elements=elements,
                    crystal_system=sys_hint,
                    a_range_A=a_range,
                    strict_chemistry=True,
                )
                if matches and matches[0].score > 0.7:
                    n_library_match += 1
                else:
                    # Record this pixel's signature as a "missing" candidate
                    if lat.a_estimate_A:
                        missing_signatures.append((
                            sys_hint or "unknown",
                            round(lat.a_estimate_A, 1),
                            sym.detected_n_fold,
                        ))
            except Exception as exc:
                logger.debug("Lattice estimation failed at (%d,%d): %s", r, c, exc)

    result.n_fold_histogram = dict(n_fold_counter)
    result.crystal_system_histogram = dict(system_counter)
    result.lattice_a_histogram = dict(lattice_bin_counter)
    if result.n_pixels_analyzed > 0:
        result.library_match_rate = n_library_match / result.n_pixels_analyzed

    # Cluster missing signatures: same (system, a_rounded, n_fold)
    if missing_signatures:
        sig_counter = Counter(missing_signatures)
        # Sort by count desc, take top 10
        result.missing_phase_clusters = [
            {
                "crystal_system": sig[0],
                "lattice_a_A": sig[1],
                "n_fold": sig[2],
                "pixel_count": count,
                "fraction_of_analyzed": count / result.n_pixels_analyzed if result.n_pixels_analyzed > 0 else 0,
            }
            for sig, count in sig_counter.most_common(10)
        ]

    result.elapsed_seconds = time.perf_counter() - t0
    return result
