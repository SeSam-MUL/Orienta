"""Quality Check (Mode 3) for Crystal Hint.

Compares the experimentally-detected pattern symmetry against the
indexed phase's expected crystal system. Flags pixels where the
detected n-fold is INCOMPATIBLE with the indexed phase's symmetry
class — a strong signal of mis-indexing or missing phase.

For each pixel:
  1. Detect symmetry → compatible_systems list (from `crystal_hint_symmetry`)
  2. Look up indexed phase via the active xmap → get its crystal system
  3. Compatible iff indexed_system ∈ detected.compatible_systems

Returns a 2D mismatch mask + summary counts + per-pixel detail.

See docs/superpowers/specs/2026-05-26-crystal-hint-feature-design.md Section 6.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from backend.api.services.crystal_hint_symmetry import detect_symmetry
from backend.api.services.crystal_hint_spherical import detect_symmetry_spherical

logger = logging.getLogger(__name__)


@dataclass
class QualityCheckResult:
    n_pixels_analyzed: int
    n_pixels_in_roi: int
    n_pixels_indexed: int
    n_pixels_unindexed: int       # phase_id == 0 or no symmetry
    n_pixels_match: int           # detected symmetry matches indexed system
    n_pixels_mismatch: int        # detected symmetry incompatible
    n_pixels_no_detection: int    # symmetry confidence too low to judge
    # Mismatch breakdown by indexed phase id → count
    mismatch_by_phase: dict[int, dict] = field(default_factory=dict)
    # Optional sampled mismatch detail: list of (row, col, detected_n_fold,
    # indexed_phase_id, indexed_phase_name, indexed_system)
    sample_mismatches: list[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_pixels_analyzed": self.n_pixels_analyzed,
            "n_pixels_in_roi": self.n_pixels_in_roi,
            "n_pixels_indexed": self.n_pixels_indexed,
            "n_pixels_unindexed": self.n_pixels_unindexed,
            "n_pixels_match": self.n_pixels_match,
            "n_pixels_mismatch": self.n_pixels_mismatch,
            "n_pixels_no_detection": self.n_pixels_no_detection,
            "mismatch_rate": (self.n_pixels_mismatch
                              / max(1, self.n_pixels_indexed - self.n_pixels_no_detection)),
            "mismatch_by_phase": {str(k): v for k, v in self.mismatch_by_phase.items()},
            "sample_mismatches": list(self.sample_mismatches),
            "elapsed_seconds": float(self.elapsed_seconds),
            "warnings": list(self.warnings),
        }


def _system_from_phase(phase) -> Optional[str]:
    """Extract canonical crystal-system string from an orix Phase object.

    Tries multiple attributes since orix versions differ.
    """
    # Newer orix: phase.point_group.crystal_system
    try:
        cs = phase.point_group.system
        if cs:
            return str(cs).lower()
    except Exception:
        pass
    try:
        cs = phase.space_group.crystal_system
        if cs:
            return str(cs).lower()
    except Exception:
        pass
    # Fallback: parse space-group number directly if available
    try:
        sg_num = phase.space_group.number
        if sg_num:
            return _system_from_sg_number(int(sg_num))
    except Exception:
        pass
    return None


def _system_from_sg_number(sg: int) -> str:
    if 1 <= sg <= 2: return "triclinic"
    if 3 <= sg <= 15: return "monoclinic"
    if 16 <= sg <= 74: return "orthorhombic"
    if 75 <= sg <= 142: return "tetragonal"
    if 143 <= sg <= 167: return "trigonal"
    if 168 <= sg <= 194: return "hexagonal"
    if 195 <= sg <= 230: return "cubic"
    return "unknown"


def _is_compatible(detected_systems: list[str], indexed_system: str) -> bool:
    """Determine whether the symmetry detector's compatible systems include
    the indexed phase's actual system.

    Trigonal and hexagonal are degenerate w.r.t. n-fold-only symmetry
    detection (both have 3-fold + 6-fold axes); treat as a match.
    """
    if not detected_systems or not indexed_system:
        return True  # can't judge → don't flag
    s = indexed_system.lower()
    if s in detected_systems:
        return True
    # Trigonal ↔ hexagonal: 3-fold sub-symmetry of 6-fold, treat as match.
    # Symmetric: if EITHER side is in {trig, hex} AND the other ALSO contains
    # one of those, accept.
    trig_hex = {"trigonal", "hexagonal"}
    if s in trig_hex and (trig_hex & set(detected_systems)):
        return True
    return False


def quality_check_region(
    get_pattern: Callable[[int, int], Optional[np.ndarray]],
    xmap,                                # orix CrystalMap, may be None
    n_rows: int,
    n_cols: int,
    roi_mask: Optional[np.ndarray] = None,
    max_pixels: int = 256,
    detector_geom: Optional[dict] = None,
    avg_radius: int = 0,
) -> QualityCheckResult:
    """Compare detected vs indexed symmetry for each pixel in the ROI.

    Args:
        get_pattern: callable (row, col) → pattern array (or None).
        xmap: orix CrystalMap from the active indexing result. If None,
            we can't compare; result will report n_pixels_indexed=0.
        n_rows, n_cols: scan shape (rows × cols).
        roi_mask: bool (n_rows, n_cols), or None = whole scan.
        max_pixels: cap on number of pixels analyzed.

    Returns QualityCheckResult.
    """
    import time
    t0 = time.perf_counter()

    if roi_mask is None:
        # Whole-scan subsample
        total = n_rows * n_cols
        if total <= max_pixels:
            coords = [(r, c) for r in range(n_rows) for c in range(n_cols)]
        else:
            step = max(1, int(np.ceil(np.sqrt(total / max_pixels))))
            coords = [(r, c) for r in range(0, n_rows, step) for c in range(0, n_cols, step)]
    else:
        flat = [(int(r), int(c)) for r, c in zip(*np.where(roi_mask))]
        if len(flat) <= max_pixels:
            coords = flat
        else:
            rng = np.random.default_rng(seed=0)
            idx = rng.choice(len(flat), size=max_pixels, replace=False)
            coords = [flat[i] for i in idx]

    n_in_roi = int(roi_mask.sum()) if roi_mask is not None else (n_rows * n_cols)
    result = QualityCheckResult(
        n_pixels_analyzed=0,
        n_pixels_in_roi=n_in_roi,
        n_pixels_indexed=0,
        n_pixels_unindexed=0,
        n_pixels_match=0,
        n_pixels_mismatch=0,
        n_pixels_no_detection=0,
    )

    if xmap is None:
        result.warnings.append("No active indexing result — cannot compare.")
        result.elapsed_seconds = time.perf_counter() - t0
        return result

    # Pre-compute phase-id → system map for all phases in the xmap.
    # orix PhaseList API: phases.ids → list of int ids; phases[id] → Phase obj.
    # Older orix versions yielded tuples on iteration; we try several access
    # patterns and fall back to enumerate() if all else fails.
    phase_system_cache: dict[int, str] = {}
    phase_name_cache: dict[int, str] = {}
    try:
        # Newest orix: explicit .ids property
        if hasattr(xmap.phases, "ids"):
            ids_iter = list(xmap.phases.ids)
            for pid in ids_iter:
                try:
                    phase = xmap.phases[pid]
                except Exception:
                    continue
                try:
                    sys = _system_from_phase(phase)
                    phase_system_cache[int(pid)] = sys or "unknown"
                    phase_name_cache[int(pid)] = getattr(phase, "name", str(pid))
                except Exception:
                    continue
        else:
            # Legacy: iter yields (id, phase) tuples — kept for old orix
            for entry in xmap.phases:
                try:
                    if isinstance(entry, tuple):
                        pid, phase = entry
                    else:
                        # Single value — could be id or Phase. Try both.
                        try:
                            phase = xmap.phases[entry]
                            pid = entry
                        except Exception:
                            phase = entry
                            pid = getattr(phase, "id", None)
                    if pid is None:
                        continue
                    sys = _system_from_phase(phase)
                    phase_system_cache[int(pid)] = sys or "unknown"
                    phase_name_cache[int(pid)] = getattr(phase, "name", str(pid))
                except Exception:
                    continue
    except Exception as exc:
        logger.warning("Failed to enumerate xmap.phases: %s", exc)

    mismatch_counts_by_phase: dict[int, int] = Counter()
    match_counts_by_phase: dict[int, int] = Counter()

    for (r, c) in coords:
        try:
            pat = get_pattern(r, c)
        except Exception:
            continue
        if pat is None:
            continue
        result.n_pixels_analyzed += 1

        # Lookup indexed phase_id at this pixel
        px_idx = r * n_cols + c
        try:
            phase_id = int(xmap.phase_id[px_idx])
        except Exception:
            result.n_pixels_unindexed += 1
            continue
        # orix convention: -1 = unindexed sentinel. 0 IS a valid indexed phase
        # (matches the project's `_inject_phase_names` logic that starts ids
        # at 0 OR 1 depending on the indexer).
        if phase_id < 0:
            result.n_pixels_unindexed += 1
            continue
        result.n_pixels_indexed += 1
        indexed_system = phase_system_cache.get(phase_id, "unknown")

        # Detect symmetry on the experimental pattern
        pat_arr = np.asarray(pat, dtype=np.float32)
        if pat_arr.ndim != 2:
            continue
        # Optional neighborhood averaging (matches Mode 1 + 2 behavior).
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
        if detector_geom is not None:
            try:
                sym = detect_symmetry_spherical(pat_arr, detector_geom)
            except Exception as exc:
                logger.debug("Spherical symmetry failed at (%d,%d): %s — "
                             "falling back to image-space", r, c, exc)
                sym = detect_symmetry(pat_arr)
        else:
            sym = detect_symmetry(pat_arr)
        if sym.confidence == "none" or not sym.compatible_systems:
            result.n_pixels_no_detection += 1
            continue

        if _is_compatible(sym.compatible_systems, indexed_system):
            result.n_pixels_match += 1
            match_counts_by_phase[phase_id] += 1
        else:
            result.n_pixels_mismatch += 1
            mismatch_counts_by_phase[phase_id] += 1
            if len(result.sample_mismatches) < 24:
                result.sample_mismatches.append({
                    "row": r, "col": c,
                    "detected_n_fold": sym.detected_n_fold,
                    "compatible_systems": list(sym.compatible_systems),
                    "indexed_phase_id": phase_id,
                    "indexed_phase_name": phase_name_cache.get(phase_id, str(phase_id)),
                    "indexed_system": indexed_system,
                })

    # Per-phase breakdown
    all_pids = set(mismatch_counts_by_phase) | set(match_counts_by_phase)
    for pid in all_pids:
        miss = mismatch_counts_by_phase.get(pid, 0)
        match = match_counts_by_phase.get(pid, 0)
        total = miss + match
        result.mismatch_by_phase[int(pid)] = {
            "phase_name": phase_name_cache.get(pid, str(pid)),
            "indexed_system": phase_system_cache.get(pid, "unknown"),
            "match_count": int(match),
            "mismatch_count": int(miss),
            "mismatch_rate": miss / total if total > 0 else 0,
        }

    result.elapsed_seconds = time.perf_counter() - t0
    return result
