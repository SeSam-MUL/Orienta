"""Group EDS pixels by composition, then match each group to the CIF library.

Per-pixel matching is unreliable on this data. Measured on SampleB, the
aluminium matrix reads a median 84.2 at% Al and 2.80 at% Fe, while
equilibrium Fe solubility in aluminium is ~0.05 at% — i.e. several at% of
background, interaction-volume bleed from nearby particles, and standardless
Cliff-Lorimer error. Nominal stoichiometries in the library are separated by
as little as 1.1 at%, so no per-pixel rule can be trusted to choose between
them.

A cluster's mean composition is far closer to a nominal stoichiometry than
any single pixel (the SampleB matrix cluster means 95-97 at% Al), so matching
happens there. The mean is taken over the cluster's ERODED interior:
``grain_phase_assignment`` records that measuring over a full region instead
of its interior made "every real particle inhomogeneous" (44.5 at% spread vs
0.0), because the interaction volume mixes at the rim.

Spec: docs/superpowers/specs/2026-08-19-eds-chemistry-phase-map-design.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from backend.api.services.chemistry_score import (
    DEFAULT_REL_REQ, has_chemistry, score_phase_vectorised,
)
from backend.api.services.cif_phase_library import (
    TIE_TOLERANCE, CifPhaseEntry, group_degenerate_entries,
)
from backend.api.services.crystal_hint_phase_fit import _CHEM_IGNORE


@dataclass
class ClusterMatch:
    """One composition cluster and the library phase it was matched to."""

    cluster_id: int
    n_pixels: int
    mean_at_pct: Dict[str, float]          # over the eroded interior
    phase_index: int                       # -1 = no acceptable match
    score: float
    runners_up: List[Tuple[int, float]] = field(default_factory=list)
    ambiguous: bool = False


def _feature_matrix(
    at_pct_per_element: Dict[str, np.ndarray], n_px: int,
) -> Tuple[List[str], np.ndarray]:
    """Per-pixel composition as fractions over the metallic elements."""
    els = sorted(el for el in at_pct_per_element if el not in _CHEM_IGNORE)
    if not els:
        return [], np.zeros((n_px, 0))
    X = np.stack(
        [np.maximum(0.0, np.asarray(at_pct_per_element[el], dtype=np.float64))
         for el in els],
        axis=1,
    )
    tot = X.sum(axis=1, keepdims=True)
    return els, X / np.where(tot > 1e-9, tot, 1.0)


def choose_k_by_bic(X: np.ndarray, k_range: Tuple[int, int] = (2, 12)) -> int:
    """Pick the cluster count by Bayesian information criterion.

    Kept for comparison and testing. NOT the default — see
    :func:`phase_coherence` for why BIC is the wrong objective here.
    """
    from sklearn.mixture import GaussianMixture

    lo, hi = k_range
    lo = max(1, lo)
    hi = max(lo, min(hi, X.shape[0] // 10))
    best_k, best_bic = lo, np.inf
    for k in range(lo, hi + 1):
        try:
            gm = GaussianMixture(
                n_components=k, covariance_type="diag",
                random_state=0, reg_covar=1e-6,
            ).fit(X)
            bic = float(gm.bic(X))
        except Exception:  # degenerate fit for this k — skip it
            continue
        if bic < best_bic:
            best_bic, best_k = bic, k
    return best_k


def phase_coherence(phase_grid: np.ndarray) -> float:
    """Fraction of 4-neighbour pixel pairs that share a phase, in [0, 1].

    This is the objective that matters for a phase map, and it is why BIC
    is not used to pick ``k``. BIC scores how well a mixture models the
    composition *density*; real EDS composition is a noisy continuum, so BIC
    keeps adding components (measured on SampleB: it chose k=12). Those extra
    components split one physical phase's noise distribution into several
    clusters that straddle the phase-decision boundary and interleave
    spatially, so the map turns to confetti even though every cluster is
    internally consistent.

    Measured on SampleB, share of pixels whose neighbourhood mostly
    disagrees: k=12 (BIC) 18.6 %, k=6 14.0 %, k=4 1.5 %, k=3 2.3 %.
    """
    g = np.asarray(phase_grid)
    same = 0
    total = 0
    if g.shape[0] > 1:
        same += int((g[1:] == g[:-1]).sum())
        total += g[1:].size
    if g.shape[1] > 1:
        same += int((g[:, 1:] == g[:, :-1]).sum())
        total += g[:, 1:].size
    return (same / total) if total else 1.0


def _interior_mask(
    cluster_grid: np.ndarray, cid: int, n_rows: int, n_cols: int,
) -> np.ndarray:
    """Eroded interior of one cluster, falling back to the full mask.

    The fallback matters for thin or small clusters: a one-pixel-wide
    feature erodes to nothing, and reporting an empty mean would be worse
    than reporting a rim-contaminated one.
    """
    m = (cluster_grid == cid).reshape(n_rows, n_cols)
    eroded = ndimage.binary_erosion(
        m, structure=np.ones((3, 3), dtype=bool), border_value=0,
    )
    return (eroded if eroded.any() else m).ravel()


def _run_for_k(
    X: np.ndarray,
    at_pct_per_element: Dict[str, np.ndarray],
    n_rows: int,
    n_cols: int,
    candidates: List[CifPhaseEntry],
    group_of: np.ndarray,
    k: int,
    min_score: float,
    rel_req: float,
) -> Tuple[np.ndarray, np.ndarray, List[ClusterMatch]]:
    """One full cluster-and-match pass at a fixed ``k``."""
    from sklearn.cluster import KMeans

    n_px = n_rows * n_cols
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    cluster_grid = km.labels_.astype(np.int32)

    phase_grid = np.full(n_px, -1, dtype=np.int32)
    matches: List[ClusterMatch] = []

    for cid in range(k):
        mask = (cluster_grid == cid)
        if not mask.any():
            continue
        interior = _interior_mask(cluster_grid, cid, n_rows, n_cols)
        mean_at = {
            el: float(np.asarray(arr, dtype=float)[interior].mean())
            for el, arr in at_pct_per_element.items()
        }

        # Score the cluster mean as a single "pixel". no_data_score=0.0 so a
        # cluster of unmeasured pixels (KMeans reliably isolates one, and its
        # mean is all zeros) scores 0 and stays unassigned instead of
        # scoring a perfect 1.0 — the highest confidence anywhere on the map.
        one = {el: np.array([v], dtype=np.float64) for el, v in mean_at.items()}
        scored = [
            (idx, float(score_phase_vectorised(
                one, entry.composition, rel_req=rel_req, no_data_score=0.0)[0]))
            for idx, entry in enumerate(candidates)
        ]
        scored.sort(key=lambda t: -t[1])

        if scored and scored[0][1] >= min_score:
            best_idx, best_s = scored[0]
            ambiguous = any(
                group_of[i] != group_of[best_idx] and (best_s - s) <= TIE_TOLERANCE
                for i, s in scored[1:]
            )
            phase_grid[mask] = best_idx
        else:
            best_idx = -1
            best_s = float(scored[0][1]) if scored else 0.0
            ambiguous = False

        matches.append(ClusterMatch(
            cluster_id=cid,
            n_pixels=int(mask.sum()),
            mean_at_pct=mean_at,
            phase_index=best_idx,
            score=best_s,
            runners_up=[(i, s) for i, s in scored[1:4]],
            ambiguous=ambiguous,
        ))

    # A matched cluster paints all of its pixels, but individual pixels with
    # no measurement inside it are still unmeasured — never claim a phase
    # for them on the strength of their neighbours' chemistry.
    phase_grid[~has_chemistry(at_pct_per_element)] = -1

    return (phase_grid.reshape(n_rows, n_cols),
            cluster_grid.reshape(n_rows, n_cols),
            matches)


def cluster_and_match(
    at_pct_per_element: Dict[str, np.ndarray],
    n_rows: int,
    n_cols: int,
    candidates: List[CifPhaseEntry],
    k: Optional[int] = None,
    k_range: Tuple[int, int] = (2, 8),
    min_score: float = 0.3,
    rel_req: float = DEFAULT_REL_REQ,
) -> Tuple[np.ndarray, np.ndarray, List[ClusterMatch], int]:
    """Cluster the composition, match each cluster, paint its pixels.

    With ``k=None`` the cluster count is chosen by the spatial coherence of
    the resulting phase map (see :func:`phase_coherence`), not by BIC —
    picking the run that actually yields grain-like regions. Degenerate
    solutions are excluded: a ``k`` collapsing the map to a single phase
    scores perfect coherence and is never what the user wants, so only runs
    producing at least two distinct phases compete. Ties go to the smaller
    ``k``.

    Returns ``(phase_grid, cluster_grid, matches, k_used)``. A cluster whose
    best match scores below ``min_score`` is left unassigned (-1) and
    reported in ``matches`` with ``phase_index == -1`` — an unmatched cluster
    is visible and honest, and must never be forced onto the nearest phase.
    """
    n_px = n_rows * n_cols
    empty = np.full((n_rows, n_cols), -1, dtype=np.int32)
    if not at_pct_per_element or n_px <= 0:
        return empty, empty.copy(), [], 0

    els, X = _feature_matrix(at_pct_per_element, n_px)
    if not els:
        return empty, empty.copy(), [], 0

    group_of = np.zeros(len(candidates), dtype=np.int32)
    for gid, members in enumerate(group_degenerate_entries(candidates)):
        for m in members:
            group_of[m] = gid

    def run(kk):
        return _run_for_k(X, at_pct_per_element, n_rows, n_cols, candidates,
                          group_of, kk, min_score, rel_req)

    if k:
        k_used = max(1, min(int(k), n_px))
        phase_grid, cluster_grid, matches = run(k_used)
        return phase_grid, cluster_grid, matches, k_used

    lo, hi = k_range
    lo = max(2, lo)
    hi = max(lo, min(hi, n_px))
    best = None
    for kk in range(lo, hi + 1):
        try:
            pg, cg, ms = run(kk)
        except Exception:      # degenerate fit at this k — try the next
            continue
        n_phases = len({int(v) for v in np.unique(pg) if v >= 0})
        coh = phase_coherence(pg)
        # Strictly greater keeps the smallest k on a tie.
        if n_phases >= 2 and (best is None or coh > best[0]):
            best = (coh, kk, pg, cg, ms)

    if best is None:           # never resolved 2+ phases — fall back
        k_used = max(1, min(lo, n_px))
        phase_grid, cluster_grid, matches = run(k_used)
        return phase_grid, cluster_grid, matches, k_used

    _coh, k_used, phase_grid, cluster_grid, matches = best
    return phase_grid, cluster_grid, matches, k_used
