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
    DEFAULT_REL_REQ, background_levels, has_chemistry, infer_matrix_element,
    score_phase_ratio,
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


#: Box width, in pixels, for the smoothing applied before clustering.
#: Zero disables it and reproduces the pre-2026-08-24 behaviour exactly.
#:
#: This is the single most important knob in this module, and leaving it at
#: zero was a real defect. KMeans on raw per-pixel at% produces confetti, and
#: the only lever against that was keeping ``k`` tiny — so ``k`` was doing
#: smoothing's job and the map collapsed to three structures no matter what
#: the element maps showed. Measured on SampleB (90x120, 8 elements), cluster
#: coherence (fraction of 4-neighbour pairs in the same cluster):
#:
#:   smoothing |  k=3    k=4    k=6    k=8    k=10   k=12
#:   ----------+------------------------------------------
#:         raw |  0.947  0.665  0.573  0.457  0.346  0.305
#:         3x3 |  0.976  0.974  0.803  0.778  0.703  0.670
#:         5x5 |  0.978  0.958  0.936  0.831  0.808  0.751
#:         7x7 |  0.979  0.960  0.940  0.938  0.838  0.811
#:
#: On the raw row coherence falls monotonically with k, so any criterion that
#: maximises it is forced to the smallest k. Connected pieces at k=8: raw gives
#: 3491 pieces with a MEDIAN SIZE OF ONE PIXEL; 5x5 gives 174, largest 3759.
#:
#: The cost is boundary resolution — features thinner than the box are absorbed
#: — which is why this is a user-facing control and not a constant.
DEFAULT_SCALE = 5


def _smooth_maps(
    at_pct_per_element: Dict[str, np.ndarray], n_rows: int, n_cols: int,
    scale: int,
) -> Dict[str, np.ndarray]:
    """Box-average each element map. ``scale <= 1`` returns the input as-is.

    Smoothing the COMPOSITION rather than the label map is deliberate: it
    denoises the quantity the clustering actually reasons about, so cluster
    means stay physically meaningful. Filtering labels afterwards would only
    hide the speckle.
    """
    if scale is None or scale <= 1:
        return at_pct_per_element
    out: Dict[str, np.ndarray] = {}
    for el, arr in at_pct_per_element.items():
        a = np.asarray(arr, dtype=np.float64).reshape(n_rows, n_cols)
        out[el] = ndimage.uniform_filter(a, size=int(scale), mode="nearest").ravel()
    return out


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


#: Two cluster means closer than this, in at% on their largest single-element
#: difference, are the same chemistry as far as this data can tell.
#:
#: Calibrated on SampleB, where the minimum pairwise gap falls
#: 14.16 (k=2) / 13.58 (k=4) / 7.16 (k=6) / 4.55 (k=7) / 1.37 (k=8) / 0.99 (k=12).
#: Any threshold from 1.5 to 4.0 selects k=7 — a plateau, not a cliff. The
#: value also has to sit above the measurement error: the same module docstring
#: records the aluminium matrix reading 2.80 at% Fe where equilibrium solubility
#: is ~0.05 at%, so differences of a few tenths of an at% are not evidence of a
#: different chemistry.
DISTINCT_AT_PCT = 2.0


def _cluster_means(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Mean feature vector per cluster, in at% (the features are fractions)."""
    out = np.zeros((k, X.shape[1]), dtype=np.float64)
    for c in range(k):
        m = labels == c
        if m.any():
            out[c] = X[m].mean(axis=0) * 100.0
    return out


def min_pairwise_gap(X: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Smallest largest-single-element difference between any two clusters.

    "Largest single element" rather than a Euclidean distance on purpose: two
    structures differing by 3 at% Si and nothing else ARE different structures,
    but that difference is diluted to near-nothing in a norm over eight
    elements dominated by Al.
    """
    means = _cluster_means(X, labels, k)
    gap = np.inf
    for i in range(k):
        for j in range(i):
            gap = min(gap, float(np.abs(means[i] - means[j]).max()))
    return 0.0 if not np.isfinite(gap) else gap


def choose_k_by_distinctness(
    X: np.ndarray, k_range: Tuple[int, int] = (2, 12),
    min_gap: float = DISTINCT_AT_PCT,
) -> int:
    """Largest ``k`` whose clusters are all still chemically distinct.

    Raise k while every cluster mean stays at least ``min_gap`` at% away from
    every other on some element; stop when clusters start duplicating each
    other. This asks the only question k should answer — "how many distinct
    chemistries are on this map" — without letting map tidiness or cluster
    area into the objective.

    Two criteria were measured and rejected, both recorded so they are not
    retried:

    * **Map coherence** (the previous default) falls monotonically with k on
      unsmoothed data, so maximising it is FORCED to the smallest k. SampleB
      collapsed to 3 structures while the element maps plainly showed more.
      Tidiness is what :data:`DEFAULT_SCALE` is for — one lever per problem.
    * **Silhouette** peaks at k=3-4 here (0.834 / 0.838) because it averages
      over pixels, and the structures that matter are 70 px out of 10 800.
      Any area-weighted criterion is dominated by the matrix. Measured, not
      assumed.

    Over-segmentation is the safer error: on SampleB k=7 splits one Si
    particle into three concentric rings (Si 24 / 36 / 52 at%) because the
    interaction volume makes a small particle read as a gradient. Merging
    those is one click; recovering a structure that was never separated is
    not. So this deliberately errs high.
    """
    from sklearn.cluster import KMeans

    lo, hi = max(2, int(k_range[0])), int(k_range[1])
    n = X.shape[0]
    if n <= lo or hi < lo:
        return lo
    hi = min(hi, n - 1)

    best = lo
    for k in range(lo, hi + 1):
        try:
            labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        except Exception:          # degenerate fit at this k
            break
        if len(np.unique(labels)) < k:
            break
        if min_pairwise_gap(X, labels, k) < min_gap:
            break
        best = k
    return best


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
    matrix_element: Optional[str] = None,
    background: Optional[Dict[str, float]] = None,
    rule_set=None,
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
            # background comes from the WHOLE map, never from `one`: a
            # single value is its own median, so the enrichment gate would
            # compare the cluster mean against itself and veto everything.
            (idx, float(score_phase_ratio(
                one, entry.composition, matrix_element=matrix_element,
                no_data_score=0.0, background=background)[0]))
            for idx, entry in enumerate(candidates)
        ]

        # User rules decide ELIGIBILITY, before the ranking. A blocked
        # phase scores 0 so it loses even to a badly-scoring phase that
        # is allowed - that is what "may not compete" means. The rule is
        # evaluated on the cluster MEAN here and on the pixel in the
        # other mode; same evaluator, different population, and the UI
        # has to say which.
        blocked_reasons = {}
        if rule_set is not None and not rule_set.is_empty:
            from backend.api.services.phase_rules import gate_scores
            regated = []
            for idx, sc in scored:
                rule = rule_set.rule_for(candidates[idx].key)
                gated, outcome = gate_scores(
                    np.array([sc], dtype=np.float64), rule, one,
                    background=background)
                if outcome is not None and not bool(outcome.allowed[0]):
                    blocked_reasons[idx] = outcome.reason
                regated.append((idx, float(gated[0])))
            scored = regated
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
    k_range: Tuple[int, int] = (2, 12),
    min_score: float = 0.3,
    scale: Optional[int] = None,
    rule_set=None,
) -> Tuple[np.ndarray, np.ndarray, List[ClusterMatch], int]:
    """Cluster the composition, match each cluster, paint its pixels.

    ``scale`` box-averages the composition first (default
    :data:`DEFAULT_SCALE`; pass 0 to disable and reproduce the old behaviour).
    Without it KMeans returns confetti — measured at k=8 on SampleB: 3491
    connected pieces with a median size of ONE pixel.

    With ``k=None`` the cluster count comes from
    :func:`choose_k_by_distinctness`, which raises k while the cluster means
    stay chemically distinguishable. It deliberately does NOT maximise :func:`phase_coherence`: coherence
    falls monotonically with k on unsmoothed data, so maximising it forces the
    smallest k and the map collapses however much structure the element maps
    show. Coherence is still the right measure of a map's tidiness, and
    smoothing is the knob for it — one lever per problem.

    Returns ``(phase_grid, cluster_grid, matches, k_used)``. A cluster whose
    best match scores below ``min_score`` is left unassigned (-1) and
    reported in ``matches`` with ``phase_index == -1`` — an unmatched cluster
    is visible and honest, and must never be forced onto the nearest phase.
    """
    n_px = n_rows * n_cols
    empty = np.full((n_rows, n_cols), -1, dtype=np.int32)
    if not at_pct_per_element or n_px <= 0:
        return empty, empty.copy(), [], 0

    scale_used = DEFAULT_SCALE if scale is None else int(scale)
    smoothed = _smooth_maps(at_pct_per_element, n_rows, n_cols, scale_used)

    els, X = _feature_matrix(smoothed, n_px)
    if not els:
        return empty, empty.copy(), [], 0

    matrix_element = infer_matrix_element(at_pct_per_element)
    background = background_levels(at_pct_per_element)
    group_of = np.zeros(len(candidates), dtype=np.int32)
    for gid, members in enumerate(group_degenerate_entries(candidates)):
        for m in members:
            group_of[m] = gid

    def run(kk):
        return _run_for_k(X, at_pct_per_element, n_rows, n_cols, candidates,
                          group_of, kk, min_score, matrix_element,
                          background, rule_set)

    if k:
        k_used = max(1, min(int(k), n_px))
        phase_grid, cluster_grid, matches = run(k_used)
        return phase_grid, cluster_grid, matches, k_used

    lo, hi = k_range
    lo = max(2, lo)
    hi = max(lo, min(hi, n_px))
    k_used = choose_k_by_distinctness(X, (lo, hi))
    k_used = max(1, min(k_used, n_px))
    try:
        phase_grid, cluster_grid, matches = run(k_used)
    except Exception:
        # A degenerate fit at the chosen k must not lose the map; the
        # smallest k is always fittable when there are pixels at all.
        k_used = max(1, min(lo, n_px))
        phase_grid, cluster_grid, matches = run(k_used)
    return phase_grid, cluster_grid, matches, k_used
