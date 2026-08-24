"""Seeded selection ("magic wand") on the EDS composition maps.

WHY SMOOTHING IS NOT A DETAIL. EDS acquired during an EBSD session is poor
data by construction — the accelerating voltage and beam current are chosen
for Kikuchi patterns, so the interaction volume is far larger than the
features being mapped, and the per-pixel composition carries several at% of
systematic error. Measured on `Test_data/ground_truth/G1_14.h5oina`
(805x602 = 484 610 px), clustering the RAW at% maps gives a phase coherence
of 0.368 and shatters every cluster into ~60 000 connected components with a
MEDIAN SIZE OF ONE PIXEL. A flood fill seeded anywhere in that would grow
exactly one pixel and the feature would be useless.

Smoothing the composition first fixes it:

    smoothing   coherence   largest component per cluster
    none          0.368       272 / 155 / 98 / 44 px
    3x3           0.735       136 227 / 18 741 / ...
    5x5           0.801       160 611 / 39 465 / ...

So the field this module builds is deliberately a SMOOTHED one. The cost is
resolution at the boundary; the alternative is no feature at all.

WHY THE SLIDER IS A PIXEL COUNT. Thresholding the distance directly is
unusable — the growth curve plateaus hard. Measured on a real 189 px Si/Mg
particle: threshold 0.5 % of range -> 98 154 px, and 2 % through 10 % all
return the IDENTICAL 98 154 px, then 25 % -> 280 340 px. Once the threshold
clears the cheapest bridge between feature and matrix, the map floods. The
caller therefore gets a growth curve and drives the slider by pixel count,
which is monotone and has no dead band.

Spec: docs/superpowers/specs/2026-08-24-eds-magic-wand-design.md
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from backend.api.services.chemistry_score import has_chemistry
from backend.api.services.crystal_hint_phase_fit import _CHEM_IGNORE

# Smoothing window over the composition maps. 5x5 from the table above: 3x3
# still leaves the second cluster fragmented, 7x7 buys little more coherence
# and costs boundary resolution.
DEFAULT_SMOOTH = 5

# Number of points on the growth curve. The slider interpolates between
# them, so this is the resolution of "how many pixels do I get".
GROWTH_STEPS = 64


def _smoothed_stack(
    at_pct_per_element: Dict[str, np.ndarray],
    n_rows: int,
    n_cols: int,
    smooth: int,
) -> Tuple[List[str], np.ndarray]:
    """(elements, (n_el, n_rows, n_cols) smoothed at% maps)."""
    els = sorted(el for el in at_pct_per_element if el not in _CHEM_IGNORE)
    if not els:
        return [], np.zeros((0, n_rows, n_cols))
    stack = np.stack([
        np.maximum(0.0, np.asarray(at_pct_per_element[el], dtype=np.float64)
                   ).reshape(n_rows, n_cols)
        for el in els
    ])
    if smooth and smooth > 1:
        stack = np.stack([
            ndimage.uniform_filter(layer, size=smooth, mode="nearest")
            for layer in stack
        ])
    return els, stack


def wand_field(
    at_pct_per_element: Dict[str, np.ndarray],
    n_rows: int,
    n_cols: int,
    seed_row: int,
    seed_col: int,
    smooth: int = DEFAULT_SMOOTH,
) -> Tuple[np.ndarray, float, List[Dict], Dict[str, float]]:
    """Distance-from-seed field, its scale, a growth curve and the seed mean.

    Returns ``(field_uint8, scale, growth, seed_composition)`` where the real
    distance in at% is ``field * scale``. Pixels with no measurement are
    pinned to 255 so no fill can ever reach them — the same veto
    ``_run_for_k`` applies when it classifies.
    """
    if not (0 <= seed_row < n_rows and 0 <= seed_col < n_cols):
        raise ValueError(f"seed ({seed_row},{seed_col}) outside {n_rows}x{n_cols}")

    els, stack = _smoothed_stack(at_pct_per_element, n_rows, n_cols, smooth)
    if not els:
        raise ValueError("no usable EDS elements in this dataset")

    seed_vec = stack[:, seed_row, seed_col]
    # L1 in at% — same space the classifier reasons in, and a number the
    # user can read ("within 3 at% of the seed").
    dist = np.abs(stack - seed_vec[:, None, None]).sum(axis=0)

    live = has_chemistry(at_pct_per_element).reshape(n_rows, n_cols)
    finite = dist[live] if live.any() else dist
    dmax = float(finite.max()) if finite.size else 1.0
    if dmax <= 0:
        dmax = 1.0
    scale = dmax / 254.0

    field = np.clip(np.round(dist / scale), 0, 254).astype(np.uint8)
    field[~live] = 255            # unreachable

    growth = _growth_curve(field, seed_row, seed_col)
    seed_comp = {el: float(stack[i, seed_row, seed_col]) for i, el in enumerate(els)}
    return field, scale, growth, seed_comp


def global_growth_curve(field: np.ndarray) -> List[Dict]:
    """Growth curve for "every pixel like this one", ignoring connectivity.

    A phase rarely occurs as one blob: dispersoids, precipitates and second
    phases are scattered across the scan. The connected fill needs one pass
    per particle; this selects them all at once. It is a chemistry-space
    selection, not a spatial one, so it has no seed component — just a
    threshold on the same field, which makes it a cumulative histogram.
    """
    reachable = field[field < 255]
    if reachable.size == 0:
        return []
    hist = np.bincount(reachable.ravel(), minlength=255)[:255]
    cum = np.cumsum(hist)
    qs = np.unique(np.quantile(
        reachable, np.linspace(0.0, 1.0, GROWTH_STEPS)).astype(int))
    out: List[Dict] = []
    seen = set()
    for t in qs:
        n = int(cum[int(t)])
        if n in seen:
            continue
        seen.add(n)
        out.append({"threshold": int(t), "n_pixels": n})
    return out


def _growth_curve(field: np.ndarray, seed_row: int, seed_col: int) -> List[Dict]:
    """How many pixels the fill reaches at each threshold.

    Lets the caller drive the slider by pixel count instead of by distance,
    which is what makes it usable — see the module docstring.

    Thresholds are sampled at QUANTILES of the field, not linearly. A linear
    sweep spends most of its steps where no pixels live: measured on
    SampleB, it produced 1 -> 298 -> 4662 px, so the whole span between a
    small feature and a large one had no slider position at all. Sampling
    where the pixels actually are puts the steps where the user needs them.
    """
    reachable = field[field < 255]
    if reachable.size == 0:
        return []
    qs = np.unique(np.quantile(
        reachable, np.linspace(0.0, 1.0, GROWTH_STEPS)).astype(int))

    out: List[Dict] = []
    seen = set()
    for t in qs:
        n = int(flood_from(field, seed_row, seed_col, int(t)).sum())
        if n in seen:
            continue          # collapse the plateaus
        seen.add(n)
        out.append({"threshold": int(t), "n_pixels": n})
    return out


def flood_from(
    field: np.ndarray, seed_row: int, seed_col: int, threshold: int,
) -> np.ndarray:
    """4-connected region containing the seed where ``field <= threshold``.

    Vectorised via :func:`scipy.ndimage.label` rather than a Python BFS: the
    equivalent per-pixel BFS measures 324 ms on a 485 k-px map against 1.8 ms
    here, and this runs once per growth-curve point.
    """
    ok = field <= threshold
    if not ok[seed_row, seed_col]:
        return np.zeros_like(ok)
    lab, _n = ndimage.label(ok)          # default structure = 4-connected
    return lab == lab[seed_row, seed_col]


def selection_stats(
    at_pct_per_element: Dict[str, np.ndarray],
    mask: np.ndarray,
    background: Optional[Dict[str, float]] = None,
) -> Dict:
    """Mean composition of a selection, and how enriched it is.

    Reported live so the user judges the selection on chemistry rather than
    on its shape. Enrichment is against the map's own background — the
    quantity the classifier itself gates on — so the number in the readout
    and the number in the classifier mean the same thing.
    """
    flat = mask.ravel()
    n = int(flat.sum())
    if n == 0:
        return {"n_pixels": 0, "mean_at_pct": {}, "enrichment": {}}
    mean = {el: float(np.asarray(v, dtype=float).ravel()[flat].mean())
            for el, v in at_pct_per_element.items()}
    enrich: Dict[str, float] = {}
    if background:
        total = sum(v for el, v in mean.items() if el not in _CHEM_IGNORE)
        if total > 1e-9:
            for el, v in mean.items():
                bg = background.get(el, 0.0)
                if bg > 1e-9:
                    enrich[el] = round((v / total) / bg, 2)
    return {
        "n_pixels": n,
        "mean_at_pct": {el: round(v, 2) for el, v in mean.items() if v >= 0.05},
        "enrichment": enrich,
    }
