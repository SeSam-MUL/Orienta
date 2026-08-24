"""Vectorised chemistry matching for the EDS phase map.

:func:`crystal_hint_phase_fit.chemistry_fit` is the authoritative scalar
scorer and is used by the indexing chemistry prior, where its constants were
calibrated on ProbeB. This module is its vectorised twin for the batch
classifier (~22 phases x 50k pixels), plus one addition the batch case needs:
a RELATIVE missing-major veto.

Why the relative veto: ``chemistry_fit``'s ``_ABSENT`` is an absolute floor
(1.2 at%). A phase requiring 11.6 at% Fe is therefore not vetoed on a pixel
measuring 4 at% Fe, because 4 > 1.2 -- so an aluminium matrix carrying a few
at% of background Fe is happily labelled an Fe-Mn-Si intermetallic. Measured
on SampleB that is 20.95 % of all pixels. The relative veto asks instead
whether the measured amount is a plausible fraction of what the phase
REQUIRES.

``rel_req=0.0`` reproduces ``chemistry_fit`` exactly, so this module can be
diffed against it in tests.

Spec: docs/superpowers/specs/2026-08-19-eds-chemistry-phase-map-design.md
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from backend.api.services.crystal_hint_phase_fit import _CHEM_IGNORE

# Mirrors chemistry_fit's constants so the two cannot drift apart silently.
# See crystal_hint_phase_fit.py:575-620 for their calibration history.
_MAJOR_REQ = 0.05
_ABSENT = 0.012
_PRESENT_SIG = 0.05

# Default relative requirement for a defining element: a phase is vetoed
# where a defining element is measured below this fraction of its nominal
# value. Kept well below 1.0 because standardless Cliff-Lorimer under-reads
# heavy elements.
#
# THE CEILING IS THE QUANTIFICATION, NOT THE CHEMISTRY. Standardless
# Cliff-Lorimer under-reads heavy elements badly: on SampleB the Fe-Mn-Si
# particle (ground truth: the connected Fe>4 at% region, 3401 px) measures a
# mean 5.33 at% Fe where sd_0302719.cif nominally requires 11.6 — a factor
# of 0.46. rel_req must therefore sit well BELOW ~0.46, or the veto starts
# eating real particles.
#
# Calibrated on SampleB (10 800 px, 22 candidates) 2026-08-19. Columns:
# share of pixels holding an Fe-phase with <25 % of its required Fe; share
# of the ground-truth particle still assigned an Fe-bearing phase; share of
# Si>60 at% particles labelled Si.cif; share of Al>=95 at% matrix as Al.cif:
#
#   rel_req   Fe-bad    particle    Si>60    matrix   note
#   0.00      12.59 %     99.6 %     100 %    100 %   plain chemistry_fit
#   0.15      11.09 %     99.6 %     100 %    100 %
#   0.20       7.16 %     99.6 %     100 %    100 %   still under-vetoing
#   0.25       0.70 %     99.6 %     100 %    100 %
#   0.30       0.00 %     99.4 %     100 %    100 %   <-- chosen
#   0.35       0.00 %     97.8 %     100 %    100 %   particle starts eroding
#   0.40       0.00 %     71.5 %     100 %    100 %   28 % of the particle lost
#
# 0.30 is the smallest value that reaches Fe-bad 0.00 % while still
# recovering 99.4 % of the real particle. An earlier pass chose 0.40 from
# per-pixel metrics alone; the visual check caught that it silently vetoed
# a quarter of the particle, because 0.40 sits right on the 0.46 under-read.
#
# KNOWN LIMIT — THIS IS A SampleB CONSTANT, NOT A UNIVERSAL ONE. A sweep
# across all 11 h5oina areas in Test_data (2026-08-19), adjudicated by the
# band-contrast drop of each assigned region, found the requirement is
# contradictory between datasets:
#   SampleB   real Fe-Mn-Si particle, ratio 0.330 -> needs rel_req <= 0.30
#   HIgh_MG   FALSE Mg17Al12 on solute-rich matrix, ratio 0.346,
#             band-contrast drop only 0.5 -> needs rel_req >= 0.40
# No single global value satisfies both. The under-read depends on the
# element, the accelerating voltage and the k-factors, so the honest fix is
# a per-dataset (or per-element) calibration, not a better global default.
# Until then 0.30 is tuned for the Al-Fe-Mn-Si case and will over-report
# dilute solute enrichment as a phase on Al-Mg data.
#
# Secondary bias, same sweep: the veto floor scales with the phase's own
# nominal content, so it eliminates Fe-rich candidates before Fe-poor ones.
# On SampleB it leaves sd_0302719 (the library's most dilute Fe entry at
# 11.6 at%) winning partly because its rivals are vetoed, not purely because
# it fits best. Reducing near-degenerate library entries matters more than
# tuning this number.
DEFAULT_REL_REQ = 0.3


def _renormalise(
    at_pct_per_element: Dict[str, np.ndarray],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Drop C/O, clamp negatives, renormalise each pixel to fractions.

    Returns ``(elements, fractions (n_el, n_px), raw_total (n_px,))``.
    """
    els = sorted(el for el in at_pct_per_element if el not in _CHEM_IGNORE)
    if not els:
        n_px = len(next(iter(at_pct_per_element.values())))
        return [], np.zeros((0, n_px)), np.zeros(n_px)
    stack = np.stack(
        [np.maximum(0.0, np.asarray(at_pct_per_element[el], dtype=np.float64))
         for el in els],
        axis=0,
    )
    total = stack.sum(axis=0)
    return els, stack / np.where(total > 1e-9, total, 1.0), total


def has_chemistry(at_pct_per_element: Dict[str, np.ndarray]) -> np.ndarray:
    """Bool mask: pixels that actually carry a usable EDS measurement.

    A pixel with no counts at all — dead detector channel, or a region of
    the frame the map never covered — has no chemistry to match against.
    Callers that CLASSIFY must treat those as unclassified; see
    ``no_data_score`` on :func:`score_phase_vectorised`.
    """
    if not at_pct_per_element:
        return np.zeros(0, dtype=bool)
    _els, _p, total = _renormalise(at_pct_per_element)
    return total > 1e-9


def score_phase_vectorised(
    at_pct_per_element: Dict[str, np.ndarray],
    phase_at_pct: Dict[str, float],
    rel_req: float = 0.0,
    no_data_score: float = 1.0,
) -> np.ndarray:
    """Score one phase against every pixel. Returns float32 in [0, 1].

    Mirrors :func:`chemistry_fit` step for step, including its early return
    on the missing-major veto (a vetoed pixel does NOT additionally receive
    the present-but-missing penalty).

    ``no_data_score`` is what a pixel with no usable chemistry scores.
    ``chemistry_fit`` returns a NEUTRAL 1.0, which is right for the indexing
    prior — that score is a multiplier, so 1.0 means "this says nothing,
    defer to the pattern evidence". It is catastrophic for a CLASSIFIER:
    1.0 is the top of the range, so every candidate ties at the maximum and
    ``argmax`` hands the pixel to whichever phase sits in row 1 of the
    spreadsheet, reported as a perfect match that ``min_score`` cannot
    filter. Measured on `Test_data/70502_t2 … Arbeitsbereich 1`, that
    fabricated a confident "Al" across 63.7 % of a map that is simply
    unmeasured. Classifiers pass ``no_data_score=0.0``; the default stays
    1.0 so parity with ``chemistry_fit`` is preserved.
    """
    if not at_pct_per_element:
        return np.zeros(0, dtype=np.float32)
    n_px = len(next(iter(at_pct_per_element.values())))

    # Clean the phase side exactly as chemistry_fit._clean does.
    q_raw = {el: max(0.0, float(v)) for el, v in (phase_at_pct or {}).items()
             if el not in _CHEM_IGNORE and v is not None}
    q_tot = sum(q_raw.values())
    if q_tot <= 1e-9:
        return np.ones(n_px, dtype=np.float32)  # fail-soft, as chemistry_fit
    q = {el: v / q_tot for el, v in q_raw.items()}

    els, p, total = _renormalise(at_pct_per_element)
    if not els:
        return np.ones(n_px, dtype=np.float32)
    p_of = {el: p[i] for i, el in enumerate(els)}
    zero = np.zeros(n_px, dtype=np.float64)

    # score = 1 - 0.5 * L1 over the union of both element sets.
    l1 = zero.copy()
    for el in set(p_of) | set(q):
        l1 += np.abs(p_of.get(el, zero) - q.get(el, 0.0))
    score = np.clip(1.0 - 0.5 * l1, 0.0, 1.0)

    # Missing-major veto. chemistry_fit returns immediately here, so the
    # present-but-missing penalty below must not touch these pixels.
    veto = np.zeros(n_px, dtype=bool)
    for el, frac in q.items():
        if frac >= _MAJOR_REQ:
            measured = p_of.get(el, zero)
            veto |= (measured < _ABSENT)
            if rel_req > 0.0:
                veto |= (measured < rel_req * frac)

    # Present-but-missing penalty: damp by the fraction of significant
    # measured composition the phase leaves unexplained.
    unexplained = zero.copy()
    for el, arr in p_of.items():
        if q.get(el, 0.0) < _ABSENT:
            unexplained += np.where(arr >= _PRESENT_SIG, arr, 0.0)
    penalised = score * np.maximum(0.0, 1.0 - unexplained)

    out = np.where(veto, np.minimum(score, 0.05), penalised)

    # A pixel with no usable chemistry. NaN counts as no data too: NumPy's
    # maximum propagates NaN where Python's max(0.0, nan) returns 0.0, so
    # without this a single NaN channel would poison the total and land
    # here anyway — make that explicit rather than incidental.
    usable = np.isfinite(total) & (total > 1e-9)
    out = np.where(usable, out, float(no_data_score))
    return np.clip(np.nan_to_num(out, nan=float(no_data_score)),
                   0.0, 1.0).astype(np.float32)
