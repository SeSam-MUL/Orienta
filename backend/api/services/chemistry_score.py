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

from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.api.services.crystal_hint_phase_fit import _CHEM_IGNORE

# Mirrors chemistry_fit's constants so the two cannot drift apart silently.
# Read their calibration history there -- it is the authoritative copy, and
# tests/test_eds_scorer_constants.py pins the two together.
#
# 2026-09-12: _ABSENT 0.012 -> 0.020 and _PRESENT_SIG 0.05 -> 0.02, because
# both are ABSOLUTE At.% floors and the k-factors were corrected that day.
# Swept on this module's own consumer as well as on the prior's:
# tasks/eds_prior_audit/10_phase_map_constants.py runs the five numbers
# tests/test_eds_phase_map_accuracy.py asserts on, over the pair, and the
# phase map does not move ON SampleB -- Fe-bad 0.611 %, particle 95.43 %,
# Si 100 %, matrix 100 %, cluster coherence 0.960, r(BC) -0.748, identical
# to three decimals at every pair tried.
#
# THAT IS ONE FILE. On the 7050 Cu-bearing scan
# (Test_data/batch_test/... Arbeitsbereich 1) the per-pixel classifier moves
# 392 of 5304 pixels, 7.4 %, and the median best score falls 0.921 -> 0.875
# (tasks/eds_prior_audit/18_mgcu2_and_7050.py). There is no reference on that
# file, so that is an UNASSESSED behaviour change, not a measured gain: the
# direction is mixed (Al7FeCu2 -> Al and Al -> Al7FeCu2 both appear) and the
# falling score is expected, since a lower unexplained floor charges every
# phase for more elements. The map is gated by the ENRICHMENT rule
# below, which asks the same question relative to the map's own
# background; these absolute floors are the scalar scorer's only gate and
# so decide the indexing prior alone (there: alpha's per-pixel top-1 rate
# 43.84 -> 89.37 %).
_MAJOR_REQ = 0.05
_ABSENT = 0.020
_PRESENT_SIG = 0.02

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
    unmeasured=None,
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
    # An element whose window could not be priced is UNMEASURED, not zero.
    # Dropping it from both sides (and renormalising the phase over what is
    # left) is what stops the gates below from ruling out precisely the
    # phases that element identifies. See chemistry_fit's `unmeasured`.
    blind = frozenset(unmeasured or ())
    q_raw = {el: max(0.0, float(v)) for el, v in (phase_at_pct or {}).items()
             if el not in _CHEM_IGNORE and el not in blind and v is not None}
    q_tot = sum(q_raw.values())
    if q_tot <= 1e-9:
        # Nothing left to compare against -- an empty phase composition, or
        # one whose every element is unmeasured. That is the SAME state as a
        # pixel with no chemistry, so it gets the same answer the caller
        # asked for: 1.0 keeps the prior's multiplier neutral, and a
        # classifier passing 0.0 gets something `min_score` can filter. A
        # flat 1.0 here is not neutral in an argmax, it is a win.
        return np.full(n_px, float(no_data_score), dtype=np.float32)
    q = {el: v / q_tot for el, v in q_raw.items()}

    els, p, total = _renormalise(
        {el: v for el, v in at_pct_per_element.items() if el not in blind}
        if blind else at_pct_per_element)
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


# ---------------------------------------------------------------------------
# Ratio matching
#
# Spec: docs/superpowers/specs/2026-08-20-eds-ratio-matching-design.md
#
# WHY THIS EXISTS. EDS acquired during an EBSD session is bad data by
# construction: the accelerating voltage and beam current are chosen for
# Kikuchi patterns, not for X-rays, so the interaction volume is large
# compared with the features being mapped. A sub-micron particle is measured
# together with the matrix around it, and the measured composition is a
# mixture, not the phase.
#
# The saving grace is that BOTH dominant errors are multiplicative on the
# non-matrix elements:
#   - interaction-volume dilution scales every non-matrix element by the
#     same fill fraction f,
#   - standardless Cliff-Lorimer scales each element by its own roughly
#     constant sensitivity factor.
# A ratio between two non-matrix elements therefore survives what the
# absolute values do not. Measured on 7050 area 1 across dilution bands:
# absolute Cu varies 6x, Mg/Cu varies 2.4x.
#
# It is NOT invariant, and this module does not pretend otherwise — 2.4x is
# real drift, worst on weak signal where the ratio runs toward the
# background's ratio. Ratios are the better instrument here, not a cure.
# ---------------------------------------------------------------------------

# A non-matrix element below this renormalised fraction is treated as not
# measured for ratio purposes: at trace level the log-ratio is dominated by
# background and counting noise, which is exactly where the drift above
# comes from. Elements below it do not veto; they simply do not vote.
_RATIO_FLOOR = 0.005

# Detection-level presence floor for the ABSOLUTE gate that survives from
# the old scorer. It answers "is this element here at all", which the
# multiplicative errors barely touch — unlike "is it here in stoichiometric
# quantity", which they make unanswerable per pixel.
_PRESENCE_FLOOR = _ABSENT

# How far above the map's OWN background a defining element must sit before
# a phase requiring it can be assigned. Replaces the relative-to-nominal
# veto, which could not be calibrated across datasets because it compared a
# measured value against a stoichiometry the quantification never reproduces.
#
# This bar is MEASURED per dataset, so it carries no k-factor assumption:
# it asks "is this element enriched HERE, relative to the rest of THIS map",
# which is the question that actually separates a particle from the matrix.
# Measured enrichment of real features over their map's median:
#   SampleB  Fe 2.7x   Si 4.4x
#   7050_R   Cu 8.4x   Mg 4.0x
#
# This is a RATIO, so it looks scale-free -- but it is not quite. The
# standardless k-factors are element-wise, and every pixel is renormalised to
# 100 %, so a particle/background ratio moves when they change. It did:
# correcting them on 2026-09-12 took SampleB's particle-over-background Fe
# from 186.1 to 173.1 and moved this whole sweep down by about 0.12.
#
# Calibrated on SampleB 2026-08-20, RE-MEASURED 2026-09-12 on the corrected
# at% scale (tasks/eds_quant_audit/08_enrichment_sweep.py). Columns: share of
# pixels holding an Fe-phase with <25 % of its required Fe, and share of the
# ground-truth Fe particle still assigned an Fe-bearing phase.
#
# The ground truth is now the phase's nominal Fe (11.6 at%), not the old
# 4 at% literal: on the corrected scale 4 at% no longer selects the particle
# but the particle plus its interaction-volume halo (6941 px against 3830).
#
#   enrich   Fe-bad   particle          2026-08-20 (old scale, Fe > 4 at%)
#   1.00     0.05 %   100.0 %
#   1.05     0.00 %   100.0 %   band starts
#   1.15     0.00 %   100.0 %   <-- chosen, middle of the band
#   1.20     0.00 %   100.0 %           1.20   1.74 %   98.8 %
#   1.25     0.00 %   100.0 %           1.25   0.76 %   98.9 %
#   1.30     0.00 %   100.0 %   band ends   1.30   0.27 %   99.0 %  <-- was
#   1.325    0.00 %    98.6 %           1.35   0.13 %   99.1 %
#   1.35     0.00 %    95.4 %           1.40   0.11 %   99.3 %
#   1.40     0.00 %    86.9 %           1.45   0.06 %   95.4 %  band ended
#   1.45     0.00 %    79.9 %           1.50   0.04 %   88.9 %
#   1.55     0.00 %    65.8 %           1.55   0.00 %   81.6 %
#
# Two things changed. The Fe-bad column is now 0.00 % everywhere above 1.00 --
# the corrected chemistry alone removes the mis-assignment this constant was
# introduced to fight. And the safe band moved from 1.20..1.45 to 1.05..1.30,
# which put the old 1.30 exactly on the cliff the comment below warns about:
# at 1.35 it already costs 4.6 points of particle retention and at 1.40, 13.
#
# 1.15 sits in the middle of the safe band rather than at the best single
# value -- the same relative position inside the band that 1.30 held in the
# old one. The predecessor constant was placed at exactly such an edge and had
# to be corrected; 1.30 had drifted onto one again without moving.
#
# Bias is deliberately low rather than high: at 1.00 the cost is 0.05 % of
# pixels mis-assigned, at 1.40 it is 13 points of a real particle eaten.
#
# If the quantification scale moves again, re-run script 08.
_ENRICHMENT = 1.15


def background_levels(
    at_pct_per_element: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Per-element background: the median over pixels that have chemistry.

    The matrix of a real map is not clean — SampleB reads a median 2.80 at%
    Fe everywhere against an equilibrium solubility of ~0.05 at%, because
    the interaction volume drags signal in from the particles around it.
    That level is what a phase has to beat to be called present.
    """
    live = has_chemistry(at_pct_per_element)
    out: Dict[str, float] = {}
    _els, p, _total = _renormalise(at_pct_per_element)
    for i, el in enumerate(_els):
        vals = p[i][live] if live.any() else p[i]
        out[el] = float(np.median(vals)) if vals.size else 0.0
    return out


def infer_matrix_element(at_pct_per_element: Dict[str, np.ndarray]) -> Optional[str]:
    """The element that dominates the map — the solvent, e.g. Al.

    Ratio matching is defined on the NON-matrix elements, so this choice
    inverts the whole metric if it is wrong. It is inferred rather than
    assumed, and callers are expected to surface it: a Mg- or Fe-based
    sample must not silently be scored as if it were an aluminium alloy.
    """
    els = [el for el in at_pct_per_element if el not in _CHEM_IGNORE]
    if not els:
        return None
    means = {el: float(np.nanmean(np.asarray(at_pct_per_element[el], dtype=np.float64)))
             for el in els}
    best = max(means, key=means.get)
    # A "matrix" that is not actually dominant is not a matrix. Below this
    # the sample is a mixture and ratio matching should use every element.
    return best if means[best] >= 40.0 else None


def score_phase_ratio(
    at_pct_per_element: Dict[str, np.ndarray],
    phase_at_pct: Dict[str, float],
    matrix_element: Optional[str] = None,
    no_data_score: float = 1.0,
    background: Optional[Dict[str, float]] = None,
    unmeasured=None,
) -> np.ndarray:
    """Score one phase against every pixel using non-matrix element ratios.

    Replaces the relative missing-major veto, which could not be calibrated:
    it demanded a defining element reach a FRACTION OF ITS NOMINAL VALUE,
    which dilution makes impossible for a thin feature. `Al7FeCu2` (20 at%
    Cu nominal) scored the veto floor on every pixel of both Cu-bearing test
    files under that rule.

    The gate that remains is absolute: a phase whose defining element is
    ABSENT is still rejected. What is dropped is the demand that it be
    present in stoichiometric quantity.

    Phases with fewer than two non-matrix elements have no ratio — `Al.cif`
    has none at all, `Al6Fe` has one. They fall back to the renormalised-L1
    of :func:`score_phase_vectorised`. That is not a corner case: it covers
    the matrix phase, which is most of a typical map.
    """
    if not at_pct_per_element:
        return np.zeros(0, dtype=np.float32)
    n_px = len(next(iter(at_pct_per_element.values())))

    # An element whose window could not be priced is UNMEASURED, not zero.
    # Dropping it from both sides (and renormalising the phase over what is
    # left) is what stops the gates below from ruling out precisely the
    # phases that element identifies. See chemistry_fit's `unmeasured`.
    blind = frozenset(unmeasured or ())
    q_raw = {el: max(0.0, float(v)) for el, v in (phase_at_pct or {}).items()
             if el not in _CHEM_IGNORE and el not in blind and v is not None}
    q_tot = sum(q_raw.values())
    if q_tot <= 1e-9:
        # See score_phase_vectorised: neutral for the prior, filterable for
        # a classifier, never a free win.
        return np.full(n_px, float(no_data_score), dtype=np.float32)
    q = {el: v / q_tot for el, v in q_raw.items()}

    els, p, total = _renormalise(
        {el: v for el, v in at_pct_per_element.items() if el not in blind}
        if blind else at_pct_per_element)
    if not els:
        return np.ones(n_px, dtype=np.float32)
    p_of = {el: p[i] for i, el in enumerate(els)}
    zero = np.zeros(n_px, dtype=np.float64)

    # L1 baseline — also the fallback for phases without a usable ratio.
    l1 = zero.copy()
    for el in set(p_of) | set(q):
        l1 += np.abs(p_of.get(el, zero) - q.get(el, 0.0))
    l1_score = np.clip(1.0 - 0.5 * l1, 0.0, 1.0)

    # Two gates, both on the DEFINING elements of the phase.
    #
    #   1. absolute presence  — is this element here at all
    #   2. enrichment         — is it enriched over this map's own background
    #
    # The matrix element is exempt from (2) by definition: it IS the
    # background, so it can never be enriched over itself, and gating on it
    # would veto the matrix phase across the entire map.
    # Enrichment needs a POPULATION to be measured against. A caller
    # scoring a single pixel (or a single cluster mean) cannot supply one
    # from its own input — the median of one value is that value, so the
    # gate would compare a number against 1.3x itself and veto every phase.
    # Such callers must pass the map-level background explicitly; when they
    # do not, the enrichment gate is skipped rather than applied to
    # nonsense. The absolute presence gate still applies.
    #
    # REFUTED, 2026-09-12 -- do not retry without a new argument. The obvious
    # repair for the 2026-08-25 Mg2Si report (a phase refused on every pixel
    # of a map made of it, characterised in
    # tests/test_eds_suggest_no_match_reason.py) is 'a bar the map's own
    # MAXIMUM cannot reach is not a bar, so stand the gate down'. It separates
    # the Mg2Si map from both real datasets cleanly
    # (tasks/eds_prior_audit/15_vacuous_gate.py: 0 pixels clear either bar
    # there, 18 575 to 25 031 clear every bar here) -- and it is still wrong,
    # because 'this map has no contrast in that element' covers TWO opposite
    # situations. The other one is
    # test_enrichment_gate_rejects_background_level_signal below: an aluminium
    # matrix carrying a flat Mg 8 / Cu 2 at% of solute has no contrast either,
    # its Mg:Cu ratio is exactly MgCu's, and with the gate stood down it reads
    # as MgCu at 0.999 on every pixel -- the precise failure this gate exists
    # to prevent. Telling the two apart needs to compare the map's BULK
    # composition with the phase, and that does not separate them either:
    # Mg2Si against its map scores 0.967 and alpha-Al(Fe,Mn)Si against
    # SampleB's bulk scores 0.938, so any threshold that admits the first
    # admits the second.
    if background is None:
        background = (background_levels(at_pct_per_element)
                      if n_px > 1 else {})
    veto = np.zeros(n_px, dtype=bool)
    for el, frac in q.items():
        if frac < _MAJOR_REQ:
            continue
        measured = p_of.get(el, zero)
        veto |= (measured < _PRESENCE_FLOOR)
        if el != matrix_element:
            bar = _ENRICHMENT * background.get(el, 0.0)
            if bar > 0.0:
                veto |= (measured < bar)

    # Ratio term over the phase's non-matrix elements.
    nm = [el for el in q if el != matrix_element and q[el] > 0.0]
    score = l1_score
    if len(nm) >= 2:
        sq_sum = zero.copy()
        n_pairs = zero.copy()
        for i in range(len(nm)):
            for j in range(i + 1, len(nm)):
                a, b = nm[i], nm[j]
                pa = p_of.get(a, zero)
                pb = p_of.get(b, zero)
                usable = (pa >= _RATIO_FLOOR) & (pb >= _RATIO_FLOOR)
                if not usable.any():
                    continue
                # Only evaluate where both are above the floor; log of a
                # trace value is noise, not evidence.
                safe_a = np.where(usable, pa, 1.0)
                safe_b = np.where(usable, pb, 1.0)
                d = np.abs(np.log(safe_a / safe_b) - np.log(q[a] / q[b]))
                sq_sum += np.where(usable, d * d, 0.0)
                n_pairs += usable
        has_ratio = n_pairs > 0
        rms = np.sqrt(sq_sum / np.where(has_ratio, n_pairs, 1.0))
        ratio_score = np.exp(-rms)

        # A phase element the pixel does not show AT ALL still counts
        # against the phase — the ratio term alone cannot see it, because
        # an absent element contributes no pair.
        missing = zero.copy()
        for el in nm:
            missing += (p_of.get(el, zero) < _RATIO_FLOOR).astype(np.float64)
        ratio_score = ratio_score * np.power(0.5, missing)

        # Take the BETTER of the two, not the ratio alone.
        #
        # Measured on the SampleB Fe-Mn-Si particle core, ratio of means
        # against nominal: Fe/Si 0.70x, Fe/Mn 0.83x, Si/Mn 1.18x, while the
        # absolute Fe is 0.66x. Ratios are the better instrument — but the
        # Cliff-Lorimer bias is per ELEMENT, so ratios containing Fe do not
        # cancel cleanly and the pure-dilution argument overstates its case.
        #
        # The two metrics fail in different places: L1 fails when dilution
        # is severe (a thin Cu ring), ratios fail when the per-element bias
        # is large (Fe against Si). A phase that fits EITHER its absolute
        # composition OR its internal ratios is a candidate worth ranking;
        # requiring both would reject the true phase in both regimes. The
        # absolute presence gate below still has to pass either way, so this
        # is more permissive about STOICHIOMETRY, never about whether the
        # defining element is there at all.
        score = np.where(has_ratio, np.maximum(ratio_score, l1_score), l1_score)

    # Unexplained-element penalty — but NOT for the matrix element.
    #
    # This is the whole dilution correction, and it is one line. A particle
    # smaller than the interaction volume is always measured together with
    # the matrix around it: the Mg-Cu phase below reads Al 71.6 at% because
    # the birne sampled the aluminium around it, not because the phase
    # contains aluminium. Charging the phase for that Al is charging it for
    # the measurement geometry, and it is what kept the correct phase at
    # 0.259 while a phase that simply "explains" the Al won at 0.599.
    #
    # A non-matrix element the phase cannot account for still counts fully:
    # that one really is evidence against the phase.
    unexplained = zero.copy()
    for el, arr in p_of.items():
        if el == matrix_element:
            continue
        if q.get(el, 0.0) < _ABSENT:
            unexplained += np.where(arr >= _PRESENT_SIG, arr, 0.0)
    score = score * np.maximum(0.0, 1.0 - unexplained)

    out = np.where(veto, np.minimum(score, 0.05), score)
    usable_px = np.isfinite(total) & (total > 1e-9)
    out = np.where(usable_px, out, float(no_data_score))
    return np.clip(np.nan_to_num(out, nan=float(no_data_score)),
                   0.0, 1.0).astype(np.float32)
