"""Decide the phase per orientation grain instead of per pixel.

The chemistry is aggregated over the grain's ERODED interior, because the
outermost ring is where the EDS interaction volume mixes two phases and where
the residual registration offset bites. A grain whose interior chemistry is not
unimodal is reported ``ambiguous`` and left alone: for a tool people take
measurements from, showing the doubt beats guessing.

Six failures were designed out of this module, and every one of them would
have produced a confident wrong answer rather than an error:

* Measuring the chemical spread over the FULL grain while aggregating over the
  eroded interior. The rim is mixed by construction -- that is why the
  aggregation erodes -- so the spread would have measured the rim blur and
  called every real particle inhomogeneous. Measured on the 5x5 particle in the
  tests: 44.5 at% over the full grain against 0.0 over its interior, i.e. every
  grain on a real map would come back ``ambiguous`` and nothing would ever be
  reassigned. The mask is now chosen in ONE place, ``_interior``, which both
  the aggregation and the spread go through.

* ``np.bincount`` over the raw phase ids. Real maps mark not-indexed pixels
  with -1 (the orix convention) and ``bincount`` raises on negative input. The
  tempting repairs are both wrong: clipping invents a phase for a pixel that
  has none, and counting -1 as a phase lets "not indexed" win a majority vote.
  Unindexed pixels are excluded from the vote, and a grain with no indexed
  pixel at all keeps ``-1`` -- it has no diffraction evidence, so it gets no
  phase.

* ``max(fits, key=...)`` returns the FIRST maximum, so two phases with the same
  nominal composition were separated by whichever ``expected_by_phase`` happened
  to list first. That is an arbitrary choice presented as a determination, on
  exactly the degenerate-phase case this step exists to arbitrate -- and it is
  not a corner case: duplicated database entries tie exactly, and so does every
  pair of candidates once ``chemistry_fit``'s missing-major veto floors them.
  A tie among strangers is now ``ambiguous``, naming every tied phase. A tie
  that CONTAINS the grain's incumbent phase goes to the incumbent -- not
  because it was listed first, but because the diffraction pattern already
  chose it, which is the same independent tiebreak the render gate applies one
  task later. Calling that ``ambiguous`` instead would let one duplicated
  database entry switch the feature off for every grain of that phase, since
  ``ambiguous`` forfeits the rim repair.

* Deciding on "does the grain's MAJORITY phase change?". The damage being
  repaired is a wrong rim around a correctly-labelled core, and on any particle
  larger than about 5x5 the core outvotes the rim -- so the majority rule left
  every large particle shelled while fixing only the small ones. The decision
  is about the whole grain: it is ``reassign`` whenever any indexed pixel in the
  grain disagrees with the proposal, which is why ``proposed_phase`` can equal
  ``current_phase`` on a reassignment.

* Deciding a grain whose eroded interior is one pixel. ``p90 - p10`` over one
  pixel is identically 0.0, so the uniformity gate could never fire for it
  while the response reported a spread of 0.0 as though it had measured
  perfect homogeneity -- and a 3x3 grain, which erodes to exactly one pixel,
  is above the default ``min_px``. Falling back to the whole grain instead is
  no better and was measured to be worse: on a 3x3 particle eight of the nine
  pixels are interface mixture and their median proposes the MATRIX phase, so
  the fallback re-creates the shelling defect it was meant to cover for. Such
  a grain is now ``ambiguous``, with the whole-grain spread reported as the
  real number it is.

* Gating a rim repair by scoring the grain under ``proposed_phase`` against
  the grain under ``current_phase``. Those are the SAME phase on a rim repair,
  so that comparison is a render against itself: the margin can never be met,
  every rim repair is downgraded to ``keep``, and the refusal reads
  ``proposed 0.200 vs current 0.200`` -- one number printed twice and
  presented as a measurement. The whole point of the module would have come
  back as a safety gate. ``verify_decisions`` therefore has TWO gates and asks
  the disputed pixels themselves; see its docstring.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from backend.api.services.crystal_hint_phase_fit import chemistry_fit

#: Phase id meaning "this pixel was not indexed" (the orix convention). Any
#: negative id is read the same way: there is no orientation here, so there is
#: no phase to keep, change or vote with.
_NOT_INDEXED = -1

#: One pixel of erosion, 8-connected: the interior is what remains after
#: dropping every pixel that touches something outside the grain, which is
#: exactly the set whose EDS signal is not mixed with a neighbour.
_ERODE_STRUCTURE = np.ones((3, 3), bool)

#: Two phases whose ``chemistry_fit`` scores are within this of each other are
#: not separated by the chemistry.
#:
#: THIS CONSTANT IS EMPIRICAL. Do NOT try to re-derive it from the formula.
#: ``chemistry_fit`` looks like ``1 - 0.5 * L1``, which would make a 1 at%
#: change in a phase's nominal composition worth at most 0.01 -- but that
#: expression is only its first half. The missing-major veto
#: (``crystal_hint_phase_fit.py``: ``if frac >= 0.05 and pixel[el] < 0.012:
#: return min(score, 0.05)``) is a CLIFF, and the present-but-missing penalty
#: after it is multiplicative, so the map is not Lipschitz anywhere near those
#: thresholds. Measured on an ordinary alpha-Al(Fe,Mn)Si-like composition --
#: every number below is reproducible from the compositions as written:
#:
#:   pixel     {Al 90.0, Si  5.0, Fe 0.5, Mn 4.5}
#:   candidate {Al 80.0, Si 10.0, Fe 4.6, Mn 5.4}  -> 0.900
#:   candidate {Al 80.0, Si 10.0, Fe 5.6, Mn 4.4}  -> 0.050
#:
#: One at% moved between two minor elements, |dfit| 0.85. A sweep confirms it
#: is not a hand-picked pair: drawing pixel and candidate over
#: [Al, Si, Fe, Mn, Cu, Mg] from Dirichlet(alpha=0.4) x 100 with
#: numpy default_rng(0), then moving 1.0 at% between two distinct elements
#: (15453 usable draws of 20000), the worst |dfit| was 0.8648 -- 86x what the
#: analytic argument predicts.
#:
#: What justifies 0.01 is measurement plus quantification error. On an
#: Al 93 / Si 1 at% pixel: pure Al scores 0.98936 against Al99.7Si0.3 at
#: 0.99236 -- a gap of 0.0030, which must read as a tie; against Al96Si4 at
#: 0.97064 -- a gap of 0.0187, which must decide. 0.01 sits 3.3x above the
#: first and 1.9x below the second, and both are pinned by tests. Below that
#: gap the candidates are closer together than the standardless Cliff-Lorimer
#: quantification feeding them can resolve -- it is documented in
#: ``crystal_hint_phase_fit.py`` as under-reading heavy elements by roughly a
#: factor of two.
#:
#: Exact ties are common rather than exotic, which is the argument for the rule
#: rather than for the number: duplicated database entries with the same
#: formula score identically (this project has three alpha-AlFeSi entries), and
#: so do all candidates once the veto floors them -- a Cu-rich pixel scores
#: 0.005494505 against BOTH Al and Si.
#:
#: Public, with ``tied_phases`` beside it, because a tie is a fact a caller
#: needs and prose in ``reason`` is not a signal. A grain whose pattern phase
#: was wrong AND whose candidates tie now reports ``keep`` where it once
#: reported ``ambiguous``; ``chem_fit`` still carries every fit, so the tie is
#: recoverable -- but only against this number, and a caller should not have
#: to hardcode it or grep the reason string to get it.
TIE_TOLERANCE = 0.01

#: How much better the forward render must be before a chemistry-driven change
#: is adopted. Same value, and the same job, as the map-wide variant unifier's
#: "clear margin" (``backend/spherical_gpu/pipeline/variant_unification.py``,
#: ``margin_clear: float = 0.03``), which arbitrates the same question one
#: level up: a score that is only marginally better has not established
#: anything, so the incumbent -- which the diffraction pattern chose -- stands.
#:
#: This is a ONE-SIDED threshold and deliberately so. It is not a claim that
#: differences below 0.03 are noise; it is a refusal to overturn the pattern on
#: a difference that small. Nothing here ever adopts a change the render scores
#: WORSE, at any margin.
#:
#: WHAT THE PRECEDENT DOES AND DOES NOT ESTABLISH. It is tempting to write that
#: 0.03 was calibrated on whole-render comparisons and therefore may not
#: transfer to a handful of pixels. That is FALSE, and checked: the unifier's
#: ``score_fn`` returns a PER-PIXEL render-NCC and its comparison is a
#: ``np.median`` over a sample of ``ns = min(score_pixels_max, max(3,
#: upix.size // 10), upix.size)`` pixels (``variant_unification.py`` around
#: lines 382-390), i.e. typically 3-8. So 0.03 already governs a small-sample
#: per-pixel render-NCC statistic and transfers here better than a first
#: reading suggests.
#:
#: What is genuinely unestablished is narrower, and is the part to check on
#: real data: BOTH ``margin_clear`` sites take a MEDIAN, while this gate takes
#: a pixel-weighted MEAN, and this gate has no minimum group size at all. A
#: one-pixel group is reachable here and casts a full, unrobust vote, where a
#: median of three would have absorbed the same bad sample.
#:
#: Two things about that precedent that are easy to get wrong, both measured
#: rather than argued. (1) Neither ``ns`` EXPRESSION guarantees three samples --
#: both take ``min(..., unit_size)``, so the ``max(3, ...)`` at line 382 is not
#: a floor on the result -- but neither SITE is ever handed a unit that small.
#: Line 492's branch rejects ``pix.size <= rescue_max_px`` (``rescue_max_px =
#: 2``); line 382's rejects the same and also ``pix.size < min_grain_px`` (3),
#: and ``plan_grain_units`` emits either a whole grain or components of at
#: least ``domain_min_px`` (8). Measured over grains of 3-39 px with 400 random
#: class labelings, the smallest unit reaching EITHER site is 3 px. So both
#: precedent sites do median at least three samples in practice; this gate
#: guarantees nothing, which makes the contrast above stronger, not weaker.
#: (2) The second site (the adoption branch, lines 492-504) is NOT the less
#: robust of the two. Swept over unit sizes 1-399 it never samples FEWER than
#: the first and samples strictly more for 4 <= n <= 79 (at n = 30, eight
#: samples against three), because line 382's ``max(3, upix.size // 10)`` only
#: lifts a term that is 0 for small units while ``upix.size`` is the binding
#: cap. Both production callers (``backend/api/routes/indexing.py:3528`` and
#: ``variant_unification.py:747``) pass the defaults these figures assume.
RENDER_ADOPT_MARGIN = 0.03

#: An eroded interior smaller than this cannot be tested for uniformity, so
#: ``decide_grains`` refuses to decide such a grain and reports it
#: ``ambiguous``.
#:
#: This gate belongs to the SPREAD and to nothing else. It is not a statement
#: that a small interior's median is bad -- measured on the 3x3 particle in the
#: tests, the median of the single interior pixel proposes the particle
#: (fits Al 0.007 / Si 0.843) while the median of the whole grain proposes the
#: MATRIX (0.277 / 0.225), because eight of those nine pixels are interface
#: mixture. One real interior pixel beats eight rim pixels. So
#: ``aggregate_grain_chemistry`` keeps using any non-empty interior, and only
#: the uniformity test is gated here. Routing the aggregator through this gate
#: made it hand out the very answer ``decide_grains`` refuses.
#:
#: ``p90 - p10`` over ONE pixel is identically 0.0 whatever the chemistry, so
#: the uniformity gate could never fire for such a grain while the response
#: reported ``si_spread = 0.0`` as though it had measured perfect homogeneity.
#: That is not rare: a 3x3 grain is 9 px, above the default ``min_px`` of 5,
#: and erodes to exactly one pixel -- and that pixel is the mixed signal the
#: erosion exists to avoid.
#:
#: Measured, p90-p10 of N samples from N(0, sigma=12), median over 4000 trials:
#:
#:   N        1      2      3      4      5      9     16     25     49
#:   spread   0.00   9.35  15.18  18.83  20.96  24.63  27.12  28.32  29.42
#:   reads 0  100%     0%     0%     0%     0%     0%     0%     0%     0%
#:
#: Only N=1 is structurally dead; the honest reading of the rest of that row is
#: that the statistic is biased LOW at every small N (0.30, 0.49, 0.61 and 0.68
#: of the large-sample value at N = 2, 3, 4, 5). So this constant removes the
#: dead case and buys a little margin -- it does NOT make a 4-pixel spread
#: trustworthy, which is what the note in the reason is for. 4 is the interior
#: of a 4x4 grain, the smallest square whose erosion leaves a patch rather than
#: a point.
_MIN_INTERIOR_PX = 4


@dataclass
class GrainDecision:
    """One decision for one grain.

    ``proposed_phase`` is what the WHOLE grain should be, so it can equal
    ``current_phase`` on a ``reassign``: that is the rim-repair case, where the
    grain's majority is already right and only its outer ring disagrees.

    ``current_phase`` is the majority over the grain's INDEXED pixels, and is
    ``-1`` only when the grain has no indexed pixel at all.

    ``si_spread`` is the p90-p10 range of the most variable element over the
    same pixels the chemistry was taken from (the name predates the fact that
    it is not Si-specific).
    """

    grain_id: int
    n_px: int
    current_phase: int
    proposed_phase: int
    decision: str              # "reassign" | "keep" | "ambiguous"
    reason: str
    chem_fit: dict[int, float] = field(default_factory=dict)
    si_spread: float = 0.0


def tied_phases(chem_fit: dict[int, float],
                tolerance: float | None = None) -> list[int]:
    """The phase ids whose fit is within ``tolerance`` of the best one, sorted.

    More than one means the chemistry did not separate them. This is the same
    predicate ``decide_grains`` applies, exported so a caller can recover the
    fact from ``GrainDecision.chem_fit`` without hardcoding the tolerance or
    reading English out of ``reason``.

    Returns ``[]`` for an empty ``chem_fit``: the branches that do not compute
    fits at all report nothing rather than a tie of nothing.

    ``tolerance=None`` reads ``TIE_TOLERANCE`` at CALL time, not as a def-time
    default. The reason strings interpolate the module global, so a bound
    default would let a caller who rebinds the constant get a decision taken at
    one tolerance and a reason quoting another -- a wrong number in
    user-visible text, on the constant that was just made public so people
    could read it.
    """
    if not chem_fit:
        return []
    tol = TIE_TOLERANCE if tolerance is None else tolerance
    best = max(chem_fit.values())
    return sorted(p for p, f in chem_fit.items() if f >= best - tol)


def _validated_maps(at_pct_maps: dict, shape: tuple) -> dict[str, np.ndarray]:
    """Return the At.% maps as arrays, or raise saying which one is wrong.

    Non-finite values are refused rather than skipped. A NaN reaching the
    median would arrive at ``chemistry_fit`` as a composition, where
    ``max(0.0, nan)`` quietly becomes 0.0 -- a phase silently rewritten as
    "this element is absent", which is the class of bug this whole step exists
    to remove.
    """
    if not at_pct_maps:
        raise ValueError(
            "at_pct_maps is empty: there is no chemistry to decide from")
    out: dict[str, np.ndarray] = {}
    for el, m in at_pct_maps.items():
        arr = np.asarray(m, dtype=np.float64)
        if arr.shape != shape:
            raise ValueError(
                f"at_pct_maps[{el!r}] has shape {arr.shape}, expected {shape}")
        if not np.isfinite(arr).all():
            n_bad = int((~np.isfinite(arr)).sum())
            raise ValueError(
                f"at_pct_maps[{el!r}] has {n_bad} non-finite value(s); "
                "quantify or mask them before deciding phases from them")
        out[el] = arr
    return out


def _window_of(labels: np.ndarray, grain_id: int) -> tuple:
    """Tight bounding box of one grain, as a tuple of slices."""
    hit = labels == grain_id
    if not hit.any():
        raise ValueError(f"grain {grain_id} is not present in the label map")
    rows = np.flatnonzero(hit.any(axis=1))
    cols = np.flatnonzero(hit.any(axis=0))
    return (slice(int(rows[0]), int(rows[-1]) + 1),
            slice(int(cols[0]), int(cols[-1]) + 1))


def _interior(labels: np.ndarray, grain_id: int, erode: bool,
              window: tuple | None = None,
              min_interior: int = 1) -> tuple[tuple, np.ndarray, bool]:
    """Choose the pixels whose chemistry describes this grain.

    Returns ``(window, interior, eroded)`` where ``interior`` is a boolean
    sub-array over ``labels[window]`` and ``eroded`` says whether the erosion
    left at least ``min_interior`` pixels or the full grain was used instead.

    ``min_interior`` is what the CALLER needs from the sample, and the two
    callers need different things. A median is worth having from a single
    interior pixel, so ``aggregate_grain_chemistry`` asks for 1. A spread is
    not: ``p90 - p10`` over one pixel is identically 0.0, so ``decide_grains``
    asks for ``_MIN_INTERIOR_PX`` before it will believe a uniformity test.

    THE single place that choice is made. The aggregation and the spread must
    see the same pixels or the spread measures the mixed rim the aggregation
    was written to avoid, and the two drifting apart is silent.

    The erosion runs inside the grain's bounding box. That is identical to
    eroding the whole map: the box is tight, so no pixel of this grain lies
    outside it, and ``binary_erosion``'s ``border_value=0`` treats everything
    past the box edge as "not this grain" -- which it is, including for a grain
    that touches the edge of the map. Verified against a full-map erosion for
    every grain in ``test_the_interior_matches_a_naive_full_map_erosion``.

    Eroding the full map per grain instead is O(grains x pixels). Measured on
    a 301x402 map with 6406 labels (~500 block grains plus 5% unindexed pixels,
    each of which segment_grains gives its own label): the whole of
    ``decide_grains`` takes 0.21 s this way, against 4.1 s extrapolated for the
    per-grain erosion and median alone -- before the separate spread pass.
    """
    if window is None:
        window = _window_of(labels, grain_id)
    sub = labels[window] == grain_id
    if not erode:
        return window, sub, False
    inner = ndi.binary_erosion(sub, _ERODE_STRUCTURE)
    if int(inner.sum()) >= min_interior:
        return window, inner, True
    return window, sub, False


def _median_chemistry(at_pct_maps: dict[str, np.ndarray], window: tuple,
                      interior: np.ndarray) -> dict[str, float]:
    """Median At.% per element over the selected pixels."""
    return {el: float(np.median(m[window][interior]))
            for el, m in at_pct_maps.items()}


def _spread(at_pct_maps: dict[str, np.ndarray], window: tuple,
            interior: np.ndarray) -> float:
    """p90 - p10 of the most variable element over the SAME pixels, in At.%."""
    worst = 0.0
    for m in at_pct_maps.values():
        v = m[window][interior]
        if v.size:
            worst = max(worst, float(np.percentile(v, 90) - np.percentile(v, 10)))
    return worst


def aggregate_grain_chemistry(labels: np.ndarray,
                              at_pct_maps: dict[str, np.ndarray],
                              grain_id: int,
                              erode: bool = True) -> dict[str, float]:
    """Median At.% over the grain, using the eroded interior when it survives.

    Any non-empty interior is used, however small: one interior pixel is a
    better description of the grain than its mixed rim, measured. Only a grain
    that erodes to NOTHING -- a single-pixel-wide line, say -- falls back to
    its full extent, rim included.

    ``decide_grains`` is stricter, but about the spread rather than about this
    median; see ``_MIN_INTERIOR_PX``.
    """
    labels = np.asarray(labels)
    maps = _validated_maps(at_pct_maps, labels.shape)
    window, interior, _ = _interior(labels, grain_id, erode)
    return _median_chemistry(maps, window, interior)


def decide_grains(labels: np.ndarray, phase_id: np.ndarray,
                  at_pct_maps: dict[str, np.ndarray],
                  expected_by_phase: dict[int, dict[str, float]],
                  min_px: int = 5,
                  spread_limit: float = 20.0) -> list[GrainDecision]:
    """One decision per grain, ordered by grain id.

    ``expected_by_phase`` maps a phase id to its nominal At.% composition (what
    ``phase_nominal_at_pct`` returns for the phase's formula).

    A grain is ``ambiguous`` when its interior chemistry is not unimodal, or
    when two candidate phases fit it equally well; ``reassign`` when the
    chemistry supports a phase that at least one of its pixels does not
    currently carry; ``keep`` otherwise. Nothing here writes to the map --
    Task 4 gates every reassignment on the forward render before it is applied.
    """
    labels = np.asarray(labels)
    phase_id = np.asarray(phase_id)
    if labels.ndim != 2:
        raise ValueError(f"expected a 2D label map, got {labels.shape}")
    if labels.shape != phase_id.shape:
        raise ValueError(f"labels {labels.shape} != phase_id {phase_id.shape}")
    if labels.size and int(labels.min()) < 0:
        raise ValueError(
            f"labels must be non-negative grain ids, found {int(labels.min())}; "
            "an unlabelled pixel has no grain and cannot be decided as one")
    maps = _validated_maps(at_pct_maps, labels.shape)
    if not expected_by_phase:
        raise ValueError(
            "expected_by_phase is empty: there is no candidate phase to choose")
    if min_px < _MIN_INTERIOR_PX:
        # This floor does NOT promise a measurable interior -- min_px counts
        # grain pixels and _MIN_INTERIOR_PX counts interior ones, and a 4-px
        # grain still erodes to nothing. What it promises is narrower and is
        # the thing that would otherwise be fabricated: the spread REPORTED
        # for a grain is measured either over an interior of at least
        # _MIN_INTERIOR_PX px or over the whole grain, so with this floor in
        # place it is never a p90-p10 over one or two pixels.
        raise ValueError(
            f"min_px={min_px} is below {_MIN_INTERIOR_PX}: the spread reported "
            f"for a grain is measured either over an interior of at least "
            f"{_MIN_INTERIOR_PX} px or over the whole grain, and p90-p10 over "
            "one pixel is identically zero, so a lower floor would report a "
            "fabricated 0.0 spread for grains too small to measure at all")

    # One pass for every grain's bounding box, rather than one full-map
    # comparison per grain.
    windows = ndi.find_objects(labels.astype(np.int64) + 1)

    out: list[GrainDecision] = []
    for gid, window in enumerate(windows):
        if window is None:
            continue
        sub_lab = labels[window] == gid
        n_px = int(sub_lab.sum())
        if n_px == 0:
            continue

        pid = phase_id[window][sub_lab]
        indexed = pid >= 0
        n_unindexed = int(n_px - indexed.sum())
        if not indexed.any():
            out.append(GrainDecision(
                gid, n_px, _NOT_INDEXED, _NOT_INDEXED, "keep",
                f"grain is entirely unindexed ({n_px} px), so it has no phase "
                "to keep or change and no pattern to support one"))
            continue
        cur = int(np.bincount(pid[indexed]).argmax())
        unindexed_note = (
            f"; {n_unindexed} of {n_px} px are unindexed and took no part"
            if n_unindexed else "")

        if n_px < min_px:
            out.append(GrainDecision(
                gid, n_px, cur, cur, "keep",
                f"grain too small ({n_px} px < {min_px}){unindexed_note}"))
            continue

        window, interior, eroded = _interior(labels, gid, True, window=window,
                                             min_interior=_MIN_INTERIOR_PX)
        spread = _spread(maps, window, interior)
        if not eroded:
            # The interior is too small to sample, so this grain's chemistry
            # cannot be separated from its mixed rim. Deciding anyway would
            # decide FROM the rim: on a 3x3 particle eight of the nine pixels
            # are interface mixture, and their median proposes the matrix
            # phase -- which is the shelling defect this step exists to
            # remove, re-created by the fallback meant to cover for it. The
            # spread reported here is over the whole grain, so it is a real
            # measurement rather than the identically-zero one a single-pixel
            # interior would have produced.
            out.append(GrainDecision(
                gid, n_px, cur, cur, "ambiguous",
                f"the erosion left fewer than {_MIN_INTERIOR_PX} px, too small "
                "to sample the interior separately from the mixed rim; over "
                f"the whole grain the spread is {spread:.1f} at%"
                f"{unindexed_note}", {}, spread))
            continue
        if spread > spread_limit:
            out.append(GrainDecision(
                gid, n_px, cur, cur, "ambiguous",
                f"chemistry spread {spread:.1f} at% > {spread_limit:.0f} -- "
                f"grain is not chemically uniform{unindexed_note}",
                {}, spread))
            continue

        chem = _median_chemistry(maps, window, interior)
        fits = {int(p): float(chemistry_fit(chem, e))
                for p, e in expected_by_phase.items()}
        # Via the public predicate, so a caller re-deriving the tie from
        # chem_fit cannot get a different answer from the one taken here.
        tied = tied_phases(fits)
        tie_note = ""
        if len(tied) > 1:
            named = ", ".join(f"{p} (fit {fits[p]:.3f})" for p in tied)
            head = (f"chemistry does not separate phases {named} -- a tie "
                    f"within {TIE_TOLERANCE:.2f}")
            if cur not in tied:
                # None of the tied candidates is what the pattern already
                # chose, so there is no non-arbitrary way to pick between
                # them. Saying so beats picking by dict order.
                out.append(GrainDecision(
                    gid, n_px, cur, cur, "ambiguous",
                    f"{head}{unindexed_note}", fits, spread))
                continue
            # The incumbent IS one of the tied candidates, and it was chosen by
            # the diffraction pattern -- an independent tiebreak, and the one
            # this architecture defers to everywhere else (the render gate in
            # Task 4 does the same thing one step later). Preferring it is not
            # dict order. It also matters: without this, ONE duplicated
            # database entry would turn the feature off for every grain of that
            # phase, since "ambiguous" forfeits the rim repair that "keep"
            # would still allow.
            tied = [cur]
            tie_note = f"; {head}, so the pattern's phase {cur} stands"

        best = tied[0]
        n_off = int(np.count_nonzero(pid[indexed] != best))
        if n_off == 0:
            reason = f"already phase {best} throughout (fit {fits[best]:.3f})"
            decision = "keep"
        elif best != cur:
            reason = (f"grain chemistry favours phase {best} "
                      f"(fit {fits[best]:.3f}) over the current majority "
                      f"phase {cur}")
            decision = "reassign"
        else:
            reason = (f"grain is mostly phase {best} already "
                      f"(fit {fits[best]:.3f}) but {n_off} of "
                      f"{int(indexed.sum())} indexed px carry another phase")
            decision = "reassign"
        out.append(GrainDecision(gid, n_px, cur, best, decision,
                                 reason + tie_note + unindexed_note,
                                 fits, spread))
    return out


def _refused(d: GrainDecision, reason: str) -> GrainDecision:
    """The same decision, downgraded to ``keep``, with all eight fields kept.

    ``proposed_phase`` becomes ``current_phase``: a refused decision must leave
    the map where the pattern put it, not where the chemistry wanted it.
    ``chem_fit`` and ``si_spread`` are carried through unchanged, because they
    are the evidence a reader needs to judge the refusal.
    """
    return GrainDecision(d.grain_id, d.n_px, d.current_phase, d.current_phase,
                         "keep", reason, d.chem_fit, d.si_spread)


def _render_score(render_scorer: Callable[..., float | None], grain_id: int,
                  phase: int, pixels: np.ndarray | None = None) -> float | None:
    """One scorer call, with the contract enforced on the way back.

    ``pixels=None`` is passed by OMITTING the argument, so a scorer that only
    implements ``(grain_id, phase_id)`` -- the brief's signature, and every
    caller that never sees a rim repair -- keeps working on the whole-grain
    path.

    A non-finite score is refused rather than compared. ``nan >= x + margin``
    is False, so a NaN would arrive as "the render does not support it" with
    ``nan`` printed where a measurement belongs -- a scorer that cannot compute
    a score says so with ``None``.
    """
    value = (render_scorer(grain_id, phase) if pixels is None
             else render_scorer(grain_id, phase, pixels))
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(
            f"render_scorer({grain_id}, {phase}) returned {value!r}; a score "
            "that cannot be computed must be reported as None, because every "
            "comparison against a non-finite value silently answers 'no'")
    return value


def _disputed_pixels(labels: np.ndarray, phase_id: np.ndarray,
                     d: GrainDecision) -> np.ndarray:
    """The pixels in this grain that disagree with its own decision.

    Those that are INDEXED and carry some phase other than ``proposed_phase``.
    Unindexed pixels are excluded: ``phase_id >= 0`` is the module-wide reading
    of ``_NOT_INDEXED``, and note it is ``>= 0`` rather than ``> 0`` -- zero is
    an ordinary phase id, only NEGATIVE ids mean "no orientation here".

    Raises rather than returning an empty mask, and is called from the
    pre-scan, so a caller learns about a mismatch before any rendering is paid
    for.
    """
    grain = labels == d.grain_id
    if not grain.any():
        raise ValueError(
            f"grain {d.grain_id} is not present in the label map")
    disputed = grain & (phase_id >= 0) & (phase_id != d.proposed_phase)
    if not disputed.any():
        raise ValueError(
            f"grain {d.grain_id} is marked 'reassign' to phase "
            f"{d.proposed_phase} but no indexed pixel in it carries any other "
            "phase, so there is nothing to repair; decisions and maps that "
            "disagree this way are not the same map, and guessing which of "
            "the two to believe is what this module refuses to do")
    return disputed


def _disputed_score(render_scorer: Callable[..., float | None],
                    phase_id: np.ndarray, d: GrainDecision,
                    disputed: np.ndarray
                    ) -> tuple[float | None, float | None, str, tuple]:
    """Score the pixels that actually disagree with the grain's decision.

    Returns ``(proposed, carried, scope_text, groups)``, the two scores being
    pixel-weighted means over the disputed pixels, or
    ``(None, None, "", partial_groups)`` when the scorer could not answer.

    The disputed pixels are grouped by the phase they carry -- a rim can border
    more than one neighbour -- and each group is scored under
    ``proposed_phase`` and under its own phase, over that group's mask alone.
    Combining the groups by pixel-weighted mean makes the verdict a statement
    about pixels rather than about how many neighbours the rim happens to
    touch. The groups PARTITION the disputed set, so the masks NAME exactly
    ``2 * n_disputed`` pixels however many groups there are. How many of those
    a scorer actually renders is its own business: the route's scorer samples
    up to ``GRAIN_RENDER_MAX_PX`` per call, so on a 28-px disputed set it
    renders 32 rather than 56.

    ``groups`` is the per-group evidence: phase, pixel count, both scores and
    the delta between them. It is what makes "adopted despite a render loss on
    4 px" a number rather than a sentence nobody reads. On the ``None`` path it
    holds only the groups scored before the scorer gave up, so it may be
    shorter than the true group count.
    """
    n_disputed = int(disputed.sum())
    weighted_new = 0.0
    weighted_cur = 0.0
    parts: list[str] = []
    groups: list[dict] = []
    for carried in sorted(int(p) for p in np.unique(phase_id[disputed])):
        mask = disputed & (phase_id == carried)
        n_px = int(mask.sum())
        s_new = _render_score(render_scorer, d.grain_id, d.proposed_phase, mask)
        s_cur = _render_score(render_scorer, d.grain_id, carried, mask)
        if s_new is None or s_cur is None:
            return None, None, "", tuple(groups)
        weighted_new += n_px * s_new
        weighted_cur += n_px * s_cur
        parts.append(f"phase {carried}: {n_px} px")
        groups.append({"phase": carried, "n_px": n_px,
                       "score_proposed": s_new, "score_current": s_cur,
                       "delta": s_new - s_cur})

    scope = f"at the {n_disputed} disputed px ({', '.join(parts)})"
    return (weighted_new / n_disputed, weighted_cur / n_disputed, scope,
            tuple(groups))


def verify_decisions(
    decisions: list[GrainDecision],
    render_scorer: Callable[..., float | None],
    margin: float = RENDER_ADOPT_MARGIN,
    labels: np.ndarray | None = None,
    phase_id: np.ndarray | None = None,
) -> list[GrainDecision]:
    """The gated decisions, and nothing else.

    A thin wrapper over ``verify_decisions_with_evidence``, whose docstring is
    the specification. Use that one if you need to know WHY a decision came out
    the way it did: an ADOPTED decision is returned by identity, so its
    ``reason`` is still the chemistry sentence ``decide_grains`` wrote, and the
    render numbers behind the adoption exist nowhere in this return value.
    """
    return verify_decisions_with_evidence(
        decisions, render_scorer, margin, labels, phase_id)[0]


def verify_decisions_with_evidence(
    decisions: list[GrainDecision],
    render_scorer: Callable[..., float | None],
    margin: float = RENDER_ADOPT_MARGIN,
    labels: np.ndarray | None = None,
    phase_id: np.ndarray | None = None,
) -> tuple[list[GrainDecision], dict[int, dict]]:
    """Keep a change only if the forward render supports it.

    Chemistry proposes, the forward model decides. A chemistry score cannot
    overturn the pattern on its own -- that is exactly how a phase with a
    convenient nominal composition took 10 % of a map it had no business in.

    THE SCORER CONTRACT, which binds whatever is passed as ``render_scorer``::

        render_scorer(grain_id: int, phase_id: int, pixels=None) -> float | None

    ``pixels`` is a boolean mask over the FULL map, the same shape as
    ``labels``, naming the pixels to score; ``None`` means the scorer's own
    default scope (the grain's eroded interior). The return is a render-NCC, or
    ``None`` when it cannot be computed -- never NaN, and never an infinity:
    ``inf >= x + margin`` is True, so an infinite score would ADOPT the change.

    The cache key is therefore ``(grain_id, phase_id, pixels)`` -- all three.
    A scorer must also be able to score an ARBITRARY pixel subset, not only its
    default scope, and must cope with a subset of one pixel (where an NCC is
    degenerate: return ``None``, not NaN).

    The masks handed to it PARTITION the disputed pixels, so a rim repair NAMES
    exactly ``2 * n_disputed`` pixels however many groups it has; only the
    per-call overhead scales with the number of groups. A scorer may render
    fewer than it is named -- the route's scorer caps each call at
    ``GRAIN_RENDER_MAX_PX`` -- so that figure is an upper bound on the work,
    not a count of it.

    The whole-grain path calls the scorer with TWO positional arguments, so a
    scorer implementing only ``(grain_id, phase_id)`` still works there. It
    will raise ``TypeError`` on a rim repair, which is the honest outcome: that
    scorer cannot answer the question that case asks.

    TWO GATES, because ``proposed_phase`` can equal ``current_phase``:

    * ``proposed != current`` -- the whole grain changes phase. Score the grain
      under each phase in the scorer's default scope and require the proposal
      to win by ``margin``.

    * ``proposed == current`` -- a rim repair: the grain's identity is already
      right and only some of its pixels disagree. Scoring the grain under
      ``proposed`` and under ``current`` here is the SAME call -- same grain,
      same phase, same scope, so any cache keyed on those returns the identical
      float -- which makes the test ``x >= x + margin``: always False, every rim
      repair downgraded to ``keep``, and the refusal reading ``proposed 0.200 vs
      current 0.200``. Instead the DISPUTED pixels are scored -- see
      ``_disputed_score`` -- which asks the honest question: at these specific
      pixels, does the grain's phase render better than what they carry now?

      A SCORER CACHE MUST STILL KEY ON THE MASK, for a different reason: a rim
      bordering two phases produces two groups scored under the SAME
      ``proposed_phase``, so a ``(grain_id, phase_id)`` key would serve the
      second group the first group's number.

    A rim repair needs ``labels`` and ``phase_id`` to find those pixels. If
    either is missing this RAISES, rather than passing the decision through
    ungated (which would apply the repair on chemistry alone, the thing this
    gate exists to prevent) or downgrading it to ``keep`` (which would hide the
    repair behind a gate that never ran). A caller who cannot supply the arrays
    cannot have this gate, and is told so rather than handed a result that
    looks gated and is not.

    THE ARRAYS ARE IGNORED WHEN NO RIM REPAIR IS PRESENT, deliberately, and are
    not validated then either. The whole-grain path never looks at a pixel, so
    requiring them there would break the brief's own no-array calls. The shape
    and 2D checks are therefore a guarantee about the RIM path, not about the
    call.

    ``None`` from any scorer call refuses that decision; nothing is guessed.
    Decisions that are not ``reassign`` are returned unchanged, by identity,
    the scorer is not called for them, and they get no evidence entry.

    THE EVIDENCE, keyed by ``grain_id``, one entry per ``reassign`` decision.
    It exists because an ADOPTED decision is returned by identity: its
    ``reason`` is still the chemistry sentence from ``decide_grains``, so a
    repair adopted at 0.713 vs 0.288 would otherwise be indistinguishable from
    one adopted at 0.031 vs 0.001 -- and the case this design deliberately
    permits, where a small group renders WORSE under the proposal and is
    repaired anyway on the pixel-weighted mean, would be invisible. Numbers
    from this pipeline reach a publication; those pixels have to be countable.

    ==================  =========================================================
    ``path``            ``"whole-grain"`` or ``"rim"``
    ``current_phase``   as decided
    ``proposed_phase``  as decided
    ``score_proposed``  the score compared, ``None`` if the scorer gave up
    ``score_current``   the score compared, ``None`` if the scorer gave up
    ``margin``          the margin actually applied
    ``verdict``         ``"adopted"`` / ``"refused"`` / ``"no-render-score"``
    ``n_disputed``      rim only; ``None`` on the whole-grain path
    ``groups``          rim only; ``()`` on the whole-grain path. Per group:
                        ``phase``, ``n_px``, ``score_proposed``,
                        ``score_current``, ``delta``. PARTIAL on a
                        ``no-render-score`` verdict.
    ``n_px_worse``      rim only: disputed px in groups whose ``delta`` is
                        negative -- the pixels adopted DESPITE the render.
                        ``None`` on the whole-grain path and on a
                        ``no-render-score`` verdict, because those did not
                        measure it; ``0`` means measured and none.
    ==================  =========================================================

    ``None`` rather than ``0`` for the unmeasured cases is not fussiness: a
    zero there is a claim the gate never checked, which is the class of
    fabricated number this module has already been corrected for once.
    """
    decisions = list(decisions)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError(
            f"margin={margin!r} is not a usable threshold: a negative margin "
            "would let the render's disagreement count in favour of the very "
            "change this gate exists to refuse, and a non-finite one makes "
            "every comparison False while the reason still quotes two numbers "
            "as though a threshold had been applied")

    gated = [d for d in decisions if d.decision == "reassign"]
    seen: set[int] = set()
    for d in gated:
        if d.grain_id in seen:
            raise ValueError(
                f"grain {d.grain_id} has more than one 'reassign' decision; "
                "the evidence is keyed by grain, so one of them would be "
                "silently overwritten and the caller would never learn which")
        seen.add(d.grain_id)

    rim_repairs = [d for d in gated
                   if d.proposed_phase == d.current_phase]
    disputed_by_grain: dict[int, np.ndarray] = {}
    if rim_repairs:
        if labels is None or phase_id is None:
            missing = " and ".join(
                n for n, v in (("labels", labels), ("phase_id", phase_id))
                if v is None)
            raise ValueError(
                f"{len(rim_repairs)} decision(s) repair pixels inside a grain "
                f"whose phase is not changing (grain {rim_repairs[0].grain_id} "
                f"is one), and that can only be gated on the disputed pixels "
                f"themselves, which needs both labels and phase_id -- "
                f"{missing} was not supplied. Without them this call would "
                "either compare a render against itself or adopt the repair on "
                "chemistry alone")
        labels = np.asarray(labels)
        phase_id = np.asarray(phase_id)
        if labels.ndim != 2:
            raise ValueError(f"expected a 2D label map, got {labels.shape}")
        if labels.shape != phase_id.shape:
            raise ValueError(
                f"labels {labels.shape} != phase_id {phase_id.shape}")
        # Before any rendering is paid for: a decision that disagrees with the
        # arrays is a mismatch, and finding out mid-loop wastes real GPU time.
        for d in rim_repairs:
            disputed_by_grain[d.grain_id] = _disputed_pixels(
                labels, phase_id, d)

    out: list[GrainDecision] = []
    evidence: dict[int, dict] = {}
    for d in decisions:
        if d.decision != "reassign":
            out.append(d)
            continue

        is_rim = d.proposed_phase == d.current_phase
        if is_rim:
            disputed = disputed_by_grain[d.grain_id]
            r_new, r_cur, scope, groups = _disputed_score(
                render_scorer, phase_id, d, disputed)
            n_disputed = int(disputed.sum())
        else:
            r_new = _render_score(render_scorer, d.grain_id, d.proposed_phase)
            r_cur = _render_score(render_scorer, d.grain_id, d.current_phase)
            scope, groups, n_disputed = "", (), None

        if r_new is None or r_cur is None:
            verdict = "no-render-score"
            out.append(_refused(
                d, "no render score available -- refusing to reassign on "
                   "chemistry alone"))
        elif r_new >= r_cur + margin:
            verdict = "adopted"
            out.append(d)
        elif scope:
            verdict = "refused"
            out.append(_refused(
                d, f"render does not support it {scope}: proposed "
                   f"{r_new:.3f} vs current {r_cur:.3f} (needs "
                   f"+{margin:.2f}); both are pixel-weighted means over those "
                   "groups"))
        else:
            verdict = "refused"
            out.append(_refused(
                d, f"render does not support it: proposed {r_new:.3f} vs "
                   f"current {r_cur:.3f} (needs +{margin:.2f})"))

        evidence[d.grain_id] = {
            "path": "rim" if is_rim else "whole-grain",
            "current_phase": d.current_phase,
            "proposed_phase": d.proposed_phase,
            "score_proposed": r_new,
            "score_current": r_cur,
            "margin": margin,
            "verdict": verdict,
            "n_disputed": n_disputed,
            "groups": groups,
            # None, not 0: the whole-grain path never looked at a pixel, and a
            # refusal for want of a score did not finish looking.
            "n_px_worse": (
                sum(g["n_px"] for g in groups if g["delta"] < 0.0)
                if is_rim and verdict != "no-render-score" else None),
        }
    return out, evidence
