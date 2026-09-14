"""Per-phase fit scores for Crystal Hint candidates.

Two heuristic scorers that turn the raw symmetry-NCC vector + Hough-band
d-spacings into per-phase fit scores (0..1, higher = better fit). Used
to discriminate between phases that all pass the categorical filters
(chemistry, system, lattice-range) — without this, every preset-expected
cubic phase tied at score 1.50 on a typical Al-alloy pixel because the
filters alone don't pick a winner.

The two scorers are INDEPENDENT and complementary:

1. `symmetry_fit(n_fold_scores, crystal_system)`
   Uses the relative NCC weight of the folds that ARE present in the
   phase's point group. A tetragonal phase on a pattern with strong
   3-fold NCC scores below a cubic phase (which has 3-fold).

2. `dspacing_fit(d_observed_A, lattice_a_A, crystal_system)`
   For each observed Hough-band d-spacing, finds the closest theoretical
   d-spacing the phase would produce. Lower mean relative error → higher
   fit. Cubic phases enumerate the standard {hkl} reflection list; non-
   cubic phases fall back to a neutral score (1.0) so they're neither
   helped nor penalised by this scorer alone.

Both return 0..1 with neutral = 1.0 (no penalty) when they can't compute
a score (missing inputs, unknown system, etc.). Multiplied into the
existing chemistry × system × preset score in `crystal_hint_local_library`.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional


# Folds that are PRIMARY rotation axes in each crystal system's
# highest-symmetry point group. Used by `symmetry_fit`. References:
# International Tables for Crystallography, Vol A, "Point groups".
#
# Conservative — we list the n-fold rotations that COULD show up at a
# zone-axis projection. Off-axis projections show lower-fold or 2-fold
# everywhere, but we only score the high-symmetry signatures.
_SYSTEM_PRIMARY_FOLDS: dict[str, frozenset[int]] = {
    "cubic":        frozenset({2, 3, 4}),
    "tetragonal":   frozenset({2, 4}),       # NO 3-fold
    "hexagonal":    frozenset({2, 3, 6}),
    "trigonal":     frozenset({2, 3}),
    "orthorhombic": frozenset({2}),
    "monoclinic":   frozenset({2}),
    "triclinic":    frozenset(),             # no rotational sym beyond identity
}


# Threshold below which the strongest observed NCC is treated as "no
# signal" — when the best fold barely beats noise, we can't discriminate
# crystal systems and return a neutral score.
_NCC_SIGNAL_FLOOR = 0.15
# A fold is "strongly present" if its NCC is at least this fraction of
# the best observed fold (relative threshold). Used both to identify
# observation-strong folds and to flag candidate-required folds that are
# missing.
_STRONG_RATIO = 0.6
# A fold is "weakly present" if below this fraction of best — counts as
# "absent" for purposes of penalising candidates that require it.
_WEAK_RATIO = 0.4
# Per-mismatch penalty weights. Tuned so:
#  - A candidate whose strongest required fold is missing AND has a
#    strong observed fold it can't explain drops to ~0.3 (heavy penalty,
#    above the 0.2 floor).
#  - A perfect match gives 1.0.
_PENALTY_MISSING_REQUIRED = 0.25
_PENALTY_UNEXPECTED_STRONG = 0.35


def symmetry_fit(
    n_fold_scores: dict[int, float],
    crystal_system: str,
) -> float:
    """Score how well a phase's primary rotation axes match the observed
    NCC pattern.

    Returns 0..1, mapped into [0.2, 1.0] so a "wrong" symmetry never
    fully zeros a candidate (we want the user to still see it).

    Discriminator logic — two penalty terms (both 0..1):
      - "missing-required": for each fold in the candidate's primary set
        whose observed NCC is *weak* (< _WEAK_RATIO × best), count one
        penalty unit. Captures e.g. m-3m candidates on patterns where
        3-fold is absent.
      - "unexpected-strong": for each observed fold that is *strong*
        (>= _STRONG_RATIO × best) but NOT in the candidate's primary
        set, count one penalty unit. Captures e.g. tetragonal/ortho
        candidates on patterns with strong 3-fold.

    Score = 1.0 − _PENALTY_MISSING_REQUIRED × missing_count
                − _PENALTY_UNEXPECTED_STRONG × unexpected_count
    Clamped to [0.2, 1.0].

    Worked example with observed {2: 0.38, 3: 0.35, 4: 0.31, 6: 0.10}
    (best=0.38; strong-threshold=0.228; weak-threshold=0.152):
      strong observed = {2, 3, 4}; weak observed = {6}

      cubic (2,3,4):
        missing-required = ∅ → 0 penalty
        unexpected-strong = ∅ → 0 penalty
        score = 1.0

      tetragonal (2,4):
        missing-required = ∅ (both 2 and 4 are strong)
        unexpected-strong = {3} → 0.35 penalty
        score = 0.65

      hexagonal (2,3,6):
        missing-required = {6} (6 is weak)   → 0.25
        unexpected-strong = {4} → 0.35
        score = 0.40

    On a pattern where 4-fold is strong but 3-fold is weak
    {2: 0.4, 3: 0.15, 4: 0.5, 6: 0.05} (best=0.5):
      strong observed = {2, 4}; weak observed = {3, 6}
      cubic (2,3,4):
        missing-required = {3} → 0.25 penalty
        unexpected-strong = ∅
        score = 0.75
      tetragonal (2,4):
        missing-required = ∅
        unexpected-strong = ∅
        score = 1.0   ← tetragonal now WINS over cubic, as it should

    Neutral score 1.0 returned when:
      - `n_fold_scores` empty / all zeros
      - `crystal_system` unknown
      - max NCC below noise floor (no usable signal)
      - candidate is triclinic (no fold info, no claim to test)
    """
    if not n_fold_scores:
        return 1.0
    system_norm = (crystal_system or "").strip().lower()
    primary = _SYSTEM_PRIMARY_FOLDS.get(system_norm)
    if primary is None:
        return 1.0  # unknown system → no penalty
    if not primary:
        return 1.0  # triclinic — nothing to claim, nothing to refute

    positives = {n: max(0.0, float(s)) for n, s in n_fold_scores.items()}
    if not positives:
        return 1.0
    best = max(positives.values())
    if best < _NCC_SIGNAL_FLOOR:
        # No usable signal — every candidate gets neutral. Don't fake
        # discrimination from noise.
        return 1.0

    strong_thresh = _STRONG_RATIO * best
    weak_thresh = _WEAK_RATIO * best

    # missing-required: primary folds whose observed NCC is below weak.
    missing_required = sum(
        1 for n in primary
        if positives.get(n, 0.0) < weak_thresh
    )
    # unexpected-strong: observed strong folds that the candidate's
    # point group does not have.
    unexpected_strong = sum(
        1 for n, s in positives.items()
        if s >= strong_thresh and n not in primary
    )

    score = (
        1.0
        - _PENALTY_MISSING_REQUIRED * missing_required
        - _PENALTY_UNEXPECTED_STRONG * unexpected_strong
    )
    return max(0.2, min(1.0, score))


def _bravais_centering(space_group: str) -> str:
    """Infer the Bravais centering letter from a space-group symbol.

    Maps:
      "P..." → P (primitive)
      "I..." → I (body-centered)
      "F..." → F (face-centered)
      "A...", "B...", "C..." → C (base-centered; treated as C-face)
      "R..." → R (rhombohedral, treated as primitive for now)
    Default "P" so unknown groups don't over-filter.
    """
    if not space_group:
        return "P"
    first = space_group.strip()[0].upper()
    if first in ("P", "I", "F", "R"):
        return first
    if first in ("A", "B", "C"):
        return "C"
    return "P"


def _is_reflection_allowed(h: int, k: int, l: int, centering: str) -> bool:
    """Structure-factor extinction rules for the lattice centering.

    Without this, the d-spacing scorer would match against forbidden
    reflections (e.g. {100} for FCC Al doesn't actually appear in the
    pattern but a=4.05 would still get credit for d=4.30 ≈ a). Filtering
    by Bravais centering is the difference between Al ranking falsely
    above α-AlFeMnSi and the correct order on the user's pixel.

    Cubic reflections only — for non-cubic the rules differ per system
    + space group (glide planes, screw axes etc.). We're conservative:
    if `centering` is unknown, allow everything.
    """
    if centering == "I":  # body-centered: h + k + l must be even
        return (h + k + l) % 2 == 0
    if centering == "F":  # face-centered: h, k, l all even or all odd
        evens = sum(1 for v in (h, k, l) if v % 2 == 0)
        return evens in (0, 3)
    if centering == "C":  # C-face centered: h + k must be even
        return (h + k) % 2 == 0
    # Primitive (P) / rhombohedral (R) / unknown: all (hkl) allowed
    return True


def _cubic_d_spacings(a_A: float, hkl_max: int = 4,
                      space_group: str = "") -> list[float]:
    """Enumerate the ALLOWED unique d-spacings for the cubic system.

    Returns sorted descending. Uses d_hkl = a / sqrt(h² + k² + l²) and
    filters by `space_group`'s Bravais centering — without that filter
    forbidden reflections (e.g. {100} for FCC, {100} for BCC) would
    pollute the fit score.
    """
    centering = _bravais_centering(space_group)
    seen_n2: set[int] = set()
    ds: list[float] = []
    for h in range(0, hkl_max + 1):
        for k in range(0, hkl_max + 1):
            for l in range(0, hkl_max + 1):
                n2 = h * h + k * k + l * l
                if n2 == 0 or n2 in seen_n2:
                    continue
                # All (hkl) with the same h²+k²+l² have the same d in
                # cubic, but extinction rules differ — pick the canonical
                # representative for the family. Check if ANY (hkl) with
                # this n2 is allowed; if so the family is observed.
                if not _is_reflection_allowed(h, k, l, centering):
                    # Try to find an allowed permutation/sign combo with
                    # the same n2 — for cubic families like {hkl} = {001}
                    # vs {010} vs {100}, all give same d but only some
                    # may be allowed. We pre-enumerate (h,k,l) ≥ 0; the
                    # cubic point-group equivalents include sign flips
                    # which all share the same extinction rule, so we
                    # only need to check permutations.
                    if not any(_is_reflection_allowed(p, q, r, centering)
                              for p, q, r in {(h, k, l), (h, l, k),
                                              (k, h, l), (k, l, h),
                                              (l, h, k), (l, k, h)}):
                        continue
                seen_n2.add(n2)
                ds.append(a_A / math.sqrt(n2))
    ds.sort(reverse=True)
    return ds


def _tetragonal_d_spacings(a_A: float, c_A: float, hkl_max: int = 4) -> list[float]:
    """Enumerate d-spacings for tetragonal system.

    d_hkl = 1 / sqrt((h^2 + k^2)/a^2 + l^2/c^2)
    """
    seen: set[float] = set()
    ds: list[float] = []
    inv_a2 = 1.0 / (a_A * a_A)
    inv_c2 = 1.0 / (c_A * c_A)
    for h in range(0, hkl_max + 1):
        for k in range(0, hkl_max + 1):
            for l in range(0, hkl_max + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                inv_d2 = (h * h + k * k) * inv_a2 + l * l * inv_c2
                if inv_d2 <= 0:
                    continue
                d = 1.0 / math.sqrt(inv_d2)
                # Quantise to 1e-4 Å to dedupe equivalent-by-symmetry triplets
                key = round(d, 4)
                if key in seen:
                    continue
                seen.add(key)
                ds.append(d)
    ds.sort(reverse=True)
    return ds


def dspacing_fit(
    d_observed_A: Iterable[float],
    lattice_a_A: Optional[float],
    crystal_system: str,
    lattice_c_A: Optional[float] = None,
    hkl_max: int = 4,
    space_group: str = "",
) -> float:
    """Score how well a phase's theoretical d-spacings match the
    observed Hough-band d-spacings.

    Returns 0..1 with 1.0 = perfect fit, near 0 = no overlap.

    For each observed d_obs, finds the closest theoretical d_calc and
    computes relative error |d_obs - d_calc| / d_obs. Score is
    1 / (1 + mean_relative_error). A 10% mean error → fit ~0.91; a 50%
    mean error → fit ~0.67.

    Neutral score 1.0 returned when:
      - No observed d-spacings (lattice estimator gave up)
      - Missing lattice_a_A
      - Non-cubic / non-tetragonal (we don't enumerate those families yet
        in this initial cut — followup to add hexagonal/orthorhombic)
    """
    d_observed = [d for d in (d_observed_A or []) if d and d > 0]
    if not d_observed or not lattice_a_A:
        return 1.0

    # Adaptive hkl_max: enumerate enough reflections to reach the smallest
    # observed d-spacing. For a=12.5 Å with d_obs_min=1.06, we need
    # hkl_max ≥ ceil(a/d_obs_min) so that a/sqrt(n²) can reach d_obs_min.
    # Without this, large-cell phases (like α-AlFeMnSi a=12.5) couldn't
    # match small d-spacings and got unfairly low scores.
    d_min = min(d_observed)
    needed_max = max(int(math.ceil(float(lattice_a_A) / max(d_min, 0.1))), hkl_max)
    needed_max = min(needed_max, 12)  # cap at 12 — beyond that we hit
                                       # diminishing returns + lots of high-
                                       # order reflections that mostly
                                       # dominate the search.

    system_norm = (crystal_system or "").strip().lower()
    if system_norm == "cubic":
        d_calc = _cubic_d_spacings(float(lattice_a_A), hkl_max=needed_max,
                                  space_group=space_group)
    elif system_norm == "tetragonal":
        c = float(lattice_c_A) if lattice_c_A else float(lattice_a_A)
        d_calc = _tetragonal_d_spacings(float(lattice_a_A), c, hkl_max=needed_max)
    else:
        # Hexagonal / orthorhombic / monoclinic / triclinic d-spacing
        # formulas need (a, b, c, α, β, γ) — we only have `a` for most
        # external candidates. Skip with neutral score until we extend.
        return 1.0

    if not d_calc:
        return 1.0

    # Weighted by d_obs magnitude — LARGER d's correspond to lower-order
    # reflections (the strongest Kikuchi bands), which are more
    # informative + less prone to Hough-detection noise. Without this,
    # a single noisy small d_obs could outweigh a strong large d.
    total_weight = 0.0
    total_weighted_err = 0.0
    for d_obs in d_observed:
        weight = d_obs  # linear weight by d magnitude
        best = min(abs(d_obs - d_c) for d_c in d_calc)
        total_weighted_err += weight * (best / d_obs)
        total_weight += weight
    mean_rel_err = total_weighted_err / total_weight if total_weight > 0 else 0.0
    return max(0.05, 1.0 / (1.0 + mean_rel_err))


def combined_phase_fit(
    n_fold_scores: dict[int, float],
    crystal_system: str,
    d_observed_A: Iterable[float],
    lattice_a_A: Optional[float],
    lattice_c_A: Optional[float] = None,
    space_group: str = "",
) -> tuple[float, float, float]:
    """Convenience wrapper. Returns (symmetry_fit, dspacing_fit, combined).

    combined = symmetry_fit × dspacing_fit. Both individual scores are
    surfaced separately so the UI can show a breakdown ("why is this
    phase ranked here").
    """
    s_fit = symmetry_fit(n_fold_scores, crystal_system)
    d_fit = dspacing_fit(d_observed_A, lattice_a_A, crystal_system,
                        lattice_c_A=lattice_c_A, space_group=space_group)
    return s_fit, d_fit, s_fit * d_fit


# ---------------------------------------------------------------------------
# Pattern-degeneracy collapse
# ---------------------------------------------------------------------------
# A phase's Kikuchi pattern is determined by its space group + lattice
# parameters, NOT its exact site occupancy. Two phases with the same
# space group and lattice (within measurement tolerance) produce the same
# pattern and CANNOT be distinguished by EBSD geometry. Listing both wastes
# a candidate slot and pushes genuinely distinct phases down the list.
#
# `collapse_degenerate` groups a scored candidate list by structural key
# and keeps one representative per group (the highest-scoring), recording
# the absorbed members so the UI can show "also matches: X, Y".

# Relative lattice tolerance: two lattice constants count as equal when
# they differ by less than this fraction. 2% comfortably covers the
# ±20% heuristic noise floor on lattice estimation while still separating
# genuinely different structures (e.g. Fm-3m Al a=4.05 vs Al2Cu a=5.77).
_LATTICE_REL_TOL = 0.02


def _lattice_close(a1: Optional[float], a2: Optional[float]) -> bool:
    """True if two lattice constants are equal within _LATTICE_REL_TOL.

    None is treated as a wildcard (matches anything) so a missing axis
    (e.g. cubic phases carrying only `a`) doesn't block a merge."""
    if a1 is None or a2 is None:
        return True
    hi = max(abs(a1), abs(a2))
    if hi < 1e-9:
        return True
    return abs(a1 - a2) / hi < _LATTICE_REL_TOL


def _degenerate(x: dict, y: dict) -> bool:
    """True if candidates x and y produce indistinguishable patterns:
    same crystal_system, same space group, lattice a/b/c all within tol.

    Requires REAL structural evidence first: a candidate with an unknown
    crystal system, or with no space group AND no lattice at all, is never
    declared degenerate — otherwise metadata-poor entries (all defaulting
    to system="unknown", sg=None, a=None) would falsely collapse into one,
    hiding genuinely distinct phases. Degeneracy is a positive claim about
    shared structure; absence of structure is not evidence of sameness."""
    sys_x = (x.get("crystal_system") or "").strip().lower()
    sys_y = (y.get("crystal_system") or "").strip().lower()
    if sys_x in ("", "unknown") or sys_y in ("", "unknown"):
        return False
    if sys_x != sys_y:
        return False
    # Need at least one concrete structural discriminator on x (and, by the
    # checks below, a matching one on y): a space group OR a lattice axis.
    has_sg = (x.get("sg_number") is not None
              or bool((x.get("space_group") or "").strip()))
    has_lat = any(x.get(k) is not None for k in ("a", "b", "c"))
    if not has_sg and not has_lat:
        return False
    # Prefer space-group NUMBER when both present; else fall back to the
    # normalised symbol. A mismatch here means different structure.
    xn, yn = x.get("sg_number"), y.get("sg_number")
    if xn is not None and yn is not None:
        if int(xn) != int(yn):
            return False
    else:
        xs = (x.get("space_group") or "").replace(" ", "").lower()
        ys = (y.get("space_group") or "").replace(" ", "").lower()
        if xs and ys and xs != ys:
            return False
    return (_lattice_close(x.get("a"), y.get("a"))
            and _lattice_close(x.get("b"), y.get("b"))
            and _lattice_close(x.get("c"), y.get("c")))


def collapse_degenerate(items: list[dict]) -> list[dict]:
    """Collapse pattern-degenerate candidates, preserving input order.

    Each item is a dict with at least::
        {"name": str, "crystal_system": str, "space_group": str,
         "sg_number": Optional[int], "a": float|None, "b": ..., "c": ...}

    Walks the list in the order given (callers pass it pre-sorted by
    score, best first). The first occurrence of each structural group is
    the representative; later degenerate members are dropped and their
    names appended to the representative's ``degenerate_with`` list.

    Returns a new list of the representatives (order preserved). Items are
    not mutated; representatives are shallow-copied with the added
    ``degenerate_with`` key. Bit-identical (modulo the empty
    ``degenerate_with``) when no two items are degenerate.
    """
    reps: list[dict] = []
    for it in items:
        merged = False
        for rep in reps:
            if _degenerate(rep, it):
                rep["degenerate_with"].append(it.get("name", "?"))
                merged = True
                break
        if not merged:
            new_rep = dict(it)
            new_rep["degenerate_with"] = []
            reps.append(new_rep)
    return reps


# ---------------------------------------------------------------------------
# EDS chemistry weighting (Option C)
# ---------------------------------------------------------------------------
# Pattern symmetry + d-spacing cannot separate chemically-distinct phases
# that share a structure (Al vs Al2Cu vs alpha-Al(Fe,Mn)Si are all cubic
# and tie). The pixel's measured EDS chemistry is the discriminator: a
# Fe-rich pixel should down-rank pure Al / Si and up-rank Fe-bearing
# phases. These two helpers turn a candidate's formula + the pixel's At%
# into a 0..1 chemistry-match score.

# Ubiquitous light elements excluded from the chemistry match: surface
# oxide (O) and carbon coating (C) are acquisition artifacts, not phase
# chemistry, and would otherwise dominate the At% vector.
_CHEM_IGNORE = frozenset({"O", "C"})


def phase_nominal_at_pct(formula: str) -> dict[str, float]:
    """Parse a chemical formula into nominal atomic-percent per element.

    Handles integer and decimal subscripts, with or without spaces:
      "Al2Cu"                          -> {Al: 66.7, Cu: 33.3}
      "Mn4.512Al127.296Fe19.488Si16.704" -> normalised At%
      "Al"                             -> {Al: 100.0}
    Returns {} when nothing parseable is found (caller treats as neutral).
    Bracketed/comma display forms like "(Al,Fe,Si)" fall back to equal
    split across the listed elements.
    """
    if not formula:
        return {}
    import re
    # Element symbol + optional (int or decimal) subscript.
    matches = re.findall(r"([A-Z][a-z]?)\s*(\d*\.?\d+)?", formula)
    counts: dict[str, float] = {}
    for sym, num in matches:
        if not sym:
            continue
        try:
            c = float(num) if num else 1.0
        except ValueError:
            c = 1.0
        counts[sym] = counts.get(sym, 0.0) + c
    total = sum(counts.values())
    if total <= 1e-9:
        return {}
    return {el: 100.0 * c / total for el, c in counts.items()}


def chemistry_fit(
    pixel_at_pct: dict[str, float],
    phase_at_pct: dict[str, float],
    unmeasured=None,
) -> float:
    """Score how well a phase's nominal composition matches a pixel's
    measured EDS chemistry. Returns 0..1 (1 = identical, 0 = disjoint).

    Both vectors are restricted to metallic elements (O/C dropped), then
    renormalised to fractions summing to 1, and compared by L1 distance:
        score = 1 - 0.5 * sum_i |pixel_i - phase_i|     (L1 in [0,2])

    ``unmeasured`` — elements the quantification could not price, whose window
    was therefore left out and the rest renormalised without it. WITHOUT THIS
    THE HOLE IS READ AS A MEASURED ZERO, and the missing-major veto below then
    floors every phase whose major element is the one nobody could measure — a
    20x suppression of exactly the phases the missing element identifies. "We
    could not measure this" and "this is not here" are opposite statements.
    Such an element is dropped from BOTH vectors and the phase side is
    renormalised over what remains, so a 20 at% Cu phase does not pay a 20 %
    penalty for an unpriceable Cu window; the elements that WERE measured still
    decide.

    Neutral 1.0 (no effect on ranking) is returned when the match can't be
    computed — empty pixel chemistry (no EDS / dead pixel) or empty phase
    composition — so chemistry weighting fails soft to the symmetry/lattice
    ranking instead of crashing or zeroing everything.
    """
    def _clean(d: dict[str, float]) -> dict[str, float]:
        kept = {el: max(0.0, float(v)) for el, v in (d or {}).items()
                if el not in _CHEM_IGNORE and v is not None}
        tot = sum(kept.values())
        if tot <= 1e-9:
            return {}
        return {el: v / tot for el, v in kept.items()}

    blind = frozenset(unmeasured or ())
    p = _clean({el: v for el, v in (pixel_at_pct or {}).items()
                if el not in blind})
    q = _clean({el: v for el, v in (phase_at_pct or {}).items()
                if el not in blind})
    if not p or not q:
        return 1.0  # fail-soft: no usable chemistry on one side
    l1 = 0.0
    for el in set(p) | set(q):
        l1 += abs(p.get(el, 0.0) - q.get(el, 0.0))
    score = max(0.0, min(1.0, 1.0 - 0.5 * l1))

    # Missing-major-element veto. Symmetric L1 gives a phase partial credit
    # for the elements it DOES share (e.g. an Fe-Al-Si intermetallic that is
    # 25% Al scores ~0.5 on a 94%-Al matrix pixel because the Al overlap
    # masks the absent Fe). But a phase that REQUIRES a major element
    # (>=15 at% nominally) which is essentially ABSENT (<2 at%) in the
    # measured pixel cannot be that phase here — no symmetry/lattice match
    # should resurrect it. Cap such phases low so 'soft' strongly damps and
    # 'filter' (threshold 0.35) drops them. This implements the documented
    # "drop chemically-impossible phases that win cubic-FCC by symmetry
    # degeneracy" rule. Pure tightening: it only ever LOWERS scores for
    # phases the pixel chemistry rules out.
    # 2026-08-04: _MAJOR_REQ lowered 0.15 -> 0.05. At 0.15 a phase whose
    # DEFINING elements are nominally below 15 at% was never vetoed when those
    # elements were absent. alpha-Al(Fe,Mn)Si (Fe 7.5 + Mn 7.5 at%) therefore
    # scored 0.596 on Al/Si mixing-boundary pixels that contain 0.2 at% Fe --
    # BETTER than Si (0.235) -- because its nominal Al/Si ratio sits on the
    # Al<->Si mixing line. That is why the alpha rim around Si particles
    # survived even a fully symmetric prior (measured: 127/168 particles still
    # shelled). Verified on ProbeB: rim fit 0.596 -> 0.050, while the real
    # 1607-px Fe/Mn particle (Fe 3.96 at%) keeps fit 0.903 and still wins over
    # Al (0.712) and Si (0.020) -- the 2 at% _ABSENT floor gives enough margin
    # even with the ~0.42x heavy-element under-read of the standardless
    # Cliff-Lorimer quantification.
    # 2026-08-05: _ABSENT lowered 0.02 -> 0.012. At 2 at% the threshold sat INSIDE
    # the Mn distribution of a real alpha-Al(Fe,Mn)Si particle (Mn 1.8-2.2 at%
    # throughout), so measurement noise punched 394 holes into a 1607-px grain —
    # 99 separate patches, 60 % of them deep in the interior, not a rim effect.
    # The pattern evidence there is identical to the core (CI 0.635 vs 0.636).
    # Matrix and Si-boundary pixels sit at Mn ~0.3 at%, so there is a wide empty
    # gap between "absent" and "present"; 1.2 at% sits in the middle of it
    # instead of on the edge of a noisy distribution. Measured on ProbeB:
    # alpha grain retained 79.8 % -> 100 %, while the Si-rim veto and the Al
    # matrix stay at 100 % vetoed.
    #
    # 2026-09-12: _ABSENT raised 0.012 -> 0.020. IT IS AN ABSOLUTE At.% FLOOR,
    # SO IT IS TIED TO THE QUANTIFICATION SCALE, and the k-factors were
    # corrected that day (tasks/eds-quantification-wrong-2026-09-10.md). The
    # 2026-08-05 note above places it by naming its two populations -- matrix
    # and Si-boundary pixels "at Mn ~0.3 at%" against a real alpha particle
    # "at Mn 1.8-2.2". Both moved. Re-measured on the corrected scale over the
    # 268x201 reference scan (tasks/eds_prior_audit/06_veto_constants.py), Mn
    # At.% per region of the pattern-only 4-phase map:
    #
    #     Al matrix    p05 0.52   median 0.78   p95 1.55
    #     Al7Cu2Fe     p05 0.97   median 1.37   p95 1.81
    #     beta         p05 1.76   median 2.24   p95 2.69
    #     alpha        p05 2.36   median 3.51   p95 5.01
    #
    # The empty gap is now 1.55 .. 2.36, and 2.0 at% is its middle -- the same
    # placement rule as before, on the numbers that exist now. Sweep of the
    # per-pixel top-1 rate of the prior used as a classifier over the four
    # reference phases (07_joint_sweep.py; the last column is the 2026-08-04
    # defect above, the share of Si-particle NEIGHBOUR pixels handed to an
    # Fe phase on the Al/Si file):
    #
    #   _ABSENT   alpha  Al7Cu2Fe   beta   matrix    ALL   Si rim -> Fe
    #    0.012    89.37    99.92   99.41   82.53   88.18      20.34 %
    #    0.016    89.37    99.92   99.41   83.41   88.64      20.34 %
    #    0.020    89.37    99.92   99.41   83.89   88.90       6.78 %   <-- chosen
    #    0.025    89.37    99.92   99.41   84.64   89.29       3.39 %
    #    0.030    42.30    99.92   99.90   85.90   78.71
    #
    # 0.025 scores better on every column and is NOT chosen: alpha's own p05 is
    # 2.36, so 2.5 at% sits inside the distribution it must not cut, and one
    # step further (3.0) costs 47 points of alpha. That is exactly the edge the
    # 2026-08-05 note was written about. 2.0 is 1.29x below it and 1.29x above
    # the matrix p95.
    #
    # _MAJOR_REQ stays at 0.05, measured and deliberate. Lowering it to 0.02
    # buys 1.6 points of matrix (06_veto_constants.py) by making alpha's 2.7 at%
    # Mn a REQUIRED element -- and that is wrong for this phase: Fe and Mn share
    # one site, so alpha in a Mn-free alloy is still alpha. A veto must not
    # demand an element the structure lets the alloy choose.
    _MAJOR_REQ, _ABSENT = 0.05, 0.020
    for el, frac in q.items():
        if frac >= _MAJOR_REQ and p.get(el, 0.0) < _ABSENT:
            return min(score, 0.05)

    # Present-but-missing penalty (the other asymmetric direction). A pixel
    # that measures a SIGNIFICANT non-trace element the phase does NOT contain
    # cannot be that phase: e.g. an intermetallic particle pixel reads Al~75%
    # Fe~10% Si~10% (the interaction volume mixes particle + surrounding
    # matrix), and plain L1 still rewards Al (the dominant element) over the
    # Fe-Al-Si intermetallic — so Crystal Hint kept suggesting "Al" on real
    # particles (the user's recurring complaint). Damp a phase by the fraction
    # of significant (>=5 at%) measured pixel composition it leaves
    # unexplained. Trace levels (matrix Fe/Si/Mn ~1 at%) are below the 5%
    # cutoff, so matrix pixels are untouched and Al still wins there; only
    # genuine multi-element particle pixels demote the pure-matrix phase and
    # let the intermetallic that DOES contain Fe/Si/Mn rank up.
    #
    # 2026-09-12: 0.05 -> 0.02. ANOTHER ABSOLUTE At.% FLOOR, so the k-factor
    # correction moved it too. It buys the alpha region and it is NOT free, and
    # the trade is this:
    #
    # WHAT IT BUYS. The 5 at% version could no longer charge a phase for an
    # element that is plainly there. That was the alpha/beta flip: the alpha
    # region measures Mn 3.5 at% against a 0.8 at% background, beta contains no
    # Mn at all, and beta was not charged for it.
    #
    # WHAT IT COSTS, AND WHY THE MATRIX FALLS. The original intent -- "trace
    # levels (matrix Fe/Si/Mn ~1 at%) are below the cutoff" -- is only half met
    # at 2 at%. The reference scan's Al matrix reads Fe 1.81 and Mn 0.86, which
    # do clear, but also Si 4.85, Cu 2.79 and Zn 2.81. At the old floor NOTHING
    # on a mean matrix pixel was charged (Si at 4.85 sat just under 5); at 2 at%
    # Al.cif is charged about 10.5 % unexplained, of which Cu and Zn are 5.6
    # points. Those are exactly the two channels that do NOT reproduce across
    # scans of one sample (Cu spreads 36-58 %, Zn 16-39 %; the Zn window is
    # contaminated by Cu L emission), so part of this cost is paid in the
    # least trustworthy currency available. It is 1.14 of the 2.01 points the
    # Al-matrix region loses; the rest is the interaction-volume halo, which
    # tasks/eds_prior_audit/14_matrix_misses.py shows is real (the pixels handed
    # over have lower band contrast, p = 1.6e-223, Fe 5.35 against 0.80 at%, and
    # sit a median 6 px from a particle). Weighting Cu and Zn down is the proper
    # repair and is a feature, not a constant.
    #
    # Sweep of the per-pixel top-1 rate over the four reference phases
    # (tasks/eds_prior_audit/07_joint_sweep.py, at _ABSENT = 0.020):
    #
    #   floor   alpha  Al7Cu2Fe   beta   matrix    ALL
    #    0.05   43.84    99.92   99.90   85.03   78.63
    #    0.03   82.77    99.92   99.90   84.89   87.87
    #    0.02   89.37    99.92   99.41   83.89   88.90   <-- chosen
    #
    # Below 0.02 the floor stops deciding: the enrichment rule below takes over
    # wherever a background is supplied, and 0.015 / 0.012 measure the same
    # 89.37 % (06_veto_constants.py). 0.02 is the lowest value that is still a
    # statement about At.%, and it is above every trace level in both test
    # datasets.
    _PRESENT_SIG = 0.02
    unexplained = sum(
        frac for el, frac in p.items()
        if frac >= _PRESENT_SIG and q.get(el, 0.0) < _ABSENT
    )
    if unexplained > 0.0:
        score *= max(0.0, 1.0 - unexplained)
    return score
