"""User-authored rules that decide which phases may compete for a region.

Why this exists, in the user's words: "EDS signal sometimes contains
ambiguities (measurement of surroundings if features are smaller than the
interaction volume of EDS). These ambiguities seem to tamper with the automatic
selection of the correct phase candidates in some cases."

Measured on SampleB, that is true but LOCAL: of seven structures, two decide
between their top two candidates on under 1 at% of margin (0.36 and 0.20),
while the other five are clear by 1.3 to 3.7. Rules therefore exist to settle
the close calls, and they are opt-in per phase so they cannot disturb the calls
that were already right.

WHAT A RULE MEANS. Exactly one thing: **a phase may compete here, or it may
not.** Not a weight, not a preference. That was a deliberate correction after a
review. The first design multiplied the score by a soft factor, justified by a
measurement that a plausible band ("Si 40-100 at%") keeps only 46 % of a real
particle — but that measurement assumed a rule DISCARDS pixels. It does not: a
pixel whose Si rule fails simply goes to the next candidate. The number
measured the weakness of the rule TYPE, not the cost of hard semantics. And a
soft weight cannot produce the one diagnostic that makes this feature honest --
"no candidate passed; nearest was Si.cif, blocked by Si >= 40, measured 25.3".

THE UNITS ARE THE TRAP. Ranges are evaluated on at% RENORMALISED over the
scored elements, never on raw at%. Measured on SampleB the raw at% total ranges
91.30 to 100.00, so an identical composition drifts up to ~8.7 % relative on
total yield alone, and a raw-at% band would pass in one region and fail in
another for a reason that is not chemistry. `chemistry_score._renormalise`
returns FRACTIONS; at% is that times 100. Getting this backwards is what once
produced "Al 49.68x enriched" on a map whose background is aluminium.

WHY ENRICHMENT IS OFFERED alongside the content ranges the user asked for: it
is simply the better instrument on this data, and the difference is large.
"Si >= 2x the map's own background" keeps 96.3 % of a real particle and admits
5.3 % of the map; "Si >= 40 at%" keeps 46.3 % and admits 0.9 %. Enrichment
carries no k-factor and no stoichiometry assumption because the bar is measured
per dataset — which also makes it the MORE portable rule between samples, not
the less: the factor is stored, the background is re-measured. Content ranges
stay necessary because enrichment can say nothing about the matrix element
itself (aluminium is 0.5x by construction), so Al-bearing phases like Al6Fe
still need them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.api.services.chemistry_score import (
    _ABSENT, _RATIO_FLOOR, _renormalise, background_levels,
)


@dataclass(frozen=True)
class ElementRange:
    """`Mg between 10 and 50 at%`. Either bound may be omitted."""

    element: str
    min_at_pct: Optional[float] = None
    max_at_pct: Optional[float] = None


@dataclass(frozen=True)
class RatioRange:
    """`Mg:Si between 0.5 and 3.0`, on renormalised at%.

    A ratio cancels any common factor, so renormalising changes nothing about
    the answer; it matters only that both legs come from one vector with C and
    O already dropped.
    """

    numerator: str
    denominator: str
    min_ratio: Optional[float] = None
    max_ratio: Optional[float] = None


@dataclass(frozen=True)
class EnrichmentRange:
    """`Si at least 2x the map's own background`."""

    element: str
    min_factor: Optional[float] = None
    max_factor: Optional[float] = None


@dataclass(frozen=True)
class PhaseRule:
    """What a region must satisfy for one phase to be allowed to compete.

    Keyed on ``phase_key`` — the ``CifPhaseEntry.key`` — and never on a phase
    index. Indices are positions in the candidate list, which changes with
    which phases are ticked and which elements the file measured; the store
    already carries hand-set pixels by key for exactly this reason.
    """

    phase_key: str
    phase_formula: str = ""
    elements: Sequence[ElementRange] = field(default_factory=tuple)
    ratios: Sequence[RatioRange] = field(default_factory=tuple)
    enrichment: Sequence[EnrichmentRange] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        return not (self.elements or self.ratios or self.enrichment)


@dataclass(frozen=True)
class RuleSet:
    """Everything the user decided, for one classification run."""

    name: str = ""
    #: Phases allowed to take part at all. ``None`` means "every candidate".
    phase_keys: Optional[Sequence[str]] = None
    #: Overrides ``infer_matrix_element``. The matrix is a sample property, so
    #: it is declared once for the map rather than per phase.
    matrix_elements: Sequence[str] = field(default_factory=tuple)
    rules: Sequence[PhaseRule] = field(default_factory=tuple)

    def rule_for(self, phase_key: str) -> Optional[PhaseRule]:
        for r in self.rules:
            if r.phase_key == phase_key:
                return r
        return None

    @property
    def is_empty(self) -> bool:
        return not self.rules and self.phase_keys is None and not self.matrix_elements


# --- evaluation -------------------------------------------------------------

@dataclass
class RuleOutcome:
    """Per-pixel verdict for one rule, plus why it says what it says."""

    allowed: np.ndarray            # bool, one per pixel
    #: Human-readable reason the rule blocked, when it blocked everywhere.
    #: Written for the user, not for a log: "Si >= 40 at% (measured 25.3)".
    reason: Optional[str] = None
    #: True when a rule could not be decided at all — e.g. a ratio whose legs
    #: sit below the detection floor. Such a rule does NOT pass: we cannot
    #: assert what we could not measure. It is reported separately so the UI
    #: can say "not evaluated" rather than "failed".
    undecidable: bool = False


def renormalised_at_pct(
    at_pct_per_element: Dict[str, np.ndarray],
) -> Tuple[List[str], np.ndarray]:
    """Composition as at% over the scored elements. C and O are dropped.

    ``_renormalise`` returns fractions; at% is that times 100. Every rule in
    this module works in at%, because that is what the user types and what the
    inspector shows.
    """
    els, frac, _total = _renormalise(at_pct_per_element)
    return els, frac * 100.0


def _series(els: Sequence[str], at_pct: np.ndarray, element: str) -> Optional[np.ndarray]:
    try:
        return at_pct[els.index(element)]
    except ValueError:
        return None


def _fmt(lo: Optional[float], hi: Optional[float], unit: str = " at%") -> str:
    if lo is not None and hi is not None:
        return f"{lo:g}-{hi:g}{unit}"
    if lo is not None:
        return f">= {lo:g}{unit}"
    if hi is not None:
        return f"<= {hi:g}{unit}"
    return "any"


def evaluate_rule(
    rule: PhaseRule,
    at_pct_per_element: Dict[str, np.ndarray],
    background: Optional[Dict[str, float]] = None,
) -> RuleOutcome:
    """Which pixels this phase is allowed to compete for.

    Every clause must hold — there is no OR, and deliberately no expression
    language. A materials scientist writing "Mg 10-50 at% and Mg:Si 0.5-3.0"
    means both, and an editor that can express more than that is an editor
    whose result nobody can predict.
    """
    els, at_pct = renormalised_at_pct(at_pct_per_element)
    n_px = at_pct.shape[1] if at_pct.size else 0
    if n_px == 0:
        return RuleOutcome(np.zeros(0, dtype=bool))
    allowed = np.ones(n_px, dtype=bool)
    if rule.is_empty:
        return RuleOutcome(allowed)

    reasons: List[str] = []
    undecidable = False

    # --- element content ----------------------------------------------------
    for er in rule.elements:
        vals = _series(els, at_pct, er.element)
        if vals is None:
            # The rule names an element this dataset did not measure. Refusing
            # is the honest answer: we cannot assert a composition we do not
            # have. Reported so the UI can say so rather than showing a phase
            # that silently never appears.
            undecidable = True
            reasons.append(f"{er.element} not measured here")
            allowed = np.zeros(n_px, dtype=bool)
            continue
        ok = np.ones(n_px, dtype=bool)
        if er.min_at_pct is not None:
            ok &= vals >= er.min_at_pct
        if er.max_at_pct is not None:
            ok &= vals <= er.max_at_pct
        if not ok.all():
            reasons.append(
                f"{er.element} {_fmt(er.min_at_pct, er.max_at_pct)} "
                f"(measured {float(np.median(vals)):.1f})"
            )
        allowed &= ok

    # --- element ratios -----------------------------------------------------
    for rr in rule.ratios:
        num = _series(els, at_pct, rr.numerator)
        den = _series(els, at_pct, rr.denominator)
        if num is None or den is None:
            undecidable = True
            missing = rr.numerator if num is None else rr.denominator
            reasons.append(f"{missing} not measured here")
            allowed = np.zeros(n_px, dtype=bool)
            continue
        # Both legs must clear the detection floor AND be present. Measured on
        # SampleB: where either leg is below the floor the ratio spans 0 to
        # 2.5e10, and a naive band check passes 1.7 % of pure noise. A ratio we
        # cannot measure does not pass.
        floor = _RATIO_FLOOR * 100.0
        absent = _ABSENT * 100.0
        measurable = (num >= floor) & (den >= floor) & (num >= absent) & (den >= absent)
        ratio = np.divide(num, den, out=np.zeros_like(num),
                          where=den > 0)
        ok = measurable.copy()
        if rr.min_ratio is not None:
            ok &= ratio >= rr.min_ratio
        if rr.max_ratio is not None:
            ok &= ratio <= rr.max_ratio
        if not measurable.any():
            undecidable = True
            reasons.append(
                f"{rr.numerator}:{rr.denominator} not measurable "
                f"(both below {absent:.1f} at%)"
            )
        elif not ok.all():
            med = float(np.median(ratio[measurable])) if measurable.any() else float("nan")
            reasons.append(
                f"{rr.numerator}:{rr.denominator} "
                f"{_fmt(rr.min_ratio, rr.max_ratio, '')} (measured {med:.2f})"
            )
        allowed &= ok

    # --- enrichment over the map's own background ---------------------------
    if rule.enrichment:
        bg = background if background is not None else background_levels(at_pct_per_element)
        for en in rule.enrichment:
            vals = _series(els, at_pct, en.element)
            base = float((bg or {}).get(en.element, 0.0) or 0.0) * 100.0
            if vals is None or base <= 1e-9:
                undecidable = True
                reasons.append(
                    f"{en.element} has no background to compare against")
                allowed = np.zeros(n_px, dtype=bool)
                continue
            factor = vals / base
            ok = np.ones(n_px, dtype=bool)
            if en.min_factor is not None:
                ok &= factor >= en.min_factor
            if en.max_factor is not None:
                ok &= factor <= en.max_factor
            if not ok.all():
                reasons.append(
                    f"{en.element} {_fmt(en.min_factor, en.max_factor, 'x background')} "
                    f"(measured {float(np.median(factor)):.2f}x)"
                )
            allowed &= ok

    return RuleOutcome(
        allowed=allowed,
        reason=" · ".join(reasons) if reasons else None,
        undecidable=undecidable,
    )


def gate_scores(
    scores: np.ndarray,
    rule: Optional[PhaseRule],
    at_pct_per_element: Dict[str, np.ndarray],
    background: Optional[Dict[str, float]] = None,
    blocked_score: float = 0.0,
) -> Tuple[np.ndarray, Optional[RuleOutcome]]:
    """Apply one phase's rule to that phase's scores.

    A gate rather than a re-score, matching the veto convention already in
    ``chemistry_score``: the calibrated constants stay untouched and the rule
    only decides eligibility. ``blocked_score`` is 0.0 rather than a small
    number because a blocked phase must lose to an unblocked one even when the
    unblocked one scores badly — that is what "may not compete" means.
    """
    if rule is None or rule.is_empty:
        return scores, None
    outcome = evaluate_rule(rule, at_pct_per_element, background)
    if outcome.allowed.shape != np.shape(scores):
        # Shapes disagreeing means the rule was evaluated against a different
        # population than the scores. Refusing loudly beats gating the wrong
        # pixels.
        raise ValueError(
            f"rule outcome shape {outcome.allowed.shape} does not match "
            f"score shape {np.shape(scores)}"
        )
    return np.where(outcome.allowed, scores, blocked_score), outcome


# --- (de)serialisation ------------------------------------------------------

def rule_set_from_dict(payload: Optional[dict]) -> Optional[RuleSet]:
    """Build a RuleSet from the frontend's JSON. Unknown keys are ignored."""
    if not payload:
        return None
    rules: List[PhaseRule] = []
    for r in payload.get("rules") or []:
        key = r.get("phase_key")
        if not key:
            continue
        rules.append(PhaseRule(
            phase_key=str(key),
            phase_formula=str(r.get("phase_formula") or ""),
            elements=tuple(
                ElementRange(str(e["element"]),
                             _num(e.get("min_at_pct")), _num(e.get("max_at_pct")))
                for e in (r.get("elements") or []) if e.get("element")
            ),
            ratios=tuple(
                RatioRange(str(x["numerator"]), str(x["denominator"]),
                           _num(x.get("min_ratio")), _num(x.get("max_ratio")))
                for x in (r.get("ratios") or [])
                if x.get("numerator") and x.get("denominator")
            ),
            enrichment=tuple(
                EnrichmentRange(str(x["element"]),
                                _num(x.get("min_factor")), _num(x.get("max_factor")))
                for x in (r.get("enrichment") or []) if x.get("element")
            ),
        ))
    keys = payload.get("phase_keys")
    return RuleSet(
        name=str(payload.get("name") or ""),
        phase_keys=(tuple(str(k) for k in keys) if keys is not None else None),
        matrix_elements=tuple(str(m) for m in (payload.get("matrix_elements") or [])),
        rules=tuple(rules),
    )


def rule_set_to_dict(rs: Optional[RuleSet]) -> Optional[dict]:
    if rs is None:
        return None
    return {
        "name": rs.name,
        "phase_keys": list(rs.phase_keys) if rs.phase_keys is not None else None,
        "matrix_elements": list(rs.matrix_elements),
        "rules": [
            {
                "phase_key": r.phase_key,
                "phase_formula": r.phase_formula,
                "elements": [
                    {"element": e.element, "min_at_pct": e.min_at_pct,
                     "max_at_pct": e.max_at_pct} for e in r.elements
                ],
                "ratios": [
                    {"numerator": x.numerator, "denominator": x.denominator,
                     "min_ratio": x.min_ratio, "max_ratio": x.max_ratio}
                    for x in r.ratios
                ],
                "enrichment": [
                    {"element": x.element, "min_factor": x.min_factor,
                     "max_factor": x.max_factor} for x in r.enrichment
                ],
            }
            for r in rs.rules
        ],
    }


def _num(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


# --- manual region definitions ----------------------------------------------
#
# The clause types above answer "may this phase compete for this pixel". The
# same three clause types answer a different question just as well: "do these
# pixels belong together". That second question is the one the automatic
# grouping sometimes gets wrong, and a user reported exactly how:
#
#   "pure Silicon and AlFeMnSi phases are not possible to separate from each
#    other because the automatic definition of regions based on composition
#    lumps them into the same category. The high amount of Al background
#    signal is likely the issue here"
#
# That is a real property of the feature space, not a bug in the fit. The
# grouping measures distance over the renormalised composition, so a matrix
# element sitting at ~85 at% dominates the distance and a few at% of iron —
# the thing that actually tells the two apart — is a rounding error next to
# it. No choice of ``k`` fixes that, because ``k`` cannot say WHICH element
# carries the distinction.
#
# Two levers follow, and they are different tools:
#   * element weights (in ``eds_clustering``) tell the automatic grouping
#     which elements to care about, and keep it automatic;
#   * a region definition here removes the question from the fit entirely —
#     the user states the composition window and the pixels inside it are a
#     region, whatever the distance metric thinks.


def _binds(clause) -> bool:
    """Does this clause actually constrain anything?

    A clause with both bounds unset is what the editor's "Add" button
    produces before the user types a number. It is a half-written edit, and
    reading it as a real clause made a definition look non-empty while
    constraining nothing - so it claimed EVERY pixel and one definition ate
    the map. Counting clauses is not enough; they have to bite.
    """
    for attr in ("min_at_pct", "max_at_pct", "min_ratio", "max_ratio",
                 "min_factor", "max_factor"):
        if getattr(clause, attr, None) is not None:
            return True
    return False


@dataclass(frozen=True)
class RegionDefinition:
    """One hand-declared region: a name plus the composition window for it.

    Deliberately the same three clause types as :class:`PhaseRule`, so
    :func:`evaluate_rule` evaluates both — one editor, one semantics, one set
    of surprises. The difference is what the result is used for: a rule gates
    a phase's eligibility, a definition claims pixels.

    ``phase_key`` is optional. Empty means "group these pixels, then let the
    library matcher name them as usual"; set means the user has already
    decided the name and the matcher must not overrule it.
    """

    name: str
    elements: Sequence[ElementRange] = field(default_factory=tuple)
    ratios: Sequence[RatioRange] = field(default_factory=tuple)
    enrichment: Sequence[EnrichmentRange] = field(default_factory=tuple)
    phase_key: str = ""

    @property
    def is_empty(self) -> bool:
        # Bound-less clauses do not count - see `_binds`.
        return not any(_binds(c) for c in
                       (*self.elements, *self.ratios, *self.enrichment))


@dataclass
class RegionAssignment:
    """Result of applying definitions to a map."""

    #: One label per pixel: index into ``defs``, or -1 for unclaimed.
    labels: np.ndarray
    #: Pixels each definition claimed, in order.
    counts: List[int]
    #: Pixels a later definition wanted but an earlier one had already taken.
    overlaps: List[int]
    #: Per definition: why it claims nothing, when it claims nothing.
    #: Human-readable English, always present as a fallback.
    reasons: List[Optional[str]]
    #: The same answer as a stable code, so the UI can say it in the user's
    #: language. `None` where the definition claimed something; the code
    #: ``"rule"`` means the text in ``reasons`` is the clause-level reason
    #: from :func:`evaluate_rule` and carries element names, so it has no
    #: fixed translation and the prose is the answer.
    reason_codes: List[Optional[str]] = field(default_factory=list)


def region_labels(
    defs: Sequence[RegionDefinition],
    at_pct_per_element: Dict[str, np.ndarray],
    background: Optional[Dict[str, float]] = None,
) -> RegionAssignment:
    """Claim pixels for hand-declared regions, first definition wins.

    Order is priority, and it is the whole conflict-resolution story. An
    overlap is normal — "Si > 40" and "Fe 2-10" can both hold on a rim pixel
    — and resolving it by anything cleverer than declared order (smallest
    region, best fit, nearest mean) would make the outcome depend on the data
    rather than on what the user wrote. The counts of what each definition
    lost to an earlier one are reported instead, so the UI can show it.

    An empty definition claims NOTHING rather than everything. A window with
    no clauses in it is a half-finished edit, and reading it as "the whole
    map" would wipe the map on the way to writing the first clause.
    """
    els, at_pct = renormalised_at_pct(at_pct_per_element)
    n_px = at_pct.shape[1] if at_pct.size else 0
    labels = np.full(n_px, -1, dtype=np.int32)
    counts: List[int] = []
    overlaps: List[int] = []
    reasons: List[Optional[str]] = []
    codes: List[Optional[str]] = []
    if n_px == 0:
        return RegionAssignment(labels, [0] * len(defs), [0] * len(defs),
                                [None] * len(defs), [None] * len(defs))

    for i, d in enumerate(defs):
        if d.is_empty:
            counts.append(0)
            overlaps.append(0)
            reasons.append("no composition window set yet")
            codes.append("empty")
            continue
        outcome = evaluate_rule(d, at_pct_per_element, background=background)
        want = np.asarray(outcome.allowed, dtype=bool)
        free = labels < 0
        take = want & free
        labels[take] = i
        counts.append(int(take.sum()))
        overlaps.append(int((want & ~free).sum()))
        if not take.any():
            if outcome.reason:
                reasons.append(outcome.reason)
                codes.append("rule")
            elif want.any():
                reasons.append("every matching pixel was already claimed above")
                codes.append("taken")
            else:
                reasons.append("no pixel matches this window")
                codes.append("nomatch")
        else:
            reasons.append(None)
            codes.append(None)
    return RegionAssignment(labels, counts, overlaps, reasons, codes)


def region_defs_from_list(payload) -> List[RegionDefinition]:
    """Parse region definitions from the wire. Unusable entries are dropped."""
    out: List[RegionDefinition] = []
    for i, item in enumerate(payload or []):
        if not isinstance(item, dict):
            continue
        out.append(RegionDefinition(
            name=str(item.get("name") or f"Region {i + 1}"),
            elements=tuple(
                ElementRange(str(e.get("element")), _num(e.get("min_at_pct")),
                             _num(e.get("max_at_pct")))
                for e in item.get("elements") or [] if e.get("element")),
            ratios=tuple(
                RatioRange(str(r.get("numerator")), str(r.get("denominator")),
                           _num(r.get("min_ratio")), _num(r.get("max_ratio")))
                for r in item.get("ratios") or []
                if r.get("numerator") and r.get("denominator")),
            enrichment=tuple(
                EnrichmentRange(str(x.get("element")), _num(x.get("min_factor")),
                                _num(x.get("max_factor")))
                for x in item.get("enrichment") or [] if x.get("element")),
            phase_key=str(item.get("phase_key") or ""),
        ))
    return out


def region_defs_to_list(defs: Sequence[RegionDefinition]) -> List[dict]:
    """Round-trip counterpart of :func:`region_defs_from_list`."""
    out: List[dict] = []
    for d in defs or []:
        out.append({
            "name": d.name,
            "phase_key": d.phase_key,
            "elements": [
                {"element": e.element, "min_at_pct": e.min_at_pct,
                 "max_at_pct": e.max_at_pct} for e in d.elements],
            "ratios": [
                {"numerator": r.numerator, "denominator": r.denominator,
                 "min_ratio": r.min_ratio, "max_ratio": r.max_ratio}
                for r in d.ratios],
            "enrichment": [
                {"element": x.element, "min_factor": x.min_factor,
                 "max_factor": x.max_factor} for x in d.enrichment],
        })
    return out
