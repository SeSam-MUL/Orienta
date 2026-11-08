"""Find or refine the pattern centre (PC) of a stack of calibration patterns.

Both searches minimise ONE cost over ALL calibration patterns at once (``pc_cost.JointPcCost``,
PyEBSDIndex's pattern-centre cost; Pang, Larsen and Schuh 2020 call this the joint fit). The
older "Global PC Refine" optimised every pattern on its own from the current PC and averaged the
nine answers; started at the placeholder (0.5, 0.5, 0.5) that landed 0.12-0.31 away from the
file's own PC on five files from three makers, the joint cost with a wide swarm 0.007-0.036.

* ``mode="find"`` ("Find PC"): no start is needed. A particle swarm over a wide box
  (``WIDE_CENTRE`` +- ``WIDE_LIMIT``, the box the measurement used) and then Nelder-Mead on the same
  cost. A result that lies on an edge of the box means the true PC may lie outside it: the box is
  moved to the result and the search repeated (at most ``MAX_RECENTRES`` times). When the patterns
  do not fix the result clearly (see ``confidence``; with no threshold set nothing is called clear,
  so the second box is always tried), the same search is made in a second box for
  a detector farther from the sample (``FAR_CENTRE``). Its result replaces the first one only when it
  is cheaper AND either clear itself or cheaper by more than the ring excess of the first result (on a
  nearly flat landscape "cheaper" alone is a coin toss; with no threshold only the second condition can
  hold); both candidates are reported. The candidate that is not the result is also reported as
  ``second_position`` when it lies more than ``SECOND_POSITION_DIST`` away in x, y or z and costs at most
  ``SECOND_POSITION_COST`` times the result's cost (``second_position_of``). A result still
  on an edge of its box is reported as such, never hidden.
* ``mode="refine"`` ("Global PC Refine"): starts at ``start`` (the PC the page holds). With
  ``method="Nelder-Mead"`` only the local polish; with ``method="PSO"`` a swarm in ``start`` +-
  ``search_limit`` first. A refine refuses (``SearchError``) an unknown ``method`` (``badMethod``), a ``start``
  that is not three numbers inside the physical box (``badStart``) and a ``search_limit`` outside [1e-4, 1]
  (``badSearchLimit``), before it builds anything. Every Nelder-Mead polish stays inside the physical box.

Afterwards, for both: ``confidence`` (does the stack fix this PC?) and the PC each calibration
pattern prefers on its own, started at the joint result (``per_pattern``, ``spread``): they say
how far the patterns disagree, they are NOT averaged into the answer.

``confidence`` answers one question: do the patterns fix this PC, or would a PC picked at random
fit them about as well? ``contrast`` is how far below the TYPICAL cost of plausible PCs (the lower
quartile, ``TYPICAL_PERCENTILE``, over 60 quasi-random points of the wide box) the result lies, as a fraction
of that typical cost. The lower quartile and not the median, because the cost is a staircase: a single pattern
fails at some PCs (its bands are not found, so it adds a fixed penalty), and the sum over the stack then takes
only a few levels. The median of the 60 box points jumps from one level to the next as soon as more than half
of the points lie on the upper level, and the contrast of the same result jumps with it. The lower quartile
sits on the level of the points that fit best and moves far less when a few points change level.
A result is "clear" from ``CONTRAST_MIN`` upward. ``CONTRAST_MIN`` is one value and ``None`` means "no
threshold": the measurement kept in the repository with its log (the controls: pure noise and a search
box without the minimum; rendered patterns; real files from three makers) gives no number that separates
the controls from the real files. While it is ``None`` the search does not judge: ``state`` is
"notJudged" and never "clear" or "unclear", ``contrast``, ``typical_cost`` and the ring values are
reported as always, and the reasons "notAMinimum" (a ring point is cheaper) and "noContrast" (the contrast
cannot be computed because a cost is not a number) stay facts. A number set later brings the judgement back
without any other change.

The ring statistics (the cost on rings of +-0.01 and +-0.02 around the result: ``excess``, ``excess_small``,
``growth``) are reported but do NOT decide: on the earlier scratch measurement pure noise had the same ring
excess as good real data, and on rough real landscapes the growth from one ring to the next was close to 1
for results within 0.02 of the file's PC (the committed log carries these columns too, so the claim can
be re-read).

The particle swarm of PyEBSDIndex draws from the global random generator and from a scrambled
Halton sequence it cannot be seeded for, so two runs differ in the last digits. The tests accept
a tolerance, never an exact number.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from backend.api.services.pc_cost import JointPcCost

logger = logging.getLogger(__name__)

#: The search box of "Find PC": centre and half-width per component (x, y, z), Bruker numbers.
#: The values the measurement behind the feature used:
#: they hold Oxford's PCz 0.85, EDAX's 0.64-0.70, NORDIF's 0.56 and PCy 0.26-0.50.
WIDE_CENTRE = (0.5, 0.4, 0.7)
WIDE_LIMIT = (0.2, 0.3, 0.35)
#: The second box, tried when the first result is unclear: a detector farther from the sample
#: (PCz up to 1.55; the kikuchipy paper's Al-10Si notebook starts at PCz 1.25).
FAR_CENTRE = (0.5, 0.3, 1.2)
#: A box is never moved outside what a detector can be: x in [0.05, 0.95], y in [0, 1], z in [0.2, 1.6].
PHYSICAL_LOW = (0.05, 0.0, 0.2)
PHYSICAL_HIGH = (0.95, 1.0, 1.6)
#: A result this close to an edge of its box (or outside it) is "on the edge": at most EDGE_TOL, and on an axis
#: where the box is narrow (a box clipped to what a detector can be) at most EDGE_FRACTION of its half width.
EDGE_TOL = 0.02
EDGE_FRACTION = 0.1
#: A Nelder-Mead start is moved this far inside the physical box (see ``_polish``).
EDGE_INSET = 1e-3
#: The smallest ``search_limit`` of a refine; below ~5e-17 start +- limit is start itself (lo == hi in the swarm's box).
MIN_SEARCH_LIMIT = 1e-4
MAX_RECENTRES = 2
N_PARTICLES = 30
N_ITERATIONS = 60
#: Nelder-Mead as in PyEBSDIndex's own ``optimize`` (the measurement's setting).
NM_OPTIONS = {"fatol": 1e-5, "xatol": 1e-4, "maxiter": 600}
NM_OPTIONS_SINGLE = {"fatol": 1e-5, "xatol": 1e-4, "maxiter": 200}
#: A cost difference below this is the optimiser's noise: the descent takes a new result only when it is cheaper
#: by more, and the ring calls a result "notAMinimum" only when a ring point is cheaper by more.
POLISH_TOL = NM_OPTIONS["fatol"]
#: The two rings of the confidence check and the number of points of the box sample.
RING_SMALL = 0.01
RING = 0.02
N_REFERENCE = 60
#: The "typical" cost of the contrast is this percentile of the cost over the ``N_REFERENCE`` box points
#: (the lower quartile; the median jumps on a cost with levels, see the module docstring).
TYPICAL_PERCENTILE = 25
#: A result is "clear" when its contrast is at least CONTRAST_MIN. A number is a decision; the measurement
#: in the repository gives none. None is a valid state, not an error: the search then does not judge
#: (state "notJudged", see the module docstring).
CONTRAST_MIN = None
#: "Find PC" needs this many calibration patterns (NORDIF's calibration set has exactly five).
MIN_PATTERNS_FIND = 5
#: A second position is a separate position more than this far from the result in x, y or z (a reasoned default,
#: not fitted: the bar the measurements call "the same PC"). Measured: the warning marks a flat scan, where the
#: patterns hardly tell two positions apart, not a wrong result. It fired on 18 of 20 runs of the one flat file and
#: on none of 12 runs of six other files, and in none of the 8 wrong runs did the second position lie within this
#: distance of the file's PC.
SECOND_POSITION_DIST = 0.05
#: ... whose cost is at most this many times the cost of the result (a reasoned default, not fitted: a cost
#: within ten percent is not clearly worse).
SECOND_POSITION_COST = 1.10

AXES = ("x", "y", "z")


class SearchError(ValueError):
    """A search that cannot start. ``code`` is what the interface translates on."""

    def __init__(self, code: str, message: str, **params):
        super().__init__(message)
        self.code = code
        self.params = params


@dataclass
class SearchReport:
    pc: tuple
    cost: float
    start: Optional[tuple]
    mode: str
    method: str
    n_patterns: int
    evaluations: int = 0
    box: Optional[dict] = None
    recentred: int = 0
    far_box_used: bool = False
    first_box_pc: Optional[tuple] = None
    far_box_pc: Optional[tuple] = None
    far_box_cost: Optional[float] = None
    on_edge: List[str] = field(default_factory=list)
    confidence: dict = field(default_factory=dict)
    per_pattern: Optional[np.ndarray] = None
    spread: Optional[dict] = None
    #: A refine only: the contrast of the PC it started from, ``1 - cost(start) / typical`` with the same
    #: ``typical`` as ``confidence["contrast"]`` (one cost evaluation more). ``None`` for a find, and when the
    #: cost at the start is not a number.
    contrast_at_start: Optional[float] = None
    #: A find only: the end of the box that was NOT adopted, ``{"pc": (x, y, z), "cost": float}``, when it is a
    #: separate position that fits almost as well (see ``second_position_of``); else ``None``.
    second_position: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "pc": [float(v) for v in self.pc], "cost": _finite_or_none(self.cost),
            "start": None if self.start is None else [float(v) for v in self.start],
            "mode": self.mode, "method": self.method, "n_patterns": int(self.n_patterns),
            "evaluations": int(self.evaluations), "box": self.box, "recentred": int(self.recentred),
            "far_box_used": bool(self.far_box_used),
            "first_box_pc": None if self.first_box_pc is None else [float(v) for v in self.first_box_pc],
            "far_box_pc": None if self.far_box_pc is None else [float(v) for v in self.far_box_pc],
            "far_box_cost": _finite_or_none(self.far_box_cost),
            "on_edge": list(self.on_edge),
            "confidence": self.confidence,
            "per_pattern": None if self.per_pattern is None
            else [[_finite_or_none(v) for v in row] for row in self.per_pattern],
            "spread": self.spread,
            "contrast_at_start": _finite_or_none(self.contrast_at_start),
            "second_position": _second_dict(self.second_position),
        }


# ------------------------------------------------------------------ pieces

def _clip_box(centre, limit):
    lo = np.maximum(np.asarray(centre, float) - limit, PHYSICAL_LOW)
    hi = np.minimum(np.asarray(centre, float) + limit, PHYSICAL_HIGH)
    return lo, hi


def _swarm(cost: JointPcCost, centre, limit, n_particles, n_iterations):
    """PyEBSDIndex's particle swarm on ``cost`` inside centre +- limit (clipped to the physical box)."""
    from pyebsdindex.pcopt import PSOOpt
    lo, hi = _clip_box(centre, np.asarray(limit, float))
    start = np.clip(np.asarray(centre, float), lo, hi)
    swarm = PSOOpt(dimensions=3, n_particles=max(5, int(n_particles)), c1=3.5, c2=3.5, w=0.8,
                   hyperparammethod="auto", early_exit=1e-4)
    _best, where = swarm.optimize(cost, start=start, bounds=(lo, hi), niter=int(n_iterations), verbose=0)
    return np.asarray(where, dtype=float).reshape(3), (lo, hi)


def _polish(cost, x0, options):
    """Nelder-Mead on ``cost`` from ``x0``, held inside what a detector can be; returns (pc, cost).

    Without the bounds a cost that keeps falling beyond PCz 1.6 would be followed out of the physical box."""
    from scipy import optimize
    low, high = np.asarray(PHYSICAL_LOW, float), np.asarray(PHYSICAL_HIGH, float)
    # scipy clips the start simplex to the bounds: a start ON an edge makes that axis' simplex a point and the
    # axis is never explored. Start a little inside (measured: minimum z 1.59, start 1.6, result 1.6).
    start = np.clip(np.asarray(x0, float), low + EDGE_INSET, high - EDGE_INSET)
    result = optimize.minimize(lambda pc: cost(pc), start, method="Nelder-Mead",
                               bounds=list(zip(low, high)), options=dict(options))
    return np.asarray(result.x, dtype=float), float(result.fun)


def _on_edge(pc, lo, hi) -> List[str]:
    tol = [min(EDGE_TOL, EDGE_FRACTION * (hi[k] - lo[k]) / 2.0) for k in range(3)]
    return [axis for k, axis in enumerate(AXES)
            if pc[k] <= lo[k] + tol[k] or pc[k] >= hi[k] - tol[k]]


def _ring_points(pc, radius) -> np.ndarray:
    """The 6 points +-radius along the axes and the 8 corners (+-radius, +-radius, +-radius)."""
    pc = np.asarray(pc, float)
    pts = []
    for k in range(3):
        for s in (-radius, radius):
            p = pc.copy()
            p[k] += s
            pts.append(p)
    for sx in (-radius, radius):
        for sy in (-radius, radius):
            for sz in (-radius, radius):
                pts.append(pc + np.array([sx, sy, sz]))
    return np.array(pts)


def _box_sample(n: int) -> np.ndarray:
    """``n`` quasi-random PCs of the wide box (a Halton sequence: the same points every time)."""
    from scipy.stats import qmc
    lo, hi = _clip_box(WIDE_CENTRE, np.asarray(WIDE_LIMIT, float))
    return qmc.scale(qmc.Halton(3, seed=5).random(n), lo, hi)


def _finite_or_none(value):
    """The number, or None when it is NaN or infinite (the report is plain JSON, which has no NaN)."""
    return None if value is None or not np.isfinite(value) else float(value)


def _second_dict(second) -> Optional[dict]:
    """``{"pc": [x, y, z], "cost": number}`` as plain JSON, or ``None``."""
    if not second:
        return None
    return {"pc": [float(v) for v in second["pc"]], "cost": _finite_or_none(second.get("cost"))}


def second_position_of(result_pc, result_cost, other_pc, other_cost) -> Optional[dict]:
    """The other candidate of a find as ``{"pc", "cost"}`` when it is a second position, else ``None``.

    A second position is more than ``SECOND_POSITION_DIST`` away from the result in x, y or z AND costs at most
    ``SECOND_POSITION_COST`` times the cost of the result. The cost is compared with the result's, which is the
    best one except when the other candidate was cheaper but not clearly enough to be adopted (its ratio is then
    below one and it qualifies). A cost that is not a finite number above zero cannot be compared: no second
    position."""
    if other_pc is None or other_cost is None:
        return None
    if result_cost is None or not np.isfinite(result_cost) or not np.isfinite(other_cost) or result_cost <= 0:
        return None
    result = np.asarray(result_pc, float).reshape(3)
    other = np.asarray(other_pc, float).reshape(3)
    if not (np.all(np.isfinite(result)) and np.all(np.isfinite(other))):
        return None
    apart = float(np.max(np.abs(other - result))) > SECOND_POSITION_DIST
    near_in_cost = float(other_cost) <= SECOND_POSITION_COST * float(result_cost)
    if not (apart and near_in_cost):
        return None
    return {"pc": tuple(float(v) for v in other), "cost": float(other_cost)}


def confidence_report(cost, pc, at_pc: float) -> dict:
    """Does the stack fix ``pc``? (``at_pc`` is the cost there.) See the module docstring: the state
    is decided by ``contrast`` (and by the ring showing no cheaper point); the ring numbers are information.
    With ``CONTRAST_MIN`` None there is no state to decide: it is "notJudged"."""
    ring_small = cost(_ring_points(pc, RING_SMALL)) - at_pc
    ring_large_points = _ring_points(pc, RING)
    ring_large = cost(ring_large_points) - at_pc
    excess_small = float(np.median(ring_small))
    excess = float(np.median(ring_large))
    typical = float(np.percentile(cost(_box_sample(N_REFERENCE)), TYPICAL_PERCENTILE))
    lowest = float(min(ring_small.min(), ring_large.min()))
    # a cost that is not a number anywhere in the comparison cannot be compared: no contrast, never "clear"
    comparable = bool(np.all(np.isfinite([at_pc, typical, excess_small, excess, lowest])))
    contrast = 0.0 if not comparable or typical <= 1e-9 else float(np.clip(1.0 - at_pc / typical, -1.0, 1.0))
    growth = excess / excess_small if excess_small > 1e-9 else None
    reasons = []
    if lowest < -POLISH_TOL:
        reasons.append("notAMinimum")
    # too low a contrast is a reason only against a threshold; a contrast that cannot be computed is one always
    if not comparable or (CONTRAST_MIN is not None and contrast < CONTRAST_MIN):
        reasons.append("noContrast")
    if CONTRAST_MIN is None:
        state = "notJudged"
    else:
        state = "clear" if not reasons else "unclear"
    return {
        "state": state, "reasons": reasons,
        "contrast": contrast, "typical_cost": _finite_or_none(typical), "growth": _finite_or_none(growth),
        "excess_small": _finite_or_none(excess_small), "excess": _finite_or_none(excess),
        "min_excess": _finite_or_none(lowest),
        "radius_small": RING_SMALL, "radius": RING, "n_reference": N_REFERENCE,
        "best_point": [float(v) for v in ring_large_points[int(np.argmin(ring_large))]],
    }


def quality_of(report: dict) -> dict:
    """What a person needs to know about how good a search result is, from ``SearchReport.to_dict()``:
    ``{"contrast", "warning", "axes", "n_patterns"}``.

    ``contrast`` is the fit contrast (a number, ``None`` when it could not be computed); ``warning`` is
    ``"edge"`` (the result lies at an edge of the search box, ``axes`` says in which axes) or
    ``"notAMinimum"`` (a point right next to the result is cheaper) or ``None``; with both, the edge is
    named. ``n_patterns`` is the number of calibration patterns the result was computed from. Stored with
    a calibration made from the result, and with an indexing result that ran at its PC.

    A find with a second position (``second_position_of``) adds ``second_position`` (``{"pc", "cost"}``) and
    ``best_cost`` (the cost of the result), and has ``warning`` ``"secondPosition"`` unless the edge or
    "notAMinimum" is named: those keep their value and the second position is still there. A result without a
    second position has neither key."""
    confidence = (report or {}).get("confidence") or {}
    edge = [str(a) for a in ((report or {}).get("on_edge") or [])]
    second = _second_dict((report or {}).get("second_position"))
    if edge:
        warning, axes = "edge", edge
    elif "notAMinimum" in (confidence.get("reasons") or []):
        warning, axes = "notAMinimum", []
    elif second:
        warning, axes = "secondPosition", []
    else:
        warning, axes = None, []
    contrast = confidence.get("contrast")
    # a contrast of a cost that is not a number is reported as 0.0 by ``confidence_report``: say none
    if contrast is None or confidence.get("typical_cost") is None or not np.isfinite(contrast):
        contrast = None
    n = (report or {}).get("n_patterns")
    out = {"contrast": None if contrast is None else float(contrast), "warning": warning, "axes": axes,
           "n_patterns": None if n is None else int(n)}
    if second:
        out["second_position"] = second
        out["best_cost"] = _finite_or_none((report or {}).get("cost"))
    return out


def contrast_at(cost, pc, typical) -> Optional[float]:
    """The contrast of ``pc`` against a ``typical`` cost that ``confidence_report`` already found (its
    ``typical_cost``): ``1 - cost(pc) / typical``, clipped to [-1, 1] as there. ``None`` when either is not a
    number or the typical cost is not above zero. One evaluation of ``cost``."""
    if typical is None or not np.isfinite(typical) or typical <= 1e-9:
        return None
    at = float(cost(np.asarray(pc, float)))
    if not np.isfinite(at):
        return None
    return float(np.clip(1.0 - at / typical, -1.0, 1.0))


def _spread(per_pattern: np.ndarray) -> dict:
    q1, med, q3 = np.percentile(per_pattern, [25, 50, 75], axis=0)
    return {"median": [float(v) for v in med], "iqr": [float(v) for v in (q3 - q1)]}


def _find_in_boxes(cost, centre, limit, n_particles, n_iterations, say):
    """The search of "Find PC" in one starting box, moving it while the result is on its edge.

    Returns ``(pc, value, (lo, hi), box_centre, recentred)`` of the best result seen."""
    centre = np.array(centre, float)
    limit = np.array(limit, float)
    best = None
    recentred = 0
    while True:
        swarm_pc, (lo, hi) = _swarm(cost, centre, limit, n_particles, n_iterations)
        pc, value = _polish(cost, swarm_pc, NM_OPTIONS)
        if best is None or value < best[1]:
            best = (pc, value, (lo, hi), centre.copy())
        say()
        if _on_edge(pc, lo, hi) and recentred < MAX_RECENTRES:
            recentred += 1
            centre = pc.copy()
            continue
        break
    pc, value, bounds, box_centre = best
    return pc, value, bounds, box_centre, recentred


# ------------------------------------------------------------------ the entry point

def check_refine(method, start, search_limit):
    """What a refine is given, checked before anything is built. Returns ``(start, "NELDER-MEAD" or "PSO", limit)``."""
    name = method.strip().upper() if isinstance(method, str) else None
    if name not in ("NELDER-MEAD", "PSO"):
        raise SearchError("badMethod", f"unknown refine method {method!r}: 'Nelder-Mead' or 'PSO'", method=str(method))
    if start is None:
        raise SearchError("needsStart", "a refine needs the PC to start from")
    try:
        arr = np.asarray(start, float)
    except (TypeError, ValueError):
        arr = None
    if arr is None or arr.shape != (3,):
        raise SearchError("badStart", f"the start must be three numbers (x, y, z), got {start!r}", axes=list(AXES))
    outside = [axis for k, axis in enumerate(AXES)           # a NaN is outside: every comparison with it is False
               if not (PHYSICAL_LOW[k] <= arr[k] <= PHYSICAL_HIGH[k])]
    if outside:
        raise SearchError(
            "badStart", f"the start {tuple(arr)} is not a pattern centre a detector can have "
            f"(x {PHYSICAL_LOW[0]}-{PHYSICAL_HIGH[0]}, y {PHYSICAL_LOW[1]}-{PHYSICAL_HIGH[1]}, "
            f"z {PHYSICAL_LOW[2]}-{PHYSICAL_HIGH[2]}); check {', '.join(outside)}", axes=outside)
    try:
        limit = float(search_limit)
    except (TypeError, ValueError):
        limit = float("nan")
    if not MIN_SEARCH_LIMIT <= limit <= 1.0:                   # False for NaN as well
        raise SearchError("badSearchLimit",
                          f"the search limit must be in [{MIN_SEARCH_LIMIT:g}, 1], got {search_limit!r}",
                          search_limit=str(search_limit))
    return tuple(float(v) for v in arr), name, limit


def run_global(indexer, patterns, *, mode: str = "find", method: str = "Nelder-Mead",
               search_limit: float = 0.05, start=None, n_particles: int = N_PARTICLES,
               n_iterations: int = N_ITERATIONS, with_spread: bool = True,
               progress: Optional[Callable[[float], None]] = None) -> SearchReport:
    """Search the PC of ``patterns`` (``(n, rows, cols)``) with ``indexer``'s phases and detector."""
    if mode not in ("find", "refine"):
        raise SearchError("badMode", f"unknown search mode {mode!r}")
    stack = np.asarray(patterns)
    n = 1 if stack.ndim == 2 else int(stack.shape[0])
    if mode == "find" and n < MIN_PATTERNS_FIND:
        raise SearchError(
            "tooFewPatterns",
            f"Find PC needs at least {MIN_PATTERNS_FIND} calibration patterns, got {n}",
            have=n, need=MIN_PATTERNS_FIND)
    start_used = None
    if mode == "refine":
        start_used, method_name, search_limit = check_refine(method, start, search_limit)
    steps = {"done": 0}
    total_steps = 6 if mode == "find" else 3

    def say(fraction=None):
        if progress is None:
            return
        steps["done"] += 1
        progress(min(0.95, steps["done"] / float(total_steps)) if fraction is None else fraction)

    cost = JointPcCost(indexer, stack)
    if progress is not None:
        progress(0.02)
    recentred = 0
    far_used = False
    box = None
    confidence = None
    first_box_pc = far_box_pc = far_box_cost = None
    if mode == "find":
        pc, value, (lo, hi), box_centre, recentred = _find_in_boxes(
            cost, WIDE_CENTRE, WIDE_LIMIT, n_particles, n_iterations, say)
        first_box_pc = tuple(float(v) for v in pc)
        first_box_cost = float(value)
        confidence = confidence_report(cost, pc, value)
        if confidence["state"] != "clear":
            far = _find_in_boxes(cost, FAR_CENTRE, WIDE_LIMIT, n_particles, n_iterations, say)
            far_box_pc, far_box_cost = tuple(float(v) for v in far[0]), float(far[1])
            far_confidence = confidence_report(cost, far[0], far[1])
            # cheaper alone is a coin toss on a nearly flat landscape: the far box must be clear itself, or
            # cheaper by more than the cost rise of a ring step around the first result. With no threshold
            # nothing is clear ("notJudged"): the far box is always tried, adopted only when clearly cheaper
            clearly_cheaper = value - far[1] > max(confidence["excess"] or 0.0, 0.0)     # None: not finite
            if far[1] < value and (far_confidence["state"] == "clear" or clearly_cheaper):
                pc, value, (lo, hi), box_centre = far[0], far[1], far[2], far[3]
                recentred += far[4]
                far_used = True
                confidence = far_confidence
        used = "PSO+NM"
        box = {"centre": [float(v) for v in box_centre], "limit": [float(v) for v in WIDE_LIMIT],
               "low": [float(v) for v in lo], "high": [float(v) for v in hi]}
    else:
        if method_name == "PSO":
            swarm_pc, (lo, hi) = _swarm(cost, np.array(start_used), np.full(3, search_limit),
                                        n_particles, n_iterations)
            say()
            pc, value = _polish(cost, swarm_pc, NM_OPTIONS)
            used = "PSO+NM"
            box = {"centre": list(start_used), "limit": [search_limit] * 3,
                   "low": [float(v) for v in lo], "high": [float(v) for v in hi]}
        else:
            pc, value = _polish(cost, np.array(start_used), NM_OPTIONS)
            used = "NM"
            lo = hi = None
    say()

    if confidence is None:                  # a refine: judged once here; a find was judged above
        confidence = confidence_report(cost, pc, value)
    if "notAMinimum" in confidence["reasons"]:
        # a point of the ring is cheaper: one more descent from it, then judge again
        pc2, value2 = _polish(cost, np.array(confidence["best_point"]), NM_OPTIONS)
        # only a new result is judged anew; the same one stays as judged ("cheaper" by less than the
        # polish tolerance is the optimiser's noise: from a start just inside the box two descents differ by ~5e-8)
        if value2 < value - POLISH_TOL:
            pc, value = pc2, value2
            confidence = confidence_report(cost, pc, value)
    edges = [] if lo is None else _on_edge(pc, lo, hi)      # of the answer, which the descent may have moved
    # a refine says how good the PC it started from was, on the scale of its own result (same typical cost)
    contrast_at_start = (contrast_at(cost, start_used, confidence.get("typical_cost"))
                         if mode == "refine" else None)

    # The two candidates of a find are the ends of its boxes; the one that is not the result may be a second position
    second_position = None
    if mode == "find" and far_box_pc is not None:
        other = (first_box_pc, first_box_cost) if far_used else (far_box_pc, far_box_cost)
        second_position = second_position_of(pc, value, *other)

    per_pattern = spread = None
    if with_spread and n > 1:
        per_pattern = np.array([_polish(cost.subset(i), pc, NM_OPTIONS_SINGLE)[0] for i in range(n)])
        spread = _spread(per_pattern)
    if progress is not None:
        progress(1.0)
    return SearchReport(
        pc=tuple(float(v) for v in pc), cost=float(value), start=start_used, mode=mode, method=used,
        n_patterns=n, evaluations=int(cost.evaluations), box=box, recentred=recentred,
        far_box_used=far_used, first_box_pc=first_box_pc, far_box_pc=far_box_pc,
        far_box_cost=far_box_cost, on_edge=edges, confidence=confidence,
        per_pattern=per_pattern, spread=spread, contrast_at_start=contrast_at_start,
        second_position=second_position)
