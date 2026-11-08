"""The pattern-centre search: its control flow on cost functions the test defines (fast), and the
whole thing on patterns rendered at a PC the test chose (``slow``: PyEBSDIndex's band detection and
a particle swarm; run them in both environments)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("pyebsdindex")

from backend.api.services import pc_search as ps  # noqa: E402


# ======================================================================================
# Fast: the search on a cost the test defines
# ======================================================================================

class Bowl:
    """A stand-in for ``JointPcCost``. Calibration pattern ``i`` has a bowl of stiffness ``coefs[i]``
    centred on ``targets[i]``, flat (``PLATEAU``) farther than ``sqrt(PLATEAU / coef)`` from its
    centre, like the real cost far from the true PC; the joint cost is the mean of the bowls.
    ``indexer`` is a plain namespace: ``target`` (the common centre), optional ``offsets`` (one
    per pattern), ``coefs`` (one per pattern, default 130) and ``flat`` (a cost with no minimum)."""

    PLATEAU = 8.5

    def __init__(self, indexer, patterns=None, *, bowls=None):
        self.indexer = indexer
        if bowls is None:
            n = 1 if np.ndim(patterns) == 2 else len(patterns)
            offsets = getattr(indexer, "offsets", None)
            coefs = getattr(indexer, "coefs", None)
            bowls = [(np.asarray(indexer.target, float) + (np.zeros(3) if offsets is None
                                                            else np.asarray(offsets[i], float)),
                      130.0 if coefs is None else float(coefs[i])) for i in range(n)]
        self.bowls = bowls
        self.evaluations = 0

    @property
    def n_patterns(self):
        return len(self.bowls)

    def subset(self, i):
        return Bowl(self.indexer, bowls=[self.bowls[i]])

    def __call__(self, pc):
        arr = np.asarray(pc, float)
        rows = np.atleast_2d(arr)
        flat = getattr(self.indexer, "flat", False)
        out = np.empty(rows.shape[0])
        for k, row in enumerate(rows):
            if flat:
                out[k] = self.PLATEAU
            else:
                out[k] = float(np.mean([0.05 + min(self.PLATEAU, coef * float(np.sum((row - t) ** 2)))
                                        for t, coef in self.bowls]))
        self.evaluations += rows.shape[0]
        return float(out[0]) if arr.ndim == 1 else out


@pytest.fixture
def bowl(monkeypatch):
    monkeypatch.setattr(ps, "JointPcCost", Bowl)

    def make(target, **kw):
        return SimpleNamespace(target=target, **kw)
    return make


PATTERNS = np.zeros((9, 8, 8), np.float32)


@pytest.fixture(autouse=True)
def _threshold_of_the_fast_tests(request, monkeypatch):
    """The fast tests judge costs the test defines: a ``Bowl`` result has a contrast above 0.9, a plateau 0.
    They run with a threshold of their own (0.5) so that the rules that depend on a judgement are tested
    whatever the module holds; a test of the state without a threshold sets ``CONTRAST_MIN`` to None itself
    (see ``THRESHOLDS``). The ``slow`` tests keep the module's value, which is None: they check the search result."""
    if request.node.get_closest_marker("slow") is None:
        monkeypatch.setattr(ps, "CONTRAST_MIN", 0.5)


#: The two states of the module: a threshold set (the search judges) and none (it does not).
THRESHOLDS = [0.5, None]


def _judged_as(threshold, judged):
    """The state a report must have: ``judged`` under a threshold, "notJudged" without one."""
    return "notJudged" if threshold is None else judged


def test_find_finds_a_minimum_inside_the_box_and_calls_it_clear(bowl):
    target = (0.52, 0.31, 0.62)
    rep = ps.run_global(bowl(target), PATTERNS, mode="find")
    assert np.abs(np.array(rep.pc) - target).max() < 0.01
    assert rep.confidence["state"] == "clear" and rep.confidence["reasons"] == []
    assert rep.recentred == 0 and rep.far_box_used is False and rep.on_edge == []
    assert rep.method == "PSO+NM" and rep.start is None and rep.n_patterns == 9


def test_find_does_not_move_a_narrow_box_for_a_minimum_a_sixth_of_its_half_width_inside_the_edge(bowl, monkeypatch):
    """A box of +-0.05 (its edge is 0.005 deep): the minimum 0.015 inside the x edge is not 'on the edge', so the box
    stays and no edge is reported. With the fixed tolerance of 0.02 it was: the box moved and x was named."""
    monkeypatch.setattr(ps, "WIDE_LIMIT", (0.05, 0.05, 0.05))
    target = (0.535, 0.40, 0.70)
    rep = ps.run_global(bowl(target), PATTERNS, mode="find")
    assert rep.recentred == 0 and rep.on_edge == []
    assert np.abs(np.array(rep.pc) - target).max() < 0.01


def test_find_moves_its_box_when_the_minimum_lies_beyond_an_edge(bowl):
    """z = 1.10 is outside the wide box (z up to 1.05) but inside the bowl's reach of its edge."""
    target = (0.50, 0.30, 1.10)
    rep = ps.run_global(bowl(target), PATTERNS, mode="find")
    assert rep.recentred >= 1
    assert np.abs(np.array(rep.pc) - target).max() < 0.01
    assert rep.box["centre"][2] > ps.WIDE_CENTRE[2]            # the box that found it was moved up


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_find_tries_the_far_box_when_the_first_result_is_unclear(bowl, monkeypatch, threshold):
    """The bowl at z = 1.4 is invisible from the first box (a plateau everywhere in it) and inside the far one.
    Under a threshold the far result is clear; without one nothing is, and the far result is adopted because
    it is cheaper than the plateau by far more than the ring excess of the first result (about 0)."""
    monkeypatch.setattr(ps, "CONTRAST_MIN", threshold)
    target = (0.50, 0.30, 1.40)
    rep = ps.run_global(bowl(target), PATTERNS, mode="find")
    assert rep.far_box_used is True
    assert np.abs(np.array(rep.pc) - target).max() < 0.02
    assert rep.confidence["state"] == _judged_as(threshold, "clear")


def test_without_a_threshold_find_runs_through_and_does_not_judge(bowl, monkeypatch):
    """CONTRAST_MIN None: no error, state "notJudged", the contrast is still a number (a Bowl result is far below
    the plateau: above 0.9), no reason, the answer is the minimum, and the far box was tried (nothing is clear)
    but not adopted: it walks to the same minimum at best, so it is never clearly cheaper. The report is plain JSON."""
    import json
    monkeypatch.setattr(ps, "CONTRAST_MIN", None)
    target = (0.52, 0.31, 0.62)
    rep = ps.run_global(bowl(target), PATTERNS, mode="find")
    assert np.abs(np.array(rep.pc) - target).max() < 0.01
    conf = rep.confidence
    assert conf["state"] == "notJudged" and conf["reasons"] == []
    assert isinstance(conf["contrast"], float) and conf["contrast"] > 0.9
    assert conf["typical_cost"] > 0 and conf["excess"] is not None
    assert rep.far_box_pc is not None and rep.far_box_cost is not None      # tried
    assert rep.far_box_used is False and rep.on_edge == []
    shown = json.loads(json.dumps(rep.to_dict(), allow_nan=False))
    assert shown["confidence"]["state"] == "notJudged"


def test_without_a_threshold_a_refine_runs_through_and_does_not_judge(bowl, monkeypatch):
    monkeypatch.setattr(ps, "CONTRAST_MIN", None)
    target = (0.52, 0.31, 0.62)
    for kw in ({"method": "Nelder-Mead"}, {"method": "PSO", "search_limit": 0.05}):
        rep = ps.run_global(bowl(target), PATTERNS, mode="refine", start=(0.50, 0.33, 0.60), **kw)
        assert np.abs(np.array(rep.pc) - target).max() < 0.005
        assert rep.confidence["state"] == "notJudged" and rep.confidence["reasons"] == []
        assert rep.confidence["contrast"] > 0.9 and rep.far_box_pc is None


def _basins(*basins):
    """A cost with several basins, ``(centre, base, stiffness)`` each: the cost at a PC is the lowest
    basin's ``base + stiffness * distance^2``, capped at the plateau (``Bowl.PLATEAU``)."""
    class Basins(Bowl):
        def __call__(self, pc):
            arr = np.asarray(pc, float)
            rows = np.atleast_2d(arr)
            out = np.array([min(b + min(self.PLATEAU - b, k * float(np.sum((r - np.asarray(c, float)) ** 2)))
                                for c, b, k in basins) for r in rows])
            self.evaluations += rows.shape[0]
            return float(out[0]) if arr.ndim == 1 else out
    return Basins


@pytest.mark.parametrize("threshold", THRESHOLDS)
@pytest.mark.parametrize("far_base, adopted", [(4.95, False), (4.7, True)])
def test_the_far_box_result_is_adopted_only_when_it_is_clear_or_clearly_cheaper(monkeypatch, far_base, adopted,
                                                                                  threshold):
    """Two basins. A lies in the first box: cost 5.0 on a plateau of 8.5, contrast 1 - 5.0 / 8.5 = 0.41, unclear
    under the fast tests' 0.5; stiffness 130, so the ring of +-0.02 around it costs 130 * 0.0004 = 0.052 on its six
    axis points and 130 * 3 * 0.0004 = 0.156 on its eight corners, median 0.156 (by hand). B lies in the far box
    and is softer (stiffness 12). B cheaper by 0.05 is less than the 0.156 the ring costs: a coin toss on a
    nearly flat landscape, not adopted. B cheaper by 0.3 is more than the ring, although B's contrast (0.447) is
    itself unclear: adopted. Either way both candidates are reported. Without a threshold the far box is tried and
    the rule is the same (neither result is clear anyway), the state is "notJudged"."""
    monkeypatch.setattr(ps, "CONTRAST_MIN", threshold)
    a, b = (0.5, 0.4, 0.7), (0.5, 0.3, 1.5)
    monkeypatch.setattr(ps, "JointPcCost", _basins((a, 5.0, 130.0), (b, far_base, 12.0)))
    rep = ps.run_global(SimpleNamespace(target=(0, 0, 0)), PATTERNS, mode="find")
    assert np.abs(np.array(rep.first_box_pc) - a).max() < 0.01
    assert np.abs(np.array(rep.far_box_pc) - b).max() < 0.02
    assert rep.far_box_cost == pytest.approx(far_base, abs=0.01)
    assert rep.far_box_used is adopted
    assert np.abs(np.array(rep.pc) - (b if adopted else a)).max() < 0.02
    assert rep.confidence["state"] == _judged_as(threshold, "unclear")


def test_a_clear_first_result_never_tries_the_far_box(bowl):
    rep = ps.run_global(bowl((0.52, 0.31, 0.62)), PATTERNS, mode="find")
    assert rep.far_box_used is False and rep.far_box_pc is None and rep.far_box_cost is None
    assert np.abs(np.array(rep.first_box_pc) - (0.52, 0.31, 0.62)).max() < 0.01


def test_without_a_threshold_the_search_gives_no_verdict_and_no_error(monkeypatch):
    """CONTRAST_MIN None: ``confidence_report`` does not raise, the state is "notJudged", the numbers are the
    ones it always reports. By hand: a quadratic bowl measured at its own centre costs 0, every box point costs
    more, so contrast = 1 - 0 / typical = 1."""
    monkeypatch.setattr(ps, "CONTRAST_MIN", None)
    target = np.array([0.5, 0.4, 0.7])
    report = ps.confidence_report(_quadratic(target), target, 0.0)
    assert report["state"] == "notJudged" and report["reasons"] == []
    assert report["contrast"] == pytest.approx(1.0) and report["typical_cost"] > 0
    assert report["excess"] == pytest.approx(0.48) and report["min_excess"] == pytest.approx(0.04)
    assert report["state"] != "clear"


def test_with_a_threshold_the_search_judges_again(monkeypatch):
    """The same report as above under a threshold: contrast 1 >= 0.5 is clear, a contrast of 0 is not."""
    monkeypatch.setattr(ps, "CONTRAST_MIN", 0.5)
    target = np.array([0.5, 0.4, 0.7])
    assert ps.confidence_report(_quadratic(target), target, 0.0)["state"] == "clear"
    flat = ps.confidence_report(lambda pc: np.full(np.atleast_2d(pc).shape[0], 3.0), target, 3.0)
    assert flat["state"] == "unclear" and flat["reasons"] == ["noContrast"]


def test_confidence_is_judged_once_for_each_result_on_the_find_path(bowl, monkeypatch):
    """A clear first result is judged once (not again after the far-box decision, which did not happen)."""
    calls = []
    real = ps.confidence_report
    monkeypatch.setattr(ps, "confidence_report", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    ps.run_global(bowl((0.52, 0.31, 0.62)), PATTERNS, mode="find")
    assert len(calls) == 1


def test_without_a_threshold_each_result_of_a_find_is_still_reported_once(bowl, monkeypatch):
    """No result is clear, so the far box is always tried: the first and the far result are each judged once."""
    monkeypatch.setattr(ps, "CONTRAST_MIN", None)
    calls = []
    real = ps.confidence_report
    monkeypatch.setattr(ps, "confidence_report", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    ps.run_global(bowl((0.52, 0.31, 0.62)), PATTERNS, mode="find")
    assert len(calls) == 2


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_find_says_so_when_no_box_holds_the_minimum(bowl, monkeypatch, threshold):
    """Beyond the physical limit of PCz (1.6) nothing is found, and the result is not called clear. Under a
    threshold the reason is the missing contrast; without one there is no verdict and no such reason."""
    monkeypatch.setattr(ps, "CONTRAST_MIN", threshold)
    rep = ps.run_global(bowl((0.5, 0.3, 2.4)), PATTERNS, mode="find")
    assert rep.confidence["state"] == _judged_as(threshold, "unclear")
    assert rep.confidence["state"] != "clear"
    assert ("noContrast" in rep.confidence["reasons"]) is (threshold is not None)
    assert rep.confidence["contrast"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_a_flat_cost_has_no_contrast_and_no_growth(bowl, monkeypatch, threshold):
    monkeypatch.setattr(ps, "CONTRAST_MIN", threshold)
    rep = ps.run_global(bowl((0.5, 0.4, 0.7), flat=True), PATTERNS, mode="find")
    assert rep.confidence["state"] == _judged_as(threshold, "unclear")
    assert rep.confidence["reasons"] == (["noContrast"] if threshold is not None else [])
    assert rep.confidence["contrast"] == pytest.approx(0.0, abs=1e-9)
    assert rep.confidence["growth"] is None                  # a ring that costs exactly what the centre does


def test_refine_with_nelder_mead_starts_at_the_given_pc(bowl):
    target = (0.52, 0.31, 0.62)
    rep = ps.run_global(bowl(target), PATTERNS, mode="refine", method="Nelder-Mead",
                        start=(0.50, 0.33, 0.60))
    assert np.abs(np.array(rep.pc) - target).max() < 0.005
    assert rep.method == "NM" and rep.start == (0.50, 0.33, 0.60) and rep.box is None


def test_a_refine_is_a_local_search_from_its_start(monkeypatch):
    """Two basins 0.4 apart: a shallow one at A and a deeper one at B (inside the wide box of a find). A refine
    that starts next to A ends at A; a search that ignored its start would fall into B."""
    A, B = np.array([0.30, 0.20, 0.40]), np.array([0.60, 0.45, 0.80])

    class TwoBasins(Bowl):
        def __call__(self, pc):
            arr = np.asarray(pc, float)
            rows = np.atleast_2d(arr)
            out = np.array([min(0.60 + min(self.PLATEAU, 130.0 * float(np.sum((r - A) ** 2))),
                                0.05 + min(self.PLATEAU, 130.0 * float(np.sum((r - B) ** 2)))) for r in rows])
            self.evaluations += rows.shape[0]
            return float(out[0]) if arr.ndim == 1 else out

    monkeypatch.setattr(ps, "JointPcCost", TwoBasins)
    rep = ps.run_global(SimpleNamespace(target=(0, 0, 0)), PATTERNS, mode="refine", method="Nelder-Mead",
                        start=(0.31, 0.21, 0.41))
    assert np.abs(np.array(rep.pc) - A).max() < 0.01


def test_refine_with_the_swarm_searches_around_the_start_only(bowl):
    target = (0.52, 0.31, 0.62)
    rep = ps.run_global(bowl(target), PATTERNS, mode="refine", method="PSO", search_limit=0.05,
                        start=(0.50, 0.33, 0.60))
    assert np.abs(np.array(rep.pc) - target).max() < 0.005
    assert rep.method == "PSO+NM" and rep.box["limit"] == [0.05, 0.05, 0.05]
    assert rep.box["low"] == pytest.approx([0.45, 0.28, 0.55])         # start - 0.05, by hand


def test_the_answer_is_the_minimum_of_the_joint_cost_not_the_mean_of_the_patterns_own_minima(bowl):
    """Four patterns have a stiff bowl (130) centred on A, one a soft bowl (13) centred 0.04 away in x.
    Each pattern alone prefers its own centre; the mean of those five is A + 0.008. The minimum of the
    JOINT cost is the stiffness-weighted mean, A + 0.04 * 13 / (4 * 130 + 13) = A + 0.00098: the
    stiff patterns decide. (An answer that averaged the patterns' own optima, the old Global Refine,
    would be 0.007 away from it.)"""
    a = np.array([0.52, 0.31, 0.62])
    offsets = [(0, 0, 0)] * 4 + [(0.04, 0, 0)]
    coefs = [130.0] * 4 + [13.0]
    rep = ps.run_global(bowl(tuple(a), offsets=offsets, coefs=coefs), PATTERNS[:5], mode="find")
    np.testing.assert_allclose(rep.pc, a + np.array([0.00098, 0, 0]), atol=0.0025)
    np.testing.assert_allclose(rep.per_pattern[:4], np.tile(a, (4, 1)), atol=0.003)
    np.testing.assert_allclose(rep.per_pattern[4], a + np.array([0.04, 0, 0]), atol=0.004)
    own_mean = rep.per_pattern.mean(axis=0)
    assert abs(own_mean[0] - rep.pc[0]) > 0.005          # what the old method would have answered


def test_the_spread_is_the_interquartile_range_of_what_each_pattern_prefers(bowl):
    """x offsets -0.02 ... +0.02 in steps of 0.005 over nine patterns: the 25th and 75th percentiles
    of nine evenly spaced values are the 3rd and the 7th, -0.01 and +0.01, so the IQR is 0.02 (by hand)."""
    offsets = [((i - 4) * 0.005, 0.0, 0.0) for i in range(9)]
    rep = ps.run_global(bowl((0.52, 0.31, 0.62), offsets=offsets), PATTERNS, mode="find")
    assert rep.spread["iqr"][0] == pytest.approx(0.02, abs=0.004)
    assert rep.spread["iqr"][1] == pytest.approx(0.0, abs=0.003)
    assert rep.spread["median"][0] == pytest.approx(0.52, abs=0.004)


def test_too_few_patterns_for_a_find_are_refused_with_a_code(bowl):
    with pytest.raises(ps.SearchError) as exc:
        ps.run_global(bowl((0.5, 0.4, 0.7)), PATTERNS[:4], mode="find")
    assert exc.value.code == "tooFewPatterns" and exc.value.params == {"have": 4, "need": 5}
    ps.run_global(bowl((0.5, 0.4, 0.7)), PATTERNS[:5], mode="find")              # five are enough


def test_a_refine_needs_a_start_and_a_mode_must_be_known(bowl):
    with pytest.raises(ps.SearchError) as exc:
        ps.run_global(bowl((0.5, 0.4, 0.7)), PATTERNS, mode="refine")
    assert exc.value.code == "needsStart"
    with pytest.raises(ps.SearchError) as exc:
        ps.run_global(bowl((0.5, 0.4, 0.7)), PATTERNS, mode="guess")
    assert exc.value.code == "badMode"


def test_progress_only_goes_forward_and_ends_at_one(bowl):
    seen = []
    ps.run_global(bowl((0.52, 0.31, 0.62)), PATTERNS, mode="find", progress=seen.append)
    assert seen and seen[-1] == 1.0 and all(b >= a for a, b in zip(seen, seen[1:]))
    assert all(0.0 <= v <= 1.0 for v in seen)


def test_the_report_is_plain_json():
    import json
    rep = ps.SearchReport(pc=(0.5, 0.4, 0.7), cost=1.0, start=None, mode="find", method="PSO+NM",
                          n_patterns=5, per_pattern=np.zeros((5, 3)), confidence={"state": "clear"})
    assert json.loads(json.dumps(rep.to_dict()))["pc"] == [0.5, 0.4, 0.7]


# ------------------------------------------------------------------ the small pieces, by hand

def test_the_box_is_clipped_to_what_a_detector_can_be():
    lo, hi = ps._clip_box((0.5, 0.1, 1.5), np.array([0.2, 0.3, 0.35]))
    # x 0.3..0.7; y -0.2..0.4 -> 0.0..0.4; z 1.15..1.85 -> 1.15..1.6
    np.testing.assert_allclose(lo, (0.3, 0.0, 1.15))
    np.testing.assert_allclose(hi, (0.7, 0.4, 1.6))


def test_in_a_narrow_box_the_edge_is_a_tenth_of_its_half_width_not_the_fixed_tolerance():
    """A box of +-0.05 around (0.5, 0.4, 0.7): the edge is 0.005 deep (a tenth of 0.05), not 0.02, which
    would call the outer 40 % of the box 'the edge'. A box whose half width is 0.2 or more keeps 0.02."""
    lo, hi = np.array([0.45, 0.35, 0.65]), np.array([0.55, 0.45, 0.75])
    assert ps._on_edge(np.array([0.53, 0.4, 0.7]), lo, hi) == []                  # 0.02 from the edge: inside
    assert ps._on_edge(np.array([0.5, 0.4, 0.72]), lo, hi) == []                  # 0.03 from it
    assert ps._on_edge(np.array([0.546, 0.4, 0.7]), lo, hi) == ["x"]              # 0.004 from it
    assert ps._on_edge(np.array([0.5, 0.354, 0.7]), lo, hi) == ["y"]              # 0.004 above the lower edge
    assert ps._on_edge(np.array([0.5, 0.4, 0.76]), lo, hi) == ["z"]               # beyond it
    # the width belongs to the axis: x +-0.2 keeps 0.02, y +-0.01 gets 0.001
    lo, hi = np.array([0.3, 0.39, 0.35]), np.array([0.7, 0.41, 1.05])
    assert ps._on_edge(np.array([0.685, 0.4, 0.7]), lo, hi) == ["x"]              # 0.015 from x's edge: near it
    assert ps._on_edge(np.array([0.675, 0.4, 0.7]), lo, hi) == []                 # 0.025 from it: not near it
    assert ps._on_edge(np.array([0.5, 0.408, 0.7]), lo, hi) == []                 # 0.002 from y's edge: y's tolerance is 0.001
    assert ps._on_edge(np.array([0.5, 0.4095, 0.7]), lo, hi) == ["y"]             # 0.0005 from it


def test_a_result_near_an_edge_or_outside_the_box_is_on_the_edge():
    lo, hi = np.array([0.3, 0.1, 0.35]), np.array([0.7, 0.7, 1.05])
    assert ps._on_edge(np.array([0.5, 0.4, 0.7]), lo, hi) == []
    assert ps._on_edge(np.array([0.69, 0.4, 0.7]), lo, hi) == ["x"]              # 0.01 from the edge
    assert ps._on_edge(np.array([0.5, 0.115, 1.06]), lo, hi) == ["y", "z"]         # 0.015 above; beyond
    assert ps._on_edge(np.array([0.5, 0.4, 1.035]), lo, hi) == ["z"]               # 0.015 below the edge
    assert ps._on_edge(np.array([0.5, 0.4, 1.02]), lo, hi) == []                    # 0.03 below: not near it


def test_the_ring_is_six_axis_points_and_eight_corners():
    pts = ps._ring_points(np.array([0.5, 0.4, 0.7]), 0.02)
    assert pts.shape == (14, 3)
    np.testing.assert_allclose(pts[0], (0.48, 0.4, 0.7))
    np.testing.assert_allclose(pts[5], (0.5, 0.4, 0.72))
    corners = np.round(pts[6:] - np.array([0.5, 0.4, 0.7]), 12)
    assert set(map(tuple, np.abs(corners).round(6))) == {(0.02, 0.02, 0.02)}
    assert len(set(map(tuple, corners))) == 8


def _quadratic(target, stiffness=400.0):
    target = np.asarray(target, float)

    def cost(pc):
        arr = np.asarray(pc, float)
        out = stiffness * np.sum((np.atleast_2d(arr) - target) ** 2, axis=1)
        return float(out[0]) if arr.ndim == 1 else out
    return cost


def test_growth_of_a_quadratic_bowl_is_four_by_hand():
    """cost = 400 |d|^2: ring points at +-0.01 along an axis cost 0.04, at a corner 0.12; the median of
    six 0.04 and eight 0.12 is 0.12 at 0.01 and 0.48 at 0.02 -> growth 4."""
    target = np.array([0.5, 0.4, 0.7])
    report = ps.confidence_report(_quadratic(target), target, 0.0)
    assert report["excess_small"] == pytest.approx(0.12) and report["excess"] == pytest.approx(0.48)
    assert report["growth"] == pytest.approx(4.0) and report["min_excess"] == pytest.approx(0.04)


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_a_ring_point_cheaper_than_the_result_means_not_a_minimum(monkeypatch, threshold):
    """Under a threshold the result is unclear; without one it is "notJudged" and still "notAMinimum"."""
    monkeypatch.setattr(ps, "CONTRAST_MIN", threshold)
    target = np.array([0.52, 0.4, 0.7])
    away = target + np.array([0.015, 0.0, 0.0])
    cost = _quadratic(target)
    report = ps.confidence_report(cost, away, cost(away))
    assert "notAMinimum" in report["reasons"] and report["state"] == _judged_as(threshold, "unclear")
    assert report["min_excess"] < 0


@pytest.mark.parametrize("below, flagged", [(1e-6, False), (1e-3, True)])
def test_a_ring_point_is_a_cheaper_point_only_beyond_the_polish_tolerance(monkeypatch, below, flagged):
    """The descent takes a new result only when it is cheaper by more than NM_OPTIONS['fatol'] (1e-5); the ring
    must not call the result 'notAMinimum' for a ring point that is cheaper by less, where the descent would
    adopt nothing. By hand: the cost is 1.0 at the result, 1.0 - below at the ring point (-0.02, 0, 0) and 2.0
    at every other ring point and box sample point, so contrast = 1 - 1/2 = 0.5 clears the threshold 0.0."""
    monkeypatch.setattr(ps, "CONTRAST_MIN", 0.0)
    pc0 = np.array([0.5, 0.4, 0.7])
    cheap = pc0 + np.array([-0.02, 0.0, 0.0])

    def cost(pc):
        arr = np.atleast_2d(np.asarray(pc, float))
        out = np.where(np.all(np.abs(arr - pc0) < 1e-12, axis=1), 1.0, 2.0)
        out[np.all(np.abs(arr - cheap) < 1e-12, axis=1)] = 1.0 - below
        return out
    report = ps.confidence_report(cost, pc0, 1.0)
    assert report["min_excess"] == pytest.approx(-below)
    assert ("notAMinimum" in report["reasons"]) is flagged
    assert report["state"] == ("unclear" if flagged else "clear")


@pytest.mark.parametrize("threshold", THRESHOLDS)
@pytest.mark.parametrize("n_low", [20, 29, 31, 40])
def test_the_contrast_does_not_jump_when_half_the_box_points_lie_on_the_upper_level(monkeypatch, n_low, threshold):
    """A cost with two levels, like the real one (single patterns fail at some PCs): the ``n_low`` box points with
    the smallest x cost 5.0, the others 15.0. The result costs 1.0.

    By hand: of the 60 sorted box costs the lower quartile is the value at position 0.25 * 59 = 14.75, which is 5.0
    for every n_low from 16 to 59 -> typical 5.0, contrast 1 - 1.0 / 5.0 = 0.8 in all four cases. The MEDIAN (position
    29.5, the mean of the 30th and 31st value) is 15.0 for n_low 20 and 29 and 5.0 for 31 and 40: the contrast would
    be 1 - 1/15 = 0.933 for the first two and 0.8 for the others, a jump at n_low 30 by 0.133 for one more point.
    The number is reported with and without a threshold; only the state differs."""
    monkeypatch.setattr(ps, "CONTRAST_MIN", threshold)
    x_sorted = np.sort(ps._box_sample(ps.N_REFERENCE)[:, 0])
    cut = 0.5 * (x_sorted[n_low - 1] + x_sorted[n_low])               # n_low points lie left of it

    def cost(pc):
        rows = np.atleast_2d(np.asarray(pc, float))
        out = np.where(rows[:, 0] <= cut, 5.0, 15.0)
        return float(out[0]) if np.ndim(pc) == 1 else out
    pc0 = np.array([0.5, 0.4, 0.7])
    report = ps.confidence_report(cost, pc0, 1.0)
    assert report["typical_cost"] == pytest.approx(5.0)
    assert report["contrast"] == pytest.approx(0.8)
    assert report["state"] == _judged_as(threshold, "clear")


def test_the_typical_cost_is_the_lower_quartile_of_the_box_points():
    """Costs 1, 2, ..., 60 in the order of the box sample (the point with the smallest x costs 1): the lower quartile
    of 1..60 is at position 14.75 -> 15 + 0.75 = 15.75 (by hand), the median would be 30.5."""
    def cost(pc):
        rows = np.atleast_2d(np.asarray(pc, float))
        if rows.shape[0] == ps.N_REFERENCE:                            # the box sample: the rank of x
            return np.argsort(np.argsort(rows[:, 0])) + 1.0
        return np.full(rows.shape[0], 100.0)                           # the rings
    report = ps.confidence_report(cost, np.array([0.5, 0.4, 0.7]), 1.0)
    assert ps.TYPICAL_PERCENTILE == 25
    assert report["typical_cost"] == pytest.approx(15.75)


def _nan_where(rows_nan, inner):
    """``inner`` everywhere, except that a call with ``rows_nan`` rows (a ring: 14; the box sample: 60) answers NaN."""
    def cost(pc):
        arr = np.asarray(pc, float)
        out = np.asarray(inner(arr), float)
        if arr.ndim == 2 and arr.shape[0] == rows_nan:
            out = np.full_like(out, np.nan)
        return out
    return cost


@pytest.mark.parametrize("threshold", [0.5, 0.0, None])
@pytest.mark.parametrize("what", ["at_pc", "box_sample", "ring"])
def test_a_cost_that_is_not_finite_is_never_called_clear_and_leaves_no_nan_in_the_json(monkeypatch, what, threshold):
    """A non-finite cost gives a non-finite contrast, and ``nan < CONTRAST_MIN`` is False: it used to be 'clear'.
    By hand: the contrast of a result that cannot be compared is 0.0, like the flat case, the reason 'noContrast'
    (also under a threshold of 0.0, which a contrast of 0.0 does not fall below: the reason must not rest on it).
    Without a threshold ("notJudged") the reason stays: that no contrast can be computed is a fact, not a verdict."""
    import json
    monkeypatch.setattr(ps, "CONTRAST_MIN", threshold)
    target = np.array([0.5, 0.4, 0.7])
    quad = _quadratic(target)
    at_pc = float("nan") if what == "at_pc" else 0.0
    cost = quad if what == "at_pc" else _nan_where(60 if what == "box_sample" else 14, quad)
    report = ps.confidence_report(cost, target, at_pc)
    assert report["state"] == _judged_as(threshold, "unclear") and "noContrast" in report["reasons"]
    assert report["state"] != "clear"
    assert report["contrast"] == 0.0
    shown = ps.SearchReport(pc=tuple(target), cost=0.0, start=None, mode="find", method="PSO+NM",
                            n_patterns=5, confidence=report)
    json.dumps(shown.to_dict(), allow_nan=False)                       # raises on NaN or Infinity


# ------------------------------------------------------------------ the polish stays in the physical box

@pytest.mark.parametrize("target, x0, expected", [
    ((0.5, 0.4, 1.75), (0.5, 0.4, 1.55), (0.5, 0.4, 1.6)),            # beyond PCz 1.6: stops at 1.6
    ((0.5, 0.4, 0.05), (0.5, 0.4, 0.3), (0.5, 0.4, 0.2)),             # below PCz 0.2
    ((-0.2, 0.4, 0.7), (0.2, 0.4, 0.7), (0.05, 0.4, 0.7)),            # left of PCx 0.05
    ((0.5, 1.3, 0.7), (0.5, 0.9, 0.7), (0.5, 1.0, 0.7)),              # above PCy 1.0
    ((0.5, 0.4, 0.7), (0.5, 0.4, 1.6), (0.5, 0.4, 0.7)),              # started ON the upper edge: still leaves it
    ((0.5, 0.4, 0.7), (0.05, 0.0, 0.2), (0.5, 0.4, 0.7)),             # started in the lower corner
])
def test_the_polish_never_leaves_what_a_detector_can_be(target, x0, expected):
    """A quadratic bowl is lowest at the point of the physical box nearest its centre: the clipped target."""
    pc, value = ps._polish(_quadratic(target), x0, ps.NM_OPTIONS)
    assert np.all(pc >= np.array(ps.PHYSICAL_LOW) - 1e-12) and np.all(pc <= np.array(ps.PHYSICAL_HIGH) + 1e-12)
    np.testing.assert_allclose(pc, expected, atol=0.003)


def test_a_polish_started_outside_the_box_is_moved_in_first_without_a_warning():
    """A ring point of a result on the edge lies outside the box; scipy would warn about such a start and clip it."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pc, _ = ps._polish(_quadratic((0.5, 0.4, 0.7)), (0.5, 0.4, 1.62), ps.NM_OPTIONS)
    np.testing.assert_allclose(pc, (0.5, 0.4, 0.7), atol=0.003)


@pytest.mark.parametrize("axis, edge, target", [
    (2, 1.6, 1.59),          # PCz upper edge, minimum 0.01 inside
    (2, 0.2, 0.21),          # PCz lower edge
    (0, 0.95, 0.94),         # PCx upper edge
    (1, 1.0, 0.99),          # PCy upper edge
])
def test_a_polish_started_exactly_on_an_edge_still_explores_that_axis(axis, edge, target):
    """scipy clips the start simplex to the bounds: a start ON the upper edge made the simplex of that axis a
    point and the axis was never explored (measured: minimum z = 1.59, start 1.6, result 1.6). By hand: the
    cost is 400 |pc - t|^2, lowest at t, which lies 0.01 inside the edge; the result must be within 1e-3 of t."""
    goal = np.array([0.5, 0.4, 0.7])
    goal[axis] = target
    x0 = goal.copy()
    x0[axis] = edge
    pc, _ = ps._polish(_quadratic(goal), x0, ps.NM_OPTIONS)
    assert abs(pc[axis] - target) < 1e-3, pc


class _NotFinite(Bowl):
    """A joint cost that is not a number anywhere (a stack whose bands no detector can see, a failed render)."""
    VALUE = float("nan")

    def subset(self, i):
        return _NotFinite(self.indexer, bowls=[self.bowls[i]])

    def __call__(self, pc):
        arr = np.asarray(pc, float)
        self.evaluations += int(np.atleast_2d(arr).shape[0])
        return self.VALUE if arr.ndim == 1 else np.full(arr.shape[0], self.VALUE)


@pytest.mark.parametrize("threshold", THRESHOLDS)
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
@pytest.mark.parametrize("kw", [
    {"mode": "find"},
    {"mode": "refine", "method": "Nelder-Mead", "start": (0.5, 0.4, 0.7)},
    {"mode": "refine", "method": "PSO", "start": (0.5, 0.4, 0.7)},
])
def test_a_search_whose_cost_is_not_finite_reports_no_cost_and_valid_json(monkeypatch, value, kw, threshold):
    """The whole search on a cost that is NaN or infinity, through to the JSON the route sends. By hand: no
    cost is a number, so ``cost`` (and ``far_box_cost`` of a find, which always tries the far box here because
    nothing can be called clear) are None, the PC stays a finite point, and ``allow_nan=False`` raises on any
    NaN or Infinity left in the text."""
    import json

    class Cost(_NotFinite):
        VALUE = value
    monkeypatch.setattr(ps, "CONTRAST_MIN", threshold)
    monkeypatch.setattr(ps, "JointPcCost", Cost)
    rep = ps.run_global(SimpleNamespace(target=(0.5, 0.4, 0.7)), PATTERNS, n_particles=5, n_iterations=3, **kw)
    shown = rep.to_dict()
    json.loads(json.dumps(shown, allow_nan=False))
    assert shown["cost"] is None and np.all(np.isfinite(shown["pc"]))
    assert shown["confidence"]["state"] == _judged_as(threshold, "unclear")
    if kw["mode"] == "find":
        assert shown["far_box_cost"] is None


def test_a_report_with_non_finite_cost_fields_is_plain_json():
    """Every number of the report that is a cost or a PC row is None when it is not finite."""
    import json
    rep = ps.SearchReport(pc=(0.5, 0.4, 0.7), cost=float("inf"), start=None, mode="find", method="PSO+NM",
                          n_patterns=2, far_box_cost=float("nan"),
                          per_pattern=np.array([[0.5, 0.4, 0.7], [float("nan"), 0.4, float("inf")]]))
    shown = rep.to_dict()
    json.dumps(shown, allow_nan=False)
    assert shown["cost"] is None and shown["far_box_cost"] is None
    assert shown["per_pattern"] == [[0.5, 0.4, 0.7], [None, 0.4, None]]


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_a_search_whose_minimum_lies_beyond_the_physical_limit_stays_inside_it(bowl, monkeypatch, threshold):
    """The bowl's centre is z = 1.7 (find) or 1.75 (refine), above PCz 1.6; the answer is held at 1.6 and a find
    does not call it clear.

    The find's bowl has stiffness 24, so it is flat farther than sqrt(8.5 / 24) = 0.595 from its centre (by hand).
    The wide box ends at z = 1.05, 0.65 from the centre: flat everywhere in it, the first result is unclear and the
    far box is tried. In the far box (z 0.85 to 1.55) the slope reaches down to z about 1.1 with x and y at the
    bowl's: a large part of the box, which the swarm cannot miss. (The refine's default stiffness 130, reach 0.256,
    would be seen from the far box only above z = 1.444, a few percent of it: 3 runs in 200 missed it and
    stayed at the middle of the wide box, z = 0.7.) The bowl at 1.7 keeps falling past 1.6, so a ring point at
    z = 1.62 is cheaper than the answer: 'notAMinimum' is the reason it is unclear, not a missing contrast.
    Without a threshold the state is "notJudged", the far box is tried all the same and 'notAMinimum' stays."""
    monkeypatch.setattr(ps, "CONTRAST_MIN", threshold)
    rep = ps.run_global(bowl((0.5, 0.3, 1.7), coefs=[24.0] * len(PATTERNS)), PATTERNS, mode="find",
                        with_spread=False)
    assert rep.pc[2] <= ps.PHYSICAL_HIGH[2] + 1e-12 and rep.pc[2] > 1.57
    assert rep.confidence["state"] == _judged_as(threshold, "unclear") and rep.far_box_used is True
    assert "notAMinimum" in rep.confidence["reasons"]
    assert ("noContrast" in rep.confidence["reasons"]) is False
    ref = ps.run_global(bowl((0.5, 0.3, 1.75)), PATTERNS, mode="refine", method="Nelder-Mead",
                        start=(0.5, 0.3, 1.55), with_spread=False)
    assert ref.pc[2] <= ps.PHYSICAL_HIGH[2] + 1e-12 and ref.pc[2] > 1.57


# ------------------------------------------------------------------ what a refine is given is checked first

@pytest.fixture
def no_cost(monkeypatch):
    """A search that is refused must be refused BEFORE any cost is built (the build is the expensive part)."""
    def refuse(*a, **k):
        pytest.fail("the cost was built for a search that should have been refused")
    monkeypatch.setattr(ps, "JointPcCost", refuse)


@pytest.mark.parametrize("method", ["Nelder-Mead", "PSO"])
@pytest.mark.parametrize("start, axes", [
    ((0.5, 0.4, 2.0), ["z"]),                          # beyond PCz 1.6
    ((0.5, -0.1, 0.7), ["y"]),
    ((0.01, 0.4, 0.7), ["x"]),                         # below PCx 0.05
    ((0.5, float("nan"), 0.7), ["y"]),
    ((float("inf"), 0.4, 0.7), ["x"]),
    ((0.5, 0.4), ["x", "y", "z"]),                     # two values
    ((0.5, 0.4, 0.7, 0.1), ["x", "y", "z"]),
    (("a", "b", "c"), ["x", "y", "z"]),
])
def test_a_refine_refuses_a_start_that_is_not_a_pc_of_the_physical_box(no_cost, method, start, axes):
    with pytest.raises(ps.SearchError) as exc:
        ps.run_global(SimpleNamespace(target=(0, 0, 0)), PATTERNS, mode="refine", method=method, start=start)
    assert exc.value.code == "badStart" and exc.value.params == {"axes": axes}


@pytest.mark.parametrize("method", ["Nelder-Mead", "PSO"])
@pytest.mark.parametrize("limit", [-0.05, 0.0, float("nan"), float("inf"), 1.5, "wide"])
def test_a_refine_refuses_a_search_limit_outside_zero_to_one(no_cost, method, limit):
    with pytest.raises(ps.SearchError) as exc:
        ps.run_global(SimpleNamespace(target=(0, 0, 0)), PATTERNS, mode="refine", method=method,
                      start=(0.5, 0.4, 0.7), search_limit=limit)
    assert exc.value.code == "badSearchLimit"


@pytest.mark.parametrize("method", ["Nelder-Mead", "PSO"])
@pytest.mark.parametrize("limit", [1e-5, 5e-17, 9.9e-5])
def test_a_refine_refuses_a_search_limit_too_small_to_make_a_box(no_cost, method, limit):
    """Below ~5e-17 start +- limit rounds to start itself, so lo == hi in the swarm's box. The smallest
    accepted limit is ps.MIN_SEARCH_LIMIT = 1e-4."""
    with pytest.raises(ps.SearchError) as exc:
        ps.run_global(SimpleNamespace(target=(0, 0, 0)), PATTERNS, mode="refine", method=method,
                      start=(0.5, 0.4, 0.7), search_limit=limit)
    assert exc.value.code == "badSearchLimit"


def test_a_search_limit_of_exactly_the_smallest_is_accepted():
    assert ps.MIN_SEARCH_LIMIT == 1e-4
    got = ps.check_refine("PSO", (0.5, 0.4, 0.7), 1e-4)
    assert got[2] == 1e-4


@pytest.mark.parametrize("method", ["Powell", "", "NM", None, "PSO+NM"])
def test_a_refine_refuses_an_unknown_method_instead_of_running_nelder_mead(no_cost, method):
    with pytest.raises(ps.SearchError) as exc:
        ps.run_global(SimpleNamespace(target=(0, 0, 0)), PATTERNS, mode="refine", method=method,
                      start=(0.5, 0.4, 0.7))
    assert exc.value.code == "badMethod"


def test_the_edges_of_what_is_accepted_are_accepted(bowl):
    """A start exactly on the physical box, a search limit of exactly 1, the method in any case.

    A search limit of 1 makes the swarm's box the whole physical box (0.9 x 1.0 x 1.4). The bowl of stiffness 5 is
    flat only farther than sqrt(8.5 / 5) = 1.30 from its centre, and no corner of that box is as far (the farthest,
    y 1.0, z 1.6, x 0.05 or 0.95, is sqrt(0.45^2 + 0.6^2 + 0.9^2) = 1.17; the start's corner is 1.08), so the slope
    is seen from every particle. (At the default stiffness 130, reach 0.256, the bowl fills 6 % of the box and 4 runs in 200
    never saw it and stayed at the start.)"""
    rep = ps.run_global(bowl((0.5, 0.4, 0.7), coefs=[5.0] * len(PATTERNS)), PATTERNS, mode="refine", method="pso",
                        search_limit=1.0, start=(0.05, 0.0, 1.6), with_spread=False)
    assert rep.method == "PSO+NM" and np.abs(np.array(rep.pc) - (0.5, 0.4, 0.7)).max() < 0.02
    rep = ps.run_global(bowl((0.5, 0.4, 0.7)), PATTERNS, mode="refine", method="  nelder-mead ",
                        start=(0.5, 0.4, 0.7), with_spread=False)
    assert rep.method == "NM"


# ------------------------------------------------------------------ the far box, the descent

def test_a_clear_far_box_result_is_adopted_although_it_is_cheaper_by_less_than_the_ring(monkeypatch):
    """Two basins of stiffness 130. A in the first box: cost 4.30 on a plateau of 8.5, contrast 1 - 4.30 / 8.5 = 0.494,
    unclear under the fast tests' 0.5; its ring of +-0.02 costs a median of 130 * 3 * 0.0004 = 0.156 (by hand: six axis
    points 0.052, eight corners 0.156). B in the far box: cost 4.20, contrast 0.506, CLEAR itself, but cheaper than A by
    only 0.10 < 0.156. The rule adopts a far result that is clear OR clearly cheaper: B is adopted.
    Without a threshold nothing is clear, B is cheaper by less than the ring: the far box is tried and NOT adopted."""
    a, b = (0.5, 0.4, 0.7), (0.5, 0.3, 1.4)
    monkeypatch.setattr(ps, "JointPcCost", _basins((a, 4.30, 130.0), (b, 4.20, 130.0)))
    rep = ps.run_global(SimpleNamespace(target=(0, 0, 0)), PATTERNS, mode="find", with_spread=False)
    assert np.abs(np.array(rep.first_box_pc) - a).max() < 0.01
    assert rep.far_box_cost == pytest.approx(4.20, abs=0.01)
    assert rep.far_box_used is True and np.abs(np.array(rep.pc) - b).max() < 0.02
    assert rep.confidence["state"] == "clear"


def test_without_a_threshold_a_far_result_cheaper_by_less_than_the_ring_is_not_adopted(monkeypatch):
    """The landscape of the test above with CONTRAST_MIN None: B (4.20) is cheaper than A (4.30) by 0.10, the ring
    around A costs a median of 0.156 (by hand, see above), and no result is clear. The far box was tried, both
    candidates are reported, A stays, and nothing is called clear."""
    monkeypatch.setattr(ps, "CONTRAST_MIN", None)
    a, b = (0.5, 0.4, 0.7), (0.5, 0.3, 1.4)
    monkeypatch.setattr(ps, "JointPcCost", _basins((a, 4.30, 130.0), (b, 4.20, 130.0)))
    rep = ps.run_global(SimpleNamespace(target=(0, 0, 0)), PATTERNS, mode="find", with_spread=False)
    assert rep.far_box_cost == pytest.approx(4.20, abs=0.01)                     # tried
    assert np.abs(np.array(rep.far_box_pc) - b).max() < 0.02
    assert rep.far_box_used is False and np.abs(np.array(rep.pc) - a).max() < 0.01
    assert rep.confidence["state"] == "notJudged"


def _with_a_pit(p, q, pit_stiffness=1e4):
    """A broad bowl (base 1.0, stiffness 400) at ``p`` and, ``q`` away, a pit of radius about 0.005 that is cheaper
    (base 0.9). A search that converges on the bowl never sees the pit; the ring of +-0.02 around ``p`` reaches it."""
    return _basins((p, 1.0, 400.0), (q, 0.9, pit_stiffness))


def _judged(monkeypatch):
    calls = []
    real = ps.confidence_report
    monkeypatch.setattr(ps, "confidence_report", lambda *a, **k: (calls.append(real(*a, **k)), calls[-1])[1])
    return calls


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_a_ring_that_shows_a_cheaper_point_gets_one_more_descent_from_it(monkeypatch, threshold):
    """Bowl at P = (0.52, 0.4, 0.7) (cost 1.0), a pit at Q = P + (0.02, 0, 0), the ring point (cost 0.9, 0.1 cheaper).
    A refine from (0.50, 0.4, 0.7) converges on P, the ring shows the pit (notAMinimum), the descent from it ends at Q;
    the answer is Q at cost 0.9, judged again and no longer 'notAMinimum'. The same without a threshold."""
    monkeypatch.setattr(ps, "CONTRAST_MIN", threshold)
    p, q = (0.52, 0.4, 0.7), (0.54, 0.4, 0.7)
    monkeypatch.setattr(ps, "JointPcCost", _with_a_pit(p, q))
    calls = _judged(monkeypatch)
    rep = ps.run_global(SimpleNamespace(target=(0, 0, 0)), PATTERNS, mode="refine", method="Nelder-Mead",
                        start=(0.50, 0.4, 0.7), with_spread=False)
    assert len(calls) == 2 and "notAMinimum" in calls[0]["reasons"]
    assert np.abs(np.array(calls[0]["best_point"]) - q).max() < 0.002          # the ring point the descent started at
    assert np.abs(np.array(rep.pc) - q).max() < 0.002 and rep.cost == pytest.approx(0.9, abs=0.002)
    assert "notAMinimum" not in rep.confidence["reasons"] and rep.confidence is calls[1]
    assert (rep.confidence["state"] == "notJudged") is (threshold is None)


def test_a_descent_that_finds_nothing_cheaper_is_not_judged_a_second_time(monkeypatch):
    """The cost falls toward z = 1.9, beyond what a detector can be: the answer is held at z = 1.6 and a ring point
    at z = 1.62 is cheaper (notAMinimum, rightly). The descent from it is held at 1.6 too, finds the same cost, and the
    unchanged result is not judged again."""
    class Slope(Bowl):
        def __call__(self, pc):
            arr = np.asarray(pc, float)
            rows = np.atleast_2d(arr)
            out = 130.0 * ((rows[:, 0] - 0.5) ** 2 + (rows[:, 1] - 0.3) ** 2) + 5.0 * (1.9 - rows[:, 2])
            self.evaluations += rows.shape[0]
            return float(out[0]) if arr.ndim == 1 else out
    monkeypatch.setattr(ps, "JointPcCost", Slope)
    calls = _judged(monkeypatch)
    rep = ps.run_global(SimpleNamespace(target=(0, 0, 0)), PATTERNS, mode="refine", method="Nelder-Mead",
                        start=(0.5, 0.3, 1.59), with_spread=False)
    assert rep.pc[2] == pytest.approx(1.6, abs=1e-9) and rep.cost == pytest.approx(5.0 * 0.3, abs=1e-6)
    assert len(calls) == 1 and "notAMinimum" in rep.confidence["reasons"]


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_the_edges_are_those_of_the_answer_after_the_descent_not_of_the_one_before(monkeypatch, threshold):
    """The wide box reaches x = 0.7; a result within 0.02 of it is 'on the edge'. Bowl at P = (0.662, 0.4, 0.5) (0.038
    from the edge), pit at Q = P + (0.02, 0, 0) = x 0.682 (0.018 from it). The swarm and Nelder-Mead end at P (no edge),
    the descent ends at Q: the answer is on the x edge. The swarm is replaced by a swarm that has found P exactly (a real
    one, or the simplex of Nelder-Mead after it, may step into the pit by chance), so the run is the same every time.
    The edges are reported with and without a threshold."""
    monkeypatch.setattr(ps, "CONTRAST_MIN", threshold)
    p, q = (0.662, 0.4, 0.5), (0.682, 0.4, 0.5)
    monkeypatch.setattr(ps, "JointPcCost", _with_a_pit(p, q))
    monkeypatch.setattr(ps, "_swarm", lambda cost, centre, limit, n, it: (
        np.array(p), ps._clip_box(centre, np.asarray(limit, float))))
    rep = ps.run_global(SimpleNamespace(target=(0, 0, 0)), PATTERNS, mode="find", with_spread=False)
    assert np.abs(np.array(rep.first_box_pc) - p).max() < 0.002                  # before the descent: P, not on an edge
    assert np.abs(np.array(rep.pc) - q).max() < 0.002
    assert rep.on_edge == ["x"]
    assert rep.box["high"][0] == pytest.approx(0.7)


# ======================================================================================
# Slow: patterns rendered at a PC the test chose, indexed with the application's indexer
# ======================================================================================

slow = pytest.mark.slow


@pytest.fixture(scope="module")
def synth():
    from tests import pc_synth
    with pc_synth.cpu_band_detection():
        yield pc_synth


def _error(rep, truth):
    return float(np.abs(np.array(rep.pc) - np.array(truth)).max())


@slow
def test_find_recovers_the_pc_the_patterns_were_rendered_at(synth):
    truth = (0.52, 0.31, 0.62)
    pats = synth.render_patterns((80, 80), truth, n=9, seed=7)
    rep = ps.run_global(synth.make_indexer((80, 80)), pats, mode="find")
    assert _error(rep, truth) < 0.02
    assert rep.recentred == 0
    assert rep.per_pattern.shape == (9, 3)


@slow
def test_find_with_the_fewest_patterns_it_accepts(synth):
    truth = (0.52, 0.31, 0.62)
    pats = synth.render_patterns((80, 80), truth, n=5, seed=7)
    rep = ps.run_global(synth.make_indexer((80, 80)), pats, mode="find")
    assert _error(rep, truth) < 0.02


@slow
def test_find_on_a_landscape_detector_with_a_large_distance(synth):
    truth = (0.50, 0.33, 0.85)                       # an Oxford-like PCz on a 60 x 80 pattern
    pats = synth.render_patterns((60, 80), truth, n=9, seed=7)
    rep = ps.run_global(synth.make_indexer((60, 80)), pats, mode="find")
    assert _error(rep, truth) < 0.02


@slow
def test_find_does_not_need_the_camera_tilt(synth):
    """The cost compares angles between bands, which a rigid tilt of the whole frame leaves unchanged
    (the measurement on three files): patterns rendered with a 10 degree camera tilt are searched
    with an indexer that knows no tilt."""
    truth = (0.52, 0.31, 0.62)
    pats = synth.render_patterns((80, 80), truth, n=9, seed=7, tilt=10.0)
    rep = ps.run_global(synth.make_indexer((80, 80), tilt=0.0), pats, mode="find")
    assert _error(rep, truth) < 0.03


@slow
def test_find_with_two_phases_in_the_indexer(synth):
    """Nickel patterns, an indexer holding nickel AND aluminium (both face-centred cubic, a = 3.52 and
    4.05 A): the cost takes, per pattern, the phase that matches best, and the PC is still found."""
    from orix.crystal_map import Phase, PhaseList
    from ebsd_utils import sanitize_cif
    al = Path(__file__).resolve().parents[1] / "Test_data" / "Al.cif"
    if not al.is_file():
        pytest.skip("Al.cif is not in this checkout's Test_data/")
    truth = (0.52, 0.31, 0.62)
    pats = synth.render_patterns((80, 80), truth, n=9, seed=7)
    phases = PhaseList([synth.ni_phase_list()[0], Phase.from_cif(sanitize_cif(str(al)))])
    rep = ps.run_global(synth.make_indexer((80, 80), phase_list=phases), pats, mode="find")
    assert _error(rep, truth) < 0.03


def _noise_patterns():
    return np.random.default_rng(2).normal(0, 1, (9, 80, 80)).astype(np.float32)


@slow
def test_pure_noise_is_not_judged_without_a_threshold(synth, monkeypatch):
    """CONTRAST_MIN None: noise gets state "notJudged" (never "clear"), the contrast is still a finite number,
    and the report is plain JSON."""
    import json
    monkeypatch.setattr(ps, "CONTRAST_MIN", None)
    rep = ps.run_global(synth.make_indexer((80, 80)), _noise_patterns(), mode="find")
    assert rep.confidence["state"] == "notJudged"
    assert np.isfinite(rep.confidence["contrast"])
    json.dumps(rep.to_dict(), allow_nan=False)


@slow
def test_pure_noise_is_never_called_clear_with_a_threshold(synth, monkeypatch):
    """With a threshold set, noise is not "clear". 0.3 is NOT a proposal for the threshold: it lies well above
    the noise contrasts measured on real EBSD (0.05 to 0.07), so this test only
    holds the rule "contrast below the threshold is never clear"."""
    monkeypatch.setattr(ps, "CONTRAST_MIN", 0.3)
    rep = ps.run_global(synth.make_indexer((80, 80)), _noise_patterns(), mode="find")
    assert rep.confidence["state"] != "clear"


# ------------------------------------------------------------------ contrast of the PC a refine started from

def test_contrast_at_is_one_minus_cost_over_typical_by_hand():
    cost = _quadratic((0.5, 0.4, 0.7))                   # 400 |d|^2
    # 0.1 away in x: 400 * 0.01 = 4; typical 8 -> 1 - 4 / 8 = 0.5
    assert ps.contrast_at(cost, (0.6, 0.4, 0.7), 8.0) == pytest.approx(0.5)
    # 0.3 away: cost 36, typical 8 -> 1 - 4.5 = -3.5, held to -1 as the result's own contrast is
    assert ps.contrast_at(cost, (0.8, 0.4, 0.7), 8.0) == -1.0
    assert ps.contrast_at(cost, (0.5, 0.4, 0.7), 8.0) == 1.0              # at the minimum: cost 0
    assert ps.contrast_at(cost, (0.6, 0.4, 0.7), None) is None
    assert ps.contrast_at(cost, (0.6, 0.4, 0.7), 0.0) is None             # no typical cost to measure against
    assert ps.contrast_at(cost, (0.6, 0.4, 0.7), float("nan")) is None


def test_a_refine_reports_the_contrast_of_its_start_on_the_scale_of_its_result(bowl):
    """All patterns share one bowl, cost = 0.05 + 130 |pc - target|^2. The start (0.50, 0.33, 0.60) is
    (0.02, 0.02, 0.02) from the target: 130 * 3 * 0.0004 = 0.156, so the cost is 0.206 (by hand)."""
    target = (0.52, 0.31, 0.62)
    rep = ps.run_global(bowl(target), PATTERNS, mode="refine", method="Nelder-Mead", start=(0.50, 0.33, 0.60))
    typical = rep.confidence["typical_cost"]
    assert rep.contrast_at_start == pytest.approx(1.0 - 0.206 / typical, abs=1e-9)
    assert rep.contrast_at_start < rep.confidence["contrast"]              # the result fits better than the start
    assert rep.to_dict()["contrast_at_start"] == pytest.approx(rep.contrast_at_start)


def test_a_find_has_no_contrast_of_a_start(bowl):
    rep = ps.run_global(bowl((0.52, 0.31, 0.62)), PATTERNS, mode="find")
    assert rep.contrast_at_start is None and rep.to_dict()["contrast_at_start"] is None


def test_the_contrast_of_the_start_costs_one_evaluation_more(bowl):
    """Same search, same start: the cost object is called once more than without it."""
    seen = {}
    real = ps.contrast_at

    def spy(cost, pc, typical):
        before = cost.evaluations
        out = real(cost, pc, typical)
        seen["extra"] = cost.evaluations - before
        return out

    import unittest.mock as mock
    with mock.patch.object(ps, "contrast_at", spy):
        ps.run_global(bowl((0.52, 0.31, 0.62)), PATTERNS, mode="refine", method="Nelder-Mead", start=(0.50, 0.33, 0.60))
    assert seen["extra"] == 1


# ------------------------------------------------------------------ quality_of: the few numbers a result is shown with

def _report(**kw):
    base = {"confidence": {"contrast": 0.8, "typical_cost": 8.0, "reasons": []}, "on_edge": [], "n_patterns": 9}
    base.update(kw)
    return base


def test_quality_of_a_clean_result_has_the_contrast_and_no_warning():
    assert ps.quality_of(_report()) == {"contrast": 0.8, "warning": None, "axes": [], "n_patterns": 9}


def test_quality_of_a_result_at_the_edge_names_the_axes():
    assert ps.quality_of(_report(on_edge=["x", "z"])) == {
        "contrast": 0.8, "warning": "edge", "axes": ["x", "z"], "n_patterns": 9}


def test_quality_of_a_result_that_is_no_minimum_has_that_warning_and_no_axes():
    out = ps.quality_of(_report(confidence={"contrast": 0.3, "typical_cost": 8.0, "reasons": ["notAMinimum"]}))
    assert out == {"contrast": 0.3, "warning": "notAMinimum", "axes": [], "n_patterns": 9}


def test_quality_of_names_the_edge_when_both_apply():
    out = ps.quality_of(_report(on_edge=["y"], confidence={"contrast": 0.3, "typical_cost": 8.0,
                                                           "reasons": ["notAMinimum", "noContrast"]}))
    assert out["warning"] == "edge" and out["axes"] == ["y"]


def test_quality_of_a_contrast_that_could_not_be_computed_is_none_not_zero():
    """confidence_report reports 0.0 where a cost is not a number; its typical cost is then None."""
    out = ps.quality_of(_report(confidence={"contrast": 0.0, "typical_cost": None, "reasons": ["noContrast"]}))
    assert out["contrast"] is None


def test_quality_of_a_report_without_what_it_needs_does_not_raise():
    assert ps.quality_of({}) == {"contrast": None, "warning": None, "axes": [], "n_patterns": None}
    assert ps.quality_of(None) == {"contrast": None, "warning": None, "axes": [], "n_patterns": None}


# ------------------------------------------------------------------ a second position

def test_the_rule_of_the_second_position_is_the_one_fixed_before_any_measurement():
    assert ps.SECOND_POSITION_DIST == 0.05 and ps.SECOND_POSITION_COST == 1.10


RESULT = (0.50, 0.40, 0.70)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("step, expected", [(0.049, False), (0.051, True)])
def test_a_candidate_is_a_second_position_only_beyond_0_05_in_x_y_or_z(axis, sign, step, expected):
    """By hand: the candidate differs from the result by `step` in one axis and by nothing in the others, at the
    same cost. 0.049 is not apart, 0.051 is; the direction does not matter."""
    other = list(RESULT)
    other[axis] += sign * step
    got = ps.second_position_of(RESULT, 2.0, other, 2.0)
    assert (got is not None) is expected
    if expected:
        np.testing.assert_allclose(got["pc"], other, atol=1e-12)
        assert got["cost"] == 2.0


def test_the_distance_is_the_largest_axis_not_the_length():
    """0.04 in all three axes is a length of 0.069 but not 0.05 in any axis: not apart."""
    assert ps.second_position_of(RESULT, 2.0, (0.54, 0.44, 0.74), 2.0) is None


@pytest.mark.parametrize("ratio, expected", [(1.09, True), (1.11, False), (1.0, True), (0.5, True), (3.0, False)])
def test_a_candidate_is_a_second_position_only_up_to_1_10_times_the_cost(ratio, expected):
    """The result costs 2.0. 1.09 x 2.0 = 2.18 qualifies, 1.11 x 2.0 = 2.22 does not; a candidate cheaper than the
    result (ratio below one) qualifies."""
    got = ps.second_position_of(RESULT, 2.0, (0.60, 0.40, 0.70), 2.0 * ratio)
    assert (got is not None) is expected
    if expected:
        assert got["cost"] == pytest.approx(2.0 * ratio)


def test_equal_positions_and_costs_that_are_not_numbers_give_no_second_position():
    assert ps.second_position_of(RESULT, 2.0, RESULT, 2.0) is None
    for bad in (float("nan"), float("inf"), None):
        assert ps.second_position_of(RESULT, 2.0, (0.6, 0.4, 0.7), bad) is None
        assert ps.second_position_of(RESULT, bad, (0.6, 0.4, 0.7), 2.0) is None
    assert ps.second_position_of(RESULT, 0.0, (0.6, 0.4, 0.7), 0.0) is None          # no ratio to a cost of zero
    assert ps.second_position_of(RESULT, 2.0, None, 2.0) is None
    assert ps.second_position_of(RESULT, 2.0, (float("nan"), 0.4, 0.7), 2.0) is None


A_BASIN, B_BASIN = (0.5, 0.4, 0.7), (0.5, 0.3, 1.5)         # one in the first box, one in the far box, 0.8 apart in z


@pytest.mark.parametrize("threshold", THRESHOLDS)
@pytest.mark.parametrize("far_base, second_is_first_box", [(4.7, True), (4.4, None)])
def test_when_the_far_box_is_adopted_the_first_box_is_the_second_position_within_1_10(monkeypatch, threshold,
                                                                                      far_base, second_is_first_box):
    """Basin A (first box) costs 5.0, basin B (far box) costs `far_base`; B is cheaper by more than the ring (0.156)
    so it is adopted. A costs 5.0 / 4.7 = 1.064 times the result: a second position, at A's centre. With B at 4.4
    A costs 5.0 / 4.4 = 1.136 times: none."""
    monkeypatch.setattr(ps, "CONTRAST_MIN", threshold)
    monkeypatch.setattr(ps, "JointPcCost", _basins((A_BASIN, 5.0, 130.0), (B_BASIN, far_base, 12.0)))
    rep = ps.run_global(SimpleNamespace(target=(0, 0, 0)), PATTERNS, mode="find")
    assert rep.far_box_used is True
    if second_is_first_box is None:
        assert rep.second_position is None and rep.to_dict()["second_position"] is None
        return
    assert np.abs(np.array(rep.second_position["pc"]) - A_BASIN).max() < 0.01
    assert rep.second_position["cost"] == pytest.approx(5.0, abs=0.01)
    assert rep.to_dict()["second_position"]["pc"] == [float(v) for v in rep.second_position["pc"]]


@pytest.mark.parametrize("threshold", THRESHOLDS)
@pytest.mark.parametrize("far_base, has_second", [(5.4, True), (5.6, False), (4.95, True)])
def test_when_the_first_box_is_kept_the_far_box_is_the_second_position_within_1_10(monkeypatch, threshold,
                                                                                    far_base, has_second):
    """Basin A costs 5.0 and stays the result. B at 5.4 is 1.08 times that: a second position at B's centre; at
    5.6 it is 1.12 times: none. B at 4.95 is cheaper than the result but not clearly (0.05 < the ring 0.156), so
    it is not adopted, and as it is cheaper it is a second position."""
    monkeypatch.setattr(ps, "CONTRAST_MIN", threshold)
    monkeypatch.setattr(ps, "JointPcCost", _basins((A_BASIN, 5.0, 130.0), (B_BASIN, far_base, 12.0)))
    rep = ps.run_global(SimpleNamespace(target=(0, 0, 0)), PATTERNS, mode="find")
    assert rep.far_box_used is False
    if not has_second:
        assert rep.second_position is None
        return
    assert np.abs(np.array(rep.second_position["pc"]) - B_BASIN).max() < 0.02
    assert rep.second_position["cost"] == pytest.approx(far_base, abs=0.01)


def test_a_find_whose_boxes_end_at_the_same_position_has_no_second_position(bowl, monkeypatch):
    """Without a threshold the far box runs and walks to the same minimum: the candidates are 0 apart."""
    monkeypatch.setattr(ps, "CONTRAST_MIN", None)
    rep = ps.run_global(bowl((0.52, 0.31, 0.62)), PATTERNS, mode="find")
    assert rep.far_box_pc is not None and rep.second_position is None


def test_a_find_that_did_not_try_the_far_box_has_no_second_position(bowl):
    rep = ps.run_global(bowl((0.52, 0.31, 0.62)), PATTERNS, mode="find")
    assert rep.far_box_pc is None and rep.second_position is None


def test_a_refine_has_no_second_position_whatever_the_landscape(monkeypatch):
    monkeypatch.setattr(ps, "JointPcCost", _basins((A_BASIN, 5.0, 130.0), (B_BASIN, 5.0, 12.0)))
    for kw in ({"method": "Nelder-Mead"}, {"method": "PSO", "search_limit": 0.5}):
        rep = ps.run_global(SimpleNamespace(target=(0, 0, 0)), PATTERNS, mode="refine", start=A_BASIN, **kw)
        assert rep.second_position is None and rep.to_dict()["second_position"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_find_whose_cost_is_not_a_number_has_no_second_position(monkeypatch, value):
    class Cost(_NotFinite):
        VALUE = value
    monkeypatch.setattr(ps, "CONTRAST_MIN", None)
    monkeypatch.setattr(ps, "JointPcCost", Cost)
    rep = ps.run_global(SimpleNamespace(target=(0.5, 0.4, 0.7)), PATTERNS, n_particles=5, n_iterations=3, mode="find")
    assert rep.far_box_pc is not None and rep.second_position is None


SECOND = {"pc": [0.40, 0.30, 1.50], "cost": 5.0}


def test_quality_of_a_result_with_a_second_position_names_it_and_the_best_cost():
    out = ps.quality_of(_report(second_position=SECOND, cost=4.7))
    assert out == {"contrast": 0.8, "warning": "secondPosition", "axes": [], "n_patterns": 9,
                   "second_position": SECOND, "best_cost": 4.7}


def test_quality_of_keeps_the_edge_or_no_minimum_warning_and_still_names_the_second_position():
    edge = ps.quality_of(_report(second_position=SECOND, cost=4.7, on_edge=["z"]))
    assert edge["warning"] == "edge" and edge["axes"] == ["z"] and edge["second_position"] == SECOND
    notmin = ps.quality_of(_report(second_position=SECOND, cost=4.7,
                                   confidence={"contrast": 0.3, "typical_cost": 8.0, "reasons": ["notAMinimum"]}))
    assert notmin["warning"] == "notAMinimum" and notmin["second_position"] == SECOND and notmin["best_cost"] == 4.7


def test_quality_of_a_result_without_one_has_no_second_position_keys():
    assert "second_position" not in ps.quality_of(_report(second_position=None, cost=4.7))
    assert "best_cost" not in ps.quality_of(_report(cost=4.7))
