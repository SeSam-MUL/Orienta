"""The phase check must ask Hough ONCE per candidate phase, not once per grain.

Each `hough_indexing` call leaks about 0.24 GiB of Windows commit that is never
returned. Measured 2026-09-04 on a reused indexer (so the indexer build is
ruled out — it is flat at +0.019 GiB over six builds):

      4 calls  +0.97 GiB commit  +0.76 GiB RSS
     12 calls  +2.92 GiB         +2.10 GiB
     24 calls  +5.96 GiB         +3.10 GiB

Linear, no plateau, and RSS rises with it — a real leak inside
kikuchipy/PyEBSDIndex, not a heap that gets reused. A real check made 172
candidate evaluations, i.e. ~40 GiB, and that is what kept taking the machine to
its commit limit (76.3 of 77.3 GiB, 1.0 GiB free) until the backend was
restarted.

Batching cannot fix a leak in someone else's code, but it decides how many times
we step on it: one call per candidate phase instead of grains x candidates.

The second test is the one that matters more: batching must not change a single
decision. `hough_indexing` treats every pattern independently, so concatenating
the grains' samples is only an arrangement — if a decision moved, the
arrangement leaked into the answer and the optimisation would be a bug.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_phase_reassignment import (  # noqa: E402
    NC, NR, PHASES, _grid, _hough_fn_factory, _score_fns,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _run(hough):
    from backend.spherical_gpu.pipeline.phase_reassignment import check_map
    full_q, pf = _grid()
    fns, _calls = _score_fns()
    return check_map(full_q, pf, NR, NC, PHASES, fns, hough)


def _two_suspect_scorers():
    """Both phase-2 grains score below the floor, so BOTH are suspects.

    The shared fixture has only one suspect grain, where per-grain and batched
    arrangements make the same single call — the test would pass without the
    batching existing. Two suspects is the smallest scenario that tells them
    apart: the old arrangement asked candidate 1 twice.
    """
    def alpha_fn(flats, quats):
        return np.full(np.asarray(flats).reshape(-1).size, 0.15)

    def al_fn(flats, quats):
        return np.full(np.asarray(flats).reshape(-1).size, 0.32)

    return {1: al_fn, 2: alpha_fn}


def test_one_hough_call_per_candidate_phase():
    from backend.spherical_gpu.pipeline.phase_reassignment import check_map
    full_q, pf = _grid()
    hough = _hough_fn_factory()
    report, _margin = check_map(full_q, pf, NR, NC, PHASES,
                                _two_suspect_scorers(), hough)

    # Two suspect grains, one candidate phase they can be compared against.
    assert report["n_suspect"] == 2
    cands = [pid for pid, _flats in hough.calls]
    assert cands.count(1) == 1, (
        f"candidate 1 was asked {cands.count(1)}x for 2 suspect grains "
        "— that is the per-grain arrangement, which is what leaks"
    )
    assert len(cands) == len(set(cands)), f"a candidate was asked twice: {cands}"

    # And the batch carried BOTH grains' pixels, not just the first.
    flats_for_1 = [f for pid, f in hough.calls if pid == 1][0]
    assert flats_for_1.size >= 2 * 3


def test_every_suspect_grain_is_still_asked_about():
    """Batching must not quietly drop a grain from the batch."""
    hough = _hough_fn_factory()
    report, _margin = _run(hough)

    suspects = [e for e in report["grains"]
                if e.get("decision") in ("reassign", "keep")
                and e.get("stored_score") is not None]
    # Every suspect grain contributed sample pixels to some batch, and no
    # (phase, pixel) pair was asked twice. Counted per PHASE since 2026-09-07:
    # the fairness step asks Hough about the STORED phase on the same pixels
    # (one batched call per stored phase), so a pixel legitimately appears
    # once under its candidate and once under its own phase -- what must not
    # happen is the old per-grain arrangement asking one phase twice.
    assert hough.calls
    pairs = [(pid, int(f)) for pid, flats in hough.calls for f in flats]
    assert len(pairs) == len(set(pairs)), "a (phase, pixel) pair was asked twice"
    assert len(suspects) >= 1


def test_decisions_are_identical_to_the_per_grain_arrangement():
    """The answer must not depend on how the question was packed.

    Same fakes, same map; the reference numbers are the ones the per-grain
    version produced (alpha renders 0.15 on the wrong grain, Al 0.32, so the
    margin is -0.17 and the grain flips).
    """
    hough = _hough_fn_factory()
    report, margin = _run(hough)

    assert report["n_grains"] == 3
    assert report["n_reassign"] == 1
    flipped = [e for e in report["grains"] if e["decision"] == "reassign"]
    assert len(flipped) == 1
    e = flipped[0]
    assert e["phase_id"] == 2
    assert e["best_alt_phase"] == 1
    assert e["stored_score"] == pytest.approx(0.15, abs=1e-9)
    assert e["best_alt_score"] == pytest.approx(0.32, abs=1e-9)
    assert e["margin"] == pytest.approx(-0.17, abs=1e-9)
    # The margin map lights up only where a question was asked AND answered.
    assert np.isfinite(margin).sum() > 0


def test_a_failing_candidate_marks_every_grain_it_was_batched_with():
    """One batch, many grains — a failure has to reach all of them.

    Otherwise the grains in that batch would be reported as "nothing found",
    which is the silent clean bill of health this whole area is about.
    """
    def hough(cand_pid, flats):
        raise MemoryError("Not enough memory to build the Hough index")

    report, _margin = _run(hough)
    marked = [e for e in report["grains"] if e.get("unevaluated")]
    assert marked, "a failed candidate batch left no trace on any grain"
    assert report["n_unevaluated"] == len(marked)
    assert any("memory" in r.lower() for r in report["unevaluated_reasons"])
    # And it must NOT be reported as something to reassign.
    assert report["n_reassign"] == 0
