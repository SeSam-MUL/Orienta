"""Applying an island finding writes exactly what the check reported.

The check and the apply are two steps with a report between them. The failure
that matters is a silent disagreement: the check says "this pixel should be
Al7FeCu2 at the neighbour's orientation" and the apply writes something else,
or writes it somewhere else. So these tests pin the correspondence, not the
mechanism.

Context: on the real 7050 map the per-grain check found 3 grains / 23 px while
the island stage found 19 islands / 22 px -- 17 of them Al pixels sitting
inside an Al7FeCu2 particle, the exact defect reported by the user
(2026-09-07). Everything below is about not corrupting that repair on the way
from the report to the map.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.spherical_gpu.pipeline.island_check import (  # noqa: E402
    apply_island_findings,
)

ROWS, COLS = 4, 5
N = ROWS * COLS


def _base():
    q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (N, 1))
    pf = np.full(N, 2, dtype=np.int64)
    return q, pf


def test_a_phase_finding_writes_the_quats_the_check_scored():
    q, pf = _base()
    px = [6, 7]
    won = [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    report = {"islands": [{"pixels": px, "decision": "phase", "winner_phase": 2,
                           "new_quats": won}]}
    new_pf, new_q, applied = apply_island_findings(q, pf, ROWS, COLS, report)

    assert np.allclose(new_q[px], won), "wrote a different orientation than reported"
    assert (new_pf[px] == 2).all()
    assert applied == [{"pixels": px, "winner_phase": 2, "decision": "phase"}]


def test_it_touches_nothing_else():
    """A one-pixel repair must be a one-pixel change."""
    q, pf = _base()
    pf[:] = 1
    report = {"islands": [{"pixels": [6], "decision": "phase", "winner_phase": 2,
                           "new_quats": [[0.0, 1.0, 0.0, 0.0]]}]}
    new_pf, new_q, _ = apply_island_findings(q, pf, ROWS, COLS, report)

    others = [i for i in range(N) if i != 6]
    assert (new_pf[others] == 1).all()
    assert np.array_equal(new_q[others], q[others])


def test_the_inputs_are_not_mutated():
    """The caller keeps the pre-change arrays -- they are the undo record."""
    q, pf = _base()
    q_before, pf_before = q.copy(), pf.copy()
    report = {"islands": [{"pixels": [3], "decision": "phase", "winner_phase": 9,
                           "new_quats": [[0.0, 1.0, 0.0, 0.0]]}]}
    apply_island_findings(q, pf, ROWS, COLS, report)
    assert np.array_equal(q, q_before) and np.array_equal(pf, pf_before)


def test_only_accepted_decisions_are_applied():
    """`keep`, `no-scorer` and an already-`applied` entry are not repairs.

    Re-applying an `applied` entry is the second-click case: the reassign marks
    what it wrote, so a second press must be a no-op rather than a rewrite.
    """
    q, pf = _base()
    pf[:] = 1
    report = {"islands": [
        {"pixels": [1], "decision": "keep", "winner_phase": 2},
        {"pixels": [2], "decision": "no-scorer"},
        {"pixels": [3], "decision": "applied", "winner_phase": 2,
         "new_quats": [[0.0, 1.0, 0.0, 0.0]]},
    ]}
    new_pf, new_q, applied = apply_island_findings(q, pf, ROWS, COLS, report)
    assert applied == []
    assert (new_pf == 1).all()
    assert np.array_equal(new_q, q)


def test_a_variant_finding_writes_that_one_orientation_to_every_island_pixel():
    q, pf = _base()
    var = [0.0, 0.0, 1.0, 0.0]
    report = {"islands": [{"pixels": [6, 7], "decision": "variant",
                           "winner_phase": 2, "variant": var}]}
    _, new_q, applied = apply_island_findings(q, pf, ROWS, COLS, report)
    assert np.allclose(new_q[[6, 7]], np.array([var, var]))
    assert applied[0]["decision"] == "variant"


def test_a_quat_count_that_does_not_match_the_pixels_is_refused():
    """Silently zipping mismatched lists would write a neighbour's orientation
    onto the wrong pixel -- a wrong map that still looks plausible."""
    q, pf = _base()
    report = {"islands": [{"pixels": [6, 7], "decision": "phase", "winner_phase": 2,
                           "new_quats": [[0.0, 1.0, 0.0, 0.0]]}]}
    with pytest.raises(ValueError, match="2 pixels but 1 quats"):
        apply_island_findings(q, pf, ROWS, COLS, report)


def test_an_out_of_range_pixel_is_refused():
    q, pf = _base()
    report = {"islands": [{"pixels": [N + 3], "decision": "phase", "winner_phase": 2,
                           "new_quats": [[0.0, 1.0, 0.0, 0.0]]}]}
    with pytest.raises(ValueError, match="out of range"):
        apply_island_findings(q, pf, ROWS, COLS, report)


def test_no_report_is_a_no_op():
    q, pf = _base()
    new_pf, new_q, applied = apply_island_findings(q, pf, ROWS, COLS, {})
    assert applied == [] and np.array_equal(new_pf, pf) and np.array_equal(new_q, q)
