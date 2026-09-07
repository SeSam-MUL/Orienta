"""Every compare-phases row must carry the orientation it is about.

`quat_wxyz` was written only onto rows for the pixel's OWN phase, because its
first consumer was the same-phase pseudo-symmetry adopt. Its second consumer —
"Assign <other phase> to this grain" — needs exactly the other case, found the
field missing, sent no seed, and fell back to Hough. The user was looking at
Al7FeCu2 scored R = 0.4231 against Al at 0.1552 and was told the assignment was
impossible, with the answer sitting unused in the same response (2026-09-07).

`disorientation_deg` stays same-phase-only: it is measured against the stored
orientation, which means nothing across a phase change.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.routes.indexing import (  # noqa: E402
    _annotate_compare_rows_with_delta,
)

STORED_PID = 1
OTHER_PID = 2
STORED_Q = np.array([1.0, 0.0, 0.0, 0.0])


def _rows():
    return [
        {"phase_id": STORED_PID, "euler_deg": [10.0, 20.0, 30.0]},
        {"phase_id": OTHER_PID, "euler_deg": [105.8, 128.3, 11.7]},
    ]


def test_every_row_gets_its_quaternion():
    rows = _rows()
    _annotate_compare_rows_with_delta(rows, STORED_Q, STORED_PID, "m-3m")
    for r in rows:
        q = r.get("quat_wxyz")
        assert q is not None, f"phase {r['phase_id']} row carries no orientation"
        assert len(q) == 4
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-9)


def test_the_quaternion_matches_the_euler_the_panel_prints():
    """It has to BE the displayed orientation, not merely some orientation."""
    from orix.quaternion import Rotation

    rows = _rows()
    _annotate_compare_rows_with_delta(rows, STORED_Q, STORED_PID, "m-3m")
    other = rows[1]
    expect = np.asarray(Rotation.from_euler(
        np.deg2rad(np.asarray(other["euler_deg"]))[None, :]).data).reshape(4)
    assert abs(float(np.dot(other["quat_wxyz"], expect))) > 1.0 - 1e-12


def test_disorientation_stays_on_the_pixels_own_phase():
    """Across a phase change it would be a number with no meaning."""
    rows = _rows()
    _annotate_compare_rows_with_delta(rows, STORED_Q, STORED_PID, "m-3m")
    assert "disorientation_deg" in rows[0]
    assert "disorientation_deg" not in rows[1]


def test_quats_are_written_even_without_a_stored_orientation():
    """No stored orientation only disables the comparison, not the field the
    assignment needs."""
    rows = _rows()
    _annotate_compare_rows_with_delta(rows, None, None, None)
    assert all(r.get("quat_wxyz") is not None for r in rows)
    assert all("disorientation_deg" not in r for r in rows)


def test_a_row_without_euler_is_left_alone_rather_than_guessed():
    rows = [{"phase_id": OTHER_PID, "euler_deg": []},
            {"phase_id": OTHER_PID}]
    _annotate_compare_rows_with_delta(rows, STORED_Q, STORED_PID, "m-3m")
    assert all("quat_wxyz" not in r for r in rows)


def test_it_never_raises_on_junk():
    """Fail-soft: annotation is a convenience, never a reason to lose the whole
    compare-phases response."""
    rows = [{"phase_id": "not-an-int", "euler_deg": [1.0, 2.0, 3.0]}]
    _annotate_compare_rows_with_delta(rows, STORED_Q, STORED_PID, "m-3m")
    assert rows[0].get("quat_wxyz") is not None
