"""
Tests for the per-phase indexing orchestrator.

Covers the branching logic of :func:`indexing_controller.run_per_phase_indexing`
without paying the cost of running real Hough/Dictionary/Spherical
passes. The expensive bit (each phase's actual indexing) is mocked
out — the orchestrator's job is to schedule the per-phase calls,
sanitise the consensus map, skip empty masks, tolerate per-phase
failures, and hand off to ``build_consensus_xmap`` with consistent
indices. That's what these tests verify.

The merge step (``build_consensus_xmap``) is already covered by the
multi-phase comparison tests, so we stop short of asserting on
xmap contents.
"""
from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from indexing_controller import (
    ComparisonConfig,
    IndexingMethod,
    IndexingResult,
    PhaseConfig,
    PhaseMethodResult,
    PixelSelectionMode,
    run_per_phase_indexing,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _stub_signal(n_rows=4, n_cols=4):
    """Tiny stand-in for a kikuchipy EBSD signal — only ``data.shape`` is read."""
    sig = MagicMock()
    sig.data.shape = (n_rows, n_cols, 60, 60)
    return sig


def _stub_phase_config(name: str) -> PhaseConfig:
    return PhaseConfig(
        name=name,
        cif_path=f"{name}.cif",
        phase_list=MagicMock(),  # build_consensus_xmap reads xmap.phases, not this
    )


def _make_stub_pmr(name: str, n_rows: int, n_cols: int, mask: np.ndarray) -> PhaseMethodResult:
    """Build a PhaseMethodResult that walks like a duck for build_consensus_xmap."""
    n_pix = int(mask.sum())
    fake_xmap = MagicMock()
    fake_xmap.rotations.data = np.tile([1.0, 0.0, 0.0, 0.0], (n_pix, 1))
    # build_consensus_xmap iterates xmap.phases.ids — give it one
    # synthetic phase entry so the logic doesn't bail early.
    phase = MagicMock()
    phase.color = "#888888"
    phase.point_group = MagicMock(name="point_group")
    phase.space_group = None
    phase.structure = None
    fake_xmap.phases.ids = [0]
    fake_xmap.phases.__getitem__ = lambda self, k: phase
    indexing_result = IndexingResult(
        xmap=fake_xmap,
        selection_mask=mask,
        original_shape=(n_rows, n_cols),
        method=IndexingMethod.HOUGH,
    )
    return PhaseMethodResult(
        phase_name=name,
        method=IndexingMethod.HOUGH,
        indexing_result=indexing_result,
        mean_score=0.9,
    )


# ---------------------------------------------------------------------------
# Argument-validation tests — pure, no patching needed
# ---------------------------------------------------------------------------


def test_run_per_phase_requires_aligned_configs_and_masks():
    """Mismatched lengths must raise — silent zip-truncation would
    silently drop a phase, and the merge would then point at the
    wrong slot."""
    sig = _stub_signal()
    with pytest.raises(ValueError, match="must align"):
        run_per_phase_indexing(
            signal=sig, detector=None,
            method=IndexingMethod.HOUGH,
            config=ComparisonConfig(phases=[], methods=[IndexingMethod.HOUGH]),
            phase_configs=[_stub_phase_config("A")],
            masks_per_phase=[],
            consensus_map=np.zeros((4, 4), dtype=np.int32),
        )


def test_run_per_phase_rejects_empty_phase_list():
    sig = _stub_signal()
    with pytest.raises(ValueError, match="at least one phase"):
        run_per_phase_indexing(
            signal=sig, detector=None,
            method=IndexingMethod.HOUGH,
            config=ComparisonConfig(phases=[], methods=[IndexingMethod.HOUGH]),
            phase_configs=[],
            masks_per_phase=[],
            consensus_map=np.zeros((4, 4), dtype=np.int32),
        )


def test_run_per_phase_rejects_consensus_map_shape_mismatch():
    sig = _stub_signal(n_rows=4, n_cols=4)
    with pytest.raises(ValueError, match="consensus_map shape"):
        run_per_phase_indexing(
            signal=sig, detector=None,
            method=IndexingMethod.HOUGH,
            config=ComparisonConfig(phases=[], methods=[IndexingMethod.HOUGH]),
            phase_configs=[_stub_phase_config("A")],
            masks_per_phase=[np.ones((4, 4), dtype=bool)],
            consensus_map=np.zeros((3, 3), dtype=np.int32),
        )


def test_run_per_phase_rejects_per_phase_mask_shape_mismatch():
    sig = _stub_signal(n_rows=4, n_cols=4)
    with pytest.raises(ValueError, match="per-phase mask shape"):
        run_per_phase_indexing(
            signal=sig, detector=None,
            method=IndexingMethod.HOUGH,
            config=ComparisonConfig(phases=[], methods=[IndexingMethod.HOUGH]),
            phase_configs=[_stub_phase_config("A")],
            masks_per_phase=[np.ones((3, 3), dtype=bool)],
            consensus_map=np.zeros((4, 4), dtype=np.int32),
        )


def test_run_per_phase_rejects_all_empty_masks():
    sig = _stub_signal()
    with pytest.raises(ValueError, match="All per-phase masks are empty"):
        run_per_phase_indexing(
            signal=sig, detector=None,
            method=IndexingMethod.HOUGH,
            config=ComparisonConfig(phases=[], methods=[IndexingMethod.HOUGH]),
            phase_configs=[_stub_phase_config("A"), _stub_phase_config("B")],
            masks_per_phase=[
                np.zeros((4, 4), dtype=bool),
                np.zeros((4, 4), dtype=bool),
            ],
            consensus_map=np.full((4, 4), -1, dtype=np.int32),
        )


# ---------------------------------------------------------------------------
# Orchestrator behaviour with mocked indexing
# ---------------------------------------------------------------------------


def test_run_per_phase_skips_empty_mask_phases():
    """A phase with an all-False mask must be silently skipped,
    not invoked. ``run_single_phase_method`` is expensive — calling
    it with an empty mask wastes time and would also potentially
    fail downstream."""
    sig = _stub_signal()
    pc_a = _stub_phase_config("A")
    pc_b = _stub_phase_config("B")

    mask_a = np.zeros((4, 4), dtype=bool)            # empty -> skipped
    mask_b = np.zeros((4, 4), dtype=bool)
    mask_b[:, :2] = True                             # left half is phase B
    consensus = np.where(mask_b, 1, -1).astype(np.int32)

    fake_pmr = _make_stub_pmr("B", 4, 4, mask_b)

    with patch("indexing_controller.run_single_phase_method", return_value=fake_pmr) as mock_run, \
         patch("indexing_controller.build_consensus_xmap", return_value=MagicMock()) as mock_merge:
        result = run_per_phase_indexing(
            signal=sig, detector=None,
            method=IndexingMethod.HOUGH,
            config=ComparisonConfig(phases=[pc_a, pc_b], methods=[IndexingMethod.HOUGH]),
            phase_configs=[pc_a, pc_b],
            masks_per_phase=[mask_a, mask_b],
            consensus_map=consensus,
        )

    # run_single_phase_method must have been called ONCE for phase B,
    # never for the empty phase A.
    assert mock_run.call_count == 1
    called_with = mock_run.call_args.kwargs
    assert called_with["phase_config"] is pc_b

    # build_consensus_xmap got the user's consensus_map verbatim (modulo
    # sanitisation, which is a no-op when all kept phases were called).
    consensus_passed = mock_merge.call_args.args[0].consensus_map
    np.testing.assert_array_equal(consensus_passed, consensus)

    # Result reports the skipped phase in metadata.
    assert result.metadata["n_phases_total"] == 2
    assert result.metadata["n_phases_run"] == 1
    assert result.metadata["n_phases_skipped"] == 1


def test_run_per_phase_tolerates_per_phase_failure():
    """If one phase blows up mid-run, the orchestrator must continue
    with the remaining phases AND erase the failed phase's pixels
    from the consensus map (otherwise the merge would try to look
    up a non-existent xmap and crash)."""
    sig = _stub_signal()
    pc_a = _stub_phase_config("A")
    pc_b = _stub_phase_config("B")

    mask_a = np.zeros((4, 4), dtype=bool); mask_a[:2, :] = True
    mask_b = np.zeros((4, 4), dtype=bool); mask_b[2:, :] = True
    consensus = np.full((4, 4), -1, dtype=np.int32)
    consensus[:2, :] = 0
    consensus[2:, :] = 1

    fake_pmr_b = _make_stub_pmr("B", 4, 4, mask_b)

    def side_effect(*args, **kwargs):
        if kwargs["phase_config"].name == "A":
            raise RuntimeError("orix bombed out on phase A")
        return fake_pmr_b

    with patch("indexing_controller.run_single_phase_method", side_effect=side_effect), \
         patch("indexing_controller.build_consensus_xmap", return_value=MagicMock()) as mock_merge:
        result = run_per_phase_indexing(
            signal=sig, detector=None,
            method=IndexingMethod.HOUGH,
            config=ComparisonConfig(phases=[pc_a, pc_b], methods=[IndexingMethod.HOUGH]),
            phase_configs=[pc_a, pc_b],
            masks_per_phase=[mask_a, mask_b],
            consensus_map=consensus,
        )

    # The consensus passed to build_consensus_xmap must have phase A's
    # pixels marked unclassified (-1) since A failed; phase B is kept.
    consensus_passed = mock_merge.call_args.args[0].consensus_map
    assert (consensus_passed[:2, :] == -1).all(), "phase A pixels must be erased on failure"
    assert (consensus_passed[2:, :] == 1).all(), "phase B pixels must survive"

    assert result.metadata["n_phases_run"] == 1
    assert result.metadata["n_phases_skipped"] == 1
    assert result.metadata["skipped"][0]["phase_index"] == 0


def test_run_per_phase_raises_if_all_phases_fail():
    """Zero successful results -> raise (rather than return an empty
    IndexingResult that callers would have to special-case)."""
    sig = _stub_signal()
    pc_a = _stub_phase_config("A")
    mask_a = np.ones((4, 4), dtype=bool)
    consensus = np.zeros((4, 4), dtype=np.int32)

    with patch("indexing_controller.run_single_phase_method",
               side_effect=RuntimeError("no candidates passed")):
        with pytest.raises(ValueError, match="all phases failed"):
            run_per_phase_indexing(
                signal=sig, detector=None,
                method=IndexingMethod.HOUGH,
                config=ComparisonConfig(phases=[pc_a], methods=[IndexingMethod.HOUGH]),
                phase_configs=[pc_a],
                masks_per_phase=[mask_a],
                consensus_map=consensus,
            )


def test_run_per_phase_progress_callback_fires_per_phase():
    sig = _stub_signal()
    pc_a = _stub_phase_config("A")
    pc_b = _stub_phase_config("B")
    mask_a = np.ones((4, 4), dtype=bool); mask_a[2:, :] = False
    mask_b = ~mask_a
    consensus = np.where(mask_a, 0, 1).astype(np.int32)
    fake = _make_stub_pmr("X", 4, 4, mask_a)
    progress_log = []

    with patch("indexing_controller.run_single_phase_method", return_value=fake), \
         patch("indexing_controller.build_consensus_xmap", return_value=MagicMock()):
        run_per_phase_indexing(
            signal=sig, detector=None,
            method=IndexingMethod.HOUGH,
            config=ComparisonConfig(phases=[pc_a, pc_b], methods=[IndexingMethod.HOUGH]),
            phase_configs=[pc_a, pc_b],
            masks_per_phase=[mask_a, mask_b],
            consensus_map=consensus,
            progress_cb=lambda msg, frac: progress_log.append((msg, frac)),
        )

    # At least one progress entry per phase + the merge step at the end
    assert len(progress_log) >= 3
    fractions = [f for _, f in progress_log]
    assert fractions == sorted(fractions), "progress fractions must be monotonic"
    assert fractions[-1] >= 0.9, "final progress should approach 1.0 at merge time"


def test_run_per_phase_returns_union_mask():
    """The IndexingResult.selection_mask must be the union of all
    per-phase masks, so the result-writer treats every classified
    pixel as 'has data'."""
    sig = _stub_signal()
    pc_a = _stub_phase_config("A")
    pc_b = _stub_phase_config("B")
    mask_a = np.zeros((4, 4), dtype=bool); mask_a[0, :] = True
    mask_b = np.zeros((4, 4), dtype=bool); mask_b[3, :] = True
    consensus = np.full((4, 4), -1, dtype=np.int32)
    consensus[0, :] = 0
    consensus[3, :] = 1
    fake_a = _make_stub_pmr("A", 4, 4, mask_a)
    fake_b = _make_stub_pmr("B", 4, 4, mask_b)

    def side_effect(*args, **kwargs):
        return fake_a if kwargs["phase_config"].name == "A" else fake_b

    with patch("indexing_controller.run_single_phase_method", side_effect=side_effect), \
         patch("indexing_controller.build_consensus_xmap", return_value=MagicMock()):
        result = run_per_phase_indexing(
            signal=sig, detector=None,
            method=IndexingMethod.HOUGH,
            config=ComparisonConfig(phases=[pc_a, pc_b], methods=[IndexingMethod.HOUGH]),
            phase_configs=[pc_a, pc_b],
            masks_per_phase=[mask_a, mask_b],
            consensus_map=consensus,
        )

    expected_union = np.zeros((4, 4), dtype=bool)
    expected_union[0, :] = True
    expected_union[3, :] = True
    np.testing.assert_array_equal(result.selection_mask, expected_union)


# ---------------------------------------------------------------------------
# The sentence the run owes the user when it lost a phase
#
# The orchestrator above is fail-soft on purpose and records
# `n_phases_skipped` / `skipped` in the metadata. Until 2026-09-26 NOTHING read
# those keys — zero readers in `frontend/src`, and the only two in this very
# file, which is what made the gap easy to miss: the metadata was covered, the
# user was not. A routed run that lost a phase therefore looked exactly like a
# run whose pixels merely failed to index.
#
# `_describe_skipped_phases` turns that metadata into the progress-log line.
# These tests are about the MESSAGE; the policy (partial map, pixels -1) is
# unchanged and is covered by `test_run_per_phase_tolerates_per_phase_failure`.
# ---------------------------------------------------------------------------
from backend.api.routes.indexing import _describe_skipped_phases  # noqa: E402


def test_nothing_skipped_says_nothing():
    """The caller's `if` must read as "was there anything to report"."""
    cfgs = [_stub_phase_config("A"), _stub_phase_config("B")]
    assert _describe_skipped_phases([], cfgs) == ""
    assert _describe_skipped_phases(None, cfgs) == ""


def test_the_skipped_phase_is_named_with_its_reason_and_the_consequence():
    cfgs = [_stub_phase_config("Al"), _stub_phase_config("Al7FeCu2"),
            _stub_phase_config("alpha")]
    msg = _describe_skipped_phases(
        [{"phase_index": 1, "reason": "CUDA out of memory"}], cfgs)
    assert "Al7FeCu2" in msg                  # which phase, by NAME not index
    assert "CUDA out of memory" in msg        # and why
    assert "UNCLASSIFIED" in msg              # and what it means for the map
    assert "1 of 3" in msg                    # and how much of the run it was
    # The phases that survived must NOT be named as skipped.
    assert "alpha" not in msg


def test_every_skipped_phase_is_listed():
    cfgs = [_stub_phase_config("A"), _stub_phase_config("B"), _stub_phase_config("C")]
    msg = _describe_skipped_phases(
        [{"phase_index": 0, "reason": "boom"}, {"phase_index": 2, "reason": "bang"}],
        cfgs)
    assert "A (boom)" in msg and "C (bang)" in msg
    assert "2 of 3" in msg


def test_a_missing_reason_still_gets_reported():
    """A report that needs every field present is a report that goes missing."""
    cfgs = [_stub_phase_config("A")]
    msg = _describe_skipped_phases([{"phase_index": 0}], cfgs)
    assert "A" in msg and "unknown" in msg


@pytest.mark.parametrize("bad_index", [7, -1, None, "1"])
def test_an_index_that_does_not_land_still_produces_a_line(bad_index):
    """Bookkeeping that disagrees with itself must not silence the warning.

    This is the whole failure mode being fixed, one level down: suppressing the
    line when the index cannot be resolved would restore the silence.
    """
    cfgs = [_stub_phase_config("A")]
    msg = _describe_skipped_phases([{"phase_index": bad_index, "reason": "x"}], cfgs)
    assert "UNCLASSIFIED" in msg
    assert f"phase {bad_index}" in msg


def test_a_bool_index_does_not_name_a_phase():
    """`isinstance(True, int)` is True in Python, so a bare int check would have
    confidently named phase 1 — a WRONG name, worse than a missing one."""
    cfgs = [_stub_phase_config("Al"), _stub_phase_config("Si")]
    for bad in (True, False):
        msg = _describe_skipped_phases([{"phase_index": bad, "reason": "b"}], cfgs)
        assert "Al" not in msg and "Si" not in msg, f"{bad!r} named a phase: {msg}"
        assert "UNCLASSIFIED" in msg          # ...but it is still reported


def test_a_numpy_integer_still_names_its_phase():
    """Any round-trip of this metadata through h5 or JSON hands back np ints, and
    a bare `int` check would degrade them to the useless "phase 1"."""
    np_idx = np.int64(1)
    cfgs = [_stub_phase_config("Al"), _stub_phase_config("Al7FeCu2")]
    msg = _describe_skipped_phases([{"phase_index": np_idx, "reason": "oom"}], cfgs)
    assert "Al7FeCu2" in msg


def test_a_traceback_sized_reason_is_cut_and_says_where_the_rest_is():
    """One log entry per run, and the EMSphInx path appends a whole traceback to
    its message. Cutting it keeps the log readable; the pointer keeps it honest."""
    cfgs = [_stub_phase_config("Al")]
    msg = _describe_skipped_phases(
        [{"phase_index": 0,
          "reason": "NML generation failed" + "\n" + "E" * 4000}], cfgs)
    assert len(msg) < 500, f"message is {len(msg)} chars — a traceback got through"
    assert "orienta.log" in msg               # where the full text lives
    assert "NML generation failed" in msg     # the useful head survives


def test_the_message_reads_the_metadata_the_orchestrator_really_writes():
    """Producer and consumer held against each other, not against my idea of them.

    Every other test here hands `_describe_skipped_phases` a dict I wrote. That
    cannot catch the two sides disagreeing about the key names or the index base
    — exactly the class of defect that made `sht_paths_by_phase` render every
    phase against its predecessor's master. So this one runs the REAL
    orchestrator with a failing phase and feeds its REAL metadata in.
    """
    sig = _stub_signal()
    pc_a = _stub_phase_config("Al7FeCu2")
    pc_b = _stub_phase_config("Al")
    mask_a = np.zeros((4, 4), dtype=bool); mask_a[:2, :] = True
    mask_b = np.zeros((4, 4), dtype=bool); mask_b[2:, :] = True
    consensus = np.full((4, 4), -1, dtype=np.int32)
    consensus[:2, :] = 0
    consensus[2:, :] = 1
    pmr_b = _make_stub_pmr("Al", 4, 4, mask_b)

    def side_effect(*args, **kwargs):
        if kwargs["phase_config"].name == "Al7FeCu2":
            raise RuntimeError("CUDA out of memory")
        return pmr_b

    with patch("indexing_controller.run_single_phase_method", side_effect=side_effect), \
         patch("indexing_controller.build_consensus_xmap", return_value=MagicMock()):
        result = run_per_phase_indexing(
            signal=sig, detector=None,
            method=IndexingMethod.HOUGH,
            config=ComparisonConfig(phases=[pc_a, pc_b],
                                    methods=[IndexingMethod.HOUGH]),
            phase_configs=[pc_a, pc_b],
            masks_per_phase=[mask_a, mask_b],
            consensus_map=consensus,
        )

    # The route passes exactly these two things.
    msg = _describe_skipped_phases(result.metadata["skipped"], [pc_a, pc_b])
    assert "Al7FeCu2" in msg, (
        "the phase the orchestrator recorded is not the one the message names — "
        f"metadata was {result.metadata['skipped']!r}"
    )
    assert "CUDA out of memory" in msg
    assert "1 of 2" in msg
    assert "UNCLASSIFIED" in msg
    # ...and the count the metadata reports agrees with the line's wording.
    assert result.metadata["n_phases_skipped"] == 1


def _route_source_tree():
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "backend" / "api" / "routes" / "indexing.py").read_text(encoding="utf-8")
    return ast.parse(src)


def _calls(node, name) -> bool:
    """True if `node` contains a call to `name`, bare or as an attribute.

    Both forms are needed: the route calls `_progress(...)` (a Name) and
    `logger.warning(...)` (an Attribute), and a check that saw only one of them
    would quietly pass on the other.
    """
    import ast
    for n in ast.walk(node):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Name) and f.id == name:
            return True
        if isinstance(f, ast.Attribute) and f.attr == name:
            return True
    return False


def _routing_warning_site(tree=None):
    """``(owner, if_node)`` for the place the route logs the skipped phases.

    ``owner`` is the node whose ``body`` directly holds those two statements, so a
    test can ask what encloses them instead of walking up to the first ancestor
    that happens to be a ``try`` — the endpoint has one of those around
    everything, and it is precisely the handler that discards the result.

    Selected by STRUCTURE, not by a source window and not by the property under
    test: find the statement block holding ``_skip_msg = _describe_skipped_phases(...)``
    and take the following ``if`` in that same block that calls ``_progress``.
    The guard condition is then something a test can assert about, rather than
    something the search already assumed.

    The first version of this took a 400-character slice after the call and
    string-matched in it. That was wrong twice: it **passed** when the call was
    made dead code (``if False and _skip_msg:``) — the one property it exists to
    guarantee — and it broke on four of five behaviour-preserving edits, one of
    them raising ``ValueError`` out of ``str.index`` instead of failing an
    assertion. Three lines of explanatory comment were enough to break it, in a
    file whose house style is a long comment above a short change.
    """
    import ast
    tree = tree if tree is not None else _route_source_tree()
    hits = []
    for parent in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(parent, field, None)
            if not isinstance(block, list):
                continue
            for i, stmt in enumerate(block):
                if not (isinstance(stmt, ast.Assign)
                        and getattr(stmt.targets[0], "id", None) == "_skip_msg"
                        and _calls(stmt, "_describe_skipped_phases")):
                    continue
                nxt = next((n for n in block[i + 1:]
                            if isinstance(n, ast.If) and _calls(n, "_progress")), None)
                assert nxt is not None, (
                    "_skip_msg is computed but no following `if` in the same block "
                    "logs it — the message is built and thrown away"
                )
                hits.append((parent, field, nxt))
    assert len(hits) == 1, f"expected exactly one report site, found {len(hits)}"
    owner, field, node = hits[0]
    assert field == "body", f"report site sits in {field}, not a plain body"
    return owner, node


def test_the_route_actually_sends_that_message_to_the_progress_log():
    """A source-level check, and here is why it is not a live one.

    `_progress` is a closure defined inside the endpoint, and reaching the
    routing branch needs a loaded signal, a phase map and a detector — so a
    "does it get called" test would cost more than it proves. But a correct
    function that nothing calls IS the defect being fixed, so the wiring is
    asserted rather than assumed. Same reasoning as
    `PhaseMapCanvas.mount.test.jsx`.
    """
    import ast

    _owner, node = _routing_warning_site()

    # The guard must be the message itself — not a constant that disables the
    # call. `if False and _skip_msg:` is the mutation the string-window version
    # of this test let through.
    assert isinstance(node.test, ast.Name) and node.test.id == "_skip_msg", (
        f"the skip message is logged under {ast.dump(node.test)}, which is not a "
        "plain check of the message — dead or conditional code would go unnoticed"
    )

    # ...and the body hands that message to the user-facing progress log.
    call = next(n for n in ast.walk(node)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "_progress")
    assert any(isinstance(a, ast.Name) and a.id == "_skip_msg" for a in call.args), (
        "_progress is called in that branch but not with the skip message"
    )
    # The percentage belongs to the end of the run: the merge reports 0.95 and
    # completion is 1.0, so anything outside that window reads as going
    # backwards in the progress bar. Nothing else pinned this.
    pct = next((a.value for a in call.args if isinstance(a, ast.Constant)), None)
    assert isinstance(pct, float) and 0.95 <= pct < 1.0, f"progress pct was {pct!r}"


def test_the_report_cannot_lose_a_finished_run():
    """The report must sit in its OWN try/except, not merely inside one.

    A malformed row makes the helper raise — deliberately, that is a contract
    violation and must be loud. But `_store_result` is ~570 lines further on, and
    the endpoint's outer `except Exception` sets status="failed" and never stores
    the result: a cosmetic log line would discard a completed indexing run, hours
    of it on the spherical paths. `_apply_particle_rescue` wrote the rule down
    first ("Never raises: a failure here must not lose an indexing result").

    The first version of this test walked up to the nearest enclosing `try` and
    was therefore satisfied by that outer handler — the very one that loses the
    run. It passed with the local guard deleted. So the guard must be the
    statements' DIRECT owner.
    """
    import ast

    owner, _node = _routing_warning_site()
    assert isinstance(owner, ast.Try), (
        "the skipped-phase report is not inside its own try/except (its direct "
        f"owner is {type(owner).__name__}) — an exception from a log line would "
        "reach the endpoint handler and discard the finished indexing result"
    )
    assert any(isinstance(h.type, ast.Name) and h.type.id == "Exception"
               for h in owner.handlers), "the guard must catch Exception"
    # And it must not swallow silently: the run has to leave a trace.
    assert any(_calls(h, "warning") or _calls(h, "exception") or _calls(h, "error")
               for h in owner.handlers), "a swallowed failure must still be logged"
