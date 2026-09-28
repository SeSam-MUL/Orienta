"""A filter that cannot find what it was asked for must not WIDEN.

`_phase_test_library_entries` said `if phase_keys:`, so `None` and `[]` both
meant "every phase" -- a selection that resolved to nothing silently ran the
phase test over the WHOLE LIBRARY. Found by 03 at `routes/indexing.py:1975`, and
it is the SECOND time this project paid for that exact shape: the
phase-collections merge fixed it at the call site while this function kept the
old meaning for every caller after it. `collectionFilter.js:26` is the same
fallback on the frontend ("renamed or deleted: filter nothing").

The four cases are now distinct, and the fourth is the one that is NOT an error:

    None            not filtered -- every testable phase
    []              an empty selection, so an empty result; no run
    [known keys]    exactly those, in order
    [unknown key]   400 `unknownPhaseKey` -- a stale reference, not a filter

    a known key with NO SHT is none of the above: a collection may legitimately
    hold `Al`, which has no SHT, so it is dropped and RETURNED as `unsupported`
    with reason `no_sht`, the same shape `resolve()` uses for `Al -> no_master`.

The danger was never the width. It is that the display kept claiming a
selection while the run used everything.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.problem import CODE_HEADER                    # noqa: E402
from backend.api.routes import indexing                        # noqa: E402


class _Entry:
    """Minimal stand-in for `LocalEntry`: only what the selector reads."""

    def __init__(self, key, sht=True):
        self.key = key
        self.sht_path = Path(f"{key}.sht") if sht else None
        self.formula = key
        self.display_formula = key
        self.space_group = "m-3m"
        self.cif_path = None


#: `Al` deliberately has no SHT -- it is the real library's case and the reason
#: "no SHT" cannot be an error.
LIBRARY = {
    "Al": _Entry("Al", sht=False),
    "alpha": _Entry("alpha"),
    "Al13Fe4": _Entry("Al13Fe4"),
}


@pytest.fixture(autouse=True)
def library(monkeypatch):
    import backend.api.services.crystal_hint_local_library as chl
    monkeypatch.setattr(chl, "get_index", lambda: LIBRARY)


def _select(keys):
    return indexing._phase_test_library_entries(keys)


def test_none_means_not_filtered():
    """What `/single-pixel-phase-test/phases` wants: everything testable."""
    entries, unsupported = _select(None)
    assert sorted(e.key for e in entries) == ["Al13Fe4", "alpha"]
    assert unsupported == []


def test_an_empty_selection_means_an_empty_result_not_everything():
    """The defect, in one line. `[]` used to be indistinguishable from `None`."""
    entries, unsupported = _select([])
    assert entries == [], "an empty selection must not widen to the library"
    assert unsupported == []


def test_a_selection_of_whitespace_is_still_empty():
    """`[""]` and `[" "]` reach this from a hand-written request or a stale
    store; they are an empty selection, not a key named "".
    """
    assert _select(["", "   "]) == ([], [])


def test_named_phases_come_back_in_the_order_given():
    entries, unsupported = _select(["Al13Fe4", "alpha"])
    assert [e.key for e in entries] == ["Al13Fe4", "alpha"]
    assert unsupported == []


def test_a_key_the_library_does_not_know_is_an_error_with_a_code():
    """A stale reference must be visible. Running on the rest would hide it, and
    running on everything -- the old behaviour -- would hide it twice."""
    with pytest.raises(HTTPException) as e:
        _select(["alpha", "GhostPhase"])
    assert e.value.status_code == 400
    assert e.value.headers.get(CODE_HEADER) == "unknownPhaseKey"
    assert "GhostPhase" in e.value.detail


def test_a_group_with_al_runs_the_two_that_can_and_names_al():
    """b9's first case, and the reason `no_sht` is not an error: a collection may
    legitimately hold `Al`."""
    entries, unsupported = _select(["Al", "alpha", "Al13Fe4"])
    assert [e.key for e in entries] == ["alpha", "Al13Fe4"]
    assert unsupported == [{"key": "Al", "reason": "no_sht"}]


def test_a_group_where_nothing_has_an_sht_is_empty_and_not_an_error():
    """b9's second case: empty result, no run, NO 400. `Al` exists -- it just
    cannot be tested this way, which is a fact to report, not a failure."""
    entries, unsupported = _select(["Al"])
    assert entries == []
    assert unsupported == [{"key": "Al", "reason": "no_sht"}]


def test_the_response_model_carries_unsupported_separately_from_excluded():
    """Two reasons in one list is how a caller reports the wrong one:
    `excluded` means "excluded by the EDS pre-filter", `unsupported` means "this
    method cannot test it at all"."""
    fields = indexing.SinglePixelPhaseTestResponse.model_fields
    assert "unsupported" in fields and "excluded" in fields
    empty = indexing.SinglePixelPhaseTestResponse(pixel_index=0, row=0, col=0)
    assert empty.unsupported == [] and empty.excluded == []


def test_every_caller_unpacks_the_pair():
    """A structural guard. The function used to return a bare list, so a caller
    left un-updated would silently bind the tuple to one name and iterate
    `(entries, unsupported)` -- two items, neither of them a phase.
    """
    import ast

    tree = ast.parse(Path(indexing.__file__).read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        call = node.value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "_phase_test_library_entries"):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Tuple) or len(target.elts) != 2:
            bad.append(getattr(node, "lineno", "?"))
    assert bad == [], (
        f"these call sites do not unpack (entries, unsupported): lines {bad}")
