"""Phases the available chemistry cannot separate must be grouped, not
decided by library row order.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.services.cif_phase_library import (
    CifPhaseEntry, group_degenerate_entries,
)


def _entry(name, comp):
    return CifPhaseEntry(
        key=name, cif_filename=name, formula=name, space_group="",
        space_group_number=None, crystal_system="", composition=comp,
        elements=sorted(comp),
    )


def test_groups_entries_that_chemistry_cannot_separate():
    """Al6Fe vs beta-AlFeSi differ by 1.1 at% in the real library."""
    cands = [
        _entry("Al6Fe.cif", {"Al": 85.7, "Fe": 14.3}),
        _entry("beta-AlFeSi.cif", {"Al": 84.6, "Fe": 15.4}),
        _entry("Al.cif", {"Al": 100.0}),
    ]
    groups = group_degenerate_entries(cands, max_at_pct_sep=3.0)
    assert sorted(len(g) for g in groups) == [1, 2]
    pair = next(g for g in groups if len(g) == 2)
    assert set(pair) == {0, 1}


def test_distinct_entries_stay_separate():
    cands = [
        _entry("Al.cif", {"Al": 100.0}),
        _entry("Si.cif", {"Si": 100.0}),
    ]
    groups = group_degenerate_entries(cands, max_at_pct_sep=3.0)
    assert sorted(len(g) for g in groups) == [1, 1]


def test_grouping_is_transitive():
    """Single linkage: A~B and B~C puts all three together."""
    cands = [
        _entry("A.cif", {"Al": 80.0, "Fe": 20.0}),
        _entry("B.cif", {"Al": 82.0, "Fe": 18.0}),
        _entry("C.cif", {"Al": 84.0, "Fe": 16.0}),
    ]
    groups = group_degenerate_entries(cands, max_at_pct_sep=2.5)
    assert len(groups) == 1 and len(groups[0]) == 3


def test_grouping_is_deterministic_and_order_preserving():
    cands = [
        _entry("A.cif", {"Al": 85.7, "Fe": 14.3}),
        _entry("B.cif", {"Al": 84.6, "Fe": 15.4}),
        _entry("C.cif", {"Al": 100.0}),
    ]
    a = group_degenerate_entries(cands, max_at_pct_sep=3.0)
    assert a == group_degenerate_entries(cands, max_at_pct_sep=3.0)
    assert a[0][0] == 0, "groups must be emitted in first-member order"


def test_empty_input():
    assert group_degenerate_entries([]) == []


def test_entry_without_composition_does_not_crash():
    cands = [_entry("empty.cif", {}), _entry("Al.cif", {"Al": 100.0})]
    groups = group_degenerate_entries(cands)
    assert sum(len(g) for g in groups) == 2
