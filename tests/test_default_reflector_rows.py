"""The default Hough reflector list: no repeated rows, and never a function of free memory.

diffsims' ``unique(use_symmetry=True).symmetrise()`` can return a row twice, from
float noise in the lattice constants of hexagonal and trigonal cells. The default
list is the strongest 70 rows, so repeats used up the budget on rows PyEBSDIndex
ignores (alpha-Ti, a = 2.95, c = 4.68: 172 rows, 106 distinct, and {0004},
{20-21}, {10-12} - each above a third of the strongest reflection - missing).

* A phase without repeats gets exactly the list the code gave before this change,
  row for row (the expectation is the old two lines, written out here).
* A phase with repeats gets the strongest 70 distinct rows; the expectation for
  hcp is the closed form of its structure factor.
* The list is a function of the phase and the rule only. Free memory decides
  whether a run happens (the refusal, ``tests/test_hough_library_budget.py``) and
  offers a shorter list as one click (``tests/test_reflector_fit.py``); it never
  changes the list silently.
"""
import contextlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("orix")
pytest.importorskip("diffsims")

import ebsd_utils as eu  # noqa: E402
import hough_reflectors as hr  # noqa: E402
from tests.data_deps import CIF_LIBRARY, need  # noqa: E402
from tests import test_phase_cell as tpc  # noqa: E402
from tests.test_phase_cell import _asymmetric_phase, _from_cif  # noqa: E402
from tests.test_structure_factor_call_sites import _all_rows, _closed_form  # noqa: E402


@pytest.fixture(autouse=True)
def clean_caches():
    def clear():
        hr.clear_specs()
        hr._CACHE.clear()
        eu._FILTERED_CACHE.clear()
        del eu._FILTERED_CACHE_ORDER[:]
    clear()
    yield
    clear()


def _old_list(phase, max_rows=70):
    """The list as ``_reflectors_for_phase`` built it before the change."""
    filt = eu._filtered_reflectors(phase, 1.0, 0.1)
    n = filt.hkl.shape[0]
    if n > max_rows:
        sf = np.abs(filt.structure_factor)
        return filt[np.argsort(-sf)[:max_rows].tolist()]
    return filt[list(range(n))]


def _rows(rlv):
    return [tuple(int(x) for x in r) for r in np.rint(np.asarray(rlv.hkl)).astype(int)]


@contextlib.contextmanager
def _ti(tmp_path, a=2.95, c=4.68):
    """alpha-Ti written with the lattice constants given (the closed form of the
    test helpers follows, they read the same table)."""
    saved = tpc.CRYSTALS["alpha-Ti"]
    tpc.CRYSTALS["alpha-Ti"] = (saved[0], (a, a, c, 90, 90, 120), saved[2], saved[3])
    try:
        yield _from_cif(tmp_path, "alpha-Ti")[0]
    finally:
        tpc.CRYSTALS["alpha-Ti"] = saved


# --------------------------------------------------------------------------
# phases without repeats: unchanged, row for row
# --------------------------------------------------------------------------

def _library_cifs():
    return sorted(p for p in CIF_LIBRARY.glob("*.cif") if "backup" not in p.name) \
        if CIF_LIBRARY.is_dir() else []


@pytest.mark.parametrize("path", _library_cifs(),
                         ids=[p.name.encode("ascii", "replace").decode() for p in _library_cifs()])
def test_a_phase_without_repeated_rows_keeps_its_list_row_for_row(path, monkeypatch):
    monkeypatch.setenv("ORIENTA_HOUGH_LIBRARY_BUDGET_MB", str(1 << 30))   # no step-down
    phase = eu.hough_phase_from_cif(str(path))
    filt = eu._filtered_reflectors(phase, 1.0, 0.1)
    keys = np.rint(np.asarray(filt.hkl)).astype(int)
    if len(np.unique(keys, axis=0)) != len(keys):
        pytest.skip("this phase has repeated rows: its list changes by design")
    new = eu._reflectors_for_phase(phase)
    old = _old_list(phase)
    assert np.array_equal(np.asarray(new.hkl), np.asarray(old.hkl))
    assert np.array_equal(np.asarray(new.structure_factor), np.asarray(old.structure_factor))


def test_the_library_holds_phases_with_and_without_repeats():
    """The two groups above are both non-empty (else the test proves nothing)."""
    need(CIF_LIBRARY / "Al.cif")
    with_rep, without = [], []
    for path in _library_cifs():
        phase = eu.hough_phase_from_cif(str(path))
        keys = np.rint(np.asarray(eu._filtered_reflectors(phase, 1.0, 0.1).hkl)).astype(int)
        (without if len(np.unique(keys, axis=0)) == len(keys) else with_rep).append(path.name)
    assert len(without) >= 20 and len(with_rep) >= 5
    assert any(n.startswith("Al.") for n in without) and any(n.startswith("Ni.") for n in without)


# --------------------------------------------------------------------------
# hcp: the strongest 70 distinct rows
# --------------------------------------------------------------------------

def _check_top_rows(name, rlv, max_rows=70):
    rows = _rows(rlv)
    assert len(rows) == len(set(rows))
    all_rows = _all_rows(name, 1.0)
    F = _closed_form(name, all_rows)
    keep = F > 0.1 * F.max()
    pool = {tuple(int(x) for x in r): f for r, f in zip(all_rows[keep], F[keep])}
    assert set(rows) <= set(pool)
    assert len(rows) == min(max_rows, len(pool))
    cut = min(pool[r] for r in rows)
    assert all(f <= cut * (1 + 1e-9) for r, f in pool.items() if r not in set(rows))


@pytest.mark.parametrize("lattice", [(2.95, 4.68), (2.951, 4.684)])
def test_hcp_list_is_the_strongest_distinct_rows(tmp_path, lattice):
    """alpha-Ti with the lattice constants that give repeats in diffsims (2.95,
    4.68: 172 rows, 106 distinct) and with those that do not."""
    with _ti(tmp_path, *lattice) as phase:
        filt = eu._filtered_reflectors(phase, 1.0, 0.1)
        n_distinct = len(np.unique(np.rint(np.asarray(filt.hkl)).astype(int), axis=0))
        if lattice == (2.95, 4.68):
            assert len(filt.hkl) > n_distinct          # the premise: repeats exist
        _check_top_rows("alpha-Ti", eu._reflectors_for_phase(phase))


def test_the_families_the_repeats_pushed_out_are_in(tmp_path):
    """{0004} (0.46 of the strongest) and {20-21} (0.43) were not in alpha-Ti's list."""
    with _ti(tmp_path) as phase:
        rows = set(_rows(eu._reflectors_for_phase(phase)))
        assert (0, 0, 4) in rows and (2, 0, 1) in rows
        old = set(_rows(_old_list(phase)))
        assert (0, 0, 4) not in old and (2, 0, 1) not in old      # what it was before


def test_describe_lists_the_missing_families(tmp_path):
    with _ti(tmp_path) as phase:
        selected = {f["label"] for f in hr.describe(phase, limit=1000)["families"]
                    if f["selected"]}
    assert {"{0004}", "{20-21}"} <= selected


# --------------------------------------------------------------------------
# the list is a function of the phase and the rule, never of free memory
# --------------------------------------------------------------------------

BUDGETS = (1 << 40, 20 << 20, 1)


@pytest.mark.parametrize("which", ["alpha-Ti-repeats", "hcp-Mg", "monoclinic-C2/m",
                                   "orthorhombic-Cmcm", "cubic-Fm-3m"])
def test_the_default_list_does_not_depend_on_free_memory(tmp_path, monkeypatch, which):
    """The same phase gives the same list, row for row, however much memory is free;
    the table, ``describe`` and the rows of a custom selection agree with it, and no
    library is predicted to make the list. Memory decides whether a run happens, not
    which reflectors it uses."""
    probed = []
    real = eu._library_rows_for
    monkeypatch.setattr(eu, "_library_rows_for",
                        lambda *a, **k: probed.append(1) or real(*a, **k))
    ctx = _ti(tmp_path) if which == "alpha-Ti-repeats" else contextlib.nullcontext(
        _from_cif(tmp_path, which)[0])
    with ctx as phase:
        spec = {"mode": "custom", "families": [[1, 0, 0], [1, 0, 2], [0, 0, 4]]}             if "hcp" in which or "Ti" in which else None
        seen = []
        for budget in BUDGETS * 2:                           # and back again
            monkeypatch.setattr(eu, "_triplet_library_budget_bytes", lambda b=budget: b)
            built = eu._reflectors_for_phase(phase)
            table_hkl, _ = hr._rule_rows(phase, hr.DEFAULT_RULE)
            assert np.array_equal(table_hkl, np.asarray(built.hkl))
            shown = hr.describe(phase)
            entry = (np.asarray(built.hkl).tolist(),
                     sorted(f["label"] for f in shown["families"] if f["selected"]),
                     hr.spec_rows(phase, spec)[0].tolist() if spec else None)
            seen.append(entry)
        assert all(e == seen[0] for e in seen)
        assert "step_down" not in shown
    assert probed == []


def _rows_cost(phase, rlv):
    # a tiny probe cap, so that the cost of a small library is measured, not "negligible"
    return eu._library_rows_for(phase, np.asarray(rlv.hkl).tolist(),
                                probe_rows=1000) * eu._TRIPLET_ROW_BYTES


def test_the_service_cost_uses_the_same_prediction(tmp_path):
    from backend.api.services import hough_reflector_service as svc
    phase = _asymmetric_phase("alpha-Ti")
    rows = np.asarray(eu._reflectors_for_phase(phase).hkl).tolist()
    assert svc._library_rows(phase, rows) == eu._library_rows_for(phase, rows)


def test_without_a_cut_the_distinct_rows_keep_the_order_of_their_first_occurrence(tmp_path):
    with _ti(tmp_path) as phase:
        filt = eu._filtered_reflectors(phase, 1.0, 0.1)
        seen, want = set(), []
        for r in _rows(filt):
            if r not in seen:
                seen.add(r)
                want.append(r)
        assert len(want) < len(filt.hkl)
        assert _rows(eu._reflectors_for_phase(phase, max_reflectors=1000)) == want


