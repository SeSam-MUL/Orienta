"""The multi-block guard, at every place that reads a CIF.

Four functions took ``structures[0]`` from pymatgen, so where a CIF's data
blocks disagree the composition of a phase was decided by block ORDER:

  * ``cif_phase_library._entry_from_cif``      — the phase library / EDS map
  * ``crystal_hint_local_library._parse_cif``  — the Crystal-Hint suggestions
  * ``crystal_structure._load_cif``            — the viewer and forward model
  * ``cif_database_builder.parse_cif_file``    — writes the spreadsheet the
                                                 other three read

``Database/CIF_Library/sd_1816951.cif`` is the file that found it: MgCu2, a
cubic Laves phase whose own atom-site table says Cu 16c + Mg 8b (Cu 66.7 /
Mg 33.3) and whose ``.xtal`` — EMsoft's own input — says the same. pymatgen
returns ``Mg4Cu`` (40 sites) and ``Mg2Cu`` (24), and neither is the compound.

There is now ONE guard, ``cif_phase_library.one_structure``. These tests pin it
at each of the four call sites with the real file, because four copies of a
rule are four chances for three of them to drift.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.services.cif_phase_library import (  # noqa: E402
    AmbiguousCifError, one_structure,
)

MGCU2_CIF = ROOT / "Database" / "CIF_Library" / "sd_1816951.cif"
needs_cif = pytest.mark.skipif(
    not MGCU2_CIF.is_file(),
    reason="Database/CIF_Library/sd_1816951.cif is not on this machine")


class _Composition:
    def __init__(self, formula):
        self.reduced_formula = formula


class _Structure:
    def __init__(self, formula):
        self.composition = _Composition(formula)


# ---------------------------------------------------------------------------
# the guard itself
# ---------------------------------------------------------------------------

def test_agreeing_blocks_give_the_structure():
    a, b = _Structure("Al"), _Structure("Al")
    assert one_structure([a, b], "x.cif") is a


def test_disagreeing_blocks_raise_with_both_formulas():
    with pytest.raises(AmbiguousCifError) as excinfo:
        one_structure([_Structure("Mg4Cu"), _Structure("Mg2Cu")],
                      "sd_1816951.cif")
    exc = excinfo.value
    assert exc.formulas == ["Mg2Cu", "Mg4Cu"]
    assert "sd_1816951.cif" in str(exc)
    for token in ("Mg2Cu", "Mg4Cu"):
        assert token in str(exc), "a caller that cannot show both cannot help"


def test_no_structure_is_a_plain_value_error():
    """Callers distinguish "unreadable" from "ambiguous" by the type."""
    with pytest.raises(ValueError) as excinfo:
        one_structure([], "empty.cif")
    assert not isinstance(excinfo.value, AmbiguousCifError)


# ---------------------------------------------------------------------------
# the four call sites, on the real file
# ---------------------------------------------------------------------------

@needs_cif
def test_phase_library_refuses_it():
    from backend.api.services import cif_phase_library as cpl
    cpl._CIF_ENTRY_CACHE.clear()
    entry, reason = cpl._entry_from_cif(MGCU2_CIF)
    assert entry is None
    # The refusal travels on the SAME channel as every other reason a file
    # did not become a phase, so the suggestion panel shows it without
    # knowing this case exists.
    assert "Mg2Cu" in reason and "Mg4Cu" in reason, reason


@needs_cif
def test_crystal_hint_refuses_it(caplog):
    from backend.api.services.crystal_hint_local_library import _parse_cif
    with caplog.at_level(logging.WARNING):
        got = _parse_cif(MGCU2_CIF)
    assert "error" in got, got
    assert "Mg2Cu" in got["error"] and "Mg4Cu" in got["error"]


@needs_cif
def test_the_structure_loader_refuses_it():
    from backend.api.services.crystal_structure import load_structure
    with pytest.raises(ValueError) as excinfo:
        load_structure(MGCU2_CIF)
    assert "sd_1816951.cif" in str(excinfo.value)


@needs_cif
def test_the_database_builder_refuses_it(caplog):
    """The builder is the one that matters most: its answer becomes the row
    every other reader trusts."""
    builder_dir = ROOT / "crystal-structures-for-ebsd-main" / "calculationxtal"
    if not (builder_dir / "cif_database_builder.py").is_file():
        pytest.skip("the database builder is not in this checkout")
    if str(builder_dir) not in sys.path:
        sys.path.insert(0, str(builder_dir))
    pytest.importorskip("pandas")
    from cif_database_builder import parse_cif_file
    with caplog.at_level(logging.WARNING):
        assert parse_cif_file(MGCU2_CIF) is None
    assert "sd_1816951.cif" in caplog.text, (
        "a rebuild that silently drops a phase is how the bad row got here")


@needs_cif
def test_a_readable_cif_still_works_everywhere():
    """The guard must not be a blanket refusal of multi-block CIFs — 18 of the
    36 shipped files declare more than one block, ten come back with more than
    one structure, and nine of those agree."""
    al = ROOT / "Database" / "CIF_Library" / "Al.cif"
    if not al.is_file():
        pytest.skip("Al.cif is not on this machine")
    from backend.api.services import cif_phase_library as cpl
    from backend.api.services.crystal_hint_local_library import _parse_cif
    from backend.api.services.crystal_structure import load_structure
    cpl._CIF_ENTRY_CACHE.clear()
    entry, reason = cpl._entry_from_cif(al)
    assert entry is not None and reason == ""
    assert "error" not in _parse_cif(al)
    assert load_structure(al) is not None
