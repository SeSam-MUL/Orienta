"""Non-standard space-group settings must not blow up the Hough indexer.

A CIF written in an alternate setting (beta-AlFeSi is "A12/a1") gives
diffpy/orix an alternate-setting space-group number (4015). PyEBSDIndex maps a
number to a Laue class with a table that only covers 1-230, so 4015 resolved to
CUBIC m-3m: every reflector was expanded into 24 poles that are all distinct on
a monoclinic lattice, and building the band-triplet library asked for

    numpy._core._exceptions._ArrayMemoryError: Unable to allocate 1.40 TiB for
    an array with shape (64177025400, 3) and data type float64

These tests pin the normalisation that keeps the correct Laue class.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffpy.structure import Atom, Lattice, Structure  # noqa: E402
from orix.crystal_map import Phase, PhaseList  # noqa: E402

from ebsd_utils import _standard_setting_phase_list  # noqa: E402


def _monoclinic_structure():
    # beta-AlFeSi cell, A-centred setting
    return Structure(
        atoms=[Atom("Al", [0.0, 0.0, 0.0]), Atom("Fe", [0.5, 0.5, 0.0])],
        lattice=Lattice(6.161, 6.175, 20.813, 90.0, 90.4, 90.0),
    )


def test_alternate_setting_maps_to_standard_number():
    """4015 ("A12/a1") -> 15 ("C2/c"), same Laue class."""
    pl = PhaseList([Phase(name="beta-AlFeSi", space_group=4015,
                          structure=_monoclinic_structure())])
    assert pl[0].space_group.number == 4015

    fixed = _standard_setting_phase_list(pl)

    assert fixed[0].space_group.number == 15
    # symmetry must be preserved, not "fixed"
    assert fixed[0].point_group.name == pl[0].point_group.name == "2/m"
    assert fixed[0].name == "beta-AlFeSi"
    np.testing.assert_allclose(
        fixed[0].structure.lattice.abcABG(),
        pl[0].structure.lattice.abcABG(),
    )
    # the caller's phase list is left alone — reflectors are computed from it
    assert pl[0].space_group.number == 4015


def test_pyebsdindex_reads_the_correct_laue_class():
    """The whole point: 4015 reads as cubic, 15 as monoclinic."""
    from pyebsdindex import crystal_sym

    assert crystal_sym.spacegroup2lauenumber(4015) == 43   # cubic m-3m — wrong
    assert crystal_sym.spacegroup2lauenumber(15) == 2      # monoclinic 2/m — right


def test_standard_space_groups_are_untouched():
    """No copy, no change for the overwhelmingly common case."""
    pl = PhaseList([Phase(name="Al", space_group=225,
                          structure=Structure(atoms=[Atom("Al", [0, 0, 0])],
                                              lattice=Lattice(4.05, 4.05, 4.05, 90, 90, 90)))])
    assert _standard_setting_phase_list(pl) is pl


def test_multi_phase_keeps_ids_and_order():
    """Reflectors are paired with phases by position — order must survive."""
    cubic = Structure(atoms=[Atom("Al", [0, 0, 0])],
                      lattice=Lattice(4.05, 4.05, 4.05, 90, 90, 90))
    pl = PhaseList([
        Phase(name="Al", space_group=225, structure=cubic),
        Phase(name="beta-AlFeSi", space_group=4015, structure=_monoclinic_structure()),
        Phase(name="Si", space_group=227, structure=cubic),
    ])

    fixed = _standard_setting_phase_list(pl)

    assert list(fixed.ids) == list(pl.ids)
    assert [p.name for _, p in fixed] == ["Al", "beta-AlFeSi", "Si"]
    assert [p.space_group.number for _, p in fixed] == [225, 15, 227]


def test_unmappable_space_group_number_raises_a_readable_error():
    """Fail loud rather than silently handing PyEBSDIndex a cubic Laue class."""
    pl = PhaseList([Phase(name="weird", space_group=225,
                          structure=_monoclinic_structure())])

    class _FakeSpaceGroup:
        number = 999999
        short_name = "??"

    pl[0]._space_group = _FakeSpaceGroup()

    with pytest.raises(ValueError, match="neither a standard space group"):
        _standard_setting_phase_list(pl)


@pytest.mark.parametrize("cif_name", ["beta-AlFeSi.cif"])
def test_real_cif_builds_an_indexer_instead_of_asking_for_1_4_tib(cif_name):
    """End-to-end regression on the file the bug was reported against."""
    cif = Path(__file__).resolve().parents[1] / "Database" / "CIF_Library" / cif_name
    if not cif.exists():
        pytest.skip(f"{cif_name} not available (Database/ is not in version control)")

    import kikuchipy as kp

    from ebsd_utils import create_indexer, prepare_reflectors, sanitize_cif

    phase = Phase.from_cif(sanitize_cif(str(cif)))
    assert phase.space_group.number == 4015, "test fixture assumes the A12/a1 setting"

    pl = PhaseList([phase])
    detector = kp.detectors.EBSDDetector(shape=(60, 60), pc=(0.5, 0.5, 0.5),
                                         sample_tilt=70.0)

    indexer = create_indexer(detector, pl, prepare_reflectors(pl), nBands=12)

    assert indexer.phaselist[0].lauecode == 2  # monoclinic, not cubic (43)
    assert phase.space_group.number == 4015, "caller's phase must not be mutated"
