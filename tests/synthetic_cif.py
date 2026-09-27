"""A symmetry-less CIF that belongs to the suite, not to the machine.

Four tests in three files needed a phase written as ``P 1``: the Hough failure
message calling that out by name, the band-triplet library it would ask for
being reported as huge, and the two reflector-registry tests that need a phase
whose full library is UNAFFORDABLE so they can assert the refusal. All four
reached into ``Database/CIF_Library/Al7FeCu2.cif`` for it, because that file
happened to be stored without symmetry.

It was repaired on 2026-09-13 (``P4/mnc``, #128, the backup is beside it), and
all four then FAILED — two wanting "point group 1" and more than a GiB, two with
"DID NOT RAISE MemoryError". Their ``skipif`` asks only whether the file EXISTS,
which it still does, so none could skip. They were asserting the defect, and
repairing the data broke them: a test that needs a library file to stay broken is
a test bound to one machine's mistake.

So the P 1 phase is synthesised here instead. Nothing is copied from any
structure database — the cell is made up, which is all these tests need: no
symmetry means no equivalence merging, so the reflector families pile up and the
library explodes exactly as the real case did.

The coordinates are deliberately irregular
------------------------------------------
The first version of this file used the F-centring set (0,0,0 / ½½0 / ½0½ / 0½½)
plus Fe at ¼¼¼ — tidy, and **more symmetric than P 1**: the declared group is 1,
but ``SpacegroupAnalyzer(structure, symprec=1e-5)``, the analyser this tree uses
in ``cif_phase_library``, DERIVES ``P222`` (#16) from those positions. The four
tests read the declared group and were unaffected, but that is one reader away
from the trap this whole file exists to escape — "P 1 only by accident of how it
was read". The positions below have no such relation and read back as ``P1`` (#1)
from the analyser as well as from the declaration.

Measured, ``ebsd_utils.predict_triplet_library``, identical on the 60x60 and the
156x128 detector (the library is sized from the phase's poles and lattice, not
from the detector — that is why the two shapes agree bit for bit):

    keep=70   1,000,494,880 rows   44.73 GiB
    keep=50     112,498,750         5.03
    keep=40      22,862,700         1.02
    keep=32       4,607,680         0.21
    keep<=24              0         negligible (built under the probe cap)

The top two steps are **the 2026-09-03 field measurement on the real P1 file to
the digit** (1,000,494,880 and 112,498,750 rows). The trimmed steps differ (that
run measured 32,412,765 and 7,126,740) because trimming keeps a different set of
families on a different lattice; the regime is what matters here, and the
unaffordable end of it is exact.

Point group reads back as ``1``, space group as ``P1`` (#1), 70 reflector
families, 5 atoms.
"""
from pathlib import Path

#: Deliberately invented, and deliberately irregular — see the docstring. A real
#: database entry would drag provenance and a licence into the test suite for no
#: gain, and would be repairable, which is the failure this file exists to avoid.
SYNTHETIC_P1_CIF = """\
data_synthetic_P1_no_symmetry
_symmetry_space_group_name_H-M   'P 1'
_symmetry_Int_Tables_number      1
_cell_length_a    10.0000
_cell_length_b    10.2000
_cell_length_c    10.4000
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0
_cell_volume      1060.80
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Al1  Al  0.1100  0.2300  0.3700  1.0
  Al2  Al  0.4300  0.6100  0.1900  1.0
  Al3  Al  0.7700  0.0900  0.5300  1.0
  Al4  Al  0.2900  0.8300  0.7100  1.0
  Fe1  Fe  0.6300  0.4700  0.8900  1.0
"""

#: What the stem becomes in a message, so a test can assert on the name.
SYNTHETIC_P1_STEM = "SyntheticP1"


def write_synthetic_p1_cif(directory) -> Path:
    """Write the P 1 CIF into ``directory`` (a ``tmp_path``) and return its path."""
    path = Path(directory) / f"{SYNTHETIC_P1_STEM}.cif"
    path.write_text(SYNTHETIC_P1_CIF, encoding="utf-8")
    return path


def forget_reflector_limit_for(cif_path) -> None:
    """Drop any remembered reflector limit for this phase, without touching disk.

    The limit registry is the SECOND machine-state input to
    ``_describe_hough_failure``: ``_hough_memory_evidence`` takes the first
    ladder step at or below the phase's registered limit, so an entry of 32 turns
    the 44.73 GiB ask into 0.21 GiB, flips "is this plausibly a memory problem",
    and deletes the symmetry sentence from the message. The registry is
    persisted, is keyed on the file STEM, and these tests share a stem with the
    two that deliberately register 32 — so without this, one test file's leftover
    decides another's answer.

    ``clear_phase_reflector_limits(persist=False)`` is the real API and marks the
    registry loaded, so a later lookup does not quietly read the file back in.
    """
    from ebsd_utils import clear_phase_reflector_limits, get_phase_reflector_limit

    clear_phase_reflector_limits(persist=False)
    assert get_phase_reflector_limit(str(cif_path)) is None, (
        "the reflector-limit registry still answers for this phase; the memory "
        "evidence would be measured against a trimmed library"
    )


def synthetic_p1_phase_list(directory):
    """``(PhaseList, cif_path)`` for the synthetic P 1 phase.

    Goes through ``sanitize_cif`` + ``Phase.from_cif`` like the production
    readers do, so the phase a test gets is the phase the code would build.
    """
    from orix.crystal_map import Phase, PhaseList

    from ebsd_utils import sanitize_cif

    path = write_synthetic_p1_cif(directory)
    phase = Phase.from_cif(sanitize_cif(str(path)))
    phase.name = path.stem
    return PhaseList(phase), path
