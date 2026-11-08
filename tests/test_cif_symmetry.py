"""CIF files stored as P 1 whose atoms have more symmetry.

A Materials Project / DFT download written without symmetry is space group P 1
to every reader, and Hough indexing then asks for a band-triplet library of tens
of GiB (a noisy P 1 cell with about 70 reflector families needs about 1.0e9
triplet rows, 44.7 GiB; see ``tests/synthetic_cif.py``). The symmetrised copy of
the same crystal needs well under 48 MiB.

Expected space-group numbers are the International Tables numbers of the group
each crystal is BUILT in (Al Fm-3m 225, Al2Cu I4/mcm 140, hcp Zr P6_3/mmc 194);
they are not read back from the analyser under test. The "noisy" variants carry
0.003 A of random displacement per coordinate, the size of a DFT relaxation
residue.
"""
import hashlib
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymatgen.core import Lattice, Structure  # noqa: E402
from pymatgen.io.cif import CifWriter  # noqa: E402

from backend.api.services import cif_symmetry as cs  # noqa: E402
from backend.api.services.cif_citation import read_citation  # noqa: E402
from tests.synthetic_cif import write_synthetic_p1_cif  # noqa: E402


def _crystals():
    return {
        "Al": (Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.05), ["Al"], [[0, 0, 0]]), 225),
        "Al2Cu": (
            Structure.from_spacegroup(
                "I4/mcm", Lattice.tetragonal(6.07, 4.87), ["Cu", "Al"],
                [[0, 0, 0.25], [0.1583, 0.6583, 0]],
            ),
            140,
        ),
        "Zr": (
            Structure.from_spacegroup("P6_3/mmc", Lattice.hexagonal(3.23, 5.15), ["Zr"], [[1 / 3, 2 / 3, 0.25]]),
            194,
        ),
    }


def _noisy(structure, amplitude_A=0.003, seed=1):
    rng = np.random.default_rng(seed)
    cart = structure.cart_coords + rng.uniform(-amplitude_A, amplitude_A, structure.cart_coords.shape)
    return Structure(structure.lattice, structure.species, structure.lattice.get_fractional_coords(cart))


def _ladder_numbers(structure):
    return [r["number"] for r in cs.symmetry_ladder(structure)]


def _stable_noisy(structure, amplitude=0.003):
    """A noisy copy whose group is the SAME at every tolerance of the ladder, found
    by trying seeds in order (and asserted, so a fixture that is stable by luck says
    so): the tests of the copy's mechanics need a cell for which a copy is offered."""
    for seed in range(100):
        candidate = _noisy(structure, amplitude, seed)
        numbers = _ladder_numbers(candidate)
        if len(set(numbers)) == 1 and numbers[0] > 1:
            return candidate
    raise AssertionError("no seed gives a stable ladder")


def _write_p1(directory: Path, name: str, structure) -> Path:
    path = directory / f"{name}.cif"
    text = str(CifWriter(structure))          # no symprec: space group P 1
    assert "'P 1'" in text, "fixture must be a P 1 file"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("variant", ["exact", "noisy"])
@pytest.mark.parametrize("name", ["Al", "Al2Cu", "Zr"])
def test_hidden_symmetry_is_recovered_from_a_p1_copy(tmp_path, name, variant):
    structure, expected = _crystals()[name]
    if variant == "noisy":
        structure = _noisy(structure)
    path = _write_p1(tmp_path, f"{name}_{variant}", structure)
    v = cs.analyse_p1(path)
    assert v.applicable, v.reason
    assert v.space_group_number == expected
    assert v.symprec == 0.01


def test_the_noise_really_defeats_the_strict_analyser(tmp_path):
    # Guards the fixture: if 0.003 A of noise no longer broke symmetry at
    # symprec 1e-5, the "noisy" cases above would prove nothing.
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    structure, _ = _crystals()["Al2Cu"]
    assert SpacegroupAnalyzer(_noisy(structure), symprec=1e-5).get_space_group_number() == 1


def test_a_genuinely_distorted_cell_is_not_flattened_to_cubic(tmp_path):
    # fcc Cu stretched along c by 2 % (c/a = 1.02): 3.61 A against 3.6822 A, a
    # difference of 0.072 A. That is body-centred tetragonal, I4/mmm (139), and
    # must stay so at 0.01 A.
    a = 3.61
    cu = Structure(
        Lattice.from_parameters(a, a, a * 1.02, 90, 90, 90),
        ["Cu"] * 4,
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )
    path = _write_p1(tmp_path, "CuTetragonal", cu)
    v = cs.analyse_p1(path)
    assert v.applicable
    assert v.space_group_number == 139
    assert v.crystal_system == "tetragonal"

    copy = Path(cs.save_symmetrised_copy(path)["path"])
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    written = Structure.from_file(str(copy))
    assert SpacegroupAnalyzer(written, symprec=1e-5).get_space_group_number() == 139
    abc = sorted(written.lattice.abc)
    assert abc[2] - abc[0] > 0.5          # not a cube in any setting
    # And it is the same crystal: volume per atom of the input, by hand.
    assert written.volume / len(written) == pytest.approx(a * a * a * 1.02 / 4, rel=1e-3)


def test_partial_occupancy_is_not_symmetrised(tmp_path):
    path = tmp_path / "Mixed.cif"
    path.write_text(
        """\
data_Mixed
_symmetry_space_group_name_H-M   'P 1'
_symmetry_Int_Tables_number      1
_cell_length_a 4.05
_cell_length_b 4.05
_cell_length_c 4.05
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Al1 Al 0.0 0.0 0.0 0.5
  Cu1 Cu 0.0 0.0 0.0 0.5
  Al2 Al 0.5 0.5 0.0 1.0
  Al3 Al 0.5 0.0 0.5 1.0
  Al4 Al 0.0 0.5 0.5 1.0
""",
        encoding="utf-8",
    )
    v = cs.analyse_p1(path)
    assert not v.applicable and v.reason == cs.DISORDERED
    with pytest.raises(cs.SymmetriseError) as err:
        cs.save_symmetrised_copy(path)
    assert err.value.code == cs.DISORDERED
    assert [p.name for p in tmp_path.iterdir()] == ["Mixed.cif"]


def test_a_truly_irregular_p1_cell_has_nothing_to_offer(tmp_path):
    path = write_synthetic_p1_cif(tmp_path)
    v = cs.analyse_p1(path)
    assert not v.applicable and v.reason == cs.NO_SYMMETRY


def test_a_file_with_real_symmetry_is_not_p1(tmp_path):
    structure, _ = _crystals()["Al"]
    path = tmp_path / "Al.cif"
    path.write_text(str(CifWriter(structure, symprec=0.01)), encoding="utf-8")
    assert cs.analyse_p1(path).reason == cs.NOT_P1


@pytest.mark.parametrize(
    "spacegroup, lattice, expected_group, conventional_atoms",
    [
        # A primitive cell has one atom and is P 1 in a CIF, yet the crystal
        # is face-centred cubic (Fm-3m, 225; conventional cell 4 atoms) or
        # body-centred cubic (Im-3m, 229; conventional cell 2 atoms). Hough
        # would need 5.03 GiB for the one-atom fcc file.
        ("Fm-3m", Lattice.cubic(4.05), 225, 4),
        ("Im-3m", Lattice.cubic(3.3), 229, 2),
    ],
)
def test_a_one_atom_primitive_cell_is_not_refused(tmp_path, spacegroup, lattice, expected_group, conventional_atoms):
    primitive = Structure.from_spacegroup(spacegroup, lattice, ["Fe"], [[0, 0, 0]]).get_primitive_structure()
    assert len(primitive) == 1
    src = _write_p1(tmp_path, "OneAtom", primitive)
    v = cs.analyse_p1(src)
    assert v.applicable and v.space_group_number == expected_group
    r = cs.save_symmetrised_copy(src)
    assert r["space_group_number"] == expected_group
    assert r["n_atoms"] == conventional_atoms
    assert len(Structure.from_file(r["path"])) == conventional_atoms


def test_an_empty_structure_is_unreadable_not_symmetric():
    from pymatgen.core import Lattice as L

    v = cs._detect(Structure(L.cubic(4.0), [], []))
    assert not v.applicable and v.reason == cs.UNREADABLE


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_copy_is_new_documented_and_keeps_the_citation(tmp_path):
    structure, _ = _crystals()["Zr"]
    src = _write_p1(tmp_path, "Zr_ICSD", _noisy(structure))
    src.write_text(
        src.read_text(encoding="utf-8")
        + "\n_publ_section_doi '10.1000/zr-original'\n"
        + "_publ_section_references\n;\nOriginal Author (1999). J. Original 1, 2.\n;\n",
        encoding="utf-8",
    )
    before = _digest(src)

    r1 = cs.save_symmetrised_copy(src, today=date(2026, 10, 11))
    assert r1["name"] == "Zr_ICSD_sym.cif"
    assert (r1["space_group_number"], r1["space_group_symbol"]) == (194, "P6_3/mmc")
    assert _digest(src) == before                       # the original is untouched

    text = Path(r1["path"]).read_text(encoding="utf-8")
    for needle in (
        "Source file: Zr_ICSD.cif",
        "Symmetrised by Orienta from P 1 file Zr_ICSD.cif",
        "P6_3/mmc (194)",
        "symprec 0.01",
        "2026-10-11",
        "_audit_creation_method",
    ):
        assert needle in text, needle
    cit = read_citation(text)
    assert cit.doi == "10.1000/zr-original"
    assert cit.reference == "Original Author (1999). J. Original 1, 2."

    # The same crystal is not written twice (see the sibling tests); the first
    # copy is untouched by the refused second attempt.
    with pytest.raises(cs.SymmetriseError):
        cs.save_symmetrised_copy(src)
    assert (tmp_path / "Zr_ICSD_sym.cif").read_text(encoding="utf-8") == text


def test_an_occupied_name_makes_the_copy_take_the_next_one(tmp_path):
    # Files that are not this crystal (here: not even a CIF) hold the first two
    # names; the copy takes the third and overwrites nothing.
    structure, _ = _crystals()["Zr"]
    src = _write_p1(tmp_path, "Zr_ICSD", _noisy(structure))
    (tmp_path / "Zr_ICSD_sym.cif").write_text("not a cif\n", encoding="utf-8")
    (tmp_path / "Zr_ICSD_sym2.cif").write_text("also not a cif\n", encoding="utf-8")
    assert cs.save_symmetrised_copy(src)["name"] == "Zr_ICSD_sym3.cif"
    assert (tmp_path / "Zr_ICSD_sym.cif").read_text() == "not a cif\n"
    assert (tmp_path / "Zr_ICSD_sym2.cif").read_text() == "also not a cif\n"


def test_the_copy_reads_back_with_the_group_through_the_pipelines_readers(tmp_path):
    # The readers the Hough builder and the Crystal Database actually use.
    from ebsd_utils import hough_phase_from_cif

    structure, expected = _crystals()["Al2Cu"]
    src = _write_p1(tmp_path, "Al2Cu_noisy", _noisy(structure))
    copy = cs.save_symmetrised_copy(src)["path"]
    assert int(hough_phase_from_cif(copy).space_group.number) == expected
    assert int(hough_phase_from_cif(str(src)).space_group.number) == 1


def test_a_changed_structure_is_refused_not_written(tmp_path, monkeypatch):
    structure, _ = _crystals()["Al"]
    src = _write_p1(tmp_path, "Al_p1", structure)
    monkeypatch.setattr(cs, "_same_crystal", lambda a, b: False)
    with pytest.raises(cs.SymmetriseError) as err:
        cs.save_symmetrised_copy(src)
    assert err.value.code == "changed_structure"
    assert [p.name for p in tmp_path.iterdir()] == ["Al_p1.cif"]


def test_same_crystal_check_rejects_a_different_composition():
    al = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.05), ["Al"], [[0, 0, 0]])
    ni = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.05), ["Ni"], [[0, 0, 0]])
    assert cs._same_crystal(al, al)
    assert not cs._same_crystal(al, ni)


def test_the_hough_library_drops_from_gibibytes_to_mebibytes(tmp_path, monkeypatch):
    # A noisy P 1 hcp cell needs more than 10 GiB of triplet library; its
    # symmetrised copy under 48 MiB. Rows are 48 bytes each.
    monkeypatch.setenv("ORIENTA_NO_PHASE_LIMIT_STORE", "1")
    import kikuchipy as kp
    from orix.crystal_map import PhaseList

    from ebsd_utils import hough_phase_from_cif, predict_triplet_library, prepare_reflectors

    structure, _ = _crystals()["Zr"]
    src = _write_p1(tmp_path, "Zr_noisy", _noisy(structure))
    copy = cs.save_symmetrised_copy(src)["path"]
    det = kp.detectors.EBSDDetector(shape=(60, 60), pc=(0.5, 0.5, 0.5), sample_tilt=70.0)

    def rows(path):
        ph = hough_phase_from_cif(str(path))
        ph.name = Path(path).stem
        pl = PhaseList(ph)
        refl = prepare_reflectors(pl)
        n = len(refl[0].hkl) if isinstance(refl, list) else len(refl.hkl)
        pred = predict_triplet_library(det, pl, refl, counts=(n,))
        return pred[0][1] or 0

    p1_rows, sym_rows = rows(src), rows(copy)
    assert p1_rows * 48 > 10 * 2**30
    assert sym_rows * 48 < 48 * 2**20


# ---------------------------------------------------------------------------
# The group that is reported is the group of the file that is written
# ---------------------------------------------------------------------------
# Two real borderline cells (found by fuzzing noisy P 1 copies of known
# crystals). The analyser at symprec 0.01 A named one "Cmcm" and the other
# "P2"; CifWriter analyses the structure again and wrote Fm-3m and Pnnn. The
# crystals were built in Fm-3m (225) and Pnnn (48, ITA).

NOISY_FCC_AL = """# generated using pymatgen
data_Al
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   4.05000000
_cell_length_b   4.05000000
_cell_length_c   4.05000000
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   1
_chemical_formula_structural   Al
_chemical_formula_sum   Al4
_cell_volume   66.43012500
_cell_formula_units_Z   4
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Al  Al0  1  0.00080359  0.00063485  -0.00048975  1
  Al  Al1  1  0.49927781  0.50034936  -0.00043409  1
  Al  Al2  1  0.49999469  0.00035912  0.49929700  1
  Al  Al3  1  -0.00090237  0.50011089  0.50001898  1
"""

NOISY_PNNN_FEO = """# generated using pymatgen
data_FeO
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   4.00000000
_cell_length_b   5.00000000
_cell_length_c   6.00000000
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   1
_chemical_formula_structural   FeO
_chemical_formula_sum   'Fe8 O8'
_cell_volume   120.00000000
_cell_formula_units_Z   8
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Fe  Fe0  1  0.89954842  0.20017705  0.69980247  1
  Fe  Fe1  1  0.40047365  0.29955970  0.19996027  1
  Fe  Fe2  1  0.89999681  0.79954113  0.30049709  1
  Fe  Fe3  1  0.10063237  0.79983941  0.70022278  1
  Fe  Fe4  1  0.10016397  0.19909464  0.30007228  1
  Fe  Fe5  1  0.59997344  0.70002883  0.19968913  1
  Fe  Fe6  1  0.39988337  0.69993825  0.80034297  1
  Fe  Fe7  1  0.60014481  0.29999808  0.80044138  1
  O  O8  1  0.69975957  0.09986510  0.39947555  1
  O  O9  1  0.20067944  0.40033405  0.90026467  1
  O  O10  1  0.70028964  0.90003816  0.60006221  1
  O  O11  1  0.29989088  0.89992947  0.40001568  1
  O  O12  1  0.30065464  0.10019250  0.59998312  1
  O  O13  1  0.79974912  0.59978003  0.90046266  1
  O  O14  1  0.20021940  0.60002340  0.09990007  1
  O  O15  1  0.79951977  0.39997684  0.10025220  1
"""


def _declared_group(cif_text: str):
    import re

    number = int(re.search(r"^_symmetry_Int_Tables_number\s+(\d+)", cif_text, re.MULTILINE).group(1))
    symbol = re.search(r"^_symmetry_space_group_name_H-M\s+'?([^'\r\n]+?)'?[ \t]*$", cif_text, re.MULTILINE).group(1)
    return number, symbol


@pytest.mark.parametrize(
    "text, number, symbol",
    [(NOISY_FCC_AL, 225, "Fm-3m"), (NOISY_PNNN_FEO, 48, "Pnnn")],
    ids=["noisy-fcc-Al", "noisy-Pnnn-FeO"],
)
def test_reported_group_is_the_group_written_into_the_copy(tmp_path, text, number, symbol):
    src = tmp_path / "Borderline.cif"
    src.write_text(text, encoding="utf-8")

    # These cells are the borderline case at 0.01 A (the analyser says another group
    # than the file CifWriter writes), and over the whole ladder the group changes,
    # so the tolerance is chosen: the point here is that the group reported is the
    # group written at that tolerance.
    assert cs.analyse_p1(src).reason == cs.AMBIGUOUS_TOLERANCE
    verdict = cs.analyse_p1(src, 0.01)          # what the interface shows before the click
    assert verdict.applicable
    assert (verdict.space_group_number, verdict.space_group_symbol) == (number, symbol)

    result = cs.save_symmetrised_copy(src, symprec=0.01)
    written = Path(result["path"]).read_text(encoding="utf-8")
    assert _declared_group(written) == (number, symbol)           # what is in the file
    assert (result["space_group_number"], result["space_group_symbol"]) == (number, symbol)
    # ... and the provenance in the file says the same thing.
    assert f"space group {symbol} ({number}). Tolerance 0.01 Angstrom chosen by the user" in written
    assert f"# Space group of this copy: {symbol} ({number})" in written


def test_the_copy_is_the_same_crystal_as_the_noisy_file(tmp_path):
    from pymatgen.analysis.structure_matcher import StructureMatcher

    src = tmp_path / "Borderline.cif"
    src.write_text(NOISY_PNNN_FEO, encoding="utf-8")
    result = cs.save_symmetrised_copy(src, symprec=0.01)
    matcher = StructureMatcher(ltol=0.01, stol=0.02, angle_tol=0.5, primitive_cell=True, scale=False)
    assert matcher.fit(Structure.from_file(str(src)), Structure.from_file(result["path"]))
    assert result["n_atoms"] == 16                                # 8 Fe + 8 O, counted by hand


@pytest.mark.parametrize("name", ["Al' x.cif", "Al ' ' y.cif", "Al'; z.cif"])
def test_a_file_name_with_quotes_does_not_break_the_copy(tmp_path, name):
    from pymatgen.io.cif import CifFile

    structure, expected = _crystals()["Al"]
    src = _write_p1(tmp_path, "tmp", _noisy(structure))
    named = tmp_path / name
    src.rename(named)
    result = cs.save_symmetrised_copy(named)
    text = Path(result["path"]).read_text(encoding="utf-8")
    block = next(iter(CifFile.from_str(text).data.values())).data      # a strict parse
    assert block["_audit_creation_method"].strip().startswith("Symmetrised by Orienta from P 1 file")
    assert name in block["_audit_creation_method"]
    assert int(block["_symmetry_Int_Tables_number"]) == expected
    assert cs.analyse_p1(result["path"]).reason == cs.NOT_P1          # still a readable, symmetric CIF


def _deny_writing_copies(monkeypatch):
    """Make creating a ``*_sym*.cif`` fail the way a read-only folder does."""
    import builtins

    real = builtins.open

    def fake_open(file, mode="r", *a, **kw):
        if "x" in str(mode) and "_sym" in Path(str(file)).name:
            raise PermissionError(13, "Permission denied", str(file))
        return real(file, mode, *a, **kw)

    monkeypatch.setattr(builtins, "open", fake_open)


def test_an_unwritable_folder_is_a_coded_refusal_without_the_local_path(tmp_path, monkeypatch):
    structure, _ = _crystals()["Al"]
    src = _write_p1(tmp_path, "Al_ro", _noisy(structure))
    _deny_writing_copies(monkeypatch)
    with pytest.raises(cs.SymmetriseError) as err:
        cs.save_symmetrised_copy(src)
    assert err.value.code == cs.NOT_WRITABLE
    assert str(tmp_path) not in str(err.value) and "Al_ro" not in str(err.value)


def test_parallel_requests_never_overwrite_each_others_copy(tmp_path, monkeypatch):
    # The name is claimed with an exclusive create. Same-content detection is
    # switched off here so every thread really tries to write: all of them must
    # end up with a file of their own, none overwritten.
    import threading

    structure, _ = _crystals()["Al"]
    src = _write_p1(tmp_path, "Al_par", _noisy(structure))
    verdict = cs.analyse_p1(src)
    results, errors = [], []

    def work(i):
        try:
            results.append(cs._create_new_file(src, "Al_par", f"copy number {i}\n"))
        except Exception as exc:   # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(24)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and verdict.applicable
    names = sorted(p.name for p in results)
    assert len(set(names)) == 24                                  # 24 different files
    written = sorted(p.read_text() for p in tmp_path.glob("Al_par_sym*.cif"))
    assert written == sorted(f"copy number {i}\n" for i in range(24))   # none overwritten


# ---------------------------------------------------------------------------
# The copy must be visible, and must not duplicate what is already there
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "source_name, copy_name",
    [
        ("Al.P1-backup-2026-09-13.cif", "Al.P1_sym.cif"),
        ("Al.orig.cif", "Al_sym.cif"),
        ("Al.bak.cif", "Al_sym.cif"),
        ("Al_BACKUP.cif", "Al_sym.cif"),
        ("backup.cif", "copy_sym.cif"),
    ],
)
def test_a_copy_does_not_inherit_the_repair_marker(tmp_path, source_name, copy_name):
    from backend.api.services.cif_phase_library import is_backup_file

    structure, _ = _crystals()["Al"]
    src = tmp_path / source_name
    src.write_text(str(CifWriter(_noisy(structure))), encoding="utf-8")
    assert is_backup_file(src)                                   # the premise
    result = cs.save_symmetrised_copy(src)
    assert result["name"] == copy_name
    assert not is_backup_file(result["name"])


def test_the_copy_of_a_backup_is_a_phase_and_is_offered_to_hough(tmp_path, monkeypatch):
    # The three P 1 files of the real library are *.P1-backup-*.cif: their
    # copies must show up in the phase library and in the Hough file list.
    import path_utils
    from backend.api.services.cif_phase_library import load_cif_phase_library_with_skips
    from indexing_controller import IndexingMethod, discover_files_for_method

    structure, _ = _crystals()["Al2Cu"]
    folder = tmp_path / "Database" / "CIF_Library"
    folder.mkdir(parents=True)
    src = folder / "Al2Cu.P1-backup-2026-09-13.cif"
    src.write_text(str(CifWriter(_noisy(structure))), encoding="utf-8")
    result = cs.save_symmetrised_copy(src)

    library, skipped = load_cif_phase_library_with_skips(tmp_path / "none.xlsx", folder)
    assert result["name"] in library
    assert [s["file"] for s in skipped] == [src.name]            # only the backup itself is skipped

    monkeypatch.setattr(path_utils, "get_local_database_path", lambda: tmp_path / "Database")
    listed = [f["filename"] for f in discover_files_for_method(IndexingMethod.HOUGH)["files"]]
    assert result["name"] in listed and src.name not in listed


def _p1_text_of(spacegroup_structure, noisy=True):
    return str(CifWriter(_noisy(spacegroup_structure) if noisy else spacegroup_structure))


def test_a_sibling_that_already_holds_the_crystal_is_reported_not_duplicated(tmp_path):
    structure, _ = _crystals()["Al2Cu"]
    (tmp_path / "Al2Cu.P1-backup-2026-09-13.cif").write_text(_p1_text_of(structure), encoding="utf-8")
    # the hand-repaired sibling, written in its real group
    (tmp_path / "Al2Cu.cif").write_text(str(CifWriter(structure, symprec=0.01)), encoding="utf-8")
    with pytest.raises(cs.SymmetriseError) as err:
        cs.save_symmetrised_copy(tmp_path / "Al2Cu.P1-backup-2026-09-13.cif")
    assert err.value.code == cs.SAME_CONTENT_EXISTS
    assert err.value.extra == {"existing": "Al2Cu.cif"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["Al2Cu.P1-backup-2026-09-13.cif", "Al2Cu.cif"]


def test_a_different_crystal_with_a_similar_name_does_not_block_the_copy(tmp_path):
    al, _ = _crystals()["Al"]
    ni = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.52), ["Ni"], [[0, 0, 0]])
    (tmp_path / "Al.P1-backup.cif").write_text(_p1_text_of(al), encoding="utf-8")
    (tmp_path / "Al.cif").write_text(str(CifWriter(ni, symprec=0.01)), encoding="utf-8")   # not the same crystal
    (tmp_path / "Alloy.cif").write_text(str(CifWriter(al, symprec=0.01)), encoding="utf-8")  # other name entirely
    result = cs.save_symmetrised_copy(tmp_path / "Al.P1-backup.cif")
    assert result["name"] == "Al.P1_sym.cif"


def test_a_second_save_of_the_same_file_names_the_first_copy(tmp_path):
    structure, _ = _crystals()["Zr"]
    src = _write_p1(tmp_path, "Zr_ICSD", _noisy(structure))
    first = cs.save_symmetrised_copy(src)
    with pytest.raises(cs.SymmetriseError) as err:
        cs.save_symmetrised_copy(src)
    assert err.value.code == cs.SAME_CONTENT_EXISTS
    assert err.value.extra["existing"] == first["name"]
    assert sorted(p.name for p in tmp_path.glob("*_sym*.cif")) == [first["name"]]        # no _sym2


def test_parallel_saves_of_one_file_write_exactly_one_copy(tmp_path, monkeypatch):
    import threading
    import time

    structure, _ = _crystals()["Al2Cu"]
    src = _write_p1(tmp_path, "Al2Cu_par", _noisy(structure))
    real = cs._same_content_sibling
    verdict = cs.analyse_p1(src)
    # The analysis takes a second of CPU per thread, which staggers the threads so
    # that the first one has written before the others look. Hand them the verdict
    # at once: all six arrive at the check together.
    monkeypatch.setattr(cs, "analyse_p1", lambda *a, **k: verdict)

    def slow_check(*args, **kwargs):
        # Widen the gap between "nothing there yet" and "written": without the
        # lock every thread sits in it at the same time.
        answer = real(*args, **kwargs)
        time.sleep(0.15)
        return answer

    monkeypatch.setattr(cs, "_same_content_sibling", slow_check)
    outcomes = []

    def work():
        try:
            outcomes.append(cs.save_symmetrised_copy(src)["name"])
        except cs.SymmetriseError as exc:
            outcomes.append(exc.code)

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(p.name for p in tmp_path.glob("*_sym*.cif")) == ["Al2Cu_par_sym.cif"]
    assert outcomes.count("Al2Cu_par_sym.cif") == 1
    assert outcomes.count(cs.SAME_CONTENT_EXISTS) == 5


def test_a_p1_file_with_the_same_crystal_is_not_a_sibling_that_blocks_the_copy(tmp_path):
    # Another P 1 file of the same crystal is no better than the source: only a
    # file in a real group counts as "already there".
    structure, _ = _crystals()["Al2Cu"]
    (tmp_path / "Al2Cu.P1-backup-2026-09-13.cif").write_text(_p1_text_of(structure), encoding="utf-8")
    (tmp_path / "Al2Cu_other_p1.cif").write_text(_p1_text_of(structure, noisy=False), encoding="utf-8")
    result = cs.save_symmetrised_copy(tmp_path / "Al2Cu.P1-backup-2026-09-13.cif")
    assert result["name"] == "Al2Cu.P1_sym.cif"


# ---------------------------------------------------------------------------
# The copy keeps the source's axes when it can, and says so when it cannot
# ---------------------------------------------------------------------------

def _cementite(permuted: bool):
    """Fe3C, Pnma (62), built from the ITA setting a=5.09, b=6.74, c=4.52; with
    ``permuted`` the same crystal in the axis order (b, c, a) = 6.74, 4.52, 5.09."""
    fe3c = Structure.from_spacegroup(
        "Pnma", Lattice.orthorhombic(5.09, 6.74, 4.52), ["Fe", "Fe", "C"],
        [[0.186, 0.063, 0.328], [0.036, 0.25, 0.852], [0.89, 0.25, 0.45]],
    )
    if not permuted:
        return fe3c
    return Structure(
        Lattice(fe3c.lattice.matrix[[1, 2, 0]]), fe3c.species, [[c[1], c[2], c[0]] for c in fe3c.frac_coords]
    )


def _same_atoms_up_to_origin(a, b, tol=0.011):
    """Do the two structures hold the same atoms at the same fractional positions
    of the SAME axes, apart from a common shift of the origin? (The standard
    setting moves the origin to a symmetry centre; that is a translation of the
    whole crystal, not a relabelling of its axes.)"""
    import numpy as np

    if len(a) != len(b):
        return False
    pa = [(str(sp), np.array(c)) for sp, c in zip(a.species, a.frac_coords)]
    pb = [(str(sp), np.array(c)) for sp, c in zip(b.species, b.frac_coords)]
    for sp_b, c_b in pb:
        if sp_b != pa[0][0]:
            continue
        shift = c_b - pa[0][1]
        if all(
            any(sp == sp2 and np.abs(((c + shift - c2 + 0.5) % 1.0) - 0.5).max() <= tol for sp2, c2 in pb)
            for sp, c in pa
        ):
            return True
    return False


@pytest.mark.parametrize("permuted, symbol_written", [(False, True), (True, False)], ids=["ITA-axes", "permuted-axes"])
@pytest.mark.parametrize("noise", [0.0, 0.003])
def test_a_conventional_source_cell_keeps_its_axes(tmp_path, permuted, symbol_written, noise):
    source = _cementite(permuted)
    if noise:
        source = _noisy(source, noise)
    src = _write_p1(tmp_path, "Fe3C", source)
    result = cs.save_symmetrised_copy(src)

    assert result["cell"] == {"kept_source_axes": True, "atoms_before": 16, "atoms_after": 16, "axes": ""}
    assert (result["space_group_number"], result["space_group_symbol"]) == (62, "Pnma")
    text = Path(result["path"]).read_text(encoding="utf-8")
    assert ("_symmetry_space_group_name_H-M" in text) is symbol_written     # no symbol for another setting
    assert "Cell and axes as in the source file." in text

    copy = Structure.from_file(result["path"])
    # the axes, by hand: a=5.09 b=6.74 c=4.52 (ITA) or 6.74, 4.52, 5.09 (permuted)
    expected = (6.74, 4.52, 5.09) if permuted else (5.09, 6.74, 4.52)
    assert tuple(round(x, 3) for x in copy.lattice.abc) == expected
    assert len(copy) == 16
    # every atom sits where it sat in the source cell, in the same axes (fractional,
    # to 0.011, apart from the origin shift to a symmetry centre)
    assert _same_atoms_up_to_origin(source, copy)


def test_the_kept_cell_goes_through_the_hough_reader_with_the_right_group(tmp_path):
    from ebsd_utils import hough_phase_from_cif

    src = _write_p1(tmp_path, "Fe3C", _cementite(True))
    phase = hough_phase_from_cif(cs.save_symmetrised_copy(src)["path"])
    assert phase.point_group.name == "mmm"
    assert len(phase.structure) == 16
    assert int(phase.space_group.number) % 1000 == 62          # 4062: diffpy's name for the non-standard setting


@pytest.mark.parametrize("name", ["Al", "Al2Cu", "Zr"])
def test_the_crystals_of_the_basic_tests_keep_their_axes(tmp_path, name):
    structure, expected = _crystals()[name]
    result = cs.save_symmetrised_copy(_write_p1(tmp_path, name, _noisy(structure)))
    assert result["cell"]["kept_source_axes"] is True
    copy = Structure.from_file(result["path"])
    assert tuple(round(x, 2) for x in copy.lattice.abc) == tuple(round(x, 2) for x in structure.lattice.abc)
    assert result["space_group_number"] == expected


@pytest.mark.parametrize(
    "label, build, before, after",
    [
        ("bcc Fe, primitive cell", lambda: Structure.from_spacegroup("Im-3m", Lattice.cubic(2.87), ["Fe"], [[0, 0, 0]]).get_primitive_structure(), 1, 2),
        ("corundum, primitive cell", lambda: Structure.from_spacegroup(
            "R-3c", Lattice.hexagonal(4.76, 12.99), ["Al", "O"], [[0, 0, 0.352], [0.306, 0, 0.25]]).get_primitive_structure(), 10, 30),
        ("hcp Zr, 2x2x1 supercell", lambda: _crystals()["Zr"][0] * (2, 2, 1), 8, 2),
    ],
)
def test_a_cell_that_cannot_be_kept_is_standardised_and_says_so(tmp_path, label, build, before, after):
    src = _write_p1(tmp_path, "Cell", _stable_noisy(build()))
    result = cs.save_symmetrised_copy(src)
    assert result["cell"]["kept_source_axes"] is False, label
    assert (result["cell"]["atoms_before"], result["cell"]["atoms_after"]) == (before, after), label
    assert result["n_atoms"] == after
    text = Path(result["path"]).read_text(encoding="utf-8")
    sentence = f"Cell converted to the conventional standard setting; atoms per cell {before} -> {after}"
    assert sentence in text
    assert text.count(sentence) == 2                               # the comment and _audit_creation_method


def test_axes_text_is_source_to_copy():
    import numpy as np

    assert cs._axes_text(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])) == "a->b, b->c, c->a"
    assert cs._axes_text(np.eye(3)) == "a->a, b->b, c->c"
    assert cs._axes_text(np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])) == "a->c, b->b, c->a"


def test_relabelled_axes_are_named_for_an_ita_setting_copy():
    # Source a=6.74 b=4.52 c=5.09, copy in the ITA setting a=5.09 b=6.74 c=4.52:
    # source a is the copy's b, source b its c, source c its a.
    assert cs._axes_of_conversion(_cementite(True), _cementite(False)) == "a->b, b->c, c->a"
    # An identity mapping is not a relabelling and is not reported as one.
    assert cs._axes_of_conversion(_cementite(False), _cementite(False)) == ""


def test_the_axes_comparison_can_tell_relabelled_axes_apart():
    # The helper behind the test above must fail for the old behaviour: the same
    # crystal with permuted axes is a different set of fractional positions.
    assert _same_atoms_up_to_origin(_cementite(True), _cementite(True))
    assert not _same_atoms_up_to_origin(_cementite(True), _cementite(False))


@pytest.mark.parametrize(
    "header, expected",
    [
        ("_symmetry_space_group_name_H-M   'P 1'\n_symmetry_Int_Tables_number   1\n", True),
        ("_symmetry_space_group_name_H-M   'P 1'\n", True),
        ("_space_group_name_H-M_alt   'P 1'\n_space_group_IT_number 1\n", True),
        ("", True),                                                          # nothing declared: read as P 1
        ("_symmetry_Int_Tables_number   227\n", False),
        ("_space_group_IT_number 62\n", False),
        ("_symmetry_space_group_name_H-M   'F d -3 m'\n", False),
        ("_symmetry_space_group_name_H-M   'P -1'\n", False),                # P-1 is not P 1
        ("data_a\n_symmetry_Int_Tables_number 1\ndata_b\n_symmetry_Int_Tables_number 225\n", False),
    ],
)
def test_the_text_precheck_tells_p1_headers_from_the_rest(tmp_path, header, expected):
    path = tmp_path / "x.cif"
    path.write_text(f"data_x\n{header}_cell_length_a 4\n", encoding="utf-8")
    assert cs.may_be_p1(path) is expected


def test_the_precheck_costs_milliseconds_where_the_analysis_costs_seconds(tmp_path):
    import time

    structure, _ = _crystals()["Al2Cu"]
    path = _write_p1(tmp_path, "Al2Cu", _noisy(structure))
    t0 = time.perf_counter()
    for _ in range(20):
        assert cs.may_be_p1(path)
    precheck = (time.perf_counter() - t0) / 20
    t0 = time.perf_counter()
    cs.analyse_p1(path)
    analysis = time.perf_counter() - t0
    assert precheck < 0.02 and precheck * 10 < analysis


def test_a_kept_cell_with_another_atom_count_is_not_accepted(tmp_path, monkeypatch):
    # The candidate written in the source's axes must have the source's atom count.
    # Here the writer describes a 2x1x1 supercell of the same crystal with the real
    # writer: it declares the group the analysis finds (140), it is the same crystal
    # for the (primitive-cell) matcher, and it is not the source cell, so only the
    # atom-count check refuses it.
    structure, _ = _crystals()["Al2Cu"]
    src = _write_p1(tmp_path, "Al2Cu", _noisy(structure))
    real = cs._write_in_cell
    monkeypatch.setattr(cs, "_write_in_cell", lambda cell, standard_setting, symprec: real(cell * (2, 1, 1), standard_setting, symprec))
    verdict = cs.analyse_p1(src)
    assert verdict.applicable
    assert verdict._cell["kept_source_axes"] is False
    assert verdict._copy_atoms == 12


# ---------------------------------------------------------------------------
# The copy must be read correctly by the reader the indexing uses
# ---------------------------------------------------------------------------

def _pipeline_atoms(path) -> int:
    from ebsd_utils import hough_phase_from_cif

    return len(hough_phase_from_cif(str(path)).structure)


def _hex_cell_with_gamma_60():
    """NiAs-type P6_3/mmc (194): Ni at 0, As at (1/3, 2/3, 1/4), a = 3.45, c = 5.87,
    4 atoms in the conventional cell. Described with gamma = 60 degrees (axes a and
    a + b), the cell shape many DFT codes write."""
    import numpy as np

    std = Structure.from_spacegroup("P6_3/mmc", Lattice.hexagonal(3.45, 5.87), ["Ni", "As"], [[0, 0, 0], [1 / 3, 2 / 3, 0.25]])
    lattice = Lattice(np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]]) @ std.lattice.matrix)
    return Structure(lattice, std.species, lattice.get_fractional_coords(std.cart_coords))


def test_a_gamma_60_hexagonal_cell_is_not_written_in_an_unreadable_setting(tmp_path):
    # Operators in this cell are not the standard ones of group 194. diffpy does
    # not recognise the setting and expands with the STANDARD operators of the
    # number in the file: 14 atoms instead of 4, and 4 reflector rows instead of 186.
    source = _hex_cell_with_gamma_60()
    assert len(source) == 4 and round(source.lattice.gamma) == 60
    src = _write_p1(tmp_path, "NiAs60", source)
    result = cs.save_symmetrised_copy(src)
    assert result["space_group_number"] == 194
    assert result["cell"]["kept_source_axes"] is False                  # the candidate was refused
    assert _pipeline_atoms(result["path"]) == 4                         # 2 Ni + 2 As, by hand
    assert result["n_atoms"] == 4


def test_the_pipeline_check_is_what_refuses_the_gamma_60_candidate(tmp_path):
    # pymatgen reads the candidate correctly (4 atoms, 194), the pipeline reads 14.
    import numpy as np
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    source = _hex_cell_with_gamma_60()
    standard = SpacegroupAnalyzer(source, symprec=0.01).get_conventional_standard_structure()
    candidate = cs._write_in_cell(cs._in_source_axes(source, standard), False, 0.01)
    back, number, _ = cs._read_back(candidate)
    assert (len(back), number) == (4, 194)
    in_phase, structure = cs._as_the_pipeline_reads_it(candidate)
    assert in_phase == 14 and len(structure) == 14


def _signed_permutations():
    import itertools

    import numpy as np

    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for i, j in enumerate(perm):
                m[i, j] = signs[i]
            if round(float(np.linalg.det(m))) == 1:
                out.append(m)
    return out


@pytest.mark.parametrize("index", range(24))
def test_every_axis_labelling_of_a_p63mc_cell_reads_back_with_its_four_atoms(tmp_path, index):
    # Wurtzite-type P6_3mc (186), 2 Zn + 2 S in the conventional cell, the cell
    # written with each of the 24 proper signed axis permutations. 20 of them made
    # the pipeline read a different number of atoms.
    import numpy as np

    std = Structure.from_spacegroup("P6_3mc", Lattice.hexagonal(3.82, 6.26), ["Zn", "S"], [[1 / 3, 2 / 3, 0], [1 / 3, 2 / 3, 0.375]])
    assert len(std) == 4
    m = _signed_permutations()[index]
    lattice = Lattice(m @ std.lattice.matrix)
    source = Structure(lattice, std.species, lattice.get_fractional_coords(std.cart_coords))
    src = _write_p1(tmp_path, "ZnS", source)
    result = cs.save_symmetrised_copy(src)
    assert result["space_group_number"] == 186
    assert _pipeline_atoms(result["path"]) == 4


# ---------------------------------------------------------------------------
# Near-equal axes: which labelling is meant is decided by the atoms
# ---------------------------------------------------------------------------

def _pnma_with_near_equal_axes(swapped: bool):
    """Pnma (62) with a = 5.10, b = 5.09, c = 6.74. With ``swapped`` the cell is
    written with its axes (b, a, -c): a = 5.09, b = 5.10 and every atom's x and y
    exchanged. The cell SHAPE is the same within 1 % either way round, so only the
    atoms say which labelling the file has."""
    import numpy as np

    std = Structure.from_spacegroup(
        "Pnma", Lattice.orthorhombic(5.10, 5.09, 6.74), ["Fe", "Fe", "C"],
        [[0.186, 0.063, 0.328], [0.036, 0.25, 0.852], [0.25, 0.39, 0.95]],
    )
    if not swapped:
        return std
    lattice = Lattice(np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]) @ std.lattice.matrix)
    return Structure(lattice, std.species, lattice.get_fractional_coords(std.cart_coords))


@pytest.mark.parametrize("noise", [0.0, 0.003])
def test_swapped_near_equal_axes_are_kept_as_the_source_has_them(tmp_path, noise):
    source = _pnma_with_near_equal_axes(swapped=True)
    assert tuple(round(x, 2) for x in source.lattice.abc) == (5.09, 5.10, 6.74)
    src = _write_p1(tmp_path, "PnmaSwapped", _noisy(source, noise) if noise else source)
    result = cs.save_symmetrised_copy(src)
    copy = Structure.from_file(result["path"])
    # The identity labelling is tried first and fits the lengths; the atoms refuse it.
    assert tuple(round(x, 2) for x in copy.lattice.abc) == (5.09, 5.10, 6.74)
    assert result["cell"]["kept_source_axes"] is True
    # The C atom is at (0.39, 0.25, ...) in the source's axes and stays there (up
    # to the origin shift): the earlier writer put it at (0.25, 0.39, ...).
    assert _same_atoms_up_to_origin(source, copy)
    assert not _same_atoms_up_to_origin(_pnma_with_near_equal_axes(swapped=False), copy)


def test_unswapped_near_equal_axes_stay_as_they_are(tmp_path):
    source = _pnma_with_near_equal_axes(swapped=False)
    result = cs.save_symmetrised_copy(_write_p1(tmp_path, "PnmaPlain", source))
    copy = Structure.from_file(result["path"])
    assert tuple(round(x, 2) for x in copy.lattice.abc) == (5.10, 5.09, 6.74)
    assert result["cell"]["kept_source_axes"] is True and _same_atoms_up_to_origin(source, copy)


def test_when_no_labelling_maps_the_atoms_the_standard_path_says_so(tmp_path, monkeypatch):
    # The standard cell of this crystal has its axes sorted by length (5.09, 5.10,
    # 6.74); the file has (5.10, 5.09, 6.74). Only the identity labelling is
    # offered, and it is the wrong one for this file: the copy must not claim
    # "cell and axes as in the source file".
    source = _pnma_with_near_equal_axes(swapped=False)
    src = _write_p1(tmp_path, "PnmaPlain", source)
    real = cs._matching_axes
    monkeypatch.setattr(cs, "_matching_axes", lambda a, b: (m for m in real(a, b) if (m == __import__("numpy").eye(3)).all()))
    result = cs.save_symmetrised_copy(src)
    assert result["cell"]["kept_source_axes"] is False
    text = Path(result["path"]).read_text(encoding="utf-8")
    assert "Cell converted to the conventional standard setting" in text
    assert "Cell and axes as in the source file" not in text
    assert "relabelled" not in text            # the labelling is the same, the atoms are what differ


def test_atoms_map_onto_is_strict_about_axes_noise_and_elements():
    plain = _pnma_with_near_equal_axes(swapped=False)
    swapped = _pnma_with_near_equal_axes(swapped=True)
    assert cs._atoms_map_onto(plain, plain)
    # same lattice shape, atoms exchanged x <-> y: not the same atoms in the same axes
    exchanged = Structure(plain.lattice, plain.species, [[c[1], c[0], c[2]] for c in plain.frac_coords])
    assert not cs._atoms_map_onto(plain, exchanged)
    assert not cs._atoms_map_onto(plain, swapped)             # the other labelling of the same crystal
    # noise: 0.01 A is within the tolerance, 0.3 A is not
    import numpy as np

    def moved(dist):
        coords = np.array(plain.frac_coords)
        coords[0] += np.array([dist / plain.lattice.a, 0, 0])
        return Structure(plain.lattice, plain.species, coords)

    assert cs._atoms_map_onto(plain, moved(0.01))
    assert not cs._atoms_map_onto(plain, moved(0.3))
    # elements
    renamed = Structure(plain.lattice, ["Ni" if str(sp) == "C" else str(sp) for sp in plain.species], plain.frac_coords)
    assert not cs._atoms_map_onto(plain, renamed)


# ---------------------------------------------------------------------------
# Oxidation states
# ---------------------------------------------------------------------------

def _charged_mgo():
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.21), ["Mg2+", "O2-"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def test_charged_type_symbols_do_not_cost_the_source_axes(tmp_path):
    source = _charged_mgo()
    text = str(CifWriter(source, symprec=None))
    assert "Mg2+" in text and "O2-" in text                    # the premise: a charged P 1 file
    src = tmp_path / "MgO.cif"
    src.write_text(text, encoding="utf-8")
    result = cs.save_symmetrised_copy(src)
    assert result["space_group_number"] == 225
    assert result["cell"] == {"kept_source_axes": True, "atoms_before": 8, "atoms_after": 8, "axes": ""}
    copy = Structure.from_file(result["path"])
    assert sorted(str(sp) for sp in copy.species) == ["Mg"] * 4 + ["O"] * 4      # elements, as by hand
    assert tuple(round(x, 2) for x in copy.lattice.abc) == (4.21, 4.21, 4.21)
    assert _same_atoms_up_to_origin(cs._plain(source), copy)


def test_a_failed_write_leaves_no_empty_copy_and_a_retry_reuses_the_name(tmp_path, monkeypatch):
    import builtins
    import errno

    structure, _ = _crystals()["Al"]
    src = _write_p1(tmp_path, "Al_full", _noisy(structure))
    real = builtins.open

    class FailingHandle:
        """The exclusive create succeeds, the write then fails: a full disk."""

        def __init__(self, handle):
            self._handle = handle

        def write(self, _text):
            raise OSError(errno.ENOSPC, "No space left on device")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._handle.close()
            return False

    def fake_open(file, mode="r", *a, **kw):
        handle = real(file, mode, *a, **kw)
        if "x" in str(mode) and "_sym" in Path(str(file)).name:
            return FailingHandle(handle)
        return handle

    monkeypatch.setattr(builtins, "open", fake_open)
    with pytest.raises(cs.SymmetriseError) as err:
        cs.save_symmetrised_copy(src)
    assert err.value.code == cs.NOT_WRITABLE
    assert list(tmp_path.glob("*_sym*.cif")) == []                 # no empty _sym.cif left behind

    monkeypatch.setattr(builtins, "open", real)                     # the disk has room again
    assert cs.save_symmetrised_copy(src)["name"] == "Al_full_sym.cif"   # not _sym2


def test_the_real_twin_is_found_among_many_similarly_named_files(tmp_path):
    # Eleven other crystals share the first name token and sort ahead of the twin
    # (the twin sorts last in either case rule: "zz"); a cap on the number of
    # candidates would miss it. Its name is in capitals, so the first-token match
    # must ignore case.
    structure, _ = _crystals()["Al2Cu"]
    (tmp_path / "Al2Cu.P1-backup-2026-09-13.cif").write_text(_p1_text_of(structure), encoding="utf-8")
    for i in range(11):
        other = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.50 + 0.05 * i), ["Ni"], [[0, 0, 0]])
        (tmp_path / f"Al2Cu_variant{i:02d}.cif").write_text(str(CifWriter(other, symprec=0.01)), encoding="utf-8")
    (tmp_path / "AL2CU_zz_twin.cif").write_text(str(CifWriter(structure, symprec=0.01)), encoding="utf-8")
    with pytest.raises(cs.SymmetriseError) as err:
        cs.save_symmetrised_copy(tmp_path / "Al2Cu.P1-backup-2026-09-13.cif")
    assert err.value.code == cs.SAME_CONTENT_EXISTS
    assert err.value.extra == {"existing": "AL2CU_zz_twin.cif"}


def test_the_mapping_is_one_to_one():
    # Two source atoms within 0.03 A of the SAME copy atom, and nothing near the
    # copy's other atom: not the copy's atoms, although every source atom is
    # close to some copy atom.
    copy = Structure(Lattice.cubic(5.0), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    doubled = Structure(Lattice.cubic(5.0), ["Fe", "Fe"], [[0, 0, 0], [0.005, 0, 0]])
    assert cs._atoms_map_onto(copy, copy)
    assert not cs._atoms_map_onto(doubled, copy)
# ---------------------------------------------------------------------------
# Each guard of the kept-axes candidate refuses on its own
# ---------------------------------------------------------------------------

def _exchanged_xy(structure):
    return Structure(structure.lattice, structure.species, [[c[1], c[0], c[2]] for c in structure.frac_coords])


def test_the_pymatgen_read_back_alone_refuses_a_candidate_with_exchanged_axes(tmp_path, monkeypatch):
    # The candidate describes the right crystal with x and y of every atom
    # exchanged. The pipeline check is switched off, so only the check on the
    # read-back can refuse it.
    source = _pnma_with_near_equal_axes(swapped=False)
    src = _write_p1(tmp_path, "Pnma", source)
    real = cs._write_in_cell
    monkeypatch.setattr(cs, "_write_in_cell", lambda cell, std, sp: real(_exchanged_xy(cell), std, sp))
    monkeypatch.setattr(cs, "_pipeline_accepts", lambda *a, **k: True)
    verdict = cs.analyse_p1(src)
    assert verdict.applicable and verdict._cell["kept_source_axes"] is False


def test_the_pipeline_atoms_alone_refuse_a_candidate_whose_positions_differ(tmp_path, monkeypatch):
    # pymatgen reads the candidate correctly; the pipeline's expansion has the atoms
    # at other positions (same count): refused.
    source = _pnma_with_near_equal_axes(swapped=False)
    src = _write_p1(tmp_path, "Pnma", source)
    real = cs._as_the_pipeline_reads_it
    calls = []

    def first_candidate_only(text):       # the standard path that follows is read for real
        calls.append(1)
        in_phase, structure = real(text)
        return (in_phase, _exchanged_xy(structure)) if len(calls) == 1 else (in_phase, structure)

    monkeypatch.setattr(cs, "_as_the_pipeline_reads_it", first_candidate_only)
    verdict = cs.analyse_p1(src)
    assert verdict.applicable and verdict._cell["kept_source_axes"] is False
    assert len(calls) == 2                # the candidate, then the standard path


def test_the_pipeline_atom_count_alone_refuses_a_candidate(tmp_path, monkeypatch):
    # The positions are right, the phase the pipeline builds has more atoms than
    # the source: refused (this is the gamma = 60 degrees failure in miniature).
    source = _pnma_with_near_equal_axes(swapped=False)
    src = _write_p1(tmp_path, "Pnma", source)
    real = cs._as_the_pipeline_reads_it
    calls = []

    def first_candidate_only(text):
        calls.append(1)
        in_phase, structure = real(text)
        return (in_phase + 10 if len(calls) == 1 else in_phase, structure)

    monkeypatch.setattr(cs, "_as_the_pipeline_reads_it", first_candidate_only)
    verdict = cs.analyse_p1(src)
    assert verdict.applicable and verdict._cell["kept_source_axes"] is False
    assert len(calls) == 2


def test_the_group_check_alone_refuses_a_candidate_declaring_another_group(tmp_path, monkeypatch):
    source = _pnma_with_near_equal_axes(swapped=False)
    src = _write_p1(tmp_path, "Pnma", source)
    real = cs._read_back
    calls = []

    def lying(text):
        back, number, symbol = real(text)
        calls.append(number)
        return back, number + 1 if len(calls) == 1 else number, symbol       # only the candidate's first read

    monkeypatch.setattr(cs, "_read_back", lying)
    verdict = cs.analyse_p1(src)
    assert verdict.applicable and verdict._cell["kept_source_axes"] is False


# ---------------------------------------------------------------------------
# The angle tolerance is stated, not inherited (pymatgen's default is 5 degrees)
# ---------------------------------------------------------------------------

def _bct_iron(c_over_a):
    """Body-centred tetragonal Fe, a = 2.87 A, 2 atoms; c/a = 1.02 is I4/mmm (139)."""
    return Structure(Lattice.tetragonal(2.87, 2.87 * c_over_a), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def _monoclinic_two_atom(beta):
    """Two Fe atoms in a monoclinic cell (P2/m with the second atom at 1/2,1/2,0 gives
    C2/m, #12, for beta != 90; at exactly 90 degrees it is orthorhombic Cmmm, #65)."""
    return Structure(Lattice.monoclinic(3.1, 3.7, 4.3, beta), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.0]])


def test_the_default_angle_tolerance_is_what_the_module_does_not_use():
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    assert cs.ANGLE_TOLERANCE == 0.5
    # the premise: with pymatgen's own default (5 degrees) these distortions vanish
    assert SpacegroupAnalyzer(_bct_iron(1.02), symprec=0.01).get_space_group_number() == 229
    assert SpacegroupAnalyzer(_monoclinic_two_atom(90.4), symprec=0.01).get_space_group_number() == 65


@pytest.mark.parametrize("c_over_a", [1.01, 1.02, 1.05])
@pytest.mark.parametrize("tolerance", [0.01, 0.02, 0.05, 0.1])
def test_a_tetragonal_distortion_of_bcc_stays_tetragonal(c_over_a, tolerance):
    assert cs.spacegroup_analyzer(_bct_iron(c_over_a), tolerance).get_space_group_number() == 139


@pytest.mark.parametrize("beta", [90.4, 91.0])
@pytest.mark.parametrize("tolerance", [0.01, 0.1])
def test_a_two_atom_monoclinic_cell_does_not_become_orthorhombic(beta, tolerance):
    assert cs.spacegroup_analyzer(_monoclinic_two_atom(beta), tolerance).get_space_group_number() == 12


def test_a_distorted_bcc_file_is_offered_as_tetragonal_not_cubic(tmp_path):
    src = _write_p1(tmp_path, "BctFe", _bct_iron(1.02))
    verdict = cs.analyse_p1(src)
    assert verdict.applicable
    assert (verdict.space_group_number, verdict.crystal_system) == (139, "tetragonal")


def test_a_slightly_monoclinic_file_is_offered_as_monoclinic(tmp_path):
    src = _write_p1(tmp_path, "MonoFe", _monoclinic_two_atom(90.4))
    verdict = cs.analyse_p1(src)
    assert verdict.applicable
    assert (verdict.space_group_number, verdict.crystal_system) == (12, "monoclinic")


# ---------------------------------------------------------------------------
# One tolerance is not a measurement: the group is looked at along a ladder
# ---------------------------------------------------------------------------

LADDER_CRYSTALS = {
    "Al": (225, lambda: Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.05), ["Al"], [[0, 0, 0]])),
    "Zr": (194, lambda: Structure.from_spacegroup("P6_3/mmc", Lattice.hexagonal(3.23, 5.15), ["Zr"], [[1 / 3, 2 / 3, 0.25]])),
    "Al2Cu": (140, lambda: Structure.from_spacegroup(
        "I4/mcm", Lattice.tetragonal(6.07, 4.87), ["Cu", "Al"], [[0, 0, 0.25], [0.1583, 0.6583, 0]])),
}


@pytest.mark.parametrize("amplitude", [0.003, 0.005, 0.008, 0.015])
@pytest.mark.parametrize("name", list(LADDER_CRYSTALS))
def test_the_ladder_never_calls_a_subgroup_stable_and_reaches_the_true_group(name, amplitude):
    # Ten random noise patterns per case. At one tolerance the answer is often a
    # subgroup (fcc Al with +-0.005 A: P1, Cm, 129, 59, 12, 7, ...); the outcome is
    # "stable" only where the group is the same along the whole ladder, and then it
    # must be the group the crystal was built in. Every ladder ends in that group at
    # its loosest tolerance (the construction is the independent expectation).
    expected, build = LADDER_CRYSTALS[name]
    outcomes = set()
    for seed in range(10):
        ladder = cs.symmetry_ladder(_noisy(build(), amplitude, seed))
        outcome = cs._ladder_outcome(ladder)
        outcomes.add(outcome)
        if outcome == "stable":
            assert ladder[0]["number"] == expected, (name, amplitude, seed, ladder)
        else:
            assert outcome == "ambiguous", (name, amplitude, seed, ladder)
        assert ladder[-1]["number"] == expected, (name, amplitude, seed, ladder)
    if amplitude >= 0.008:
        assert "ambiguous" in outcomes                      # the ladder really does trigger


def test_a_stable_ladder_offers_a_copy_and_an_unstable_one_does_not(tmp_path):
    # 0.003 A of noise on fcc Al: some seeds are stable, some are not. For every
    # seed, a verdict that offers a copy names the true group; the others ask.
    seen = set()
    for seed in range(10):
        src = _write_p1(tmp_path, f"Al_{seed}", _noisy(LADDER_CRYSTALS["Al"][1](), 0.003, seed))
        verdict = cs.analyse_p1(src)
        if verdict.applicable:
            assert verdict.space_group_number == 225 and verdict._copy_atoms == 4
            seen.add("offered")
        else:
            assert verdict.reason == cs.AMBIGUOUS_TOLERANCE
            assert verdict.ladder[-1]["number"] == 225
            seen.add("asked")
    assert seen == {"offered", "asked"}                       # both happen over these ten seeds


def _al_with_eight_thousandths(seed=1):
    return _noisy(LADDER_CRYSTALS["Al"][1](), 0.008, seed)


def test_the_ladder_is_shown_and_no_copy_is_written_until_a_tolerance_is_chosen(tmp_path):
    # fcc Al, +-0.008 A, seed 1: Cm (8) at 0.01 and 0.02 A, Fm-3m (225) from 0.05 A.
    # The earlier behaviour wrote the Cm copy (8 atoms instead of 4, Hough point
    # group 1m1 instead of m-3m) as "the hidden symmetry".
    src = _write_p1(tmp_path, "AlNoisy", _al_with_eight_thousandths())
    verdict = cs.analyse_p1(src)
    assert not verdict.applicable and verdict.reason == cs.AMBIGUOUS_TOLERANCE
    assert [(r["symprec"], r["number"], r["symbol"]) for r in verdict.ladder] == [
        (0.01, 8, "Cm"), (0.02, 8, "Cm"), (0.05, 225, "Fm-3m"), (0.1, 225, "Fm-3m")]
    assert cs.ladder_text(verdict.ladder) == "0.01 A: Cm (8); 0.02 A: Cm (8); 0.05 A: Fm-3m (225); 0.1 A: Fm-3m (225)"

    with pytest.raises(cs.SymmetriseError) as err:
        cs.save_symmetrised_copy(src)
    assert err.value.code == cs.AMBIGUOUS_TOLERANCE
    assert err.value.extra["ladder"][2]["symbol"] == "Fm-3m"
    assert "0.05 A: Fm-3m (225)" in str(err.value)
    assert list(tmp_path.glob("*_sym*.cif")) == []                         # nothing was written


def test_the_user_chooses_the_tolerance_and_it_is_written_into_the_provenance(tmp_path):
    from ebsd_utils import hough_phase_from_cif

    src = _write_p1(tmp_path, "AlNoisy", _al_with_eight_thousandths())
    result = cs.save_symmetrised_copy(src, symprec=0.05)
    assert (result["space_group_number"], result["n_atoms"]) == (225, 4)
    assert result["tolerance_chosen"] is True and result["symprec"] == 0.05
    assert [r["number"] for r in result["ladder"]] == [8, 8, 225, 225]
    text = Path(result["path"]).read_text(encoding="utf-8")
    assert "Tolerance 0.05 Angstrom chosen by the user from the ladder" in text
    assert "0.01 A: Cm (8)" in text and "0.05 A: Fm-3m (225)" in text
    assert text.count("chosen by the user") == 2                       # the comment and _audit_creation_method
    assert hough_phase_from_cif(result["path"]).point_group.name == "m-3m"


def test_the_user_may_also_choose_the_subgroup_and_gets_what_was_chosen(tmp_path):
    src = _write_p1(tmp_path, "AlNoisy", _al_with_eight_thousandths())
    result = cs.save_symmetrised_copy(src, symprec=0.01)
    assert result["space_group_number"] == 8 and result["tolerance_chosen"] is True


@pytest.mark.parametrize("bad", [0.03, 0.5, 0.0, -0.01])
def test_a_tolerance_outside_the_ladder_is_refused(tmp_path, bad):
    src = _write_p1(tmp_path, "AlNoisy", _al_with_eight_thousandths())
    with pytest.raises(cs.SymmetriseError) as err:
        cs.save_symmetrised_copy(src, symprec=bad)
    assert err.value.code == cs.INVALID_TOLERANCE


def test_an_unchosen_stable_file_says_so_in_its_provenance(tmp_path):
    src = _write_p1(tmp_path, "AlStable", _stable_noisy(LADDER_CRYSTALS["Al"][1]()))
    result = cs.save_symmetrised_copy(src)
    assert result["tolerance_chosen"] is False
    text = Path(result["path"]).read_text(encoding="utf-8")
    assert "The same group at every tolerance of 0.01, 0.02, 0.05, 0.1 Angstrom" in text


def test_p1_at_every_tolerance_is_no_symmetry(tmp_path):
    src = write_synthetic_p1_cif(tmp_path)
    assert cs.analyse_p1(src).reason == cs.NO_SYMMETRY


# ---------------------------------------------------------------------------
# The standard path is checked against the pipeline reader too
# ---------------------------------------------------------------------------

def _corundum_primitive():
    return Structure.from_spacegroup(
        "R-3c", Lattice.hexagonal(4.76, 12.99), ["Al", "O"], [[0, 0, 0.352], [0.306, 0, 0.25]]).get_primitive_structure()


def _displace_first_atom(structure):
    frac = np.array(structure.frac_coords)
    frac[0] += [0.0, 0.0, 0.2]            # about 2.6 A: no operation and common shift maps this back
    return Structure(structure.lattice, structure.species, frac)


def test_the_standard_path_is_unchanged_where_the_pipeline_agrees(tmp_path):
    # Control for the two refusals below: the same cell, the real pipeline reader.
    src = _write_p1(tmp_path, "Corundum", _stable_noisy(_corundum_primitive()))
    verdict = cs.analyse_p1(src)
    assert verdict.applicable and verdict._cell["kept_source_axes"] is False
    assert (verdict.space_group_number, verdict._copy_atoms) == (167, 30)


def test_the_standard_path_refuses_a_copy_the_pipeline_reads_with_other_atom_count(tmp_path, monkeypatch):
    src = _write_p1(tmp_path, "Corundum", _stable_noisy(_corundum_primitive()))
    real = cs._as_the_pipeline_reads_it
    monkeypatch.setattr(cs, "_as_the_pipeline_reads_it", lambda text: (lambda r: (r[0] + 6, r[1]))(real(text)))
    verdict = cs.analyse_p1(src)
    assert not verdict.applicable and verdict.reason == cs.CHANGED_STRUCTURE
    with pytest.raises(cs.SymmetriseError) as err:
        cs.save_symmetrised_copy(src)
    assert err.value.code == cs.CHANGED_STRUCTURE
    assert list(tmp_path.glob("*_sym*.cif")) == []


def test_the_standard_path_refuses_a_copy_the_pipeline_reads_with_other_positions(tmp_path, monkeypatch):
    src = _write_p1(tmp_path, "Corundum", _stable_noisy(_corundum_primitive()))
    real = cs._as_the_pipeline_reads_it
    monkeypatch.setattr(cs, "_as_the_pipeline_reads_it", lambda text: (lambda r: (r[0], _displace_first_atom(r[1])))(real(text)))
    verdict = cs.analyse_p1(src)
    assert not verdict.applicable and verdict.reason == cs.CHANGED_STRUCTURE


# ---------------------------------------------------------------------------
# A header that contradicts itself is not P 1, and not the group it names
# ---------------------------------------------------------------------------

_AL_HEAD = (
    "data_Al\n_cell_length_a 4.05\n_cell_length_b 4.05\n_cell_length_c 4.05\n"
    "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
)
_AL_ATOM = (
    "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
    "_atom_site_fract_y\n_atom_site_fract_z\nAl1 Al 0 0 0\n"
)


def _header_text(symbol_line: str, number_line: str, operators: str = "'x, y, z'") -> str:
    return (
        _AL_HEAD + symbol_line + number_line
        + f"loop_\n_symmetry_equiv_pos_as_xyz\n{operators}\n" + _AL_ATOM
    )


def test_number_1_with_the_symbol_of_fcc_and_only_the_identity_is_a_contradiction(tmp_path):
    # Number 1 and the operator x,y,z say P 1; the symbol says F m -3 m. Read as P 1
    # the one atom has the symmetry Pm-3m (221), a different crystal from the fcc cell
    # of four atoms the symbol describes.
    path = tmp_path / "Al_contradiction.cif"
    path.write_text(_header_text("_symmetry_space_group_name_H-M 'F m -3 m'\n", "_symmetry_Int_Tables_number 1\n"), encoding="utf-8")
    verdict = cs.analyse_p1(path)
    assert not verdict.applicable and verdict.reason == cs.CONTRADICTORY_HEADER
    assert verdict.header == {"number": 1, "symbol": "F m -3 m"}
    with pytest.raises(cs.SymmetriseError) as err:
        cs.save_symmetrised_copy(path)
    assert err.value.code == cs.CONTRADICTORY_HEADER
    assert [p.name for p in tmp_path.iterdir()] == ["Al_contradiction.cif"]


@pytest.mark.parametrize("symbol", ["'P 2_1/c'", '"Fd -3 m :1"', "Pnma", "'R -3 c'"])
def test_any_other_symbol_with_number_1_is_a_contradiction(tmp_path, symbol):
    path = tmp_path / "X.cif"
    path.write_text(_header_text(f"_symmetry_space_group_name_H-M {symbol}\n", "_space_group_IT_number 1\n"), encoding="utf-8")
    assert cs.analyse_p1(path).reason == cs.CONTRADICTORY_HEADER


@pytest.mark.parametrize("symbol", ["'P 1'", '"P 1"', "P1", "'P  1'"])
def test_p1_spelled_in_any_way_is_no_contradiction(symbol):
    assert cs.header_contradiction(f"_symmetry_space_group_name_H-M {symbol}\n_symmetry_Int_Tables_number 1\n") is None


def test_a_consistent_or_incomplete_header_is_no_contradiction():
    assert cs.header_contradiction("_symmetry_space_group_name_H-M 'F m -3 m'\n_symmetry_Int_Tables_number 225\n") is None
    assert cs.header_contradiction("_symmetry_space_group_name_H-M 'F m -3 m'\n") is None
    assert cs.header_contradiction("_symmetry_Int_Tables_number 1\n") is None
    assert cs.header_contradiction("data_x\n") is None


def test_a_comment_or_text_field_is_not_a_header():
    text = "# _symmetry_Int_Tables_number 1\ndata_x\n_symmetry_space_group_name_H-M 'F m -3 m'\n"
    assert cs.header_contradiction(text) is None


# ---------------------------------------------------------------------------
# The sibling scan does not hold the global lock
# ---------------------------------------------------------------------------

def test_the_sibling_scan_of_one_save_does_not_queue_a_save_in_another_folder(tmp_path, monkeypatch):
    import threading
    import time

    folder_a, folder_b = tmp_path / "a", tmp_path / "b"
    folder_a.mkdir()
    folder_b.mkdir()
    structure, _ = _crystals()["Al2Cu"]
    noisy = _stable_noisy(structure)
    src_a = _write_p1(folder_a, "Al2Cu_a", noisy)
    src_b = _write_p1(folder_b, "Al2Cu_b", noisy)
    # A sibling in a real group in folder A: the scan has something to read.
    (folder_a / "Al2Cu_reference.cif").write_text(str(CifWriter(structure, symprec=0.01)), encoding="utf-8")
    verdicts = {src_a: cs.analyse_p1(src_a), src_b: cs.analyse_p1(src_b)}
    monkeypatch.setattr(cs, "analyse_p1", lambda path, *a, **k: verdicts[Path(path)])

    from backend.forward_sim.crystal import cif_origin
    real_read = cif_origin.structure_from_cif

    def slow_read(path, *a, **k):
        if Path(path).name == "Al2Cu_reference.cif":
            time.sleep(1.5)                       # one slow file in folder A
        return real_read(path, *a, **k)

    monkeypatch.setattr(cif_origin, "structure_from_cif", slow_read)
    done = {}

    def save(key, src):
        try:
            cs.save_symmetrised_copy(src)
        except cs.SymmetriseError:
            pass                                  # A may find its sibling: irrelevant here
        done[key] = time.monotonic()

    start = time.monotonic()
    thread_a = threading.Thread(target=save, args=("a", src_a))
    thread_a.start()
    time.sleep(0.3)                               # A is inside its scan
    save("b", src_b)
    thread_a.join()
    assert done["b"] < done["a"], "the save in folder B waited for the scan in folder A"
    assert done["b"] - start < 1.2


def test_a_copy_that_appears_during_the_scan_is_still_found_inside_the_lock(tmp_path, monkeypatch):
    # The scan found nothing; a parallel request then wrote the copy. The re-check
    # inside the lock looks at what appeared since.
    structure, _ = _crystals()["Al2Cu"]
    src = _write_p1(tmp_path, "Al2Cu_race", _stable_noisy(structure))
    verdict = cs.analyse_p1(src)
    monkeypatch.setattr(cs, "analyse_p1", lambda *a, **k: verdict)
    real = cs._same_content_sibling
    calls = []

    def scan_then_a_parallel_write(src_, base, back, skip=frozenset()):
        answer = real(src_, base, back, skip)
        calls.append(sorted(skip))
        if len(calls) == 1:                       # the scan outside the lock is done: another request writes
            (src_.parent / "Al2Cu_race_sym.cif").write_text(str(CifWriter(structure, symprec=0.01)), encoding="utf-8")
        return answer

    monkeypatch.setattr(cs, "_same_content_sibling", scan_then_a_parallel_write)
    with pytest.raises(cs.SymmetriseError) as err:
        cs.save_symmetrised_copy(src)
    assert err.value.code == cs.SAME_CONTENT_EXISTS
    assert err.value.extra["existing"] == "Al2Cu_race_sym.cif"
    assert calls == [[], []]                      # (skip is a frozenset of names already looked at: none existed)


def test_each_sibling_is_read_once_per_save(tmp_path, monkeypatch):
    # The re-check inside the lock looks only at files that appeared after the scan;
    # repeating the whole scan there would hold the lock for every slow read again.
    structure, _ = _crystals()["Al2Cu"]
    src = _write_p1(tmp_path, "Al2Cu_once", _stable_noisy(structure))
    other = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.05), ["Al"], [[0, 0, 0]])
    (tmp_path / "Al2Cu_other_crystal.cif").write_text(str(CifWriter(other, symprec=0.01)), encoding="utf-8")
    verdict = cs.analyse_p1(src)
    monkeypatch.setattr(cs, "analyse_p1", lambda *a, **k: verdict)

    from backend.forward_sim.crystal import cif_origin
    real_read = cif_origin.structure_from_cif
    reads = []

    def counting(path, *a, **k):
        reads.append(Path(path).name)
        return real_read(path, *a, **k)

    monkeypatch.setattr(cif_origin, "structure_from_cif", counting)
    result = cs.save_symmetrised_copy(src)
    assert result["name"] == "Al2Cu_once_sym.cif"
    assert reads.count("Al2Cu_other_crystal.cif") == 1


# ---------------------------------------------------------------------------
# The Hall symbol counts, and so does the opposite contradiction
# ---------------------------------------------------------------------------

def test_number_1_with_the_hall_symbol_of_fcc_is_a_contradiction(tmp_path):
    path = tmp_path / "Al_hall.cif"
    path.write_text(_header_text("_symmetry_space_group_name_Hall '-F 4 2 3'\n", "_symmetry_Int_Tables_number 1\n"), encoding="utf-8")
    verdict = cs.analyse_p1(path)
    assert verdict.reason == cs.CONTRADICTORY_HEADER and not verdict.applicable
    assert verdict.header == {"number": 1, "symbol": "-F 4 2 3 (Hall)"}
    assert cs.may_be_p1(path)                       # it goes on to analyse_p1, which says why


def test_number_225_with_the_symbol_p1_and_only_the_identity_is_a_contradiction(tmp_path):
    # The Hough pipeline reads this file as P 1 with one atom; the number says Fm-3m.
    path = tmp_path / "Al_p1_symbol.cif"
    path.write_text(_header_text("_symmetry_space_group_name_H-M 'P 1'\n", "_symmetry_Int_Tables_number 225\n"), encoding="utf-8")
    assert cs.may_be_p1(path)                       # not "not P 1": the header is not trusted
    verdict = cs.analyse_p1(path)
    assert verdict.reason == cs.CONTRADICTORY_HEADER and not verdict.applicable
    assert verdict.header == {"number": 225, "symbol": "P 1"}
    with pytest.raises(cs.SymmetriseError) as err:
        cs.save_symmetrised_copy(path)
    assert err.value.code == cs.CONTRADICTORY_HEADER


def test_number_225_with_a_hall_symbol_of_p1_is_a_contradiction():
    assert cs.header_contradiction("_symmetry_Int_Tables_number 225\n_symmetry_space_group_name_Hall 'P 1'\n") == {
        "number": 225, "symbol": "P 1 (Hall)"}


@pytest.mark.parametrize(
    "header",
    [
        "_symmetry_Int_Tables_number 225\n_symmetry_space_group_name_Hall '-F 4 2 3'\n",
        "_symmetry_Int_Tables_number 225\n_symmetry_space_group_name_H-M 'F m -3 m'\n_symmetry_space_group_name_Hall '-F 4 2 3'\n",
        "_symmetry_Int_Tables_number 1\n_symmetry_space_group_name_Hall 'P 1'\n",
        "_space_group_IT_number 14\n_space_group_name_H-M_alt 'P 21/c'\n_space_group_name_Hall '-P 2ybc'\n",
        "_symmetry_space_group_name_Hall '-F 4 2 3'\n",                       # no number: not judged
        "_symmetry_Int_Tables_number 1\n_symmetry_Int_Tables_number 225\n_symmetry_space_group_name_H-M 'P 1'\n",   # two numbers: not judged
    ],
)
def test_consistent_or_unjudgeable_headers_are_no_contradiction(header):
    assert cs.header_contradiction(header) is None


# ---------------------------------------------------------------------------
# A rung that fails is a failure in the ladder, not a rung that was not asked
# ---------------------------------------------------------------------------

def _failing_at(monkeypatch, *failing):
    """``spacegroup_analyzer`` raising at the given tolerances only."""
    real = cs.spacegroup_analyzer

    def analyzer(structure, symprec=cs.SYMPREC):
        if failing == () or symprec in failing:
            raise RuntimeError("spglib got no cell")
        return real(structure, symprec)

    monkeypatch.setattr(cs, "spacegroup_analyzer", analyzer)


def test_a_failing_rung_stays_in_the_ladder_and_does_not_make_the_rest_stable(tmp_path, monkeypatch):
    structure = _stable_noisy(LADDER_CRYSTALS["Al"][1]())
    src = _write_p1(tmp_path, "AlStable", structure)
    assert cs.analyse_p1(src).applicable                       # control: stable at all four rungs
    _failing_at(monkeypatch, 0.1)
    ladder = cs.symmetry_ladder(structure)
    assert [r["number"] for r in ladder] == [225, 225, 225, None]
    assert ladder[3]["symprec"] == 0.1 and ladder[3]["error"] == "RuntimeError: spglib got no cell"
    assert cs._ladder_outcome(ladder) == "incomplete"
    verdict = cs.analyse_p1(src)
    assert not verdict.applicable and verdict.reason == cs.AMBIGUOUS_TOLERANCE
    assert cs.ladder_text(verdict.ladder).endswith("0.1 A: not determined")
    with pytest.raises(cs.SymmetriseError) as err:
        cs.save_symmetrised_copy(src)
    assert err.value.code == cs.AMBIGUOUS_TOLERANCE
    chosen = cs.save_symmetrised_copy(src, symprec=0.05)         # a tolerance that worked can still be chosen
    assert chosen["space_group_number"] == 225 and chosen["tolerance_chosen"] is True


def test_a_failure_at_every_rung_is_unreadable_with_the_reason(tmp_path, monkeypatch):
    src = _write_p1(tmp_path, "AlStable", _stable_noisy(LADDER_CRYSTALS["Al"][1]()))
    _failing_at(monkeypatch)
    verdict = cs.analyse_p1(src)
    assert verdict.reason == cs.UNREADABLE and verdict.detail == "RuntimeError: spglib got no cell"
    with pytest.raises(cs.SymmetriseError) as err:
        cs.save_symmetrised_copy(src)
    assert err.value.code == cs.UNREADABLE
    assert "RuntimeError: spglib got no cell" in str(err.value) and err.value.extra["detail"] == "RuntimeError: spglib got no cell"


def test_a_failure_at_the_chosen_tolerance_names_its_reason(tmp_path, monkeypatch):
    src = _write_p1(tmp_path, "AlStable", _stable_noisy(LADDER_CRYSTALS["Al"][1]()))
    _failing_at(monkeypatch, 0.02)
    verdict = cs.analyse_p1(src, 0.02)
    assert verdict.reason == cs.UNREADABLE and "spglib got no cell" in verdict.detail


def test_every_failed_rung_is_logged_as_a_warning(tmp_path, monkeypatch, caplog):
    import logging

    src = _write_p1(tmp_path, "AlStable", _stable_noisy(LADDER_CRYSTALS["Al"][1]()))
    _failing_at(monkeypatch, 0.05)
    with caplog.at_level(logging.WARNING, logger="backend.api.services.cif_symmetry"):
        cs.analyse_p1(src)
    assert [r.levelno for r in caplog.records if "symprec 0.05" in r.getMessage()] == [logging.WARNING]


def test_a_ladder_of_failures_is_unreadable_even_where_a_single_tolerance_would_work(tmp_path, monkeypatch):
    # The ladder is replaced by four failed rungs; the analysis at 0.01 A itself works.
    # Reading the verdict off that one tolerance anyway would offer a copy for a ladder
    # that said nothing.
    src = _write_p1(tmp_path, "AlStable", _stable_noisy(LADDER_CRYSTALS["Al"][1]()))
    failed = [{"symprec": t, "number": None, "symbol": "", "error": "RuntimeError: no cell"} for t in cs.LADDER]
    monkeypatch.setattr(cs, "symmetry_ladder", lambda structure: failed)
    verdict = cs.analyse_p1(src)
    assert not verdict.applicable and verdict.reason == cs.UNREADABLE
    assert verdict.detail == "RuntimeError: no cell"
