"""Hough builds its phases with diffpy, and diffpy expands in origin choice 1.

The master-pattern path was repaired on 2026-09-25 (`cif_origin.structure_from_cif`):
a CIF in one of the 24 space groups with two origin choices is now expanded in
the origin it was written in. The Hough path never used that repair. It built
every phase with ``orix Phase.from_cif(sanitize_cif(path))``, which is diffpy's
P_cif, and diffpy's operators for those 24 groups are origin choice 1. Measured
on 2026-10-02 on the library files that are written in choice 2:

    Si.cif          16 Si             should be  8 Si
    sd_1816951.cif  Mg32 Cu8          should be  Mg8 Cu16

The atoms feed the structure factors that choose Hough's reflectors. For
silicon the wrong cell put the diamond-forbidden {222} into the band list and
dropped {220} and {224} — {220} being one of silicon's strongest bands.

`ebsd_utils.hough_phase_from_cif` is the one builder every Hough site uses now.
The tests below drive it, and three of the call sites themselves, mostly with
synthetic cells so they mean something in a checkout without ``Database/``.
"""
from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("pymatgen")
pytest.importorskip("spglib")

from orix.crystal_map import Phase, PhaseList  # noqa: E402

from backend.forward_sim.crystal.cif_origin import (  # noqa: E402
    CifOriginError,
    structure_from_cif,
)
from ebsd_utils import (  # noqa: E402
    hough_phase_from_cif,
    prepare_reflectors,
    sanitize_cif,
)

LIBRARY = ROOT / "Database" / "CIF_Library"


# --- helpers -----------------------------------------------------------------

_HEADER = """data_{block}
_cell_length_a {a}
_cell_length_b {a}
_cell_length_c {a}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_Int_Tables_number 227
_symmetry_space_group_name_H-M '{symbol}'
loop_
_atom_site_label
_atom_site_type_symbol
{wyckoff_column}_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
"""


def _cif(tmp_path: Path, name: str, *, a: float, symbol: str,
         sites: list[tuple[str, str, str, str, str, str]]) -> Path:
    """A minimal Fd-3m CIF. An empty Wyckoff symbol on every site drops the column.

    The block name has capitals on purpose: diffpy's CIF reader lowercases block
    names and pymatgen's does not, and the helper has to find the same block in
    both. An all-lowercase name would let a case-sensitive match pass here.
    """
    with_wyckoff = any(s[2] for s in sites)
    text = _HEADER.format(block="Synthetic_" + Path(name).stem, a=a, symbol=symbol,
                          wyckoff_column="_atom_site_Wyckoff_symbol\n" if with_wyckoff else "")
    for label, element, wyckoff, x, y, z in sites:
        cells = [label, element] + ([wyckoff or "."] if with_wyckoff else []) + [x, y, z, "1"]
        text += " ".join(cells) + "\n"
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _si_choice_2(tmp_path: Path) -> Path:
    """8a at (1/8,1/8,1/8): how the databases publish silicon (origin choice 2)."""
    return _cif(tmp_path, "si_choice2.cif", a=5.4309, symbol="Fd-3m",
                sites=[("Si", "Si", "8a", "0.125", "0.125", "0.125")])


def _si_choice_1(tmp_path: Path) -> Path:
    """The same crystal written in origin choice 1: 8a at the origin."""
    return _cif(tmp_path, "si_choice1.cif", a=5.4309, symbol="Fd-3m O1",
                sites=[("Si", "Si", "8a", "0", "0", "0")])


def _mgcu2_choice_2(tmp_path: Path) -> Path:
    """sd_1816951's cell: Mg 8a and Cu 16d in origin choice 2."""
    return _cif(tmp_path, "mgcu2_choice2.cif", a=7.0309, symbol="Fd-3m",
                sites=[("Mg", "Mg", "8a", "0.125", "0.125", "0.125"),
                       ("Cu", "Cu", "16d", "0.5", "0.5", "0.5")])


def _mgcu2_choice_1(tmp_path: Path) -> Path:
    """MgCu2 in origin choice 1 (ITA): Mg 8a (0,0,0), Cu 16d (5/8,5/8,5/8)."""
    return _cif(tmp_path, "mgcu2_choice1.cif", a=7.0309, symbol="Fd-3m O1",
                sites=[("Mg", "Mg", "8a", "0", "0", "0"),
                       ("Cu", "Cu", "16d", "0.625", "0.625", "0.625")])


def _composition(phase) -> dict[str, float]:
    counts: Counter = Counter()
    for atom in phase.structure:
        counts[str(atom.element)] += float(atom.occupancy)
    return dict(counts)


def _reflectors(phase) -> dict[tuple[int, int, int], float]:
    """hkl -> |F| of the reflector list Hough would be given for this phase."""
    ref = prepare_reflectors(PhaseList(phase.deepcopy()))
    hkl = np.rint(ref.hkl).astype(int)
    return {tuple(int(v) for v in h): float(f)
            for h, f in zip(hkl, np.abs(ref.structure_factor))}


def _families(hkl_set) -> set[tuple[int, ...]]:
    return {tuple(sorted(abs(v) for v in h)) for h in hkl_set}


def _assert_identical(a, b):
    """Bit for bit: what Phase.from_cif built is what the helper returns.

    Not the name: for a CIF with text before its first ``data_`` (every
    pymatgen-written file), `sanitize_cif` writes a fresh temp copy per call and
    orix names the phase after that copy, so two calls never agree on it. The
    call sites rename the phase afterwards; names are checked where no copy is
    made.
    """
    assert a.space_group.number == b.space_group.number
    assert a.point_group.name == b.point_group.name
    assert np.array_equal(a.structure.lattice.abcABG(), b.structure.lattice.abcABG())
    assert np.array_equal(a.structure.lattice.base, b.structure.lattice.base)
    assert len(a.structure) == len(b.structure)
    for x, y in zip(a.structure, b.structure):
        assert (x.element, x.label) == (y.element, y.label)
        assert np.array_equal(x.xyz, y.xyz)
        assert x.occupancy == y.occupancy
        assert np.array_equal(x.U, y.U)


def _library_cif(name: str) -> Path:
    path = LIBRARY / name
    if not path.is_file():
        pytest.skip(f"Database/CIF_Library/{name} is not in this checkout")
    return path


# --- synthetic cells -----------------------------------------------------------

def test_diffpy_still_doubles_a_choice_2_silicon_cell(tmp_path):
    """The premise, measured. If this starts failing, diffpy has learned to read
    the origin choice and the correction in the helper can go."""
    assert len(Phase.from_cif(str(_si_choice_2(tmp_path))).structure) == 16


def test_choice_2_silicon_comes_back_as_the_diamond_cell(tmp_path):
    phase = hough_phase_from_cif(_si_choice_2(tmp_path))
    assert _composition(phase) == {"Si": 8.0}


def test_choice_2_silicon_gives_hough_the_reflectors_of_real_silicon(tmp_path):
    """What Hough is handed. Diamond-cubic silicon has {220} and no {222} (the
    textbook forbidden reflection); the origin-blind cell had it backwards."""
    fixed = _reflectors(hough_phase_from_cif(_si_choice_2(tmp_path)))
    families = _families(fixed)
    assert (0, 2, 2) in families
    assert (2, 2, 2) not in families

    # and it is the reflector list of the same crystal written in choice 1
    reference = _reflectors(Phase.from_cif(str(_si_choice_1(tmp_path))))
    assert set(fixed) == set(reference)
    for hkl, f in reference.items():
        assert fixed[hkl] == pytest.approx(f, rel=1e-9)


def test_choice_2_mgcu2_is_not_inverted_to_mg2cu(tmp_path):
    path = _mgcu2_choice_2(tmp_path)
    assert _composition(Phase.from_cif(str(path))) == {"Mg": 16.0, "Cu": 8.0}

    phase = hough_phase_from_cif(path)
    assert _composition(phase) == {"Mg": 8.0, "Cu": 16.0}

    fixed = _reflectors(phase)
    reference = _reflectors(Phase.from_cif(str(_mgcu2_choice_1(tmp_path))))
    assert set(fixed) == set(reference)
    for hkl, f in reference.items():
        assert fixed[hkl] == pytest.approx(f, rel=1e-9)


def test_the_correction_changes_only_the_atoms(tmp_path):
    path = _si_choice_2(tmp_path)
    before = Phase.from_cif(str(path))
    after = hough_phase_from_cif(path)
    assert after.name == before.name
    assert after.space_group.number == before.space_group.number == 227
    assert after.point_group.name == before.point_group.name
    assert np.array_equal(after.structure.lattice.abcABG(), before.structure.lattice.abcABG())
    assert np.array_equal(after.structure.lattice.base, before.structure.lattice.base)


def test_a_choice_1_cif_is_returned_untouched(tmp_path):
    path = _si_choice_1(tmp_path)
    phase = hough_phase_from_cif(path)
    _assert_identical(phase, Phase.from_cif(str(path)))
    assert phase.name == "si_choice1"


def test_an_undecidable_origin_is_refused_not_guessed(tmp_path):
    """No operators, no Wyckoff symbols, no formula, no marker on the symbol:
    `cif_origin` refuses this for the master pattern, so Hough refuses it too."""
    path = _cif(tmp_path, "bare_si.cif", a=5.4309, symbol="Fd-3m",
                sites=[("Si", "Si", "", "0.125", "0.125", "0.125")])
    with pytest.raises(CifOriginError, match="bare_si"):
        hough_phase_from_cif(path)


# --- the library ---------------------------------------------------------------

def test_library_silicon_has_eight_atoms():
    path = _library_cif("Si.cif")
    phase = hough_phase_from_cif(path)
    structure, _ = structure_from_cif(path)
    assert _composition(phase) == {"Si": 8.0} == structure.composition.get_el_amt_dict()


def test_library_mgcu2_is_mg8_cu16():
    path = _library_cif("sd_1816951.cif")
    phase = hough_phase_from_cif(path)
    structure, _ = structure_from_cif(path)
    assert _composition(phase) == {"Mg": 8.0, "Cu": 16.0}
    assert structure.composition.get_el_amt_dict() == {"Mg": 8.0, "Cu": 16.0}


def test_library_aluminium_is_bit_identical_to_before():
    path = _library_cif("Al.cif")
    phase = hough_phase_from_cif(path)
    assert _composition(phase) == {"Al": 4.0}
    before = Phase.from_cif(sanitize_cif(str(path)))
    _assert_identical(phase, before)
    assert phase.name == before.name == "Al"


def test_library_fd3m_file_already_in_choice_1_is_untouched():
    """Al3Fe2Si is in Fd-3m too, but its explicit operators are choice 1."""
    path = _library_cif("Al3Fe2Si_mp-1190708_symmetrized.cif")
    _assert_identical(hough_phase_from_cif(path), Phase.from_cif(sanitize_cif(str(path))))


def test_only_the_two_choice_2_library_files_change():
    """The regression net: every other library CIF comes out exactly as before."""
    if not LIBRARY.is_dir():
        pytest.skip("Database/CIF_Library is not in this checkout")
    changed = []
    for path in sorted(LIBRARY.glob("*.cif")):
        if ".P1-backup-" in path.name:
            continue
        before = Phase.from_cif(sanitize_cif(str(path)))
        after = hough_phase_from_cif(path)
        try:
            _assert_identical(after, before)
        except AssertionError:
            changed.append(path.stem)
    assert sorted(changed) == ["Si", "sd_1816951"], changed


# --- the call sites ------------------------------------------------------------

def test_pc_refinement_loads_the_corrected_phase(tmp_path):
    from pc_controller import PCController

    ctrl = PCController()
    ctrl.load_phase(str(_si_choice_2(tmp_path)))
    assert _composition(ctrl.phase) == {"Si": 8.0}
    hkl = {tuple(int(v) for v in h) for h in np.rint(ctrl.reflectors.hkl).astype(int)}
    assert (2, 2, 2) not in _families(hkl)


def test_the_per_phase_hough_config_carries_the_corrected_phase(tmp_path):
    from backend.api.routes import indexing as route

    req = SimpleNamespace(method="hough", cif_paths=[str(_si_choice_2(tmp_path))],
                          master_h5_paths=[], sht_paths=[])
    (cfg,) = route._build_phase_configs(req)
    (phase,) = [p for _, p in cfg.phase_list]
    assert _composition(phase) == {"Si": 8.0}


def test_the_cached_hough_indexer_is_built_from_the_corrected_phase(tmp_path):
    pytest.importorskip("pyebsdindex")
    from backend.api.routes import indexing as route

    det = {"pat_height": 60, "pat_width": 60, "sample_tilt": 70.0, "tilt": 0.0,
           "pc_x": 0.5, "pc_y": 0.5, "pc_z": 0.5, "binning": 1}
    pl, _det, _indexer = route._hough_indexer_for(str(_si_choice_2(tmp_path)), det)
    (phase,) = [p for _, p in pl]
    assert _composition(phase) == {"Si": 8.0}


#: Hough modules, and the functions in them that may still call
#: ``Phase.from_cif`` directly — each reads only the point group or space
#: group (which no origin choice changes) or does not feed Hough at all.
_HOUGH_MODULES = {
    "backend/api/routes/indexing.py": {
        "_phase_point_group_and_name": "point group only",
        "_describe_hough_failure": "point group for a failure message",
        "assign_phase_to_grain": "rebuilds a pruned xmap phase; not a Hough run",
    },
    "backend/api/services/batch_manager.py": {},
    "backend/spherical_gpu/pipeline/resolution.py": {},
    "pc_controller.py": {},
}


def test_no_hough_site_builds_a_phase_around_the_helper():
    """Every other direct ``from_cif`` in these modules is a site that skips the
    correction — the way a new call site would silently bring the defect back."""
    offenders = []
    for rel, allowed in _HOUGH_MODULES.items():
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))

        def visit(node, owner):
            for child in ast.iter_child_nodes(node):
                here = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else owner
                if (isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                        and child.func.attr == "from_cif" and here not in allowed):
                    offenders.append(f"{rel}:{child.lineno} in {here}")
                visit(child, here)

        visit(tree, "<module>")
    assert offenders == []

    # A positive anchor, so a refactor that renames the helper cannot leave this
    # guard checking nothing.
    text = (ROOT / "backend/api/routes/indexing.py").read_text(encoding="utf-8")
    assert text.count("hough_phase_from_cif(") >= 8
