"""Every Hough phase has BOTH properties: the CIF's origin choice and the frame symmetry.

Two defects were fixed on separate branches and both live in the one place that
builds a Hough phase from a CIF:

* orix reads a CIF through diffpy, and diffpy expands the 24 space groups with
  two origin choices in choice 1 whatever the file says. ``Si.cif`` came out
  with 16 atoms, ``sd_1816951.cif`` (MgCu2) as Mg32 Cu8 (``test_hough_cif_origin``).
* orix knows one monoclinic point group per class, the one with the two-fold axis
  along Z. In Orienta's crystal frame (X || a, Z || c*) a b-unique phase has it
  along Y, which is what ``crystal_symmetry.FramePhase`` decides
  (``test_crystal_symmetry``).

``ebsd_utils.hough_phase_from_cif`` must give both. If it hands back a plain orix
``Phase`` the monoclinic phases (Al13Fe4, beta-AlFeSi) silently lose the Y axis
again, and nothing about the atoms would show it. These tests pin the
combination on the library CIFs (skipped without ``Database/``), on synthetic
cells that run anywhere, and on every site that builds a Hough phase.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("pymatgen")
pytest.importorskip("spglib")

from collections import Counter  # noqa: E402

from orix.crystal_map import Phase  # noqa: E402

from crystal_symmetry import FramePhase  # noqa: E402
from ebsd_utils import hough_phase_from_cif, sanitize_cif  # noqa: E402

LIBRARY = ROOT / "Database" / "CIF_Library"


# --- helpers -----------------------------------------------------------------

def _library_cif(name: str) -> Path:
    path = LIBRARY / name
    if not path.is_file():
        pytest.skip(f"Database/CIF_Library/{name} is not in this checkout")
    return path


def _composition(phase) -> dict[str, float]:
    counts: Counter = Counter()
    for atom in phase.structure:
        counts[str(atom.element)] += float(atom.occupancy)
    return dict(counts)


def _two_fold_axis(phase) -> str:
    """Which of x, y, z the proper two-fold rotation of the phase's group is about."""
    data = np.asarray(phase.point_group.data, dtype=float)
    found = []
    for axis, name in ((0, "x"), (1, "y"), (2, "z")):
        vec = np.zeros(4)
        vec[axis + 1] = 1.0                       # a half turn about this axis
        if any(abs(float(np.dot(q, vec))) > 0.999 for q in data):
            found.append(name)
    assert len(found) == 1, f"{phase.name}: two-fold axes {found}"
    return found[0]


def _assert_frame_symmetry_y(phase):
    assert isinstance(phase, FramePhase), type(phase)
    assert phase.point_group.proper_subgroup.name == "121"
    assert _two_fold_axis(phase) == "y"


_SI_CHOICE_2 = """data_Synthetic_Silicon
_cell_length_a 5.4309
_cell_length_b 5.4309
_cell_length_c 5.4309
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_Int_Tables_number 227
_symmetry_space_group_name_H-M 'Fd-3m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_Wyckoff_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Si Si 8a 0.125 0.125 0.125 1
"""

#: P 1 21/c 1 (14), unique axis b, beta != 90: a monoclinic cell with one Fe.
_MONOCLINIC = """data_Synthetic_Monoclinic
_cell_length_a 5.0
_cell_length_b 6.0
_cell_length_c 7.0
_cell_angle_alpha 90
_cell_angle_beta 100
_cell_angle_gamma 90
_symmetry_Int_Tables_number 14
_symmetry_space_group_name_H-M 'P 1 21/c 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Fe1 Fe 0.1 0.2 0.3 1
"""


@pytest.fixture()
def silicon_cif(tmp_path) -> Path:
    path = tmp_path / "si_choice2.cif"
    path.write_text(_SI_CHOICE_2, encoding="utf-8", newline="\n")
    return path


@pytest.fixture()
def monoclinic_cif(tmp_path) -> Path:
    path = tmp_path / "mono_b.cif"
    path.write_text(_MONOCLINIC, encoding="utf-8", newline="\n")
    return path


# --- the combination, library files --------------------------------------------

def test_library_silicon_has_eight_atoms_and_the_cubic_laue_group():
    phase = hough_phase_from_cif(_library_cif("Si.cif"))
    assert _composition(phase) == {"Si": 8.0}
    assert isinstance(phase, FramePhase)
    assert phase.point_group.laue.name == "m-3m"


def test_library_mgcu2_is_mg8_cu16_and_a_frame_phase():
    phase = hough_phase_from_cif(_library_cif("sd_1816951.cif"))
    assert _composition(phase) == {"Mg": 8.0, "Cu": 16.0}
    assert isinstance(phase, FramePhase)
    assert phase.point_group.laue.name == "m-3m"


@pytest.mark.parametrize("name,space_group", [("Al13Fe4.cif", 12), ("beta-AlFeSi.cif", 4015)])
def test_library_monoclinic_phases_keep_the_y_two_fold(name, space_group):
    phase = hough_phase_from_cif(_library_cif(name))
    assert phase.space_group.number == space_group
    _assert_frame_symmetry_y(phase)
    # What orix alone would give is the Z axis: the defect this pins.
    plain = Phase.from_cif(sanitize_cif(str(_library_cif(name))))
    assert plain.point_group.proper_subgroup.name == "112"


def test_library_monoclinic_atoms_are_untouched_by_the_origin_handling():
    path = _library_cif("Al13Fe4.cif")
    phase = hough_phase_from_cif(path)
    plain = Phase.from_cif(sanitize_cif(str(path)))
    assert len(phase.structure) == len(plain.structure) == 102
    assert np.array_equal(phase.structure.xyz, plain.structure.xyz)


# --- the combination, synthetic cells (run without Database/) -------------------

def test_synthetic_choice_2_silicon_is_the_diamond_cell_and_a_frame_phase(silicon_cif):
    phase = hough_phase_from_cif(silicon_cif)
    assert _composition(phase) == {"Si": 8.0}
    assert isinstance(phase, FramePhase)
    assert phase.point_group.laue.name == "m-3m"


def test_synthetic_b_unique_monoclinic_cell_has_its_two_fold_along_y(monoclinic_cif):
    _assert_frame_symmetry_y(hough_phase_from_cif(monoclinic_cif))


def test_a_plain_orix_phase_would_put_the_two_fold_along_z(monoclinic_cif):
    """The negative control of the test above: the check can tell the two apart."""
    plain = Phase.from_cif(sanitize_cif(str(monoclinic_cif)))
    assert not isinstance(plain, FramePhase)
    assert _two_fold_axis(plain) == "z"


# --- the sites ----------------------------------------------------------------------

def test_hough_reflectors_phase_from_cif_is_the_combined_builder(silicon_cif, monoclinic_cif):
    import hough_reflectors as hr

    si = hr.phase_from_cif(silicon_cif)
    assert _composition(si) == {"Si": 8.0}
    assert isinstance(si, FramePhase)
    mono = hr.phase_from_cif(monoclinic_cif)
    _assert_frame_symmetry_y(mono)
    assert mono.name == "mono_b"


def test_pc_refinement_phase_has_both_properties(silicon_cif, monoclinic_cif):
    from pc_controller import PCController

    ctrl = PCController()
    ctrl.load_phase(str(silicon_cif))
    assert _composition(ctrl.phase) == {"Si": 8.0}
    assert isinstance(ctrl.phase, FramePhase)
    ctrl = PCController()
    ctrl.load_phase(str(monoclinic_cif))
    _assert_frame_symmetry_y(ctrl.phase)


def test_the_per_phase_hough_config_has_both_properties(silicon_cif, monoclinic_cif):
    from backend.api.routes import indexing as route

    req = SimpleNamespace(method="hough", cif_paths=[str(silicon_cif), str(monoclinic_cif)],
                          master_h5_paths=[], sht_paths=[])
    configs = route._build_phase_configs(req)
    phases = [p for cfg in configs for _, p in cfg.phase_list]
    assert len(phases) == 2
    assert _composition(phases[0]) == {"Si": 8.0} and isinstance(phases[0], FramePhase)
    _assert_frame_symmetry_y(phases[1])


def test_the_cached_hough_indexer_has_both_properties(monoclinic_cif, monkeypatch):
    """The phase the cached indexer is built from. The indexer itself is a stub:
    a real one for this cell needs ~5 GiB of band triplets, and whether that fits
    depends on the machine's free memory, not on the phase."""
    import ebsd_utils
    from backend.api.routes import indexing as route

    monkeypatch.setattr(route, "_HOUGH_INDEXER_CACHE", {})
    monkeypatch.setattr(ebsd_utils, "create_indexer", lambda *a, **kw: object())
    monkeypatch.setattr(ebsd_utils, "prepare_reflectors", lambda pl, *a, **kw: object())
    det = {"pat_height": 60, "pat_width": 60, "sample_tilt": 70.0, "tilt": 0.0,
           "pc_x": 0.5, "pc_y": 0.5, "pc_z": 0.5, "binning": 1}
    pl, _det, _indexer = route._hough_indexer_for(str(monoclinic_cif), det)
    (phase,) = [p for _, p in pl]
    _assert_frame_symmetry_y(phase)


# --- guards: no site builds a Hough phase around the builder -------------------------

_MACHINERY = {"create_indexer", "prepare_reflectors", "get_indexer"}

#: Directories (first path component) that hold no shipped Hough code.
_SKIP = {"tests", "tasks", "research", "docs", "scripts", "node_modules", "Database",
         "frontend", "electron", ".claude", "crystal-structures-for-ebsd-main",
         "Test_data", "dxa", "Linux_Maker", "branding", "logs", "licenses",
         "diagnostic_runs", "examples", "images_for_presentation", "screenshots"}


def _source_trees():
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if rel.parts and rel.parts[0] in _SKIP:
            continue
        try:
            yield rel.as_posix(), ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue


def _called_names(node) -> set[str]:
    names = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                names.add(f.attr)
        elif isinstance(n, ast.Name) and n.id == "hough_phase_from_cif":
            names.add("hough_phase_from_cif")     # handed to asyncio.to_thread
    return names


def _functions(tree):
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


#: Functions that set up Hough machinery and read a CIF themselves, and why that
#: is not a phase for Hough. Empty: every such function goes through the builder.
_MAY_READ_A_CIF_THEMSELVES: dict[str, str] = {}


def test_no_function_that_sets_up_hough_builds_its_phase_with_from_cif():
    """A function that calls create_indexer / prepare_reflectors / get_indexer AND
    ``from_cif`` builds a Hough phase around ``hough_phase_from_cif``: it would
    bring back the 16-atom silicon or the Z-axis monoclinic group."""
    offenders, with_machinery = [], 0
    for rel, tree in _source_trees():
        for fn in _functions(tree):
            names = _called_names(fn)
            if not names & _MACHINERY:
                continue
            with_machinery += 1
            if "from_cif" in names and f"{rel}::{fn.name}" not in _MAY_READ_A_CIF_THEMSELVES:
                offenders.append(f"{rel}:{fn.lineno} {fn.name}")
    assert offenders == []
    # A positive anchor: the scan found the Hough machinery at all.
    assert with_machinery >= 8, with_machinery


def test_the_builder_is_built_on_frame_phase_and_not_on_orix_phase():
    tree = ast.parse((ROOT / "ebsd_utils.py").read_text(encoding="utf-8"))
    (builder,) = [f for f in _functions(tree) if f.name == "hough_phase_from_cif"]
    imported = {(n.module, a.name)
                for n in ast.walk(builder) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert ("crystal_symmetry", "FramePhase") in imported
    assert ("orix.crystal_map", "Phase") not in imported
    assert "from_cif" in _called_names(builder)


def test_hough_reflectors_has_no_second_way_to_build_the_phase():
    tree = ast.parse((ROOT / "hough_reflectors.py").read_text(encoding="utf-8"))
    (fn,) = [f for f in _functions(tree) if f.name == "phase_from_cif"]
    names = _called_names(fn)
    assert "hough_phase_from_cif" in names
    assert "from_cif" not in names and "sanitize_cif" not in names


def test_every_hough_site_calls_the_builder():
    """The sites the origin fix named, counted by call (or hand-over to a thread)."""
    sites = {
        "backend/api/routes/indexing.py": 8,
        "backend/api/services/batch_manager.py": 1,
        "backend/spherical_gpu/pipeline/resolution.py": 1,
        "pc_controller.py": 1,
        "hough_reflectors.py": 1,
    }
    for rel, minimum in sites.items():
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        uses = 0
        for fn in _functions(tree):
            if "hough_phase_from_cif" in _called_names(fn):
                uses += 1
        assert uses >= minimum, (rel, uses)

