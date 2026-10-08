"""Every place that reduces a monoclinic phase's orientations uses its real axis.

``test_crystal_symmetry.py`` pins the decision. These tests pin that the code
which builds phases and reduces orientations actually asks it: a b-unique phase
whose two pixels differ by its true two-fold axis (about Y in the X||a, Z||c*
frame) is ONE orientation for grain reconstruction, boundary angles, the
pseudo-symmetry helpers, the dictionary grid and the Hough reflector orbits.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from diffpy.structure import Lattice, Structure
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Rotation
from orix.quaternion.symmetry import C2h, get_point_group

import crystal_symmetry as cs
from crystal_symmetry import FramePhase, frame_symmetry

ROOT = Path(__file__).resolve().parents[1]
LATTICE_B = Lattice(15.49, 8.08, 12.48, 90, 107.67, 90)
ROT_Y = Rotation.from_axes_angles([0, 1, 0], np.pi)


def _plain_monoclinic_phase(space_group=12):
    return Phase("Al13Fe4", space_group=space_group,
                 structure=Structure(lattice=LATTICE_B))


def _twin_map(phase, n=8, extra_deg=0.0):
    """Left half one orientation, right half the same crystal via the Y two-fold,
    optionally tilted by ``extra_deg`` on top (a real, small misorientation)."""
    rng = np.random.default_rng(11)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    base = Rotation(q)
    twin = ROT_Y * Rotation.from_axes_angles([1, 2, 3], np.radians(extra_deg)) * base
    quats = np.empty((n * n, 4))
    for r in range(n):
        for c in range(n):
            quats[r * n + c] = (base if c < n // 2 else twin).data[0]
    ys, xs = np.divmod(np.arange(n * n), n)
    return CrystalMap(rotations=Rotation(quats), phase_id=np.ones(n * n, int),
                      x=xs.astype(float), y=ys.astype(float),
                      phase_list=PhaseList(phases=[phase], ids=[1]))


def test_boundary_angles_see_no_boundary_across_the_two_fold():
    from backend.api.services.grain_boundaries import boundary_angles

    xmap = _twin_map(_plain_monoclinic_phase())
    ang = boundary_angles(xmap, 8, 8)
    assert np.nanmax(ang["h"]) < 1e-3
    assert np.nanmax(ang["v"]) < 1e-3


def test_boundary_angles_would_draw_a_wall_with_the_z_axis_group():
    """The same map under orix's own group: the wall this change removes."""
    from orix.quaternion import Misorientation, Orientation

    xmap = _twin_map(_plain_monoclinic_phase())
    q = xmap.rotations.data.reshape(8, 8, 4)
    a, b = q[:, 3], q[:, 4]
    z = Misorientation(Orientation(b, symmetry=C2h) * ~Orientation(a, symmetry=C2h),
                       symmetry=(C2h, C2h)).map_into_symmetry_reduced_zone()
    assert np.degrees(np.asarray(z.angle)).min() > 100


def test_grain_reconstruction_merges_the_two_fold_twin_into_one_grain():
    from analysis.ebsd_dataset import EBSDDataset
    from analysis.grain_analysis import reconstruct_grains

    ds = EBSDDataset(_twin_map(_plain_monoclinic_phase()), step_size=1.0)
    assert ds.phase.point_group.proper_subgroup.name == "121"
    grains = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=1)
    assert grains.n_grains == 1


def test_kam_sees_the_real_misorientation_across_the_two_fold():
    from analysis.deformation_analysis import calculate_kam
    from analysis.ebsd_dataset import EBSDDataset

    ds = EBSDDataset(_twin_map(_plain_monoclinic_phase(), extra_deg=1.0), step_size=1.0)
    kam = np.asarray(calculate_kam(ds, order=1, threshold_deg=7.0), dtype=float)
    # The interface carries the 1 degree that is really there. With the Z-axis
    # group the pair reads 180 degrees, is excluded as a boundary, and the
    # interface pixels show no strain at all.
    assert np.nanmax(kam[:, 3:5]) > 0.1
    assert np.nanmax(kam[:, :2]) < 1e-2


def test_cubic_grain_reconstruction_is_unchanged():
    """Nothing but monoclinic phases may move."""
    from analysis.ebsd_dataset import EBSDDataset

    ph = Phase("Al", space_group=225)
    ds = EBSDDataset(_twin_map(ph), step_size=1.0)
    assert ds.phase.point_group is get_point_group(225)


def test_store_result_gives_every_stored_xmap_the_real_axis():
    from backend.api.routes import indexing

    xmap = _twin_map(_plain_monoclinic_phase())
    assert xmap.phases[1].point_group.proper_subgroup.name == "112"
    result = SimpleNamespace(xmap=xmap, metadata={})
    rid = indexing._store_result(result, "hough")
    try:
        assert result.xmap.phases[1].point_group.proper_subgroup.name == "121"
    finally:
        indexing._result_registry.pop(rid, None)


@pytest.mark.parametrize("direction", ["X", "Y", "Z"])
def test_one_grain_is_one_ipf_colour_across_the_two_fold(direction):
    from tools.phase_map_generator import compute_ipf_colors

    xmap = _twin_map(_plain_monoclinic_phase())          # a plain orix Phase
    rgb = np.asarray(compute_ipf_colors(xmap, direction)).reshape(8, 8, 3)
    assert np.abs(rgb[:, :4] - rgb[:, 4:]).max() < 1e-6


def test_pseudosym_reduces_with_the_b_axis_group():
    from backend.spherical_gpu.pseudosym import _sym_quats, same_orientation_angle_deg

    s = _sym_quats("2/m")
    two_fold = [row for row in s if abs(row[0]) < 1e-9]
    assert len(two_fold) >= 1
    assert all(abs(abs(row[2]) - 1.0) < 1e-9 for row in two_fold)    # about Y
    rng = np.random.default_rng(5)
    q = rng.normal(size=(6, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    twin = ROT_Y.data[0]
    from backend.spherical_gpu.pseudosym import _qmul
    q2 = _qmul(twin[None, :], q)
    for a, b in zip(q, q2):
        assert same_orientation_angle_deg(a[None, :], b, "2/m")[0] < 1e-3


def test_pseudosym_leaves_every_other_group_alone():
    from orix.quaternion.symmetry import _groups
    from backend.spherical_gpu.pseudosym import _sym_quats

    for name in ("m-3m", "m-3", "mmm", "4/mmm", "6/mmm", "-3m", "-1", "1", "432"):
        ref = next(g for g in _groups if g.name == name)
        assert np.array_equal(_sym_quats(name), np.asarray(ref.data, dtype=np.float64))


def test_dictionary_grid_covers_the_fundamental_zone_of_the_b_group():
    from orix.sampling import get_sample_fundamental
    from backend.dict_gpu.pipeline.grid import sample_orientations

    got = sample_orientations("2/m", 25.0)
    ref = get_sample_fundamental(25.0, point_group=frame_symmetry("2/m"))
    z = get_sample_fundamental(25.0, point_group=C2h)
    assert got.size == ref.size
    assert np.allclose(got.data, ref.data)
    assert not np.allclose(np.sort(got.data, axis=0), np.sort(z.data, axis=0)) or got.size != z.size


@pytest.mark.parametrize("sg, name", [(5, "121"), (8, "1m1"), (12, "2/m"), (15, "2/m"),
                                       (14, "2/m"), (3, "121"), (9, "1m1")])
def test_sht_point_group_names_carry_the_axis(sg, name):
    from backend.spherical_gpu.pipeline.sht_io import _SG_TO_POINT_GROUP

    assert _SG_TO_POINT_GROUP[sg] == name


def test_sht_point_group_table_is_otherwise_unchanged():
    from backend.spherical_gpu.pipeline.sht_io import _SG_TO_POINT_GROUP

    for n in range(1, 231):
        if cs.monoclinic_class_of_space_group(n) is None:
            assert _SG_TO_POINT_GROUP[n] == get_point_group(n).name or n in range(1, 3)


def test_hough_reflector_orbits_use_the_b_axis_group():
    """(h k l) is equivalent to (-h k -l) in a b-unique crystal, not (-h -k l)."""
    import hough_reflectors as hr
    from diffsims.crystallography import ReciprocalLatticeVector

    def orbit(phase):
        rlv = ReciprocalLatticeVector(phase, hkl=np.array([[1.0, 2.0, 3.0]]))
        rows, _ = hr._orbit_rows_of(phase, rlv)
        return {tuple(int(x) for x in r) for r in rows}

    lattice = Structure(lattice=LATTICE_B)
    assert orbit(FramePhase("p", space_group=12, structure=lattice)) == {
        (1, 2, 3), (-1, 2, -3), (-1, -2, -3), (1, -2, 3)}
    assert orbit(Phase("p", space_group=12, structure=lattice)) == {
        (1, 2, 3), (-1, -2, 3), (-1, -2, -3), (1, 2, -3)}


# ----- the guard: nobody builds a monoclinic-blind Phase any more ----------

_IMPORT = re.compile(r"from orix\.crystal_map import ([^\n#]+)")

#: Files that may import orix's ``Phase`` directly, and why that is safe.
_ALLOWED = {
    "crystal_symmetry.py": "defines FramePhase",
    "analysis/ebsd_dataset.py": "type annotation only; frames the xmap in __init__",
    "backend/api/routes/database.py": "reads CIF metadata, reduces no orientation",
    "backend/api/services/ang_export.py": "reads a CIF's lattice only",
    "backend/api/services/result_exporter.py": "builds the batch export map; "
                                               "_store_result frames stored results",
    "backend/dictionary_gpu/pipeline.py": "type annotation; the group comes from "
                                          "phase_point_group",
}


def _source_files():
    skip = {"tests", "tasks", "research", "docs", "scripts", "node_modules", "Database",
            "frontend", "electron", ".claude", "crystal-structures-for-ebsd-main",
            "Test_data", "dxa", "Linux_Maker", "branding", "logs", "licenses"}
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT)
        if rel.parts and rel.parts[0] in skip:
            continue
        yield p, rel.as_posix()


def test_no_module_imports_orix_phase_without_a_reason():
    offenders = []
    for path, rel in _source_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in _IMPORT.finditer(text):
            names = [n.strip() for n in m.group(1).split(",")]
            if any(n == "Phase" or n.startswith("Phase as ") for n in names):
                if rel not in _ALLOWED:
                    offenders.append(rel)
                    break
    assert not offenders, (
        "These modules import orix's Phase, whose point group for a monoclinic "
        "phase has its two-fold axis along Z. Import crystal_symmetry.FramePhase "
        "instead (or add the file to _ALLOWED with the reason): %s" % offenders)


# ----- the library: real CIF, real master, real indexer --------------------

_DB = Path(__file__).resolve().parents[1] / "Database"


def _library(*parts):
    base = Path(os.environ.get("ORIENTA_TEST_DB", str(_DB)))
    path = base.joinpath(*parts)
    if not path.is_file():
        pytest.skip(f"library file not available: {path}")
    return path


def test_library_cif_of_a_b_unique_phase_gets_the_y_axis():
    from ebsd_utils import sanitize_cif

    cif = _library("CIF_Library", "Al13Fe4.cif")
    phase = FramePhase.from_cif(sanitize_cif(str(cif)))
    assert phase.space_group.number == 12
    assert phase.point_group.proper_subgroup.name == "121"
    assert Phase.from_cif(sanitize_cif(str(cif))).point_group.proper_subgroup.name == "112"


def test_library_master_header_names_the_b_unique_group():
    from backend.spherical_gpu.pipeline.sht_io import read_sht_master

    found = sorted((_DB / "EBSD_SHT_Database" / "Al13Fe4").glob("*.sht"))
    if not found:
        pytest.skip("Al13Fe4 master not available")
    master = read_sht_master(str(found[0]), device="cpu")
    assert master.space_group == 12 and master.z_rot == 1
    assert master.point_group == "2/m"
    assert cs.unique_axis(frame_symmetry(master.point_group)) == "b"


def test_the_spherical_indexer_answers_in_the_b_unique_frame():
    """The premise of this whole module, measured: a rendered Al13Fe4 pattern
    comes back as the true orientation or as its image under the two-fold along
    Y -- not along Z. Sixteen random orientations through the real renderer and
    the real GPU indexer."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    found = sorted((_DB / "EBSD_SHT_Database" / "Al13Fe4").glob("*.sht"))
    if not found:
        pytest.skip("Al13Fe4 master not available")
    from orix.quaternion import Orientation
    from tests.test_spherical_gpu.test_m3_variant_selection import DET, _render
    from backend.spherical_gpu.backend import (BackendConfig, PhaseConfig,
                                               SphericalGPUBackend)

    rng = np.random.default_rng(7)
    quats = rng.normal(size=(16, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    pats = np.stack([_render(str(found[0]), q) for q in quats])
    backend = SphericalGPUBackend(BackendConfig(phases=[
        PhaseConfig(sht_file=str(found[0]), bandwidth=88, refine=True)]))
    res = backend.index_array(np.ascontiguousarray(pats, dtype=np.float32), DET)
    got = Rotation.from_euler(np.asarray(res.euler_xyz.numpy(), dtype=np.float64).reshape(-1, 3))
    truth = Rotation(quats)

    def within_2deg(sym):
        ang = Orientation(got, symmetry=sym).angle_with(Orientation(truth, symmetry=sym), degrees=True)
        return int((np.asarray(ang).ravel() < 2.0).sum())

    right, orix_group = within_2deg(frame_symmetry("2/m")), within_2deg(C2h)
    assert right >= 13, f"only {right}/16 match under the Y-axis group"
    assert orix_group <= right - 3, (
        f"Z-axis group matches {orix_group}/16, Y-axis group {right}/16: the "
        "indexer would no longer be answering in the b-unique frame")


# ----- exports: the axis-named point groups must still have a TSL / CTF code --

def _checkpoint_with_phase(tmp_path, point_group, space_group, lattice):
    import h5py
    from backend.api.services.checkpoint_writer import CheckpointWriter

    source = tmp_path / "Scan.h5oina"
    with h5py.File(source, "w") as f:
        f.create_dataset("Manufacturer", data=np.array([b"Oxford Instruments"]))
        h = f.create_group("1/EBSD/Header")
        h.create_dataset("Beam Voltage", data=np.float32(20.0))
        h.create_dataset("Tilt Angle", data=np.float32(np.deg2rad(70.0)))
        h.create_dataset("Scanning Rotation Angle", data=np.float32(np.pi))
        h.create_dataset("Working Distance", data=np.float32(16.5))
        f.create_group("1/EBSD/Data")
    cw = CheckpointWriter(str(source))
    cw.init_metadata(grid_shape=(3, 4), batch_id="mono", method="spherical",
                     step_size_um=0.4)
    meta = {"phase_file": "gone.sht", "ci_mean": 0.7, "duration_sec": 1.0,
            "point_group": point_group, "lattice_constants": lattice}
    if space_group is not None:
        meta["space_group"] = space_group
    rng = np.random.default_rng(1)
    cw.write_phase_result("Mono", np.full((3, 4), 0.7, np.float32),
                          rng.uniform(0, 3, (3, 4, 3)).astype(np.float32), meta)
    cw.compute_auto_assignment(confidence_threshold=0.0)
    return source, cw


@pytest.mark.parametrize("point_group, space_group", [
    ("121", None), ("1m1", None), ("2/m", None), ("2/m", 12), ("121", 5)])
def test_batch_ctf_and_ang_export_of_axis_named_monoclinic_phases(
        tmp_path, point_group, space_group):
    from backend.api.services.result_exporter import export_ang_ctf

    source, cw = _checkpoint_with_phase(
        tmp_path, point_group, space_group, [15.49, 8.08, 12.48, 90, 107.67, 90])
    ang, ctf = export_ang_ctf(cw.checkpoint_path, str(tmp_path / "o"),
                              source_h5_path=str(source))
    assert Path(ang).is_file() and Path(ctf).is_file()
    rows = Path(ctf).read_text(encoding="utf-8").splitlines()
    phase_row = rows[next(i for i, r in enumerate(rows) if r.startswith("Phases")) + 1]
    assert phase_row.split("\t")[1] == "2" or "\t2\t" in phase_row   # Oxford Laue class 2


def test_grain_segmentation_named_2m_uses_the_b_axis_group():
    from backend.api.services.grain_segmentation import segment_grains

    xmap = _twin_map(_plain_monoclinic_phase())
    eul = np.asarray(xmap.rotations.to_euler()).reshape(8, 8, 3)
    labels = segment_grains(eul, "2/m", threshold_deg=5.0)
    assert len(np.unique(labels)) == 1


def test_project_manager_reader_frames_loaded_xmaps(tmp_path):
    import project_manager
    from indexing_controller import IndexingMethod, IndexingResult

    xmap = _twin_map(_plain_monoclinic_phase())
    entry = project_manager.GalleryEntry(
        name="mono", method="Spherical", phase_data=None,
        indexing_result=IndexingResult(
            xmap=xmap, selection_mask=np.ones((8, 8), bool), original_shape=(8, 8),
            method=IndexingMethod.SPHERICAL))
    project_manager.save_project(str(tmp_path / "proj"), [entry], {})
    _meta, entries = project_manager.load_project(str(tmp_path / "proj"))
    loaded = entries[0].indexing_result.xmap
    assert loaded.phases[1].point_group.proper_subgroup.name == "121"


# ----- the guard for name -> symmetry tables -------------------------------

_Z_AXIS_ROUTES = re.compile(r"\bC2h\b|get_point_group|spacegroup2pointgroup")

#: Files that may name orix's Z-axis monoclinic group, and why that is safe.
_Z_AXIS_ALLOWED = {
    "backend/api/services/grain_segmentation.py":
        "name table; segment_grains swaps '2/m' for frame_symmetry()",
    "backend/dict_gpu/pipeline/grid.py":
        "name table; sample_orientations resolves monoclinic names with "
        "frame_symmetry() before it looks at the table",
}


def _code_lines_naming_the_z_group(text: str):
    for line in text.splitlines():
        code = line.split("#", 1)[0]
        if _Z_AXIS_ROUTES.search(code):
            yield line.strip()


def test_the_scan_for_z_axis_routes_sees_what_it_is_meant_to_see():
    assert list(_code_lines_naming_the_z_group(
        "from orix.quaternion.symmetry import C2h\n".replace("\n", "\n")))
    assert list(_code_lines_naming_the_z_group("pg = get_point_group(12)"))
    assert not list(_code_lines_naming_the_z_group("x = 1  # C2h is not code here"))


def test_nothing_else_routes_a_monoclinic_name_to_orixs_z_axis_group():
    offenders = {}
    for path, rel in _source_files():
        if rel == "crystal_symmetry.py" or rel in _Z_AXIS_ALLOWED:
            continue
        hits = list(_code_lines_naming_the_z_group(
            path.read_text(encoding="utf-8", errors="replace")))
        if hits:
            offenders[rel] = hits
    assert not offenders, (
        "These modules name orix's C2h / get_point_group, whose monoclinic groups "
        "have the two-fold axis along Z. Ask crystal_symmetry.frame_symmetry "
        "(or add the file to _Z_AXIS_ALLOWED with the reason): %s" % offenders)
