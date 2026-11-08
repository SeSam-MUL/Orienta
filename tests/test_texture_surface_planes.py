"""Surface-plane fractions: the script's number (one direction per pixel) and the share with any plane of the family.

``TextureAnalysis_mtex6.m`` lines 441-523 count ``angle(ori * Miller(h,k,l,CS), vector3d.Z) < 15 deg``: ONE specimen
vector per pixel, the angle to +Z not antipodal. The reference angles below were measured with MTEX for Python on
orientations built as ``orientation.byMiller(hkl, uvw)`` (hkl parallel to ND = Z, uvw parallel to RD = X); the
orientations here are built from an explicit rotation matrix and scipy, independently of the code under test.
"""
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("orix")

import analysis.texture_analysis as ta                                           # noqa: E402

# byMiller(hkl, uvw) -> angle in degrees of Miller (1,1,1) / (1,0,0) / (1,1,0) to Z (MTEX for Python)
TABLE = [
    ((1, 1, 1), (1, 1, -2), (0.0, 54.7356, 35.2644)),
    ((-1, 1, 1), (1, 1, 0), (70.5288, 125.2644, 90.0)),
    ((1, -1, 1), (1, 1, 0), (70.5288, 54.7356, 90.0)),
    ((1, 1, -1), (-1, 1, 0), (70.5288, 54.7356, 35.2644)),
    ((0, 0, 1), (1, 0, 0), (54.7356, 90.0, 90.0)),
    ((1, 0, 0), (0, 1, 1), (54.7356, 0.0, 45.0)),
    ((0, 1, 1), (1, 0, 0), (35.2644, 90.0, 60.0)),
    ((-1, -1, -1), (1, 1, -2), (180.0, 125.2644, 144.7356)),
]
FAMILIES = ([1, 1, 1], [1, 0, 0], [1, 1, 0])


def orix_quaternion(hkl, uvw) -> np.ndarray:
    """orix quaternion (w, x, y, z) of the orientation with hkl along Z and uvw along X.

    The MTEX rotation is the matrix that takes the crystal frame to the specimen frame; its rows are the crystal
    axes a = uvw, b = c x a, c = hkl (unit). An orix quaternion is the conjugate of the MTEX rotation.
    """
    from scipy.spatial.transform import Rotation
    a = np.asarray(uvw, float) / np.linalg.norm(uvw)
    c = np.asarray(hkl, float) / np.linalg.norm(hkl)
    b = np.cross(c, a)
    m = np.stack([a, b, c])                                         # specimen = m @ crystal
    assert np.allclose(m @ m.T, np.eye(3)) and np.isclose(np.linalg.det(m), 1.0)
    x, y, z, w = Rotation.from_matrix(m).as_quat()                  # scipy: (x, y, z, w)
    return np.array([w, -x, -y, -z])                                # conjugate


def dataset_of(quats, point_group="m-3m", phase_ids=None, names=("Al",)):
    """A one-row map of the given orix quaternions (one pixel each)."""
    from orix.crystal_map import CrystalMap, PhaseList
    from orix.quaternion import Rotation
    from analysis.ebsd_dataset import EBSDDataset
    q = np.asarray(quats, float).reshape(-1, 4)
    n = len(q)
    ids = np.zeros(n, int) if phase_ids is None else np.asarray(phase_ids)
    xmap = CrystalMap(rotations=Rotation(q), phase_id=ids, x=np.arange(n, dtype=float), y=np.zeros(n),
                      phase_list=PhaseList(names=list(names), point_groups=[point_group] * len(names)),
                      prop={"bc": np.full(n, 150.0)})
    return EBSDDataset(xmap, step_size=1.0)


def script(ds, family, tol, **kw):
    return ta.calculate_surface_plane_fraction(ds, family, tol, **kw)


def family(ds, fam, tol, **kw):
    return ta.calculate_surface_plane_family_fraction(ds, fam, tol, **kw)


# ------------------------------------------------------------------------------------------------------------------
# the script's number against the MTEX table
# ------------------------------------------------------------------------------------------------------------------
def test_each_orientation_alone_gives_its_table_angle_at_any_tolerance_that_separates_it():
    angles = np.array([row[2] for row in TABLE])                    # (8, 3)
    for hkl, uvw, expected in TABLE:
        ds = dataset_of([orix_quaternion(hkl, uvw)])
        for col, fam in enumerate(FAMILIES):
            for tol in (15, 36, 40, 46, 50, 55, 59, 61, 71, 89, 91, 126, 145, 181):
                want = 100.0 if expected[col] < tol else 0.0
                assert script(ds, fam, tol) == want, (hkl, uvw, fam, tol, expected[col])
    assert angles.shape == (8, 3)


def test_the_eight_orientations_together_give_the_table_shares_at_tolerances_that_pin_every_angle():
    ds = dataset_of([orix_quaternion(h, u) for h, u, _ in TABLE])
    angles = np.array([row[2] for row in TABLE])
    for tol in (15, 36, 40, 46, 50, 55, 59, 61, 71, 89, 91, 126, 145, 181):
        want = (angles < tol).mean(axis=0) * 100.0
        got = [script(ds, fam, tol) for fam in FAMILIES]
        assert got == pytest.approx(list(want), abs=1e-9), tol
    # the stated 15 deg case: 1 of 8 for {111} and {100}, none for {110}
    assert [script(ds, fam, 15.0) for fam in FAMILIES] == [12.5, 12.5, 0.0]


def test_the_angle_to_z_is_not_antipodal():
    # (111) of the last table row points to -Z: 180 deg, not 0
    ds = dataset_of([orix_quaternion((-1, -1, -1), (1, 1, -2))])
    assert script(ds, [1, 1, 1], 15.0) == 0.0
    assert family(ds, [1, 1, 1], 15.0) == 100.0                     # the same plane, seen from the other side


@pytest.mark.parametrize("point_group", ["m-3", "23", "m-3m", "432"])
def test_a_plane_and_its_negative_are_one_plane_whatever_the_proper_group(point_group):
    # specimen Z along -(111) in the crystal frame. The proper groups of m-3 and 23 (12 rotations) have no operator
    # that takes a normal to its negative, so only the "+/- is one plane" rule puts this pixel into the {111} family.
    ds = dataset_of([orix_quaternion((-1, -1, -1), (1, 1, -2))], point_group=point_group)
    assert script(ds, [1, 1, 1], 15.0) == 0.0                       # 180 deg from +Z
    assert family(ds, [1, 1, 1], 15.0) == 100.0


# ------------------------------------------------------------------------------------------------------------------
# ideal components
# ------------------------------------------------------------------------------------------------------------------
def test_shear2_has_a_111_plane_in_the_surface_and_no_100_or_110():
    ds = dataset_of([orix_quaternion((1, 1, 1), (1, 1, -2))])       # Shear 2 {111}<112>
    assert script(ds, [1, 1, 1], 15.0) == 100.0
    assert family(ds, [1, 1, 1], 15.0) == 100.0
    assert family(ds, [1, 0, 0], 15.0) == 0.0
    assert family(ds, [1, 1, 0], 15.0) == 0.0


def test_rot_cube_nd45_has_a_100_plane_in_the_surface():
    ds = dataset_of([orix_quaternion((1, 0, 0), (0, 1, 1))])        # Rot.Cube ND45 {100}<011>
    assert script(ds, [1, 0, 0], 15.0) == 100.0
    assert family(ds, [1, 0, 0], 15.0) == 100.0
    assert family(ds, [1, 1, 1], 15.0) == 0.0


def test_cube_script_100_is_zero_because_the_direction_lies_along_x_but_the_family_is_complete():
    ds = dataset_of([[1.0, 0.0, 0.0, 0.0]])                         # identity: crystal axes = specimen axes
    assert script(ds, [1, 0, 0], 15.0) == 0.0                       # the direction (1,0,0) is along X
    assert family(ds, [1, 0, 0], 15.0) == 100.0                     # (0,0,1) is the surface normal
    assert script(ds, [0, 0, 1], 15.0) == 100.0


def test_the_family_is_symmetric_equivalent_invariant_the_script_number_is_not():
    # orix: s * q is the same crystal orientation; here s = a 90 deg turn about the crystal x axis
    q = orix_quaternion((1, 1, 1), (1, 1, -2))
    s = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
    from analysis.orientation_math import qmul
    q2 = qmul(s, q)
    for fam in FAMILIES:
        assert family(dataset_of([q]), fam, 15.0) == family(dataset_of([q2]), fam, 15.0)
    assert script(dataset_of([q]), [1, 1, 1], 15.0) != script(dataset_of([q2]), [1, 1, 1], 15.0)


def test_other_phases_and_not_indexed_pixels_are_not_counted_and_not_in_the_denominator():
    cube = [1.0, 0.0, 0.0, 0.0]
    shear = orix_quaternion((1, 1, 1), (1, 1, -2))
    # phase 0: 1 cube + 1 shear; phase 1: three shear pixels that would triple {111}; two not indexed
    ds = dataset_of([cube, shear] + [shear] * 3 + [cube] * 2, phase_ids=[0, 0, 1, 1, 1, -1, -1], names=("Al", "Other"))
    assert script(ds, [1, 1, 1], 15.0, texture_phase=0) == 50.0
    assert family(ds, [1, 1, 1], 15.0, texture_phase=0) == 50.0
    # the default is the phase with the most indexed pixels: phase 1, three shear pixels
    assert script(ds, [1, 1, 1], 15.0) == 100.0


def test_nan_orientation_is_in_the_denominator_and_in_no_count(monkeypatch):
    good = orix_quaternion((1, 1, 1), (1, 1, -2))
    ds = dataset_of([good, good])
    monkeypatch.setattr(ta, "_phase_quaternions", lambda _ds, _pid: np.array([good, [np.nan] * 4]))
    assert script(ds, [1, 1, 1], 15.0) == 50.0
    assert family(ds, [1, 1, 1], 15.0) == 50.0


def test_analyze_texture_fills_the_family_fields_next_to_the_script_numbers():
    from analysis.texture_components import get_texture_preset
    ds = dataset_of([orix_quaternion((1, 1, 1), (1, 1, -2))] * 4)
    res = ta.analyze_texture(ds, {"Goss": get_texture_preset("FCC_Rolling")["Goss"]})
    assert res.plane_111_fraction == 100.0 and res.plane_111_family_fraction == 100.0
    assert res.plane_100_family_fraction == 0.0 and res.plane_110_family_fraction == 0.0


# ------------------------------------------------------------------------------------------------------------------
# the workbook
# ------------------------------------------------------------------------------------------------------------------
def test_sheet1_has_the_script_rows_and_the_three_family_rows(tmp_path):
    from analysis.excel_exporter import ExcelExporter, PLANE_FAMILY_LABELS, PLANE_LABELS, RX_CRITERIA_KEYS
    from analysis.texture_components import get_texture_preset
    # cube: script {100} = 0 % (the direction (1,0,0) lies along X), family {100} = 100 %; swapped groups read wrong
    ds = dataset_of([[1.0, 0.0, 0.0, 0.0]] * 4)
    res = ta.analyze_texture(ds, {"Goss": get_texture_preset("FCC_Rolling")["Goss"]})
    ex = ExcelExporter(str(tmp_path / "d.xlsx"))
    ex.export_dataset("ds", {"basic_info": {}, "texture": res})
    ws = ex.wb[ex.wb.sheetnames[0]]
    assert [ws.cell(row=r, column=1).value for r in (45, 46, 47)] == list(PLANE_LABELS)
    assert all("one direction, as the script" in label for label in PLANE_LABELS)
    assert [ws.cell(row=r, column=1).value for r in (64, 65, 66)] == list(PLANE_FAMILY_LABELS)
    assert all("any plane of the family" in label for label in PLANE_FAMILY_LABELS)
    assert ws.cell(row=60 + len(RX_CRITERIA_KEYS) - 1, column=1).value == RX_CRITERIA_KEYS[-1]     # rows 60-63 untouched
    assert [ws.cell(row=r, column=2).value for r in (45, 46, 47)] == [0.0, 0.0, 0.0]
    assert [ws.cell(row=r, column=2).value for r in (64, 65, 66)] == [0.0, 100.0, 0.0]


def test_sheet1_without_family_fields_leaves_the_family_rows_empty(tmp_path):
    from types import SimpleNamespace
    from analysis.excel_exporter import ExcelExporter
    old = SimpleNamespace(texture_index=1.0, entropy=1.0, component_fractions={}, plane_111_fraction=1.0,
                          plane_100_fraction=2.0, plane_110_fraction=3.0)
    ex = ExcelExporter(str(tmp_path / "d.xlsx"))
    ex.export_dataset("ds", {"basic_info": {}, "texture": old})
    ws = ex.wb[ex.wb.sheetnames[0]]
    assert [ws.cell(row=r, column=2).value for r in (45, 46, 47)] == [1.0, 2.0, 3.0]
    assert [ws.cell(row=r, column=2).value for r in (64, 65, 66)] == [None, None, None]


# ------------------------------------------------------------------------------------------------------------------
# real data: the script's numbers equal MTEX's on Orienta's own orientations
# ------------------------------------------------------------------------------------------------------------------
REAL = {
    "sampleB": ("EBSD_SampleB_extrusion_withPattern Sample_B Arbeitsbereich 1 Elementverteilungsdaten 1.h5oina",
                8593, (4.457116, 0.0, 0.0)),
    "g114": ("ground_truth/G1_14.h5oina", 169745, (8.646499, 0.0, 0.0)),
}


@pytest.mark.slow
@pytest.mark.parametrize("name", list(REAL))
def test_script_numbers_equal_mtex_on_real_maps(name):
    from analysis.ebsd_dataset import EBSDDataset
    from analysis.xmap_loader import load_xmap_file
    from tests.data_deps import TEST_DATA_DIR, first_missing
    base = Path(os.environ.get("ORIENTA_TEST_DATA") or TEST_DATA_DIR)
    fname, n_px, want = REAL[name]
    path = base / fname
    if first_missing(path) is not None:
        pytest.skip(f"needs {path} (set ORIENTA_TEST_DATA to a Test_data directory that has it)")
    xmap, step, _meta = load_xmap_file(str(path))
    ds = EBSDDataset(xmap, step_size=step)
    assert int((np.asarray(ds.xmap.phase_id) >= 0).sum()) == n_px                  # the pixels of the dump: all indexed pixels
    got = [script(ds, fam, 15.0) for fam in FAMILIES]
    assert got == pytest.approx(list(want), abs=1e-6)
    # the family fraction is at least the single direction's and a share
    for fam, single in zip(FAMILIES, got):
        assert single <= family(ds, fam, 15.0) <= 100.0
