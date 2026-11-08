"""Trigonal phases whose two-fold axes or mirrors lie along Y, not along X.

Orienta's crystal frame is X || a, Z || c*. orix knows one point group per
trigonal class, the one with the two-fold axis (or mirror normal) along ``a``:
``D3`` "32", ``C3v`` "3m", ``D3d`` "-3m". The space groups 149, 151, 153 (312),
157, 159 (31m) and 162, 163 (-31m) have them 30 degrees further, perpendicular
to ``a``, along Y. ``crystal_symmetry`` hands those seven a group whose operators
and tag say which setting it is, named "312" (orix' name of ``D3y``), "3m" and
"-3m".

Every expectation here is computed from the operations of the diffpy space group
and a cell matrix built in this file, never from the code under test.
"""
from __future__ import annotations

import copy
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pytest
from diffpy.structure import Lattice, Structure
from diffpy.structure.spacegroups import GetSpaceGroup
from orix import io as orix_io
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.plot import IPFColorKeyTSL
from orix.quaternion import Orientation, Rotation, Symmetry
from orix.quaternion.symmetry import C3v, D3, D3d, D3y, get_point_group
from orix.sampling import get_sample_fundamental
from orix.vector import Vector3d
from scipy.spatial import cKDTree

import crystal_symmetry as cs
from crystal_symmetry import (
    FramePhase,
    frame_symmetry,
    phase_point_group,
    point_group_key,
)

TARGET = {149: "312", 151: "312", 153: "312", 157: "31m", 159: "31m",
          162: "-31m", 163: "-31m"}
NAME = {"312": "312", "31m": "3m", "-31m": "-3m"}
CONTROL = (150, 152, 156, 164, 166, 167)
HEX = Lattice(4.8, 4.8, 4.5, 90, 90, 120)


# ------------------------------------------------------------ the truth ---
def _cell_matrix(a, b, c, alpha, beta, gamma):
    """Columns a, b, c in the frame X || a, Z || c*."""
    al, be, ga = np.radians([alpha, beta, gamma])
    av = np.array([a, 0.0, 0.0])
    bv = np.array([b * np.cos(ga), b * np.sin(ga), 0.0])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    return np.stack([av, bv, np.array([cx, cy, np.sqrt(c * c - cx * cx - cy * cy)])],
                    axis=1)


def _key(m):
    return tuple(np.round(m, 6).ravel() + 0.0)


def _true_point_group(sg):
    """Cartesian 3x3 matrices of the point group of space group ``sg``."""
    m = _cell_matrix(4.8, 4.8, 4.5, 90, 90, 120)
    minv = np.linalg.inv(m)
    out = {}
    for op in GetSpaceGroup(int(sg)).iter_symops():
        r = m @ np.asarray(op.R, float) @ minv
        out[_key(r)] = r
    return out


def _matrices(sym):
    """Cartesian matrices of a symmetry (improper elements enter as -R)."""
    mats = np.asarray(sym.to_matrix())
    imp = np.asarray(sym.improper, dtype=bool)
    out = {}
    for mat, flip in zip(mats, imp):
        r = -mat if flip else mat
        out[_key(r)] = r
    return out


def _rot(axis, deg):
    return Rotation.from_axes_angles(axis, np.radians(deg))


def _orientations(sym, n=40, seed=3):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return Orientation(Rotation(q), symmetry=sym)


def _angle_deg(a, b):
    return np.degrees(a.angle_with(b))


GROUPS = {t: frame_symmetry(t) for t in ("312", "31m", "-31m")}


# ------------------------------------------------------------ operators ---
@pytest.mark.parametrize("sg", sorted(TARGET))
def test_operators_equal_those_of_the_diffpy_space_group(sg):
    got = frame_symmetry(space_group=sg)
    truth = _true_point_group(sg)
    assert set(_matrices(got)) == set(truth)
    assert got.size == len(truth)


@pytest.mark.parametrize("sg", CONTROL)
def test_controls_use_orix_group_and_the_same_conversion(sg):
    """The conversion is right if orix' own group matches it where orix is right."""
    got = frame_symmetry(space_group=sg)
    assert got is get_point_group(sg)
    assert set(_matrices(got)) == set(_true_point_group(sg))


@pytest.mark.parametrize("sg", sorted(TARGET))
def test_orix_group_of_the_seven_does_not_match(sg):
    """Why the groups exist: orix' group differs from the space group's."""
    assert set(_matrices(get_point_group(sg))) != set(_true_point_group(sg))


@pytest.mark.parametrize("sg", sorted(TARGET))
def test_laue_group_is_the_true_laue_group(sg):
    truth = _true_point_group(sg)
    laue = {}
    for k, r in truth.items():
        laue[k] = r
        laue[_key(-r)] = -r
    got = frame_symmetry(space_group=sg).laue
    assert set(_matrices(got)) == set(laue)
    assert got.name == "-3m"
    assert getattr(got, cs._TRIGONAL_TAG) == "-31m"


@pytest.mark.parametrize("sg, setting", sorted(TARGET.items()))
def test_names_are_orix_names_and_the_tag_names_the_setting(sg, setting):
    sym = frame_symmetry(space_group=sg)
    assert sym.name == NAME[setting]
    assert getattr(sym, cs._TRIGONAL_TAG) == setting
    assert sym.system == "trigonal"
    # proper subgroup 312 (orders 6) or 3: the Euler region of the rotations alone
    assert sym.euler_fundamental_region == ((360, 180, 120) if setting == "31m"
                                            else (360, 90, 120))


def test_proper_subgroup_of_minus_31m_is_orix_312():
    sg = GROUPS["-31m"]
    assert sg.proper_subgroup.name == "312"
    assert sg.proper_subgroup.size == 6
    assert GROUPS["312"].proper_subgroup.name == "312"
    assert GROUPS["31m"].proper_subgroup.name == "3"


# ----------------------------------------------------------- misorientation
@pytest.mark.parametrize("setting", ["312", "-31m"])
@pytest.mark.parametrize("axis", [(0, 1, 0), (np.cos(np.pi / 6), np.sin(np.pi / 6), 0),
                                  (np.cos(np.pi / 6), -np.sin(np.pi / 6), 0)])
def test_orientations_related_by_a_true_two_fold_are_equal(setting, axis):
    sym = GROUPS[setting]
    o = _orientations(sym)
    twin = Orientation(_rot(axis, 180) * Rotation(o), symmetry=sym)
    assert _angle_deg(o, twin).max() < 1e-4


def test_two_fold_of_the_31m_group_is_a_mirror_of_its_laue_group():
    sym = GROUPS["31m"]
    o = _orientations(sym.laue)
    twin = Orientation(_rot((0, 1, 0), 180) * Rotation(o), symmetry=sym.laue)
    assert _angle_deg(o, twin).max() < 1e-4


@pytest.mark.parametrize("setting", ["312", "-31m"])
def test_the_two_fold_along_x_is_not_a_symmetry_operation(setting):
    """orix' group has it; the true group has a rotation by 60 degrees there."""
    sym = GROUPS[setting]
    o = _orientations(sym)
    twin = Orientation(_rot((1, 0, 0), 180) * Rotation(o), symmetry=sym)
    assert np.allclose(_angle_deg(o, twin), 60.0, atol=1e-3)
    o_orix = Orientation(o.data, symmetry=D3d if setting == "-31m" else D3)
    twin_orix = Orientation(twin.data, symmetry=D3d if setting == "-31m" else D3)
    assert _angle_deg(o_orix, twin_orix).max() < 1e-4


def test_rotation_by_60_degrees_about_c_is_not_a_symmetry_of_minus_31m():
    sym = GROUPS["-31m"]
    o = _orientations(sym)
    turned = Orientation(_rot((0, 0, 1), 60) * Rotation(o), symmetry=sym)
    assert np.allclose(_angle_deg(o, turned), 60.0, atol=1e-3)


# ------------------------------------------------------ fundamental sector
@pytest.mark.parametrize("setting", ["312", "31m", "-31m"])
@pytest.mark.parametrize("which", ["group", "laue"])
def test_fundamental_sector_holds_exactly_one_image_of_every_direction(setting, which):
    sym = GROUPS[setting]
    sym = sym if which == "group" else sym.laue
    sector = sym.fundamental_sector
    rng = np.random.default_rng(4)
    v = Vector3d(rng.normal(size=(600, 3))).unit
    images = sym.outer(v)                       # (group size, 600)
    inside = np.array([np.asarray(sector >= images[i]).reshape(-1)
                       for i in range(images.shape[0])])
    count = inside.sum(axis=0)
    assert set(np.unique(count)) == {1}, np.bincount(count)


@pytest.mark.parametrize("setting", ["312", "31m", "-31m"])
@pytest.mark.parametrize("which", ["group", "laue"])
def test_in_fundamental_sector_maps_to_an_image_inside_the_sector(setting, which):
    """orix switches on the name here too ("32" takes a branch of its own)."""
    sym = GROUPS[setting]
    sym = sym if which == "group" else sym.laue
    rng = np.random.default_rng(5)
    v = Vector3d(rng.normal(size=(300, 3))).unit
    w = v.in_fundamental_sector(sym)
    assert np.asarray(w <= sym.fundamental_sector).all()
    images = sym.outer(v)
    dist = np.stack([np.linalg.norm(images[i].data - w.data, axis=-1)
                     for i in range(images.shape[0])])
    assert (dist.min(axis=0) < 1e-6).all()


def test_fundamental_sector_of_minus_31m_sits_between_mirrors_not_rotated():
    """Mirrors of -31m contain X (azimuth 0) and azimuth 60: the sector is that wedge."""
    sector = GROUPS["-31m"].fundamental_sector
    az = np.degrees(sector.vertices.azimuth) % 360
    on_equator = np.isclose(sector.vertices.polar, np.pi / 2)
    assert {round(a) for a in az[on_equator]} == {0, 60}


def test_orix_sector_of_its_own_minus_3m_is_the_one_we_do_not_use():
    az = np.degrees(D3d.fundamental_sector.vertices.azimuth) % 360
    on_equator = np.isclose(D3d.fundamental_sector.vertices.polar, np.pi / 2)
    assert {round(a) for a in az[on_equator]} == {30, 330}


# --------------------------------------------------------------- IPF colour
@pytest.mark.parametrize("setting", ["312", "31m", "-31m"])
def test_ipf_colour_is_the_same_for_symmetry_equivalent_orientations(setting):
    sym = GROUPS[setting]
    o = _orientations(sym, n=30)
    laue = sym.laue
    proper = [s for s in laue if not bool(s.improper)]
    assert len(proper) == 6
    for d in (Vector3d.xvector(), Vector3d.yvector(), Vector3d.zvector()):
        key = IPFColorKeyTSL(sym, direction=d)
        ref = key.orientation2color(o)
        for s in proper:
            eq = Orientation(Rotation(s) * Rotation(o), symmetry=sym)
            assert np.abs(key.orientation2color(eq) - ref).max() < 1e-6


def test_ipf_colour_differs_for_the_two_fold_along_x():
    """Sanity of the test above: a rotation that is no symmetry changes the colour."""
    sym = GROUPS["-31m"]
    o = _orientations(sym, n=30)
    twin = Orientation(_rot((1, 0, 0), 180) * Rotation(o), symmetry=sym)
    key = IPFColorKeyTSL(sym, direction=Vector3d.zvector())
    assert np.abs(key.orientation2color(o) - key.orientation2color(twin)).max() > 0.05


# ----------------------------------------------------------------- orix H5
def _xmap(phase):
    return CrystalMap(
        rotations=Rotation.identity((4,)), phase_id=np.ones(4, int),
        x=np.arange(4.0), y=np.zeros(4),
        phase_list=PhaseList(phases=[phase], ids=[1]),
    )


def test_orix_h5_roundtrip_with_a_space_group_returns_the_marked_group(tmp_path):
    ph = FramePhase("Cr2N", space_group=162, structure=Structure(lattice=HEX))
    path = tmp_path / "x.h5"
    orix_io.save(str(path), _xmap(ph))
    back = orix_io.load(str(path))
    assert back.phases[1].point_group.name == "-3m"
    cs.frame_xmap(back)
    assert back.phases[1].point_group is frame_symmetry("-31m")
    assert back.phases[1].space_group.number == 162


@pytest.mark.parametrize("sg, setting", sorted(TARGET.items()))
def test_orix_h5_roundtrip_returns_the_operators_of_every_one_of_the_seven(
        tmp_path, sg, setting):
    """The path of ``orix.io.save``/``load`` and the one of ``project_manager``."""
    ph = FramePhase("p", space_group=sg, structure=Structure(lattice=HEX))
    truth = set(_true_point_group(sg))
    path = tmp_path / "x.h5"
    orix_io.save(str(path), _xmap(ph))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        back = orix_io.load(str(path))
    _assert_back_is_the_marked_group(back, sg, setting, truth, caught)

    import project_manager
    from indexing_controller import IndexingMethod, IndexingResult

    entry = project_manager.GalleryEntry(
        name="e", method="Spherical", phase_data=None,
        indexing_result=IndexingResult(
            xmap=_xmap(ph), selection_mask=np.ones((2, 2), bool),
            original_shape=(2, 2), method=IndexingMethod.SPHERICAL))
    project_manager.save_project(str(tmp_path / "proj"), [entry])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _meta, entries = project_manager.load_project(str(tmp_path / "proj"))
    _assert_back_is_the_marked_group(
        entries[0].indexing_result.xmap, sg, setting, truth, caught)


def _assert_back_is_the_marked_group(back, sg, setting, truth, caught):
    warned = [w for w in caught if "Setting space group to 'None'" in str(w.message)]
    if setting == "312":
        # orix reads "312" as D3y, but the space group derives "32": it drops it.
        assert len(warned) == 1
        assert back.phases[1].space_group is None
    else:
        assert warned == []
        assert back.phases[1].space_group.number == sg
    cs.frame_xmap(back)
    got = back.phases[1].point_group
    assert got is GROUPS[setting]
    assert point_group_key(got) == setting
    assert set(_matrices(got)) == truth
    assert set(_matrices(got.laue)) >= truth


def test_orix_h5_forgets_the_tag_of_a_phase_without_a_space_group(tmp_path):
    """Known limit: orix writes only ``point_group.name``, which is "-3m" for both
    settings, and a phase without a space group has nothing else to restore it
    from. The caller that saves such a phase has to carry the setting itself."""
    ph = FramePhase("x", point_group="-31m", structure=Structure(lattice=HEX))
    assert ph.point_group is frame_symmetry("-31m")
    path = tmp_path / "y.h5"
    orix_io.save(str(path), _xmap(ph))
    back = cs.frame_xmap(orix_io.load(str(path)))
    got = back.phases[1].point_group
    assert got.name == "-3m"
    assert getattr(got, cs._TRIGONAL_TAG, None) is None
    assert got == D3d                      # orix' group, not the one with the tag


# ------------------------------------------------------------------ copies
@pytest.mark.parametrize("setting", ["312", "31m", "-31m"])
def test_copies_pickles_and_slices_neither_crash_nor_lose_the_group(setting):
    sym = GROUPS[setting]
    for dup in (copy.deepcopy(sym), copy.copy(sym), pickle.loads(pickle.dumps(sym))):
        assert getattr(dup, cs._TRIGONAL_TAG) == setting
        assert dup.name == sym.name
        assert set(_matrices(dup)) == set(_matrices(sym))
        assert point_group_key(dup) == setting
        assert getattr(dup.laue, cs._TRIGONAL_TAG) == "-31m"
    # Copies orix makes without knowing the tag: nothing may raise.
    for new in (sym[:3], sym[sym.improper], sym[::-1], Symmetry(sym), sym.flatten(),
                sym.unique()):
        new.laue
        new.fundamental_sector
        new.proper_subgroup
        new.system


def test_slice_of_a_marked_group_is_a_plain_group_without_the_tag():
    part = GROUPS["-31m"][:3]
    assert getattr(part, cs._TRIGONAL_TAG, None) is None


def test_a_marked_group_survives_a_process_boundary():
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        keys = [pool.submit(point_group_key, GROUPS[s]).result(timeout=120)
                for s in ("312", "31m", "-31m")]
    assert keys == ["312", "31m", "-31m"]


def test_phase_deepcopy_and_framephase_deepcopy_keep_the_group():
    ph = FramePhase("p", space_group=162, structure=Structure(lattice=HEX))
    for dup in (ph.deepcopy(), copy.deepcopy(ph)):
        assert dup.point_group is frame_symmetry("-31m")
    ph2 = FramePhase("p", point_group="31m", structure=Structure(lattice=HEX))
    for dup in (ph2.deepcopy(), copy.deepcopy(ph2)):
        assert point_group_key(dup.point_group) == "31m"
        assert getattr(dup.point_group, cs._TRIGONAL_TAG) == "31m"


def test_framephase_survives_the_orix_containers():
    ph = FramePhase("Cr2N", space_group=162, structure=Structure(lattice=HEX))
    xmap = _xmap(ph)
    for obj in (copy.deepcopy(xmap), xmap.deepcopy(), xmap[xmap.phase_id == 1]):
        assert point_group_key(obj.phases[1].point_group) == "-31m"
    assert point_group_key(xmap.phases_in_data[1].point_group) == "-31m"
    assert point_group_key(xmap[xmap.phase_id == 1].orientations.symmetry) == "-31m"


# ---------------------------------------------------------- frame_symmetry
@pytest.mark.parametrize("sg, setting", sorted(TARGET.items()))
def test_frame_symmetry_by_space_group_returns_the_marked_object(sg, setting):
    got = frame_symmetry(space_group=sg)
    assert got is GROUPS[setting]
    assert frame_symmetry(space_group=str(sg)) is got


@pytest.mark.parametrize("sg", sorted(TARGET))
def test_alternative_setting_numbers_are_not_the_seven(sg):
    """1162 and 2149 are no diffpy space groups: they behave as before the groups
    existed (orix' lookup), never as the marked group."""
    for alt in (sg + 1000, sg + 2000):
        try:
            got = frame_symmetry(space_group=alt)
        except Exception:
            continue                                  # failing as before is fine
        assert getattr(got, cs._TRIGONAL_TAG, None) is None
        assert got is get_point_group(alt)


@pytest.mark.parametrize("text", ["312", "31m", "-31m"])
def test_frame_symmetry_by_text_returns_the_same_objects(text):
    got = frame_symmetry(text)
    assert got is GROUPS[text]
    assert frame_symmetry(f"  {text} ") is got
    assert frame_symmetry(got) is got                       # already marked
    assert frame_symmetry(point_group=got) is got


@pytest.mark.parametrize("sg, setting", sorted(TARGET.items()))
def test_the_group_of_a_text_equals_the_group_of_its_space_groups(sg, setting):
    assert frame_symmetry(setting) is frame_symmetry(space_group=sg)


@pytest.mark.parametrize("name", ["32", "3m", "-3m", "321", "3m1", "-3m1", "3", "-3"])
def test_the_other_trigonal_names_are_unchanged(name):
    assert frame_symmetry(name) == name


def test_orix_objects_pass_through_unchanged():
    for sym in (D3, C3v, D3d):
        assert frame_symmetry(sym) is sym


def test_plain_orix_312_becomes_the_marked_312_group():
    """D3y has the Y two-folds; unmarked, orix picks its sector 30 degrees off."""
    assert frame_symmetry(D3y) is GROUPS["312"]
    assert point_group_key(frame_symmetry(D3y)) == "312"


def test_a_group_that_only_carries_the_name_312_is_left_alone():
    fake = Symmetry(D3)                       # D3's operators (two-folds along X)
    fake.name = "312"
    assert frame_symmetry(fake) is fake
    other = Symmetry(D3y)[:3]
    other.name = "312"
    assert frame_symmetry(other) is other


def test_space_group_decides_over_a_text():
    assert frame_symmetry("31m", space_group=150) is get_point_group(150)
    assert frame_symmetry("-3m", space_group=162) is GROUPS["-31m"]


def test_marked_group_is_not_orix_group_of_the_same_name():
    for setting, name in (("31m", "3m"), ("-31m", "-3m")):
        sym = GROUPS[setting]
        ref = {"3m": C3v, "-3m": D3d}[name]
        assert sym.name == ref.name
        assert sym != ref
        assert hash(sym) != hash(ref)


def test_the_312_group_is_named_after_its_setting_not_after_d3():
    """"32" is orix' name of D3 (two-folds along X); the Y setting has its own name."""
    sym = GROUPS["312"]
    assert sym.name == "312" == D3y.name
    assert sym.name != D3.name
    assert sym != D3
    assert set(_matrices(sym)) == set(_matrices(D3y)) != set(_matrices(D3))


# --------------------------------------------------------- point_group_key
def test_point_group_key():
    assert point_group_key(GROUPS["-31m"]) == "-31m"
    assert point_group_key(GROUPS["31m"]) == "31m"
    assert point_group_key(GROUPS["312"]) == "312"
    assert point_group_key(D3d) == "-3m"
    assert point_group_key(D3) == "32"
    assert point_group_key("-3m") == "-3m"
    assert point_group_key("  m-3m ") == "m-3m"
    assert point_group_key(get_point_group(225)) == "m-3m"
    assert point_group_key(None) == ""


@pytest.mark.parametrize("setting", ["312", "31m", "-31m"])
def test_the_key_goes_through_frame_symmetry_and_back(setting):
    assert frame_symmetry(point_group_key(GROUPS[setting])) is GROUPS[setting]


# --------------------------------------------------------------- FramePhase
@pytest.mark.parametrize("sg, setting", sorted(TARGET.items()))
def test_framephase_with_a_space_group_gets_the_marked_group(sg, setting):
    ph = FramePhase("p", space_group=sg, structure=Structure(lattice=HEX))
    assert ph.point_group is GROUPS[setting]
    assert phase_point_group(ph) is GROUPS[setting]
    assert Phase("p", space_group=sg, structure=Structure(lattice=HEX)).point_group \
        is get_point_group(sg)
    assert phase_point_group(Phase("p", space_group=sg)) is GROUPS[setting]


@pytest.mark.parametrize("text", ["312", "31m", "-31m"])
def test_framephase_setter_takes_the_three_texts(text):
    ph = FramePhase("p", point_group=text, structure=Structure(lattice=HEX))
    assert ph.point_group is GROUPS[text]
    assert phase_point_group(ph) is GROUPS[text]
    other = FramePhase("q", structure=Structure(lattice=HEX))
    other.point_group = text
    assert other.point_group is GROUPS[text]
    assert other.point_group.name == NAME[text]


def test_framephase_setter_takes_a_marked_object():
    ph = FramePhase("p", structure=Structure(lattice=HEX))
    ph.point_group = GROUPS["31m"]
    assert ph.point_group is GROUPS["31m"]


def test_framephase_setter_with_a_matching_space_group_does_not_warn():
    ph = FramePhase("p", space_group=157, structure=Structure(lattice=HEX))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ph.point_group = "31m"
    assert ph.space_group.number == 157
    assert ph.point_group is GROUPS["31m"]


def test_framephase_setter_with_another_group_drops_the_space_group_as_orix_does():
    ph = FramePhase("p", space_group=157, structure=Structure(lattice=HEX))
    with pytest.warns(UserWarning):
        ph.point_group = "-31m"
    assert ph.space_group is None
    assert ph.point_group is GROUPS["-31m"]


def test_framephase_orix_names_still_mean_orix_groups():
    assert FramePhase("p", point_group="-3m").point_group is D3d
    # (orix reads the text "32" as the TSL alias of "321", which has D3's operators.)
    assert FramePhase("p", point_group="32").point_group == D3
    assert FramePhase("p", point_group="3m").point_group is C3v


def test_framephase_other_groups_are_untouched():
    assert FramePhase("Al", space_group=225).point_group is get_point_group(225)
    assert FramePhase("p", space_group=150).point_group is get_point_group(150)
    assert FramePhase("p", space_group=187).point_group is get_point_group(187)
    assert FramePhase("p", space_group=12,
                      structure=Structure(lattice=Lattice(15.49, 8.08, 12.48, 90, 107.67, 90))
                      ).point_group.proper_subgroup.name == "121"


def test_frame_xmap_repairs_a_plain_orix_phase_with_one_of_the_seven():
    plain = PhaseList(phases=[Phase("Cr2N", space_group=162,
                                    structure=Structure(lattice=HEX))], ids=[1])
    xmap = CrystalMap(rotations=Rotation.identity((3,)), phase_id=np.ones(3, int),
                      x=np.arange(3.0), y=np.zeros(3), phase_list=plain)
    assert xmap.phases[1].point_group is D3d
    cs.frame_xmap(xmap)
    assert xmap.phases[1].point_group is GROUPS["-31m"]


# --------------------------------------------------------- one source of truth
def test_the_space_group_set_is_one_list():
    import phase_cell

    assert set(TARGET) == set(cs.TRIGONAL_Y_SETTING_SPACE_GROUPS)
    assert phase_cell.PER_REFLECTION_SPACE_GROUPS == frozenset(TARGET)
    assert phase_cell.PER_REFLECTION_SPACE_GROUPS is cs.TRIGONAL_Y_SETTING_SPACE_GROUPS


# ------------------------------------------------------------------ sampling
@pytest.mark.parametrize("setting, sister", [("312", D3), ("31m", C3v), ("-31m", D3d)])
def test_get_sample_fundamental_runs_and_is_as_large_as_orix_sister(setting, sister):
    ours = get_sample_fundamental(8, point_group=GROUPS[setting])
    theirs = get_sample_fundamental(8, point_group=sister)
    assert ours.size > 0
    assert abs(ours.size - theirs.size) <= 0.05 * theirs.size


def test_sample_orientations_cover_a_minus_31m_group_in_the_y_setting():
    """The grid of the true group reaches every orientation within one step."""
    from backend.dict_gpu.pipeline.grid import sample_orientations

    sym = GROUPS["-31m"]
    q = np.asarray(sample_orientations(sym, 8.0).data)[:, :4]
    rng = np.random.default_rng(7)
    probe = rng.normal(size=(1500, 4))
    probe /= np.linalg.norm(probe, axis=1, keepdims=True)
    tree = cKDTree(np.vstack([q, -q]))
    best = np.full(len(probe), np.inf)
    for s in sym.laue:
        if not bool(s.improper):
            d, _ = tree.query(_left(np.asarray(s.data).reshape(4), probe))
            best = np.minimum(best, d)
    ang = np.degrees(4 * np.arcsin(np.clip(best / 2, 0, 1)))
    assert (ang > 8.0).mean() < 0.02


def _left(s, probe):
    sw, sx, sy, sz = s
    pw, px, py, pz = probe.T
    return np.stack([sw * pw - sx * px - sy * py - sz * pz,
                     sw * px + sx * pw + sy * pz - sz * py,
                     sw * py - sx * pz + sy * pw + sz * px,
                     sw * pz + sx * py - sy * px + sz * pw], axis=-1)
