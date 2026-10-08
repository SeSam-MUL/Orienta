"""The point group of a monoclinic phase must carry its real two-fold axis.

Orienta's crystal frame is X||a, Z||c*. A b-unique monoclinic crystal has its
two-fold axis along Y, but orix only ships the unique-axis-c group. These tests
pin that ``crystal_symmetry`` hands out the right group, that nothing but
monoclinic phases changes, and that the group survives the orix containers
phases travel in.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
from diffpy.structure import Lattice, Structure
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.plot import IPFColorKeyTSL
from orix.quaternion import Orientation, Rotation
from orix.quaternion.symmetry import C2h, get_point_group
from orix.vector import Vector3d

import crystal_symmetry as cs
from crystal_symmetry import FramePhase, frame_symmetry, unique_axis

#: Cell of Al13Fe4 (C2/m, b-unique): only beta differs from 90 degrees.
LATTICE_B = Lattice(15.49, 8.08, 12.48, 90, 107.67, 90)


def _random_orientations(sym, n=40, seed=3):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return Orientation(Rotation(q), symmetry=sym)


def _equivalents(o, sym):
    """Every orientation the symmetry says is the same crystal as ``o``."""
    return [Orientation(Rotation(s) * Rotation(o), symmetry=sym)
            for s in sym if np.linalg.norm(np.asarray(s.data)[..., 1:]) > 0.5
            or bool(s.improper)]


@pytest.mark.parametrize("sg, axis", [(12, "b"), (15, "b"), (14, "b"), (13, "b"),
                                       (4015, "b"), (2012, "b"), (1012, "c")])
def test_unique_axis_comes_from_the_space_group(sg, axis):
    sym = frame_symmetry(space_group=sg)
    assert unique_axis(sym) == axis
    assert sym.name == "2/m"
    assert sym.laue.name == "2/m"


@pytest.mark.parametrize("sg, cls, name", [(5, "2", "121"), (8, "m", "1m1")])
def test_non_centrosymmetric_classes_get_orix_axis_names(sg, cls, name):
    sym = frame_symmetry(space_group=sg)
    assert sym.name == name
    assert sym.laue.name == "2/m"
    assert unique_axis(sym) == "b"


def test_the_b_group_is_the_one_with_the_two_fold_along_y():
    sym = frame_symmetry(space_group=12)
    assert sym.proper_subgroup.name == "121"
    assert sym.size == 4 and sym.contains_inversion
    y = Vector3d.yvector()
    axes = sym[sym.angle > 0.1]
    two_fold = [a for a in axes if not bool(a.improper)]
    assert len(two_fold) == 1
    assert np.allclose(np.abs(two_fold[0].axis.data), y.data)
    # The orix group it replaces has its axis along z.
    assert C2h.proper_subgroup.name == "112"


def test_b_unique_orientations_related_by_the_two_fold_are_equal():
    sym = frame_symmetry(space_group=12)
    o = _random_orientations(sym)
    rot_y = Rotation.from_axes_angles([0, 1, 0], np.pi)
    twin = Orientation(rot_y * Rotation(o), symmetry=sym)
    assert np.degrees(o.angle_with(twin)).max() < 1e-4
    # ... which the group orix would use calls 180 degrees apart.
    o_z = Orientation(o.data, symmetry=C2h)
    twin_z = Orientation(twin.data, symmetry=C2h)
    assert np.degrees(o_z.angle_with(twin_z)).min() > 1.0


def test_every_symmetry_equivalent_has_zero_disorientation_and_one_colour():
    sym = frame_symmetry(space_group=15)
    o = _random_orientations(sym, n=25)
    key = {d: IPFColorKeyTSL(sym, direction=d)
           for d in (Vector3d.xvector(), Vector3d.yvector(), Vector3d.zvector())}
    for eq in _equivalents(o, sym):
        assert np.degrees(o.angle_with(eq)).max() < 1e-4
        for k in key.values():
            assert np.abs(k.orientation2color(o) - k.orientation2color(eq)).max() < 1e-6


def test_ipf_colour_of_the_twin_differs_under_the_old_group():
    """Sanity of the test above: with C2h (z) the same twin gets another colour."""
    sym = frame_symmetry(space_group=12)
    o = _random_orientations(sym, n=25)
    rot_y = Rotation.from_axes_angles([0, 1, 0], np.pi)
    twin = Orientation(rot_y * Rotation(o), symmetry=C2h)
    old = Orientation(o.data, symmetry=C2h)
    k = IPFColorKeyTSL(C2h, direction=Vector3d.xvector())
    assert np.abs(k.orientation2color(old) - k.orientation2color(twin)).max() > 0.05


def test_lattice_decides_when_the_space_group_cannot():
    assert unique_axis(frame_symmetry("2/m", lattice=(5, 6, 7, 90, 90, 100))) == "c"
    assert unique_axis(frame_symmetry("2/m", lattice=(5, 6, 7, 100, 90, 90))) == "a"
    assert unique_axis(frame_symmetry("2/m", lattice=(5, 6, 7, 90, 100, 90))) == "b"
    # Nothing to go on: the convention of the indexers.
    assert unique_axis(frame_symmetry("2/m")) == "b"
    assert unique_axis(frame_symmetry(C2h)) == "b"


def test_explicit_axis_names_are_kept():
    assert unique_axis(frame_symmetry("112")) == "c"
    assert unique_axis(frame_symmetry("211")) == "a"
    assert frame_symmetry("121").name == "121"


def test_resolving_twice_changes_nothing():
    once = frame_symmetry(space_group=1012)
    again = frame_symmetry(once)
    assert unique_axis(again) == "c"
    assert again is once


@pytest.mark.parametrize("n", [n for n in range(1, 231)
                               if cs.monoclinic_class_of_space_group(n) is None])
def test_every_other_space_group_is_returned_exactly_as_orix_does(n):
    got = frame_symmetry(space_group=n)
    ref = get_point_group(n)
    assert got is ref


@pytest.mark.parametrize("name", ["1", "-1", "222", "mm2", "mmm", "4", "4/mmm",
                                  "3", "-3m", "6", "6/mmm", "23", "m-3", "432",
                                  "-43m", "m-3m"])
def test_other_point_group_names_pass_through(name):
    assert frame_symmetry(name) == name
    assert frame_symmetry(None) is None


def test_framephase_keeps_its_space_group_and_gets_the_b_group():
    phase = FramePhase("Al13Fe4", space_group=12, structure=Structure(lattice=LATTICE_B))
    assert phase.space_group.number == 12
    assert phase.point_group.proper_subgroup.name == "121"
    assert Phase("x", space_group=12).point_group.proper_subgroup.name == "112"


def test_framephase_from_point_group_string_follows_the_indexers_convention():
    assert FramePhase("p", point_group="2/m").point_group.proper_subgroup.name == "121"
    # (orix reads the bare string "2" as the TSL code of the Laue group "2/m".)
    assert FramePhase("p", point_group="121").point_group.name == "121"
    assert FramePhase("p", point_group="1m1").point_group.name == "1m1"
    assert FramePhase("p", point_group="112").point_group.proper_subgroup.name == "112"


def test_framephase_leaves_cubic_alone():
    ph = FramePhase("Al", space_group=225)
    assert ph.point_group is get_point_group(225)


def test_framephase_survives_the_orix_containers():
    ph = FramePhase("Al13Fe4", space_group=12, structure=Structure(lattice=LATTICE_B))
    pl = PhaseList(phases=[ph, FramePhase("Al", space_group=225)], ids=[1, 2])
    xmap = CrystalMap(
        rotations=Rotation.identity((4,)),
        phase_id=np.array([1, 1, 2, 2]),
        x=np.arange(4.0), y=np.zeros(4), phase_list=pl,
    )
    for obj in (copy.deepcopy(xmap), xmap.deepcopy(), xmap[xmap.phase_id == 1]):
        assert obj.phases[1].point_group.proper_subgroup.name == "121"
    assert xmap.phases_in_data[1].point_group.proper_subgroup.name == "121"
    assert xmap.phases_in_data[2].point_group.name == "m-3m"
    single = xmap[xmap.phase_id == 1]
    assert single.orientations.symmetry.proper_subgroup.name == "121"


def test_frame_xmap_repairs_plain_orix_phases_in_place():
    plain = PhaseList(phases=[Phase("Al13Fe4", space_group=12,
                                    structure=Structure(lattice=LATTICE_B))], ids=[1])
    xmap = CrystalMap(rotations=Rotation.identity((3,)), phase_id=np.ones(3, int),
                      x=np.arange(3.0), y=np.zeros(3), phase_list=plain)
    assert xmap.phases[1].point_group.proper_subgroup.name == "112"
    cs.frame_xmap(xmap)
    assert xmap.phases[1].point_group.proper_subgroup.name == "121"
    assert xmap.phases[1].space_group.number == 12
    # Already framed: nothing is rebuilt.
    phases = xmap.phases
    cs.frame_xmap(xmap)
    assert xmap.phases is phases


def test_fundamental_zone_of_the_b_group_is_the_b_unique_region():
    sym = frame_symmetry(space_group=12)
    assert sym.euler_fundamental_region == (360, 90, 360)
    assert C2h.euler_fundamental_region == (360, 180, 180)


@pytest.mark.parametrize("axis", ["a", "b", "c"])
def test_ipf_sector_holds_exactly_one_image_of_every_direction(axis):
    """The Laue group's fundamental sector must be a fundamental domain: of the
    images of a direction under the group, exactly one lies inside it."""
    from orix.vector import Vector3d

    laue = cs._TABLE[axis]["2/m"].laue
    sector = laue.fundamental_sector
    rng = np.random.default_rng(4)
    v = Vector3d(rng.normal(size=(400, 3))).unit
    images = laue.outer(v)                       # (group size, 400)
    inside = np.array([np.asarray(sector >= images[i]).reshape(-1)
                       for i in range(images.shape[0])])
    count = inside.sum(axis=0)
    assert set(np.unique(count)) == {1}, np.bincount(count)


@pytest.mark.parametrize("axis", ["a", "b", "c"])
@pytest.mark.parametrize("cls", ["2", "m", "2/m"])
def test_the_laue_group_keeps_the_unique_axis(axis, cls):
    sym = cs._TABLE[axis][cls]
    assert unique_axis(sym) == axis
    assert unique_axis(sym.laue) == axis
    assert sym.laue.name == "2/m"


def test_space_group_axis_is_cached():
    cs._space_group_axis.cache_clear()
    cs._space_group_axis(12)
    cs._space_group_axis(12)
    assert cs._space_group_axis.cache_info().hits >= 1


@pytest.mark.parametrize("axis", ["a", "b", "c"])
def test_ipf_colours_agree_across_every_equivalent_for_every_axis(axis):
    sym = cs._TABLE[axis]["2/m"]
    o = _random_orientations(sym, n=30)
    for d in (Vector3d.xvector(), Vector3d.yvector(), Vector3d.zvector()):
        key = IPFColorKeyTSL(sym, direction=d)
        for eq in _equivalents(o, sym):
            assert np.abs(key.orientation2color(o) - key.orientation2color(eq)).max() < 1e-6
