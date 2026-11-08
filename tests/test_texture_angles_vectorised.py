"""The angles of the texture components with the vectorised orientation arithmetic.

``analysis.texture_analysis.get_component_angles`` computes the angle of every indexed pixel to every variant of every
component with ``analysis.orientation_math``. The orix implementation it replaced (``Misorientation(...).reduce()``
per ideal orientation) is kept in ``tests/texture_reference_impl.py`` as the reference of the ARITHMETIC; every angle
must equal it, and so must everything derived from it (fractions, split, map, texture index, entropy) wherever the
angle is not within ``MARGIN`` of the tolerance. How the variants are built is not compared here (the reference is
handed the variants of the code under test); that is checked against MTEX in test_texture_components_mtex.py.
"""
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("orix")

import analysis.texture_analysis as ta                                           # noqa: E402
from analysis.texture_components import TextureComponent, get_texture_preset      # noqa: E402
from tests import texture_reference_impl as old                                  # noqa: E402

MARGIN = 1e-9          # rad
TOL_DEG = 10.0


@pytest.fixture(autouse=True, scope="module")
def _reference_gets_the_variants_of_the_code_under_test():
    old.use_variants_of_the_code_under_test()


def _fcc():
    return dict(get_texture_preset("FCC_Rolling"))


def _dataset_from_quaternions(quats, phase_name="Al", point_group="m-3m"):
    from orix.crystal_map import CrystalMap, PhaseList
    from orix.quaternion import Rotation
    from analysis.ebsd_dataset import EBSDDataset
    n = len(quats)
    nx = 50
    ny = -(-n // nx)
    pad = ny * nx - n
    data = np.concatenate([quats, np.tile(quats[:1], (pad, 1))]) if pad else quats
    xmap = CrystalMap(rotations=Rotation(data), phase_id=np.zeros(len(data), dtype=int),
                      x=np.tile(np.arange(nx, dtype=float), ny), y=np.repeat(np.arange(ny, dtype=float), nx),
                      phase_list=PhaseList(names=[phase_name], point_groups=[point_group]),
                      prop={"bc": np.full(len(data), 150.0)})
    return EBSDDataset(xmap, step_size=1.0)


def _ideal_orientations(dataset, components):
    """The variants (an orix Orientation array) of every component, as the code under test builds them."""
    from orix.quaternion import Orientation
    return {name: Orientation(ta.component_variants(c, dataset.phase)) for name, c in components.items()}


def _near_tolerance_quaternions(components, seed=7, per_component=19):
    """Orientations at (tolerance + d) from a random variant of every component, presented through a random crystal
    symmetry operator: d on a ladder from 1e-3 rad down to 1e-12 rad on both sides, plus exact ones and some at
    random angles."""
    from orix.quaternion import Rotation
    from orix.quaternion.symmetry import Oh
    dummy = _dataset_from_quaternions(np.array([[1.0, 0, 0, 0]]))
    ideals = _ideal_orientations(dummy, components)
    rng = np.random.default_rng(seed)
    tol = np.radians(TOL_DEG)
    deltas = np.concatenate([[0.0], np.logspace(-12, -3, 9), -np.logspace(-12, -3, 9)])
    crystal = Oh.proper_subgroup
    out = []
    for name, ideal in ideals.items():
        for i in range(per_component):
            axis = rng.normal(size=3)
            ang = tol + deltas[i % len(deltas)] if i % 3 else rng.uniform(0, np.radians(60))
            small = Rotation.from_axes_angles(axis, ang)
            variant = ideal[int(rng.integers(ideal.size))]
            cr = crystal[int(rng.integers(crystal.size))]
            o = cr * (small * variant)
            out.append(np.asarray(o.data, dtype=float).reshape(4))
    q = np.array(out)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def synthetic():
    comps = _fcc()
    dataset = _dataset_from_quaternions(_near_tolerance_quaternions(comps))
    return dataset, comps, old.component_angles(dataset, comps)       # the orix angles, computed once


def _assert_angles_equal(new, ref):
    assert new.shape == ref.shape
    assert np.array_equal(np.isnan(new), np.isnan(ref))
    diff = np.abs(new - ref)
    assert np.nanmax(diff) < MARGIN, f"largest angle difference {np.nanmax(diff):.3e} rad"


def test_every_angle_equals_the_orix_implementation_on_orientations_near_the_tolerance(synthetic):
    dataset, comps, ref = synthetic
    dataset._component_angles = None
    new = ta.get_component_angles(dataset, comps).min_angles
    assert new.shape[0] == 14
    _assert_angles_equal(new, ref)
    tol = np.radians(TOL_DEG)
    assert (np.abs(ref - tol) < 1e-4).sum() > 14 * 4        # the map really sits at the tolerance boundary


def _best_component(angles, tol):
    """0 = none within the tolerance, else the 1-based nearest component; a later component must be closer by more
    than MARGIN to take a pixel from an earlier one, so a tie goes to the lowest index (the rule of the component map,
    written out on its own)."""
    a = np.where(np.isnan(angles), np.inf, angles)
    best_angle = np.full(a.shape[1], np.inf)
    best = np.zeros(a.shape[1], dtype=int)
    for k in range(a.shape[0]):
        take = (a[k] < best_angle - MARGIN) & (a[k] < tol)
        best_angle[take] = a[k][take]
        best[take] = k + 1
    return best


def test_fractions_and_map_follow_from_the_orix_angles(synthetic):
    """Everything derived from the angles equals what the orix angles give, wherever no angle is within MARGIN of the
    tolerance. (The orix pipelines end to end are compared on a four-grain map in test_texture_angles_reuse.py.)
    A component's fraction is the sum over its variants of the share of pixels within the tolerance of that variant."""
    dataset, comps, ref_angles = synthetic
    dataset._component_angles = None
    tol = np.radians(TOL_DEG)
    ref_variants = old.variant_angles(dataset, comps)
    dist = np.abs(ref_angles - tol)
    near = int(sum((np.abs(v - tol) < MARGIN).sum() for v in ref_variants))
    n = ref_angles.shape[1]
    ref = {name: sum((np.sum(row < tol) / n) * 100.0 for row in ref_variants[k]) for k, name in enumerate(comps)}
    assert max(ref.values()) > 100.0 / n * 5              # the set puts pixels near every component
    new = ta.calculate_texture_component_fractions(dataset, comps, TOL_DEG)
    assert new.keys() == ref.keys()
    res = ta.analyze_texture(dataset, comps)
    if near == 0:                                              # no angle inside the margin: exactly equal
        assert new == ref
        assert res.component_fractions == ref
    else:                                                      # an angle inside it may fall either side
        for k in ref:
            assert abs(new[k] - ref[k]) <= near * 100.0 / n + 1e-12
    new_map, names = ta.calculate_texture_component_map(dataset, comps, TOL_DEG)
    assert names == list(comps)
    flat = np.flatnonzero(np.asarray(dataset.indexed_mask_2d).reshape(-1))
    got = np.asarray(new_map).reshape(-1)[flat]
    want = _best_component(ref_angles, tol)
    differing = got != want
    # a different pixel assignment needs an angle within MARGIN of the tolerance (no two components are the same
    # orientation any more, so no tie can hide a wrong rule)
    if differing.any():
        assert (dist[:, differing].min(axis=0) < MARGIN).all()


def test_a_component_that_cannot_be_built_stays_nan_and_the_rest_is_unchanged(synthetic):
    dataset, comps, ref_angles = synthetic
    three = dict(list(comps.items())[:3])
    three["Broken"] = TextureComponent(name="Broken", hkl=[0, 0, 0], uvw=[0, 0, 0])
    dataset._component_angles = None
    got = ta.get_component_angles(dataset, three)
    assert got.failed == [False, False, False, True]
    assert np.isnan(got.min_angles[3]).all()
    _assert_angles_equal(got.min_angles[:3], ref_angles[:3])


def test_a_noncubic_phase_uses_its_own_symmetry():
    from orix.quaternion import Rotation
    q = np.asarray(Rotation.random(60).data, dtype=float)
    dataset = _dataset_from_quaternions(q, phase_name="Mg", point_group="6/mmm")
    comps = {"Euler": TextureComponent(name="Euler", euler_deg=[30.0, 20.0, 10.0])}
    dataset._component_angles = None
    _assert_angles_equal(ta.get_component_angles(dataset, comps).min_angles, old.component_angles(dataset, comps))


def test_min_angle_without_symmetry_matches_orix():
    from orix.quaternion import Orientation, Rotation
    ori = Orientation(Rotation.random(500).data)
    ideal = Orientation.from_euler(np.radians([[10.0, 20.0, 30.0], [100.0, 50.0, 20.0]]))
    new = ta._min_angle_to_equivalents(ori, ideal, None)
    ref = old.min_angle_to_equivalents(ori, ideal, None)
    assert np.abs(new - ref).max() < MARGIN


# ---------------------------------------------------------------------------------------------
# real data: the Oxford reference scan, all 14 components. The new code runs on every indexed pixel, the orix reference (about 4 minutes
# for the whole map) on an evenly spaced subset of them.
# ---------------------------------------------------------------------------------------------
def _real_dataset(filename):
    from tests.data_deps import TEST_DATA_DIR
    from analysis.ebsd_dataset import EBSDDataset
    from analysis.xmap_loader import load_xmap_file
    path = Path(os.environ.get("ORIENTA_TEST_DATA", TEST_DATA_DIR)) / filename
    if not path.is_file():
        pytest.skip(f"needs {path} (set ORIENTA_TEST_DATA to a Test_data directory that has it)")
    xmap, step, _ = load_xmap_file(str(path))
    return EBSDDataset(xmap, step_size=step, source=str(path))


def test_sampleb_all_fourteen_components_equal_the_orix_implementation():
    dataset = _real_dataset("EBSD_SampleB_extrusion_withPattern Sample_B Arbeitsbereich 1 "
                            "Elementverteilungsdaten 1.h5oina")
    comps = _fcc()
    dataset._component_angles = None
    new = ta.get_component_angles(dataset, comps).min_angles
    assert new.shape[0] == 14 and not np.isnan(new).any()
    sel = np.linspace(0, new.shape[1] - 1, 250).astype(int)
    ori_all = dataset.orientations[dataset.indexed_mask]
    ref = old.component_angles(dataset, comps, orientations=ori_all[sel])
    _assert_angles_equal(new[:, sel], ref)
