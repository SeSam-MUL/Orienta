"""Texture components exactly as a reference MTEX script builds and counts them.

The script builds every component as ``unique(specimenSymmetry('mmm') * orientation.byMiller(hkl, uvw, CS))`` (or
``byEuler``) and sums ``volume(oriR, variant, 10 deg)`` over the first 1, 2 or 4 variants. The numbers of that
computation are in ``tests/fixtures/analysis_mtex_parity/texture_components.npz`` (MTEX for Python 1.0.0, run in
its own environment, numbers only; see the fixtures README). An orix quaternion is the conjugate of the MTEX rotation of the same orientation.
"""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("orix")

import analysis.texture_analysis as ta                                           # noqa: E402
from analysis.orientation_math import disorientation_angle, proper_ops, qconj, qmul      # noqa: E402
from analysis.texture_components import TextureComponent, get_texture_preset      # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "analysis_mtex_parity" / "texture_components.npz"
N_VARIANTS = [2, 4, 2, 1, 1, 1, 2, 2, 2, 1, 4, 2, 4, 2]       # what the script sums, per FCC preset component
CLASS_TOL = 1e-7        # rad; MTEX's own ideal orientations carry an error up to 4e-8 rad (measured)


@pytest.fixture(scope="module")
def mtex():
    return np.load(FIXTURE)


def _fcc():
    return dict(get_texture_preset("FCC_Rolling"))


def _dataset(quats, phase=None):
    from orix.crystal_map import CrystalMap, PhaseList
    from orix.quaternion import Rotation
    from analysis.ebsd_dataset import EBSDDataset
    quats = np.asarray(quats, dtype=float).reshape(-1, 4)
    n = len(quats)
    phase_list = PhaseList(names=["Al"], point_groups=["m-3m"]) if phase is None else PhaseList([phase])
    xmap = CrystalMap(rotations=Rotation(quats), phase_id=np.zeros(n, dtype=int),
                      x=np.arange(n, dtype=float), y=np.zeros(n), phase_list=phase_list,
                      prop={"bc": np.full(n, 150.0)})
    return EBSDDataset(xmap, step_size=1.0)


@pytest.fixture(scope="module")
def al_phase():
    return _dataset(np.array([[1.0, 0, 0, 0]])).phase


def _class_distance(a, b, ops):
    """Smallest rotation angle between orientation sets ``a`` (n, 4) and ``b`` (m, 4) under the crystal operators
    ``ops`` (orix: crystal symmetry from the left): for each row of ``a`` the smallest angle to any row of ``b``."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    return np.array([min(disorientation_angle(x[None], y[None], ops)[0] for y in b) for x in a])


# --------------------------------------------------------------------------------------------------------------
# the variants
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("k", range(14))
def test_the_variants_are_the_variants_mtex_lists(mtex, al_phase, k):
    comps = _fcc()
    name = list(comps)[k]
    assert str(mtex["names"][k]) == name
    theirs = qconj(mtex["variants_mtex"][k, :int(mtex["n_used"][k])])          # MTEX rotation -> orix quaternion
    mine = ta.component_variants(comps[name], al_phase)
    assert mine.shape == (N_VARIANTS[k], 4) == (len(theirs), 4)
    ops = proper_ops(al_phase.point_group)
    assert _class_distance(theirs, mine, ops).max() < CLASS_TOL, name          # every MTEX variant is one of mine
    assert _class_distance(mine, theirs, ops).max() < CLASS_TOL, name          # and the other way round


def test_the_script_sums_exactly_the_distinct_variants(mtex):
    """The script takes the first 1, 2 or 4 of ``unique(...)``: with that many distinct variants every component's
    choice is the whole set, so MTEX's order of the list is not needed."""
    assert list(mtex["n_distinct"]) == list(mtex["n_used"]) == N_VARIANTS


def test_symmetry_multiplicity_of_the_preset_is_the_number_of_variants(al_phase):
    for k, (name, comp) in enumerate(_fcc().items()):
        assert comp.symmetry_multiplicity == N_VARIANTS[k], name
        assert len(ta.component_variants(comp, al_phase)) == comp.symmetry_multiplicity, name


def test_no_two_components_are_the_same_orientation(al_phase):
    """The former construction made Inv. Goss equal to Cube and Goss equal to Rot.Cube ND45."""
    comps = _fcc()
    ops = proper_ops(al_phase.point_group)
    ideal = {n: ta.component_variants(c, al_phase) for n, c in comps.items()}
    names = list(comps)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert _class_distance(ideal[a], ideal[b], ops).min() > np.radians(5.0), (a, b)
    assert _class_distance(ideal["Inv. Goss"], ideal["Cube"], ops).min() > np.radians(40.0)
    assert _class_distance(ideal["Goss"], ideal["Rot.Cube ND45"], ops).min() > np.radians(40.0)


@pytest.mark.parametrize("name", ["Brass", "S", "Copper", "Goss", "Inv. Goss", "Rot.Cube ND45", "Rot.Cube ND22",
                                  "Cube", "Q", "P", "R", "Shear 2"])
def test_the_ideal_orientation_puts_hkl_on_nd_and_an_equivalent_of_uvw_on_rd(name, al_phase):
    """{hkl}<uvw> as written in the script: the plane normal along ND (z), a crystal equivalent of uvw that is
    perpendicular to it along RD (x). (``Miller`` / ``byMiller`` accept pairs that are not perpendicular as
    written, like Brass {011}<211>; the equivalent <2 1 -1> is.)"""
    comp = _fcc()[name]
    q = ta.ideal_orientation(comp, al_phase)
    w, x, y, z = q
    m = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    h = np.array(comp.hkl, float)
    u = np.array(comp.uvw, float)
    nd, rd = m[:, 2], m[:, 0]          # specimen z and x axes expressed in the crystal frame (orix quaternion)
    assert np.allclose(nd, h / np.linalg.norm(h), atol=1e-12)
    assert abs(nd @ rd) < 1e-12
    assert np.allclose(np.sort(np.abs(rd)), np.sort(np.abs(u)) / np.linalg.norm(u), atol=1e-12)      # cubic equivalent


@pytest.mark.parametrize("name", list(_fcc()))
def test_the_specimen_symmetry_acts_on_the_specimen_side(name, al_phase):
    """The orientation turned by a 180 deg rotation about RD, TD or ND of the SPECIMEN is the same component."""
    comp = _fcc()[name]
    ideal = ta.component_variants(comp, al_phase)
    ops = proper_ops(al_phase.point_group)
    for s in ([0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]):
        turned = qmul(ideal[0], np.array(s, float))         # orix: the specimen operator acts from the right
        assert _class_distance(turned[None], ideal, ops)[0] < 1e-6, (name, s)      # arccos of 1 - eps is good to 3e-8


# --------------------------------------------------------------------------------------------------------------
# the counts
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("tol", [5, 10, 15])
def test_fractions_equal_the_mtex_volumes_on_a_random_and_clustered_orientation_set(mtex, tol):
    ds = _dataset(qconj(mtex["quats_mtex"]))
    comps = _fcc()
    got = ta.calculate_texture_component_fractions(ds, comps, float(tol))
    want = mtex[f"vol_pct_{tol}"]
    angles = ta.get_component_angles(ds, comps)
    # a pixel within 1e-6 rad of the tolerance could fall either side of MTEX's slightly different ideal
    near = [int(np.sum(np.abs(v - np.radians(tol)) < 1e-6)) for v in
            (a for var in angles.variant_angles for a in var)]
    assert sum(near) == 0, near
    n = len(mtex["quats_mtex"])
    for k, name in enumerate(comps):
        assert got[name] == pytest.approx(want[k], abs=1e-9), name
    assert any(w > 1.0 for w in want)                      # the set really hits every component
    assert max(got.values()) < 100.0 and n == 9000


def test_overlapping_variants_are_counted_twice_as_the_script_does():
    """volEBSD = volume(v1) + volume(v2): a pixel within the tolerance of two variants counts twice."""
    comp = TextureComponent(name="Close", euler_deg=[3.0, 0.0, 0.0], symmetry_multiplicity=2)
    ds = _dataset(np.tile([1.0, 0, 0, 0], (5, 1)))
    assert len(ta.component_variants(comp, ds.phase)) == 2
    assert ta.calculate_texture_component_fractions(ds, {"Close": comp}, 10.0) == {"Close": 200.0}


def test_masked_fractions_and_the_split_use_the_same_sum(mtex):
    ds = _dataset(qconj(mtex["quats_mtex"]))
    comps = _fcc()
    mask = np.zeros(len(mtex["quats_mtex"]), dtype=bool)
    mask[::2] = True
    sub = ta.calculate_texture_component_fractions(ds, comps, 10.0, pixel_mask=mask)
    ds_half = _dataset(qconj(mtex["quats_mtex"][mask]))
    assert sub == ta.calculate_texture_component_fractions(ds_half, comps, 10.0)


def test_volume_semantics_of_mtex_are_the_ones_used(mtex):
    """MTEX volume(list, centre, tol): count / list length, NaN entries counted in the length only, the boundary
    crossed within floating-point noise."""
    assert list(mtex["edge_volume"]) == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]          # tol + (-1e-6 ... 1e-6)
    q = np.array(mtex["nan_quats"])
    ds = _dataset(qconj(q))
    comp = {"Cube": TextureComponent(name="Cube", euler_deg=[0.0, 0.0, 0.0])}
    assert ta.calculate_texture_component_fractions(ds, comp, 10.0)["Cube"] == pytest.approx(
        100.0 * float(mtex["nan_volume"]), abs=1e-12)
    for eps, want in zip(mtex["edge_eps"], mtex["edge_volume"]):
        a = np.radians(10.0) + float(eps)
        ds = _dataset([[np.cos(a / 2), 0, 0, np.sin(a / 2)]])
        got = ta.calculate_texture_component_fractions(ds, comp, 10.0)["Cube"]
        if abs(eps) >= 1e-9:                                      # the margin the parity claim is made with
            assert got == 100.0 * want, eps


def test_a_nan_orientation_gives_nan_angles_counts_in_no_fraction_and_claims_no_component():
    comps = {"Cube": TextureComponent(name="Cube", euler_deg=[0.0, 0.0, 0.0]),
             "Cube again": TextureComponent(name="Cube again", euler_deg=[0.0, 0.0, 0.0])}
    ds = _dataset([[1.0, 0, 0, 0], [np.nan] * 4, [1.0, 0, 0, 0], [0.0, 0, 0, 1.0]])
    angles = ta.get_component_angles(ds, comps)
    assert np.isnan(angles.min_angles[:, 1]).all() and np.isfinite(angles.min_angles[:, [0, 2, 3]]).all()
    assert ta.calculate_texture_component_fractions(ds, comps, 10.0)["Cube"] == 75.0     # of 4 pixels, the NaN one counted
    cmap, _ = ta.calculate_texture_component_map(ds, comps, 10.0)
    assert list(cmap.reshape(-1)) == [1, 0, 1, 1]


def test_the_map_gives_a_tie_to_the_lowest_component_index():
    """Two components 1.7e-10 rad apart (a rotation of 1e-8 deg about ND): for a pixel 5 deg either side of them the
    nearer one is nearer by less than 1e-9 rad, so it is a tie and the first component keeps the pixel. (At an angle
    near 0 the angle itself is only good to about 3e-8 rad, so the pixels sit at 5 deg.)"""
    comps = {"A": TextureComponent(name="A", euler_deg=[0.0, 0.0, 0.0]),
             "B": TextureComponent(name="B", euler_deg=[1e-8, 0.0, 0.0])}
    half = np.radians(5.0) / 2
    ds = _dataset([[np.cos(half), 0, 0, np.sin(half)], [np.cos(half), 0, 0, -np.sin(half)]])
    angles = ta.get_component_angles(ds, comps).min_angles
    assert (np.abs(angles[0] - angles[1]) < 1e-9).all() and (angles[0] != angles[1]).all()      # a tie, not an equality
    cmap, _ = ta.calculate_texture_component_map(ds, comps, 10.0)
    assert list(cmap.reshape(-1)) == [1, 1]


def test_the_angles_are_per_variant_and_the_component_minimum_is_their_minimum(mtex):
    ds = _dataset(qconj(mtex["quats_mtex"][:3000]))
    comps = _fcc()
    angles = ta.get_component_angles(ds, comps)
    assert [len(v) for v in angles.variant_angles] == N_VARIANTS
    for k, var in enumerate(angles.variant_angles):
        assert np.array_equal(angles.min_angles[k], np.min(var, axis=0))


# --------------------------------------------------------------------------------------------------------------
# other presets and phases
# --------------------------------------------------------------------------------------------------------------
def test_the_bcc_preset_is_built_by_the_same_rule(al_phase):
    bcc = get_texture_preset("BCC_Rolling")
    assert [len(ta.component_variants(c, al_phase)) for c in bcc.values()] == [1, 1]
    ds = _dataset([ta.ideal_orientation(bcc["Goss"], al_phase)])
    assert ta.calculate_texture_component_fractions(ds, bcc, 10.0)["Goss"] == 100.0


def test_a_hexagonal_phase_builds_miller_components_from_its_own_lattice():
    from diffpy.structure import Lattice, Structure
    from orix.crystal_map import Phase
    phase = Phase(name="Mg", point_group="6/mmm",
                  structure=Structure(lattice=Lattice(3.2, 3.2, 5.2, 90, 90, 120)))
    comp = TextureComponent(name="Basal", hkl=[0, 0, 1], uvw=[1, 0, 0])
    q = ta.ideal_orientation(comp, phase)
    ds = _dataset([q, qmul(q, np.array([np.cos(np.radians(2.5)), 0, 0, np.sin(np.radians(2.5))]))], phase=phase)
    angles = ta.get_component_angles(ds, {"Basal": comp}).min_angles[0]
    assert angles[0] < 1e-9 and angles[1] == pytest.approx(np.radians(5.0), abs=1e-9)
