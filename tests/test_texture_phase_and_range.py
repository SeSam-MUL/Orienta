"""The texture step counts the pixels of the texture phase only, and a sharp texture reads as sharp.

The script computes everything on ``oriR = ebsd(ph).orientations``: the pixels of ONE phase. The components (and the
crystal symmetry) are those of that phase (here Al, which ties with the other phase and so is the default), so the
pixels of any other phase are neither counted nor in the denominator. Which phase the default picks, and the explicit
choice, are in tests/test_texture_phase_choice.py.
"""
import dataclasses

import numpy as np
import pytest

pytest.importorskip("orix")

import analysis.texture_analysis as ta                                           # noqa: E402
from analysis.orientation_math import disorientation_angle, proper_ops           # noqa: E402
from analysis.texture_components import TextureComponent, get_texture_preset      # noqa: E402


def _two_phase_dataset(al_quats, other_quats, n_unindexed=3, second_point_group="m-3m"):
    """Al (phase 0), a second phase (phase 1) and some not-indexed pixels (-1), side by side in one row."""
    from orix.crystal_map import CrystalMap, PhaseList
    from orix.quaternion import Rotation
    from analysis.ebsd_dataset import EBSDDataset
    q = np.concatenate([np.asarray(al_quats, float).reshape(-1, 4), np.asarray(other_quats, float).reshape(-1, 4),
                        np.tile([1.0, 0, 0, 0], (n_unindexed, 1))])
    ids = np.concatenate([np.zeros(len(al_quats), int), np.ones(len(other_quats), int), np.full(n_unindexed, -1)])
    n = len(q)
    xmap = CrystalMap(rotations=Rotation(q), phase_id=ids, x=np.arange(n, dtype=float), y=np.zeros(n),
                      phase_list=PhaseList(names=["Al", "Other"], point_groups=["m-3m", second_point_group]),
                      prop={"bc": np.full(n, 150.0)})
    return EBSDDataset(xmap, step_size=1.0), ids


def _comps():
    fcc = get_texture_preset("FCC_Rolling")
    return {"Goss": fcc["Goss"], "Cube": fcc["Cube"]}


@pytest.fixture(scope="module")
def world():
    """Al: 2 pixels on Goss, 8 on Cube. Other phase: 10 pixels on Goss (they would triple Goss if counted)."""
    probe = _two_phase_dataset([[1.0, 0, 0, 0]], [[1.0, 0, 0, 0]])[0]
    goss = ta.ideal_orientation(_comps()["Goss"], probe.phase)
    al = np.vstack([np.tile(goss, (2, 1)), np.tile([1.0, 0, 0, 0], (8, 1))])
    ds, ids = _two_phase_dataset(al, np.tile(goss, (10, 1)))
    return ds, ids


def test_angles_fractions_and_the_denominator_are_those_of_the_components_phase(world):
    ds, ids = world
    angles = ta.get_component_angles(ds, _comps())
    assert angles.min_angles.shape == (2, 10)                       # the 10 Al pixels, not the 20 indexed ones
    fr = ta.calculate_texture_component_fractions(ds, _comps(), 10.0)
    assert fr == {"Goss": 20.0, "Cube": 80.0}


def test_pixel_mask_and_the_split_count_the_components_phase_only(world):
    ds, ids = world
    mask = np.ones(len(ids), bool)
    assert ta.calculate_texture_component_fractions(ds, _comps(), 10.0, pixel_mask=mask) == {"Goss": 20.0, "Cube": 80.0}
    only_other = ids == 1                                           # a mask that selects nothing of the phase: no share to give
    assert ta.calculate_texture_component_fractions(ds, _comps(), 10.0, pixel_mask=only_other) == {"Goss": None, "Cube": None}

    class Grains:                                                   # two groups: everything in grain 1 / grain 2
        grain_ids = np.where(np.arange(len(ids)) < 5, 1, 2)
    out = ta.calculate_texture_split_by_rx(ds, Grains(), [1], _comps(), 10.0)
    assert out["n_rx_pixels"] == 5 and out["n_substructure_pixels"] == 5          # Al pixels only (10 of 25 pixels)
    assert out["rx"] == {"Goss": 40.0, "Cube": 60.0}                # pixels 0-4: 2 Goss, 3 Cube
    assert out["substructure"] == {"Goss": 0.0, "Cube": 100.0}


def test_the_map_gives_the_other_phase_no_component(world):
    ds, ids = world
    cmap, _ = ta.calculate_texture_component_map(ds, _comps(), 10.0)
    flat = cmap.reshape(-1)
    assert list(flat[:10]) == [1, 1] + [2] * 8
    assert (flat[10:] == 0).all()


def test_surface_plane_fractions_count_the_components_phase_only():
    half = np.radians(45.0) / 2
    tilted = [np.cos(half), np.sin(half), 0, 0]                     # (001) 45 deg off ND
    ds, _ = _two_phase_dataset(np.tile([1.0, 0, 0, 0], (4, 1)), np.tile(tilted, (4, 1)))
    assert ta.calculate_surface_plane_fraction(ds, [0, 0, 1], 15.0) == 100.0


def test_a_tie_between_phases_goes_to_the_lower_id(world):
    """Al (10 pixels) and the other phase (10 pixels) tie: the default is the lower id. (Which phase the default picks
    on a map whose matrix has the higher id is in tests/test_texture_phase_choice.py.)"""
    ds, _ = world
    assert ta.resolve_texture_phase(ds) == 0
    assert ds.phase.name == "Al"


# ------------------------------------------------------------------------------------------------------------------
# index and entropy
# ------------------------------------------------------------------------------------------------------------------
def test_a_sharp_cube_texture_reads_as_sharp_in_index_and_entropy():
    """3000 orientations within 3 deg of Cube lie within 10 deg of Cube, Rot.Cube ND10 and RD10: the component
    fractions sum to about 286 % (they overlap). The index and entropy are the kernel density's, not the fractions':
    the index is far above the uniform 1 and the entropy below the uniform 0, the same as for any other sharp texture."""
    from orix.crystal_map import CrystalMap, PhaseList
    from orix.quaternion import Rotation
    from analysis.ebsd_dataset import EBSDDataset
    rng = np.random.default_rng(3)
    n = 3000
    ang = np.radians(rng.uniform(0, 3, n))
    ax = rng.normal(size=(n, 3))
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    q = np.concatenate([np.cos(ang / 2)[:, None], ax * np.sin(ang / 2)[:, None]], axis=1)
    xmap = CrystalMap(rotations=Rotation(q), phase_id=np.zeros(n, int), x=np.arange(n, dtype=float), y=np.zeros(n),
                      phase_list=PhaseList(names=["Al"], point_groups=["m-3m"]), prop={"bc": np.full(n, 150.0)})
    comps = get_texture_preset("FCC_Rolling")
    res = ta.analyze_texture(EBSDDataset(xmap, step_size=1.0), comps)
    assert sum(res.component_fractions.values()) > 250.0
    assert res.texture_index > 50.0
    assert res.entropy < -2.0


# ------------------------------------------------------------------------------------------------------------------
# the stored angles, the BCC preset
# ------------------------------------------------------------------------------------------------------------------
def test_the_angles_keep_one_array_per_component_and_derive_the_minimum(world):
    ds, _ = world
    fields = {f.name: f for f in dataclasses.fields(ta.ComponentAngles)}
    assert "min_angles" not in fields                                # derived, not a second copy
    assert fields["variant_angles"].default is dataclasses.MISSING   # not optional
    angles = ta.get_component_angles(ds, _comps())
    assert isinstance(angles.variant_angles, list) and len(angles.variant_angles) == 2
    for k, var in enumerate(angles.variant_angles):
        assert np.array_equal(angles.min_angles[k], var.min(axis=0))


def test_the_bcc_rotated_goss_is_the_textbook_one_and_goss_turned_by_90_degrees_about_nd():
    """{110}<1-10> is Goss {110}<001> turned by 90 deg about ND. (Modulo the cubic symmetry that is a disorientation of
    62.8 deg, not 90: the symmetry reduces the angle. It must not be 0, as the former {110}<001> was.)"""
    from analysis.orientation_math import qmul
    bcc = get_texture_preset("BCC_Rolling")
    assert bcc["Rot. Goss"].hkl == [1, 1, 0] and bcc["Rot. Goss"].uvw == [1, -1, 0]
    ds, _ = _two_phase_dataset([[1.0, 0, 0, 0]], [[1.0, 0, 0, 0]])
    phase = ds.phase
    goss, rot = ta.component_variants(bcc["Goss"], phase), ta.component_variants(bcc["Rot. Goss"], phase)
    ops = proper_ops(phase.point_group)
    d = min(disorientation_angle(g[None], r[None], ops)[0] for g in goss for r in rot)
    assert np.degrees(d) == pytest.approx(62.8, abs=0.05)
    turned = qmul(goss[:1], np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]))        # 90 deg about ND
    assert min(disorientation_angle(turned, r[None], ops)[0] for r in rot) < 1e-6


def test_the_bcc_rotated_goss_is_inv_goss_of_the_fcc_preset():
    ds, _ = _two_phase_dataset([[1.0, 0, 0, 0]], [[1.0, 0, 0, 0]])
    a = ta.component_variants(get_texture_preset("BCC_Rolling")["Rot. Goss"], ds.phase)
    b = ta.component_variants(get_texture_preset("FCC_Rolling")["Inv. Goss"], ds.phase)
    ops = proper_ops(ds.phase.point_group)
    assert disorientation_angle(a[:1], b[:1], ops)[0] < 1e-6
