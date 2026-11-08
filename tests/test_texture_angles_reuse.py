"""One per-pixel component assignment for the fractions, the RX / substructure split and the component map.

The minimum angle of every indexed pixel to every component is computed once per dataset
(``analysis.texture_analysis.get_component_angles``). Everything derived from it must equal what the earlier
code returned, bit for bit: that code is kept in ``tests/texture_reference_impl.py`` and run here on the same map.
The reference is handed the variants of the code under test (their construction is checked against MTEX in
test_texture_components_mtex.py) and counts as the script of the owner counts: the sum over the variants.
"""
import numpy as np
import pytest

pytest.importorskip("orix")

import analysis.texture_analysis as ta                                           # noqa: E402
from analysis.grain_analysis import reconstruct_grains                           # noqa: E402
from analysis.texture_components import TextureComponent, get_texture_preset      # noqa: E402
from tests import texture_reference_impl as old                                  # noqa: E402
from tests.test_excel_exporter import _four_grain_map                            # noqa: E402


def _components():
    # three real components keep the run short; the code path is the same for all 14
    comps = {k: v for k, v in get_texture_preset("FCC_Rolling").items() if k in ("Brass", "Goss", "Cube")}
    # a component orix cannot build: its fraction is None (a missing number) and it claims no pixel in the map
    comps["Broken"] = TextureComponent(name="Broken", hkl=[0, 0, 0], uvw=[0, 0, 0])
    return comps


@pytest.fixture(autouse=True, scope="module")
def _reference_gets_the_variants_of_the_code_under_test():
    old.use_variants_of_the_code_under_test()


@pytest.fixture(scope="module")
def world():
    dataset = _four_grain_map()
    grains = reconstruct_grains(dataset, angle_threshold_deg=5.0, min_pixels=3)
    comps = _components()
    return dataset, grains, comps


@pytest.fixture
def counting(monkeypatch):
    """How often the expensive misorientation step runs (one count per variant of a component)."""
    calls = []
    real = ta._min_angle_to_equivalents

    def counted(orientations, ideals, symmetry=None):
        calls.append(orientations.size)
        return real(orientations, ideals, symmetry)

    monkeypatch.setattr(ta, "_min_angle_to_equivalents", counted)
    return calls


def test_the_unbuildable_component_is_really_skipped(world):
    dataset, _, comps = world
    assert old.calculate_texture_component_fractions(dataset, comps, 10.0)["Broken"] is None


def test_fractions_equal_the_earlier_implementation_bit_for_bit(world):
    dataset, _, comps = world
    dataset._component_angles = None
    new = ta.calculate_texture_component_fractions(dataset, comps, 10.0)
    ref = old.calculate_texture_component_fractions(dataset, comps, 10.0)
    assert list(new) == list(ref) and new == ref
    assert new["Broken"] is None and 0.0 < new["Brass"] + new["Goss"] + new["Cube"]
    for tol in (3.0, 15.0):                                        # the angles do not depend on the tolerance
        assert ta.calculate_texture_component_fractions(dataset, comps, tol) == \
            old.calculate_texture_component_fractions(dataset, comps, tol)


def test_masked_fractions_and_the_split_equal_the_earlier_implementation(world):
    dataset, grains, comps = world
    ids = np.asarray(grains.grain_ids).reshape(-1)
    for rx_ids in ([1, 2], [3], list(range(1, grains.n_grains + 1)), []):
        new = ta.calculate_texture_split_by_rx(dataset, grains, rx_ids, comps, 10.0)
        ref = old.calculate_texture_split_by_rx(dataset, grains, rx_ids, comps, 10.0)
        assert new == ref, rx_ids
    mask = np.isin(ids, [1, 4])
    assert ta.calculate_texture_component_fractions(dataset, comps, 10.0, pixel_mask=mask) == \
        old.calculate_texture_component_fractions(dataset, comps, 10.0, pixel_mask=mask)


def test_the_component_map_equals_the_earlier_implementation(world):
    dataset, _, comps = world
    new_map, new_names = ta.calculate_texture_component_map(dataset, comps, 10.0)
    ref_map, ref_names = old.calculate_texture_component_map(dataset, comps, 10.0)
    assert new_names == ref_names == list(comps)
    assert np.array_equal(new_map, ref_map) and new_map.dtype == ref_map.dtype
    assert set(np.unique(new_map)) - {0} and 4 not in set(np.unique(new_map))      # "Broken" (4) claims nothing


def test_analyze_texture_equals_the_earlier_implementation(world):
    dataset, _, comps = world
    new = ta.analyze_texture(dataset, comps)
    ref = old.analyze_texture(dataset, comps)
    assert new.component_fractions == ref.component_fractions
    # index and entropy are the kernel density's (the earlier implementation had an approximation from the fractions)
    from analysis.texture_odf import texture_index_and_entropy
    assert (new.texture_index, new.entropy) == texture_index_and_entropy(
        dataset, np.asarray(dataset.xmap.phase_id) == new.phase_id, halfwidth_deg=5.0)
    assert (new.plane_111_fraction, new.plane_100_fraction, new.plane_110_fraction) == \
        (ref.plane_111_fraction, ref.plane_100_fraction, ref.plane_110_fraction)


def test_the_angles_are_computed_once_for_fractions_split_and_map(world, counting):
    dataset, grains, comps = world
    dataset._component_angles = None
    ta.analyze_texture(dataset, comps)
    one_pass = len(counting)
    assert one_pass > 0
    del counting[:]
    ta.calculate_texture_split_by_rx(dataset, grains, [1, 2], comps, 10.0)
    ta.calculate_texture_component_map(dataset, comps, 10.0)
    ta.calculate_texture_component_fractions(dataset, comps, 7.0)
    assert counting == []                                   # not a single misorientation step again


def test_a_split_without_a_texture_step_computes_the_angles_once(world, counting):
    dataset, grains, comps = world
    dataset._component_angles = None
    ta.calculate_texture_split_by_rx(dataset, grains, [1, 2], comps, 10.0)
    one_pass = len(counting)
    assert one_pass > 0                                     # the old code ran this pass once per group
    ta.calculate_texture_split_by_rx(dataset, grains, [3], comps, 10.0)
    assert len(counting) == one_pass


def test_other_components_or_another_phase_geometry_compute_again(world, counting):
    dataset, _, comps = world
    dataset._component_angles = None
    ta.calculate_texture_component_fractions(dataset, comps, 10.0)
    first = len(counting)
    fewer = {k: v for k, v in comps.items() if k != "Cube"}
    ta.calculate_texture_component_fractions(dataset, fewer, 10.0)
    assert len(counting) > first


def test_a_quality_filter_invalidates_the_angles():
    dataset = _four_grain_map()
    comps = _components()
    before = ta.calculate_texture_component_fractions(dataset, comps, 10.0)
    cached = dataset._component_angles
    bc = np.asarray(dataset.band_contrast_raw())
    dataset.apply_quality_filter(bc_min=float(np.median(bc)))             # unindexes the lower half
    after = ta.calculate_texture_component_fractions(dataset, comps, 10.0)
    assert dataset._component_angles is not cached
    assert dataset._component_angles.min_angles.shape[1] == int(dataset.indexed_mask.sum()) < cached.min_angles.shape[1]
    assert after == old.calculate_texture_component_fractions(dataset, comps, 10.0)
    assert after != before


def test_other_orientations_invalidate_the_angles():
    dataset = _four_grain_map()
    comps = _components()
    ta.calculate_texture_component_fractions(dataset, comps, 10.0)
    cached = dataset._component_angles
    from orix.quaternion import Rotation
    n = dataset.size
    dataset.xmap._rotations = Rotation.from_axes_angles([0, 0, 1], np.deg2rad(np.linspace(0, 80, n)))
    after = ta.calculate_texture_component_fractions(dataset, comps, 10.0)
    assert dataset._component_angles is not cached
    assert after == old.calculate_texture_component_fractions(dataset, comps, 10.0)
