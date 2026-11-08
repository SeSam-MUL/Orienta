"""Per-grain band contrast and EDX means read measurements only.

A closed not-indexed pixel carries its host grain's id, and real files store a
(usually low) band contrast and a chemistry value for it. Neither belongs to the
grain's mean.
"""
import numpy as np
import pytest
from orix.crystal_map import CrystalMap, PhaseList
from orix.quaternion import Rotation

from analysis.bc_analysis import grain_average_bc
from analysis.ebsd_dataset import EBSDDataset, GrainSet
from analysis.grain_analysis import reconstruct_grains
from analysis import rx_analysis

NY = NX = 12


def _scratch_map():
    """One grain, a closed 2 x 6 scratch; BC and Fe are 200 / 10 on the indexed pixels
    and 10 / 100 in the scratch."""
    scratch = np.zeros((NY, NX), bool)
    scratch[5:7, 3:9] = True
    pid = np.where(scratch, -1, 0).reshape(-1)
    q = np.tile(Rotation.identity(1).data, (NY * NX, 1))
    bc = np.where(scratch, 10.0, 200.0).reshape(-1)
    fe = np.where(scratch, 100.0, 10.0).reshape(-1)
    xmap = CrystalMap(rotations=Rotation(q), phase_id=pid,
                      x=np.tile(np.arange(NX, dtype=float), NY),
                      y=np.repeat(np.arange(NY, dtype=float), NX),
                      phase_list=PhaseList(names=["Al"], point_groups=["m-3m"]),
                      prop={"bc": bc, "Fe": fe})
    ds = EBSDDataset(xmap, step_size=1.0)
    grains = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=1, fill=False)
    return ds, grains, scratch


def test_the_scratch_is_closed_into_the_one_grain():
    _, grains, scratch = _scratch_map()
    assert grains.n_grains == 1
    assert grains.closed_mask[scratch].all()
    assert (grains.grain_ids[scratch] == 1).all()


def test_grain_average_bc_ignores_closed_pixels():
    ds, grains, _ = _scratch_map()
    g_bc, ids = grain_average_bc(ds, grains)
    assert ids.tolist() == [1]
    assert g_bc[0] == 200.0          # 184.17 when the closed pixels are counted


def test_grain_average_bc_fallback_ignores_closed_pixels(monkeypatch):
    import analysis.bc_analysis as bca
    monkeypatch.setattr(bca, "SCIPY_AVAILABLE", False)
    ds, grains, _ = _scratch_map()
    g_bc, _ = grain_average_bc(ds, grains)
    assert g_bc[0] == 200.0


def test_hand_built_grainset_without_closed_mask_counts_every_labelled_pixel():
    ds, grains, _ = _scratch_map()
    by_hand = GrainSet(grain_ids=grains.grain_ids, n_grains=1)
    g_bc, _ = grain_average_bc(ds, by_hand)
    assert g_bc[0] == pytest.approx(np.mean(ds.bc))


def test_grain_average_edx_ignores_closed_pixels():
    ds, grains, _ = _scratch_map()
    g_fe = rx_analysis.calculate_grain_average_edx(ds.edx_channels["Fe"], grains)
    assert g_fe.tolist() == [10.0]    # 22.5 with the scratch counted


def test_analyze_recrystallization_hands_the_measured_fe_mean_to_the_classifier(monkeypatch):
    ds, grains, _ = _scratch_map()
    seen = {}

    def spy(grains_, dataset_, g_kam, g_bc, *args, **kw):
        seen["g_fe"] = args[-1]
        seen["g_bc"] = g_bc
        return np.array([], dtype=int)

    monkeypatch.setattr(rx_analysis, "classify_rx_grains", spy)
    kam = np.zeros((NY, NX))
    rx_analysis.analyze_recrystallization(ds, grains, kam, edx_fe_channel="Fe")
    assert seen["g_fe"].tolist() == [10.0]
    assert seen["g_bc"].tolist() == [200.0]
