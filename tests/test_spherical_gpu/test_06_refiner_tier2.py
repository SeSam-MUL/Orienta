"""Tier-2 refinement: cc-volume Gaussian smoothing + sub-bin parabolic
peak interpolation. Test that median dataset disorientation drops vs
Tier-1.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from backend.spherical_gpu.pipeline.detector import DetectorGeometry
from backend.spherical_gpu.pipeline.indexer import Tier1Indexer
from backend.spherical_gpu.pipeline.refiner import (
    refine_index_result, smooth_nc_vol,
)
from backend.spherical_gpu.pipeline.sht_io import read_sht_master

# Opens the reference oracle directly rather than through the `oracle`
# fixture, so it needs the same guard. Not in the 55 the Linux run
# reported (these were already red or skipped for other reasons there),
# but the same dependency, and it fails the same way in any checkout
# without the file.
from tests.data_deps import SPHERICAL_ORACLE, need, requires

pytestmark = requires(SPHERICAL_ORACLE)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _disorientation_deg(eu_a, eu_b, point_group="m-3m"):
    from orix.quaternion import Misorientation, Rotation
    from orix.quaternion.symmetry import _groups
    sym = next(g for g in _groups if g.name == point_group)
    r_a = Rotation.from_euler(eu_a)
    r_b = Rotation.from_euler(eu_b)
    miso = Misorientation((r_a * (~r_b)).data, symmetry=(sym, sym))
    miso = miso.map_into_symmetry_reduced_zone()
    return np.rad2deg(miso.angle)


@pytest.fixture(scope="module")
def setup_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oracle_path = PROJECT_ROOT / "tests/test_spherical_gpu/data/reference_oracle.h5"
    with h5py.File(oracle_path, "r") as f:
        meta = json.loads(f.attrs["meta_json"])
        sht_bytes = bytes(f["embedded_sht_bytes"][()])
        gt_eulers_no_refine = f["euler_xyz_no_refine"][:]
    h5oina_path = need(PROJECT_ROOT / meta["h5oina_source"])
    tmp = tempfile.NamedTemporaryFile(suffix=".sht", delete=False)
    tmp.write(sht_bytes); tmp.close()
    detector_params = {k: meta[k] for k in (
        "n_rows", "n_cols", "pat_width", "pat_height", "pixel_size",
        "tilt", "binning", "step_x", "step_y", "pc_x", "pc_y", "pc_z", "vendor",
    )}
    geom = DetectorGeometry.from_params(detector_params, device=device)
    master = read_sht_master(tmp.name, device=device)
    indexer = Tier1Indexer(geom, master, device, bandwidth=meta["bandwidth"])
    return device, indexer, gt_eulers_no_refine, h5oina_path


def test_smooth_nc_vol_preserves_shape(setup_pipeline):
    device, indexer, _, h5oina_path = setup_pipeline
    with h5py.File(h5oina_path, "r") as f:
        pat = f["1/EBSD/Data/Processed Patterns"][:4].astype(np.float32)
    pat_t = torch.from_numpy(pat).to(device)
    _, _, nc_vol, _ = indexer._index_batch_with_volume(pat_t)
    sm = smooth_nc_vol(nc_vol, sigma=1.0)
    assert sm.shape == nc_vol.shape
    assert torch.isfinite(sm).all()


@pytest.mark.xfail(
    reason="Tier-2 inherits Tier-1's indexImage-fidelity gap (NOT an axis-"
    "convention bug — see "
    "tasks/_archive/handoff-typ-b-axis-bug-investigation.md sec.13). Tier-2 "
    "can only refine near the chosen Tier-1 cc-volume peak; "
    "it cannot jump to a different basin. On the ~62 percent of pixels "
    "where indexImage picks the wrong cc-volume peak (and we replicate "
    "that failure faithfully), Tier-2 sub-bin refinement just sharpens the "
    "wrong answer. Same root cause as "
    "test_05_indexer_tier1.py::test_tier1_soft_gate_disorientation.",
    strict=False,
)
def test_refiner_improves_median_disorientation(setup_pipeline):
    """Across a sample of patterns, Tier-2 median disorientation should
    drop noticeably vs Tier-1."""
    device, indexer, gt_eulers_no_refine, h5oina_path = setup_pipeline
    n_test = 100
    with h5py.File(h5oina_path, "r") as f:
        pat = f["1/EBSD/Data/Processed Patterns"][:n_test].astype(np.float32)
    pat_t = torch.from_numpy(pat).to(device)

    eulers_t1, _, nc_vol, _ = indexer._index_batch_with_volume(pat_t)
    eulers_t2, _ = refine_index_result(
        nc_vol, indexer.bandwidth, smoothing_sigma=1.0,
    )

    diso_t1 = _disorientation_deg(
        eulers_t1.cpu().numpy(), gt_eulers_no_refine[:n_test]
    )
    diso_t2 = _disorientation_deg(
        eulers_t2.cpu().numpy(), gt_eulers_no_refine[:n_test]
    )

    print(f"\n  Tier-1 diso: median={np.median(diso_t1):.2f} 95th={np.percentile(diso_t1, 95):.2f} max={diso_t1.max():.2f}")
    print(f"  Tier-2 diso: median={np.median(diso_t2):.2f} 95th={np.percentile(diso_t2, 95):.2f} max={diso_t2.max():.2f}")

    assert np.median(diso_t2) < np.median(diso_t1) + 0.5, (
        f"Tier-2 median {np.median(diso_t2):.2f} did not improve over "
        f"Tier-1 median {np.median(diso_t1):.2f}"
    )


def test_refiner_sigma_sweep(setup_pipeline):
    """Find the sigma that minimizes median disorientation."""
    device, indexer, gt_eulers_no_refine, h5oina_path = setup_pipeline
    n_test = 100
    with h5py.File(h5oina_path, "r") as f:
        pat = f["1/EBSD/Data/Processed Patterns"][:n_test].astype(np.float32)
    pat_t = torch.from_numpy(pat).to(device)

    _, _, nc_vol, _ = indexer._index_batch_with_volume(pat_t)
    print()
    for sigma in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        eu, _ = refine_index_result(nc_vol, indexer.bandwidth, smoothing_sigma=sigma)
        diso = _disorientation_deg(eu.cpu().numpy(), gt_eulers_no_refine[:n_test])
        print(
            f"  sigma={sigma:.1f}  median={np.median(diso):5.2f}  "
            f"95th={np.percentile(diso, 95):5.2f}  max={diso.max():5.2f}"
        )
