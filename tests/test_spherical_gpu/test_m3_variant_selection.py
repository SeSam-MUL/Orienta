"""Closed loop on an m-3 master: render, index, get the SAME orientation back.

Background (2026-09-09, tasks/spherical-indexer-fidelity-2026-09-09.md and
tasks/fidelity/01_reproduce.md)
-----------------------------------------------------------------------
On the SampleB alpha particle Orienta's raw spherical answer is, pixel for
pixel, the 90-degree cubic coset partner of what EMSphInx returns from the SAME
master (20/20 sampled pixels: ~88.4 deg apart under m-3, ~1.8 deg under m-3m).
That is reproducible WITHOUT any measured data and without the resolver: render
an m-3 master at a known orientation through Orienta's own forward model
(sht_pattern_renderer.PatternRenderer) and feed the synthetic pattern straight
into Orienta's own Tier1 indexer (SphericalGPUBackend.index_array). The indexer
returns the 90-degree partner, 8/8 on both alpha masters, while the same loop on
the m-3m Al master closes to within the L=88 bin quantisation (<= 1.9 deg).

Read the numbers this way: the two renders are visibly DIFFERENT images (pixel
NCC 0.20-0.39), and at the true orientation Orienta's own cc volume sits at
0.20-0.26 (noise floor) while the partner peaks at 0.46-0.72. So this is not a
near-tie the indexer loses -- the forward model and the correlation chain
disagree about the crystal frame by a 90-degree cube rotation.

Why this test reaches into Database/EBSD_SHT_Database (the conftest asks tests
not to): the defect only exists for point group m-3, and the SHT embedded in the
reference oracle is Al, m-3m -- it cannot express the bug. The test skips
cleanly when the alpha master is absent, and the path is overridable via
ORIENTA_TEST_M3_SHT.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


# SampleB geometry, verbatim from the EMSphInx export used all evening
# (tasks/fidelity/data_s11/emsphinx_s11.h5, attr 'detector_geometry').
DET = {
    "pc_x": 0.5026211263239384,
    "pc_y": 0.3272601428659012,
    "pc_z": 0.8458379050406317,
    "pat_width": 156,
    "pat_height": 128,
    "n_cols": 120,
    "n_rows": 90,
    "pixel_size": 70.0,
    "tilt": 4.322593688964844,
    "binning": 1,
    "step_x": 0.5,
    "step_y": 0.5,
    "vendor": "Bruker",
    "sample_tilt": 70.00314331054688,
}

_DEFAULT_M3_SHT = (
    Path(__file__).resolve().parents[2]
    / "Database" / "EBSD_SHT_Database" / "sd_0302719"
    / "Mn0.5Fe0.5Al5Si0.68 (sd_0302719) [cI168] {20kV}.sht"
)

# Three FIXED orientations (w, x, y, z), generic -- none of them near a cube
# axis, so a 90-degree partner can never be mistaken for the orientation itself.
FIXED_QUATS = np.array([
    [0.6320, 0.2571, -0.5305, 0.5019],
    [0.7409, -0.4160, 0.2380, -0.4694],
    [0.5527, 0.5905, 0.4460, 0.3839],
])
FIXED_QUATS /= np.linalg.norm(FIXED_QUATS, axis=1, keepdims=True)

BANDWIDTH = 88
TOL_DEG = 3.0


def _m3_sht() -> str:
    p = Path(os.environ.get("ORIENTA_TEST_M3_SHT", str(_DEFAULT_M3_SHT)))
    if not p.is_file():
        pytest.skip(f"m-3 alpha master not available at {p} "
                    "(set ORIENTA_TEST_M3_SHT to point at one)")
    return str(p)


def _render(sht_path: str, quat: np.ndarray) -> np.ndarray:
    from backend.api.services import sht_pattern_renderer as svc
    from backend.spherical_gpu.pipeline.detector import convert_pc_to_emsoft

    H, W = int(DET["pat_height"]), int(DET["pat_width"])
    xpc, ypc, L_um = convert_pc_to_emsoft(
        pc=(DET["pc_x"], DET["pc_y"], DET["pc_z"]), vendor=DET["vendor"],
        pat_width=W, pat_height=H, pixel_size=DET["pixel_size"],
        binning=DET["binning"],
    )
    rnd = svc.get_renderer()
    grid = svc.load_or_get_phase(sht_path, max_bandwidth=128)
    qt = torch.tensor(np.asarray(quat, dtype=np.float64)[:4], dtype=torch.float64)
    with torch.no_grad():
        sim = rnd.render(grid, qt, (float(xpc), float(ypc), float(L_um)), (H, W),
                         DET["pixel_size"], tilt_deg=DET["sample_tilt"],
                         det_tilt_deg=DET["tilt"])
    return np.asarray(sim.numpy(), dtype=np.float32)


def _index(sht_path: str, patterns: np.ndarray) -> np.ndarray:
    from orix.quaternion import Rotation

    from backend.spherical_gpu.backend import (BackendConfig, PhaseConfig,
                                               SphericalGPUBackend)
    bk = SphericalGPUBackend(BackendConfig(phases=[
        PhaseConfig(sht_file=sht_path, bandwidth=BANDWIDTH, refine=True)]))
    res = bk.index_array(np.ascontiguousarray(patterns, dtype=np.float32), DET)
    eu = np.asarray(res.euler_xyz.numpy(), dtype=np.float64).reshape(-1, 3)
    return np.asarray(Rotation.from_euler(eu).data, dtype=np.float64)


def _angle_deg(qa, qb, group):
    """Misorientation in degrees under `group`, via Orientation.angle_with --
    never a hand formula (lessons.md 2026-09-09: symmetry acts on the LEFT and
    a hand-rolled reduction confirmed its own error)."""
    from orix.quaternion import Orientation, Rotation
    a = Orientation(Rotation(np.atleast_2d(qa)), symmetry=group)
    b = Orientation(Rotation(np.atleast_2d(qb)), symmetry=group)
    return np.asarray(a.angle_with(b, degrees=True), dtype=float)


@pytest.fixture(scope="module")
def closed_loop():
    """Render the m-3 master at the three fixed orientations, index the
    synthetic patterns, return (rendered quats, indexed quats)."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA: renders + indexes three 128x156 patterns")
    sht = _m3_sht()
    pats = np.stack([_render(sht, q) for q in FIXED_QUATS])
    return FIXED_QUATS, _index(sht, pats)


def test_m3_closed_loop_returns_the_rendered_orientation(closed_loop):
    """THE defect. Currently returns the 90-degree partner on all three."""
    from orix.quaternion import symmetry as sym
    q_ren, q_idx = closed_loop
    ang = _angle_deg(q_ren, q_idx, sym.Th)          # m-3
    assert np.all(ang < TOL_DEG), (
        "Tier1 spherical indexing does not return the orientation its own "
        f"forward model rendered: angles under m-3 = {np.round(ang, 2)} deg "
        f"(tolerance {TOL_DEG}). Partner check under m-3m: "
        f"{np.round(_angle_deg(q_ren, q_idx, sym.Oh), 2)} deg -- near zero "
        "there means the answer is the 90-degree cubic coset partner."
    )


def test_m3_closed_loop_is_right_up_to_the_cubic_coset(closed_loop):
    """Guard rail, passes today: whatever else changes, the answer must stay
    inside the cubic coset of the rendered orientation. If this ever fails the
    frame calibration itself moved and the test above means something else."""
    from orix.quaternion import symmetry as sym
    q_ren, q_idx = closed_loop
    ang = _angle_deg(q_ren, q_idx, sym.Oh)          # m-3m
    assert np.all(ang < TOL_DEG), (
        f"closed loop broke outside the cubic coset: {np.round(ang, 2)} deg "
        "under m-3m -- not the 90-degree-partner defect but something worse"
    )
