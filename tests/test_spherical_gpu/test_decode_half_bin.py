"""The cc-volume decode must place the half turn at pi, not at floor(size/2) bins.

Background (2026-09-10, tasks/fidelity/03_fix_round2.md)
--------------------------------------------------------
The alpha and gamma axes of the SO(3) cc volume carry a half-turn offset.  All
four copies of the decode expressed it as an INTEGER number of bins -- ``size //
2`` in ``indexer._decode_peak`` / ``_decode_peak_skip_vol``, ``L - 1`` in
``refiner.decode_cell_to_zxz`` and ``resolution._decode_bins`` -- but the axis
length ``size = 2L - 1`` is odd, so half a turn is ``size / 2`` bins (87.5 at
L = 88).  The floor was short by exactly half a bin on alpha and on gamma; beta
has no offset and was correct, which is how the defect was localised
(``tasks/fidelity/exp_r2_subbin.py``: the ground-truth cell sat at argmax -0.51
bins on alpha, -0.49 on gamma, -0.03 on beta).

Cost: a ~1.6 deg orientation error at L = 88 that scaled with the bin (1.4 deg
at L = 68, 1.2 deg at L = 128), on EVERY phase and every bandwidth -- it is
independent of symmetry, unlike the crystal-frame C2<1 -1 0> of
``test_m3_variant_selection.py``.  It is why the raw spherical answer trailed
EMSphInx by 1.7 deg after that fix (tasks/fidelity/04_verify.md criterion 2).
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from backend.spherical_gpu.pipeline._frame import (apply_frame_fix_zxz,
                                                   decode_cells_to_zxz)


def _origin_triple():
    """The emitted triple for (alpha, beta, gamma) = (0, 0, 0).

    Written out rather than imported so the test states the convention it is
    pinning: ZYZ -> ZXZ contributes phi1 += pi/2 + 3pi/2 and phi2 -= pi/2, and
    the crystal-frame constant is applied last.
    """
    two_pi = 2.0 * math.pi
    phi1 = ((0.0 + math.pi / 2.0) % two_pi + 1.5 * math.pi) % two_pi
    phi2 = (0.0 - math.pi / 2.0) % two_pi
    return apply_frame_fix_zxz(phi1, 0.0, phi2)


def _circ(a, b):
    d = abs(float(a) - float(b)) % (2.0 * math.pi)
    return min(d, 2.0 * math.pi - d)


@pytest.mark.parametrize("L", [8, 68, 88, 128])
def test_axis_midpoint_is_the_decode_origin(L):
    """alpha = gamma = 0 must sit at cell ``size / 2`` -- a HALF-INTEGER cell,
    because ``size = 2L - 1`` is odd, i.e. not at any sampled bin.

    This is the property the integer floor could not have.  With the shipped
    ``off = L - 1 = size // 2`` the origin sat on a bin and the midpoint
    decoded to half a bin of alpha and of gamma; see
    ``test_floored_origin_is_exactly_half_a_bin_off`` below for that number.
    """
    size = 2 * L - 1
    mid = size / 2.0
    got = decode_cells_to_zxz(np.array([mid]), np.array([0.0]),
                              np.array([mid]), L)
    for g, w, name in zip(got, _origin_triple(), ("phi1", "Phi", "phi2")):
        d = _circ(g[0], w)
        assert d < 1e-12, (
            f"L={L}: decoding cell size/2 = {mid} gives {name}="
            f"{float(g[0]):.9f}, expected the origin value {float(w):.9f} "
            f"(off by {math.degrees(d):.4f} deg) -- the decode's half turn is "
            f"not at size/2 bins")


@pytest.mark.parametrize("L", [68, 88, 128])
def test_floored_origin_is_exactly_half_a_bin_off(L):
    """Pin the size of the regression: decoding at the FLOORED midpoint
    ``size // 2`` -- what every copy of the decode used until 2026-09-10 --
    misses the origin by exactly half a bin on alpha and on gamma, and not at
    all on beta.  1.03 deg each at L = 88."""
    size = 2 * L - 1
    scale = 2.0 * math.pi / size
    floored = float(size // 2)
    assert size / 2.0 - floored == 0.5
    got = decode_cells_to_zxz(np.array([floored]), np.array([0.0]),
                              np.array([floored]), L)
    want = _origin_triple()
    for g, w, name, expect in ((got[0], want[0], "phi1", 0.5 * scale),
                               (got[1], want[1], "Phi", 0.0),
                               (got[2], want[2], "phi2", 0.5 * scale)):
        d = _circ(g[0] if np.ndim(g) else g, w)
        assert abs(d - expect) < 1e-12, (
            f"L={L}: floored origin misses {name} by {math.degrees(d):.4f} deg, "
            f"expected {math.degrees(expect):.4f}")


# ----------------------------------------------------------------------
# Closed loop on real masters -- the mechanism, not one phase.
# ----------------------------------------------------------------------
_DB = Path(__file__).resolve().parents[2] / "Database" / "EBSD_SHT_Database"

# (tag, sht relative to _DB, orix Laue group name, z_rot)
_MASTERS = [
    ("m-3   sd_0302719",
     "sd_0302719/Mn0.5Fe0.5Al5Si0.68 (sd_0302719) [cI168] {20kV}.sht", "Th"),
    ("m-3m  Al", "Al/Al (Al) [cF4] {20kV}.sht", "Oh"),
    ("mmm   Al6Fe",
     "Al/Al6Fe (Al6Fe_mp-570001_symmetrized) {20kV}.sht", "D2h"),
]

# Measured after the fix at L=88: 0.178 / 0.258 / 0.290 deg.  Before it:
# 1.601 / 1.548 / 1.599 -- the half bin is 1.03 deg on alpha AND on gamma.
_TOL_DEG = 0.8
_BANDWIDTH = 88

_QUATS = np.array([
    [0.6320, 0.2571, -0.5305, 0.5019],
    [0.7409, -0.4160, 0.2380, -0.4694],
    [0.5527, 0.5905, 0.4460, 0.3839],
])
_QUATS /= np.linalg.norm(_QUATS, axis=1, keepdims=True)


@pytest.mark.parametrize("tag,rel,group_name", _MASTERS)
def test_closed_loop_lands_well_inside_the_bin(tag, rel, group_name):
    """Render at a known orientation, index it, get it back to a small
    FRACTION of the cc bin -- not to ~0.8 of it.

    Reaches into Database/EBSD_SHT_Database (the conftest discourages it) for
    the same reason test_m3_variant_selection.py does: the bundled oracle
    master is a single phase and this test is about the decode being wrong for
    every phase at every bandwidth.  Skips cleanly when the database is absent.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA: renders + indexes three 128x156 patterns")
    sht = Path(os.environ.get("ORIENTA_TEST_SHT_DB", str(_DB))) / rel
    if not sht.is_file():
        pytest.skip(f"master not available at {sht}")

    from orix.quaternion import Rotation, Orientation
    import orix.quaternion.symmetry as sym

    from tests.test_spherical_gpu.test_m3_variant_selection import DET, _render
    from backend.spherical_gpu.backend import (BackendConfig, PhaseConfig,
                                               SphericalGPUBackend)

    pats = np.stack([_render(str(sht), q) for q in _QUATS])
    bk = SphericalGPUBackend(BackendConfig(phases=[
        PhaseConfig(sht_file=str(sht), bandwidth=_BANDWIDTH, refine=True)]))
    res = bk.index_array(np.ascontiguousarray(pats, dtype=np.float32), DET)
    eu = np.asarray(res.euler_xyz.numpy(), dtype=np.float64).reshape(-1, 3)
    q = np.asarray(Rotation.from_euler(eu).data, dtype=np.float64)

    group = getattr(sym, group_name)
    ang = np.asarray(
        Orientation(Rotation(q), symmetry=group).angle_with(
            Orientation(Rotation(_QUATS), symmetry=group), degrees=True),
        dtype=float).ravel()
    bin_deg = 360.0 / (2 * _BANDWIDTH - 1)
    assert float(np.median(ang)) < _TOL_DEG, (
        f"{tag}: closed loop lands {np.round(ang, 3)} deg from the rendered "
        f"orientation (bin = {bin_deg:.3f} deg, tolerance {_TOL_DEG}). A median "
        f"near 0.8 * bin means the decode's half turn is back to an integer "
        f"number of bins -- see backend/spherical_gpu/pipeline/_frame.py")
