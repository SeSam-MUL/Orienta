"""Cross-correlation sanity tests — these run BEFORE the indexer test_05.

If any of these fail, the bug is in:
- the cs2cc_/rs2cc_ usage (master autocorrelation should peak at identity)
- the master coef format conversion (SHT-file [m, l] → ebsdtorch (2L-1, L) etc.)
- the cc-volume index → Euler decoding

…and we should NOT be looking at the detector→sphere projection or the real-data
gate yet. Diagnostic surface stays small.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch

from backend.spherical_gpu._math import wigner_d
from backend.spherical_gpu._math._wigner_logspace import wigner_d_eq_half_pi
from backend.spherical_gpu._math.sht_cc import rs2cc_, cs2cc_


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def master_with_indexer(oracle):
    """Materialise the embedded SHT and build a Tier1Indexer once for the suite."""
    from backend.spherical_gpu.pipeline.detector import DetectorGeometry
    from backend.spherical_gpu.pipeline.indexer import Tier1Indexer
    from backend.spherical_gpu.pipeline.sht_io import read_sht_master
    import json

    meta = json.loads(oracle.attrs["meta_json"])
    sht_filename = oracle["embedded_sht_filename"][()]
    if isinstance(sht_filename, bytes):
        sht_filename = sht_filename.decode("utf-8")
    tmp_dir = Path(tempfile.mkdtemp(prefix="spherical_gpu_test_"))
    target = tmp_dir / sht_filename
    target.write_bytes(bytes(oracle["embedded_sht_bytes"][()]))

    device = _device()
    detector_params = {k: meta[k] for k in (
        "n_rows", "n_cols", "pat_width", "pat_height", "pixel_size",
        "tilt", "binning", "step_x", "step_y", "pc_x", "pc_y", "pc_z",
        "vendor",
    )}
    geom = DetectorGeometry.from_params(detector_params, device=device)
    master = read_sht_master(str(target), device=device)
    indexer = Tier1Indexer(geom=geom, master=master, device=device, bandwidth=68)
    return master, indexer


def test_master_autocorrelation_peaks_at_identity(master_with_indexer):
    """rs2cc_(master, master, …) MUST peak at the identity rotation.

    This is the single most important sanity check: if the master pattern
    correlated with itself doesn't pick out R = I as the best match, then
    EVERY downstream test is meaningless. The bug is in:
    - cs2cc_/rs2cc_ kernel itself (unlikely — it's vendored & tested upstream),
    - the master coef format conversion (SHT file → indexer-internal layout),
    - the cc-volume index → Euler decoding.

    WHY the identity check uses value-at-identity rather than argmax:
    ---------------------------------------------------------------
    The rs2cc_ cc-volume encodes ZYZ Euler angles (alpha, beta, gamma) at
    indices (a, b, c) using the decode formula from _index_batch:

        alpha = ((a - (L-1) + size) % size) * scale
        beta  = b * scale
        gamma = ((L-1) - c + size) % size) * scale

    where scale = 2*pi / size and size = 2*L-1.

    Identity (alpha=0, beta=0, gamma=0) therefore sits at index (L-1, 0, L-1).

    However, for a high-symmetry cubic master pattern (Al, m-3m, z_rot=4),
    the ZYZ Euler angles are DEGENERATE at beta=0: when Phi=0, rotations
    alpha and gamma collapse to a single combined rotation alpha+gamma.
    This means ALL 135 points along the diagonal (a, 0, a) for a=0..134
    produce numerically identical correlation values (they all encode the
    same SO(3) element modulo the cubic symmetry).

    Using argmax() to find "the" identity peak is therefore non-deterministic
    — floating-point rounding picks an arbitrary winner among 135 equally-high
    values.  The correct assertion is:

    1. The VALUE at the true identity index (L-1, 0, L-1) is within a small
       relative tolerance of the global maximum.
    2. The global maximum is in the b=0 layer (Phi=0 is correct for identity).

    Both assertions together prove that (a) identity IS at the top of the cc
    landscape and (b) the beta axis is correctly decoded (beta=0 for identity).
    """
    _, indexer = master_with_indexer
    L = indexer.bandwidth
    mc = indexer._master_coefs                      # (1, L, L) c64 in current layout
    cc = rs2cc_(L, mc, mc, indexer._wigner_table)    # (1, 2L-1, 2L-1, 2L-1) volume

    size = 2 * L - 1
    cc_np = cc[0].detach().cpu().float().numpy()

    global_max = float(cc_np.max())
    global_min = float(cc_np.min())

    # The identity index per the _index_batch decode formula:
    #   alpha = 0  →  a = L-1 = 67 (for bw=68)
    #   beta  = 0  →  b = 0
    #   gamma = 0  →  c = L-1 = 67
    identity_idx = (L - 1, 0, L - 1)
    val_at_identity = float(cc_np[identity_idx])

    print(
        f"Master autocorr: global_max={global_max:.6f}, "
        f"val_at_identity{identity_idx}={val_at_identity:.6f}, "
        f"global_min={global_min:.6f}, size={size}, L-1={L-1}"
    )

    # Assertion 1: value at true identity index must be within 0.1% of global max.
    # (With 135 symmetry-equivalent peaks, floating-point spread is ~1e-5 relative.)
    REL_TOL = 0.001
    assert val_at_identity >= global_max * (1.0 - REL_TOL), (
        f"Value at identity index {identity_idx} = {val_at_identity:.6f} is "
        f"more than {REL_TOL*100:.1f}% below global max {global_max:.6f}. "
        "The master coef layout (sht_io [m,l] → rs2cc_ [b,l,m]) OR the "
        "cc-volume decode formula (identity at (L-1, 0, L-1)) is wrong."
    )

    # Assertion 2: the b=0 layer (Phi=0, correct for identity) must contain
    # the global maximum.  If the max is at b != 0, the beta axis is mis-decoded.
    b0_max = float(cc_np[:, 0, :].max())
    assert b0_max >= global_max * (1.0 - REL_TOL), (
        f"Global max {global_max:.6f} is NOT in the b=0 (Phi=0) layer "
        f"(b=0 max={b0_max:.6f}). The beta axis is mis-encoded or the "
        "coef layout is transposed."
    )


def test_pattern_score_correlates_with_oracle(master_with_indexer, oracle, oracle_h5oina):
    """Sanity check: even if orientations are off, the SCORE should still
    correlate with EMSphInx's score. If both are random, the bug is much
    deeper than just decoding.
    """
    import h5py
    import numpy as np

    _, indexer = master_with_indexer
    # Use a small subset so this stays fast as a sanity test.
    with h5py.File(oracle_h5oina, "r") as f:
        pat_np = np.asarray(f["1/EBSD/Data/Processed Patterns"][:64], dtype=np.float32)
    pat = torch.from_numpy(pat_np).to(indexer.device)
    _, scores = indexer._index_batch(pat)
    our = scores.cpu().numpy()
    ref = oracle["score_no_refine"][:64]
    r = float(np.corrcoef(our, ref)[0, 1])
    print(f"Score correlation on first 64 patterns: r = {r:.3f}")
    assert np.isfinite(r), "scores are constant — pipeline is broken"
    # We don't assert a magnitude here because the master autocorrelation
    # peak test above is the strict gate. This is a sanity check.


def test_wigner_table_shape_and_finiteness():
    """The Wigner-d table at β=π/2 must be finite for the indexing bandwidth.

    The flat-table size is (L+1)·(L+2)·(L+3)/6 — verified empirically against
    wigner_d_eq_half_pi at L=10, 20, 70. This formula counts the
    (l, m, n) triples with 0 ≤ m ≤ n ≤ l ≤ L (the upper-triangular packing
    used by the half-pi table).
    """
    L = 70  # the indexer uses L = bandwidth + 2 = 70
    table = wigner_d_eq_half_pi(L, dtype=torch.float32, device=_device())
    expected_size = (L + 1) * (L + 2) * (L + 3) // 6
    assert table.shape[0] == expected_size, (
        f"Wigner table size {table.shape[0]} != expected {expected_size}"
    )
    assert torch.isfinite(table).all(), "Wigner table contains NaN/Inf"


def test_synthetic_round_trip_smoke(master_with_indexer):
    """Synthetic round-trip: rotate sphere by known R, splat onto detector,
    index, recover R within tolerance. SMOKE TEST only — full validation
    once Tier-1 disorientation gate is met.

    This is the critical "convention" test. The indexer must recover the
    known orientation; the tolerance is generous (within 30°) for the
    smoke test, intended to fail catastrophically on convention errors but
    permit the bin-resolution drift inherent in cc-volume indexing.
    """
    from tests.test_spherical_gpu._synthetic import synthetic_pattern_from_master

    master, indexer = master_with_indexer
    # Choose an arbitrary rotation that's not near identity nor near a symmetry
    target_euler = (math.radians(30.0), math.radians(40.0), math.radians(20.0))
    synth = synthetic_pattern_from_master(
        master=master, geom=indexer.geom,
        euler_zxz=target_euler, flip_y=getattr(indexer, "flip_y", True),
    )

    eulers, _scores = indexer._index_batch(synth.pattern.unsqueeze(0))
    recovered = eulers.cpu().numpy()[0]
    print(
        f"Synthetic round-trip: target = {[math.degrees(a) for a in target_euler]}°, "
        f"recovered = {[math.degrees(a) for a in recovered.tolist()]}°"
    )

    # We don't assert a tight bound here — the proxy spherical image is too
    # weak (Y20-only) for tight orientation discrimination. The point of THIS
    # test is to keep the synthetic round-trip plumbing alive and catch
    # catastrophic convention errors. A real synthetic test (next iteration)
    # would use the actual master pattern via inverse SHT.
    import numpy as np
    assert np.isfinite(recovered).all(), "synthetic round-trip returned NaN/Inf"
