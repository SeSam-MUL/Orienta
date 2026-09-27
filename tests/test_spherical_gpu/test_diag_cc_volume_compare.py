"""Iter-15A: bin-by-bin compare EMSphInx cc volume vs our cc volume for Pixel 0.

EMSphInx cc volume layout (Bunge ZXZ): shape (135, 135, 135), axes
(Phi, phi2, phi1) with phi1 fastest. From the dumptopk_pixel0.json
"cc_bunge_flat" field.

Our cc volume nc_vol: shape (B, 2L-1, 2L-1, 2L-1). Axis decoder per
Tier1Indexer._decode_peak:
  alpha_zyz = ((a - off + size) % size) * scale
  Phi_zxz   = b * scale
  gamma_zyz = ((off - c + size) % size) * scale
  phi1_zxz  = (alpha + 3*pi/2 + pi/2) % 2pi = alpha
  phi2_zxz  = (gamma - pi/2) % 2pi
So (a, b, c) maps to ZXZ via the formula above.

To compare bin-by-bin, we transform our nc_vol into EMSphInx's Bunge
layout (Phi, phi2, phi1) and compute element-wise statistics.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

# Opens the reference oracle directly rather than through the `oracle`
# fixture, so it needs the same guard. Not in the 55 the Linux run
# reported (these were already red or skipped for other reasons there),
# but the same dependency, and it fails the same way in any checkout
# without the file.
from tests.data_deps import SPHERICAL_ORACLE, need, requires

pytestmark = requires(SPHERICAL_ORACLE)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _our_to_emsphinx_bunge(nc_vol: np.ndarray, L: int) -> np.ndarray:
    """Reorder our nc_vol[a, b, c] into EMSphInx Bunge layout
    (Phi_idx, phi2_idx, phi1_idx).

    From _decode_peak:
      alpha = ((a - off + size) % size) * scale
      Phi   = b * scale
      gamma = ((off - c + size) % size) * scale
      phi1 = (alpha + 3pi/2 + pi/2) % 2pi
      phi2 = (gamma - pi/2) % 2pi

    EMSphInx Bunge layout:
      cc_bunge[Phi_idx, phi2_idx, phi1_idx] where idx_x = round(angle * size / 2pi)

    Inverse map: for given (Phi_idx, phi2_idx, phi1_idx):
      Phi   = Phi_idx * 2pi/size
      phi1  = phi1_idx * 2pi/size
      phi2  = phi2_idx * 2pi/size
    Then convert ZXZ -> our (a, b, c):
      b = Phi_idx (direct)
      alpha = (phi1 - 2pi) mod 2pi = phi1 (since phi1 in [0, 2pi))
        but in our decode phi1 = (alpha + 2pi) mod 2pi = alpha mod 2pi
        so alpha = phi1
      a = ((alpha / scale) + off) mod size = ((phi1_idx) + off) mod size
      gamma = (phi2 + pi/2) mod 2pi
      c = (off - gamma/scale) mod size = (off - (phi2_idx + size/4)) mod size
        wait scale = 2pi/size, so phi2_idx = phi2 / scale
        gamma_idx = phi2_idx + (pi/2)/scale = phi2_idx + size/4
        c = (off - gamma_idx) mod size
    """
    size = 2 * L - 1
    off = L - 1
    out = np.zeros((size, size, size), dtype=nc_vol.dtype)
    # Map every (Phi_idx, phi2_idx, phi1_idx) to (a, b, c) in our layout
    for Phi_idx in range(size):
        b = Phi_idx
        for phi2_idx in range(size):
            # phi2 = phi2_idx * 2pi/size
            # gamma = phi2 + pi/2 = (phi2_idx + size/4) * 2pi/size, in our scale
            gamma_off = phi2_idx + size / 4.0
            # c = (off - gamma_off) mod size, but gamma_off may be fractional
            # The current decoder uses ((off - c + size) % size) * scale = gamma
            # so c = round((off - gamma/scale) mod size)
            c = int(round((off - gamma_off) % size)) % size
            for phi1_idx in range(size):
                # alpha = phi1 = phi1_idx * scale
                # a = (alpha/scale + off) mod size = (phi1_idx + off) mod size
                a = (phi1_idx + off) % size
                out[Phi_idx, phi2_idx, phi1_idx] = nc_vol[a, b, c]
    return out


def test_cc_volume_bin_by_bin_compare():
    from backend.spherical_gpu.pipeline.detector import DetectorGeometry
    from backend.spherical_gpu.pipeline.indexer import Tier1Indexer
    from backend.spherical_gpu.pipeline.sht_io import read_sht_master

    # Load EMSphInx cc volume from DumpTopK. This is a DIAGNOSTIC comparison
    # (no asserts) that needs a current EMSphInx top-k dump as its reference.
    # If that dump is absent or predates the cc_cube_size/cc_bunge_flat schema,
    # the diagnostic can't run — skip (don't fail) so the suite stays green;
    # regenerate the dump to re-enable the comparison.
    dump_path = PROJECT_ROOT / "tasks/dumptopk_pixel0.json"
    if not dump_path.is_file():
        pytest.skip(f"EMSphInx reference dump not present: {dump_path.name}")
    with open(dump_path) as f:
        dump = json.load(f)
    if "cc_cube_size" not in dump or "cc_bunge_flat" not in dump:
        pytest.skip(
            "tasks/dumptopk_pixel0.json predates the cc_cube_size/cc_bunge_flat "
            "schema — regenerate the EMSphInx top-k dump to run this diagnostic."
        )
    cube = int(dump["cc_cube_size"])
    em_cc = np.array(dump["cc_bunge_flat"], dtype=np.float64).reshape(cube, cube, cube)
    print(f"\nEMSphInx cc shape: {em_cc.shape}, range=[{em_cc.min():.4f}, {em_cc.max():.4f}]")
    print(f"  argmax (Phi, phi2, phi1) = "
          f"{np.unravel_index(em_cc.argmax(), em_cc.shape)}, "
          f"value={em_cc.max():.4f}")

    # Build our pipeline + cc volume for pixel 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oracle_path = PROJECT_ROOT / "tests/test_spherical_gpu/data/reference_oracle.h5"
    with h5py.File(oracle_path, "r") as f:
        meta = json.loads(f.attrs["meta_json"])
        sht_bytes = bytes(f["embedded_sht_bytes"][()])
    h5oina_path = need(PROJECT_ROOT / meta["h5oina_source"])
    with h5py.File(h5oina_path, "r") as f:
        pat = f["1/EBSD/Data/Processed Patterns"][0:1].astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix=".sht", delete=False)
    tmp.write(sht_bytes); tmp.close()
    detector_params = {k: meta[k] for k in (
        "n_rows", "n_cols", "pat_width", "pat_height", "pixel_size",
        "tilt", "binning", "step_x", "step_y", "pc_x", "pc_y", "pc_z", "vendor",
    )}
    geom = DetectorGeometry.from_params(detector_params, device=device)
    master = read_sht_master(tmp.name, device=device)
    indexer = Tier1Indexer(geom, master, device, bandwidth=meta["bandwidth"])

    pat_t = torch.from_numpy(pat).to(device).float()
    eulers, scores, nc_vol, _ = indexer._index_batch_with_volume(pat_t)
    L = indexer.bandwidth
    size = 2 * L - 1
    assert size == cube, f"size mismatch: ours={size}, em={cube}"
    our_cc = nc_vol[0].cpu().numpy().astype(np.float64)
    print(f"\nOur cc shape: {our_cc.shape}, range=[{our_cc.min():.4f}, {our_cc.max():.4f}]")
    a_a = np.unravel_index(our_cc.argmax(), our_cc.shape)
    print(f"  our argmax (a, b, c) = {a_a}, value={our_cc.max():.4f}")

    # Reorder our cc to EMSphInx Bunge layout
    our_cc_bunge = _our_to_emsphinx_bunge(our_cc, L)
    print(f"  our_cc_bunge argmax (Phi, phi2, phi1) = "
          f"{np.unravel_index(our_cc_bunge.argmax(), our_cc_bunge.shape)}, "
          f"value={our_cc_bunge.max():.4f}")

    # Compare element-wise on the OVERLAPPING bins (no padding for both)
    # Both are (cube, cube, cube). Compute correlation, scale, etc.
    em_flat = em_cc.flatten()
    our_flat = our_cc_bunge.flatten()
    r = float(np.corrcoef(em_flat, our_flat)[0, 1])
    # Linear fit em = a * ours + b
    A = np.vstack([our_flat, np.ones_like(our_flat)]).T
    coef, *_ = np.linalg.lstsq(A, em_flat, rcond=None)
    print(f"\n=== bin-by-bin EMSphInx vs ours (Bunge layout) ===")
    print(f"  Pearson correlation r = {r:.6f}")
    print(f"  Linear fit em = {coef[0]:.4f} * ours + {coef[1]:.6f}")
    print(f"  std EM/ours = {em_flat.std() / our_flat.std():.4f}")

    # Top-K agreement: are the K brightest cells in roughly the same locations?
    K = 50
    em_top_idx = np.argsort(-em_flat)[:K]
    our_top_idx = np.argsort(-our_flat)[:K]
    overlap = len(set(em_top_idx.tolist()) & set(our_top_idx.tolist()))
    print(f"  Top-{K} cell overlap: {overlap}/{K}")

    # Check value at EMSphInx argmax in our cc
    em_max_idx = em_cc.argmax()
    em_argmax = np.unravel_index(em_max_idx, em_cc.shape)
    em_val = float(em_cc[em_argmax])
    our_val_at_em = float(our_cc_bunge[em_argmax])
    print(f"\n  At EMSphInx argmax {em_argmax}:  EM={em_val:.4f}  Ours={our_val_at_em:.4f}")

    # Check value at our argmax in EMSphInx cc
    our_max_idx = our_cc_bunge.argmax()
    our_argmax = np.unravel_index(our_max_idx, our_cc_bunge.shape)
    em_val_at_our = float(em_cc[our_argmax])
    our_val = float(our_cc_bunge[our_argmax])
    print(f"  At our argmax     {our_argmax}:  EM={em_val_at_our:.4f}  Ours={our_val:.4f}")

    # Save artifacts
    np.savez(
        PROJECT_ROOT / "tasks/cc_volume_compare.npz",
        em_cc=em_cc, our_cc_bunge=our_cc_bunge,
    )
    print(f"\nArtifacts saved to tasks/cc_volume_compare.npz")
