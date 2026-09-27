"""Iter-14 final bisection: compare our pipeline's stage-1 sphere image
against EMSphInx WSL-DumpTopK's BackProjector::unproject() output.

Procedure:
1. Load EMSphInx sphere from tasks/dumptopk_pixel0.json (5041 north cells).
2. Replicate stage 1 of our `_direct_sht_coefs` for the same Pixel 0:
   - Preprocess pattern (gausbckg + nregions)
   - Sample at GL inside cells (north + south)
   - Normalize over combined (north + south) statistics
   - Scatter into a (dim, dim) buffer; south cells ADD to north idx
3. Compare cell-by-cell. Find first significant mismatch.

Output: tasks/sphere_compare.log with bin-by-bin diff stats.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

# Opens the reference oracle directly rather than through the `oracle`
# fixture, so it needs the same guard. Not in the 55 the Linux run
# reported (these were already red or skipped for other reasons there),
# but the same dependency, and it fails the same way in any checkout
# without the file.
from tests.data_deps import SPHERICAL_ORACLE, need, requires

pytestmark = requires(SPHERICAL_ORACLE)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_sphere_image_compare_emsphinx():
    from backend.spherical_gpu.pipeline.detector import DetectorGeometry
    from backend.spherical_gpu.pipeline.indexer import Tier1Indexer
    from backend.spherical_gpu.pipeline.sht_io import read_sht_master
    from backend.spherical_gpu.pipeline.preprocessing import gausbckg, nregions

    # Load EMSphInx ground truth sphere from DumpTopK output
    dump_path = PROJECT_ROOT / "tasks/dumptopk_pixel0.json"
    if not dump_path.exists():
        import pytest
        pytest.skip("DumpTopK output not generated yet — run scripts/run_dumptopk.py first")
    with open(dump_path) as f:
        dump = json.load(f)
    grid_dim = int(dump["grid_dim"])
    bw = int(dump["bandwidth"])
    assert bw == 68, f"unexpected bandwidth {bw}"
    assert grid_dim == 71, f"unexpected grid_dim {grid_dim}"
    sphere_flat = np.array(dump["sphere_flat"], dtype=np.float64)
    sph_north_em = sphere_flat[: grid_dim * grid_dim].reshape(grid_dim, grid_dim)
    sph_south_em = sphere_flat[grid_dim * grid_dim :].reshape(grid_dim, grid_dim)

    print(f"\n=== EMSphInx unproject output (pixel 0) ===")
    print(f"grid_dim={grid_dim}, bandwidth={bw}")
    print(f"north: nonzero={np.count_nonzero(sph_north_em)}, "
          f"min={sph_north_em.min():.4f}, max={sph_north_em.max():.4f}, "
          f"mean={sph_north_em.mean():.6f}, std={sph_north_em.std():.6f}")
    print(f"south: nonzero={np.count_nonzero(sph_south_em)} "
          f"(should be 0 — EMSphInx writes only to north hemisphere)")

    # Load our setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oracle_path = PROJECT_ROOT / "tests/test_spherical_gpu/data/reference_oracle.h5"
    with h5py.File(oracle_path, "r") as f:
        meta = json.loads(f.attrs["meta_json"])
        sht_bytes = bytes(f["embedded_sht_bytes"][()])
    h5oina_path = need(PROJECT_ROOT / meta["h5oina_source"])
    with h5py.File(h5oina_path, "r") as f:
        pattern_0 = f["1/EBSD/Data/Processed Patterns"][0].astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix=".sht", delete=False)
    tmp.write(sht_bytes); tmp.close()

    detector_params = {k: meta[k] for k in (
        "n_rows", "n_cols", "pat_width", "pat_height", "pixel_size",
        "tilt", "binning", "step_x", "step_y", "pc_x", "pc_y", "pc_z", "vendor",
    )}
    geom = DetectorGeometry.from_params(detector_params, device=device)
    master = read_sht_master(tmp.name, device=device)
    indexer = Tier1Indexer(geom, master, device, bandwidth=meta["bandwidth"])

    L = indexer.bandwidth
    dim = indexer._gl_dim
    assert dim == grid_dim, f"GL grid dim mismatch: ours={dim}, EMSphInx={grid_dim}"

    # Replicate stage 1 of _direct_sht_coefs
    pat_t = torch.from_numpy(pattern_0[None, :, :]).to(device).float()
    prep = nregions(gausbckg(pat_t), n=10)

    omega   = indexer._dp_omega.to(device)
    gxy     = indexer._dp_grid_xy.to(device)
    s_gxy   = indexer._dp_south_grid_xy.to(device)
    s_omega = indexer._dp_south_omega.to(device)
    omega_sum = (omega.sum() + s_omega.sum()).clamp_min(1e-12)

    pat_4d = prep.unsqueeze(1).float()
    H, W = prep.shape[1:]
    scale = indexer._rescale_factor
    w_out = max(1, round(W * scale))
    h_out = max(1, round(H * scale))
    if w_out < W or h_out < H:
        pat_4d = F.interpolate(
            pat_4d, (h_out, w_out),
            mode="bicubic", align_corners=False, antialias=True,
        )

    grid_4d = gxy.unsqueeze(0).unsqueeze(0).expand(1, 1, -1, -1)
    sampled = F.grid_sample(
        pat_4d, grid_4d, mode="bilinear",
        padding_mode="zeros", align_corners=True,
    ).squeeze(1).squeeze(1)  # (1, N_inside)

    if s_gxy.shape[0] > 0:
        grid_s_4d = s_gxy.unsqueeze(0).unsqueeze(0).expand(1, 1, -1, -1)
        sampled_s = F.grid_sample(
            pat_4d, grid_s_4d, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        ).squeeze(1).squeeze(1)
    else:
        sampled_s = torch.empty(1, 0, dtype=sampled.dtype, device=device)

    # Combined stats
    n_term = (sampled   * omega.unsqueeze(0)  ).sum(dim=1, keepdim=True)
    s_term = (sampled_s * s_omega.unsqueeze(0)).sum(dim=1, keepdim=True)
    mu = (n_term + s_term) / omega_sum
    res_n = sampled   - mu
    res_s = sampled_s - mu
    v_n = (res_n ** 2 * omega.unsqueeze(0)  ).sum(dim=1, keepdim=True)
    v_s = (res_s ** 2 * s_omega.unsqueeze(0)).sum(dim=1, keepdim=True)
    var = (v_n + v_s) / omega_sum
    std = var.sqrt().clamp_min(1e-6)
    f_norm = (res_n / std).cpu().numpy()[0]    # (N_inside,)
    f_norm_s = (res_s / std).cpu().numpy()[0]  # (N_south_inside,)

    inside_flat   = indexer._dp_inside_flat.cpu().numpy()
    # Reconstruct south-inside flat indices (in north grid):
    # The south scatter happens per-ring inside Stage 4. To get a single (dim,dim)
    # sphere image like EMSphInx, we need to walk the south rings.
    south_ring_gxy = indexer._gl_south_ring_gxy
    south_ring_inside = indexer._gl_south_ring_inside
    ring_cell_flat = indexer._gl_ring_cell_flat

    sph_ours = np.zeros((dim, dim), dtype=np.float64)
    sph_flat_ours = sph_ours.reshape(-1)
    sph_flat_ours[inside_flat] = f_norm

    # Walk south rings: for each ring ar, sample south cells (where inside),
    # normalize using mu/std, ADD to the north hemisphere positions.
    south_idx_ptr = 0
    for ar_idx in range(len(ring_cell_flat)):
        ar = ar_idx + 1
        s_ins = south_ring_inside[ar_idx].cpu().numpy().astype(bool)
        n_inside = int(s_ins.sum())
        if n_inside == 0:
            continue
        ring_flat = ring_cell_flat[ar_idx].cpu().numpy().astype(np.int64)
        # Take the next n_inside values from f_norm_s (matches order from
        # _build_flat_south_tensors that walks rings in the same loop)
        vals = f_norm_s[south_idx_ptr : south_idx_ptr + n_inside]
        south_idx_ptr += n_inside
        # Add to north positions where s_ins is True
        target_idx = ring_flat[s_ins]
        np.add.at(sph_flat_ours, target_idx, vals)

    print(f"\n=== Our stage-1 sphere image (pixel 0) ===")
    print(f"north: nonzero={np.count_nonzero(sph_ours)}, "
          f"min={sph_ours.min():.4f}, max={sph_ours.max():.4f}, "
          f"mean={sph_ours.mean():.6f}, std={sph_ours.std():.6f}")

    # Compare cell-by-cell
    nonzero_em = sph_north_em != 0
    nonzero_ours = sph_ours != 0

    common_nonzero = nonzero_em & nonzero_ours
    only_em = nonzero_em & ~nonzero_ours
    only_ours = ~nonzero_em & nonzero_ours

    print(f"\n=== Cell-coverage comparison ===")
    print(f"both nonzero      : {int(common_nonzero.sum())}")
    print(f"EMSphInx-only     : {int(only_em.sum())}")
    print(f"Ours-only         : {int(only_ours.sum())}")

    # Optimal scale to match (handle global linear scaling)
    if common_nonzero.sum() > 10:
        em_v = sph_north_em[common_nonzero].astype(np.float64)
        ours_v = sph_ours[common_nonzero].astype(np.float64)
        # Best linear fit y = a*x + b
        A = np.vstack([ours_v, np.ones_like(ours_v)]).T
        coef, *_ = np.linalg.lstsq(A, em_v, rcond=None)
        a, b = coef
        residual = em_v - (a * ours_v + b)
        rms = float(np.sqrt(np.mean(residual ** 2)))
        # Pearson correlation
        r = float(np.corrcoef(em_v, ours_v)[0, 1])
        print(f"\nLinear fit EM = a * Ours + b on {int(common_nonzero.sum())} common cells:")
        print(f"  a = {a:.6f}  b = {b:.6f}")
        print(f"  rms residual = {rms:.6f}")
        print(f"  Pearson correlation r = {r:.6f}")

        # If we just multiply by a single constant, what's the relative error?
        # (Captures the "are values proportional" question.)
        ratio = em_v / np.where(np.abs(ours_v) > 1e-6, ours_v, 1)
        median_ratio = float(np.median(ratio[np.abs(ours_v) > 1e-3]))
        print(f"  median ratio EM/Ours (where |Ours|>1e-3) = {median_ratio:.6f}")

    # Detailed bin-level diff for first few mismatches
    print(f"\n=== First 10 cells where EMSphInx writes but we don't ===")
    em_only_idx = np.argwhere(only_em)
    for i, (r, c) in enumerate(em_only_idx[:10]):
        print(f"  ({r},{c}): EM={sph_north_em[r,c]:.4f}, ours={sph_ours[r,c]:.4f}")

    print(f"\n=== First 10 cells where we write but EMSphInx doesn't ===")
    ours_only_idx = np.argwhere(only_ours)
    for i, (r, c) in enumerate(ours_only_idx[:10]):
        print(f"  ({r},{c}): EM={sph_north_em[r,c]:.4f}, ours={sph_ours[r,c]:.4f}")

    # If common cells exist, dump per-cell diff distribution
    if common_nonzero.sum() > 0:
        em_v = sph_north_em[common_nonzero]
        ours_v = sph_ours[common_nonzero]
        diff = em_v - ours_v
        rel = np.abs(diff) / (np.abs(em_v) + 1e-9)
        print(f"\n=== Per-cell diff stats over {int(common_nonzero.sum())} common cells ===")
        print(f"  abs diff: mean={np.abs(diff).mean():.4f}, "
              f"max={np.abs(diff).max():.4f}, "
              f"p95={np.percentile(np.abs(diff), 95):.4f}")
        print(f"  rel diff: median={np.median(rel) * 100:.2f}%, "
              f"p95={np.percentile(rel, 95) * 100:.2f}%, "
              f"max={rel.max() * 100:.2f}%")

    # Save artifacts for later analysis
    artifacts_path = PROJECT_ROOT / "tasks/sphere_compare_data.npz"
    np.savez(
        artifacts_path,
        sph_north_em=sph_north_em,
        sph_ours=sph_ours,
        nonzero_em=nonzero_em,
        nonzero_ours=nonzero_ours,
    )
    print(f"\nArtifacts saved to: {artifacts_path}")
