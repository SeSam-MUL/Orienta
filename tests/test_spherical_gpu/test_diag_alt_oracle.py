"""Iter-15B: Use BatchIndexDump (indexImage path) as ALT-ORACLE.

We have 3 reference orientation sets for the same patterns:
  - oracle.no_refine (saved in reference_oracle.h5)  — IndexEBSD CLI, ref=False
  - oracle.refined   (saved in reference_oracle.h5)  — IndexEBSD CLI, ref=True
  - alt-oracle (tasks/batch_indeximage_refTrue.csv) — direct indexImage, ref=True

The first two are the same EMSphInx codebase but go through the IndexEBSD
binary's pixel iteration. The third uses the same Indexer code but is called
directly from C++ (no IndexEBSD post-processing).

We compare ALL of:
  (a) alt-oracle vs oracle.refined         — disagreement reveals what
      IndexEBSD does that indexImage doesn't.
  (b) Our pipeline vs oracle.refined       — current Tier-1 gap.
  (c) Our pipeline vs alt-oracle            — Tier-1 gap to a DIFFERENT
      reference. If we match alt-oracle better than refined, we just
      have the IndexEBSD-specific extra step missing.
"""
from __future__ import annotations

import csv
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
ALT_ORACLE_CSV = PROJECT_ROOT / "tasks/batch_indeximage_refTrue.csv"


def _disorientation_deg(eu_a: np.ndarray, eu_b: np.ndarray) -> np.ndarray:
    from orix.quaternion import Misorientation, Rotation
    from orix.quaternion.symmetry import _groups
    sym = next(g for g in _groups if g.name == "m-3m")
    r_a = Rotation.from_euler(eu_a)
    r_b = Rotation.from_euler(eu_b)
    miso = Misorientation((r_a * (~r_b)).data, symmetry=(sym, sym))
    miso = miso.map_into_symmetry_reduced_zone()
    return np.rad2deg(miso.angle)


def _quat_to_zxz_eulers(qu: np.ndarray) -> np.ndarray:
    """Convert (N, 4) quaternion array (w, x, y, z) to (N, 3) Bunge ZXZ Euler in radians."""
    from orix.quaternion import Rotation
    rot = Rotation(qu)
    return rot.to_euler()


@pytest.mark.skipif(
    not ALT_ORACLE_CSV.is_file(),
    reason=(
        f"Alt-oracle CSV {ALT_ORACLE_CSV.name} not present — regenerate "
        "via the WSL EMSphInx BatchIndexDump CLI to enable this diagnostic. "
        "Tracked in tasks/phase_a_followups.md item #1."
    ),
)
def test_alt_oracle_vs_main_oracle():
    """Compare BatchIndexDump output (alt-oracle) vs the main oracle's
    no_refine and refined arrays. Reports disorientation distribution."""
    csv_path = ALT_ORACLE_CSV
    oracle_path = PROJECT_ROOT / "tests/test_spherical_gpu/data/reference_oracle.h5"

    rows = []
    with open(csv_path) as f:
        rdr = csv.reader(f)
        header = next(rdr)
        for row in rdr:
            rows.append(row)

    pixels = np.array([int(r[0]) for r in rows])
    quats = np.array([[float(r[4]), float(r[5]), float(r[6]), float(r[7])] for r in rows])
    corrs = np.array([float(r[2]) for r in rows])

    print(f"\nLoaded {len(rows)} pixels from BatchIndexDump.")
    assert (pixels == np.arange(len(rows))).all(), "Pixel order is not 0..N-1!"

    alt_eulers = _quat_to_zxz_eulers(quats)

    with h5py.File(oracle_path, "r") as f:
        no_ref = f["euler_xyz_no_refine"][:]
        refined = f["euler_xyz_refined"][:]

    diso_alt_vs_norefine = _disorientation_deg(alt_eulers, no_ref)
    diso_alt_vs_refined  = _disorientation_deg(alt_eulers, refined)

    print(f"\nAlt-oracle (indexImage ref=true) vs oracle.no_refine:")
    print(f"  median = {np.median(diso_alt_vs_norefine):.2f} deg")
    print(f"  95th-pct = {np.percentile(diso_alt_vs_norefine, 95):.2f} deg")
    print(f"  fraction < 1 deg: {float((diso_alt_vs_norefine < 1.0).mean()):.2%}")
    print(f"  fraction < 3 deg: {float((diso_alt_vs_norefine < 3.0).mean()):.2%}")
    print(f"  fraction < 5 deg: {float((diso_alt_vs_norefine < 5.0).mean()):.2%}")

    print(f"\nAlt-oracle (indexImage ref=true) vs oracle.refined:")
    print(f"  median = {np.median(diso_alt_vs_refined):.2f} deg")
    print(f"  95th-pct = {np.percentile(diso_alt_vs_refined, 95):.2f} deg")
    print(f"  fraction < 1 deg: {float((diso_alt_vs_refined < 1.0).mean()):.2%}")
    print(f"  fraction < 3 deg: {float((diso_alt_vs_refined < 3.0).mean()):.2%}")
    print(f"  fraction < 5 deg: {float((diso_alt_vs_refined < 5.0).mean()):.2%}")


@pytest.mark.skipif(
    not ALT_ORACLE_CSV.is_file(),
    reason=(
        f"Alt-oracle CSV {ALT_ORACLE_CSV.name} not present — regenerate "
        "via the WSL EMSphInx BatchIndexDump CLI to enable this diagnostic. "
        "Tracked in tasks/phase_a_followups.md item #1."
    ),
)
def test_our_pipeline_vs_alt_oracle():
    """Run our Tier-1 pipeline on all 2842 patterns, compare against alt-oracle."""
    from backend.spherical_gpu.pipeline.detector import DetectorGeometry
    from backend.spherical_gpu.pipeline.indexer import Tier1Indexer
    from backend.spherical_gpu.pipeline.sht_io import read_sht_master

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oracle_path = PROJECT_ROOT / "tests/test_spherical_gpu/data/reference_oracle.h5"
    with h5py.File(oracle_path, "r") as f:
        meta = json.loads(f.attrs["meta_json"])
        sht_bytes = bytes(f["embedded_sht_bytes"][()])
        no_ref = f["euler_xyz_no_refine"][:]
        refined = f["euler_xyz_refined"][:]
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

    # Index all patterns (will take ~1-2 min on GPU)
    result = indexer.index_h5oina(str(h5oina_path), batch_size=64)
    our_eulers = result.euler_xyz.cpu().numpy()

    # Load alt-oracle quaternions
    csv_path = PROJECT_ROOT / "tasks/batch_indeximage_refTrue.csv"
    with open(csv_path) as f:
        rdr = csv.reader(f); next(rdr)
        rows = list(rdr)
    quats = np.array([[float(r[4]), float(r[5]), float(r[6]), float(r[7])] for r in rows])
    alt_eulers = _quat_to_zxz_eulers(quats)

    diso_our_vs_alt = _disorientation_deg(our_eulers, alt_eulers)
    diso_our_vs_refined = _disorientation_deg(our_eulers, refined)
    diso_our_vs_norefine = _disorientation_deg(our_eulers, no_ref)

    print()
    print(f"Our pipeline vs alt-oracle (indexImage ref=true):")
    print(f"  median = {np.median(diso_our_vs_alt):.2f} deg")
    print(f"  95th-pct = {np.percentile(diso_our_vs_alt, 95):.2f} deg")
    print(f"  fraction < 1 deg: {float((diso_our_vs_alt < 1.0).mean()):.2%}")
    print(f"  fraction < 3 deg: {float((diso_our_vs_alt < 3.0).mean()):.2%}")
    print(f"  fraction < 5 deg: {float((diso_our_vs_alt < 5.0).mean()):.2%}")

    print()
    print(f"Our pipeline vs oracle.refined:")
    print(f"  median = {np.median(diso_our_vs_refined):.2f} deg")
    print(f"  95th-pct = {np.percentile(diso_our_vs_refined, 95):.2f} deg")

    print()
    print(f"Our pipeline vs oracle.no_refine:")
    print(f"  median = {np.median(diso_our_vs_norefine):.2f} deg")
    print(f"  95th-pct = {np.percentile(diso_our_vs_norefine, 95):.2f} deg")
