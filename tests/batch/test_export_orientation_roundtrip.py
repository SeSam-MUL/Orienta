"""Exporting the same CrystalMap to .ang and .ctf must round-trip consistently.

EDAX/TSL .ang historically uses a bottom-origin Y; Oxford/HKL .ctf uses a
top-origin Y. If the exporter writes the same Y array to both formats
without per-format handling, re-loading the two files in orix can produce
orientations that are flipped relative to each other, even though the
underlying CrystalMap was the same.

This test drives the production ``export_ang_ctf`` end-to-end via a small
synthetic multiphase checkpoint, then loads both written files back
through orix and checks that orientations at matching ``(x, y)`` map
positions agree to within a fraction of a degree (symmetry-reduced
disorientation under m-3m).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from orix.io import load as orix_load
from orix.quaternion import Orientation
from orix.quaternion import symmetry as _sym

from backend.api.services.checkpoint_writer import CheckpointWriter
from backend.api.services.result_exporter import export_ang_ctf


@pytest.fixture
def asymmetric_checkpoint(tmp_path: Path):
    """A 4×3 (rows×cols) single-phase checkpoint with twelve distinct,
    asymmetric per-pixel orientations.

    Asymmetry matters: if every pixel held the same orientation we would
    not see a Y-flip because mirroring an all-identical map produces the
    same map. By choosing 12 unique Eulers that vary with pixel index we
    make any reordering (including a Y-flip) detectable.
    """
    source = tmp_path / "FakeOrientation.h5oina"
    n_rows, n_cols = 4, 3
    n_pixels = n_rows * n_cols
    with h5py.File(source, "w") as f:
        ebsd_data = f.create_group("1/EBSD/Data")
        # BC sequence — identifies any row/col reordering on inspection.
        ebsd_data.create_dataset(
            "Band Contrast",
            data=np.arange(10, 10 + n_pixels, dtype=np.uint8),
        )
        hdr = f.create_group("1/EBSD/Header")
        hdr.create_dataset("Beam Voltage", data=np.float32(20.0))
        hdr.create_dataset("Tilt Angle", data=np.float32(70.0))
        hdr.create_dataset("Magnification", data=np.float32(500.0))

    cw = CheckpointWriter(str(source))
    cw.init_metadata(
        grid_shape=(n_rows, n_cols),
        batch_id="orient-test",
        method="spherical",
        step_size_um=0.5,
    )
    # Per-pixel Euler triples in radians — picked so each pixel is
    # distinct in all three Euler components.
    flat_eulers = np.deg2rad(
        np.array(
            [[i * 11.0, 20.0 + i, 30.0 + 2 * i] for i in range(n_pixels)],
            dtype=np.float64,
        )
    )
    ori = flat_eulers.reshape(n_rows, n_cols, 3).astype(np.float32)
    ci = np.full((n_rows, n_cols), 0.8, dtype=np.float32)

    cw.write_phase_result(
        "Al", ci, ori,
        {
            "phase_file": "Al.cif",
            "ci_mean": 0.8,
            "duration_sec": 1.0,
            "space_group": 225,
            "point_group": "m-3m",
        },
    )
    cw.compute_auto_assignment(confidence_threshold=0.0)
    return source, cw, (n_rows, n_cols)


def _sorted_by_xy(xmap):
    """Sort the xmap's rotations and coordinates by (y, x) so two xmaps
    that loaded with different internal ordering can be compared
    pixel-for-pixel."""
    xs = np.asarray(xmap.x).astype(float).ravel()
    ys = np.asarray(xmap.y).astype(float).ravel()
    order = np.lexsort((xs, ys))
    rots = xmap.rotations[order]
    return rots, xs[order], ys[order]


def test_ang_ctf_roundtrip_orientations_match(asymmetric_checkpoint, tmp_path):
    source, cw, grid_shape = asymmetric_checkpoint
    out = tmp_path / "out"
    out.mkdir()

    ang, ctf = export_ang_ctf(
        cw.checkpoint_path, str(out),
        write_ang=True, write_ctf=True,
        source_h5_path=str(source),
    )
    assert ang is not None, "exporter failed to produce .ang"
    assert ctf is not None, "exporter failed to produce .ctf"

    ang_xmap = orix_load(str(ang))
    ctf_xmap = orix_load(str(ctf))

    r_ang, xa, ya = _sorted_by_xy(ang_xmap)
    r_ctf, xc, yc = _sorted_by_xy(ctf_xmap)

    assert r_ang.size == r_ctf.size, (
        f".ang/.ctf pixel count mismatch: {r_ang.size} vs {r_ctf.size}"
    )
    assert r_ang.size == grid_shape[0] * grid_shape[1], (
        f"unexpected pixel count {r_ang.size}; expected {grid_shape}"
    )
    assert np.allclose(xa, xc, atol=1e-4) and np.allclose(ya, yc, atol=1e-4), (
        f".ang/.ctf x/y coordinate sets diverge after sort:\n"
        f"  ang x={xa}\n  ctf x={xc}\n  ang y={ya}\n  ctf y={yc}"
    )

    o_ang = Orientation(r_ang, _sym.Oh)
    o_ctf = Orientation(r_ctf, _sym.Oh)
    deg = np.rad2deg(o_ang.angle_with(o_ctf)).ravel()
    median = float(np.median(deg))
    p95 = float(np.percentile(deg, 95))
    worst = float(np.max(deg))

    assert median < 1.0, (
        f".ang vs .ctf orientations diverge:\n"
        f"  median {median:.3f} deg, 95th-pctile {p95:.3f} deg, "
        f"max {worst:.3f} deg\n"
        f"  → formats may be writing Y with different conventions "
        f"(EDAX bottom-origin vs Oxford top-origin)"
    )
