"""SP1 validation — NCC of our GPU master pattern vs the EMsoft oracle (Task 12).

This is the harness the SP0+SP1 gate (design §5, §4.7) drives to: it builds our
dynamical master pattern with :func:`backend.forward_sim.dynamical.master_builder.build_master`
at a chosen ``npx``, loads the matching EMsoft oracle slice ``mLPNH[0, E_idx]``,
**block-mean downsamples** the 1001x1001 oracle to our ``(2*npx+1, 2*npx+1)``
resolution, masks the inscribed disc the modified-Lambert projection actually
covers, and reports the zero-mean normalised cross-correlation (NCC) over that
disc.

Why downsample the oracle (not upsample ours)
----------------------------------------------
The full EMsoft gate is ``npx=500`` (1001x1001), but the dynamical core
(per-direction ``torch.linalg.eig`` / ``matrix_exp``) is sequential on CUDA, so a
full-resolution run is minutes-to-hours.  For the *first* NCC measurement we run
at a tractable ``npx`` (50 → 101x101 directions) and compare like-for-like by
averaging each oracle block down to one of our pixels.  Both grids share the same
EMsoft modified-Lambert pixel convention (centre at ``npx``, coordinate
``(i-npx)/npx`` ∈ [-1, 1]), so the block-mean is a pixel-aligned area average.

NCC definition (zero-mean normalised cross-correlation)
-------------------------------------------------------
Over the masked disc pixels ``D``::

    a = ours[D] - mean(ours[D])
    b = emsoft[D] - mean(emsoft[D])
    NCC = Σ a·b / sqrt(Σ a² · Σ b²)      ∈ [-1, 1]

This is invariant to per-image offset and scale (the master pattern is defined up
to an overall normalisation), so it measures whether the *contrast pattern* (band
geometry + relative intensities) matches — exactly the gate the spec sets.

CLI
---
``python -m backend.forward_sim.validate.ncc_vs_emsoft Ni --npx 50``
``python -m backend.forward_sim.validate.ncc_vs_emsoft Al --npx 50 --energy 20``
``python -m backend.forward_sim.validate.ncc_vs_emsoft Ni --npx 50 --uniform-lambda``
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from ..crystal.structure_matrix import read_bethe_parameters
from ..crystal.xtal_io import read_crystal_structure
from ..dynamical.master_builder import build_master
from ..mc.emsoft_mc_input import load_mc
from ..runtime import ForwardSimError, get_device

# Oracle master-pattern files (verified present 2026-06-10).  Both carry the
# oracle (mLPNH), the MC input (MCOpenCL) and the crystal (CrystalData) in one .h5.
_ORACLE = {
    "Ni": "Database/EBSD_H5_Cache/Ni/Ni_master_E20kV_npx500.h5",
    "Al": "Database/EBSD_H5_Cache/Al/Al_master_E20kV_npx500.h5",
}

_ITER_DIR = Path("tasks/forward_sim/iterations/3")


def _project_root() -> Path:
    """Repo root (this file is ``backend/forward_sim/validate/ncc_vs_emsoft.py``)."""
    return Path(__file__).resolve().parents[3]


def _resolve_oracle(phase: str) -> Path:
    """Absolute path to the EMsoft oracle ``.h5`` for ``phase`` (Ni / Al)."""
    if phase not in _ORACLE:
        raise ValueError(
            f"unknown phase {phase!r}; known oracles: {sorted(_ORACLE)}"
        )
    p = _project_root() / _ORACLE[phase]
    if not p.is_file():
        raise ValueError(f"oracle file not found for {phase!r}: {p}")
    return p


def _energy_index(ekevs: np.ndarray, energy_kV: float) -> int:
    """Index of the energy bin closest to ``energy_kV`` in ``EkeVs``."""
    return int(np.argmin(np.abs(ekevs.ravel() - energy_kV)))


def block_mean_downsample(arr: np.ndarray, out_size: int) -> np.ndarray:
    """Block-mean downsample a square ``(S, S)`` array to ``(out_size, out_size)``.

    Each output pixel ``(i, j)`` averages the source pixels whose **row/col index**
    falls in the half-open block ``[i·S/out, (i+1)·S/out)`` (and likewise for
    columns).  For ``S = out_size`` this is the identity; otherwise it is the
    standard area-average block-mean (the source need not be an integer multiple of
    the target — boundary blocks just contain a few more/fewer source pixels).

    Args:
        arr: ``(S, S)`` float array (the EMsoft 1001x1001 oracle slice).
        out_size: target side length (``2*npx+1``).

    Returns:
        ``(out_size, out_size)`` float64 array.
    """
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"expected square 2D array, got {arr.shape}")
    S = arr.shape[0]
    if out_size <= 0:
        raise ValueError(f"out_size must be > 0, got {out_size}")
    if out_size > S:
        raise ValueError(
            f"out_size {out_size} > source size {S}; this is a DOWNsampler"
        )
    a = arr.astype(np.float64, copy=False)
    # Block boundaries along each axis (out_size+1 edges spanning [0, S]).
    edges = np.linspace(0, S, out_size + 1).astype(int)
    out = np.empty((out_size, out_size), dtype=np.float64)
    # Average rows first, then columns (separable block-mean).
    row_blocks = np.empty((out_size, S), dtype=np.float64)
    for i in range(out_size):
        r0, r1 = edges[i], edges[i + 1]
        if r1 <= r0:
            r1 = r0 + 1
        row_blocks[i] = a[r0:r1].mean(axis=0)
    for j in range(out_size):
        c0, c1 = edges[j], edges[j + 1]
        if c1 <= c0:
            c1 = c0 + 1
        out[:, j] = row_blocks[:, c0:c1].mean(axis=1)
    return out


def _inscribed_disc_mask(m: int) -> np.ndarray:
    """Inscribed-disc boolean mask for an ``(m, m)`` Lambert grid (``m = 2*npx+1``).

    Pixel ``(row, col)`` maps to Lambert coordinate ``((col-npx)/npx, (row-npx)/npx)``;
    the modified-Lambert projection covers ``|xy|₂ <= 1`` (the inscribed circle), so
    the square corners are excluded — matching ``build_master``'s ``inside`` mask and
    EMsoft's undefined mLPNH corners.
    """
    npx = (m - 1) // 2
    idx = np.arange(m, dtype=np.float64)
    rr, cc = np.meshgrid(idx, idx, indexing="ij")
    x = (cc - npx) / npx
    y = (rr - npx) / npx
    return (x * x + y * y) <= 1.0 + 1e-9


def ncc_zero_mean(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Zero-mean normalised cross-correlation of ``a``, ``b`` over ``mask`` pixels.

    Returns a value in ``[-1, 1]``; ``nan`` only if a variance is exactly zero
    (a constant master over the disc — which would itself be a red flag).
    """
    av = a[mask].astype(np.float64).ravel()
    bv = b[mask].astype(np.float64).ravel()
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = math.sqrt(float(np.dot(av, av)) * float(np.dot(bv, bv)))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(av, bv) / denom)


def _save_png(path: Path, arr: np.ndarray, mask: np.ndarray | None = None) -> None:
    """Save a 2D array as a grayscale PNG, scaled to its in-disc min/max.

    Pixels outside ``mask`` (if given) are set to black so the inscribed disc reads
    cleanly.  Uses Pillow (already a project dependency) — no matplotlib figure
    overhead.
    """
    from PIL import Image

    a = arr.astype(np.float64).copy()
    if mask is not None:
        vals = a[mask]
    else:
        vals = a.ravel()
    lo = float(np.nanmin(vals)) if vals.size else 0.0
    hi = float(np.nanmax(vals)) if vals.size else 1.0
    if hi <= lo:
        hi = lo + 1.0
    norm = (a - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, 1.0)
    if mask is not None:
        norm = np.where(mask, norm, 0.0)
    img = (norm * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="L").save(str(path))


def ncc_vs_emsoft(
    phase: str,
    *,
    npx: int,
    energy_kV: float = 20.0,
    uniform_lambda: bool = False,
    device: torch.device | str | None = None,
    dmin: float = 0.05,
    save_images: bool = True,
    save_npy: bool = False,
    absflg: int = 1,
) -> dict:
    """Measure the NCC of our master pattern vs the EMsoft oracle for ``phase``.

    Builds our dynamical master at ``npx``, loads the matching EMsoft
    ``mLPNH[0, E_idx]`` slice, block-mean downsamples it to ``(2*npx+1)``, masks the
    inscribed disc, and computes the zero-mean NCC over the disc.

    Args:
        phase: ``"Ni"`` or ``"Al"`` (selects the oracle ``.h5``).
        npx: Lambert half-grid size; the grid is ``(2*npx+1)²``.  Keep small (≈50)
            for a tractable first measurement; the full gate is ``npx=500``.
        energy_kV: accelerating voltage; the closest ``EkeVs`` bin is used.
        uniform_lambda: if True, replace the MC depth profile with a flat one — the
            geometry-only sanity mode (design §2 ``uniform``).
        device: compute device; defaults to :func:`backend.forward_sim.runtime.get_device`
            (CUDA if available, else honours ``FORWARD_SIM_DEVICE``).
        dmin: reflection-list resolution limit in nm (design default 0.05).
        save_images: if True, write ``our.png``/``emsoft.png``/``absdiff.png`` under
            ``tasks/forward_sim/iterations/3/`` (named per phase + mode).
        save_npy: if True (default False), additionally write the raw float64 arrays
            ``our_<phase>_npx<N>.npy`` (our master) and ``emsoft_<phase>_npx<N>.npy``
            (the block-mean downsampled oracle) into the iteration dir — for cheap
            re-scoring without re-running the build.  Default behaviour unchanged.
        absflg: EMsoft absorption flag forwarded to :func:`build_master`
            (``1`` = phonon only, default — no behaviour change; ``3`` = phonon +
            core-loss).

    Returns:
        A dict with keys ``phase``, ``energy_kV``, ``energy_idx``, ``npx``,
        ``uniform_lambda``, ``device``, ``ncc``, ``wall_s``, and (when images are
        saved) ``heatmaps`` (list of saved PNG paths); when ``save_npy`` is True a
        ``npy`` key lists the two saved ``.npy`` paths.

    Raises:
        ValueError: for an unknown phase or a missing oracle file.
    """
    oracle = _resolve_oracle(phase)
    dev = torch.device(device) if device is not None else get_device()

    structure = read_crystal_structure(str(oracle))
    mc = load_mc(str(oracle))
    bethe = read_bethe_parameters(str(oracle))

    # Energy bin in OUR MC EkeVs (same array EMsoft stored).
    ekevs = mc.EkeVs.detach().cpu().numpy()
    e_idx = _energy_index(ekevs, energy_kV)
    energy_used = float(ekevs.ravel()[e_idx])

    t0 = time.time()
    ours_t = build_master(
        structure,
        mc,
        npx=npx,
        energy_idx=e_idx,
        dmin=dmin,
        device=dev,
        uniform_lambda=uniform_lambda,
        bethe_params=bethe,
        absflg=absflg,
    )
    wall_s = time.time() - t0
    ours = ours_t.detach().cpu().numpy().astype(np.float64)
    m = ours.shape[0]  # 2*npx+1

    # --- Load + block-mean downsample the EMsoft oracle slice -------------------
    with h5py.File(str(oracle), "r") as f:
        mlpnh = f["EMData/EBSDmaster/mLPNH"]  # (1, nE, 1001, 1001)
        oracle_ek = np.asarray(f["EMData/EBSDmaster/EkeVs"][()]).ravel()
        oe_idx = _energy_index(oracle_ek, energy_kV)
        oracle_slice = np.asarray(mlpnh[0, oe_idx, :, :], dtype=np.float64)

    emsoft = block_mean_downsample(oracle_slice, m)

    # --- Mask + NCC -------------------------------------------------------------
    # CORNER FIX (2026-06-22): the modified-Lambert square is FULLY covered (the
    # corners are valid equatorial directions EMsoft also fills), so the honest
    # comparison is over the WHOLE square. We report all three so a corner-region
    # regression can never again hide behind a disc-only mask (the old blind spot
    # that let four black-diamond corner holes pass at "0.998 disc NCC").
    mask = _inscribed_disc_mask(m)              # disc (legacy, back-compat key)
    full_mask = np.ones((m, m), dtype=bool)     # whole square (the honest metric)
    corner_mask = ~mask                         # the previously-zeroed corners
    ncc = ncc_zero_mean(ours, emsoft, mask)
    ncc_full = ncc_zero_mean(ours, emsoft, full_mask)
    ncc_corners = ncc_zero_mean(ours, emsoft, corner_mask)

    result = {
        "phase": phase,
        "energy_kV": energy_used,
        "energy_idx": e_idx,
        "npx": npx,
        "uniform_lambda": bool(uniform_lambda),
        "device": str(dev),
        "absflg": int(absflg),
        "ncc": ncc,                # disc-only (legacy)
        "ncc_full": ncc_full,      # whole square (honest)
        "ncc_corners": ncc_corners,
        "wall_s": wall_s,
    }

    if save_npy:
        # Raw float64 arrays for cheap re-scoring (no re-build).  Named per phase +
        # npx; written into the active iteration dir.
        our_npy = _ITER_DIR / f"our_{phase}_npx{npx}.npy"
        em_npy = _ITER_DIR / f"emsoft_{phase}_npx{npx}.npy"
        our_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(our_npy), ours)
        np.save(str(em_npy), emsoft)
        result["npy"] = [str(our_npy), str(em_npy)]

    if save_images:
        suffix = f"{phase}_npx{npx}" + ("_uniform" if uniform_lambda else "")
        our_png = _ITER_DIR / f"our_{suffix}.png"
        em_png = _ITER_DIR / f"emsoft_{suffix}.png"
        diff_png = _ITER_DIR / f"absdiff_{suffix}.png"
        # Normalise each (zero-mean / unit-std over the disc) before the abs-diff so
        # the difference reflects pattern mismatch, not the arbitrary overall scale.
        ours_n = _disc_normalise(ours, mask)
        em_n = _disc_normalise(emsoft, mask)
        absdiff = np.abs(ours_n - em_n)
        _save_png(our_png, ours, mask)
        _save_png(em_png, emsoft, mask)
        _save_png(diff_png, absdiff, mask)
        result["heatmaps"] = [str(our_png), str(em_png), str(diff_png)]

    return result


def _disc_normalise(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero-mean / unit-std normalise an array over its masked disc (for abs-diff)."""
    a = arr.astype(np.float64).copy()
    vals = a[mask]
    mu = float(vals.mean()) if vals.size else 0.0
    sd = float(vals.std()) if vals.size else 1.0
    if sd == 0.0:
        sd = 1.0
    out = (a - mu) / sd
    return np.where(mask, out, 0.0)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NCC of our GPU master pattern vs the EMsoft oracle."
    )
    p.add_argument("phase", choices=sorted(_ORACLE), help="phase (Ni or Al)")
    p.add_argument(
        "--npx", type=int, default=50, help="Lambert half-grid size (grid=2*npx+1)"
    )
    p.add_argument(
        "--energy", type=float, default=20.0, help="accelerating voltage (kV)"
    )
    p.add_argument("--dmin", type=float, default=0.05, help="reflection dmin (nm)")
    p.add_argument(
        "--absflg",
        type=int,
        default=1,
        choices=(1, 3),
        help="absorption flag (1 = phonon only, 3 = phonon + core-loss)",
    )
    p.add_argument(
        "--uniform-lambda",
        action="store_true",
        help="geometry-only sanity mode (flat depth profile)",
    )
    p.add_argument(
        "--save-npy",
        action="store_true",
        help="also write our_<phase>_npx<N>.npy + emsoft_<phase>_npx<N>.npy",
    )
    p.add_argument(
        "--device",
        default=None,
        help="torch device override (e.g. cpu, cuda); else runtime default",
    )
    p.add_argument(
        "--no-images", action="store_true", help="skip writing the PNG heatmaps"
    )
    return p


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        res = ncc_vs_emsoft(
            args.phase,
            npx=args.npx,
            energy_kV=args.energy,
            uniform_lambda=args.uniform_lambda,
            device=args.device,
            dmin=args.dmin,
            save_images=not args.no_images,
            save_npy=args.save_npy,
            absflg=args.absflg,
        )
    except ForwardSimError as exc:  # no-CUDA fail-loud → actionable message
        print(f"ERROR: {exc}")
        return 2
    mode = " (uniform-λ)" if res["uniform_lambda"] else ""
    print(
        f"NCC({res['phase']}, {res['energy_kV']:.0f}kV{mode}) "
        f"full={res['ncc_full']:.4f} disc={res['ncc']:.4f} "
        f"corners={res['ncc_corners']:.4f}  "
        f"[npx={res['npx']}, device={res['device']}, wall={res['wall_s']:.1f}s]"
    )
    if "heatmaps" in res:
        for h in res["heatmaps"]:
            print(f"  wrote {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
