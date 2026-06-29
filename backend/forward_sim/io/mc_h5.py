"""SP5 — serialise our GPU Monte-Carlo output into an EMsoft-format MC ``.h5``.

EMsoft's ``EMMCOpenCL`` writes the Monte-Carlo energy/depth histogram into a
standalone ``.h5`` *before* the master-pattern run reads it back.  Our GPU
forward-sim produces the same information as an :class:`~backend.forward_sim.mc.emsoft_mc_input.MCData`,
but until now it was only ever held in memory and folded into the master file.
For the GPU engine to be a true drop-in for the whole workflow — in particular
so ``scan_missing`` / ``crystal_picker`` detect a GPU run as having produced an
MC file (``has_mc``), and so "Simulate All Missing" converges — we must emit the
MC ``.h5`` in EMsoft's layout.

Layout (verified against the real Ni oracle ``Ni_E20kV_sig70_n501_o0.h5``,
2026-06-16)::

    EMData/MCOpenCL/accum_e      (numsx, numsy, nE)        int32
    EMData/MCOpenCL/accum_z      (nx, ny, nz, nE)          int32   (depth axis 2)
    EMData/MCOpenCL/accumSP      (numsx, numsy, nE)        float32
    EMData/MCOpenCL/numEbins     (1,)                      int32
    EMData/MCOpenCL/numzbins     (1,)                      int32
    EMData/MCOpenCL/multiplier   (1,)                      int32
    EMData/MCOpenCL/totnum_el    (1,)                      int32
    EMData/EBSDmaster/EkeVs      (nE,)                     float32   (load_mc reads this)
    NMLparameters/MCCLNameList/{Ehistmin, Ebinsize, EkeV, depthstep, depthmax,
                                sig, omega, numsx, totnum_el, multiplier, MCmode,
                                mode, xtalname, dataname}
    CrystalData/*                                          (from our structure)
    EMheader/MCOpenCL/ProgramName = "EMMCOpenCL.f90"

This is exactly the set :func:`backend.forward_sim.mc.emsoft_mc_input.load_mc`
reads back (``EMData/MCOpenCL/{accum_e, accum_z}``, ``EMData/EBSDmaster/EkeVs``,
``NMLparameters/MCCLNameList/{depthstep, depthmax}``), so ``write_mc_h5`` →
``load_mc`` round-trips the accumulators and the depth binning exactly.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from ..crystal.xtal_io import CrystalStructure
from ..mc.emsoft_mc_input import MCData
# Reuse the EMsoft-string / crystal / header helpers (single source of truth).
from .master_h5 import (
    _MC_PROGRAM,
    _str_ds,
    _write_crystal_data,
    _write_emheader,
)


def _accum_to_int32(t: torch.Tensor) -> np.ndarray:
    """Coerce an accumulator tensor (int-valued float or int) to int32 numpy.

    ``run_gpu_mc`` stores ``accum_e`` / ``accum_z`` as int-valued tensors (the
    counts are produced by ``index_put_(accumulate=True)``); EMsoft stores them
    as ``int32``.  Round-then-cast is exact for integral values.
    """
    arr = t.detach().cpu().numpy()
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.rint(arr)
    return arr.astype(np.int32)


def write_mc_h5(
    out_path: str,
    mc: MCData,
    structure: CrystalStructure,
    *,
    sig: float,
    omega: float = 0.0,
    numsx: int | None = None,
    totnum_el: int,
    Ehistmin: float,
    Ebinsize: float,
    EkeV: float | None = None,
    xtalname: str = "forward_sim.xtal",
) -> str:
    """Write a :class:`MCData` into an EMsoft-MCOpenCL-format ``.h5``.

    The produced file is read back by
    :func:`backend.forward_sim.mc.emsoft_mc_input.load_mc` (accum arrays + EkeVs
    + depth binning) and detected as an MC file by ``crystal_picker`` /
    ``scan_missing`` (a non-``_master_`` ``.h5`` whose name carries the stem and
    ``E{kv}kV``).

    Args:
        out_path: destination file path (overwritten if it exists).
        mc: the Monte-Carlo data — supplies ``accum_e``, ``accum_z`` and ``EkeVs``.
        structure: the crystal — written into ``CrystalData``.
        sig: sample tilt in degrees (EBSD standard 70).
        omega: secondary tilt in degrees (usually 0).
        numsx: Lambert direction-grid side for ``accum_e`` / ``numsx`` NML.
            Defaults to ``accum_e``'s leading axis.
        totnum_el: total electrons simulated (stored in ``MCOpenCL/totnum_el``
            and the NML).
        Ehistmin: lowest energy bin centre (kV) — stored in the NML.
        Ebinsize: energy bin width (kV) — stored in the NML.
        EkeV: beam (accelerating) voltage (kV).  Defaults to the top ``EkeVs`` bin.
        xtalname: crystal file name string stored in the file (cosmetic).

    Returns:
        ``out_path`` (the written file path).

    Raises:
        ValueError: if the accumulators have an unexpected rank.
    """
    accum_e_np = _accum_to_int32(mc.accum_e)
    accum_z_np = _accum_to_int32(mc.accum_z)
    if accum_e_np.ndim != 3:
        raise ValueError(
            f"accum_e must be (numsx, numsy, nE), got shape {accum_e_np.shape}"
        )
    if accum_z_np.ndim != 4:
        raise ValueError(
            f"accum_z must be (nx, ny, nz, nE), got shape {accum_z_np.shape}"
        )

    ekevs_np = mc.EkeVs.detach().cpu().numpy().astype(np.float32).ravel()
    nE = int(ekevs_np.shape[0])
    nz = int(accum_z_np.shape[2])
    if numsx is None:
        numsx = int(accum_e_np.shape[0])
    if EkeV is None:
        EkeV = float(ekevs_np[-1]) if nE else float(Ehistmin)

    # accumSP (the Lambert-projected back-scatter map) is not produced by our MC;
    # write a float32 placeholder with the same direction×energy shape EMsoft uses.
    accumsp_np = accum_e_np.astype(np.float32)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        # --- CrystalData ----------------------------------------------------
        _write_crystal_data(f, structure)

        # --- EMData/MCOpenCL + EMData/EBSDmaster/EkeVs ----------------------
        emdata = f.create_group("EMData")
        mcg = emdata.create_group("MCOpenCL")
        mcg.create_dataset("accum_e", data=accum_e_np)
        mcg.create_dataset("accum_z", data=accum_z_np)
        mcg.create_dataset("accumSP", data=accumsp_np)
        mcg.create_dataset("numEbins", data=np.array([nE], dtype=np.int32))
        mcg.create_dataset("numzbins", data=np.array([nz], dtype=np.int32))
        mcg.create_dataset("multiplier", data=np.array([1], dtype=np.int32))
        mcg.create_dataset(
            "totnum_el", data=np.array([int(totnum_el)], dtype=np.int32)
        )
        # load_mc reads EkeVs from EMData/EBSDmaster/EkeVs.
        ebsd = emdata.create_group("EBSDmaster")
        ebsd.create_dataset("EkeVs", data=ekevs_np)

        # --- NMLparameters/MCCLNameList (load_mc reads depthstep/depthmax) --
        nmlp = f.create_group("NMLparameters")
        g = nmlp.create_group("MCCLNameList")
        g.create_dataset("Ehistmin", data=np.array([float(Ehistmin)], dtype=np.float64))
        g.create_dataset("Ebinsize", data=np.array([float(Ebinsize)], dtype=np.float64))
        g.create_dataset("EkeV", data=np.array([float(EkeV)], dtype=np.float64))
        g.create_dataset(
            "depthstep", data=np.array([float(mc.depth_step)], dtype=np.float64)
        )
        g.create_dataset(
            "depthmax", data=np.array([float(mc.depth_max)], dtype=np.float64)
        )
        g.create_dataset("sig", data=np.array([float(sig)], dtype=np.float64))
        g.create_dataset("omega", data=np.array([float(omega)], dtype=np.float64))
        g.create_dataset("numsx", data=np.array([int(numsx)], dtype=np.int32))
        g.create_dataset(
            "totnum_el", data=np.array([int(totnum_el)], dtype=np.int32)
        )
        g.create_dataset("multiplier", data=np.array([1], dtype=np.int32))
        _str_ds(g, "MCmode", "CSDA")
        _str_ds(g, "mode", "full")
        _str_ds(g, "xtalname", xtalname)
        _str_ds(g, "dataname", out.name)

        # --- EMheader (ProgramName = EMMCOpenCL.f90) -----------------------
        header = f.create_group("EMheader")
        _write_emheader(header, "MCOpenCL", _MC_PROGRAM)

    return out_path
