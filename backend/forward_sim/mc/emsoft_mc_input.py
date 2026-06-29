"""SP1 — read EMsoft Monte-Carlo output (``EMData/MCOpenCL``) into ``MCData``.

For SP0+SP1 the Monte-Carlo energy/depth distribution is **not** recomputed on
the GPU; it is read from EMsoft's pre-existing master ``.h5`` as a fixed,
validated input (the GPU MC port is deferred to SP2).  This module provides
that reader.

Confirmed HDF5 layout (verified against the real Ni oracle ``.h5``,
2026-06-10):

* ``EMData/MCOpenCL/accum_e`` — shape ``(numsx, numsy, nE)`` int32,
  direction x energy back-scatter electron counts.  Here ``(501, 501, 11)``.
* ``EMData/MCOpenCL/accum_z`` — shape ``(nx, ny, nz, nE)`` int32, the depth
  histogram.  Here ``(51, 51, 101, 11)`` with the **depth axis = axis 2**
  (``nz = 101``) and the **energy axis = axis 3** (``nE = 11``).
* ``EMData/EBSDmaster/EkeVs`` — shape ``(nE,)`` float, the bin energies (kV);
  here ``[10, 11, ..., 20]``.
* ``NMLparameters/MCCLNameList/{depthstep, depthmax, Ebinsize, Ehistmin}`` —
  the Monte-Carlo depth/energy binning (``depthstep = 1.0`` nm,
  ``depthmax = 100.0`` nm for Ni).

The depth axis of ``accum_z`` decays from a near-surface peak to zero at
``depthmax`` (verified on Ni: argmax at bin 1, tail bins all zero), so the
normalised :meth:`MCData.lambda_z` is a sensible escape-depth distribution.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import h5py
import torch

from backend.forward_sim.runtime import get_device

# HDF5 paths (constants so the layout is documented in one place).
_MC_GROUP = "EMData/MCOpenCL"
_ACCUM_E = f"{_MC_GROUP}/accum_e"
_ACCUM_Z = f"{_MC_GROUP}/accum_z"
_EKEVS = "EMData/EBSDmaster/EkeVs"
_NML = "NMLparameters/MCCLNameList"

# Axis indices into accum_z (nx, ny, nz, nE).
_ACCUM_Z_DEPTH_AXIS = 2
_ACCUM_Z_ENERGY_AXIS = 3
# Axis index into accum_e (numsx, numsy, nE).
_ACCUM_E_ENERGY_AXIS = 2


@dataclass
class MCData:
    """EMsoft Monte-Carlo energy/depth distribution for one phase.

    Attributes:
        accum_e: ``(numsx, numsy, nE)`` direction x energy back-scatter counts.
        accum_z: ``(nx, ny, nz, nE)`` depth histogram (depth axis 2, energy 3).
        EkeVs: ``(nE,)`` bin energies in kV.
        depth_step: depth-bin width in nm (``depthstep``).
        depth_max: maximum tracked depth in nm (``depthmax``).
    """

    accum_e: torch.Tensor
    accum_z: torch.Tensor
    EkeVs: torch.Tensor
    depth_step: float
    depth_max: float

    @property
    def n_energy(self) -> int:
        """Number of energy bins (``nE``)."""
        return int(self.EkeVs.shape[0])

    def _check_energy_idx(self, energy_idx: int) -> None:
        if not 0 <= energy_idx < self.n_energy:
            raise ValueError(
                f"energy_idx {energy_idx} out of range [0, {self.n_energy})"
            )

    def lambda_z(self, energy_idx: int) -> torch.Tensor:
        """Normalised depth distribution ``lambda(z)`` for one energy bin.

        The depth profile is ``accum_z`` summed over the two direction axes
        ``(nx, ny)`` for the given energy, then normalised to sum to 1 over the
        depth axis.

        Args:
            energy_idx: index into the energy axis (``0 .. nE-1``).

        Returns:
            A ``(nz,)`` float tensor that sums to 1.

        Raises:
            ValueError: if ``energy_idx`` is out of range, or if the depth
                histogram is empty for this energy (zero total counts).
        """
        self._check_energy_idx(energy_idx)
        # accum_z: (nx, ny, nz, nE) -> select energy -> (nx, ny, nz).
        slab = self.accum_z[..., energy_idx].to(torch.float64)
        # Sum over the two direction axes (0, 1) -> depth profile (nz,).
        depth_profile = slab.sum(dim=(0, 1))
        total = depth_profile.sum()
        if total <= 0:
            raise ValueError(
                f"accum_z has zero total counts for energy_idx {energy_idx}; "
                "cannot normalise depth distribution"
            )
        return (depth_profile / total).to(torch.float32)

    def energy_weight(self, energy_idx: int) -> torch.Tensor:
        """Per-direction energy weight grid for one energy bin (from accum_e).

        Args:
            energy_idx: index into the energy axis (``0 .. nE-1``).

        Returns:
            A ``(numsx, numsy)`` non-negative float tensor — the back-scatter
            electron count per Lambert direction at this energy.

        Raises:
            ValueError: if ``energy_idx`` is out of range.
        """
        self._check_energy_idx(energy_idx)
        return self.accum_e[..., energy_idx].to(torch.float32)


def _scalar_float(value) -> float:
    """Coerce an HDF5 scalar / length-1 array to a Python float."""
    import numpy as np

    arr = np.asarray(value).ravel()
    return float(arr[0])


def _ekevs_from_mc_nml(f, accum_e_np, src: str):
    """Reconstruct the bin energies from the Monte-Carlo NML group.

    A standalone EMMCOpenCL ``.h5`` has no ``EMData/EBSDmaster/EkeVs`` (that is
    written by the master step); the energy grid is instead derivable from
    ``NMLparameters/MCCLNameList/{Ehistmin, Ebinsize}`` and the number of bins
    (``EMData/MCOpenCL/numEbins`` or, as a last resort, the energy axis of
    ``accum_e``).  EMsoft fills ``EkeVs[i] = Ehistmin + i*Ebinsize``.

    Args:
        f: open ``h5py.File``.
        accum_e_np: the loaded ``accum_e`` array (energy axis = last axis).
        src: source path (for the error message).

    Returns:
        A ``numpy`` 1-D array of bin energies (kV).

    Raises:
        ValueError: if neither ``EkeVs`` nor the MC NML energy params exist.
    """
    import numpy as np

    nml = f.get(_NML)
    if nml is None or "Ehistmin" not in nml or "Ebinsize" not in nml:
        raise ValueError(
            f"{_EKEVS} dataset missing in {src!r} and the Monte-Carlo NML "
            f"({_NML}/Ehistmin,Ebinsize) is not present either — cannot "
            f"reconstruct the bin energies."
        )
    ehistmin = _scalar_float(nml["Ehistmin"][()])
    ebinsize = _scalar_float(nml["Ebinsize"][()])

    n_e = None
    num_ebins_ds = f.get(f"{_MC_GROUP}/numEbins")
    if num_ebins_ds is not None:
        n_e = int(np.asarray(num_ebins_ds[()]).ravel()[0])
    if not n_e or n_e <= 0:
        n_e = int(accum_e_np.shape[_ACCUM_E_ENERGY_AXIS])

    return ehistmin + np.arange(n_e, dtype="float64") * ebinsize


def load_mc(master_or_mc_h5: str) -> MCData:
    """Read EMsoft Monte-Carlo data from a master / MC ``.h5``.

    Reads ``EMData/MCOpenCL/{accum_e, accum_z}`` and
    ``NMLparameters/MCCLNameList/{depthstep, depthmax}``.  The bin energies are
    taken from ``EMData/EBSDmaster/EkeVs`` when present (master ``.h5``); for a
    standalone EMMCOpenCL ``.h5`` (which has NO master group yet) they are
    reconstructed from the Monte-Carlo NML as ``Ehistmin + i*Ebinsize`` for
    ``i in [0, numEbins)`` — the same grid EMEBSDmaster writes to ``EkeVs``.

    Args:
        master_or_mc_h5: path to an EMsoft master ``.h5`` (which also carries
            the MC output) or a standalone EMMCOpenCL MC ``.h5``.

    Returns:
        A populated :class:`MCData` with tensors on the forward-sim device.

    Raises:
        ValueError: if the ``EMData/MCOpenCL`` group (or a required dataset
            within it) is missing, or if the bin energies can neither be read
            from ``EkeVs`` nor reconstructed from the MC NML.
    """
    if not os.path.exists(master_or_mc_h5):
        raise ValueError(f"MC file not found: {master_or_mc_h5!r}")

    device = get_device()

    with h5py.File(master_or_mc_h5, "r") as f:
        if _MC_GROUP not in f:
            raise ValueError(
                f"EMData/MCOpenCL group missing in {master_or_mc_h5!r}"
            )
        for ds in (_ACCUM_E, _ACCUM_Z):
            if ds not in f:
                raise ValueError(
                    f"{ds} dataset missing in {master_or_mc_h5!r}"
                )

        accum_e_np = f[_ACCUM_E][()]
        accum_z_np = f[_ACCUM_Z][()]

        nml = f.get(_NML)
        if nml is not None and "depthstep" in nml:
            depth_step = _scalar_float(nml["depthstep"][()])
        else:
            depth_step = 1.0
        if nml is not None and "depthmax" in nml:
            depth_max = _scalar_float(nml["depthmax"][()])
        else:
            depth_max = float(accum_z_np.shape[_ACCUM_Z_DEPTH_AXIS]) * depth_step

        # Bin energies: prefer the master's EkeVs; otherwise reconstruct from
        # the MC NML (a fresh EMMCOpenCL .h5 has no EBSDmaster group).
        if _EKEVS in f:
            ekevs_np = f[_EKEVS][()]
        else:
            ekevs_np = _ekevs_from_mc_nml(f, accum_e_np, master_or_mc_h5)

    accum_e = torch.as_tensor(accum_e_np, device=device)
    accum_z = torch.as_tensor(accum_z_np, device=device)
    ekevs = torch.as_tensor(ekevs_np, device=device).to(torch.float32).ravel()

    return MCData(
        accum_e=accum_e,
        accum_z=accum_z,
        EkeVs=ekevs,
        depth_step=depth_step,
        depth_max=depth_max,
    )
