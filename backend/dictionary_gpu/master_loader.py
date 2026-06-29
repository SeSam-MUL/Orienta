"""Load an EMsoft master pattern h5 into a torch tensor.

We expect:
    /EMData/EBSDmaster/mLPNH  shape (nE, 2*npx+1, 2*npx+1) float32
    /EMData/EBSDmaster/mLPSH  shape (nE, 2*npx+1, 2*npx+1) float32
    /EMData/EBSDmaster/EkeVs  shape (nE,)            float64

If the file uses kikuchipy's saved Lambert projection format, the dataset
names are mLPNH / mLPSH (Lambert north / south hemisphere). We pick the
energy plane closest to the requested kV.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch

from backend.forward_sim.io.master_validation import is_lambert_disc_masked


@dataclass
class LoadedMasterPattern:
    """Master pattern data ready for GPU projection."""
    hemispheres: torch.Tensor   # (2, M, M) float32 on device, [0]=upper, [1]=lower
    npx: int                    # half-grid size (M = 2*npx + 1)
    energy_kv: float            # selected energy
    space_group: int            # ITC SG number, 0 if missing
    file_path: str


def load_master_pattern(
    file_path: str,
    energy_kv: Optional[float] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> LoadedMasterPattern:
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"master pattern not found: {p}")

    with h5py.File(p, "r") as f:
        em = f["EMData/EBSDmaster"]
        upper_arr = em["mLPNH"][...]   # (nE, M, M) or (numset, nE, M, M) for EMsoft raw
        lower_arr = em["mLPSH"][...]
        # EMsoft stores (numset, numEbins, M, M); kikuchipy-saved files drop numset to (nE, M, M).
        # Squeeze leading length-1 atom-set axis if present.
        if upper_arr.ndim == 4 and upper_arr.shape[0] == 1:
            upper_arr = upper_arr[0]
            lower_arr = lower_arr[0]
        if upper_arr.ndim != 3 or lower_arr.ndim != 3:
            raise ValueError(
                f"unexpected master pattern shape: upper={upper_arr.shape}, lower={lower_arr.shape}"
            )

        energies = np.asarray(em["EkeVs"][...]).reshape(-1)
        if energy_kv is None:
            idx = int(len(energies) - 1)
            energy_used = float(energies[idx])
        else:
            idx = int(np.argmin(np.abs(energies - energy_kv)))
            energy_used = float(energies[idx])

        upper = upper_arr[idx]
        lower = lower_arr[idx]

        # FAIL LOUD on the disc-masked-Lambert corner bug (pre-2026-06-22 forward-sim
        # masters zeroed the square corners). Such a master projects black holes into
        # Dictionary indexing → wrong NCC → mis-indexing, so refuse it instead of
        # silently producing garbage. The phase must be regenerated.
        if is_lambert_disc_masked(upper):
            raise ValueError(
                f"Master pattern '{p.name}' has a disc-masked Lambert square "
                f"(corners zeroed) — a stale artifact from before the 2026-06-22 "
                f"corner fix. It would produce black-hole patterns and mis-index. "
                f"Delete and regenerate this phase's master pattern."
            )

        sg = 0
        if "CrystalData/SpaceGroupNumber" in f:
            sg = int(np.asarray(f["CrystalData/SpaceGroupNumber"][...]).reshape(-1)[0])

    hemispheres_np = np.stack([upper, lower], axis=0).astype(np.float32, copy=False)
    M = hemispheres_np.shape[-1]
    npx = (M - 1) // 2

    hemispheres = torch.from_numpy(hemispheres_np).to(device=device, dtype=dtype)

    return LoadedMasterPattern(
        hemispheres=hemispheres,
        npx=npx,
        energy_kv=energy_used,
        space_group=sg,
        file_path=str(p),
    )
