"""Inverse SHT: complex SH coefficients -> real spherical signal on DH grid.

Input convention matches ``backend.spherical_gpu.pipeline.sht_io.SHTMasterFile``:
    coefs_ml: complex tensor (bw, bw), indexed [m, l], with l < m entries
    forced to zero by the EMSphInx packing convention.

Output convention: a `(2L, 2L)` real tensor `signal[ti, pj]` where
    theta_ti = pi * (2*ti) / (2*L),     ti in [0, 2L)
    phi_pj   = pi * (2*pj + 1) / (4*L), pj in [0, 2L)
matching ``backend.spherical_gpu._math.sht.grid_DriscollHealy``. ``L`` is the
band-limit actually used for reconstruction (truncated from the file's bw if
``max_bandwidth`` is set).

Implementation:
    Reuses the existing ``CSHT.isht`` (built on the project's stable
    `wigner_d` / log-space Legendre tables). We pack the EMSphInx-format
    `[m, l]` complex coefficient block into the `(b, 2L-1, L)` layout that
    `CSHT.isht` expects, exploit Hermitian symmetry for real-signal
    reconstruction (``a_{l,-m} = (-1)^m * conj(a_{l,m})``), and take the
    real part of the (theoretically real) spherical signal.

Phase A truncates the file's bandwidth to ``max_bandwidth=128`` by default
to keep the CSHT setup memory bounded (Wigner tables scale O(L^3)). For
visual pattern comparison this is plenty (corresponds to ~1.4 deg
detail on the unit sphere). Higher fidelity is a Phase B option.
"""
from __future__ import annotations

from typing import Optional

import torch

from .sht import CSHT


# Default truncation bandwidth. Wigner-d tables for CSHT scale as O(L^3).
# At L=128 the table is ~16 MB at single precision (~32 MB double); at L=384
# it is >400 MB which OOM-ed the conda env on first attempt.
DEFAULT_MAX_BANDWIDTH = 128


def _pack_to_csht_layout(
    coefs_ml: torch.Tensor, bw: int, device: torch.device,
) -> torch.Tensor:
    """Pack `(bw, bw)` `[m, l]` complex coefs into `(1, 2bw-1, bw)` layout
    expected by ``CSHT.isht``.

    The center index `bw - 1` corresponds to m = 0; positive m fills the
    upper half (`bw-1+m`) directly, negative m is reconstructed via
    Hermitian conjugate symmetry for real signals.
    """
    coefs = coefs_ml.to(device=device, dtype=torch.complex128)
    Psi = torch.zeros((1, 2 * bw - 1, bw), dtype=torch.complex128, device=device)

    # m = 0
    Psi[0, bw - 1, :] = coefs[0, :]

    # m > 0: positive frequencies and their Hermitian conjugates
    if bw > 1:
        m_idx = torch.arange(1, bw, device=device)              # (bw-1,)
        signs = ((-1.0) ** m_idx.to(torch.float64)).to(torch.complex128)  # (bw-1,)
        # Positive m
        Psi[0, bw - 1 + m_idx, :] = coefs[m_idx, :]
        # Negative m: a_{l,-m} = (-1)^m * conj(a_{l,m})
        Psi[0, bw - 1 - m_idx, :] = signs[:, None] * coefs[m_idx, :].conj()

    return Psi


def inverse_sht_to_dh_grid(
    coefs_ml: torch.Tensor,
    bandwidth: int,
    device: Optional[torch.device] = None,
    max_bandwidth: int = DEFAULT_MAX_BANDWIDTH,
) -> torch.Tensor:
    """Reconstruct real spherical signal on a Driscoll-Healy `(2L, 2L)` grid.

    Parameters
    ----------
    coefs_ml : torch.Tensor
        Complex coefficients, shape `(bw, bw)`, indexed `[m, l]`. Lower-tri
        entries `l < m` must be zero.
    bandwidth : int
        Bandlimit `bw`; must equal `coefs_ml.shape[0]`.
    device : torch.device, optional
        Output tensor device. Defaults to the device of `coefs_ml`.
    max_bandwidth : int
        Truncate `bw` to at most this value. The reconstructed grid is
        `(2L, 2L)` where `L = min(bw, max_bandwidth)`.

    Returns
    -------
    torch.Tensor
        Real-valued tensor, shape `(2*L, 2*L)`, dtype float32.

    Raises
    ------
    ValueError
        If `coefs_ml.shape != (bw, bw)`.
    """
    bw = int(bandwidth)
    if coefs_ml.shape != (bw, bw):
        raise ValueError(
            f"coefs_ml shape {tuple(coefs_ml.shape)} does not match "
            f"(bandwidth={bw}, bandwidth={bw})"
        )
    if device is None:
        device = coefs_ml.device

    L = min(bw, int(max_bandwidth))
    if L < 1:
        raise ValueError(f"truncated bandwidth must be >= 1, got {L}")

    # Truncate
    coefs_trunc = coefs_ml[:L, :L]

    # Pack into CSHT-expected (b, 2L-1, L) layout with Hermitian conjugate
    # symmetry for real-signal reconstruction
    Psi = _pack_to_csht_layout(coefs_trunc, L, device=device)

    # Reuse the project's CSHT.isht — it builds Wigner tables internally
    csht = CSHT(L=L, device=device, precision="double")
    psi_complex = csht.isht(Psi)        # (1, 2L, 2L) complex
    return psi_complex[0].real.to(torch.float32).contiguous()
