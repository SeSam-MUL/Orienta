"""SP3 (part 2) — serialise our GPU master into an EMSphInx ``.sht`` file.

The output is a faithful EMSphInx **spherical-harmonics** master file (the binary
``*sht`` format authored by William C. Lenthe, 2019; the same format EMSphInx's
``mp2sht`` produces and the project reader
:mod:`backend.spherical_gpu.pipeline.sht_io` consumes).  We do **not** re-run
EMsoft / EMSphInx — we take the northern-hemisphere Lambert master
:func:`backend.forward_sim.dynamical.master_builder.build_master` produced,
forward-transform it to spherical-harmonic coefficients, and write the exact byte
layout the reader (and the EMSphInx C++ ``sht::File::read``) expect.

What the format is (verified against ``Database/EBSD_SHT_Database/Ni/...sht``)
----------------------------------------------------------------------------
Self-contained little-endian binary (NOT HDF5).  Block order (the C++
``sht::File::write`` order — :file:`tasks/_research_sht/sht_file.in.hpp` and the
project reader :mod:`...sht_io`)::

    FileHeader            40 bytes fixed + doi + notes (each 8-byte padded)
        magic '*sht', version (1,1), reserved(2), software(8), modality(1),
        reserved(3), beamEnergy f4, primaryAngle f4, secondaryAngle f4,
        reservedParam f4, doiLen i2, noteLen i2
    MasterPatternData      8 bytes fixed
        numXtal i1, sgEff u1, pijk i1=+1, rotSense i1=112('p'),
        modality i1, vendor i1=EMsoft(1), simMetaSize i2
      per crystal:
        CrystalData       72 bytes fixed + atoms(32 each) + 5 padded strings
            sgNum u1, sgSet i1, sgAxis i1=1, sgCell i1=1, oriXYZ 3f4,
            lattice 6f4 (a,b,c in nm, angles deg), rot 4f4 (w,x,y,z),
            weight f4, numAtoms i2, {formula,name,symbol,refs,note}Len 5i2
        AtomData(32)      x,y,z in 24ths (3f4), occ f4, charge f4, debWal f4
                          (nm^2), resFp f4, atZ i1, reserved 3 bytes
      per crystal:
        SimulationData    simMetaSize bytes (EMsoftED = 88)
    HarmonicsData          8 bytes fixed + doubCnt*8 (LE f8, symmetry-packed)
        bw i2, zRot i1, cmpFlg i1, doubCnt i4
    CRC-32C trailer        u4  (the EMSphInx-specific table; see _crc32c_emsoft)

The harmonic coefficients are the **symmetry-compressed** real-double packing the
reader's ``UnpackHarm`` inverts: only the non-systematically-zero ``alm[m, l]``
(``l >= m``) entries are stored, in the order ``PackHarm`` writes them, governed by
``(zRot, cmpFlg)`` derived from the effective space group.

Coefficient convention (the load-bearing part)
----------------------------------------------
The reader hands its ``coefs_ml`` straight to
:func:`backend.spherical_gpu._math.sht_inverse.inverse_sht_to_dh_grid`, which packs
``[m, l]`` into the CSHT layout and runs ``CSHT.isht``.  We therefore produce
``coefs_ml`` as the **exact inverse of that path**: reconstruct the master on the
Driscoll-Healy ``(2L, 2L)`` grid, run the project's own ``CSHT.fsht``, and take the
``m >= 0`` half of the resulting ``(2L-1, L)`` coefficients —
``coefs_ml[m, l] = Psi[(L-1)+m, l]``.  Because the forward and inverse use the same
project SHT machinery, ``inverse_sht_to_dh_grid(write-then-read)`` reproduces the
input grid to FP precision (NCC ~ 1.0), regardless of EMSphInx's internal
normalisation — the roundtrip is closed within our own pipeline, which is exactly
what the SP-GPU indexer/forward path needs.

Master → sphere reconstruction
------------------------------
``our_master_NH`` lives on the modified-Lambert square (NH only).  For each DH grid
direction we map to the Lambert square via the kikuchipy/EMsoft-validated
:func:`backend.dictionary_gpu.lambert.direction_to_lambert` (using ``|z|`` so the
southern hemisphere mirrors the NH — exact for the centrosymmetric/cubic SP0+SP1
scope) and bilinearly sample (``align_corners=True``, the dictionary pipeline's
sampler).  This reuses the same primitive ``master_h5._stereographic_from_lambert``
uses.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..crystal.xtal_io import CrystalStructure
from backend.dictionary_gpu.lambert import direction_to_lambert
from backend.spherical_gpu._math.sht import CSHT, grid_DriscollHealy, theta_phi_to_xyz
from backend.spherical_gpu._math._wigner_logspace import csht_weights_half_pi

# --- format constants (sht_file.in.hpp) --------------------------------------
_MAGIC_LE = b"*sht"
_VERSION = (1, 1)
_MODALITY_EBSD = 1
_VENDOR_EMSOFT = 1
_PIJK_EMSOFT = 1          # EMsoft is always pijk +1
_ROTSENSE_PASSIVE = 112   # ascii 'p'
_SIMMETA_EMSOFTED = 88    # size of the EMsoftED SimulationData block

# EMSphInx production band-limit: the real EMsoft .sht files in
# ``Database/EBSD_SHT_Database`` (e.g. the Ni cF4 20 kV oracle) are written at
# bw=384.  Sharp Kikuchi bands are NOT band-limited at low bw (the iter-9 bw=88
# write reconstructed at only NCC≈0.83 — a band-limit ceiling, not a serialiser
# defect), so 384 is the default that captures full band sharpness (Ni recon
# NCC≈0.99 from the npx=500 master, == the pure-CSHT band ceiling).
_PRODUCTION_BANDWIDTH = 384


# --- device resolution + Wigner-table warm-up (LEVER 2) -----------------------
def _resolve_sht_device(device: torch.device | str | None) -> torch.device:
    """Resolve the SHT compute device.

    ``None`` (the sensible default) means **CUDA if available, else CPU** — the
    CSHT/DLT transform is a sparse-dense matmul + FFTs that runs far faster on the
    GPU, and EMsoft's own ``.sht`` writer (``EMEBSDmasterSHT``/``DSHT.f90``) is
    serial-CPU with no GPU path, so any correct CUDA transform structurally beats
    it.  An explicit ``"cuda"`` that is unavailable falls back to CPU (fail-safe,
    with a warning) rather than crashing the write.

    NB (reproducibility): the CUDA and CPU transforms agree to ~3e-10 (fp64
    reduction-order jitter) but are NOT byte-identical, so a ``.sht`` written on GPU
    vs CPU carries a *different CRC* (the coefficients are value-equal, both files
    load and index correctly; ``sht_io`` parses but does not enforce the CRC).  Pass
    ``device="cpu"`` if you need a machine-independent, byte-reproducible ``.sht``.
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        import warnings

        warnings.warn(
            "sht_writer: device='cuda' requested but CUDA is unavailable; "
            "falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")
    return dev


def warm_sht_cache(
    bandwidth: int = _PRODUCTION_BANDWIDTH,
    device: torch.device | str | None = None,
) -> torch.device:
    """Pre-build (and cache) the O(L³) Wigner half-π table for ``write_sht``.

    The CSHT/DLT transform's dominant cold cost is building the Wigner little-``d``
    table (``csht_weights_half_pi``), ~22 s at ``bw=384`` and rebuilt fresh in every
    fresh process.  That table is a pure function of ``(L, device, precision)`` and is
    already LRU-memoized in :mod:`backend.spherical_gpu._math._wigner_logspace`.  This
    helper triggers that one build up-front so a subsequent batch of ``.sht`` writes at
    the same ``(bandwidth, device)`` all hit the warm cache (table cost ≈ 0 after the
    first).  ``write_sht`` uses precision ``"double"``, so the table is warmed in
    double precision to match the key the writer hits.

    Args:
        bandwidth: band-limit ``bw`` to warm (the ``.sht`` ``bw``; default production 384).
        device: torch device; ``None`` => CUDA-if-available else CPU (same rule as
            :func:`write_sht`).

    Returns:
        The resolved :class:`torch.device` the table was warmed on.
    """
    dev = _resolve_sht_device(device)
    csht_weights_half_pi(int(bandwidth), device=dev, precision="double")
    return dev


# Element symbols indexed by Z-1 (mp2sht.cpp formula estimate table).
_AT_SYB = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al",
    "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]

# HarmonicsData::SpaceGroupRot — z rotational order per space group (1..230).
# Verbatim port of the 230-entry table (sht_file.in.hpp:1838).
_SG_ROT = (
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 3, 3, 3, 3, 6, 6, 6, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4,
    4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
)

# HarmonicsData::SpaceGroupCmp — compression flag per space group (1..230).
# Verbatim port of the 230-entry table (sht_file.in.hpp:1858).
_SG_CMP = (
    0x0, 0x1, 0x0, 0x0, 0x0, 0x4, 0x4, 0x4, 0x4, 0x5, 0x5, 0x5, 0x5, 0x5,
    0x5, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x4, 0x4, 0x4, 0x4,
    0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4,
    0x4, 0x4, 0x4, 0x4, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7,
    0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7,
    0x7, 0x7, 0x7, 0x7, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x3, 0x3,
    0x3, 0x3, 0x3, 0x3, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x8, 0x8,
    0x8, 0x8, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x8, 0x8, 0x7, 0x7, 0x7, 0x7,
    0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7,
    0x7, 0x7, 0x0, 0x0, 0x0, 0x0, 0x1, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x8, 0x4, 0x8, 0x4, 0x8, 0x8, 0x5, 0x5, 0x9, 0x9, 0x9, 0x9, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x2, 0x3, 0x3, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x4, 0x4, 0x4, 0x4, 0xA, 0xA, 0x6, 0x6, 0x7, 0x7, 0x7, 0x7, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x7, 0x7, 0x7, 0x7,
    0x7, 0x7, 0x7, 0x7, 0x7, 0x7,
)


# --- EMSphInx-specific CRC-32C (the embedded table in sht_file.in.hpp) --------
def _build_emsoft_crc_lut() -> list[int]:
    """Build the exact 256-entry CRC LUT EMSphInx embeds (sht_file.in.hpp:967).

    This is NOT the standard reflected CRC-32C: the table is generated by
    shifting right but XORing with the **non-reflected** polynomial
    ``0x1edc6f41`` (so entries stay < 2**29, e.g. LUT[1] = 0x0a5f4d75).
    Verified bit-exact against the real Ni ``.sht`` trailer.
    """
    poly = 0x1EDC6F41
    lut: list[int] = []
    for i in range(256):
        v = i
        for _ in range(8):
            xr = v & 1
            v >>= 1
            if xr:
                v ^= poly
            v &= 0xFFFFFFFF
        lut.append(v)
    return lut


_CRC_LUT = _build_emsoft_crc_lut()


def _crc32c_emsoft(data: bytes, crc: int = 0) -> int:
    """EMSphInx ``detail::crc32c`` (sht_file.in.hpp:947), table-driven.

    ``crc = ~crc; for b: crc = (crc>>8) ^ LUT[(crc & 0xFF) ^ b]; return ~crc``.
    Chains contiguously (``crc32c(B, crc32c(A)) == crc32c(A+B)``), so the whole
    file body ``raw[:-4]`` can be hashed in a single call — verified bit-exact
    against the real Ni ``.sht`` (stored trailer 0xee994930).
    """
    c = (~crc) & 0xFFFFFFFF
    for b in data:
        c = ((c >> 8) ^ _CRC_LUT[(c & 0xFF) ^ b]) & 0xFFFFFFFF
    return (~c) & 0xFFFFFFFF


def _pad8(n: int) -> int:
    return n if (n % 8 == 0) else n + (8 - (n % 8))


def _pad8_bytes(s: bytes) -> bytes:
    """Right-pad ``s`` with NUL to a multiple of 8 (the format's string rule)."""
    return s + b"\x00" * (_pad8(len(s)) - len(s))


# --- master (Lambert NH [+ SH]) -> Driscoll-Healy FULL sphere -----------------
def _lambert_nh_to_dh_grid(
    lambert_nh: torch.Tensor,
    npx: int,
    L: int,
    device: torch.device,
    lambert_sh: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample the Lambert master(s) onto the FULL ``(2L, 2L)`` Driscoll-Healy sphere.

    The DH grid covers the **whole** sphere — ``theta`` spans ``[0, π)``, so its
    lower rows (``theta > π/2``) are southern-hemisphere directions (``z < 0``).
    EMsoft's ``SH_analyze`` likewise transforms the full sphere from BOTH Lambert
    hemispheres (``mLPNH``, ``mLPSH``), encoding the equator-antisymmetric (odd
    ``l+m``) harmonics that distinguish a non-centrosymmetric crystal.

    For each DH grid point ``(theta, phi)`` we form the unit direction and map it to
    a Lambert-square coordinate via :func:`direction_to_lambert` (which uses ``|z|``,
    so a northern and its mirror-southern direction land on the SAME square cell of
    their respective hemisphere array).  Directions with ``z ≥ 0`` are sampled from
    ``lambert_nh``; directions with ``z < 0`` from ``lambert_sh``.  Bilinear sample
    (``align_corners=True``); ``padding_mode='zeros'`` only fires for the unreachable
    ``|coord| > 1`` (the DH directions map to ``|coord| ≤ 0.997`` in practice, never
    the exact square corner).  The Lambert master itself is now FULLY populated —
    its square CORNERS are valid equatorial directions filled by ``build_master``
    (the 2026-06-22 corner fix; previously they were zeroed, producing four black
    equatorial diamonds in this reconstruction).

    When ``lambert_sh`` is ``None`` the southern directions are sampled from
    ``lambert_nh`` as well — i.e. the master is treated as equator-symmetric
    (``NH == SH``), which is **exact** for a centrosymmetric/cubic cell and is
    **bit-identical** to the legacy NH-only behaviour (every DH direction samples
    ``lambert_nh`` at the same grid coordinate as before).

    Args:
        lambert_nh: ``(2·npx+1, 2·npx+1)`` northern master on the Lambert square.
        npx: Lambert half-grid size.
        L: target band-limit (output grid is ``(2L, 2L)``).
        device: torch device for the sampling.
        lambert_sh: optional ``(2·npx+1, 2·npx+1)`` southern master; when ``None``
            the northern array is reused for the southern hemisphere (NH == SH).

    Returns:
        ``(2L, 2L)`` float64 real signal on the full-sphere DH grid.
    """
    m = 2 * npx + 1
    dh = grid_DriscollHealy(L, torch.float64).to(device)        # (2L, 2L, 2)
    xyz = theta_phi_to_xyz(dh)                                  # (2L, 2L, 3)
    dirs = xyz.reshape(-1, 3)
    xy = direction_to_lambert(dirs)                            # (P, 2) in [-1, 1]
    grid = xy.reshape(1, 2 * L, 2 * L, 2).to(torch.float32)
    img_nh = lambert_nh.to(device).reshape(1, 1, m, m).to(torch.float32)
    samp_nh = F.grid_sample(
        img_nh, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    ).reshape(2 * L, 2 * L)
    if lambert_sh is None:
        # NH == SH (centrosymmetric/cubic): every DH direction samples the NH
        # array, identical to the legacy NH-only path — bit-identical output.
        return samp_nh.to(torch.float64)
    # Full sphere: southern (z < 0) directions sample the SH array.
    img_sh = lambert_sh.to(device).reshape(1, 1, m, m).to(torch.float32)
    samp_sh = F.grid_sample(
        img_sh, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    ).reshape(2 * L, 2 * L)
    z = dirs[:, 2].reshape(2 * L, 2 * L)
    samp = torch.where(z >= 0.0, samp_nh, samp_sh)
    return samp.to(torch.float64)


# --- forward SHT: DH grid -> coefs_ml[m, l] ----------------------------------
def _forward_sht_coefs(
    dh_signal: torch.Tensor, bw: int, device: torch.device
) -> torch.Tensor:
    """Forward spherical-harmonic transform of a real DH-grid signal.

    Runs the project's :class:`CSHT` ``fsht`` (the exact transform whose inverse
    the reader path :func:`inverse_sht_to_dh_grid` uses) and extracts the
    ``m >= 0`` half into the reader's ``coefs_ml[m, l]`` layout.  The ``CSHT``
    coefficient block is ``(2bw-1, bw)`` with centre index ``bw-1`` = ``m = 0``,
    so ``coefs_ml[m, l] = Psi[(bw-1)+m, l]``.  Lower-triangle ``l < m`` is forced
    to 0 (it is ~0 already; the reader treats it as systematic zero).

    Args:
        dh_signal: ``(2bw, 2bw)`` real signal on the DH grid.
        bw: band-limit.
        device: torch device.

    Returns:
        ``(bw, bw)`` complex128 coefficients indexed ``[m, l]``.
    """
    csht = CSHT(L=bw, device=device, precision="double")
    psi = csht.fsht(dh_signal.to(torch.float64).unsqueeze(0))   # (1, 2bw-1, bw)
    psi0 = psi[0]
    m_idx = torch.arange(bw, device=device)
    coefs = psi0.index_select(0, (bw - 1) + m_idx).to(torch.complex128)  # (bw, bw)
    # Zero the systematic lower triangle (l < m).
    ll = torch.arange(bw, device=device)
    mask = ll[None, :] >= m_idx[:, None]
    coefs = torch.where(mask, coefs, torch.zeros_like(coefs))
    return coefs.contiguous()


# --- harmonic compression (port of HarmonicsData::PackHarm) -------------------
def _num_harm(bw: int, n: int, f: int) -> int:
    """Port of ``HarmonicsData::NumHarm`` (sht_file.in.hpp:1672)."""
    inv = bool(f & 0x01)
    mir_z = bool(f & 0x02)
    mir_y = bool(f & 0x04)
    mir_x = bool(f & 0x08)
    if mir_x and mir_y:
        raise ValueError("compression flags 0x04 and 0x08 are mutually exclusive")
    num = 0
    for m in range(bw):
        if n > 1 and m % n != 0:
            continue
        for ll in range(m, bw):
            if inv and ll % 2 != 0:
                continue
            if mir_z and (ll + m) % 2 != 0:
                continue
            num += 1
    if not (mir_x or mir_y):
        num *= 2
    return num


def _pack_harm(coefs_ml: np.ndarray, bw: int, n: int, f: int) -> np.ndarray:
    """Port of ``HarmonicsData::PackHarm`` (sht_file.in.hpp:1707).

    Compresses the dense ``(bw, bw)`` ``[m, l]`` complex coefficients into the
    real-double packed array the reader's ``UnpackHarm`` inverts.

    **Vectorised (LEVER 4) — bit-identical to the reference (m, l) double loop.**
    The packed-double order is preserved exactly: outer loop over ``m`` ascending
    (rows with ``m % n != 0`` dropped when ``n>1``), inner over ``l`` ascending
    (``l >= m`` minus the ``inv``/``mir_z`` parity drops), and for a complex
    (``t==0``) row the ``[real, imag]`` interleave per surviving ``l``.  Only the
    inner ``l`` sweep is vectorised with a numpy boolean mask + a column-stacked
    interleave (``ravel`` of the ``(k, 2)`` real/imag stack reproduces the loop's
    ``append(real); append(imag)`` order byte-for-byte); the ``m`` loop is left as
    a cheap Python loop (≤ ``bw`` = 384 iterations).  ``float(v.real)`` →
    ``v.real`` is an exact IEEE-754-double identity, so the bytes match.

    Args:
        coefs_ml: ``(bw, bw)`` complex128 array indexed ``[m, l]``.
        bw, n, f: bandwidth, z-rotational order, compression flag.

    Returns:
        1-D float64 array of length ``NumHarm(bw, n, f)``.
    """
    inv = bool(f & 0x01)
    mir_z = bool(f & 0x02)
    mir_y = bool(f & 0x04)
    mir_x = bool(f & 0x08)
    if mir_x and mir_y:
        raise ValueError("compression flags 0x04 and 0x08 are mutually exclusive")

    coefs = np.ascontiguousarray(coefs_ml, dtype=np.complex128)
    real = coefs.real
    imag = coefs.imag
    ll_all = np.arange(bw)                      # l indices 0..bw-1

    chunks: list[np.ndarray] = []
    for m in range(bw):
        if n > 1 and m % n != 0:
            continue
        # row type: 0 complex, 1 real, 2 imaginary (identical to the loop).
        if mir_y:
            t = 1
        elif mir_x:
            t = 1 if (m % (n * 2) == 0) else 2
        else:
            t = 0
        # surviving l for this row: l >= m, minus the inv / mir_z parity drops.
        keep = ll_all >= m
        if inv:
            keep &= (ll_all % 2) == 0
        if mir_z:
            keep &= ((ll_all + m) % 2) == 0
        sel = np.nonzero(keep)[0]
        if sel.size == 0:
            continue
        if t == 0:
            # [real, imag] interleaved per surviving l: column_stack then ravel
            # reproduces append(real); append(imag) exactly.
            pair = np.empty((sel.size, 2), dtype="<f8")
            pair[:, 0] = real[m, sel]
            pair[:, 1] = imag[m, sel]
            chunks.append(pair.reshape(-1))
        elif t == 1:
            chunks.append(np.ascontiguousarray(real[m, sel], dtype="<f8"))
        else:
            chunks.append(np.ascontiguousarray(imag[m, sel], dtype="<f8"))

    arr = (
        np.concatenate(chunks).astype("<f8", copy=False)
        if chunks
        else np.empty(0, dtype="<f8")
    )
    expected = _num_harm(bw, n, f)
    if arr.shape[0] != expected:
        raise ValueError(
            f"PackHarm produced {arr.shape[0]} doubles, NumHarm expects {expected}"
        )
    return arr


# --- binary block builders ---------------------------------------------------
def _build_file_header(
    energy_kV: float, primary_tilt_deg: float, secondary_tilt_deg: float,
    doi: bytes, notes: bytes, software: bytes,
) -> bytes:
    """FileHeader: 40 fixed bytes + padded doi + padded notes."""
    sw = (software + b"\x00" * 8)[:8]
    fixed = bytearray(40)
    fixed[0:4] = _MAGIC_LE
    fixed[4] = _VERSION[0]
    fixed[5] = _VERSION[1]
    # reserved [6:8] = 0
    fixed[8:16] = sw
    fixed[16] = _MODALITY_EBSD
    # reserved [17:20] = 0
    struct.pack_into(
        "<ffff", fixed, 20,
        float(energy_kV), float(primary_tilt_deg), float(secondary_tilt_deg), 0.0,
    )
    struct.pack_into("<hh", fixed, 36, len(doi), len(notes))
    return bytes(fixed) + _pad8_bytes(doi) + _pad8_bytes(notes)


def _build_master_pattern_data(
    num_xtal: int, sg_eff: int, sim_meta_size: int
) -> bytes:
    """MasterPatternData: 8 fixed bytes."""
    fixed = bytearray(8)
    fixed[0] = num_xtal & 0xFF
    fixed[1] = sg_eff & 0xFF
    struct.pack_into("<b", fixed, 2, _PIJK_EMSOFT)
    struct.pack_into("<b", fixed, 3, _ROTSENSE_PASSIVE)
    fixed[4] = _MODALITY_EBSD
    fixed[5] = _VENDOR_EMSOFT
    struct.pack_into("<h", fixed, 6, sim_meta_size)
    return bytes(fixed)


def _coord_24ths(x: float) -> float:
    """Fractional coord -> 24ths, EMsoft's special-casing of 6ths (addDataEMsoft)."""
    x = float(np.fmod(x, 1.0))
    if x < 0.0:
        x += 1.0
    if abs(x - 1.0 / 6.0) < 1e-12:
        return 4.0
    if abs(x - 1.0 / 3.0) < 1e-12:
        return 8.0
    if abs(x - 2.0 / 3.0) < 1e-12:
        return 16.0
    if abs(x - 5.0 / 6.0) < 1e-12:
        return 20.0
    return x * 24.0


def _build_atom(at) -> bytes:
    """AtomData: 32 bytes (x,y,z in 24ths, occ, charge=0, debWal, resFp=0, atZ)."""
    buf = bytearray(32)
    x, y, z = at.xyz
    struct.pack_into(
        "<7f", buf, 0,
        _coord_24ths(x), _coord_24ths(y), _coord_24ths(z),
        float(at.occ), 0.0, float(at.B), 0.0,
    )
    # atZ is a signed int8 in the format; atomic numbers (1..118) all fit.
    struct.pack_into("<b", buf, 28, int(at.Z))
    return bytes(buf)


def _build_crystal_data(
    structure: CrystalStructure,
    formula: bytes, name: bytes, symbol: bytes, refs: bytes, note: bytes,
) -> bytes:
    """CrystalData: 72 fixed bytes + atoms + 5 padded strings."""
    a, b, c, al, be, ga = (float(v) for v in structure.lattice)
    fixed = bytearray(72)
    fixed[0] = int(structure.space_group) & 0xFF
    struct.pack_into("<b", fixed, 1, 1)   # sgSet = 1 (default origin)
    struct.pack_into("<b", fixed, 2, 1)   # sgAxis = Default
    struct.pack_into("<b", fixed, 3, 1)   # sgCell = Default
    struct.pack_into("<fff", fixed, 4, 0.0, 0.0, 0.0)            # origin shift
    struct.pack_into("<6f", fixed, 16, a, b, c, al, be, ga)     # lattice
    struct.pack_into("<4f", fixed, 40, 1.0, 0.0, 0.0, 0.0)     # rotation (identity)
    struct.pack_into("<f", fixed, 56, 1.0)                      # weight
    struct.pack_into("<h", fixed, 60, len(structure.atoms))     # numAtoms
    struct.pack_into(
        "<5h", fixed, 62,
        len(formula), len(name), len(symbol), len(refs), len(note),
    )
    body = bytearray()
    for at in structure.atoms:
        body += _build_atom(at)
    body += _pad8_bytes(formula)
    body += _pad8_bytes(name)
    body += _pad8_bytes(symbol)
    body += _pad8_bytes(refs)
    body += _pad8_bytes(note)
    return bytes(fixed) + bytes(body)


def _build_emsoft_ed(
    energy_kV: float, primary_tilt_deg: float, secondary_tilt_deg: float,
    dmin: float, npx: int, numsx: int, totnum_el: int, bethe_params,
    emsoft_version: bytes,
) -> bytes:
    """SimulationData (EMsoftED): 88 bytes (sht_file.in.hpp:741)."""
    buf = bytearray(88)
    buf[0:8] = (emsoft_version + b"\x00" * 8)[:8]
    c1, c2, c3, sgdbdiff = (float(v) for v in bethe_params)
    # sigStart/End/Step, omega, keV, eHistMin, eBinSize, depthMax, depthStep, thickness
    struct.pack_into(
        "<10f", buf, 8,
        float(primary_tilt_deg), 0.0, 0.0, float(secondary_tilt_deg),
        float(energy_kV), 10.0, 1.0, 100.0, 1.0, float("inf"),
    )
    struct.pack_into("<q", buf, 48, int(totnum_el))            # totNumEl (i8)
    struct.pack_into("<h", buf, 56, int(numsx))                # numSx (i2)
    # reserved [58:60] = 0
    struct.pack_into("<4f", buf, 60, c1, c2, c3, sgdbdiff)     # c1,c2,c3,sigDbDiff
    struct.pack_into("<f", buf, 76, float(dmin))               # dMin
    struct.pack_into("<h", buf, 80, int(npx))                  # numPx (i2)
    struct.pack_into("<b", buf, 82, 1)                         # latGridType = 1 (Lambert)
    # reserved [83:88] = 0
    return bytes(buf)


def _build_harmonics(coefs_ml: np.ndarray, bw: int, z_rot: int, cmp_flg: int) -> bytes:
    """HarmonicsData: 8 fixed bytes + doubCnt*8 packed real doubles."""
    packed = _pack_harm(coefs_ml, bw, z_rot, cmp_flg)
    doub_cnt = packed.shape[0]
    fixed = bytearray(8)
    struct.pack_into("<h", fixed, 0, bw)
    struct.pack_into("<b", fixed, 2, z_rot)
    struct.pack_into("<b", fixed, 3, cmp_flg)
    struct.pack_into("<i", fixed, 4, doub_cnt)
    return bytes(fixed) + packed.tobytes()


def _formula_from_atoms(structure: CrystalStructure) -> str:
    """Estimate the chemical formula string (mp2sht.cpp: ordered set of symbols)."""
    seen: list[int] = []
    for at in structure.atoms:
        z = int(at.Z)
        if z not in seen:
            seen.append(z)
    seen.sort()
    out = ""
    for z in seen:
        if 1 <= z <= len(_AT_SYB):
            out += _AT_SYB[z - 1]
        else:
            out += f"Z{z}"
    return out


def write_sht(
    out_path: str,
    our_master_NH: torch.Tensor | np.ndarray,
    structure: CrystalStructure,
    *,
    our_master_SH: torch.Tensor | np.ndarray | None = None,
    bandwidth: int = _PRODUCTION_BANDWIDTH,
    energy_kV: float,
    npx: Optional[int] = None,
    dmin: float = 0.05,
    primary_tilt_deg: float = 70.0,
    secondary_tilt_deg: float = 0.0,
    sg_eff: Optional[int] = None,
    formula: Optional[str] = None,
    name: str = "",
    symbol: str = "",
    refs: str = "",
    note: str = "created with forward_sim sht_writer",
    doi: str = "https://doi.org/10.1016/j.ultramic.2019.112841",
    numsx: int = 501,
    totnum_el: int = 1_000_000_000,
    bethe_params: Tuple[float, float, float, float] = (4.0, 8.0, 50.0, 1.0),
    software_version: bytes = b"fwd_sim0",
    emsoft_version: bytes = b"5_0_0_0",
    device: torch.device | str | None = None,
) -> str:
    """Serialise our GPU master into an EMSphInx ``.sht`` spherical-harmonics file.

    Reconstructs the master on a Driscoll-Healy ``(2·bw, 2·bw)`` **full sphere**,
    forward-transforms it with the project's ``CSHT`` to the reader's
    ``coefs_ml[m, l]`` convention, symmetry-compresses per the effective space
    group, and writes the exact EMSphInx ``*sht`` binary
    (:mod:`backend.spherical_gpu.pipeline.sht_io` reads it back).

    Full sphere (NH + SH).  EMsoft's ``EMEBSDmasterSHT`` builds the harmonics from
    BOTH Lambert hemispheres (``transformer%analyze(finalmLPNH, finalmLPSH, alm)``):
    even-``l+m`` coefficients use ``NH + SH`` (equator-symmetric) and odd-``l+m`` use
    ``NH − SH`` (equator-antisymmetric).  When ``our_master_SH`` is given, the DH
    sphere's southern rows (``z < 0``) are sampled from it, so the transform encodes
    the genuine full sphere — **required** for a non-centrosymmetric crystal where
    ``NH ≠ SH`` (its ``NH − SH`` odd harmonics are non-zero).  When ``our_master_SH``
    is ``None`` the master is treated as equator-symmetric (``NH == SH``), which is
    exact for a centrosymmetric/cubic cell and is **bit-identical** to the previous
    NH-only serialisation (the southern DH rows sample the same NH array).

    Args:
        out_path: destination file path (overwritten if it exists).
        our_master_NH: northern-hemisphere Lambert master, shape
            ``(2·npx+1, 2·npx+1)`` — exactly what
            :func:`backend.forward_sim.dynamical.master_builder.build_master`
            returns.
        structure: the crystal (lattice + atoms + space group); written into the
            ``CrystalData`` block and used to derive the harmonic compression.
        our_master_SH: optional **southern**-hemisphere Lambert master, same shape
            ``(2·npx+1, 2·npx+1)`` as ``our_master_NH`` — the true ``mLPSH`` for a
            **non-centrosymmetric** crystal, produced by a second
            :func:`build_master` call with ``hemisphere="south"``.  When provided the
            spherical-harmonic transform uses the full sphere (NH + SH).  When
            ``None`` (default) the master is treated as equator-symmetric (NH == SH),
            the exact and bit-identical behaviour for a centrosymmetric/cubic cell.
        bandwidth: spherical-harmonic band-limit ``bw`` (the file's ``bw``; the
            stored coefficient block is ``(bw, bw)``).  Defaults to the EMSphInx
            production band-limit ``384`` (matches the real EMsoft ``.sht`` files
            and captures full Kikuchi-band sharpness — needs a high-res master,
            ``npx>=200``, to feed it); smaller is valid and faster but loses band
            sharpness for sharp masters.
        energy_kV: beam energy (keV) the master was computed at.
        npx: Lambert half-grid size; defaults to ``(side − 1) // 2`` inferred from
            ``our_master_NH``.  Used only for the cosmetic ``numPx`` field and the
            shape guard.
        dmin: resolution limit (nm) stored in the EMsoftED simulation block.
        primary_tilt_deg: sample primary tilt (sigma); EBSD default 70.
        secondary_tilt_deg: sample secondary tilt (omega); default 0.
        sg_eff: effective space group number (1..230) used for the harmonic
            compression flags and stored as ``sgEff``.  Defaults to
            ``structure.space_group`` (the right choice for the cubic SP0+SP1
            scope; for averaged/effective-symmetry masters pass it explicitly).
        formula: chemical-formula string; defaults to an estimate from the atom
            types (mp2sht convention).
        name, symbol, refs, note: crystal metadata strings (all optional).
        doi: file DOI string.
        numsx, totnum_el, bethe_params, software_version, emsoft_version:
            cosmetic simulation/header metadata stored in the file.
        device: torch device for the SHT (the CSHT/DLT transform + DH sampling).
            ``None`` (default) selects **CUDA if available, else CPU** — the
            transform is far faster on the GPU and is the wall this serialiser pays.
            An explicit ``"cuda"`` that is unavailable falls back to CPU with a
            warning.  The coefficients are computed in fp64 and are equal to ~1e-9
            CPU-vs-CUDA, so the written file is device-independent (bit-faithful
            header/packing/CRC).  Use :func:`warm_sht_cache` once before a batch of
            writes to amortise the O(L³) Wigner-table build.

    Returns:
        ``out_path`` (the written file path).

    Raises:
        ValueError: if ``bandwidth`` < 1, the master shape is not square, the
            inferred ``npx`` is inconsistent, ``our_master_SH`` (when given) does not
            match ``our_master_NH``'s shape, or ``sg_eff`` is out of ``[1, 230]``.
    """
    if int(bandwidth) < 1:
        raise ValueError(f"bandwidth must be >= 1, got {bandwidth}")
    bw = int(bandwidth)

    # LEVER 2: run the SHT on CUDA by default (None => cuda-if-available else CPU);
    # an explicit unavailable "cuda" falls back to CPU (fail-safe).
    dev = _resolve_sht_device(device)
    nh = torch.as_tensor(our_master_NH, dtype=torch.float32, device=dev)
    if nh.ndim != 2 or nh.shape[0] != nh.shape[1]:
        raise ValueError(
            f"our_master_NH must be square (2·npx+1)², got {tuple(nh.shape)}"
        )
    side = int(nh.shape[0])
    if side % 2 == 0:
        raise ValueError(
            f"our_master_NH side {side} must be odd (= 2·npx+1)"
        )
    inferred_npx = (side - 1) // 2
    if npx is None:
        npx = inferred_npx
    elif int(npx) != inferred_npx:
        raise ValueError(
            f"npx={npx} inconsistent with master side {side} (= 2·{inferred_npx}+1)"
        )
    npx = int(npx)

    sh = None
    if our_master_SH is not None:
        sh = torch.as_tensor(our_master_SH, dtype=torch.float32, device=dev)
        if sh.shape != nh.shape:
            raise ValueError(
                f"our_master_SH shape {tuple(sh.shape)} must match our_master_NH "
                f"{tuple(nh.shape)}"
            )

    sg = int(sg_eff) if sg_eff is not None else int(structure.space_group)
    if not 1 <= sg <= 230:
        raise ValueError(f"sg_eff {sg} out of range [1, 230]")
    z_rot = int(_SG_ROT[sg - 1])
    cmp_flg = int(_SG_CMP[sg - 1])

    # 1) Lambert master(s) -> Driscoll-Healy (2bw, 2bw) FULL-sphere signal.  When an
    #    SH master is given the southern DH rows sample it (full sphere, NH != SH);
    #    when None the master is equator-symmetric (NH == SH) — bit-identical to the
    #    legacy NH-only path.
    dh_signal = _lambert_nh_to_dh_grid(nh, npx, bw, dev, lambert_sh=sh)

    # 2) Forward SHT -> coefs_ml[m, l] (reader convention).
    coefs_ml = _forward_sht_coefs(dh_signal, bw, dev)
    coefs_np = coefs_ml.detach().cpu().numpy().astype(np.complex128)

    # 3) Assemble the binary blocks in the C++ File::write order.
    if formula is None:
        formula = _formula_from_atoms(structure)

    # The FileHeader-level notes string is kept empty (matching EMsoft's mp2sht);
    # the per-crystal ``note`` carries the human text. ``doi`` is the header DOI.
    header_b = _build_file_header(
        energy_kV, primary_tilt_deg, secondary_tilt_deg,
        doi.encode("utf-8"), b"", software_version,
    )

    mp_b = _build_master_pattern_data(1, sg, _SIMMETA_EMSOFTED)

    crystal_b = _build_crystal_data(
        structure,
        formula.encode("utf-8"), name.encode("utf-8"), symbol.encode("utf-8"),
        refs.encode("utf-8"), note.encode("utf-8"),
    )

    sim_b = _build_emsoft_ed(
        energy_kV, primary_tilt_deg, secondary_tilt_deg, dmin, npx, numsx,
        totnum_el, bethe_params, emsoft_version,
    )

    harm_b = _build_harmonics(coefs_np, bw, z_rot, cmp_flg)

    body = header_b + mp_b + crystal_b + sim_b + harm_b
    crc = _crc32c_emsoft(body, 0)
    full = body + struct.pack("<I", crc)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(full)
    return str(out)
