"""SP3 (part 1) — serialise our GPU master into an EMsoft-format ``master.h5``.

The output is a faithful EMsoft ``EBSDmaster`` HDF5 file: a **drop-in** for
:func:`kikuchipy.load` (``projection="lambert"`` or ``"stereographic"``) and the
kikuchipy dictionary-indexing pipeline.  We do **not** re-run EMsoft — we take the
northern-hemisphere Lambert master :func:`backend.forward_sim.dynamical.master_builder.build_master`
produced and wrap it in the exact group/dataset layout kikuchipy's reader expects.

What kikuchipy's reader actually requires
-----------------------------------------
Studied directly from ``kikuchipy.io.plugins._emsoft_master_pattern`` (v0.11.3):

* ``EMheader/EBSDmaster/ProgramName`` MUST decode to ``"EMEBSDmaster.f90"`` —
  ``check_file_format`` raises ``IOError`` otherwise (this is the format gate).
* ``EMData/EBSDmaster/{EkeVs, numset, mLPNH, mLPSH, masterSPNH, masterSPSH}`` —
  the energy axis + the four master arrays.  For the **Lambert** projection the
  reader sums ``mLP*`` over the leading ``numset`` (asymmetric-position) axis, so
  ``mLPNH`` is shaped ``(numset, nE, 2·npx+1, 2·npx+1)``; the **stereographic**
  arrays ``masterSP*`` are ``(nE, 2·npx+1, 2·npx+1)`` (no ``numset`` axis).
* ``NMLparameters/EBSDMasterNameList/npx`` — half the master side length.
* ``NMLparameters/MCCLNameList/Ebinsize`` — energy-axis scale (keV/bin).
* ``CrystalData/*`` — parsed by ``_crystaldata2phase`` into an orix ``Phase``
  (``AtomData`` ``(5, N)``, ``Atomtypes`` ``(N,)``, ``LatticeParameters`` ``(6,)``,
  ``SpaceGroupNumber``, ``Natomtypes``).

Single-energy choice (documented)
---------------------------------
Our forward model builds **one** master per energy bin (``energy_idx``).  EMsoft's
file always carries the full MC energy axis (e.g. 11 bins, 10–20 kV).  We replicate
the same single-energy master into **every** energy bin of ``mLPNH/SH`` (and the
stereographic arrays).  Rationale: kikuchipy's reader, when asked for a single
``energy=`` (or a range), slices the energy axis and the master it returns is
identical regardless of which bin it lands in — so replication makes
``kp.load(..., energy=E)`` return our master for **any** requested energy, which is
the behaviour a downstream dictionary-indexing call wants.  (Asking for the full
energy range would stack identical copies; that is harmless and explicitly
documented.)  ``EkeVs`` and the MC carry-through come from the source MC ``.h5``.

Southern hemisphere
-------------------
For a **centrosymmetric / cubic** crystal the southern Lambert hemisphere equals
the northern one (verified on the real Ni oracle: ``mLPNH == mLPSH`` bit-for-bit).
We therefore copy NH→SH by default (``mirror_sh="copy"``), which is exact for the
cubic Ni/Al validation gate.  ``mirror_sh="grid"`` instead resamples the sphere's
southern hemisphere from the NH grid via the Lambert mapping (a true mirror for
non-centrosymmetric cells); the copy is the safe default for the SP0+SP1 cubic
scope.  For a genuinely non-centrosymmetric crystal (SP6), pass an **explicit**
``our_master_SH`` — a second :func:`build_master` call with ``hemisphere="south"``
— which is written verbatim as ``mLPSH`` (and its stereographic projection),
overriding ``mirror_sh``.

Stereographic arrays
--------------------
``masterSPNH/SPSH`` are the EMsoft **stereographic** projection of the same sphere.
We build them by, for each stereographic pixel ``(row, col)`` →
``X=(col−npx)/npx, Y=(row−npx)/npx``, inverting the stereographic projection to a
unit direction ``(2X, 2Y, 1−q)/(1+q)`` (``q=X²+Y²``) and sampling the Lambert NH
master at that direction (reusing the kikuchipy/EMsoft-validated
:func:`backend.dictionary_gpu.lambert.direction_to_lambert` + an
``align_corners=True`` bilinear ``grid_sample`` — the very sampler the dictionary
pipeline uses).  Pixels with ``q>1`` (outside the inscribed disc) are 0, matching
EMsoft.  Reproducing the real Ni ``masterSPNH`` from its own ``mLPNH`` this way
gives NCC 0.99995 (residual = bilinear interp vs EMsoft's exact recompute).

MC + header carry-through
-------------------------
``CrystalData``, ``EMData/MCOpenCL/*`` and the ``MCOpenCL`` NML/header are copied
verbatim from ``mc_source_h5`` when present (the master file the MC came from), so
the MC geometry travels with the master.  If a group is absent we synthesise a
minimal valid placeholder.  The ``EBSDmaster`` NML/header are always (re)written by
us so the format gate + ``npx``/``dmin``/``Ebinsize`` are correct for our master.
"""
from __future__ import annotations

import datetime as _dt
import getpass
import socket
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from ..crystal.xtal_io import CrystalStructure
from backend.dictionary_gpu.lambert import direction_to_lambert

# EMsoft program names — the Lambert/stereo reader gates on the EBSDmaster one.
_EBSD_MASTER_PROGRAM = b"EMEBSDmaster.f90"
_MC_PROGRAM = b"EMMCOpenCL.f90"
_EMSOFT_VERSION = b"5_0_forward_sim"

# Default EMsoft Bethe parameters [c1, c2, c3, sgdbdiff].
_DEFAULT_BETHE = (4.0, 8.0, 50.0, 1.0)


def _str_ds(group: h5py.Group, name: str, value) -> None:
    """Write a length-1 variable-length-bytes dataset (EMsoft string convention)."""
    if isinstance(value, str):
        value = value.encode()
    group.create_dataset(name, data=np.array([value], dtype=h5py.special_dtype(vlen=bytes)))


def _now_strings() -> tuple[bytes, bytes]:
    """(date, full-timestamp) byte strings in an EMsoft-ish format."""
    now = _dt.datetime.now()
    date = now.strftime("%b %d %Y").encode()
    full = now.strftime("%b %d %Y, %I:%M:%S.%f %p").encode()
    return date, full


def _stereographic_from_lambert(
    lambert_nh: torch.Tensor, npx: int, device: torch.device
) -> np.ndarray:
    """EMsoft stereographic NH projection of a Lambert NH master.

    For each stereographic pixel ``(row, col)`` with ``X=(col−npx)/npx``,
    ``Y=(row−npx)/npx`` and ``q=X²+Y²``, invert the stereographic projection to the
    upper-hemisphere unit direction ``(2X, 2Y, 1−q)/(1+q)`` and sample
    ``lambert_nh`` there (Lambert mapping + ``align_corners=True`` bilinear
    ``grid_sample``).  Pixels with ``q>1`` are 0 (outside the disc, EMsoft parity).

    Args:
        lambert_nh: ``(2·npx+1, 2·npx+1)`` float master on the Lambert square.
        npx: Lambert half-grid size.
        device: torch device for the sampling.

    Returns:
        ``(2·npx+1, 2·npx+1)`` float32 numpy array — the stereographic NH master.
    """
    m = 2 * npx + 1
    idx = torch.arange(m, dtype=torch.float64, device=device)
    cols, rows = torch.meshgrid(idx, idx, indexing="xy")  # cols varies along axis-1
    x = (cols - npx) / npx
    y = (rows - npx) / npx
    q = x * x + y * y
    inside = q <= 1.0
    denom = 1.0 + q
    dx = 2.0 * x / denom
    dy = 2.0 * y / denom
    dz = (1.0 - q) / denom
    dirs = torch.stack([dx.reshape(-1), dy.reshape(-1), dz.reshape(-1)], dim=-1)
    xy = direction_to_lambert(dirs)  # (P, 2) in [-1, 1]
    grid = xy.reshape(1, m, m, 2).to(torch.float32)
    img = lambert_nh.to(device).reshape(1, 1, m, m).to(torch.float32)
    samp = F.grid_sample(
        img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    ).reshape(m, m)
    samp = torch.where(inside, samp, torch.zeros_like(samp))
    return samp.detach().cpu().numpy().astype(np.float32)


def _copy_group(src: h5py.Group, dst_parent: h5py.Group, name: str) -> bool:
    """Copy ``src[name]`` into ``dst_parent`` if it exists. Returns True if copied."""
    if name in src:
        src.copy(src[name], dst_parent, name=name)
        return True
    return False


def _write_crystal_data(f: h5py.File, structure: CrystalStructure) -> None:
    """Write a minimal valid ``CrystalData`` group from our structure.

    ``AtomData`` is ``(5, N)`` rows ``[x, y, z, occ, B]`` (EMsoft convention),
    matching what :func:`backend.forward_sim.crystal.xtal_io.read_crystal_structure`
    reads back and what kikuchipy's ``_crystaldata2phase`` consumes.
    """
    cd = f.create_group("CrystalData")
    atoms = structure.atoms
    n = len(atoms)
    atom_data = np.zeros((5, n), dtype=np.float32)
    atomtypes = np.zeros(n, dtype=np.int32)
    for i, at in enumerate(atoms):
        atom_data[0, i], atom_data[1, i], atom_data[2, i] = at.xyz
        atom_data[3, i] = at.occ
        atom_data[4, i] = at.B
        atomtypes[i] = int(at.Z)
    cd.create_dataset("AtomData", data=atom_data)
    cd.create_dataset("Atomtypes", data=atomtypes)
    cd.create_dataset("Natomtypes", data=np.array([n], dtype=np.int32))
    cd.create_dataset(
        "LatticeParameters", data=np.asarray(structure.lattice, dtype=np.float64)
    )
    cd.create_dataset(
        "SpaceGroupNumber", data=np.array([int(structure.space_group)], dtype=np.int32)
    )
    cd.create_dataset("SpaceGroupSetting", data=np.array([1], dtype=np.int32))
    cd.create_dataset(
        "CrystalSystem",
        data=np.array([int(structure.crystal_system)], dtype=np.int32),
    )
    date, _ = _now_strings()
    _str_ds(cd, "CreationDate", date)
    _str_ds(cd, "CreationTime", _now_strings()[1])
    try:
        creator = getpass.getuser()
    except Exception:
        creator = "forward_sim"
    _str_ds(cd, "Creator", creator)
    _str_ds(cd, "ProgramName", "EMmkxtal.f90")
    _str_ds(cd, "Source", "forward_sim")


def write_master_h5(
    out_path: str,
    our_master_NH: torch.Tensor | np.ndarray,
    structure: CrystalStructure,
    mc_source_h5: str | None = None,
    *,
    npx: int,
    dmin: float,
    energy_kV: float,
    n_energy: int | None = None,
    energy_idx: int | None = None,
    EkeVs: np.ndarray | None = None,
    Ebinsize: float = 1.0,
    xtalname: str = "forward_sim.xtal",
    bethe_params: tuple = _DEFAULT_BETHE,
    mirror_sh: str = "copy",
    our_master_SH: torch.Tensor | np.ndarray | None = None,
    device: torch.device | str | None = None,
) -> str:
    """Serialise our GPU master into a kikuchipy-loadable EMsoft ``EBSDmaster`` .h5.

    Args:
        out_path: destination file path (overwritten if it exists).
        our_master_NH: northern-hemisphere Lambert master, shape
            ``(2·npx+1, 2·npx+1)`` — exactly what
            :func:`backend.forward_sim.dynamical.master_builder.build_master`
            returns (a single energy bin).
        structure: the crystal (lattice + atoms + space group) — written into
            ``CrystalData`` and used by kikuchipy to build the orix ``Phase``.
        mc_source_h5: optional path to the EMsoft master/MC ``.h5`` the Monte-Carlo
            came from.  ``CrystalData`` (if not overridden), ``EMData/MCOpenCL/*``,
            ``EkeVs`` and the ``MCOpenCL`` NML/header are carried through from it.
            If ``None`` (or a group is missing), minimal valid placeholders are
            synthesised.
        npx: Lambert half-grid size (the master side length is ``2·npx+1``).
        dmin: resolution limit (nm) — written into the EBSDmaster NML.
        energy_kV: the beam energy (keV) our single master was computed at.
        n_energy: number of energy bins in the output file.  Defaults to the source
            MC's ``EkeVs`` length, else the length of ``EkeVs``, else 1.
        energy_idx: which energy bin our master corresponds to (for documentation /
            ``lastEnergy``).  Our master is replicated into **all** bins regardless
            (see module docstring), so this only labels ``lastEnergy``.  Defaults to
            the bin of ``EkeVs`` closest to ``energy_kV``.
        EkeVs: explicit energy axis (keV) of length ``n_energy``.  Defaults to the
            source MC's ``EkeVs``; else a 1-keV-spaced axis ending at ``energy_kV``.
        Ebinsize: energy-axis scale (keV/bin) written into the MCCL NML (kikuchipy
            reads this as the energy axis scale).
        xtalname: crystal file name string stored in the file (cosmetic).
        bethe_params: ``[c1, c2, c3, sgdbdiff]`` Bethe cutoffs (stored, cosmetic).
        mirror_sh: ``"copy"`` (default) sets ``mLPSH = mLPNH`` (exact for
            centrosymmetric/cubic cells); ``"grid"`` resamples the southern
            hemisphere from the NH grid via the Lambert mapping.  **Ignored when
            ``our_master_SH`` is given** (an explicit SH array is always used
            verbatim).
        our_master_SH: optional explicit **southern**-hemisphere Lambert master,
            same shape ``(2·npx+1, 2·npx+1)`` as ``our_master_NH`` — the true
            ``mLPSH`` for a **non-centrosymmetric** crystal, produced by a second
            :func:`build_master` call with ``hemisphere="south"``.  When provided it
            is written as ``mLPSH`` directly (and its stereographic projection as
            ``masterSPSH``), overriding ``mirror_sh``.  When ``None`` (default), the
            ``mirror_sh`` policy fills the SH from the NH (correct for cubic).
        device: torch device for the stereographic resampling; defaults to CPU.

    Returns:
        ``out_path`` (the written file path).

    Raises:
        ValueError: if ``our_master_NH`` is not square ``(2·npx+1)²``, ``npx<1``,
            or ``mirror_sh`` is invalid.
    """
    if npx < 1:
        raise ValueError(f"npx must be >= 1, got {npx}")
    if mirror_sh not in ("copy", "grid"):
        raise ValueError(f"mirror_sh must be 'copy' or 'grid', got {mirror_sh!r}")

    dev = torch.device(device) if device is not None else torch.device("cpu")
    m = 2 * npx + 1

    nh = torch.as_tensor(our_master_NH, dtype=torch.float32, device=dev)
    if nh.shape != (m, m):
        raise ValueError(
            f"our_master_NH must be ({m}, {m}) for npx={npx}, got {tuple(nh.shape)}"
        )

    sh_explicit = None
    if our_master_SH is not None:
        sh_explicit = torch.as_tensor(our_master_SH, dtype=torch.float32, device=dev)
        if sh_explicit.shape != (m, m):
            raise ValueError(
                f"our_master_SH must be ({m}, {m}) for npx={npx}, got "
                f"{tuple(sh_explicit.shape)}"
            )

    # ---- Energy axis -------------------------------------------------------
    src = h5py.File(mc_source_h5, "r") if mc_source_h5 else None
    try:
        if EkeVs is None and src is not None and "EMData/EBSDmaster/EkeVs" in src:
            EkeVs = np.asarray(src["EMData/EBSDmaster/EkeVs"][()], dtype=np.float32).ravel()
        if EkeVs is not None:
            EkeVs = np.asarray(EkeVs, dtype=np.float32).ravel()
            nE = len(EkeVs)
        else:
            nE = int(n_energy) if n_energy else 1
            # 1-keV-spaced axis ending at energy_kV.
            EkeVs = np.array(
                [energy_kV - (nE - 1 - i) * Ebinsize for i in range(nE)],
                dtype=np.float32,
            )
        if n_energy is not None and int(n_energy) != nE:
            # Resize the axis to the requested length (truncate/extend at the high end).
            nE = int(n_energy)
            EkeVs = np.array(
                [energy_kV - (nE - 1 - i) * Ebinsize for i in range(nE)],
                dtype=np.float32,
            )
        if energy_idx is None:
            energy_idx = int(np.abs(EkeVs - energy_kV).argmin())

        # ---- Master arrays (replicate single-energy master into every bin) -
        nh_np = nh.detach().cpu().numpy().astype(np.float32)
        if sh_explicit is not None:
            # True SH array (non-centrosymmetric crystal): use it verbatim,
            # overriding the NH-mirror policy.
            sh_t = sh_explicit
            sh_np = sh_explicit.detach().cpu().numpy().astype(np.float32)
        elif mirror_sh == "copy":
            sh_t = None  # SH == NH; stereo SH reuses the NH stereo below.
            sh_np = nh_np.copy()
        else:  # "grid" — resample SH from the sphere's lower hemisphere
            sh_np = _stereographic_lower_from_lambert(nh, npx, dev)
            sh_t = torch.as_tensor(sh_np)

        # Lambert arrays carry the leading numset (asymmetric-position) axis = 1.
        mLPNH = np.broadcast_to(nh_np, (1, nE, m, m)).astype(np.float32).copy()
        mLPSH = np.broadcast_to(sh_np, (1, nE, m, m)).astype(np.float32).copy()

        # Stereographic arrays (no numset axis).
        spnh = _stereographic_from_lambert(nh, npx, dev)
        spsh = (
            spnh.copy()
            if sh_t is None
            else _stereographic_from_lambert(sh_t, npx, dev)
        )
        masterSPNH = np.broadcast_to(spnh, (nE, m, m)).astype(np.float32).copy()
        masterSPSH = np.broadcast_to(spsh, (nE, m, m)).astype(np.float32).copy()

        # ---- Write the file ------------------------------------------------
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out, "w") as f:
            # --- CrystalData (prefer source carry-through for fidelity) -----
            if src is not None and "CrystalData" in src:
                src.copy(src["CrystalData"], f, name="CrystalData")
            else:
                _write_crystal_data(f, structure)

            # --- EMData/EBSDmaster ------------------------------------------
            emdata = f.create_group("EMData")
            ebsd = emdata.create_group("EBSDmaster")
            ebsd.create_dataset("mLPNH", data=mLPNH)
            ebsd.create_dataset("mLPSH", data=mLPSH)
            ebsd.create_dataset("masterSPNH", data=masterSPNH)
            ebsd.create_dataset("masterSPSH", data=masterSPSH)
            ebsd.create_dataset("EkeVs", data=EkeVs.astype(np.float32))
            ebsd.create_dataset(
                "BetheParameters", data=np.asarray(bethe_params, dtype=np.float32)
            )
            ebsd.create_dataset("numEbins", data=np.array([nE], dtype=np.int32))
            ebsd.create_dataset("numset", data=np.array([1], dtype=np.int32))
            ebsd.create_dataset(
                "lastEnergy", data=np.array([int(energy_idx) + 1], dtype=np.int32)
            )
            ebsd.create_dataset("Z2percent", data=np.array([100.0], dtype=np.float32))
            _str_ds(ebsd, "xtalname", xtalname)

            # --- EMData/MCOpenCL (carry through, else synthesise) -----------
            if not (src is not None and _copy_group(src, emdata, "MCOpenCL")):
                _synthesise_mc(emdata, nE)

            # --- NMLparameters ----------------------------------------------
            nmlp = f.create_group("NMLparameters")
            _write_ebsd_master_nml(nmlp, npx, dmin, bethe_params, xtalname)
            _write_mccl_nml(nmlp, src, energy_kV, Ebinsize, xtalname, nE)
            bethe = nmlp.create_group("BetheList")
            bethe.create_dataset("c1", data=np.array([bethe_params[0]], dtype=np.float32))
            bethe.create_dataset("c2", data=np.array([bethe_params[1]], dtype=np.float32))
            bethe.create_dataset("c3", data=np.array([bethe_params[2]], dtype=np.float32))
            bethe.create_dataset(
                "sgdbdiff", data=np.array([bethe_params[3]], dtype=np.float32)
            )

            # --- NMLfiles (cosmetic text blobs) -----------------------------
            nmlf = f.create_group("NMLfiles")
            _str_array_ds(
                nmlf,
                "EBSDmasterNML",
                [
                    "&EBSDmastervars",
                    f" npx = {npx},",
                    f" dmin = {dmin},",
                    f" energyfile = '{Path(out_path).name}',",
                    "/",
                ],
            )
            _str_array_ds(nmlf, "MCOpenCLNML", ["&MCCLdata", "/"])

            # --- EMheader (the format gate lives here) ----------------------
            header = f.create_group("EMheader")
            _write_emheader(header, "EBSDmaster", _EBSD_MASTER_PROGRAM)
            _write_emheader(header, "MCOpenCL", _MC_PROGRAM)
    finally:
        if src is not None:
            src.close()

    return out_path


def _stereographic_lower_from_lambert(
    lambert_nh: torch.Tensor, npx: int, device: torch.device
) -> np.ndarray:
    """Southern Lambert hemisphere resampled from the NH grid (mirror via −z).

    Used only for ``mirror_sh='grid'``: the SH Lambert pixel for direction
    ``(x, y, z)`` is the NH master sampled at the mirror ``(x, y, −z)``.  For a
    centrosymmetric cell this equals NH; for non-centrosymmetric it is the true
    lower hemisphere of the (NH-only) computed master and is an approximation.
    """
    m = 2 * npx + 1
    idx = torch.arange(m, dtype=torch.float64, device=device)
    cols, rows = torch.meshgrid(idx, idx, indexing="xy")
    x = (cols - npx) / npx
    y = (rows - npx) / npx
    from backend.dictionary_gpu.lambert import lambert_to_direction

    xy = torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)
    dirs, _ = lambert_to_direction(xy)  # (P, 3), z >= 0
    dirs[:, 2] = -dirs[:, 2]  # mirror to lower hemisphere
    xy2 = direction_to_lambert(dirs)  # maps back onto the same NH square via |z|
    grid = xy2.reshape(1, m, m, 2).to(torch.float32)
    img = lambert_nh.to(device).reshape(1, 1, m, m).to(torch.float32)
    samp = F.grid_sample(
        img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    ).reshape(m, m)
    return samp.detach().cpu().numpy().astype(np.float32)


def _synthesise_mc(emdata: h5py.Group, nE: int) -> None:
    """Write a minimal valid ``EMData/MCOpenCL`` group (placeholder MC)."""
    mc = emdata.create_group("MCOpenCL")
    n = 501
    mc.create_dataset("accum_e", data=np.zeros((n, n, nE), dtype=np.int32))
    mc.create_dataset("accum_z", data=np.zeros((51, 51, 101, nE), dtype=np.int32))
    mc.create_dataset("accumSP", data=np.zeros((n, n, nE), dtype=np.float32))
    mc.create_dataset("numEbins", data=np.array([nE], dtype=np.int32))
    mc.create_dataset("numzbins", data=np.array([101], dtype=np.int32))
    mc.create_dataset("multiplier", data=np.array([1], dtype=np.int32))
    mc.create_dataset("totnum_el", data=np.array([0], dtype=np.int32))


def _write_ebsd_master_nml(
    nmlp: h5py.Group, npx: int, dmin: float, bethe_params, xtalname: str
) -> None:
    """Write ``NMLparameters/EBSDMasterNameList`` (kikuchipy reads ``npx`` here)."""
    g = nmlp.create_group("EBSDMasterNameList")
    g.create_dataset("npx", data=np.array([int(npx)], dtype=np.int32))
    g.create_dataset("dmin", data=np.array([float(dmin)], dtype=np.float32))
    g.create_dataset("Esel", data=np.array([-1], dtype=np.int32))
    g.create_dataset("combinesites", data=np.array([0], dtype=np.int32))
    g.create_dataset("doLegendre", data=np.array([0], dtype=np.int32))
    g.create_dataset("nthreads", data=np.array([1], dtype=np.int32))
    g.create_dataset("restart", data=np.array([0], dtype=np.int32))
    g.create_dataset("stdout", data=np.array([6], dtype=np.int32))
    g.create_dataset("uniform", data=np.array([0], dtype=np.int32))
    g.create_dataset("useEnergyWeighting", data=np.array([0], dtype=np.int32))
    _str_ds(g, "latgridtype", "Lambert")
    _str_ds(g, "energyfile", "forward_sim_master.h5")
    _str_ds(g, "BetheParametersFile", "BetheParameters.nml")
    _str_ds(g, "copyfromenergyfile", "undefined")


def _write_mccl_nml(
    nmlp: h5py.Group,
    src: h5py.File | None,
    energy_kV: float,
    Ebinsize: float,
    xtalname: str,
    nE: int,
) -> None:
    """Write ``NMLparameters/MCCLNameList`` (kikuchipy reads ``Ebinsize`` here).

    Carry the source MC NML through verbatim when available (so the MC geometry
    metadata travels with the master); otherwise synthesise a minimal valid one.
    """
    if src is not None and "NMLparameters/MCCLNameList" in src:
        src.copy(src["NMLparameters/MCCLNameList"], nmlp, name="MCCLNameList")
        return
    g = nmlp.create_group("MCCLNameList")
    g.create_dataset("Ebinsize", data=np.array([float(Ebinsize)], dtype=np.float64))
    g.create_dataset("EkeV", data=np.array([float(energy_kV)], dtype=np.float64))
    g.create_dataset(
        "Ehistmin",
        data=np.array([float(energy_kV) - (nE - 1) * Ebinsize], dtype=np.float64),
    )
    g.create_dataset("depthmax", data=np.array([100.0], dtype=np.float64))
    g.create_dataset("depthstep", data=np.array([1.0], dtype=np.float64))
    g.create_dataset("numsx", data=np.array([501], dtype=np.int32))
    g.create_dataset("sig", data=np.array([70.0], dtype=np.float64))
    g.create_dataset("omega", data=np.array([0.0], dtype=np.float64))
    g.create_dataset("totnum_el", data=np.array([0], dtype=np.int32))
    g.create_dataset("multiplier", data=np.array([1], dtype=np.int32))
    _str_ds(g, "MCmode", "CSDA")
    _str_ds(g, "mode", "full")
    _str_ds(g, "xtalname", xtalname)
    _str_ds(g, "dataname", "forward_sim_mc.h5")


def _write_emheader(parent: h5py.Group, name: str, program: bytes) -> None:
    """Write an ``EMheader/<name>`` subgroup (ProgramName is the format gate)."""
    g = parent.create_group(name)
    date, full = _now_strings()
    _str_ds(g, "ProgramName", program)
    _str_ds(g, "Version", _EMSOFT_VERSION)
    _str_ds(g, "Date", date)
    _str_ds(g, "StartTime", full)
    _str_ds(g, "StopTime", full)
    try:
        host = socket.gethostname()
        user = getpass.getuser()
    except Exception:
        host, user = "forward_sim", "forward_sim"
    _str_ds(g, "HostName", host)
    _str_ds(g, "UserName", user)
    _str_ds(g, "UserEmail", "")
    _str_ds(g, "UserLocation", host)
    if name == "EBSDmaster":
        g.create_dataset("Duration", data=np.array([0.0], dtype=np.float32))


def _str_array_ds(group: h5py.Group, name: str, lines: list[str]) -> None:
    """Write a 1-D variable-length-bytes dataset (one entry per line)."""
    data = np.array([s.encode() for s in lines], dtype=h5py.special_dtype(vlen=bytes))
    group.create_dataset(name, data=data)
