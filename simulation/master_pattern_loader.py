"""
Load Master Pattern H5 and SHT metadata without GUI dependencies.

Supports three file types:
- Master H5 (*_master*.h5): Full hemispheric projections via kikuchipy or h5py
- MC H5 (*_E*kV*.h5): Monte Carlo electron distribution via h5py
- SHT (*.sht): Binary spherical harmonics — metadata from filename only
"""

import logging
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MasterPatternData:
    """Container for loaded master pattern data and metadata."""

    file_path: str
    file_type: str  # "master", "h5", "sht"
    metadata: Dict[str, Any] = field(default_factory=dict)
    upper_hemisphere: Optional[np.ndarray] = None  # 2D float array
    lower_hemisphere: Optional[np.ndarray] = None  # 2D float array
    has_image: bool = False


def detect_file_type(file_path: str) -> str:
    """Determine file type from name/extension.

    Returns:
        "master", "sht", or "h5"
    """
    p = Path(file_path)
    if p.suffix.lower() == ".sht":
        return "sht"
    if "_master" in p.name.lower():
        return "master"
    return "h5"


def load_file(file_path: str) -> MasterPatternData:
    """Auto-detect type and load the appropriate data."""
    ft = detect_file_type(file_path)
    if ft == "master":
        return load_master_pattern(file_path)
    elif ft == "sht":
        return load_sht_metadata(file_path)
    else:
        return load_mc_h5(file_path)


# ---------------------------------------------------------------------------
# Master Pattern H5
# ---------------------------------------------------------------------------

def load_master_pattern(file_path: str) -> MasterPatternData:
    """Load Master Pattern H5 via kikuchipy, fallback to h5py.

    Master H5 structure (EMsoft):
        /EMData/EBSDmaster/masterSPNH  — (nE, ny, nx) float32 stereographic north
        /EMData/EBSDmaster/masterSPSH  — (nE, ny, nx) float32 stereographic south
        /CrystalData/Atomtypes         — atomic numbers
        /CrystalData/SpaceGroupNumber  — ITC space group
        /CrystalData/LatticeParameters — [a, b, c, alpha, beta, gamma]
        /NMLparameters/EBSDMasterNameList/npx — half-grid size
    """
    data = MasterPatternData(file_path=file_path, file_type="master")

    # Try kikuchipy first
    try:
        data = _load_via_kikuchipy(file_path, data)
        if data.has_image:
            return data
    except Exception as e:
        logger.debug(f"kikuchipy load failed, trying h5py: {e}")

    # Fallback: h5py direct
    try:
        data = _load_master_via_h5py(file_path, data)
    except Exception as e:
        logger.error(f"Failed to load master pattern {file_path}: {e}")
        _extract_metadata_from_filename(file_path, data)

    return data


def _load_via_kikuchipy(file_path: str, data: MasterPatternData) -> MasterPatternData:
    """Load master pattern using kikuchipy."""
    import kikuchipy as kp

    try:
        mp = kp.load(file_path, signal_type="EBSDMasterPattern")
    except Exception as e:
        logger.debug("Loading as EBSDMasterPattern failed, trying generic: %s", e)
        mp = kp.load(file_path)

    # Extract image — take highest energy slice
    arr = mp.data
    if arr.ndim == 3:
        # (nE, ny, nx) — take last energy (highest keV)
        data.upper_hemisphere = arr[-1].astype(np.float32)
    elif arr.ndim == 2:
        data.upper_hemisphere = arr.astype(np.float32)
    data.has_image = data.upper_hemisphere is not None

    # Try loading lower hemisphere
    try:
        mp_lower = kp.load(file_path, signal_type="EBSDMasterPattern", hemisphere="lower")
        arr_l = mp_lower.data
        if arr_l.ndim == 3:
            data.lower_hemisphere = arr_l[-1].astype(np.float32)
        elif arr_l.ndim == 2:
            data.lower_hemisphere = arr_l.astype(np.float32)
    except Exception as e:
        logger.debug("Lower hemisphere not available: %s", e)

    # Extract metadata from kikuchipy object
    try:
        phase = mp.phase
        if phase is not None:
            data.metadata["phase_name"] = str(phase.name) if hasattr(phase, "name") else ""
            if hasattr(phase, "space_group"):
                sg = phase.space_group
                if sg is not None:
                    data.metadata["space_group"] = str(sg.short_name) if hasattr(sg, "short_name") else str(sg)
                    data.metadata["space_group_number"] = int(sg.number) if hasattr(sg, "number") else 0
            if hasattr(phase, "point_group"):
                pg = phase.point_group
                if pg is not None:
                    data.metadata["point_group"] = str(pg.name) if hasattr(pg, "name") else str(pg)
    except Exception as e:
        logger.debug(f"Could not extract phase metadata: {e}")

    _extract_metadata_from_filename(file_path, data)

    if data.upper_hemisphere is not None:
        data.metadata["image_shape"] = f"{data.upper_hemisphere.shape[0]}x{data.upper_hemisphere.shape[1]}"

    return data


def _load_master_via_h5py(file_path: str, data: MasterPatternData) -> MasterPatternData:
    """Load master pattern directly via h5py."""
    import h5py

    with h5py.File(file_path, "r") as f:
        # Read stereographic projections
        for key, attr in [
            ("/EMData/EBSDmaster/masterSPNH", "upper_hemisphere"),
            ("/EMData/EBSDmaster/masterSPSH", "lower_hemisphere"),
        ]:
            if key in f:
                arr = f[key][()]
                if arr.ndim == 3:
                    arr = arr[-1]  # highest energy
                setattr(data, attr, arr.astype(np.float32))

        data.has_image = data.upper_hemisphere is not None

        # Crystal metadata
        if "/CrystalData/Atomtypes" in f:
            atomtypes = f["/CrystalData/Atomtypes"][()]
            data.metadata["atom_types"] = [int(z) for z in atomtypes]

        if "/CrystalData/SpaceGroupNumber" in f:
            data.metadata["space_group_number"] = int(f["/CrystalData/SpaceGroupNumber"][()])

        if "/CrystalData/LatticeParameters" in f:
            lp = f["/CrystalData/LatticeParameters"][()]
            data.metadata["lattice_params"] = [float(x) for x in lp]

        # Simulation params
        nml_base = "/NMLparameters/EBSDMasterNameList"
        if f"{nml_base}/npx" in f:
            data.metadata["npx"] = int(f[f"{nml_base}/npx"][()])
        if f"{nml_base}/dmin" in f:
            data.metadata["dmin"] = float(f[f"{nml_base}/dmin"][()])

        # Energy info
        if "/EMData/EBSDmaster/EkeVs" in f:
            energies = f["/EMData/EBSDmaster/EkeVs"][()]
            data.metadata["energies_keV"] = [float(e) for e in energies]

    _extract_metadata_from_filename(file_path, data)

    if data.upper_hemisphere is not None:
        data.metadata["image_shape"] = f"{data.upper_hemisphere.shape[0]}x{data.upper_hemisphere.shape[1]}"

    return data


# ---------------------------------------------------------------------------
# SHT files (binary, NOT HDF5)
# ---------------------------------------------------------------------------

def load_sht_metadata(file_path: str) -> MasterPatternData:
    """Extract metadata from SHT file.

    SHT is a custom binary format (magic: *sht), NOT HDF5.
    Metadata is extracted from the filename convention:
        {formula} ({phasename}) [{pearsonsymbol}] {{kVvoltage [tiltdeg]}}.sht
    Example: "Al (Al) [cF4] {20kV}.sht"
    """
    data = MasterPatternData(file_path=file_path, file_type="sht")
    data.has_image = False

    name = Path(file_path).stem

    # Parse: formula (phase) [pearson] {kV}
    m = re.match(
        r'^(?P<formula>[^(\[{]+?)\s+'
        r'(?:\((?P<phase>[^)]+)\)\s*)?'
        r'(?:\[(?P<pearson>[^\]]+)\]\s*)?'
        r'(?:\{(?P<kv>\d+)kV(?:\s+(?P<tilt>\d+)deg)?\})?',
        name,
    )
    if m:
        data.metadata["formula"] = m.group("formula").strip()
        if m.group("phase"):
            data.metadata["phase_name"] = m.group("phase").strip()
        if m.group("pearson"):
            data.metadata["pearson_symbol"] = m.group("pearson").strip()
        if m.group("kv"):
            data.metadata["kV"] = int(m.group("kv"))
        if m.group("tilt"):
            data.metadata["tilt_deg"] = int(m.group("tilt"))
    else:
        data.metadata["formula"] = name

    # File size
    try:
        data.metadata["file_size_bytes"] = Path(file_path).stat().st_size
    except OSError:
        pass

    data.metadata["note"] = "SHT files contain spherical harmonic coefficients (no 2D preview available)"
    return data


# ---------------------------------------------------------------------------
# Monte Carlo H5
# ---------------------------------------------------------------------------

def load_mc_h5(file_path: str) -> MasterPatternData:
    """Load Monte Carlo H5 file.

    MC H5 structure (EMsoft):
        /EMData/MCOpenCL/accum_e  — (nE, ny, nx) int32 accumulated electrons
        /CrystalData/...          — same as master
    """
    data = MasterPatternData(file_path=file_path, file_type="h5")

    try:
        import h5py

        with h5py.File(file_path, "r") as f:
            # Read accumulated electron distribution
            for key in ["/EMData/MCOpenCL/accum_e", "/EMData/MCOpenCL/accumSP"]:
                if key in f:
                    arr = f[key][()]
                    if arr.ndim == 3:
                        # Sum over energies for overview
                        arr = arr.sum(axis=0)
                    data.upper_hemisphere = arr.astype(np.float32)
                    data.has_image = True
                    break

            # Crystal metadata
            if "/CrystalData/Atomtypes" in f:
                atomtypes = f["/CrystalData/Atomtypes"][()]
                data.metadata["atom_types"] = [int(z) for z in atomtypes]

            if "/CrystalData/SpaceGroupNumber" in f:
                data.metadata["space_group_number"] = int(f["/CrystalData/SpaceGroupNumber"][()])

            # MC params
            nml_base = "/NMLparameters/MCCLNameList"
            if f"{nml_base}/numsx" in f:
                data.metadata["numsx"] = int(f[f"{nml_base}/numsx"][()])
            if f"{nml_base}/totnum_el" in f:
                data.metadata["total_electrons"] = int(f[f"{nml_base}/totnum_el"][()])

    except Exception as e:
        logger.error(f"Failed to load MC H5 {file_path}: {e}")

    _extract_metadata_from_filename(file_path, data)

    if data.upper_hemisphere is not None:
        data.metadata["image_shape"] = f"{data.upper_hemisphere.shape[0]}x{data.upper_hemisphere.shape[1]}"

    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_metadata_from_filename(file_path: str, data: MasterPatternData):
    """Extract kV and material info from standardized filename."""
    name = Path(file_path).stem

    # kV from filename: *_E20kV* or {20kV}
    kv_match = re.search(r'_E(\d+)kV|[{](\d+)kV', name)
    if kv_match and "kV" not in data.metadata:
        data.metadata["kV"] = int(kv_match.group(1) or kv_match.group(2))

    # npx from filename: *_npx500*
    npx_match = re.search(r'_npx(\d+)', name)
    if npx_match and "npx" not in data.metadata:
        data.metadata["npx"] = int(npx_match.group(1))

    # Material from parent folder
    parent = Path(file_path).parent.name
    if parent and parent not in (".", "EBSD_H5_Cache", "EBSD_SHT_Database"):
        data.metadata.setdefault("material", parent)

    data.metadata.setdefault("filename", Path(file_path).name)
    data.metadata.setdefault("file_type_label", {
        "master": "Master Pattern H5",
        "h5": "Monte Carlo H5",
        "sht": "SHT (Spherical Harmonics)",
    }.get(data.file_type, data.file_type.upper()))
