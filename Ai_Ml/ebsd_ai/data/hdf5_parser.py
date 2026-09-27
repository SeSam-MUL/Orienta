"""Parsers for HDF5-based EBSD file formats.

Supports three HDF5 sub-formats:

H5OINA (Oxford Instruments)
----------------------------
- Top-level numbered groups (``1/``, ``2/``, …) with ``EBSD/Header`` and
  ``EBSD/Data`` sub-groups.
- Euler angles in **radians** (Bunge ZXZ).
- Phase IDs: 0 = not indexed, 1…N = phases.
- Quality metrics: Band Contrast (uint8 0-255), MAD (degrees).
- Patterns in ``Data/Processed Patterns`` (4D: rows, cols, H, W).

kikuchipy
---------
- ``Scan 1/EBSD/`` (or ``Scan N``) with ``Header`` and ``Data`` sub-groups.
- Primarily a pattern-storage format.  Indexed orientations may live in a
  ``crystal_map`` sub-group as quaternions (w, x, y, z).
- Patterns in ``Data/patterns`` (4D: rows, cols, H, W).

EMsoft
------
- ``EMData/EBSD/`` for dictionary-indexing results, or
  ``EMData/EBSDmaster/`` for master patterns.
- Euler angles in **radians** (Bunge ZXZ).
- Quality metrics: CI, IQ, AvDotProductMap.
- Patterns in ``EMData/EBSD/EBSDPatterns`` (3D: N, H, W).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ebsd_ai.config import mad_to_confidence, sanitize_array
from ebsd_ai.data.scan_import import ScanData, ScanFormat

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_hdf5(filepath: str | Path) -> ScanData:
    """Parse an HDF5-based EBSD file into a :class:`ScanData`.

    Automatically detects the sub-format (H5OINA, kikuchipy, EMsoft)
    and delegates to the appropriate internal parser.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.h5``, ``.hdf5``, or ``.h5oina`` file.

    Returns
    -------
    ScanData
        Parsed scan data.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ImportError
        If ``h5py`` is not installed.
    ValueError
        If the file cannot be parsed or format is not recognised.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    try:
        import h5py  # noqa: F811
    except ImportError as exc:
        raise ImportError(
            "h5py is required for HDF5 file parsing. "
            "Install it with: pip install h5py"
        ) from exc

    with h5py.File(path, "r") as f:
        fmt = _detect_subformat(f)
        if fmt == ScanFormat.H5OINA:
            return _parse_h5oina(f, path)
        if fmt == ScanFormat.KIKUCHIPY:
            return _parse_kikuchipy(f, path)
        if fmt == ScanFormat.EMSOFT:
            return _parse_emsoft(f, path)

    raise ValueError(
        f"Cannot determine HDF5 sub-format for: {path}. "
        "Expected H5OINA, kikuchipy, or EMsoft structure."
    )


def parse_h5oina(filepath: str | Path) -> ScanData:
    """Parse an Oxford Instruments H5OINA file.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.h5oina`` file.

    Returns
    -------
    ScanData
        Parsed scan data.
    """
    return _parse_hdf5_with_format(filepath, ScanFormat.H5OINA)


def parse_kikuchipy(filepath: str | Path) -> ScanData:
    """Parse a kikuchipy HDF5 file.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.h5`` file.

    Returns
    -------
    ScanData
        Parsed scan data.
    """
    return _parse_hdf5_with_format(filepath, ScanFormat.KIKUCHIPY)


def parse_emsoft(filepath: str | Path) -> ScanData:
    """Parse an EMsoft HDF5 file.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.h5`` file.

    Returns
    -------
    ScanData
        Parsed scan data.
    """
    return _parse_hdf5_with_format(filepath, ScanFormat.EMSOFT)


# ---------------------------------------------------------------------------
# Internal: format routing
# ---------------------------------------------------------------------------


def _parse_hdf5_with_format(
    filepath: str | Path, expected: ScanFormat
) -> ScanData:
    """Open HDF5 file and delegate to format-specific parser."""
    import h5py

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    with h5py.File(path, "r") as f:
        if expected == ScanFormat.H5OINA:
            return _parse_h5oina(f, path)
        if expected == ScanFormat.KIKUCHIPY:
            return _parse_kikuchipy(f, path)
        if expected == ScanFormat.EMSOFT:
            return _parse_emsoft(f, path)

    raise ValueError(f"Unsupported HDF5 format: {expected}")  # pragma: no cover


def _detect_subformat(f: Any) -> ScanFormat:
    """Detect HDF5 sub-format from group structure.

    Detection order matters — check most-specific signatures first:

    1. **EMsoft**: ``EMData`` or ``NMLparameters`` at top level.
    2. **kikuchipy**: ``Scan N`` groups at top level.
    3. **H5OINA**: numbered top-level groups with ``EBSD`` sub-group.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle.

    Returns
    -------
    ScanFormat
        Detected sub-format.
    """
    import h5py

    # EMsoft: has EMData or NMLparameters at top level
    if "EMData" in f or "NMLparameters" in f:
        return ScanFormat.EMSOFT

    # kikuchipy: has "Scan N" groups at top level
    if any(k.startswith("Scan ") for k in f.keys()):
        return ScanFormat.KIKUCHIPY

    # H5OINA: numbered top-level groups with EBSD sub-group
    for key in f.keys():
        grp = f[key]
        if isinstance(grp, h5py.Group) and "EBSD" in grp:
            return ScanFormat.H5OINA

    return ScanFormat.UNKNOWN


# ---------------------------------------------------------------------------
# H5OINA parser
# ---------------------------------------------------------------------------


def _parse_h5oina(f: Any, path: Path) -> ScanData:
    """Parse Oxford Instruments H5OINA structure."""
    import h5py

    # Find the first measurement group with EBSD data
    ebsd_grp = None
    for key in f.keys():
        grp = f[key]
        if isinstance(grp, h5py.Group) and "EBSD" in grp:
            ebsd_grp = grp["EBSD"]
            break

    if ebsd_grp is None:
        raise ValueError(f"No EBSD group found in H5OINA file: {path}")

    meta: dict[str, object] = {}
    header = ebsd_grp.get("Header")
    data = ebsd_grp.get("Data")

    if data is None:
        raise ValueError(f"No EBSD/Data group in H5OINA file: {path}")

    # --- Header metadata ---
    xcells = _read_scalar(header, "X Cells", int) if header else None
    ycells = _read_scalar(header, "Y Cells", int) if header else None
    step_x = _read_scalar(header, "X Step", float) if header else None
    step_y = _read_scalar(header, "Y Step", float) if header else None

    if xcells is not None:
        meta["xcells"] = xcells
    if ycells is not None:
        meta["ycells"] = ycells

    # --- Phase definitions ---
    phase_names: list[str] = []
    if header is not None and "Phases" in header:
        phases_grp = header["Phases"]
        for phase_key in sorted(phases_grp.keys(), key=_int_sort_key):
            phase_grp = phases_grp[phase_key]
            name = _read_string(phase_grp, "Name")
            if not name:
                name = _read_string(phase_grp, "Formula")
            if not name:
                name = f"Phase_{phase_key}"
            phase_names.append(name)

    all_phase_names = ["Not indexed"] + phase_names

    # --- Data arrays ---
    # Euler angles (radians, Bunge ZXZ)
    euler = _read_dataset(data, "Euler")
    if euler is not None and euler.ndim == 2 and euler.shape[1] == 3:
        euler_rad = sanitize_array(
            euler.astype(np.float64), name="euler_angles (H5OINA)"
        )
    elif euler is not None and euler.ndim == 1:
        # Some files store flat, reshape to (N, 3)
        euler_rad = sanitize_array(
            euler.reshape(-1, 3).astype(np.float64),
            name="euler_angles (H5OINA)",
        )
    else:
        euler_rad = np.zeros((0, 3), dtype=np.float64)

    n = euler_rad.shape[0] if euler_rad.size > 0 else 0

    # Phase IDs
    phase_raw = _read_dataset(data, "Phase")
    if phase_raw is not None:
        phase_ids = phase_raw.flatten().astype(np.int32)
        if n == 0:
            n = len(phase_ids)
    else:
        phase_ids = np.zeros(n, dtype=np.int32)

    phase_ids = np.clip(phase_ids, 0, len(all_phase_names) - 1)

    # Fix empty euler if we got n from phase_ids
    if euler_rad.size == 0 and n > 0:
        euler_rad = np.zeros((n, 3), dtype=np.float64)

    # Confidence: prefer MAD → exp(-MAD), fall back to Band Contrast / 255
    mad = _read_dataset(data, "Mean Angular Deviation")
    bc = _read_dataset(data, "Band Contrast")
    if mad is not None:
        mad_flat = sanitize_array(
            mad.flatten().astype(np.float64), name="MAD (H5OINA)"
        )
        confidence = mad_to_confidence(mad_flat).astype(np.float32)
        meta["mad"] = mad_flat
    elif bc is not None:
        bc_flat = sanitize_array(
            bc.flatten().astype(np.float64), name="Band Contrast (H5OINA)"
        )
        confidence = (bc_flat / 255.0).astype(np.float32)
        meta["band_contrast"] = bc_flat
    else:
        confidence = np.zeros(n, dtype=np.float32)

    # Patterns (optional)
    patterns = _read_dataset(data, "Processed Patterns")
    if patterns is not None:
        if patterns.ndim == 4:
            # (rows, cols, H, W) → (N, H, W)
            nrows_p, ncols_p, ph, pw = patterns.shape
            patterns = patterns.reshape(nrows_p * ncols_p, ph, pw)
        patterns = patterns.astype(np.uint8)

    # Grid shape
    scan_shape: tuple[int, int] | None = None
    if isinstance(ycells, int) and ycells > 0:
        if isinstance(xcells, int) and xcells > 0:
            scan_shape = (ycells, xcells)
    if scan_shape is None and n > 0:
        if isinstance(xcells, int) and xcells > 0:
            computed = n // xcells
            if computed > 0:
                scan_shape = (computed, xcells)

    step_sizes: tuple[float, float] | None = None
    if step_x is not None and step_y is not None:
        step_sizes = (float(step_x), float(step_y))

    return ScanData(
        patterns=patterns,
        phase_ids=phase_ids,
        phase_names=all_phase_names,
        euler_angles=euler_rad,
        confidence_scores=confidence,
        eds_data=None,
        scan_shape=scan_shape,
        step_sizes=step_sizes,
        source_format=ScanFormat.H5OINA,
        source_file=str(path),
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# kikuchipy parser
# ---------------------------------------------------------------------------


def _parse_kikuchipy(f: Any, path: Path) -> ScanData:
    """Parse kikuchipy HDF5 structure."""
    # Find scan group
    scan_grp = None
    for key in sorted(f.keys()):
        if key.startswith("Scan "):
            scan_grp = f[key]
            break

    if scan_grp is None:
        raise ValueError(f"No 'Scan N' group found in kikuchipy file: {path}")

    meta: dict[str, object] = {}
    ebsd_grp = scan_grp.get("EBSD")

    # --- Patterns ---
    patterns = None
    n = 0
    nrows = 0
    ncols = 0

    if ebsd_grp is not None:
        data_grp = ebsd_grp.get("Data")
        header_grp = ebsd_grp.get("Header")

        if data_grp is not None:
            raw_patterns = _read_dataset(data_grp, "patterns")
            if raw_patterns is not None:
                if raw_patterns.ndim == 4:
                    nrows, ncols = raw_patterns.shape[:2]
                    ph, pw = raw_patterns.shape[2:]
                    n = nrows * ncols
                    patterns = raw_patterns.reshape(n, ph, pw)
                elif raw_patterns.ndim == 3:
                    n = raw_patterns.shape[0]
                    patterns = raw_patterns
                if patterns is not None:
                    patterns = patterns.astype(np.uint8)

        # --- Header ---
        if header_grp is not None:
            # Grid dimensions
            n_patterns = _read_dataset(header_grp, "Number of patterns")
            if n_patterns is not None:
                arr = n_patterns.flatten()
                if len(arr) >= 2:
                    nrows = int(arr[0])
                    ncols = int(arr[1])
                    if n == 0:
                        n = nrows * ncols

            step_arr = _read_dataset(header_grp, "Step sizes")
            if step_arr is not None:
                step_flat = step_arr.flatten()
                if len(step_flat) >= 2:
                    meta["step_y"] = float(step_flat[0])
                    meta["step_x"] = float(step_flat[1])

            # Phase definitions
            phase_names: list[str] = []
            if "Phases" in header_grp:
                phases_grp = header_grp["Phases"]
                for pk in sorted(phases_grp.keys(), key=_int_sort_key):
                    p = phases_grp[pk]
                    name = _read_string(p, "name")
                    if not name:
                        name = f"Phase_{pk}"
                    phase_names.append(name)
    else:
        phase_names = []

    # --- Crystal map (indexed orientations) ---
    euler_rad = np.zeros((max(n, 0), 3), dtype=np.float64)
    phase_ids = np.zeros(max(n, 0), dtype=np.int32)
    confidence = np.zeros(max(n, 0), dtype=np.float32)

    cm_grp = scan_grp.get("crystal_map")
    if cm_grp is not None:
        # Rotations as quaternions (w, x, y, z) → convert to Euler
        rots = _read_dataset(cm_grp, "rotations")
        if rots is not None and rots.ndim == 2 and rots.shape[1] == 4:
            rots_clean = sanitize_array(
                rots.astype(np.float64), name="quaternions (kikuchipy)"
            )
            euler_rad = _quaternion_to_euler(rots_clean).astype(np.float64)
            if n == 0:
                n = rots.shape[0]

        pid = _read_dataset(cm_grp, "phase_id")
        if pid is not None:
            phase_ids = pid.flatten().astype(np.int32)
            if n == 0:
                n = len(phase_ids)

        # Quality properties
        props = cm_grp.get("properties")
        if props is not None:
            iq = _read_dataset(props, "iq")
            dp = _read_dataset(props, "dp")
            if dp is not None:
                # Dot product already in [0, 1]
                confidence = dp.flatten().astype(np.float32)
                meta["dot_product"] = dp.flatten().astype(np.float64)
            elif iq is not None:
                iq_flat = sanitize_array(
                    iq.flatten().astype(np.float64),
                    name="IQ (kikuchipy)",
                )
                iq_max = iq_flat.max() if iq_flat.size > 0 else 1.0
                if np.isfinite(iq_max) and iq_max > 0:
                    confidence = (iq_flat / iq_max).astype(np.float32)
                meta["image_quality"] = iq_flat

    all_phase_names = ["Not indexed"] + phase_names
    phase_ids = np.clip(phase_ids, 0, len(all_phase_names) - 1)

    # Ensure arrays are consistent size
    if euler_rad.shape[0] != n and n > 0:
        euler_rad = np.zeros((n, 3), dtype=np.float64)
    if len(phase_ids) != n and n > 0:
        phase_ids = np.zeros(n, dtype=np.int32)
    if len(confidence) != n and n > 0:
        confidence = np.zeros(n, dtype=np.float32)

    scan_shape: tuple[int, int] | None = None
    if nrows > 0 and ncols > 0:
        scan_shape = (nrows, ncols)

    step_sizes: tuple[float, float] | None = None
    sx = meta.get("step_x")
    sy = meta.get("step_y")
    if sx is not None and sy is not None:
        step_sizes = (float(sx), float(sy))  # type: ignore[arg-type]

    return ScanData(
        patterns=patterns,
        phase_ids=phase_ids,
        phase_names=all_phase_names,
        euler_angles=euler_rad,
        confidence_scores=confidence,
        eds_data=None,
        scan_shape=scan_shape,
        step_sizes=step_sizes,
        source_format=ScanFormat.KIKUCHIPY,
        source_file=str(path),
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# EMsoft parser
# ---------------------------------------------------------------------------


def _parse_emsoft(f: Any, path: Path) -> ScanData:
    """Parse EMsoft HDF5 structure."""
    meta: dict[str, object] = {}

    emdata = f.get("EMData")
    if emdata is None:
        raise ValueError(f"No EMData group found in EMsoft file: {path}")

    # Prefer EBSD indexing results over master patterns
    ebsd_grp = emdata.get("EBSD")
    master_grp = emdata.get("EBSDmaster")

    if ebsd_grp is None and master_grp is None:
        raise ValueError(
            f"No EMData/EBSD or EMData/EBSDmaster group in: {path}"
        )

    # --- NML parameters (step sizes, grid info) ---
    nml = f.get("NMLparameters")
    step_x: float | None = None
    step_y: float | None = None
    ipf_wd: int | None = None
    ipf_ht: int | None = None

    if nml is not None:
        idx_nml = nml.get("EBSDIndexingNameListType")
        if idx_nml is not None:
            step_x = _read_scalar(idx_nml, "step_x", float)
            step_y = _read_scalar(idx_nml, "step_y", float)
            ipf_wd = _read_scalar(idx_nml, "ipf_wd", int)
            ipf_ht = _read_scalar(idx_nml, "ipf_ht", int)

    # --- Crystal data ---
    phase_names: list[str] = []
    crystal = f.get("CrystalData")
    if crystal is not None:
        sg = _read_scalar(crystal, "SpaceGroupNumber", int)
        if sg is not None:
            meta["space_group"] = sg
        lp = _read_dataset(crystal, "LatticeParameters")
        if lp is not None:
            meta["lattice_parameters"] = lp.flatten().tolist()
        # Use space group as phase name fallback
        phase_name = _read_string(crystal, "PhaseName")
        if not phase_name:
            phase_name = f"SG_{sg}" if sg is not None else "Phase_1"
        phase_names.append(phase_name)

    all_phase_names = ["Not indexed"] + phase_names

    # --- EBSD indexing data ---
    n = 0
    euler_rad = np.zeros((0, 3), dtype=np.float64)
    phase_ids = np.zeros(0, dtype=np.int32)
    confidence = np.zeros(0, dtype=np.float32)
    patterns = None

    if ebsd_grp is not None:
        # Euler angles (radians, Bunge ZXZ)
        euler = _read_dataset(ebsd_grp, "EulerAngles")
        if euler is not None:
            if euler.ndim == 3:
                # (rows, cols, 3) → (N, 3)
                euler_rad = sanitize_array(
                    euler.reshape(-1, 3).astype(np.float64),
                    name="euler_angles (EMsoft)",
                )
            elif euler.ndim == 2 and euler.shape[1] == 3:
                euler_rad = sanitize_array(
                    euler.astype(np.float64),
                    name="euler_angles (EMsoft)",
                )
            n = euler_rad.shape[0]

        # Phase IDs
        phase_raw = _read_dataset(ebsd_grp, "Phase")
        if phase_raw is not None:
            phase_ids = phase_raw.flatten().astype(np.int32)
            if n == 0:
                n = len(phase_ids)

        # Quality metrics (prefer CI, then IQ, then AvDotProductMap)
        ci = _read_dataset(ebsd_grp, "CI")
        iq = _read_dataset(ebsd_grp, "IQ")
        dp = _read_dataset(ebsd_grp, "AvDotProductMap")

        if ci is not None:
            confidence = ci.flatten().astype(np.float32)
            meta["ci"] = ci.flatten().astype(np.float64)
        elif dp is not None:
            confidence = dp.flatten().astype(np.float32)
            meta["dot_product"] = dp.flatten().astype(np.float64)
        elif iq is not None:
            iq_flat = sanitize_array(
                iq.flatten().astype(np.float64), name="IQ (EMsoft)"
            )
            iq_max = iq_flat.max() if iq_flat.size > 0 else 1.0
            if np.isfinite(iq_max) and iq_max > 0:
                confidence = (iq_flat / iq_max).astype(np.float32)
            else:
                confidence = np.zeros(len(iq_flat), dtype=np.float32)
            meta["image_quality"] = iq_flat

        # Patterns
        raw_patterns = _read_dataset(ebsd_grp, "EBSDPatterns")
        if raw_patterns is not None:
            if raw_patterns.ndim == 3:
                patterns = raw_patterns.astype(np.uint8)
            elif raw_patterns.ndim == 4:
                nr, nc, ph, pw = raw_patterns.shape
                patterns = raw_patterns.reshape(nr * nc, ph, pw).astype(
                    np.uint8
                )
            if patterns is not None and n == 0:
                n = patterns.shape[0]

    elif master_grp is not None:
        # Master pattern file — no per-point data, just metadata
        meta["file_type"] = "master_pattern"
        master_nh = _read_dataset(master_grp, "masterSPNH")
        if master_nh is not None:
            meta["master_pattern_shape"] = list(master_nh.shape)

    # Ensure consistent sizes
    if euler_rad.shape[0] == 0 and n > 0:
        euler_rad = np.zeros((n, 3), dtype=np.float64)
    if phase_ids.size == 0 and n > 0:
        phase_ids = np.zeros(n, dtype=np.int32)
    if confidence.size == 0 and n > 0:
        confidence = np.zeros(n, dtype=np.float32)

    phase_ids = np.clip(phase_ids, 0, len(all_phase_names) - 1)

    # Grid shape
    scan_shape: tuple[int, int] | None = None
    if ipf_ht is not None and ipf_wd is not None:
        if ipf_ht > 0 and ipf_wd > 0:
            scan_shape = (ipf_ht, ipf_wd)
    if scan_shape is None and n > 0:
        # Try to infer from Phase array shape if 2D
        phase_raw = _read_dataset(
            ebsd_grp, "Phase"
        ) if ebsd_grp is not None else None
        if phase_raw is not None and phase_raw.ndim == 2:
            scan_shape = (phase_raw.shape[0], phase_raw.shape[1])

    step_sizes: tuple[float, float] | None = None
    if step_x is not None and step_y is not None:
        step_sizes = (float(step_x), float(step_y))

    return ScanData(
        patterns=patterns,
        phase_ids=phase_ids,
        phase_names=all_phase_names,
        euler_angles=euler_rad,
        confidence_scores=confidence,
        eds_data=None,
        scan_shape=scan_shape,
        step_sizes=step_sizes,
        source_format=ScanFormat.EMSOFT,
        source_file=str(path),
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Quaternion ↔ Euler conversion
# ---------------------------------------------------------------------------


def _quaternion_to_euler(quats: np.ndarray) -> np.ndarray:
    """Convert quaternions (w, x, y, z) to Bunge Euler angles.

    Inverse of ``euler_to_quaternion``.

    Parameters
    ----------
    quats : np.ndarray
        Quaternions, shape ``(N, 4)``, order ``(w, x, y, z)``.

    Returns
    -------
    np.ndarray
        Euler angles ``(phi1, Phi, phi2)`` in radians, shape ``(N, 3)``.
    """
    q = np.asarray(quats, dtype=np.float64)
    if q.ndim == 1:
        q = q[np.newaxis, :]

    # Normalize quaternions to unit length for numerical safety
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    q = q / norms

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Bunge ZXZ: inverse of the forward formula
    # sigma = atan2(z, w)     → (phi1 + phi2) / 2
    # delta = atan2(y, x)     → (phi1 - phi2) / 2
    # Phi = 2 * acos(sqrt(w^2 + z^2))
    sigma = np.arctan2(z, w)
    delta = np.arctan2(y, x)

    chi = np.sqrt(w**2 + z**2)
    chi = np.clip(chi, 0.0, 1.0)
    big_phi = 2.0 * np.arccos(chi)

    phi1 = sigma + delta
    phi2 = sigma - delta

    # Normalize to [0, 2π)
    phi1 = phi1 % (2.0 * np.pi)
    phi2 = phi2 % (2.0 * np.pi)

    result = np.stack([phi1, big_phi, phi2], axis=-1)
    # Replace any residual NaN/Inf from degenerate quaternions
    result = np.where(np.isfinite(result), result, 0.0)
    return result


# ---------------------------------------------------------------------------
# HDF5 reading helpers
# ---------------------------------------------------------------------------


def _read_dataset(group: Any, name: str) -> np.ndarray | None:
    """Read an HDF5 dataset, returning None if not found."""
    if group is None or name not in group:
        return None
    import h5py

    ds = group[name]
    if isinstance(ds, h5py.Dataset):
        return np.asarray(ds[()])
    return None


def _read_scalar(
    group: Any,
    name: str,
    dtype: type = float,
) -> Any:
    """Read a scalar value from an HDF5 dataset.

    Returns ``None`` if the dataset does not exist.
    """
    arr = _read_dataset(group, name)
    if arr is None:
        return None
    try:
        return dtype(arr.flat[0])
    except (ValueError, IndexError):
        return None


def _read_string(group: Any, name: str) -> str:
    """Read a string dataset, decoding bytes if needed."""
    if group is None or name not in group:
        return ""
    import h5py

    ds = group[name]
    if not isinstance(ds, h5py.Dataset):
        return ""
    val = ds[()]
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, np.bytes_):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, str):
        return val
    if isinstance(val, np.ndarray):
        if val.size == 1:
            item = val.flat[0]
            if isinstance(item, bytes):
                return item.decode("utf-8", errors="replace")
            return str(item)
    return str(val)


def _int_sort_key(s: str) -> int:
    """Sort key that treats numeric strings as integers."""
    try:
        return int(s)
    except ValueError:
        return 0
