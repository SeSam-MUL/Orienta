"""Parser for Oxford/HKL Channel Text File (.ctf) format.

The CTF format is a plain-text columnar format produced by Oxford
Instruments / HKL Channel 5 EBSD software.  The file has a header
section followed by tab-delimited data rows.

Header structure
----------------
The header consists of keyword lines (not prefixed with ``#``).
Important keywords:

- ``Channel Text File`` — identifies the format (first line)
- ``XCells`` / ``YCells`` — grid dimensions
- ``XStep`` / ``YStep`` — step sizes in micrometers
- ``AcqE1`` — accelerating voltage (kV)
- ``KV`` — alternative kV field
- ``Phases`` — number of phases, followed by phase definition lines

Phase definition lines appear after ``Phases\\tN`` and have the format:

    a;b;c\\talpha;beta;gamma\\tName\\tGroup\\tSpace Group\\t...

Data columns (tab-delimited)
-----------------------------
Phase, X, Y, Bands, Error, Euler1, Euler2, Euler3, MAD, BC, BS

- **Euler angles are in DEGREES** (Bunge convention).
- Phase 0 = not indexed.
- MAD (Mean Angular Deviation) is the fit quality metric.
- BC (Band Contrast) and BS (Band Slope) are pattern quality metrics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ebsd_ai.config import mad_to_confidence, sanitize_array
from ebsd_ai.data.scan_import import ScanData, ScanFormat, degrees_to_radians

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum number of data columns for a valid CTF data row
_MIN_DATA_COLUMNS = 8

# Header keywords
_KW_XCELLS = "XCells"
_KW_YCELLS = "YCells"
_KW_XSTEP = "XStep"
_KW_YSTEP = "YStep"
_KW_ACQE1 = "AcqE1"
_KW_KV = "KV"
_KW_PHASES = "Phases"
_KW_MAG = "Mag"
_KW_COVERAGE = "Coverage"
_KW_DEVICE = "Device"
_KW_AUTHOR = "Author"
_KW_JOB_MODE = "JobMode"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_ctf(filepath: str | Path) -> ScanData:
    """Parse an Oxford/HKL .ctf file into a :class:`ScanData`.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.ctf`` file.

    Returns
    -------
    ScanData
        Parsed scan data.  ``patterns`` is always ``None``
        (CTF files do not contain patterns).

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file cannot be parsed (no data rows, bad format).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CTF file not found: {path}")

    header_lines, data_lines = _split_header_data(path)
    meta, phase_names = _parse_header(header_lines)

    if not data_lines:
        raise ValueError(f"No data rows found in CTF file: {path}")

    data = _parse_data_rows(data_lines)
    n = data.shape[0]

    # Column layout: Phase, X, Y, Bands, Error, Euler1, Euler2, Euler3, ...
    raw_phase = data[:, 0].astype(np.int32)

    # Euler angles in degrees → convert to radians
    euler_deg = sanitize_array(data[:, 5:8], name="euler_angles (CTF)")
    euler_rad = degrees_to_radians(euler_deg)

    # MAD (Mean Angular Deviation) — used as inverse confidence
    # Lower MAD = better fit. Normalize to [0, 1] confidence.
    mad = (
        data[:, 8]
        if data.shape[1] > 8
        else np.zeros(n, dtype=np.float64)
    )
    mad = sanitize_array(mad, name="MAD (CTF)")
    confidence = mad_to_confidence(mad)

    # Band Contrast (BC) — pattern quality metric
    bc = (
        data[:, 9]
        if data.shape[1] > 9
        else np.zeros(n, dtype=np.float64)
    )

    # Band Slope (BS)
    bs = (
        data[:, 10]
        if data.shape[1] > 10
        else np.zeros(n, dtype=np.float64)
    )

    # Phase names: CTF uses 0 = not indexed, 1..N = phases
    all_phase_names = ["Not indexed"] + phase_names
    phase_ids = np.clip(raw_phase, 0, len(all_phase_names) - 1)

    # Grid shape from header
    scan_shape = _get_grid_shape(meta, n)

    step_x = meta.get("xstep")
    step_y = meta.get("ystep")
    step_sizes: tuple[float, float] | None = None
    if step_x is not None and step_y is not None:
        step_sizes = (
            float(step_x),  # type: ignore[arg-type]
            float(step_y),  # type: ignore[arg-type]
        )

    # Store quality metrics in metadata
    meta["band_contrast"] = bc
    meta["band_slope"] = bs
    meta["mad"] = mad

    return ScanData(
        patterns=None,
        phase_ids=phase_ids,
        phase_names=all_phase_names,
        euler_angles=euler_rad,
        confidence_scores=confidence.astype(np.float32),
        eds_data=None,
        scan_shape=scan_shape,
        step_sizes=step_sizes,
        source_format=ScanFormat.CTF,
        source_file=str(path),
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_header_data(
    path: Path,
) -> tuple[list[str], list[str]]:
    """Read a CTF file and split into header and data lines.

    The data section starts after all header keywords are consumed.
    We detect the transition by looking for the first line whose first
    field is a valid integer (the Phase column).
    """
    header: list[str] = []
    data: list[str] = []
    in_data = False

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue

            if in_data:
                data.append(stripped)
                continue

            # Try to detect transition to data section.
            # Data rows start with an integer (Phase column).
            # Header lines start with a keyword or phase definition.
            if _is_data_row(stripped):
                in_data = True
                data.append(stripped)
            else:
                header.append(stripped)

    return header, data


def _is_data_row(line: str) -> bool:
    """Check if a line looks like a CTF data row.

    Data rows are tab-delimited and start with an integer (Phase).
    We require at least _MIN_DATA_COLUMNS tab-separated fields,
    with the first being a non-negative integer.
    """
    parts = line.split("\t")
    if len(parts) < _MIN_DATA_COLUMNS:
        # Also try space-delimited (some CTF variants)
        parts = line.split()
        if len(parts) < _MIN_DATA_COLUMNS:
            return False

    try:
        phase_val = int(parts[0])
        # Phase column is a small non-negative integer
        if phase_val < 0 or phase_val > 100:
            return False
        # Second and third columns should be numeric (X, Y)
        float(parts[1])
        float(parts[2])
        return True
    except (ValueError, IndexError):
        return False


def _parse_header(
    header_lines: list[str],
) -> tuple[dict[str, object], list[str]]:
    """Parse CTF header lines into metadata dict and phase names.

    Returns
    -------
    tuple[dict[str, object], list[str]]
        (metadata dict, list of phase names)
    """
    meta: dict[str, object] = {}
    phase_names: list[str] = []
    n_phases_expected = 0
    reading_phases = False
    phases_read = 0

    for line in header_lines:
        # Phase definition lines (after "Phases\tN")
        if reading_phases and phases_read < n_phases_expected:
            name = _extract_phase_name(line)
            if name:
                phase_names.append(name)
            else:
                phase_names.append(f"Phase_{phases_read + 1}")
            phases_read += 1
            if phases_read >= n_phases_expected:
                reading_phases = False
            continue

        # Split on tab for keyword\tvalue pairs
        parts = line.split("\t", 1)
        key = parts[0].strip()
        value = parts[1].strip() if len(parts) > 1 else ""

        if key == _KW_XCELLS:
            meta["xcells"] = _safe_int(value)
        elif key == _KW_YCELLS:
            meta["ycells"] = _safe_int(value)
        elif key == _KW_XSTEP:
            meta["xstep"] = _safe_float(value)
        elif key == _KW_YSTEP:
            meta["ystep"] = _safe_float(value)
        elif key == _KW_ACQE1:
            meta["kv"] = _safe_float(value)
        elif key == _KW_KV:
            # Alternative KV field
            if "kv" not in meta:
                meta["kv"] = _safe_float(value)
        elif key == _KW_PHASES:
            n_phases_expected = _safe_int(value)
            reading_phases = True
            phases_read = 0
        elif key == _KW_MAG:
            meta["magnification"] = _safe_float(value)
        elif key == _KW_COVERAGE:
            meta["coverage"] = _safe_int(value)
        elif key == _KW_DEVICE:
            meta["device"] = value
        elif key == _KW_AUTHOR:
            meta["author"] = value
        elif key == _KW_JOB_MODE:
            meta["job_mode"] = value

    return meta, phase_names


def _extract_phase_name(line: str) -> str:
    """Extract phase name from a CTF phase definition line.

    Phase lines have semicolon-separated lattice parameters in the
    first field, then tab-separated fields.  The phase name is
    typically the third tab-separated field::

        a;b;c\\talpha;beta;gamma\\tPhaseName\\tLaueGroup\\t...

    Falls back to simpler parsing if the format varies.
    """
    parts = line.split("\t")
    # Standard layout: parts[2] is the phase name
    if len(parts) >= 3:
        name = parts[2].strip()
        if name:
            return name

    # Fallback: try to find a non-numeric field
    for part in parts:
        part = part.strip()
        if part and not _looks_numeric(part):
            return part

    return ""


def _looks_numeric(s: str) -> bool:
    """Check if a string looks like a number or semicolon-separated numbers."""
    # "3.24;3.24;5.18" or "90;90;120" are numeric-like
    for chunk in s.split(";"):
        try:
            float(chunk.strip())
        except ValueError:
            return False
    return True


def _parse_data_rows(lines: list[str]) -> np.ndarray:
    """Parse CTF data rows (tab or space delimited).

    Parameters
    ----------
    lines : list[str]
        Data lines from the CTF file.

    Returns
    -------
    np.ndarray
        Shape ``(N, C)`` float64 array.

    Raises
    ------
    ValueError
        If no valid data rows are found.
    """
    rows: list[list[float]] = []
    ncols: int | None = None

    for line in lines:
        # CTF uses tabs, but some files use spaces
        parts = line.split("\t")
        if len(parts) < _MIN_DATA_COLUMNS:
            parts = line.split()
        if len(parts) < _MIN_DATA_COLUMNS:
            continue

        try:
            values = [float(p) for p in parts]
        except ValueError:
            continue

        if ncols is None:
            ncols = len(values)
        elif len(values) != ncols:
            continue

        rows.append(values)

    if not rows:
        raise ValueError(
            "No valid data rows found in CTF file "
            f"(need >= {_MIN_DATA_COLUMNS} columns per row)"
        )

    return np.array(rows, dtype=np.float64)


def _get_grid_shape(
    meta: dict[str, object],
    n_points: int,
) -> tuple[int, int] | None:
    """Get (nrows, ncols) from header metadata.

    Returns
    -------
    tuple[int, int] or None
        ``(ycells, xcells)`` if available, else ``None``.
    """
    xcells = meta.get("xcells")
    ycells = meta.get("ycells")

    if isinstance(xcells, int) and xcells > 0:
        if isinstance(ycells, int) and ycells > 0:
            return (ycells, xcells)
        computed = n_points // xcells
        if computed > 0:
            return (computed, xcells)

    if isinstance(ycells, int) and ycells > 0:
        computed = n_points // ycells
        if computed > 0:
            return (ycells, computed)

    return None


def _safe_float(s: str) -> float:
    """Parse a string to float, returning 0.0 on failure."""
    try:
        return float(s)
    except ValueError:
        return 0.0


def _safe_int(s: str) -> int:
    """Parse a string to int, returning 0 on failure."""
    try:
        return int(s)
    except ValueError:
        return 0
