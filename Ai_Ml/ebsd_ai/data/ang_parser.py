"""Parser for EDAX/OIM .ang files.

The ANG format is a plain-text columnar format produced by EDAX/OIM
EBSD acquisition and analysis software.  Each file consists of a
header (lines starting with ``#``) followed by whitespace-delimited
data rows.

Header fields
-------------
- ``# GRID: SqrGrid`` or ``HexGrid``
- ``# XSTEP:`` and ``# YSTEP:`` — step sizes in micrometers
- ``# NCOLS_ODD:`` / ``# NCOLS_EVEN:`` and ``# NROWS:`` — grid dimensions
- ``# Phase N`` blocks — phase definitions (name, formula, etc.)

Data columns (standard 10-column layout)
-----------------------------------------
phi1, Phi, phi2, x, y, IQ, CI, PhaseID, SEM_signal, Fit

- Euler angles are in **radians**, Bunge convention.
- PhaseID is 1-based; 0 means not indexed.
- CI (Confidence Index) is in [0, 1].
- IQ (Image Quality) is an arbitrary positive float.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ebsd_ai.config import sanitize_array
from ebsd_ai.data.scan_import import ScanData, ScanFormat

# ---------------------------------------------------------------------------
# ANG header keys
# ---------------------------------------------------------------------------

_KEY_GRID = "# GRID:"
_KEY_XSTEP = "# XSTEP:"
_KEY_YSTEP = "# YSTEP:"
_KEY_NCOLS_ODD = "# NCOLS_ODD:"
_KEY_NCOLS_EVEN = "# NCOLS_EVEN:"
_KEY_NROWS = "# NROWS:"
_KEY_PHASE = "# Phase"
_KEY_PHASE_NAME = "# MaterialName"
_KEY_PHASE_FORMULA = "# Formula"
_KEY_PHASE_SYMMETRY = "# Symmetry"
_KEY_KV = "# KV:"
_KEY_WD = "# WORKING_DISTANCE:"
_KEY_SAMPLE_TILT = "# SampleTiltAngle:"

# Minimum number of data columns expected in a valid ANG row
_MIN_DATA_COLUMNS = 8


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_ang(filepath: str | Path) -> ScanData:
    """Parse an EDAX/OIM .ang file into a :class:`ScanData`.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.ang`` file.

    Returns
    -------
    ScanData
        Parsed scan data.  ``patterns`` is always ``None``
        (ANG files do not contain patterns).

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file cannot be parsed (no data rows, bad column count).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"ANG file not found: {path}")

    header, data_lines = _split_header_data(path)
    meta = _parse_header(header)
    _raw_names = meta.pop("_phase_names")
    phase_names: list[str] = _raw_names if isinstance(_raw_names, list) else []

    if not data_lines:
        raise ValueError(f"No data rows found in ANG file: {path}")

    data = _parse_data_rows(data_lines)

    n = data.shape[0]

    euler_angles = sanitize_array(data[:, 0:3], name="euler_angles (ANG)")
    # x, y in columns 3, 4 — not stored separately
    iq = data[:, 5] if data.shape[1] > 5 else np.zeros(n, dtype=np.float64)
    ci = data[:, 6] if data.shape[1] > 6 else np.zeros(n, dtype=np.float64)
    ci = sanitize_array(ci, name="CI (ANG)")
    raw_phase = (
        data[:, 7].astype(np.int32)
        if data.shape[1] > 7
        else np.zeros(n, dtype=np.int32)
    )

    # ANG phase IDs are 1-based; 0 = not indexed.
    # We shift to 0-based and ensure phase_names[0] = "Not indexed".
    all_phase_names = ["Not indexed"] + phase_names
    phase_ids = np.clip(raw_phase, 0, len(all_phase_names) - 1)

    # Infer grid shape
    scan_shape = _infer_grid_shape(meta, n)
    step_x = meta.get("xstep")
    step_y = meta.get("ystep")
    step_sizes: tuple[float, float] | None = None
    if step_x is not None and step_y is not None:
        step_sizes = (
            float(step_x),  # type: ignore[arg-type]
            float(step_y),  # type: ignore[arg-type]
        )

    # Store IQ in metadata
    meta["image_quality"] = iq

    return ScanData(
        patterns=None,
        phase_ids=phase_ids,
        phase_names=all_phase_names,
        euler_angles=euler_angles.astype(np.float64),
        confidence_scores=ci.astype(np.float32),
        eds_data=None,
        scan_shape=scan_shape,
        step_sizes=step_sizes,
        source_format=ScanFormat.ANG,
        source_file=str(path),
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_header_data(path: Path) -> tuple[list[str], list[str]]:
    """Read file and separate header lines from data lines."""
    header: list[str] = []
    data: list[str] = []

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                header.append(stripped)
            else:
                data.append(stripped)

    return header, data


def _parse_header(header_lines: list[str]) -> dict[str, object]:
    """Extract metadata from ANG header lines.

    Returns a dict with keys like ``grid``, ``xstep``, ``ystep``,
    ``ncols``, ``nrows``, ``kv``, ``working_distance``, ``sample_tilt``,
    and the special key ``_phase_names`` (list of str).
    """
    meta: dict[str, object] = {}
    phase_names: list[str] = []
    current_phase_name: str | None = None

    for line in header_lines:
        if line.startswith(_KEY_GRID):
            meta["grid"] = line[len(_KEY_GRID):].strip()

        elif line.startswith(_KEY_XSTEP):
            meta["xstep"] = _parse_float(line, _KEY_XSTEP)

        elif line.startswith(_KEY_YSTEP):
            meta["ystep"] = _parse_float(line, _KEY_YSTEP)

        elif line.startswith(_KEY_NCOLS_ODD):
            meta["ncols"] = _parse_int(line, _KEY_NCOLS_ODD)

        elif line.startswith(_KEY_NCOLS_EVEN):
            # For hex grids; store but prefer ncols_odd
            if "ncols" not in meta:
                meta["ncols"] = _parse_int(line, _KEY_NCOLS_EVEN)

        elif line.startswith(_KEY_NROWS):
            meta["nrows"] = _parse_int(line, _KEY_NROWS)

        elif line.startswith(_KEY_PHASE) and not line.startswith(
            _KEY_PHASE_NAME
        ):
            # New phase block: "# Phase N"
            if current_phase_name is not None:
                phase_names.append(current_phase_name)
            current_phase_name = None

        elif line.startswith(_KEY_PHASE_NAME):
            name = line[len(_KEY_PHASE_NAME):].strip()
            current_phase_name = name if name else None

        elif line.startswith(_KEY_PHASE_FORMULA):
            # Use formula as fallback if no MaterialName
            if current_phase_name is None:
                formula = line[len(_KEY_PHASE_FORMULA):].strip()
                if formula:
                    current_phase_name = formula

        elif line.startswith(_KEY_KV):
            meta["kv"] = _parse_float(line, _KEY_KV)

        elif line.startswith(_KEY_WD):
            meta["working_distance"] = _parse_float(line, _KEY_WD)

        elif line.startswith(_KEY_SAMPLE_TILT):
            meta["sample_tilt"] = _parse_float(line, _KEY_SAMPLE_TILT)

    # Don't forget the last phase
    if current_phase_name is not None:
        phase_names.append(current_phase_name)

    meta["_phase_names"] = phase_names
    return meta


def _parse_float(line: str, prefix: str) -> float:
    """Extract a float value after a header prefix."""
    try:
        return float(line[len(prefix):].strip())
    except ValueError:
        return 0.0


def _parse_int(line: str, prefix: str) -> int:
    """Extract an int value after a header prefix."""
    try:
        return int(line[len(prefix):].strip())
    except ValueError:
        return 0


def _parse_data_rows(lines: list[str]) -> np.ndarray:
    """Parse whitespace-delimited numeric data rows.

    Parameters
    ----------
    lines : list[str]
        Non-header, non-empty lines from the ANG file.

    Returns
    -------
    np.ndarray
        Shape ``(N, C)`` where C is the number of columns.

    Raises
    ------
    ValueError
        If no valid data rows are found or columns are inconsistent.
    """
    rows: list[list[float]] = []
    ncols: int | None = None

    for line in lines:
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
            # Skip rows with inconsistent column count
            continue

        rows.append(values)

    if not rows:
        raise ValueError(
            "No valid data rows found in ANG file "
            f"(need >= {_MIN_DATA_COLUMNS} columns per row)"
        )

    return np.array(rows, dtype=np.float64)


def _infer_grid_shape(
    meta: dict[str, object],
    n_points: int,
) -> tuple[int, int] | None:
    """Infer (nrows, ncols) from header or data point count.

    Returns
    -------
    tuple[int, int] or None
        ``(nrows, ncols)`` if determinable, else ``None``.
    """
    ncols = meta.get("ncols")
    nrows = meta.get("nrows")

    if isinstance(ncols, int) and ncols > 0:
        if isinstance(nrows, int) and nrows > 0:
            return (nrows, ncols)
        # Infer nrows from point count
        computed_nrows = n_points // ncols
        if computed_nrows > 0:
            return (computed_nrows, ncols)

    if isinstance(nrows, int) and nrows > 0:
        computed_ncols = n_points // nrows
        if computed_ncols > 0:
            return (nrows, computed_ncols)

    return None
