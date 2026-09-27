"""Common import infrastructure for EBSD scan data.

Provides a unified ``ScanData`` container, Euler-to-quaternion conversion,
automatic file format detection, and a high-level ``import_to_store()``
function for importing parsed scans into a
:class:`~ebsd_ai.data.training_store.TrainingStore`.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from ebsd_ai.config import DetectorInfo
    from ebsd_ai.data.training_store import TrainingStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File format detection
# ---------------------------------------------------------------------------


class ScanFormat(enum.Enum):
    """Supported EBSD scan file formats."""

    ANG = "ANG"        # EDAX/OIM .ang files
    CTF = "CTF"        # Oxford/HKL .ctf files
    H5OINA = "H5OINA"  # Oxford HDF5 .h5oina files
    KIKUCHIPY = "KIKUCHIPY"  # kikuchipy HDF5 .h5 files
    EMSOFT = "EMSOFT"  # EMsoft HDF5 .h5 files
    UNKNOWN = "UNKNOWN"


# Magic bytes / signatures for HDF5 format detection
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def detect_format(filepath: str | Path) -> ScanFormat:
    """Detect the EBSD scan file format from extension and content.

    Parameters
    ----------
    filepath : str or Path
        Path to the scan file.

    Returns
    -------
    ScanFormat
        Detected file format.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    # Plain-text formats by extension
    if suffix == ".ang":
        return ScanFormat.ANG
    if suffix == ".ctf":
        return ScanFormat.CTF

    # HDF5-based formats
    if suffix in (".h5", ".hdf5", ".h5oina"):
        return _detect_hdf5_subformat(path)

    return ScanFormat.UNKNOWN


def _detect_hdf5_subformat(path: Path) -> ScanFormat:
    """Identify which HDF5 variant an EBSD file is.

    Checks for characteristic group/dataset names to distinguish
    H5OINA, kikuchipy, and EMsoft HDF5 files.

    Parameters
    ----------
    path : Path
        Path to an HDF5 file.

    Returns
    -------
    ScanFormat
        One of ``H5OINA``, ``KIKUCHIPY``, ``EMSOFT``, or ``UNKNOWN``.
    """
    # Verify HDF5 magic bytes first
    try:
        with open(path, "rb") as fh:
            magic = fh.read(8)
        if magic != _HDF5_MAGIC:
            return ScanFormat.UNKNOWN
    except OSError:
        return ScanFormat.UNKNOWN

    try:
        import h5py
    except ImportError:
        # If h5py is not available, fall back to extension only
        if path.suffix.lower() == ".h5oina":
            return ScanFormat.H5OINA
        return ScanFormat.UNKNOWN

    try:
        with h5py.File(path, "r") as f:
            # EMsoft: has "EMData" or "NMLparameters" at top level
            if "EMData" in f or "NMLparameters" in f:
                return ScanFormat.EMSOFT

            # kikuchipy: has "Scan N" groups at top level
            if any(k.startswith("Scan ") for k in f.keys()):
                return ScanFormat.KIKUCHIPY

            # H5OINA: numbered top-level groups with "EBSD" subgroup
            for key in f.keys():
                grp = f[key]
                if isinstance(grp, h5py.Group):
                    if "EBSD" in grp:
                        return ScanFormat.H5OINA

    except OSError:
        return ScanFormat.UNKNOWN

    return ScanFormat.UNKNOWN


# ---------------------------------------------------------------------------
# Euler angle conversion
# ---------------------------------------------------------------------------


def euler_to_quaternion(
    euler_angles: np.ndarray,
    convention: str = "bunge",
) -> np.ndarray:
    """Convert Euler angles to unit quaternions.

    Parameters
    ----------
    euler_angles : np.ndarray
        Euler angles in radians. Shape ``(3,)`` for a single orientation
        or ``(N, 3)`` for a batch. Columns are (phi1, Phi, phi2).
    convention : str
        Euler angle convention. Currently only ``"bunge"`` is supported
        (ZXZ rotation, the EBSD standard).

    Returns
    -------
    np.ndarray
        Unit quaternions as float64 array. Shape ``(4,)`` for a single
        input or ``(N, 4)`` for a batch. Quaternion order is
        ``(w, x, y, z)`` (scalar-first).

    Raises
    ------
    ValueError
        If *convention* is not supported or *euler_angles* has wrong shape.

    Notes
    -----
    The Bunge convention defines Euler angles as three successive rotations:

    1. Rotate by phi1 about the Z axis
    2. Rotate by Phi about the new X axis
    3. Rotate by phi2 about the new Z axis

    This is the standard in EBSD and crystallographic texture analysis.
    The conversion follows the Bunge ZXZ passive convention used by
    MTEX, DREAM.3D, and orix.
    """
    if convention.lower() != "bunge":
        raise ValueError(
            f"Unsupported Euler convention: {convention!r}. "
            f"Only 'bunge' is currently supported."
        )

    angles = np.asarray(euler_angles, dtype=np.float64)
    single = angles.ndim == 1
    if single:
        angles = angles[np.newaxis, :]

    if angles.ndim != 2 or angles.shape[1] != 3:
        raise ValueError(
            f"Expected euler_angles with shape (N, 3) or (3,), "
            f"got {euler_angles.shape}"
        )

    phi1 = angles[:, 0]
    big_phi = angles[:, 1]
    phi2 = angles[:, 2]

    # Half-angles for the Bunge ZXZ decomposition:
    #   q = q_z(phi1) * q_x(Phi) * q_z(phi2)
    #
    # Using the closed-form:
    #   sigma = (phi1 + phi2) / 2
    #   delta = (phi1 - phi2) / 2
    sigma = (phi1 + phi2) / 2.0
    delta = (phi1 - phi2) / 2.0
    c_phi = np.cos(big_phi / 2.0)
    s_phi = np.sin(big_phi / 2.0)

    w = c_phi * np.cos(sigma)
    x = s_phi * np.cos(delta)
    y = s_phi * np.sin(delta)
    z = c_phi * np.sin(sigma)

    quats = np.stack([w, x, y, z], axis=-1)

    # Ensure positive scalar part (canonical form)
    neg_mask = quats[:, 0] < 0
    quats[neg_mask] *= -1.0

    # Normalize to unit quaternion (handle numerical drift)
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    quats = quats / norms

    if single:
        result: np.ndarray = quats[0]
        return result
    return quats


def degrees_to_radians(degrees: np.ndarray) -> np.ndarray:
    """Convert angles from degrees to radians.

    Parameters
    ----------
    degrees : np.ndarray
        Angles in degrees.

    Returns
    -------
    np.ndarray
        Angles in radians (float64).
    """
    result: np.ndarray = np.deg2rad(np.asarray(degrees, dtype=np.float64))
    return result


# ---------------------------------------------------------------------------
# ScanData container
# ---------------------------------------------------------------------------


@dataclass
class ScanData:
    """Unified container for parsed EBSD scan data.

    Holds all data needed to feed into
    :class:`~ebsd_ai.data.training_store.TrainingStore`.
    Not all fields are required; optional fields may be ``None`` when
    the source format does not provide them.

    Parameters
    ----------
    patterns : np.ndarray or None
        EBSD patterns with shape ``(N, H, W)``, uint8 or uint16.
        May be ``None`` if the file only contains indexed results.
    phase_ids : np.ndarray
        Integer phase IDs per point, shape ``(N,)``. Index into
        ``phase_names``. Use 0 for unindexed/unknown.
    phase_names : list[str]
        Phase name lookup table. ``phase_names[phase_ids[i]]`` gives
        the phase name for point *i*.
    euler_angles : np.ndarray
        Euler angles in radians, shape ``(N, 3)``, Bunge convention
        (phi1, Phi, phi2).
    confidence_scores : np.ndarray
        Confidence index per point, shape ``(N,)``, typically in [0, 1].
    eds_data : dict[str, np.ndarray] or None
        Element symbol → per-point concentration array of shape ``(N,)``.
        ``None`` if no EDS data is available.
    scan_shape : tuple[int, int] or None
        (n_rows, n_cols) grid dimensions, if the scan is a regular grid.
        ``None`` for irregular point clouds.
    step_sizes : tuple[float, float] or None
        (step_x, step_y) in micrometers, if known.
    source_format : ScanFormat
        Format the data was parsed from.
    source_file : str
        Original file path for provenance.
    metadata : dict
        Additional format-specific metadata (detector info, kV, etc.).
    """

    patterns: Optional[np.ndarray] = None
    phase_ids: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int32)
    )
    phase_names: list[str] = field(default_factory=lambda: ["Unknown"])
    euler_angles: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float64)
    )
    confidence_scores: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    eds_data: Optional[dict[str, np.ndarray]] = None
    scan_shape: Optional[tuple[int, int]] = None
    step_sizes: Optional[tuple[float, float]] = None
    source_format: ScanFormat = ScanFormat.UNKNOWN
    source_file: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        """Number of scan points."""
        if self.patterns is not None:
            n: int = self.patterns.shape[0]
            return n
        return len(self.phase_ids)

    @property
    def has_patterns(self) -> bool:
        """Whether EBSD patterns are available."""
        return self.patterns is not None and self.patterns.size > 0

    @property
    def has_eds(self) -> bool:
        """Whether EDS data is available."""
        return self.eds_data is not None and len(self.eds_data) > 0

    @property
    def orientations(self) -> np.ndarray:
        """Euler angles converted to unit quaternions (w, x, y, z).

        Returns
        -------
        np.ndarray
            Shape ``(N, 4)``, float64. Returns zero quaternions if
            no Euler angles are set.
        """
        if self.euler_angles.size == 0:
            return np.zeros((self.n_points, 4), dtype=np.float64)
        return euler_to_quaternion(self.euler_angles, convention="bunge")

    def validate(self) -> list[str]:
        """Check internal consistency, returning a list of issues.

        Returns
        -------
        list[str]
            Empty if data is consistent. Otherwise, one string per issue.
        """
        issues: list[str] = []
        n = self.n_points

        if n == 0:
            issues.append("ScanData has 0 points")
            return issues

        if self.phase_ids.shape != (n,):
            issues.append(
                f"phase_ids shape {self.phase_ids.shape} != ({n},)"
            )

        if self.euler_angles.size > 0 and self.euler_angles.shape != (n, 3):
            issues.append(
                f"euler_angles shape {self.euler_angles.shape} != ({n}, 3)"
            )

        if self.confidence_scores.size > 0 and self.confidence_scores.shape != (n,):
            issues.append(
                f"confidence_scores shape {self.confidence_scores.shape} != ({n},)"
            )

        if self.patterns is not None and self.patterns.ndim != 3:
            issues.append(
                f"patterns ndim {self.patterns.ndim} != 3 (expected N, H, W)"
            )

        if self.patterns is not None and self.patterns.shape[0] != n:
            issues.append(
                f"patterns shape[0] {self.patterns.shape[0]} != n_points {n}"
            )

        # Check phase_ids are within range of phase_names
        if self.phase_ids.size > 0:
            max_id = int(self.phase_ids.max())
            if max_id >= len(self.phase_names):
                issues.append(
                    f"max phase_id {max_id} >= len(phase_names) "
                    f"{len(self.phase_names)}"
                )
            if int(self.phase_ids.min()) < 0:
                issues.append("phase_ids contains negative values")

        # Check EDS arrays have consistent length
        if self.eds_data is not None:
            for el, arr in self.eds_data.items():
                if arr.shape != (n,):
                    issues.append(
                        f"eds_data['{el}'] shape {arr.shape} != ({n},)"
                    )

        # Check for non-finite values in numerical arrays
        if self.euler_angles.size > 0 and not np.all(
            np.isfinite(self.euler_angles)
        ):
            n_bad = int(np.sum(~np.isfinite(self.euler_angles)))
            issues.append(
                f"euler_angles contains {n_bad} non-finite value(s)"
            )
        if self.confidence_scores.size > 0 and not np.all(
            np.isfinite(self.confidence_scores)
        ):
            n_bad = int(np.sum(~np.isfinite(self.confidence_scores)))
            issues.append(
                f"confidence_scores contains {n_bad} non-finite value(s)"
            )

        return issues


# ---------------------------------------------------------------------------
# High-level parse / import
# ---------------------------------------------------------------------------


def parse_scan(filepath: str | Path) -> ScanData:
    """Parse an EBSD scan file into a :class:`ScanData`.

    Detects the file format automatically and delegates to the
    appropriate parser (ANG, CTF, or HDF5).

    Parameters
    ----------
    filepath : str or Path
        Path to the scan file (``.ang``, ``.ctf``, ``.h5``,
        ``.hdf5``, or ``.h5oina``).

    Returns
    -------
    ScanData
        Parsed scan data.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the format is unknown or the file cannot be parsed.
    """
    path = Path(filepath)
    fmt = detect_format(path)

    if fmt == ScanFormat.ANG:
        from ebsd_ai.data.ang_parser import parse_ang

        return parse_ang(path)

    if fmt == ScanFormat.CTF:
        from ebsd_ai.data.ctf_parser import parse_ctf

        return parse_ctf(path)

    if fmt in (ScanFormat.H5OINA, ScanFormat.KIKUCHIPY, ScanFormat.EMSOFT):
        from ebsd_ai.data.hdf5_parser import parse_hdf5

        return parse_hdf5(path)

    raise ValueError(
        f"Unknown or unsupported EBSD scan format for: {path}. "
        f"Detected: {fmt.value}. "
        f"Supported extensions: .ang, .ctf, .h5, .hdf5, .h5oina"
    )


@dataclass
class ImportResult:
    """Summary of an ``import_to_store`` operation.

    Parameters
    ----------
    source_file : str
        Path to the imported scan file.
    format : ScanFormat
        Detected file format.
    n_points_total : int
        Total number of scan points in the file.
    n_points_imported : int
        Number of points that passed the CI threshold and were stored.
    n_points_skipped_no_pattern : int
        Points skipped because patterns were not available in the file.
    phase_counts : dict[str, int]
        Number of imported points per phase name.
    ci_threshold : float
        CI threshold that was applied.
    has_eds : bool
        Whether EDS data was present in the scan.
    warnings : list[str]
        Any non-fatal issues encountered during import.
    """

    source_file: str = ""
    format: ScanFormat = ScanFormat.UNKNOWN
    n_points_total: int = 0
    n_points_imported: int = 0
    n_points_skipped_no_pattern: int = 0
    phase_counts: dict[str, int] = field(default_factory=dict)
    ci_threshold: float = 0.3
    has_eds: bool = False
    warnings: list[str] = field(default_factory=list)


def _build_detector_info(
    scan: ScanData,
) -> DetectorInfo:
    """Build a :class:`DetectorInfo` from ScanData metadata.

    Uses metadata keys ``kv``, ``working_distance``, ``sample_tilt``
    when available; falls back to defaults otherwise.  The detector
    manufacturer is inferred from the source format.
    """
    from ebsd_ai.config import (
        DetectorConvention,
        DetectorInfo,
        DetectorManufacturer,
    )

    meta = scan.metadata

    # Map source format → manufacturer / convention
    mfr_map: dict[ScanFormat, DetectorManufacturer] = {
        ScanFormat.ANG: DetectorManufacturer.EDAX,
        ScanFormat.CTF: DetectorManufacturer.OXFORD,
        ScanFormat.H5OINA: DetectorManufacturer.OXFORD,
        ScanFormat.KIKUCHIPY: DetectorManufacturer.OTHER,
        ScanFormat.EMSOFT: DetectorManufacturer.OTHER,
    }
    conv_map: dict[ScanFormat, DetectorConvention] = {
        ScanFormat.ANG: DetectorConvention.EDAX,
        ScanFormat.CTF: DetectorConvention.OXFORD,
        ScanFormat.H5OINA: DetectorConvention.OXFORD,
        ScanFormat.KIKUCHIPY: DetectorConvention.KIKUCHIPY,
        ScanFormat.EMSOFT: DetectorConvention.EMSOFT,
    }

    manufacturer = mfr_map.get(scan.source_format, DetectorManufacturer.OTHER)
    convention = conv_map.get(scan.source_format, DetectorConvention.KIKUCHIPY)

    kv = float(meta.get("kv", 20.0))  # type: ignore[arg-type]
    wd = float(meta.get("working_distance", 15.0))  # type: ignore[arg-type]
    tilt = float(meta.get("sample_tilt", 70.0))  # type: ignore[arg-type]

    return DetectorInfo(
        manufacturer=manufacturer,
        pc_convention=convention,
        kv=kv,
        working_distance=wd,
        sample_tilt=tilt,
    )


_DEFAULT_IMPORT_CHUNK_SIZE = 500
"""Default number of scan points processed per chunk during import."""


def import_to_store(
    filepath: str | Path,
    store: TrainingStore,
    *,
    ci_threshold: float = 0.3,
    scan_data: Optional[ScanData] = None,
    target_size: int = 128,
    chunk_size: int = _DEFAULT_IMPORT_CHUNK_SIZE,
) -> ImportResult:
    """Import an EBSD scan file into a :class:`TrainingStore`.

    Parses the file (or uses a pre-parsed :class:`ScanData`), filters
    points by confidence index, and stores the high-confidence samples
    in the training store.  Large scans are processed in chunks to
    limit peak memory usage.

    Parameters
    ----------
    filepath : str or Path
        Path to the scan file.  Used for format detection and
        provenance, even when *scan_data* is supplied.
    store : TrainingStore
        Target training store.
    ci_threshold : float
        Minimum confidence score for a point to be imported.
        Points below this threshold are skipped.
    scan_data : ScanData, optional
        Pre-parsed scan data.  If ``None``, the file is parsed
        automatically via :func:`parse_scan`.
    target_size : int
        Normalized pattern size passed to the training store.
    chunk_size : int
        Number of scan points to process per chunk.  Larger values
        are faster but use more memory.  Default: 500.

    Returns
    -------
    ImportResult
        Summary of the import operation.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist and *scan_data* is not provided.
    ValueError
        If the file format is unknown, the scan has no data, or
        patterns are required but missing.
    """
    path = Path(filepath)
    result = ImportResult(source_file=str(path), ci_threshold=ci_threshold)

    # Parse if not pre-supplied
    if scan_data is None:
        scan_data = parse_scan(path)

    result.format = scan_data.source_format
    result.n_points_total = scan_data.n_points
    result.has_eds = scan_data.has_eds

    # Validate
    issues = scan_data.validate()
    if issues:
        result.warnings.extend(issues)

    if scan_data.n_points == 0:
        result.warnings.append("Scan has 0 points, nothing to import")
        return result

    # Must have patterns to import into the training store
    if not scan_data.has_patterns:
        result.n_points_skipped_no_pattern = scan_data.n_points
        result.warnings.append(
            f"No patterns in {path.name}. "
            f"Only files with embedded EBSD patterns can be imported."
        )
        return result

    assert scan_data.patterns is not None  # for type narrowing

    # Build detector info from metadata
    detector = _build_detector_info(scan_data)

    # Compute orientations (Euler → quaternion)
    orientations = scan_data.orientations  # (N, 4)

    # Import in chunks to limit peak memory
    n_total = scan_data.n_points
    n_added = 0

    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        chunk_slice = slice(start, end)

        chunk_eds: dict[str, np.ndarray] | None = None
        if scan_data.eds_data is not None:
            chunk_eds = {
                el: arr[chunk_slice] for el, arr in scan_data.eds_data.items()
            }

        n_chunk = store.add_from_indexing(
            patterns=scan_data.patterns[chunk_slice],
            phase_ids=scan_data.phase_ids[chunk_slice],
            phase_names=scan_data.phase_names,
            orientations=orientations[chunk_slice],
            confidence_scores=scan_data.confidence_scores[chunk_slice],
            detector_info=detector,
            ci_threshold=ci_threshold,
            eds_data=chunk_eds,
            source_file=str(path),
            target_size=target_size,
        )
        n_added += n_chunk

    result.n_points_imported = n_added

    # Count phases among imported points
    mask = scan_data.confidence_scores >= ci_threshold
    if mask.any():
        for idx in np.where(mask)[0]:
            pid = int(scan_data.phase_ids[idx])
            name = scan_data.phase_names[pid]
            result.phase_counts[name] = result.phase_counts.get(name, 0) + 1

    logger.info(
        "Imported %d/%d points from %s (CI >= %.2f)",
        n_added,
        scan_data.n_points,
        path.name,
        ci_threshold,
    )

    return result
