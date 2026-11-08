"""
EBSDDataset — Unified wrapper for orix.crystal_map.CrystalMap.

This module provides a thin adapter that normalizes quality metrics across different
indexing methods (Hough, Dictionary, Spherical, h5oina) and provides convenience methods.

Key features:
- Unified quality metric access (bc/pq/scores → normalized quality)
- EDX channel detection and access
- Phase/symmetry shortcuts
- Orientation → Orientation with symmetry conversion
- Indexed pixel masking

Reference: the MTEX-equivalence design notes §1.2
"""

import copy
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, TYPE_CHECKING
import logging
import uuid
import numpy as np

if TYPE_CHECKING:
    from analysis.grain_cells import Cells

logger = logging.getLogger(__name__)

try:
    from orix.crystal_map import CrystalMap, Phase
    from orix.quaternion import Orientation, Rotation
    ORIX_AVAILABLE = True
except ImportError:
    ORIX_AVAILABLE = False
    CrystalMap = None
    Phase = None
    Orientation = None
    Rotation = None


# The one wording for "this map has no band contrast", shared by the Excel export, the batch and the
# BC routes. It names the cause; it does not claim anything about the file's format.
NO_BAND_CONTRAST = "no band contrast in this map"


def _reduce_to_best_match(xmap: 'CrystalMap', rpp: int) -> 'CrystalMap':
    """Reduce a multi-match CrystalMap (keep_n > 1) to best match only.

    Dictionary indexing with keep_n=N stores N orientations per pixel.
    This creates a new CrystalMap with only the best (first) orientation,
    which is what all downstream analysis expects.

    Parameters
    ----------
    xmap : CrystalMap
        Multi-match crystal map with rotations_per_point > 1.
    rpp : int
        Rotations per point (keep_n value).

    Returns
    -------
    CrystalMap
        Single-match crystal map (1 orientation per pixel).
    """
    # Take the best (first) rotation per pixel.
    # xmap.rotations.data shape is (n_pixels, rpp, 4) for quaternions
    rot_data = xmap.rotations.data
    if rot_data.ndim == 3:
        # (n_pixels, rpp, 4) → take first match per pixel
        best_rotations = Rotation(rot_data[:, 0, :])
    elif rot_data.ndim == 2 and rot_data.shape[0] == xmap.size * rpp:
        # Flat layout: (n_pixels * rpp, 4) → take every rpp-th
        best_rotations = Rotation(rot_data[::rpp])
    else:
        # Unknown layout, return as-is
        logger.warning("Unexpected rotation shape %s with rpp=%d, skipping reduce", rot_data.shape, rpp)
        return xmap

    n_pixels = best_rotations.shape[0]

    # Build new CrystalMap with single orientations
    new_xmap = CrystalMap(
        rotations=best_rotations,
        phase_id=xmap.phase_id[:n_pixels] if len(xmap.phase_id) > n_pixels else xmap.phase_id,
        x=xmap.x[:n_pixels] if hasattr(xmap, 'x') and xmap.x is not None else None,
        y=xmap.y[:n_pixels] if hasattr(xmap, 'y') and xmap.y is not None else None,
        phase_list=xmap.phases,
    )

    # Copy properties, reducing multi-match dims
    if hasattr(xmap, '_prop'):
        for key in list(xmap._prop.keys()):
            try:
                val = xmap.prop[key]
            except Exception as e:
                logger.debug("Skipping property %s during reduce: %s", key, e)
                continue
            if val.ndim == 2 and val.shape[0] == n_pixels:
                # (n_pixels, keep_n) → take best match column
                new_xmap.prop[key] = val[:, 0]
            elif val.ndim == 1 and len(val) == n_pixels:
                new_xmap.prop[key] = val
            # Skip unexpected shapes

    return new_xmap


@dataclass
class GrainSet:
    """
    Grain reconstruction results.

    Attributes:
        grain_ids: Per-pixel grain labels (same shape as EBSD map), 0 = not in a grain
        n_grains: Total number of grains (excluding grain_id=0)

        Per-grain properties (arrays of length n_grains):
        mean_orientations: Grain-average orientations (orix.Orientation), from measurements only
        grain_size: Number of measured pixels per grain
        area: Grain area in µm², the area of the grain's cells * step_size² (covered not-indexed
            pixels inside a grain belong to the cells around them)
        ecd: Equivalent circular diameter = 2 * sqrt(area / π) in µm
        perimeter: Length of the grain's outer boundary loop in µm
        perimeter_with_inclusions: Length of all boundary loops of the grain (holes included) in µm
        aspect_ratio: Long over short principal axis of the grain shape (MTEX aspectRatio), >= 1
        gos: Grain orientation spread in radians, from measurements only

        Grain boundary info (cell edges between measurements of different grains of one phase,
        one entry per pair of measurements):
        boundary_segments: Unused, kept for compatibility; the boundary data lives in boundary_type
            and boundary_length
        boundary_type: Per-segment disorientation angle in degrees
        boundary_length: Per-segment length in µm, same order as boundary_type
        boundary_pairs: (K, 2) flat pixel indices a < b of the two measurements of each segment,
            same order as boundary_type and boundary_length

        cells: The exact cells of the last segmentation pass (None for a GrainSet built by hand)

    Reference: the MTEX-equivalence design notes §2.1
    """
    grain_ids: np.ndarray
    n_grains: int

    # Per-grain properties (length n_grains)
    mean_orientations: Optional['Orientation'] = None
    grain_size: Optional[np.ndarray] = None
    area: Optional[np.ndarray] = None
    ecd: Optional[np.ndarray] = None
    perimeter: Optional[np.ndarray] = None
    perimeter_with_inclusions: Optional[np.ndarray] = None
    aspect_ratio: Optional[np.ndarray] = None
    gos: Optional[np.ndarray] = None

    # Reconstruction details; None for GrainSets built elsewhere
    n_closed: Optional[np.ndarray] = None         # closed pixels per grain
    closed_mask: Optional[np.ndarray] = None      # (ny, nx) in a grain, not a measurement of it
    culled_mask: Optional[np.ndarray] = None      # (ny, nx) measurements dropped by min_pixels
    dropped_mask: Optional[np.ndarray] = None     # (ny, nx) the part of culled_mask dropped after the second pass
    cells: Optional['Cells'] = None               # cells of the last segmentation pass (analysis/grain_cells.py)
    grain_phase: Optional[np.ndarray] = None      # (n_grains,) phase id per grain
    alpha: Optional[float] = None                 # closing radius used, in pixels
    filled_rotations: Optional[np.ndarray] = None # (ny, nx, 4) fill result, NaN where not filled
    filled_mask: Optional[np.ndarray] = None      # (ny, nx) pixels that received a filled orientation

    # Grain boundary info
    boundary_segments: list = field(default_factory=list)
    boundary_type: Optional[np.ndarray] = None
    boundary_length: Optional[np.ndarray] = None   # um per entry of boundary_type, same order
    boundary_pairs: Optional[np.ndarray] = None    # (K, 2) flat pixel indices a < b per entry, same order

    def __post_init__(self):
        """Validate grain_ids shape."""
        if not isinstance(self.grain_ids, np.ndarray):
            raise TypeError(f"grain_ids must be np.ndarray, got {type(self.grain_ids)}")
        if self.grain_ids.ndim != 2:
            raise ValueError(f"grain_ids must be 2D, got shape {self.grain_ids.shape}")


class EBSDDataset:
    """
    Unified interface for indexed EBSD data, independent of indexing method.

    Wraps orix.crystal_map.CrystalMap and normalizes quality metric names.
    Provides convenience methods for common operations.

    Supported sources:
        - Hough indexing (kikuchipy)
        - Dictionary indexing (kikuchipy)
        - Spherical indexing (EMSphinx via IndexEBSD)
        - Loaded from .h5oina, .ang, .ctf (orix.io.load)

    Attributes:
        xmap: Original orix CrystalMap
        step_size: Pixel step size in µm
        source: Indexing method identifier
        dataset_token: uuid4 hex, fixed for the life of this object (``GET /api/analysis/status`` reports it)
        grains: GrainSet from grain reconstruction (None until reconstructed)
        grain_recon_params: What the grains were reconstructed with (threshold, min pixels, alpha,
            min intercept), set by the reconstruct route; None whenever ``grains`` is None or was
            made elsewhere

    Reference: the MTEX-equivalence design notes §1.1, §1.2
    """

    def __init__(self, xmap: 'CrystalMap', step_size: float, source: str = "unknown"):
        """
        Initialize EBSDDataset.

        Args:
            xmap: orix CrystalMap from indexing or file load
            step_size: Pixel step size in µm
            source: Indexing method ('hough', 'dictionary', 'spherical', 'h5oina', 'ang', 'ctf')
        """
        if not ORIX_AVAILABLE:
            raise ImportError("orix is required for EBSDDataset. Install with: pip install orix")

        if not isinstance(xmap, CrystalMap):
            raise TypeError(f"xmap must be orix.CrystalMap, got {type(xmap)}")

        # Dictionary indexing with keep_n > 1 stores multiple orientations per pixel.
        # Reduce to best match (index 0) so all downstream analysis sees 1 orientation/pixel.
        rpp = getattr(xmap, 'rotations_per_point', 1)
        if rpp > 1:
            xmap = _reduce_to_best_match(xmap, rpp)

        # Monoclinic phases must reduce their orientations with the group whose
        # two-fold axis is the crystal's own (see crystal_symmetry).
        from crystal_symmetry import frame_xmap
        self.xmap = frame_xmap(xmap)
        self.step_size = step_size
        self.source = source
        self.grains: Optional[GrainSet] = None
        self.grain_recon_params: Optional[dict] = None
        # Identity of this map for a page that must notice a swap: a new EBSDDataset (a load) gets a new
        # token, the same object (stashed and restored on a file switch) keeps its own.
        self.dataset_token: str = uuid.uuid4().hex
        # The grid this map sits on (a ScanGeometry dict). None = unknown, never "square": set by
        # whoever knows it (the result a dataset was made from), copied by every derived dataset.
        self.scan_geometry: Optional[dict] = None

    # ─── Shape & Dimensions ──────────────────────────────────────────────────

    @property
    def nx(self) -> int:
        """Number of pixels in x direction (columns).

        Falls back to 1 for 1D line-scan xmaps (shape == (n,)) so downstream
        callers don't IndexError when running analysis on a line scan.
        """
        sh = self.xmap.shape
        return int(sh[1]) if len(sh) >= 2 else 1

    @property
    def ny(self) -> int:
        """Number of pixels in y direction (rows)."""
        sh = self.xmap.shape
        return int(sh[0]) if len(sh) >= 1 else 1

    @property
    def shape(self) -> Tuple[int, int]:
        """Map shape as (ny, nx). Always 2-tuple even for 1D line scans."""
        sh = self.xmap.shape
        if len(sh) >= 2:
            return (int(sh[0]), int(sh[1]))
        if len(sh) == 1:
            return (int(sh[0]), 1)
        return (1, 1)

    @property
    def size(self) -> int:
        """Total number of pixels."""
        return self.xmap.size

    # ─── Quality Metrics (Normalized) ────────────────────────────────────────

    @property
    def quality(self) -> np.ndarray:
        """
        Normalized quality metric (0-1 range), regardless of indexing method.

        Maps to: bc (h5oina), pq (Hough), scores (Dictionary).
        Returns array of shape (size,). Returns ones if no quality data
        is present in xmap.prop or if the array is empty (avoids
        "zero-size array to reduction operation" crash on degenerate
        xmaps).
        """
        for key in (self._band_contrast_key(), 'pq'):
            if key is not None and key in self.xmap.prop:
                arr = self.xmap.prop[key]
                if arr.size == 0:
                    continue
                amax = float(arr.max())
                return arr / amax if amax > 0 else arr
        if 'scores' in self.xmap.prop:
            scores = self.xmap.prop['scores']
            # Dictionary indexing stores (n_pixels, keep_n) — take best match only
            return scores[:, 0] if scores.ndim > 1 else scores
        # Fallback: all ones
        return np.ones(self.xmap.size)

    @property
    def bc(self) -> np.ndarray:
        """
        Band contrast or equivalent quality metric as raw values.

        Tries: bc (any capitalisation, orix keeps the .ctf column as "BC") → pq → scores → fallback (128).
        Returns array of shape (size,).
        """
        native = self._band_contrast_key()
        for key in ([native] if native is not None else []) + ['pq', 'scores']:
            if key in self.xmap.prop:
                val = self.xmap.prop[key]
                # Dictionary indexing stores scores as (n_pixels, keep_n) — take best match
                return val[:, 0] if val.ndim > 1 else val
        # Fallback: mid-range gray value
        return np.ones(self.xmap.size) * 128

    def _band_contrast_key(self) -> Optional[str]:
        """The key of the band-contrast array in ``xmap.prop`` (any capitalisation, one value per pixel), or None.

        Oxford ``.ctf`` files reach orix with the column under ``"BC"``; H5OINA and Orienta's own exports use
        ``"bc"``.
        """
        prop = getattr(self.xmap, "prop", {}) or {}
        found = []
        try:
            for key in prop:
                if str(key).lower() == "bc" and np.asarray(prop[key]).size == self.xmap.size:
                    found.append(key)
        except Exception:
            return None
        if not found:
            return None
        # not dependent on the order of the properties: an exact "bc" first, then the other spellings sorted
        return "bc" if "bc" in found else sorted(found, key=str)[0]

    def band_contrast_raw(self) -> Optional[np.ndarray]:
        """The native band-contrast array as stored (one value per pixel, any spelling of the key), or None."""
        key = self._band_contrast_key()
        return np.asarray(self.xmap.prop[key]) if key is not None else None

    @property
    def has_native_bc(self) -> bool:
        """True only when the xmap carries a real native Band Contrast array.
        pq / scores are NOT native BC — the BC-calibrated GMM/quality-filter
        must fail loud rather than run on those.

        INVARIANT: xmap.prop['bc'] (or 'BC' from a .ctf) must only ever hold REAL native Band
        Contrast (e.g. bridged from an Oxford H5OINA source file), never a
        computed/surrogate value (CI, scores, FFT image quality). This property
        trusts prop['bc'] as native — writing a surrogate there would silently
        make the BC-only analyses run on fake data."""
        return self._band_contrast_key() is not None

    @property
    def bc_2d(self) -> np.ndarray:
        """Band contrast reshaped to 2D map (ny, nx)."""
        return self.bc.reshape(self.shape)

    # ─── Orientations & Phase ────────────────────────────────────────────────

    @property
    def orientations(self) -> 'Orientation':
        """
        All orientations as orix.Orientation (with crystal symmetry).

        Converts xmap.rotations to Orientation using the primary phase symmetry.
        Returns Orientation of shape (size,).
        """
        return Orientation(self.xmap.rotations, symmetry=self.phase.point_group)

    @property
    def orientations_2d(self) -> 'Orientation':
        """Orientations reshaped to 2D map (ny, nx)."""
        ori_flat = self.orientations
        # orix Orientation reshaping
        return ori_flat.reshape(*self.shape)

    @property
    def phase(self) -> 'Phase':
        """
        Primary phase.

        Returns the first indexed phase (phase_id >= 0).
        """
        # xmap.phases.ids can be a list or ndarray
        phase_ids = np.array(self.xmap.phases.ids)
        valid_ids = phase_ids[phase_ids >= 0]
        if len(valid_ids) == 0:
            raise ValueError("No indexed phases found in xmap")
        from crystal_symmetry import frame_phase
        return frame_phase(self.xmap.phases[int(valid_ids[0])])

    @property
    def indexed_mask(self) -> np.ndarray:
        """Boolean mask: True where indexed. Shape (size,)."""
        return self.xmap.phase_id >= 0

    @property
    def indexed_mask_2d(self) -> np.ndarray:
        """Boolean mask: True where indexed. Shape (ny, nx)."""
        return self.indexed_mask.reshape(self.shape)

    # ─── EDX Channels ────────────────────────────────────────────────────────

    @property
    def edx_channels(self) -> Dict[str, np.ndarray]:
        """
        Available EDX channels as {element_name: np.ndarray}.

        Scans xmap.prop for keys containing common element names.
        Returns dict with arrays of shape (size,).
        """
        edx = {}
        element_keywords = [
            'Fe', 'Si', 'Mn', 'Mg', 'Cu', 'Zn', 'Al', 'Cr', 'Sn', 'Ag',
            'Ni', 'Ti', 'V', 'C', 'O', 'N', 'S', 'P'
        ]
        for key, val in self.xmap.prop.items():
            if any(elem in key for elem in element_keywords):
                edx[key] = val
        return edx

    def has_edx_element(self, element: str) -> bool:
        """Check if EDX data contains a specific element (case-insensitive)."""
        return any(element.lower() in key.lower() for key in self.edx_channels.keys())

    @property
    def phase_names(self) -> list:
        """List of phase names from CrystalMap."""
        try:
            return list(self.xmap.phases.names)
        except Exception as e:
            logger.debug("Could not read phase names: %s", e)
            return ["unknown"]

    # ─── Convenience Methods ─────────────────────────────────────────────────

    def __repr__(self) -> str:
        """String representation."""
        indexed_count = np.sum(self.indexed_mask)
        indexed_pct = (indexed_count / self.size) * 100 if self.size > 0 else 0

        edx_str = ""
        if self.edx_channels:
            edx_str = f", EDX: {list(self.edx_channels.keys())}"

        grains_str = ""
        if self.grains is not None:
            grains_str = f", {self.grains.n_grains} grains"

        return (
            f"EBSDDataset(source={self.source}, "
            f"shape={self.shape}, step={self.step_size}µm, "
            f"indexed={indexed_pct:.1f}%{edx_str}{grains_str})"
        )

    def crop_to_indexed(self) -> 'EBSDDataset':
        """
        Return new EBSDDataset containing only indexed pixels.

        Useful for removing non-indexed regions before analysis.
        """
        xmap_indexed = self.xmap[self.indexed_mask]
        cropped = EBSDDataset(xmap_indexed, self.step_size, self.source)
        cropped.scan_geometry = copy.deepcopy(self.scan_geometry)      # same grid, its own copy
        return cropped

    # ─── Quality Filter ──────────────────────────────────────────────────────

    def apply_quality_filter(
        self,
        bc_min: float = 0,
        bands_min: int = 0,
    ) -> dict:
        """Mark low-quality pixels as unindexed so downstream stats skip them.

        Reads ``bc`` and ``bands`` from xmap.prop (carried in from the source
        h5oina by export_result_h5_light) and sets ``phase_id = -1`` wherever
        a pixel falls below either threshold. Grain reconstruction, KAM, GOS,
        texture components and Excel sheets all already exclude unindexed
        pixels, so this single mutation removes them from every statistic
        without touching the analysis modules themselves.

        The mutation is destructive on the in-memory xmap. Reload the source
        file to reset. Resetting via ``apply_quality_filter(0, 0)`` is a
        no-op — it cannot resurrect pixels that were already unindexed
        before the filter ran.

        Parameters
        ----------
        bc_min : float
            Minimum Band Contrast (0..255). Pixels with bc < bc_min get
            unindexed. ``0`` disables this criterion.
        bands_min : int
            Minimum number of detected Hough bands. Pixels with bands <
            bands_min get unindexed. ``0`` disables this criterion.

        Returns
        -------
        dict with keys:
            indexed_before : int
            indexed_after : int
            filtered_by_bc : int
            filtered_by_bands : int
            filtered_total : int
            filtered_fraction : float (0..1)
        """
        bc = self.band_contrast_raw()
        bands = self.xmap.prop.get("bands")

        before = int(np.sum(self.xmap.phase_id >= 0))

        bad_bc = (
            (np.asarray(bc).ravel() < bc_min)
            if bc is not None and bc_min > 0
            else np.zeros(self.size, dtype=bool)
        )
        bad_bands = (
            (np.asarray(bands).ravel() < bands_min)
            if bands is not None and bands_min > 0
            else np.zeros(self.size, dtype=bool)
        )

        # Pixel is rejected if EITHER criterion fails — Aztec convention.
        bad = bad_bc | bad_bands

        # Honour pre-existing unindexed (don't count those as newly filtered).
        already_unindexed = self.xmap.phase_id < 0
        newly_filtered = bad & ~already_unindexed

        # Mutate phase_id directly via _phase_id. orix's public setter does
        # `if value == -1` which raises on ndarray inputs (orix bug — works
        # only for scalars). Going through the private attribute side-steps
        # that. We also have to register the "not_indexed" phase manually
        # since the public setter would normally do it.
        new_pid = np.asarray(self.xmap.phase_id).copy()
        new_pid[newly_filtered] = -1
        try:
            if "not_indexed" not in self.xmap.phases.names:
                self.xmap.phases.add_not_indexed()
            self.xmap._phase_id[self.xmap.is_in_data] = new_pid
        except Exception as e:
            raise RuntimeError(
                f"orix refused phase_id mutation ({e}). Reload the file and try again."
            ) from e

        # Invalidate cached grain reconstruction — a quality filter changes
        # which pixels participate, so existing grains are stale.
        self.grains = None
        self.grain_recon_params = None

        after = int(np.sum(self.xmap.phase_id >= 0))
        return {
            "indexed_before": before,
            "indexed_after": after,
            "filtered_by_bc": int(np.sum(bad_bc & ~already_unindexed)),
            "filtered_by_bands": int(np.sum(bad_bands & ~already_unindexed)),
            "filtered_total": int(np.sum(newly_filtered)),
            "filtered_fraction": float(np.sum(newly_filtered) / max(self.size, 1)),
        }

    def get_phase_mask(self, phase_id: int) -> np.ndarray:
        """Get boolean mask for specific phase. Shape (size,)."""
        return self.xmap.phase_id == phase_id

    def get_phase_mask_2d(self, phase_id: int) -> np.ndarray:
        """Get boolean mask for specific phase. Shape (ny, nx)."""
        return self.get_phase_mask(phase_id).reshape(self.shape)
