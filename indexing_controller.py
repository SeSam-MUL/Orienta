"""
Indexing Controller - Manages Dictionary, Hough, and Spherical indexing workflows.

Provides:
- Pixel selection (full image, region, chemistry-based via EDS)
- Dictionary Indexing via kikuchipy
- Hough Indexing via kikuchipy (wraps existing ebsd_utils)
- Partial indexing with result back-mapping to original grid
- Integration with Phase Map Generator (FEAT-12)

CRITICAL: When indexing a subset of pixels, results MUST be correctly
mapped back to their original positions in the full scan grid.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.api.log_broadcast import timed_step

logger = logging.getLogger(__name__)


def _release_cuda_cache() -> None:
    """Free GPU tensors that are still in PyTorch's caching allocator.

    Called from ``finally`` blocks around every Dictionary-GPU and
    Spherical-GPU run so the next run starts with the full free VRAM
    budget reported by ``torch.cuda.mem_get_info``. Without this, a
    dict-indexing run leaves its ~8 GB pattern tensor cached and a
    subsequent Spherical-GPU run sees free_memory=0 → batch=1 → 0 pat/s.
    Safe to call when CUDA is not available — silent no-op.
    """
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("torch.cuda.empty_cache() failed: %s", exc)


class IndexingMethod(Enum):
    """Available indexing methods."""
    HOUGH = "hough"
    DICTIONARY = "dictionary"
    SPHERICAL = "spherical"
    EMBEDDING = "embedding"


class PixelSelectionMode(Enum):
    """How pixels are selected for indexing."""
    FULL = "full"           # Index all pixels
    REGION = "region"       # Rectangular region (row_start:row_end, col_start:col_end)
    MASK = "mask"           # Boolean mask (from EDS chemistry or manual selection)


@dataclass
class IndexingConfig:
    """Configuration for an indexing job."""
    method: IndexingMethod = IndexingMethod.HOUGH
    selection_mode: PixelSelectionMode = PixelSelectionMode.FULL

    # Hough-specific
    n_bands: int = 12
    t_sigma: float = 2.0
    r_sigma: float = 2.0

    # Dictionary-specific
    metric: str = 'ncc'     # 'ncc' or 'ndp'
    keep_n: int = 20        # Top N matches to keep
    n_per_iteration: Optional[int] = None  # Memory control
    # Orientation sampling step (deg) when a raw master must be projected into
    # a dictionary. Used by the CPU path (and matches the GPU Path-A default).
    angular_step_deg: float = 1.5

    # GPU compute mode for dictionary indexing — see backend/dict_gpu/
    # "auto" → use GPU if a CUDA device is detected, else CPU
    # "gpu"  → require GPU; raise GpuDictError if no CUDA detected
    # "cpu"  → force the existing kikuchipy CPU path
    compute_mode: str = "auto"

    # Spherical-specific (EMSphinx)
    bandwidth: int = 88     # Spherical harmonic bandwidth
    normed: bool = True
    refine: bool = True
    sht_file: str = ""      # Path to .sht master pattern file
    nregions: int = 10      # Adaptive histogram equalization regions
    circmask: int = -1      # Circular mask: -1=off, 0=inscribed, >0=radius in px
    gausbckg: bool = True    # Subtract 2D Gaussian background (matches EMSphInx default + oracle generator)
    cc_fp64: bool = False   # FP64 cross-correlation volume (closes ~10pp alt-oracle gap)
    # Spherical backend selector (Phase 5):
    #   "emsphinx"      - the WSL EMSphInx CPU CLI (the original implementation)
    #   "spherical_gpu" - the in-process PyTorch GPU pipeline added in this branch
    # The GPU path is ~11x faster on identical data and is the recommended
    # default once validated on the user's hardware.
    backend: str = "emsphinx"

    # Region selection (used when selection_mode == REGION)
    row_start: int = 0
    row_end: int = -1
    col_start: int = 0
    col_end: int = -1

    # EDS chemistry prior (per-phase, keyed by phase file path). Empty => off.
    eds_phase_strengths: Dict[str, float] = field(default_factory=dict)
    eds_expected_overrides: Dict[str, dict] = field(default_factory=dict)


@dataclass
class IndexingResult:
    """Result of an indexing operation with back-mapping info."""
    xmap: object  # orix CrystalMap
    selection_mask: np.ndarray   # Boolean mask: True = indexed
    original_shape: Tuple[int, int]  # (n_rows, n_cols)
    method: IndexingMethod
    confidence_scores: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)
    eds_counts: Optional[Dict] = None           # {'counts': {elem: 1D array}, 'n_rows', 'n_cols'}
    phase_compositions: Optional[Dict] = None   # {'PhaseName': {'Fe': 82.0, 'O': 3.0}, ...}


@dataclass
class PhaseConfig:
    """Configuration for a single phase in multi-phase comparison."""
    name: str                          # Display name (e.g. "Fe-bcc")
    cif_path: str = ""                 # For Hough indexing
    master_h5_path: str = ""           # For Dictionary indexing
    sht_path: str = ""                 # For Spherical indexing
    phase_list: object = None          # orix PhaseList (loaded from CIF)
    dictionary: object = None          # kikuchipy EBSD signal (for dictionary indexing)
    color: str = ""                    # Display color (auto-assigned if empty)


@dataclass
class ComparisonConfig:
    """Configuration for a multi-phase multi-method comparison batch."""
    phases: List[PhaseConfig]
    methods: List[IndexingMethod]      # Subset of [HOUGH, DICTIONARY, SPHERICAL]
    selection_mode: PixelSelectionMode = PixelSelectionMode.FULL

    # Shared parameters (same across phases for fair comparison)
    n_bands: int = 12
    t_sigma: float = 2.0
    r_sigma: float = 2.0
    metric: str = 'ncc'
    keep_n: int = 20
    bandwidth: int = 88
    normed: bool = True
    refine: bool = True
    nregions: int = 10
    circmask: int = -1
    gausbckg: bool = True   # matches EMSphInx default + oracle generator (mirrors IndexingConfig)
    # Phase 5: spherical backend selector — same field as IndexingConfig.
    # Carried through ComparisonConfig so multi-phase runs honour the
    # GPU/CPU choice. Default "emsphinx" preserves legacy behaviour for
    # callers that don't set it explicitly.
    backend: str = "emsphinx"

    # Region selection
    row_start: int = 0
    row_end: int = -1
    col_start: int = 0
    col_end: int = -1


@dataclass
class PhaseMethodResult:
    """Result for one (phase, method) combination."""
    phase_name: str
    method: IndexingMethod
    indexing_result: IndexingResult     # Full result with xmap, scores, etc.
    mean_score: float                  # Mean confidence/NCC/SCC over indexed pixels
    score_map_2d: Optional[np.ndarray] = None  # 2D score map (n_rows, n_cols)


@dataclass
class ComparisonResult:
    """Aggregated result of a multi-phase multi-method batch."""
    results: List[PhaseMethodResult]   # All (phase, method) results
    phases: List[PhaseConfig]
    methods: List[IndexingMethod]
    original_shape: Tuple[int, int]
    selection_mask: Optional[np.ndarray] = None
    best_phase_per_pixel: Optional[np.ndarray] = None  # (n_rows, n_cols) phase index
    consensus_map: Optional[np.ndarray] = None  # (n_rows, n_cols) majority-vote phase
    eds_counts: Optional[Dict] = None
    phase_compositions: Optional[Dict] = None
    n_eds_adjusted: int = 0  # pixels whose winner flipped due to EDS phase_weights


def discover_files_for_method(
    method: IndexingMethod,
    material_hint: str = "",
    current_pc: Optional[list] = None,
) -> Dict:
    """Scan Database/ for files matching the indexing method.

    Parameters
    ----------
    method : IndexingMethod
        Which indexing method to find files for.
    material_hint : str
        Material name to prioritize in results (e.g. "Fe", "Al").

    Returns
    -------
    list of dict
        Each dict has keys: path (Path), filename (str), material (str).
        Sorted with material_hint matches first.
    """
    from path_utils import get_local_database_path, DATABASE_SUBFOLDERS

    db_root = get_local_database_path()
    results: List[Dict] = []

    def _scan_folder(base: Path, extensions: set, name_filter=None):
        """Scan a folder and its material subfolders for matching files."""
        found = []
        if not base.is_dir():
            return found
        try:
            entries = list(base.iterdir())
        except (OSError, PermissionError) as e:
            logger.warning(f"Error listing {base}: {e}")
            return found
        for f in entries:
            try:
                if f.is_file() and f.suffix.lower() in extensions:
                    if name_filter and name_filter not in f.name.lower():
                        continue
                    found.append({"path": f, "filename": f.name, "material": "-"})
                elif f.is_dir():
                    material = f.name
                    for sub_f in f.iterdir():
                        if sub_f.is_file() and sub_f.suffix.lower() in extensions:
                            if name_filter and name_filter not in sub_f.name.lower():
                                continue
                            found.append({"path": sub_f, "filename": sub_f.name, "material": material})
            except (OSError, PermissionError) as e:
                logger.warning(f"Error scanning {f}: {e}")
        return found

    if method == IndexingMethod.HOUGH:
        # CIF files for Hough indexing
        results = _scan_folder(db_root / DATABASE_SUBFOLDERS["cif_library"], {".cif"})

    elif method == IndexingMethod.DICTIONARY:
        # 1. Pre-generated dictionaries from Dictionary_Library
        dict_lib = db_root / DATABASE_SUBFOLDERS["dictionary_library"]
        results = _scan_folder(dict_lib, {".h5"})
        for entry in results:
            entry["file_type"] = "dictionary"

        # 2. Master patterns from H5 Cache (can generate dictionary from these)
        h5_cache = db_root / DATABASE_SUBFOLDERS["h5_cache"]
        masters = _scan_folder(h5_cache, {".h5"}, name_filter="_master")
        # Exclude dict files whose names also contain "_master_" in their stem
        # (e.g. "Al_master_E20kV_npx500_dict_20kV_128x156_2.0deg.h5")
        masters = [m for m in masters if "_dict_" not in m["filename"].lower()]
        for entry in masters:
            entry["file_type"] = "master"
        results.extend(masters)

    elif method == IndexingMethod.SPHERICAL:
        # SHT files for Spherical indexing
        results = _scan_folder(db_root / DATABASE_SUBFOLDERS["sht_database"], {".sht"})

    else:
        return {"files": [], "groups": []}

    # Sort: material_hint matches first, dictionaries before masters, then alpha
    hint_lower = material_hint.lower()

    def sort_key(entry):
        mat_match = 0 if hint_lower and entry["material"].lower() == hint_lower else 1
        ftype = entry.get("file_type", "")
        type_order = 0 if ftype == "dictionary" else (1 if ftype == "master" else 2)
        return (mat_match, type_order, entry["material"], entry["filename"])

    results.sort(key=sort_key)

    # Enrich dictionary entries with JSON sidecar metadata (energy, PC, shape, etc.)
    for entry in results:
        if entry.get("file_type") == "dictionary":
            json_path = entry["path"].with_suffix(".json")
            if json_path.is_file():
                try:
                    import json as _json
                    meta = _json.loads(json_path.read_text(encoding="utf-8"))
                    entry["energy_kv"] = meta.get("energy_kv", 0)
                    entry["pc"] = meta.get("pc", [])
                    entry["detector_shape"] = meta.get("detector_shape", [])
                    entry["resolution_deg"] = meta.get("resolution_deg", 0)
                    entry["n_orientations"] = meta.get("n_orientations", 0)
                    # None, not 70: "unknown" and "explicitly 70 deg" must not
                    # look the same to the UI's geometry check.
                    entry["sample_tilt"] = meta.get("sample_tilt")
                    # Camera geometry. Missing on older sidecars — surfaced as
                    # None (unknown) rather than 0, because "unknown" and
                    # "explicitly flat" must not be confused when the UI decides
                    # whether a dictionary fits the current detector.
                    entry["detector_tilt"] = meta.get("detector_tilt")
                    entry["azimuthal"] = meta.get("azimuthal")
                    if meta.get("material"):
                        entry["composition"] = meta["material"]
                except Exception:
                    pass

    # Calculate PC deviation for dictionary entries
    if current_pc and method == IndexingMethod.DICTIONARY:
        try:
            from backend.api.services.pc_utils import pc_deviation_percent, pc_match_color
        except ImportError:
            try:
                import sys as _sys
                _sys.path.insert(0, str(Path(__file__).parent / "backend" / "api" / "services"))
                from pc_utils import pc_deviation_percent, pc_match_color
            except ImportError:
                pc_deviation_percent = None
                pc_match_color = None
        if pc_deviation_percent:
            for entry in results:
                dict_pc = entry.get("pc")
                if dict_pc:
                    delta = pc_deviation_percent(current_pc, dict_pc)
                    entry["pc_delta_percent"] = delta
                    entry["pc_match"] = pc_match_color(delta)
                else:
                    entry["pc_delta_percent"] = None
                    entry["pc_match"] = "unknown"

    # Enrich results with phase metadata (chemical formula, space group, etc.)
    try:
        from phase_metadata import (
            get_phase_metadata, extract_elements, compute_element_group,
            get_crystal_system, composition_for_cif,
        )
        cif_dir = db_root / DATABASE_SUBFOLDERS["cif_library"]
        for entry in results:
            meta = get_phase_metadata(entry["path"], cif_library_dir=cif_dir)
            # display_label = the Crystal-Database composition (pymatgen
            # reduced_formula + subscripts, cached) so every picker shows the
            # SAME nice formula as the Crystal Database table. Crystal system is
            # sent separately and shown alongside.
            comp = composition_for_cif(meta.cif_path) if meta.cif_path else ""
            entry["display_label"] = comp or meta.display_label
            entry["formula"] = meta.formula
            entry["space_group"] = meta.space_group
            entry["pearson"] = meta.pearson
            entry["phase_name"] = meta.phase_name
            entry["metadata_source"] = meta.source
            entry["crystal_system"] = get_crystal_system(meta.space_group) if meta.space_group else ""
            # Degeneracy-detection fields (consumed by the frontend
            # phase-degeneracy detector).
            entry["lattice_a"] = meta.lattice_a
            entry["lattice_b"] = meta.lattice_b
            entry["lattice_c"] = meta.lattice_c
            entry["lattice_alpha"] = meta.lattice_alpha
            entry["lattice_beta"] = meta.lattice_beta
            entry["lattice_gamma"] = meta.lattice_gamma
            entry["space_group_number"] = meta.space_group_number
            entry["laue_class"] = meta.laue_class
            entry["centering"] = meta.centering
            # Element grouping
            elements = extract_elements(meta.formula)
            entry["elements"] = elements
            entry["element_group"] = compute_element_group(elements)
    except Exception:
        for entry in results:
            if "display_label" not in entry:
                entry["display_label"] = entry["filename"]
            entry.setdefault("element_group", "Sonstiges")

    # Build sorted group list for frontend
    groups_seen = []
    for entry in results:
        g = entry.get("element_group", "Sonstiges")
        if g not in groups_seen:
            groups_seen.append(g)

    def _group_sort_key(g):
        if g == "Reine Elemente":
            return (0, g)
        if g == "Sonstiges":
            return (2, g)
        return (1, g)
    groups_seen.sort(key=_group_sort_key)

    return {"files": results, "groups": groups_seen}


def find_matching_dicts(
    master_path: str,
    signal_shape: Optional[Tuple[int, int]] = None,
    material_hint: str = "",
) -> List[Dict]:
    """Find dictionary files that match a given master pattern.

    Matching criteria:
    1. Filename contains the master stem (e.g. "Al_master_E20kV_npx500")
    2. If signal_shape provided: filename contains "{rows}x{cols}" pattern

    Parameters
    ----------
    master_path : str
        Path to the master pattern H5 file.
    signal_shape : tuple of (rows, cols), optional
        EBSD detector dimensions to filter by (e.g. (128, 156)).
    material_hint : str
        Material name for sorting.

    Returns
    -------
    list of dict
        Each dict has: path, filename, material, file_type='dictionary'
    """
    from pathlib import Path as _Path

    master_stem = _Path(master_path).stem  # e.g. "Al_master_E20kV_npx500"

    # Get all dictionary files
    all_dicts = discover_files_for_method(IndexingMethod.DICTIONARY, material_hint)
    dict_files = [e for e in all_dicts if e.get("file_type") == "dictionary"]

    # Filter: filename must contain master_stem
    matching = [e for e in dict_files if master_stem in e["filename"]]

    # Further filter by signal shape if provided
    if signal_shape is not None and len(matching) > 0:
        rows, cols = signal_shape
        size_tag_1 = f"{rows}x{cols}"
        size_tag_2 = f"{cols}x{rows}"  # some tools reverse dimensions
        shape_matching = [
            e for e in matching
            if size_tag_1 in e["filename"] or size_tag_2 in e["filename"]
        ]
        # Only apply shape filter if it yields results; otherwise show all stem-matches
        if shape_matching:
            matching = shape_matching

    return matching


def create_selection_mask(
    n_rows: int,
    n_cols: int,
    mode: PixelSelectionMode,
    region: Optional[Tuple[int, int, int, int]] = None,
    chemistry_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Create a boolean selection mask for which pixels to index.

    Parameters
    ----------
    n_rows, n_cols : int
        Scan dimensions.
    mode : PixelSelectionMode
        Selection strategy.
    region : tuple, optional
        (row_start, row_end, col_start, col_end) for REGION mode.
    chemistry_mask : np.ndarray, optional
        Pre-computed boolean mask (True = index) for MASK mode.
        Typically from EDS filtering (FEAT-10).

    Returns
    -------
    np.ndarray
        Boolean mask of shape (n_rows, n_cols). True = pixel will be indexed.
    """
    if mode == PixelSelectionMode.FULL:
        return np.ones((n_rows, n_cols), dtype=bool)

    elif mode == PixelSelectionMode.REGION:
        mask = np.zeros((n_rows, n_cols), dtype=bool)
        if region is None:
            raise ValueError("Region must be provided for REGION mode")
        r0, r1, c0, c1 = region
        if r1 == -1:
            r1 = n_rows
        if c1 == -1:
            c1 = n_cols
        r0 = max(0, min(r0, n_rows))
        r1 = max(0, min(r1, n_rows))
        c0 = max(0, min(c0, n_cols))
        c1 = max(0, min(c1, n_cols))
        mask[r0:r1, c0:c1] = True  # Python-standard: exclusive end
        return mask

    elif mode == PixelSelectionMode.MASK:
        if chemistry_mask is None:
            raise ValueError("Chemistry mask must be provided for MASK mode")
        mask = np.asarray(chemistry_mask, dtype=bool)
        if mask.shape != (n_rows, n_cols):
            raise ValueError(
                f"Mask shape {mask.shape} does not match scan shape ({n_rows}, {n_cols})"
            )
        return mask

    raise ValueError(f"Unknown selection mode: {mode}")


class IndexingCancelled(Exception):
    """Raised when the user cancels an indexing job."""
    pass


# Backward-compatible alias used by the new Dict-GPU / Spherical-GPU
# cancel paths. Pointing it at the existing class means the route's
# ``except IndexingCancelled`` block catches every cancel uniformly.
CancelledIndexingError = IndexingCancelled


# Minimum number of patterns before Ray parallel indexing is worthwhile.
# Below this threshold, single-thread with OpenCL GPU is faster due to Ray startup overhead.
# Ray creates N separate OpenCL contexts which fight for GPU — keep threshold high.
RAY_MIN_PATTERNS = 10000

# Max Ray workers — too many workers cause GPU contention and massive startup overhead.
RAY_MAX_WORKERS = 4


def _hough_use_ray(n_selected, pc_rows, ray_available, min_patterns=RAY_MIN_PATTERNS):
    """Decide whether to use PyEBSDIndex's Ray distributed Hough path.

    Ray is used only for large jobs with a SINGLE pattern centre. A per-pixel
    PC (``pc_rows > 1``, e.g. from PC refinement) MUST go single-thread:
    ``index_pats_distributed`` runs an internal ``npats=1`` test-flight and
    per-chunk workers that both feed the full ``(N, 3)`` PC into a smaller slot,
    raising "could not broadcast input array from shape (N,3) into shape (1,3)".
    The single-thread ``index_pats`` handles a per-pixel PC correctly because it
    indexes all N patterns in one call (``npats == N`` == PC rows).
    """
    return bool(ray_available and n_selected >= min_patterns and pc_rows <= 1)


def hough_index_patterns(
    signal,
    phase_list,
    detector,
    config: IndexingConfig,
    selection_mask: Optional[np.ndarray] = None,
    progress_callback=None,
    cancel_check=None,
) -> IndexingResult:
    """Run Hough indexing on selected patterns.

    Parameters
    ----------
    signal : kikuchipy.signals.EBSD
        Full EBSD signal (4D: n_rows, n_cols, height, width).
    phase_list : orix.crystal_map.PhaseList
        Phases to index against.
    detector : kikuchipy.detectors.EBSDDetector
        Detector geometry with PC.
    config : IndexingConfig
        Indexing parameters.
    selection_mask : np.ndarray, optional
        Boolean mask (n_rows, n_cols). None = index all.
    progress_callback : callable, optional
        Called with (message, pct) where pct is a float 0.0-1.0.
    cancel_check : callable, optional
        Returns True if the job was cancelled. Checked between steps.

    Returns
    -------
    IndexingResult
        With xmap and back-mapping information.
    """
    from ebsd_utils import prepare_reflectors, create_indexer

    def _progress(msg, pct=None):
        if progress_callback:
            progress_callback(msg, pct)

    def _check_cancel():
        if cancel_check and cancel_check():
            raise IndexingCancelled("Indexing cancelled by user")

    n_rows, n_cols = signal.data.shape[:2]

    if selection_mask is None:
        selection_mask = np.ones((n_rows, n_cols), dtype=bool)

    n_selected = int(selection_mask.sum())

    # --- Per-pixel PC slicing for partial selections ---
    # When the detector carries a *per-pixel* PC (one PC vector per scan
    # pixel — e.g. Oxford/Aztec h5oina stores the PC drift across the scan,
    # or the user ran pixel-wise PC refinement), the PC array has shape
    # (n_rows*n_cols, 3). For a full-grid index that lines up with the
    # patterns. But for a REGION / MASK selection we extract only
    # ``signal.data[selection_mask]`` (n_selected patterns) below, while the
    # indexer is still built from the full-grid detector — pyebsdindex then
    # tries to broadcast the (n_total, 3) PC onto n_selected patterns and
    # dies with "could not broadcast (N,3) into (M,3)". Slice the PC down to
    # the selected pixels here, in the same C-order the pattern extraction
    # uses (selection_mask.ravel()), so each region pattern keeps its own PC.
    if not selection_mask.all() and detector is not None:
        try:
            pc_arr = np.asarray(detector.pc)
            n_pc = pc_arr.size // 3
            if n_pc == (n_rows * n_cols) and n_pc != n_selected:
                pc_sliced = pc_arr.reshape(-1, 3)[selection_mask.ravel()]
                detector = detector.deepcopy()
                detector.pc = pc_sliced
                logger.info(
                    "Hough: sliced per-pixel PC %s -> %s for partial selection",
                    pc_arr.shape, pc_sliced.shape,
                )
        except Exception:
            # Fail-soft: if anything about the PC shape is unexpected, leave
            # the detector untouched. A single/broadcastable PC indexes fine;
            # only a full-grid per-pixel PC hits the broadcast error above.
            logger.warning("Hough: per-pixel PC slicing skipped", exc_info=True)

    # --- Step 1: Prepare reflectors (20% -> 30%) ---
    _progress(f"Hough: preparing reflectors for {len(list(phase_list)) if phase_list else 0} phase(s)...", 0.22)
    with timed_step("Hough: prepare reflectors"):
        reflectors = prepare_reflectors(phase_list)
    _check_cancel()

    # --- Step 2: Create indexer (30% -> 35%) ---
    _progress(f"Hough: creating indexer (bands={config.n_bands}, tSigma={config.t_sigma}, rSigma={config.r_sigma})...", 0.30)
    with timed_step("Hough: create indexer"):
        indexer = create_indexer(
            detector, phase_list, reflectors,
            nBands=config.n_bands,
            tSigma=config.t_sigma,
            rSigma=config.r_sigma,
        )
    _check_cancel()

    # --- Step 3: Extract patterns (35% -> 40%) ---
    if selection_mask.all():
        with timed_step("Hough: reshape patterns (full grid)"):
            sig_shape = signal.data.shape[2:]
            patterns = signal.data.reshape((-1,) + sig_shape)
            nav_shape = (n_rows, n_cols)
            is_partial = False
    else:
        _progress(f"Hough: extracting {n_selected} selected patterns (out of {n_rows*n_cols})...", 0.35)
        with timed_step(f"Hough: extract {n_selected} patterns"):
            # Reshape to (n_total, H, W) and select by integer positions rather
            # than `signal.data[selection_mask]`: a 2-D boolean mask indexing a
            # 4-D LAZY (dask) array is mishandled by dask (it treats the mask's
            # total size as a 1-D index over axis 0 → "Boolean array with size
            # n_rows*n_cols is not long enough for axis 0 with size n_rows").
            # Integer fancy-indexing over the flattened nav axis is dask-safe and
            # preserves row-major order (matches the mask ravel + back-mapping).
            sig_shape = signal.data.shape[2:]
            _sel_idx = np.where(np.asarray(selection_mask, dtype=bool).ravel())[0]
            patterns = signal.data.reshape((-1,) + sig_shape)[_sel_idx]  # (n_selected, H, W)
            nav_shape = (n_selected,)
            is_partial = True
    _check_cancel()

    # PyEBSDIndex's indexers (both index_pats_distributed and index_pats) only
    # accept a real numpy array / EBSDPatterns / h5py.Dataset — NOT a lazy dask
    # array and NOT a np.memmap. Both are silently rejected ("Unrecognized input
    # data type" → returns None → "cannot unpack non-iterable NoneType object").
    #   * Lazy-loaded signals (large files) give a dask array.
    #   * EDAX UP1/UP2 (and other eager kikuchipy readers) give a np.memmap
    #     backed by the file — reshaping keeps it a memmap, so it reaches the
    #     Ray path and returns None (the "Hough broke on .up1" bug).
    # Materialise either into a plain contiguous in-RAM ndarray. This is the
    # same footprint eager loading always used, and indexing every pattern needs
    # them in RAM anyway.
    if hasattr(patterns, "compute") or isinstance(patterns, np.memmap):
        _progress(f"Hough: loading {n_selected} patterns into memory...", 0.38)
        with timed_step(f"Hough: materialise {n_selected} patterns"):
            # np.array (copy=True) forces a genuine in-RAM buffer, decoupled
            # from the memory-mapped file — ascontiguousarray would leave an
            # already-contiguous memmap file-backed, and the mmap file handle
            # is not valid inside spawned Ray worker processes.
            patterns = np.ascontiguousarray(np.array(patterns))

    # --- Step 4: Run Hough transform (40% -> 85%) ---
    _progress(f"Hough: indexing {n_selected} patterns...", 0.40)

    # A per-pixel PC (one PC per pattern, from PC refinement) is unsupported by
    # PyEBSDIndex's Ray path (see _hough_use_ray). Detect it from the detector
    # and route to single-thread, which handles a per-pixel PC correctly.
    pc_rows = int(np.asarray(detector.pc).size // 3) if detector is not None else 1
    ray_available = False
    try:
        from pyebsdindex._ebsd_index_parallel import index_pats_distributed
        ray_available = True
    except ImportError:
        pass
    use_ray = _hough_use_ray(n_selected, pc_rows, ray_available)

    if use_ray:
        import os, logging as _logging
        _logging.getLogger("ray.worker").setLevel(_logging.ERROR)
        ncpu = min(max(1, os.cpu_count() - 2), RAY_MAX_WORKERS)
        _progress(f"Hough: Ray parallel indexing ({ncpu} workers, {n_selected} patterns)...", 0.42)
        with timed_step(f"Hough: Ray indexing ({n_selected} patterns, {ncpu} workers)"):
            index_data, band_data = index_pats_distributed(
                patsin=patterns,
                ebsd_indexer_obj=indexer,
                ncpu=ncpu,
                verbose=0,
            )
    else:
        if pc_rows > 1 and n_selected >= RAY_MIN_PATTERNS:
            _progress(f"Hough: single-thread indexing ({n_selected} patterns, per-pixel PC — Ray parallel not supported with per-pixel PC)...", 0.42)
        elif n_selected < RAY_MIN_PATTERNS:
            _progress(f"Hough: single-thread indexing ({n_selected} patterns — below Ray threshold of {RAY_MIN_PATTERNS})...", 0.42)
        else:
            _progress(f"Hough: single-thread indexing ({n_selected} patterns)...", 0.42)
        with timed_step(f"Hough: single-thread indexing ({n_selected} patterns)"):
            index_data, band_data, _, _ = indexer.index_pats(
                patsin=patterns, verbose=0, chunksize=528,
            )

    _check_cancel()

    # --- Step 5: Build CrystalMap (85% -> 95%) ---
    _progress("Hough: building crystal map...", 0.85)
    with timed_step("Hough: build CrystalMap"):
        from kikuchipy.indexing._hough_indexing import xmap_from_hough_indexing_data

        if is_partial:
            xmap_sub = xmap_from_hough_indexing_data(
                data=index_data,
                phase_list=phase_list,
                navigation_shape=nav_shape,
                step_sizes=(1.0,),
            )
            _progress("Hough: back-mapping results to original grid positions...", 0.90)

            from orix.crystal_map import CrystalMap

            selected_indices = np.argwhere(selection_mask)  # (N, 2) of (row, col)
            xs = selected_indices[:, 1].astype(float)
            ys = selected_indices[:, 0].astype(float)

            xmap = CrystalMap(
                rotations=xmap_sub.rotations,
                phase_id=xmap_sub.phase_id,
                x=xs,
                y=ys,
                phase_list=phase_list,
            )
        else:
            step_sizes = tuple(
                a.scale for a in signal.axes_manager.navigation_axes[::-1]
            ) if hasattr(signal, 'axes_manager') else (1.0, 1.0)
            xmap = xmap_from_hough_indexing_data(
                data=index_data,
                phase_list=phase_list,
                navigation_shape=nav_shape,
                step_sizes=step_sizes,
            )

    # --- Step 6: Extract confidence (95% -> 100%) ---
    _progress("Hough: extracting confidence index (CI) scores...", 0.95)
    with timed_step("Hough: extract confidence"):
        confidence = _extract_confidence(index_data)
    return IndexingResult(
        xmap=xmap,
        selection_mask=selection_mask,
        original_shape=(n_rows, n_cols),
        method=IndexingMethod.HOUGH,
        confidence_scores=confidence,
        metadata={'band_data': band_data},
    )


def _dictionary_signal_from_master(master, detector, angular_step_deg, *,
                                   energy: float = 20.0, progress=None):
    """Project a master pattern into a detector-geometry dictionary (CPU path).

    kikuchipy's ``dictionary_indexing`` needs a *simulated dictionary* whose
    pattern shape matches the experimental detector — NOT a raw master. The
    indexing route hands us a master (``kp.load(path)``), so project it here.
    ``get_patterns`` (like the GPU projection) requires the square-Lambert,
    both-hemisphere master, so reload from the source file if the master was
    loaded otherwise (the plain ``kp.load`` default is stereographic/upper).
    """
    import kikuchipy as kp
    from backend.dict_gpu._pcadi.master_to_dict import _recover_master_path
    from backend.dict_gpu.pipeline.grid import sample_orientations

    m = master
    if (getattr(m, "hemisphere", None) != "both"
            or getattr(m, "projection", None) != "lambert"):
        path = _recover_master_path(m)
        if path is None:
            raise ValueError(
                "Dictionary indexing needs the master in the square-Lambert "
                "projection with both hemispheres, but it was loaded as "
                f"{getattr(m, 'projection', None)!r}/{getattr(m, 'hemisphere', None)!r} "
                "and its source file could not be located to reload."
            )
        m = kp.load(path, projection="lambert", hemisphere="both")

    rotations = sample_orientations(m.phase.point_group, angular_step_deg)
    if progress:
        progress(f"Dictionary: projecting {rotations.size} simulated patterns "
                 f"from master at {angular_step_deg}° (CPU)...")
    return m.get_patterns(rotations=rotations, detector=detector,
                          energy=energy, compute=True)


def dictionary_index_patterns(
    signal,
    dictionary,
    config: IndexingConfig,
    selection_mask: Optional[np.ndarray] = None,
    progress_callback=None,
    cancel_check=None,
    detector=None,
) -> IndexingResult:
    """Run Dictionary Indexing on selected patterns.

    Parameters
    ----------
    signal : kikuchipy.signals.EBSD
        Experimental EBSD signal.
    dictionary : kikuchipy.signals.EBSD
        Pre-computed dictionary of simulated patterns.
    config : IndexingConfig
        DI parameters (metric, keep_n, etc.).
    selection_mask : np.ndarray, optional
        Boolean mask (n_rows, n_cols). None = index all.
    progress_callback : callable, optional
        Called with status strings during processing.

    Returns
    -------
    IndexingResult
    """
    def _progress(msg):
        if progress_callback:
            progress_callback(msg)

    def _check_cancel():
        if cancel_check is not None and cancel_check():
            raise CancelledIndexingError("Dictionary indexing cancelled by user")

    # User may have cancelled in the gap between submit and worker pickup —
    # fail fast before we touch the GPU.
    _check_cancel()

    # GPU dispatcher — when CUDA is present and compute_mode is "auto" or
    # "gpu", route to backend/dict_gpu/api.gpu_dictionary_index_patterns.
    # When compute_mode is "gpu" but no CUDA device is available, raise.
    # Otherwise (auto + no GPU, or cpu) fall through to the CPU path.
    #
    # Always release the PyTorch CUDA cache in a ``finally`` once the GPU
    # path returns (success OR failure OR cancel) so the next indexing run
    # starts with the full VRAM budget. Without this the dict-tensor lives
    # on in the caching allocator and starves Spherical-GPU / batch runs.
    if config.compute_mode in ("auto", "gpu"):
        from backend.dict_gpu.runtime import detect_gpu
        from backend.dict_gpu.exceptions import GpuDictError
        gpu = detect_gpu()
        if gpu.available:
            from backend.dict_gpu.api import gpu_dictionary_index_patterns
            try:
                return gpu_dictionary_index_patterns(
                    experimental_signal=signal,
                    master_pattern_or_path=dictionary,
                    # Prefer the calibration-store detector (carries the REFINED PC after a
                    # Global PC Refine); signal.detector is a separate, stale channel (P2-B).
                    detector=detector if detector is not None else getattr(signal, "detector", None),
                    metric=config.metric,
                    keep_n=config.keep_n,
                    selection_mask=selection_mask,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                )
            finally:
                _release_cuda_cache()
        elif config.compute_mode == "gpu":
            raise GpuDictError(
                "compute_mode='gpu' was requested but no CUDA device is "
                "available. Set compute_mode='auto' or 'cpu'."
            )
        # auto + no GPU: silently fall through to CPU

    n_rows, n_cols = signal.data.shape[:2]

    if selection_mask is None:
        selection_mask = np.ones((n_rows, n_cols), dtype=bool)

    n_selected = int(selection_mask.sum())
    dict_size = len(dictionary.data) if hasattr(dictionary, 'data') else '?'
    _progress(
        f"Dictionary: matching {n_selected} patterns against {dict_size} simulated patterns "
        f"(metric={config.metric}, keep_n={config.keep_n})..."
    )

    # navigation_mask for kikuchipy: True = SKIP, opposite of our mask
    nav_mask = ~selection_mask if not selection_mask.all() else None

    if nav_mask is not None:
        _progress(f"Dictionary: applying navigation mask — {n_selected}/{n_rows*n_cols} pixels active...")

    # Capture stdout from kikuchipy/dask progress bars and forward to logging
    # so they appear in the Dev-Panel and frontend logs.
    import io, sys, logging as _logging
    _di_logger = _logging.getLogger("indexing.dictionary")

    class _StdoutCapture(io.TextIOBase):
        """Tee stdout to both original stream and logger."""
        def __init__(self, original):
            self.original = original
        def write(self, s):
            if s and s.strip():
                _di_logger.info(s.strip())
            return self.original.write(s)
        def flush(self):
            self.original.flush()

    _check_cancel()
    # Pull the active circular signal mask, if the user enabled one in the
    # viewer. Converted to kikuchipy's ``signal_mask`` convention here:
    # True = EXCLUDE (corners), False = include (disc).
    sig_mask_kp = None
    try:
        from backend.api.routes.ebsd_viewer import get_active_include_mask
        include = get_active_include_mask()
        if include is not None:
            sig_mask_kp = ~include
            _progress(f"Dictionary: circular signal mask active "
                      f"({int((~sig_mask_kp).sum())}/{sig_mask_kp.size} px used)")
    except Exception:
        logger.debug("Could not fetch active signal mask", exc_info=True)

    # kikuchipy's dictionary_indexing needs a simulated DICTIONARY whose pattern
    # shape matches the experimental detector. The indexing route loads a raw
    # MASTER (kp.load), so project it into a dictionary here when needed. A
    # master pattern exposes get_patterns(); a pre-generated dictionary signal
    # does not — that's how we tell them apart. (The GPU path does the
    # equivalent on-device; this is the CPU counterpart so compute_mode='cpu'
    # works with the same master input instead of raising a shape-mismatch.)
    if hasattr(dictionary, "get_patterns"):
        det_for_dict = detector if detector is not None else getattr(signal, "detector", None)
        if det_for_dict is None:
            raise ValueError("Dictionary indexing from a master needs a detector "
                             "(PC + geometry); none was available.")
        _check_cancel()
        dictionary = _dictionary_signal_from_master(
            dictionary, det_for_dict, config.angular_step_deg, progress=_progress)
        _check_cancel()

    old_stdout = sys.stdout
    sys.stdout = _StdoutCapture(old_stdout)
    try:
        di_kwargs = dict(
            metric=config.metric,
            keep_n=config.keep_n,
            n_per_iteration=config.n_per_iteration,
            navigation_mask=nav_mask,
            rechunk=True,  # Split into chunks → dask uses all CPU threads
        )
        if sig_mask_kp is not None:
            di_kwargs["signal_mask"] = sig_mask_kp
        xmap_raw = signal.dictionary_indexing(dictionary, **di_kwargs)
    finally:
        sys.stdout = old_stdout

    # Cancel after the kikuchipy call too — the user may have hit Stop
    # while the Dask loop was running (we can't interrupt that loop from
    # outside, but at least we skip the back-mapping + result assembly).
    _check_cancel()
    _progress("Dictionary: extracting NCC scores from best matches...")
    # Extract scores from the first (best) match
    scores = None
    if hasattr(xmap_raw, 'prop') and 'scores' in xmap_raw.prop:
        scores_all = xmap_raw.prop['scores']
        if scores_all.ndim == 2:
            scores = scores_all[:, 0]  # Best match score
        else:
            scores = scores_all

    # Back-map xmap to original grid positions (same pattern as Hough)
    # kikuchipy returns compressed xmap with only indexed pixels — we need
    # to restore original (row, col) coordinates for downstream analysis.
    if nav_mask is not None and not selection_mask.all():
        _progress("Dictionary: back-mapping results to original grid positions...")
        from orix.crystal_map import CrystalMap
        selected_indices = np.argwhere(selection_mask)  # (N, 2) of (row, col)

        # xmap_raw may have fewer entries than selected_indices if some
        # pixels failed to index — truncate to match
        n_xmap = xmap_raw.size
        xs = selected_indices[:n_xmap, 1].astype(float)
        ys = selected_indices[:n_xmap, 0].astype(float)

        props = {}
        for key in xmap_raw.prop:
            val = xmap_raw.prop[key]
            if hasattr(val, '__len__') and len(val) == n_xmap:
                props[key] = val
            else:
                props[key] = val

        xmap = CrystalMap(
            rotations=xmap_raw.rotations,
            phase_id=xmap_raw.phase_id,
            x=xs,
            y=ys,
            phase_list=xmap_raw.phases_in_data,
            prop=props,
            scan_unit="px",
        )
        # Truncate scores to match xmap size
        if scores is not None and len(scores) > n_xmap:
            scores = scores[:n_xmap]
    else:
        xmap = xmap_raw

    return IndexingResult(
        xmap=xmap,
        selection_mask=selection_mask,
        original_shape=(n_rows, n_cols),
        method=IndexingMethod.DICTIONARY,
        confidence_scores=scores,
        metadata={
            'dictionary': dictionary,   # retained for pattern comparison viewer
            'signal': signal,           # experimental signal for pixel lookup
            'metric': config.metric,
            'keep_n': config.keep_n,
        },
    )


def refine_orientations(
    result,
    master_h5_path: str,
    energy: float,
    pc: tuple,
    trust_region: float = 5.0,
    progress_callback=None,
):
    """Refine orientations from a dictionary indexing result.

    Uses kikuchipy's signal.refine_orientation() (Nelder-Mead) to improve
    the initial coarse dictionary match to sub-degree accuracy.

    Parameters
    ----------
    result : IndexingResult
    master_h5_path : str
        Path to the .h5 master pattern file (EBSDMasterPattern in Lambert projection).
    energy : float
        Accelerating voltage in kV.
    pc : tuple
        Pattern centre (PCx, PCy, PCz).
    trust_region : float
        +/- angular bound in degrees for all 3 Euler angles. Default 5.0.
    progress_callback : callable, optional

    Returns
    -------
    orix.crystal_map.CrystalMap
        Refined CrystalMap. Replace result.xmap with this.
    """
    import kikuchipy as kp
    from kikuchipy.detectors import EBSDDetector

    def _p(msg):
        if progress_callback:
            progress_callback(msg)

    signal = result.metadata['signal']
    pattern_shape = signal.data.shape[2:]   # (n_rows_det, n_cols_det)

    _p(f"Loading master pattern from {master_h5_path} ...")
    master = kp.load(master_h5_path)

    detector = EBSDDetector(
        shape=pattern_shape,
        pc=list(pc),
        sample_tilt=70.0,
        tilt=0.0,
    )

    n_pixels = int(result.selection_mask.sum())

    # For partial indexing (region/mask), we must crop the signal to match
    # the xmap shape — kikuchipy requires signal nav_shape == xmap shape.
    if not result.selection_mask.all():
        mask = result.selection_mask
        # Find bounding box of the selection
        rows_any = np.any(mask, axis=1)
        cols_any = np.any(mask, axis=0)
        r_min, r_max = np.where(rows_any)[0][[0, -1]]
        c_min, c_max = np.where(cols_any)[0][[0, -1]]

        _p(f"Cropping signal to region [{r_min}:{r_max+1}, {c_min}:{c_max+1}]...")
        cropped_signal = signal.inav[c_min:c_max+1, r_min:r_max+1]

        # Build a full-grid xmap for the cropped region
        from orix.crystal_map import CrystalMap
        crop_h = r_max - r_min + 1
        crop_w = c_max - c_min + 1

        # Re-create xmap with coordinates relative to cropped region
        xs = result.xmap.x - c_min if result.xmap.x is not None else None
        ys = result.xmap.y - r_min if result.xmap.y is not None else None

        cropped_xmap = CrystalMap(
            rotations=result.xmap.rotations,
            phase_id=result.xmap.phase_id,
            x=xs, y=ys,
            phase_list=result.xmap.phases_in_data,
            prop=result.xmap.prop,
        )

        # Create nav_mask for the cropped region (skip pixels outside selection)
        sub_mask = mask[r_min:r_max+1, c_min:c_max+1]
        nav_mask = ~sub_mask
        use_signal = cropped_signal
        use_xmap = cropped_xmap
    else:
        nav_mask = None
        use_signal = signal
        use_xmap = result.xmap

    _p(
        f"Refining {n_pixels} orientations "
        f"(trust_region=±{trust_region}°, energy={energy} kV) ..."
    )

    # Pass-through the circular signal mask if the user has it enabled in
    # the viewer — refine_orientation skips masked pixels when computing
    # NCC, which is exactly what we want for the detector corners.
    refine_sig_mask = None
    try:
        from backend.api.routes.ebsd_viewer import get_active_include_mask
        _inc = get_active_include_mask()
        if _inc is not None:
            refine_sig_mask = ~_inc
    except Exception:
        logger.debug("Could not fetch signal mask for refinement", exc_info=True)

    refine_kwargs = dict(
        xmap=use_xmap,
        detector=detector,
        master_pattern=master,
        energy=energy,
        navigation_mask=nav_mask,
        trust_region=[trust_region, trust_region, trust_region],
        compute=True,
    )
    if refine_sig_mask is not None:
        refine_kwargs["signal_mask"] = refine_sig_mask
    refined_xmap = use_signal.refine_orientation(**refine_kwargs)

    _p("Refinement complete.")
    return refined_xmap


def _find_indexebsd() -> Optional[str]:
    """Find the IndexEBSD binary path.

    On Windows: searches via WSL (where EMSphInx is installed).
    On Linux/WSL: searches local known paths and $PATH.

    Returns
    -------
    str or None
        Full path to IndexEBSD (WSL/Linux path), or None if not found.
    """
    import sys as _sys
    from path_utils import _is_valid_unix_path

    if _sys.platform == "win32":
        # On Windows, EMSphInx lives inside WSL
        try:
            # Try 'which' via WSL first
            r = subprocess.run(
                ['wsl', 'bash', '-lc', 'which IndexEBSD 2>/dev/null'],
                capture_output=True, text=True, timeout=15,
            )
            path = r.stdout.strip().replace('\x00', '')
            if path and _is_valid_unix_path(path):
                return path
        except Exception:
            pass

        # Try known WSL paths
        try:
            r = subprocess.run(
                ['wsl', 'bash', '-lc', 'whoami'],
                capture_output=True, text=True, timeout=5,
            )
            user = r.stdout.strip().replace('\x00', '')
        except Exception:
            user = ""

        if user:
            candidates = [
                f"/home/{user}/EMSphInx/build/IndexEBSD",
                f"/home/{user}/emsoft/builds/EMSphInx-Release/IndexEBSD",
                f"/usr/local/bin/IndexEBSD",
            ]
            for cand in candidates:
                try:
                    r = subprocess.run(
                        ['wsl', 'bash', '-c', f'test -x "{cand}" && echo "{cand}"'],
                        capture_output=True, text=True, timeout=5,
                    )
                    if r.stdout.strip() and _is_valid_unix_path(r.stdout.strip()):
                        return r.stdout.strip()
                except Exception:
                    pass
        return None

    # Native Linux/WSL: search locally
    known_paths = [
        Path("/usr/local/bin/IndexEBSD"),
        Path.home() / "emsoft" / "builds" / "EMSphInx-Release" / "IndexEBSD",
        Path.home() / "EMSphInx" / "build" / "IndexEBSD",
    ]
    for p in known_paths:
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)

    # Fallback: check $PATH
    which = shutil.which("IndexEBSD")
    if which:
        return which
    return None


def _build_indexebsd_wsl_cmd(outdir_wsl: str, indexebsd: str, nml_wsl: str) -> list:
    """Build the ``wsl`` command list that runs IndexEBSD inside WSL (on Windows).

    SECURITY: the paths are passed as bash *positional parameters* ($1/$2/$3),
    never interpolated into the script string, so an EBSD filename containing
    ``"``, ``$(...)``, backticks, ``;`` etc. cannot break out and execute
    arbitrary shell. ($0 is just a label used in bash error messages.)

    CRITICAL: the ``-e``/``--exec`` flag is REQUIRED. Without it ``wsl.exe``
    swallows every argument *after* the ``-c`` script, so bash runs
    ``cd "$1" && "$2" "$3"`` with $1/$2/$3 all empty — the empty $2 then yields
    ``: command not found`` (exit 127), which looked like the IndexEBSD binary
    was missing even though it was found. Verified on Windows 11 / WSL2::

        wsl    bash -lc 'echo $#' A B   -> 0   (args dropped)
        wsl -e bash -lc 'echo $#' A B   -> 2   (args forwarded)
    """
    return [
        'wsl', '-e', 'bash', '-lc', 'cd "$1" && "$2" "$3"',
        'emsphinx', outdir_wsl, indexebsd, nml_wsl,
    ]


def _get_emsoft_data_path() -> Path:
    """Get the EMdatapathname from EMsoft config or default.

    On Windows: reads the config via WSL, returns a local temp directory
    (since IndexEBSD runs in WSL and needs accessible paths).
    """
    import sys as _sys

    if _sys.platform == "win32":
        # On Windows, use project-local temp dir that both Windows and WSL can access.
        # Return the base dir only — callers add subdirectories (e.g. "SphericalIndexing").
        local_dir = Path(__file__).parent / "TempRuns"
        local_dir.mkdir(parents=True, exist_ok=True)
        return local_dir

    config_path = Path.home() / ".config" / "EMsoft" / "EMsoftConfig.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            return Path(config.get("EMdatapathname", str(Path.home() / "EMsoftData")))
        except (json.JSONDecodeError, KeyError):
            pass
    return Path.home() / "EMsoftData"


def _get_pattern_dataset_path(h5_path: str) -> str:
    """Determine HDF5 dataset path to EBSD patterns.

    Supports Oxford H5OINA (``1/EBSD/Data/Processed Patterns``) and
    EDAX/TSL HDF5 (``ScanName/EBSD/Data/Pattern``).

    Returns the full HDF5 internal path string for IndexEBSD's ``patdset``
    parameter.
    """
    try:
        import h5py
        with h5py.File(h5_path, 'r') as f:
            # Oxford H5OINA: /1/EBSD/Data/Processed Patterns or /1/EBSD/Data/Raw Patterns
            for scan_key in ['1', '2', '3']:
                for pat_name in ['Processed Patterns', 'Raw Patterns']:
                    path = f"{scan_key}/EBSD/Data/{pat_name}"
                    if path in f:
                        return path

            # EDAX/TSL HDF5: /<ScanName>/EBSD/Data/Pattern
            for key in f.keys():
                edax_path = f"{key}/EBSD/Data/Pattern"
                if edax_path in f:
                    return edax_path
    except Exception as e:
        logger.warning(f"Could not auto-detect pattern dataset path: {e}")

    # Default fallback
    return '1/EBSD/Data/Processed Patterns'


def _detect_rectangular_crop(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Detect if a boolean mask is a contiguous rectangle.

    Returns (row_start, row_end, col_start, col_end) if the mask is
    rectangular, or None if it is irregular.
    """
    rows_any = np.any(mask, axis=1)
    cols_any = np.any(mask, axis=0)

    row_indices = np.where(rows_any)[0]
    col_indices = np.where(cols_any)[0]

    if len(row_indices) == 0 or len(col_indices) == 0:
        return None

    r0, r1 = int(row_indices[0]), int(row_indices[-1]) + 1
    c0, c1 = int(col_indices[0]), int(col_indices[-1]) + 1

    # Verify the bounding box is fully filled (i.e. truly rectangular)
    expected_count = (r1 - r0) * (c1 - c0)
    if int(mask[r0:r1, c0:c1].sum()) == expected_count:
        return (r0, r1, c0, c1)

    return None


def _export_patterns_for_indexebsd(
    src_h5: str, dst_h5: str, patdset: str,
    progress_callback=None,
    crop_region=None,
    n_cols_full: int = 0,
    source_vendor: str = "",
) -> None:
    """Re-export pattern dataset to HDF5 1.8 compatible file.

    IndexEBSD is built with HDF5 1.8.20 and cannot read newer HDF5 files
    (fixed-length string dtypes cause type conversion errors).  This
    helper copies the pattern dataset plus the ``Manufacturer`` tag into
    a fresh file written with ``libver='earliest'``.

    Parameters
    ----------
    crop_region : tuple, optional
        (row_start, row_end, col_start, col_end) to export only a
        rectangular sub-region. Patterns are stored in row-major order.
    n_cols_full : int
        Number of columns in the full scan (needed for crop indexing).
    """
    import h5py

    def _progress(msg):
        if progress_callback:
            progress_callback(msg)
        logger.info(msg)

    _progress(f"Converting patterns to HDF5 1.8 format...")
    with h5py.File(src_h5, 'r') as src:
        ds = src[patdset]

        if crop_region is not None and n_cols_full > 0:
            r0, r1, c0, c1 = crop_region
            crop_rows = r1 - r0
            crop_cols = c1 - c0
            _progress(f"  Cropping to region [{r0}:{r1}, {c0}:{c1}] "
                       f"({crop_rows}×{crop_cols} = {crop_rows * crop_cols} patterns)")
            # Build indices for the rectangular crop in row-major order
            indices = []
            for row in range(r0, r1):
                for col in range(c0, c1):
                    indices.append(row * n_cols_full + col)
            patterns = ds[indices]
        else:
            n_patterns = ds.shape[0]
            _progress(f"  Reading {n_patterns} patterns ({ds.shape})...")
            patterns = ds[:]

        # Read vendor name — IndexEBSD requires Manufacturer to exist.
        # EMSphInx applies `vendorFlip` based on this attribute per
        # include/modality/ebsd/pattern.hpp:463-470 of the upstream source:
        #     EDAX / EMsoft  -> vendorFlip = true   (apply vertical flip)
        #     Oxford / Bruker / Bruker Nano / DREAM.3D -> vendorFlip = false
        #
        # We tag everything as 'EDAX' so EMSphInx applies vendorFlip=true.
        # Empirical justification (cross-method audit on NiLowGain
        # 2026-05-21, EDAX-source OIM H5):
        #   * Manufacturer='EDAX' (vendorFlip=true):  5.4 deg median vs Hough  ✓
        #   * Manufacturer='Bruker' (vendorFlip=false): 40.3 deg median vs Hough  ✗
        # An earlier hypothesis — that kikuchipy patterns are "already-flipped"
        # and need vendorFlip=false — was empirically refuted (see commit log
        # for 1ffef20 -> revert). kikuchipy patterns ARE in the EDAX camera
        # convention that EMSphInx EXPECTS as input under Manufacturer='EDAX';
        # the vendorFlip then converts them to EMSphInx's internal Bruker-
        # canonical for SHT correlation.
        vendor = b'EDAX'
        flip_state = 'true'
        logger.info(f"  Manufacturer tag: {vendor.decode()} "
                    f"(source: {source_vendor!r}, vendorFlip={flip_state})")

    _progress(f"  Writing {patterns.nbytes / 1024 / 1024:.0f} MB to compat file...")
    with h5py.File(dst_h5, 'w', libver='earliest') as dst:
        dst.create_dataset('Manufacturer', data=vendor)

        grp = dst
        parts = patdset.split('/')
        for part in parts[:-1]:
            grp = grp.require_group(part)
        grp.create_dataset(parts[-1], data=patterns)

    _progress(f"  HDF5 conversion done: {dst_h5}")


def generate_emsphinx_nml(
    config: IndexingConfig,
    h5_path: str,
    detector_params: Dict,
    output_dir: Path,
    progress_callback=None,
    crop_region=None,
) -> Path:
    """Generate an EMSphInx NML file for IndexEBSD.

    The NML format matches the ``IndexEBSD -t`` template (namelist
    ``&EMSphInx``).  All paths are written as absolute paths so the
    NML is self-contained and does not depend on the working directory
    or ``EMdatapathname`` being set.

    Parameters
    ----------
    config : IndexingConfig
        Indexing configuration with spherical parameters.
    h5_path : str
        Path to the experimental H5OINA file.
    detector_params : dict
        Must contain: 'pc_x', 'pc_y', 'pc_z' (Oxford convention),
        'n_cols', 'n_rows' (scan dims), 'pat_width', 'pat_height',
        'pixel_size' (microns), 'tilt' (degrees), 'binning',
        'step_x', 'step_y' (microns).
    output_dir : Path
        Directory for output files.
    crop_region : tuple, optional
        (row_start, row_end, col_start, col_end) for rectangular crop.
        When given, only cropped patterns are exported and scandims adjusted.

    Returns
    -------
    Path
        Path to the generated .nml file.
    """
    import sys as _sys

    def _to_wsl_path(win_path: str) -> str:
        """Convert a Windows path to WSL mount path (e.g. E:\\foo → /mnt/e/foo)."""
        p = str(win_path).replace('\\', '/')
        if len(p) >= 2 and p[1] == ':':
            drive = p[0].lower()
            return f"/mnt/{drive}{p[2:]}"
        return p

    # Use absolute paths throughout — IndexEBSD resolves them correctly
    sht = str(Path(config.sht_file).resolve())
    h5_abs = str(Path(h5_path).resolve())
    output_dir = Path(output_dir).resolve()

    # IndexEBSD (HDF5 1.8.20) can't read newer HDF5 files or .h5oina extension.
    # Re-export pattern data to a compatible HDF5 1.8 file.
    # When crop_region is set, always re-export (cropped data changes).
    if os.path.isfile(h5_abs):
        crop_suffix = ""
        if crop_region is not None:
            r0, r1, c0, c1 = crop_region
            crop_suffix = f"_crop_{r0}_{r1}_{c0}_{c1}"
        h5_compat = str(output_dir / (Path(h5_abs).stem + crop_suffix + '_compat.h5'))
        n_cols_full = detector_params.get('n_cols', 0)
        # Always re-export: old compat files may have wrong pattern count
        if True:
            patdset_src = _get_pattern_dataset_path(h5_path)
            _export_patterns_for_indexebsd(
                h5_abs, h5_compat, patdset_src,
                progress_callback=progress_callback,
                crop_region=crop_region,
                n_cols_full=n_cols_full,
                source_vendor=detector_params.get('source_vendor', ''),
            )
        h5_abs = h5_compat

    # Auto-detect HDF5 dataset path for patterns
    patdset = _get_pattern_dataset_path(h5_path)

    # Output filenames (absolute)
    base_name = Path(h5_path).stem
    datafile = str(output_dir / f"{base_name}_spherical.h5")
    vendorfile = str(output_dir / f"{base_name}_spherical.ang")

    # On Windows, convert all paths to WSL mount paths for the NML
    # (IndexEBSD runs in WSL and needs /mnt/... paths)
    if _sys.platform == "win32":
        sht = _to_wsl_path(sht)
        h5_abs = _to_wsl_path(h5_abs)
        datafile = _to_wsl_path(datafile)
        vendorfile = _to_wsl_path(vendorfile)

    # Extract detector parameters
    pc_x = detector_params.get('pc_x', 0.5)
    pc_y = detector_params.get('pc_y', 0.5)
    pc_z = detector_params.get('pc_z', 0.5)
    n_cols = detector_params.get('n_cols', 0)
    n_rows = detector_params.get('n_rows', 0)
    pat_w = detector_params.get('pat_width', 640)
    pat_h = detector_params.get('pat_height', 480)
    delta = detector_params.get('pixel_size', 55.0)
    thetac = detector_params.get('tilt', 10.0)
    binning = detector_params.get('binning', 1)
    step_x = detector_params.get('step_x', 0.1)
    step_y = detector_params.get('step_y', 0.1)
    vendor = detector_params.get('vendor', 'EMsoft')

    # Adjust scandims for cropped region
    if crop_region is not None:
        r0, r1, c0, c1 = crop_region
        n_rows = r1 - r0
        n_cols = c1 - c0

    nml_content = f""" &EMSphInx
!#################################################################
! Input Files
!#################################################################
 patfile    = '{h5_abs}',
 patdset    = '{patdset}',
 masterfile = '{sht}',

!#################################################################
! Pattern Processing
!#################################################################
 patdims    = {pat_w}, {pat_h},
 circmask   = {config.circmask},
 gausbckg   = {'.TRUE.' if config.gausbckg else '.FALSE.'},
 nregions   = {config.nregions},

!#################################################################
! Camera Calibration
!#################################################################
 delta      = {delta},
 pctr       = {pc_x}, {pc_y}, {pc_z},
 vendor     = '{vendor}',
 thetac     = {thetac},

!#################################################################
! Scan Information
!#################################################################
 scandims   = {n_cols}, {n_rows}, {step_x}, {step_y},
 roimask    = '',

!#################################################################
! Indexing Parameters
!#################################################################
 bw         = {config.bandwidth},
 normed     = {'.TRUE.' if config.normed else '.FALSE.'},
 refine     = {'.TRUE.' if config.refine else '.FALSE.'},
 nthread    = 0,
 batchsize  = 0,

!#################################################################
! Output Files
!#################################################################
 datafile   = '{datafile}',
 vendorfile = '{vendorfile}',
 /
"""

    nml_path = Path(output_dir) / f"{base_name}_emsphinx.nml"
    # CRITICAL: Unix line endings — IndexEBSD runs in WSL and chokes on \r\n
    # CRITICAL: UTF-8 encoding — SHT filenames may contain Greek letters (α, τ)
    nml_path.write_text(nml_content, encoding='utf-8', newline='\n')
    logger.info(f"Generated EMSphInx NML: {nml_path}")
    return nml_path


def _parse_emsphinx_output(output_dir: Path, base_name: str) -> Optional[object]:
    """Parse EMSphinx output (.ang or .ctf) into a CrystalMap.

    Tries .ang first (IndexEBSD default vendorfile format), then .ctf.

    Returns
    -------
    CrystalMap or None
    """
    from orix.io import load as orix_load

    for ext in ['.ang', '.ctf']:
        out_file = output_dir / f"{base_name}_spherical{ext}"
        if out_file.exists() and out_file.stat().st_size > 0:
            try:
                xmap = orix_load(str(out_file))
                # EMSphinx writes 8-column .ang (not standard TSL 10-col).
                # orix puts the confidence column into 'unknown2' instead
                # of 'ci'.  Rename so score extraction finds it.
                if hasattr(xmap, 'prop') and 'unknown2' in xmap.prop and 'ci' not in xmap.prop:
                    xmap.prop['ci'] = xmap.prop.pop('unknown2')
                logger.info(f"Parsed EMSphinx result from {out_file}")
                return xmap
            except Exception as e:
                logger.warning(f"Could not parse {out_file}: {e}")

    # Try HDF5 output as last resort
    h5_out = output_dir / f"{base_name}_spherical.h5"
    if h5_out.exists() and h5_out.stat().st_size > 0:
        try:
            import h5py
            with h5py.File(str(h5_out), 'r') as f:
                # EMSphinx HDF5 format: Scan 1/EBSD/Data/{Phi1, Phi, Phi2, Metric}
                data_path = None
                for scan_key in ['Scan 1', 'Scan 2', 'Scan1', 'Scan2']:
                    p = f"{scan_key}/EBSD/Data"
                    if p in f:
                        data_path = p
                        break

                if data_path and f"{data_path}/Phi1" in f:
                    phi1 = f[f"{data_path}/Phi1"][:].ravel()  # radians
                    phi = f[f"{data_path}/Phi"][:].ravel()
                    phi2 = f[f"{data_path}/Phi2"][:].ravel()
                    metric = f[f"{data_path}/Metric"][:].ravel() if f"{data_path}/Metric" in f else None

                    from orix.quaternion import Rotation
                    from orix.crystal_map import CrystalMap

                    # Euler angles are in radians from EMSphinx
                    euler_rad = np.column_stack([phi1, phi, phi2])
                    rotations = Rotation.from_euler(euler_rad)
                    n_points = len(phi1)

                    prop = {}
                    if metric is not None:
                        prop['ci'] = metric[:n_points]

                    xmap = CrystalMap(
                        rotations=rotations,
                        x=np.arange(n_points) if n_points > 1 else np.array([0]),
                        prop=prop,
                    )
                    logger.info(f"Parsed EMSphinx HDF5 result from {h5_out} ({n_points} points)")
                    return xmap
                else:
                    logger.warning(f"EMSphinx HDF5 {h5_out}: no Euler data found at expected paths")
        except Exception as e:
            logger.warning(f"Could not parse HDF5 {h5_out}: {e}")

    return None


def _get_processed_patterns_if_dirty(
    selection_mask: Optional[np.ndarray], roi_mode: bool,
) -> Optional[np.ndarray]:
    """Return the active in-memory EBSD patterns IF the viewer modified them.

    Frame averaging / background removal / autocontrast change the in-memory
    signal but not the file on disk. Spherical indexing reads patterns from
    the file, so without this it would silently ignore those edits. When the
    active signal is flagged dirty this returns the processed patterns (full
    grid, or the ROI subset when ``roi_mode``); otherwise returns ``None`` and
    callers read the original H5 exactly as before.
    """
    try:
        from backend.api.routes.ebsd_viewer import (
            is_active_signal_dirty, _get_active_signal,
        )
    except Exception:
        return None
    if not is_active_signal_dirty():
        return None
    signal = _get_active_signal()
    data = getattr(signal, "data", None)
    if data is None:
        return None
    data = np.asarray(data)
    if data.ndim != 4:
        return None
    sig_h, sig_w = int(data.shape[2]), int(data.shape[3])
    flat = data.reshape(-1, sig_h, sig_w)
    if roi_mode and selection_mask is not None:
        mask_flat = np.asarray(selection_mask).ravel()
        if mask_flat.size != flat.shape[0]:
            return None  # shape mismatch — fall back to reading the file
        return flat[mask_flat]
    return flat


def _resolve_cif_for_sht(sht_path: str) -> str:
    """Best-effort map an SHT master-pattern path to its source CIF (for the
    Hough anchor used in pseudo-symmetry resolution). Returns "" on any failure
    so the caller fails safe to the raw spherical orientations."""
    try:
        from pathlib import Path as _P
        from path_utils import get_local_database_path, DATABASE_SUBFOLDERS
        from phase_metadata import get_phase_metadata
        cif_dir = get_local_database_path() / DATABASE_SUBFOLDERS["cif_library"]
        meta = get_phase_metadata(_P(sht_path), cif_library_dir=cif_dir)
        return getattr(meta, "cif_path", "") or ""
    except Exception:
        return ""


class _PatternStackTooLarge(Exception):
    """Raised when the full pattern stack would exceed the in-memory read budget
    for streaming-path pseudo-symmetry resolution (see
    :func:`_materialize_patterns_for_resolution`). Carries the estimated size so
    the caller can log an honest message and fall back to raw orientations."""

    def __init__(self, nbytes: int):
        self.nbytes = int(nbytes)
        super().__init__(f"pattern stack ~{nbytes / 1e9:.1f} GB exceeds read budget")


def _resolve_read_budget_bytes() -> int:
    """Max bytes we'll pull into RAM to build the Hough anchor for pseudo-symmetry
    resolution on the streaming (index_h5) path. Half of *available* RAM (leaves
    headroom for the Hough indexer + the CrystalMap), min 1 GB. Falls back to 4 GB
    if psutil is unavailable. This is a safety valve: a 27 GB lazy-loaded map never
    triggers a giant read — it just keeps the raw orientations and tells the user
    to use an ROI or the Hough method instead."""
    try:
        import psutil
        return max(1 * 1024 ** 3, int(0.5 * psutil.virtual_memory().available))
    except Exception:
        return 4 * 1024 ** 3


def _read_h5_pattern_stack(
    h5_path: str,
    indices: Optional[np.ndarray] = None,
    max_bytes: Optional[int] = None,
) -> np.ndarray:
    """Read the EBSD pattern dataset from an H5OINA / EDAX .h5 into a NumPy array.

    ``indices=None`` reads the full ``(N, H, W)`` stack in file order; otherwise
    reads only those flat pattern indices (must be ascending for efficient
    chunked reads). Dataset discovery mirrors
    ``Tier1Indexer._open_pattern_dataset`` so both Oxford H5OINA
    (``1/EBSD/Data/Processed Patterns``) and EDAX H5
    (``<Scan>/EBSD/Data/Pattern``) work.

    ``max_bytes`` (full read only) raises :class:`_PatternStackTooLarge` before
    allocating if the stack would exceed the budget — used by the streaming-path
    resolver so a huge lazy-loaded file can't OOM the run.
    """
    import h5py
    with h5py.File(h5_path, "r") as f:
        candidates = [
            "1/EBSD/Data/Processed Patterns",
            "1/EBSD/Data/Patterns",
            "1/EBSD/Data/Raw Patterns",
        ]
        for top_name in list(f.keys()):
            top = f[top_name]
            if not isinstance(top, h5py.Group):
                continue
            for sub in ("EBSD/Data/Pattern", "EBSD/Data/Patterns"):
                full = f"{top_name}/{sub}"
                if full in f and isinstance(f[full], h5py.Dataset):
                    candidates.append(full)
        dset = None
        for ds_path in candidates:
            if ds_path in f and isinstance(f[ds_path], h5py.Dataset) and f[ds_path].ndim == 3:
                dset = f[ds_path]
                break
        if dset is None:
            raise FileNotFoundError(
                f"No EBSD pattern dataset in {h5_path} (tried: {candidates})"
            )
        if indices is None:
            if max_bytes is not None:
                nbytes = int(np.prod(dset.shape)) * int(dset.dtype.itemsize)
                if nbytes > max_bytes:
                    raise _PatternStackTooLarge(nbytes)
            return np.asarray(dset[:])
        idx = np.asarray(indices)
        return np.asarray(dset[idx])


def _materialize_patterns_for_resolution(
    h5_path: str, masters_meta, phase_id, progress=None,
) -> Optional[np.ndarray]:
    """Provide in-memory patterns for pseudo-symmetry resolution on the streaming
    (``index_h5``) full-map path, which otherwise has none.

    On the streaming path there are no in-memory patterns, so the Hough
    pseudo-symmetry resolver + variant unification were silently skipped and a
    ``z_rot==2`` intermetallic (cubic approximant m-3/23, cubic -43m, orthorhombic
    mmm/222/mm2) kept its WRONG spherical pseudo-variant on every pixel. If such a
    phase actually won pixels, read the pattern stack from the H5 now (as the
    Hough anchor) so the auto-correction runs on a plain full-map index too — no
    viewer pre-processing / ROI required.

    Returns ``(N, H, W)`` patterns in result order, or ``None`` to keep the raw
    spherical orientations (nothing to resolve, file too large for the RAM budget,
    or any read error — all fail safe to the prior behaviour).
    """
    from backend.spherical_gpu.pseudosym import spherical_unreliable
    try:
        present = {int(x) for x in np.unique(np.asarray(phase_id).reshape(-1))}
    except Exception:
        return None
    need = any(
        spherical_unreliable((m or {}).get("z_rot"), (m or {}).get("point_group"))
        and (i + 1) in present
        for i, m in enumerate(masters_meta or [])
    )
    if not need or not h5_path:
        return None
    try:
        pats = _read_h5_pattern_stack(h5_path, max_bytes=_resolve_read_budget_bytes())
    except _PatternStackTooLarge as e:
        if progress:
            progress(
                f"Spherical-GPU: pattern stack ~{e.nbytes / 1e9:.1f} GB exceeds the "
                f"in-memory cap — skipping automatic pseudo-symmetry resolution. "
                f"Index a region (ROI) or use the Hough method for intermetallic "
                f"maps.", 0.905)
        return None
    except Exception:
        logger.warning(
            "Could not materialise patterns for pseudo-symmetry resolution on the "
            "streaming path; keeping raw spherical orientations", exc_info=True)
        return None
    if progress:
        progress(
            "Spherical-GPU: loaded patterns from H5 for pseudo-symmetry resolution "
            "(z_rot==2 phase present)...", 0.905)
    return pats


def spherical_gpu_index_patterns(
    h5_path: str,
    config: IndexingConfig,
    detector_params: Dict,
    selection_mask: Optional[np.ndarray] = None,
    progress_callback=None,
    sht_paths: Optional[List[str]] = None,
    cancel_check=None,
    phase_weights: Optional[np.ndarray] = None,
) -> IndexingResult:
    """Run Spherical Indexing via the in-process PyTorch GPU pipeline.

    Drop-in alternative to ``spherical_index_patterns`` that uses
    ``backend.spherical_gpu.SphericalGPUBackend``. ~11x faster than WSL
    EMSphInx on the validated HiGainNi case, with bit-identical math.

    Multi-phase support: pass a list of SHT paths via ``sht_paths`` (or
    set ``config.sht_file`` for the single-phase legacy path). When given
    multiple SHTs the backend builds one Tier1Indexer per phase and
    re-uses the H5 read + indexer state across all phases — much cheaper
    than calling this function once per phase as the legacy multi-phase
    route did (which rebuilt the SHT-loaded state every phase).

    Progress messages emit per-phase progress + measured pat/s rate so
    they show up in the IndexingPage log box (the FastAPI route routes
    them through ``_indexing_tasks[task_id]['pending_log']`` for poll
    delivery).
    """
    from backend.spherical_gpu.backend import (
        BackendConfig, PhaseConfig, SphericalGPUBackend,
    )
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    import time as _time

    def _progress(msg, pct=None):
        if progress_callback:
            progress_callback(msg, pct)
        logger.info(msg)

    # Normalize sht inputs: explicit list wins, else fall back to config.sht_file.
    files = list(sht_paths or [])
    if not files and config.sht_file:
        files = [config.sht_file]
    if not files:
        raise ValueError(
            "spherical_gpu_index_patterns requires sht_paths or config.sht_file"
        )
    n_phases = len(files)

    _progress(
        f"Spherical-GPU: building backend "
        f"({'multi-phase: ' + str(n_phases) if n_phases > 1 else 'single phase'}, "
        f"L={config.bandwidth}, refine={'yes' if config.refine else 'no'}, "
        f"gausbckg={'yes' if config.gausbckg else 'no'}, "
        f"circmask={config.circmask}, nregions={config.nregions})",
        0.02,
    )
    phases = [
        PhaseConfig(
            sht_file=f,
            bandwidth=int(config.bandwidth),
            normed=bool(config.normed),
            refine=bool(config.refine),
            circmask=int(config.circmask),
            gausbckg=bool(config.gausbckg),
            nregions=int(config.nregions),
        )
        for f in files
    ]
    bcfg = BackendConfig(phases=phases)
    backend = SphericalGPUBackend(bcfg)
    # Plumb cancel hook into the backend so its batch loops can break
    # out cooperatively. Backend uses ``CancelledIndexingError`` from
    # this module to signal the cancel, which the route catches and
    # turns into task status "cancelled".
    if cancel_check is not None:
        backend.set_cancel_check(cancel_check)
    _progress(
        f"Spherical-GPU: device={backend.runtime.device_name}, "
        f"free_memory={backend.runtime.free_memory_gb:.1f} GB",
        0.05,
    )
    # Fail loud at start if VRAM has been eaten by a prior run — the
    # user probably hit Stop on a Dict-GPU run that left ~8 GB cached
    # and 0.0 GB free here means we'd run at batch=1 / ~0 pat/s. Better
    # to bail than silently grind.
    if (
        backend.runtime.cuda_available
        and backend.runtime.free_memory_gb < 1.0
    ):
        raise RuntimeError(
            f"Spherical-GPU: only "
            f"{backend.runtime.free_memory_gb*1024:.0f} MiB free VRAM on "
            f"{backend.runtime.device_name} — a previous indexing run is "
            f"still holding the GPU. Restart the backend (python "
            f"start_app.py) to release memory, or run the same indexing "
            f"on CPU."
        )

    # Two indexing paths:
    #   - ROI mode (selection_mask given and not all-True): load patterns
    #     via kikuchipy, fancy-index by the mask, and run backend.index_array
    #     on just the selected stack. Wall-time scales with n_selected.
    #   - Full-image mode (mask is None or all-True): stream straight from
    #     the H5 through backend.index_h5 — slightly cheaper than the ROI
    #     path because it skips the fancy-indexing copy.
    roi_mode = selection_mask is not None and not bool(selection_mask.all())

    # When the viewer modified the patterns in-memory (frame averaging /
    # background removal / autocontrast), index THOSE patterns — otherwise
    # the edits are silently ignored because we read straight from the file.
    processed = _get_processed_patterns_if_dirty(selection_mask, roi_mode)
    if processed is not None:
        _progress(
            "Spherical-GPU: patterns source = PROCESSED in-memory signal "
            "(viewer pre-processing applied, not the raw file)."
        )

    n_rows_hint = int(detector_params.get("n_rows", 0) or 0)
    n_cols_hint = int(detector_params.get("n_cols", 0) or 0)
    n_total_hint = n_rows_hint * n_cols_hint or None

    if roi_mode:
        # Validate mask shape against the detector grid up front — fail loud
        # rather than silently indexing the wrong region.
        if n_rows_hint > 0 and n_cols_hint > 0 and selection_mask.shape != (
            n_rows_hint, n_cols_hint
        ):
            raise ValueError(
                f"selection_mask shape {selection_mask.shape} does not match "
                f"detector grid ({n_rows_hint}, {n_cols_hint})"
            )
        n_selected = int(selection_mask.sum())
        n_total_for_log = n_total_hint or int(selection_mask.size)
        if processed is not None:
            # Viewer pre-processing (frame averaging / background removal)
            # modified the in-memory signal; index THOSE patterns, not the
            # stale raw file. `processed` already holds the ROI subset.
            patterns_roi = processed
            _progress(
                f"Spherical-GPU: ROI mode — {patterns_roi.shape[0]} processed "
                f"in-memory patterns (viewer pre-processing applied)...",
                0.08,
            )
        else:
            _progress(
                f"Spherical-GPU: ROI mode — loading {n_selected} of "
                f"{n_total_for_log} patterns from H5...",
                0.08,
            )
            # Load only the selected patterns directly via h5py. We *cannot*
            # use kikuchipy.load here because the Oxford reader trips on H5OINA
            # files with non-standard metadata (e.g. missing 'binning') and
            # because we already do dataset discovery elsewhere — mirror the
            # logic from Tier1Indexer._open_pattern_dataset so EDAX H5 and
            # H5OINA both work. h5py supports integer fancy-indexing along
            # axis 0, so we only read the n_selected patterns we actually need.
            import h5py
            with h5py.File(h5_path, "r") as f:
                candidates = [
                    "1/EBSD/Data/Processed Patterns",
                    "1/EBSD/Data/Patterns",
                    "1/EBSD/Data/Raw Patterns",
                ]
                for top_name in list(f.keys()):
                    top = f[top_name]
                    if not isinstance(top, h5py.Group):
                        continue
                    for sub in ("EBSD/Data/Pattern", "EBSD/Data/Patterns"):
                        full = f"{top_name}/{sub}"
                        if full in f and isinstance(f[full], h5py.Dataset):
                            candidates.append(full)
                dset = None
                for ds_path in candidates:
                    if ds_path in f and isinstance(f[ds_path], h5py.Dataset):
                        if f[ds_path].ndim == 3:
                            dset = f[ds_path]
                            break
                if dset is None:
                    raise FileNotFoundError(
                        f"No EBSD pattern dataset in {h5_path} (tried: {candidates})"
                    )
                n_total_h5 = int(dset.shape[0])
                mask_flat = selection_mask.ravel()
                if mask_flat.size != n_total_h5:
                    raise ValueError(
                        f"selection_mask flat size {mask_flat.size} does not "
                        f"match H5 pattern count {n_total_h5} in dataset "
                        f"{ds_path!r}"
                    )
                # Sorted int-array indexing is the most efficient h5py path
                # for chunked datasets (one seek per chunk, no fancy mask copy).
                indices = np.where(mask_flat)[0]
                patterns_roi = np.asarray(dset[indices])  # (n_selected, ph, pw)
        _progress(
            f"Spherical-GPU: indexing {patterns_roi.shape[0]} patterns "
            f"across {n_phases} phase{'s' if n_phases > 1 else ''}...",
            0.10,
        )
    else:
        if n_total_hint:
            _progress(
                f"Spherical-GPU: indexing {n_total_hint} patterns "
                f"across {n_phases} phase{'s' if n_phases > 1 else ''}...",
                0.10,
            )
        else:
            _progress("Spherical-GPU: indexing all patterns...", 0.10)

    # Per-phase wall-clock tracker for reporting pat/s rate. The backend's
    # built-in callback reports "[phase i/N name] Indexed X/Y patterns" for
    # multi-phase and "Indexed X/Y patterns" for single. We parse that
    # message format and append the measured rate.
    _phase_t0 = {"i": -1, "t": _time.perf_counter(), "last_n": 0}

    def _bcb(msg, frac):
        # frac is 0..1 across the whole multi-phase run (per the backend's
        # _index_with_phases wrapping). Map to 0.10..0.95 of overall.
        overall = 0.10 + 0.85 * float(frac)

        # Detect phase index from "[phase i/N name] ..." prefix.
        ph_idx = -1
        if msg.startswith("[phase ") and "]" in msg:
            try:
                inner = msg[len("[phase "):msg.index("]")]
                ph_idx = int(inner.split("/")[0])
            except Exception:
                pass

        # Reset per-phase timer when we see a new phase index.
        nonlocal_state = _phase_t0
        if ph_idx != nonlocal_state["i"]:
            nonlocal_state["i"] = ph_idx
            nonlocal_state["t"] = _time.perf_counter()
            nonlocal_state["last_n"] = 0

        # Try to extract "Indexed X/Y" and compute pat/s.
        rate_str = ""
        try:
            if "Indexed " in msg and "/" in msg:
                tail = msg.split("Indexed ", 1)[1]
                num = tail.split("/", 1)[0].strip()
                n_done = int(num)
                dt = max(_time.perf_counter() - nonlocal_state["t"], 1e-3)
                rate = n_done / dt if dt > 0 else 0.0
                if rate > 0:
                    rate_str = f"  ({rate:.0f} pat/s)"
        except Exception:
            pass

        _progress(msg + rate_str, overall)

    # Cancel between ROI load and the (expensive) backend build/index.
    if cancel_check is not None and cancel_check():
        raise CancelledIndexingError("Spherical-GPU indexing cancelled before run")

    # If the user has the circular signal mask active in the viewer, zero
    # out the detector corners on the patterns before SHT correlation. This
    # only helps for the index_array paths — index_h5 reads straight from
    # disk and would need backend changes. In practice users who enable the
    # mask have also run autocontrast / dynamic-BG, which already zeroed the
    # corners, so the h5 path is fine for the common case.
    _sp_include = None
    try:
        from backend.api.routes.ebsd_viewer import get_active_include_mask
        _sp_include = get_active_include_mask()
    except Exception:
        logger.debug("Could not fetch signal mask for spherical-GPU", exc_info=True)

    def _apply_mask(patterns):
        if _sp_include is None:
            return patterns
        # patterns: (N, H, W). Multiply by 2D mask broadcast across N.
        if patterns.shape[-2:] != _sp_include.shape:
            return patterns  # mask shape mismatch — skip silently rather than crash
        return patterns * _sp_include.astype(patterns.dtype)

    masters_meta = []
    # Pseudo-symmetry resolution state (see the resolver block below). The
    # in-memory patterns are reused for the Hough anchor; index_h5 streams from
    # disk so resolution is skipped there (no in-memory patterns).
    _sp_patterns_for_resolve = None
    _resolved_eulers = None
    try:
        if roi_mode:
            # patterns_roi is the processed in-memory subset when the signal
            # was dirty, otherwise the H5-read subset (see the ROI block).
            _sp_patterns_for_resolve = _apply_mask(patterns_roi)
            result = backend.index_array(
                patterns=_sp_patterns_for_resolve,
                detector_params=detector_params,
                progress_callback=_bcb,
                phase_weights=phase_weights,
            )
        elif processed is not None:
            # Full grid, viewer pre-processing applied.
            _sp_patterns_for_resolve = _apply_mask(processed)
            result = backend.index_array(
                patterns=_sp_patterns_for_resolve,
                detector_params=detector_params,
                progress_callback=_bcb,
                phase_weights=phase_weights,
            )
        else:
            result = backend.index_h5(
                h5_path=h5_path,
                detector_params=detector_params,
                progress_callback=_bcb,
                phase_weights=phase_weights,
            )
        # Snapshot only the small metadata fields we need to build the
        # CrystalMap's PhaseList. CRITICAL: do NOT keep references to the
        # `ix.master` objects themselves — they own the (~GB) harmonics
        # tensors on the GPU. Holding a Python ref keeps those tensors
        # alive across backend.invalidate() in `finally`, which leaks
        # VRAM and starves the next indexing run (a previous version of
        # this code held the live master objects and produced a ~8.5GB
        # leak after a single spherical-GPU run on a 12GB card).
        masters_meta = [
            {
                "formula": getattr(ix.master, "formula", None),
                "point_group": getattr(ix.master, "point_group", None),
                # z_rot is the authoritative "spherical-unreliable" determinant
                # (== 2 → flat cc); snapshot it so we never re-touch the GPU
                # master after invalidate().
                "z_rot": getattr(ix.master, "z_rot", None),
            }
            for ix in backend._indexers
        ]

        # NOTE: pseudo-symmetry resolution runs AFTER the `finally` below — it
        # uses Hough (CPU), not the GPU indexer, so we free the GPU first.
    finally:
        # Drop indexer-internal tensors and release the PyTorch cache so
        # the next indexing run gets the full free-VRAM budget back.
        try:
            backend.invalidate()
        except Exception:
            pass
        _release_cuda_cache()
    n_indexed = int(result.euler_xyz.shape[0])
    rate_total = (n_indexed * n_phases) / max(result.runtime_seconds, 1e-3)
    _progress(
        f"Spherical-GPU: indexing complete — {n_indexed} patterns × "
        f"{n_phases} phase{'s' if n_phases > 1 else ''} in "
        f"{result.runtime_seconds:.1f}s ({rate_total:.0f} pat/s effective).",
        0.90,
    )

    # --- Pseudo-symmetry / low-symmetry resolution (GPU already freed) --------
    # The SHT SO(3) correlation can't index z_rot==2 masters (cubic approximants
    # m-3/23 with full cubic band geometry, cubic -43m, and orthorhombic
    # mmm/222/mm2): it lands on a WRONG basin (the green-vs-purple IPF +
    # poor-pattern-match bug). resolve_eulers_multiphase replaces those
    # orientations with Hough band-geometry indexing PER PHASE — for each
    # z_rot==2 phase it re-indexes ONLY that phase's assigned pixels (phase_id),
    # so MULTI-PHASE runs are now handled too (the phase assignment itself is
    # reliable — the spherical score ranks the right phase even with a wrong
    # orientation; measured on 7050). It is a strict NO-OP (raw eulers) for
    # high-symmetry-only phase sets. Uses Hough (CPU) + the in-memory patterns,
    # NOT the GPU indexer — so it runs here, after the `finally` freed the GPU.
    # Streams progress; fail-safe per phase. Provenance is surfaced in the result
    # metadata so the Pattern Match view + Phase Test can label the source.
    # On the streaming (index_h5) full-map path there are no in-memory patterns,
    # so the resolver below (and variant unification) would be skipped and a
    # z_rot==2 intermetallic would keep its wrong spherical pseudo-variant on
    # every pixel. Materialise the pattern stack from the H5 now (fail-safe:
    # None on nothing-to-resolve / too-large / read error) so the auto-Hough
    # correction runs on a plain full-map index too — no ROI / viewer edit needed.
    if _sp_patterns_for_resolve is None:
        _streamed = _materialize_patterns_for_resolution(
            h5_path, masters_meta, result.phase_id.numpy(), progress=_progress)
        if _streamed is not None:
            _sp_patterns_for_resolve = _apply_mask(_streamed)

    _orientation_source = "spherical"
    _orientation_source_reason = ""
    _resolved_phase_ids = set()
    if _sp_patterns_for_resolve is not None:
        try:
            from backend.spherical_gpu.pipeline.resolution import resolve_eulers_multiphase
            from backend.spherical_gpu.pseudosym import is_pseudosymmetric
            _raw_eul = result.euler_xyz.numpy().astype(np.float64)
            _phase_id_resolve = result.phase_id.numpy().reshape(-1)
            _cif_paths = [_resolve_cif_for_sht(f) for f in files]
            _res_eul, _mp_info = resolve_eulers_multiphase(
                _sp_patterns_for_resolve, _raw_eul, _phase_id_resolve,
                masters_meta, _cif_paths, detector_params, progress=_progress)
            _resolved_phase_ids = _mp_info.get("resolved_phase_ids", set())
            if _res_eul is not None:
                _resolved_eulers = _res_eul
                _orientation_source = "hough"
                _nfb = int(_mp_info.get("n_fallback", 0) or 0)
                _fb_note = (f" ({_nfb} px kept spherical where Hough failed)"
                            if _nfb else "")
                _labels = _mp_info.get("labels", [])
                if len(_labels) == 1:
                    _lbl, _pg = _labels[0]
                    if is_pseudosymmetric(_pg):
                        _orientation_source_reason = (
                            f"Pseudo-symmetry (point group {_pg}): the SHT-spherical "
                            f"SO(3) correlation lands on a wrong pseudo-symmetric "
                            f"variant for this cubic-approximant phase, so the "
                            f"orientations are taken from Hough band-geometry "
                            f"indexing" + _fb_note + ".")
                    else:
                        _orientation_source_reason = (
                            f"Point group {_pg} (z-rotational symmetry order 2): the "
                            f"SHT-spherical SO(3) correlation cannot form a sharp "
                            f"orientation peak for this class, so the orientations "
                            f"are taken from Hough band-geometry indexing"
                            + _fb_note + ".")
                else:
                    _names = ", ".join(_l for _l, _ in _labels)
                    _orientation_source_reason = (
                        f"Low-symmetry phases {_names} (z-rotational symmetry order "
                        f"2): the SHT-spherical SO(3) correlation cannot form a "
                        f"sharp orientation peak for them, so their orientations are "
                        f"taken from Hough band-geometry indexing" + _fb_note + ".")
        except Exception:
            logger.warning(
                "Multi-phase pseudo-symmetry resolution failed; using raw "
                "spherical orientations", exc_info=True)
            _resolved_eulers = None

    # --- Map-wide pseudo-symmetry variant unification ------------------------
    # Hough is variant-blind (band geometry has the holohedral symmetry), so
    # the substituted orientations can be per-pixel arbitrary pseudo-variants →
    # IPF salt-and-pepper inside physical grains. Unify per phase: segment
    # grains modulo the SUPERGROUP, decide the variant per grain by aggregated
    # render-NCC with the spatial-coherence twin-protection policy, rescue
    # Hough-failure orphans. Fail-safe: any problem keeps the resolver output.
    eulers = (_resolved_eulers if _resolved_eulers is not None
              else result.euler_xyz.numpy().astype(np.float64))
    _vu_reports = None
    if _resolved_phase_ids and _sp_patterns_for_resolve is not None:
        try:
            from backend.spherical_gpu.pipeline.variant_unification import (
                unify_after_hough_resolve,
            )
            _progress("Pseudo-symmetry: unifying variant speckle per grain "
                      "(render-NCC verified)...", 0.93)
            _eul_u, _vu_reports = unify_after_hough_resolve(
                eulers, result.phase_id.numpy().reshape(-1),
                _sp_patterns_for_resolve, files, masters_meta,
                detector_params, selection_mask, roi_mode,
                _resolved_phase_ids, progress=lambda m: _progress(m, 0.95))
            if _eul_u is not None:
                eulers = _eul_u
                _tot_flip = sum(r.get("n_flipped_units", 0)
                                for r in _vu_reports.values() if isinstance(r, dict))
                _tot_amb = sum(r.get("n_ambiguous", 0)
                               for r in _vu_reports.values() if isinstance(r, dict))
                _tot_resc = sum(r.get("n_rescued", 0)
                                for r in _vu_reports.values() if isinstance(r, dict))
                _progress(
                    f"Pseudo-symmetry: variant unification done — "
                    f"{_tot_flip} unit(s) unified/flipped, {_tot_amb} ambiguous, "
                    f"{_tot_resc} px rescued", 0.96)
                _orientation_source_reason += (
                    " Variant speckle was unified per grain (supergroup "
                    "segmentation + aggregated render-NCC; coherent twin "
                    "domains only flipped on a clear margin).")
        except Exception:
            logger.warning("Variant unification failed; keeping resolver "
                           "orientations", exc_info=True)

    _progress("Spherical-GPU: building CrystalMap...", 0.97)

    # --- Build CrystalMap (matches the EMSphInx-flow output shape) -----------
    n_points = eulers.shape[0]
    rotations = Rotation.from_euler(eulers)

    n_rows = int(detector_params.get("n_rows") or 0)
    n_cols = int(detector_params.get("n_cols") or 0)

    if roi_mode:
        # In ROI mode the rotations array covers only the selected pixels.
        # Use argwhere(mask) to place them back at their original (row, col)
        # positions — matches the Hough partial-selection path above.
        selected_indices = np.argwhere(selection_mask)  # (N, 2) of (row, col)
        xs = selected_indices[:, 1].astype(float)
        ys = selected_indices[:, 0].astype(float)
    else:
        if n_rows * n_cols != n_points:
            n_rows = 1
            n_cols = n_points
        xs = np.tile(np.arange(n_cols), n_rows)[:n_points]
        ys = np.repeat(np.arange(n_rows), n_cols)[:n_points]

    # Build a Phase per master pattern (so multi-phase runs label correctly).
    #
    # IMPORTANT: do NOT silently fall back to point_group="1" (triclinic)
    # when the master pattern lacks one. That used to render IPF maps as
    # near-monochrome grey because IPFColorKeyTSL on the trivial fundamental
    # zone collapses every orientation to roughly the same RGB. The user
    # then saw a flat grey rectangle and had no clue what went wrong.
    # Fail loud here — the SHT loader must provide point_group.
    #
    # Defensive fallback: if the metadata snapshot is empty (shouldn't
    # happen given the snapshot above, but guards against future
    # invalidate-order changes inside the backend), re-read the masters
    # onto CPU device so we don't pull harmonics tensors back into VRAM
    # after invalidate() just released them. Drop the reference right
    # after extracting metadata so Python GC can release the tensors.
    if not masters_meta:
        from backend.spherical_gpu.pipeline.sht_io import read_sht_master
        for ph in backend.config.phases:
            m = read_sht_master(ph.sht_file, device="cpu")
            masters_meta.append({
                "formula": getattr(m, "formula", None),
                "point_group": getattr(m, "point_group", None),
                "z_rot": getattr(m, "z_rot", None),
            })
            del m

    # Phases that actually won pixels — so we never warn about a z_rot==2 phase
    # that got zero pixels (nothing about it is wrong). None → couldn't tell, warn.
    try:
        _phase_ids_present = {int(x) for x in np.unique(result.phase_id.numpy())}
    except Exception:
        _phase_ids_present = None
    phase_objs = []
    _lowsym_broken = []
    from backend.spherical_gpu.pseudosym import spherical_unreliable as _sph_unrel
    for _i, m in enumerate(masters_meta):
        pg = m.get("point_group")
        if not pg:
            raise RuntimeError(
                f"Master pattern for phase {m.get('formula')!r} has no point_group. "
                "Spherical-GPU indexing cannot build a CrystalMap with proper "
                "IPF colouring from a master that lacks crystallographic "
                "symmetry. Regenerate the SHT with a CIF that carries a valid "
                "space group, or fix the SHT loader to propagate point_group."
            )
        phase_objs.append(Phase(name=m.get("formula") or "Phase", point_group=pg))
        # The SHT-spherical SO(3) correlation can't index z_rot==2 masters
        # (cubic m-3/23/-43m + orthorhombic mmm/222/mm2) — it lands in a wrong
        # basin. The in-memory path AUTO-CORRECTS these above PER PHASE (Hough
        # orientations for each z_rot==2 phase's pixels). Flag only the
        # spherical-unreliable phases that won pixels but the resolver did NOT
        # cover (the index_h5 streaming path with no in-memory patterns, or a
        # phase where Hough failed on every assigned pixel) so the user knows to
        # use Hough. A z_rot==2 phase that won zero pixels is not flagged.
        if (_sph_unrel(m.get("z_rot"), pg) and (_i + 1) not in _resolved_phase_ids
                and (_phase_ids_present is None or (_i + 1) in _phase_ids_present)):
            _lowsym_broken.append(str(m.get("formula") or pg))
    if _lowsym_broken:
        logger.warning(
            "⚠ SPHERICAL INDEXING is UNRELIABLE for low-symmetry phase(s) %s "
            "(point group mmm / m-3 / 23, z_rot=2): the SHT-spherical correlation "
            "mis-indexes them and the automatic Hough substitution did not run for "
            "this run (multi-phase, or streamed from disk). Use HOUGH indexing for "
            "these phases, or run them single-phase with in-memory patterns so the "
            "orientations are auto-taken from Hough.",
            _lowsym_broken,
        )
    # Build PhaseList with EXPLICIT 1-indexed ids to match the CPU/.ang
    # convention. _attach_indexing_metadata in backend/api/routes/indexing.py
    # keys sht_paths_by_phase with `i + 1` (1-indexed), so the Pattern Match
    # dialog looks up SHTs by xmap.phase_id values 1, 2, ... — if the xmap
    # uses 0-indexed phase_ids the lookup misses and the dialog can't
    # render the best-match simulated pattern.
    phase_list = PhaseList(
        phases=phase_objs,
        ids=list(range(1, len(phase_objs) + 1)),
    )
    # IndexResult.phase_id is already 1-indexed from the backend, matching
    # the PhaseList ids above. Don't shift to 0 — that's what was breaking
    # the Pattern Match dialog (phase_id=0 sht-map lookup with no key 0).
    phase_id_arr = result.phase_id.numpy().astype(np.int32)

    # Mirror score into "ci" so phase_map's CI-threshold cleanup (which reads
    # xmap.prop["ci"]) lights up on spherical-gpu results — without this, the
    # CI slider in the Phase Map page silently no-ops on this method.
    _score_np = result.score.numpy()
    prop = {
        "scores": _score_np,
        "ci": _score_np,
        "mad": np.full(n_points, np.nan, dtype=np.float64),
    }
    # Reference-frame correction — shared with the EMSphInx path.
    # SOURCE_INTRINSIC_OFFSET["spherical_gpu"] is identity under CS1, so this
    # is currently a passthrough for the orientations themselves; the call
    # routes SP-GPU through the central function for vendor-generality, the
    # fail-loud unverified-Bruker guard, and future-proofing.
    from backend.api.services.orientation_frame import apply_reference_frame_correction
    rotations = apply_reference_frame_correction(
        rotations,
        vendor=detector_params.get("source_vendor", "oxford"),
        header_meta=detector_params.get("frame_header_meta", {}),
        source="spherical_gpu",
    )
    xmap = CrystalMap(
        rotations=rotations,
        phase_id=phase_id_arr,
        x=xs,
        y=ys,
        phase_list=phase_list,
        prop=prop,
    )
    if selection_mask is None:
        selection_mask = np.ones((n_rows, n_cols), dtype=bool)
    if roi_mode and (n_rows == 0 or n_cols == 0):
        # detector_params didn't carry grid hints — fall back to the mask shape
        # so downstream callers can still reconstruct the original frame.
        n_rows, n_cols = selection_mask.shape
    _progress(f"Spherical-GPU: done — {rate_total:.0f} pat/s effective", 1.0)
    return IndexingResult(
        xmap=xmap,
        selection_mask=selection_mask,
        original_shape=(n_rows, n_cols),
        method=IndexingMethod.SPHERICAL,
        confidence_scores=result.score.numpy(),
        metadata={
            "backend": "spherical_gpu",
            "bandwidth": int(config.bandwidth),
            "n_patterns": n_points,
            "n_phases": n_phases,
            "runtime_seconds": float(result.runtime_seconds),
            "rate_pat_per_sec": float(rate_total),
            "device": str(backend.runtime.device),
            "device_name": backend.runtime.device_name,
            "orientation_source": _orientation_source,
            "orientation_source_reason": _orientation_source_reason,
            # EDS chemistry prior: pixels whose winning phase flipped vs the
            # unweighted argmax (0 when the prior is off — see Task 3/8).
            "eds_n_adjusted": int(getattr(result, "n_adjusted", 0)),
            # Map-wide variant unification provenance (per resolved phase):
            # grain count, flipped/ambiguous/rescued totals + per-grain
            # decisions (mode, margin, centroid) for the UI / diagnostics.
            "variant_unification": _vu_reports,
        },
    )


def _sweep_stale_preproc_temps(temp_dir: Path) -> None:
    """Delete leftover ``_preproc_*.h5`` temp files from earlier crashed runs."""
    try:
        for stale in temp_dir.glob("_preproc_*.h5"):
            try:
                stale.unlink()
            except Exception:
                pass
    except Exception:
        pass


def _make_processed_temp_h5(h5_path: str, progress_callback=None) -> Optional[str]:
    """Write a temp H5 with the viewer-processed patterns, if the signal is dirty.

    The external EMSphInx binary can only read files, so when the active EBSD
    signal was modified in the viewer (frame averaging / background removal /
    autocontrast) we copy the original H5 and overwrite its pattern dataset
    in-place with the processed patterns. Returns the temp path, or ``None``
    when the signal is unmodified (callers then use the original file).
    The caller MUST delete the returned file.
    """
    try:
        from backend.api.routes.ebsd_viewer import (
            is_active_signal_dirty, _get_active_signal,
        )
    except Exception:
        return None
    if not is_active_signal_dirty():
        return None
    if not h5_path or not Path(h5_path).exists():
        return None
    signal = _get_active_signal()
    data = getattr(signal, "data", None)
    if data is None:
        return None
    data = np.asarray(data)
    if data.ndim != 4:
        return None
    sig_h, sig_w = int(data.shape[2]), int(data.shape[3])
    flat = data.reshape(-1, sig_h, sig_w)

    import os as _os
    import shutil as _shutil
    import time as _t
    import h5py as _h5py

    def _p(msg):
        if progress_callback:
            try:
                progress_callback(msg)
            except Exception:
                pass

    temp_h5 = None
    try:
        temp_dir = _get_emsoft_data_path() / "SphericalIndexing" / "_preproc"
        temp_dir.mkdir(parents=True, exist_ok=True)
        _sweep_stale_preproc_temps(temp_dir)
        temp_h5 = temp_dir / (
            f"_preproc_{Path(h5_path).stem}_{_os.getpid()}_{int(_t.time())}.h5"
        )
        _p("Spherical: viewer pre-processing detected — writing processed "
           "patterns to a temporary file for EMSphInx...")
        _shutil.copy2(h5_path, temp_h5)
        ds_path = _get_pattern_dataset_path(str(temp_h5))
        with _h5py.File(temp_h5, "r+") as f:
            dset = f[ds_path]
            if int(np.prod(dset.shape[:-2]) if dset.ndim >= 3 else dset.shape[0]) \
                    != flat.shape[0]:
                raise ValueError(
                    f"processed pattern count {flat.shape[0]} does not match "
                    f"file dataset {ds_path!r} shape {dset.shape}"
                )
            orig_dtype = dset.dtype
            if np.issubdtype(orig_dtype, np.integer):
                # Processed patterns may be float (background removal yields
                # negatives) — rescale globally into the integer range so the
                # dataset's dtype is preserved (EMSphInx re-normalises anyway).
                info = np.iinfo(orig_dtype)
                pmin, pmax = float(flat.min()), float(flat.max())
                if pmax > pmin:
                    scaled = (flat.astype(np.float64) - pmin) / (pmax - pmin) * info.max
                else:
                    scaled = np.zeros_like(flat, dtype=np.float64)
                out = scaled.astype(orig_dtype)
            else:
                out = flat.astype(orig_dtype)
            # Same shape + dtype -> in-place write, no file-size bloat.
            dset[...] = out.reshape(dset.shape)
        _p(f"Spherical: patterns source = PROCESSED in-memory signal "
           f"(temp file {temp_h5.name}).")
        return str(temp_h5)
    except Exception as e:
        _p(f"Spherical: could not build processed temp H5 ({e}); "
           f"indexing the raw file instead.")
        try:
            if temp_h5 is not None and temp_h5.exists():
                temp_h5.unlink()
        except Exception:
            pass
        return None


def spherical_index_patterns(
    h5_path: str,
    config: IndexingConfig,
    detector_params: Dict,
    selection_mask: Optional[np.ndarray] = None,
    progress_callback=None,
) -> IndexingResult:
    """Run Spherical Indexing via EMSphInx IndexEBSD.

    Thin wrapper over :func:`_spherical_index_patterns_impl`. When the active
    EBSD signal was modified in the viewer (frame averaging / background
    removal / autocontrast), it first writes the processed patterns to a
    temporary H5 so the external EMSphInx binary indexes those instead of the
    stale raw file. The temp file is always deleted afterwards.
    """
    temp_h5 = _make_processed_temp_h5(h5_path, progress_callback)
    try:
        return _spherical_index_patterns_impl(
            h5_path=temp_h5 or h5_path,
            config=config,
            detector_params=detector_params,
            selection_mask=selection_mask,
            progress_callback=progress_callback,
        )
    finally:
        if temp_h5:
            try:
                Path(temp_h5).unlink()
            except Exception:
                pass


def _spherical_index_patterns_impl(
    h5_path: str,
    config: IndexingConfig,
    detector_params: Dict,
    selection_mask: Optional[np.ndarray] = None,
    progress_callback=None,
) -> IndexingResult:
    """Run Spherical Indexing via EMSphinx IndexEBSD.

    Parameters
    ----------
    h5_path : str
        Path to the Oxford H5OINA experimental pattern file.
    config : IndexingConfig
        Must have sht_file set.
    detector_params : dict
        Detector geometry parameters (see generate_emsphinx_nml docs).
    selection_mask : np.ndarray, optional
        Boolean mask (n_rows, n_cols). None = index all.
    progress_callback : callable, optional
        Called with progress strings.

    Returns
    -------
    IndexingResult

    Raises
    ------
    FileNotFoundError
        If IndexEBSD binary or SHT file not found.
    RuntimeError
        If IndexEBSD execution fails.
    """
    def _progress(msg, pct=None):
        if progress_callback:
            progress_callback(msg, pct)
        logger.info(msg)

    # Validate prerequisites
    import sys as _sys_check
    _progress("Spherical: locating IndexEBSD binary...", 0.02)
    indexebsd = _find_indexebsd()
    if indexebsd is None:
        if _sys_check.platform == "win32":
            raise FileNotFoundError(
                "IndexEBSD binary not found. WSL is required for spherical indexing on Windows. "
                "Install EMSphInx inside WSL: https://github.com/EMsoft-org/EMSphInx"
            )
        raise FileNotFoundError(
            "IndexEBSD binary not found. Install EMSphInx: https://github.com/EMsoft-org/EMSphInx"
        )
    _progress(f"Spherical: IndexEBSD found at {indexebsd}", 0.04)

    sht_path = Path(config.sht_file) if config.sht_file else None
    if sht_path is None or not sht_path.exists():
        raise FileNotFoundError(
            f"SHT master file not found: {config.sht_file!r}. "
            "Generate an SHT master pattern in the Simulation module first."
        )

    if not Path(h5_path).exists():
        raise FileNotFoundError(f"Experimental file not found: {h5_path}")

    n_rows = detector_params.get('n_rows', 0)
    n_cols = detector_params.get('n_cols', 0)

    if selection_mask is None:
        selection_mask = np.ones((n_rows, n_cols), dtype=bool)

    # Detect rectangular crop from selection mask to avoid indexing all pixels
    crop_region = None
    if not selection_mask.all():
        crop_region = _detect_rectangular_crop(selection_mask)
        if crop_region is not None:
            r0, r1, c0, c1 = crop_region
            crop_rows = r1 - r0
            crop_cols = c1 - c0
            _progress(
                f"Rectangular selection detected: [{r0}:{r1}, {c0}:{c1}] "
                f"({crop_rows}×{crop_cols} = {crop_rows * crop_cols} pixels)"
            )
        else:
            _progress(
                f"Non-rectangular mask: will index all {n_rows}×{n_cols} "
                f"pixels and filter afterwards"
            )

    # Create output directory
    emdata = _get_emsoft_data_path()
    output_dir = emdata / "SphericalIndexing"
    output_dir.mkdir(parents=True, exist_ok=True)

    _progress(f"Spherical: generating NML configuration (bandwidth={config.bandwidth})...", 0.10)

    # Generate NML (includes HDF5 conversion if needed)
    try:
        with timed_step("Spherical: generate NML"):
            nml_path = generate_emsphinx_nml(
                config, h5_path, detector_params, output_dir,
                progress_callback=progress_callback,
                crop_region=crop_region,
            )
    except Exception as e:
        import traceback
        raise RuntimeError(
            f"NML generation failed: {e}\n{traceback.format_exc()}"
        ) from e

    # Run IndexEBSD with real-time output streaming
    _progress("Spherical: running IndexEBSD...", 0.30)
    _progress(f"  binary : {indexebsd}")
    _progress(f"  SHT    : {config.sht_file}")
    _progress(f"  patterns: {h5_path}")
    _progress(f"  NML    : {nml_path}")

    try:
        import sys as _sys

        def _to_wsl(win_path) -> str:
            """Convert Windows path to WSL /mnt/ path."""
            p = str(win_path).replace('\\', '/')
            if len(p) >= 2 and p[1] == ':':
                return f"/mnt/{p[0].lower()}{p[2:]}"
            return p

        if _sys.platform == "win32":
            # On Windows, run IndexEBSD via WSL.
            nml_wsl = _to_wsl(nml_path)
            outdir_wsl = _to_wsl(output_dir)
            cmd = _build_indexebsd_wsl_cmd(outdir_wsl, indexebsd, nml_wsl)
        else:
            cmd = [indexebsd, str(nml_path)]

        popen_kwargs = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Only set cwd on non-WSL (WSL handles cd in bash command)
        if _sys.platform != "win32":
            popen_kwargs['cwd'] = str(output_dir)

        with timed_step("Spherical: IndexEBSD execution"):
            proc = subprocess.Popen(cmd, **popen_kwargs)

            # Stream stdout in real-time
            stdout_lines = []
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    stdout_lines.append(line)
                    _progress(f"  [IndexEBSD] {line}")

            proc.wait(timeout=3600)

        # Capture any remaining stderr
        stderr_text = proc.stderr.read().strip()
        if stderr_text:
            # Only show non-HDF5-diagnostic lines to avoid noise
            for line in stderr_text.splitlines():
                if 'HDF5-DIAG' not in line and '#0' not in line:
                    _progress(f"  [IndexEBSD stderr] {line}")

    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError("IndexEBSD timed out after 1 hour")

    if proc.returncode != 0:
        stderr = stderr_text if stderr_text else "No error output"
        stdout_tail = "\n".join(stdout_lines[-20:]) if stdout_lines else ""
        raise RuntimeError(
            f"IndexEBSD failed (exit code {proc.returncode}):\n"
            f"stderr: {stderr}\nstdout: {stdout_tail}"
        )

    _progress("Spherical: parsing output...", 0.85)

    # Parse output
    base_name = Path(h5_path).stem
    with timed_step("Spherical: parse output"):
        xmap = _parse_emsphinx_output(output_dir, base_name)

    if xmap is None:
        raise RuntimeError(
            f"Could not parse IndexEBSD output. Check {output_dir} for results."
        )

    # Reference-frame correction — shared with the Spherical-GPU path.
    # SOURCE_INTRINSIC_OFFSET["emsphinx"] = identity (no rotation) for all
    # vendors. The historical inline R(+90 deg, Z) was empirically refuted
    # 2026-05-21: cross-method audit on NiLowGain shows it diverges to 41 deg
    # vs Hough; identity closes the gap to 5.4 deg.
    from backend.api.services.orientation_frame import apply_reference_frame_correction
    rot_corrected = apply_reference_frame_correction(
        xmap.rotations,
        vendor=detector_params.get("source_vendor", "oxford"),
        header_meta=detector_params.get("frame_header_meta", {}),
        source="emsphinx",
    )
    try:
        xmap._rotations = rot_corrected
    except Exception:
        from orix.crystal_map import CrystalMap as _CrystalMap
        xmap = _CrystalMap(
            rotations=rot_corrected,
            phase_id=xmap.phase_id,
            x=getattr(xmap, "x", None),
            y=getattr(xmap, "y", None),
            phase_list=xmap.phases,
            prop=dict(xmap.prop),
            scan_unit=getattr(xmap, "scan_unit", "um"),
        )
    logger.info(
        "Spherical: reference-frame correction applied via orientation_frame "
        "(source=emsphinx)."
    )

    # Extract confidence scores from the xmap properties.
    # EMSphinx writes its quality metric into 'ci' (Confidence Index) in the
    # .ang output — NOT into 'scores'.  Check in order of preference:
    #   'scores' — generic key used by kikuchipy dictionary indexing
    #   'ci'     — Confidence Index from EMSphinx / EDAX .ang files  [0, 1]
    #   'fit'    — Pattern Fit (secondary, lower is better for some parsers)
    scores = None
    if hasattr(xmap, 'prop'):
        for _score_key in ('scores', 'ci', 'fit'):
            if _score_key in xmap.prop:
                _raw = np.asarray(xmap.prop[_score_key])
                scores = _raw[:, 0] if _raw.ndim == 2 else _raw
                logger.debug(f"Spherical indexing: extracted scores from xmap.prop['{_score_key}']")
                break

    # When crop_region was used, IndexEBSD already processed only the
    # selected pixels — the xmap matches the crop dimensions directly.
    # For non-rectangular masks (crop_region is None but mask is partial),
    # we must post-filter the full IndexEBSD result.
    if not selection_mask.all() and crop_region is None:
        flat_mask = selection_mask.ravel()
        selected_indices = np.where(flat_mask)[0]
        n_selected = len(selected_indices)
        _progress(f"Applying selection mask: {n_selected}/{flat_mask.size} pixels kept")

        from orix.crystal_map import CrystalMap
        from orix.quaternion import Rotation

        all_rotations = xmap.rotations.data
        all_phase_ids = xmap.phase_id

        sel_rotations = all_rotations[selected_indices]
        sel_phase_ids = all_phase_ids[selected_indices]
        if scores is not None:
            scores = scores[selected_indices]

        xmap = CrystalMap(
            rotations=Rotation(sel_rotations),
            phase_id=sel_phase_ids,
            x=np.arange(n_selected, dtype=float),
            y=np.zeros(n_selected, dtype=float),
            phase_list=xmap.phases,
        )

    _progress(f"Spherical indexing complete: {xmap.size} points indexed")

    # WSL Cleanup — WSL is only a workspace, nothing stays there.
    # Remove compat HDF5 (can be huge), NML, and result files.
    try:
        if output_dir.exists():
            for f in output_dir.iterdir():
                try:
                    f.unlink()
                except Exception:
                    pass
            _progress("WSL SphericalIndexing workspace cleaned up.")
    except Exception as e:
        logger.warning("SphericalIndexing WSL cleanup failed: %s", e)

    return IndexingResult(
        xmap=xmap,
        selection_mask=selection_mask,
        original_shape=(n_rows, n_cols),
        method=IndexingMethod.SPHERICAL,
        confidence_scores=scores,
        metadata={
            'bandwidth': config.bandwidth,
        },
    )


def merge_partial_result(
    result: IndexingResult,
    phase_list,
) -> object:
    """Create a full CrystalMap with unindexed pixels set to -1.

    For partial indexing results, this creates a complete grid where
    unindexed pixels have phase_id = -1.

    CRITICAL: Preserves exact pixel positions so results can be
    combined with the original dataset.

    Parameters
    ----------
    result : IndexingResult
        Partial indexing result with selection_mask.
    phase_list : PhaseList
        Phase list for the full CrystalMap.

    Returns
    -------
    orix.crystal_map.CrystalMap
        Full grid CrystalMap with -1 for unindexed pixels.
    """
    from orix.crystal_map import CrystalMap
    from orix.quaternion import Rotation

    n_rows, n_cols = result.original_shape
    n_total = n_rows * n_cols

    # If full indexing, return xmap as-is
    if result.selection_mask.all():
        return result.xmap

    xmap_partial = result.xmap

    # Use phases from indexing output when GUI phase_list is None
    # (e.g. Spherical Indexing doesn't require a CIF file)
    if phase_list is None:
        phase_list = xmap_partial.phases

    # Create arrays for full grid
    full_phase_ids = np.full(n_total, -1, dtype=int)
    full_rotations = np.zeros((n_total, 4), dtype=float)
    full_rotations[:, 0] = 1.0  # Identity quaternion (w=1, x=0, y=0, z=0)

    # Map partial results back to original positions
    flat_mask = result.selection_mask.ravel()
    indexed_positions = np.where(flat_mask)[0]

    partial_phase_ids = xmap_partial.phase_id
    partial_rotations = xmap_partial.rotations.data

    # Dictionary indexing returns top-N orientations per pixel (shape: n, N, 4).
    # We only need the best match (index 0) for the merged grid.
    if partial_rotations.ndim == 3:
        partial_rotations = partial_rotations[:, 0, :]

    # Ensure shapes match
    n_indexed = min(len(indexed_positions), len(partial_phase_ids))

    full_phase_ids[indexed_positions[:n_indexed]] = partial_phase_ids[:n_indexed]
    full_rotations[indexed_positions[:n_indexed]] = partial_rotations[:n_indexed]

    # Build coordinate arrays
    rows, cols = np.mgrid[0:n_rows, 0:n_cols]
    x = cols.ravel().astype(float)
    y = rows.ravel().astype(float)

    full_xmap = CrystalMap(
        rotations=Rotation(full_rotations),
        phase_id=full_phase_ids,
        x=x,
        y=y,
        phase_list=phase_list,
    )

    # Copy properties if available.
    # Use prop[key] (__getitem__) not prop.items() (raw dict.items) so that
    # is_in_data filtering is applied.  This ensures we always get n_indexed
    # values in indexed order, regardless of whether kikuchipy returned a
    # full-grid xmap (n_total rows) or a partial xmap (n_indexed rows).
    if hasattr(xmap_partial, '_prop'):
        for key in list(xmap_partial._prop.keys()):
            try:
                values = xmap_partial.prop[key]  # filtered by is_in_data → n_indexed rows
            except Exception:
                continue
            # Dictionary indexing may return 3D props (n, top_N, ...);
            # flatten to best match only
            if values.ndim == 3:
                values = values[:, 0, :]
            if values.ndim == 1:
                full_prop = np.zeros(n_total, dtype=values.dtype)
                full_prop[indexed_positions[:n_indexed]] = values[:n_indexed]
            elif values.ndim == 2:
                full_prop = np.zeros((n_total, values.shape[1]), dtype=values.dtype)
                full_prop[indexed_positions[:n_indexed]] = values[:n_indexed]
            else:
                continue  # skip unexpected shapes
            full_xmap.prop[key] = full_prop

    return full_xmap


def compute_indexing_quality_summary(result) -> dict:
    """Compute a human-readable quality summary for an IndexingResult.

    Returns
    -------
    dict with keys:
        mean_score, pct_good, pct_acceptable, method_label,
        good_threshold, acceptable_threshold, phase_fractions (dict),
        color ('green'/'orange'/'red'), summary_line (str)
    """
    summary = {
        'mean_score': None,
        'pct_good': 0.0,
        'pct_acceptable': 0.0,
        'method_label': 'CI',
        'good_threshold': 0.1,
        'acceptable_threshold': 0.05,
        'phase_fractions': {},
        'color': 'red',
        'summary_line': 'No quality data available.',
    }

    if result is None:
        return summary

    # Method-specific thresholds
    if result.method == IndexingMethod.DICTIONARY:
        summary['method_label'] = result.metadata.get('metric', 'NCC').upper()
        summary['good_threshold'] = 0.3
        summary['acceptable_threshold'] = 0.15
    elif result.method == IndexingMethod.HOUGH:
        summary['method_label'] = 'CI'
        summary['good_threshold'] = 0.1
        summary['acceptable_threshold'] = 0.05
    else:
        summary['method_label'] = 'CI'
        summary['good_threshold'] = 0.1
        summary['acceptable_threshold'] = 0.05

    # Score statistics
    scores = result.confidence_scores
    if scores is not None:
        flat = np.asarray(scores).ravel()
        valid = flat[np.isfinite(flat) & (flat > 0)]
        if len(valid) > 0:
            mean_s = float(np.mean(valid))
            pct_good = float(np.mean(valid >= summary['good_threshold']) * 100)
            pct_acc = float(np.mean(valid >= summary['acceptable_threshold']) * 100)
            summary['mean_score'] = mean_s
            summary['pct_good'] = pct_good
            summary['pct_acceptable'] = pct_acc

            if pct_good >= 60:
                summary['color'] = 'green'
            elif pct_acc >= 40:
                summary['color'] = 'orange'
            else:
                summary['color'] = 'red'

    # Phase distribution from xmap
    try:
        xmap = result.xmap
        if xmap is not None and hasattr(xmap, 'phase_id'):
            phase_ids = np.asarray(xmap.phase_id)
            mask = phase_ids >= 0
            if mask.sum() > 0:
                phases = xmap.phases if hasattr(xmap, 'phases') else None
                for pid in np.unique(phase_ids[mask]):
                    frac = float((phase_ids == pid).sum()) / float(mask.sum()) * 100
                    name = str(pid)
                    if phases is not None:
                        try:
                            name = phases[int(pid)].name
                        except Exception:
                            pass
                    summary['phase_fractions'][name] = frac
    except Exception:
        pass

    # Build summary line
    lbl = summary['method_label']
    if summary['mean_score'] is not None:
        thr = summary['good_threshold']
        pct = summary['pct_good']
        line = f"Quality: Mean {lbl}={summary['mean_score']:.3f} | {lbl}>{thr}: {pct:.0f}%"
        if summary['phase_fractions']:
            phase_str = ', '.join(
                f"{n} {v:.0f}%" for n, v in summary['phase_fractions'].items()
            )
            line += f" | Phases: {phase_str}"
        summary['summary_line'] = line

    return summary


def result_to_phase_map_data(
    xmap,
    step_x: float = 1.0,
    step_y: float = 1.0,
    title: str = "Phase Map",
):
    """Convert a CrystalMap to PhaseMapData for visualization.

    Bridge between indexing results and FEAT-12 phase map viewer.

    Parameters
    ----------
    xmap : orix.crystal_map.CrystalMap
        Indexing result (full or merged).
    step_x, step_y : float
        Pixel size in micrometers.
    title : str
        Display title.

    Returns
    -------
    PhaseMapData
        Ready for tools.phase_map_generator.render_phase_map().
    """
    from tools.phase_map_generator import extract_phase_map_from_xmap

    phase_data = extract_phase_map_from_xmap(xmap)
    phase_data.step_x = step_x
    phase_data.step_y = step_y
    phase_data.title = title
    return phase_data


@dataclass
class QuickTestResult:
    """Result of a single-pattern quick test."""
    row: int
    col: int
    phase_name: str
    ci: float                       # Confidence index from Hough
    euler_deg: Tuple[float, float, float]  # Euler angles (phi1, Phi, phi2) in degrees
    experimental: np.ndarray        # 2D pattern
    simulated: Optional[np.ndarray] = None  # Best-match simulated pattern (if available)
    ncc: Optional[float] = None     # NCC between experimental and simulated


def quick_test_single_pattern(
    signal,
    detector,
    phase_list,
    row: int,
    col: int,
    n_bands: int = 12,
    t_sigma: float = 2.0,
    r_sigma: float = 2.0,
) -> QuickTestResult:
    """Run Hough indexing on a single pattern and return results.

    This is the fast-feedback function for FEAT-20: user picks a pixel,
    gets phase + orientation + CI within seconds.

    Parameters
    ----------
    signal : kikuchipy.signals.EBSD
        Full EBSD signal.
    detector : kikuchipy.detectors.EBSDDetector
        Detector geometry with PC.
    phase_list : orix.crystal_map.PhaseList
        Phases to test against.
    row, col : int
        Pixel coordinates in the scan.
    n_bands, t_sigma, r_sigma : float
        Hough indexing parameters.

    Returns
    -------
    QuickTestResult
    """
    from ebsd_utils import prepare_reflectors, create_indexer

    # Extract pattern
    pattern = np.asarray(signal.data[row, col], dtype=np.float32)

    # Build indexer
    reflectors = prepare_reflectors(phase_list)
    indexer = create_indexer(
        detector, phase_list, reflectors,
        nBands=n_bands, tSigma=t_sigma, rSigma=r_sigma,
    )

    # Index single pattern
    from hyperspy.signals import Signal2D
    from kikuchipy.signals import EBSD as EBSD_Signal

    sig2d = Signal2D(pattern)
    ebsd = EBSD_Signal(sig2d, detector=detector)
    xmap, index_data, band_data = ebsd.hough_indexing(
        phase_list, indexer,
        return_index_data=True,
        return_band_data=True,
        verbose=0,
    )

    # Extract results
    ci = float(index_data['cm'].ravel()[0])
    phase_id = int(xmap.phase_id[0])
    phase_name = str(xmap.phases[phase_id].name) if phase_id >= 0 else "Unindexed"

    # Euler angles in degrees
    from orix.quaternion import Rotation
    rot = xmap.rotations[0]
    euler = rot.to_euler(degrees=True)
    euler_flat = np.asarray(euler).ravel()
    phi1 = float(euler_flat[0])
    Phi = float(euler_flat[1])
    phi2 = float(euler_flat[2])

    # Try to generate a simulated pattern for comparison
    simulated = None
    ncc = None
    try:
        simulated = _simulate_pattern_for_orientation(
            detector, phase_list, rot, phase_id
        )
        if simulated is not None:
            from tools.pattern_comparison import compute_ncc_scalar
            ncc = compute_ncc_scalar(pattern, simulated)
    except Exception as e:
        logger.debug(f"Could not generate simulated pattern: {e}")

    return QuickTestResult(
        row=row,
        col=col,
        phase_name=phase_name,
        ci=ci,
        euler_deg=(phi1, Phi, phi2),
        experimental=pattern,
        simulated=simulated,
        ncc=ncc,
    )


def multi_phase_quick_test(
    signal,
    detector,
    phase_list_per_phase: list,
    row: int,
    col: int,
    n_bands: int = 12,
    t_sigma: float = 2.0,
    r_sigma: float = 2.0,
    progress_callback=None,
) -> List[QuickTestResult]:
    """Run Hough indexing on a single pattern against multiple phases.

    Tests each phase independently and returns ranked results sorted by CI.

    Parameters
    ----------
    signal : kikuchipy.signals.EBSD
        Full EBSD signal.
    detector : kikuchipy.detectors.EBSDDetector
        Detector geometry with PC.
    phase_list_per_phase : list of orix.crystal_map.PhaseList
        Each entry is a PhaseList containing a single phase to test.
    row, col : int
        Pixel coordinates in the scan.
    n_bands, t_sigma, r_sigma : float
        Hough indexing parameters.
    progress_callback : callable, optional
        Called with (phase_index, total_phases, phase_name) during execution.

    Returns
    -------
    list of QuickTestResult
        Results sorted by CI (descending), best match first.
    """
    results = []
    total = len(phase_list_per_phase)

    for i, phase_list in enumerate(phase_list_per_phase):
        phase_name = "Unknown"
        try:
            phase_name = str(phase_list[0].name)
        except Exception:
            pass

        if progress_callback is not None:
            progress_callback(i, total, phase_name)

        try:
            result = quick_test_single_pattern(
                signal=signal,
                detector=detector,
                phase_list=phase_list,
                row=row,
                col=col,
                n_bands=n_bands,
                t_sigma=t_sigma,
                r_sigma=r_sigma,
            )
            results.append(result)
        except Exception as e:
            logger.warning(f"Quick test failed for phase '{phase_name}': {e}")
            # Create a failed result so the user sees all phases
            pattern = np.asarray(signal.data[row, col], dtype=np.float32)
            results.append(QuickTestResult(
                row=row, col=col,
                phase_name=phase_name,
                ci=0.0,
                euler_deg=(0.0, 0.0, 0.0),
                experimental=pattern,
                simulated=None,
                ncc=None,
            ))

    # Sort by CI descending (best match first)
    results.sort(key=lambda r: r.ci, reverse=True)
    return results


def _simulate_pattern_for_orientation(detector, phase_list, rotation, phase_id):
    """Attempt to simulate a pattern for a given orientation.

    Uses kikuchipy's geometrical simulation to generate Kikuchi band
    positions overlaid as a line image — not a full master-pattern
    projection (that would require a master file).

    Returns None if simulation is not possible.
    """
    try:
        from kikuchipy.simulations import KikuchiPatternSimulator

        phase = phase_list[phase_id]
        from diffsims.crystallography import ReciprocalLatticeVector
        from ebsd_utils import _normalize_element_labels
        # Same robustness as ebsd_utils._reflectors_for_phase: resolve messy CIF
        # element labels (oxidation states / mixed-occupancy sites) so structure
        # factors aren't silently zero, and tolerate diffsims' unimplemented
        # ``.allowed`` for primitive-hexagonal space groups (absent reflections
        # carry F≈0 and render invisibly).
        _normalize_element_labels(phase)
        ref = ReciprocalLatticeVector.from_min_dspacing(phase, 1.0)
        try:
            ref = ref[ref.allowed]
        except NotImplementedError:
            pass
        ref = ref.unique(use_symmetry=True).symmetrise()
        ref.sanitise_phase()
        ref.calculate_structure_factor()

        sim = KikuchiPatternSimulator(ref)
        geo = sim.on_detector(detector, rotation)

        # Render to image
        pat_h, pat_w = detector.shape
        img = np.zeros((pat_h, pat_w), dtype=np.float32)
        lines = geo.as_collections(zone_axes=False)
        if lines:
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            fig = Figure(figsize=(pat_w / 100, pat_h / 100), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, pat_w)
            ax.set_ylim(pat_h, 0)
            ax.set_facecolor('black')
            for lc in lines:
                ax.add_collection(lc)
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(pat_h, pat_w, 3)
            img = buf.mean(axis=2).astype(np.float32)
            plt.close(fig)
        return img
    except Exception as e:
        logger.debug(f"Geometrical simulation failed: {e}")
        return None


def export_results(xmap, path: str) -> Path:
    """Export indexing results to .ang or .ctf format.

    Uses orix's built-in save functionality which supports
    EDAX .ang and Oxford/HKL .ctf formats natively.

    Parameters
    ----------
    xmap : orix.crystal_map.CrystalMap
        Indexing result to export.
    path : str
        Output file path. Extension determines format (.ang or .ctf).

    Returns
    -------
    Path
        Path to the saved file.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    from orix.io import save
    from orix.crystal_map import CrystalMap
    from orix.quaternion import Rotation

    out = Path(path)
    if out.suffix.lower() not in ('.ang',):
        raise ValueError(
            f"Unsupported export format '{out.suffix}'. "
            "Use .ang (EDAX TSL format)."
        )

    # orix .ang/.ctf writer needs a full rectangular grid.
    # Our back-mapped xmap may be sparse (only indexed pixels with x/y coords).
    # Expand to full grid with identity rotations for unindexed pixels.
    try:
        shape = xmap.shape
        if len(shape) < 2 or shape[0] == 0 or shape[1] == 0:
            raise ValueError("xmap has no 2D grid shape")
        save(out, xmap)
    except (ValueError, IndexError):
        # Fallback: build a full-grid xmap from sparse coordinates
        logger.info("Building full-grid xmap for .ang/.ctf export...")
        if xmap.x is not None and xmap.y is not None:
            max_y = int(np.max(xmap.y)) + 1
            max_x = int(np.max(xmap.x)) + 1
        else:
            max_y, max_x = xmap.shape if len(xmap.shape) >= 2 else (xmap.size, 1)

        n_total = max_y * max_x
        # Create full grid of identity rotations
        full_rot = Rotation.identity(n_total)
        full_phase_id = np.full(n_total, -1, dtype=int)
        full_props = {}

        # Place indexed pixels into full grid
        for i in range(xmap.size):
            xi = int(xmap.x[i]) if xmap.x is not None else i % max_x
            yi = int(xmap.y[i]) if xmap.y is not None else i // max_x
            idx = yi * max_x + xi
            if 0 <= idx < n_total:
                full_rot[idx] = xmap.rotations[i]
                full_phase_id[idx] = xmap.phase_id[i]

        # Copy properties
        for key in xmap.prop:
            val = xmap.prop[key]
            if hasattr(val, '__len__') and len(val) == xmap.size:
                full_val = np.zeros(n_total, dtype=val.dtype)
                for i in range(xmap.size):
                    xi = int(xmap.x[i]) if xmap.x is not None else i % max_x
                    yi = int(xmap.y[i]) if xmap.y is not None else i // max_x
                    idx = yi * max_x + xi
                    if 0 <= idx < n_total:
                        full_val[idx] = val[i]
                full_props[key] = full_val

        ys, xs = np.mgrid[0:max_y, 0:max_x]
        full_xmap = CrystalMap(
            rotations=full_rot,
            phase_id=full_phase_id,
            x=xs.ravel().astype(float),
            y=ys.ravel().astype(float),
            phase_list=xmap.phases,
            prop=full_props,
            scan_unit="px",
        )
        save(out, full_xmap)

    logger.info(f"Exported indexing results to {out}")
    return out


# ---------------------------------------------------------------------------
# Multi-Phase Multi-Method Comparison
# ---------------------------------------------------------------------------

def extract_score_map(result: IndexingResult) -> Optional[np.ndarray]:
    """Extract a 2D score map from any IndexingResult, regardless of method.

    Returns scores as float array (n_rows, n_cols) with NaN for unindexed pixels.
    Works for Hough (CI), Dictionary (NCC top-1), and Spherical (from xmap props).
    """
    from tools.pattern_comparison import get_ncc_score_map

    # Try the existing score map extractor (works for all methods with confidence_scores)
    score_map = get_ncc_score_map(result)
    if score_map is not None:
        return score_map

    # Fallback: try to extract scores from xmap properties
    if result.xmap is not None and hasattr(result.xmap, 'prop'):
        for key in ('scores', 'ncc', 'ci', 'cm', 'fit'):
            if key in result.xmap.prop:
                vals = np.asarray(result.xmap.prop[key])
                if vals.ndim == 2:
                    vals = vals[:, 0]  # Take top-1 score
                n_rows, n_cols = result.original_shape
                n_total = n_rows * n_cols
                if vals.size == n_total:
                    return vals.reshape(n_rows, n_cols)
                # Partial result: embed into full grid
                n_selected = int(result.selection_mask.sum())
                if vals.size == n_selected:
                    full = np.full(n_total, np.nan)
                    full[result.selection_mask.ravel()] = vals
                    return full.reshape(n_rows, n_cols)
    return None


def run_single_phase_method(
    signal,
    detector,
    phase_config: PhaseConfig,
    method: IndexingMethod,
    config: ComparisonConfig,
    selection_mask: np.ndarray,
    h5_path: str = "",
    detector_params: Optional[Dict] = None,
) -> PhaseMethodResult:
    """Run indexing for one (phase, method) combination.

    Dispatches to the correct indexing function and wraps the result.
    """
    n_rows, n_cols = signal.data.shape[:2]
    original_shape = (n_rows, n_cols)

    # Build IndexingConfig from shared comparison params
    idx_config = IndexingConfig(
        method=method,
        selection_mode=config.selection_mode,
        n_bands=config.n_bands,
        t_sigma=config.t_sigma,
        r_sigma=config.r_sigma,
        metric=config.metric,
        keep_n=config.keep_n,
        bandwidth=config.bandwidth,
        normed=config.normed,
        refine=config.refine,
        nregions=config.nregions,
        circmask=config.circmask,
        gausbckg=config.gausbckg,
        # Phase 5 fix: carry the spherical backend selector through. Without
        # this, multi-phase spherical runs always fell back to EMSphInx CPU
        # because idx_config.backend defaulted to "emsphinx" regardless of
        # what the user picked.
        backend=getattr(config, "backend", "emsphinx"),
        sht_file=phase_config.sht_path,
        row_start=config.row_start,
        row_end=config.row_end,
        col_start=config.col_start,
        col_end=config.col_end,
    )

    # Get phase_list for this phase
    phase_list = phase_config.phase_list

    if method == IndexingMethod.HOUGH:
        if phase_list is None:
            raise ValueError(f"Phase '{phase_config.name}': CIF/phase_list required for Hough indexing")
        result = hough_index_patterns(signal, phase_list, detector, idx_config, selection_mask)

    elif method == IndexingMethod.DICTIONARY:
        if phase_config.dictionary is None:
            raise ValueError(f"Phase '{phase_config.name}': Dictionary signal required for Dictionary indexing")
        if phase_list is None:
            raise ValueError(f"Phase '{phase_config.name}': phase_list required for Dictionary indexing")
        result = dictionary_index_patterns(signal, phase_config.dictionary, idx_config, selection_mask)

    elif method == IndexingMethod.SPHERICAL:
        if not phase_config.sht_path:
            raise ValueError(f"Phase '{phase_config.name}': SHT file required for Spherical indexing")
        # Override sht_file with this phase's SHT path
        idx_config.sht_file = phase_config.sht_path
        # Phase 5: dispatch on backend selector. Default is "emsphinx" (the
        # WSL CPU path) to preserve existing behaviour; "spherical_gpu" runs
        # the in-process PyTorch backend.
        backend_choice = getattr(idx_config, "backend", "emsphinx")
        if backend_choice == "spherical_gpu":
            result = spherical_gpu_index_patterns(
                h5_path=h5_path,
                config=idx_config,
                detector_params=detector_params or {},
                selection_mask=selection_mask,
            )
        else:
            result = spherical_index_patterns(
                h5_path=h5_path,
                config=idx_config,
                detector_params=detector_params or {},
                selection_mask=selection_mask,
            )

    else:
        raise ValueError(f"Unknown indexing method: {method}")

    # Extract score map and compute mean score
    score_map = extract_score_map(result)
    if score_map is not None:
        valid_scores = score_map[~np.isnan(score_map)]
        mean_score = float(np.mean(valid_scores)) if valid_scores.size > 0 else 0.0
    else:
        mean_score = 0.0

    return PhaseMethodResult(
        phase_name=phase_config.name,
        method=method,
        indexing_result=result,
        mean_score=mean_score,
        score_map_2d=score_map,
    )


def compute_comparison_maps(
    comparison: ComparisonResult,
    phase_weights: Optional[np.ndarray] = None,
) -> ComparisonResult:
    """Compute best_phase_per_pixel and consensus_map from all results.

    For each pixel:
    - best_phase_per_pixel: phase with highest score across ALL (phase, method) combos
    - consensus_map: phase that wins in majority of methods (majority vote)

    Parameters
    ----------
    phase_weights : np.ndarray | None
        Optional (n_phases, n_pixels) multiplier applied to ``scores_3d``
        before the winner argmax. Row ``i`` corresponds to phase index ``i``
        (phase-id order); columns are in full-grid raveled pixel order — the
        SAME order ``score_map_2d.ravel()`` produces. When ``None`` (default)
        this function is byte-for-byte identical to its previous behaviour and
        ``ComparisonResult.n_eds_adjusted`` stays 0.
    """
    n_rows, n_cols = comparison.original_shape
    n_phases = len(comparison.phases)
    n_methods = len(comparison.methods)

    if not comparison.results:
        return comparison

    # Build 3D score array: (n_phases, n_methods, n_pixels)
    n_pixels = n_rows * n_cols
    scores_3d = np.full((n_phases, n_methods, n_pixels), np.nan)

    phase_names = [p.name for p in comparison.phases]
    method_list = comparison.methods

    for pmr in comparison.results:
        try:
            pi = phase_names.index(pmr.phase_name)
            mi = method_list.index(pmr.method)
        except ValueError:
            continue
        if pmr.score_map_2d is not None:
            scores_3d[pi, mi, :] = pmr.score_map_2d.ravel()

    # Optional EDS chemistry prior: reweight scores per phase before the
    # winner argmax. Bit-identical when phase_weights is None (block skipped).
    _base = None
    if phase_weights is not None:
        pw = np.asarray(phase_weights, dtype=np.float64)   # (P, n_pixels)
        # Unweighted winner (baseline), for the adjusted-count comparison.
        _flat0 = scores_3d.reshape(n_phases * n_methods, n_pixels)
        _has0 = ~np.all(np.isnan(_flat0), axis=0)
        _base = np.full(n_pixels, -1, dtype=int)
        if _has0.any():
            with np.errstate(invalid='ignore'):
                _base[_has0] = np.nanargmax(_flat0[:, _has0], axis=0) // n_methods
        scores_3d = scores_3d * pw[:, None, :]   # broadcast over methods; NaN*w=NaN

    # Best phase per pixel: highest score across all (phase, method) combos
    # Reshape to (n_phases * n_methods, n_pixels)
    flat_scores = scores_3d.reshape(n_phases * n_methods, n_pixels)
    all_nan = np.all(np.isnan(flat_scores), axis=0)
    best_phase_flat = np.full(n_pixels, -1, dtype=int)
    has_data = ~all_nan
    if has_data.any():
        with np.errstate(invalid='ignore'):
            best_flat_idx = np.nanargmax(flat_scores[:, has_data], axis=0)
        best_phase_flat[has_data] = best_flat_idx // n_methods
    comparison.best_phase_per_pixel = best_phase_flat.reshape(n_rows, n_cols)

    # Count pixels whose winner flipped due to the EDS reweight (both >= 0).
    if _base is not None:
        comparison.n_eds_adjusted = int(np.sum(
            (best_phase_flat >= 0) & (_base >= 0) & (best_phase_flat != _base)))

    # Consensus map: for each pixel, which phase wins in majority of methods?
    # Per method, find the best phase
    votes = np.full((n_methods, n_pixels), -1, dtype=int)
    for mi in range(n_methods):
        method_scores = scores_3d[:, mi, :]  # (n_phases, n_pixels)
        has_data = ~np.all(np.isnan(method_scores), axis=0)
        if has_data.any():
            with np.errstate(invalid='ignore'):
                best_phase = np.nanargmax(method_scores[:, has_data], axis=0)
            votes[mi, has_data] = best_phase

    # Count votes per phase
    consensus = np.full(n_pixels, -1, dtype=int)
    for px in range(n_pixels):
        valid_votes = votes[:, px]
        valid_votes = valid_votes[valid_votes >= 0]
        if valid_votes.size > 0:
            counts = np.bincount(valid_votes, minlength=n_phases)
            consensus[px] = int(np.argmax(counts))

    comparison.consensus_map = consensus.reshape(n_rows, n_cols)
    return comparison


def build_consensus_xmap(comparison: ComparisonResult):
    """Build a unified multi-phase CrystalMap from consensus voting results.

    For each pixel, takes the orientation from the best-scoring
    PhaseMethodResult for the consensus-assigned phase.

    Parameters
    ----------
    comparison : ComparisonResult
        Must have consensus_map and results populated.

    Returns
    -------
    orix.crystal_map.CrystalMap or None
        Multi-phase CrystalMap ready for IPF coloring.
    """
    if comparison.consensus_map is None or not comparison.results:
        return None

    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation

    n_rows, n_cols = comparison.original_shape
    n_total = n_rows * n_cols
    consensus_flat = comparison.consensus_map.ravel()

    # 1. For each phase, find the best PhaseMethodResult (highest mean_score)
    best_per_phase: Dict[int, PhaseMethodResult] = {}
    for phase_idx, phase_config in enumerate(comparison.phases):
        candidates = [
            pmr for pmr in comparison.results
            if pmr.phase_name == phase_config.name
            and pmr.indexing_result is not None
            and pmr.indexing_result.xmap is not None
        ]
        if candidates:
            best_per_phase[phase_idx] = max(candidates, key=lambda r: r.mean_score)

    if not best_per_phase:
        logger.warning("build_consensus_xmap: no valid xmaps found in results")
        return None

    # 2. Build combined PhaseList from individual xmaps (preserving point_group)
    combined_phases = []
    for phase_idx in range(len(comparison.phases)):
        if phase_idx in best_per_phase:
            src_xmap = best_per_phase[phase_idx].indexing_result.xmap
            src_phases = src_xmap.phases
            found = False
            for pid in src_phases.ids:
                if pid >= 0:
                    p = src_phases[pid]
                    # EMSphinx .ang sets point_group directly (from TSL
                    # symmetry number) without space_group.  Copy whichever
                    # is available so IPF coloring works.
                    pg = p.point_group
                    sg = p.space_group
                    if sg is not None:
                        new_phase = Phase(
                            name=comparison.phases[phase_idx].name,
                            space_group=sg,
                            color=p.color,
                        )
                    elif pg is not None:
                        new_phase = Phase(
                            name=comparison.phases[phase_idx].name,
                            point_group=pg,
                            color=p.color,
                        )
                    else:
                        new_phase = Phase(
                            name=comparison.phases[phase_idx].name,
                            color=p.color,
                        )
                    if hasattr(p, 'structure') and p.structure is not None:
                        new_phase.structure = p.structure
                    logger.info(
                        f"  Phase {phase_idx} '{new_phase.name}': "
                        f"point_group={new_phase.point_group}, "
                        f"space_group={new_phase.space_group}"
                    )
                    combined_phases.append(new_phase)
                    found = True
                    break
            if not found:
                logger.warning(f"  Phase {phase_idx}: no valid phase found in xmap")
                combined_phases.append(Phase(name=comparison.phases[phase_idx].name))
        else:
            logger.warning(f"  Phase {phase_idx}: no xmap available")
            combined_phases.append(Phase(name=comparison.phases[phase_idx].name))

    combined_phase_list = PhaseList(combined_phases)

    # 3. Create full-grid arrays
    full_rotations = np.zeros((n_total, 4), dtype=float)
    full_rotations[:, 0] = 1.0  # Identity quaternion
    full_phase_ids = np.full(n_total, -1, dtype=int)

    # 4. For each phase, extract orientations for pixels assigned to it
    for phase_idx, pmr in best_per_phase.items():
        phase_pixels = np.where(consensus_flat == phase_idx)[0]
        if phase_pixels.size == 0:
            continue

        src_xmap = pmr.indexing_result.xmap
        src_mask = pmr.indexing_result.selection_mask
        src_rotations = src_xmap.rotations.data

        # Handle 3D rotations from dictionary indexing (top-N)
        if src_rotations.ndim == 3:
            src_rotations = src_rotations[:, 0, :]

        if src_mask is not None and not src_mask.all():
            # Partial indexing: build position -> xmap_entry mapping
            flat_mask = src_mask.ravel()
            indexed_positions = np.where(flat_mask)[0]
            position_to_idx = np.full(n_total, -1, dtype=int)
            position_to_idx[indexed_positions] = np.arange(len(indexed_positions))

            # Only include pixels that were actually indexed
            valid_mask = np.isin(phase_pixels, indexed_positions)
            valid_pixels = phase_pixels[valid_mask]
            xmap_indices = position_to_idx[valid_pixels]

            # Clamp to valid range
            in_range = xmap_indices < len(src_rotations)
            valid_pixels = valid_pixels[in_range]
            xmap_indices = xmap_indices[in_range]

            if valid_pixels.size > 0:
                full_rotations[valid_pixels] = src_rotations[xmap_indices]
                full_phase_ids[valid_pixels] = phase_idx
        else:
            # Full indexing: direct index mapping
            valid = phase_pixels < len(src_rotations)
            valid_pixels = phase_pixels[valid]
            if valid_pixels.size > 0:
                full_rotations[valid_pixels] = src_rotations[valid_pixels]
                full_phase_ids[valid_pixels] = phase_idx

    # 5. Build CrystalMap
    rows, cols = np.mgrid[0:n_rows, 0:n_cols]
    consensus_xmap = CrystalMap(
        rotations=Rotation(full_rotations),
        phase_id=full_phase_ids,
        x=cols.ravel().astype(float),
        y=rows.ravel().astype(float),
        phase_list=combined_phase_list,
    )

    n_assigned = int((full_phase_ids >= 0).sum())
    logger.info(
        f"build_consensus_xmap: {n_assigned}/{n_total} pixels assigned to "
        f"{len(best_per_phase)} phases"
    )
    return consensus_xmap


# --- Internal helpers ---

def _extract_confidence(index_data) -> Optional[np.ndarray]:
    """Extract confidence metric from Hough index_data.

    pyebsdindex can return index_data in several shapes depending on version
    and phase_list size:
      - a list/tuple of solutions: [best, 2nd, ...] -> take [0]
      - a structured array of shape (n_pixels,)  -> use directly
      - a structured array of shape (n_phases + 1, n_pixels) -> take the
        LAST row, which is the best-across-phases consensus row that
        pyebsdindex writes at indxData[-1] = indxData[0] for single-phase
        or the true per-pixel winner for multi-phase. Previous code raveled
        the full 2D array, which produced n_rows × n_pixels and crashed the
        downstream reshape to the navigation grid (BUG-M, 2026-04-21).
    """
    if index_data is None:
        return None
    try:
        data = index_data[0] if isinstance(index_data, (list, tuple)) else index_data
        cm = np.asarray(data['cm'])
        # Collapse any extra leading axis (solutions/phases) down to 1D per pixel.
        # For single-phase pyebsdindex, cm[-1] == cm[0]. For multi-phase, cm[-1]
        # is the consensus best-per-pixel row.
        if cm.ndim >= 2:
            cm = cm[-1]
        return cm.ravel()
    except (KeyError, IndexError, TypeError):
        return None


def _index_selected_hough(signal, selected_indices, phase_list, indexer, detector):
    """Index specific pixels via Hough and create a CrystalMap.

    Indexes each selected pixel individually and assembles a CrystalMap
    with correct spatial coordinates.
    """
    from hyperspy.signals import Signal2D
    from kikuchipy.signals import EBSD
    from orix.crystal_map import CrystalMap
    from orix.quaternion import Rotation

    rotations_list = []
    phase_ids = []
    xs = []
    ys = []

    for row, col in selected_indices:
        try:
            pat = signal.data[row, col]
            sig2d = Signal2D(pat)
            ebsd = EBSD(sig2d, detector=detector)
            xmap_single, _, _ = ebsd.hough_indexing(
                phase_list, indexer,
                return_index_data=True,
                return_band_data=True,
                verbose=0,
            )
            rotations_list.append(xmap_single.rotations.data[0])
            phase_ids.append(int(xmap_single.phase_id[0]))
        except Exception:
            # Failed pixel: identity quaternion + unindexed
            rotations_list.append(np.array([1.0, 0.0, 0.0, 0.0]))
            phase_ids.append(-1)
        xs.append(float(col))
        ys.append(float(row))

    all_rotations = np.array(rotations_list)
    all_phase_ids = np.array(phase_ids)

    xmap = CrystalMap(
        rotations=Rotation(all_rotations),
        phase_id=all_phase_ids,
        x=np.array(xs),
        y=np.array(ys),
        phase_list=phase_list,
    )
    return xmap


def run_per_phase_indexing(
    signal,
    detector,
    method: IndexingMethod,
    config: ComparisonConfig,
    phase_configs: List[PhaseConfig],
    masks_per_phase: List[np.ndarray],
    consensus_map: np.ndarray,
    h5_path: str = "",
    detector_params: Optional[Dict] = None,
    progress_cb: Optional[callable] = None,
) -> IndexingResult:
    """Run indexing once per phase, each restricted to that phase's pixels.

    Unlike the comparison batch (where every phase competes for every pixel),
    this routine takes the assignment as ground truth — typically the
    EDS phase-map produced by the EDS page — and runs each phase only
    on the pixels classified as that phase. Per-phase results are then
    stitched together by ``build_consensus_xmap`` using the supplied
    ``consensus_map`` (rather than re-deriving consensus from scores,
    which would defeat the point).

    Why this is faster + cleaner than the competitive path:

    - Each indexing pass touches ~1/N as many pixels (N = phase count).
      For Hough on 100k pixels and 5 phases that's a 5× wall-clock win.
    - Pixels never get an orientation from the wrong phase just because
      that phase happened to score highest on a particular pattern.
      The chemistry already told us which phase the pixel is — we
      only need the orientation.

    Parameters
    ----------
    masks_per_phase
        One 2D bool mask per phase, in the same order as ``phase_configs``.
        Pixels that should not be indexed at all (i.e. ``consensus_map < 0``)
        must be False in every mask.
    consensus_map
        2D int array (n_rows, n_cols) where each pixel holds its phase
        index (matching the position in ``phase_configs``) or ``-1`` for
        unclassified pixels. ``build_consensus_xmap`` reads this to pull
        the right per-phase orientation into the merged xmap.
    progress_cb
        Optional callback ``progress_cb(message: str, fraction: float)``
        invoked at the start of each phase pass + at merge time, so the
        caller can pipe progress to a long-running task UI.

    Returns
    -------
    IndexingResult
        ``xmap`` is a multi-phase CrystalMap covering only the
        classified pixels. ``selection_mask`` is the union of all
        per-phase masks, so consumers like the result writer can still
        fill the original-grid lattice correctly.
    """
    if len(phase_configs) != len(masks_per_phase):
        raise ValueError(
            f"phase_configs and masks_per_phase must align: got "
            f"{len(phase_configs)} configs and {len(masks_per_phase)} masks"
        )
    if not phase_configs:
        raise ValueError("run_per_phase_indexing requires at least one phase")

    n_rows, n_cols = signal.data.shape[:2]
    if consensus_map.shape != (n_rows, n_cols):
        raise ValueError(
            f"consensus_map shape {consensus_map.shape} does not match "
            f"signal navigation shape {(n_rows, n_cols)}"
        )

    union_mask = np.zeros((n_rows, n_cols), dtype=bool)
    for m in masks_per_phase:
        if m.shape != (n_rows, n_cols):
            raise ValueError(
                f"per-phase mask shape {m.shape} does not match signal "
                f"navigation shape {(n_rows, n_cols)}"
            )
        union_mask |= m

    if not union_mask.any():
        raise ValueError(
            "All per-phase masks are empty — nothing to index. Check that "
            "the phase map covers some pixels for at least one selected phase."
        )

    # Run each phase on its own pixels. Skipped phases (empty mask or
    # known-failure CIF) are tracked so the merge doesn't get a half-
    # populated PhaseList that misnumbers consensus indices.
    per_phase_results: List[PhaseMethodResult] = []
    skipped: List[Tuple[int, str]] = []
    for i, (pc, mask) in enumerate(zip(phase_configs, masks_per_phase)):
        if not mask.any():
            skipped.append((i, "empty mask"))
            logger.info("Per-phase indexing: skipping phase %s (empty mask)", pc.name)
            continue
        if progress_cb:
            progress_cb(
                f"Phase {i + 1}/{len(phase_configs)}: {pc.name} "
                f"({int(mask.sum())} pixels)",
                0.05 + 0.85 * (i / max(len(phase_configs), 1)),
            )
        try:
            pmr = run_single_phase_method(
                signal=signal,
                detector=detector,
                phase_config=pc,
                method=method,
                config=config,
                selection_mask=mask,
                h5_path=h5_path if method == IndexingMethod.SPHERICAL else "",
                detector_params=detector_params if method == IndexingMethod.SPHERICAL else None,
            )
            per_phase_results.append(pmr)
        except Exception as exc:
            logger.warning("Per-phase indexing: phase %s failed: %s", pc.name, exc)
            skipped.append((i, str(exc)))
            continue

    if not per_phase_results:
        details = "; ".join(f"{phase_configs[i].name}: {why}" for i, why in skipped) or "unknown"
        raise ValueError(f"Per-phase indexing produced no results — all phases failed: {details}")

    # Build a synthetic ComparisonResult so build_consensus_xmap can
    # do the heavy lifting of stitching the per-phase xmaps together.
    # The crucial difference from the score-based path: we pre-set
    # ``consensus_map`` to the user's classification so the merge
    # respects the EDS assignment exactly.
    sanitised_consensus = consensus_map.astype(np.int32, copy=True)
    # If a phase failed, its pixels in the consensus map can't be
    # rendered — mark them unclassified so build_consensus_xmap doesn't
    # try to look up an absent xmap.
    if skipped:
        skipped_indices = {i for i, _ in skipped}
        for skipped_idx in skipped_indices:
            sanitised_consensus[sanitised_consensus == skipped_idx] = -1

    comparison = ComparisonResult(
        results=per_phase_results,
        phases=phase_configs,
        methods=[method],
        original_shape=(n_rows, n_cols),
        selection_mask=union_mask,
        consensus_map=sanitised_consensus,
    )

    if progress_cb:
        progress_cb("Merging per-phase results…", 0.95)
    merged_xmap = build_consensus_xmap(comparison)
    if merged_xmap is None:
        raise RuntimeError(
            "build_consensus_xmap returned None — per-phase merge failed even "
            "though individual phases produced xmaps. Check the per-phase "
            "selection_mask shapes and the consensus_map indices."
        )

    metadata = {
        "method": method.value,
        "per_phase_routing": True,
        "n_phases_total": len(phase_configs),
        "n_phases_run": len(per_phase_results),
        "n_phases_skipped": len(skipped),
        "skipped": [{"phase_index": i, "reason": why} for i, why in skipped],
        "per_phase_pixel_counts": {
            pmr.phase_name: int(masks_per_phase[
                next(
                    (j for j, pc in enumerate(phase_configs) if pc.name == pmr.phase_name),
                    0,
                )
            ].sum())
            for pmr in per_phase_results
        },
    }

    return IndexingResult(
        xmap=merged_xmap,
        selection_mask=union_mask,
        original_shape=(n_rows, n_cols),
        method=method,
        metadata=metadata,
    )
