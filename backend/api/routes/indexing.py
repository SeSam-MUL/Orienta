"""
Indexing API Routes

Wraps IndexingController for:
- Hough, Dictionary, and Spherical (EMSphinx) indexing
- Pixel selection (full, region, mask)
- Multi-phase comparison
- Result retrieval with back-mapping
"""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.api.services.image_utils import array_to_base64_png, array_to_base64_raw, colormap_array_to_base64
from backend.api.services.calibration_store import calibration_store
from backend.api.services import state_version

logger = logging.getLogger(__name__)
router = APIRouter()

import time as _time
from collections import OrderedDict

# Cap _indexing_tasks at MAX_TRACKED_TASKS so a long-running session that
# fires many indexing jobs (PC refinement, Quick Test, batch retries) doesn't
# leak memory by keeping every task forever. Oldest evicted on overflow.
MAX_TRACKED_TASKS = 50
_indexing_tasks: OrderedDict = OrderedDict()


def _track_task(task_id: str, initial_state: dict) -> None:
    """Insert a new task while enforcing MAX_TRACKED_TASKS LRU cap."""
    _indexing_tasks[task_id] = initial_state
    while len(_indexing_tasks) > MAX_TRACKED_TASKS:
        _indexing_tasks.popitem(last=False)


# Result registry — stores multiple indexing results keyed by ID
# OrderedDict so we can evict the oldest results once past the cap. Each entry
# holds a full CrystalMap (orientations + scores for every pixel), so an
# unbounded registry leaks RAM across a long multi-index session.
MAX_STORED_RESULTS = 12
_result_registry: "OrderedDict[str, object]" = OrderedDict()  # {result_id: IndexingResult}
_active_result_id: str | None = None


def _get_result(result_id: str | None = None):
    """Get a specific or the active indexing result."""
    rid = result_id or _active_result_id
    if rid is None:
        return None
    return _result_registry.get(rid)


def _resolve_step_size_um(active) -> float | None:
    """Best-effort µm/pixel step size for an indexing result, or None.

    Order of preference:
      1. ``result.metadata['step_size_um']`` — set on re-import of a light
         file and the most robust source (survives signal unload/switch).
      2. The active EBSD signal's navigation-axis scale — what kikuchipy
         records from the source h5oina; reliable right after indexing.

    Returns None when neither is available; callers that need step size for a
    grid (e.g. the MTEX-facing exports) must fail loud rather than guess a
    default — a wrong step silently mis-scales the whole map.
    """
    # 1. Result metadata.
    try:
        md = getattr(active, "metadata", None) or {}
        s = md.get("step_size_um")
        if s is not None and float(s) > 0:
            return float(s)
    except (TypeError, ValueError):
        pass

    # 2. Active signal navigation-axis scale.
    try:
        from backend.api.routes.ebsd_viewer import _get_active_signal
        signal = _get_active_signal()
        if signal is not None:
            scales = [a.scale for a in signal.axes_manager.navigation_axes]
            if scales and float(scales[0]) > 0:
                return float(scales[0])
    except Exception:
        logger.debug("step-size lookup from signal failed", exc_info=True)

    return None


def _resolve_source_vendor(active, src_path_str: str | None = None) -> str:
    """Resolve the source vendor ("oxford"/"edax"/"bruker"/"") for an export.

    Used to put exported orientations into the vendor's stored Euler frame so
    MTEX/Aztec read them aligned (see orientation_frame.to_vendor_export_frame).
    Order: result metadata -> source h5 Manufacturer. Returns "" when unknown,
    which the frame helpers treat as "leave native" (no silent mis-rotation).
    """
    def _norm(s: str) -> str:
        s = (s or "").lower()
        if "oxford" in s:
            return "oxford"
        if "edax" in s or "tsl" in s or "ametek" in s:
            return "edax"
        if "bruker" in s:
            return "bruker"
        return ""

    md = getattr(active, "metadata", None) or {}
    v = _norm(str(md.get("source_vendor", "")))
    if v:
        return v

    if src_path_str and Path(src_path_str).is_file():
        try:
            import h5py
            with h5py.File(src_path_str, "r") as f:
                if "Manufacturer" in f:
                    raw = f["Manufacturer"][()]
                    s = raw[0] if np.ndim(raw) else raw
                    s = s.decode() if isinstance(s, bytes) else str(s)
                    return _norm(s)
        except Exception:
            logger.debug("vendor resolve from source failed", exc_info=True)
    return ""


def _euler_ndarray_to_vendor(euler_arr, vendor: str, r_user=None):
    """Right-multiply an (..., 3) Bunge-radian Euler array into the vendor frame.

    Preserves shape; used for PerPhase Euler grids that aren't orix Rotations.
    ``r_user`` (opt-in per-file coordinate-system rotation) is threaded through
    to ``to_vendor_export_frame``; ``None`` (default) is byte-identical to today.
    """
    from orix.quaternion import Rotation as _Rot
    from backend.api.services.orientation_frame import to_vendor_export_frame
    arr = np.asarray(euler_arr, dtype=np.float64)
    shp = arr.shape
    rot = to_vendor_export_frame(
        _Rot.from_euler(arr.reshape(-1, 3)), vendor, r_user=r_user)
    return rot.to_euler().reshape(shp).astype(np.float32)


def _store_result(result, method_name: str) -> str:
    """Store a result and make it active. Returns the result_id.

    Also tags the result with the currently-active source file path so the
    frontend gallery can show which file each entry belongs to and the
    activate endpoint can fail loud when the user tries to activate a
    result whose file isn't loaded.
    """
    global _active_result_id
    result_id = f"{method_name}_{int(_time.time())}_{uuid.uuid4().hex[:8]}"
    _result_registry[result_id] = result
    _result_registry.move_to_end(result_id)
    _active_result_id = result_id
    # Bound RAM: evict the oldest results past the cap, but never the active
    # (just-stored) one. Reactivation on file-switch degrades gracefully to an
    # empty map + re-index if a very old result was evicted.
    while len(_result_registry) > MAX_STORED_RESULTS:
        oldest = next(iter(_result_registry))
        if oldest == _active_result_id:
            break
        _result_registry.pop(oldest, None)
    # Tag with source file so multi-file workflows can route correctly.
    try:
        from backend.api.routes.ebsd_viewer import _ebsd_file_path
        if not hasattr(result, "metadata") or result.metadata is None:
            try:
                result.metadata = {}
            except Exception:
                pass
        if hasattr(result, "metadata") and result.metadata is not None:
            result.metadata["source_file"] = str(_ebsd_file_path) if _ebsd_file_path else None
    except Exception:
        logger.debug("could not tag result with source_file", exc_info=True)
    # Result is now registered + active → tell polling clients to refetch.
    state_version.bump()
    return result_id


def reactivate_result_for_source(path: str) -> "str | None":
    """Re-activate the most recent indexing result computed for ``path``.

    Called on file (re)load. Previously load_ebsd hard-reset
    ``_active_result_id = None`` on every file change, so returning to a file
    you had already indexed showed an EMPTY phase map until you manually
    re-picked the result from the gallery — which reads as "my indexing
    result vanished". The results were never actually lost (they stay in
    ``_result_registry`` tagged with their ``source_file``), just deactivated.

    This restores the result automatically when you come back to its file,
    mirroring the analysis-dataset stash/restore. It only ever activates a
    result whose tagged source file canonically matches the file being
    loaded, so there's no risk of showing a previous file's (wrong-shape)
    xmap; if nothing matches it sets None, exactly like before.
    """
    global _active_result_id
    try:
        from backend.api.routes.ebsd_viewer import _canonical_path
        canon = _canonical_path(path)
    except Exception:
        _active_result_id = None
        return None
    match = None
    # dict preserves insertion order → last match wins = most recent result.
    for rid, res in _result_registry.items():
        md = getattr(res, "metadata", None) or {}
        src = md.get("source_file") if isinstance(md, dict) else None
        if src and _canonical_path(src) == canon:
            match = rid
    _active_result_id = match
    if match:
        logger.info("Re-activated indexing result %s for %s", match, path)
    return match


def _attach_indexing_metadata(
    result,
    method_name: str,
    sht_paths: list | None = None,
    det_params: dict | None = None,
) -> None:
    """Populate result.metadata fields needed by the SHT forward renderer.

    Sets:
      - ``indexing_method``: "spherical" | "dictionary" | "hough"
      - ``sht_paths_by_phase``: {phase_id: absolute_sht_path} (spherical only)
      - ``detector_geometry``: vendor-normalized PC + detector shape + tilt
        + pixel size + binning, in the same dict shape produced upstream by
        the indexing route.

    Idempotent and safe to call before ``_store_result``. Adds nothing for
    methods other than spherical that don't have an SHT.
    """
    if result is None:
        return
    if not hasattr(result, "metadata") or result.metadata is None:
        try:
            result.metadata = {}
        except Exception:
            return
    md = result.metadata
    md["indexing_method"] = method_name
    if method_name == "spherical" and sht_paths:
        # Phase ids in xmap conventionally start at 1; if there's a
        # not_indexed entry at id=-1 we still align positively to the
        # supplied sht_paths in order.
        md["sht_paths_by_phase"] = {
            i + 1: str(Path(p).resolve())
            for i, p in enumerate(sht_paths)
        }
    if det_params:
        # Store a defensive copy so later mutations of det_params don't
        # leak into the registered result.
        md["detector_geometry"] = dict(det_params)


import re as _re

# Match EMsoft-convention SHT filenames: "Name (Formula) [Pearson] {kV}"
# Example: "Ni (Ni) [cF4] {20kV}" -> prefix "Ni"
# We only strip the trailing "(...) [...] {...}" block so legitimate names
# containing parentheses (like "alpha-(AlFeSi) (Fe23Al81Si15)") are preserved.
_SHT_SUFFIX_RE = _re.compile(
    r"\s*\([^)]+\)\s*\[[^\]]+\]\s*\{[^}]+\}\s*$"
)


def _smart_phase_name_from_path(path: Path) -> str:
    """Return a short, human-friendly phase name derived from a file path.

    Heuristics (in order):
      - Dictionary/master files like "Ni_master_E20kV_npx500_dict_20kV_60x60_5.0deg.h5"
        contain "_master_" or "_dict_" separators — take the prefix ("Ni").
      - SHT files follow the EMsoft convention
        "Name (Formula) [Pearson] {kV}.sht" — strip that trailing block.
        Names with genuine parentheses like
        "alpha-(AlFeSi) (Fe23Al81Si15) [cI128] {20kV}" keep "alpha-(AlFeSi)".
      - Otherwise use the stem as-is.
    """
    stem = path.stem
    for sep in ("_master_", "_dict_"):
        if sep in stem:
            return stem.split(sep, 1)[0]
    # Strip the SHT "(Formula) [Pearson] {kV}" tail but only if it's actually
    # at the end. Names containing parentheses elsewhere survive intact.
    stripped = _SHT_SUFFIX_RE.sub("", stem).strip()
    if stripped:
        return stripped
    return stem


def _inject_phase_names(result, req) -> None:
    """Set meaningful phase names in xmap from the request file paths.

    Handles three cases that previously silently failed:
      1. orix ``xmap.phases`` may yield ``(id, Phase)`` tuples on 0.12+ or
         bare Phase objects on older versions. Setting ``.name`` on a tuple
         raised AttributeError and the injection was a no-op — empty phase
         names in the result JSON (BUG-I).
      2. For dictionary .h5 files that lack CrystalData,
         ``extract_metadata_from_h5`` fell back to the full stem (e.g.
         ``Ni_master_E20kV_npx500_dict_20kV_60x60_5.0deg``). We now parse
         the stem to return a short "Ni".
      3. PREVIOUS FIX WAS STILL BUGGY: aligned by *position* rather than
         by *phase id*. When ``phases`` contains a leading ``not_indexed``
         entry (id=-1), the first real phase (id=0) lives at position 1.
         Iterating ``enumerate`` and assigning paths[0] to position 0 then
         renamed ``not_indexed`` to "Ni" and skipped the real phase. Fix:
         only touch entries with ``id >= 0``, consume paths in order.
    """
    if result is None or result.xmap is None:
        return
    try:
        from phase_metadata import get_phase_metadata
        from pathlib import Path

        paths = req.cif_paths or req.master_h5_paths or req.sht_paths or []
        if not paths:
            return

        cif_dir = None
        try:
            from path_utils import get_local_database_path, DATABASE_SUBFOLDERS
            cif_dir = get_local_database_path() / DATABASE_SUBFOLDERS["cif_library"]
        except Exception:
            pass

        phase_names: list[str] = []
        for p in paths:
            path_obj = Path(p)
            name = ""
            try:
                meta = get_phase_metadata(path_obj, cif_library_dir=cif_dir)
                if meta.source == "stem" or not meta.formula:
                    name = _smart_phase_name_from_path(path_obj)
                else:
                    name = meta.formula
            except Exception:
                name = _smart_phase_name_from_path(path_obj)
            phase_names.append(name)

        try:
            phases_iter = list(result.xmap.phases)
        except Exception as e:
            logger.warning(f"_inject_phase_names: could not iterate xmap.phases: {e}")
            return

        # Filter to real phases (id >= 0) preserving order; tuple or Phase-obj.
        real_entries: list = []
        for entry in phases_iter:
            if isinstance(entry, tuple) and len(entry) == 2:
                pid, phase_obj = entry
                try:
                    if int(pid) < 0:
                        continue
                except Exception:
                    pass
            else:
                phase_obj = entry
                pid = getattr(entry, "id", 0)
                try:
                    if int(pid) < 0:
                        continue
                except Exception:
                    pass
            real_entries.append(phase_obj)

        # Assign names in order. If the user sent fewer paths than real
        # phases, only rename the first N.
        for phase_obj, name in zip(real_entries, phase_names):
            if not name:
                continue
            try:
                phase_obj.name = name
            except (AttributeError, TypeError) as e:
                logger.debug(f"_inject_phase_names: could not set name: {e}")
    except Exception as e:
        logger.warning(f"Could not set phase names: {e}")


def _build_phase_configs(req):
    """Build PhaseConfig list from request file paths.

    Returns a list of PhaseConfig objects for multi-phase indexing.
    Each phase is identified by its file path and enriched with metadata.

    Raises ValueError when the selected method has no matching files —
    e.g. Spherical with only CIFs attached. Previously this silently
    returned [], and the caller then raised the opaque
    "All phases failed during multi-phase indexing: unknown" because
    phase_errors also stayed empty (the per-phase loop never ran).
    """
    from indexing_controller import PhaseConfig
    from phase_metadata import get_phase_metadata
    from pathlib import Path

    configs = []
    paths = []
    expected_ext = ""
    if req.method == 'hough':
        paths = req.cif_paths
        expected_ext = "CIF"
    elif req.method == 'dictionary':
        paths = req.master_h5_paths
        expected_ext = "master .h5"
    elif req.method == 'spherical':
        paths = req.sht_paths
        expected_ext = "SHT"

    if not paths:
        raise ValueError(
            f"{req.method} indexing needs {expected_ext} files, but got "
            f"cif={len(req.cif_paths)}, master_h5={len(req.master_h5_paths)}, "
            f"sht={len(req.sht_paths)}. Check that the selected phases have a "
            f"{expected_ext} file in the database for this method."
        )

    cif_dir = None
    try:
        from path_utils import get_local_database_path, DATABASE_SUBFOLDERS
        cif_dir = get_local_database_path() / DATABASE_SUBFOLDERS["cif_library"]
    except Exception:
        pass

    for p in paths:
        try:
            meta = get_phase_metadata(Path(p), cif_library_dir=cif_dir)
            name = meta.formula or Path(p).stem
        except Exception:
            name = Path(p).stem

        pc = PhaseConfig(
            name=name,
            cif_path=p if req.method == 'hough' else '',
            master_h5_path=p if req.method == 'dictionary' else '',
            sht_path=p if req.method == 'spherical' else '',
        )

        # For Hough: load phase from CIF
        if req.method == 'hough':
            from orix.crystal_map import Phase
            from ebsd_utils import sanitize_cif
            pc.phase_list = Phase.from_cif(sanitize_cif(p))

        configs.append(pc)

    return configs


def get_last_indexing_result():
    """Accessor for the active indexing result. Use this instead of importing _last_result directly."""
    return _get_result()


def build_spherical_det_params(signal, detector, ebsd_file_path: str, pixel_rc=None) -> dict:
    """Detector-geometry dict for the SHT forward operator + spherical indexer.

    Same shape/values the spherical indexing path produces. Includes the
    BUG-J auto-scale of pixel_size when the detector width falls outside
    EMSphinx's [5, 90] mm range, and source-vendor detection.

    PC selection from a per-pixel detector map (F3):
      - ``pixel_rc=(row, col)`` → the PC at THAT pixel (single-pixel phase test
        uses the clicked pixel's PC — the first rung toward per-pixel indexing);
      - otherwise → the MEAN PC over the map (was pixel (0,0), which is neither
        the mean nor the tested pixel — the F3 bug);
      - a single PC (shape (1,3)) is used as-is.
    """
    from backend.spherical_gpu.pipeline.detector import DEFAULT_PIXEL_SIZE_UM
    pc_full = np.asarray(detector.pc, dtype=float)
    if pc_full.ndim == 3 and pixel_rc is not None:
        nr, nc = pc_full.shape[0], pc_full.shape[1]
        r = min(max(int(pixel_rc[0]), 0), nr - 1)
        c = min(max(int(pixel_rc[1]), 0), nc - 1)
        pc_arr = pc_full[r, c]
    elif pc_full.size > 3:
        pc_arr = pc_full.reshape(-1, 3).mean(axis=0)
    else:
        pc_arr = pc_full.flatten()[:3]
    sig_shape = signal.axes_manager.signal_shape
    nav_shape_raw = signal.axes_manager.navigation_shape
    step_sizes = [a.scale for a in signal.axes_manager.navigation_axes]
    det_params = {
        'pc_x': float(pc_arr[0]),
        'pc_y': float(pc_arr[1]),
        'pc_z': float(pc_arr[2]),
        'pat_width': int(sig_shape[0]),
        'pat_height': int(sig_shape[1]),
        'n_cols': int(nav_shape_raw[0]),
        'n_rows': int(nav_shape_raw[1]) if len(nav_shape_raw) > 1 else 1,
        # MUST equal the PC-Refinement Forward-Sim default so the same phase at
        # the same PC builds the same detector distance L in both tools. Shared
        # constant — do not hardcode a divergent literal here (see detector.py).
        'pixel_size': DEFAULT_PIXEL_SIZE_UM,
        'tilt': 10.0,
        'binning': 1,
        'step_x': float(step_sizes[0]) if step_sizes else 0.1,
        'step_y': float(step_sizes[1]) if len(step_sizes) > 1 else 0.1,
        'vendor': 'Bruker',
    }
    if hasattr(detector, 'shape'):
        det_params['pat_height'] = int(detector.shape[0])
        det_params['pat_width'] = int(detector.shape[1])
    if hasattr(detector, 'px_size') and detector.px_size > 1.0:
        det_params['pixel_size'] = float(detector.px_size)
    if hasattr(detector, 'tilt'):
        det_params['tilt'] = float(detector.tilt)
    # SAMPLE tilt is separate from detector tilt and is what the
    # SHT forward renderer needs (FEAT-SHT-FWD-A).
    if hasattr(detector, 'sample_tilt'):
        det_params['sample_tilt'] = float(detector.sample_tilt)
    if hasattr(detector, 'binning'):
        det_params['binning'] = int(detector.binning)

    # EMSphinx's IndexEBSD rejects detector widths outside [5, 90] mm
    # (BUG-J). Datasets loaded without an explicit pixel_size fell
    # back to 55 µm — on a 60×60 binned detector that's 3.3 mm, below
    # the threshold, and spherical would fail with
    # "unreasonable EBSD detector width".
    # Auto-scale pixel_size so total detector width lands in the
    # middle of the valid range (~15 mm) when the header/store value
    # would otherwise produce an out-of-range width. We log the
    # substitution so users can set a correct value in PC Refinement
    # if they want the geometry to be physically accurate.
    det_width_mm = det_params['pixel_size'] * det_params['pat_width'] / 1000.0
    if det_width_mm < 5.0 or det_width_mm > 90.0:
        old_px = det_params['pixel_size']
        target_mm = 15.0
        det_params['pixel_size'] = (target_mm * 1000.0) / max(1, det_params['pat_width'])
        logger.warning(
            "Spherical: detector width %.1f mm out of EMSphinx range "
            "[5, 90] mm (pixel_size=%.1f µm × %d px). Substituting "
            "pixel_size=%.1f µm → width %.1f mm. Set a real pixel_size "
            "in PC Refinement for physically accurate geometry.",
            det_width_mm, old_px, det_params['pat_width'],
            det_params['pixel_size'], target_mm,
        )

    def _detect_source_vendor(h5_path):
        try:
            import h5py as _h5
            with _h5.File(h5_path, 'r') as f:
                for scan_key in ['1', '2', '3']:
                    if f"{scan_key}/EBSD/Data" in f:
                        return "oxford"
                for key in f.keys():
                    if f"{key}/EBSD/Data/Pattern" in f:
                        return "edax"
        except Exception:
            pass
        return "unknown"

    det_params['source_vendor'] = _detect_source_vendor(ebsd_file_path or '')
    return det_params


# --- Single-Pixel Phase Test seams ------------------------------------------
# These thin wrappers exist so the route stays testable: the test suite
# monkeypatches them to inject a synthetic signal / detector / chemistry /
# library instead of standing up the full EBSD-viewer + calibration stack.

def _get_active_signal_for_phase_test():
    from backend.api.routes.ebsd_viewer import _get_active_signal
    return _get_active_signal()


def _bg_remove_phase_test(exp):
    """Dynamic-background-remove the measured pattern (matches the clean
    simulated dynamical pattern → band-vs-band NCC + a clean preview).

    Fail-soft: on any error fall back to the raw pattern (logged), never 500.
    """
    import numpy as np
    try:
        from kikuchipy.pattern import remove_dynamic_background
        return remove_dynamic_background(
            exp.astype("float32"), operation="subtract",
            filter_domain="frequency").astype("float32")
    except Exception as _e:  # noqa: BLE001 — fail-soft, raw fallback
        logger.warning(
            "phase-test: dynamic BG removal failed, using raw pattern: %s", _e)
        return np.asarray(exp, dtype=np.float32)


def _get_detector_for_phase_test(pixel_index=None):
    """Return (calibration_entry, det_params) for the active dataset, or
    (None, None) if no detector is registered.

    When ``pixel_index`` is given and the active dataset carries a per-pixel PC
    map, det_params uses the PC at THAT pixel (F3 tested-pixel); otherwise the
    map mean (or the single PC)."""
    from backend.api.routes.ebsd_viewer import _active_dataset, _ebsd_file_path
    from backend.api.services.calibration_store import calibration_store
    detector = calibration_store.get_detector(_active_dataset)
    entry = calibration_store.get_entry(_active_dataset)
    if detector is None:
        return None, None
    signal = _get_active_signal_for_phase_test()
    pixel_rc = None
    if pixel_index is not None and signal is not None:
        try:
            n_cols = int(signal.axes_manager.navigation_shape[0])
            if n_cols > 0:
                pixel_rc = divmod(int(pixel_index), n_cols)
        except Exception:
            pixel_rc = None
    det_params = build_spherical_det_params(signal, detector, _ebsd_file_path, pixel_rc=pixel_rc)
    return entry, det_params


def get_pixel_at_pct_for_phase_test(pixel_index: int):
    from backend.api.services.eds_pixel_chemistry import get_pixel_at_pct
    return get_pixel_at_pct(pixel_index)


def _phase_test_library_entries(phase_keys):
    from backend.api.services.crystal_hint_local_library import get_index
    entries = [e for e in get_index().values() if e.sht_path is not None]
    if phase_keys:
        wanted = set(phase_keys)
        entries = [e for e in entries if e.key in wanted]
    return entries


def _dictionary_library_files():
    """All dictionary/master ``.h5`` files in the Dictionary_Library (for the
    'Use for indexing' Dictionary-method availability check). Best effort."""
    try:
        from path_utils import get_local_database_path, DATABASE_SUBFOLDERS
        d = get_local_database_path() / DATABASE_SUBFOLDERS["dictionary_library"]
        if d.is_dir():
            return list(d.rglob("*.h5"))
    except Exception:
        logger.debug("Dictionary_Library scan failed", exc_info=True)
    return []


def _master_h5_for_phase(entry, files=None):
    """Best-effort match of a phase to a dictionary/master ``.h5`` by name, so the
    Phase-Test 'Use for indexing' picker can show whether Dictionary indexing is
    available for it. Returns a path string or None (None -> Dictionary disabled).

    Matching is intentionally conservative (substring of the canonical key/formula
    stem in the filename); a miss just disables Dictionary for that phase — the
    user can still generate a master pattern via the Dictionary/Simulation tools.
    """
    if files is None:
        files = _dictionary_library_files()
    if not files:
        return None
    key = (getattr(entry, "key", "") or "").strip().lower()
    formula = (getattr(entry, "formula", "") or "").strip().lower()
    # Distinctive tokens only (>= 4 chars) so short formulas like "ni"/"al" don't
    # over-match an unrelated master and falsely show Dictionary as available. A
    # miss just disables Dictionary (✗) — far safer than attaching a wrong master.
    cands = [s for s in (key, key.split(" ")[0], formula.split(" ")[0]) if s and len(s) >= 4]
    for f in files:
        nm = f.name.lower()
        if any(c in nm for c in cands):
            return str(f)
    return None


class IndexingStartRequest(BaseModel):
    method: str = "hough"  # "hough", "dictionary", "spherical"
    selection_mode: str = "full"  # "full", "region", "mask"

    # Phase files
    cif_paths: List[str] = []
    sht_paths: List[str] = []
    master_h5_paths: List[str] = []

    # Hough params
    n_bands: int = 12
    t_sigma: float = 2.0
    r_sigma: float = 2.0

    # Dictionary params
    metric: str = "ncc"
    keep_n: int = 20
    # GPU compute mode for dictionary indexing.
    # "auto" | "gpu" | "cpu" — ignored by hough/spherical methods.
    compute_mode: str = "auto"

    # Spherical params
    bandwidth: int = 88
    normed: bool = True
    refine: bool = True
    nregions: int = 10
    # "emsphinx" (WSL EMSphInx CPU) or "spherical_gpu" (in-process PyTorch GPU).
    # Default stays on the CPU path until the frontend toggles to GPU.
    backend: str = "emsphinx"
    circmask: int = -1
    gausbckg: bool = False

    # Region selection
    row_start: int = 0
    row_end: int = -1
    col_start: int = 0
    col_end: int = -1

    # Mask (flat boolean array as list of 0/1)
    mask: Optional[List[int]] = None

    # Quick test (single-pixel indexing)
    quick_test: bool = False
    qt_row: int = 0
    qt_col: int = 0

    # Extra fields from frontend (ignored but accepted)
    dataset: Optional[str] = None
    send_to_phase_map: bool = True

    # Per-phase routing from the EDS phase map. When true, the
    # backend reads the live PhaseMapStore, builds one (phase, mask)
    # pair per CIF in cif_paths, and runs each phase only on its
    # classified pixels. mask + selection_mode are ignored in that
    # mode — the per-phase masks come from the store, not the
    # request.
    use_phase_map_routing: bool = False


class SinglePixelPhaseTestRequest(BaseModel):
    pixel_index: int = Field(..., ge=0)
    eds_weighting: Literal["off", "soft", "filter"] = "filter"
    eds_filter_threshold: float = Field(0.35, ge=0.0, le=1.0)
    aperture: Literal["auto", "circular", "full"] = "auto"
    aperture_radius: float = Field(1.0, ge=0.3, le=1.0)
    max_bandwidth: int = Field(128, ge=64, le=512)
    phase_keys: Optional[list[str]] = None
    bg_remove: bool = True


class PhaseTestRemaskRequest(BaseModel):
    job_id: str
    aperture: Literal["auto", "circular", "full"] = "circular"
    aperture_radius: float = Field(1.0, ge=0.3, le=1.0)


class PhaseTestCandidate(BaseModel):
    phase_key: str
    formula: str = ""
    display_formula: str = ""
    space_group: str = ""
    crystal_system: str = "unknown"
    r_score: Optional[float] = None
    ncc_score: Optional[float] = None
    chemistry_fit: Optional[float] = None
    rank: int = 0
    euler_deg: list[float] = []
    simulated_png: Optional[str] = None
    ncc_diff_png: Optional[str] = None
    excluded_by_eds: bool = False
    # Indexing-method availability for the "Use for indexing" button: the file
    # path this phase has for each method (None = not available -> that method is
    # disabled in the picker). cif=Hough, sht=Spherical, master_h5=Dictionary.
    cif_path: Optional[str] = None
    sht_path: Optional[str] = None
    master_h5_path: Optional[str] = None


class SinglePixelPhaseTestResponse(BaseModel):
    pixel_index: int
    row: int
    col: int
    experimental_png: Optional[str] = None
    pixel_at_pct: Optional[dict[str, float]] = None
    eds_weighting_effective: str = "off"
    mask_applied: bool = False
    bandwidth: int = 128
    candidates: list[PhaseTestCandidate] = []
    excluded: list[PhaseTestCandidate] = []


class ComparisonRequest(BaseModel):
    """Multi-phase, multi-method comparison batch."""
    phases: List[Dict]  # [{name, cif_path, sht_path, master_h5_path}, ...]
    methods: List[str]  # ["hough", "dictionary", "spherical"]
    n_bands: int = 12
    bandwidth: int = 88


class RefineRequest(BaseModel):
    master_h5_path: str
    energy_kv: float = 20.0
    trust_region: float = 5.0


@router.post("/start")
async def start_indexing(req: IndexingStartRequest):
    """Start an indexing job as a background task."""
    from backend.api.routes.ebsd_viewer import _get_active_signal
    if _get_active_signal() is None:
        raise HTTPException(status_code=400, detail="No EBSD data loaded")

    # Fail loud on an unsupported method. The run_indexing task maps the method
    # name with a HOUGH default, so e.g. "embedding" (sent by the frontend's
    # Embedding/FAISS option) silently ran Hough instead — the user thought
    # they'd run neural indexing. Reject it up front until it's implemented.
    _supported_methods = {"hough", "dictionary", "spherical"}
    if req.method not in _supported_methods:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported indexing method {req.method!r}. Supported: "
                   f"{', '.join(sorted(_supported_methods))}. "
                   "(Embedding/FAISS indexing is not implemented in this build.)",
        )

    task_id = str(uuid.uuid4())
    _track_task(task_id, {
        "status": "running",
        "progress": 0.0,
        "message": "Starting indexing...",
        "result": None,
        "error": None,
        # Pending log lines for the user-facing log box. Drained on every
        # /status poll. Keeps a small backlog so we never lose messages
        # between polls — frontend appends each as a separate line.
        "pending_log": [],
    })

    def run_indexing():
        global _active_result_id
        try:
            from indexing_controller import (
                IndexingMethod, PixelSelectionMode, IndexingConfig,
                create_selection_mask, hough_index_patterns,
                dictionary_index_patterns, spherical_index_patterns,
            )

            method_map = {
                "hough": IndexingMethod.HOUGH,
                "dictionary": IndexingMethod.DICTIONARY,
                "spherical": IndexingMethod.SPHERICAL,
            }
            selection_map = {
                "full": PixelSelectionMode.FULL,
                "region": PixelSelectionMode.REGION,
                "mask": PixelSelectionMode.MASK,
            }

            indexing_method = method_map.get(req.method, IndexingMethod.HOUGH)
            selection_mode = selection_map.get(req.selection_mode, PixelSelectionMode.FULL)

            config = IndexingConfig(
                method=indexing_method,
                selection_mode=selection_mode,
                n_bands=req.n_bands,
                t_sigma=req.t_sigma,
                r_sigma=req.r_sigma,
                metric=req.metric,
                keep_n=req.keep_n,
                compute_mode=req.compute_mode,
                bandwidth=req.bandwidth,
                normed=req.normed,
                refine=req.refine,
                nregions=req.nregions,
                circmask=req.circmask,
                gausbckg=req.gausbckg,
                backend=req.backend,
                row_start=req.row_start,
                row_end=req.row_end,
                col_start=req.col_start,
                col_end=req.col_end,
            )

            if req.sht_paths:
                config.sht_file = req.sht_paths[0]

            def _progress(msg, pct=None):
                _indexing_tasks[task_id]["message"] = msg
                if pct is not None:
                    _indexing_tasks[task_id]["progress"] = pct
                # Append to user-facing log queue so meaningful events
                # (per-phase progress, pat/s rate, etc.) show up in the
                # Indexing-page log box. Drained on /status poll.
                pending = _indexing_tasks[task_id].setdefault("pending_log", [])
                pending.append(msg)
                # Cap the queue so a stuck client poll doesn't grow it
                # unbounded — keep the last 500 lines.
                if len(pending) > 500:
                    del pending[:-500]
                logger.info("Indexing progress: %s (%.0f%%)", msg,
                            (_indexing_tasks[task_id]["progress"]) * 100)

            def _cancel_check():
                """Polled by Hough / Dictionary-GPU / Spherical-GPU loops
                to honour the user clicking Stop. ``/stop/{task_id}`` sets
                status to ``failed`` — we just mirror that as the cancel
                signal so the loops can break out and the route's
                ``except IndexingCancelled`` handler does the cleanup.
                """
                return _indexing_tasks.get(task_id, {}).get("status") == "failed"

            # Probe CUDA up front and stash to the task state. The frontend
            # polls /status and renders this as a banner ("Running on CUDA
            # RTX 4070 · 11.2 GB free" vs. "CPU only"), so the user can
            # see at a glance whether the run will be fast and whether a
            # previous run is still hogging VRAM.
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _idx = 0
                    _free_b, _total_b = _torch.cuda.mem_get_info(_idx)
                    _props = _torch.cuda.get_device_properties(_idx)
                    _indexing_tasks[task_id]["runtime"] = {
                        "device": "cuda",
                        "device_name": _props.name,
                        "free_vram_gb": round(_free_b / (1024 ** 3), 2),
                        "total_vram_gb": round(_total_b / (1024 ** 3), 2),
                        "compute_mode": getattr(req, "compute_mode", "auto"),
                        "low_vram_warning": _free_b < (1 << 30),  # <1 GB free
                    }
                else:
                    _indexing_tasks[task_id]["runtime"] = {
                        "device": "cpu",
                        "device_name": "CPU",
                        "free_vram_gb": 0.0,
                        "total_vram_gb": 0.0,
                        "compute_mode": getattr(req, "compute_mode", "auto"),
                        "low_vram_warning": False,
                    }
            except Exception as _exc:
                logger.debug("CUDA probe failed: %s", _exc)
                _indexing_tasks[task_id]["runtime"] = {
                    "device": "unknown",
                    "device_name": "unknown",
                    "free_vram_gb": 0.0,
                    "total_vram_gb": 0.0,
                    "compute_mode": getattr(req, "compute_mode", "auto"),
                    "low_vram_warning": False,
                }

            # Re-fetch the signal here (closures capture names, not values)
            from backend.api.routes.ebsd_viewer import _get_active_signal
            signal = _get_active_signal()
            nav_shape = signal.axes_manager.navigation_shape
            # kikuchipy navigation_shape is (cols, rows) - reverse for (rows, cols)
            if len(nav_shape) >= 2:
                n_rows, n_cols = nav_shape[1], nav_shape[0]
            else:
                n_rows, n_cols = nav_shape[0], 1

            _indexing_tasks[task_id]["message"] = "Building selection mask..."
            _indexing_tasks[task_id]["progress"] = 0.05

            # Build selection mask
            chemistry_mask = None
            if req.mask is not None:
                chemistry_mask = np.array(req.mask, dtype=bool).reshape(n_rows, n_cols)

            region = None
            if selection_mode == PixelSelectionMode.REGION:
                region = (req.row_start, req.row_end, req.col_start, req.col_end)

            # Quick test: override selection to single pixel
            if req.quick_test:
                selection_mask = np.zeros((n_rows, n_cols), dtype=bool)
                r, c = req.qt_row, req.qt_col
                if 0 <= r < n_rows and 0 <= c < n_cols:
                    selection_mask[r, c] = True
                else:
                    raise ValueError(f"Quick test pixel ({r},{c}) out of range ({n_rows}x{n_cols})")
            else:
                selection_mask = create_selection_mask(
                    n_rows=n_rows,
                    n_cols=n_cols,
                    mode=selection_mode,
                    region=region,
                    chemistry_mask=chemistry_mask,
                )

            _indexing_tasks[task_id]["progress"] = 0.1

            # Get detector from CalibrationStore (single source of truth)
            from backend.api.routes.ebsd_viewer import _active_dataset
            detector = calibration_store.get_detector(_active_dataset)

            if detector is None:
                # Fallback: signal's own detector
                detector = getattr(signal, 'detector', None)

            if detector is None:
                # Last resort: build minimal detector from signal shape
                from kikuchipy.detectors import EBSDDetector
                sig_shape = signal.axes_manager.signal_shape
                detector = EBSDDetector(shape=(sig_shape[1], sig_shape[0]))

            # Build detector params for Spherical indexing (needed in both single and multi-phase paths)
            det_params = None
            if indexing_method == IndexingMethod.SPHERICAL:
                from backend.api.routes.ebsd_viewer import _ebsd_file_path
                det_params = build_spherical_det_params(signal, detector, _ebsd_file_path)

            result = None
            n_phase_files = len(req.cif_paths) + len(req.master_h5_paths) + len(req.sht_paths)

            # ============================================================
            # PER-PHASE ROUTING (EDS phase map): each phase runs only
            # on its own classified pixels. Takes precedence over the
            # competitive multi-phase path because the user already
            # decided which phase belongs to which pixel — rerunning
            # everything against everything would just add noise.
            # ============================================================
            if req.use_phase_map_routing:
                from backend.api.services.phase_map_store import get_phase_map_store
                from indexing_controller import (
                    ComparisonConfig, PhaseConfig,
                    run_per_phase_indexing,
                )

                store = get_phase_map_store()
                state = store.get_state()
                if state is None:
                    raise ValueError(
                        "Per-phase routing requested, but no phase map is loaded. "
                        "Auto-classify on the EDS page first, then send to indexing."
                    )

                # Match the request's CIFs to phase-map entries by basename. The
                # phase map only knows filenames; the request brings full paths.
                cif_path_by_filename: Dict[str, str] = {}
                for p in req.cif_paths:
                    try:
                        cif_path_by_filename[Path(p).name] = p
                    except Exception:
                        continue

                # Walk the phase map IN ORDER so the resulting phase indices
                # in consensus_map match the position in phase_configs.
                # Phases without a matching CIF in the request are dropped
                # along with their pixels (fail-loud rather than silently
                # mis-attributing them).
                ordered_configs: List[PhaseConfig] = []
                ordered_masks: List[np.ndarray] = []
                kept_indices: List[int] = []
                for i, entry in enumerate(state.phase_entries):
                    cif_path = cif_path_by_filename.get(entry.cif_filename)
                    if cif_path is None:
                        continue
                    mask_2d = (state.phase_grid == i)
                    if not mask_2d.any():
                        continue
                    pc = PhaseConfig(name=entry.cif_filename, cif_path=cif_path)
                    # Hough/Dictionary need a phase_list; spherical uses sht_path
                    # which the phase map doesn't track. Build a phase_list once
                    # per CIF using the existing helper so per-phase indexing
                    # gets the same data as the legacy single-phase path.
                    try:
                        from ebsd_utils import sanitize_cif
                        from orix.crystal_map import Phase, PhaseList
                        sanitized = sanitize_cif(cif_path)
                        phase = Phase.from_cif(sanitized)
                        if phase.name != Path(cif_path).stem:
                            phase.name = Path(cif_path).stem
                        pc.phase_list = PhaseList(phases=[phase])
                    except Exception as exc:
                        logger.warning(
                            "Per-phase routing: cannot load phase_list for %s: %s",
                            cif_path, exc,
                        )
                        continue
                    ordered_configs.append(pc)
                    ordered_masks.append(mask_2d)
                    kept_indices.append(i)

                if not ordered_configs:
                    raise ValueError(
                        "Per-phase routing: none of the phase-map CIFs are present "
                        "in the request's cif_paths AND have non-empty masks. "
                        "Check that the Indexing page actually pre-selected the "
                        "matching CIFs from the phase map."
                    )

                # Build a consensus_map whose values are positions in
                # ordered_configs (not the original phase_grid indices). Pixels
                # that pointed at a dropped phase become unclassified.
                index_remap = {old: new for new, old in enumerate(kept_indices)}
                consensus = np.full(state.phase_grid.shape, -1, dtype=np.int32)
                for old_idx, new_idx in index_remap.items():
                    consensus[state.phase_grid == old_idx] = new_idx

                comparison_config = ComparisonConfig(
                    phases=ordered_configs,
                    methods=[indexing_method],
                    selection_mode=PixelSelectionMode.MASK,
                    n_bands=req.n_bands,
                    t_sigma=req.t_sigma,
                    r_sigma=req.r_sigma,
                    metric=req.metric,
                    keep_n=req.keep_n,
                    bandwidth=req.bandwidth,
                    normed=req.normed,
                    refine=req.refine,
                    nregions=req.nregions,
                )

                _progress(
                    f"Per-phase routing: {len(ordered_configs)} phases, "
                    f"{int(np.sum([m.sum() for m in ordered_masks]))} pixels total",
                    0.05,
                )
                result = run_per_phase_indexing(
                    signal=signal,
                    detector=detector,
                    method=indexing_method,
                    config=comparison_config,
                    phase_configs=ordered_configs,
                    masks_per_phase=ordered_masks,
                    consensus_map=consensus,
                    h5_path=_ebsd_file_path if indexing_method == IndexingMethod.SPHERICAL else "",
                    detector_params=det_params if indexing_method == IndexingMethod.SPHERICAL else None,
                    progress_cb=_progress,
                )
                # Stash the per-phase metadata so the result UI can show
                # what got routed where.
                if result.metadata is None:
                    result.metadata = {}
                result.metadata["selection_mode"] = "phase_map_routing"

            # ============================================================
            # MULTI-PHASE PATH: >1 file for Dictionary or Spherical
            # (Hough handles multi-CIF natively via PhaseList)
            # ============================================================
            elif n_phase_files > 1 and indexing_method in (IndexingMethod.DICTIONARY, IndexingMethod.SPHERICAL):
                from indexing_controller import (
                    ComparisonConfig, ComparisonResult, PhaseConfig,
                    run_single_phase_method, compute_comparison_maps,
                    build_consensus_xmap, extract_score_map,
                    spherical_gpu_index_patterns,
                )

                # Fast path: spherical_gpu can handle multi-phase natively
                # in a single backend (one SHT load + one H5 read pass per
                # batch shared across phases). Skips the per-phase Python
                # loop that rebuilds Wigner-d / GL-grid / etc. for every
                # phase, which alone was about 10-20 seconds of overhead
                # per phase at L=88. Other backends (EMSphInx CPU,
                # Dictionary) keep the legacy per-phase loop below.
                gpu_fast_path_done = False
                all_results = None
                if indexing_method == IndexingMethod.SPHERICAL and req.backend == "spherical_gpu":
                    _progress(
                        f"Spherical-GPU multi-phase: {req.sht_paths and len(req.sht_paths) or 0} phase(s)",
                        0.10,
                    )
                    result = spherical_gpu_index_patterns(
                        h5_path=_ebsd_file_path or '',
                        config=config,
                        detector_params=det_params,
                        selection_mask=selection_mask,
                        progress_callback=_progress,
                        sht_paths=list(req.sht_paths or []),
                        cancel_check=_cancel_check,
                    )
                    # Phase names for the legend/exports are set centrally by
                    # _inject_phase_names() after the method dispatch (clean,
                    # export-safe names, consistent across all methods). Removed a
                    # dead block here that imported a non-existent
                    # indexing_controller._SHT_PREFIX_RE — the ImportError was
                    # swallowed, so it never ran.
                    _progress("Multi-phase indexing complete", 0.95)
                    gpu_fast_path_done = True
                else:
                    phase_configs = _build_phase_configs(req)
                    comparison_config = ComparisonConfig(
                        phases=phase_configs,
                        methods=[indexing_method],
                        selection_mode=selection_mode,
                        bandwidth=req.bandwidth,
                        normed=req.normed,
                        refine=req.refine,
                        nregions=req.nregions,
                        circmask=req.circmask,
                        gausbckg=req.gausbckg,
                        backend=req.backend,
                        row_start=req.row_start,
                        row_end=req.row_end,
                        col_start=req.col_start,
                        col_end=req.col_end,
                    )

                    all_results = []
                    phase_errors = []
                    for i, pc in enumerate(phase_configs):
                        _progress(
                            f"Phase {i+1}/{len(phase_configs)}: {pc.name} — {req.method} indexing...",
                            0.1 + 0.8 * (i / len(phase_configs)),
                        )
                        try:
                            pmr = run_single_phase_method(
                                signal=signal,
                                detector=detector,
                                phase_config=pc,
                                method=indexing_method,
                                config=comparison_config,
                                selection_mask=selection_mask,
                                h5_path=_ebsd_file_path if indexing_method == IndexingMethod.SPHERICAL else '',
                                detector_params=det_params if indexing_method == IndexingMethod.SPHERICAL else None,
                            )
                            all_results.append(pmr)
                        except Exception as e:
                            import traceback
                            logger.warning(f"Phase {pc.name} failed: {e}\n{traceback.format_exc()}")
                            _progress(f"Phase {pc.name} FAILED: {e}")
                            phase_errors.append(f"{pc.name}: {e}")
                            continue

                    if not all_results:
                        error_details = "; ".join(phase_errors) if phase_errors else "unknown"
                        raise ValueError(f"All phases failed during multi-phase indexing: {error_details}")

                # GPU fast-path already produced a complete IndexingResult;
                # skip the legacy per-phase merge code that needs all_results.
                if gpu_fast_path_done:
                    pass
                else:
                    _progress("Merging results...", 0.95)

                    comparison = ComparisonResult(
                        results=all_results,
                        phases=phase_configs,
                        methods=[indexing_method],
                        original_shape=(n_rows, n_cols),
                        selection_mask=selection_mask,
                    )
                    comparison = compute_comparison_maps(comparison)
                    merged_xmap = build_consensus_xmap(comparison)

                # Skip the per-phase merge bookkeeping for the GPU fast path
                # — its IndexingResult is already complete with phase_id and
                # multi-phase metadata baked in.
                if not gpu_fast_path_done:
                    # Build per-phase stats
                    per_phase_stats = []
                    total_indexed = int(selection_mask.sum()) if selection_mask is not None else 0
                    if comparison.best_phase_per_pixel is not None:
                        bp = comparison.best_phase_per_pixel[selection_mask] if selection_mask is not None else comparison.best_phase_per_pixel.ravel()
                        for pi, pc in enumerate(phase_configs):
                            px_count = int((bp == pi).sum())
                            pmr_match = [r for r in all_results if r.phase_name == pc.name]
                            mean_ci = float(pmr_match[0].mean_score) if pmr_match else 0.0
                            per_phase_stats.append({
                                'name': pc.name,
                                'ci': round(mean_ci, 4),
                                'pixels': px_count,
                                'fraction': round(px_count / max(total_indexed, 1) * 100, 1),
                            })
                        per_phase_stats.sort(key=lambda x: x['pixels'], reverse=True)

                    best_pmr = max(all_results, key=lambda r: r.mean_score)

                    # Preserve per-phase results for rich export
                    per_phase_data = {}
                    n_total = int(np.prod(comparison.original_shape))
                    sel_mask_flat = None
                    if comparison.selection_mask is not None:
                        sm = np.asarray(comparison.selection_mask).ravel().astype(bool)
                        if sm.size == n_total and not sm.all():
                            sel_mask_flat = sm

                    for pmr in all_results:
                        try:
                            pmr_euler = pmr.indexing_result.xmap.rotations.to_euler(degrees=False)
                            pmr_ci = pmr.indexing_result.confidence_scores

                            euler_full = None
                            if pmr_euler is not None:
                                euler_arr = np.asarray(pmr_euler).reshape(-1, 3).astype(np.float32)
                                if sel_mask_flat is not None and euler_arr.shape[0] == int(sel_mask_flat.sum()):
                                    euler_full = np.full((n_total, 3), np.nan, dtype=np.float32)
                                    euler_full[sel_mask_flat] = euler_arr
                                elif euler_arr.shape[0] == n_total:
                                    euler_full = euler_arr
                                else:
                                    raise ValueError(
                                        f"per-phase euler size {euler_arr.shape[0]} matches "
                                        f"neither mask ({int(sel_mask_flat.sum()) if sel_mask_flat is not None else '—'}) "
                                        f"nor full grid ({n_total})"
                                    )
                                euler_full = euler_full.reshape((*comparison.original_shape, 3))

                            ci_full = None
                            if pmr_ci is not None:
                                ci_arr = np.asarray(pmr_ci).astype(np.float32).ravel()
                                if sel_mask_flat is not None and ci_arr.shape[0] == int(sel_mask_flat.sum()):
                                    ci_full = np.full(n_total, np.nan, dtype=np.float32)
                                    ci_full[sel_mask_flat] = ci_arr
                                elif ci_arr.shape[0] == n_total:
                                    ci_full = ci_arr
                                else:
                                    raise ValueError(
                                        f"per-phase CI size {ci_arr.shape[0]} matches "
                                        f"neither mask nor full grid ({n_total})"
                                    )
                                ci_full = ci_full.reshape(comparison.original_shape)

                            per_phase_data[pmr.phase_name] = {
                                'euler': euler_full,
                                'ci': ci_full,
                                'ci_mean': float(pmr.mean_score),
                            }
                        except Exception as e:
                            logger.warning("Could not extract per-phase data for %s: %s", pmr.phase_name, e)

                    from indexing_controller import IndexingResult
                    result = IndexingResult(
                        xmap=merged_xmap,
                        selection_mask=selection_mask,
                        original_shape=(n_rows, n_cols),
                        method=indexing_method,
                        confidence_scores=best_pmr.indexing_result.confidence_scores,
                        metadata={
                            'multi_phase': True,
                            'per_phase_stats': per_phase_stats,
                            'n_phases': len(phase_configs),
                            'per_phase_data': per_phase_data,
                        },
                    )

            # ============================================================
            # SINGLE-PHASE PATH: original logic (unchanged)
            # ============================================================

            # --- Hough indexing ---
            elif indexing_method == IndexingMethod.HOUGH:
                if not req.cif_paths:
                    raise ValueError("Hough indexing requires at least one CIF file path")

                from orix.crystal_map import Phase, PhaseList
                from ebsd_utils import sanitize_cif

                from pathlib import Path as P

                phases = []
                for cif_path in req.cif_paths:
                    _progress(f"Loading phase from {cif_path}...")
                    phase = Phase.from_cif(sanitize_cif(cif_path))
                    # Restore original name if sanitize_cif created a temp file
                    original_stem = P(cif_path).stem
                    if phase.name != original_stem:
                        phase.name = original_stem
                    phases.append(phase)
                phase_list = PhaseList(phases)

                _indexing_tasks[task_id]["progress"] = 0.2
                _progress("Running Hough indexing...")

                def _cancel_check():
                    return _indexing_tasks.get(task_id, {}).get("status") == "failed"

                result = hough_index_patterns(
                    signal=signal,
                    phase_list=phase_list,
                    detector=detector,
                    config=config,
                    selection_mask=selection_mask,
                    progress_callback=_progress,
                    cancel_check=_cancel_check,
                )

            # --- Dictionary indexing ---
            elif indexing_method == IndexingMethod.DICTIONARY:
                if not req.master_h5_paths:
                    raise ValueError("Dictionary indexing requires a master H5 file path")

                import kikuchipy as kp
                _progress(f"Loading dictionary from {req.master_h5_paths[0]}...")
                dictionary = kp.load(req.master_h5_paths[0])

                _indexing_tasks[task_id]["progress"] = 0.2
                _progress("Running dictionary indexing...")

                result = dictionary_index_patterns(
                    signal=signal,
                    dictionary=dictionary,
                    config=config,
                    selection_mask=selection_mask,
                    progress_callback=_progress,
                    cancel_check=_cancel_check,
                    detector=detector,   # store-resolved (refined PC) — P2-B
                )

            # --- Spherical indexing (EMSphinx, single phase) ---
            elif indexing_method == IndexingMethod.SPHERICAL:
                if not req.sht_paths:
                    raise ValueError("Spherical indexing requires at least one SHT file path")

                _indexing_tasks[task_id]["progress"] = 0.2

                # Phase 5: dispatch on backend selector. Default "emsphinx" runs the
                # WSL EMSphInx CLI (legacy CPU path); "spherical_gpu" runs the
                # in-process PyTorch backend (~11x faster, GPU required).
                backend_choice = getattr(config, "backend", "emsphinx")
                if backend_choice == "spherical_gpu":
                    _progress("Running spherical (PyTorch GPU) indexing...")
                    from indexing_controller import spherical_gpu_index_patterns
                    result = spherical_gpu_index_patterns(
                        h5_path=_ebsd_file_path or '',
                        config=config,
                        detector_params=det_params,
                        selection_mask=selection_mask,
                        progress_callback=_progress,
                        cancel_check=_cancel_check,
                    )
                else:
                    _progress("Running spherical (EMSphinx CPU) indexing...")
                    result = spherical_index_patterns(
                        h5_path=_ebsd_file_path or '',
                        config=config,
                        detector_params=det_params,
                        selection_mask=selection_mask,
                        progress_callback=_progress,
                    )

            else:
                raise ValueError(f"Unknown indexing method: {req.method}")

            # Inject actual phase names into xmap before storing
            _inject_phase_names(result, req)

            # Attach metadata needed by the SHT forward renderer in
            # /api/indexing/pattern-match (Phase A of FEAT-SHT-FWD).
            _attach_indexing_metadata(
                result,
                method_name=req.method,
                sht_paths=req.sht_paths if req.method == "spherical" else None,
                det_params=det_params,
            )

            # Don't overwrite a multi-pixel result with a single-pixel quick test
            if not (req.quick_test and _get_result() is not None):
                _store_result(result, req.method)
            _indexing_tasks[task_id]["status"] = "completed"
            _indexing_tasks[task_id]["progress"] = 1.0
            _indexing_tasks[task_id]["message"] = "Indexing complete"
            task_result = {
                "n_indexed": int(result.selection_mask.sum()),
                "original_shape": list(result.original_shape),
                "method": result.method.value,
                "has_confidence": result.confidence_scores is not None,
            }
            if result.confidence_scores is not None:
                # Defensive collapse for 2D pyebsdindex cm — same fix as
                # commit c474f41 for /results endpoint. Without this the
                # mean_ci shown after indexing averages phase-0 + consensus
                # rows together, inflating the displayed value.
                arr = np.asarray(result.confidence_scores)
                if arr.ndim >= 2:
                    arr = arr[-1]
                valid = arr[~np.isnan(arr)] if hasattr(arr, '__len__') else arr
                if len(valid) > 0:
                    task_result["mean_ci"] = float(np.nanmean(valid))
            # Include phase names for Quick Test display (exclude not_indexed)
            if hasattr(result, 'xmap') and result.xmap is not None:
                try:
                    phases_in = result.xmap.phases_in_data
                    task_result["phases"] = [
                        {"id": int(pid), "name": name}
                        for pid, name in zip(phases_in.ids, phases_in.names)
                        if pid != -1
                    ]
                except Exception:
                    pass
            # Include per-phase stats for multi-phase results
            if hasattr(result, 'metadata') and result.metadata.get('multi_phase'):
                task_result["multi_phase"] = True
                task_result["per_phase_stats"] = result.metadata.get('per_phase_stats', [])
            _indexing_tasks[task_id]["result"] = task_result

        except Exception as e:
            import traceback
            # Check if this was a user cancellation (not a real error)
            from indexing_controller import IndexingCancelled, _release_cuda_cache
            if isinstance(e, IndexingCancelled):
                logger.info("Indexing cancelled by user")
                _indexing_tasks[task_id]["status"] = "failed"
                _indexing_tasks[task_id]["message"] = "Cancelled by user"
                # Release any GPU tensors the cancelled run had allocated
                # so the next indexing attempt sees the full VRAM budget.
                # Without this Spherical-GPU after a cancelled Dict-GPU
                # sees ~0 GB free and runs at batch=1.
                _release_cuda_cache()
                return
            tb = traceback.format_exc()
            logger.exception("Indexing failed")
            _indexing_tasks[task_id]["status"] = "failed"
            _indexing_tasks[task_id]["error"] = f"{e}\n{tb}"
            # Same defensive cleanup on hard failures (e.g. OOM partway
            # through). Cheap when CUDA isn't in use.
            try:
                _release_cuda_cache()
            except Exception:
                pass

    import asyncio
    asyncio.get_event_loop().run_in_executor(None, run_indexing)
    return {"task_id": task_id, "status": "running"}


@router.get("/status/{task_id}")
async def get_indexing_status(task_id: str):
    """Check indexing task status. Drains the pending log queue so the
    frontend can pull every progress message between polls — joins the
    queued lines with newlines into a single ``log`` string the
    IndexingPage frontend appends to its log box."""
    if task_id not in _indexing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    state = _indexing_tasks[task_id]
    # Atomically grab and clear the pending log queue.
    pending = state.get("pending_log") or []
    log_str = "\n".join(pending) if pending else None
    if pending:
        state["pending_log"] = []
    # Return a shallow copy with the joined log string injected. Don't
    # mutate the underlying state's "log" field (other code may rely on
    # the absence of that key for non-task fields).
    response = dict(state)
    if log_str is not None:
        response["log"] = log_str
    response.pop("pending_log", None)
    return response


@router.post("/stop/{task_id}")
async def stop_indexing(task_id: str):
    """Cancel a running indexing task.

    Sets the task status to ``failed`` with message ``Cancelled by user``.
    The worker thread polls this flag inside Hough / Dictionary-GPU /
    Spherical-GPU loops and raises ``IndexingCancelled``, which the route
    catches and runs VRAM cleanup. Returns immediately so the UI is not
    blocked on the worker actually noticing — the worker may be inside a
    non-interruptible kikuchipy / kp.load() call and won't see the flag
    until that call returns.
    """
    if task_id not in _indexing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    _indexing_tasks[task_id]["status"] = "failed"
    _indexing_tasks[task_id]["message"] = "Cancelled by user"
    # Best-effort: also release any GPU memory that's already free in
    # the PyTorch caching allocator. PyTorch's allocator is thread-safe;
    # this only reclaims unused blocks (won't touch tensors a still-
    # running batch is using). On the next batch boundary the worker
    # raises CancelledIndexingError and runs a fuller cleanup.
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.debug("stop_indexing GPU cleanup failed: %s", exc)
    return {"success": True, "task_id": task_id}


@router.get("/gpu-status")
async def gpu_status():
    """Probe CUDA without changing anything — used by the frontend banner
    on mount so the user sees free VRAM before starting an indexing run.
    """
    try:
        import torch
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info(0)
            props = torch.cuda.get_device_properties(0)
            free_gb = round(free_b / (1024 ** 3), 2)
            total_gb = round(total_b / (1024 ** 3), 2)
            return {
                "device": "cuda",
                "device_name": props.name,
                "free_vram_gb": free_gb,
                "total_vram_gb": total_gb,
                "low_vram_warning": free_b < (1 << 30),
            }
    except Exception as exc:
        logger.debug("gpu_status probe failed: %s", exc)
    return {
        "device": "cpu",
        "device_name": "CPU",
        "free_vram_gb": 0.0,
        "total_vram_gb": 0.0,
        "low_vram_warning": False,
    }


@router.post("/release-gpu")
async def release_gpu_memory():
    """Force-release GPU memory held by PyTorch's caching allocator AND by the
    module-level phase-test / pattern-compare backend cache.

    Emergency button for when an indexing run — or a Single-Pixel Phase Test —
    left tensors on the GPU. The Phase Test caches ``SphericalGPUBackend``
    objects (per phase-set / bandwidth) that hold *live* GPU references, so a
    bare ``empty_cache()`` would reclaim nothing ("0 MiB freed") and the card
    stays pinned at ~0 free. We therefore EVICT those caches first, then
    empty_cache, so the freed blocks actually return to the driver. Returns the
    free VRAM before/after + how many backends/tables were evicted so the
    frontend can confirm something happened.

    Does NOT interrupt a running task — only reclaims memory not in active use.
    To stop a running run, use ``/stop/{task_id}``.
    """
    info = {"device": "cpu", "freed_mb": 0, "free_before_gb": 0.0,
            "free_after_gb": 0.0, "backends_evicted": 0, "tables_cleared": 0}

    # Drop the live-referenced GPU caches FIRST. This is the actual fix —
    # without it empty_cache() frees nothing after a Phase Test. Done
    # unconditionally so the references die even on a CPU-only box.
    try:
        info["backends_evicted"] = _evict_phase_compare_backends()
    except Exception as exc:
        logger.warning("release_gpu_memory: backend eviction failed: %s", exc)
    # Drop the SHT renderer's per-phase LambertGrid cache + singleton — a
    # high-bandwidth grid is several GB and is held by the Pattern Match, the
    # variant-flip tool and the Phase Test. empty_cache() alone can't reclaim it
    # (live refs), so without this the button reports "0 MiB freed".
    try:
        from backend.api.services.sht_pattern_renderer import release_gpu_caches
        info["render_phases_cleared"] = release_gpu_caches()
    except Exception as exc:
        logger.warning("release_gpu_memory: renderer cache clear failed: %s", exc)
    try:
        from backend.spherical_gpu._math._wigner_logspace import clear_half_pi_caches
        info["tables_cleared"] = clear_half_pi_caches()
    except Exception as exc:
        logger.debug("release_gpu_memory: wigner cache clear failed: %s", exc)

    try:
        import torch, gc
        if torch.cuda.is_available():
            free_before, _ = torch.cuda.mem_get_info(0)
            gc.collect()
            torch.cuda.empty_cache()
            free_after, _ = torch.cuda.mem_get_info(0)
            info["device"] = "cuda"
            info["device_name"] = torch.cuda.get_device_properties(0).name
            info["free_before_gb"] = round(free_before / (1024 ** 3), 2)
            info["free_after_gb"] = round(free_after / (1024 ** 3), 2)
            info["freed_mb"] = round((free_after - free_before) / (1024 ** 2), 1)
        else:
            gc.collect()
    except Exception as exc:
        logger.warning("release_gpu_memory failed: %s", exc)
        info["error"] = str(exc)
    return info


@router.post("/refine")
async def refine_orientations_endpoint(req: RefineRequest):
    """Refine orientations from the last dictionary indexing result using Nelder-Mead."""
    if _get_result() is None:
        raise HTTPException(status_code=400, detail="No indexing result available")
    if _get_result().method.value != "dictionary":
        raise HTTPException(status_code=400, detail="Refinement only available after dictionary indexing")

    if not Path(req.master_h5_path).exists():
        raise HTTPException(status_code=400, detail=f"Master pattern file not found: {req.master_h5_path}")

    # Get PC from CalibrationStore (with fallback to the signal detector).
    # The PC is the quantity indexing/refinement stands and falls on — never
    # refine against a fabricated (0.5,0.5,0.5). Doing so produced a
    # geometrically wrong refinement that silently overwrote the result and
    # reported "done" (the user-reported "refinement after indexing doesn't
    # work"). If no real PC is available, fail loud.
    try:
        from backend.api.routes.ebsd_viewer import _active_dataset
        pc = calibration_store.get_pc(_active_dataset)
    except Exception:
        logger.exception("refine: calibration_store.get_pc failed")
        pc = None

    if pc is None:
        try:
            from backend.api.routes.ebsd_viewer import _get_active_signal
            sig = _get_active_signal()
            pc = tuple(sig.detector.pc_flattened.mean(axis=0)) if sig is not None else None
        except Exception:
            logger.exception("refine: failed to read PC from active signal detector")
            pc = None

    if pc is None:
        raise HTTPException(
            status_code=400,
            detail="No pattern center available for the active dataset — "
                   "calibrate / refine the PC (PC Refinement page) before refining orientations.",
        )
    logger.info("Refinement using PC %s", tuple(round(float(x), 4) for x in pc))

    task_id = str(uuid.uuid4())
    _track_task(task_id, {
        "status": "running",
        "progress": 0.0,
        "message": "Starting refinement...",
        "result": None,
        "error": None,
    })

    def run_refine():
        try:
            from indexing_controller import refine_orientations

            def _progress(msg, pct=None):
                _indexing_tasks[task_id]["message"] = msg
                if pct is not None:
                    _indexing_tasks[task_id]["progress"] = pct
                logger.info("Refine: %s", msg)

            refined_xmap = refine_orientations(
                result=_get_result(),
                master_h5_path=req.master_h5_path,
                energy=req.energy_kv,
                pc=tuple(pc),
                trust_region=req.trust_region,
                progress_callback=_progress,
            )

            # Update the stored result with refined xmap
            _get_result().xmap = refined_xmap
            _indexing_tasks[task_id]["status"] = "done"
            _indexing_tasks[task_id]["progress"] = 1.0
            _indexing_tasks[task_id]["message"] = "Refinement complete"

        except Exception as e:
            import traceback
            logger.exception("Refinement failed")
            _indexing_tasks[task_id]["status"] = "failed"
            _indexing_tasks[task_id]["error"] = f"{e}\n{traceback.format_exc()}"

    import asyncio
    asyncio.get_event_loop().run_in_executor(None, run_refine)
    return {"task_id": task_id, "status": "running"}


@router.get("/pattern-match")
async def get_pattern_match(
    row: int, col: int, rank: int = 0, result_id: str = None,
    max_bandwidth: Optional[int] = None,
    aperture: str = "auto",
    aperture_radius: float = 1.0,
    compare_phases: bool = False,
):
    """Get experimental vs simulated pattern comparison with NCC image and R-score.

    Parameters
    ----------
    max_bandwidth : Optional[int]
        For spherical results: truncate the SHT to this bandwidth when
        rendering the simulated pattern. Default = 128 (fast).
        256 = sharper bands (~1.7 GB VRAM), 384 = sharpest (~5.7 GB).
    aperture : str
        Circular-aperture handling for the R/NCC computation:
        ``"auto"`` (default) heuristically detects a black-corner aperture,
        ``"circular"`` forces the mask on (use this for EDAX phosphor patterns
        whose corners are dark-grey, not black, so auto-detection misses them),
        ``"full"`` forces it off. Anything else falls back to ``"auto"``.
    aperture_radius : float
        Radius of the circular mask as a fraction of the inscribed circle,
        clamped to ``[0.3, 1.0]``. Only used when the mask is active.
    compare_phases : bool
        When True AND the active result is a multi-phase spherical run
        (i.e. ``result.metadata["sht_paths_by_phase"]`` has > 1 entry),
        also re-index this single pixel against EVERY phase's SHT, render
        the per-phase simulated pattern, and return them as
        ``phase_results`` in the response (sorted by R-score, best first).
        The dialog uses this to let the user step through "what would
        Phase X have looked like at this pixel" — the key misindex
        diagnostic for chemically-degenerate phase candidates. ~0.5-0.8s
        per click on a 12-phase result after the per-result backend is
        warm (subsequent clicks reuse the cached backend).
    """
    aperture = str(aperture).lower()
    if aperture not in ("auto", "circular", "full"):
        aperture = "auto"
    try:
        aperture_radius = float(aperture_radius)
    except (TypeError, ValueError):
        aperture_radius = 1.0
    aperture_radius = min(max(aperture_radius, 0.3), 1.0)
    result = _get_result(result_id)
    if result is None:
        raise HTTPException(status_code=400, detail="No indexing result available")

    from tools.pattern_comparison import (
        get_experimental_pattern, get_best_match_pattern,
        compute_ncc_image, compute_ncc_scalar,
        detect_circular_aperture, circular_mask, apply_circular_mask,
        compute_ncc_scalar_masked,
    )

    n_rows, n_cols = result.original_shape
    if not (0 <= row < n_rows and 0 <= col < n_cols):
        raise HTTPException(status_code=400, detail=f"Pixel ({row}, {col}) out of bounds")

    exp = get_experimental_pattern(result, row, col)

    # Convert patterns to base64 PNG
    import io, base64
    from PIL import Image

    def _pattern_to_b64(arr):
        if arr is None:
            return None
        arr_norm = ((arr - arr.min()) / max(float(np.ptp(arr)), 1e-7) * 255).astype(np.uint8)
        img = Image.fromarray(arr_norm, mode='L')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('ascii')

    # Determine which forward path produces the simulated pattern.
    # Spherical Indexing has no in-memory dictionary; instead we render
    # directly from the SHT file using the GPU forward operator.
    indexing_method = (
        (result.metadata or {}).get("indexing_method", "dictionary")
        if hasattr(result, "metadata") else "dictionary"
    )
    simulated_source: str | None = None
    simulated_error: str | None = None
    sim_b64_spherical: str | None = None

    if indexing_method == "spherical":
        sim = None
        try:
            from backend.api.services.sht_pattern_renderer import (
                SHTRenderError as _SHTRenderError,
                render_pattern_to_png_b64 as _render_sht_b64,
            )
            import torch as _torch
            from backend.spherical_gpu.pipeline.detector import (
                convert_pc_to_emsoft as _conv_emsoft,
            )

            # Locate xmap pixel index (mirrors logic used further below)
            xmap_local = result.xmap
            flat_mask_local = result.selection_mask.ravel()
            flat_idx_local = row * n_cols + col
            n_xmap_local = xmap_local.rotations.size
            n_total_local = n_rows * n_cols
            if n_xmap_local == n_total_local:
                px_idx_local = flat_idx_local
            elif flat_mask_local[flat_idx_local]:
                px_idx_local = int(flat_mask_local[:flat_idx_local].sum())
            else:
                raise _SHTRenderError(
                    f"Pixel ({row},{col}) was not indexed in this run."
                )

            phase_id_local = int(xmap_local.phase_id[px_idx_local])
            sht_map = (result.metadata or {}).get("sht_paths_by_phase") or {}
            # Accept both int and str keys (JSON round-trip safety)
            sht_path_local = sht_map.get(phase_id_local) or sht_map.get(str(phase_id_local))
            if not sht_path_local:
                raise _SHTRenderError(
                    f"No SHT registered for phase_id={phase_id_local}. "
                    f"Re-run indexing to populate sht_paths_by_phase metadata."
                )

            det = (result.metadata or {}).get("detector_geometry")
            if not det:
                raise _SHTRenderError(
                    "detector_geometry missing from result.metadata; "
                    "re-run indexing on this dataset."
                )

            # orix Rotation -> quaternion (w,x,y,z)
            rot_obj = xmap_local.rotations[px_idx_local]
            q_data = np.asarray(rot_obj.data).reshape(-1)[:4]
            quat_t = _torch.tensor(q_data, dtype=_torch.float64)

            # Convert vendor PC to EMsoft (xpc, ypc, L) for the renderer
            xpc, ypc, L_um = _conv_emsoft(
                pc=(float(det["pc_x"]), float(det["pc_y"]), float(det["pc_z"])),
                vendor=str(det.get("vendor", "Bruker")),
                pat_width=int(det["pat_width"]),
                pat_height=int(det["pat_height"]),
                pixel_size=float(det.get("pixel_size", 70.0)),
                binning=int(det.get("binning", 1)),
            )

            # render() expects SAMPLE tilt as ``tilt_deg`` (the parameter
            # uses sample-tilt semantics post FEAT-SHT-FWD-A fix). Prefer
            # the explicit ``sample_tilt`` field; ``tilt`` is the detector
            # tilt and MUST be passed separately as ``det_tilt_deg`` —
            # Tier1Indexer's alpha = 90 - sample_tilt + det_tilt. If we
            # omit det_tilt_deg here, the renderer uses 0 and the
            # simulated pattern is rotated by exactly det.tilt away from
            # the experimental, producing visual NCC ~ 0 even when the
            # indexer's orientation is correct. (Bug 2026-05-22: broke
            # EDAX datasets with non-zero detector tilt, e.g. LoGainNi.h5
            # with det.tilt=10 deg gave R = 0.03 instead of R > 0.4.)
            sim_b64_spherical = _render_sht_b64(
                sht_path=sht_path_local,
                orientation_quat=quat_t,
                pc_emsoft=(float(xpc), float(ypc), float(L_um)),
                detector_shape=(int(det["pat_height"]), int(det["pat_width"])),
                pixel_size_um=float(det.get("pixel_size", 70.0)),
                tilt_deg=float(det.get("sample_tilt", 70.0)),
                det_tilt_deg=float(det.get("tilt", 0.0)),
                max_bandwidth=max_bandwidth,
            )
            simulated_source = "sht_forward"

            # Decode b64 PNG back to grayscale ndarray for NCC computation
            from PIL import Image as _ImageDecode
            buf_dec = io.BytesIO(base64.b64decode(sim_b64_spherical))
            sim = np.asarray(_ImageDecode.open(buf_dec), dtype=np.float32)
        except Exception as e:
            # Fail-loud per spec: surface the error message in the response
            simulated_error = str(e)
            sim = None
            logger.warning(
                f"[pattern-match] SHT forward render failed for ({row},{col}): {e}"
            )
    else:
        sim = get_best_match_pattern(result, row, col, rank=rank)
        if sim is not None:
            simulated_source = "dictionary"

    # Circular detector aperture (EDAX-style).
    # When the experimental pattern was recorded through a round phosphor its
    # corners carry no Kikuchi signal. The simulated pattern + NCC image are
    # full squares and would otherwise "show beyond the detector" and dilute
    # the R-score; mask them so the comparison only covers the physical
    # detector area.
    #
    # `aperture` controls this: "auto" detects black corners (only fires when
    # corners are near-black); "circular" forces the mask on — needed for EDAX
    # phosphor patterns whose corners are dark-grey, not black, so the
    # heuristic misses them; "full" forces it off.
    aperture_circular = False
    aperture_mask = None
    if exp is not None and sim is not None:
        try:
            if np.asarray(exp).shape == np.asarray(sim).shape:
                if aperture == "circular":
                    aperture_circular = True
                elif aperture == "full":
                    aperture_circular = False
                else:  # "auto"
                    aperture_circular = detect_circular_aperture(exp)
                if aperture_circular:
                    aperture_mask = circular_mask(
                        np.asarray(exp).shape, radius_frac=aperture_radius,
                    )
        except Exception as e:
            logger.warning(f"[pattern-match] aperture handling failed: {e}")
            aperture_circular = False
            aperture_mask = None

    # Mask the simulated pattern so its corners match the experimental
    # circular aperture before it is encoded to PNG further below.
    if aperture_circular and aperture_mask is not None and sim is not None:
        sim = apply_circular_mask(sim, aperture_mask)

    # Compute NCC image and R-score
    ncc_image_b64 = None
    r_score = None
    r_quality = None
    if exp is not None and sim is not None:
        ncc_img = compute_ncc_image(exp, sim)
        if aperture_circular and aperture_mask is not None:
            # Recompute NCC over only the circular region so the matching
            # black corners don't dilute the score, and zero the NCC image
            # corners for a consistent display.
            r_score = compute_ncc_scalar_masked(exp, sim, aperture_mask)
            ncc_img = apply_circular_mask(ncc_img, aperture_mask)
        else:
            r_score = compute_ncc_scalar(exp, sim)

        # Render NCC image with RdBu_r colormap
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.cm as cm

        # Normalize NCC to [0,1] for colormap (NCC range is ~[-1, 1])
        vmax = max(abs(float(np.nanmax(ncc_img))), abs(float(np.nanmin(ncc_img))), 0.01)
        vmax = min(vmax, 1.0)
        norm_img = (ncc_img + vmax) / (2 * vmax)  # map [-vmax, vmax] to [0, 1]
        colored = (cm.RdBu_r(norm_img) * 255).astype(np.uint8)[:, :, :3]
        if aperture_circular and aperture_mask is not None:
            # Black out the colormapped corners (RdBu_r maps 0.5->white).
            colored = apply_circular_mask(colored, aperture_mask)
        img_ncc = Image.fromarray(colored)
        buf_ncc = io.BytesIO()
        img_ncc.save(buf_ncc, format='PNG')
        ncc_image_b64 = base64.b64encode(buf_ncc.getvalue()).decode('ascii')

        # R quality classification
        if r_score >= 0.3:
            r_quality = "good"
        elif r_score >= 0.15:
            r_quality = "acceptable"
        else:
            r_quality = "poor"

    # Get score, phase, euler from xmap
    xmap = result.xmap
    ncc_score = None
    phase_name = None
    euler_angles = None
    total_ranks = 1

    try:
        flat_mask = result.selection_mask.ravel()
        flat_idx = row * n_cols + col

        # Compute correct index into xmap arrays (partial vs full grid)
        n_xmap = xmap.rotations.size
        n_total = n_rows * n_cols
        if n_xmap == n_total:
            px_idx = flat_idx
        else:
            if not flat_mask[flat_idx]:
                px_idx = None
            else:
                px_idx = int(flat_mask[:flat_idx].sum())

        if px_idx is not None:
            if 'scores' in xmap.prop:
                scores = xmap.prop['scores']
                if scores.ndim == 2:
                    total_ranks = scores.shape[1]
                    r = min(rank, total_ranks - 1)
                    ncc_score = float(scores[px_idx, r])
                else:
                    ncc_score = float(scores[px_idx])
            elif result.confidence_scores is not None:
                ncc_score = float(result.confidence_scores[px_idx])

            # Phase name — direct lookup via phase_id
            try:
                phase_id = int(xmap.phase_id[px_idx])
                phase_name = xmap.phases[phase_id].name
            except Exception:
                # Fallback: first phase name
                try:
                    phase_name = next(iter(xmap.phases_in_data.names))
                except Exception:
                    pass

            rot = xmap.rotations[px_idx]
            euler_angles = list(map(float, np.degrees(rot.to_euler()[0])))
    except Exception as e:
        logger.warning(f"[pattern-match] metadata extraction failed for ({row},{col}): {e}")

    # --- Multi-phase score comparison ---
    # Collect R-scores from ALL results in the registry for this pixel
    phase_scores = []
    for rid, res in _result_registry.items():
        if res is result:
            # Already have this result's score
            if phase_name and r_score is not None:
                phase_scores.append({
                    "phase": phase_name,
                    "r_score": r_score,
                    "result_id": rid,
                    "is_active": True,
                })
            continue
        # Check if this pixel was indexed in this other result
        try:
            other_mask = res.selection_mask.ravel()
            if not other_mask[flat_idx]:
                continue
            other_exp = get_experimental_pattern(res, row, col)
            other_sim = get_best_match_pattern(res, row, col, rank=0)
            if other_exp is None or other_sim is None:
                continue
            # Use the same aperture masking as the active result so the
            # phase-comparison bars are computed on the same basis.
            if (
                aperture_circular and aperture_mask is not None
                and np.asarray(other_exp).shape == aperture_mask.shape
                and np.asarray(other_sim).shape == aperture_mask.shape
            ):
                other_r = compute_ncc_scalar_masked(
                    other_exp, other_sim, aperture_mask,
                )
            else:
                other_r = compute_ncc_scalar(other_exp, other_sim)
            # Get phase name from this result
            other_phase = None
            try:
                other_xmap = res.xmap
                n_other = other_xmap.rotations.size
                n_other_total = res.original_shape[0] * res.original_shape[1]
                if n_other == n_other_total:
                    other_px = flat_idx
                else:
                    other_px = int(other_mask[:flat_idx].sum())
                other_pid = int(other_xmap.phase_id[other_px])
                other_phase = other_xmap.phases[other_pid].name
            except Exception:
                other_phase = rid.split('_')[0]  # fallback to method name
            phase_scores.append({
                "phase": other_phase or rid,
                "r_score": float(other_r),
                "result_id": rid,
                "is_active": False,
            })
        except Exception:
            continue

    # Sort by R-score descending (best first)
    phase_scores.sort(key=lambda x: x["r_score"], reverse=True)

    # For spherical we already have a base64-encoded PNG from the SHT renderer;
    # normally avoid re-encoding the decoded ndarray (which would lose the
    # color-depth baseline). But if a circular aperture mask was applied, the
    # corners of `sim` were zeroed in-place and we MUST re-encode the masked
    # ndarray so the displayed simulated pattern matches the experimental
    # aperture. `sim` for spherical is already uint8 (decoded from PNG).
    if sim_b64_spherical is not None and not (
        aperture_circular and aperture_mask is not None
    ):
        sim_field = sim_b64_spherical
    else:
        sim_field = _pattern_to_b64(sim)

    # Mask the displayed experimental pattern too, so the user visually sees
    # the same aperture that the R-score was computed over.
    exp_display = exp
    if aperture_circular and aperture_mask is not None and exp is not None:
        try:
            exp_display = apply_circular_mask(exp, aperture_mask)
        except Exception as e:
            # Show the unmasked pattern rather than failing the whole
            # diagnostic, but log loudly — masking the display is a pure
            # array op and should never fail; if it does, something is
            # off-shape and the score the user sees may not reflect what
            # they expect.
            logger.warning(
                "[pattern-match] aperture mask on display failed: %s "
                "(falling back to unmasked display; R-score is still masked)",
                e,
            )
            exp_display = exp

    # --- Per-phase comparison (Misindex-diagnose) ------------------------------
    # When ``compare_phases=true`` and we have a multi-phase spherical run,
    # re-index THIS SINGLE PIXEL against every phase's SHT independently
    # and return per-phase (R-score, Euler, simulated PNG). Lets the user
    # step through "what if phase X had won here?" in the dialog. See
    # ``_compute_phase_compare_results`` below for the heavy lifting.
    phase_results = None
    if (
        compare_phases
        and indexing_method == "spherical"
        and exp is not None
    ):
        try:
            sht_map_local = (result.metadata or {}).get("sht_paths_by_phase") or {}
            det_local = (result.metadata or {}).get("detector_geometry")
            if sht_map_local and det_local and len(sht_map_local) >= 1:
                phase_results = _compute_phase_compare_results(
                    exp_pattern=exp,
                    detector_geometry=det_local,
                    sht_paths_by_phase=sht_map_local,
                    phase_list=result.xmap.phases,
                    aperture_mask=aperture_mask if aperture_circular else None,
                    max_bandwidth=max_bandwidth,
                )
        except Exception as e:
            logger.warning(
                "[pattern-match] compare_phases failed for (%d,%d): %s",
                row, col, e,
            )
            phase_results = None

    return {
        "experimental": _pattern_to_b64(exp_display),
        "simulated": sim_field,
        "ncc_image": ncc_image_b64,
        "ncc_score": ncc_score,
        "r_score": r_score,
        "r_quality": r_quality,
        "phase_name": phase_name,
        "euler_angles": euler_angles,
        "total_ranks": total_ranks,
        "rank": rank,
        "pixel_row": row,
        "pixel_col": col,
        "phase_scores": phase_scores,
        "phase_results": phase_results,
        "indexing_method": indexing_method,
        "simulated_source": simulated_source,
        "simulated_error": simulated_error,
        # Provenance: for pseudo-symmetric (cubic-approximant) phases the spherical
        # SO(3) correlation lands on a wrong variant, so the stored orientation
        # comes from Hough band-geometry indexing instead. Surface that here so the
        # UI can label "Orientation: Hough (pseudo-symmetry)".
        "orientation_source": (getattr(result, "metadata", None) or {}).get(
            "orientation_source", "spherical"),
        "orientation_source_reason": (getattr(result, "metadata", None) or {}).get(
            "orientation_source_reason", ""),
        "circular_aperture": aperture_circular,
        "aperture_mode": aperture,
        "aperture_radius": aperture_radius,
    }


# ======================================================================
# Universal manual pseudo-symmetry correction (user-driven grain flip)
# ======================================================================
# When the Pattern Match shows a wrong pseudo-symmetric variant for a phase the
# automatic resolver doesn't cover, the user can: (1) GET /pattern-match/variants
# to see the candidate orientations (current + crystallographic pseudo-variants +
# Hough) each rendered + render-NCC, pick the one that matches; (2) POST
# /pattern-match/apply-to-grain to propagate that correction to the whole grain.
# Works for ANY pseudo-symmetry because the candidate is chosen by render-NCC (the
# ground-truth discriminator) + the user's eye, not by a static symmetry table.

class GrainApplyRequest(BaseModel):
    row: int
    col: int
    quat: list           # chosen corrected orientation [w,x,y,z] at the clicked pixel
    threshold_deg: float = 5.0
    max_total_deg: float = 15.0   # anti-drift cap of the snap flood fill
    refine: bool = False          # guarded Newton polish after the snap
    result_id: str | None = None


def _spherical_pixel_ctx(result, row: int, col: int):
    """Resolve (xmap, px_idx, phase_id, sht_path, det, point_group) for a spherical
    result pixel, raising HTTPException on any problem."""
    if (getattr(result, "metadata", None) or {}).get("indexing_method") != "spherical":
        raise HTTPException(status_code=400,
                            detail="Variant flip is only available for spherical results")
    xmap = result.xmap
    n_rows, n_cols = result.original_shape
    if not (0 <= row < n_rows and 0 <= col < n_cols):
        raise HTTPException(status_code=400, detail=f"Pixel ({row},{col}) out of bounds")
    flat_mask = result.selection_mask.ravel()
    flat_idx = row * n_cols + col
    n_xmap = xmap.rotations.size
    if n_xmap == n_rows * n_cols:
        px_idx = flat_idx
    elif flat_mask[flat_idx]:
        px_idx = int(flat_mask[:flat_idx].sum())
    else:
        raise HTTPException(status_code=400, detail=f"Pixel ({row},{col}) was not indexed")
    phase_id = int(xmap.phase_id[px_idx])
    sht_map = (result.metadata or {}).get("sht_paths_by_phase") or {}
    sht_path = sht_map.get(phase_id) or sht_map.get(str(phase_id))
    det = (result.metadata or {}).get("detector_geometry")
    if not sht_path or not det:
        raise HTTPException(status_code=400,
                            detail="Result missing SHT/detector metadata; re-run indexing")
    pg = None
    try:
        pg = xmap.phases[phase_id].point_group.name
    except Exception:
        try:
            pg = list(xmap.phases)[0].point_group.name
        except Exception:
            pg = None
    return xmap, px_idx, phase_id, str(sht_path), det, pg


# The v1 rigid-C flood fill (`_grain_floodfill`) lived here; grain-flip v2
# replaced it with the per-pixel coset-snap fill
# `backend.spherical_gpu.pseudosym.grain_snap_floodfill` (pure, unit-tested in
# tests/test_grain_flip.py) — one apply now also fixes grains whose pixels are
# scattered across DIFFERENT wrong variants, while preserving intra-grain
# distortion (each pixel keeps its own measured orientation, flipped to the
# correct variant).


@router.get("/pattern-match/variants")
async def get_pattern_match_variants(
    row: int, col: int, result_id: str = None,
    max_bandwidth: int = 128, aperture: str = "auto", aperture_radius: float = 1.0,
):
    """Candidate orientations for the clicked pixel — current + crystallographic
    pseudo-variants + Hough — each rendered through the SHT forward model with its
    render-NCC vs the experimental pattern, sorted best-first. The user picks the
    one whose simulated pattern matches, then applies it to the grain."""
    result = _get_result(result_id)
    if result is None:
        raise HTTPException(status_code=400, detail="No indexing result available")
    xmap, px_idx, phase_id, sht_path, det, pg = _spherical_pixel_ctx(result, row, col)

    from tools.pattern_comparison import (
        get_experimental_pattern, compute_ncc_scalar,
        compute_ncc_scalar_masked, detect_circular_aperture, circular_mask,
    )
    from backend.api.services.sht_pattern_renderer import (
        render_pattern_to_png_b64 as _render_sht_b64,
    )
    from backend.spherical_gpu.pipeline.detector import convert_pc_to_emsoft as _conv_emsoft
    from backend.spherical_gpu.pseudosym import pseudosym_variant_quats
    from orix.quaternion import Rotation as _R
    import torch as _torch
    from PIL import Image as _Img
    import io as _io, base64 as _b64

    exp = get_experimental_pattern(result, row, col)
    if exp is None:
        raise HTTPException(status_code=400, detail="No experimental pattern for this pixel")
    exp = np.asarray(exp, dtype=np.float32)

    q_cur = np.asarray(xmap.rotations[px_idx].data).reshape(-1)[:4].astype(np.float64)
    variants = pseudosym_variant_quats(q_cur, pg or "1")     # (K,4), current first
    labels = (["current"] + [f"variant {i}" for i in range(1, len(variants))])
    quats = [np.asarray(v, dtype=np.float64) for v in variants]
    # Hough candidate (band geometry — the universal correct orientation)
    try:
        from indexing_controller import _resolve_cif_for_sht
        _cif = _resolve_cif_for_sht(sht_path)
        eu_h = _hough_euler_for_phase(exp, _cif, det) if _cif else None
        if eu_h is not None:
            qh = np.asarray(_R.from_euler(np.asarray(eu_h, float).reshape(1, 3)).data
                            ).reshape(-1)[:4].astype(np.float64)
            quats.append(qh)
            labels.append("Hough")
    except Exception:
        logger.debug("[variants] Hough candidate unavailable", exc_info=True)

    xpc, ypc, L_um = _conv_emsoft(
        pc=(float(det["pc_x"]), float(det["pc_y"]), float(det["pc_z"])),
        vendor=str(det.get("vendor", "Bruker")),
        pat_width=int(det["pat_width"]), pat_height=int(det["pat_height"]),
        pixel_size=float(det.get("pixel_size", 70.0)), binning=int(det.get("binning", 1)),
    )
    ds = (int(det["pat_height"]), int(det["pat_width"]))
    mask = None
    ap = str(aperture).lower()
    if ap == "circular" or (ap == "auto" and detect_circular_aperture(exp)):
        mask = circular_mask(ds, radius_frac=min(max(float(aperture_radius), 0.3), 1.0))

    cands = []
    for lab, q in zip(labels, quats):
        try:
            qt = _torch.tensor(np.asarray(q)[:4], dtype=_torch.float64)
            b64 = _render_sht_b64(
                sht_path=sht_path, orientation_quat=qt,
                pc_emsoft=(float(xpc), float(ypc), float(L_um)), detector_shape=ds,
                pixel_size_um=float(det.get("pixel_size", 70.0)),
                tilt_deg=float(det.get("sample_tilt", 70.0)),
                det_tilt_deg=float(det.get("tilt", 0.0)), max_bandwidth=int(max_bandwidth),
            )
            sim = np.asarray(_Img.open(_io.BytesIO(_b64.b64decode(b64))), dtype=np.float32)
            if mask is not None and sim.shape == exp.shape:
                r_val = compute_ncc_scalar_masked(exp, sim, mask)
            else:
                r_val = compute_ncc_scalar(exp, sim)
            eu = list(map(float, np.degrees(
                _R(np.asarray(q)[:4]).to_euler().reshape(-1)[:3])))
            cands.append({"label": lab, "r_score": float(r_val), "thumbnail": b64,
                          "quat": [float(x) for x in np.asarray(q)[:4]], "euler": eu})
        except Exception as e:
            logger.warning("[variants] render failed for %s: %s", lab, e)
    cands.sort(key=lambda d: -(d["r_score"] if d["r_score"] is not None else -1.0))
    return {
        "pixel_row": row, "pixel_col": col, "phase_id": phase_id, "point_group": pg,
        "current_quat": [float(x) for x in q_cur], "candidates": cands,
    }


def _render_sim_for_ncc(sht_path, quat, det, max_bandwidth: int = 128):
    """Render the SHT forward pattern at `quat` and return it as a float32
    array shaped like the detector — for render-NCC checks (refine gate)."""
    import torch as _torch
    import io as _io
    import base64 as _b64
    from PIL import Image as _Img
    from backend.api.services.sht_pattern_renderer import (
        render_pattern_to_png_b64 as _render_sht_b64,
    )
    from backend.spherical_gpu.pipeline.detector import convert_pc_to_emsoft as _conv

    xpc, ypc, L_um = _conv(
        pc=(float(det["pc_x"]), float(det["pc_y"]), float(det["pc_z"])),
        vendor=str(det.get("vendor", "Bruker")),
        pat_width=int(det["pat_width"]), pat_height=int(det["pat_height"]),
        pixel_size=float(det.get("pixel_size", 70.0)), binning=int(det.get("binning", 1)),
    )
    b64 = _render_sht_b64(
        sht_path=sht_path,
        orientation_quat=_torch.tensor(np.asarray(quat, dtype=np.float64)[:4],
                                       dtype=_torch.float64),
        pc_emsoft=(float(xpc), float(ypc), float(L_um)),
        detector_shape=(int(det["pat_height"]), int(det["pat_width"])),
        pixel_size_um=float(det.get("pixel_size", 70.0)),
        tilt_deg=float(det.get("sample_tilt", 70.0)),
        det_tilt_deg=float(det.get("tilt", 0.0)),
        max_bandwidth=int(max_bandwidth),
    )
    return np.asarray(_Img.open(_io.BytesIO(_b64.b64decode(b64))), dtype=np.float32)


def _grain_newton_refine(result, det, sht_path, point_group, coords, quats,
                         *, max_move_deg: float = 2.0, sample_n: int = 8,
                         bandwidth: int = 128):
    """Guarded per-pixel Newton polish of the snapped grain orientations.

    PHYSICS CAUTION: for the z_rot==2 phases this tool exists for, the SHT
    SO(3) cc surface is exactly what is unreliable (2026-06-27 root cause), so
    refinement is accept-only-if-provably-safe:

    - per pixel: accepted only if Newton CONVERGED, moved <= `max_move_deg`
      (stays inside the ~2 deg render-NCC basin) AND the cc value improved;
    - globally: a render-NCC spot check on up to `sample_n` accepted pixels
      (snapped vs refined, through the SHT forward renderer vs the
      experimental pattern). If the median does NOT improve, ALL refinements
      are DISCARDED and the summary says so — the snap result stands.

    Returns ``(refined_by_flat | None, summary dict)`` — `refined_by_flat`
    maps flat grid index -> (4,) quaternion for ACCEPTED pixels only. Never
    raises: any failure returns ``(None, {"status": "error", ...})``.
    """
    import torch
    from orix.quaternion import Rotation as _R
    from tools.pattern_comparison import (
        get_experimental_pattern, compute_ncc_scalar,
    )
    from backend.spherical_gpu._math.sht_newton import (
        newton_refine, cc_at_rotation, _zxz_to_zyz,
    )
    try:
        n_cols = int(result.original_shape[1])
        pairs = []                      # (flat_idx, exp_pattern, quat)
        for (r, c) in coords:
            fi = r * n_cols + c
            if fi not in quats:
                continue
            exp = get_experimental_pattern(result, r, c)
            if exp is None:
                continue
            pairs.append((fi, np.asarray(exp, dtype=np.float32), quats[fi]))
        if not pairs:
            return None, {"status": "skipped", "reason": "no experimental patterns"}

        # Cached single-phase backend (same key shape as the Phase Test /
        # forward-sim preview so geometries share the Wigner/Legendre tables).
        def _sht_sig(p):
            try:
                return (p, Path(p).stat().st_mtime_ns)
            except OSError:
                return (p, 0)
        sample_tilt = float(det.get("sample_tilt", 70.0))
        det_tilt = float(det.get("tilt", 0.0))
        pixel_size = float(det.get("pixel_size", 70.0))
        cache_key = ((_sht_sig(sht_path),), int(bandwidth), sample_tilt,
                     det_tilt, pixel_size)
        backend = _get_phase_compare_backend(cache_key)
        if backend is None:
            # Reclaim VRAM before allocating a fresh backend. The main indexing
            # run + variant renders often leave the GPU near-full (reserved but
            # unallocated); without this the refine backend build OOMs and the
            # whole polish is skipped. empty_cache() returns the reserved pool
            # to the driver so the ~tens-of-MiB refine build fits.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            from backend.spherical_gpu.backend import (
                BackendConfig, PhaseConfig, SphericalGPUBackend,
            )
            cfg = PhaseConfig(
                sht_file=sht_path, bandwidth=int(bandwidth), normed=True,
                refine=True, circmask=0, gausbckg=True, nregions=10,
                sample_tilt_deg=sample_tilt,
            )
            backend = SphericalGPUBackend(BackendConfig(phases=[cfg]))
            _put_phase_compare_backend(cache_key, backend)
        backend._ensure_built(det)
        indexer = backend._indexers[0]
        L = indexer.bandwidth

        pats = np.stack([p for (_fi, p, _q) in pairs]).astype(np.float32)
        pats_t = torch.as_tensor(pats, device=indexer.device)
        prep = indexer._run_preprocessing(pats_t)
        gln = indexer._direct_sht_coefs(prep).to(torch.complex128)
        flm = indexer._master_coefs.to(torch.complex128)
        if flm.dim() == 3:
            flm = flm.squeeze(0)

        seed_q = np.stack([q for (_fi, _p, q) in pairs])
        eu_seed = torch.as_tensor(
            np.asarray(_R(seed_q).to_euler(), dtype=np.float64),
            device=indexer.device)

        refined_by_flat: dict[int, np.ndarray] = {}
        moves, n_guard = [], 0
        for i, (fi, _p, q0) in enumerate(pairs):
            cc0 = float(cc_at_rotation(flm, gln[i], _zxz_to_zyz(eu_seed[i]), L))
            eu_i, cc_i, converged = newton_refine(flm, gln[i], eu_seed[i], L)
            move = None
            if converged and float(cc_i) > cc0:
                q_i = np.asarray(
                    _R.from_euler(eu_i.cpu().numpy().reshape(1, 3)).data
                ).reshape(4)
                dot = abs(float(np.dot(q_i, q0)))
                move = float(np.degrees(2.0 * np.arccos(min(dot, 1.0))))
            if move is not None and move <= max_move_deg:
                refined_by_flat[fi] = q_i
                moves.append(move)
            else:
                n_guard += 1
        if not refined_by_flat:
            return None, {"status": "rejected", "reason": "guards",
                          "n_refined": 0, "n_guard_rejected": n_guard}

        # Render-NCC spot check — the honest gate. Median must improve.
        acc = sorted(refined_by_flat.keys())
        step = max(1, len(acc) // int(sample_n))
        sample = acc[::step][: int(sample_n)]
        exp_by_flat = {fi: p for (fi, p, _q) in pairs}
        deltas = []
        for fi in sample:
            exp = exp_by_flat[fi]
            n_before = compute_ncc_scalar(
                exp, _render_sim_for_ncc(sht_path, quats[fi], det, bandwidth))
            n_after = compute_ncc_scalar(
                exp, _render_sim_for_ncc(sht_path, refined_by_flat[fi], det, bandwidth))
            deltas.append(float(n_after) - float(n_before))
        median_delta = float(np.median(deltas)) if deltas else 0.0
        if median_delta <= 0.0:
            return None, {
                "status": "rejected", "reason": "render_ncc_gate",
                "n_refined": 0, "n_guard_rejected": n_guard,
                "median_render_ncc_delta": median_delta,
                "n_sampled": len(deltas),
            }
        return refined_by_flat, {
            "status": "applied",
            "n_refined": len(refined_by_flat),
            "n_guard_rejected": n_guard,
            "median_render_ncc_delta": median_delta,
            "mean_move_deg": float(np.mean(moves)) if moves else 0.0,
            "n_sampled": len(deltas),
        }
    except Exception as e:  # noqa: BLE001 — refine must never break the apply
        logger.warning("[grain-flip] refine failed (snap results kept): %s",
                       e, exc_info=True)
        msg = str(e)
        if "out of memory" in msg.lower() or "CUDA out of memory" in msg:
            # GPU is full from the indexing run / variant renders. Try once
            # more to reclaim it so the NEXT apply can refine, and give the
            # user an actionable message instead of the raw torch dump.
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return None, {"status": "skipped", "reason": (
                "GPU out of memory — click 'Release GPU' on the Indexing page, "
                "then re-apply (or uncheck 'Refine'). The variant flip itself "
                "was applied.")}
        return None, {"status": "error", "reason": msg[:200]}


MAX_REFINE_GRAIN_PX = 1500


def _free_interactive_gpu_caches() -> None:
    """Free the VRAM held by the interactive Pattern-Match / variant-render
    caches so a following allocation (the refine backend) fits, WITHOUT the user
    having to press 'Release GPU'.

    The SHT renderer's per-phase LambertGrid (used for the variant thumbnails)
    is several GB and is the dominant holder; clearing it + empty_cache()
    reclaims the room the refine build needs. This is self-healing: the only
    cost is that the next variant render / pattern-match click rebuilds its grid
    once (~a few seconds). The already-displayed thumbnails are unaffected (they
    are decoded client-side). Best-effort — never raises."""
    try:
        from backend.api.services.sht_pattern_renderer import release_gpu_caches
        release_gpu_caches()
    except Exception:
        logger.debug("refine pre-free: renderer cache clear failed", exc_info=True)
    try:
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.debug("refine pre-free: empty_cache failed", exc_info=True)


@router.post("/pattern-match/apply-to-grain")
async def apply_variant_to_grain(req: GrainApplyRequest):
    """Propagate the chosen orientation correction from the clicked pixel to its
    whole grain (v2). Every grain pixel is snapped to ITS OWN best pseudo-variant
    (gradient-tracking flood fill, `pseudosym.grain_snap_floodfill`), so one apply
    also fixes grains scattered across DIFFERENT wrong variants and intra-grain
    distortion is preserved. Optionally runs the guarded Newton refine
    (`req.refine`). Stores a one-level undo in the result metadata. Modifies the
    stored CrystalMap."""
    result = _get_result(req.result_id)
    if result is None:
        raise HTTPException(status_code=400, detail="No indexing result available")
    xmap, px_idx, phase_id, sht_path, det, pg = _spherical_pixel_ctx(result, req.row, req.col)
    from backend.spherical_gpu.pseudosym import grain_snap_floodfill
    from orix.quaternion import Rotation as _R

    n_rows, n_cols = result.original_shape
    flat_mask = result.selection_mask.ravel()
    n_xmap = xmap.rotations.size
    qdata = np.asarray(xmap.rotations.data).reshape(-1, 4).astype(np.float64)
    pid = np.asarray(xmap.phase_id).reshape(-1)

    # Build full-grid orientation + phase arrays (NaN / -1 for unindexed).
    full_q = np.full((n_rows * n_cols, 4), np.nan)
    phase_full = np.full(n_rows * n_cols, -1)
    if n_xmap == n_rows * n_cols:
        full_q = qdata.copy()
        phase_full = pid.copy()
    else:
        idxs = np.where(flat_mask)[0]
        full_q[idxs] = qdata
        phase_full[idxs] = pid

    q_click = full_q[req.row * n_cols + req.col]
    if np.isnan(q_click[0]):
        raise HTTPException(status_code=400, detail="Clicked pixel was not indexed")
    q_target = np.asarray(req.quat, dtype=np.float64).reshape(-1)[:4]
    q_target = q_target / (np.linalg.norm(q_target) + 1e-12)

    grain, new_q_map, snap_stats = grain_snap_floodfill(
        full_q, phase_full, n_rows, n_cols, (req.row, req.col), phase_id, pg,
        q_target, threshold_deg=float(req.threshold_deg),
        max_total_deg=float(req.max_total_deg))

    # Optional guarded Newton polish (fail-safe: snap results kept on any issue).
    refine_summary = None
    if req.refine:
        if len(grain) > MAX_REFINE_GRAIN_PX:
            refine_summary = {"status": "skipped",
                              "reason": f"grain > {MAX_REFINE_GRAIN_PX} px"}
        else:
            # Self-heal the OOM: free the interactive variant-render VRAM right
            # before the refine allocates, so the user never has to press
            # 'Release GPU' first. Cost is one grid rebuild on the next render.
            _free_interactive_gpu_caches()
            refined, refine_summary = _grain_newton_refine(
                result, det, sht_path, pg, grain, new_q_map)
            if refined:
                new_q_map.update(refined)

    # Precompute flat -> xmap index once (the old per-pixel flat_mask[:fi].sum()
    # was O(N) per lookup — quadratic on big ROI grains).
    if n_xmap == n_rows * n_cols:
        xmap_index_of = None                      # identity mapping
    else:
        xmap_index_of = np.full(n_rows * n_cols, -1, dtype=np.int64)
        xmap_index_of[np.where(flat_mask)[0]] = np.arange(n_xmap)

    new_q = qdata.copy()
    undo_idx, undo_old = [], []
    changed = 0
    for (r, c) in grain:
        fi = r * n_cols + c
        xi = fi if xmap_index_of is None else int(xmap_index_of[fi])
        if xi < 0 or fi not in new_q_map:
            continue
        undo_idx.append(int(xi))
        undo_old.append([float(v) for v in qdata[xi]])
        new_q[xi] = new_q_map[fi]
        changed += 1
    xmap._rotations = _R(new_q)   # write back (same pattern as the frame-correction step)

    # One-level undo, stored on the result (in-memory registry — survives until
    # re-index / backend restart, same lifetime as the correction itself).
    md = getattr(result, "metadata", None)
    if isinstance(md, dict):
        md["grain_flip_undo"] = {"xmap_indices": undo_idx, "old_quats": undo_old}

    return {"n_changed": changed, "grain_size": len(grain),
            "threshold_deg": float(req.threshold_deg),
            "max_total_deg": float(req.max_total_deg),
            "snap": snap_stats, "refine": refine_summary,
            "undo_available": isinstance(md, dict) and bool(undo_idx)}


class UnifyVariantsRequest(BaseModel):
    result_id: str | None = None
    threshold_deg: float = 5.0


@router.post("/pseudosym/unify")
async def unify_pseudosym_variants(req: UnifyVariantsRequest):
    """Map-wide pseudo-symmetry variant unification on an EXISTING spherical
    result (post-processing twin of the automatic pipeline stage): per phase,
    segment grains modulo the supergroup, unify variant speckle per grain by
    aggregated render-NCC (coherent twin domains only flip on a clear margin),
    rescue tiny orphans. Stores one-level undo (same slot as the grain flip,
    so the existing Undo button works). Runs for EVERY phase with ≥2 variant
    classes — verification semantics keep correctly-indexed phases safe."""
    result = _get_result(req.result_id)
    if result is None:
        raise HTTPException(status_code=400, detail="No indexing result available")
    md = getattr(result, "metadata", None) or {}
    if md.get("indexing_method") != "spherical":
        raise HTTPException(status_code=400,
                            detail="Variant unification is only available for spherical results")
    sht_map = md.get("sht_paths_by_phase") or {}
    det = md.get("detector_geometry")
    if not sht_map or not det:
        raise HTTPException(status_code=400,
                            detail="Result missing SHT/detector metadata; re-run indexing")

    from backend.spherical_gpu.pipeline.variant_unification import (
        build_render_score_fn, unify_map,
    )
    from tools.pattern_comparison import get_experimental_pattern
    from orix.quaternion import Rotation as _R

    xmap = result.xmap
    n_rows, n_cols = result.original_shape
    flat_mask = np.asarray(result.selection_mask, dtype=bool).ravel()
    n_xmap = xmap.rotations.size
    qdata = np.asarray(xmap.rotations.data).reshape(-1, 4).astype(np.float64)
    pid = np.asarray(xmap.phase_id).reshape(-1)

    if n_xmap == n_rows * n_cols:
        flat_of_row = np.arange(n_xmap)
    else:
        flat_of_row = np.flatnonzero(flat_mask)
        if flat_of_row.size != n_xmap:
            raise HTTPException(status_code=409,
                                detail="Selection mask does not match the result size")
    full_q = np.full((n_rows * n_cols, 4), np.nan)
    full_q[flat_of_row] = qdata
    phase_full = np.full(n_rows * n_cols, -1, dtype=np.int64)
    phase_full[flat_of_row] = pid

    def _get_pattern(flat):
        r, c = divmod(int(flat), n_cols)
        p = get_experimental_pattern(result, r, c)
        return None if p is None else np.asarray(p, dtype=np.float32)

    def _work():
        reports = {}
        cur = full_q
        changed = False
        for phase_id_val in sorted({int(x) for x in np.unique(pid)}):
            try:
                pg = xmap.phases[phase_id_val].point_group.name
            except Exception:
                continue
            sht = sht_map.get(phase_id_val) or sht_map.get(str(phase_id_val))
            if not sht:
                continue
            try:
                score_fn = build_render_score_fn(str(sht), det, _get_pattern)
                new_full, rep = unify_map(cur, phase_full, n_rows, n_cols,
                                          phase_id_val, pg, score_fn)
            except Exception as e:  # fail safe per phase
                logger.warning("[unify] phase %s failed: %s", phase_id_val, e,
                               exc_info=True)
                reports[phase_id_val] = {"error": str(e)[:200]}
                continue
            reports[phase_id_val] = rep
            if new_full is not None:
                cur = new_full
                changed = True
        return cur, changed, reports

    new_full_q, changed, reports = await asyncio.to_thread(_work)

    n_changed = 0
    if changed:
        new_rows = new_full_q[flat_of_row]
        moved = ~np.all(np.isclose(new_rows, qdata, atol=1e-12), axis=1)
        idxs = np.flatnonzero(moved)
        if idxs.size:
            if isinstance(md, dict):
                md["grain_flip_undo"] = {
                    "xmap_indices": [int(i) for i in idxs],
                    "old_quats": [[float(v) for v in qdata[i]] for i in idxs],
                }
            xmap._rotations = _R(new_rows)
            n_changed = int(idxs.size)

    summary = {
        "n_changed": n_changed,
        "n_grains": sum(int(r.get("n_grains", 0)) for r in reports.values()
                        if isinstance(r, dict)),
        "n_flipped_units": sum(int(r.get("n_flipped_units", 0)) for r in reports.values()
                               if isinstance(r, dict)),
        "n_ambiguous": sum(int(r.get("n_ambiguous", 0)) for r in reports.values()
                           if isinstance(r, dict)),
        "n_rescued": sum(int(r.get("n_rescued", 0)) for r in reports.values()
                         if isinstance(r, dict)),
        "ambiguous_grains": [
            {"phase_id": p, "centroid": g["centroid"], "margin": g["margin"]}
            for p, r in reports.items() if isinstance(r, dict)
            for g in r.get("grains", []) if "ambiguous" in g.get("decision", "")
        ][:50],
        "undo_available": n_changed > 0,
    }
    if isinstance(md, dict):
        md["variant_unification"] = reports
    return summary


class GrainUndoRequest(BaseModel):
    result_id: str | None = None


@router.post("/pattern-match/undo-grain")
async def undo_grain_flip(req: GrainUndoRequest):
    """Restore the orientations changed by the LAST apply-to-grain (one level)."""
    result = _get_result(req.result_id)
    if result is None:
        raise HTTPException(status_code=400, detail="No indexing result available")
    md = getattr(result, "metadata", None)
    undo = (md or {}).get("grain_flip_undo") if isinstance(md, dict) else None
    if not undo or not undo.get("xmap_indices"):
        raise HTTPException(status_code=400, detail="Nothing to undo")
    from orix.quaternion import Rotation as _R
    xmap = result.xmap
    qdata = np.asarray(xmap.rotations.data).reshape(-1, 4).astype(np.float64)
    idxs = np.asarray(undo["xmap_indices"], dtype=np.int64)
    olds = np.asarray(undo["old_quats"], dtype=np.float64).reshape(-1, 4)
    if idxs.size != olds.shape[0] or idxs.size == 0 or idxs.max() >= qdata.shape[0]:
        raise HTTPException(status_code=409, detail="Undo data no longer matches the result")
    qdata[idxs] = olds
    xmap._rotations = _R(qdata)
    md.pop("grain_flip_undo", None)
    return {"n_restored": int(idxs.size)}


# Cache of (sht_paths tuple, bandwidth) -> SphericalGPUBackend. First call
# pays the ~6s wigner-d / SHT setup; subsequent per-pixel re-index calls
# are ~30-50 ms per phase. Keyed by the SORTED tuple so phase ordering
# doesn't fragment the cache.
# LRU-capped: each SphericalGPUBackend holds GPU tensors + Wigner tables, so an
# unbounded cache leaks VRAM as the user pattern-match-compares across phases.
MAX_PHASE_COMPARE_BACKENDS = 8
_phase_compare_backends: "OrderedDict" = OrderedDict()


def _free_gpu_backend(backend) -> None:
    """Best-effort release of a SphericalGPUBackend's GPU memory.

    Drops the backend's per-phase indexers + geometry (the GPU tensors) so the
    references die and a following ``empty_cache()`` can return the VRAM to the
    driver. ``invalidate()`` is the real teardown on ``SphericalGPUBackend``
    (it sets ``_indexers=[]`` / ``_geom=None``); ``close()`` is kept only as a
    fallback for any backend that grows one later. The old code called the
    non-existent ``close()`` and nothing else, so eviction was a silent no-op
    and the per-phase master patterns stayed pinned on the GPU.
    """
    for meth in ("invalidate", "close"):
        fn = getattr(backend, meth, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                logger.debug("phase-compare backend %s() failed", meth, exc_info=True)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _get_phase_compare_backend(key):
    """LRU-get: touch the entry so it's marked most-recently-used."""
    b = _phase_compare_backends.get(key)
    if b is not None:
        _phase_compare_backends.move_to_end(key)
    return b


def _put_phase_compare_backend(key, backend):
    """Insert + evict oldest beyond the cap, freeing evicted GPU backends."""
    _phase_compare_backends[key] = backend
    _phase_compare_backends.move_to_end(key)
    while len(_phase_compare_backends) > MAX_PHASE_COMPARE_BACKENDS:
        _old_key, _old_backend = _phase_compare_backends.popitem(last=False)
        _free_gpu_backend(_old_backend)


def _evict_phase_compare_backends() -> int:
    """Free + drop ALL cached phase-test / pattern-compare backends.

    Each entry holds GPU tensors via *live* references, so a bare
    ``torch.cuda.empty_cache()`` cannot reclaim their VRAM — the cache must be
    cleared first. This is what makes the Release-GPU button actually work
    after a Single-Pixel Phase Test. Returns the number of backends evicted.
    """
    n = len(_phase_compare_backends)
    while _phase_compare_backends:
        _key, _backend = _phase_compare_backends.popitem(last=False)
        _free_gpu_backend(_backend)
    return n


def spherical_orientation_for_pattern(
    *,
    exp_pattern,
    sht_path: str,
    det_params: dict,
    max_bandwidth: int | None = None,
) -> list[float] | None:
    """Best-fit orientation of ONE experimental pattern against ONE phase SHT
    using the SHT-spherical (Tier1) indexer — the SAME orientation source the
    Single-Pixel Phase Test uses.

    Returns Euler angles in DEGREES (Bunge ZXZ), or ``None`` on any failure /
    shape mismatch (caller falls back to its own orientation source).

    Why this exists: the PC-Refinement Forward-Sim preview defaulted to Hough
    for its reference orientation, but Hough (PyEBSDIndex Radon band detection)
    structurally fails on low-resolution patterns (~128×156) and falls back to
    identity — so the preview rendered the phase at the WRONG orientation while
    the Phase Test rendered the real spherical one, making the same phase look
    completely different in the two tools. Routing the preview through this
    helper makes both tools use ONE orientation source. Shares the cached
    ``SphericalGPUBackend`` pool so a repeated geometry reuses the Wigner/
    Legendre tables instead of rebuilding them.
    """
    import torch
    import numpy as _np
    from backend.spherical_gpu.backend import (
        BackendConfig, PhaseConfig, SphericalGPUBackend,
    )
    try:
        pat_h = int(det_params["pat_height"])
        pat_w = int(det_params["pat_width"])
        pat = _np.asarray(exp_pattern, dtype=_np.float32)
        if pat.shape != (pat_h, pat_w):
            logger.warning(
                "spherical_orientation_for_pattern: pattern shape %s != detector "
                "(%d,%d) — skipping (caller falls back)", pat.shape, pat_h, pat_w)
            return None
        bw = int(max_bandwidth) if max_bandwidth else 128
        sample_tilt = float(det_params.get("sample_tilt", 70.0))
        det_tilt = float(det_params.get("tilt", 0.0))
        pixel_size = float(det_params.get("pixel_size", 70.0))

        def _sht_sig(p):
            try:
                return (p, Path(p).stat().st_mtime_ns)
            except OSError:
                return (p, 0)

        # Same cache-key shape as the Phase Test, so a single-phase preview
        # backend and the Phase Test's multi-phase one never collide.
        cache_key = ((_sht_sig(sht_path),), bw, sample_tilt, det_tilt, pixel_size)
        backend = _get_phase_compare_backend(cache_key)
        if backend is None:
            cfg = PhaseConfig(
                sht_file=sht_path, bandwidth=bw, normed=True, refine=True,
                circmask=0, gausbckg=True, nregions=10,
                sample_tilt_deg=sample_tilt,
            )
            backend = SphericalGPUBackend(BackendConfig(phases=[cfg]))
            _put_phase_compare_backend(cache_key, backend)
        backend._ensure_built(det_params)
        pat_t = torch.from_numpy(pat[_np.newaxis]).to(backend.device)
        eulers_i, _scores = backend._indexers[0]._index_batch(pat_t)
        eu_rad = eulers_i[0].detach().cpu().numpy().astype(float)  # (3,) radians
        return [float(_np.rad2deg(x)) for x in eu_rad]
    except Exception as e:  # noqa: BLE001 — fail-soft, caller falls back
        logger.warning("spherical_orientation_for_pattern failed: %s", e)
        return None


def _hough_euler_for_phase(exp_pattern, cif_path, det_params):
    """Hough orientation (radians, Bunge ZXZ) for ONE pattern + ONE phase's CIF.

    The SHT-spherical (Tier1) correlation lands on a high-symmetry pole for
    low-symmetry point groups (m-3, mmm — the intermetallic phases: z_rot=2),
    so for those phases we take the orientation from Hough indexing instead,
    which is reliable on this app's data (root-caused 2026-06-27). The SHT
    forward render at THIS orientation then provides the phase-discrimination
    NCC. Returns a (3,) ndarray in radians, or None on any failure.
    """
    try:
        import numpy as _np
        from orix.crystal_map import Phase, PhaseList
        from ebsd_utils import sanitize_cif, prepare_reflectors, create_indexer
        from kikuchipy.detectors import EBSDDetector
        from kikuchipy.signals import EBSD
        phase = Phase.from_cif(sanitize_cif(str(cif_path)))
        try:
            phase.name = Path(cif_path).stem
        except Exception:
            pass
        pl = PhaseList(phase)
        H = int(det_params["pat_height"]); W = int(det_params["pat_width"])
        det = EBSDDetector(
            shape=(H, W),
            sample_tilt=float(det_params.get("sample_tilt", 70.0)),
            tilt=float(det_params.get("tilt", 0.0)),
            pc=(float(det_params["pc_x"]), float(det_params["pc_y"]), float(det_params["pc_z"])),
            convention="bruker",
            binning=int(det_params.get("binning", 1)),
        )
        refl = prepare_reflectors(pl)
        indexer = create_indexer(det, pl, refl, nBands=12)
        ebsd = EBSD(_np.asarray(exp_pattern, dtype=_np.float32)[None, None], detector=det)
        xmap, _idx, _bands = ebsd.hough_indexing(
            pl, indexer, return_index_data=True, return_band_data=True, verbose=0)
        return _np.asarray(xmap.rotations.to_euler()[0], dtype=float)  # radians
    except Exception as e:  # noqa: BLE001 — fail-soft, caller keeps spherical
        logger.warning("hough orientation for phase failed (%s): %s", cif_path, e)
        return None


def _compute_phase_compare_results(
    *,
    exp_pattern,
    detector_geometry: dict,
    sht_paths_by_phase: dict,
    phase_list=None,
    phase_names: dict | None = None,
    cif_paths_by_phase: dict | None = None,
    aperture_mask=None,
    max_bandwidth: int | None = None,
    progress_cb=None,
    should_cancel=None,
    collect_arrays: dict | None = None,
) -> list[dict]:
    """Re-index a single experimental pattern against every phase SHT
    and render the per-phase simulated pattern.

    Returns a list of dicts sorted by R-score descending:
        [{"rank": 1, "phase_id": 4, "phase_name": "Al4FeSi",
          "r_score": 0.42, "ncc_score": 0.71,
          "euler_deg": [..., ..., ...],
          "simulated_b64": "<png>", "ncc_diff_b64": "<png>"}, ...]

    The first call for a given ``(sht_paths, bandwidth)`` builds and
    caches a ``SphericalGPUBackend``; subsequent calls reuse it so the
    per-click cost is dominated by the per-pixel SHT correlation
    (~30-50 ms × N phases ≈ 0.5 s for 12 phases).
    """
    import torch
    import numpy as np
    from tools.pattern_comparison import (
        compute_ncc_scalar, compute_ncc_scalar_masked,
    )
    from backend.api.services.sht_pattern_renderer import (
        render_pattern_to_png_b64,
    )
    from backend.spherical_gpu.backend import (
        BackendConfig, PhaseConfig, SphericalGPUBackend,
    )
    from backend.spherical_gpu.pipeline.detector import (
        convert_pc_to_emsoft,
    )
    from orix.quaternion import Rotation as _Rotation

    det = detector_geometry
    pat_h = int(det["pat_height"])
    pat_w = int(det["pat_width"])
    pixel_size = float(det.get("pixel_size", 70.0))
    binning = int(det.get("binning", 1))
    vendor = str(det.get("vendor", "Bruker"))
    sample_tilt = float(det.get("sample_tilt", 70.0))
    det_tilt = float(det.get("tilt", 0.0))
    bw = int(max_bandwidth) if max_bandwidth is not None else 128

    # Normalise pattern to (1, H, W) float32 in the indexer's expected layout
    pat = np.asarray(exp_pattern, dtype=np.float32)
    if pat.shape != (pat_h, pat_w):
        # Mismatch with stored detector geometry; bail out — the result
        # would be meaningless and probably worse than not showing it.
        return []
    pat_batch = pat[np.newaxis]  # (1, H, W)

    # Re-mask support (additive, opt-in): when the caller passes a dict, stash
    # the UNMASKED measured + per-phase simulated float32 arrays so a later
    # /remask call can recompute masked R + diffs WITHOUT re-rendering. Default
    # collect_arrays=None → these assignments never run (existing behaviour).
    if collect_arrays is not None:
        collect_arrays["measured"] = pat
        collect_arrays.setdefault("sims", {})

    # Build / reuse the cached backend for THIS combination of phase SHTs.
    # Key on the SORTED tuple so the cache survives phase-id permutations.
    phase_items = sorted(
        ((int(pid) if str(pid).lstrip("-").isdigit() else pid, p)
         for pid, p in sht_paths_by_phase.items()),
        key=lambda kv: str(kv[0]),
    )
    # Key on (path, mtime) per .sht so a RE-SIMULATED master (same path, newer
    # mtime) forces a fresh backend instead of serving a stale cached one.
    def _sht_sig(p):
        try:
            return (p, Path(p).stat().st_mtime_ns)
        except OSError:
            return (p, 0)
    cache_key = (tuple(_sht_sig(p) for _, p in phase_items), bw, sample_tilt, det_tilt, pixel_size)
    backend = _get_phase_compare_backend(cache_key)
    if backend is None:
        phases_cfg = [
            PhaseConfig(
                sht_file=p,
                bandwidth=bw,
                normed=True,
                refine=True,
                circmask=0,
                gausbckg=True,
                nregions=10,
                sample_tilt_deg=sample_tilt,
            )
            for _, p in phase_items
        ]
        backend = SphericalGPUBackend(BackendConfig(phases=phases_cfg))
        _put_phase_compare_backend(cache_key, backend)

    det_params = {
        "pc_x": float(det["pc_x"]), "pc_y": float(det["pc_y"]), "pc_z": float(det["pc_z"]),
        "pat_width": pat_w, "pat_height": pat_h,
        "n_cols": 1, "n_rows": 1,
        "pixel_size": pixel_size, "tilt": det_tilt,
        "sample_tilt": sample_tilt, "binning": binning,
        "step_x": 1.0, "step_y": 1.0,
        "vendor": vendor,
    }

    # Convert PC for the renderer. xpc/ypc are pure pixel offsets so the
    # value is independent of pixel_size; we still compute L_um from it.
    xpc, ypc, L_um = convert_pc_to_emsoft(
        pc=(det_params["pc_x"], det_params["pc_y"], det_params["pc_z"]),
        vendor=vendor,
        pat_width=pat_w, pat_height=pat_h,
        pixel_size=pixel_size, binning=binning,
    )

    # Index the single pattern against each phase ONE AT A TIME, but reuse
    # the SINGLE multi-phase backend cached above: it builds the phase-
    # independent SHT tables (Wigner-d / GL-Legendre, geometry + sample-tilt)
    # ONCE and shares them across phases via SharedSphericalTables, giving
    # one table-build instead of N. Each phase still gets its own
    # Tier1Indexer (master-pattern-specific data stays per-phase), so the
    # per-phase (euler, score) we read here is the actual best-orientation-
    # for-THIS-phase — NOT the consensus winner that ``index_array`` would
    # argmax-select.
    #
    # Bit-identity: a solo backend's ``index_array`` routes a single-pattern
    # batch through ``_index_array_via_indexer`` →
    # ``indexer._index_batch(from_numpy(pat).to(device))``. We replicate that
    # exact call on each per-phase indexer, so the numbers match the old
    # per-phase single-phase-backend path by construction.
    backend._ensure_built(det_params)   # builds backend._indexers (shared tables)
    pat_t = torch.from_numpy(pat_batch).to(backend.device)   # (1, H, W) float32
    if len(backend._indexers) != len(phase_items):
        raise RuntimeError(
            f"phase-compare: {len(backend._indexers)} indexers != "
            f"{len(phase_items)} phases — ordering/cardinality assumption broken"
        )

    out_rows = []
    for (pid_raw, sht_path), indexer in zip(phase_items, backend._indexers):
        if should_cancel is not None and should_cancel():
            break
        try:
            pid = int(pid_raw)
        except (TypeError, ValueError):
            pid = -1
        try:
            eulers_i, scores_i = indexer._index_batch(pat_t)
        except Exception as e:
            logger.warning("[pattern-match compare] phase %d index failed: %s", pid, e)
            continue

        eu = eulers_i[0].detach().cpu().numpy().astype(float)  # (3,) radians
        ncc = float(scores_i.reshape(-1)[0].detach().cpu())

        # Low-symmetry intermetallics (m-3 / mmm, z_rot=2): the SHT-spherical
        # correlation lands on a high-symmetry pole (root-caused 2026-06-27 —
        # the CC volume is flat for these point groups). The orientation from
        # Hough is reliable for them, so override eu with the Hough solution and
        # render the SHT at THAT orientation. The forward-NCC (r_score, the sort
        # key) is then meaningful. High-symmetry phases keep the spherical
        # orientation (it works for z_rot=4 / 2/m). Fail-soft: any Hough error
        # keeps the spherical eu.
        orientation_source = "spherical"
        try:
            _zrot = int(getattr(getattr(indexer, "master", None), "z_rot", 4) or 4)
        except Exception:
            _zrot = 4
        _cif_p = None
        if cif_paths_by_phase:
            _cif_p = cif_paths_by_phase.get(pid_raw, cif_paths_by_phase.get(pid))
        if _zrot == 2 and _cif_p:
            eu_h = _hough_euler_for_phase(pat, _cif_p, det_params)
            if eu_h is not None:
                eu = eu_h
                orientation_source = "hough"

        # Render simulated pattern via the SHT renderer using THIS phase's
        # winning Euler. det_tilt is passed so the renderer matches the
        # indexer's geometry (fix bundled with the d5a8b8c det_tilt commit).
        try:
            rot_obj = _Rotation.from_euler(eu.reshape(1, 3))
            q_arr = np.asarray(rot_obj.data).reshape(-1)[:4]
            q_t = torch.tensor(q_arr, dtype=torch.float64)
            sim_b64 = render_pattern_to_png_b64(
                sht_path=str(sht_path),
                orientation_quat=q_t,
                pc_emsoft=(float(xpc), float(ypc), float(L_um)),
                detector_shape=(pat_h, pat_w),
                pixel_size_um=pixel_size,
                tilt_deg=sample_tilt,
                det_tilt_deg=det_tilt,
                max_bandwidth=bw,
            )
        except Exception as e:
            logger.warning("[pattern-match compare] phase %d render failed: %s", pid, e)
            continue

        # Decode the simulated PNG, apply the same aperture as the parent
        # request, then NCC vs experimental.
        try:
            from PIL import Image
            import io, base64
            from tools.pattern_comparison import ncc_diff_to_png_b64
            sim_arr = np.asarray(
                Image.open(io.BytesIO(base64.b64decode(sim_b64))),
                dtype=np.float32,
            )
            # Capture the UNMASKED simulated array for re-masking (before any
            # apply_circular_mask for display). Only phases that rendered+decoded
            # successfully reach here — exactly the ones that produce a row.
            if collect_arrays is not None and sim_arr.shape == pat.shape:
                collect_arrays["sims"][pid] = sim_arr
            _use_mask = aperture_mask is not None and sim_arr.shape == pat.shape
            if _use_mask:
                r_val = compute_ncc_scalar_masked(pat, sim_arr, aperture_mask)
            else:
                r_val = compute_ncc_scalar(pat, sim_arr)
            diff_b64 = ncc_diff_to_png_b64(pat, sim_arr, aperture_mask if _use_mask else None)
        except Exception as e:
            logger.warning("[pattern-match compare] phase %d NCC failed: %s", pid, e)
            r_val = float("nan")
            diff_b64 = None

        # Phase name lookup: explicit names map wins (the library/standalone
        # path has no orix phase_list); else fall back to the xmap phase_list.
        if phase_names is not None and pid in phase_names:
            phase_name = phase_names[pid]
        else:
            try:
                phase_name = phase_list[pid].name
            except Exception:
                phase_name = f"Phase {pid}"

        # Mask the DISPLAYED simulated too (corners -> 0) so all panels match
        if aperture_mask is not None and sim_arr.shape == pat.shape:
            from tools.pattern_comparison import apply_circular_mask
            import io as _io, base64 as _b64
            from PIL import Image as _Image
            _sm = apply_circular_mask(sim_arr, aperture_mask)
            _a = ((_sm - _sm.min()) / max(float(np.ptp(_sm)), 1e-7) * 255).astype(np.uint8)
            _buf = _io.BytesIO(); _Image.fromarray(_a).save(_buf, format="PNG")
            sim_b64 = _b64.b64encode(_buf.getvalue()).decode("ascii")

        out_rows.append({
            "phase_id": pid,
            "phase_name": str(phase_name),
            "r_score": float(r_val) if r_val == r_val else None,  # NaN -> None
            "ncc_score": float(ncc),
            "euler_deg": [float(x) for x in np.degrees(eu)],
            "orientation_source": orientation_source,
            "simulated_b64": sim_b64,
            "ncc_diff_b64": diff_b64,
        })
        if progress_cb is not None:
            progress_cb(len(out_rows), len(phase_items), str(phase_name))

    # Sort by R-score (best first); items with None R go last
    def _sort_key(r):
        v = r.get("r_score")
        return -v if (v is not None and v == v) else float("inf")
    out_rows.sort(key=_sort_key)
    for i, r in enumerate(out_rows):
        r["rank"] = i + 1
    return out_rows


def _assemble_phase_test_candidates(rows, id_to_entry, chem_by_key, eff_mode, excluded):
    """Join compare rows back to library entries + chemistry, rank, and build
    the candidate/excluded PhaseTestCandidate lists. Returns (out_candidates, out_excluded)."""
    from backend.api.services.phase_test import combined_rank_score

    _dict_files = _dictionary_library_files()   # for Dictionary-method availability

    def _avail(e):
        """(cif_path, sht_path, master_h5_path) availability for the 'Use for
        indexing' picker — the file this phase has for each method, or None."""
        cif = str(e.cif_path) if getattr(e, "cif_path", None) else None
        sht = str(e.sht_path) if getattr(e, "sht_path", None) else None
        master = _master_h5_for_phase(e, _dict_files)
        return cif, sht, master

    candidates = []
    for r in rows:
        e = id_to_entry.get(r["phase_id"])
        if e is None:
            continue
        chem = chem_by_key.get(e.key)
        _cif, _sht, _master = _avail(e)
        candidates.append((combined_rank_score(r.get("r_score"), chem, eff_mode),
                           PhaseTestCandidate(
            phase_key=e.key, formula=e.formula or e.key,
            display_formula=getattr(e, "display_formula", "") or e.formula or e.key,
            space_group=getattr(e, "space_group", "") or "",
            crystal_system=getattr(e, "crystal_system", "unknown") or "unknown",
            r_score=r.get("r_score"), ncc_score=r.get("ncc_score"),
            chemistry_fit=chem, euler_deg=r.get("euler_deg", []),
            simulated_png=r.get("simulated_b64"),
            ncc_diff_png=r.get("ncc_diff_b64"),
            cif_path=_cif, sht_path=_sht, master_h5_path=_master)))
    candidates.sort(key=lambda t: t[0], reverse=True)
    out_candidates = []
    for i, (_, c) in enumerate(candidates):
        c.rank = i + 1
        out_candidates.append(c)

    out_excluded = []
    for e in excluded:
        _cif, _sht, _master = _avail(e)
        out_excluded.append(PhaseTestCandidate(
            phase_key=e.key, formula=e.formula or e.key,
            display_formula=getattr(e, "display_formula", "") or e.formula or e.key,
            space_group=getattr(e, "space_group", "") or "",
            crystal_system=getattr(e, "crystal_system", "unknown") or "unknown",
            chemistry_fit=chem_by_key.get(e.key), excluded_by_eds=True,
            cif_path=_cif, sht_path=_sht, master_h5_path=_master))

    return out_candidates, out_excluded


@router.post("/single-pixel-phase-test", response_model=SinglePixelPhaseTestResponse)
async def single_pixel_phase_test(req: SinglePixelPhaseTestRequest):
    """Test one pixel against every SHT-library phase; rank by pattern-fit (R)
    and EDS chemistry-fit. Needs a loaded file + a valid PC — no prior
    indexing run required.

    Why SHT/spherical-GPU forward only (no Hough/Dictionary toggle here):
    this tool's whole value is showing the user *two patterns side by side*
    (measured vs simulated) per phase. A simulated pattern only exists if you
    have a forward model — the SHT master pattern projected through the
    detector geometry. Hough indexing detects bands + looks them up; it
    produces an orientation but NO renderable pattern, so it cannot power the
    Measured | Simulated | Difference comparison. The method choice
    (Hough / Dictionary / Spherical, CPU / GPU) belongs to the full-map
    indexing run, not this per-pixel visual phase tester."""
    import io, base64
    import numpy as np
    from PIL import Image
    from tools.pattern_comparison import (
        circular_mask, apply_circular_mask, detect_circular_aperture,
    )
    from backend.api.services.phase_test import eds_prefilter

    signal = _get_active_signal_for_phase_test()
    if signal is None:
        raise HTTPException(status_code=400, detail="No EBSD file loaded — open one first")

    entry, det_params = _get_detector_for_phase_test(pixel_index=req.pixel_index)
    if det_params is None:
        raise HTTPException(status_code=400, detail="No detector/calibration for the active dataset")
    if entry is not None and getattr(entry, "pc_source", None) == "missing":
        raise HTTPException(
            status_code=400,
            detail="Pattern Center is not set for this dataset. Set/refine the PC "
                   "first (EBSD Viewer / PC Refinement) — a simulated pattern with "
                   "a placeholder PC would be misleading.")

    n_cols = int(det_params["n_cols"]); n_rows = int(det_params["n_rows"])
    if req.pixel_index >= n_rows * n_cols:
        raise HTTPException(status_code=400, detail="pixel_index out of range")
    row, col = divmod(req.pixel_index, n_cols)

    try:
        exp = np.asarray(signal.data[row, col], dtype=np.float32)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Could not read pattern at ({row},{col})")

    # Dynamic-background-remove the measured pattern so it correlates band-vs-band
    # against the clean simulated dynamical pattern (and the preview looks clean).
    if req.bg_remove:
        exp = _bg_remove_phase_test(exp)

    # Aperture mask
    ap = req.aperture
    if ap == "auto":
        circular = detect_circular_aperture(exp)
    else:
        circular = (ap == "circular")
    mask = circular_mask(exp.shape, req.aperture_radius) if circular else None
    exp_display = apply_circular_mask(exp, mask) if mask is not None else exp

    # EDS pre-filter
    pixel_at_pct = get_pixel_at_pct_for_phase_test(req.pixel_index)
    entries = _phase_test_library_entries(req.phase_keys)
    tested, excluded, chem_by_key, eff_mode = eds_prefilter(
        entries, pixel_at_pct, req.eds_weighting, req.eds_filter_threshold)

    # Synthetic ids for the compare engine
    id_to_entry = {i: e for i, e in enumerate(tested)}
    sht_paths_by_phase = {i: str(e.sht_path) for i, e in id_to_entry.items()}
    phase_names = {i: (e.display_formula or e.formula or e.key)
                   for i, e in id_to_entry.items()}
    # CIF per phase → lets the compare engine take the Hough orientation for
    # low-symmetry intermetallics (where the SHT-spherical CC mis-indexes).
    cif_paths_by_phase = {i: str(e.cif_path) for i, e in id_to_entry.items()
                          if getattr(e, "cif_path", None)}

    rows = []
    if sht_paths_by_phase:
        rows = _compute_phase_compare_results(
            exp_pattern=exp, detector_geometry=det_params,
            sht_paths_by_phase=sht_paths_by_phase, phase_names=phase_names,
            cif_paths_by_phase=cif_paths_by_phase,
            aperture_mask=mask, max_bandwidth=req.max_bandwidth)

    # Join compare rows back to library entries + chemistry; build candidates
    out_candidates, out_excluded = _assemble_phase_test_candidates(
        rows, id_to_entry, chem_by_key, eff_mode, excluded)

    def _png(arr):
        a = ((arr - arr.min()) / max(float(np.ptp(arr)), 1e-7) * 255).astype(np.uint8)
        buf = io.BytesIO(); Image.fromarray(a).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    return SinglePixelPhaseTestResponse(
        pixel_index=req.pixel_index, row=row, col=col,
        experimental_png=_png(exp_display), pixel_at_pct=pixel_at_pct,
        eds_weighting_effective=eff_mode, mask_applied=mask is not None,
        bandwidth=req.max_bandwidth, candidates=out_candidates, excluded=out_excluded)


# ---------------------------------------------------------------------------
# Non-blocking background-job variant of the single-pixel phase test.
#
# The sync endpoint above runs N cold ~6 s SphericalGPUBackend builds inline,
# which blocks the event loop and makes the frontend time out at 5 min for a
# large phase list. The job API below kicks the heavy compute onto a worker
# thread (asyncio.to_thread), reports cumulative per-phase progress, and can be
# cancelled mid-run. The sync endpoint + its assembly helper are reused verbatim.
# ---------------------------------------------------------------------------
_phase_test_jobs: "dict[str, dict]" = {}
_PHASE_TEST_JOB_MAX = 24

# Re-mask cache: per completed job, the UNMASKED measured + per-phase simulated
# float32 arrays (from _compute_phase_compare_results' collect_arrays) plus the
# slim metadata needed to rebuild candidates WITHOUT re-rendering/re-indexing.
# Bounded + LRU-evicting so the base64-free numpy arrays (a handful of MB per
# job) can't grow without bound. Keyed by job id.
_phase_test_remask_cache: "OrderedDict[str, dict]" = OrderedDict()
_REMASK_CACHE_MAX = 8


def _store_remask_cache(jid: str, entry: dict) -> None:
    """Insert a re-mask cache entry, refresh LRU order, evict oldest beyond cap."""
    _phase_test_remask_cache[jid] = entry
    _phase_test_remask_cache.move_to_end(jid)
    while len(_phase_test_remask_cache) > _REMASK_CACHE_MAX:
        _phase_test_remask_cache.popitem(last=False)


def _new_phase_test_job() -> str:
    # Evict oldest jobs beyond the cap so the registry (which holds full
    # results incl. base64 PNGs, MBs each) cannot grow without bound.
    # dict preserves insertion order, so the first key is the oldest.
    while len(_phase_test_jobs) >= _PHASE_TEST_JOB_MAX:
        oldest = next(iter(_phase_test_jobs))
        _phase_test_jobs.pop(oldest, None)
    jid = uuid.uuid4().hex[:12]
    _phase_test_jobs[jid] = {
        "status": "running", "done": 0, "total": 0,
        "current": None, "result": None, "error": None, "cancelled": False,
    }
    return jid


async def _run_phase_test_job(jid: str, *, exp, det_params, sht_paths_by_phase,
                              phase_names, mask, max_bandwidth,
                              id_to_entry, chem_by_key, eff_mode, excluded,
                              row, col, pixel_index, exp_display, pixel_at_pct):
    job = _phase_test_jobs[jid]
    # The first run builds the multi-phase spherical backend (~25 s) inside
    # _compute_phase_compare_results BEFORE any per-phase progress tick. Surface
    # a building message via the poll so the dialog isn't frozen at "0/N".
    # The per-phase progress_cb overwrites job["current"] once phases complete.
    job["current"] = "Building phase models — first run is slower…"
    try:
        def _progress(done, total, current=None):
            job["done"] = done
            job["total"] = total
            if current is not None:
                job["current"] = current

        def _should_cancel():
            return job.get("cancelled", False)

        rows = []
        arrays: dict = {}
        # CIF per phase → Hough orientation for low-symmetry intermetallics.
        cif_paths_by_phase = {i: str(e.cif_path) for i, e in id_to_entry.items()
                              if getattr(e, "cif_path", None)}
        if sht_paths_by_phase:
            rows = await asyncio.to_thread(
                _compute_phase_compare_results,
                exp_pattern=exp, detector_geometry=det_params,
                sht_paths_by_phase=sht_paths_by_phase, phase_names=phase_names,
                cif_paths_by_phase=cif_paths_by_phase,
                aperture_mask=mask, max_bandwidth=max_bandwidth,
                progress_cb=_progress, should_cancel=_should_cancel,
                collect_arrays=arrays)
        if job.get("cancelled"):
            job["status"] = "cancelled"
            return
        out_candidates, out_excluded = _assemble_phase_test_candidates(
            rows, id_to_entry, chem_by_key, eff_mode, excluded)
        job["result"] = SinglePixelPhaseTestResponse(
            pixel_index=pixel_index, row=row, col=col,
            experimental_png=exp_display, pixel_at_pct=pixel_at_pct,
            eds_weighting_effective=eff_mode, mask_applied=mask is not None,
            bandwidth=max_bandwidth, candidates=out_candidates, excluded=out_excluded
        ).model_dump()
        job["status"] = "done"

        # Cache the UNMASKED arrays + slim per-phase rows so a later /remask can
        # recompute masked R + diff/sim PNGs WITHOUT re-render/re-index. We keep
        # only what the remask needs: ncc_score/euler/name come from the rows;
        # the actual R + PNGs are recomputed from sims[pid] under the new mask.
        if "measured" in arrays and arrays.get("sims"):
            slim_rows = [
                {"phase_id": r["phase_id"], "phase_name": r["phase_name"],
                 "euler_deg": r.get("euler_deg", []), "ncc_score": r.get("ncc_score")}
                for r in rows
            ]
            _store_remask_cache(jid, {
                "measured": arrays["measured"],
                "sims": arrays["sims"],
                "id_to_entry": id_to_entry,
                "chem_by_key": chem_by_key,
                "eff_mode": eff_mode,
                "excluded": excluded,
                "row": row, "col": col,
                "pixel_index": pixel_index,
                "pixel_at_pct": pixel_at_pct,
                "bandwidth": max_bandwidth,
                "rows": slim_rows,
            })
    except Exception as e:
        logger.exception("[phase-test job %s] failed", jid)
        job["status"] = "error"
        job["error"] = str(e)


@router.post("/single-pixel-phase-test/start")
async def single_pixel_phase_test_start(req: SinglePixelPhaseTestRequest):
    """Kick off a single-pixel phase test as a background job. Returns a
    job_id + the experimental pattern + total phase count immediately; poll
    /progress/{job_id} for per-phase progress and the final result."""
    import io, base64
    import numpy as np
    from PIL import Image
    from tools.pattern_comparison import (circular_mask, apply_circular_mask, detect_circular_aperture)
    from backend.api.services.phase_test import eds_prefilter

    signal = _get_active_signal_for_phase_test()
    if signal is None:
        raise HTTPException(status_code=400, detail="No EBSD file loaded — open one first")
    entry, det_params = _get_detector_for_phase_test(pixel_index=req.pixel_index)
    if det_params is None:
        raise HTTPException(status_code=400, detail="No detector/calibration for the active dataset")
    if entry is not None and getattr(entry, "pc_source", None) == "missing":
        raise HTTPException(status_code=400, detail="Pattern Center is not set for this dataset. Set/refine the PC first.")
    n_cols = int(det_params["n_cols"]); n_rows = int(det_params["n_rows"])
    if req.pixel_index >= n_rows * n_cols:
        raise HTTPException(status_code=400, detail="pixel_index out of range")
    row, col = divmod(req.pixel_index, n_cols)
    try:
        exp = np.asarray(signal.data[row, col], dtype=np.float32)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Could not read pattern at ({row},{col})")
    # Dynamic-background-remove the measured pattern (band-vs-band NCC + clean preview).
    if req.bg_remove:
        exp = _bg_remove_phase_test(exp)
    ap = req.aperture
    circular = detect_circular_aperture(exp) if ap == "auto" else (ap == "circular")
    mask = circular_mask(exp.shape, req.aperture_radius) if circular else None
    exp_display_arr = apply_circular_mask(exp, mask) if mask is not None else exp

    pixel_at_pct = get_pixel_at_pct_for_phase_test(req.pixel_index)
    entries = _phase_test_library_entries(req.phase_keys)
    tested, excluded, chem_by_key, eff_mode = eds_prefilter(
        entries, pixel_at_pct, req.eds_weighting, req.eds_filter_threshold)
    id_to_entry = {i: e for i, e in enumerate(tested)}
    sht_paths_by_phase = {i: str(e.sht_path) for i, e in id_to_entry.items()}
    phase_names = {i: (e.display_formula or e.formula or e.key) for i, e in id_to_entry.items()}

    def _png(arr):
        a = ((arr - arr.min()) / max(float(np.ptp(arr)), 1e-7) * 255).astype(np.uint8)
        buf = io.BytesIO(); Image.fromarray(a).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    exp_png = _png(exp_display_arr)

    jid = _new_phase_test_job()
    _phase_test_jobs[jid]["total"] = len(tested)
    asyncio.create_task(_run_phase_test_job(
        jid, exp=exp, det_params=det_params, sht_paths_by_phase=sht_paths_by_phase,
        phase_names=phase_names, mask=mask, max_bandwidth=req.max_bandwidth,
        id_to_entry=id_to_entry, chem_by_key=chem_by_key, eff_mode=eff_mode,
        excluded=excluded, row=row, col=col, pixel_index=req.pixel_index,
        exp_display=exp_png, pixel_at_pct=pixel_at_pct))
    return {"job_id": jid, "total": len(tested), "excluded_count": len(excluded),
            "eds_weighting_effective": eff_mode, "experimental_png": exp_png,
            "pixel_at_pct": pixel_at_pct, "row": row, "col": col}


@router.get("/single-pixel-phase-test/progress/{job_id}")
async def single_pixel_phase_test_progress(job_id: str):
    job = _phase_test_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return {"status": job["status"], "done": job["done"], "total": job["total"],
            "current": job["current"], "error": job["error"], "result": job["result"]}


@router.post("/single-pixel-phase-test/cancel/{job_id}")
async def single_pixel_phase_test_cancel(job_id: str):
    job = _phase_test_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    job["cancelled"] = True
    return {"ok": True}


@router.get("/single-pixel-phase-test/pixel-chemistry/{pixel_index}")
async def single_pixel_phase_test_pixel_chemistry(pixel_index: int):
    """Per-pixel EDS atomic-% for the active dataset — lightweight (no GPU, no
    detector, no pattern read; just one EDS-counts lookup).

    The phase-test dialog calls this whenever the crosshair moves so the EDS
    readout (and thus the user's mental model of which phases the chemistry
    filter will keep/drop) follows the selected pixel LIVE, instead of being
    frozen on the pixel of the last completed run — which previously made the
    chemistry look stuck on the first pixel and led to filtering against the
    wrong composition. Returns ``pixel_at_pct: null`` when there is no EDS,
    the grid is misaligned, or the index is out of range (fail-soft)."""
    if pixel_index < 0:
        raise HTTPException(status_code=400, detail="pixel_index must be >= 0")
    at_pct = get_pixel_at_pct_for_phase_test(pixel_index)
    return {"pixel_index": pixel_index, "pixel_at_pct": at_pct}


def _recompute_remask(cache: dict, aperture: str, aperture_radius: float) -> dict:
    """Recompute masked R + diff/sim PNGs from cached UNMASKED arrays for a new
    mask. Pure numpy/PNG work — no render, no index. Returns kwargs for
    SinglePixelPhaseTestResponse (candidates/excluded already assembled).

    Runs on a worker thread (called via asyncio.to_thread) so the per-phase
    NCC + PNG encode loop never touches the event loop, even though it's fast.
    """
    import io, base64
    import numpy as np
    from PIL import Image
    from tools.pattern_comparison import (
        circular_mask, apply_circular_mask, detect_circular_aperture,
        compute_ncc_scalar, compute_ncc_scalar_masked, ncc_diff_to_png_b64,
    )

    measured = cache["measured"]

    # Resolve the aperture exactly like the live endpoints do.
    if aperture == "auto":
        circular = detect_circular_aperture(measured)
    else:
        circular = (aperture == "circular")
    mask = circular_mask(measured.shape, aperture_radius) if circular else None

    # Same normalisation helper used for experimental_png/simulated_png elsewhere.
    def _png(arr):
        a = ((arr - arr.min()) / max(float(np.ptp(arr)), 1e-7) * 255).astype(np.uint8)
        buf = io.BytesIO(); Image.fromarray(a).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    exp_display = apply_circular_mask(measured, mask) if mask is not None else measured
    exp_png = _png(exp_display)

    sims = cache["sims"]
    rows = []
    for meta in cache["rows"]:
        pid = meta["phase_id"]
        sim_arr = sims.get(pid)
        if sim_arr is None:
            # Phase rendered no array (shouldn't happen for cached rows) — skip.
            continue
        _use_mask = mask is not None and sim_arr.shape == measured.shape
        if _use_mask:
            r_val = compute_ncc_scalar_masked(measured, sim_arr, mask)
        else:
            r_val = compute_ncc_scalar(measured, sim_arr)
        diff_b64 = ncc_diff_to_png_b64(measured, sim_arr, mask if _use_mask else None)
        sim_disp = apply_circular_mask(sim_arr, mask) if _use_mask else sim_arr
        rows.append({
            "phase_id": pid,
            "phase_name": meta["phase_name"],
            "r_score": float(r_val) if r_val == r_val else None,  # NaN -> None
            "ncc_score": meta.get("ncc_score"),
            "euler_deg": meta.get("euler_deg", []),
            "simulated_b64": _png(sim_disp),
            "ncc_diff_b64": diff_b64,
        })

    out_candidates, out_excluded = _assemble_phase_test_candidates(
        rows, cache["id_to_entry"], cache["chem_by_key"],
        cache["eff_mode"], cache["excluded"])

    return {
        "pixel_index": cache["pixel_index"], "row": cache["row"], "col": cache["col"],
        "experimental_png": exp_png, "pixel_at_pct": cache["pixel_at_pct"],
        "eds_weighting_effective": cache["eff_mode"], "mask_applied": mask is not None,
        "bandwidth": cache["bandwidth"],
        "candidates": out_candidates, "excluded": out_excluded,
    }


@router.post("/single-pixel-phase-test/remask", response_model=SinglePixelPhaseTestResponse)
async def single_pixel_phase_test_remask(req: PhaseTestRemaskRequest):
    """Recompute masked R + diff/simulated PNGs from the cached (unmasked)
    rendered patterns for a previous job, for a new mask — NO re-render.

    Toggling the detector mask or its radius only changes which pixels feed the
    NCC; the simulated patterns are unchanged. This reuses the arrays already
    rendered by the original job, so it's tens of ms instead of the multi-second
    re-index+render of a fresh run."""
    cache = _phase_test_remask_cache.get(req.job_id)
    if cache is None:
        raise HTTPException(
            status_code=404,
            detail="Re-run the test first — cached patterns expired")
    # Refresh LRU so an actively re-masked job isn't evicted out from under us.
    _phase_test_remask_cache.move_to_end(req.job_id)
    kwargs = await asyncio.to_thread(
        _recompute_remask, cache, req.aperture, req.aperture_radius)
    return SinglePixelPhaseTestResponse(**kwargs)


@router.get("/single-pixel-phase-test/phases")
async def single_pixel_phase_test_phases():
    """List the library phases that can be tested (those with an SHT)."""
    entries = _phase_test_library_entries(None)
    return [{"key": e.key, "label": (e.display_formula or e.formula or e.key),
             "space_group": getattr(e, "space_group", "") or ""} for e in entries]


@router.get("/ncc-heatmap")
async def get_ncc_heatmap(result_id: str = None):
    """Get NCC score heatmap as base64 PNG image."""
    result = _get_result(result_id)
    if result is None:
        raise HTTPException(status_code=400, detail="No indexing result available")

    from tools.pattern_comparison import get_ncc_score_map
    score_map = get_ncc_score_map(result)
    if score_map is None:
        raise HTTPException(status_code=400, detail="No score data available")

    n_rows, n_cols = result.original_shape

    import io as _io, base64
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    score_clipped = np.clip(score_map, 0, None)

    # Crop to bounding box of indexed (non-NaN) pixels
    valid_mask = np.isfinite(score_clipped)
    if valid_mask.any():
        rows_any = np.where(valid_mask.any(axis=1))[0]
        cols_any = np.where(valid_mask.any(axis=0))[0]
        crop_r0, crop_r1 = int(rows_any[0]), int(rows_any[-1]) + 1
        crop_c0, crop_c1 = int(cols_any[0]), int(cols_any[-1]) + 1
    else:
        crop_r0, crop_r1, crop_c0, crop_c1 = 0, n_rows, 0, n_cols

    score_cropped = score_clipped[crop_r0:crop_r1, crop_c0:crop_c1]
    crop_rows, crop_cols = score_cropped.shape

    vmax = min(float(np.nanmax(score_cropped)), 1.0)
    vmax = max(vmax, 0.01)

    # Full heatmap with colorbar (cropped)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    im = ax.imshow(score_cropped, cmap="RdYlGn", vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title("NCC Score Map")
    plt.colorbar(im, ax=ax, label="NCC")
    ax.axis("off")
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    heatmap_b64 = base64.b64encode(buf.read()).decode("ascii")

    # Clean heatmap for click mapping — direct PIL (no matplotlib margins)
    from PIL import Image as _PILImage
    import matplotlib.cm as _cm

    # Normalize scores to [0,1] for colormap, handle NaN
    score_norm = np.copy(score_cropped)
    nan_mask = ~np.isfinite(score_norm)
    score_norm[nan_mask] = 0.0
    score_norm = score_norm / vmax
    score_norm = np.clip(score_norm, 0.0, 1.0)

    # Apply RdYlGn colormap → RGBA → RGB
    colored = (_cm.RdYlGn(score_norm) * 255).astype(np.uint8)[:, :, :3]
    # Set NaN pixels to dark background
    colored[nan_mask] = [40, 42, 54]

    # Scale up for better click precision (min 400px wide)
    scale = max(1, 400 // max(crop_cols, 1))
    if scale > 1:
        colored = np.repeat(np.repeat(colored, scale, axis=0), scale, axis=1)

    clean_img = _PILImage.fromarray(colored)
    buf2 = _io.BytesIO()
    clean_img.save(buf2, format="PNG")
    buf2.seek(0)
    clean_b64 = base64.b64encode(buf2.read()).decode("ascii")

    valid = score_cropped[np.isfinite(score_cropped)]
    return {
        "heatmap": heatmap_b64,
        "heatmap_clean": clean_b64,
        "n_rows": crop_rows,
        "n_cols": crop_cols,
        "crop_row_offset": crop_r0,
        "crop_col_offset": crop_c0,
        "original_rows": n_rows,
        "original_cols": n_cols,
        "min_score": float(valid.min()) if len(valid) else 0.0,
        "max_score": float(valid.max()) if len(valid) else 0.0,
        "mean_score": float(valid.mean()) if len(valid) else 0.0,
    }


# ===========================================================================
# Phase B: Forward-NCC quality map (SHT-forward simulated vs experimental)
# ===========================================================================

def _stats_of_map(arr: np.ndarray) -> dict:
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "n_valid": 0}
    return {
        "min":  float(valid.min()),
        "max":  float(valid.max()),
        "mean": float(valid.mean()),
        "n_valid": int(len(valid)),
    }


@router.get("/forward-ncc/status")
async def get_forward_ncc_status(result_id: str = None):
    """Whether a forward-NCC map is already cached on the active result."""
    result = _get_result(result_id)
    if result is None:
        return {"ready": False, "reason": "no result"}
    md = result.metadata or {}
    fwd = md.get("forward_ncc_map")
    if fwd is None:
        return {"ready": False, "method": md.get("indexing_method")}
    stats = _stats_of_map(np.asarray(fwd))
    return {
        "ready": True,
        "method": md.get("indexing_method"),
        "bandwidth": md.get("forward_ncc_bandwidth"),
        **stats,
        "shape": list(np.asarray(fwd).shape),
    }


@router.post("/forward-ncc/compute")
# Declared `def` (not async) so FastAPI runs this 10-30s GPU-SHT forward-map
# computation in its worker threadpool instead of on the event loop — on the
# loop it froze every other request (incl. /health) for the whole compute.
# The body has no awaits.
def post_forward_ncc_compute(
    result_id: str = None,
    max_bandwidth: int = 256,
    force: bool = False,
):
    """Compute (and cache) the forward-NCC quality map for the active result.

    Synchronous: waits until compute finishes (~10-30 s on RTX 4070
    depending on map size + number of phases). The map is stored on
    ``result.metadata["forward_ncc_map"]`` so subsequent /heatmap calls
    are O(1).

    Parameters
    ----------
    max_bandwidth : int
        SHT truncation bandwidth (128/256/384). Higher = sharper bands
        in the simulated patterns but more GPU memory.
    force : bool
        Recompute even if a cached map already exists.
    """
    result = _get_result(result_id)
    if result is None:
        raise HTTPException(status_code=400, detail="No indexing result available")
    md = result.metadata or {}
    if not force and md.get("forward_ncc_map") is not None:
        cached = np.asarray(md["forward_ncc_map"])
        return {
            "status": "cached",
            "bandwidth": md.get("forward_ncc_bandwidth"),
            **_stats_of_map(cached),
        }

    from backend.api.services.sht_pattern_renderer import (
        compute_forward_ncc_map, SHTRenderError,
    )
    try:
        arr = compute_forward_ncc_map(result, max_bandwidth=int(max_bandwidth))
    except SHTRenderError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "status": "computed",
        "bandwidth": int(max_bandwidth),
        **_stats_of_map(arr),
    }


@router.get("/forward-ncc/heatmap")
async def get_forward_ncc_heatmap(result_id: str = None):
    """Return the forward-NCC map as a base64 PNG heatmap.

    Requires that POST /forward-ncc/compute has been called first;
    otherwise returns 404 so the frontend can show a "Compute" button.
    """
    result = _get_result(result_id)
    if result is None:
        raise HTTPException(status_code=400, detail="No indexing result available")
    md = result.metadata or {}
    arr = md.get("forward_ncc_map")
    if arr is None:
        raise HTTPException(
            status_code=404,
            detail="Forward-NCC map not computed yet. POST /forward-ncc/compute first.",
        )

    import io as _io, base64
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as _cm
    from PIL import Image as _PILImage

    arr = np.asarray(arr, dtype=np.float32)
    n_rows, n_cols = arr.shape

    # Crop bbox of indexed pixels (mirror /ncc-heatmap)
    valid_mask = np.isfinite(arr)
    if valid_mask.any():
        rows_any = np.where(valid_mask.any(axis=1))[0]
        cols_any = np.where(valid_mask.any(axis=0))[0]
        crop_r0, crop_r1 = int(rows_any[0]), int(rows_any[-1]) + 1
        crop_c0, crop_c1 = int(cols_any[0]), int(cols_any[-1]) + 1
    else:
        crop_r0, crop_r1, crop_c0, crop_c1 = 0, n_rows, 0, n_cols
    cropped = arr[crop_r0:crop_r1, crop_c0:crop_c1]
    crop_rows, crop_cols = cropped.shape

    # Use viridis like the CI map so the user can read the two side by side
    valid = cropped[np.isfinite(cropped)]
    vmin = float(valid.min()) if len(valid) else 0.0
    vmax = float(valid.max()) if len(valid) else 1.0
    if vmax - vmin < 1e-6:
        vmin, vmax = -1.0, 1.0

    norm = (cropped - vmin) / max(vmax - vmin, 1e-6)
    norm = np.clip(norm, 0.0, 1.0)
    nan_mask = ~np.isfinite(cropped)
    norm[nan_mask] = 0.0

    colored = (_cm.viridis(norm) * 255).astype(np.uint8)[:, :, :3]
    colored[nan_mask] = [40, 42, 54]

    # Upscale for click precision
    scale = max(1, 400 // max(crop_cols, 1))
    if scale > 1:
        colored = np.repeat(np.repeat(colored, scale, axis=0), scale, axis=1)
    img = _PILImage.fromarray(colored)
    buf = _io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    clean_b64 = base64.b64encode(buf.read()).decode("ascii")

    return {
        "heatmap_clean": clean_b64,
        "n_rows": crop_rows,
        "n_cols": crop_cols,
        "crop_row_offset": crop_r0,
        "crop_col_offset": crop_c0,
        "original_rows": n_rows,
        "original_cols": n_cols,
        "min_score": vmin,
        "max_score": vmax,
        "mean_score": float(valid.mean()) if len(valid) else 0.0,
        "bandwidth": md.get("forward_ncc_bandwidth"),
    }


@router.get("/result/last")
async def get_last_result():
    """Get summary of the most recent indexing result."""
    active = _get_result()
    if active is None:
        return {"available": False}
    resp = {
        "result_id": _active_result_id,
        "n_indexed": int(active.selection_mask.sum()),
        "original_shape": list(active.original_shape),
        "method": active.method.value,
        "has_confidence": active.confidence_scores is not None,
    }
    if active.confidence_scores is not None:
        # Defensive collapse for 2D pyebsdindex cm — same fix as
        # commit c474f41 (/results) and 1a2c123 (/start completion).
        # Without this, mean_ci/min_ci/max_ci returned by /result/last mix
        # phase-0 and consensus values for multi-phase pyebsdindex output.
        arr = np.asarray(active.confidence_scores)
        if arr.ndim >= 2:
            arr = arr[-1]
        valid = arr[~np.isnan(arr)] if hasattr(arr, '__len__') else arr
        if len(valid) > 0:
            resp["mean_ci"] = float(np.nanmean(valid))
            resp["min_ci"] = float(np.nanmin(valid))
            resp["max_ci"] = float(np.nanmax(valid))
    # Include phase names if available (exclude not_indexed, align with backend PHASE_COLORS)
    if hasattr(active, 'xmap') and active.xmap is not None:
        try:
            phases_in = active.xmap.phases_in_data
            # Build list with phase_id so frontend can map colors correctly
            resp["phases"] = [
                {"id": int(pid), "name": name}
                for pid, name in zip(phases_in.ids, phases_in.names)
                if pid != -1
            ]
            # Count not_indexed pixels
            n_not_indexed = int(np.sum(active.xmap.phase_id == -1))
            if n_not_indexed > 0:
                resp["n_not_indexed"] = n_not_indexed
        except Exception:
            pass

    # For single-pixel results (quick test), include per-pixel detail
    if resp["n_indexed"] == 1 and hasattr(active, 'xmap') and active.xmap is not None:
        try:
            xmap = active.xmap
            # Phase name of the single indexed pixel
            if hasattr(xmap, 'phases_in_data') and len(xmap.phases_in_data) > 0:
                resp["phase_name"] = xmap.phases_in_data[0].name
            # CI / confidence
            if active.confidence_scores is not None:
                valid = active.confidence_scores[~np.isnan(active.confidence_scores)]
                if len(valid) > 0:
                    resp["ci"] = float(valid[0])
            # Euler angles in degrees
            if hasattr(xmap, 'rotations') and len(xmap.rotations) > 0:
                euler = xmap.rotations[0].to_euler(degrees=True)
                resp["euler_deg"] = [float(e) for e in euler.flatten()[:3]]
        except Exception:
            pass

    return resp


@router.get("/results")
async def list_results():
    """List all stored indexing results with metadata."""
    results = []
    for rid, res in _result_registry.items():
        source_file = None
        if hasattr(res, "metadata") and isinstance(res.metadata, dict):
            source_file = res.metadata.get("source_file")
        entry = {
            "id": rid,
            "method": res.method.value,
            "n_indexed": int(res.selection_mask.sum()),
            "original_shape": list(res.original_shape),
            "is_active": rid == _active_result_id,
            "source_file": source_file,
            "phases": [],
            "mean_ci": None,
        }
        # Add phase names from xmap (exclude not_indexed, include phase_id)
        if res.xmap is not None:
            try:
                phases_in = res.xmap.phases_in_data
                entry["phases"] = [
                    {"id": int(pid), "name": name}
                    for pid, name in zip(phases_in.ids, phases_in.names)
                    if pid != -1
                ]
            except Exception:
                entry["phases"] = []
        # Add mean confidence score. Defensive collapse: if a stored result
        # somehow has 2D confidence_scores (bypassed _extract_confidence),
        # use only the consensus row so the displayed mean_ci isn't an
        # average across all-phases-plus-consensus (would overstate CI).
        if res.confidence_scores is not None:
            try:
                arr = np.asarray(res.confidence_scores)
                if arr.ndim >= 2:
                    arr = arr[-1]
                scores = arr[~np.isnan(arr)]
                entry["mean_ci"] = round(float(scores.mean()), 4) if len(scores) > 0 else None
            except Exception:
                pass
        results.append(entry)
    return {"results": results, "active_id": _active_result_id}


@router.post("/results/activate/{result_id}")
async def activate_result(result_id: str):
    """Set the active indexing result.

    If the result's source_file differs from the currently-loaded file,
    auto-switch the file first (via the same load pipeline as
    /api/ebsd/switch-file). This way the frontend can click any gallery
    entry without manually switching files first; the backend keeps the
    file and the active result consistent.

    Returns 404 if the result is gone, 409 (with a hint) if the source
    file is no longer reachable on disk.
    """
    global _active_result_id
    if result_id not in _result_registry:
        raise HTTPException(status_code=404, detail=f"Result '{result_id}' not found")
    res = _result_registry[result_id]
    source_file = None
    if hasattr(res, "metadata") and isinstance(res.metadata, dict):
        source_file = res.metadata.get("source_file")
    if source_file:
        from backend.api.routes import ebsd_viewer as _ev
        if str(_ev._ebsd_file_path) != str(source_file):
            # Auto-switch the file so render/pattern-match see the right
            # xmap. We MUST set _active_result_id AFTER the load, because
            # load_ebsd resets it to None on every file change.
            from pathlib import Path as _Path
            if not _Path(source_file).is_file():
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": "Result's source file no longer on disk",
                        "result_source_file": source_file,
                        "current_file": str(_ev._ebsd_file_path) if _ev._ebsd_file_path else None,
                    },
                )
            try:
                await _ev.load_ebsd(_ev.LoadEBSDRequest(path=source_file))
            except Exception as e:
                logger.exception("auto-switch failed during activate_result")
                raise HTTPException(
                    status_code=500,
                    detail=f"failed to auto-switch to result's source file: {e}",
                )
    _active_result_id = result_id
    state_version.bump()
    return {"status": "ok", "active_id": result_id, "auto_switched_file": source_file}


@router.post("/results/deactivate")
async def deactivate_result():
    """Clear the active indexing result so Phase Map / render fall back to
    the _analysis_dataset (populated by /api/analysis/load). Used when the
    user switches to a gallery entry that was loaded from an .ang / .h5
    file rather than produced by the current indexing run."""
    global _active_result_id
    _active_result_id = None
    state_version.bump()
    return {"status": "ok", "active_id": None}


@router.delete("/results/{result_id}")
async def delete_result(result_id: str):
    """Delete a stored indexing result."""
    global _active_result_id
    if result_id not in _result_registry:
        raise HTTPException(status_code=404, detail=f"Result '{result_id}' not found")
    del _result_registry[result_id]
    if _active_result_id == result_id:
        _active_result_id = next(iter(_result_registry), None)
    return {"status": "ok", "remaining": len(_result_registry)}


class ImportH5Request(BaseModel):
    path: str


@router.post("/import-h5")
async def import_h5_result(req: ImportH5Request):
    """Import a rich/light .h5 export back into the active indexing session.

    Three things happen in one call so the user gets a full Save-as-able
    gallery entry from a single click in "Add file…":

    1. Parse ``/Indexing`` via the existing rich-h5 loader (handles both the
       Assignment-nested layout and the flat single-phase layout).
    2. Wrap the resulting CrystalMap in an ``IndexingResult`` and register
       it. The returned ``result_id`` makes ``Save as…`` work — including
       the new Light .h5 export.
    3. If the file carries ``/1/EBSD`` (rich .h5 inherits it from the source
       h5oina via ``shutil.copy2``), also call ``load_ebsd_file`` so the
       EBSD Viewer picks up patterns. EDS routes read ``/1/EDS`` straight
       off ``_ebsd_file_path`` so they automatically pick the new source up.

    Returns a capability dict so the UI can tell the user what's available
    in the loaded file. Light files report ``patterns=False, eds=False``;
    rich files report ``True/True`` (or whichever the source had).
    """
    import h5py
    from pathlib import Path as _P

    p = _P(req.path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {p}")

    capabilities = {
        "indexing": False,
        "patterns": False,
        "eds": False,
        "electron_image": False,
    }
    try:
        with h5py.File(str(p), "r") as f:
            capabilities["indexing"] = "Indexing" in f
            # Either name is valid — h5oina uses "Processed Patterns",
            # raw kikuchipy datasets use "Patterns". Rich h5 inherits
            # whatever the source had.
            capabilities["patterns"] = (
                "1/EBSD/Data/Processed Patterns" in f
                or "1/EBSD/Data/Patterns" in f
            )
            capabilities["eds"] = "1/EDS" in f
            capabilities["electron_image"] = "1/Electron Image" in f
    except OSError as e:
        raise HTTPException(
            status_code=400, detail=f"Cannot open as HDF5: {e}"
        )

    if not capabilities["indexing"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{p.name} has no /Indexing group — not a GUI rich/light "
                ".h5 export. To import a raw .h5oina, use /api/ebsd/load."
            ),
        )

    # Build the CrystalMap via the same loader Phase Map uses today.
    from backend.api.routes.analysis import _load_kikuchipy_rich_h5
    try:
        xmap = _load_kikuchipy_rich_h5(str(p))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to parse /Indexing: {e}"
        )

    # Pull the IndexingResult-shaped fields (selection_mask, CI, method)
    # straight out of the file. The loader returns only the xmap.
    with h5py.File(str(p), "r") as f:
        idx = f["Indexing"]
        method_raw = idx.attrs.get("method", "spherical")
        if isinstance(method_raw, bytes):
            method_raw = method_raw.decode("utf-8", errors="replace")
        method_str = str(method_raw).lower()

        gs_attr = idx.attrs.get("grid_shape", None)
        if gs_attr is not None and len(list(gs_attr)) == 2:
            grid_shape = (int(gs_attr[0]), int(gs_attr[1]))
        else:
            grid_shape = tuple(xmap.shape)

        n_pix = int(np.prod(grid_shape))

        # Selection mask — try flat first, then Assignment, then default-all.
        if "selection_mask" in idx:
            sel = np.asarray(idx["selection_mask"]).astype(bool)
        elif "Assignment" in idx and "selection_mask" in idx["Assignment"]:
            sel = np.asarray(idx["Assignment/selection_mask"]).astype(bool)
        else:
            sel = np.ones(grid_shape, dtype=bool)
        if sel.size != n_pix:
            sel = np.ones(grid_shape, dtype=bool)
        selection_mask = sel.flatten()

        # Confidence — same precedence (Assignment first since that's the
        # canonical multi-phase location, then flat single-phase fallback).
        if "Assignment" in idx and "confidence_index" in idx["Assignment"]:
            ci_raw = np.asarray(idx["Assignment/confidence_index"]).flatten()
        elif "confidence_index" in idx:
            ci_raw = np.asarray(idx["confidence_index"]).flatten()
        else:
            ci_raw = np.asarray(xmap.prop.get("ci", np.zeros(xmap.size, dtype=np.float32)))
        if ci_raw.size != n_pix:
            ci_raw = np.zeros(n_pix, dtype=np.float32)

    from indexing_controller import IndexingMethod, IndexingResult
    method_map = {
        "spherical": IndexingMethod.SPHERICAL,
        "dictionary": IndexingMethod.DICTIONARY,
        "hough": IndexingMethod.HOUGH,
    }
    method = method_map.get(method_str, IndexingMethod.SPHERICAL)

    result = IndexingResult(
        xmap=xmap,
        selection_mask=selection_mask,
        original_shape=grid_shape,
        method=method,
        confidence_scores=ci_raw.astype(np.float32),
        metadata={"source_file": str(p), "imported_from_h5": True},
    )

    # Carry per-phase CI maps over from the xmap (loader stashes them) so
    # subsequent re-exports re-emit /Indexing/PerPhase/ unchanged.
    per_phase = getattr(xmap, "_per_phase_data", None)
    if per_phase:
        result.metadata["per_phase_data"] = per_phase

    result_id = _store_result(result, method.value)

    # If the file has /1/EBSD, also bring its patterns into the EBSD viewer.
    # Failure here is non-fatal — the indexing result is still usable, the
    # user just won't see patterns until they load the source separately.
    ebsd_signal_loaded = False
    if capabilities["patterns"]:
        try:
            from backend.api.routes.ebsd_viewer import load_ebsd_file
            ebsd_signal_loaded = load_ebsd_file(str(p))
        except Exception as e:
            logger.warning(
                "import-h5: EBSD signal load failed (%s) — patterns won't "
                "appear in EBSD Viewer until you load the source separately",
                e,
            )

    # Also open the h5_session and register the calibration. The EDS page
    # reads through ``h5_session`` (NOT ``_ebsd_file_path``), so without this
    # explicit open the EDS routes still see the previously-loaded file (or
    # nothing) even though ``/1/EDS`` sits right there in the rich h5. The
    # regular ``/api/ebsd/load`` route does both of these steps — mirror
    # them so an imported rich .h5 is fully equivalent to "loaded the
    # source h5oina" for every downstream page.
    if ebsd_signal_loaded:
        try:
            from backend.api.services.h5_session import (
                open_file as h5_open, is_open as h5_is_open,
                close_file as h5_close, get_current_path,
            )
            if h5_is_open() and get_current_path() != str(p):
                h5_close()
            if not h5_is_open():
                h5_open(str(p))
        except Exception as e:
            logger.warning(
                "import-h5: h5_session open failed (%s) — EDS page will be empty",
                e,
            )
        try:
            from backend.api.routes.ebsd_viewer import _get_active_signal
            from backend.api.services.calibration_store import calibration_store
            signal = _get_active_signal()
            if signal is not None:
                calibration_store.clear()
                calibration_store.register(p.stem, signal)
        except Exception as e:
            logger.warning(
                "import-h5: calibration_store register failed (%s) — detector PC "
                "may need manual re-entry",
                e,
            )

    try:
        ci_for_mean = ci_raw[selection_mask] if selection_mask.any() else ci_raw
        ci_mean = float(np.mean(ci_for_mean)) if ci_for_mean.size else 0.0
    except Exception:
        ci_mean = 0.0

    phase_names: list = []
    try:
        for entry in xmap.phases:
            ph = entry[1] if isinstance(entry, tuple) else entry
            nm = getattr(ph, "name", "") or ""
            if nm and nm.lower() != "not_indexed":
                phase_names.append(str(nm))
    except Exception:
        pass

    label = (
        f"{method.value} — {' / '.join(phase_names) if phase_names else 'unknown'} "
        f"— CI: {ci_mean:.3f} — {int(np.sum(selection_mask))} px"
    )

    return {
        "result_id": result_id,
        "label": label,
        "shape": list(grid_shape),
        "phases": phase_names,
        "ci_mean": ci_mean,
        "n_pixels": int(np.sum(selection_mask)),
        "capabilities": capabilities,
        "ebsd_signal_loaded": ebsd_signal_loaded,
        "source_path": str(p),
    }


@router.get("/methods")
async def list_methods():
    """List available indexing methods with their requirements."""
    return {
        "methods": [
            {
                "id": "hough",
                "name": "Hough Indexing",
                "requires": "CIF file",
                "speed": "Standard",
            },
            {
                "id": "dictionary",
                "name": "Dictionary Indexing",
                "requires": "Master H5 file",
                "speed": "Slow (but accurate)",
            },
            {
                "id": "spherical",
                "name": "Spherical Indexing (EMSphinx)",
                "requires": "SHT file + WSL",
                "speed": "Fast",
            },
        ]
    }


@router.get("/files/{method}")
async def discover_files(method: str, material_hint: str = "", current_pc: str = ""):
    """Discover available files for a specific indexing method."""
    try:
        from indexing_controller import IndexingMethod, discover_files_for_method

        method_map = {
            "hough": IndexingMethod.HOUGH,
            "dictionary": IndexingMethod.DICTIONARY,
            "spherical": IndexingMethod.SPHERICAL,
        }

        if method not in method_map:
            raise HTTPException(status_code=400, detail=f"Unknown method: {method}")

        # Parse current_pc from comma-separated string
        pc_list = None
        if current_pc:
            try:
                pc_list = [float(x.strip()) for x in current_pc.split(",")]
                if len(pc_list) != 3:
                    pc_list = None
            except ValueError:
                pc_list = None

        result = discover_files_for_method(
            method_map[method], material_hint,
            current_pc=pc_list,
        )
        files = result["files"] if isinstance(result, dict) else result
        groups = result.get("groups", []) if isinstance(result, dict) else []
        # Convert Path objects to strings
        for f in files:
            f["path"] = str(f["path"])
        return {"files": files, "groups": groups}
    except HTTPException:
        # Preserve original status — without this, the "Unknown method"
        # 400 raised above became a 500 with detail "400: Unknown method".
        raise
    except ImportError:
        return {"files": [], "groups": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ExportRequest(BaseModel):
    format: str = "ang"  # "ang", "ctf", "h5", "h5_light"
    filename: Optional[str] = None  # auto-generated if not provided
    include_eds: bool = True
    include_detector: bool = True


@router.post("/export")
async def export_indexing_result(req: ExportRequest):
    """Export indexing result as .ang, rich .h5, or light .h5.

    .ang:      Standard orientation file via orix (MTEX compatible).
    .h5:       Copies original h5oina + adds /Indexing/ with per-phase Euler/CI.
               Multi-GB if the source is multi-GB. Inherits source's HDF5
               superblock version — can be unreadable in MATLAB < R2020a.
    .h5_light: Same /Indexing structure but NO source copy. Fresh file with
               superblock v0 → MATLAB-compatible on all versions. Typical
               5–50 MB regardless of source size. Recommended for sharing.
    """
    active = _get_result()
    if active is None:
        raise HTTPException(status_code=400, detail="No indexing result available")

    # Opt-in per-file coordinate-system rotation. When the file's FrameSpec has
    # apply_to_export set, compose R_user into the exported orientation frame;
    # otherwise _r_user stays None and the export is byte-identical to today.
    from backend.api.services import reference_frame_state as rfs
    _spec = rfs.get_frame((active.metadata or {}).get("source_file"))
    _r_user = rfs.resolve_r_user(_spec) if _spec.get("apply_to_export") else None

    import tempfile
    from pathlib import Path

    fmt = req.format.lower().lstrip(".")

    # User-facing warning surfaced on the response when the .ang opt-in path
    # cannot honor the coordinate-system rotation. Stays None on every other
    # path (incl. default apply_to_export=false) → response is unchanged.
    _frame_warning = None

    # Determine output path
    if req.filename:
        out_path = Path(req.filename)
    else:
        # Default name mirrors the user-visible suffix: foo.ang, foo.h5,
        # foo_light.h5. Without the _light suffix the two h5 variants would
        # collide on the same default path.
        if fmt == "ang":
            tail = ".ang"
        elif fmt == "h5_light":
            tail = "_light.h5"
        else:
            tail = ".h5"
        out_path = Path(tempfile.mkdtemp()) / f"indexing_result_{active.method.value}{tail}"

    try:
        if fmt == "ang":
            # Standard orix export — MTEX compatible
            from indexing_controller import export_results

            # KNOWN LIMITATION (loud, not silent): the orix .ang writer is
            # called directly and does NOT compose the per-file coordinate-
            # system rotation (_r_user). Only the .h5 / Light .h5 paths thread
            # r_user through to_vendor_export_frame. So when the user turned on
            # "Apply to exports", the .ang stays in the vendor frame while the
            # .h5 exports are rotated. Threading r_user into the .ang path is a
            # deeper follow-up; for now warn in the log AND on the response so
            # the discrepancy is never silent. Default path (_r_user is None)
            # leaves _frame_warning None → response byte-identical to today.
            if _r_user is not None:
                _frame_warning = (
                    ".ang export does not apply the coordinate-system "
                    "rotation; use Light .h5 for coordinate-system-correct "
                    "orientations, or disable 'Apply to exports'."
                )
                logger.warning(
                    "Export: .ang format does not yet honor the per-file "
                    "coordinate-system rotation (apply_to_export=True); only "
                    ".h5 exports are rotated. The exported .ang stays in the "
                    "vendor frame."
                )

            # Inject confidence_scores into xmap.prop BEFORE the orix .ang
            # writer pulls them — otherwise every CI column is 0.0 because
            # our IndexingResult stores CI on the result object, not on the
            # xmap itself. User-visible: exported .ang had mean_ci=0.37 but
            # every row had confidence_index=0.000.
            if (active.confidence_scores is not None
                    and active.xmap is not None
                    and 'scores' not in active.xmap.prop
                    and 'ci' not in active.xmap.prop):
                ci_arr = np.asarray(active.confidence_scores).ravel()
                if ci_arr.ndim >= 2:
                    ci_arr = ci_arr[-1]  # consensus row for 2D cm
                n_xmap = active.xmap.rotations.size
                if ci_arr.size >= n_xmap:
                    # Trim to the xmap size; sparse back-mapped xmaps
                    # need the leading n_xmap values (one per indexed pixel)
                    active.xmap.prop['ci'] = ci_arr[:n_xmap].astype(np.float32)
                    logger.info("Injected ci into xmap.prop before .ang export (size=%d)", n_xmap)

            export_results(active.xmap, str(out_path))

        elif fmt == "h5":
            # Rich H5 export: original data + per-phase indexing results
            import h5py

            # Get source file path for copying original data
            source_path = None
            try:
                from backend.api.routes.ebsd_viewer import _ebsd_file_path
                source_path = _ebsd_file_path
            except Exception:
                pass

            if source_path and Path(source_path).is_file():
                # Copy original h5oina as base
                import shutil
                shutil.copy2(str(source_path), str(out_path))
            else:
                # No source file — create empty H5
                with h5py.File(str(out_path), "w") as f:
                    pass

            # Write indexing results
            with h5py.File(str(out_path), "a") as f:
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc).isoformat()

                if "Indexing" in f:
                    del f["Indexing"]

                idx = f.create_group("Indexing")
                idx.attrs["method"] = active.method.value
                idx.attrs["software"] = "Orienta"
                idx.attrs["created"] = now
                idx.attrs["grid_shape"] = list(active.original_shape)

                # Export-frame conversion (see orientation_frame): write Euler
                # in the source vendor's stored frame so MTEX/Aztec read it
                # aligned. The reader inverts via source_vendor on re-import.
                from backend.api.services.orientation_frame import (
                    to_vendor_export_frame as _to_vendor_frame,
                    export_frame_offset as _export_frame_offset,
                )
                _export_vendor = _resolve_source_vendor(
                    active, str(source_path) if source_path else None
                )
                idx.attrs["source_vendor"] = _export_vendor or "unknown"
                idx.attrs["orientation_reference_frame"] = (
                    "vendor_stored (Aztec/MTEX default import)"
                    if float(_export_frame_offset(_export_vendor).angle.max()) > 1e-6
                    else "native (EMsoft/kikuchipy common)"
                )

                n_rows, n_cols = active.original_shape
                per_phase_data = active.metadata.get('per_phase_data', {})

                if per_phase_data:
                    # Multi-phase: write per-phase results
                    per_phase_grp = idx.create_group("PerPhase")
                    per_phase_grp.attrs["description"] = (
                        "Results for each phase tested. Each pixel has orientation + CI "
                        "for every phase, allowing post-processing phase reassignment."
                    )
                    phase_names = list(per_phase_data.keys())

                    for i, name in enumerate(phase_names):
                        pd = per_phase_data[name]
                        pg = per_phase_grp.create_group(name)
                        pg.attrs["phase_id"] = i + 1
                        pg.attrs["ci_mean"] = pd.get("ci_mean", 0.0)

                        if pd.get("euler") is not None:
                            ds = pg.create_dataset(
                                "euler_angles",
                                data=_euler_ndarray_to_vendor(
                                    pd["euler"], _export_vendor, r_user=_r_user),
                                dtype=np.float32, compression="gzip")
                            ds.attrs["unit"] = "radians"
                            ds.attrs["convention"] = "Bunge (ZXZ)"
                            ds.attrs["description"] = "Euler angles if this phase is the correct assignment"

                        if pd.get("ci") is not None:
                            ds = pg.create_dataset("confidence_index", data=pd["ci"], dtype=np.float32, compression="gzip")
                            ds.attrs["description"] = "Confidence Index [0,1] for this phase at each pixel"

                    # Auto-assignment from consensus xmap
                    aa_grp = idx.create_group("Assignment")
                    aa_grp.attrs["description"] = "Best phase per pixel (highest CI wins)"

                    if hasattr(active.xmap, "rotations"):
                        euler = _to_vendor_frame(
                            active.xmap.rotations, _export_vendor, r_user=_r_user
                        ).to_euler(degrees=False)
                        euler_arr = np.array(euler).reshape((*active.original_shape, 3)).astype(np.float32)
                        ds = aa_grp.create_dataset("euler_angles", data=euler_arr, dtype=np.float32, compression="gzip")
                        ds.attrs["unit"] = "radians"
                        ds.attrs["convention"] = "Bunge (ZXZ)"

                    if hasattr(active.xmap, "phase_id"):
                        # Convert orix-native phase_id (-1=unindexed, 0..N-1=phase)
                        # to the on-disk convention the loader expects
                        # (0=unindexed, 1..N=phase). Writing the raw orix values
                        # made load_xmap treat Al (orix id 0) as unindexed and
                        # dropped one full phase on re-import.
                        raw_pid = np.array(active.xmap.phase_id).reshape(active.original_shape)
                        written_pid = np.where(raw_pid < 0, 0, raw_pid + 1).astype(np.uint8)
                        ds = aa_grp.create_dataset("phase_id", data=written_pid, compression="gzip")
                        ds.attrs["phase_names"] = phase_names

                    if active.confidence_scores is not None:
                        ds = aa_grp.create_dataset("confidence_index", data=active.confidence_scores.reshape(active.original_shape).astype(np.float32), compression="gzip")

                else:
                    # Single-phase: write directly
                    if hasattr(active.xmap, "rotations"):
                        euler = _to_vendor_frame(
                            active.xmap.rotations, _export_vendor, r_user=_r_user
                        ).to_euler(degrees=False)
                        ds = idx.create_dataset("euler_angles", data=np.array(euler).reshape((*active.original_shape, 3)).astype(np.float32), compression="gzip")
                        ds.attrs["unit"] = "radians"
                        ds.attrs["convention"] = "Bunge (ZXZ)"

                    if hasattr(active.xmap, "phase_id"):
                        # Same convention shift as multi-phase — see comment above.
                        raw_pid = np.array(active.xmap.phase_id).reshape(active.original_shape)
                        written_pid = np.where(raw_pid < 0, 0, raw_pid + 1).astype(np.uint8)
                        idx.create_dataset("phase_id", data=written_pid, compression="gzip")

                    if active.confidence_scores is not None:
                        idx.create_dataset("confidence_index", data=active.confidence_scores.reshape(active.original_shape).astype(np.float32), compression="gzip")

                # Selection mask
                idx.create_dataset("selection_mask", data=active.selection_mask.astype(np.uint8), compression="gzip")

                # Phase table /Indexing/Phases/<1..N> — the Analysis loader
                # (load_xmap_result) reads names + symmetry from here.
                # Without this group, reloaded phases fall back to generic
                # "phase_1/phase_2" names and triclinic symmetry, which
                # breaks the legend and IPF colouring.
                try:
                    phases_tbl = idx.create_group("Phases")
                    xmap_phases = list(active.xmap.phases) if active.xmap is not None else []
                    i_counter = 0
                    for entry in xmap_phases:
                        # orix 0.12+ yields (id, Phase); older versions yield a Phase
                        if isinstance(entry, tuple) and len(entry) == 2:
                            pid, phase_obj = entry
                            try:
                                if int(pid) < 0:
                                    continue
                            except Exception:
                                pass
                        else:
                            phase_obj = entry
                            pid = getattr(entry, "id", None)
                            try:
                                if pid is not None and int(pid) < 0:
                                    continue
                            except Exception:
                                pass
                        i_counter += 1
                        pg = phases_tbl.create_group(str(i_counter))
                        name = getattr(phase_obj, "name", "") or f"phase_{i_counter}"
                        pg.attrs["name"] = str(name)
                        sg = getattr(phase_obj, "space_group", None)
                        if sg is not None:
                            try:
                                pg.attrs["space_group"] = int(getattr(sg, "number", sg))
                            except (TypeError, ValueError):
                                pass
                        pgrp = getattr(phase_obj, "point_group", None)
                        if pgrp is not None:
                            try:
                                pg.attrs["point_group"] = str(getattr(pgrp, "name", pgrp))
                            except Exception:
                                pass
                except Exception as e:
                    logger.warning("Export: failed to write /Indexing/Phases: %s", e)

                # Detector
                if req.include_detector:
                    try:
                        from backend.api.routes.ebsd_viewer import _get_active_signal, _active_dataset
                        from backend.api.services.calibration_store import calibration_store
                        if "Detector" not in f:
                            det_grp = f.create_group("Detector")
                        else:
                            det_grp = f["Detector"]
                        signal = _get_active_signal()
                        if signal is not None:
                            det = getattr(signal, "detector", None)
                            if det is not None:
                                # PC from the calibration store so the export carries the
                                # REFINED PC (a Global PC Refine writes the store; signal.detector
                                # is a separate channel and would be stale). Mean for a per-pixel
                                # map, matching batch export; fall back to signal.detector.
                                store_pc = calibration_store.get_pc(_active_dataset)
                                pc_out = store_pc if store_pc is not None else np.array(det.pc).flatten()[:3]
                                if "pc" not in det_grp:
                                    det_grp.create_dataset("pc", data=np.asarray(pc_out, dtype=float).flatten()[:3])
                                det_grp.attrs["sample_tilt"] = float(det.sample_tilt)
                                det_grp.attrs["shape"] = list(det.shape)
                    except Exception as e:
                        logger.warning("Export: detector info failed: %s", e)

                # Documentation
                if "Documentation" not in f:
                    from backend.api.services.result_exporter import README_TEXT
                    doc = f.create_group("Documentation")
                    doc.attrs["format_version"] = "1.0"
                    doc.create_dataset("README", data=README_TEXT)

        elif fmt == "h5_light":
            # Light H5 export — fresh file, NO source copy. Same /Indexing
            # structure as the rich branch above, but the source patterns
            # are NOT replicated. Two upsides:
            #   1. ~5–50 MB output instead of multi-GB → can be shared.
            #   2. libver='earliest' keeps the superblock at version 0 →
            #      MATLAB R2019b / HDF5 1.8.x can open it. The rich branch
            #      inherits the source's v2 superblock (h5oina ships v2),
            #      which trips old MATLABs with "H5Fget_obj_count not a
            #      file id" inside h5info.
            # Code is duplicated from the rich branch on purpose: a refactor
            # would touch the working rich path. Sync changes by hand if
            # the rich /Indexing layout ever changes.
            import h5py
            from backend.api.services.result_exporter import (
                FORMAT_VERSION as _LIGHT_FMT_VERSION,
                README_TEXT as _LIGHT_README,
                _copy_quality_fields,
            )
            from datetime import datetime, timezone

            # Source path is optional — if the original h5oina has been
            # moved/deleted, we still produce a valid light file (just
            # without the BC/PC/bands quality fields and without the
            # absolute path in /SourceReference).
            src_path_str = None
            try:
                from backend.api.routes.ebsd_viewer import _ebsd_file_path
                src_path_str = _ebsd_file_path
            except Exception:
                pass

            # Resolve step size BEFORE opening the file so a failure here
            # doesn't leave a stray partial .h5 on disk. Fail loud rather than
            # write a 0-step file that silently mis-scales the map in MTEX
            # (Irmi's 2026-06-02 feedback).
            step_size_um = _resolve_step_size_um(active)
            if step_size_um is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Cannot determine the scan step size for export. "
                        "Load the source EBSD data (so its calibration is "
                        "available) and try again."
                    ),
                )

            # Resolve the source vendor so we can write Euler in the vendor's
            # stored frame (Aztec/MTEX default import). Our indexers emit the
            # EMsoft/kikuchipy common frame, which is 90deg about ND from
            # Oxford/Bruker — see orientation_frame.to_vendor_export_frame.
            from backend.api.services.orientation_frame import (
                to_vendor_export_frame as _to_vendor_frame,
                export_frame_offset as _export_frame_offset,
            )
            _export_vendor = _resolve_source_vendor(active, src_path_str)

            with h5py.File(str(out_path), "w", libver="earliest") as f:
                now = datetime.now(timezone.utc).isoformat()

                ref = f.create_group("SourceReference")
                ref.attrs["description"] = (
                    "Pointer to the original source file. Data was NOT "
                    "copied to save disk. Open the source directly if "
                    "you need raw patterns."
                )
                if src_path_str:
                    sp = Path(src_path_str)
                    ref.attrs["source_file_path"] = str(sp.absolute())
                    ref.attrs["source_file_name"] = sp.name
                    ref.attrs["source_file_stem"] = sp.stem
                    try:
                        ref.attrs["source_size_mb"] = round(
                            sp.stat().st_size / (1024**2), 1
                        )
                    except OSError:
                        pass

                idx = f.create_group("Indexing")
                idx.attrs["method"] = active.method.value
                idx.attrs["software"] = "Orienta"
                idx.attrs["created"] = now
                idx.attrs["grid_shape"] = list(active.original_shape)
                idx.attrs["format_version"] = _LIGHT_FMT_VERSION

                # step_size_um resolved above (before the file was opened).
                # Per-pixel X/Y µm coordinates let MTEX place the orientations
                # on a grid without the source h5oina.
                idx.attrs["step_size_um"] = float(step_size_um)
                _n_rows, _n_cols = active.original_shape
                _cc, _rr = np.meshgrid(
                    np.arange(_n_cols), np.arange(_n_rows)
                )
                # X varies with column, Y with row — same convention as the
                # .ang writer (result_exporter.py): coord = index · step.
                X_ds = idx.create_dataset(
                    "X", data=(_cc * step_size_um).astype(np.float32),
                    compression="gzip",
                )
                X_ds.attrs["unit"] = "um"
                X_ds.attrs["description"] = "Sample X (column · step_size_um)"
                Y_ds = idx.create_dataset(
                    "Y", data=(_rr * step_size_um).astype(np.float32),
                    compression="gzip",
                )
                Y_ds.attrs["unit"] = "um"
                Y_ds.attrs["description"] = "Sample Y (row · step_size_um)"

                # Orientation reference-frame tag. Euler datasets below are
                # written in the vendor's stored frame so a raw read (MTEX
                # default import / h5read) matches the vendor (Aztec) solution.
                # The reader (analysis._load_kikuchipy_rich_h5) inverts this on
                # re-import via source_vendor so working maps stay native.
                _frame_converted = bool(
                    _export_frame_offset(_export_vendor).angle.max() > 1e-6
                )
                idx.attrs["source_vendor"] = _export_vendor or "unknown"
                idx.attrs["orientation_reference_frame"] = (
                    "vendor_stored (Aztec/MTEX default import)"
                    if _frame_converted else "native (EMsoft/kikuchipy common)"
                )

                per_phase_data = active.metadata.get('per_phase_data', {})

                if per_phase_data:
                    per_phase_grp = idx.create_group("PerPhase")
                    per_phase_grp.attrs["description"] = (
                        "Per-phase results. Each pixel has orientation + CI "
                        "for every phase tested, allowing post-processing "
                        "phase reassignment."
                    )
                    phase_names_list = list(per_phase_data.keys())
                    for i, name in enumerate(phase_names_list):
                        pd = per_phase_data[name]
                        pg = per_phase_grp.create_group(name)
                        pg.attrs["phase_id"] = i + 1
                        pg.attrs["ci_mean"] = pd.get("ci_mean", 0.0)
                        if pd.get("euler") is not None:
                            ds = pg.create_dataset(
                                "euler_angles",
                                data=_euler_ndarray_to_vendor(
                                    pd["euler"], _export_vendor, r_user=_r_user),
                                dtype=np.float32, compression="gzip",
                            )
                            ds.attrs["unit"] = "radians"
                            ds.attrs["convention"] = "Bunge (ZXZ)"
                        if pd.get("ci") is not None:
                            pg.create_dataset(
                                "confidence_index", data=pd["ci"],
                                dtype=np.float32, compression="gzip",
                            )

                    aa_grp = idx.create_group("Assignment")
                    aa_grp.attrs["description"] = (
                        "Best phase per pixel (highest CI wins). 1-based "
                        "phase_id with 0 = unindexed."
                    )
                    if hasattr(active.xmap, "rotations"):
                        euler = _to_vendor_frame(
                            active.xmap.rotations, _export_vendor, r_user=_r_user
                        ).to_euler(degrees=False)
                        euler_arr = np.array(euler).reshape(
                            (*active.original_shape, 3)
                        ).astype(np.float32)
                        ds = aa_grp.create_dataset(
                            "euler_angles", data=euler_arr,
                            dtype=np.float32, compression="gzip",
                        )
                        ds.attrs["unit"] = "radians"
                        ds.attrs["convention"] = "Bunge (ZXZ)"
                    if hasattr(active.xmap, "phase_id"):
                        # Same 0-based → 1-based shift as the rich branch.
                        raw_pid = np.array(active.xmap.phase_id).reshape(
                            active.original_shape
                        )
                        written_pid = np.where(
                            raw_pid < 0, 0, raw_pid + 1
                        ).astype(np.uint8)
                        ds = aa_grp.create_dataset(
                            "phase_id", data=written_pid, compression="gzip",
                        )
                        ds.attrs["phase_names"] = phase_names_list
                    if active.confidence_scores is not None:
                        aa_grp.create_dataset(
                            "confidence_index",
                            data=active.confidence_scores.reshape(
                                active.original_shape
                            ).astype(np.float32),
                            dtype=np.float32, compression="gzip",
                        )
                else:
                    # Single-phase — datasets sit flat under /Indexing/
                    if hasattr(active.xmap, "rotations"):
                        euler = _to_vendor_frame(
                            active.xmap.rotations, _export_vendor, r_user=_r_user
                        ).to_euler(degrees=False)
                        ds = idx.create_dataset(
                            "euler_angles",
                            data=np.array(euler).reshape(
                                (*active.original_shape, 3)
                            ).astype(np.float32),
                            compression="gzip",
                        )
                        ds.attrs["unit"] = "radians"
                        ds.attrs["convention"] = "Bunge (ZXZ)"
                    if hasattr(active.xmap, "phase_id"):
                        raw_pid = np.array(active.xmap.phase_id).reshape(
                            active.original_shape
                        )
                        written_pid = np.where(
                            raw_pid < 0, 0, raw_pid + 1
                        ).astype(np.uint8)
                        idx.create_dataset(
                            "phase_id", data=written_pid, compression="gzip",
                        )
                    if active.confidence_scores is not None:
                        idx.create_dataset(
                            "confidence_index",
                            data=active.confidence_scores.reshape(
                                active.original_shape
                            ).astype(np.float32),
                            compression="gzip",
                        )

                idx.create_dataset(
                    "selection_mask",
                    data=active.selection_mask.astype(np.uint8),
                    compression="gzip",
                )

                # /Indexing/Phases/<1..N>/ — phase names + symmetry. The
                # Analysis loader (load_xmap_result) needs this; without
                # it phases fall back to "phase_1/2" + triclinic and the
                # IPF map becomes colour noise.
                try:
                    phases_tbl = idx.create_group("Phases")
                    xmap_phases = (
                        list(active.xmap.phases)
                        if active.xmap is not None
                        else []
                    )
                    i_counter = 0
                    for entry in xmap_phases:
                        if isinstance(entry, tuple) and len(entry) == 2:
                            pid, phase_obj = entry
                            try:
                                if int(pid) < 0:
                                    continue
                            except Exception:
                                pass
                        else:
                            phase_obj = entry
                            pid = getattr(entry, "id", None)
                            try:
                                if pid is not None and int(pid) < 0:
                                    continue
                            except Exception:
                                pass
                        i_counter += 1
                        pg = phases_tbl.create_group(str(i_counter))
                        name = getattr(phase_obj, "name", "") or f"phase_{i_counter}"
                        pg.attrs["name"] = str(name)
                        sg = getattr(phase_obj, "space_group", None)
                        if sg is not None:
                            try:
                                pg.attrs["space_group"] = int(
                                    getattr(sg, "number", sg)
                                )
                            except (TypeError, ValueError):
                                pass
                        pgrp = getattr(phase_obj, "point_group", None)
                        if pgrp is not None:
                            try:
                                pg.attrs["point_group"] = str(
                                    getattr(pgrp, "name", pgrp)
                                )
                            except Exception:
                                pass
                except Exception as e:
                    logger.warning(
                        "Light export: /Indexing/Phases write failed: %s", e
                    )

                # Copy per-pixel quality fields (BC, PC, bands) from the
                # source h5oina into /Indexing/Assignment/ — MTEX-style
                # readers expect them. Silently skip if source is gone or
                # in single-phase mode where there's no Assignment group
                # (Phase Map still works on the flat layout in that case).
                if src_path_str and Path(src_path_str).is_file():
                    target_grp = idx.get("Assignment")
                    if target_grp is not None:
                        try:
                            _copy_quality_fields(
                                src_path_str, target_grp, active.original_shape
                            )
                        except Exception as e:
                            logger.warning(
                                "Light export: quality-field copy failed: %s", e
                            )

                # /Detector — geometry only, no patterns.
                if req.include_detector:
                    try:
                        from backend.api.routes.ebsd_viewer import _get_active_signal, _active_dataset
                        from backend.api.services.calibration_store import calibration_store
                        det_grp = f.create_group("Detector")
                        # Mirror step size here too — same as the batch helper,
                        # so a reader that looks under /Detector also finds it.
                        det_grp.attrs["step_size_um"] = float(step_size_um)
                        signal = _get_active_signal()
                        if signal is not None:
                            det = getattr(signal, "detector", None)
                            if det is not None:
                                # Refined PC from the store (see rich export above).
                                store_pc = calibration_store.get_pc(_active_dataset)
                                pc_out = store_pc if store_pc is not None else np.array(det.pc).flatten()[:3]
                                det_grp.create_dataset(
                                    "pc", data=np.asarray(pc_out, dtype=float).flatten()[:3]
                                )
                                det_grp.attrs["sample_tilt"] = float(det.sample_tilt)
                                det_grp.attrs["shape"] = list(det.shape)
                    except Exception as e:
                        logger.warning(
                            "Light export: detector info failed: %s", e
                        )

                doc = f.create_group("Documentation")
                doc.attrs["format_version"] = _LIGHT_FMT_VERSION
                doc.attrs["description"] = (
                    "Lightweight EBSD indexing results — no source patterns "
                    "copied. Superblock v0 for MATLAB compatibility."
                )
                doc.create_dataset("README", data=_LIGHT_README)

        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported format '{fmt}'. "
                    "Use 'ang', 'h5' (rich), or 'h5_light' (compact)."
                ),
            )

        from fastapi.responses import FileResponse
        # The endpoint streams the file itself (no JSON body), so the
        # user-facing frame_warning rides along as a response header. Only
        # set on the .ang opt-in case; otherwise headers are unchanged.
        _resp_headers = (
            {"X-Frame-Warning": _frame_warning}
            if _frame_warning is not None
            else None
        )
        return FileResponse(
            path=str(out_path),
            filename=out_path.name,
            media_type="application/octet-stream",
            headers=_resp_headers,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Export failed")
        raise HTTPException(status_code=500, detail=str(e))


class PreprocessingPreviewRequest(BaseModel):
    row: int = 0
    col: int = 0
    gausbckg: bool = True
    circmask: int = -1   # -1 = no mask, 0 = auto-radius, >0 = explicit radius
    nregions: int = 10   # AHE grid divisions; 0 = skip AHE


def _to_base64_png(arr: "np.ndarray") -> str:
    """Convert a 2-D uint8 array to a base64-encoded PNG string."""
    import io
    import base64
    from PIL import Image

    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@router.post("/preview-preprocessing")
async def preview_preprocessing(req: PreprocessingPreviewRequest):
    """Preview EMSphinx preprocessing on a single EBSD pattern.

    Returns three base64 PNG images:
    - ``original``      — raw pattern normalised to 0-255
    - ``gaussian_bg``   — gaussian-background-subtracted version
    - ``circmask_ahe``  — circular-masked + adaptive histogram equalisation
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter

    from backend.api.routes.ebsd_viewer import _get_active_signal

    signal = _get_active_signal()
    if signal is None:
        raise HTTPException(status_code=400, detail="No EBSD data loaded")

    nav_shape = signal.axes_manager.navigation_shape
    # kikuchipy navigation_shape is (cols, rows)
    if len(nav_shape) >= 2:
        n_rows, n_cols = int(nav_shape[1]), int(nav_shape[0])
    else:
        n_rows, n_cols = int(nav_shape[0]), 1

    if not (0 <= req.row < n_rows and 0 <= req.col < n_cols):
        raise HTTPException(
            status_code=400,
            detail=f"Row/col ({req.row}, {req.col}) out of range ({n_rows}x{n_cols})",
        )

    pattern = np.array(signal.data[req.row, req.col], dtype=np.float32)

    # --- Image 1: original ---
    pmin, pmax = pattern.min(), pattern.max()
    span = max(float(pmax - pmin), 1e-7)
    original_uint8 = ((pattern - pmin) / span * 255).astype(np.uint8)

    # --- Image 2: gaussian background subtracted ---
    if req.gausbckg:
        sigma = min(pattern.shape) // 4
        bg = gaussian_filter(pattern, sigma=float(sigma))
        gauss = pattern - bg
    else:
        gauss = pattern.copy()

    gmin, gmax = gauss.min(), gauss.max()
    gspan = max(float(gmax - gmin), 1e-7)
    gauss_uint8 = ((gauss - gmin) / gspan * 255).astype(np.uint8)

    # --- Image 3: circular mask + AHE ---
    result = gauss.copy()

    if req.circmask != -1:
        h, w = result.shape
        cy, cx = h / 2.0, w / 2.0
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        radius = min(h, w) / 2.0 if req.circmask == 0 else float(req.circmask)
        result *= (dist <= radius).astype(result.dtype)

    if req.nregions > 0:
        from skimage.exposure import equalize_adapthist

        kernel_h = max(1, result.shape[0] // req.nregions)
        kernel_w = max(1, result.shape[1] // req.nregions)
        rmin, rmax = result.min(), result.max()
        rspan = max(float(rmax - rmin), 1e-7)
        result_norm = (result - rmin) / rspan
        result_eq = equalize_adapthist(
            result_norm, kernel_size=(kernel_h, kernel_w), clip_limit=0.03
        )
        circmask_ahe_uint8 = (result_eq * 255).astype(np.uint8)
    else:
        rmin, rmax = result.min(), result.max()
        rspan = max(float(rmax - rmin), 1e-7)
        circmask_ahe_uint8 = ((result - rmin) / rspan * 255).astype(np.uint8)

    return {
        "original": _to_base64_png(original_uint8),
        "gaussian_bg": _to_base64_png(gauss_uint8),
        "circmask_ahe": _to_base64_png(circmask_ahe_uint8),
    }


# ---------------------------------------------------------------------------
# Batch Indexing — multi-dataset sequential processing with memory management
# ---------------------------------------------------------------------------

_batch_state = {
    "running": False,
    "job_id": None,
    "total": 0,
    "completed": 0,
    "current_dataset": "",
    "results": [],  # [{dataset, method, n_indexed, mean_ci, export_path, error}]
    "log": [],
}

class BatchDatasetConfig(BaseModel):
    """Configuration for one dataset in a batch job."""
    file_path: str  # Path to H5OINA/H5 file
    method: str = "hough"  # "hough", "dictionary", "spherical"
    cif_paths: List[str] = []
    master_h5_paths: List[str] = []
    sht_paths: List[str] = []
    selection_mode: str = "full"  # "full", "region", "mask"
    row_start: int = 0
    row_end: int = -1
    col_start: int = 0
    col_end: int = -1
    # Per-dataset PC override (None = use auto-detected)
    pc: Optional[List[float]] = None  # [pcx, pcy, pcz]
    # Indexing params
    n_bands: int = 12
    metric: str = "ncc"
    keep_n: int = 20


class BatchRequest(BaseModel):
    """Batch indexing job across multiple datasets."""
    datasets: List[BatchDatasetConfig]
    auto_export: bool = True  # Auto-export each result as .h5
    export_dir: str = ""  # Directory for exports (empty = next to source file)
    cleanup_after_export: bool = True  # Unload signal after export to save RAM


@router.post("/batch/start")
async def start_batch_indexing(req: BatchRequest):
    """Start batch indexing across multiple EBSD files.

    For each dataset:
    1. Load the EBSD file
    2. Apply per-dataset PC if provided
    3. Run indexing (Hough/Dictionary/Spherical)
    4. Optionally export result as .h5
    5. Optionally unload signal to free memory
    """
    if _batch_state["running"]:
        raise HTTPException(status_code=409, detail="A batch job is already running")

    job_id = str(uuid.uuid4())
    _batch_state.update({
        "running": True,
        "job_id": job_id,
        "total": len(req.datasets),
        "completed": 0,
        "current_dataset": "",
        "results": [],
        "log": [],
    })

    # Cap _batch_state["log"] at MAX_BATCH_LOG_LINES so a long batch
    # (multi-file Hough run with verbose progress callbacks) doesn't
    # accumulate hundreds of MB of strings in memory + the JSON
    # response of /batch/status.
    MAX_BATCH_LOG_LINES = 500

    def _log(msg):
        log = _batch_state["log"]
        log.append(msg)
        if len(log) > MAX_BATCH_LOG_LINES:
            # Drop oldest, keep latest. Use slice-assignment to mutate in place.
            del log[: len(log) - MAX_BATCH_LOG_LINES]
        logger.info("Batch: %s", msg)

    def run_batch():
        global _active_result_id
        import gc

        _log(f"Starting batch job {job_id}: {len(req.datasets)} datasets")

        for i, ds_config in enumerate(req.datasets):
            dataset_name = Path(ds_config.file_path).stem
            _batch_state["current_dataset"] = dataset_name
            _batch_state["completed"] = i
            _log(f"[{i+1}/{len(req.datasets)}] Loading {dataset_name}...")

            result_entry = {
                "dataset": dataset_name,
                "file_path": ds_config.file_path,
                "method": ds_config.method,
                "n_indexed": 0,
                "mean_ci": None,
                "export_path": None,
                "error": None,
            }

            try:
                # 1. Load EBSD file
                from backend.api.routes.ebsd_viewer import load_ebsd_file
                load_result = load_ebsd_file(ds_config.file_path)
                if not load_result:
                    raise RuntimeError(f"Failed to load {ds_config.file_path}")

                # Also open H5 session for EDS
                try:
                    from backend.api.services.h5_session import open_file
                    open_file(ds_config.file_path)
                except Exception:
                    pass

                # 2. Get signal and apply PC override
                from backend.api.routes.ebsd_viewer import _get_active_signal
                signal = _get_active_signal()
                if signal is None:
                    raise RuntimeError("No signal after loading")

                nav_shape = signal.axes_manager.navigation_shape
                if len(nav_shape) >= 2:
                    n_rows, n_cols = nav_shape[1], nav_shape[0]
                else:
                    n_rows, n_cols = nav_shape[0], 1

                # Get detector: explicit config PC > CalibrationStore > signal.detector > default
                detector = calibration_store.get_detector(dataset_name)
                if detector is None:
                    detector = getattr(signal, 'detector', None)
                if detector is None:
                    from kikuchipy.detectors import EBSDDetector
                    sig_shape = signal.axes_manager.signal_shape
                    detector = EBSDDetector(shape=(sig_shape[1], sig_shape[0]))

                # Override PC if explicit config provides one
                if ds_config.pc and len(ds_config.pc) == 3:
                    pc_to_apply = ds_config.pc
                    _log(f"  Applying explicit PC: ({pc_to_apply[0]:.4f}, {pc_to_apply[1]:.4f}, {pc_to_apply[2]:.4f})")
                    detector.pc = np.array([pc_to_apply])

                # 3. Build selection mask
                from indexing_controller import (
                    IndexingMethod, PixelSelectionMode, IndexingConfig,
                    create_selection_mask, hough_index_patterns,
                    dictionary_index_patterns,
                )

                method_map = {"hough": IndexingMethod.HOUGH, "dictionary": IndexingMethod.DICTIONARY, "spherical": IndexingMethod.SPHERICAL}
                sel_map = {"full": PixelSelectionMode.FULL, "region": PixelSelectionMode.REGION, "mask": PixelSelectionMode.MASK}

                indexing_method = method_map.get(ds_config.method, IndexingMethod.HOUGH)
                selection_mode = sel_map.get(ds_config.selection_mode, PixelSelectionMode.FULL)

                region = None
                if selection_mode == PixelSelectionMode.REGION:
                    region = (ds_config.row_start, ds_config.row_end, ds_config.col_start, ds_config.col_end)

                selection_mask = create_selection_mask(
                    n_rows=n_rows, n_cols=n_cols,
                    mode=selection_mode, region=region,
                )

                config = IndexingConfig(
                    method=indexing_method,
                    selection_mode=selection_mode,
                    n_bands=ds_config.n_bands,
                    metric=ds_config.metric,
                    keep_n=ds_config.keep_n,
                )

                n_selected = int(selection_mask.sum())
                _log(f"  Indexing {n_selected}/{n_rows*n_cols} pixels with {ds_config.method}...")

                # 4. Run indexing
                result = None
                if indexing_method == IndexingMethod.HOUGH:
                    if not ds_config.cif_paths:
                        raise ValueError("Hough requires CIF files")
                    from orix.crystal_map import Phase, PhaseList
                    from ebsd_utils import sanitize_cif
                    phases = []
                    for cif in ds_config.cif_paths:
                        p = Phase.from_cif(sanitize_cif(cif))
                        p.name = Path(cif).stem
                        phases.append(p)
                    phase_list = PhaseList(phases)

                    result = hough_index_patterns(
                        signal=signal, phase_list=phase_list,
                        detector=detector, config=config,
                        selection_mask=selection_mask,
                        progress_callback=lambda msg, pct=None: _log(f"  {msg}"),
                    )

                elif indexing_method == IndexingMethod.DICTIONARY:
                    if not ds_config.master_h5_paths:
                        raise ValueError("Dictionary requires master H5 file")
                    import kikuchipy as kp
                    dictionary = kp.load(ds_config.master_h5_paths[0])
                    result = dictionary_index_patterns(
                        signal=signal, dictionary=dictionary,
                        config=config, selection_mask=selection_mask,
                        progress_callback=lambda msg, pct=None: _log(f"  {msg}"),
                        detector=detector,   # store-resolved (refined PC) — P2-B
                    )
                    del dictionary
                    gc.collect()

                elif indexing_method == IndexingMethod.SPHERICAL:
                    det_params = {
                        'pctr': list(detector.pc.flatten()[:3]),
                        'thetac': float(detector.sample_tilt),
                        'delta': float(getattr(detector, 'pixel_size', detector.shape[1] / 2)),
                        'numsx': int(detector.shape[1]),
                        'numsy': int(detector.shape[0]),
                    }
                    backend_choice = getattr(config, "backend", "emsphinx")
                    if backend_choice == "spherical_gpu":
                        from indexing_controller import spherical_gpu_index_patterns
                        result = spherical_gpu_index_patterns(
                            h5_path=ds_config.file_path,
                            config=config, detector_params=det_params,
                            selection_mask=selection_mask,
                            progress_callback=lambda msg, pct=None: _log(f"  {msg}"),
                        )
                    else:
                        from indexing_controller import spherical_index_patterns
                        result = spherical_index_patterns(
                            h5_path=ds_config.file_path,
                            config=config, detector_params=det_params,
                            selection_mask=selection_mask,
                            progress_callback=lambda msg, pct=None: _log(f"  {msg}"),
                        )

                if result is None:
                    raise RuntimeError("Indexing returned no result")

                # Attach metadata for the SHT forward renderer (Phase A).
                _attach_indexing_metadata(
                    result,
                    method_name=ds_config.method,
                    sht_paths=(
                        getattr(ds_config, "sht_paths", None)
                        if ds_config.method == "spherical" else None
                    ),
                    det_params=det_params,
                )

                _store_result(result, ds_config.method)
                n_indexed = int(result.selection_mask.sum())
                mean_ci = None
                if result.confidence_scores is not None:
                    valid = result.confidence_scores[~np.isnan(result.confidence_scores)]
                    if len(valid) > 0:
                        mean_ci = float(np.nanmean(valid))

                result_entry["n_indexed"] = n_indexed
                result_entry["mean_ci"] = mean_ci
                _log(f"  Indexed {n_indexed} pixels, mean CI={mean_ci:.4f}" if mean_ci else f"  Indexed {n_indexed} pixels")

                # 5. Auto-export
                if req.auto_export:
                    export_dir = req.export_dir or str(Path(ds_config.file_path).parent)
                    export_path = str(Path(export_dir) / f"{dataset_name}_indexed.h5")
                    try:
                        # Reuse the H5 export logic
                        import h5py
                        with h5py.File(export_path, "w") as f:
                            idx_grp = f.create_group("Indexing")
                            idx_grp.attrs["method"] = result.method.value
                            idx_grp.attrs["n_indexed"] = n_indexed
                            idx_grp.attrs["original_shape"] = list(result.original_shape)
                            if hasattr(result.xmap, "rotations"):
                                euler = result.xmap.rotations.to_euler(degrees=False)
                                idx_grp.create_dataset("euler_angles", data=np.array(euler))
                            if hasattr(result.xmap, "phase_id"):
                                idx_grp.create_dataset("phase_id", data=result.xmap.phase_id)
                            if result.confidence_scores is not None:
                                idx_grp.create_dataset("confidence", data=np.array(result.confidence_scores))
                            idx_grp.create_dataset("selection_mask", data=result.selection_mask.astype(np.uint8))
                            if hasattr(result.xmap, "x") and result.xmap.x is not None:
                                idx_grp.create_dataset("x", data=np.array(result.xmap.x))
                                idx_grp.create_dataset("y", data=np.array(result.xmap.y))
                            # EDS
                            try:
                                from backend.api.services.h5_session import get_extractor, is_open
                                if is_open():
                                    ext = get_extractor()
                                    elements = ext.get_available_elements()
                                    if elements:
                                        eds_grp = f.create_group("EDS")
                                        eds_grp.attrs["elements"] = elements
                                        for el_name in elements:
                                            data = ext.get_element_map(el_name)
                                            if data is not None:
                                                eds_grp.create_dataset(el_name.replace(" ", "_"), data=data, compression="gzip")
                            except Exception:
                                pass
                            # Detector
                            det_grp = f.create_group("Detector")
                            det_grp.attrs["shape"] = list(detector.shape)
                            # Refined PC from the store (mean for a per-pixel map) — same
                            # source as the interactive export, so both H5 export paths
                            # carry an identical /Detector/pc; fall back to the detector.
                            _bpc = calibration_store.get_pc(dataset_name)
                            det_grp.create_dataset("pc", data=np.asarray(
                                _bpc if _bpc is not None else detector.pc.flatten()[:3],
                                dtype=float).flatten()[:3])
                            det_grp.attrs["sample_tilt"] = float(detector.sample_tilt)
                            # Scan metadata
                            meta = f.create_group("Scan")
                            meta.attrs["grid_shape"] = [n_rows, n_cols]
                            meta.attrs["source_file"] = ds_config.file_path

                        result_entry["export_path"] = export_path
                        _log(f"  Exported to {export_path}")
                    except Exception as ex:
                        _log(f"  Export failed: {ex}")

                # 6. Memory cleanup
                if req.cleanup_after_export and req.auto_export:
                    try:
                        from backend.api.routes.ebsd_viewer import _raw_signals
                        for name in list(_raw_signals.keys()):
                            del _raw_signals[name]
                        gc.collect()
                        _log(f"  Freed memory for {dataset_name}")
                    except Exception:
                        pass

            except Exception as ex:
                import traceback
                result_entry["error"] = str(ex)
                _log(f"  ERROR: {ex}")
                logger.exception(f"Batch indexing failed for {dataset_name}")

            _batch_state["results"].append(result_entry)

        _batch_state["completed"] = len(req.datasets)
        _batch_state["current_dataset"] = ""
        _batch_state["running"] = False
        _log(f"Batch complete: {sum(1 for r in _batch_state['results'] if not r['error'])}/{len(req.datasets)} succeeded")

    def run_batch_safely():
        # Wrapper ensures _batch_state["running"] is always reset on exit,
        # even if run_batch raises an unhandled exception. Without this, a
        # crash inside run_batch (e.g. import failure of indexing_controller,
        # NameError, etc.) left _batch_state["running"]=True forever and
        # every subsequent /batch/start returned 409.
        try:
            run_batch()
        except Exception:
            logger.exception("Batch job %s crashed unexpectedly", job_id)
            _batch_state["results"].append({
                "dataset": "(batch runner)",
                "error": "Unhandled exception — see backend logs",
            })
        finally:
            _batch_state["running"] = False
            _batch_state["current_dataset"] = ""

    import asyncio
    asyncio.get_event_loop().run_in_executor(None, run_batch_safely)
    return {"job_id": job_id, "status": "started", "total": len(req.datasets)}


@router.get("/batch/status")
async def get_batch_status():
    """Get current batch job status."""
    return {
        "running": _batch_state["running"],
        "job_id": _batch_state["job_id"],
        "total": _batch_state["total"],
        "completed": _batch_state["completed"],
        "current_dataset": _batch_state["current_dataset"],
        "results": _batch_state["results"],
        "log": _batch_state["log"][-50:],  # Last 50 log entries
    }


@router.post("/batch/stop")
async def stop_batch():
    """Cancel the running batch job."""
    if not _batch_state["running"]:
        return {"status": "no_job_running"}
    _batch_state["running"] = False
    return {"status": "stopping"}


@router.post("/pc/store")
async def store_pc_for_dataset(dataset_name: str, pc: List[float]):
    """Store a refined PC for a specific dataset name.

    Used by the batch workflow to apply per-dataset PCs.
    Dataset must be loaded (i.e. registered in CalibrationStore).
    """
    if len(pc) != 3:
        raise HTTPException(status_code=400, detail="PC must have 3 values [pcx, pcy, pcz]")

    if not calibration_store.update_pc(dataset_name, pc, source="manual"):
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not loaded. Load it first.")

    return {"stored": True, "dataset": dataset_name, "pc": pc}


@router.get("/pc/stored")
async def get_stored_pcs():
    """Get all stored per-dataset PCs from CalibrationStore."""
    pcs = {}
    for name, entry in calibration_store.get_all().items():
        pcs[name] = list(float(v) for v in entry.pc_single)
    return {"pcs": pcs}
