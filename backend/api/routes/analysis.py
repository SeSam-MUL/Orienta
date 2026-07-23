"""
EBSD Analysis API Routes

Wraps the analysis/ module (MTEX-equivalent post-processing):
- EBSDDataset + GrainSet management
- Grain analysis (reconstruction, size, boundaries)
- Deformation analysis (KAM, grain boundaries)
- Texture analysis (ODF, components, pole figures)
- Recrystallization analysis
- Batch processing & Excel export
"""

import functools
import io
import logging
import threading
import uuid
from typing import Optional, List, Dict

import numpy as np

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.api.services.image_utils import array_to_base64_png, colormap_array_to_base64

logger = logging.getLogger(__name__)

# The heavy map/analysis endpoints below are declared `def` (not `async def`)
# so FastAPI runs them in its worker threadpool instead of on the event loop —
# they do CPU-bound KAM/GOS/texture math + matplotlib rendering with no awaits,
# and on the loop they froze every other request. matplotlib's pyplot is not
# thread-safe, so @_serialize_mpl serialises these handlers under one lock; the
# event loop still stays free (other requests proceed) while analysis ops take
# turns among themselves.
_mpl_lock = threading.Lock()


def _serialize_mpl(fn):
    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        with _mpl_lock:
            return fn(*args, **kwargs)
    return _wrapped
router = APIRouter()

_dataset = None  # EBSDDataset instance

# Per-file stash so analysis results survive a file round-trip (A -> B -> A).
# Before this, every file switch ran ``_dataset = None`` and grain
# reconstruction / per-pixel results were lost for good. We now keep them
# keyed by the EBSD file path they were built against, mirroring how the
# indexing module preserves its result registry across switches. EBSDDataset
# holds an xmap (orientations/scores per pixel), not the multi-GB pattern
# array, so stashing a handful of files costs a few MB each, not GB.
_dataset_by_file: dict = {}


def stash_dataset_for_file(file_path) -> None:
    """Save the current analysis ``_dataset`` under ``file_path`` so it can be
    restored when the user switches back. No-op when nothing is loaded or the
    path is empty."""
    if _dataset is not None and file_path:
        _dataset_by_file[file_path] = _dataset


def restore_dataset_for_file(file_path) -> bool:
    """Make the analysis dataset previously built for ``file_path`` active
    again (or ``None`` if there is none). Returns True if one was restored."""
    global _dataset
    _dataset = _dataset_by_file.get(file_path)
    return _dataset is not None

# Cap analysis-task tracking at MAX_TRACKED_TASKS so a long backend session
# that triggers many grain reconstructions / Excel exports doesn't leak.
from collections import OrderedDict as _OrderedDict
MAX_TRACKED_TASKS = 50
_analysis_tasks: _OrderedDict = _OrderedDict()


def _track_task(task_id: str, initial_state: dict) -> None:
    """Insert a new task while enforcing MAX_TRACKED_TASKS LRU cap."""
    _analysis_tasks[task_id] = initial_state
    while len(_analysis_tasks) > MAX_TRACKED_TASKS:
        _analysis_tasks.popitem(last=False)


class _RXExportData:
    """Minimal RX result container for Excel export."""
    __slots__ = ('rx_fraction', 'high_bc_fraction', 'low_gkam_fraction',
                 'low_kam_fraction', 'gos_histogram', 'gkam_histogram')

    def __init__(self, rx_fraction, high_bc_fraction, low_gkam_fraction,
                 low_kam_fraction, gos_histogram, gkam_histogram):
        self.rx_fraction = rx_fraction
        self.high_bc_fraction = high_bc_fraction
        self.low_gkam_fraction = low_gkam_fraction
        self.low_kam_fraction = low_kam_fraction
        self.gos_histogram = gos_histogram
        self.gkam_histogram = gkam_histogram


def get_analysis_dataset():
    """Accessor for the analysis dataset. Use this instead of importing _dataset directly."""
    return _dataset


def _load_kikuchipy_rich_h5(path: str):
    """Build a CrystalMap from our own .h5 export schema.

    Handles three export layouts produced by the GUI (see result_exporter.py
    and indexing.py /api/indexing/export):

    **Rich/Light multi-phase**  ``result_<stem>.h5`` / ``..._light.h5``:
       - ``/Indexing/Assignment/phase_id``      (uint8, 0=not indexed, 1..N=phase)
       - ``/Indexing/Assignment/euler_angles``  (float32, radians, Bunge)
       - ``/Indexing/Assignment/confidence_index``
       - ``/Indexing/Phases/<1..N>`` with attrs.name

    **Rich single-phase**       ``result_<stem>.h5``:
       - same fields, but written FLAT under ``/Indexing/`` (no Assignment
         sub-group) because indexing.py's single-phase branch skips the
         consensus group when there's only one phase. The writer/reader
         used to disagree on this layout — see the Assignment-or-flat
         fallback below.

    orix.io.load doesn't recognise any of these; this walks the h5 directly
    and returns a CrystalMap that Analysis / Phase Map can consume.
    """
    import h5py
    import numpy as np
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation

    with h5py.File(path, "r") as f:
        idx = f.get("Indexing")
        if idx is None:
            raise ValueError("file has no /Indexing group — not a GUI export")
        asg = idx.get("Assignment")
        if asg is None:
            # Single-phase rich-h5 layout: datasets sit flat under /Indexing/
            # rather than nested in /Indexing/Assignment/. Treat /Indexing/
            # itself as the assignment group — the dataset names are
            # identical, just one level up. Quality fields are absent in
            # this layout but the loop below handles that gracefully.
            if "phase_id" not in idx or "euler_angles" not in idx:
                raise ValueError(
                    "/Indexing exists but neither /Indexing/Assignment nor "
                    "flat phase_id+euler_angles datasets are present"
                )
            asg = idx

        phase_id = np.asarray(asg["phase_id"])  # 0 = not indexed, 1..N = phase
        euler = np.asarray(asg["euler_angles"])  # shape (rows, cols, 3) in radians
        ci = np.asarray(asg.get("confidence_index", np.zeros_like(phase_id, dtype=np.float32)))
        # Step size in µm — exporter (format_version >= 1.1) writes this
        # into /Indexing.attrs. Older files don't carry it; we set 0.0
        # here and let the caller's _fallback_step_size pick up the
        # active EBSD signal's axes manager. Keeping 0.0 (not 1.0) lets
        # the load_xmap detection logic distinguish "file says 1 µm"
        # from "file doesn't say".
        try:
            step_size_um = float(idx.attrs.get("step_size_um", 0.0))
        except (TypeError, ValueError):
            step_size_um = 0.0
        # Orientation export-frame tag (format_version >= 1.3). When present,
        # the stored Euler is in the vendor's frame (Aztec/MTEX) and must be
        # inverted back to our native (EMsoft/kikuchipy) frame so the working
        # map matches freshly-indexed results and the renderer. Absent on
        # older files, which were always written native.
        _src_vendor = idx.attrs.get("source_vendor", None)
        if isinstance(_src_vendor, bytes):
            _src_vendor = _src_vendor.decode("utf-8", errors="replace")
        # Optional per-pixel quality fields carried through from the source
        # h5oina by export_result_h5_light. Older light files don't have
        # them; in that case the prop key is simply absent.
        quality_keys = ("band_contrast", "mad", "pc_x", "pc_y", "dd", "bands")
        quality_data: dict = {
            k: np.asarray(asg[k]) for k in quality_keys if k in asg
        }

        grid_shape = phase_id.shape
        n_pixels = int(np.prod(grid_shape))

        # Per-phase CI maps — loaded for the "CI — <phase>" and uncertainty
        # views in Phase Map. Lives under /Indexing/PerPhase/<name>/ in the
        # h5 schema written by result_exporter.export_result_h5_light /
        # export_result_h5. Absent on .ang/.ctf (single-CI formats).
        per_phase_data: dict = {}
        per_phase_grp = idx.get("PerPhase")
        if per_phase_grp is not None:
            for pname in per_phase_grp:
                sub = per_phase_grp[pname]
                entry: dict = {}
                if "confidence_index" in sub:
                    entry["ci"] = np.array(sub["confidence_index"]).astype(np.float32)
                if "euler_angles" in sub:
                    pe = np.array(sub["euler_angles"]).astype(np.float32)
                    if _src_vendor and str(_src_vendor) not in ("", "unknown"):
                        from backend.api.services.orientation_frame import (
                            from_vendor_export_frame as _from_vendor,
                        )
                        _shp = pe.shape
                        pe = _from_vendor(
                            Rotation.from_euler(pe.reshape(-1, 3).astype(np.float64)),
                            str(_src_vendor),
                        ).to_euler().reshape(_shp).astype(np.float32)
                    entry["euler"] = pe
                if entry:
                    per_phase_data[pname] = entry

        # Phase list — pick up space_group / point_group attrs so the
        # rebuilt Phase objects carry the crystallographic symmetry. With
        # them missing, orix treats every phase as triclinic and IPF
        # coloring degenerates into noise, which was exactly what the
        # batch "Open in Phase Map" handoff produced until we started
        # writing these attrs on export.
        phases_grp = idx.get("Phases") or idx.get("phases")
        phase_names = []
        phase_sgs: list = []      # int or None
        phase_pgs: list = []      # str or None
        if phases_grp is not None:
            # Keys are "1", "2", ... — iterate in numeric order
            for key in sorted(phases_grp.keys(), key=lambda k: int(k) if k.isdigit() else 0):
                nm = phases_grp[key].attrs.get("name", f"phase_{key}")
                if isinstance(nm, bytes):
                    nm = nm.decode("utf-8", errors="replace")
                phase_names.append(str(nm))
                sg = phases_grp[key].attrs.get("space_group", None)
                pg = phases_grp[key].attrs.get("point_group", None)
                if isinstance(pg, bytes):
                    pg = pg.decode("utf-8", errors="replace")
                phase_sgs.append(int(sg) if sg is not None else None)
                phase_pgs.append(str(pg) if pg else None)
        if not phase_names:
            # Fall back: derive from max phase_id
            n = int(phase_id.max()) if phase_id.size else 0
            phase_names = [f"phase_{i+1}" for i in range(n)]
            phase_sgs = [None] * n
            phase_pgs = [None] * n

    # Build flat arrays
    phase_id_flat = phase_id.flatten().astype(int)
    euler_flat = euler.reshape(n_pixels, 3)
    ci_flat = ci.flatten().astype(np.float32)

    # Pixel coordinates → µm coordinates so xmap.step_sizes returns the
    # real value. Without this every downstream consumer (EBSDDataset,
    # Phase Map, PC Refinement) reconstructs an axes manager with
    # scale=1.0 even though the file knows the real step. When the file
    # didn't carry a step (older exports), use 1.0 unit-coordinates and
    # let the caller substitute from the active EBSD signal.
    scale = step_size_um if step_size_um > 0 else 1.0
    y_coords, x_coords = np.mgrid[0:grid_shape[0], 0:grid_shape[1]]
    x_flat = (x_coords.flatten() * scale).astype(float)
    y_flat = (y_coords.flatten() * scale).astype(float)

    # Phase list — Phase 0 represents "not indexed"; orix handles that via id -1.
    # Build each Phase with its persisted symmetry (space_group preferred,
    # point_group fallback) so IPF coloring has something to reduce against.
    built_phases = []
    for i, nm in enumerate(phase_names):
        sg = phase_sgs[i] if i < len(phase_sgs) else None
        pg = phase_pgs[i] if i < len(phase_pgs) else None
        ph = None
        if sg is not None:
            try: ph = Phase(name=nm, space_group=int(sg))
            except Exception: ph = None
        if ph is None and pg:
            try: ph = Phase(name=nm, point_group=str(pg))
            except Exception: ph = None
        if ph is None:
            ph = Phase(name=nm)
        built_phases.append(ph)
    phase_list = PhaseList(built_phases)

    # Remap: our 0 = unindexed → orix's phase_id convention usually uses 0..N-1.
    # Shift real phases by -1 so orix phase 0 = first real phase.
    # Unindexed pixels get -1 which orix treats as "not indexed".
    orix_phase_id = np.where(phase_id_flat == 0, -1, phase_id_flat - 1)

    rotations = Rotation.from_euler(euler_flat, degrees=False)
    # Invert the export-frame rotation so re-imported maps return to native.
    if _src_vendor and str(_src_vendor) not in ("", "unknown"):
        from backend.api.services.orientation_frame import from_vendor_export_frame
        rotations = from_vendor_export_frame(rotations, str(_src_vendor))
    # Map disk-key → xmap.prop key. 'band_contrast' on disk shows up as
    # 'bc' in prop because that's the convention the analysis routes use.
    DISK_TO_PROP = {
        "band_contrast": "bc",
        "mad": "mad",
        "pc_x": "pc_x",
        "pc_y": "pc_y",
        "dd": "dd",
        "bands": "bands",
    }
    prop_dict = {"ci": ci_flat}
    for disk_key, prop_key in DISK_TO_PROP.items():
        arr = quality_data.get(disk_key)
        if arr is not None and arr.size == n_pixels:
            prop_dict[prop_key] = arr.flatten().astype(np.float32)
    xmap = CrystalMap(
        rotations=rotations,
        phase_id=orix_phase_id,
        x=x_flat,
        y=y_flat,
        phase_list=phase_list,
        prop=prop_dict,
        scan_unit="um",
    )
    # Stash per-phase CI maps on the xmap so load_xmap can mirror them
    # onto the EBSDDataset. Attribute name matches the one indexing
    # already uses on IndexingResult.metadata['per_phase_data'].
    try:
        xmap._per_phase_data = per_phase_data or None
    except Exception:
        pass
    return xmap


class LoadXmapRequest(BaseModel):
    xmap_path: str  # Path to .h5 CrystalMap file


class GrainReconstructionRequest(BaseModel):
    misorientation_threshold: float = 5.0  # degrees
    min_grain_size: int = 5  # minimum pixels
    min_intercept_um: float = 1.5  # minimum grain intercept in µm


class ExportExcelRequest(BaseModel):
    output_path: str
    dataset_name: str = "Dataset"


class BatchProcessRequest(BaseModel):
    folder_path: str
    output_path: str


@router.post("/load")
async def load_xmap(req: LoadXmapRequest):
    """Load a CrystalMap into EBSDDataset."""
    global _dataset
    try:
        from analysis.ebsd_dataset import EBSDDataset

        # Helper: try to get step size from EBSD signal/file if xmap doesn't have it
        def _fallback_step_size():
            try:
                from backend.api.routes.ebsd_viewer import _get_active_signal, _extract_step_size
                sig = _get_active_signal()
                if sig is not None:
                    ss = _extract_step_size(sig)
                    if ss and ss.get('x', 1.0) != 1.0:
                        return float(ss['x'])
            except Exception:
                pass
            return 1.0

        # Special case: load from last indexing result
        if req.xmap_path == '__from_indexing__' or req.xmap_path == '__last_indexing__':
            from backend.api.routes.indexing import get_last_indexing_result
            _last_result = get_last_indexing_result()
            if _last_result is None:
                raise HTTPException(status_code=400, detail="No indexing result available")
            xmap = _last_result.xmap
            # NOTE: We deliberately do NOT synthesise a Band Contrast array from
            # the confidence/CI scores. Confidence is match reliability, not
            # image quality — the BC-calibrated GMM and quality-filter must fail
            # loud (see EBSDDataset.has_native_bc) rather than run on a fake BC.
            #
            # But we DO bridge the result's OWN source-file native Band Contrast
            # into xmap.prop['bc'] so has_native_bc / bc_gmm / quality-filter
            # work for Oxford H5OINA indexing results (same pattern the
            # phase-map BC layer uses). read_native_band_contrast returns None
            # for EDAX / synthetic / stripped sources — in which case
            # has_native_bc stays False and the BC-only analyses fail loud
            # *truthfully*.
            if 'bc' not in xmap.prop:
                try:
                    from backend.api.services.pattern_quality import read_native_band_contrast
                    src = None
                    meta = getattr(_last_result, "metadata", None)
                    if isinstance(meta, dict):
                        src = meta.get("source_file")
                    if src:
                        n_rows, n_cols = xmap.shape
                        native_bc = read_native_band_contrast(src, n_rows, n_cols)
                        if native_bc is not None:
                            bc_flat = native_bc.ravel()
                            if bc_flat.size == xmap.size:
                                xmap.prop['bc'] = bc_flat.astype('float32')
                except Exception:
                    logger.debug("native BC bridge for indexing result failed", exc_info=True)
            step_size = getattr(xmap, 'dx', None)
            if not step_size or step_size == 1.0:
                step_size = _fallback_step_size()
            _dataset = EBSDDataset(xmap, step_size=step_size, source="indexing_result")
        else:
            # Load from file.
            # Try orix first (native .ang/.ctf/.h5 with orix structure). If the
            # file is one of our custom exports (rich result_*.h5 with
            # /Indexing/, or light result_*_light.h5 with /SourceReference/),
            # orix's rsciio chain fails with a cryptic "NoneType has no
            # attribute 'file_reader'". Fall back to parsing our schema.
            from orix.io import load as orix_load
            try:
                xmap = orix_load(req.xmap_path)
            except Exception as orix_err:
                xmap = None
                try:
                    xmap = _load_kikuchipy_rich_h5(req.xmap_path)
                except Exception as our_err:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"orix.io.load failed: {orix_err}\n"
                            f"Custom .h5 fallback failed: {our_err}"
                        ),
                    )
            # orix CrystalMap exposes the µm step as ``dx``/``dy``, not
            # the legacy ``step_sizes`` array used by older orix builds.
            # When the file said nothing about step (older exports), dx
            # comes back as 1.0 from the unit-pixel coordinates we built;
            # _fallback_step_size pulls the real value off the active
            # signal in that case.
            step_size = getattr(xmap, 'dx', None)
            if not step_size or step_size == 1.0:
                step_size = _fallback_step_size()
            _dataset = EBSDDataset(xmap, step_size=step_size, source=req.xmap_path)
            # Forward per-phase CI maps extracted by our custom .h5 reader
            # onto the dataset so Phase Map's "CI — <phase>" and
            # "Uncertainty" modes work after a file load (orix .ang/.ctf
            # don't carry this; our .h5 formats do).
            ppd = getattr(xmap, '_per_phase_data', None)
            if ppd:
                try: _dataset.per_phase_data = ppd
                except Exception: pass

        # Build response carefully to avoid PhaseList len() issue
        resp = {"success": True}
        try:
            resp["shape"] = list(_dataset.shape)
        except Exception as ex:
            resp["shape_error"] = str(ex)
        try:
            phase_list = list(_dataset.xmap.phases)
            resp["n_phases"] = len(phase_list)
            formatted = []
            for p in phase_list:
                if isinstance(p, tuple) and len(p) == 2:
                    phase_obj = p[1]
                else:
                    phase_obj = p
                name = getattr(phase_obj, 'name', str(phase_obj))
                sg = getattr(phase_obj, 'space_group', None)
                sg_str = f" ({sg.short_name})" if sg and hasattr(sg, 'short_name') else ""
                formatted.append(f"{name}{sg_str}")
            resp["phases"] = formatted
        except Exception as ex:
            resp["phases_error"] = str(ex)
            resp["n_phases"] = 0
            resp["phases"] = []
        return resp
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to load xmap")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/status")
async def analysis_status():
    """Get current analysis dataset status."""
    if _dataset is None:
        return {"loaded": False}

    phases = []
    try:
        if hasattr(_dataset, 'xmap') and hasattr(_dataset.xmap, 'phases'):
            for p in _dataset.xmap.phases:
                # p may be a (id, Phase) tuple or a Phase object
                if isinstance(p, tuple) and len(p) == 2:
                    phase_obj = p[1]
                else:
                    phase_obj = p
                name = getattr(phase_obj, 'name', str(phase_obj))
                sg = getattr(phase_obj, 'space_group', None)
                sg_str = f" ({sg.short_name})" if sg and hasattr(sg, 'short_name') else ""
                phases.append(f"{name}{sg_str}")
    except Exception:
        pass

    step_size = getattr(_dataset, 'step_size', None)

    return {
        "loaded": True,
        "shape": list(_dataset.shape) if hasattr(_dataset, 'shape') else [],
        "has_grains": hasattr(_dataset, 'grains') and _dataset.grains is not None,
        "phases": phases,
        "step_size": float(step_size) if step_size is not None else None,
    }


class QualityFilterRequest(BaseModel):
    """Quality filter thresholds. Both default to 0 (disabled)."""
    bc_min: float = 0.0       # Minimum Band Contrast (0..255)
    bands_min: int = 0        # Minimum Hough bands per pixel


class PcSeedRequest(BaseModel):
    """Push the per-pixel PC into the calibration store for a dataset.

    target_dataset: which dataset name in the EBSD viewer's calibration
        store should receive this PC map. Defaults to the currently active
        EBSD viewer dataset.
    """
    target_dataset: Optional[str] = None


@router.post("/pc-drift-to-seed")
async def pc_drift_to_seed(req: PcSeedRequest):
    """Stack the analysis dataset's per-pixel PC (pc_x, pc_y, dd) into a
    (rows, cols, 3) map and write it to the calibration store, where it
    becomes the starting point for the next indexing / PC refinement run.

    Conventions: pc_x/pc_y/dd come straight from the source h5oina (Oxford
    fractional 0..1). The downstream consumer (PC Refinement controller +
    indexer) expects the same convention. No conversion is applied.

    The target dataset name defaults to the EBSD viewer's currently active
    one, so the typical workflow "load h5oina in EBSD viewer → load light
    h5 in Analysis → push PC seed" works without any extra plumbing.
    """
    if _dataset is None:
        raise HTTPException(status_code=400, detail="No analysis dataset loaded")

    keys = ("pc_x", "pc_y", "dd")
    arrays = []
    for k in keys:
        arr = _dataset.xmap.prop.get(k) if hasattr(_dataset, "xmap") else None
        if arr is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Field '{k}' missing — light h5 was exported before "
                    "quality-field carry-over. Re-export from checkpoint or "
                    "open the original h5oina."
                ),
            )
        arrays.append(np.asarray(arr).ravel().astype(np.float32))

    n_total = int(_dataset.size)
    if any(a.size != n_total for a in arrays):
        raise HTTPException(
            status_code=400,
            detail="Per-pixel PC arrays don't match dataset shape — corrupted file?",
        )

    pc_map = np.stack(arrays, axis=-1).reshape(*_dataset.shape, 3)

    target = req.target_dataset
    if not target:
        try:
            from backend.api.routes.ebsd_viewer import _active_dataset
            target = _active_dataset
        except Exception:
            target = None
    if not target:
        raise HTTPException(
            status_code=400,
            detail=(
                "No target dataset specified and no EBSD viewer dataset is "
                "active. Load the source h5oina in EBSD Viewer first, or "
                "pass target_dataset explicitly."
            ),
        )

    from backend.api.services.calibration_store import calibration_store
    ok = calibration_store.update_pc_map(target, pc_map)
    if not ok:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Dataset '{target}' is not in the calibration store. "
                "Open it in EBSD Viewer first to register it."
            ),
        )

    mean_pc = pc_map.reshape(-1, 3).mean(axis=0)
    return {
        "success": True,
        "target_dataset": target,
        "pc_map_shape": list(pc_map.shape),
        "mean_pc": [float(v) for v in mean_pc],
        "source": "aztec_per_pixel",
    }


@router.get("/aztec-comparison")
@_serialize_mpl
def aztec_comparison():
    """Compare Aztec's indexing against ours: where they disagree on phase,
    and (where they agree) by how much the orientations differ.

    Reads Aztec's per-pixel Phase + Euler from the source h5oina (path
    stored in our light h5's /SourceReference) and confronts them with the
    in-memory analysis dataset's phase_id + euler.

    Returns a 3-panel PNG (Aztec phase | Our phase | Misorientation deg)
    plus stats: agreement %, mean / 95p misorientation on matching pixels.

    400 if no analysis dataset, no source pointer, or the source file is
    not reachable on disk.
    """
    if _dataset is None:
        raise HTTPException(status_code=400, detail="No analysis dataset loaded")

    import base64
    import io
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Locate source h5oina via the SourceReference written at light-h5 export
    src_path = getattr(_dataset, "source", None)
    src_h5oina = None
    try:
        if src_path and Path(src_path).exists():
            with h5py.File(src_path, "r") as f:
                if "SourceReference" in f:
                    p = f["SourceReference"].attrs.get("source_file_path", "")
                    if isinstance(p, bytes):
                        p = p.decode("utf-8", errors="replace")
                    if p and Path(p).exists():
                        src_h5oina = p
                # Also try: same dir, same stem, .h5oina
                if src_h5oina is None:
                    cand = Path(src_path).parent / (Path(src_path).stem.replace("_light", "").replace("result_", "") + ".h5oina")
                    if cand.exists():
                        src_h5oina = str(cand)
    except Exception as e:
        logger.debug("Source pointer lookup failed: %s", e)

    if not src_h5oina:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not locate the source h5oina (SourceReference missing or "
                "the file moved). Place the original .h5oina next to the light "
                "h5 with matching stem and try again."
            ),
        )

    # Load Aztec phase + Euler
    grid = _dataset.shape
    n_total = int(np.prod(grid))
    try:
        with h5py.File(src_h5oina, "r") as f:
            aztec_phase = np.asarray(f["/1/EBSD/Data/Phase"]).reshape(grid)
            aztec_euler = np.asarray(f["/1/EBSD/Data/Euler"]).reshape(*grid, 3)
            # Build name→id map for Aztec
            aztec_phase_names: dict = {}
            phases_grp = f.get("/1/EBSD/Header/Phases")
            if phases_grp is not None:
                for key in phases_grp:
                    nm = phases_grp[key].get("Phase Name")
                    if nm is None:
                        continue
                    name = nm[()]
                    if hasattr(name, "tolist"):
                        name = name.tolist()
                    if isinstance(name, list) and name:
                        name = name[0]
                    if isinstance(name, bytes):
                        name = name.decode("utf-8", errors="replace")
                    aztec_phase_names[int(key)] = str(name).strip()
    except (KeyError, OSError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read Aztec Phase/Euler from {src_h5oina}: {e}",
        )

    # Build Aztec → Ours phase mapping by NAME (case-insensitive substring match
    # so Aztec's "Aluminium" lines up with our "Al"). Pixels with phase_id 0 in
    # Aztec are unindexed and stay -1 in our convention.
    our_phase_id_2d = np.asarray(_dataset.xmap.phase_id).reshape(grid)
    our_phase_names = {}
    try:
        for entry in _dataset.xmap.phases:
            if isinstance(entry, tuple) and len(entry) == 2:
                pid, ph = entry
            else:
                ph = entry
                pid = getattr(ph, "id", -1)
            nm = getattr(ph, "name", "") or ""
            if int(pid) >= 0 and nm and nm.lower() != "not_indexed":
                our_phase_names[int(pid)] = nm.strip()
    except Exception as e:
        logger.debug("Could not enumerate our phases: %s", e)

    def _matches(a: str, b: str) -> bool:
        a, b = a.lower(), b.lower()
        return a == b or a.startswith(b) or b.startswith(a)

    aztec_to_ours: dict = {}
    for az_id, az_name in aztec_phase_names.items():
        for our_id, our_name in our_phase_names.items():
            if _matches(az_name, our_name):
                aztec_to_ours[az_id] = our_id
                break

    # Re-map Aztec phase_id to our phase_id space; unmapped → -1 (no comparison)
    aztec_remapped = np.full(grid, -1, dtype=np.int16)
    for az_id, our_id in aztec_to_ours.items():
        aztec_remapped[aztec_phase == az_id] = our_id

    both_indexed = (aztec_remapped >= 0) & (our_phase_id_2d >= 0)
    agree_phase = both_indexed & (aztec_remapped == our_phase_id_2d)
    disagree_phase = both_indexed & (aztec_remapped != our_phase_id_2d)

    n_both = int(both_indexed.sum())
    n_agree = int(agree_phase.sum())
    n_disagree = int(disagree_phase.sum())
    agree_pct = (n_agree / max(n_both, 1)) * 100

    # Misorientation on agreeing pixels — compute symmetry-aware angle
    # between Aztec Euler and our Euler. For multi-phase scans we use each
    # pixel's phase symmetry; common case (single phase) is fast.
    misori_deg = np.full(grid, np.nan, dtype=np.float32)
    try:
        from orix.quaternion import Rotation, Misorientation
        if n_agree > 0:
            our_euler_2d = np.asarray(_dataset.xmap.rotations.to_euler()).reshape(*grid, 3)
            r_aztec = Rotation.from_euler(aztec_euler[agree_phase])
            r_ours = Rotation.from_euler(our_euler_2d[agree_phase])
            # Misorientation = ours^-1 * aztec — symmetry handled per-phase
            # below, but for the cubic-only case Rotation.angle_with works
            mis = (~r_ours) * r_aztec
            angles = np.degrees(mis.angle.data) if hasattr(mis.angle, "data") else np.degrees(np.asarray(mis.angle))
            # Apply symmetry: for cubic m-3m the smallest equivalent angle
            # is obtained via symmetry reduction. Use orix Misorientation
            # class which knows about Phase symmetries.
            try:
                first_phase = next(iter(_dataset.xmap.phases))
                if isinstance(first_phase, tuple):
                    first_phase = first_phase[1]
                pg = getattr(first_phase, "point_group", None)
                if pg is not None:
                    mo = Misorientation(mis, symmetry=(pg, pg))
                    mo = mo.map_into_symmetry_reduced_zone()
                    angles = np.degrees(np.asarray(mo.angle.data if hasattr(mo.angle, "data") else mo.angle))
            except Exception as sym_err:
                logger.debug("Symmetry reduction failed (using raw angle): %s", sym_err)
            misori_deg[agree_phase] = angles.astype(np.float32)
    except Exception as e:
        logger.warning("Misorientation calc failed: %s", e)

    finite = misori_deg[np.isfinite(misori_deg)]
    misori_stats = {
        "mean": float(finite.mean()) if finite.size else 0.0,
        "median": float(np.median(finite)) if finite.size else 0.0,
        "p95": float(np.percentile(finite, 95)) if finite.size else 0.0,
        "max": float(finite.max()) if finite.size else 0.0,
    }

    # Render: 3-panel side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=120)

    # Same Dracula palette as Phase Map for visual consistency
    DRACULA = np.array([
        [0.373, 0.898, 0.482], [0.510, 0.667, 1.000], [0.969, 0.549, 0.424],
        [0.784, 0.576, 0.918], [1.000, 0.796, 0.420], [0.537, 0.867, 0.976],
        [1.000, 0.325, 0.439], [0.765, 0.910, 0.553],
    ])
    UNINDEXED = np.array([0.267, 0.278, 0.353])

    def _phase_to_rgb(phase_2d, name_lookup):
        rgb = np.full((*phase_2d.shape, 3), UNINDEXED, dtype=np.float32)
        for pid, name in name_lookup.items():
            if pid < 0:
                continue
            # Same hashing as backend.api.routes.phase_map.stable_color_idx
            h = 0
            for c in (name or ""):
                h = ((h << 5) - h + ord(c)) & 0xFFFFFFFF
                if h >= 0x80000000:
                    h -= 0x100000000
            slot = abs(h) % len(DRACULA)
            mask = (phase_2d == pid) if isinstance(pid, int) else (phase_2d == int(pid))
            rgb[mask] = DRACULA[slot]
        return rgb

    # Panel 1: Aztec phase
    aztec_rgb = _phase_to_rgb(aztec_phase, aztec_phase_names)
    axes[0].imshow(aztec_rgb)
    axes[0].set_title(f"Aztec Phase\n{len(aztec_phase_names)} phase(s)", color='white', fontsize=11)
    axes[0].axis('off')

    # Panel 2: Our phase (same colors via name-hash so visually comparable)
    ours_rgb = _phase_to_rgb(our_phase_id_2d, our_phase_names)
    axes[1].imshow(ours_rgb)
    axes[1].set_title(f"Ours Phase\n{len(our_phase_names)} phase(s)", color='white', fontsize=11)
    axes[1].axis('off')

    # Panel 3: misorientation heatmap (only on agreeing pixels)
    cmap = plt.get_cmap('plasma').copy()
    cmap.set_bad('#1a1b26')
    if finite.size > 0:
        vmax = float(np.percentile(finite, 98))
        if vmax <= 0:
            vmax = 1.0
    else:
        vmax = 1.0
    im = axes[2].imshow(misori_deg, cmap=cmap, vmin=0, vmax=vmax)
    axes[2].set_title(
        f"Misorientation (deg)\nagree {agree_pct:.1f}%, disagree {n_disagree} px",
        color='white', fontsize=11,
    )
    axes[2].axis('off')
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='white', labelsize=8)

    fig.patch.set_facecolor('#1a1b26')
    for ax in axes:
        ax.set_facecolor('#1a1b26')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)

    return {
        "image": base64.b64encode(buf.read()).decode('utf-8'),
        "shape": list(grid),
        "n_pixels_both_indexed": n_both,
        "n_agree_phase": n_agree,
        "n_disagree_phase": n_disagree,
        "agree_pct": agree_pct,
        "misorientation_deg": misori_stats,
        "phase_mapping": {
            str(az_id): {"aztec_name": az_name, "our_id": aztec_to_ours.get(az_id, None)}
            for az_id, az_name in aztec_phase_names.items()
        },
        "source_h5oina": src_h5oina,
    }


@router.get("/pc-drift-image")
async def pc_drift_image():
    """Render the per-pixel Pattern Center drift as a 3-panel PNG.

    Reads pc_x / pc_y / dd from xmap.prop (carried in from the source h5oina)
    and renders them side-by-side with auto-stretched colormaps. Returns the
    image as base64 plus per-channel summary stats so the UI can show
    "ΔPCx = 1.2%" etc. next to the figure.

    Returns 400 with a clear message when the dataset wasn't loaded with
    quality fields (older light h5 without the carry-over).
    """
    if _dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")

    import base64
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    keys = ("pc_x", "pc_y", "dd")
    arrays = {}
    for k in keys:
        arr = _dataset.xmap.prop.get(k) if hasattr(_dataset, "xmap") else None
        if arr is None:
            continue
        arr_np = np.asarray(arr).ravel().astype(np.float32)
        if arr_np.size != _dataset.size:
            continue
        arrays[k] = arr_np.reshape(_dataset.shape)

    if not arrays:
        raise HTTPException(
            status_code=400,
            detail=(
                "No per-pixel PC data on this dataset. The light h5 was "
                "exported before quality-field carry-over — re-export from "
                "the checkpoint or open the original h5oina."
            ),
        )

    n = len(arrays)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), dpi=120)
    if n == 1:
        axes = [axes]

    titles = {"pc_x": "PC X", "pc_y": "PC Y", "dd": "Detector Distance"}
    stats: dict = {}
    for ax, (key, data) in zip(axes, arrays.items()):
        finite = data[np.isfinite(data) & (data != 0)]  # drop zeros (unindexed)
        if finite.size > 0:
            vmin = float(np.percentile(finite, 2))
            vmax = float(np.percentile(finite, 98))
            if vmax <= vmin:
                vmin, vmax = float(finite.min()), float(finite.max() or 1.0)
            mean = float(finite.mean())
            std = float(finite.std())
            rng = float(finite.max() - finite.min())
        else:
            vmin, vmax, mean, std, rng = 0.0, 1.0, 0.0, 0.0, 0.0

        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad('#1a1b26')
        plot_data = np.where(data == 0, np.nan, data)  # zeros render as bg
        im = ax.imshow(plot_data, cmap=cmap, vmin=vmin, vmax=vmax)
        # Drift as percentage of the mean (a relative "how much variation")
        drift_pct = (rng / mean * 100) if mean else 0.0
        ax.set_title(
            f"{titles[key]}\nrange={rng:.4f}  Δ={drift_pct:.1f}%",
            color='white', fontsize=11,
        )
        ax.axis('off')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors='white', labelsize=8)

        stats[key] = {
            "min": float(finite.min()) if finite.size else 0.0,
            "max": float(finite.max()) if finite.size else 0.0,
            "mean": mean,
            "std": std,
            "range": rng,
            "drift_pct": drift_pct,
        }

    fig.patch.set_facecolor('#1a1b26')
    for ax in axes:
        ax.set_facecolor('#1a1b26')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)

    return {
        "image": base64.b64encode(buf.read()).decode('utf-8'),
        "shape": list(_dataset.shape),
        "stats": stats,
    }


@router.get("/quality-stats")
async def quality_stats():
    """Return distribution stats for BC and Bands so the UI can pick smart
    threshold defaults (5th/50th/95th percentile, range, mean).

    Returns ``{"bc": null, "bands": null}`` for fields the dataset doesn't
    carry (older light h5 without quality-field copy).
    """
    if _dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    out = {"bc": None, "bands": None}
    for key in ("bc", "bands"):
        arr = _dataset.xmap.prop.get(key) if hasattr(_dataset, "xmap") else None
        if arr is None:
            continue
        flat = np.asarray(arr).ravel()
        if flat.size == 0:
            continue
        out[key] = {
            "min": float(flat.min()),
            "max": float(flat.max()),
            "mean": float(flat.mean()),
            "p05": float(np.percentile(flat, 5)),
            "p50": float(np.percentile(flat, 50)),
            "p95": float(np.percentile(flat, 95)),
        }
    return out


@router.post("/quality-filter")
async def apply_quality_filter(req: QualityFilterRequest):
    """Mark low-quality pixels as unindexed for downstream statistics.

    Mutates the in-memory dataset destructively. Reload to reset. Returns
    counts of filtered pixels by reason so the UI can show "X removed by BC,
    Y by Bands, Z total (W%)" feedback.
    """
    if _dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    if not _dataset.has_native_bc:
        raise HTTPException(
            status_code=400,
            detail=("Band Contrast quality filter requires native Band Contrast (Oxford H5OINA). "
                    "This dataset has only computed Pattern Quality."),
        )
    try:
        stats = _dataset.apply_quality_filter(
            bc_min=req.bc_min,
            bands_min=req.bands_min,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"success": True, "filters": req.model_dump(), **stats}


@router.post("/grains/reconstruct")
async def reconstruct_grains(req: GrainReconstructionRequest, background_tasks: BackgroundTasks):
    """Reconstruct grains from indexed data."""
    if _dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")

    task_id = str(uuid.uuid4())
    _track_task(task_id, {"status": "running", "progress": 0.0})

    def run():
        try:
            from analysis.grain_analysis import reconstruct_grains
            grain_set = reconstruct_grains(
                _dataset,
                angle_threshold_deg=req.misorientation_threshold,
                min_pixels=req.min_grain_size,
            )
            _dataset.grains = grain_set
            n_grains = grain_set.n_grains if grain_set else 0
            result = {"n_grains": n_grains, "grain_count": n_grains}
            # Enrich with grain-size stats so the UI's completion handler shows
            # real ECD / aspect-ratio (it reads mean_ecd, median_ecd, std_ecd,
            # mean_aspect_ratio) instead of "undefined". Best-effort: if it
            # fails, reconstruction still succeeds and /grain-size can be called.
            if n_grains > 0:
                try:
                    import numpy as _np
                    from analysis.grain_analysis import analyze_grain_size
                    gs = analyze_grain_size(_dataset, grain_set)
                    if hasattr(gs, "avg_ECD"):
                        result["mean_ecd"] = float(gs.avg_ECD)
                        result["std_ecd"] = float(gs.std_ECD)
                    if hasattr(gs, "avg_AR"):
                        result["mean_aspect_ratio"] = float(gs.avg_AR)
                    epg = getattr(gs, "ecd_per_grain", None)
                    if epg is not None and len(epg):
                        result["median_ecd"] = float(_np.median(epg))
                except Exception:
                    logger.warning("grain-size summary after reconstruct failed", exc_info=True)
            _analysis_tasks[task_id]["result"] = result
            _analysis_tasks[task_id]["grain_count"] = n_grains
            # Set progress=1.0 BEFORE status=completed so polling clients that
            # check progress first don't see "completed at 0%" (which the
            # React spinner rendered as stuck indefinitely).
            _analysis_tasks[task_id]["progress"] = 1.0
            _analysis_tasks[task_id]["status"] = "completed"
        except Exception as e:
            _analysis_tasks[task_id]["progress"] = 1.0
            _analysis_tasks[task_id]["status"] = "failed"
            _analysis_tasks[task_id]["error"] = str(e)

    background_tasks.add_task(run)
    return {"task_id": task_id}


@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in _analysis_tasks:
        raise HTTPException(status_code=404)
    return _analysis_tasks[task_id]


@router.get("/map/{map_type}")
@_serialize_mpl
def get_analysis_map(map_type: str, cmap: str = "viridis"):
    """Get an analysis map (BC, IPF, KAM, GOS, ECD, etc.)."""
    if _dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")

    # Validate dataset has a 2D shape (not a scalar from single-pixel indexing)
    ds_shape = _dataset.shape
    if not ds_shape or len(ds_shape) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset has shape {ds_shape} — need a 2D map (run full indexing, not single-pixel Quick Test)"
        )

    # Normalize: accept hyphens or underscores (ipf-z → ipf_z)
    map_type = map_type.replace("-", "_").lower()

    try:
        import numpy as np
        extra_fields = {}  # Additional data to include in response (e.g. RX stats)

        if map_type == "bc":
            # Band Contrast map from xmap
            if hasattr(_dataset.xmap, 'prop') and 'bc' in _dataset.xmap.prop:
                data = _dataset.xmap.prop['bc'].reshape(_dataset.shape)
            else:
                # Fallback: use Band Contrast from xmap properties
                data = np.zeros(_dataset.shape)
        elif map_type in ("ipf", "ipf_x", "ipf_y", "ipf_z"):
            from tools.phase_map_generator import compute_ipf_colors
            direction = map_type.split("_")[1].upper() if "_" in map_type else "Z"
            colors = compute_ipf_colors(_dataset.xmap, direction_str=direction)
            import io, base64
            from PIL import Image
            n_rows, n_cols = _dataset.shape
            rgb = (colors.reshape(n_rows, n_cols, 3) * 255).astype(np.uint8)
            img = Image.fromarray(rgb)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            return {
                "image": base64.b64encode(buf.read()).decode('utf-8'),
                "map_type": map_type,
                "shape": [n_rows, n_cols],
            }
        elif map_type == "kam":
            from analysis.deformation_analysis import calculate_kam
            data = calculate_kam(_dataset)
        elif map_type == "gos":
            if _dataset.grains is None:
                raise HTTPException(status_code=400, detail="Reconstruct grains first for GOS map")
            # Map per-grain GOS values back to pixel grid
            gos_per_grain = _dataset.grains.gos  # radians, per grain
            grain_ids = _dataset.grains.grain_ids
            data = np.zeros(_dataset.shape, dtype=float)
            if gos_per_grain is not None:
                for i, gos_val in enumerate(gos_per_grain):
                    data[grain_ids == (i + 1)] = np.degrees(gos_val)
        elif map_type == "ecd":
            if _dataset.grains is None:
                raise HTTPException(status_code=400, detail="Reconstruct grains first for ECD map")
            ecd_per_grain = _dataset.grains.ecd  # µm, per grain
            grain_ids = _dataset.grains.grain_ids
            data = np.zeros(_dataset.shape, dtype=float)
            if ecd_per_grain is not None:
                for i, ecd_val in enumerate(ecd_per_grain):
                    data[grain_ids == (i + 1)] = ecd_val
        elif map_type == "rx":
            if _dataset.grains is None:
                raise HTTPException(status_code=400, detail="Reconstruct grains first for RX map")
            from analysis.rx_analysis import classify_rx
            data = classify_rx(_dataset, _dataset.grains)
            # Compute RX statistics from the pixel map (0=RX, 1=Sub, 2=Def)
            try:
                total_px = data.size
                rx_px = int(np.sum(data == 0))
                extra_fields["rx_fraction"] = float(rx_px / max(total_px, 1) * 100)
                extra_fields["total_grains"] = int(_dataset.grains.n_grains)
                # Count unique grain IDs in RX pixels
                grain_ids = _dataset.grains.grain_ids
                rx_grain_ids = np.unique(grain_ids[data == 0])
                rx_grain_ids = rx_grain_ids[rx_grain_ids > 0]  # exclude background
                extra_fields["rx_grain_count"] = int(len(rx_grain_ids))
                # BC-based fractions from xmap properties
                if hasattr(_dataset.xmap, 'prop') and 'bc' in _dataset.xmap.prop:
                    bc = _dataset.xmap.prop['bc']
                    bc_max = float(np.nanmax(bc)) if bc.size > 0 else 1.0
                    extra_fields["high_bc_fraction"] = float(np.sum(bc > 0.7 * bc_max) / max(bc.size, 1) * 100)
                else:
                    extra_fields["high_bc_fraction"] = 0.0
                extra_fields["low_gkam_fraction"] = 0.0
                extra_fields["low_kam_fraction"] = 0.0
            except Exception as rx_err:
                logger.warning("Could not compute RX statistics: %s", rx_err)
        elif map_type == "texture_component":
            data = np.zeros(_dataset.shape, dtype=float)
        elif map_type == "grain_boundaries":
            if _dataset.grains is None:
                raise HTTPException(status_code=400, detail="Reconstruct grains first for grain boundary map")
            # Grain boundary map: 1 at boundaries, 0 elsewhere
            grain_ids = _dataset.grains.grain_ids
            gb = np.zeros(_dataset.shape, dtype=float)
            gb[:-1, :] += (grain_ids[:-1, :] != grain_ids[1:, :]).astype(float)
            gb[:, :-1] += (grain_ids[:, :-1] != grain_ids[:, 1:]).astype(float)
            data = np.clip(gb, 0, 1)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown map type: {map_type}")

        response = {
            "image": colormap_array_to_base64(data, cmap),
            "map_type": map_type,
            "shape": list(data.shape),
            "min_val": float(np.nanmin(data)),
            "max_val": float(np.nanmax(data)),
        }
        response.update(extra_fields)
        return response
    except HTTPException:
        # Preserve the original status/detail. Without this, inner 400 errors
        # like "Reconstruct grains first for ECD map" were caught by the
        # broad except-Exception below and re-raised as 500 with the message
        # "400: Reconstruct grains first..." — confusing both HTTP-status-
        # aware clients and humans.
        raise
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Module not available: {e}")
    except Exception as e:
        logger.exception("Failed to compute map")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/grain-size")
@_serialize_mpl
def analyze_grain_size():
    """Run grain size analysis."""
    if _dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    try:
        from analysis.grain_analysis import analyze_grain_size
        if _dataset.grains is None:
            raise HTTPException(status_code=400, detail="Reconstruct grains first")
        results = analyze_grain_size(_dataset, _dataset.grains)
        # Serialize GrainSizeResult
        response = {"success": True}
        if hasattr(results, 'avg_ECD'):
            response["mean_ecd"] = float(results.avg_ECD)
            response["std_ecd"] = float(results.std_ECD)
        if hasattr(results, 'avg_maxDimX'):
            response["mean_maxDimX"] = float(results.avg_maxDimX)
            response["mean_maxDimY"] = float(results.avg_maxDimY)
        if hasattr(results, 'avg_AR'):
            response["mean_aspect_ratio"] = float(results.avg_AR)
        # Generate histogram image if available
        if hasattr(results, 'ecd_per_grain') and results.ecd_per_grain is not None:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import io, base64
            fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
            ax.hist(results.ecd_per_grain, bins=30, color='#8be9fd', edgecolor='#44475a')
            ax.set_xlabel('ECD (µm)')
            ax.set_ylabel('Count')
            ax.set_title('Grain Size Distribution')
            fig.set_facecolor('#282a36')
            ax.set_facecolor('#1e1f29')
            ax.tick_params(colors='#f8f8f2')
            ax.xaxis.label.set_color('#f8f8f2')
            ax.yaxis.label.set_color('#f8f8f2')
            ax.title.set_color('#f8f8f2')
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            buf.seek(0)
            response["histogram_image"] = base64.b64encode(buf.read()).decode('utf-8')
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deformation")
@_serialize_mpl
def analyze_deformation():
    """Run deformation analysis (KAM, GB classification)."""
    if _dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    try:
        from analysis.deformation_analysis import analyze_deformation
        if _dataset.grains is None:
            raise HTTPException(status_code=400, detail="Reconstruct grains first")
        results = analyze_deformation(_dataset, _dataset.grains)
        # Serialize DeformationResult to JSON-compatible dict
        response = {"success": True}
        if hasattr(results, 'kam') and results.kam is not None:
            response["kam_image"] = colormap_array_to_base64(results.kam, "hot")
            import numpy as np
            valid_kam = results.kam[~np.isnan(results.kam)] if results.kam.size > 0 else results.kam
            if valid_kam.size > 0:
                response["mean_kam"] = float(np.nanmean(valid_kam))
        if hasattr(results, 'total_gb_length'):
            response["hagb_length_um"] = float(results.hagb_length)
            response["sagb_length_um"] = float(results.sagb_length)
            response["gb_classification"] = {
                "total_gb_length": float(results.total_gb_length),
                "hagb": float(results.hagb_length / max(results.total_gb_length, 0.001)),
                "sagb": float(results.sagb_length / max(results.total_gb_length, 0.001)),
            }
        if hasattr(results, 'sphericity') and results.sphericity is not None:
            import numpy as np
            sph = results.sphericity
            if hasattr(sph, '__len__') and len(sph) > 0:
                response["sphericity"] = float(np.nanmean(sph))
            elif not hasattr(sph, '__len__'):
                response["sphericity"] = float(sph)
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bc_gmm")
@_serialize_mpl
def fit_bc_gmm():
    """Fit 3-Gaussian Mixture Model to Band Contrast distribution."""
    if _dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    if not _dataset.has_native_bc:
        raise HTTPException(
            status_code=400,
            detail=("Band Contrast GMM requires native Band Contrast (Oxford H5OINA). "
                    "This dataset has only computed Pattern Quality."),
        )

    try:
        from analysis.bc_analysis import fit_bc_gmm as _fit_gmm, calculate_bc_histogram
        import numpy as np

        # Calculate histogram
        hist_result = calculate_bc_histogram(_dataset)

        # Fit GMM
        gmm_result = _fit_gmm(_dataset)
        if gmm_result is None:
            raise HTTPException(status_code=400, detail="GMM fit failed — not enough data")

        return {
            "converged": gmm_result.converged,
            "n_samples": gmm_result.n_samples,
            "components": {
                "low_bc": {
                    "center": float(gmm_result.bc_low_center),
                    "fraction": float(gmm_result.bc_low_frac),
                    "label": "Deformed",
                },
                "mid_bc": {
                    "center": float(gmm_result.bc_mid_center),
                    "fraction": float(gmm_result.bc_mid_frac),
                    "label": "Recovered",
                },
                "high_bc": {
                    "center": float(gmm_result.bc_high_center),
                    "fraction": float(gmm_result.bc_high_frac),
                    "label": "Recrystallized",
                },
            },
            "weights": [float(w) for w in gmm_result.weights],
            "means": [float(m) for m in gmm_result.means],
            "stds": [float(s) for s in gmm_result.stds],
            "histogram": {
                "bin_centers": [float(c) for c in hist_result.bin_centers],
                "counts": [float(c) for c in hist_result.counts],
                "bin_width": float(hist_result.bin_width),
            },
        }
    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {e}")
    except Exception as e:
        logger.exception("BC GMM fit failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/texture")
@_serialize_mpl
def analyze_texture():
    """Run texture analysis (ODF, components)."""
    if _dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    try:
        from analysis.texture_analysis import analyze_texture
        from analysis.texture_components import TEXTURE_PRESETS
        # Use FCC_Rolling as default preset
        preset = TEXTURE_PRESETS.get("FCC_Rolling", {})
        results = analyze_texture(_dataset, preset)
        response = {"success": True}
        if hasattr(results, 'component_fractions'):
            response["components"] = [
                {"name": name, "fraction": float(frac)}
                for name, frac in results.component_fractions.items()
            ]
        if hasattr(results, 'texture_index') and results.texture_index is not None:
            response["texture_index"] = float(results.texture_index)
        if hasattr(results, 'entropy') and results.entropy is not None:
            response["texture_entropy"] = float(results.entropy)
        if hasattr(results, 'plane_111_fraction'):
            response["surface_planes"] = {
                "{111}": float(results.plane_111_fraction),
                "{100}": float(results.plane_100_fraction),
                "{110}": float(results.plane_110_fraction),
            }
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/texture/components")
async def get_texture_components(preset: str = "FCC_Rolling"):
    """Get available texture components for a preset."""
    try:
        from analysis.texture_components import TEXTURE_PRESETS
        if preset not in TEXTURE_PRESETS:
            return {"presets": list(TEXTURE_PRESETS.keys())}
        return {
            "preset": preset,
            "components": list(TEXTURE_PRESETS[preset].keys()),
        }
    except ImportError:
        return {"error": "Texture components module not available"}


def _collect_export_results() -> dict:
    """Collect all available analysis results from _dataset for Excel export."""
    import numpy as np
    results = {}

    if _dataset is None:
        return results

    # Basic info
    shape = _dataset.xmap.shape if hasattr(_dataset.xmap, 'shape') else (0, 0)
    dim_y = shape[0] if len(shape) >= 1 else 0
    dim_x = shape[1] if len(shape) >= 2 else 0
    n_grains = _dataset.grains.n_grains if _dataset.grains else 0
    results["basic_info"] = {
        "step_micron": float(_dataset.step_size),
        "dim_x": dim_x,
        "dim_y": dim_y,
        "n_grains": n_grains,
        "rotation_deg_1": 0, "rotation_deg_2": 0, "rotation_deg_3": 0,
        "gb_threshold_deg": 5.0,
        "min_pixels": 5,
        "min_intercept_um": 1.5,
    }

    if _dataset.grains is None:
        return results

    # Grain size
    try:
        from analysis.grain_analysis import analyze_grain_size
        gs = analyze_grain_size(_dataset, _dataset.grains)
        results["grain_size"] = gs
    except Exception as e:
        logger.warning("Excel export: grain_size failed: %s", e)

    # Deformation
    try:
        from analysis.deformation_analysis import analyze_deformation
        deform = analyze_deformation(_dataset, _dataset.grains)
        results["deformation"] = deform
    except Exception as e:
        logger.warning("Excel export: deformation failed: %s", e)

    # BC GMM
    try:
        from analysis.bc_analysis import fit_bc_gmm as _fit_gmm, calculate_bc_histogram
        hist_result = calculate_bc_histogram(_dataset)
        results["bc_histogram"] = hist_result
        gmm_result = _fit_gmm(_dataset)
        if gmm_result is not None:
            results["bc_gmm"] = gmm_result
    except Exception as e:
        logger.warning("Excel export: bc_gmm failed: %s", e)

    # Texture
    try:
        from analysis.texture_analysis import analyze_texture
        from analysis.texture_components import TEXTURE_PRESETS
        preset = TEXTURE_PRESETS.get("FCC_Rolling", {})
        tx = analyze_texture(_dataset, preset)
        results["texture"] = tx
    except Exception as e:
        logger.warning("Excel export: texture failed: %s", e)

    # RX — use pixel-level classify_rx to compute basic stats
    try:
        import numpy as np
        from analysis.rx_analysis import classify_rx

        rx_map = classify_rx(_dataset, _dataset.grains)
        total_px = rx_map.size
        rx_px = int(np.sum(rx_map == 0))
        rx_fraction = float(rx_px / max(total_px, 1) * 100)

        # Compute high BC fraction
        high_bc_fraction = 0.0
        if hasattr(_dataset.xmap, 'prop') and 'bc' in _dataset.xmap.prop:
            bc = _dataset.xmap.prop['bc']
            bc_max = float(np.nanmax(bc)) if bc.size > 0 else 1.0
            high_bc_fraction = float(np.sum(bc > 0.7 * bc_max) / max(bc.size, 1) * 100)

        # Build GOS histogram from grains
        gos_hist = (np.array([]), np.array([]))
        gkam_hist = (np.array([]), np.array([]))
        if hasattr(_dataset.grains, 'gos') and _dataset.grains.gos is not None:
            gos_deg = np.degrees(_dataset.grains.gos)
            gos_valid = gos_deg[np.isfinite(gos_deg)]
            if len(gos_valid) > 0:
                counts, edges = np.histogram(gos_valid, bins=20)
                centers = (edges[:-1] + edges[1:]) / 2
                # Area-weight: use grain sizes as weights
                if hasattr(_dataset.grains, 'grain_size'):
                    counts_w, _ = np.histogram(gos_valid, bins=edges,
                                               weights=_dataset.grains.grain_size[np.isfinite(gos_deg)])
                    gos_hist = (counts_w, centers)
                else:
                    gos_hist = (counts, centers)

        results["rx"] = _RXExportData(
            rx_fraction=rx_fraction,
            high_bc_fraction=high_bc_fraction,
            low_gkam_fraction=0.0,
            low_kam_fraction=0.0,
            gos_histogram=gos_hist,
            gkam_histogram=gkam_hist,
        )
    except Exception as e:
        logger.warning("Excel export: rx failed: %s", e)

    return results


@router.post("/export/excel")
async def export_excel(req: ExportExcelRequest, background_tasks: BackgroundTasks):
    """Export analysis results to Excel."""
    if _dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")

    task_id = str(uuid.uuid4())
    _track_task(task_id, {"status": "running", "progress": 0.0})

    def run():
        try:
            from analysis.excel_exporter import export_to_excel
            results = _collect_export_results()
            export_to_excel(req.output_path, req.dataset_name, results)
            _analysis_tasks[task_id]["progress"] = 1.0
            _analysis_tasks[task_id]["status"] = "completed"
            _analysis_tasks[task_id]["result"] = {"path": req.output_path}
        except Exception as e:
            _analysis_tasks[task_id]["progress"] = 1.0
            _analysis_tasks[task_id]["status"] = "failed"
            _analysis_tasks[task_id]["error"] = str(e)

    background_tasks.add_task(run)
    return {"task_id": task_id}


@router.get("/export/excel-download")
async def export_excel_download(name: str = "Dataset"):
    """Generate Excel in a temp file and return as a browser download."""
    if _dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    try:
        import tempfile, os
        from analysis.excel_exporter import export_to_excel

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            results = _collect_export_results()
            export_to_excel(tmp_path, name, results)
            with open(tmp_path, "rb") as fh:
                data = fh.read()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return StreamingResponse(
            io.BytesIO(data),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": 'attachment; filename="EBSD_Analysis.xlsx"'},
        )
    except Exception as e:
        logger.exception("Failed to generate Excel for download")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def batch_process(req: BatchProcessRequest, background_tasks: BackgroundTasks):
    """Run batch processing on a folder of files."""
    task_id = str(uuid.uuid4())
    _track_task(task_id, {"status": "running", "progress": 0.0})

    def run():
        try:
            from analysis.batch_processor import batch_process_folder
            batch_process_folder(req.folder_path, req.output_path)
            _analysis_tasks[task_id]["progress"] = 1.0
            _analysis_tasks[task_id]["status"] = "completed"
        except Exception as e:
            _analysis_tasks[task_id]["progress"] = 1.0
            _analysis_tasks[task_id]["status"] = "failed"
            _analysis_tasks[task_id]["error"] = str(e)

    background_tasks.add_task(run)
    return {"task_id": task_id}
