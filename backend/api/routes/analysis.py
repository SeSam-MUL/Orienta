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

import copy
import functools
import io
import json
import logging
import re
import threading
import urllib.parse
import uuid
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, StrictInt

from analysis.ebsd_dataset import NO_BAND_CONTRAST
from analysis.deformation_analysis import DEFAULT_DEFORMATION_PARAMS
from analysis.rx_analysis import DEFAULT_RX_CRITERIA, RX_CRITERIA_RANGES, rx_phase_info
from backend.api.problem import problem
from backend.api.services.image_utils import array_to_base64_png, colormap_array_to_base64, colormap_stops
from backend.api.services.scan_geometry import ScanGeometry

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

# The scalar maps /map/{type} draws with a colormap scaled to the data's own minimum and maximum, and the unit of
# their values. They answer ``value_range`` and ``colormap_stops``; the maps with fixed colours (IPF, RX, texture
# components, grain boundaries) do not.
SCALAR_MAP_UNITS = {"bc": "", "kam": "°", "gos": "°", "ecd": "µm"}

# Edge-grain rule, ECD basis and phase of the grain-size statistics, as last used by a grain
# reconstruction or a /grain-size call. The Excel export and the ECD map follow them, so the numbers on
# the page, in the map and in the file come from the same choice. Reset to the defaults wherever _dataset
# is replaced. ``stats_phase`` is None (all phases), a phase id or "most_measured".
# (The parameters the grains were reconstructed with live on the dataset: ``_dataset.grain_recon_params``.)
EdgeRule = Literal["matlab_script", "mtex", "outer_row", "keep_all"]
EcdBasis = Literal["cell_area", "measured_pixels"]
StatsPhase = Union[StrictInt, Literal["most_measured"]]
# the phase the texture is computed on: "most_measured" (the phase with the most indexed pixels) or a phase id
TexturePhase = Union[StrictInt, Literal["most_measured"]]
# the phase the RX analysis is made on: the same choice
RxPhase = Union[StrictInt, Literal["most_measured"]]
DEFAULT_GRAIN_STATS = {"edge_rule": "matlab_script", "ecd_basis": "cell_area", "stats_phase": None}
_grain_stats_settings: dict = dict(DEFAULT_GRAIN_STATS)

# The texture presets the page offers, and the largest tolerance angle it accepts (degrees).
TexturePreset = Literal["FCC_Rolling", "BCC_Rolling"]
TEXTURE_TOLERANCE_MAX_DEG = 30.0
TEXTURE_PRESET_NAME = "FCC_Rolling"        # the default preset
TEXTURE_TOLERANCE_DEG = 10.0               # the default tolerance angle, degrees

# What the last texture analysis (POST /texture) was run with: the phase ("most_measured" or a phase id), the preset and
# the tolerance angle. The Excel export, the RX / substructure texture split, the "Texture Components" map and /status
# follow it, so the numbers on the page, in the map and in the file are of the same choice. Reset with the grain
# statistics settings, under the same lock, wherever _dataset is replaced.
DEFAULT_TEXTURE_SETTINGS = {"phase": "most_measured", "preset": TEXTURE_PRESET_NAME, "tolerance_deg": TEXTURE_TOLERANCE_DEG}
_texture_settings: dict = dict(DEFAULT_TEXTURE_SETTINGS)
# One texture run at a time (the kernel density needs about 1 GB at its peak). Nothing else waits for it: the texture
# code draws nothing with matplotlib, so it does not take ``_mpl_lock``.
_texture_lock = threading.Lock()

# The four RX criteria the Recrystallization tab sends with "Classify" (POST /rx): the GOS threshold (degrees), the
# gBC fraction of the largest gBC, the gKAM threshold (degrees) and the minimum grain radius (um). The RX map,
# the panel numbers, the Excel export and /status follow them, so the numbers on the page, in the map and in
# the file come from the same choice. They are reset together with the grain statistics settings, under the same
# lock, wherever _dataset is replaced.
_rx_criteria: dict = dict(DEFAULT_RX_CRITERIA)

# The phase of the RX analysis ("most_measured" or a phase id), remembered with the criteria it was classified with
# (same lock, same reset): the RX map, the panel, the Excel export and the RX / texture split use it.
DEFAULT_RX_SETTINGS = {"phase": "most_measured"}
_rx_settings: dict = dict(DEFAULT_RX_SETTINGS)

# The deformation parameters of the last POST /deformation: the KAM order, the KAM threshold (degrees) and the angle
# from which a boundary is HAGB (degrees; SAGB runs from 3 degrees up to it). The KAM map, the RX analysis' KAM, the
# Excel export and /status follow them. Reset with the others, under the same lock.
_deformation_settings: dict = dict(DEFAULT_DEFORMATION_PARAMS)

# The last RX result (ids of the RX grains and the panel fields), so that "Classify" (POST /rx) and the map fetch
# that follows it (GET /map/rx) compute the KAM and the RX analysis once. It holds the dataset and the grains it
# was made for and is used only for the same dataset object, the same grains object, the same criteria and an
# unchanged map (band contrast present, number of indexed pixels): a reconstruction makes new grains, another
# map is another dataset, other criteria another key. It is cleared where the map is replaced and by the
# quality filter, which unindexes pixels. Read and written under ``_grain_state_lock``.
_rx_cache: dict = {}


# Held by POST /grain-size and the reconstruct task while they read or write the grains of ``_dataset``,
# the parameters they were made with and the stored statistics settings, so a settings write from an
# analysis of older grains cannot land after a newer reconstruction. ``GET /status`` and the start of
# ``_collect_export_results`` take their snapshot (dataset, grains, parameters, settings) under it, so
# they see the grains of one reconstruction with its own parameters; the export then works on that
# snapshot for its whole run (it does not see a reconstruction that finishes meanwhile). ``/load`` and
# ``restore_dataset_for_file`` replace ``_dataset`` and reset the settings without it, in two statements:
# a ``/status`` between them can show the new map with the settings of the old one (microseconds; the
# next call is right).
_grain_state_lock = threading.Lock()


def _reset_grain_stats_settings() -> None:
    # one update, never an empty dict in between: a reader on another thread must always find every key
    _grain_stats_settings.update(DEFAULT_GRAIN_STATS)
    _texture_settings.update(DEFAULT_TEXTURE_SETTINGS)
    _rx_criteria.update(DEFAULT_RX_CRITERIA)
    _rx_settings.update(DEFAULT_RX_SETTINGS)
    _deformation_settings.update(DEFAULT_DEFORMATION_PARAMS)
    _rx_cache.clear()


def _edge_fields(gs=None) -> dict:
    """The fields that say what a grain-size result is made of: rule, basis, phase, the counts."""
    if gs is None:
        # the stored choice as it is: None, a phase id or "most_measured" (no grains, so no name to give)
        return {"edge_rule": _grain_stats_settings["edge_rule"], "ecd_basis": _grain_stats_settings["ecd_basis"],
                "stats_phase": _grain_stats_settings.get("stats_phase"), "stats_phase_name": None,
                "n_grains_total": 0, "n_edge_grains": 0, "n_grains_used": 0, "edge_safeguard_applied": False}
    return {
        "edge_rule": gs.edge_rule, "ecd_basis": gs.ecd_basis,
        "stats_phase": gs.stats_phase, "stats_phase_name": gs.stats_phase_name,
        "n_grains_total": int(gs.n_grains_total), "n_edge_grains": int(gs.n_edge_grains),
        "n_grains_used": int(gs.n_grains_used), "edge_safeguard_applied": bool(gs.edge_safeguard_applied),
    }


def _recon_fields(recon):
    """The reconstruction parameters in the names of the page's controls, or None without any."""
    if not recon:
        return None
    return {"threshold": recon.get("gb_threshold_deg"), "min_pixels": recon.get("min_pixels"),
            "min_intercept": recon.get("min_intercept_um"), "alpha": recon.get("alpha"),
            "fill": recon.get("fill")}


def _phase_options(dataset=None) -> list:
    """The phases of ``dataset`` (default: the loaded map) as ``[{"id", "name"}]`` (not-indexed left out),
    for the page's phase list."""
    from analysis.grain_analysis import phase_ids, phase_name
    if dataset is None:
        dataset = _dataset
    if dataset is None:
        return []
    try:
        from analysis.texture_analysis import cubic_phase_ids
        cubic = cubic_phase_ids(dataset)      # the texture components are cubic: the Texture tab marks the others
        return [{"id": i, "name": phase_name(dataset, i), "cubic": i in cubic} for i in phase_ids(dataset)]
    except Exception as e:
        # an empty list hides the page's phase dropdowns: say why in the log, with the traceback
        logger.warning("could not list the phases of the analysis dataset (the phase lists stay empty): %s", e,
                       exc_info=True)
        return []


def _check_stats_phase(stats_phase, dataset=None) -> None:
    """400 with the known phase ids when ``stats_phase`` is a phase id ``dataset`` (default: the loaded map)
    does not have."""
    if dataset is None:
        dataset = _dataset
    if isinstance(stats_phase, int) and dataset is not None:
        known = [o["id"] for o in _phase_options(dataset)]
        if stats_phase not in known:
            raise HTTPException(
                status_code=400,
                detail=f"stats_phase {stats_phase} is not a phase of this map; the phase ids are {known}")

def _check_chosen_phase_present(dataset, choice, kind: str) -> None:
    """400 when a chosen phase id has no indexed pixel in ``dataset`` any more (the quality filter can unindex every
    pixel of a phase that the RX tab or the Texture tab had chosen). ``choice`` is ``"most_measured"`` (never gone)
    or a phase id; ``kind`` is ``"RX"`` or ``"texture"``. An id the map never had is the caller's own 400 that lists
    the ids; this one is for a phase that was there."""
    if isinstance(choice, bool) or not isinstance(choice, int):
        return
    if np.any(np.asarray(dataset.xmap.phase_id) == choice):
        return
    from analysis.texture_analysis import _indexed_phase_ids, _phase_label
    known = choice in _indexed_phase_ids(dataset)
    name = _phase_label(dataset, choice) if known else str(choice)
    raise HTTPException(
        status_code=400,
        detail=f"the chosen {kind} phase {name} is no longer in this map; choose a phase")


def _check_stored_phases_present(dataset) -> None:
    """The same check for the RX phase and the texture phase as remembered (the export uses both)."""
    with _grain_state_lock:
        rx_phase, texture_phase = _rx_settings["phase"], _texture_settings["phase"]
    _check_chosen_phase_present(dataset, rx_phase, "RX")
    _check_chosen_phase_present(dataset, texture_phase, "texture")


# Per-file stash so analysis results survive a file round-trip (A -> B -> A).
# Before this, every file switch ran ``_dataset = None`` and grain
# reconstruction / per-pixel results were lost for good. We now keep them
# keyed by the EBSD file path they were built against, mirroring how the
# indexing module preserves its result registry across switches. EBSDDataset
# holds an xmap (orientations/scores per pixel), not the multi-GB pattern
# array, so stashing a handful of files costs a few MB each, not GB.
_dataset_by_file: dict = {}


def _stash_key(file_path, variant_key=None):
    """A file under its default variant keeps the plain path as its key (as it always was); another
    variant (measurement x pattern source) of the same file is its own entry."""
    return file_path if variant_key is None else (file_path, tuple(variant_key))


def stash_dataset_for_file(file_path, variant_key=None) -> None:
    """Save the current analysis ``_dataset`` under ``file_path`` so it can be
    restored when the user switches back. No-op when nothing is loaded or the
    path is empty.

    ``variant_key`` is the viewer's key of the variant the file was loaded as (None: the default).
    The dataset is built from the indexing result of ONE measurement: loading another measurement
    of the same file must not bring it back."""
    if _dataset is not None and file_path:
        _dataset_by_file[_stash_key(file_path, variant_key)] = _dataset


def restore_dataset_for_file(file_path, variant_key=None) -> bool:
    """Make the analysis dataset previously built for ``file_path`` (as ``variant_key``) active
    again (or ``None`` if there is none). Returns True if one was restored."""
    global _dataset
    with _grain_state_lock:
        _dataset = _dataset_by_file.get(_stash_key(file_path, variant_key))
        _reset_grain_stats_settings()
    return _dataset is not None

# Cap analysis-task tracking at MAX_TRACKED_TASKS so a long backend session
# that triggers many grain reconstructions / Excel exports doesn't leak.
from collections import OrderedDict as _OrderedDict
MAX_TRACKED_TASKS = 50
_analysis_tasks: _OrderedDict = _OrderedDict()


def _track_task(task_id: str, initial_state: dict) -> None:
    """Insert a new task while enforcing MAX_TRACKED_TASKS LRU cap."""
    _analysis_tasks[task_id] = initial_state
    # the oldest task that is no longer running goes first: a run in flight is never forgotten (its poller
    # and the one-run-per-map guard both look it up here), even if that leaves more than the cap
    while len(_analysis_tasks) > MAX_TRACKED_TASKS:
        for old_id, old_state in _analysis_tasks.items():
            if old_state.get("status") != "running":
                del _analysis_tasks[old_id]
                break
        else:
            break


# Grain reconstructions in flight: task id -> the dataset the run works on. One run per dataset: a second
# POST /grains/reconstruct answers 409 while the recorded task is still "running" in ``_analysis_tasks``;
# a task that finished, failed or fell out of the registry no longer counts. The run removes its own entry
# in a ``finally``, so a crash cannot block the dataset for good. GET /status names the run on the loaded
# map, so a remounted Grain tab can attach to its poll again.
_reconstruct_runs: dict = {}
_reconstruct_lock = threading.Lock()
RECONSTRUCT_BUSY = "A grain reconstruction is already running; wait for it to finish."


def _prune_reconstruct_runs_locked() -> None:
    for task_id in list(_reconstruct_runs):
        task = _analysis_tasks.get(task_id)
        if task is None or task.get("status") != "running":
            del _reconstruct_runs[task_id]


def _clear_reconstruct_marker() -> None:
    with _reconstruct_lock:
        _reconstruct_runs.clear()


def _running_reconstruct(dataset):
    """``{"task_id", "status": "running", "progress"}`` of the reconstruction running on ``dataset``, or None."""
    with _reconstruct_lock:
        _prune_reconstruct_runs_locked()
        for task_id, run_dataset in _reconstruct_runs.items():
            if run_dataset is dataset:
                task = _analysis_tasks.get(task_id) or {}
                return {"task_id": task_id, "status": "running", "progress": float(task.get("progress", 0.0))}
    return None


def get_analysis_dataset():
    """Accessor for the analysis dataset. Use this instead of importing _dataset directly."""
    return _dataset


# The reader of Orienta's own result .h5 lives with the other crystal-map readers; this name is kept
# for the callers that import it from here (indexing.py, the tests).
from analysis.xmap_loader import load_orienta_result_h5 as _load_kikuchipy_rich_h5  # noqa: E402,F401


def read_index_stamp(path: str) -> Optional[ScanGeometry]:
    """The scan geometry stamped on ``/Indexing`` of an Orienta ``.h5``, or None.

    None for everything that states no geometry: a file that is not HDF5, one without ``/Indexing``
    (the caller's own loaders report that), and an ``/Indexing`` without a stamp (a square or older
    export). A stamp that cannot be read (a missing key, a shift that is not +-0.5, a grid that is
    not the stored data) is a 400, so that a damaged hexagonal file is never read as a square one."""
    import h5py
    from pathlib import Path
    from backend.api.services import orienta_hex_h5
    try:
        if not h5py.is_hdf5(path):
            return None
        with h5py.File(path, "r") as f:
            if "Indexing" not in f:
                return None
    except OSError:
        return None
    try:
        return orienta_hex_h5.read_index_geometry(path)
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        raise HTTPException(status_code=400, detail=(
            f"The scan geometry stored in {Path(path).name} cannot be read ({exc}). "
            "Nothing was stored."))


class LoadXmapRequest(BaseModel):
    xmap_path: str  # Path to .h5 CrystalMap file


class GrainReconstructionRequest(BaseModel):
    misorientation_threshold: float = 5.0  # degrees
    min_grain_size: int = 5  # minimum pixels
    # only recorded in the workbook (Sheet 1 row 11); None = not given, and the cell stays empty
    min_intercept_um: Optional[float] = Field(None, ge=0.0)
    alpha: float = Field(3.1, ge=0.0, le=20.0)  # Voronoi alpha-shape radius factor; 0 closes nothing
    fill: bool = True  # build the display-only fill of closed pixels
    edge_rule: EdgeRule = "matlab_script"   # which grains the grain-size statistics leave out
    ecd_basis: EcdBasis = "cell_area"       # ECD from the cell area or from the measured pixels
    stats_phase: Optional[StatsPhase] = None  # None = all phases, a phase id, or "most_measured"


class GrainSizeRequest(BaseModel):
    edge_rule: Optional[EdgeRule] = None    # None = the setting of the last run
    ecd_basis: Optional[EcdBasis] = None
    # left out = the setting of the last run; an explicit null = all phases
    stats_phase: Optional[StatsPhase] = None


def _rx_field(name: str):
    low, high = RX_CRITERIA_RANGES[name]
    return Field(None, ge=low, le=high)


class RXCriteriaRequest(BaseModel):
    """The RX criteria of the Recrystallization tab. A field left out (or null) keeps the stored value."""
    rx_gos_threshold: Optional[float] = _rx_field("rx_gos_threshold")      # degrees, 0 to 20
    rx_gbc_fraction: Optional[float] = _rx_field("rx_gbc_fraction")        # fraction of the largest gBC, 0 to 1
    rx_gkam_threshold: Optional[float] = _rx_field("rx_gkam_threshold")    # degrees, 0 to 10
    rx_min_radius_um: Optional[float] = _rx_field("rx_min_radius_um")      # micrometres, 0 to 100
    # "most_measured" (the phase with the most indexed pixels) or a phase id; left out (or null) keeps the stored phase
    # ("most_measured" until a choice was made)
    phase: Optional[RxPhase] = None


class DeformationRequest(BaseModel):
    """The parameters of the Deformation tab. A field left out (or null) takes the choice of the last run (the script's
    values until then): KAM order 2, KAM threshold 7 degrees, HAGB from 15 degrees (SAGB from 3 degrees up to it)."""
    kam_order: Optional[int] = Field(None, ge=1, le=5, strict=True)
    kam_threshold_deg: Optional[float] = Field(None, gt=0.0, le=30.0, strict=True)
    hagb_min_deg: Optional[float] = Field(None, gt=3.0, le=90.0, strict=True)


class TextureRequest(BaseModel):
    # a field left out (or null) takes the choice of the last run ("most_measured", FCC_Rolling, 10 degrees until then)
    phase: Optional[TexturePhase] = None
    preset: Optional[TexturePreset] = None
    tolerance_deg: Optional[float] = Field(None, gt=0.0, le=TEXTURE_TOLERANCE_MAX_DEG, strict=True)


class ExportExcelRequest(BaseModel):
    output_path: str
    dataset_name: str = "Dataset"


class BatchProcessRequest(BaseModel):
    folder_path: str
    output_path: str
    grain_edge_rule: Optional[EdgeRule] = None   # None = the batch default (matlab_script)
    grain_ecd_basis: Optional[EcdBasis] = None   # None = the batch default (cell_area)
    grain_stats_phase: Optional[Literal["most_measured"]] = None   # None = all phases
    texture_phase: Optional[TexturePhase] = None   # None = the batch default (most_measured, per file)
    texture_preset: Optional[TexturePreset] = None   # None = the batch default (FCC_Rolling)
    texture_tolerance_deg: Optional[float] = Field(None, gt=0.0, le=TEXTURE_TOLERANCE_MAX_DEG, strict=True)   # None = 10 degrees
    # the reconstruction of every file; None = the processor's default (4 deg, 3 px, alpha 3.1, fill)
    misorientation_threshold: Optional[float] = Field(None, gt=0.0)
    min_grain_size: Optional[int] = Field(None, ge=1)
    alpha: Optional[float] = Field(None, ge=0.0, le=20.0)
    fill: Optional[bool] = None
    min_intercept_um: Optional[float] = Field(None, ge=0.0)   # only recorded in the workbook: 0 is a value, not an error
    # the RX criteria of every file; None = the batch default (1.25 deg, 0.7, 0.55 deg, 0.4 um)
    rx_gos_threshold: Optional[float] = _rx_field("rx_gos_threshold")
    rx_gbc_fraction: Optional[float] = _rx_field("rx_gbc_fraction")
    rx_gkam_threshold: Optional[float] = _rx_field("rx_gkam_threshold")
    rx_min_radius_um: Optional[float] = _rx_field("rx_min_radius_um")
    # the phase of the RX analysis of every file: "most_measured" (per file, the default) or a phase id
    rx_phase: Optional[RxPhase] = None
    # the deformation parameters of every file; None = the script's values (2, 7 degrees, 15 degrees)
    kam_order: Optional[int] = Field(None, ge=1, le=5, strict=True)
    kam_threshold_deg: Optional[float] = Field(None, gt=0.0, le=30.0, strict=True)
    hagb_min_deg: Optional[float] = Field(None, gt=3.0, le=90.0, strict=True)


NO_BC_LAYER_DETAIL = "This file has no band contrast; choose another layer."


def _file_name_only(path) -> Optional[str]:
    """The last component of a file path, whichever separator it uses; None for anything that names no file."""
    if not isinstance(path, str):
        return None
    name = re.split(r"[\\/]", path.strip().rstrip("\\/"))[-1]
    return name or None


def _source_name(dataset) -> Optional[str]:
    """The name (not the folder) of the file the dataset was read from: what the page offers as the default
    dataset name of an export. None when the dataset does not come from a file."""
    name = getattr(dataset, "source_name", None)
    if name:
        return name
    src = getattr(dataset, "source", None)
    if src in (None, "", "unknown", "indexing_result"):
        return None
    return _file_name_only(src)


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

        load_meta: dict = {}        # what the file reader says about the file (the indexing result has none)

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
                        # The measurement the result was indexed on (None = the file's default).
                        from backend.api.services import variant_registry
                        native_bc = read_native_band_contrast(
                            src, n_rows, n_cols,
                            variant=variant_registry.variant_for_source(_last_result, src))
                        if native_bc is not None:
                            bc_flat = native_bc.ravel()
                            if bc_flat.size == xmap.size:
                                xmap.prop['bc'] = bc_flat.astype('float32')
                except Exception:
                    logger.debug("native BC bridge for indexing result failed", exc_info=True)
            step_size = getattr(xmap, 'dx', None)
            if not step_size or step_size == 1.0:
                step_size = _fallback_step_size()
            new_dataset = EBSDDataset(xmap, step_size=step_size, source="indexing_result")
            _meta = getattr(_last_result, "metadata", None)
            new_dataset.source_name = _file_name_only(_meta.get("source_file")) if isinstance(_meta, dict) else None
            # Which grid the grains and the KAM will be computed on: the one the result ran on.
            # Its own copy: the dataset must not change what the result says about itself.
            new_dataset.scan_geometry = copy.deepcopy(
                (_meta if isinstance(_meta, dict) else {}).get("scan_geometry"))
            with _grain_state_lock:           # the new map and the reset settings appear together
                _dataset = new_dataset
                _reset_grain_stats_settings()
        else:
            # Load from file: one function for the page and the batch (analysis/xmap_loader.py). Aztec
            # .h5oina is read from /1/EBSD; .ang/.ctf go through orix (a HexGrid .ang is kept as a
            # hexagonal scan or refused by name, never read as a square map), and Orienta's own result .h5
            # (rich result_*.h5 with /Indexing/, light result_*_light.h5 with /SourceReference/), which
            # orix's rsciio chain cannot read, through our schema reader.
            from analysis.ang_hexgrid import AngGridRefused
            from analysis.xmap_loader import load_xmap_file
            try:
                xmap, step_size, load_meta = load_xmap_file(req.xmap_path)
            except AngGridRefused as refused:
                raise problem(400, refused.code, str(refused))
            except ValueError as load_err:
                raise HTTPException(status_code=400, detail=str(load_err))
            hex_grid = load_meta.get("hex_grid")    # the grid a hexagonal .ang states
            # orix CrystalMap exposes the µm step as ``dx``/``dy``, not
            # the legacy ``step_sizes`` array used by older orix builds.
            # When the file said nothing about step (older exports), dx
            # comes back as 1.0 from the unit-pixel coordinates we built;
            # _fallback_step_size pulls the real value off the active
            # signal in that case. An Aztec header states its step: that is used as it is, and so is
            # the step a hexagonal .ang states.
            # An Orienta .h5 states its grid on /Indexing (a damaged stamp is a 400, before anything
            # is stored); a file with no stamp states none.
            file_stamp = None if hex_grid is not None else read_index_stamp(req.xmap_path)
            if not load_meta.get("step_from_file") and (not step_size or step_size == 1.0):
                step_size = _fallback_step_size()
            new_dataset = EBSDDataset(xmap, step_size=step_size, source=req.xmap_path)
            # A hexagonal .ang states its grid, and so does an Orienta .h5 with a stamp; any other
            # file does not state it (unknown, not square)
            if hex_grid is not None:
                new_dataset.scan_geometry = ScanGeometry.hex(
                    hex_grid["n_rows"], hex_grid["n_cols"], hex_grid["step_x"], hex_grid["step_y"],
                    odd_row_shift=hex_grid["shift"], shift_confirmed=hex_grid["confirmed"]).to_dict()
            else:
                new_dataset.scan_geometry = None if file_stamp is None else file_stamp.to_dict()
            with _grain_state_lock:           # the new map and the reset settings appear together
                _dataset = new_dataset
                _reset_grain_stats_settings()
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
        # what the reader did with the file: slices beyond the first are not read, phases without pixels and
        # without a usable symmetry are left out, and whether the map has band contrast of its own
        resp["n_slices"] = load_meta.get("n_slices")
        resp["skipped_phases"] = list(load_meta.get("skipped_phases", []))
        resp["has_native_bc"] = bool(_dataset.has_native_bc)
        resp["source_name"] = _source_name(_dataset)
        resp["step_size"] = float(_dataset.step_size) if _dataset.step_size is not None else None
        logger.info("analysis load: %s format=%s n_slices=%s skipped_phases=%s has_native_bc=%s",
                    req.xmap_path, load_meta.get("format"), resp["n_slices"], resp["skipped_phases"],
                    resp["has_native_bc"])
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
            resp["phase_options"] = _phase_options()
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
    dataset = _dataset      # read once: everything below comes from this object, whatever is loaded meanwhile
    if dataset is None:
        return {"loaded": False}

    phases = []
    try:
        if hasattr(dataset, 'xmap') and hasattr(dataset.xmap, 'phases'):
            for p in dataset.xmap.phases:
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

    step_size = getattr(dataset, 'step_size', None)

    # one snapshot of the grains, what they were reconstructed with and the statistics settings
    with _grain_state_lock:
        has_grains = getattr(dataset, "grains", None) is not None
        recon = getattr(dataset, "grain_recon_params", None) if has_grains else None
        settings = dict(_grain_stats_settings)
        rx_criteria = dict(_rx_criteria)
        rx_settings = dict(_rx_settings)
        deformation_settings = dict(_deformation_settings)
        texture_settings = dict(_texture_settings)
    return {
        "loaded": True,
        "shape": list(dataset.shape) if hasattr(dataset, 'shape') else [],
        "has_grains": has_grains,
        "phases": phases,
        "phase_options": _phase_options(dataset),
        "step_size": float(step_size) if step_size is not None else None,
        # whether the map has band contrast of its own (the BC layer, the GMM and the RX analysis need it)
        "has_native_bc": bool(dataset.has_native_bc),
        # the file the map was read from, without its folder; null when it does not come from a file
        "source_name": _source_name(dataset),
        # what the page's grain-size controls follow: the settings of the last reconstruction or
        # /grain-size call (stats_phase: null = all phases, a phase id, or "most_measured")
        "edge_rule": settings["edge_rule"],
        "ecd_basis": settings["ecd_basis"],
        "stats_phase": settings["stats_phase"],
        # the phase of the last texture analysis ("most_measured" or a phase id), which the Excel export follows
        "texture_phase": texture_settings["phase"],
        # and the preset and tolerance angle (degrees) of that run
        "texture_preset": texture_settings["preset"],
        "texture_tolerance_deg": texture_settings["tolerance_deg"],
        # the RX criteria of the last "Classify" (the defaults until then): GOS and gKAM in degrees, the gBC
        # fraction of the largest gBC, the minimum radius in um
        **rx_criteria,
        # the phase of the last "Classify" ("most_measured" or a phase id)
        "rx_phase": rx_settings["phase"],
        # the deformation parameters of the last run (the script's values until then)
        "deformation_params": deformation_settings,
        # what the grains were reconstructed with; null when they were not reconstructed in this session
        "recon_params": _recon_fields(recon),
        # identity of the loaded map: changes when another map replaces it, not otherwise
        "dataset_token": getattr(dataset, "dataset_token", None),
        # the grain reconstruction running on this map ({"task_id", "status", "progress"}), else null
        "reconstruct": _running_reconstruct(dataset),
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

    Conventions: pc_x/pc_y/dd come straight from the source h5oina, in the OXFORD
    convention. The calibration store, the PC Refinement controller and the
    indexers hold the BRUKER one (kikuchipy's), so the map is converted with
    kikuchipy's own Oxford to Bruker conversion (PCy = 1 - PCy_Oxford * aspect,
    PCz = PCz_Oxford * aspect, aspect = columns / rows of the pattern) before it
    is stored. Stored unconverted, the Oxford reference scan's PCy 0.552 and PCz 0.694 stood in for
    the real 0.327 and 0.846. The map is the FILE's own PC map, not a refinement,
    so it is stored as a file ("header") map: a later refine moves it as it does
    the map of a loaded file.

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
    entry = calibration_store.get_entry(target)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Dataset '{target}' is not in the calibration store. "
                "Open it in EBSD Viewer first to register it."
            ),
        )
    # The pattern shape fixes the aspect ratio the conversion needs.
    from kikuchipy.detectors import EBSDDetector
    pc_map = np.asarray(
        EBSDDetector(shape=tuple(int(v) for v in entry.detector_shape),
                     pc=pc_map.astype(np.float64), convention="oxford").pc,
        dtype=np.float32).reshape(pc_map.shape)
    calibration_store.update_pc_map(target, pc_map, source="header")

    # The PC Refinement page works on its own controller, which only reads the store
    # the first time it is opened for a dataset. A page that is already open would
    # keep refining from the old PC, so the seed is handed to it as well.
    in_page = _seed_pc_page(target, pc_map)

    # the value the store holds (the mean of the map as stored, in double precision),
    # not a float32 mean of the array: the page and the store must say the same number
    mean_pc = calibration_store.get_entry(target).pc_single
    return {
        "success": True,
        "target_dataset": target,
        "pc_map_shape": list(pc_map.shape),
        "mean_pc": [float(v) for v in mean_pc],
        "source": "aztec_per_pixel",
        "pc_page_updated": in_page,
    }


def _seed_pc_page(target: str, pc_map) -> bool:
    """Give an already opened PC Refinement session of ``target`` the seed map.

    The geometry (tilts, pixel size, binning) stays the page's own; only the PC
    becomes the map. Returns whether a session was updated.
    """
    from backend.api.routes import pcrefinement as _pcr
    sess = _pcr._sessions.get(target)
    if sess is None or sess.ctrl.detector is None:
        return False
    from kikuchipy.detectors import EBSDDetector
    det = sess.ctrl.detector
    seeded = EBSDDetector(
        shape=tuple(int(v) for v in det.shape), px_size=det.px_size,
        binning=det.binning, tilt=det.tilt, azimuthal=det.azimuthal,
        sample_tilt=det.sample_tilt, pc=np.asarray(pc_map, dtype=np.float64))
    with _pcr._state_lock:
        sess.ctrl.apply_pixelwise_pc(seeded)
        sess.sim_cache.clear()
    return True


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
    # Which measurement of the source the light h5 was indexed on: its own /Indexing attributes
    # say (none = the default, "1" as before). Aztec's Phase/Euler must be that measurement's.
    from backend.api.routes.indexing import _read_variant_provenance
    from backend.api.services.load_choices import measurement_groups, variant_from_stamp
    _source_variant = variant_from_stamp(_read_variant_provenance(src_path))
    try:
        with h5py.File(src_h5oina, "r") as f:
            _grp = measurement_groups(f, _source_variant, default="1")[0]
            aztec_phase = np.asarray(f[f"/{_grp}/EBSD/Data/Phase"]).reshape(grid)
            aztec_euler = np.asarray(f[f"/{_grp}/EBSD/Data/Euler"]).reshape(*grid, 3)
            # Build name→id map for Aztec
            aztec_phase_names: dict = {}
            phases_grp = f.get(f"/{_grp}/EBSD/Header/Phases")
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

    # Build Aztec → Orienta phase mapping by NAME (case-insensitive substring match
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
                from crystal_symmetry import phase_point_group
                pg = phase_point_group(first_phase)
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
    axes[1].set_title(f"Orienta Phase\n{len(our_phase_names)} phase(s)", color='white', fontsize=11)
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
        if key == "bc":
            arr = _dataset.band_contrast_raw()
        else:
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
            detail=(f"Band Contrast quality filter: {NO_BAND_CONTRAST}. It needs band contrast from the "
                    "measurement (Oxford H5OINA or .ctf); computed Pattern Quality is not a substitute."),
        )
    try:
        stats = _dataset.apply_quality_filter(
            bc_min=req.bc_min,
            bands_min=req.bands_min,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    with _grain_state_lock:
        _rx_cache.clear()           # pixels were unindexed: an RX result made before is for other pixels
    return {"success": True, "filters": req.model_dump(), **stats}


@router.post("/grains/reconstruct")
async def reconstruct_grains(req: GrainReconstructionRequest, background_tasks: BackgroundTasks):
    """Reconstruct grains from indexed data.

    ``stats_phase`` here: omitted or null means all phases (it overwrites the stored choice). POST
    /grain-size differs: omitted keeps the last setting, an explicit null means all phases.
    """
    if _dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    dataset = _dataset      # the run works on the dataset it was started for, whatever is loaded later
    _check_stats_phase(req.stats_phase, dataset)

    task_id = str(uuid.uuid4())
    with _reconstruct_lock:
        _prune_reconstruct_runs_locked()
        if any(run_dataset is dataset for run_dataset in _reconstruct_runs.values()):
            raise HTTPException(status_code=409, detail=RECONSTRUCT_BUSY)
        _track_task(task_id, {"status": "running", "progress": 0.0})
        _reconstruct_runs[task_id] = dataset

    def run():
        try:
            from analysis.grain_analysis import reconstruct_grains
            grain_set = reconstruct_grains(
                dataset,
                angle_threshold_deg=req.misorientation_threshold,
                min_pixels=req.min_grain_size,
                alpha=req.alpha,
                fill=req.fill,
            )
            with _grain_state_lock:
                if _dataset is not dataset:
                    raise RuntimeError("the dataset was replaced while the grains were being reconstructed; "
                                       "the result was dropped")
                dataset.grains = grain_set
                _rx_cache.clear()           # the cached RX result is for the grains that were just replaced
                dataset.grain_recon_params = {
                    "gb_threshold_deg": req.misorientation_threshold, "min_pixels": req.min_grain_size,
                    "alpha": req.alpha, "min_intercept_um": req.min_intercept_um, "fill": req.fill,
                }
                _grain_stats_settings.update(edge_rule=req.edge_rule, ecd_basis=req.ecd_basis,
                                             stats_phase=req.stats_phase)
            n_grains = grain_set.n_grains if grain_set else 0
            filled_mask = getattr(grain_set, "filled_mask", None) if grain_set else None
            result = {
                "n_grains": n_grains,
                "grain_count": n_grains,
                "alpha": req.alpha,
                "fill": req.fill,
                "n_filled": int(filled_mask.sum()) if filled_mask is not None else 0,
                "edge_rule": req.edge_rule,
                "ecd_basis": req.ecd_basis,
            }
            if n_grains == 0:
                result.update(_edge_fields())      # no grains: the counts are truly zero
            # Enrich with grain-size stats so the UI's completion handler shows
            # real ECD / aspect-ratio (it reads mean_ecd, median_ecd, std_ecd,
            # mean_aspect_ratio) instead of "undefined". Best-effort: if it
            # fails, reconstruction still succeeds and /grain-size can be called.
            if n_grains > 0:
                try:
                    from analysis.grain_analysis import analyze_grain_size, grain_size_summary
                    gs = analyze_grain_size(dataset, grain_set, edge_rule=req.edge_rule,
                                            ecd_basis=req.ecd_basis, stats_phase=req.stats_phase)
                    result.update(_edge_fields(gs))
                    if hasattr(gs, "avg_ECD"):
                        result["mean_ecd"] = float(gs.avg_ECD)
                        result["std_ecd"] = float(gs.std_ECD)
                    if hasattr(gs, "avg_AR"):
                        result["mean_aspect_ratio"] = float(gs.avg_AR)
                    result.update(grain_size_summary(gs))
                except Exception as e:
                    # no counts at all: zeros would read like a map without grains
                    logger.warning("grain-size summary after reconstruct failed", exc_info=True)
                    result["grain_stats_error"] = str(e)
            _analysis_tasks[task_id]["result"] = result
            _analysis_tasks[task_id]["grain_count"] = n_grains
            # Set progress=1.0 BEFORE status=completed so polling clients that
            # check progress first don't see "completed at 0%" (which the
            # React spinner rendered as stuck indefinitely).
            _analysis_tasks[task_id]["progress"] = 1.0
            _analysis_tasks[task_id]["status"] = "completed"
        except Exception as e:
            _analysis_tasks[task_id]["progress"] = 1.0
            _analysis_tasks[task_id]["error"] = str(e)       # before the status: a poller never sees "failed" without it
            _analysis_tasks[task_id]["status"] = "failed"
        finally:
            with _reconstruct_lock:
                _reconstruct_runs.pop(task_id, None)

    background_tasks.add_task(run)
    return {"task_id": task_id}


@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in _analysis_tasks:
        raise HTTPException(status_code=404)
    return _analysis_tasks[task_id]


def _per_grain_image(values, grain_ids) -> np.ndarray:
    """Image of one value per grain: ``values[i]`` on every pixel of grain ``i + 1``, NaN elsewhere.

    One lookup instead of a pass over the map per grain. A label without a value (above
    ``len(values)``) and label 0 (no grain) are NaN: a pixel that belongs to no grain has no grain quantity, and
    the map draws it in the colormap's "bad" colour and leaves it out of the value range.
    """
    values = np.asarray(values, dtype=float)
    grain_ids = np.asarray(grain_ids)
    lookup = np.concatenate(([np.nan], values))
    return lookup[np.where(grain_ids <= len(values), grain_ids, 0)]


_NO_GRAIN_GREY = (128, 128, 128)
# The RX map has its own fixed colours, whatever colormap the page offers: red means "RX by the analysis" and
# nothing else, and the GOS bands around it must not borrow a red from a colormap. Index = class of rx_map_classes.
_RX_CLASS_COLOURS = {
    0: (166, 206, 227),       # GOS below 1.5 degrees, not RX by the analysis: light blue
    1: (254, 224, 139),       # substructured, GOS 1.5 to 7.5 degrees: yellow
    2: (140, 86, 40),         # deformed, GOS from 7.5 degrees: brown
    3: (215, 48, 39),         # RX by the analysis: red
}


# Fixed colours of the texture components (no greys: grey is "no component", black "not of the texture phase"); the
# i-th component of the preset takes the i-th colour.
_TEXTURE_COMPONENT_COLOURS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5",
)
_NO_COMPONENT_COLOUR = "#808080"
_OUTSIDE_PHASE_COLOUR = "#000000"


def _hex_to_rgb(colour: str):
    return tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))


def _texture_component_map_response(dataset) -> dict:
    """The answer of ``/map/texture_component``: each pixel of the texture phase in the colour of the component it is
    closest to (within the tolerance), "no component" grey for the other pixels of that phase, black for pixels of
    other phases and not-indexed pixels, plus the legend ``[{name, color}]``.

    Phase, preset and tolerance are those of the last texture run (``_texture_settings``), so the map shows the pixels
    the numbers on the Texture tab were counted on. 400 when the
    phase cannot be used (not cubic, unknown id)."""
    import base64
    import io
    from PIL import Image
    from analysis.texture_analysis import _phase_label, calculate_texture_component_map, resolve_texture_phase
    from analysis.texture_components import TEXTURE_PRESETS
    with _grain_state_lock:
        settings = dict(_texture_settings)
    try:
        phase_id = resolve_texture_phase(dataset, settings["phase"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _check_chosen_phase_present(dataset, settings["phase"], "texture")
    preset = TEXTURE_PRESETS[settings["preset"]]
    # (no _texture_lock: the angles are cached on the dataset after a texture run, and a cold call is a couple of seconds)
    component_map, names = calculate_texture_component_map(
        dataset, preset, settings["tolerance_deg"], texture_phase=phase_id)
    in_phase = (np.asarray(dataset.xmap.phase_id) == phase_id).reshape(component_map.shape)
    palette = [_TEXTURE_COMPONENT_COLOURS[i % len(_TEXTURE_COMPONENT_COLOURS)] for i in range(len(names))]
    rgb = np.empty(component_map.shape + (3,), dtype=np.uint8)
    rgb[:] = _hex_to_rgb(_OUTSIDE_PHASE_COLOUR)
    rgb[in_phase] = _hex_to_rgb(_NO_COMPONENT_COLOUR)
    for k, colour in enumerate(palette, start=1):
        rgb[component_map == k] = _hex_to_rgb(colour)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return {
        "image": base64.b64encode(buf.getvalue()).decode("utf-8"),
        "map_type": "texture_component",
        "shape": list(component_map.shape),
        "min_val": 0.0,
        "max_val": float(len(names)),
        "legend": [{"name": n, "color": c} for n, c in zip(names, palette)],
        "no_component_color": _NO_COMPONENT_COLOUR,
        "outside_color": _OUTSIDE_PHASE_COLOUR,
        "texture_phase": {"id": phase_id, "name": _phase_label(dataset, phase_id)},
        "texture_preset": settings["preset"],                 # what the colours were counted with
        "texture_tolerance_deg": settings["tolerance_deg"],
    }


def _rx_class_image_to_base64(classes):
    """PNG of the RX map classes (``analysis.rx_analysis.rx_map_classes``) in fixed colours: red for the grains the
    RX analysis classified as recrystallized, light blue / yellow / brown for the GOS bands of the other grains,
    neutral grey for pixels without a grain."""
    import base64
    import io
    from PIL import Image
    classes = np.asarray(classes)
    rgb = np.empty(classes.shape + (3,), dtype=np.uint8)
    rgb[:] = _NO_GRAIN_GREY
    for cls, colour in _RX_CLASS_COLOURS.items():
        rgb[classes == cls] = colour
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _rx_cache_key(dataset, criteria: dict, phase_id: int, deformation: dict) -> tuple:
    return (getattr(dataset, "dataset_token", None), tuple(float(criteria[name]) for name in DEFAULT_RX_CRITERIA),
            int(phase_id), int(deformation["kam_order"]), float(deformation["kam_threshold_deg"]),
            bool(dataset.has_native_bc), int(dataset.indexed_mask.sum()))


def _rx_state() -> tuple:
    """``(criteria, rx_phase, deformation)`` as remembered right now, one snapshot under the lock."""
    with _grain_state_lock:
        return dict(_rx_criteria), _rx_settings["phase"], dict(_deformation_settings)


def _rx_panel(dataset, grains, criteria: dict, strict: bool = False, rx_phase="most_measured",
              deformation: Optional[dict] = None):
    """The numbers of the RX tab for ``criteria`` (the names of ``DEFAULT_RX_CRITERIA``): the ids of the grains the
    RX analysis classified as RX, and the response fields (the criteria used, the grain counts and, when the
    analysis ran, its fractions).

    The ids are None when the analysis was not run: without native band contrast (gBC would be a surrogate) there
    is no RX fraction, no RX grain count and no red, only the GOS bands, because a GOS-only number would read
    like the analysis'. When the analysis fails, ``strict`` raises its error; otherwise it is logged and the
    numbers are left out as well. The result of the last call is reused for the same dataset, grains, criteria,
    RX phase and deformation parameters (``_rx_cache``); a failure is never cached. The fields come back as a copy.

    ``rx_phase``: ``"most_measured"`` or a phase id; the fields name the phase it resolves to (``rx_phase``:
    ``{"id", "name"}``). ``ValueError`` for an id the map does not have. ``deformation``: the KAM order and threshold
    (``kam_order``, ``kam_threshold_deg``) the RX analysis' gKAM is made with; default the script's.
    """
    deformation = dict(DEFAULT_DEFORMATION_PARAMS if deformation is None else deformation)
    phase = rx_phase_info(dataset, rx_phase)
    key = _rx_cache_key(dataset, criteria, phase["id"], deformation)
    with _grain_state_lock:
        hit = _rx_cache.get("entry")
        if hit is not None and hit["dataset"] is dataset and hit["grains"] is grains and hit["key"] == key:
            return hit["ids"], dict(hit["fields"])
    fields = {"total_grains": int(grains.n_grains), "rx_needs_bc": not bool(dataset.has_native_bc),
              "rx_phase": phase,
              **{name: float(criteria[name]) for name in DEFAULT_RX_CRITERIA}}
    rx_ids = None
    failed = False
    if dataset.has_native_bc:
        try:
            from analysis.deformation_analysis import calculate_kam
            from analysis.rx_analysis import analyze_recrystallization
            rx_kam = calculate_kam(dataset, order=deformation["kam_order"],
                                   threshold_deg=deformation["kam_threshold_deg"],
                                   grains=grains if grains.n_grains > 0 else None)
            rx_res = analyze_recrystallization(
                dataset, grains, rx_kam, gos_threshold_deg=criteria["rx_gos_threshold"],
                gbc_min_fraction=criteria["rx_gbc_fraction"], gkam_threshold_deg=criteria["rx_gkam_threshold"],
                min_grain_radius_um=criteria["rx_min_radius_um"], rx_phase=phase["id"])
            rx_ids = rx_res.rx_grain_ids
            fields["rx_grain_count"] = int(len(rx_res.rx_grain_ids))
            # a number the analysis could not make (None: no pixel to divide by) is left out, like a number of an
            # analysis that did not run, never sent as a 0
            for name, value in (("rx_fraction", rx_res.rx_fraction), ("high_bc_fraction", rx_res.high_bc_fraction),
                                ("low_gkam_fraction", rx_res.low_gkam_fraction),
                                ("low_kam_fraction", rx_res.low_kam_fraction)):
                if value is not None:
                    fields[name] = float(value)
        except Exception as rx_err:
            if strict:
                raise
            failed = True
            logger.warning("RX map: the RX numbers are left out and no grain is drawn RX: %s", rx_err)
    fields["rx_drawn"] = rx_ids is not None
    if not failed:
        with _grain_state_lock:
            if _dataset is not dataset:
                return rx_ids, fields       # the map was replaced meanwhile: the cache must not keep it alive
            _rx_cache["entry"] = {"dataset": dataset, "grains": grains, "key": key, "ids": rx_ids, "fields": dict(fields)}
    return rx_ids, fields


@router.get("/map/{map_type}")
@_serialize_mpl
def get_analysis_map(map_type: str, cmap: str = "viridis"):
    """Get an analysis map (BC, IPF, KAM, GOS, ECD, etc.)."""
    # One snapshot of the map and its grains for the whole build: a reconstruction that finishes meanwhile
    # replaces ``dataset.grains``, and the ids of one set of grains must not be drawn on another.
    with _grain_state_lock:
        dataset = _dataset
        grains = dataset.grains if dataset is not None else None
    if dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")

    # Validate dataset has a 2D shape (not a scalar from single-pixel indexing)
    ds_shape = dataset.shape
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
            bc_raw = dataset.band_contrast_raw()
            if bc_raw is None:
                # an all-zero map would be drawn black and read as a broken image
                raise HTTPException(status_code=400, detail=NO_BC_LAYER_DETAIL)
            data = bc_raw.reshape(dataset.shape)
        elif map_type in ("ipf", "ipf_x", "ipf_y", "ipf_z"):
            from tools.phase_map_generator import compute_ipf_colors
            direction = map_type.split("_")[1].upper() if "_" in map_type else "Z"
            xm = dataset.xmap
            g = grains
            if g is not None and g.filled_mask is not None and g.filled_mask.any():
                # Display copy: closed pixels take their grain's orientation and phase.
                from orix.crystal_map import CrystalMap
                from orix.quaternion import Rotation
                q = np.asarray(xm.rotations.data, dtype=float).reshape(-1, 4).copy()
                pid = np.asarray(xm.phase_id).reshape(-1).copy()
                f = g.filled_mask.reshape(-1)
                q[f] = g.filled_rotations.reshape(-1, 4)[f]
                pid[f] = g.grain_phase[g.grain_ids.reshape(-1)[f] - 1]
                xm = CrystalMap(rotations=Rotation(q), phase_id=pid, x=xm.x, y=xm.y,
                                phase_list=xm.phases)
            colors = compute_ipf_colors(xm, direction_str=direction)
            import io, base64
            from PIL import Image
            n_rows, n_cols = dataset.shape
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
            g = grains
            # A GrainSet without grains (every grain culled) would mask every pixel out. The order and threshold are
            # those of the last deformation run, so the map shows the KAM the numbers were made of.
            with _grain_state_lock:
                deformation = dict(_deformation_settings)
            data = calculate_kam(dataset, order=deformation["kam_order"], threshold_deg=deformation["kam_threshold_deg"],
                                 grains=g if g is not None and g.n_grains > 0 else None)
        elif map_type == "gos":
            if grains is None:
                raise HTTPException(status_code=400, detail="Reconstruct grains first for GOS map")
            # Map per-grain GOS values back to pixel grid
            gos_per_grain = grains.gos  # radians, per grain
            grain_ids = grains.grain_ids
            data = np.zeros(dataset.shape, dtype=float)
            if gos_per_grain is not None:
                data = _per_grain_image(np.degrees(gos_per_grain), grain_ids)
        elif map_type == "ecd":
            if grains is None:
                raise HTTPException(status_code=400, detail="Reconstruct grains first for ECD map")
            ecd_per_grain = grains.ecd  # µm, per grain
            grain_ids = grains.grain_ids
            data = np.zeros(dataset.shape, dtype=float)
            ecd_basis = _grain_stats_settings["ecd_basis"]
            if ecd_per_grain is not None:
                from analysis.grain_analysis import grain_ecd_and_area
                try:
                    ecd_per_grain, _ = grain_ecd_and_area(dataset, grains, ecd_basis)
                except ValueError as e:      # e.g. measured_pixels on grains without grain_size
                    raise HTTPException(status_code=400, detail=str(e))
                data = _per_grain_image(ecd_per_grain, grain_ids)
            extra_fields["ecd_basis"] = ecd_basis     # the range line names it
        elif map_type == "rx":
            if grains is None:
                raise HTTPException(status_code=400, detail="Reconstruct grains first for RX map")
            from analysis.rx_analysis import rx_map_classes
            # The red of the map and the numbers of the panel are the RX analysis' own (the Excel export and the
            # batch use the same call), with the criteria of the last "Classify" (POST /rx; the defaults until
            # then), so the map, the page and the workbook agree. A GET never changes the criteria. When the
            # analysis cannot run, no number is sent and no grain is drawn red.
            criteria, rx_phase, deformation = _rx_state()
            _check_chosen_phase_present(dataset, rx_phase, "RX")
            rx_ids, rx_fields = _rx_panel(dataset, grains, criteria, rx_phase=rx_phase, deformation=deformation)
            extra_fields.update(rx_fields)
            # without native band contrast rx_ids is None: only the GOS bands are drawn; the grains of the other
            # phases are drawn neutral either way (they are not part of the analysis)
            data = rx_map_classes(dataset, grains, rx_ids, phase_id=rx_fields["rx_phase"]["id"])   # -1 no grain, 0..2 GOS, 3 RX
        elif map_type == "texture_component":
            return _texture_component_map_response(dataset)
        elif map_type == "grain_boundaries":
            if grains is None:
                raise HTTPException(status_code=400, detail="Reconstruct grains first for grain boundary map")
            # Grain boundary map: 1 at boundaries, 0 elsewhere
            grain_ids = grains.grain_ids
            gb = np.zeros(dataset.shape, dtype=float)
            gb[:-1, :] += (grain_ids[:-1, :] != grain_ids[1:, :]).astype(float)
            gb[:, :-1] += (grain_ids[:, :-1] != grain_ids[:, 1:]).astype(float)
            data = np.clip(gb, 0, 1)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown map type: {map_type}")

        if map_type == "rx":
            # fixed colours per class (the colormap does not apply); pixels without a grain (-1) are drawn neutral
            image = _rx_class_image_to_base64(data)
            classified = data[data >= 0]
            lo, hi = (float(classified.min()), float(classified.max())) if classified.size else (0.0, 0.0)
        else:
            image = colormap_array_to_base64(data, cmap)
            finite = np.isfinite(data)
            lo, hi = (float(np.min(data[finite])), float(np.max(data[finite]))) if finite.any() else (0.0, 0.0)
            if map_type in SCALAR_MAP_UNITS and np.isfinite(lo) and np.isfinite(hi):
                # the numbers the image was scaled with (colormap_array_to_base64 uses nanmin / nanmax of ``data``) and
                # the colours of the colormap, so the viewer can draw a scale bar under the map
                extra_fields["value_range"] = {"min": lo, "max": hi, "unit": SCALAR_MAP_UNITS[map_type]}
                extra_fields["colormap_stops"] = colormap_stops(cmap)
        response = {
            "image": image,
            "map_type": map_type,
            "shape": list(data.shape),
            "min_val": lo,
            "max_val": hi,
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


@router.post("/rx")
def classify_recrystallization(req: Optional[RXCriteriaRequest] = None):
    """Classify the RX grains with the criteria of the Recrystallization tab and remember them.

    A field left out (or null) takes the stored value. The criteria used are remembered for the RX map, the Excel
    export and /status, and answered with the numbers of the panel (``rx_fraction``, ``rx_grain_count``,
    ``total_grains``, ``high_bc_fraction``, ``low_gkam_fraction``, ``low_kam_fraction``, ``rx_needs_bc``,
    ``rx_drawn``). Without native band contrast the criteria are remembered and no number is sent. 409 when the
    map or its grains were replaced while the call was computing (nothing is stored then).

    ``phase``: ``"most_measured"`` (the phase with the most indexed pixels, the default) or a phase id; only the
    grains of that phase can be RX and its pixels are the denominator of the RX fraction (RxxAnalysis.m,
    ``RXedGrains(ph)``). Left out it takes the stored phase. 400 for an id the map does not have (the message lists
    the ids), 422 for any other type. The answer names the phase (``rx_phase``: ``{"id", "name"}``) and /status
    carries the choice. The gKAM is made with the KAM order and threshold of the Deformation tab.
    """
    with _grain_state_lock:
        dataset = _dataset
        if dataset is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        grains = dataset.grains
        if grains is None:
            raise HTTPException(status_code=400, detail="Reconstruct grains first for the RX classification")
        criteria = dict(_rx_criteria)
        rx_phase = _rx_settings["phase"]
        deformation = dict(_deformation_settings)
        if req is not None:
            criteria.update({name: float(getattr(req, name)) for name in DEFAULT_RX_CRITERIA
                             if getattr(req, name) is not None})
            if req.phase is not None:
                rx_phase = req.phase
    try:
        rx_phase_info(dataset, rx_phase)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _check_chosen_phase_present(dataset, rx_phase, "RX")
    try:
        _, fields = _rx_panel(dataset, grains, criteria, strict=True, rx_phase=rx_phase, deformation=deformation)
    except Exception as e:
        logger.exception("RX classification failed")
        raise HTTPException(status_code=500, detail=str(e))
    with _grain_state_lock:
        # remembered only if these are still the grains the criteria were chosen for
        if _dataset is not dataset or dataset.grains is not grains:
            raise HTTPException(
                status_code=409,
                detail="The map changed while classifying; the criteria were not stored. Press Classify again.")
        _rx_criteria.update(criteria)
        _rx_settings["phase"] = rx_phase
    return {"success": True, **fields}


@router.post("/grain-size")
@_serialize_mpl
def analyze_grain_size(req: Optional[GrainSizeRequest] = None):
    """Run grain size analysis. A field left out takes the setting of the last run; the settings used are
    remembered for the Excel export and the ECD map."""
    if _dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    try:
        from analysis.grain_analysis import analyze_grain_size, grain_size_summary
        with _grain_state_lock:
            dataset, grains = _dataset, _dataset.grains
            if grains is None:
                raise HTTPException(status_code=400, detail="Reconstruct grains first")
            edge_rule = (req.edge_rule if req and req.edge_rule else _grain_stats_settings["edge_rule"])
            ecd_basis = (req.ecd_basis if req and req.ecd_basis else _grain_stats_settings["ecd_basis"])
            # a field left out keeps the last setting; for the phase an explicit null means all phases
            stats_phase = (req.stats_phase if req and "stats_phase" in req.model_fields_set
                           else _grain_stats_settings["stats_phase"])
        _check_stats_phase(stats_phase, dataset)
        try:
            results = analyze_grain_size(dataset, grains, edge_rule=edge_rule, ecd_basis=ecd_basis,
                                         stats_phase=stats_phase)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        with _grain_state_lock:
            # remembered only if these are still the grains the settings were chosen for: a reconstruction
            # that finished meanwhile has stored its own settings
            if _dataset is dataset and dataset.grains is grains:
                _grain_stats_settings.update(edge_rule=edge_rule, ecd_basis=ecd_basis, stats_phase=stats_phase)
        # Serialize GrainSizeResult
        response = {"success": True, **_edge_fields(results)}
        if hasattr(results, 'avg_ECD'):
            response["mean_ecd"] = float(results.avg_ECD)
            response["std_ecd"] = float(results.std_ECD)
        if hasattr(results, 'avg_maxDimX'):
            response["mean_maxDimX"] = float(results.avg_maxDimX)
            response["mean_maxDimY"] = float(results.avg_maxDimY)
            response["std_maxDimX"] = float(results.std_maxDimX)
            response["std_maxDimY"] = float(results.std_maxDimY)
        if hasattr(results, 'avg_AR'):
            response["mean_aspect_ratio"] = float(results.avg_AR)
            response["std_aspect_ratio"] = float(results.std_AR)
        response["area_weighted_ecd"] = float(results.area_weighted_ECD)
        response.update(grain_size_summary(results))
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
def analyze_deformation(req: Optional[DeformationRequest] = None):
    """Run deformation analysis (KAM, GB classification).

    ``kam_order`` (1 to 5), ``kam_threshold_deg`` (more than 0, at most 30) and ``hagb_min_deg`` (more than 3, at most
    90; SAGB is 3 degrees up to it, as DeformationAnalysis.m): a field left out takes the choice of the last run (2, 7
    and 15 until then); 422 when one is out of range. The choices are remembered for the KAM map, the RX analysis'
    gKAM, the Excel export and /status, and the answer carries them as ``deformation_params``."""
    with _grain_state_lock:
        dataset = _dataset
        if dataset is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        asked = req if req is not None else DeformationRequest()
        params = {name: getattr(asked, name) if getattr(asked, name) is not None else _deformation_settings[name]
                  for name in DEFAULT_DEFORMATION_PARAMS}
    try:
        from analysis.deformation_analysis import analyze_deformation
        if dataset.grains is None:
            raise HTTPException(status_code=400, detail="Reconstruct grains first")
        results = analyze_deformation(dataset, dataset.grains, kam_order=params["kam_order"],
                                      kam_threshold=params["kam_threshold_deg"], hagb_min=params["hagb_min_deg"])
        with _grain_state_lock:
            # remembered only if the map is still the one the choice was made for
            if _dataset is dataset:
                _deformation_settings.update(params)
        # Serialize DeformationResult to JSON-compatible dict
        response = {"success": True, "deformation_params": dict(params)}
        if hasattr(results, 'kam') and results.kam is not None:
            response["kam_image"] = colormap_array_to_base64(results.kam, "hot")
            import numpy as np
            valid_kam = results.kam[~np.isnan(results.kam)] if results.kam.size > 0 else results.kam
            if valid_kam.size > 0:
                response["mean_kam"] = float(np.nanmean(valid_kam))
        if getattr(results, 'total_gb_length', None) is not None:     # None: no grain, no boundary
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
            detail=(f"Band Contrast GMM: {NO_BAND_CONTRAST}. It needs band contrast from the "
                    "measurement (Oxford H5OINA or .ctf); computed Pattern Quality is not a substitute."),
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
def analyze_texture(req: Optional[TextureRequest] = None):
    """Run texture analysis (component fractions, kernel-density index and entropy, surface planes).

    ``phase``: ``"most_measured"`` (the phase with the most indexed pixels, the default) or a phase id.
    ``preset``: one of the presets the page offers (FCC_Rolling, BCC_Rolling). ``tolerance_deg``: more than 0 and
    at most 30. A field left out takes the choice of the last run. 400 for a phase id the map does not have (the
    message lists the ids) and for a phase that is not cubic (the components are cubic); 422 for any other type,
    an unknown preset (the message lists the presets) and a tolerance out of range. The choices are remembered for
    the Excel export, the component map and /status; the answer names the phase (``texture_phase``) and the preset
    and tolerance used. Like the other analysis
    routes this is a plain ``def``, so it runs in the worker pool and does not block the event loop."""
    with _grain_state_lock:
        dataset = _dataset
        if dataset is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        asked = req if req is not None else TextureRequest()
        choice = {
            "phase": asked.phase if asked.phase is not None else _texture_settings["phase"],
            "preset": asked.preset if asked.preset is not None else _texture_settings["preset"],
            "tolerance_deg": asked.tolerance_deg if asked.tolerance_deg is not None else _texture_settings["tolerance_deg"],
        }
    try:
        from analysis.texture_analysis import analyze_texture, resolve_texture_phase
        from analysis.texture_components import TEXTURE_PRESETS
        try:
            phase_id = resolve_texture_phase(dataset, choice["phase"])
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        _check_chosen_phase_present(dataset, choice["phase"], "texture")
        preset = TEXTURE_PRESETS[choice["preset"]]
        with _texture_lock:             # one run at a time; map requests do not wait for it
            results = analyze_texture(dataset, preset, texture_phase=phase_id, tolerance_deg=choice["tolerance_deg"],
                                      preset_name=choice["preset"])
        with _grain_state_lock:
            # remembered only if the map is still the one the choice was made for
            if _dataset is dataset:
                _texture_settings.update(choice)
        response = {"success": True, "texture_phase": {"id": results.phase_id, "name": results.phase_name},
                    "texture_preset": choice["preset"], "texture_tolerance_deg": choice["tolerance_deg"]}
        if hasattr(results, 'component_fractions'):
            response["components"] = [
                {"name": name, "fraction": None if frac is None else float(frac)}   # None: not buildable
                for name, frac in results.component_fractions.items()
            ]
        if hasattr(results, 'texture_index') and results.texture_index is not None:
            response["texture_index"] = float(results.texture_index)
        if hasattr(results, 'entropy') and results.entropy is not None:
            response["texture_entropy"] = float(results.entropy)
        if hasattr(results, 'plane_111_fraction'):
            # the script's single-direction numbers
            response["surface_planes"] = {
                "{111}": None if results.plane_111_fraction is None else float(results.plane_111_fraction),
                "{100}": None if results.plane_100_fraction is None else float(results.plane_100_fraction),
                "{110}": None if results.plane_110_fraction is None else float(results.plane_110_fraction),
            }
        if getattr(results, "plane_111_family_fraction", None) is not None:
            # the share of pixels with any plane of the family in the surface
            response["surface_planes_family"] = {
                "{111}": float(results.plane_111_family_fraction),
                "{100}": float(results.plane_100_family_fraction),
                "{110}": float(results.plane_110_family_fraction),
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


_WARNING_REASON_MAX = 300      # characters of one export warning (orix and sklearn messages can be long)


def _collect_export_results(warnings: Optional[list] = None) -> dict:
    """Collect all available analysis results from the loaded dataset for Excel export.

    The dataset, its grains, their reconstruction parameters and the statistics settings are read once under
    ``_grain_state_lock`` and used throughout, so the file never mixes two reconstructions. A step that
    fails is left out of the results; with ``warnings`` (a list) given it appends ``"<step>: <reason>"``,
    so the caller can tell the user which part of the workbook is empty.
    """
    import numpy as np
    results = {}

    def warn(step, exc):
        logger.warning("Excel export: %s failed: %s", step, exc)
        if warnings is not None:
            # an exception is named with its class, so an empty message still says something
            reason = f"{type(exc).__name__}: {exc}" if isinstance(exc, BaseException) else str(exc)
            if len(reason) > _WARNING_REASON_MAX:
                reason = reason[:_WARNING_REASON_MAX - 3] + "..."
            warnings.append(f"{step}: {reason}")

    with _grain_state_lock:
        dataset = _dataset
        if dataset is None:
            return results
        grains = dataset.grains
        recon = (getattr(dataset, "grain_recon_params", None) or {}) if grains is not None else {}
        settings = dict(_grain_stats_settings)
        rx_criteria = dict(_rx_criteria)
        rx_phase = _rx_settings["phase"]
        deformation_settings = dict(_deformation_settings)
        texture_settings = dict(_texture_settings)
    texture_phase = texture_settings["phase"]

    # Basic info
    shape = dataset.xmap.shape if hasattr(dataset.xmap, 'shape') else (0, 0)
    dim_y = shape[0] if len(shape) >= 1 else 0
    dim_x = shape[1] if len(shape) >= 2 else 0
    n_grains = grains.n_grains if grains else 0
    results["basic_info"] = {
        "step_micron": float(dataset.step_size),
        "dim_x": dim_x,
        "dim_y": dim_y,
        "n_grains": n_grains,
        "rotation_deg_1": 0, "rotation_deg_2": 0, "rotation_deg_3": 0,
        # what the grains were reconstructed with; empty when this session did not reconstruct them
        "gb_threshold_deg": recon.get("gb_threshold_deg", ""),
        "min_pixels": recon.get("min_pixels", ""),
        "min_intercept_um": recon.get("min_intercept_um", ""),
        "grain_alpha": recon.get("alpha", ""),
    }

    # The BC histogram, the BC fit and the RX block are calibrated on native band contrast. Without it
    # ``dataset.bc`` is a surrogate (pattern quality, scores or a constant), so these steps are left out and
    # their cells stay unwritten, as the BC routes refuse such a map.
    native_bc = dataset.has_native_bc

    # BC GMM
    if not native_bc:
        warn("bc_gmm", NO_BAND_CONTRAST)
    else:
        try:
            from analysis.bc_analysis import fit_bc_gmm as _fit_gmm, calculate_bc_histogram
            hist_result = calculate_bc_histogram(dataset)
            results["bc_histogram"] = hist_result
            gmm_result = _fit_gmm(dataset)
            if gmm_result is not None:
                results["bc_gmm"] = gmm_result
            else:
                warn("bc_gmm", "too few points for the fit")
        except Exception as e:
            warn("bc_gmm", e)

    if grains is None:
        warn("grains", "not reconstructed in this session; the grain-based sheets are not written")
        return results

    # Grain size
    try:
        from analysis.grain_analysis import analyze_grain_size, grain_size_basic_info
        gs = analyze_grain_size(dataset, grains, edge_rule=settings["edge_rule"],
                                ecd_basis=settings["ecd_basis"],
                                stats_phase=settings["stats_phase"])
        results["grain_size"] = gs
        results["basic_info"].update(grain_size_basic_info(gs))
    except Exception as e:
        warn("grain_size", e)

    # Deformation
    try:
        from analysis.deformation_analysis import analyze_deformation
        deform = analyze_deformation(dataset, grains, kam_order=deformation_settings["kam_order"],
                                     kam_threshold=deformation_settings["kam_threshold_deg"],
                                     hagb_min=deformation_settings["hagb_min_deg"])
        results["deformation"] = deform
    except Exception as e:
        warn("deformation", e)

    # Texture
    preset = {}
    try:
        from analysis.texture_analysis import analyze_texture
        from analysis.texture_components import TEXTURE_PRESETS
        preset = TEXTURE_PRESETS[texture_settings["preset"]]
        with _texture_lock:
            tx = analyze_texture(dataset, preset, texture_phase=texture_phase,
                                 tolerance_deg=texture_settings["tolerance_deg"], preset_name=texture_settings["preset"])
        results["texture"] = tx
    except Exception as e:
        warn("texture", e)

    # RX: the same analysis as the batch, so every number of the block (including the two low-KAM fractions)
    # comes from the RX result and none is a placeholder
    if not native_bc:
        warn("rx", NO_BAND_CONTRAST)
    elif grains.n_grains == 0:
        # analyze_recrystallization would return all-zero fractions that read like measurements
        warn("rx", "the reconstruction found no grains, so there is nothing to classify")
    elif "deformation" not in results:
        warn("rx", "needs the KAM of the deformation step, which failed")
    else:
        try:
            from analysis.rx_analysis import analyze_recrystallization
            results["rx"] = analyze_recrystallization(
                dataset, grains, results["deformation"].kam,
                gos_threshold_deg=rx_criteria["rx_gos_threshold"], gbc_min_fraction=rx_criteria["rx_gbc_fraction"],
                gkam_threshold_deg=rx_criteria["rx_gkam_threshold"],
                min_grain_radius_um=rx_criteria["rx_min_radius_um"], rx_phase=rx_phase)
        except Exception as e:
            warn("rx", e)

    # Texture of the RX grains and of the substructure grains: the sheet "SmallVSLargeGrain" (same components
    # and tolerance as the texture step above)
    if "texture" in results and "rx" in results:
        from analysis.texture_analysis import split_phase_conflict
        conflict = split_phase_conflict(results["rx"], results["texture"])
        if conflict:
            # the split counts texture-phase pixels in RX grains: with another RX phase it would be wrong, not empty
            warn("SmallVSLargeGrain", conflict)
            results["texture_by_rx_left_out"] = True
        else:
            try:
                from analysis.texture_analysis import calculate_texture_split_by_rx
                results["texture_by_rx"] = calculate_texture_split_by_rx(
                    dataset, grains, results["rx"].rx_grain_ids, preset, texture_settings["tolerance_deg"],
                    texture_phase=texture_phase)
            except Exception as e:
                warn("texture_by_rx", e)

    return results


def _requested_name_has_content(requested) -> bool:
    """True if a workbook with content is at the requested name ("already existed"). False: the name was taken
    by a run that is writing right now (an empty reservation), not by a workbook."""
    try:
        return Path(requested).stat().st_size > 0
    except OSError:
        return False


@router.post("/export/excel")
async def export_excel(req: ExportExcelRequest, background_tasks: BackgroundTasks):
    """Export analysis results to Excel."""
    if _dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    _check_stored_phases_present(_dataset)

    task_id = str(uuid.uuid4())
    _track_task(task_id, {"status": "running", "progress": 0.0})

    def run():
        target = None
        try:
            from analysis.excel_exporter import export_to_excel, free_workbook_path, release_workbook_reservation
            warnings: list = []
            results = _collect_export_results(warnings)
            # an existing workbook is never opened or overwritten: a second export goes to <name>_2.xlsx
            target = free_workbook_path(req.output_path)
            export_to_excel(str(target), req.dataset_name, results)
            task = _analysis_tasks[task_id]
            task["progress"] = 1.0
            # path: the file that was written; requested_path: what the request asked for (the same unless it
            # already existed); warnings: the steps that failed and are therefore empty ("<step>: <reason>")
            # both as the OS spells them: "C:/o/a.xlsx" and "C:\o\a.xlsx" are one file, not a rename
            requested = Path(req.output_path)
            task["result"] = {"path": str(target), "requested_path": str(requested), "warnings": warnings}
            release_workbook_reservation(target)             # written: the name is no longer a reservation
            if Path(target) != requested:
                why = "already exists" if _requested_name_has_content(requested) else "is in use by another run"
                task["message"] = f"{requested} {why}: the export was written to {target}"
            task["status"] = "completed"
        except Exception as e:
            text = str(e)
            if target is not None:
                from analysis.excel_exporter import incomplete_workbook_note, release_workbook_reservation
                text += incomplete_workbook_note(target)     # a half-written file must not pass as a workbook
                release_workbook_reservation(target)         # nothing was written: the reserved name goes back
            _analysis_tasks[task_id]["progress"] = 1.0
            _analysis_tasks[task_id]["error"] = text         # before the status: a poller never sees "failed" without it
            _analysis_tasks[task_id]["status"] = "failed"

    background_tasks.add_task(run)
    return {"task_id": task_id}


@router.get("/export/excel-download")
def export_excel_download(name: str = "Dataset"):
    """Generate Excel in a temp file and return as a browser download."""
    if _dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    _check_stored_phases_present(_dataset)
    try:
        import tempfile, os
        from analysis.excel_exporter import export_to_excel

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = tmp.name

        warnings: list = []
        try:
            results = _collect_export_results(warnings)
            export_to_excel(tmp_path, name, results)
            with open(tmp_path, "rb") as fh:
                data = fh.read()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        headers = {"Content-Disposition": 'attachment; filename="EBSD_Analysis.xlsx"'}
        if warnings:
            # a file answer has no JSON body: the failed steps travel in a header, as a JSON list of
            # "<step>: <reason>" strings, percent-encoded (url-decode, then JSON-parse)
            headers["X-Orienta-Warnings"] = urllib.parse.quote(json.dumps(warnings), safe="")
        return StreamingResponse(
            io.BytesIO(data),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=headers,
        )
    except Exception as e:
        logger.exception("Failed to generate Excel for download")
        raise HTTPException(status_code=500, detail=str(e))


STEP_SIZE_NOTE_PREFIX = "step_size:"      # the batch warning for a file without a recorded step size


def _cut(text, limit=_WARNING_REASON_MAX) -> str:
    """``text`` cut to ``limit`` characters (the last three are "...")."""
    text = str(text)
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _batch_reasons(items, limit=3) -> str:
    """``name (reason)`` for the first few files of a failed or skipped list; a reason is cut like a warning."""
    shown = [f"{Path(item['file']).name} ({_cut(item['reason'])})" for item in items[:limit]]
    if len(items) > limit:
        shown.append(f"{len(items) - limit} more")
    return "; ".join(shown)


def _batch_left_out(warnings, limit=3) -> str:
    """``name: step, step`` for the first few files that had analysis steps left out."""
    by_file: dict = {}
    for item in warnings:
        by_file.setdefault(item["file"], []).append(item["reason"].split(":", 1)[0])
    shown = [f"{Path(file).name}: {', '.join(steps)}" for file, steps in list(by_file.items())[:limit]]
    if len(by_file) > limit:
        shown.append(f"{len(by_file) - limit} more")
    return "; ".join(shown)


def _batch_failure_text(folder_path, outcome) -> str:
    """The message of a batch in which no file gave a result."""
    if outcome.n_files == 0:
        return f"No EBSD files (.h5, .h5oina, .ang, .ctf) found in {folder_path}"
    parts = []
    if outcome.failed:
        parts.append(f"{len(outcome.failed)} failed: {_batch_reasons(outcome.failed)}")
    if outcome.skipped:
        parts.append(f"{len(outcome.skipped)} skipped: {_batch_reasons(outcome.skipped)}")
    return (f"None of the {outcome.n_files} files gave a result, no workbook was written. " + ". ".join(parts))


def _batch_summary_text(outcome) -> str:
    """The completion message of a batch: how many files went into the workbook and what happened to the rest."""
    text = f"Processed {outcome.n_processed} of {outcome.n_files} files"
    rest = []
    if outcome.skipped:
        rest.append(f"{len(outcome.skipped)} skipped ({_batch_reasons(outcome.skipped)})")
    if outcome.failed:
        rest.append(f"{len(outcome.failed)} failed ({_batch_reasons(outcome.failed)})")
    # the step-size note says an assumption was made, not that a step is missing: it stays in the warnings list only
    left_out = [item for item in outcome.warnings if not item["reason"].startswith(STEP_SIZE_NOTE_PREFIX)]
    if left_out:
        n_files = len({item["file"] for item in left_out})
        rest.append(f"{n_files} file(s) with analysis steps left out ({_batch_left_out(left_out)})")
    return text + (": " + ", ".join(rest) if rest else "") + "."


@router.post("/batch")
async def batch_process(req: BatchProcessRequest, background_tasks: BackgroundTasks):
    """Run batch processing on a folder of files."""
    task_id = str(uuid.uuid4())
    _track_task(task_id, {"status": "running", "progress": 0.0})

    def run():
        try:
            from analysis.batch_processor import batch_process_folder
            running = _analysis_tasks.get(task_id)

            def on_progress(message, current, total):
                # only the per-file calls carry a total; the step messages inside a file have -1
                if running is not None and total > 0 and message.startswith("Processing file"):
                    running["current_file"] = message
                    running["progress"] = max(0.0, min(0.99, (current - 1) / total))

            options = {key: value for key, value in (("grain_edge_rule", req.grain_edge_rule),
                                                     ("grain_ecd_basis", req.grain_ecd_basis),
                                                     ("grain_stats_phase", req.grain_stats_phase),
                                                     ("texture_phase", req.texture_phase),
                                                     ("texture_preset", req.texture_preset),
                                                     ("tolerance_deg", req.texture_tolerance_deg),
                                                     ("gb_threshold_deg", req.misorientation_threshold),
                                                     ("min_grain_pixels", req.min_grain_size),
                                                     ("grain_alpha", req.alpha),
                                                     ("grain_fill", req.fill),
                                                     ("min_intercept_um", req.min_intercept_um),
                                                     ("rx_gos_threshold", req.rx_gos_threshold),
                                                     ("rx_gbc_fraction", req.rx_gbc_fraction),
                                                     ("rx_gkam_threshold", req.rx_gkam_threshold),
                                                     ("rx_min_radius_um", req.rx_min_radius_um),
                                                     ("rx_phase", req.rx_phase),
                                                     ("kam_order", req.kam_order),
                                                     ("kam_threshold_deg", req.kam_threshold_deg),
                                                     ("hagb_min_deg", req.hagb_min_deg))
                       if value is not None}
            outcome = batch_process_folder(req.folder_path, req.output_path, progress_callback=on_progress, **options)
            task = _analysis_tasks[task_id]
            task["result"] = outcome.to_dict()
            task["progress"] = 1.0
            task["current_file"] = None                       # the batch is over: no file is "current" any more
            if outcome.save_error:
                task["error"] = outcome.save_error
                task["status"] = "failed"
            elif outcome.n_processed == 0:
                # no workbook was written: say why instead of announcing a finished batch
                task["error"] = _batch_failure_text(req.folder_path, outcome)
                task["status"] = "failed"
            else:
                text = _batch_summary_text(outcome)
                if outcome.workbook_path:
                    renamed = Path(outcome.workbook_path) != Path(req.output_path)
                    why = ("already existed" if _requested_name_has_content(req.output_path)
                           else "was in use by another run")
                    text += f" Workbook: {outcome.workbook_path}" + (
                        f" ({Path(req.output_path).name} {why}, nothing was overwritten)." if renamed else ".")
                task["message"] = text
                task["status"] = "completed"
        except Exception as e:
            _analysis_tasks[task_id]["progress"] = 1.0
            _analysis_tasks[task_id]["current_file"] = None
            _analysis_tasks[task_id]["error"] = str(e)       # before the status: a poller never sees "failed" without it
            _analysis_tasks[task_id]["status"] = "failed"

    background_tasks.add_task(run)
    return {"task_id": task_id}
