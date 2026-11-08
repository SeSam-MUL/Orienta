"""REST API for the multi-phase batch indexing system (v2).

Provides endpoints for:
- Scanning folders for EBSD files
- Quick-loading files (metadata only, no full pattern cache)
- PC status queries and copying
- Batch creation, execution, and monitoring
"""
import asyncio
import gc
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.api.problem import problem
from backend.api.services.batch_manager import BatchManager
from backend.api.services.calibration_store import calibration_store
from display_names import display_stem

logger = logging.getLogger(__name__)
router = APIRouter()

# Module-level batch manager (singleton)
_manager: BatchManager = None

# Extensions considered as EBSD data files (.up1/.up2: EDAX pattern files, placed by the APEX metadata beside them)
_EBSD_EXTENSIONS = {'.h5oina', '.h5', '.hdf5', '.hdf', '.ebsp', '.up1', '.up2'}

# Lightweight signal cache for quick-loaded files (avoid full loading)
_quick_loaded: Dict[str, Any] = {}


def _get_manager() -> BatchManager:
    global _manager
    if _manager is None:
        _manager = BatchManager()
    return _manager


# -----------------------------------------------------------------------
# Folder scanning
# -----------------------------------------------------------------------

class ScanFolderRequest(BaseModel):
    folder: str
    recursive: bool = True


@router.post("/scan-folder")
async def scan_folder(req: ScanFolderRequest):
    """Scan a folder for EBSD data files (.h5oina, .h5, .hdf5, .up1, .up2, etc.).

    Returns file list with basic metadata (size, extension). An EDAX APEX metadata file (an ``.h5``
    that lists measurements and holds no patterns) is not a scan and is left out: its pattern files
    (.up1/.up2) are the scans.
    """
    from edax_apex import is_apex_file
    folder = Path(req.folder)
    if not folder.is_dir():
        raise HTTPException(400, f"Not a valid directory: {req.folder}")
    pattern = '**/*' if req.recursive else '*'
    found = []
    for p in sorted(folder.glob(pattern)):
        if not p.is_file() or p.suffix.lower() not in _EBSD_EXTENSIONS:
            continue
        # Skip checkpoint files from previous batch runs
        if '_multiphase' in p.stem or '_checkpoint' in p.stem:
            continue
        if is_apex_file(p):
            continue
        found.append({
                "path": str(p),
                "name": p.stem,
                "extension": p.suffix.lower(),
                "size_mb": round(p.stat().st_size / (1024 * 1024), 1),
            })
    return {"files": [f["path"] for f in found], "details": found, "count": len(found)}


# -----------------------------------------------------------------------
# Quick-load: load signal metadata + register in CalibrationStore
# -----------------------------------------------------------------------

class QuickLoadRequest(BaseModel):
    file_path: str


def _inherit_viewer_calibration(dataset_name: str, fp: Path) -> None:
    """Batch v2 reads the processed patterns, so its entry is its own; when the viewer holds the same
    measurement as another pattern source, the PC the user made there (typed, refined) describes the
    same acquisition and is handed over, marked ``inherited`` from the viewer's entry (the way copy-pc
    marks it). Another measurement has a PC of its own and hands nothing over. The header PC stays when
    the user made none. Only the single PC travels; a per-pixel map does not."""
    from backend.api.services.batch_manager import holds_default_measurement, variant_entry_of_file
    parent = variant_entry_of_file(fp, calibration_store)
    if parent is None or not holds_default_measurement(parent, fp):
        return
    source = calibration_store.get_entry(parent)
    own = calibration_store.get_entry(dataset_name)
    if (source is None or own is None or source.pc_source not in calibration_store.USER_MADE_SOURCES
            or tuple(int(v) for v in source.detector_shape) != tuple(int(v) for v in own.detector_shape)):
        return
    calibration_store.update_pc(dataset_name, source.pc_single, source="inherited",
                                refinement=source.pc_refinement)
    own.parent_name = parent


@router.post("/quick-load")
async def quick_load(req: QuickLoadRequest):
    """Quick-load an EBSD file: extract metadata and register PC in CalibrationStore.

    Does NOT build a full pattern cache. Just enough to get grid shape,
    pattern shape, PC, and EDS element availability.
    """
    from edax_apex import ApexError, safe_text
    from edax_up1 import EdaxHexUpError
    fp = Path(req.file_path)
    if not fp.is_file():
        raise HTTPException(400, f"File not found: {req.file_path}")

    # The name THIS FILE has in the store: its stem, or stem + path hash when an entry of
    # that stem belongs to another file (the viewer's, say). The answer carries it, and the
    # page uses it for pc-status and copy-pc.
    from backend.api.services.batch_manager import store_name_for_file
    dataset_name = store_name_for_file(fp)
    file_key = os.path.normcase(os.path.abspath(str(fp)))

    try:
        from safe_loader import load_ebsd_safe
        signal = load_ebsd_safe(str(fp))

        # What the page shows for the file: the grid the loader read and the APEX measurement it was
        # paired with. Read before the loader's copy of the geometry is taken off (a kept signal must
        # not carry it: it describes the whole scan and goes stale on a crop). The run loads the file
        # again by path and pairs it the same way (by the GUID in its header), so what is shown here is
        # what the run uses; the viewer's remembered pairing plays no part.
        from backend.api.services.scan_geometry import apex_of_signal, drop_signal_mirror, geometry_of_signal
        geometry = geometry_of_signal(signal)
        apex = apex_of_signal(signal)
        drop_signal_mirror(signal)

        # Register in CalibrationStore - but never over a calibration the user made
        # (typed, refined, a grid map) for THIS file: quick-load is only a look at the
        # file. What the file itself gave (header, missing) is refreshed: the file may
        # have changed (an .osc put beside it). An entry of another file is never touched
        # or shown: this file has its own name.
        existing = calibration_store.get_entry(dataset_name)
        if (existing is not None and existing.file_key == file_key
                and existing.pc_source in calibration_store.USER_MADE_SOURCES):
            cal_entry = existing
        else:
            cal_entry = calibration_store.register(dataset_name, signal, file_key=file_key)
            _inherit_viewer_calibration(dataset_name, fp)

        # Extract grid shape
        nav_shape = signal.axes_manager.navigation_shape[::-1]  # (rows, cols)
        sig_shape = signal.axes_manager.signal_shape[::-1]  # (h, w)

        # Detect EDS availability via H5 viewer backend
        has_eds = False
        eds_elements = []
        try:
            from tools.h5_viewer_backend import H5OINADataExtractor
            extractor = H5OINADataExtractor(str(fp))
            eds_elements = extractor.get_available_elements() or []
            has_eds = len(eds_elements) > 0
        except Exception:
            pass

        # Get PC info
        pc_value = None
        pc_source = "none"
        if cal_entry and cal_entry.pc_source != "missing":
            pc_value = [float(v) for v in cal_entry.pc_single]
            pc_source = cal_entry.pc_source
        # A file whose PC nobody supplied ("missing": kikuchipy's placeholder is all
        # there is) is "none" to the batch page, with no PC to show or to tick.

        # Store lightweight reference for later use
        _quick_loaded[dataset_name] = {
            "file_path": str(fp),
            "signal": signal,
            "grid_shape": list(nav_shape),
            "pattern_shape": list(sig_shape),
        }

        # The convention the file's own vendor writes PC numbers in ("oxford", "tsl", "bruker"),
        # as the PC-Refinement page's header already reads it — for the Batch page to preselect
        # the same convention in its "set PC manually" dialog, instead of assuming Bruker.
        convention = getattr(cal_entry, "convention", None) if cal_entry else None
        convention_known = bool(convention) and bool(getattr(cal_entry, "convention_known", False))

        return {
            "dataset_name": dataset_name,
            "file_path": str(fp),
            "grid_shape": list(nav_shape),
            "pattern_shape": list(sig_shape),
            "pattern_count": int(signal.axes_manager.navigation_size),
            "pc": pc_value,
            "pc_source": pc_source,
            "convention": convention or "bruker",
            "convention_known": convention_known,
            "has_eds": has_eds,
            "eds_elements": eds_elements,
            "grid_kind": geometry.kind,
            "scan_geometry": geometry.to_dict(),
            "apex": apex,
        }

    except (ApexError, EdaxHexUpError) as e:
        # A coded refusal (a hexagonal pattern file no metadata names, an APEX metadata file added as
        # a scan): the code travels in X-Orienta-Code and the page words it in its own language.
        raise problem(400, e.code, safe_text(e))
    except Exception as e:
        logger.exception("Quick-load failed for %s", req.file_path)
        raise HTTPException(500, f"Failed to load: {e}")


# -----------------------------------------------------------------------
# PC management for batch workflows
# -----------------------------------------------------------------------

class CopyPCRequest(BaseModel):
    source_dataset: str
    target_datasets: List[str]
    #: The files themselves (paths), parallel to ``source_dataset`` / ``target_datasets``.
    #: Required: the entry is the one registered for THAT file. A stem names two files when
    #: two folders hold a scan of one name. (Optional in the model only so that a request
    #: without them gets the coded 400 instead of a bare 422.)
    source_file_path: Optional[str] = None
    target_file_paths: Optional[List[str]] = None


def _resolve_dataset_name(name: str) -> str:
    """Accept either a bare dataset_name (file stem) or a full filename
    (e.g. ``Sample.h5oina``) and return the canonical CalibrationStore key
    which is the path stem.

    The Batch frontend stores files as ``file_name`` *with* extension;
    CalibrationStore registers by stem. Without this normalisation every
    /copy-pc call from BatchPage 404'd because the lookup never matched.
    """
    if not name:
        return name
    # display_stem, not Path.stem: this key has to come out the same as the
    # one CalibrationStore was registered under, on every platform. See
    # display_names.py.
    return display_stem(name)


@router.post("/copy-pc")
async def copy_pc(req: CopyPCRequest):
    """Copy PC from one dataset to one or more target datasets.

    The files are addressed by PATH (``source_file_path``, ``target_file_paths``), never by
    name: a request without them is refused (400, code ``copy_pc_needs_files``). Source and
    targets must already be registered in CalibrationStore (via quick-load).
    """
    from backend.api.problem import problem
    from backend.api.services.batch_manager import default_variant_entry_of_file
    paths =req.target_file_paths or []
    if (not req.source_file_path or len(paths) != len(req.target_datasets)
            or not all(paths)):
        raise problem(
            400, "copy_pc_needs_files",
            "copy-pc addresses files, not names: send source_file_path and one entry of "
            "target_file_paths per target (two files with one stem share a name).")
    # batch v2 reads the default variant: the entry of those patterns, not the viewer's entry of
    # another variant of the same file
    source_key = default_variant_entry_of_file(req.source_file_path, calibration_store)
    if source_key is None:
        raise HTTPException(
            404,
            f"Source file for '{req.source_dataset}' is not in the CalibrationStore. "
            "Quick-load the file first via /api/batch-v2/quick-load.")
    source_pc = calibration_store.get_pc(source_key)
    source_entry = calibration_store.get_entry(source_key)
    if source_pc is None and source_entry is not None:
        # Registered, but the file carries no PC (the store answers None for it):
        # copying the placeholder to other files would spread a made-up PC.
        raise HTTPException(
            400,
            f"Source dataset '{req.source_dataset}' has no pattern centre (its file "
            "carries none). Set or refine its PC first, then copy it.",
        )
    if source_pc is None:
        raise HTTPException(
            404,
            f"Source dataset '{req.source_dataset}' (key={source_key!r}) not in "
            "CalibrationStore. Quick-load the file first via /api/batch-v2/quick-load.",
        )

    results = []
    for target, target_path in zip(req.target_datasets, paths):
        # the entry registered for this FILE; never the one a stem happens to name
        target_key = default_variant_entry_of_file(target_path, calibration_store)
        if target_key is not None and target_key == source_key:
            # a file is not copied onto itself (it would become its own parent)
            results.append({
                "dataset": target, "key": target_key, "success": True, "skipped": True,
                "pc": [float(v) for v in source_pc],
                "note": "this is the source file: nothing to copy",
            })
            continue
        if target_key is None:
            results.append({
                "dataset": target, "key": None, "success": False,
                "error": "this file is not in the store (quick-load it first)",
            })
            continue
        entry = calibration_store.get_entry(target_key)
        if entry is None:
            results.append({
                "dataset": target,
                "key": target_key,
                "success": False,
                "error": f"not in store (need quick-load for key={target_key!r})",
            })
            continue
        calibration_store.update_pc(
            target_key, source_pc, source="inherited",
            refinement=(source_entry.pc_refinement if source_entry else None))
        entry = calibration_store.get_entry(target_key)
        # A family is a tree. When the target is an ancestor of the source (the source
        # inherited from it, directly or through others: the copy back after "refine B,
        # copy to A"), the source becomes the root first; otherwise parent_name would
        # form a cycle and the family walks, which start at the refined file, would
        # disagree about the root and miss copies of the other file.
        if calibration_store.is_ancestor(target_key, source_key):
            calibration_store.detach_from_parent(source_key)
        entry.parent_name = source_key
        results.append({
            "dataset": target,
            "key": target_key,
            "success": True,
            "pc": [float(v) for v in source_pc],
        })

    return {
        "source": req.source_dataset,
        "source_key": source_key,
        "pc": [float(v) for v in source_pc],
        "results": results,
    }


class SetPCRequest(BaseModel):
    file_path: str
    #: Bruker convention, already converted (the frontend does the conversion the same way the
    #: PC-Refinement page does, via /api/pc/convert — never a second formula here).
    pc: List[float]


@router.post("/set-pc")
async def set_pc(req: SetPCRequest):
    """Set a dataset's PC by typing it in, addressed by file path like ``copy-pc``.

    Two files that share a stem are two different entries (see ``copy-pc``); addressing by name
    would silently write the typed PC onto whichever of them the store happens to answer for,
    the same hazard ``copy-pc`` was given a file path to avoid. The file must already be
    registered in CalibrationStore (quick-load it first).
    """
    if len(req.pc) != 3:
        raise HTTPException(400, "PC must have exactly 3 values [pcx, pcy, pcz]")
    from backend.api.services.batch_manager import default_variant_entry_of_file
    key = default_variant_entry_of_file(req.file_path, calibration_store)
    if key is None:
        raise HTTPException(
            404,
            f"'{req.file_path}' is not in the CalibrationStore. "
            "Quick-load the file first via /api/batch-v2/quick-load.")
    # on_refined_map="shift": a file with a per-pixel grid-calibrated PC map must not keep
    # running on that map's old mean while the page tells the user their typed PC is in use —
    # the map moves rigidly so its mean IS the typed PC, the same choice the PC-Refinement
    # page's own manual-PC endpoint makes (see detector/update-pc's _writeback_refined_pc).
    calibration_store.update_pc(key, req.pc, source="manual", on_refined_map="shift")
    entry = calibration_store.get_entry(key)
    return {"success": True, "dataset": key, "pc": [float(v) for v in entry.pc_single]}


@router.get("/pc-status")
async def pc_status(files: str = ""):
    """Get PC status for multiple datasets.

    Query param `files` is a comma-separated list of dataset names (file stems).
    If empty, returns all entries in CalibrationStore.
    """
    if files:
        names = [n.strip() for n in files.split(",") if n.strip()]
    else:
        names = list(calibration_store.get_all().keys())

    results = []
    for name in names:
        entry = calibration_store.get_entry(name)
        if entry:
            no_pc = entry.pc_source == "missing"
            results.append({
                "dataset_name": name,
                "pc": None if no_pc else [float(v) for v in entry.pc_single],
                "pc_source": "none" if no_pc else entry.pc_source,
                "parent_name": entry.parent_name,
                "detector_shape": list(entry.detector_shape),
                "sample_tilt": entry.sample_tilt,
            })
        else:
            results.append({
                "dataset_name": name,
                "pc": None,
                "pc_source": "none",
                "parent_name": None,
                "detector_shape": None,
                "sample_tilt": None,
            })
    return results


@router.post("/unload")
async def unload_quick_loaded(dataset_name: str = ""):
    """Unload a quick-loaded signal to free memory."""
    if dataset_name and dataset_name in _quick_loaded:
        del _quick_loaded[dataset_name]
        gc.collect()
        return {"unloaded": dataset_name}
    elif not dataset_name:
        count = len(_quick_loaded)
        _quick_loaded.clear()
        gc.collect()
        return {"unloaded_all": count}
    return {"error": "not found"}


class CreateBatchRequest(BaseModel):
    files: List[str]
    phases: List[Dict[str, str]]  # [{name, path, method}]
    config: Dict[str, Any] = {}


@router.post("/create")
async def create_batch(req: CreateBatchRequest):
    if not req.files:
        raise HTTPException(400, "No files provided")
    if not req.phases:
        raise HTTPException(400, "No phases provided")
    mgr = _get_manager()
    batch_id, report = mgr.create_batch(req.files, req.phases, req.config)
    return {
        "batch_id": batch_id,
        "preflight": {
            "checks": report.checks,
            "can_start": report.can_start,
            "warnings": report.warnings,
            "estimated_disk_gb": report.estimated_disk_gb,
        },
        "total_jobs": len(req.files) * len(req.phases),
    }


@router.post("/{batch_id}/start")
async def start_batch(batch_id: str):
    mgr = _get_manager()
    status = mgr.get_status(batch_id)
    if status is None:
        raise HTTPException(404, "Batch not found")
    if status["status"] == "running":
        raise HTTPException(409, "Batch already running")

    # A file already registered without a PC would be indexed at the placeholder. Each
    # file by its own name: files that share a stem with another file of the batch have no
    # entry of their own yet (the store holds one entry per stem, and it may be the other
    # file's); they are decided when they are loaded, from their own header.
    from backend.api.services.batch_manager import assign_batch_dataset_names
    from backend.api.services.pc_guard import require_pc
    jobs = mgr.get_jobs(batch_id)
    names = assign_batch_dataset_names([j["file_path"] for j in jobs])
    for job in jobs:
        key = os.path.normcase(os.path.abspath(str(job["file_path"])))
        require_pc(names[key][0], calibration_store, file_key=key)

    def run():
        mgr.run_batch_sync(batch_id)

    asyncio.get_event_loop().run_in_executor(None, run)
    return {"status": "started", "batch_id": batch_id}


@router.get("/{batch_id}/exports")
async def batch_exports(batch_id: str):
    """What each file of the batch exported, and under which name suffix (two files with
    the same stem get ``__<hash>`` in their output names so they do not overwrite each
    other)."""
    mgr = _get_manager()
    if mgr.get_status(batch_id) is None:
        raise HTTPException(404, "Batch not found")
    return {"batch_id": batch_id,
            "exports": [{"file_path": fp, **info}
                        for fp, info in (getattr(mgr, "exports", {}).get(batch_id) or {}).items()]}


@router.post("/{batch_id}/pause")
async def pause_batch(batch_id: str):
    mgr = _get_manager()
    mgr.pause()
    return {"status": "pausing"}


@router.post("/{batch_id}/resume")
async def resume_batch(batch_id: str):
    mgr = _get_manager()
    status = mgr.get_status(batch_id)
    if status is None:
        raise HTTPException(404, "Batch not found")

    def run():
        mgr.resume()
        mgr.run_batch_sync(batch_id)

    asyncio.get_event_loop().run_in_executor(None, run)
    return {"status": "resumed"}


@router.post("/{batch_id}/stop")
async def stop_batch(batch_id: str):
    mgr = _get_manager()
    mgr.stop()
    return {"status": "stopping"}


@router.get("/{batch_id}/status")
async def get_batch_status(batch_id: str):
    mgr = _get_manager()
    status = mgr.get_status(batch_id)
    if status is None:
        raise HTTPException(404, "Batch not found")
    memory = mgr.guardian.get_current_usage()

    # Surface what preprocessing actually ran, aggregated across files.
    # The dashboard displays this so the user can confirm that e.g. their
    # "frame averaging 3x3" actually took effect and wasn't silently skipped.
    applied_summary: Dict[str, set] = {}
    try:
        from backend.api.services.checkpoint_writer import CheckpointWriter
        jobs = mgr.get_jobs(batch_id)
        for fp in {j.get("file_path") for j in jobs if j.get("file_path")}:
            cw = CheckpointWriter(fp)
            meta = cw.get_preprocessing_meta()
            if not meta:
                continue
            for k, v in (meta.get("applied") or {}).items():
                applied_summary.setdefault(k, set()).add(str(v))
    except Exception:
        pass
    preprocessing = {k: sorted(list(v)) for k, v in applied_summary.items()}

    return {**status, "memory": memory, "preprocessing_applied": preprocessing}


@router.get("/{batch_id}/jobs")
async def get_batch_jobs(batch_id: str):
    mgr = _get_manager()
    jobs = mgr.get_jobs(batch_id)
    return jobs


@router.post("/{batch_id}/retry-failed")
async def retry_failed(batch_id: str):
    mgr = _get_manager()
    count = mgr.queue.retry_failed(batch_id)
    return {"retried": count}


@router.delete("/{batch_id}")
async def delete_batch(batch_id: str):
    mgr = _get_manager()
    mgr.queue.delete_batch(batch_id)
    return {"deleted": True}


@router.get("/list")
async def list_batches():
    mgr = _get_manager()
    return mgr.queue.list_batches()


@router.get("/{batch_id}/report")
async def get_batch_report(batch_id: str):
    """Generate an HTML report for a completed batch.

    All user-controlled values (file names, phase names, error messages,
    batch_id from URL) are HTML-escaped to prevent XSS via crafted filenames.
    Status comes from a closed enum, but it's still escaped defensively.
    """
    from html import escape
    from fastapi.responses import HTMLResponse
    mgr = _get_manager()
    status = mgr.get_status(batch_id)
    if status is None:
        raise HTTPException(404, "Batch not found")
    jobs = mgr.get_jobs(batch_id)

    done_jobs = [j for j in jobs if j["status"] == "done"]
    total_duration = sum(j.get("duration_sec", 0) or 0 for j in done_jobs)

    # Collect preprocessing info from each file's checkpoint
    preprocessing_section = ""
    seen_files = set()
    for j in jobs:
        fp = j.get("file_path")
        if fp in seen_files or not fp:
            continue
        seen_files.add(fp)
        try:
            from backend.api.services.checkpoint_writer import CheckpointWriter
            cw = CheckpointWriter(fp)
            meta = cw.get_preprocessing_meta()
            if meta and meta.get("applied"):
                applied_str = ", ".join(
                    f"{escape(str(k))}={escape(str(v))}" for k, v in meta["applied"].items()
                ) or "(none)"
                preprocessing_section += (
                    f'<tr><td>{escape(str(j.get("file_name", "")))}</td>'
                    f'<td>{applied_str}</td></tr>'
                )
        except Exception:
            pass
    preprocessing_html = ""
    if preprocessing_section:
        preprocessing_html = (
            '<h2>Preprocessing Applied</h2>'
            '<table><thead><tr><th>File</th><th>Steps</th></tr></thead>'
            f'<tbody>{preprocessing_section}</tbody></table>'
        )

    rows_html = ""
    for j in jobs:
        ci_str = f'{j["ci_mean"]:.3f}' if j.get("ci_mean") is not None else "&mdash;"
        dur_str = f'{j["duration_sec"]:.0f}s' if j.get("duration_sec") else "&mdash;"
        status_color = "#50fa7b" if j["status"] == "done" else "#ff5555" if j["status"] == "failed" else "#6272a4"
        err_msg = j.get("error_msg") or ""
        err_str = f'<br><small style="color:#ff5555">{escape(err_msg)}</small>' if err_msg else ""
        rows_html += f"""
        <tr>
          <td>{escape(str(j.get("file_name", "")))}</td>
          <td>{escape(str(j.get("phase_name", "")))}</td>
          <td style="color:{status_color};font-weight:bold">{escape(str(j.get("status", "")))}</td>
          <td style="text-align:right">{ci_str}</td>
          <td style="text-align:right">{dur_str}</td>
          <td>{err_str}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Batch Report</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #1e1f29; color: #f8f8f2; padding: 24px; }}
  h1 {{ color: #bd93f9; }} h2 {{ color: #8be9fd; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #44475a; padding: 6px 10px; text-align: left; font-size: 10pt; }}
  th {{ background: #282a36; color: #8be9fd; }}
  .stat {{ display: inline-block; margin-right: 24px; }}
  .stat b {{ color: #50fa7b; }}
</style></head><body>
<h1>Batch Report</h1>
<p>Batch ID: <code>{escape(batch_id[:8])}...</code> &mdash; Status: <b>{escape(str(status.get("status", "")))}</b></p>
<div>
  <span class="stat">Total: <b>{int(status.get("total_jobs", 0))}</b></span>
  <span class="stat">Done: <b>{int(status.get("completed", 0))}</b></span>
  <span class="stat">Failed: <b style="color:#ff5555">{int(status.get("failed", 0))}</b></span>
  <span class="stat">Duration: <b>{total_duration/60:.1f} min</b></span>
</div>
{preprocessing_html}
<h2>Jobs</h2>
<table>
  <thead><tr><th>File</th><th>Phase</th><th>Status</th><th>CI</th><th>Time</th><th>Error</th></tr></thead>
  <tbody>{rows_html}</tbody>
</table>
<p style="color:#6272a4;font-size:9pt;margin-top:24px">Generated by Orienta Multi-Phase Batch System</p>
</body></html>"""

    return HTMLResponse(content=html)
