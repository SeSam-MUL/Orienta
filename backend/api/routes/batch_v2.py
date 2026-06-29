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
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.api.services.batch_manager import BatchManager
from backend.api.services.calibration_store import calibration_store

logger = logging.getLogger(__name__)
router = APIRouter()

# Module-level batch manager (singleton)
_manager: BatchManager = None

# Extensions considered as EBSD data files
_EBSD_EXTENSIONS = {'.h5oina', '.h5', '.hdf5', '.hdf', '.ebsp'}

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
    """Scan a folder for EBSD data files (.h5oina, .h5, .hdf5, etc.).

    Returns file list with basic metadata (size, extension).
    """
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


@router.post("/quick-load")
async def quick_load(req: QuickLoadRequest):
    """Quick-load an EBSD file: extract metadata and register PC in CalibrationStore.

    Does NOT build a full pattern cache. Just enough to get grid shape,
    pattern shape, PC, and EDS element availability.
    """
    fp = Path(req.file_path)
    if not fp.is_file():
        raise HTTPException(400, f"File not found: {req.file_path}")

    dataset_name = fp.stem

    try:
        from safe_loader import load_ebsd_safe
        signal = load_ebsd_safe(str(fp))

        # Register in CalibrationStore
        cal_entry = calibration_store.register(dataset_name, signal)

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
        if cal_entry:
            pc_value = [float(v) for v in cal_entry.pc_single]
            pc_source = cal_entry.pc_source

        # Store lightweight reference for later use
        _quick_loaded[dataset_name] = {
            "file_path": str(fp),
            "signal": signal,
            "grid_shape": list(nav_shape),
            "pattern_shape": list(sig_shape),
        }

        return {
            "dataset_name": dataset_name,
            "file_path": str(fp),
            "grid_shape": list(nav_shape),
            "pattern_shape": list(sig_shape),
            "pattern_count": int(signal.axes_manager.navigation_size),
            "pc": pc_value,
            "pc_source": pc_source,
            "has_eds": has_eds,
            "eds_elements": eds_elements,
        }

    except Exception as e:
        logger.exception("Quick-load failed for %s", req.file_path)
        raise HTTPException(500, f"Failed to load: {e}")


# -----------------------------------------------------------------------
# PC management for batch workflows
# -----------------------------------------------------------------------

class CopyPCRequest(BaseModel):
    source_dataset: str
    target_datasets: List[str]


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
    from pathlib import Path as _P
    return _P(name).stem


@router.post("/copy-pc")
async def copy_pc(req: CopyPCRequest):
    """Copy PC from one dataset to one or more target datasets.

    Source/target identifiers may be either dataset names (file stems) or
    full filenames with extension — both are normalised to the stem.
    Targets must already be registered in CalibrationStore (via quick-load).
    """
    source_key = _resolve_dataset_name(req.source_dataset)
    source_pc = calibration_store.get_pc(source_key)
    if source_pc is None:
        raise HTTPException(
            404,
            f"Source dataset '{req.source_dataset}' (key={source_key!r}) not in "
            "CalibrationStore. Quick-load the file first via /api/batch-v2/quick-load.",
        )

    results = []
    for target in req.target_datasets:
        target_key = _resolve_dataset_name(target)
        entry = calibration_store.get_entry(target_key)
        if entry is None:
            results.append({
                "dataset": target,
                "key": target_key,
                "success": False,
                "error": f"not in store (need quick-load for key={target_key!r})",
            })
            continue
        calibration_store.update_pc(target_key, source_pc, source="inherited")
        entry = calibration_store.get_entry(target_key)
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
            results.append({
                "dataset_name": name,
                "pc": [float(v) for v in entry.pc_single],
                "pc_source": entry.pc_source,
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

    def run():
        mgr.run_batch_sync(batch_id)

    asyncio.get_event_loop().run_in_executor(None, run)
    return {"status": "started", "batch_id": batch_id}


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
