"""System-level endpoints (GPU detection, version, diagnostics, frontend errors)."""
from __future__ import annotations

import io
import json
import logging
import platform
import sys
import time
import zipfile
from pathlib import Path

from fastapi import APIRouter, Body
from fastapi.responses import Response

from backend.dict_gpu.runtime import detect_gpu
from backend.api.services.app_version import PROJECT_ROOT, get_version_info
from backend.api import file_log

router = APIRouter()

logger = logging.getLogger(__name__)

# Errors reported by the frontend land under this logger so they are easy to
# spot in logs/orienta.log and in the Dev Panel.
frontend_logger = logging.getLogger("frontend")


@router.get("/gpu", summary="GPU detection and VRAM probe")
def gpu_status() -> dict:
    """Return CUDA device info or {available: false} if no CUDA."""
    s = detect_gpu()
    return {
        "available": s.available,
        "name": s.name,
        "vram_total_gb": s.vram_total_gb,
        "vram_free_gb": s.vram_free_gb,
    }


@router.get("/version", summary="App version identity (git commit based)")
def app_version() -> dict:
    return get_version_info()


_MAX_FIELD = 8000  # keep a hostile/huge payload from bloating the log


def _clip(value, limit=_MAX_FIELD) -> str:
    text = str(value) if value is not None else ""
    return text[:limit]


@router.post("/frontend-error", summary="Log an error reported by the frontend")
def report_frontend_error(payload: dict = Body(...)) -> dict:
    """Persist a frontend-side error (React crash, unhandled rejection, ...).

    The frontend has no durable log of its own — routing its errors through
    this endpoint puts them into logs/orienta.log next to the backend events
    they usually belong to.
    """
    kind = _clip(payload.get("kind") or "unknown", 60)
    message = _clip(payload.get("message"))
    stack = _clip(payload.get("stack"))
    component_stack = _clip(payload.get("componentStack"))
    page = _clip(payload.get("page"), 300)
    where = _clip(payload.get("where"), 200)
    user_agent = _clip(payload.get("userAgent"), 300)
    breadcrumbs = _clip(payload.get("breadcrumbs"))

    lines = [f"[{kind}] {message}"]
    if where:
        lines.append(f"  shown in: {where}")
    if page:
        lines.append(f"  page: {page}")
    if user_agent:
        lines.append(f"  userAgent: {user_agent}")
    if breadcrumbs:
        # What happened before — the difference between a diagnosable report
        # and a screenshot with one line of text.
        lines.append("  what happened before:\n" + breadcrumbs)
    if stack:
        lines.append("  stack:\n" + stack)
    if component_stack:
        lines.append("  componentStack:\n" + component_stack)
    frontend_logger.error("\n".join(lines))
    return {"logged": True}


def _package_versions() -> dict:
    from importlib import metadata

    packages = {}
    for name in (
        "numpy", "scipy", "h5py", "hyperspy", "kikuchipy", "orix", "diffsims",
        "pyebsdindex", "torch", "cupy-cuda12x", "fastapi", "uvicorn",
    ):
        try:
            packages[name] = metadata.version(name)
        except Exception:
            packages[name] = None
    return packages


def _loaded_files_snapshot() -> dict:
    """Loaded-file registry + active file, defensively (never raises)."""
    try:
        from backend.api.routes import ebsd_viewer

        return {
            "active_file": getattr(ebsd_viewer, "_ebsd_file_path", None),
            "loaded_files": [
                e.get("path") for e in getattr(ebsd_viewer, "_loaded_files", [])
            ],
        }
    except Exception:
        return {"active_file": None, "loaded_files": []}


def _gpu_snapshot() -> dict:
    try:
        return gpu_status()
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _diagnostics_info() -> dict:
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": get_version_info(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": _package_versions(),
        "gpu": _gpu_snapshot(),
        **_loaded_files_snapshot(),
    }


def _recent_sim_logs(limit: int = 3) -> list[Path]:
    sim_dir = PROJECT_ROOT / "data" / "sim_logs"
    try:
        logs = sorted(
            sim_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        return logs[:limit]
    except Exception:
        return []


def _report_text(payload: dict) -> str:
    """The human half of a bug report: what the user did and expected."""
    description = _clip(payload.get("description"), 10000)
    page = _clip(payload.get("page"), 300)
    breadcrumbs = _clip(payload.get("breadcrumbs"), 20000)

    lines = ["Orienta problem report", "=" * 40, ""]
    lines.append(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if page:
        lines.append(f"Page at time of report: {page}")
    lines.append("")
    lines.append("What the user reported")
    lines.append("-" * 40)
    lines.append(description if description else "(no description given)")
    lines.append("")
    lines.append("What happened before (most recent last)")
    lines.append("-" * 40)
    lines.append(breadcrumbs if breadcrumbs else "(no activity recorded)")
    return "\n".join(lines) + "\n"


@router.api_route(
    "/diagnostics/export",
    methods=["GET", "POST"],
    summary="Bug-report bundle (zip)",
)
def export_diagnostics(payload: dict | None = Body(None)) -> Response:
    """One zip with everything a bug report needs.

    Contents: report.txt (the user's own description plus the breadcrumb
    trail — the half no log can reconstruct), info.json (version,
    environment, packages, GPU, loaded files), logs/orienta.log incl.
    rotations, the Electron-captured process log (backend-console.log) if
    present, and the newest simulation job logs.

    GET works without any context (nothing but the logs); POST carries the
    description and breadcrumbs collected in the browser.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if payload:
            zf.writestr("report.txt", _report_text(payload))
            # Also into the log, so the description sits next to the events
            # it describes even if the zip is never sent.
            desc = _clip(payload.get("description"), 2000)
            if desc:
                frontend_logger.error(
                    "[problem-report] %s\n  what happened before:\n%s",
                    desc,
                    _clip(payload.get("breadcrumbs"), 8000),
                )
        zf.writestr("info.json", json.dumps(_diagnostics_info(), indent=2, default=str))

        for p in file_log.get_log_paths():
            try:
                zf.write(p, arcname=f"logs/{p.name}")
            except Exception:
                logger.warning("Diagnostics export: could not add %s", p)

        # Process-level capture written by electron/main.js and start_app.py —
        # holds import-time crashes, print() output and CUDA-level stderr.
        for name in ("backend-console.log", "backend-console.log.1"):
            p = file_log.DEFAULT_LOG_DIR / name
            if p.exists():
                try:
                    zf.write(p, arcname=f"logs/{name}")
                except Exception:
                    logger.warning("Diagnostics export: could not add %s", p)

        for p in _recent_sim_logs():
            try:
                zf.write(p, arcname=f"sim_logs/{p.name}")
            except Exception:
                logger.warning("Diagnostics export: could not add %s", p)

    filename = "orienta-diagnostics-" + time.strftime("%Y%m%d-%H%M%S") + ".zip"
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
