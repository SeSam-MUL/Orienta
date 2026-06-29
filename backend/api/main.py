"""
FastAPI Backend for Orienta
Wraps existing Python EBSD analysis logic with a REST + WebSocket API
"""

import sys
import os
import asyncio
import json
import logging
import mimetypes
from pathlib import Path
from typing import Dict, Set
from contextlib import asynccontextmanager

# Disable Numba caching to avoid cache corruption on Python 3.13,
# but keep JIT enabled — disabling JIT causes ~100x slowdown in pyebsdindex.
if "NUMBA_CACHE_DIR" not in os.environ:
    os.environ["NUMBA_DISABLE_CACHING"] = "1"

# Fix Windows mimetypes not recognizing .js as JavaScript
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Register numpy JSON encoders at import time so every route automatically
# handles numpy scalars/arrays in responses. Fixes the whole bug class of
# "numpy.int32 in detector shape / pc -> 500 Internal Server Error".
from backend.api.services.numpy_json import register_numpy_encoders
register_numpy_encoders()

# Add project root to Python path so we can import existing modules
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)
from backend.api.log_broadcast import install_ws_log_handler


class ConnectionManager:
    """Manages WebSocket connections for real-time progress updates."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: dict):
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)
        for conn in dead:
            self.active_connections.discard(conn)

    async def send_progress(self, task_id: str, progress: float, message: str = ""):
        await self.broadcast({
            "type": "progress",
            "task_id": task_id,
            "progress": progress,
            "message": message,
        })

    async def send_result(self, task_id: str, result: dict):
        await self.broadcast({
            "type": "result",
            "task_id": task_id,
            "data": result,
        })


ws_manager = ConnectionManager()


_frontend_heartbeat_task = None


async def _frontend_watchdog():
    """Shut down if no frontend is connected for the configured grace period.

    Only active when KIKUCHIPY_WATCHDOG=1.  Any other value (including unset,
    "0", "false", "no") disables the watchdog — required for long-running
    batches and simulations where a browser/renderer freeze would otherwise
    kill the backend after 60 seconds and wipe an in-flight job.

    Grace period is 10 minutes by default so transient renderer hangs
    (Electron renderer paging out, dev panel struggling under accumulated
    logs, browser tab paused by the OS) don't trigger a shutdown — only a
    truly absent frontend does.  Override with KIKUCHIPY_WATCHDOG_GRACE_SEC.
    """
    flag = os.environ.get("KIKUCHIPY_WATCHDOG", "").strip().lower()
    if flag != "1":
        logger.info("Frontend watchdog disabled (KIKUCHIPY_WATCHDOG=%r)", flag or "<unset>")
        return
    try:
        grace_period = int(os.environ.get("KIKUCHIPY_WATCHDOG_GRACE_SEC", "600"))
    except ValueError:
        grace_period = 600
    logger.info("Frontend watchdog enabled, grace period = %ds", grace_period)
    await asyncio.sleep(grace_period)  # initial grace period for startup
    while True:
        await asyncio.sleep(30)  # check every 30s
        if len(ws_manager.active_connections) == 0:
            logger.warning(
                "No frontend connected — waiting %ds before shutdown "
                "(set KIKUCHIPY_WATCHDOG=0 or use --headless to disable)",
                grace_period,
            )
            # Wait the full grace period, re-checking in case frontend reconnects
            waited = 0
            while waited < grace_period:
                await asyncio.sleep(15)
                waited += 15
                if len(ws_manager.active_connections) > 0:
                    logger.info("Frontend reconnected after %ds, watchdog resets.", waited)
                    break
            if len(ws_manager.active_connections) == 0:
                logger.info("No frontend reconnected after %ds — shutting down backend.",
                            grace_period)
                os._exit(0)


async def _prewarm_kikuchipy_imports():
    """Pre-import the heavy scientific stack in a background thread.

    Why: the first EBSD load triggers ``from safe_loader import load_ebsd_safe``
    inside ``_load_ebsd_blocking``, which transitively imports kikuchipy,
    hyperspy, dask, scikit-image and friends. On a cold Python interpreter
    that's 5-20 s of pure-Python work that holds the GIL — during which
    the event loop cannot answer the LoadProgressModal's progress polls,
    so the modal sits frozen at "Reading metadata — Starting… — 0.0 s
    elapsed" through the entire cold-start. Pre-warming at server startup
    pushes that cost out of the user-visible critical path.

    Runs in a thread so server startup itself stays fast — the prewarm
    finishes in the background while the user sees the UI come up.
    """
    def _do_import():
        try:
            import safe_loader  # noqa: F401 — triggers kikuchipy chain
            safe_loader._kp()  # force the lazy kikuchipy import
            logger.info("Prewarm: kikuchipy + safe_loader imports complete")
        except Exception:
            logger.exception("Prewarm of kikuchipy imports failed (non-fatal)")
        # Crystal Hint local library: pymatgen parses 22 CIFs on first
        # access (~8 s). Pre-warming avoids the "first analyze-pixel takes
        # 10 s, every subsequent one is instant" surprise.
        try:
            from backend.api.services.crystal_hint_local_library import get_index
            n = len(get_index())
            logger.info("Prewarm: crystal_hint local library cached (%d entries)", n)
        except Exception:
            logger.exception("Prewarm of crystal_hint library failed (non-fatal)")
    await asyncio.to_thread(_do_import)


async def _reap_orphaned_emsoft():
    """Reap EMsoft processes orphaned by a previous backend session.

    EMsoft runs inside the WSL2 VM via ``wsl bash -lc "... EMEBSDmaster..."``.
    A backend restart/kill only tears down the Windows-side wsl.exe relay —
    the actual compute process keeps running inside the VM forever, pinning
    CPU (observed: a 10-hour orphan after a restart). At startup nothing is
    legitimately running yet, so anything found is an orphan. Runs in a
    thread (subprocess calls block, and a cold WSL may take seconds) and is
    fire-and-forget so it never delays server readiness. Non-fatal on error.
    """
    try:
        from simulation.simulation_controller import reap_orphaned_emsoft_processes
        reaped = await asyncio.to_thread(reap_orphaned_emsoft_processes)
        if reaped:
            logger.warning(
                "Startup reaper killed %d orphaned EMsoft process(es): %s",
                len(reaped), ", ".join(r["name"] for r in reaped),
            )
    except Exception:
        logger.exception("Startup reaper failed (non-fatal)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _frontend_heartbeat_task
    logger.info("Orienta Backend starting...")
    _frontend_heartbeat_task = asyncio.create_task(_frontend_watchdog())
    # Fire-and-forget prewarm — the first load no longer pays import cost.
    asyncio.create_task(_prewarm_kikuchipy_imports())
    # Fire-and-forget reaper — clean up EMsoft processes left running in WSL
    # by a previous backend session that was restarted/killed without cleanup.
    asyncio.create_task(_reap_orphaned_emsoft())
    install_ws_log_handler(ws_manager)
    logger.info("Dev-Panel WebSocket log handler installed")
    yield
    logger.info("Orienta Backend shutting down...")
    if _frontend_heartbeat_task:
        _frontend_heartbeat_task.cancel()


from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import time as _time


class HTTPTimingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with method, path, status, and duration."""

    _EXCLUDED_PREFIXES = ("/api/health", "/ws", "/assets/", "/.vite/")
    _EXCLUDED_EXTENSIONS = (".js", ".css", ".png", ".svg", ".ico", ".woff", ".woff2", ".map")

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if any(path.startswith(p) for p in self._EXCLUDED_PREFIXES):
            return await call_next(request)
        if any(path.endswith(ext) for ext in self._EXCLUDED_EXTENSIONS):
            return await call_next(request)

        start = _time.perf_counter()
        response = await call_next(request)
        duration_ms = (_time.perf_counter() - start) * 1000

        if ws_manager.active_connections:
            payload = {
                "type": "dev_log",
                "category": "http",
                "method": request.method,
                "path": path,
                "status": response.status_code,
                "duration_ms": round(duration_ms),
                "timestamp": _time.strftime("%H:%M:%S", _time.localtime())
                             + f".{int((_time.perf_counter() % 1) * 1000):03d}",
            }
            try:
                import asyncio as _asyncio
                _asyncio.ensure_future(ws_manager.broadcast(payload))
            except RuntimeError:
                pass

        return response


app = FastAPI(
    title="Orienta Backend",
    description="REST + WebSocket API for EBSD Pattern Analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for local React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(HTTPTimingMiddleware)


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "project_root": PROJECT_ROOT,
        "python_executable": sys.executable,
        "python_version": sys.version,
    }


@app.post("/api/shutdown")
async def shutdown():
    """Gracefully shut down the backend server.

    Only honoured when KIKUCHIPY_WATCHDOG=1 (Electron mode).  In dev /
    headless / browser mode this is a no-op so a page refresh, a stray
    request from a dying renderer, or a misbehaving extension can't drop
    a 12-hour batch in one POST.
    """
    if os.environ.get("KIKUCHIPY_WATCHDOG", "").strip().lower() != "1":
        logger.info("Shutdown requested but ignored (KIKUCHIPY_WATCHDOG != '1')")
        return {"status": "ignored", "reason": "watchdog disabled"}
    logger.info("Shutdown requested via API — exiting.")

    async def _delayed_exit():
        await asyncio.sleep(0.5)
        os._exit(0)

    asyncio.create_task(_delayed_exit())
    return {"status": "shutting_down"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time progress updates."""
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Client can send commands via WebSocket
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                logger.warning("WebSocket received non-JSON message, ignoring")
                continue
            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception:
        # Any other exception (oversized frame, send_json failure on a
        # half-closed connection, etc.) — log and still disconnect.
        # Without this, a leaked entry in ws_manager.active_connections
        # would prevent _frontend_watchdog from ever triggering Electron
        # auto-shutdown when no real client is left.
        logger.exception("WebSocket endpoint raised — cleaning up connection")
    finally:
        ws_manager.disconnect(websocket)


# Import and register route modules
from backend.api.routes import h5_viewer, ebsd_viewer, pcrefinement, simulation
from backend.api.routes import indexing, phase_map, analysis, ml_hub, database, eds
from backend.api.routes import calibration, settings, batch_v2, refinement
from backend.api.routes import install, virtual_images, system
from backend.api.routes import dictionary_gpu
from backend.api.routes import forward_diagnostics as forward_diagnostics_routes
from backend.api.routes import crystal_hint
from backend.api.routes import reference_frame as reference_frame_routes
from backend.api.routes import pole_figure as pole_figure_routes

app.include_router(h5_viewer.router, prefix="/api/h5", tags=["HDF5 Viewer"])
app.include_router(ebsd_viewer.router, prefix="/api/ebsd", tags=["EBSD Viewer"])
# Virtual BSE + Band Contrast share the /api/ebsd prefix because they
# are EBSD-derived (sum-of-detector-intensity, pattern-quality) — not
# EDS, even though the EDS Analysis page is what surfaces them.
app.include_router(virtual_images.router, prefix="/api/ebsd", tags=["Virtual Images"])
app.include_router(pcrefinement.router, prefix="/api/pc", tags=["PC Refinement"])
app.include_router(simulation.router, prefix="/api/simulation", tags=["Simulation"])
app.include_router(indexing.router, prefix="/api/indexing", tags=["Indexing"])
app.include_router(phase_map.router, prefix="/api/phasemap", tags=["Phase Map"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(ml_hub.router, prefix="/api/ml", tags=["ML Hub"])
app.include_router(database.router, prefix="/api/database", tags=["Database"])
app.include_router(eds.router, prefix="/api/eds", tags=["EDS"])
app.include_router(calibration.router, prefix="/api/calibration", tags=["Calibration"])
app.include_router(settings.router, prefix="/api/settings", tags=["Settings"])
app.include_router(batch_v2.router, prefix="/api/batch-v2", tags=["Batch v2"])
app.include_router(refinement.router, prefix="/api/refinement", tags=["Phase Refinement"])
app.include_router(install.router, prefix="/api/install", tags=["Install Wizard"])
app.include_router(dictionary_gpu.router, prefix="/api/dictionary-gpu", tags=["Dictionary GPU"])
app.include_router(forward_diagnostics_routes.router, prefix="/api/forward-diagnostics", tags=["forward-diagnostics"])
app.include_router(system.router, prefix="/api/system", tags=["System"])
app.include_router(crystal_hint.router, prefix="/api/crystal-hint", tags=["Crystal Hint"])
# reference_frame.router self-prefixes "/api" (frame + state-version live at /api/*).
app.include_router(reference_frame_routes.router)
# pole_figure.router self-prefixes "/api" (route lives at /api/pole-figure).
app.include_router(pole_figure_routes.router)

# Serve built React frontend in production mode
FRONTEND_DIST = Path(PROJECT_ROOT) / "frontend" / "dist"
if FRONTEND_DIST.exists():
    # Custom static-files class that adds no-cache headers to `index.html`
    # so the browser always fetches the latest entrypoint. Hashed assets
    # (vendor-XXXX.js, main-YYYY.css) keep their normal long-cache headers
    # because their filenames change on every build — those CAN be cached
    # safely. Without this, the browser holds onto the old index.html
    # which references the old hashed bundles, and the user never sees
    # frontend updates without a manual hard-reload.
    class _NoCacheIndexStaticFiles(StaticFiles):
        async def get_response(self, path: str, scope):
            response = await super().get_response(path, scope)
            if path in ("", "/", "index.html"):
                response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
            return response

    # Mount the entire dist folder as a catch-all static mount.
    # html=True serves index.html for directory requests (SPA fallback).
    # This must be last since mount() is checked after explicit routes.
    app.mount("/", _NoCacheIndexStaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
