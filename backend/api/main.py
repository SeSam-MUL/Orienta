"""
FastAPI Backend for Orienta
Wraps existing Python EBSD analysis logic with a REST + WebSocket API
"""

import sys
import os
import time
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

# Persistent rotating log file (logs/orienta.log). MUST happen at import time,
# not in the lifespan: anything logged during module import / route import /
# prewarm would otherwise be lost, and a lifespan-installed handler never runs
# at all when the app fails to import.
from backend.api.file_log import install_file_logging
_LOG_FILE = install_file_logging()
if _LOG_FILE is not None:
    from backend.api.services.app_version import version_line
    logger.info("=" * 60)
    logger.info("%s — session start (pid %d)", version_line(), os.getpid())
    logger.info("Python %s on %s", sys.version.split()[0], sys.platform)
    logger.info("Log file: %s", _LOG_FILE)

# macOS: PyEBSDIndex's band detector on the CPU path, before any route imports
# it (see pyebsdindex_mode). After the file log exists, so a Mac bug report
# shows which detector ran. No-op on other platforms.
from pyebsdindex_mode import force_cpu_band_detection
force_cpu_band_detection()

from backend.api.log_broadcast import install_ws_log_handler, uninstall_ws_log_handler


# Its own name so the dev-panel handler can suppress exactly this one without
# silencing anything else — see log_broadcast._SUPPRESSED_LOGGERS.
_ws_prune_logger = logging.getLogger("backend.api.ws_prune")


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
        # Over a snapshot, not the live set: every send suspends, and that is
        # where the /ws handler of a client that just went away gets its turn
        # to call disconnect() — mutating the set being iterated.
        #     RuntimeError: Set changed size during iteration
        # The broadcast died at that point, so every connection after the
        # departing one lost the message (progress, streamed log lines) in a
        # window that was still open. Fired and forgotten, it surfaced only as
        # "Task exception was never retrieved" in the log.
        dead = []
        for connection in tuple(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception as exc:
                dead.append((connection, exc))
        for conn, exc in dead:
            self.active_connections.discard(conn)
            # SAY WHY. This prune is what emptied active_connections on the M5
            # tester's Mac while the window was open: the log records four
            # "connection open" lines and not one "connection closed", so the
            # reason a live socket was dropped had to be inferred. One line here
            # turns the next diagnostics bundle into an answer.
            #
            # Logged under its own logger name, which log_broadcast suppresses
            # for the WebSocket handler: a record broadcast from inside broadcast
            # reaches the same dead socket, raises again, and logs again — the
            # feedback loop fixed in d370a976, rebuilt by hand. The file log
            # still gets it, which is where a bug report reads it.
            _ws_prune_logger.warning(
                "Dropped a WebSocket from the broadcast set: %s: %s",
                type(exc).__name__, exc,
            )

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


# Set by note_http_activity; read by the frontend watchdog.
_last_http_activity = 0.0
_parent_watchdog_task = None
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
            if len(ws_manager.active_connections) != 0:
                continue
            # Two second opinions before killing a running app.
            #
            # Measured on the M5 tester's Mac (2026-09-25, report point 4, log
            # in tasks/mac-test-m5-2026-09-25/): this watchdog shut the backend
            # down at 09:54:00 with the window OPEN and in use. Two WebSockets
            # opened at 09:33:27 and the log never records either closing, yet
            # the count was zero by 09:43:56. What emptied it is an INFERENCE
            # from what the log does not contain: uvicorn logged "connection
            # open" for every session and "connection closed" for none, and the
            # /ws handler's own cleanup path logs "WebSocket endpoint raised",
            # which never appears - so the silent prune in
            # ConnectionManager.broadcast (which discards a connection whose
            # send raises) is the only remaining route to an empty set. That
            # prune now logs its reason, so the next bundle will say rather than
            # imply. Meanwhile the backend went
            # on serving that same "absent" frontend: CIF parsing at 09:48:58, a
            # simulation at 09:49:18, spherical at 09:53:04 - all AFTER the
            # countdown had started. And the Electron parent, PID 3350, was
            # alive across all four backend restarts that morning.
            #
            # So an empty WebSocket count is not evidence that the app is gone.
            if _ui_parent_is_alive():
                logger.info("No WebSocket, but the window-owning parent is alive "
                            "- not shutting down. _parent_watchdog owns that case.")
                continue
            idle = seconds_since_http_activity()
            if idle < grace_period:
                logger.info("No WebSocket, but an HTTP request arrived %.0fs ago "
                            "- not shutting down.", idle)
                continue
            logger.info("No frontend reconnected after %ds and no HTTP for %.0fs "
                        "- shutting down backend.", grace_period, idle)
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


def _ensure_addon_dir():
    """Create ~/.orienta/addons once, at startup. Never fatal.

    Imported lazily and wrapped: nothing about a missing add-on folder should
    be able to stop the backend from starting, and the discovery path reads a
    missing directory as empty anyway.
    """
    try:
        from backend.api.services.addons.discovery import ensure_user_addon_dir
        ensure_user_addon_dir()
    except Exception:
        logger.exception("Could not prepare the add-on directory (non-fatal)")


def _ui_parent_is_alive() -> bool:
    """Is a parent that OWNS A WINDOW still there?

    The frontend watchdog treats this as proof that a user is present, so the
    question is deliberately narrower than _parent_watchdog's. Both launchers
    pass KIKUCHIPY_PARENT_PID, but only Electron passes KIKUCHIPY_UI_PARENT,
    and the difference matters:

      * Electron holds the window. While it lives, someone can see the app,
        even when the renderer has dropped off ws_manager.
      * start_app.py opens a browser tab and then blocks in
        ``backend_proc.wait()``. Its liveness is implied by the backend's own,
        so trusting it would mean the watchdog can never fire: launcher waits
        for backend, parent watchdog waits for launcher, frontend watchdog
        defers to the parent watchdog. The backend would hold port 8000, the
        loaded dataset and its HDF5 handles forever after the tab is closed —
        the 19-hour backend with dead file handles from 2026-08-05, and the
        opposite of what start_app.py and INSTALL.md promise.

    On that path the HTTP clock decides instead, which is what a closed tab
    actually changes: App.jsx stops polling /api/health.

    No create_time() pin here, unlike _parent_watchdog: a recycled PID can only
    be believed for the <=3 s until that watchdog notices the identity mismatch
    and exits the process outright. The two can only disagree in the direction
    "do not exit yet", and the parent watchdog overrides that by exiting.
    """
    if os.environ.get("KIKUCHIPY_UI_PARENT", "").strip() != "1":
        return False
    raw = os.environ.get("KIKUCHIPY_PARENT_PID", "").strip()
    if not raw:
        return False
    try:
        import psutil
        return psutil.pid_exists(int(raw))
    except Exception:
        return False


def note_http_activity() -> None:
    """Record that a client just asked us for something."""
    global _last_http_activity
    _last_http_activity = time.monotonic()


def seconds_since_http_activity() -> float:
    """Seconds since the last request, or inf when there has never been one."""
    if not _last_http_activity:
        return float("inf")
    return time.monotonic() - _last_http_activity


async def _parent_watchdog():
    """Exit as soon as the process that launched us is gone.

    The desktop app spawns this backend as a child of Electron and passes its
    own PID as ``KIKUCHIPY_PARENT_PID``. When Electron exits — normally, after a
    crash, or because the user closed the console window that hosts it — this
    backend must go too. Nothing else reliably covers all of those paths:

      * Electron's own ``before-quit`` kill never runs when Electron itself is
        killed (closing the console kills the whole npm/concurrently tree).
      * ``_frontend_watchdog`` keys on ``ws_manager.active_connections``, and a
        half-open WebSocket from a dead renderer keeps that count above zero
        indefinitely — which is exactly how a backend survived 19 h and then
        served stale, dead file handles after the data drive was replugged.

    Checking the parent process is deterministic and immune to both. Disabled
    when the variable is unset, so ``start_app.py``, --headless and plain
    uvicorn runs (no parent to speak of) are unaffected.
    """
    raw = os.environ.get("KIKUCHIPY_PARENT_PID", "").strip()
    if not raw:
        logger.info("Parent watchdog disabled (KIKUCHIPY_PARENT_PID unset)")
        return
    try:
        parent_pid = int(raw)
    except ValueError:
        logger.warning("Parent watchdog disabled — KIKUCHIPY_PARENT_PID=%r is not a PID", raw)
        return

    try:
        import psutil
        parent = psutil.Process(parent_pid)
        # Pin the identity: a PID alone is not enough, the OS reuses them.
        parent_started_at = parent.create_time()
    except Exception:
        logger.warning("Parent watchdog: parent PID %d not found at startup — "
                       "exiting rather than outliving it", parent_pid)
        os._exit(0)

    interval = 3.0
    logger.info("Parent watchdog enabled — exiting when PID %d goes away "
                "(checked every %.0fs)", parent_pid, interval)
    while True:
        await asyncio.sleep(interval)
        try:
            if not psutil.pid_exists(parent_pid):
                gone = "process no longer exists"
            elif psutil.Process(parent_pid).create_time() != parent_started_at:
                gone = "PID was reused by a different process"
            else:
                continue
        except Exception:
            gone = "parent process is no longer inspectable"
        logger.info("Parent (PID %d) gone (%s) — shutting the backend down.",
                    parent_pid, gone)
        os._exit(0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _frontend_heartbeat_task
    logger.info("Orienta Backend starting...")
    _frontend_heartbeat_task = asyncio.create_task(_frontend_watchdog())
    # Hard guarantee that closing the desktop app (or its console) takes the
    # backend with it — see _parent_watchdog for why the other two paths fail.
    # Reference held: asyncio may collect a task nobody keeps, and the frontend
    # watchdog now DELEGATES to this one ("_parent_watchdog owns that case"). If
    # this task ever failed to run, the backend would have no watchdog at all
    # rather than one.
    global _parent_watchdog_task
    _parent_watchdog_task = asyncio.create_task(_parent_watchdog())
    # Fire-and-forget prewarm — the first load no longer pays import cost.
    asyncio.create_task(_prewarm_kikuchipy_imports())
    # Fire-and-forget reaper — clean up EMsoft processes left running in WSL
    # by a previous backend session that was restarted/killed without cleanup.
    asyncio.create_task(_reap_orphaned_emsoft())
    # The add-on folder, so the one the empty Add-ons page NAMES exists and a
    # user can open it. Here and not in discovery: GET requests are ungated on
    # Origin because "they change nothing", and a listing that created a
    # directory would have made that false.
    _ensure_addon_dir()
    install_ws_log_handler(ws_manager)
    logger.info("Dev-Panel WebSocket log handler installed")
    try:
        yield
    finally:
        # `finally`, because an abrupt shutdown throws CancelledError into the
        # generator at `yield`; without it the tear-down below is skipped and
        # the handler outlives its loop after exactly the kind of shutdown that
        # is hardest to debug. (Two paths still skip it: a second Ctrl+C makes
        # uvicorn drop lifespan.shutdown entirely, and the watchdogs call
        # os._exit. Both are covered by the other brakes in log_broadcast.)
        logger.info("Orienta Backend shutting down...")
        if _frontend_heartbeat_task:
            _frontend_heartbeat_task.cancel()
        # Detach the dev-panel log handler BEFORE the loop closes. Left
        # attached, it turns every later log record into a scheduling attempt
        # against a dead loop — the runaway measured on the Linux runner.
        # Safe here: uvicorn shuts every connection down, including the dev
        # panel's WebSocket, before it awaits lifespan.shutdown.
        uninstall_ws_log_handler()


from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import time as _time


class HTTPTimingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with method, path, status, and duration."""

    _EXCLUDED_PREFIXES = ("/api/health", "/ws", "/assets/", "/.vite/")
    _EXCLUDED_EXTENSIONS = (".js", ".css", ".png", ".svg", ".ico", ".woff", ".woff2", ".map")

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Before the exclusions: a request is a sign of life whoever made it
        # and whatever it was for. /api/health is excluded from the dev-panel
        # LOG, and it is exactly the request a live frontend makes every 30 s —
        # so putting this call after the exclusion list would make a healthy
        # frontend's poll count for nothing. (Counterfactual, not a cause: no
        # activity clock existed when the 09:54 shutdown was measured.)
        note_http_activity()

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


from backend.api.services.app_version import get_version_info

app = FastAPI(
    title="Orienta Backend",
    description="REST + WebSocket API for EBSD Pattern Analysis",
    version=get_version_info()["version"],
    lifespan=lifespan,
)

# CORS for local React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # The error code the interface translates on. The default dev server is
    # same-origin (vite proxies /api to 8000), so this matters for a frontend
    # pointed straight at the backend with VITE_API_URL — where, without it,
    # errors would be translated in the packaged app and English in dev.
    expose_headers=["X-Orienta-Code"],
)
app.add_middleware(HTTPTimingMiddleware)

# Outermost (added last): refuse cross-site writes, foreign WebSocket origins
# and DNS-rebound Host headers before anything else sees the request. See
# backend/api/security.py and tests/test_api_origin_guard.py.
from backend.api.security import LocalOriginGuard
app.add_middleware(LocalOriginGuard)


from fastapi import HTTPException as _FastAPIHTTPException
from fastapi.exception_handlers import http_exception_handler as _default_http_handler
from starlette.exceptions import HTTPException as _StarletteHTTPException

# Paths whose failures are noise: the frontend polls these constantly, and a
# backend that is simply busy would otherwise fill the log with 404s.
_QUIET_ERROR_PATHS = ("/api/health", "/api/system/frontend-error")


@app.exception_handler(_StarletteHTTPException)
async def _log_http_exception(request: Request, exc: _StarletteHTTPException):
    """Log every error response the user is about to see.

    The codebase raises HTTPException in ~543 places; each returns a message
    to the client and, until now, left no server-side trace at all. Those
    messages are exactly what users screenshot, so they belong in the log
    next to the events that led to them.

    The response itself is produced by FastAPI's default handler — behaviour
    for every existing client is unchanged.
    """
    path = request.url.path
    if not any(path.startswith(p) for p in _QUIET_ERROR_PATHS):
        log = logger.error if exc.status_code >= 500 else logger.warning
        log("HTTP %s on %s %s: %s", exc.status_code, request.method, path, exc.detail)
    return await _default_http_handler(request, exc)


from fastapi.exceptions import RequestValidationError as _RequestValidationError
from fastapi.responses import JSONResponse as _JSONResponse


@app.exception_handler(_RequestValidationError)
async def _readable_validation_error(request: Request,
                                     exc: _RequestValidationError):
    """A rejected request must say what is wrong IN WORDS.

    FastAPI's default body is ``detail: [{type, loc, msg, input, ctx}, …]``.
    Every error panel in this app reads ``err.response.data.detail`` and puts
    it in a sentence, so a list of dicts arrives on screen as ``[object
    Object]`` — a refusal the user cannot act on, for a request that was
    refused for a perfectly nameable reason (e.g. "phase alpha carries no
    file: give it a cif, an sht or a master .h5").

    ``detail`` is therefore a string: one line per problem, each naming the
    field it belongs to. A JSON-SAFE reduction of the structured list stays
    under ``errors`` so anything that wants to inspect it still can — nothing
    in this repo does (checked: no frontend reader, no test asserts the old
    shape).

    Only ``type``/``loc``/``msg`` are carried over. pydantic's own entries also
    hold ``ctx`` (which contains the raw exception OBJECT) and ``input`` (the
    offending value, e.g. a numpy array) — handing those to JSONResponse
    raises inside the response and the client gets an EMPTY body, which is
    worse than the list this handler exists to replace. Measured.
    """
    lines = []
    safe = []
    for err in exc.errors():
        loc_parts = [str(p) for p in err.get("loc", ()) if p not in ("body",)]
        # ASCII separator on purpose: this string is logged too, and a
        # Windows console on cp1252 cannot encode an arrow.
        loc = " > ".join(loc_parts)
        msg = str(err.get("msg", "invalid value"))
        # pydantic v2 prefixes custom ValueErrors with "Value error, ".
        msg = msg[len("Value error, "):] if msg.startswith("Value error, ") else msg
        lines.append(f"{loc}: {msg}" if loc else msg)
        safe.append({"type": str(err.get("type", "")),
                     "loc": loc_parts, "msg": msg})
    detail = "; ".join(lines) or "The request body is not valid."
    logger.warning("HTTP 422 on %s %s: %s",
                   request.method, request.url.path, detail)
    return _JSONResponse(status_code=422,
                         content={"detail": detail, "errors": safe})


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "project_root": PROJECT_ROOT,
        "python_executable": sys.executable,
        "python_version": sys.version,
    }


def _shutdown_grace_s() -> float:
    try:
        return float(os.environ.get("KIKUCHIPY_SHUTDOWN_GRACE_SEC", "5"))
    except ValueError:
        return 5.0


async def schedule_shutdown(manager, grace_s: float, exit_fn=os._exit) -> bool:
    """Exit after `grace_s` — unless a WebSocket client is connected by then.

    The page posts /api/shutdown from its `unload` handler so that closing
    the window stops the backend. But `unload` also fires on a page RELOAD:
    a Vite hot reload during development, F5 in the browser, and Electron's
    own crash recovery (`mainWindow.reload()`) — which therefore killed the
    very backend it was written to protect. Measured 2026-09-09: a merge that
    touched App.jsx reloaded the page and the log read "Shutdown requested
    via API — exiting" one second later.

    A reloaded page opens its WebSocket within a second or two; a closed
    window never does. So the exit waits, and a connected client cancels it.
    Returns True when the process is going down (for tests: exit_fn is
    injected).
    """
    await asyncio.sleep(grace_s)
    if manager.active_connections:
        logger.info("Shutdown cancelled — a client reconnected within %.1f s "
                    "(page reload, not a close).", grace_s)
        return False
    logger.info("Shutdown: no client reconnected within %.1f s — exiting.", grace_s)
    exit_fn(0)
    return True


_shutdown_task = None


@app.post("/api/shutdown")
async def shutdown():
    """Gracefully shut down the backend server.

    Only honoured when KIKUCHIPY_WATCHDOG=1 (Electron mode).  In dev /
    headless / browser mode this is a no-op so a page refresh, a stray
    request from a dying renderer, or a misbehaving extension can't drop
    a 12-hour batch in one POST.

    The exit itself is deferred by KIKUCHIPY_SHUTDOWN_GRACE_SEC (default 5)
    and cancelled if a client reconnects — see schedule_shutdown.
    """
    global _shutdown_task
    if os.environ.get("KIKUCHIPY_WATCHDOG", "").strip().lower() != "1":
        logger.info("Shutdown requested but ignored (KIKUCHIPY_WATCHDOG != '1')")
        return {"status": "ignored", "reason": "watchdog disabled"}
    grace = _shutdown_grace_s()
    if _shutdown_task is not None and not _shutdown_task.done():
        return {"status": "shutting_down", "grace_s": grace}
    logger.info("Shutdown requested via API — exiting in %.0f s unless a client reconnects.", grace)
    _shutdown_task = asyncio.create_task(schedule_shutdown(ws_manager, grace))
    return {"status": "shutting_down", "grace_s": grace}


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
from backend.api.routes import eds_export
from backend.api.routes import calibration, settings, batch_v2, refinement
from backend.api.routes import install, virtual_images, system
from backend.api.routes import dictionary_gpu
from backend.api.routes import forward_diagnostics as forward_diagnostics_routes
from backend.api.routes import crystal_hint
from backend.api.routes import reference_frame as reference_frame_routes
from backend.api.routes import pole_figure as pole_figure_routes
from backend.api.routes import citations
from backend.api.routes import addons
from backend.api.routes import phase_collections

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
app.include_router(eds_export.router, prefix="/api/eds", tags=["EDS Export"])
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
app.include_router(citations.router, prefix="/api/citations", tags=["Citations"])
# The prefix must stay equal to addons.API_PREFIX: a map output's values_url
# is built from it, and a test fetches that URL.
app.include_router(addons.router, prefix="/api/addons", tags=["Add-ons"])
app.include_router(phase_collections.router, prefix="/api/phase-collections", tags=["Phase Collections"])

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
