"""Reconstruct an EMSphInx ``.sht`` master pattern onto a sphere grid for 3D display.

Pure helper for the Database Browser's interactive master-pattern sphere viewer.  It reuses
the EXACT inverse-SHT the indexer/forward path already runs
(:func:`backend.api.services.sht_pattern_renderer.load_or_get_phase` →
``LambertGrid.grid``, a full-sphere Driscoll-Healy grid), downsamples it for the web,
and returns plain JSON-able data plus crystal metadata.

No new physics: the grid is the same reconstruction the renderer caches; we only
sample angles and pack arrays.
"""
from __future__ import annotations

import logging
import math
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Project root: this file is backend/api/services/sht_sphere.py
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SHT_DB = _PROJECT_ROOT / "Database" / "EBSD_SHT_Database"


class SphereError(RuntimeError):
    """Raised when a ``.sht`` cannot be turned into a sphere (bad/missing file)."""


class SphereCancelled(Exception):
    """The client that asked for this sphere has gone; the work was not started."""


class SphereOutOfMemoryError(SphereError):
    """The sphere does not fit in memory at the requested resolution.

    Not a broken file: a lower resolution may well work, and the interface
    says so (code ``sphereOutOfMemory``).
    """


def _resolve_sht(filename: str) -> Path:
    """Resolve ``filename`` to an existing ``.sht`` under the SHT database.

    Mirrors the resolution used by ``database.preview_file``: try the path as given
    (relative to the SHT DB), then a recursive basename match.  Fail loud if absent.
    """
    if not filename:
        raise FileNotFoundError("empty filename")
    # Absolute path that is actually inside the SHT DB (defensive: stay in-tree).
    cand = (_SHT_DB / filename)
    if cand.is_file():
        _RESOLVED[filename] = cand
        return cand
    # Fall back to a literal basename match. NB: do NOT use rglob(name) — the SHT
    # filenames contain glob-special characters (``[Pearson]``, ``{kV}``), which the
    # glob engine would (mis)interpret as character classes / patterns. Iterate over
    # all .sht and compare the literal name instead.
    if _SHT_DB.is_dir():
        target = Path(filename).name
        for p in _SHT_DB.rglob("*.sht"):
            if p.name == target:
                _RESOLVED[filename] = p
                return p
    raise FileNotFoundError(f"SHT file not found under EBSD_SHT_Database: {filename}")


# Where a name the route was asked for resolved to, so that looking at the cache
# does not repeat the recursive search of the fallback above.
_RESOLVED: "dict[str, Path]" = {}


def _cheap_path(filename: str) -> Optional[Path]:
    """The file ``filename`` is known to be, or None -- one ``stat`` at most."""
    if not filename:
        return None
    path = _RESOLVED.get(filename) or (_SHT_DB / filename)
    try:
        return path if path.is_file() else None
    except OSError:
        return None


def cached_sphere(filename: str, max_bandwidth: int, target_size: int) -> Optional[dict]:
    """A sphere that is already finished, or None. Never computes, never queues.

    The route answers a hit from here, on the event loop, before any worker
    thread is involved: every request used to go through the sphere executor
    first, so a hit waited behind up to four running or queued jobs (measured:
    4.0 s for an answer that takes 30 ms).
    """
    path = _cheap_path(filename)
    if path is None:
        return None
    return _cache_get((str(path), int(max_bandwidth), int(target_size)))


def _downsample_stride(n: int, target: int) -> int:
    """Largest integer stride that keeps ≥ ``target`` samples along an ``n``-axis."""
    if target <= 0 or n <= target:
        return 1
    return max(1, n // target)


# The reconstruction runs in the compute worker (backend/worker). What stays in
# the backend: the finished-sphere cache, the queue (_COMPUTE_LOCK) and the
# sharing of identical requests (_INFLIGHT) -- they decide WHETHER a job goes
# to the worker. The RAM/VRAM preflight and the phase cache run in the worker,
# which has the memory they describe.

# Results of finished reconstructions, newest last. A cache hit must not wait
# behind a running job, so the cache is read without taking _COMPUTE_LOCK --
# which is why it is a plain dict under its own small lock instead of an
# ``lru_cache`` (that cannot be peeked at without computing on a miss).
_RESULTS_CAP = 8
_RESULTS: "OrderedDict[tuple, dict]" = OrderedDict()
_STATE = threading.RLock()

# One reconstruction at a time. Every click on a file or a resolution used to
# start its own job on the server, and nothing stopped them: six High requests
# at once took a CPU backend from 3 GB to 13.7 GB of RAM, four Max requests
# took a 12 GB GPU to 11.8 GB. Queued instead, the peak is that of ONE job.
_COMPUTE_LOCK = threading.Lock()

# A request that waits for the queue (or for an identical request that leads)
# waits in a thread. On the interpreter's shared default executor, which about
# eighty other call sites use, a queue of sphere jobs would take every thread
# and starve unrelated routes -- the structure viewer, the file listing. The
# sphere gets its own small executor instead; requests beyond its threads wait
# in the executor's queue, where they hold nothing.
SPHERE_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sphere")


class _Flight:
    """One reconstruction that is running or queued, shared by every request
    for the same sphere: the first request leads, the others wait for its
    outcome instead of queuing a second identical job."""

    __slots__ = ("event", "result", "error", "followers")

    def __init__(self) -> None:
        self.event = threading.Event()
        #: Requests that wait for this flight's outcome (guarded by _STATE). The
        #: running job is only cancelled when the leader's client AND all of
        #: these have gone.
        self.followers = 0
        self.result: Optional[dict] = None
        self.error: Optional[BaseException] = None


_INFLIGHT: "dict[tuple, _Flight]" = {}


def clear_cache() -> None:
    """Forget every finished reconstruction (used by tests)."""
    with _STATE:
        _RESULTS.clear()


def _cache_get(key: tuple) -> Optional[dict]:
    with _STATE:
        hit = _RESULTS.get(key)
        if hit is not None:
            _RESULTS.move_to_end(key)
        return hit


def _cache_put(key: tuple, payload: dict) -> None:
    with _STATE:
        _RESULTS[key] = payload
        _RESULTS.move_to_end(key)
        while len(_RESULTS) > _RESULTS_CAP:
            _RESULTS.popitem(last=False)


def _build_payload(resolved_path: str, max_bandwidth: int, target_size: int) -> dict:
    """The expensive part: reconstruct and pack. Returns the JSON-able payload."""
    # Imports are local so importing this module is cheap (no torch at import time).
    from backend.api.services.sht_pattern_renderer import (  # noqa: PLC0415
        load_or_get_phase, SHTRenderError, SHTOutOfMemoryError,
    )
    from backend.spherical_gpu.pipeline.sht_io import read_sht_master  # noqa: PLC0415

    p = Path(resolved_path)
    try:
        lg = load_or_get_phase(str(p), max_bandwidth=max_bandwidth)
    except SHTOutOfMemoryError as e:
        # The numbers in ``e`` belong in the log; the caller gets a fixed sentence.
        logger.warning("sphere %s at bw=%s: %s", p.name, max_bandwidth, e)
        raise SphereOutOfMemoryError(
            f"not enough memory for the sphere of {p.name} at this resolution"
        ) from e
    except SHTRenderError as e:
        raise SphereError(f"could not reconstruct sphere from {p.name}: {e}") from e

    grid = lg.grid.detach().cpu().numpy().astype(np.float64)  # (2L, 2L): rows=azimuth, cols=polar
    n_phi, n_theta = grid.shape

    # Downsample for the web surface (keep ~target_size per axis).
    sp = _downsample_stride(n_phi, target_size)
    st = _downsample_stride(n_theta, target_size)
    g = grid[::sp, ::st]
    np_, nt_ = g.shape

    # Driscoll-Healy angle convention (see forward.py): row i → azimuth φ = π·i/L over
    # the ORIGINAL grid; col j → polar θ = (2j+1)·π/(4L).  L = n_phi/2.
    L = n_phi / 2.0
    rows = np.arange(0, n_phi, sp)[:np_]
    cols = np.arange(0, n_theta, st)[:nt_]
    phi = (math.pi * rows / L).tolist()                    # azimuth ∈ [0, 2π)
    theta = ((2.0 * cols + 1.0) * math.pi / (4.0 * L)).tolist()  # polar ∈ (0, π)

    meta: dict = {}
    try:
        m = read_sht_master(str(p), device="cpu")
        meta = {
            "formula": m.formula or "",
            "space_group": int(m.space_group),
            "point_group": m.point_group,
            "voltage_kV": round(float(m.voltage_kv), 2),
        }
    except Exception:
        meta = {}

    return {
        "filename": p.name,
        "grid": g.tolist(),            # (n_phi, n_theta) downsampled DH grid
        "phi": phi,                    # length n_phi (azimuth)
        "theta": theta,                # length n_theta (polar)
        "bandwidth": int(lg.bandwidth),
        "file_bandwidth": int(lg.file_bandwidth),
        "meta": meta,
    }


#: How often a request that is waiting (for the queue, or for an identical
#: request that leads) looks again at whether its client has gone.
_CANCEL_POLL_S = 0.2

#: The worker job that builds a sphere; a module constant so a test can point
#: it at a stand-in.
_SPHERE_JOB = "sphere"


def _compute(key: tuple, cancelled) -> dict:
    """The expensive step, in the compute worker (in-process only under ORIENTA_NO_WORKER=1)."""
    from backend.worker import manager as wm
    if not wm.worker_enabled():
        return _build_payload(*key)
    path, bw, tgt = key
    # On the CPU the RAM check is asked HERE, before a worker is started for a
    # job that would be refused anyway (the check decides whether a job goes
    # to the worker). The device question does not create a CUDA context in
    # the backend. The VRAM check needs a context and runs in the worker.
    from backend.api.services.sht_pattern_renderer import (  # noqa: PLC0415
        SHTOutOfMemoryError, _check_ram_for_bandwidth, _device_from_env,
    )
    if _device_from_env().type == "cpu":
        try:
            _check_ram_for_bandwidth(bw)
        except SHTOutOfMemoryError as e:
            logger.warning("sphere %s at bw=%s refused before the worker: %s",
                           Path(path).name, bw, e)
            raise SphereOutOfMemoryError(
                "not enough memory for the sphere at this resolution") from e
    try:
        return wm.get_manager().run(
            _SPHERE_JOB, {"path": path, "max_bandwidth": int(bw), "target_size": int(tgt)},
            cancelled=cancelled)
    except wm.JobCancelled as e:
        raise SphereCancelled(path) from e
    except wm.JobFailed as e:
        if e.code == "sphereOutOfMemory":
            raise SphereOutOfMemoryError(e.message) from e
        if e.code == "notFound":
            raise FileNotFoundError(e.message) from e
        if e.code == "workerStopped":
            # Arrives as a failed job when the manager stopped while ours was
            # queued; the route answers it with 503 and this code.
            raise wm.WorkerError(e.message, code="workerStopped") from e
        if e.code == "sphereFailed":
            raise SphereError(e.message) from e
        # An unexpected exception in the worker (code jobFailed): one line, in
        # the wording the route used when the backend computed in-process.
        first = (e.message.strip().splitlines() or [e.code])[0][:300]
        raise SphereError(f"sphere reconstruction failed: {first}") from e


def sphere_from_sht(
    filename: str,
    *,
    max_bandwidth: int = 128,
    target_size: int = 128,
    cancelled=None,
) -> dict:
    """Return JSON-able sphere data for an EMSphInx ``.sht`` master pattern.

    Parameters
    ----------
    filename : str
        Path (relative to ``Database/EBSD_SHT_Database``) or bare basename of a ``.sht``.
    max_bandwidth : int
        Inverse-SHT truncation bandwidth for the reconstruction (default 128 → a
        256×256 sphere grid; higher = sharper bands, slower).
    target_size : int
        Downsample the grid to ≈ this many samples per axis for the web surface.
    cancelled : callable, optional
        Returns True once the client that asked has gone. Checked while the
        request waits and again right before the expensive step starts; a
        request that is cancelled by then raises :class:`SphereCancelled`
        instead of spending minutes and gigabytes on an answer nobody reads.
        A job that is already running is cancelled in the compute worker, which
        is then ended -- unless other requests wait for the same sphere: then it
        runs on for them, and is cancelled once the last of them has gone too.

    Returns
    -------
    dict
        ``{filename, grid (n_phi×n_theta), phi[], theta[], bandwidth, file_bandwidth,
        meta:{formula, space_group, point_group, voltage_kV}}``.

    Raises
    ------
    FileNotFoundError
        If the ``.sht`` does not exist under the SHT database.
    SphereError
        If the file exists but cannot be reconstructed (corrupt / unreadable).
    SphereCancelled
        If ``cancelled()`` turned true before the reconstruction started.
    """
    p = _resolve_sht(filename)
    key = (str(p), int(max_bandwidth), int(target_size))
    while True:
        with _STATE:
            hit = _cache_get(key)
            if hit is not None:
                return hit
            flight = _INFLIGHT.get(key)
            leader = flight is None
            if leader:
                flight = _Flight()
                _INFLIGHT[key] = flight
        if leader:
            break
        # Someone is already building this exact sphere (the same file at the
        # same resolution, e.g. a double click or a re-render): wait for that
        # one result instead of queuing a second identical job.
        with _STATE:
            flight.followers += 1
        try:
            while not flight.event.wait(_CANCEL_POLL_S):
                if cancelled is not None and cancelled():
                    raise SphereCancelled(filename)
        finally:
            with _STATE:
                flight.followers -= 1
        if isinstance(flight.error, SphereCancelled):
            continue        # the leader's client left, ours is still here: lead
        if flight.error is not None:
            raise flight.error
        if flight.result is not None:
            return flight.result

    try:
        # Queue for the expensive step. Waiting is cut short, and the job never
        # starts, if the client has gone in the meantime.
        while not _COMPUTE_LOCK.acquire(timeout=_CANCEL_POLL_S):
            if cancelled is not None and cancelled():
                raise SphereCancelled(filename)
        try:
            if cancelled is not None and cancelled():
                raise SphereCancelled(filename)
            def _nobody_waits() -> bool:
                # The leader's client has gone AND no follower waits for the
                # same sphere: only then is the running job worth ending.
                if cancelled is None or not cancelled():
                    return False
                with _STATE:
                    return flight.followers == 0

            payload = _compute(key, _nobody_waits)
            _cache_put(key, payload)
        finally:
            _COMPUTE_LOCK.release()
        flight.result = payload
        return payload
    except BaseException as e:
        flight.error = e
        raise
    finally:
        with _STATE:
            _INFLIGHT.pop(key, None)
        flight.event.set()
