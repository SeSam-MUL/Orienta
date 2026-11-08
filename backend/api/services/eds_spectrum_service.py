"""The spectrum-fit job of the open file and the extractor that serves its result.

A file whose EDS group carries a per-pixel spectrum cube gets its EDS numbers
from NET line intensities (``backend/eds_spectrum``) instead of AZtec's window
integrals. This module runs that evaluation and hands it to the rest of the
app:

* :func:`start_for_file` -- called by ``h5_session.open_file``. A file with a
  cube gets a job that reads the sum spectrum, fits it and projects every
  pixel (stages ``reading`` -> ``fitting`` -> ``projecting`` -> ``done`` or
  ``failed``). The work runs in a CHILD PROCESS (``backend.eds_spectrum.job``,
  :data:`RUN_IN_CHILD`), watched by a daemon thread here: in the backend's
  own process the cube read stalled every request (29 GB scan: health 9.6 s,
  a pattern 20.8 s). The child opens its
  OWN ``h5py.File``; nothing touches the session's handle. A file without a
  cube is recorded as ``"none"`` and keeps the window path unchanged.
* :func:`status` -- cheap, never blocks; what the EDS page polls.
* :func:`get_result` -- the finished :class:`SpectrumNetResult`, ``None`` for
  a file without a cube, :class:`SpectrumFitPending` (HTTP 503) while the job
  runs, :class:`SpectrumFitFailed` (HTTP 500) when it failed.
* :class:`SpectrumNetExtractor` -- wraps the raw ``H5OINADataExtractor`` so
  ``get_element_map`` returns net maps. ``h5_session.get_active_extractor``
  puts it between the raw extractor and ``CroppedExtractor``; the H5 viewer's
  ``get_extractor()`` stays raw.

Jobs are keyed by canonical path + size + mtime, one job per key, and results
are kept for the two most recently started keys (memory: the 29 GB scan's maps
are ~60 MB). The key comes from the file itself, never from "the current
job", so a job for file A cannot deliver its maps to file B, whatever order
they finish in. A child is terminated when its job is evicted, when the
session's file is closed and no file is open :data:`CLOSE_GRACE_S` later (a
switch to another file is not a close), at exit, and -- by itself -- when
the backend process is gone.

No window numbers are ever served for a file that has a cube: until the job is
done every map read raises :class:`SpectrumFitPending`. Both exceptions are
``fastapi.HTTPException`` subclasses, so a route's existing
``except HTTPException: raise`` passes them through and FastAPI renders
``{"detail": {"code": ..., ...}}``.
"""
from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import pickle
import queue
import subprocess
import sys
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import h5py
import numpy as np
from fastapi import HTTPException

from backend.eds_spectrum import job as job_mod
from backend.eds_spectrum.cube import has_spectrum_cube, read_meta, sum_spectrum
from backend.eds_spectrum.fit import (Detection, FileFit, QualityWarning,
                                      fit_sum_spectrum, project_cube)
from backend.eds_spectrum.model import default_families, label as family_label

logger = logging.getLogger(__name__)

#: Run each job in a child process (the product). False runs it in a thread
#: of this process -- for tests that hold or break the fit by patching
#: ``fit_sum_spectrum`` / ``project_cube`` here, which a child cannot see.
RUN_IN_CHILD = True
#: A job of the session's file stops when no file has been open this long
#: (``h5_session.close_file``). Loaders close the old file just before they
#: open the new one; that gap is not a close.
CLOSE_GRACE_S = 2.0
#: How often a job's watcher looks at its child, its cancel and the session.
_POLL_S = 0.2
#: The checkout the child imports ``backend`` from.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Elements fitted always but never reported as "detected, not exported"
#:: a carbon coat and a surface oxide are in every spectrum.
_NEVER_ADDED = ("C", "N", "O")
#: Label written for a fitted family AZtec did not export, by family.
_GENERATED_LABEL = {"K": "{} Kα1", "L": "{} Lα1,2", "M": "{} Mα"}
#: Results are kept for this many files (most recently started first).
MAX_FILES = 2


# ---------------------------------------------------------------------------
# errors the routes pass through
# ---------------------------------------------------------------------------
class SpectrumFitPending(HTTPException):
    """The spectrum fit of this file is still running (HTTP 503)."""

    def __init__(self, progress: float, stage: Optional[str] = None):
        self.progress = float(progress)
        self.stage = stage
        pct = int(round(100 * self.progress))
        super().__init__(
            status_code=503,
            detail={
                "code": "eds_spectrum_fit_running",
                "progress": self.progress,
                "stage": stage,
                "message": (f"The spectra of this map are still being "
                            f"evaluated ({pct} %). EDS numbers are available "
                            f"when that is done."),
            },
            headers={"Retry-After": "1"},
        )


class SpectrumFitFailed(HTTPException):
    """The spectrum fit of this file failed (HTTP 500)."""

    def __init__(self, error: str):
        self.error = str(error)
        super().__init__(
            status_code=500,
            detail={
                "code": "eds_spectrum_fit_failed",
                "error": self.error,
                "message": (f"The spectrum fit failed: {self.error}. EDS "
                            f"numbers are not available for this file."),
            },
        )


#: Both errors, for call sites that have to tell them from anything else.
SPECTRUM_FIT_ERRORS = (SpectrumFitPending, SpectrumFitFailed)


def fit_status_marker(exc: HTTPException) -> dict:
    """The ``eds_status`` a composite payload carries instead of raising.

    A response whose main purpose is not EDS numbers (the phase-map state,
    the hover probe's other layers) leaves out only its EDS-derived parts
    while the fit runs or after it failed, and says so with this.
    """
    if isinstance(exc, SpectrumFitPending):
        return {"state": "pending", "code": exc.detail["code"],
                "progress": exc.progress, "stage": exc.stage}
    return {"state": "failed", "code": exc.detail["code"],
            "error": exc.detail.get("error")}


class _Cancelled(Exception):
    """Raised inside a job that was evicted, closed or stopped."""


class _ChildFailed(Exception):
    """The child process reported an error, or ended without an answer.

    ``str()`` is the message the status and the 500 carry; ``details`` the
    child's traceback or the last lines it wrote to stderr, for the log.
    """

    def __init__(self, message: str, details: str = ""):
        super().__init__(message)
        self.details = details or ""


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------
@dataclass
class SpectrumNetResult:
    """What the job produced for one file.

    ``maps``: family label (``"Fe K"``) -> flat float32 net counts per live
    second per pixel, UNCLIPPED (see ``project_cube``), file order.
    ``labels``: the COMPOSITION's labels -> family label: every fitted family
    of an element AZtec exported, nothing else (revised
    2026-10-25). ``detected_labels``: the same for the elements the spectrum
    shows but AZtec did not export -- their net maps are reachable (the
    notice's "add as map"), they never enter a quantification.
    ``detected_at_pct``: each detected element's at% in the composition of
    the whole map's summed net counts with it added -- the one number the
    notice gives about its size.
    ``exported``: AZtec's window labels. ``exported_not_fitted``: exported
    labels whose family the model does not carry (no table line, edge not
    excited) -- they have no net map, and the status says so.
    ``live_time``: seconds per pixel, flat float64, file order -- what turns
    a net map back into counts. A region's composition is quantified on
    ``sum(net cps x live time)`` over its pixels: cps alone would weight a
    pixel by its dead time, which differs between phases.
    """
    maps: Dict[str, np.ndarray]
    fit: FileFit
    detections: List[Detection]
    quality: List[QualityWarning]
    exported: List[str]
    labels: Dict[str, str] = field(default_factory=dict)
    exported_not_fitted: List[str] = field(default_factory=list)
    detected_not_exported: List[Detection] = field(default_factory=list)
    detected_labels: Dict[str, str] = field(default_factory=dict)
    detected_at_pct: Dict[str, Optional[float]] = field(default_factory=dict)
    lines: Dict[str, Dict[str, str]] = field(default_factory=dict)
    live_time: Optional[np.ndarray] = None

    def family_for(self, name: str) -> Optional[str]:
        """Family label of an extractor label, or None if it was not fitted.

        An offered label maps directly. Anything else is parsed the way the
        quantification parses window names. A bare symbol ("Cu") means the
        line the quantification uses for that element (``lines``, made by
        ``build_counts_by_element(..., source="spectrum")``; with the matrix
        correction Cu K, without it Cu L on the Oxford reference scan),
        so a map asked for by symbol shows the intensity its at% come from;
        for an element outside the composition, the first fitted of K, L, M.
        """
        fam = self.labels.get(name) or self.detected_labels.get(name)
        if fam is not None:
            return fam
        from eds_utils import parse_element_name
        key = parse_element_name(name)
        fitted = {tuple(f) for f in self.fit.families}
        chosen = None if key.line else self.lines.get(str(key))
        if chosen and (str(key), chosen["family"]) in fitted:
            return family_label((str(key), chosen["family"]))
        for line in ((key.line,) if key.line else ("K", "L", "M")):
            if (str(key), line) in fitted:
                return family_label((str(key), line))
        return None


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
@dataclass(eq=False)          # identity: jobs live in a set
class _Job:
    key: Tuple[str, int, int]
    path: str
    state: str                       # "none" | "running" | "ready" | "failed"
    stage: Optional[str] = None
    progress: float = 0.0
    error: Optional[str] = None
    result: Optional[SpectrumNetResult] = None
    summary: Optional[dict] = None
    started: float = field(default_factory=time.monotonic)
    finished: Optional[float] = None
    root: Optional[str] = None
    done: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None
    cancel: threading.Event = field(default_factory=threading.Event)
    #: The child process (a ``subprocess.Popen``) once started; None in a
    #: thread of this process.
    process: Optional[subprocess.Popen] = None
    #: The composition's labels from the header (no fit needed), for
    #: ``get_available_elements`` while the job runs or after it failed.
    header_labels: Optional[List[str]] = None
    #: This job's file is the session's open file: the job stops when the
    #: session closes it and opens nothing within :data:`CLOSE_GRACE_S`.
    #: Per key -- opening another file unbinds every other job
    #: (:func:`_bind_session`), so closing B never stops A's fit.
    session_bound: bool = False
    closed_since: Optional[float] = None
    #: Where the work runs: ``"child"``, ``"in_process"`` (``RUN_IN_CHILD``
    #: off) or ``"in_process_fallback"`` (the child could not be started; the
    #: backend reads the cube itself and answers slowly meanwhile), with
    #: ``runner_reason`` saying why. ``None`` until the job has started.
    runner: Optional[str] = None
    runner_reason: Optional[str] = None


_registry: "OrderedDict[Tuple[str, int, int], _Job]" = OrderedDict()
#: Jobs whose thread has not finished, registered or not (an evicted job runs
#: on until it notices its cancel).
_live: set = set()
_lock = threading.Lock()

_force_window = 0
_force_lock = threading.Lock()


def _canonical(path: str) -> str:
    """Same normalisation as the loaded-files registry: no disk I/O."""
    return os.path.normcase(os.path.normpath(os.path.abspath(str(path))))


def _key(path: str) -> Optional[Tuple[str, int, int]]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (_canonical(path), int(st.st_size), int(st.st_mtime_ns))


def _lookup(path: str) -> Optional[_Job]:
    key = _key(path)               # disk I/O, outside the lock
    with _lock:
        if key is not None:
            return _registry.get(key)
        # The file cannot be stat'ed right now (a removable drive, say). A
        # result that was made from it is still the right one to serve;
        # anything else is not ours to guess.
        canon = _canonical(path)
        for job in reversed(_registry.values()):
            if job.key[0] == canon:
                return job
    return None


def _current_canonical() -> Optional[str]:
    """Canonical path of the file the h5 session has open, if any."""
    from backend.api.services.h5_session import get_current_path
    cur = get_current_path()
    return _canonical(cur) if cur else None


def _current_key() -> Optional[Tuple[str, int, int]]:
    """Registry key of the file the h5 session has open, as it is on disk now."""
    from backend.api.services.h5_session import get_current_path
    cur = get_current_path()
    return _key(cur) if cur else None


def _root_key(f: h5py.File) -> Optional[str]:
    """The acquisition slot, chosen like ``H5OINADataExtractor._find_root_key``.

    The fit has to read the cube of the same slot the window maps come from.
    """
    groups = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
    for k in groups:
        if f"{k}/EBSD" in f:
            return k
    for k in groups:
        if f"{k}/EDS" in f or f"{k}/Electron Image" in f:
            return k
    return None


def _probe(path: str) -> Tuple[bool, Optional[str]]:
    with h5py.File(path, "r") as f:
        root = _root_key(f)
        if root is None:
            return False, None
        return has_spectrum_cube(f, root), root


def _header_labels(path: str, root: str) -> Optional[List[str]]:
    """The composition's labels, from the header alone (see
    :func:`composition_labels`); None when the header cannot be read -- the
    job reports that itself, and until then the labels wait for it."""
    try:
        with h5py.File(path, "r") as f:
            meta = read_meta(f, root)
            exported, _n = job_mod.window_labels(f, root)
        return list(composition_labels(exported,
                                       default_families(exported, meta)))
    except Exception:
        logger.warning("EDS spectrum fit: could not read the element labels "
                       "of %s from its header", path, exc_info=True)
        return None


def _keeps(job: Optional[_Job]) -> bool:
    """One job per key -- except that a failed one is tried again."""
    return job is not None and job.state != "failed"


def _bind_session(key: Tuple[str, int, int]) -> None:
    """``key`` is the session's open file now; no other job is.

    Called under ``_lock`` whenever the session's file gets (or keeps) a job.
    A job unbound here runs on (a switch is not a close) and the eviction
    rules decide its fate; only the bound job stops when the session closes
    its file (:func:`_closed_too_long`).
    """
    for job in list(_registry.values()) + list(_live):
        job.session_bound = job.key == key
        if job.session_bound:
            job.closed_since = None


def start_for_file(path: str) -> None:
    """Start the spectrum fit of ``path`` unless this file already has a job.

    Probing for the cube opens the file briefly in the caller's thread, so
    ``status`` is definitive ("none" or "running") as soon as this returns.
    A file that cannot be probed is recorded as ``failed``: whether it has a
    cube is unknown, and the window path would be a guess.

    One job per key while it is running, ready or "none". A FAILED job (or
    probe) is replaced: opening the file again is the user's retry, and the
    cause (memory, a drive that was not ready) may be gone. Reads never
    retry -- ``get_result`` only starts a job for a file that has none.

    When the registry is full the oldest entry goes, but never the open
    file's current one (by key: a version of it with an older mtime is just
    an old entry, unreachable by any lookup).
    """
    key = _key(path)
    if key is None:
        raise FileNotFoundError(f"cannot stat {path!r}")
    current = _current_key()
    with _lock:
        if _keeps(_registry.get(key)):
            _registry.move_to_end(key)
            if key == current:
                _bind_session(key)
            return
    try:
        has_cube, root = _probe(path)
        state, error = ("running" if has_cube else "none"), None
    except Exception as exc:     # recorded and served as SpectrumFitFailed
        logger.exception("EDS spectrum fit: could not probe %s", path)
        has_cube, root = False, None
        state, error = "failed", f"could not read the EDS group: {exc}"
    labels = _header_labels(path, root) if has_cube else None
    current = _current_key()
    with _lock:
        if _keeps(_registry.get(key)):     # another thread got here first
            _registry.move_to_end(key)
            if key == current:
                _bind_session(key)
            return
        _registry.pop(key, None)           # a failed job: replaced
        job = _Job(key=key, path=str(path), state=state, root=root, error=error,
                   stage={"running": "reading", "failed": "failed"}.get(state),
                   header_labels=labels)
        if state != "running":
            job.finished = job.started
            job.done.set()
        _registry[key] = job
        if key == current:
            _bind_session(key)
        while len(_registry) > MAX_FILES:
            victim = next((k for k in _registry
                           if k != key and k != current), None)
            if victim is None:
                break
            _registry.pop(victim).cancel.set()
    if has_cube:
        job.thread = threading.Thread(
            target=_run, args=(job,), daemon=True,
            name=f"eds-spectrum-fit:{os.path.basename(path)}")
        with _lock:
            _live.add(job)
        job.thread.start()
        logger.info("EDS spectrum fit started for %s (%s)", path,
                    "child process" if RUN_IN_CHILD else "in this process")


def _job_for(path: Optional[str], start: bool = True) -> Optional[_Job]:
    """The job of ``path``; started first when ``start`` and there is none."""
    if not path:
        return None
    job = _lookup(path)
    if job is None and start and _key(path) is not None:
        start_for_file(path)
        job = _lookup(path)
    return job


def reset(wait_s: float = 30.0) -> None:
    """Cancel every job, forget every result, and wait for the jobs to end.

    For tests and scripts. A child process is terminated at once; a job in a
    thread of this process notices the cancel at its next progress report
    (every channel block) -- the sum fit itself reports nothing, so such a
    job inside it finishes the fit first.
    """
    with _lock:
        _registry.clear()
        jobs = list(_live)
    _stop(jobs, wait_s)


def _stop(jobs, wait_s: float) -> None:
    for job in jobs:
        job.cancel.set()
        _terminate(job.process)         # its watcher would, within _POLL_S
    deadline = time.monotonic() + wait_s
    for job in jobs:
        if job.thread is not None:
            job.thread.join(max(0.0, deadline - time.monotonic()))


@atexit.register
def _stop_at_exit() -> None:
    """Stop the jobs before the interpreter finalizes.

    Children are terminated; a job in a thread still reading the cube or
    logging while the interpreter shuts down can be frozen mid-call, so it is
    cancelled and joined for at most 10 s. A backend that leaves by
    ``os._exit`` runs no atexit; its children exit by themselves (stdin,
    ``backend.eds_spectrum.job``).
    """
    with _lock:
        jobs = list(_live)
    _stop(jobs, 10.0)


# ---------------------------------------------------------------------------
# the job
# ---------------------------------------------------------------------------
def _closed_too_long(job: _Job) -> bool:
    """True when ``job``'s file was the session's open file and the session
    has had no file open for :data:`CLOSE_GRACE_S` since. A switch to another
    file is not a close: the job is unbound (here, or at once by
    :func:`_bind_session` when the other file's job starts) and runs on;
    the eviction rules decide (``start_for_file``)."""
    if not job.session_bound:
        return False
    current = _current_canonical()
    if current is not None:
        job.closed_since = None
        if current != job.key[0]:
            job.session_bound = False
        return False
    now = time.monotonic()
    if job.closed_since is None:
        job.closed_since = now
    return now - job.closed_since >= CLOSE_GRACE_S


def _check(job: _Job) -> None:
    """Raise :class:`_Cancelled` when the job has to stop."""
    if job.cancel.is_set():
        raise _Cancelled("evicted or stopped")
    if _closed_too_long(job):
        job.cancel.set()
        raise _Cancelled("the file was closed")


def _set(job: _Job, stage: str, progress: float) -> None:
    _check(job)
    job.stage = stage
    job.progress = float(min(max(progress, job.progress), 1.0))


def _run(job: _Job) -> None:
    """The job's thread: the work in a child (or here), then the result."""
    t0 = time.perf_counter()
    try:
        _check(job)
        if RUN_IN_CHILD:
            out, where = _compute_in_child(job)
        else:
            job.runner = "in_process"
            out, where = _compute_here(job), "this process"
        _install(job, out, time.perf_counter() - t0, where)
    except _Cancelled as exc:
        job.error, job.finished = "cancelled", time.monotonic()
        job.stage = "failed"
        job.state = "failed"
        logger.info("EDS spectrum fit of %s cancelled (%s)", job.path, exc)
    except _ChildFailed as exc:
        job.error = str(exc)
        job.finished = time.monotonic()
        job.stage = "failed"
        job.state = "failed"
        logger.error("EDS spectrum fit of %s failed: %s\n%s", job.path, exc,
                     exc.details)
    except Exception as exc:
        job.error = f"{type(exc).__name__}: {exc}"
        job.finished = time.monotonic()
        job.stage = "failed"
        job.state = "failed"
        logger.exception("EDS spectrum fit of %s failed", job.path)
    finally:
        job.done.set()
        with _lock:
            _live.discard(job)


def _compute_here(job: _Job) -> dict:
    """The work in this thread. The fit's functions are looked up HERE, at
    call time, so a test that patches them reaches the job."""
    return job_mod.compute(
        job.path, job.root, lambda stage, x: _set(job, stage, x),
        sum_spectrum=sum_spectrum, fit_sum_spectrum=fit_sum_spectrum,
        project_cube=project_cube)


def _install(job: _Job, out: dict, elapsed: float, where: str) -> None:
    ff = out["fit"]
    result = _build_result(out["maps"], ff, out["detections"], out["quality"],
                           out["exported"], live_time=out["live_time"])
    job.summary = _summary(result)
    job.result = result
    job.finished = time.monotonic()
    job.stage, job.progress = "done", 1.0
    job.state = "ready"
    sec = out.get("seconds") or {}
    logger.info(
        "EDS spectrum fit of %s done in %.1f s in %s (read %.1f, fit %.1f, "
        "project %.1f): chi2_red %.2f, %d families, detected not exported: "
        "%s", job.path, elapsed, where, sec.get("read", float("nan")),
        sec.get("fit", float("nan")), sec.get("project", float("nan")),
        ff.chi2_red, len(ff.families),
        ", ".join(f"{d.element} {d.family} ({d.z:.1f} sigma)"
                  for d in result.detected_not_exported) or "none")


# ---------------------------------------------------------------------------
# the child process (protocol: backend.eds_spectrum.job)
# ---------------------------------------------------------------------------
def _spawn(job: _Job) -> subprocess.Popen:
    """Start ``python -c job.CHILD_CODE <path> <root>`` for ``job``.

    ``sys.executable``: the interpreter the backend runs on, installed or
    not. No console window on Windows, below normal priority (the requests
    of the page come first).
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_PROJECT_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
    return subprocess.Popen(
        [sys.executable, "-c", job_mod.CHILD_CODE, job.path, str(job.root)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(_PROJECT_ROOT), env=env, creationflags=flags)


def _compute_in_child(job: _Job) -> Tuple[dict, str]:
    """The work in a child process; this thread relays progress and waits.

    Returns ``(compute's dict, where)``. Raises :class:`_Cancelled` (the
    child is terminated) or :class:`_ChildFailed`. A child that cannot even
    be started is an environment problem, not the file's: the work then runs
    here, as it did before the child process existed, and the log AND :func:`status` say so
    (``runner: "in_process_fallback"`` with ``runner_reason``) -- the backend
    stalls while it reads the cube, and the page can tell the user why.
    """
    try:
        proc = _spawn(job)
    except Exception as exc:
        job.runner_reason = (f"its own process could not be started "
                             f"({type(exc).__name__}: {exc})")
        job.runner = "in_process_fallback"
        logger.error("EDS spectrum fit: could not start its process for %s "
                     "(%s: %s); fitting inside the backend instead -- the "
                     "backend answers slowly while it reads the cube",
                     job.path, type(exc).__name__, exc)
        return _compute_here(job), "this process (the child could not start)"
    job.process = proc
    job.runner = "child"
    events: "queue.Queue" = queue.Queue()
    tail: deque = deque(maxlen=40)
    name = os.path.basename(job.path)
    readers = [
        threading.Thread(target=_read_protocol, args=(proc.stdout, events),
                         daemon=True, name=f"eds-spectrum-fit-out:{name}"),
        threading.Thread(target=_read_stderr, args=(proc.stderr, tail, name),
                         daemon=True, name=f"eds-spectrum-fit-err:{name}"),
    ]
    for t in readers:
        t.start()
    graceful = False
    try:
        while True:
            _check(job)
            try:
                kind, payload = events.get(timeout=_POLL_S)
            except queue.Empty:
                continue
            if kind == "progress":
                _set(job, str(payload.get("stage")),
                     float(payload.get("progress", 0.0)))
            elif kind == "result":
                graceful = True
                return payload, f"a child process (pid {proc.pid})"
            elif kind == "error":
                graceful = True
                raise _ChildFailed(str(payload.get("error")),
                                   str(payload.get("traceback") or ""))
            else:                       # "eof" / "broken": no answer at all
                code = _wait(proc, 10.0)
                for t in readers:
                    t.join(2.0)
                # A child that dies before its first protocol line (an import
                # that fails in this interpreter, say) leaves its reason only
                # on stderr: the last line goes into the message the user
                # sees, the rest stays in the log.
                last = tail[-1] if tail else ""
                raise _ChildFailed(
                    f"the spectrum-fit process ended without an answer "
                    f"(exit code {code}{'; ' + payload if payload else ''})"
                    + (f": {last}" if last else ""),
                    "\n".join(tail))
    finally:
        if not graceful or _wait(proc, 10.0) is None:
            _terminate(proc)
        for t in readers:
            t.join(2.0)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                stream.close()
            except Exception:
                pass


def _wait(proc: subprocess.Popen, timeout: float) -> Optional[int]:
    """The exit code, or None if ``proc`` is still running after ``timeout``."""
    try:
        return proc.wait(timeout)
    except subprocess.TimeoutExpired:
        return None


def _terminate(proc: Optional[subprocess.Popen]) -> None:
    """End ``proc`` now (idempotent; ``None`` and an ended process are fine)."""
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
    except OSError:
        pass
    if _wait(proc, 10.0) is None:
        try:
            proc.kill()
        except OSError:
            pass
        _wait(proc, 10.0)


def _read_protocol(stream, events: "queue.Queue") -> None:
    """Reader thread: the child's protocol lines into ``events``.

    Ends with exactly one of ``("result", dict)``, ``("error", dict)``,
    ``("eof", None)`` or ``("broken", message)``. A line that is not one of
    the protocol's objects is skipped (nothing else should reach this
    descriptor -- the child points its stdout at stderr first).
    """
    try:
        for raw in iter(stream.readline, b""):
            try:
                msg = json.loads(raw)
            except ValueError:
                logger.debug("EDS spectrum fit: not a protocol line: %r", raw[:200])
                continue
            if not isinstance(msg, dict):
                continue
            if "stage" in msg:
                events.put(("progress", msg))
            elif "error" in msg:
                events.put(("error", msg))
                return
            elif msg.get("result"):
                events.put(("result", pickle.load(stream)))
                return
    except Exception as exc:
        events.put(("broken", f"{type(exc).__name__}: {exc}"))
        return
    events.put(("eof", None))


def _read_stderr(stream, tail: deque, name: str) -> None:
    """Reader thread: what the child writes to stderr, into the log (and the
    last lines into ``tail``, for the error of a child that died)."""
    try:
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                tail.append(line)
                logger.info("EDS spectrum fit of %s (child): %s", name, line)
    except Exception:
        pass


def _alpha_first(sub_line: str) -> int:
    """0 for an alpha (or unqualified) sub-line, 1 for anything else."""
    s = (sub_line or "").lower()
    return 0 if not s or s.startswith(("a", "α")) else 1


def _exported_families(exported: List[str], fitted) -> Tuple[
        Dict[Tuple[str, str], str], set, List[str]]:
    """AZtec's label per exported family (an alpha label wins over a beta
    one), the exported element symbols, and the exported labels whose family
    the model does not carry."""
    from eds_utils import parse_element_name

    by_family: Dict[Tuple[str, str], str] = {}
    exported_symbols = set()
    exported_not_fitted = []
    for lab in exported:
        key = parse_element_name(lab)
        sym = str(key)
        exported_symbols.add(sym)
        lines = (key.line,) if key.line else ("K", "L", "M")
        if not any((sym, ln) in fitted for ln in lines):
            exported_not_fitted.append(lab)
            continue
        if key.line:
            prev = by_family.get((sym, key.line))
            if prev is None or (_alpha_first(key.sub_line)
                                < _alpha_first(parse_element_name(prev).sub_line)):
                by_family[(sym, key.line)] = lab
    return by_family, exported_symbols, exported_not_fitted


def _labels_of(symbols, fitted, by_family) -> Dict[str, str]:
    """Label -> family label for every fitted family of ``symbols``: AZtec's
    own label where it exported that family, else a generated one."""
    out: Dict[str, str] = {}
    for sym in sorted(symbols):
        for fam in ("K", "L", "M"):
            if (sym, fam) not in fitted:
                continue
            lab = by_family.get((sym, fam)) or _GENERATED_LABEL[fam].format(sym)
            out[lab] = family_label((sym, fam))
    return out


def composition_labels(exported: List[str], families) -> Dict[str, str]:
    """The composition's labels -> family label (``SpectrumNetResult.labels``)
    for AZtec's window labels ``exported`` and the families the fit models.

    Needs no fit: the job fits exactly ``default_families(exported, meta)``
    (``FileFit.families`` is that list), so the labels are known from the
    header the moment the file is opened, and ``get_available_elements``
    answers with them while the job runs.
    """
    fitted = {tuple(f) for f in families}
    by_family, symbols, _ = _exported_families(exported, fitted)
    return dict(sorted(_labels_of(symbols, fitted, by_family).items()))


def _build_result(maps, ff: FileFit, detections, warnings,
                  exported: List[str],
                  live_time: Optional[np.ndarray] = None) -> SpectrumNetResult:
    """Labels, the detected
    elements, the lines.

    The composition is AZtec's exported elements, exactly as on the window
    path. An element the spectrum shows that AZtec did not export is NOT
    added: a sum-level detection (Ti K at 5.2 sigma on the Oxford reference scan) is counting
    noise per pixel, and per-pixel at% clips its negative half at 0 -- on
    The Oxford reference scan 17.9 % of the pixels read Ti >= 2 at% and a pure-Si pixel 1.15 at%.
    It gets a label and a net map
    (``detected_labels``) and its at% in the whole map's summed composition
    (``detected_at_pct``), for the notice.
    """
    from eds_utils import build_counts_by_element

    fitted = {tuple(fam) for fam in ff.families}
    by_family, exported_symbols, exported_not_fitted = _exported_families(
        exported, fitted)
    if exported_not_fitted:
        logger.warning("EDS spectrum fit: exported windows without a fitted "
                       "family (no net map, left out): %s",
                       ", ".join(exported_not_fitted))

    extra = [d for d in detections
             if d.element not in exported_symbols and d.element not in _NEVER_ADDED]
    labels = _labels_of(exported_symbols, fitted, by_family)
    detected_labels = _labels_of({d.element for d in extra}, fitted, by_family)

    # The line each element is quantified through, by the production policy
    # itself (the label stands in for the counts; only the choice is read).
    chosen = build_counts_by_element([(lab, lab) for lab in labels],
                                     beam_kv=ff.meta.beam_kv, source="spectrum")
    lines = {str(el): {"label": lab, "family": el.line}
             for el, lab in chosen.items()}
    lt = (None if live_time is None
          else np.asarray(live_time, dtype=np.float64).ravel())
    return SpectrumNetResult(
        maps=maps, fit=ff, detections=list(detections), quality=list(warnings),
        exported=list(exported), labels=dict(sorted(labels.items())),
        exported_not_fitted=exported_not_fitted, detected_not_exported=extra,
        detected_labels=dict(sorted(detected_labels.items())),
        detected_at_pct=_detected_at_pct(maps, labels, detected_labels, extra,
                                         lt, ff.meta.beam_kv),
        lines=dict(sorted(lines.items())), live_time=lt)


def _detected_at_pct(maps, labels, detected_labels, extra, live_time,
                     beam_kv) -> Dict[str, Optional[float]]:
    """Each detected element's at% in the whole map, were it added.

    The composition of the map's summed net counts (cps x live time over all
    pixels -- the projection is linear, so this is the sum spectrum's own
    fit), quantified on the spectrum curve with the exported elements plus
    every detected one. A size for the notice, not a number any consumer
    uses. ``None`` for an element that cannot be priced.
    """
    from eds_utils import (build_counts_by_element, counts_to_weight_pct,
                           weight_pct_to_atomic_pct)
    if not extra:
        return {}
    try:
        pairs = []
        for lab, fam in list(labels.items()) + list(detected_labels.items()):
            cps = np.asarray(maps[fam], dtype=np.float64)
            total = (float(np.sum(cps * live_time)) if live_time is not None
                     and live_time.size == cps.size else float(np.sum(cps)))
            pairs.append((lab, np.array([total])))
        counts = build_counts_by_element(pairs, beam_kv=beam_kv,
                                         source="spectrum")
        at = weight_pct_to_atomic_pct(counts_to_weight_pct(counts,
                                                           beam_kv=beam_kv))
        got = {str(el): float(v[0]) for el, v in at.items()}
    except Exception:
        logger.warning("could not size the detected elements", exc_info=True)
        got = {}
    return {d.element: (round(got[d.element], 3) if d.element in got
                        and np.isfinite(got[d.element]) else None)
            for d in extra}


def _finite(x) -> Optional[float]:
    x = float(x)
    return x if np.isfinite(x) else None


def _summary(res: SpectrumNetResult) -> dict:
    """The JSON ``fit`` block of :func:`status`."""
    p = res.fit.params
    return {
        "model_version": res.fit.model_version,
        "chi2_red": _finite(res.fit.chi2_red),
        "params": {"offset_kev": _finite(p.offset_kev), "gain": _finite(p.gain),
                   "fwhm_mnka_kev": _finite(p.fwhm_mnka_kev),
                   "absorption_a": _finite(p.absorption_a)},
        "beam_kv": _finite(res.fit.meta.beam_kv),
        "lines": res.lines,
        # Listed, never quantified: ``label``
        # is the net map "add as map" shows, ``at_pct_map_sum`` its size.
        "detected_not_exported": [
            {"element": d.element, "family": d.family, "z": _finite(d.z),
             "label": next((lab for lab, fam in res.detected_labels.items()
                            if fam == family_label((d.element, d.family))),
                           None),
             "at_pct_map_sum": res.detected_at_pct.get(d.element)}
            for d in res.detected_not_exported],
        "quality_warnings": [
            {"family": q.family, "residual_fraction": _finite(q.residual_fraction)}
            for q in res.quality],
        "exported_not_fitted": list(res.exported_not_fitted),
    }


# ---------------------------------------------------------------------------
# public reads
# ---------------------------------------------------------------------------
def status(path: Optional[str]) -> dict:
    """State of the spectrum fit of ``path``. Cheap; never blocks or starts.

    ``state``: ``"none"`` (no cube: window path), ``"running"``, ``"ready"``,
    ``"failed"``. ``fit`` is filled when ready. ``runner``: where the work
    runs (``"child"``, ``"in_process"``, ``"in_process_fallback"``; ``None``
    before it started and for a file without a cube), ``runner_reason`` why
    it fell back.
    """
    job = _job_for(path, start=False)
    if job is None:
        return {"state": "none", "progress": 0.0, "stage": None, "error": None,
                "elapsed_s": None, "fit": None, "runner": None,
                "runner_reason": None}
    end = job.finished if job.finished is not None else time.monotonic()
    return {
        "state": job.state,
        "progress": round(float(job.progress), 4),
        "stage": job.stage,
        "error": job.error,
        "elapsed_s": round(end - job.started, 3),
        "fit": job.summary if job.state == "ready" else None,
        "runner": job.runner,
        "runner_reason": job.runner_reason,
    }


def get_result(path: str, wait_s: float = 0.0,
               start: bool = True) -> Optional[SpectrumNetResult]:
    """The finished result for ``path``; ``None`` when the file has no cube.

    Raises :class:`SpectrumFitPending` when the job has not finished within
    ``wait_s`` seconds (default 0: request paths never block), and
    :class:`SpectrumFitFailed` when it failed. A file without a job yet gets
    one started here unless ``start`` is False; then a missing job is an
    error, not "no cube" (a caller passing ``start=False`` knows the file has
    one).
    """
    job = _job_for(path, start=start)
    if job is None and not start:
        raise SpectrumFitFailed(
            f"{os.path.basename(str(path))} is no longer the open file and "
            f"its spectrum fit was dropped; open it again")
    if job is None or job.state == "none":
        return None
    if job.state == "running" and wait_s > 0:
        job.done.wait(wait_s)
    if job.state == "running":
        raise SpectrumFitPending(job.progress, job.stage)
    if job.state == "failed":
        raise SpectrumFitFailed(job.error or "unknown error")
    return job.result


def spectrum_applies(path: Optional[str]) -> bool:
    """True when ``path`` has a spectrum cube (job running, ready or failed)."""
    job = _job_for(path, start=True)
    return job is not None and job.state != "none"


@contextlib.contextmanager
def force_window_source() -> Iterator[None]:
    """While active, ``get_active_extractor()`` serves window maps.

    PROCESS-WIDE, not per thread: route handlers run on other threads than a
    test or a script that drives them (TestClient's portal, ``to_thread``), so
    a thread-local switch would not reach them. For tests and the before/after
    compare script only -- never use it in a request path. Nests.
    """
    global _force_window
    with _force_lock:
        _force_window += 1
    try:
        yield
    finally:
        with _force_lock:
            _force_window -= 1


def window_source_forced() -> bool:
    return _force_window > 0


def window_maps(extractor):
    """``(labels, read)`` for AZtec's RAW window maps of ``extractor``.

    For consumers whose thresholds or file format mean raw counts -- the
    preflight's counts-per-pixel check, the batch export's ``/EDS`` group.
    Needs no fit: on a spectrum-source extractor it reads
    ``get_window_elements`` / ``get_window_map`` (cut to the crop by
    ``CroppedExtractor``); on any other the element maps ARE the windows.
    """
    if quant_source_of(extractor) == "spectrum":
        return list(extractor.get_window_elements()), extractor.get_window_map
    return list(extractor.get_available_elements()), extractor.get_element_map


def detected_only_labels(extractor) -> List[str]:
    """Labels of the elements the spectrum shows but AZtec did not export.

    Their net maps are served by ``get_element_map``; they are in no
    composition. Empty on the window source.
    Raises the fit's 503 / 500 like every other spectrum read.
    """
    if quant_source_of(extractor) != "spectrum":
        return []
    return list(extractor.get_detected_only_elements())


def quant_source_of(extractor) -> str:
    """Which intensities ``extractor`` serves: ``"spectrum"`` or ``"window"``.

    A :class:`SpectrumNetExtractor` (also under a ``CroppedExtractor``, which
    forwards the attribute) declares ``"spectrum"``; everything else is the
    vendor's window maps. Only a declared string counts: a test double that
    answers every attribute (``MagicMock``) is a window extractor.
    """
    src = getattr(extractor, "quant_source", "window")
    return src if isinstance(src, str) else "window"


# ---------------------------------------------------------------------------
# the extractor
# ---------------------------------------------------------------------------
class SpectrumNetExtractor:
    """A raw extractor whose EDS element maps are net line intensities.

    ``get_element_map(name)``: flat float32 net counts per live second of the
    family ``name`` belongs to, UNCLIPPED, a fresh copy per call (callers may
    clip in place). ``get_available_elements()``: the COMPOSITION -- per
    element AZtec exported one label per fitted family, AZtec's own label
    where it exported that family, else ``"<El> Kα1"`` / ``"<El> Lα1,2"`` /
    ``"<El> Mα"``; every quantification reads these. They are known from the
    header (:func:`composition_labels`), so this answers while the fit runs
    and after it failed; the maps and numbers wait.
    ``get_detected_only_elements()``:
    the labels of the elements the spectrum shows but AZtec did not export
    -- ``get_element_map`` serves their net
    maps, nothing quantifies them. ``get_window_map(name)``
    and ``get_window_elements()``: the raw window maps and AZtec's labels for
    them (no fit needed). ``get_live_time()``: seconds per pixel, flat, the
    factor from net cps to net counts. Everything else is the wrapped
    extractor's.

    Every method that needs the fit asks for it with ``wait_s=0``: while the
    job runs it raises :class:`SpectrumFitPending` at once, after a failure
    :class:`SpectrumFitFailed`.
    """

    quant_source = "spectrum"

    def __init__(self, extractor, path: str):
        object.__setattr__(self, "_ext", extractor)
        object.__setattr__(self, "_path", str(path))

    @property
    def raw(self):
        """The wrapped extractor (window maps, the file as it is)."""
        return self._ext

    @property
    def source_path(self) -> str:
        return self._path

    def __getattr__(self, name: str):
        return getattr(self._ext, name)

    def __repr__(self) -> str:
        return f"<SpectrumNetExtractor {os.path.basename(self._path)!r}>"

    def _may_start(self) -> bool:
        # Only the open file's view may start a job (the first read after a
        # failed start). A view of a file the session has since left -- a
        # request that began before the switch -- must not refit it.
        return _canonical(self._path) == _current_canonical()

    def _result(self) -> SpectrumNetResult:
        res = get_result(self._path, wait_s=0, start=self._may_start())
        if res is None:
            raise RuntimeError(
                f"{self._path!r} has no spectrum fit, but its extractor was "
                f"wrapped for one")
        return res

    def require_fit(self) -> None:
        """Raise the fit's 503 / 500 unless the net maps are ready.

        For endpoints whose purpose is EDS numbers and that change state:
        they call this first, so they refuse before anything is written.
        """
        self._result()

    def spectrum_fit_summary(self) -> dict:
        """The ``fit`` block of :func:`status` (raises while not ready)."""
        self._result()
        return status(self._path)["fit"]

    def get_available_elements(self) -> List[str]:
        job = _job_for(self._path, start=self._may_start())
        if (job is not None and job.state in ("running", "failed")
                and job.header_labels is not None):
            return list(job.header_labels)
        return list(self._result().labels)

    def get_detected_only_elements(self) -> List[str]:
        """Labels of detected-not-exported elements: a net map, no at%."""
        return list(self._result().detected_labels)

    def get_window_elements(self) -> List[str]:
        """AZtec's window labels (what ``get_window_map`` reads; needs no fit)."""
        return self._ext.get_available_elements()

    def get_element_map(self, element_name):
        res = self._result()
        fam = res.family_for(element_name)
        if fam is None or fam not in res.maps:
            logger.warning("EDS net map not available: %r is not a fitted "
                           "family of this file", element_name)
            return None
        return res.maps[fam].copy()

    def get_element_map_2d(self, element_name):
        data = self.get_element_map(element_name)
        if data is None:
            return None
        if data.size == self._ext.flat_point_count:
            return self._ext._flat_to_map(data)
        return data

    def get_window_map(self, element_name):
        """AZtec's window map, untouched (needs no fit)."""
        return self._ext.get_element_map(element_name)

    def get_live_time(self) -> np.ndarray:
        """Live time per pixel in seconds, flat float64 in file order (a copy).

        Net map x live time = net counts, the quantity a region's composition
        sums (``routes.eds._SummedNetComposition``). Read with the fit, so it
        always belongs to the maps it is multiplied with.
        """
        res = self._result()
        if res.live_time is None:
            raise RuntimeError(
                f"the spectrum fit of {self._path!r} carries no live time")
        return res.live_time.copy()
