"""Jobs for add-on runs, because a run is longer than a request.

axios gives up at 300 s (services/api.js:18) and the run route had no progress
channel (routes/addons.py:670-671), so a six-minute analysis aborted in the
client while the thread kept going and stored a result nobody fetched.

In-memory and per-process, like the rest of this backend's state: a job is
meaningful only while the process computing it lives.
"""
from __future__ import annotations

import copy
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_JOBS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.RLock()

#: Progress messages are polled as JSON and written by a stranger's code, so
#: they are cut. 500 characters is a sentence and a number; anything longer is
#: a traceback or a dumped array, and it would be re-sent on every poll.
_MESSAGE_LIMIT = 500


def _safe_message(message: Any) -> Optional[str]:
    """Render a progress message, or None if it cannot be rendered.

    ``str(x)`` runs ``x.__str__``, which is the add-on's code and may raise --
    the one call in ``report_progress`` that could still travel back into a
    stranger's run and be counted against it by ``_note_crash``. None means
    "keep the previous message", the same rule the fraction follows.
    """
    try:
        text = str(message)
    except Exception:                       # noqa: BLE001 -- see docstring
        logger.warning("add-on reported a progress message whose __str__ "
                       "raised; keeping the previous message", exc_info=True)
        return None
    if len(text) > _MESSAGE_LIMIT:
        return text[:_MESSAGE_LIMIT] + " [...]"
    return text


def start_job(name: str, analysis_key: str, result_id: str) -> str:
    # Prune here rather than on a timer: this is the only moment the registry
    # is known to be about to grow, and a sweep nobody calls is dead code.
    prune_jobs()
    job_id = uuid.uuid4().hex
    with _LOCK:
        _JOBS[job_id] = {
            "id": job_id, "name": name, "analysis_key": analysis_key,
            "result_id": result_id, "state": "running", "message": "",
            "fraction": None, "started_at": time.time(), "finished_at": None,
            "outputs": None, "ignored_params": None, "context": None,
            "error": None, "reason": None,
        }
    return job_id


def report_progress(job_id: str, message: str,
                    fraction: Optional[float] = None) -> None:
    """Record progress. Nothing in here may raise into add-on code.

    An unknown or finished job is ignored, because the add-on holds this
    callback and a pruned job must not become an exception inside a stranger's
    code. An out-of-range fraction is DROPPED for the same reason, one step
    further: raising would travel unwrapped into ``run_analysis``'s
    ``except Exception``, be counted by ``_note_crash``, and push the add-on
    towards the automatic disable at CRASH_LIMIT -- an arithmetic slip in a
    progress bar destroying a six-minute analysis and then the add-on.

    Not clamped either: 1.5 displayed as 100 % turns the bug into a claim.
    The last good fraction stands and the bad one is logged.
    """
    ok = True
    if fraction is not None:
        try:
            value = float(fraction)
        except (TypeError, ValueError):
            ok = False
        else:
            # `not (0 <= x <= 1)` is not the same as `x < 0 or x > 1`: NaN
            # fails every comparison, so it would pass the second form.
            ok = 0.0 <= value <= 1.0
    if not ok:
        logger.warning("add-on reported an unusable progress fraction (%r) "
                       "for job %s; keeping the previous value",
                       fraction, job_id)
    text = _safe_message(message)
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None or job["state"] != "running":
            return
        if text is not None:
            job["message"] = text
        if fraction is not None and ok:
            job["fraction"] = value


def get_job(job_id: str, *,
            include_outputs: bool = True) -> Optional[Dict[str, Any]]:
    """A DEEP copy: a caller that mutates what it reads must not edit the registry.

    ``dict(job)`` would be shallow, and the two entries a caller is most likely
    to touch -- ``outputs`` and ``context`` -- are exactly the nested ones it
    would then be sharing with the registry. The route that serialises a
    finished job walks into both.

    ``include_outputs=False`` drops that list BEFORE the copy, for the poll,
    which discards it anyway. Measured on a 50k-row table output: 70.8 ms per
    call, spent on the event loop, once per tick of a progress bar -- so a poll
    that exists to be cheap was blocking the health check with a copy it threw
    away. Dropped before, not after: copying and then deleting saves nothing.

    PRUNES FIRST, and that is what makes the routes' 404 sentence true. It
    says a finished job is dropped an hour after it finished; with the sweep
    only in ``start_job``, a user who ran once and never again kept that job --
    and its outputs -- for the life of the process, so the sentence described a
    policy that had not run. Here it runs at the one moment the answer is
    about to be read, which is the only moment the claim is observable. The
    cost is a dict scan over a handful of entries, under a lock the call takes
    anyway.
    """
    prune_jobs()
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        if include_outputs:
            return copy.deepcopy(job)
        return copy.deepcopy({k: v for k, v in job.items() if k != "outputs"})


def finish_job(job_id: str, *, outputs: Optional[List[dict]] = None,
               ignored_params: Optional[Tuple[str, ...]] = None,
               context: Optional[dict] = None,
               error: Optional[str] = None,
               reason: Optional[str] = None) -> None:
    """Put a job into its terminal state. The FIRST terminal write wins.

    Two refusals, both logged rather than silent -- unlike ``report_progress``,
    nothing here is held by add-on code, so there is no one to protect from an
    exception and no reason to swallow the event:

    * an unknown id means a finished analysis has nowhere to go. It is a bug
      in the caller (or a job pruned mid-run), and a result vanishing without
      a line in the log is how it would stay a bug.
    * an already-finished job is left alone so that a worker's ``finally`` can
      call this unconditionally: the guarantee that no job stays ``running``
      after its thread ends (see ``prune_jobs``) is bought by that call, and it
      must not be able to overwrite the real result it is insuring.
    """
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            logger.warning("finish_job for unknown job %s; the result of this "
                           "run is discarded (error=%r)", job_id, error)
            return
        if job["state"] != "running":
            logger.warning("finish_job for job %s, which is already %s; "
                           "keeping the first result", job_id, job["state"])
            return
        job["state"] = "failed" if error else "done"
        job["outputs"] = outputs
        job["ignored_params"] = ignored_params
        job["context"] = context
        job["error"] = error
        job["reason"] = reason
        job["finished_at"] = time.time()


def prune_jobs(max_age_s: float = 3600) -> int:
    """Drop finished jobs older than max_age_s. Running jobs are never dropped.

    A running job is never dropped and there is no stale-running cutoff, which
    is a decision and not an omission. A cutoff cannot tell a dead worker from
    a slow one, and the slow one is the case this whole module exists for: a
    dictionary run on a large map legitimately outlives any threshold worth
    setting, and killing its job would report a failure that did not happen.

    What makes the omission safe is the worker, not a timer: the job route's
    thread finishes its job in a ``finally``, so a worker that dies of an
    exception -- at any depth, including one raised in the add-on's own
    ``__del__`` -- still lands in a terminal state. The one case the ``finally``
    cannot cover is the process itself going away, and this registry is
    in-memory, so it goes with it. A job left ``running`` therefore means a
    thread that is still alive.
    """
    cutoff = time.time() - max_age_s
    with _LOCK:
        dead = [k for k, j in _JOBS.items()
                if j["state"] != "running" and (j["finished_at"] or 0) < cutoff]
        for k in dead:
            del _JOBS[k]
    return len(dead)
