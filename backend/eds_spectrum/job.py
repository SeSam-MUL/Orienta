"""One file's spectrum fit as a job: the work, and the child process it runs in.

:func:`compute` is the work -- read the sum spectrum, fit it, project every
pixel -- reporting its stage and progress through a callback. The backend
(``backend.api.services.eds_spectrum_service``) runs it in a CHILD PROCESS
(:func:`run_child`): the cube read decompresses for tens of seconds on a
large map, and inside the backend's own process that work shares the
interpreter lock and h5py's global lock with every request thread. Measured
on the 29 GB scan with the job in a thread: ``/api/health`` 9.6-11.4 s and a
pattern request 20.8 s while the cube was read.

The child is started as ``python -c`` (``CHILD_CODE``), not with
``multiprocessing``: a spawned multiprocessing child runs the parent's
``__main__`` script again, and a backend started from a script that calls
``uvicorn.run`` at module level (the app-run harness does) would start a
second server inside the child.

Protocol, on a descriptor of the child's own (the code dups stdout before
anything is imported and points stdout at stderr, so nothing a library prints
can reach it): one JSON object per line, ``{"stage": s, "progress": x}``
while it works, then either ``{"error": message, "traceback": text}`` or
``{"result": true}`` followed by a pickle of the dict :func:`compute`
returns. The child exits as soon as its stdin reaches end-of-file: the
parent never writes to it, so EOF means the parent closed it or is gone --
a backend that leaves by ``os._exit`` (its watchdogs) or is killed takes the
child with it (exit code :data:`EXIT_PARENT_GONE`).

Nothing here imports FastAPI or the app.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import threading
import time
import traceback
from typing import Callable, Dict, List, Tuple

import h5py

from .cube import live_time as read_live_time
from .cube import read_meta
from .cube import sum_spectrum as _sum_spectrum
from .fit import detect, quality
from .fit import fit_sum_spectrum as _fit_sum_spectrum
from .fit import project_cube as _project_cube
from .model import default_families

#: Share of the progress bar each stage owns (the fit itself reports none).
P_READ, P_FIT, P_PROJECT = 0.30, 0.45, 0.99
#: Exit code of a child whose parent went away (stdin reached end-of-file).
EXIT_PARENT_GONE = 86
#: Exit code of a child whose job raised (the message went to the parent).
EXIT_FAILED = 1
#: What the parent runs: ``sys.executable -c CHILD_CODE <path> <root>``.
#: Duplicates stdout as the protocol channel BEFORE importing anything.
CHILD_CODE = ("import os, sys; fd = os.dup(1); os.dup2(2, 1); "
              "from backend.eds_spectrum.job import run_child; "
              "run_child(fd, sys.argv[1:])")

#: ``report(stage, progress)``; may raise to stop the job.
Report = Callable[[str, float], None]


def window_labels(f: h5py.File, root: str) -> Tuple[List[str], Dict[str, int]]:
    """AZtec's window labels (as ``get_available_elements`` lists them) and
    each window's pixel count."""
    grp = f.get(f"{root}/EDS/Data/Window Integral")
    out: Dict[str, int] = {}
    if isinstance(grp, h5py.Group):
        for name in grp.keys():
            item = grp[name]
            if isinstance(item, h5py.Dataset):
                out[name] = int(item.shape[0]) if item.ndim else 1
            elif isinstance(item, h5py.Group) and "Counts" in item:
                out[name] = int(item["Counts"].shape[0])
    return sorted(out), out


def compute(path: str, root: str, report: Report, *,
            sum_spectrum=_sum_spectrum, fit_sum_spectrum=_fit_sum_spectrum,
            project_cube=_project_cube) -> dict:
    """The spectrum fit of ``path``'s acquisition slot ``root``.

    Returns ``maps`` (family label -> flat float32 net cps, unclipped),
    ``fit`` (the :class:`FileFit`), ``detections``, ``quality``, ``exported``
    (AZtec's window labels), ``live_time`` (flat, file order) and ``seconds``
    per stage. The three keyword functions are the backend's, so a test that
    patches them there reaches a job run in the backend's own thread.
    """
    t0 = time.perf_counter()
    with h5py.File(path, "r") as f:
        report("reading", 0.0)
        meta = read_meta(f, root)
        exported, n_window = window_labels(f, root)
        n_cube = int(f[f"{root}/EDS/Data/Spectrum"].shape[0])
        bad = {k: n for k, n in n_window.items() if n != n_cube}
        if bad:
            raise ValueError(
                f"the spectrum cube has {n_cube} pixels but the window "
                f"maps have {sorted(set(bad.values()))}")
        lt = read_live_time(f, root)
        if lt.size != n_cube:
            raise ValueError(f"{lt.size} live times for {n_cube} pixels")
        y = sum_spectrum(f, root, progress=lambda x: report(
            "reading", P_READ * x))
        t_read = time.perf_counter() - t0
        report("fitting", P_READ)
        ff = fit_sum_spectrum(y, meta, default_families(exported, meta))
        t_fit = time.perf_counter() - t0 - t_read
        report("projecting", P_FIT)
        maps = project_cube(f, ff, root, y_sum=y, progress=lambda x: report(
            "projecting", P_FIT + (P_PROJECT - P_FIT) * x))
    t_project = time.perf_counter() - t0 - t_read - t_fit
    return {"maps": maps, "fit": ff, "detections": detect(ff),
            "quality": quality(ff, y), "exported": exported, "live_time": lt,
            "seconds": {"read": t_read, "fit": t_fit, "project": t_project}}


# ---------------------------------------------------------------------------
# the child process
# ---------------------------------------------------------------------------
def _exit_when_parent_goes() -> None:
    """Exit at once when stdin reaches end-of-file (see the module docstring).

    ``os.read`` on the descriptor, not ``sys.stdin``: a daemon thread blocked
    inside the buffered reader holds its lock, and interpreter shutdown would
    then die on it.
    """
    def watch():
        try:
            while os.read(0, 4096):
                pass
        except OSError:
            pass
        os._exit(EXIT_PARENT_GONE)

    threading.Thread(target=watch, name="eds-fit-parent-watch",
                     daemon=True).start()


def run_child(proto_fd: int, argv: List[str]) -> None:
    """Entry point of the child process (see ``CHILD_CODE``). Never returns.

    ``argv``: the file and the acquisition slot. Below normal priority on
    POSIX (``nice``); on Windows the parent creates the process that way.
    """
    _exit_when_parent_goes()
    if hasattr(os, "nice"):
        try:
            os.nice(5)
        except OSError:
            pass
    proto = os.fdopen(proto_fd, "wb")
    last = {"stage": None, "progress": -1.0}

    def send(obj) -> None:
        proto.write(json.dumps(obj).encode("utf-8") + b"\n")
        proto.flush()

    def report(stage: str, progress: float) -> None:
        # every stage change, and every half per cent: a few hundred lines
        if stage != last["stage"] or progress - last["progress"] >= 0.005:
            last["stage"], last["progress"] = stage, float(progress)
            send({"stage": stage, "progress": float(progress)})

    code = EXIT_FAILED
    try:
        path, root = argv[0], argv[1]
        out = compute(path, root, report)
        send({"result": True})
        pickle.dump(out, proto, protocol=pickle.HIGHEST_PROTOCOL)
        proto.flush()
        code = 0
    except Exception as exc:
        try:
            send({"error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc()})
        except Exception:
            pass                    # the parent is gone; nothing to tell
    try:
        proto.close()
    except Exception:
        pass
    for stream in (sys.stdout, sys.stderr):     # both are stderr by now
        try:
            stream.flush()
        except Exception:
            pass
    # No interpreter teardown: the maps are freed by the OS, and a daemon
    # thread still blocked in os.read cannot trip it.
    os._exit(code)
