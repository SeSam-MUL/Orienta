"""Keep PyEBSDIndex's band detection on the CPU (numba) path on macOS.

PyEBSDIndex picks its band detector once, at import: if ``pyopencl`` imports
and OpenCL reports a GPU, it uses OpenCL, otherwise the CPU/numba path. There
is no switch in PyEBSDIndex. On macOS that means Apple's OpenCL, deprecated
since macOS 10.14 (PyEBSDIndex's changelog records a workaround for
Apple-Silicon OpenCL runs in 0.3.7). It cannot be avoided by installation:
conda-forge's ``pyebsdindex-base`` depends on ``pyopencl`` on every platform.

The Windows installer ships no pyopencl, so Windows users run the CPU path.
Forcing the same path on macOS keeps what is tested and what is shipped the
same implementation.

Ray needs nothing extra: ``index_pats_distributed`` asks the (rebound) CPU
detector for OpenCL parameters, gets none, and its workers use the pickled
indexer whose plan is the CPU ``BandDetect``.
"""
from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

#: Modules that bind their band detector from ``_pyopencl_installed`` at import.
_BINDING_MODULES = ("pyebsdindex._ebsd_index_single", "pyebsdindex._ebsd_index_parallel")


def force_cpu_band_detection(platform: str | None = None) -> bool:
    """On macOS, make PyEBSDIndex use its CPU band detector. Idempotent.

    Returns True when the CPU path was forced, False on other platforms or
    when PyEBSDIndex is not installed. Call it before the first EBSDIndexer
    is built: an indexer that already exists keeps the plan it was built with.
    """
    platform = sys.platform if platform is None else platform
    if platform != "darwin":
        return False
    try:
        import pyebsdindex
        from pyebsdindex import band_detect as cpu_band_detect
    except ImportError:
        return False

    had_gpu = bool(pyebsdindex._pyopencl_installed)
    pyebsdindex._pyopencl_installed = False
    rebound = []
    # A module imported before this call has already bound the OpenCL
    # detector; rebinding the module global is enough, EBSDIndexer looks it
    # up when it is constructed.
    for name in _BINDING_MODULES:
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "band_detect", None) is not cpu_band_detect:
            mod.band_detect = cpu_band_detect
            rebound.append(name)
    if had_gpu or rebound:
        logger.info(
            "PyEBSDIndex band detection: CPU (forced on macOS; OpenCL GPU %s)",
            "found and not used" if had_gpu else "not found")
    return True


# --------------------------------------------------------------------------
# One OpenCL context for all band detection in the process
# --------------------------------------------------------------------------
#
# ``pyebsdindex.opencl.band_detect_cl.BandDetect.find_bands`` builds a new
# ``OpenClParam`` (context, program and every kernel object) whenever it is
# called without ``clparams``, and ``index_pats`` always calls it that way.
# The kernel objects of those parameter sets are not given back: measured with
# pyopencl 2026.1 on an NVIDIA driver under Windows, ``Program.all_kernels()``
# alone commits about 250 MiB and 23 handles per call and keeps them after
# ``del``, ``gc.collect()`` and dropping every Python reference, whereas a
# context, a queue or a program build alone returns its memory. After about 55
# ``index_pats`` calls in one process the machine's commit limit was used up
# and the next OpenCL call raised ``Context failed: OUT_OF_HOST_MEMORY``; Index
# All, PC refinement and the reflector probes all call ``index_pats``.
#
# The fix keeps PyEBSDIndex's own code and gives every call that would have
# made a private parameter set one shared set per GPU. The kernels are the
# same objects built from the same source, so the bands are bit-identical.
# A shared kernel object holds its arguments between ``set_arg`` and the
# enqueue, so shared calls are serialised with a lock (one GPU runs them one
# at a time anyway). Callers that pass their own ``clparams`` - the Ray actors
# of ``index_pats_distributed`` - are untouched.

_SHARED_MARK = "_orienta_shared_clparams"


def _share_clparams(original, make_params):
    """Wrap a ``find_bands`` so that ``clparams=None`` means a shared set.

    ``make_params(gpu_id)`` builds a parameter set with its queue; it is called
    once per ``gpu_id`` and the result is reused by every later call.
    """
    import functools
    import threading

    lock = threading.RLock()
    shared: dict = {}

    @functools.wraps(original)
    def find_bands(self, patternsIn, verbose=0, clparams=None, chunksize=528,
                   useCPU=None, gpu_id=None, **kwargs):
        cpu = self.useCPU if useCPU is None else useCPU
        if clparams is not None or cpu:
            return original(self, patternsIn, verbose=verbose, clparams=clparams,
                            chunksize=chunksize, useCPU=useCPU, gpu_id=gpu_id,
                            **kwargs)
        with lock:
            if gpu_id not in shared:
                shared[gpu_id] = make_params(gpu_id)
            return original(self, patternsIn, verbose=verbose,
                            clparams=shared[gpu_id], chunksize=chunksize,
                            useCPU=useCPU, gpu_id=gpu_id, **kwargs)

    setattr(find_bands, _SHARED_MARK, True)
    return find_bands


def share_opencl_context() -> bool:
    """Make PyEBSDIndex's OpenCL band detector reuse one context. Idempotent.

    Returns True when the OpenCL detector is in use and shares its context
    (now or already), False when there is nothing to do: no OpenCL GPU, the
    CPU detector forced on macOS, or PyEBSDIndex not installed.
    """
    try:
        import pyebsdindex
    except ImportError:
        return False
    if not pyebsdindex._pyopencl_installed:
        return False
    try:
        from pyebsdindex.opencl import band_detect_cl, openclparam
    except ImportError:
        return False

    cls = band_detect_cl.BandDetect
    if getattr(cls.find_bands, _SHARED_MARK, False):
        return True

    def make_params(gpu_id):
        params = openclparam.OpenClParam()
        params.get_queue(gpu_id=gpu_id)
        return params

    cls.find_bands = _share_clparams(cls.find_bands, make_params)
    logger.info("PyEBSDIndex OpenCL band detection: one shared context per GPU")
    return True
