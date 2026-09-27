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
