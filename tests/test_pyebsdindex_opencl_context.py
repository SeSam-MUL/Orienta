"""PyEBSDIndex's OpenCL band detector must not build a new context per call.

``BandDetect.find_bands`` (``pyebsdindex/opencl/band_detect_cl.py``) creates a
fresh ``OpenClParam`` -- context, program and all 21 kernel objects -- whenever
it is called without ``clparams``, which is how every ``index_pats`` call
reaches it. The kernel objects of such a parameter set are never given back
(measured with pyopencl 2026.1 on an NVIDIA driver under Windows: about 250 MiB
of committed memory and 23 handles per ``all_kernels()``, unchanged by ``del``,
``gc.collect()`` or releasing every Python reference). After roughly 55 calls a
process had used the machine's whole commit limit and the next call failed with
``Context failed: OUT_OF_HOST_MEMORY``.

``pyebsdindex_mode.share_opencl_context`` therefore gives every such call one
shared parameter set. These tests pin the wrapper's contract with stand-ins
(they need no GPU) and, where an OpenCL GPU exists, that the real detector
builds exactly one context for many indexers and returns bit-identical bands.
"""
import threading
import time

import numpy as np
import pytest

pyebsdindex = pytest.importorskip("pyebsdindex")

import pyebsdindex_mode  # noqa: E402


class _Params:
    """Stands in for ``OpenClParam``."""

    made = 0

    def __init__(self, gpu_id=None):
        type(self).made += 1
        self.gpu_id = gpu_id


def _recorder():
    calls = []

    def original(self, patternsIn, verbose=0, clparams=None, chunksize=528,
                 useCPU=None, gpu_id=None, **kwargs):
        calls.append({"clparams": clparams, "useCPU": useCPU, "gpu_id": gpu_id,
                      "verbose": verbose, "chunksize": chunksize, "kw": kwargs})
        return "bands"

    return original, calls


class _Detector:
    useCPU = False


def _wrapped(factory=_Params):
    original, calls = _recorder()
    return pyebsdindex_mode._share_clparams(original, factory), calls


def test_calls_without_clparams_share_one_parameter_set():
    _Params.made = 0
    find_bands, calls = _wrapped()
    for _ in range(5):
        assert find_bands(_Detector(), np.zeros((1, 4, 4))) == "bands"
    assert _Params.made == 1
    assert len({id(c["clparams"]) for c in calls}) == 1
    assert calls[0]["clparams"] is not None


def test_each_gpu_id_gets_its_own_parameter_set():
    _Params.made = 0
    find_bands, calls = _wrapped()
    find_bands(_Detector(), None, gpu_id=0)
    find_bands(_Detector(), None, gpu_id=1)
    find_bands(_Detector(), None, gpu_id=0)
    assert _Params.made == 2
    assert calls[0]["clparams"] is calls[2]["clparams"]
    assert calls[0]["clparams"] is not calls[1]["clparams"]


def test_an_explicit_parameter_set_is_passed_through_untouched():
    _Params.made = 0
    find_bands, calls = _wrapped()
    mine = object()
    find_bands(_Detector(), None, clparams=mine)
    assert calls[0]["clparams"] is mine
    assert _Params.made == 0


def test_the_cpu_route_never_creates_a_parameter_set():
    _Params.made = 0
    find_bands, calls = _wrapped()
    find_bands(_Detector(), None, useCPU=True)
    cpu = _Detector()
    cpu.useCPU = True
    find_bands(cpu, None)
    assert _Params.made == 0
    assert all(c["clparams"] is None for c in calls)


def test_arguments_reach_the_original_unchanged():
    find_bands, calls = _wrapped()
    find_bands(_Detector(), None, 2, None, 64, False, 3, extra=7)
    c = calls[0]
    assert (c["verbose"], c["chunksize"], c["useCPU"], c["gpu_id"], c["kw"]) == (
        2, 64, False, 3, {"extra": 7})


def test_shared_calls_are_serialised():
    """One shared kernel set must never be driven by two threads at once."""
    active = []
    overlap = []

    def original(self, patternsIn, verbose=0, clparams=None, chunksize=528,
                 useCPU=None, gpu_id=None, **kwargs):
        active.append(1)
        if len(active) > 1:
            overlap.append(True)
        time.sleep(0.02)
        active.pop()
        return "bands"

    find_bands = pyebsdindex_mode._share_clparams(original, _Params)
    threads = [threading.Thread(target=find_bands, args=(_Detector(), None))
               for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not overlap


def test_install_is_idempotent_and_reports_whether_it_applied(monkeypatch):
    cl_module = pytest.importorskip("pyebsdindex.opencl.band_detect_cl")
    cls = cl_module.BandDetect
    monkeypatch.setattr(cls, "find_bands", cls.find_bands)  # restored afterwards
    monkeypatch.setattr(pyebsdindex, "_pyopencl_installed", True)
    assert pyebsdindex_mode.share_opencl_context() is True
    first = cls.find_bands
    assert pyebsdindex_mode.share_opencl_context() is True
    assert cls.find_bands is first                    # not wrapped twice


def test_without_an_opencl_gpu_nothing_is_changed(monkeypatch):
    cl_module = pytest.importorskip("pyebsdindex.opencl.band_detect_cl")
    cls = cl_module.BandDetect
    before = cls.find_bands
    monkeypatch.setattr(pyebsdindex, "_pyopencl_installed", False)
    assert pyebsdindex_mode.share_opencl_context() is False
    assert cls.find_bands is before


# ---- the real detector, only where an OpenCL GPU exists ------------------

def _real_opencl():
    if not getattr(pyebsdindex, "_pyopencl_installed", False):
        pytest.skip("PyEBSDIndex found no OpenCL GPU on this machine")
    return pytest.importorskip("pyebsdindex.opencl.openclparam")


def _nickel_indexer(cif_path):
    import kikuchipy as kp
    from orix.crystal_map import Phase, PhaseList

    from ebsd_utils import create_indexer, prepare_reflectors, sanitize_cif

    s = kp.data.nickel_ebsd_small()           # real Ni patterns shipped with kikuchipy
    det = s.detector
    phase = Phase.from_cif(sanitize_cif(str(cif_path)))
    phase.name = "Ni"
    pl = PhaseList(phase)
    return s.data.reshape(-1, *s.data.shape[-2:]), create_indexer(
        det, pl, prepare_reflectors(pl), nBands=9)


def test_many_indexers_build_one_context_and_bands_stay_bit_identical(
        monkeypatch, ni_cif_path):
    openclparam = _real_opencl()

    pats, first = _nickel_indexer(ni_cif_path)          # create_indexer installs the sharing
    # Reference: what PyEBSDIndex does by itself -- a private parameter set per call.
    private = openclparam.OpenClParam()
    private.get_queue()
    reference = first.bandDetectPlan.find_bands.__wrapped__(
        first.bandDetectPlan, pats, clparams=private)

    contexts = []
    real_get_context = openclparam.OpenClParam.get_context

    def counting(self, *a, **k):
        contexts.append(1)
        return real_get_context(self, *a, **k)

    monkeypatch.setattr(openclparam.OpenClParam, "get_context", counting)
    for _ in range(6):
        _, ix = _nickel_indexer(ni_cif_path)
        out = ix.bandDetectPlan.find_bands(pats)
        np.testing.assert_array_equal(out, reference)
    assert len(contexts) <= 1, f"{len(contexts)} OpenCL contexts for 6 indexers"
