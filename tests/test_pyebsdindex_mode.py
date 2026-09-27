"""macOS runs PyEBSDIndex's CPU band detector, like the Windows installer does.

PyEBSDIndex decides OpenCL-or-CPU once at import, from whether pyopencl finds a
GPU; conda-forge always installs pyopencl, so on an Apple-Silicon Mac it would
run Apple's deprecated OpenCL while the Windows installer ships the CPU path.
"""
import logging

import pytest

pyebsdindex = pytest.importorskip("pyebsdindex")

from pyebsdindex import band_detect as cpu_band_detect  # noqa: E402

import pyebsdindex_mode  # noqa: E402


class _FakeOpenCLDetector:
    """Stands in for pyebsdindex.opencl.band_detect_cl, which needs a GPU."""

    class BandDetect(cpu_band_detect.BandDetect):
        pass


def test_other_platforms_are_left_alone(monkeypatch):
    import pyebsdindex._ebsd_index_single as single

    monkeypatch.setattr(pyebsdindex, "_pyopencl_installed", True)
    monkeypatch.setattr(single, "band_detect", _FakeOpenCLDetector)
    for platform in ("win32", "linux"):
        assert pyebsdindex_mode.force_cpu_band_detection(platform) is False
    assert pyebsdindex._pyopencl_installed is True
    assert single.band_detect is _FakeOpenCLDetector


def test_macos_forces_the_flag_and_rebinds_already_imported_modules(monkeypatch):
    pytest.importorskip("ray")  # _ebsd_index_parallel imports ray at top level
    import pyebsdindex._ebsd_index_single as single
    import pyebsdindex._ebsd_index_parallel as parallel

    monkeypatch.setattr(pyebsdindex, "_pyopencl_installed", True)
    monkeypatch.setattr(single, "band_detect", _FakeOpenCLDetector)
    monkeypatch.setattr(parallel, "band_detect", _FakeOpenCLDetector)

    assert pyebsdindex_mode.force_cpu_band_detection("darwin") is True
    assert pyebsdindex._pyopencl_installed is False
    assert single.band_detect is cpu_band_detect
    assert parallel.band_detect is cpu_band_detect


def test_an_indexer_built_afterwards_uses_the_cpu_detector(monkeypatch):
    import pyebsdindex._ebsd_index_single as single

    monkeypatch.setattr(pyebsdindex, "_pyopencl_installed", True)
    monkeypatch.setattr(single, "band_detect", _FakeOpenCLDetector)
    pyebsdindex_mode.force_cpu_band_detection("darwin")

    indexer = single.EBSDIndexer(patDim=(60, 60))
    assert type(indexer.bandDetectPlan) is cpu_band_detect.BandDetect


def test_forcing_twice_from_the_opencl_state_gives_the_same_result(monkeypatch):
    import pyebsdindex._ebsd_index_single as single

    monkeypatch.setattr(pyebsdindex, "_pyopencl_installed", True)
    monkeypatch.setattr(single, "band_detect", _FakeOpenCLDetector)
    assert pyebsdindex_mode.force_cpu_band_detection("darwin") is True
    assert pyebsdindex_mode.force_cpu_band_detection("darwin") is True
    assert pyebsdindex._pyopencl_installed is False
    assert single.band_detect is cpu_band_detect


def test_macos_with_a_gpu_says_so_in_the_log(monkeypatch, caplog):
    """A Mac bug report must show which band detector ran."""
    import pyebsdindex._ebsd_index_single as single

    monkeypatch.setattr(pyebsdindex, "_pyopencl_installed", True)
    monkeypatch.setattr(single, "band_detect", _FakeOpenCLDetector)
    with caplog.at_level(logging.INFO, logger="pyebsdindex_mode"):
        pyebsdindex_mode.force_cpu_band_detection("darwin")
    assert any("CPU (forced on macOS; OpenCL GPU found and not used)"
               in r.getMessage() for r in caplog.records)
