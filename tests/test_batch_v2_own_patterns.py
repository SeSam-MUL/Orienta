"""A Spherical-GPU run indexes the patterns of ITS OWN signal, never another dataset the viewer holds.

Batch v2 (``batch_manager.run_single_indexing_job``) loads each file into a signal of its own and
never activates it in the viewer. The Spherical-GPU path used to take its in-memory patterns (a source
file h5py cannot open, or the viewer's edited patterns) from the VIEWER's active signal, so a batch job
indexed whatever the viewer happened to hold, under the batch file's name.

The fake backend of ``test_hex_indexing_paths`` records which entry point the run used and the pattern
stack ``index_array`` was handed.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_hex_indexing_paths import (  # noqa: F401  (complete, fake: fixtures)
    SEEN_TOPOLOGY, _FakeBackend, _Stop, complete, fake)

N_ROWS, N_COLS, PH, PW = 2, 2, 8, 8


def _stack_a():
    """The batch file's own patterns: pattern k (row-major) is filled with 0..63 + 64 * k."""
    return np.arange(N_ROWS * N_COLS * PH * PW, dtype=np.uint8).reshape(N_ROWS, N_COLS, PH, PW)


def _stack_b():
    """The viewer's other dataset, same size: every byte is 255 minus the batch file's."""
    return (255 - _stack_a()).astype(np.uint8)


@pytest.fixture
def batch_signal():
    kp = pytest.importorskip("kikuchipy")
    from kikuchipy.detectors import EBSDDetector

    detector = EBSDDetector(
        shape=(PH, PW), pc=(0.5073, 0.2621, 0.5585), sample_tilt=75.7, tilt=10.0, convention="bruker")
    return kp.signals.EBSD(_stack_a(), detector=detector)


def _viewer_holds(monkeypatch, data, dirty):
    """The viewer's active signal is ``data`` (one stable object), edited or not."""
    from backend.api.routes import ebsd_viewer
    viewer_signal = SimpleNamespace(data=data)
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda: viewer_signal)
    monkeypatch.setattr(ebsd_viewer, "is_active_signal_dirty", lambda: dirty)
    monkeypatch.setattr(ebsd_viewer, "get_active_include_mask", lambda: None)
    return viewer_signal


def _non_hdf5(tmp_path, name="batch_file.ebsp"):
    path = tmp_path / name
    path.write_bytes(b"\x01\x00\x00\x00" + bytes(range(60)))                # not HDF5
    return path


def _hdf5(tmp_path):
    import h5py
    path = tmp_path / "batch_file.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("1/EBSD/Data/Patterns", data=np.zeros((N_ROWS * N_COLS, PH, PW), "u1"))
    return path


def _run_batch_job(signal, file_path):
    from backend.api.services.batch_manager import run_single_indexing_job
    return run_single_indexing_job(
        signal=signal, file_path=str(file_path), phase_name="Al", phase_path="Al.sht",
        method="spherical", grid_shape=(N_ROWS, N_COLS),
        job_config={"backend": "spherical_gpu", "spherical_refine": False})


def test_a_batch_job_over_a_non_hdf5_file_indexes_its_own_patterns_not_the_viewers(
        fake, monkeypatch, tmp_path, batch_signal):
    _viewer_holds(monkeypatch, _stack_b(), dirty=False)
    with pytest.raises(_Stop):
        _run_batch_job(batch_signal, _non_hdf5(tmp_path))
    assert fake.instances[0].calls == [("index_array", (N_ROWS * N_COLS, PH, PW))]
    handed = fake.instances[0].arrays[0]
    # Hand count: the batch file's pattern k is bytes 64 * k .. 64 * k + 63 (8 x 8 = 64 bytes each).
    for k in range(N_ROWS * N_COLS):
        np.testing.assert_array_equal(
            handed[k], np.arange(64 * k, 64 * k + 64, dtype=np.uint8).reshape(PH, PW))
    # and not the viewer's: pattern 0 of the viewer's dataset starts at byte 255, counting down
    assert handed[0, 0, 0] == 0 and handed[0, 0, 1] == 1


def test_a_batch_job_over_an_hdf5_file_streams_it_although_the_viewer_edited_another_dataset(
        fake, monkeypatch, tmp_path, batch_signal):
    _viewer_holds(monkeypatch, _stack_b(), dirty=True)
    source = _hdf5(tmp_path)
    with pytest.raises(_Stop):
        _run_batch_job(batch_signal, source)
    assert fake.instances[0].calls == [("index_h5", str(source))]
    assert fake.instances[0].arrays == []                   # index_array was never called


# ------------------------------- the hexagonal note follows the batch file, not the viewer's dataset

def _complete_batch_job(monkeypatch, tmp_path, batch_signal):
    """A whole (faked) batch-v2 Spherical-GPU job over a non-HDF5 file, with the viewer's active
    dataset "ds" registered HEXAGONAL. Returns the job's progress lines and the
    ``orientation_source_reason`` the Spherical run wrote on its result."""
    import indexing_controller
    from backend.api.services import scan_geometry
    from backend.api.services.scan_geometry import ScanGeometry

    _viewer_holds(monkeypatch, _stack_b(), dirty=False)
    scan_geometry.set_geometry("ds", ScanGeometry.hex(N_ROWS, N_COLS, 0.5))
    real = indexing_controller.spherical_gpu_index_patterns
    reasons = []

    def recording(*args, **kwargs):
        result = real(*args, **kwargs)
        reasons.append(result.metadata.get("orientation_source_reason", ""))
        return result

    monkeypatch.setattr(indexing_controller, "spherical_gpu_index_patterns", recording)
    lines = []
    from backend.api.services.batch_manager import run_single_indexing_job
    run_single_indexing_job(
        signal=batch_signal, file_path=str(_non_hdf5(tmp_path, "batch_file.up1")),     # .up1: the vendor is known
        phase_name="Al", phase_path="Al.sht", method="spherical", grid_shape=(N_ROWS, N_COLS),
        job_config={"backend": "spherical_gpu", "spherical_refine": False},
        progress_callback=lambda msg, pct=None: lines.append(msg))
    assert len(reasons) == 1
    return lines, reasons[0]


def test_a_square_batch_file_gets_no_hexagonal_note_although_the_viewer_holds_a_hexagonal_scan(
        complete, monkeypatch, tmp_path, batch_signal):
    # premise: the pseudo-symmetry stage did run (the stand-in unification was asked), so a note COULD
    # have been written; the batch file's own grid is square, and that decides.
    lines, reason = _complete_batch_job(monkeypatch, tmp_path, batch_signal)
    assert complete == [True]
    assert not any("all six neighbours" in m for m in lines)
    assert "all six neighbours" not in reason


def test_the_note_is_the_batch_signals_own_geometry_when_that_one_is_hexagonal(
        complete, monkeypatch, tmp_path, batch_signal):
    # The positive control: batch v2 refuses a hexagonal file at load, but the job function itself
    # follows whatever geometry its signal carries; here the viewer's dataset is square, the signal's
    # is hexagonal, and the note appears.
    from backend.api.services import scan_geometry
    from backend.api.services.scan_geometry import ScanGeometry

    batch_signal.metadata.set_item("Signal.scan_geometry", ScanGeometry.hex(N_ROWS, N_COLS, 0.5).to_dict())
    monkeypatch.setattr(scan_geometry, "geometry_for",
                        lambda _dataset: ScanGeometry.square(N_ROWS, N_COLS, 0.5))   # the viewer's: square
    lines, reason = _complete_batch_job(monkeypatch, tmp_path, batch_signal)
    assert any("all six neighbours" in m for m in lines)
    assert "all six neighbours" in reason
    assert [t.kind for t in SEEN_TOPOLOGY] == ["hex"]           # the stand-in unification got the hex topology


def test_an_unreadable_geometry_of_the_batch_file_is_unknown_never_a_read_of_the_viewers_dataset(
        complete, monkeypatch, tmp_path, batch_signal):
    from backend.api.services import scan_geometry

    def broken(_signal):
        raise RuntimeError("no geometry for this signal")

    monkeypatch.setattr(scan_geometry, "geometry_of_signal", broken)
    lines, reason = _complete_batch_job(monkeypatch, tmp_path, batch_signal)
    assert complete == [True]
    assert not any("all six neighbours" in m for m in lines)
    assert "all six neighbours" not in reason


# ----- the per-phase helper's Spherical-GPU branch: which callers can reach it (they cannot)

@pytest.mark.parametrize("backend, indexer", [("spherical_gpu", "spherical_gpu_index_patterns"),
                                              ("emsphinx", "spherical_index_patterns")])
def test_a_routed_spherical_gpu_run_indexes_its_own_signal(monkeypatch, tmp_path, backend, indexer):
    """Phase-map routing reaches the Spherical branches of ``run_single_phase_method``, one SHT file per
    phase. Driven through the real route and the real ``run_per_phase_indexing``: the indexer gets the
    run's scan stamp, and the GPU one the run's signal, as every other caller hands them. Left out,
    the indexer would take the viewer's live ones, which may belong to another file by then."""
    import asyncio

    import torch
    from kikuchipy.signals import EBSD

    import indexing_controller
    import phase_metadata
    from backend.api.routes import ebsd_viewer
    from backend.api.routes import indexing as route
    from backend.api.services import phase_map_store
    from tests.test_hough_cif_origin import _si_choice_2

    cif = _si_choice_2(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    signal = EBSD(np.zeros((2, 3, 8, 8), dtype=np.float32))
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda: signal)
    state = SimpleNamespace(phase_entries=[SimpleNamespace(cif_filename=cif.name)],
                            phase_grid=np.zeros((2, 3), dtype=int))
    monkeypatch.setattr(phase_map_store, "get_phase_map_store",
                        lambda: SimpleNamespace(get_state=lambda: state))
    monkeypatch.setattr(phase_metadata, "strict_linked_cif", lambda _file, lib: Path(lib) / cif.name)
    monkeypatch.setattr(route, "build_spherical_det_params", lambda *a, **k: {"det": "params"})
    run_stamp = {"scan_geometry": object()}
    monkeypatch.setattr(route, "_scan_stamp_at_run_start", lambda *a, **k: run_stamp)

    seen = []

    def stop(*_a, **k):
        seen.append(k)
        raise RuntimeError("stop after the call")

    monkeypatch.setattr(indexing_controller, indexer, stop)
    req = route.IndexingStartRequest(method="spherical", backend=backend, gausbckg=False,
                                     sht_paths=[str(tmp_path / "si.sht")], use_phase_map_routing=True)
    route._indexing_tasks.pop(asyncio.run(route.start_indexing(req))["task_id"])

    assert len(seen) == 1
    assert seen[0]["scan_geometry"] is run_stamp["scan_geometry"]
    if backend == "spherical_gpu":
        assert seen[0]["signal"] is signal


# --------------------------------------------------------------- /start keeps its behaviour

def _start_run(monkeypatch, tmp_path, source_is_hdf5, dirty):
    """Drive the real ``POST /api/indexing/start`` (spherical, GPU backend, one phase) on dataset "ds".
    The viewer holds ``sig`` (the run's own signal); ``dirty`` says whether it was edited. The fake
    backend ends the run at the first indexing call. Returns the fake backend and ``sig``."""
    import kikuchipy as kp
    from fastapi.testclient import TestClient
    from kikuchipy.detectors import EBSDDetector
    from backend.api.main import app
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.routes import indexing as I
    from backend.api.services import scan_geometry
    from backend.api.services.calibration_store import CalibrationStore
    from tests.test_result_keeps_its_run_pc import RUN_PC, SHAPE

    sig = kp.data.nickel_ebsd_small()
    sig.detector = EBSDDetector(shape=SHAPE, pc=RUN_PC, sample_tilt=70.0, tilt=3.5)
    store = CalibrationStore()
    store.register("ds", sig)
    monkeypatch.setattr(I, "calibration_store", store)
    monkeypatch.setattr(ev, "calibration_store", store)
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    monkeypatch.setattr(ev, "_active_dataset", "ds")
    monkeypatch.setattr(ev, "_get_active_signal", lambda *a, **k: sig)
    monkeypatch.setattr(ev, "is_active_signal_dirty", lambda: dirty)
    monkeypatch.setattr(ev, "get_active_include_mask", lambda: None)
    if source_is_hdf5:
        import h5py
        source = tmp_path / "scan.h5"
        n = int(np.prod(sig.data.shape[:2]))
        with h5py.File(source, "w") as f:
            f.create_dataset("1/EBSD/Data/Patterns", data=np.zeros((n,) + sig.data.shape[2:], "u1"))
    else:
        source = tmp_path / "scan.up1"
        source.write_bytes(b"not an hdf5 file")
    monkeypatch.setattr(ev, "_ebsd_file_path", str(source), raising=False)
    monkeypatch.setattr(I, "_submit_run_guarded", lambda fn, kind="indexing": fn())
    scan_geometry.clear_all()
    saved, saved_active = dict(I._result_registry), I._active_result_id
    try:
        resp = TestClient(app).post("/api/indexing/start", json={
            "method": "spherical", "backend": "spherical_gpu", "sht_paths": ["a.sht"]})
        assert resp.status_code == 200, resp.text
    finally:
        scan_geometry.clear_all()
        I._result_registry.clear()
        I._result_registry.update(saved)
        I._active_result_id = saved_active
    return _FakeBackend.instances[0], sig, source


def test_start_with_a_dirty_viewer_on_an_hdf5_source_still_indexes_the_processed_patterns(
        fake, monkeypatch, tmp_path):
    backend, sig, _source = _start_run(monkeypatch, tmp_path, source_is_hdf5=True, dirty=True)
    n = int(np.prod(sig.data.shape[:2]))
    assert backend.calls == [("index_array", (n,) + tuple(sig.data.shape[2:]))]
    np.testing.assert_array_equal(
        backend.arrays[0], np.asarray(sig.data).reshape((n,) + tuple(sig.data.shape[2:])))


def test_start_with_a_clean_viewer_on_an_hdf5_source_still_streams_the_file(
        fake, monkeypatch, tmp_path):
    backend, _sig, source = _start_run(monkeypatch, tmp_path, source_is_hdf5=True, dirty=False)
    assert backend.calls == [("index_h5", str(source))]


def test_start_on_a_non_hdf5_source_still_indexes_the_viewers_signal(fake, monkeypatch, tmp_path):
    backend, sig, _source = _start_run(monkeypatch, tmp_path, source_is_hdf5=False, dirty=False)
    n = int(np.prod(sig.data.shape[:2]))
    assert backend.calls == [("index_array", (n,) + tuple(sig.data.shape[2:]))]
    np.testing.assert_array_equal(
        backend.arrays[0], np.asarray(sig.data).reshape((n,) + tuple(sig.data.shape[2:])))
