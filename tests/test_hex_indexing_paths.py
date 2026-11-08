"""What indexing does with a pattern file h5py cannot open, and with a hexagonal scan.

1. The Spherical full-map path streams its patterns from the source file with h5py
   (``backend.index_h5``). A ``.up1``/``.up2`` (or any non-HDF5 source) is not HDF5: the stream died
   with a bare h5py error, and a hexagonal reshape (``Signal.data`` of the loader) could never reach
   it. For such a source the active signal IS the pattern store, so it is indexed like the
   "processed" patterns (``index_array``), under the same in-memory ceiling.
2. The one automatic neighbourhood pass of an indexing run, the EDS particle rescue, compares each
   pixel with its four neighbours on the index grid; on a hexagonal scan it is skipped, visibly.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import indexing_controller as ic  # noqa: E402
from backend.api.services import scan_geometry  # noqa: E402
from backend.api.services.scan_geometry import ScanGeometry  # noqa: E402
from backend.spherical_gpu.pipeline import variant_unification as _vu_module  # noqa: E402

_REAL_UNIFY = _vu_module.unify_after_hough_resolve        # before the fixtures replace it

N_ROWS, N_COLS, PH, PW = 3, 4, 5, 6


class _Stop(Exception):
    """Raised by the fake backend once it has been asked to index: nothing after that is under test."""


class _FakeBackend:
    """Stands in for SphericalGPUBackend. Records which of the two entry points was used."""
    instances = []

    def __init__(self, cfg):
        self.calls = []
        self.arrays = []                    # the pattern stacks index_array was handed, in call order
        self.kwargs = []                    # and the keyword arguments of each of those calls
        self.runtime = SimpleNamespace(device="cpu", device_name="fake", free_memory_gb=99.0,
                                       cuda_available=False)
        _FakeBackend.instances.append(self)

    def set_cancel_check(self, fn):
        pass

    def invalidate(self):
        pass

    def index_array(self, patterns=None, **kwargs):
        self.calls.append(("index_array", np.asarray(patterns).shape))
        self.arrays.append(np.array(patterns))
        self.kwargs.append(dict(kwargs))
        raise _Stop

    def index_h5(self, h5_path=None, **kwargs):
        self.calls.append(("index_h5", str(h5_path)))
        raise _Stop


@pytest.fixture
def fake(monkeypatch):
    import backend.spherical_gpu.backend as sgb
    _FakeBackend.instances = []
    monkeypatch.setattr(sgb, "SphericalGPUBackend", _FakeBackend)
    monkeypatch.setattr(sgb, "BackendConfig", lambda phases: SimpleNamespace(phases=phases))
    monkeypatch.setattr(sgb, "PhaseConfig", lambda **kw: SimpleNamespace(**kw))
    return _FakeBackend


def _active_signal(monkeypatch, data, dirty=False, ceiling=None):
    from backend.api.routes import ebsd_viewer
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda: SimpleNamespace(data=data))
    monkeypatch.setattr(ebsd_viewer, "is_active_signal_dirty", lambda: dirty)
    monkeypatch.setattr(ebsd_viewer, "get_active_include_mask", lambda: None)
    if ceiling is not None:
        monkeypatch.setattr(ebsd_viewer, "_CROP_MATERIALIZE_MAX_BYTES", ceiling)


def _run(source, mask=None):
    cfg = SimpleNamespace(sht_file="master.sht", bandwidth=88, normed=True, refine=False,
                          circmask=0, gausbckg=False, nregions=10)
    det = {"n_rows": N_ROWS, "n_cols": N_COLS}
    return ic.spherical_gpu_index_patterns(
        str(source), cfg, det, selection_mask=mask, sht_paths=["master.sht"])


def _up_file(tmp_path):
    path = tmp_path / "scan.up1"
    path.write_bytes(b"\x01\x00\x00\x00" + bytes(range(60)))                 # not HDF5
    return path


def _hdf5_file(tmp_path):
    import h5py
    path = tmp_path / "scan.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("1/EBSD/Data/Patterns", data=np.zeros((N_ROWS * N_COLS, PH, PW), "u1"))
    return path


def _stack():
    return np.arange(N_ROWS * N_COLS * PH * PW, dtype=np.uint8).reshape(N_ROWS, N_COLS, PH, PW)


def test_a_full_map_of_a_non_hdf5_source_indexes_the_active_signal_and_never_streams(fake, monkeypatch, tmp_path):
    _active_signal(monkeypatch, _stack())                       # not dirty: the viewer did not touch it
    with pytest.raises(_Stop):
        _run(_up_file(tmp_path))
    assert fake.instances[0].calls == [("index_array", (N_ROWS * N_COLS, PH, PW))]


def test_a_roi_of_a_non_hdf5_source_indexes_only_the_selected_patterns(fake, monkeypatch, tmp_path):
    _active_signal(monkeypatch, _stack())
    mask = np.zeros((N_ROWS, N_COLS), dtype=bool)
    mask[1, 1:3] = True
    with pytest.raises(_Stop):
        _run(_up_file(tmp_path), mask)
    assert fake.instances[0].calls == [("index_array", (2, PH, PW))]
    # WHICH two: navigation positions (1, 1) and (1, 2) are flat patterns 1 * 4 + 1 = 5 and 6 of the
    # row-major stack. `_stack` counts bytes upwards, 30 bytes (5 x 6) per pattern, so pattern 5 is
    # bytes 150..179 and pattern 6 is bytes 180..209 (hand-counted, not read off the stack).
    first, second = fake.instances[0].arrays[0]
    np.testing.assert_array_equal(first, np.arange(150, 180, dtype=np.uint8).reshape(PH, PW))
    np.testing.assert_array_equal(second, np.arange(180, 210, dtype=np.uint8).reshape(PH, PW))


def test_an_hdf5_source_that_is_not_dirty_still_streams_from_the_file(fake, monkeypatch, tmp_path):
    """The path that existed before is untouched."""
    _active_signal(monkeypatch, _stack())
    source = _hdf5_file(tmp_path)
    with pytest.raises(_Stop):
        _run(source)
    assert fake.instances[0].calls == [("index_h5", str(source))]


def test_a_non_hdf5_source_without_patterns_in_memory_says_so_instead_of_crashing_in_h5py(fake, monkeypatch, tmp_path):
    _active_signal(monkeypatch, None)
    with pytest.raises(ValueError, match="not an HDF5 file"):
        _run(_up_file(tmp_path))
    assert fake.instances[0].calls == []


@pytest.mark.parametrize("data, mask_shape, fragment", [
    (None, None, "holds no patterns of it in memory"),
    # a 3-D stack of 12 patterns: not (rows, columns, height, width)
    (np.zeros((12, PH, PW), np.uint8), None, "pattern stack has shape (12, 5, 6)"),
    # a 2 x 5 = 10-pattern stack against a 3 x 4 = 12-position selection (hand-counted)
    (np.zeros((2, 5, PH, PW), np.uint8), (N_ROWS, N_COLS),
     "selection has 12 positions but the viewer holds 10 patterns"),
], ids=["nothing in memory", "not a 4-D stack", "selection does not fit the stack"])
def test_the_refusal_for_a_non_hdf5_source_names_the_real_reason(
        fake, monkeypatch, tmp_path, data, mask_shape, fragment):
    _active_signal(monkeypatch, data)
    mask = None
    if mask_shape is not None:
        mask = np.zeros(mask_shape, dtype=bool)
        mask[1, 1:3] = True                                     # a ROI, so the selection is compared
    with pytest.raises(ValueError, match="not an HDF5 file") as err:
        _run(_up_file(tmp_path), mask)
    assert fragment in str(err.value)
    others = {"holds no patterns of it in memory", "pattern stack has shape", "positions but the viewer holds"}
    assert not any(o in str(err.value) for o in others if o not in fragment)
    assert fake.instances[0].calls == []


def test_a_dirty_viewer_with_an_unusable_stack_still_reads_the_hdf5_file(fake, monkeypatch, tmp_path):
    """The reasons are only asked for under `force`; the dirty-viewer path falls back to the file."""
    _active_signal(monkeypatch, np.zeros((12, PH, PW), np.uint8), dirty=True)
    source = _hdf5_file(tmp_path)
    with pytest.raises(_Stop):
        _run(source)
    assert fake.instances[0].calls == [("index_h5", str(source))]


def test_a_lazy_stack_above_the_in_memory_ceiling_is_refused_not_allocated(fake, monkeypatch, tmp_path):
    import dask.array as da
    lazy = da.zeros((N_ROWS, N_COLS, PH, PW), dtype="u1", chunks=(1, N_COLS, PH, PW))
    _active_signal(monkeypatch, lazy, ceiling=100)              # 360 bytes > 100
    with pytest.raises(ValueError, match="still lazy"):
        _run(_up_file(tmp_path))
    assert fake.instances[0].calls == []


def test_the_viewers_own_edits_still_win_for_an_hdf5_source(fake, monkeypatch, tmp_path):
    _active_signal(monkeypatch, _stack(), dirty=True)
    with pytest.raises(_Stop):
        _run(_hdf5_file(tmp_path))
    assert fake.instances[0].calls == [("index_array", (N_ROWS * N_COLS, PH, PW))]


def test_a_failing_hex_check_answers_not_hexagonal_and_leaves_a_warning(monkeypatch, caplog):
    import logging

    def broken(_dataset):
        raise RuntimeError("the geometry registry is gone")

    monkeypatch.setattr(scan_geometry, "geometry_for", broken)
    with caplog.at_level(logging.WARNING):
        assert ic._active_scan_is_hex() is False                # unknown is not hexagonal
    warned = [r for r in caplog.records if r.levelno == logging.WARNING and r.exc_info
              and "the geometry registry is gone" in str(r.exc_info[1])]
    assert len(warned) == 1


# ---------------------------------------------------------------- the particle rescue on hex

@pytest.fixture
def armed(monkeypatch):
    """A fake indexing run (Al matrix, Si particle) the rescue would change, and a way to say which
    scan geometry the active dataset has."""
    from tests.test_eds_particle_rescue import _fake_run
    from backend.api.routes import ebsd_viewer
    si = np.full((20, 20), 1.6)
    si[8:11, 8:12] = 30.0
    route, result, req, blob = _fake_run(monkeypatch, si, {"Al.sht": 0.75, "Si.sht": 0.75})
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "ds")
    scan_geometry.clear_all()

    def arm(geometry):
        if geometry is not None:
            scan_geometry.set_geometry("ds", geometry)
        return route, result, req, blob

    yield arm
    scan_geometry.clear_all()


def test_the_particle_rescue_is_skipped_on_a_hexagonal_scan_and_says_so(armed):
    route, result, req, blob = armed(ScanGeometry.hex(20, 20, 0.5))
    before = np.asarray(result.xmap.phase_id).copy()
    lines = []
    report = route._apply_particle_rescue(result, req, result.selection_mask, lines.append)
    assert report == {"skipped": "hex_grid"}
    assert result.metadata["eds_particle_rescue"] == {"skipped": "hex_grid"}
    assert (np.asarray(result.xmap.phase_id) == before).all()
    assert len(lines) == 1 and "hexagonal" in lines[0] and "skipped" in lines[0]


@pytest.mark.parametrize("geometry", [ScanGeometry.square(20, 20, 0.5), None])
def test_the_particle_rescue_runs_as_before_on_a_square_or_unknown_scan(armed, geometry):
    route, result, req, blob = armed(geometry)
    report = route._apply_particle_rescue(result, req, result.selection_mask)
    assert report is not None and report["n_to_particle"] == blob.sum()
    assert "skipped" not in result.metadata["eds_particle_rescue"]


def test_with_the_prior_off_nothing_is_recorded_on_a_hexagonal_scan(monkeypatch):
    """The skip is a statement about a pass that would have run; with the prior off none would."""
    from tests.test_eds_particle_rescue import _fake_run
    from backend.api.routes import ebsd_viewer
    si = np.full((20, 20), 1.6)
    route, result, req, blob = _fake_run(monkeypatch, si, {"Al.sht": 0.0, "Si.sht": 0.0})
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "ds")
    scan_geometry.set_geometry("ds", ScanGeometry.hex(20, 20, 0.5))
    try:
        assert route._apply_particle_rescue(result, req, result.selection_mask) is None
        assert "eds_particle_rescue" not in result.metadata
    finally:
        scan_geometry.clear_all()


@pytest.mark.parametrize("geometry", [ScanGeometry.square(20, 20, 0.5), ScanGeometry.hex(20, 20, 0.5)],
                         ids=["square", "hexagonal"])
def test_with_no_degenerate_pair_nothing_is_recorded_on_either_grid(monkeypatch, geometry):
    """The skip is a statement about a pass that would have run. Two phases of different point groups
    (cubic m-3m and -43m) form no degenerate pair, so the pass would have done nothing: the hexagonal
    scan records and says exactly what the square one does, which is nothing."""
    from tests.test_eds_particle_rescue import _fake_run
    from backend.api.routes import ebsd_viewer
    si = np.full((20, 20), 1.6)
    si[8:11, 8:12] = 30.0
    route, result, req, blob = _fake_run(
        monkeypatch, si, {"Al.sht": 0.75, "Si.sht": 0.75}, pg_si="-43m")
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "ds")
    scan_geometry.set_geometry("ds", geometry)
    lines = []
    try:
        assert route._apply_particle_rescue(result, req, result.selection_mask, lines.append) is None
        assert "eds_particle_rescue" not in result.metadata
        assert lines == []
    finally:
        scan_geometry.clear_all()


# ------------------------------------- the variant unification, inside a whole (faked) Spherical run

class _CompleteBackend(_FakeBackend):
    """A backend whose run finishes: one -43m phase (z_rot 2) that wins every pixel."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.config = cfg
        self._indexers = [SimpleNamespace(master=SimpleNamespace(
            formula="PhaseX", point_group="-43m", z_rot=2))]

    def index_array(self, patterns=None, **kwargs):
        import torch
        n = int(np.asarray(patterns).shape[0])
        self.calls.append(("index_array", np.asarray(patterns).shape))
        return SimpleNamespace(
            euler_xyz=torch.zeros((n, 3), dtype=torch.float64),
            phase_id=torch.ones(n, dtype=torch.int64),
            score=torch.full((n,), 0.5, dtype=torch.float64),
            runtime_seconds=1.0)


#: The ``topology`` the stand-in unification was asked with, one entry per call (cleared by ``complete``).
SEEN_TOPOLOGY = []


@pytest.fixture
def complete(monkeypatch):
    """The Spherical run with the GPU backend and the two pseudo-symmetry stages replaced: the resolver
    claims it substituted Hough orientations for phase 1, and the unification records that it was asked
    and hands the orientations back as if it had unified them (so the run takes its "unified" branch)."""
    import backend.spherical_gpu.backend as sgb
    import backend.spherical_gpu.pipeline.resolution as resolution
    import backend.spherical_gpu.pipeline.variant_unification as vu
    asked = []
    SEEN_TOPOLOGY.clear()
    monkeypatch.setattr(sgb, "SphericalGPUBackend", _CompleteBackend)
    monkeypatch.setattr(sgb, "BackendConfig", lambda phases: SimpleNamespace(phases=phases))
    monkeypatch.setattr(sgb, "PhaseConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(
        resolution, "resolve_eulers_multiphase",
        lambda patterns, raw, phase_id, meta, cifs, det, progress=None: (
            np.asarray(raw), {"resolved_phase_ids": {1}, "labels": [("PhaseX", "-43m")], "n_fallback": 0}))
    monkeypatch.setattr(
        vu, "unify_after_hough_resolve",
        lambda eulers, *a, **k: (asked.append(True), SEEN_TOPOLOGY.append(k.get("topology")),
                                 (np.asarray(eulers), {}))[2])
    scan_geometry.clear_all()
    from backend.api.routes import ebsd_viewer
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "ds")
    yield asked
    scan_geometry.clear_all()


def _run_complete(tmp_path, monkeypatch, geometry, **extra):
    """``geometry`` is the LIVE active dataset's; ``extra`` goes to the function (a run's own
    ``scan_geometry`` stamp, when the test gives one)."""
    _active_signal(monkeypatch, _stack())
    if geometry is not None:
        scan_geometry.set_geometry("ds", geometry)
    lines = []
    cfg = SimpleNamespace(sht_file="master.sht", bandwidth=88, normed=True, refine=False,
                          circmask=0, gausbckg=False, nregions=10)
    result = ic.spherical_gpu_index_patterns(
        str(_up_file(tmp_path)), cfg, {"n_rows": N_ROWS, "n_cols": N_COLS},
        sht_paths=["master.sht"], progress_callback=lambda msg, pct=None: lines.append(msg),
        **extra)
    return result, lines


def test_the_variant_unification_runs_on_a_hexagonal_scan_and_joins_grains_through_all_six_neighbours(
        complete, monkeypatch, tmp_path):
    result, lines = _run_complete(tmp_path, monkeypatch, ScanGeometry.hex(N_ROWS, N_COLS, 0.5))
    assert complete == [True]                                   # the pass was asked
    assert any("all six neighbours" in m for m in lines)
    assert "all six neighbours" in result.metadata["orientation_source_reason"]
    assert len(SEEN_TOPOLOGY) == 1                              # ... with the hexagonal topology of the scan
    assert (SEEN_TOPOLOGY[0].kind, SEEN_TOPOLOGY[0].n_rows, SEEN_TOPOLOGY[0].n_cols,
            SEEN_TOPOLOGY[0].odd_row_shift) == ("hex", N_ROWS, N_COLS, 0.5)


@pytest.mark.parametrize("geometry", [ScanGeometry.square(N_ROWS, N_COLS, 0.5), None])
def test_the_variant_unification_runs_as_before_on_a_square_or_unknown_scan(
        complete, monkeypatch, tmp_path, geometry):
    result, lines = _run_complete(tmp_path, monkeypatch, geometry)
    assert complete == [True]
    assert not any("all six neighbours" in m for m in lines)
    assert "all six neighbours" not in result.metadata.get("orientation_source_reason", "")
    assert SEEN_TOPOLOGY == [None]


# ------------------------- a run decides "hexagonal?" from ITS OWN stamp, not from the live dataset

HEX_STAMP = ScanGeometry.hex(20, 20, 0.5).to_dict()          # what `_scan_stamp_at_run_start` stores
SQUARE_STAMP = ScanGeometry.square(20, 20, 0.5).to_dict()


def test_the_rescue_is_skipped_on_the_runs_hexagonal_stamp_although_the_live_dataset_is_square(armed):
    # A crop or a file switch during the run made a square dataset active; the run was hexagonal.
    route, result, req, blob = armed(ScanGeometry.square(20, 20, 0.5))
    before = np.asarray(result.xmap.phase_id).copy()
    report = route._apply_particle_rescue(
        result, req, result.selection_mask, None, scan_geometry=HEX_STAMP)
    assert report == {"skipped": "hex_grid"}
    assert (np.asarray(result.xmap.phase_id) == before).all()


@pytest.mark.parametrize("stamp", [SQUARE_STAMP, None], ids=["square stamp", "unknown stamp"])
def test_the_rescue_runs_on_a_square_or_unknown_stamp_although_the_live_dataset_is_hexagonal(armed, stamp):
    # None is "unknown", which is not hexagonal: it must never fall back to reading the live dataset.
    route, result, req, blob = armed(ScanGeometry.hex(20, 20, 0.5))
    report = route._apply_particle_rescue(
        result, req, result.selection_mask, None, scan_geometry=stamp)
    assert report is not None and report["n_to_particle"] == blob.sum()
    assert "skipped" not in result.metadata["eds_particle_rescue"]


def test_the_spherical_note_follows_the_runs_hexagonal_stamp_although_the_live_dataset_is_square(
        complete, monkeypatch, tmp_path):
    result, lines = _run_complete(
        tmp_path, monkeypatch, ScanGeometry.square(N_ROWS, N_COLS, 0.5), scan_geometry=HEX_STAMP)
    assert complete == [True]                                   # the pass runs on hex, never skipped
    assert any("all six neighbours" in m for m in lines)
    assert "all six neighbours" in result.metadata["orientation_source_reason"]
    assert SEEN_TOPOLOGY[0].kind == "hex"


@pytest.mark.parametrize("stamp", [SQUARE_STAMP, None], ids=["square stamp", "unknown stamp"])
def test_the_spherical_note_is_absent_on_a_square_or_unknown_stamp_although_the_live_dataset_is_hexagonal(
        complete, monkeypatch, tmp_path, stamp):
    result, lines = _run_complete(
        tmp_path, monkeypatch, ScanGeometry.hex(N_ROWS, N_COLS, 0.5), scan_geometry=stamp)
    assert complete == [True]
    assert not any("all six neighbours" in m for m in lines)
    assert "all six neighbours" not in result.metadata.get("orientation_source_reason", "")
    assert SEEN_TOPOLOGY == [None]


@pytest.mark.parametrize("stamp, why", [
    (ScanGeometry.hex(N_ROWS + 1, N_COLS, 0.5).to_dict(), "4 x 4"),                # a stamp of another grid
    ({"kind": "hex", "n_cols": N_COLS, "odd_row_shift": 0.5}, "cannot be read"),     # a stamp without n_rows
], ids=["raster does not fit", "stamp without n_rows"])
def test_a_skipped_unification_keeps_the_run_and_says_so_in_the_result_not_a_claim_of_six_neighbours(
        complete, monkeypatch, tmp_path, stamp, why):
    # After the GPU work: the run must end with its result. The step that could not run is named in the result
    # (the user reads the reason there, not in the log), and nothing says the grains were joined through six.
    import backend.spherical_gpu.pipeline.variant_unification as vu
    monkeypatch.setattr(vu, "unify_after_hough_resolve", _REAL_UNIFY)          # the real raster check
    result, lines = _run_complete(tmp_path, monkeypatch, ScanGeometry.hex(N_ROWS, N_COLS, 0.5),
                                  scan_geometry=stamp)
    assert result is not None and result.xmap is not None
    reason = result.metadata["orientation_source_reason"]
    assert "variant unification was skipped" in reason and why in reason
    assert not any("six neighbours" in m for m in lines)
    assert "six neighbours" not in reason


def _run_complete_with_a_failing_progress_line(tmp_path, monkeypatch, fails_on, **extra):
    """The whole faked run, but the progress callback raises on the lines containing ``fails_on``."""
    _active_signal(monkeypatch, _stack())
    scan_geometry.set_geometry("ds", ScanGeometry.hex(N_ROWS, N_COLS, 0.5))
    cfg = SimpleNamespace(sht_file="master.sht", bandwidth=88, normed=True, refine=False,
                          circmask=0, gausbckg=False, nregions=10)

    def callback(msg, pct=None):
        if fails_on in msg:
            raise RuntimeError("the progress sink is gone")

    return ic.spherical_gpu_index_patterns(
        str(_up_file(tmp_path)), cfg, {"n_rows": N_ROWS, "n_cols": N_COLS},
        sht_paths=["master.sht"], progress_callback=callback, **extra)


def test_an_error_after_the_orientations_were_unified_does_not_say_the_step_was_skipped(
        complete, monkeypatch, tmp_path):
    # The first line after the replacement raises: the map IS unified, so the note must not claim the
    # orientations are those of the Hough resolver.
    result = _run_complete_with_a_failing_progress_line(tmp_path, monkeypatch, "variant unification done")
    assert complete == [True] and result is not None
    reason = result.metadata.get("orientation_source_reason", "")
    assert "variant unification was skipped" not in reason
    assert "those of the Hough resolver" not in reason


def test_an_error_before_the_orientations_were_unified_still_says_the_step_was_skipped(
        complete, monkeypatch, tmp_path):
    result = _run_complete_with_a_failing_progress_line(tmp_path, monkeypatch, "unifying variant speckle")
    assert complete == []                                       # the unification was never asked
    reason = result.metadata["orientation_source_reason"]
    assert "variant unification was skipped" in reason and "(see the log)" in reason


# ---- the call sites: every caller that holds a run stamp hands its geometry to the two stages
# Route-level, as tests/test_result_scan_geometry.py does it for the stamp itself: the stubbed
# Spherical function switches the active dataset to a square one mid-run, and records what it was given.

def _hexagonal_run_that_ends_on_a_square_dataset(monkeypatch, tmp_path, sht_paths):
    """Drive the real ``POST /api/indexing/start`` (method spherical, GPU backend) on dataset "ds",
    hexagonal. The stubbed Spherical function makes the square dataset "other" the active one, as a
    crop or a file switch during a long run would. Returns what the stubs saw."""
    import indexing_controller
    import kikuchipy as kp
    from fastapi.testclient import TestClient
    from kikuchipy.detectors import EBSDDetector
    from backend.api.main import app
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.routes import indexing as I
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
    up1 = tmp_path / "scan.up1"
    up1.write_bytes(b"not an hdf5 file")
    monkeypatch.setattr(ev, "_ebsd_file_path", str(up1), raising=False)
    monkeypatch.setattr(I, "_submit_run_guarded", lambda fn, kind="indexing": fn())
    seen = {"spherical": [], "rescue": [], "signal": sig}

    def spherical(h5_path, config, detector_params, selection_mask=None, **kwargs):
        seen["spherical"].append(kwargs)
        monkeypatch.setattr(ev, "_active_dataset", "other")          # the switch, during the run
        mask = np.asarray(selection_mask, dtype=bool)
        return ic.IndexingResult(
            xmap=None, selection_mask=mask, original_shape=mask.shape,
            method=ic.IndexingMethod.SPHERICAL,
            confidence_scores=np.full(mask.size, 0.5, dtype=np.float32))

    monkeypatch.setattr(indexing_controller, "spherical_gpu_index_patterns", spherical)
    monkeypatch.setattr(I, "_apply_particle_rescue", lambda *a, **k: seen["rescue"].append(k))
    scan_geometry.clear_all()
    scan_geometry.set_geometry("ds", ScanGeometry.hex(3, 3, 0.5))
    scan_geometry.set_geometry("other", ScanGeometry.square(3, 3, 0.5))
    saved, saved_active = dict(I._result_registry), I._active_result_id
    try:
        resp = TestClient(app).post("/api/indexing/start", json={
            "method": "spherical", "backend": "spherical_gpu", "sht_paths": sht_paths})
        assert resp.status_code == 200, resp.text
        task = I._indexing_tasks[resp.json()["task_id"]]
        assert task["status"] == "completed", task
        assert ev._active_dataset == "other"                          # premise: the switch happened
        return seen
    finally:
        scan_geometry.clear_all()
        I._result_registry.clear()
        I._result_registry.update(saved)
        I._active_result_id = saved_active


@pytest.mark.parametrize("sht_paths", [["a.sht"], ["a.sht", "b.sht"]],
                         ids=["single-phase call", "multi-phase call"])
def test_start_hands_the_runs_own_hexagonal_stamp_to_the_spherical_run_and_to_the_rescue(
        monkeypatch, tmp_path, sht_paths):
    seen = _hexagonal_run_that_ends_on_a_square_dataset(monkeypatch, tmp_path, sht_paths)
    assert len(seen["spherical"]) == 1
    assert seen["spherical"][0]["scan_geometry"]["kind"] == "hex"     # not the square "other"
    assert seen["spherical"][0]["signal"] is seen["signal"]           # the run's own signal, handed on
    assert len(seen["rescue"]) == 1
    assert seen["rescue"][0]["scan_geometry"]["kind"] == "hex"


def test_a_legacy_batch_item_hands_its_own_hexagonal_stamp_to_the_spherical_run(
        monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import indexing_controller
    from backend.api.main import app
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.routes import indexing as I
    from tests.apex_helpers import make_apex_h5, make_up, measurement_spec, wipe_backend

    folder = tmp_path / "hexbatch"
    folder.mkdir()
    make_apex_h5(folder / "s.h5", [measurement_spec(n_rows=N_ROWS, n_cols=N_COLS, step_x=0.5)])
    up = make_up(folder / "scan_hex.up1", n=N_ROWS * N_COLS, nx=N_COLS, ny=N_ROWS)
    seen = []

    def spherical(h5_path, config, detector_params, selection_mask, **kwargs):
        seen.append(dict(kwargs, signal_is_the_active_one=kwargs.get("signal") is ev._get_active_signal()))
        scan_geometry.set_geometry("other", ScanGeometry.square(2, 2, 0.25))
        monkeypatch.setattr(ev, "_active_dataset", "other")           # the switch, during the item
        mask = np.asarray(selection_mask, dtype=bool)
        return ic.IndexingResult(
            xmap=None, selection_mask=mask, original_shape=mask.shape,
            method=ic.IndexingMethod.SPHERICAL,
            confidence_scores=np.full(mask.size, 0.5, dtype=np.float32))

    monkeypatch.setattr(indexing_controller, "spherical_gpu_index_patterns", spherical)
    monkeypatch.setattr(I, "_attach_indexing_metadata", lambda *a, **k: None)
    monkeypatch.setattr(I, "_submit_run_guarded", lambda fn, kind="": fn())
    saved_batch, saved_results = dict(I._batch_state), dict(I._result_registry)
    wipe_backend()
    try:
        resp = TestClient(app).post("/api/indexing/batch/start", json={"datasets": [{
            "file_path": str(up), "method": "spherical", "selection_mode": "full",
            "export_result": False, "backend": "spherical_gpu",
            "sht_paths": [str(tmp_path / "master.sht")]}]})
        assert resp.status_code == 200, resp.text
        entry = I._batch_state["results"][-1]
        assert entry["error"] is None, entry
        assert len(seen) == 1
        assert seen[0]["scan_geometry"]["kind"] == "hex"              # the file's, not "other"'s
        assert seen[0]["signal_is_the_active_one"] is True            # the item's own, just loaded
    finally:
        scan_geometry.clear("other")
        wipe_backend()
        I._batch_state.clear()
        I._batch_state.update(saved_batch)
        I._result_registry.clear()
        I._result_registry.update(saved_results)
