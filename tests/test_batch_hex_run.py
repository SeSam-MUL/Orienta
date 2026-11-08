"""The batch indexes a hexagonal scan on the geometry it loaded.

The load used to refuse a hexagonal file and then drop the loader's copy of its geometry, after which the
signal read as a square scan. Now the run reads the geometry of the file once, stamps its checkpoint with it
and hands it to preprocessing and to the indexer. Frame averaging of a hexagonal scan runs on the six
neighbours of a point (not on kikuchipy's rectangular window), and a refused or failed averaging fails that
file alone, with its code.

The run goes through ``BatchManager._run_batch_sync_impl`` with a real ``_load_signal`` (only the file reader is
replaced by a signal built here) and a recording stand-in for the indexer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.api.services import batch_manager as M
from backend.api.services import frame_average, scan_geometry
from backend.api.services.checkpoint_writer import read_checkpoint_geometry
from backend.api.services.scan_geometry import ScanGeometry

N_ROWS, N_COLS = 5, 5
HEX_STEP = 0.5


def _signal(data, geometry, *, mirror=True, lazy=False):
    """A kikuchipy signal of ``data`` (n_rows, n_cols, h, w) with the navigation axes of ``geometry`` and, like
    the loaders of a hexagonal file, the geometry as ``Signal.scan_geometry``."""
    import kikuchipy as kp
    signal = kp.signals.EBSD(np.array(data))
    x_axis, y_axis = signal.axes_manager.navigation_axes
    x_axis.scale, y_axis.scale = geometry.step_x, geometry.step_y
    x_axis.units = y_axis.units = "um"
    if mirror:
        signal.metadata.set_item("Signal.scan_geometry", geometry.to_dict())
    return signal.as_lazy() if lazy else signal


def _hex(shift=0.5, n_rows=N_ROWS, n_cols=N_COLS):
    return ScanGeometry.hex(n_rows, n_cols, HEX_STEP, odd_row_shift=shift)


class _Indexer:
    """Stands in for ``run_single_indexing_job``: it records what the run hands over."""

    def __init__(self, ci_maps=None):
        self.calls = []
        self.ci_maps = ci_maps or {}          # phase name -> CI map; every other phase is 0.5 everywhere

    def __call__(self, signal, file_path, phase_name, phase_path, method, grid_shape, job_config=None,
                 progress_callback=None, scan_geometry=None):
        self.calls.append({"file": file_path, "grid_shape": tuple(grid_shape), "scan_geometry": scan_geometry,
                           "data": np.array(signal.data)})
        ci = self.ci_maps.get(phase_name)
        ci = np.full(grid_shape, 0.5, dtype=np.float32) if ci is None else np.asarray(ci, dtype=np.float32)
        return (ci, np.zeros(tuple(grid_shape) + (3,), dtype=np.float32),
                {"ci_mean": 0.5, "ci_median": 0.5, "duration_sec": 0.1, "space_group": 225, "point_group": "m-3m",
                 "lattice_constants": [3.52, 3.52, 3.52, 90.0, 90.0, 90.0]})


@pytest.fixture
def run(monkeypatch, tmp_path):
    """``run(signals, preprocessing=None, postprocessing=None, phases=("Ni",), indexer=None, method="hough")`` ->
    (manager, batch id, indexer). ``signals``: file name -> signal. ``indexer=False`` leaves the real
    ``run_single_indexing_job`` in place."""
    import safe_loader
    monkeypatch.setattr(M, "_batch_names", {})
    monkeypatch.setattr(M, "_register_signal_calibration", lambda signal, file_path: None)

    def go(signals, preprocessing=None, postprocessing=None, phases=("Ni",), indexer=None, method="hough"):
        files = {str(tmp_path / name): signal for name, signal in signals.items()}
        monkeypatch.setattr(safe_loader, "load_ebsd_safe", lambda path: files[str(path)])
        if indexer is not False:
            indexer = indexer or _Indexer()
            monkeypatch.setattr(M, "run_single_indexing_job", indexer)
        mgr = M.BatchManager(db_path=str(tmp_path / "jobs.db"))
        config = {"auto_export": False}
        if preprocessing is not None:
            config["preprocessing"] = preprocessing
        if postprocessing is not None:
            config["postprocessing"] = postprocessing
        cifs = []
        for name in phases:
            cif = tmp_path / f"{name}.cif"
            cif.write_text(f"data_{name}\n", encoding="utf-8")
            cifs.append({"name": name, "path": str(cif), "method": method})
        batch_id = mgr.queue.create_batch(list(files), cifs, config)
        mgr._run_batch_sync_impl(batch_id)
        return mgr, batch_id, indexer

    return go


def _checkpoint(tmp_path, name):
    return tmp_path / f"{name}_multiphase.h5"


# ---------------------------------------------------------------- the geometry the run loaded

def test_batch_load_keeps_the_mirror_of_a_hexagonal_scan(monkeypatch):
    """``_load_signal`` neither refuses nor takes the loader's copy off: the run reads it afterwards."""
    import safe_loader
    signal = _signal(np.zeros((N_ROWS, N_COLS, 1, 3), dtype=np.uint8), _hex())
    monkeypatch.setattr(safe_loader, "load_ebsd_safe", lambda path: signal)
    assert M._load_signal("scan.up1") is signal
    assert signal.metadata.has_item("Signal.scan_geometry")
    assert scan_geometry.geometry_of_signal(signal).kind == "hex"


def test_batch_hex_keeps_geometry(run, tmp_path, monkeypatch):
    """The checkpoint is stamped hexagonal and the indexer is handed the hexagonal stamp, although another
    (square) dataset is the one the viewer has open."""
    from backend.api.routes import ebsd_viewer
    scan_geometry.set_geometry("other", ScanGeometry.square(3, 3, 1.0))
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "other")
    geometry = _hex(shift=-0.5)
    data = np.zeros((N_ROWS, N_COLS, 1, 3), dtype=np.uint8)
    try:
        mgr, batch_id, indexer = run({"hexscan.h5oina": _signal(data, geometry)})
    finally:
        scan_geometry.clear_all()
    (job,) = mgr.get_jobs(batch_id)
    assert job["status"] == "done", job
    stamp = read_checkpoint_geometry(_checkpoint(tmp_path, "hexscan"))
    assert stamp is not None and stamp.kind == "hex"
    assert stamp.grid_shape == (N_ROWS, N_COLS)
    assert (stamp.odd_row_shift, stamp.shift_confirmed) == (-0.5, False)
    assert (stamp.step_x, stamp.step_y) == (geometry.step_x, geometry.step_y)
    (call,) = indexer.calls
    assert call["scan_geometry"] == geometry.to_dict()


def test_batch_square_file_is_stamped_square_and_handed_the_square_stamp(run, tmp_path):
    geometry = ScanGeometry.square(4, 3, 0.25)
    data = np.zeros((4, 3, 1, 3), dtype=np.uint8)
    mgr, batch_id, indexer = run({"sq.h5oina": _signal(data, geometry, mirror=False)})
    assert [j["status"] for j in mgr.get_jobs(batch_id)] == ["done"]
    stamp = read_checkpoint_geometry(_checkpoint(tmp_path, "sq"))
    assert stamp is not None and stamp.kind == "square" and stamp.grid_shape == (4, 3)
    assert indexer.calls[0]["scan_geometry"] == geometry.to_dict()


def test_batch_square_step_read_elsewhere_is_kept_and_not_refused_against_the_axes(run, tmp_path, monkeypatch):
    """Navigation axes without a scale read as a square scan with placeholder steps (1.0). When the batch
    finds the step elsewhere (the file's header: here 0.5), the checkpoint keeps that step as it always has,
    is not refused for the difference, and carries no stamp (a placeholder is not stamped)."""
    import h5py
    import kikuchipy as kp
    signal = kp.signals.EBSD(np.zeros((2, 3, 1, 3), dtype=np.uint8))
    monkeypatch.setattr(M, "_extract_step_size_um", lambda sig, path: 0.5)
    mgr, batch_id, indexer = run({"bare.h5oina": signal})
    assert [j["status"] for j in mgr.get_jobs(batch_id)] == ["done"]
    assert indexer.calls[0]["scan_geometry"]["kind"] == "square"
    assert read_checkpoint_geometry(_checkpoint(tmp_path, "bare")) is None
    with h5py.File(_checkpoint(tmp_path, "bare"), "r") as f:
        assert f["metadata"].attrs["step_size_um"] == 0.5


# ---------------------------------------------------------------- frame averaging on the six neighbours

#: Neighbours of the interior point (2, 2) of a 5 x 5 hexagonal scan, by hand. Rows are y = row * step_y; with
#: +0.5 the odd rows sit half a step to the right: row 2 holds x = col, row 1 and row 3 hold x = col + 0.5, so
#: the points of rows 1 and 3 half a step from x = 2 are the columns 1 and 2. With -0.5 they sit half a step
#: to the left (x = col - 0.5): the columns 2 and 3.
SIX = {
    0.5: [(2, 1), (2, 3), (1, 1), (1, 2), (3, 1), (3, 2)],
    -0.5: [(2, 1), (2, 3), (1, 2), (1, 3), (3, 2), (3, 3)],
}


def _marked_scan(shift):
    """Every pattern (0, 0, 70), except the point (2, 2) itself (70, 0, 0) and its six neighbours (0, 70, 0)."""
    data = np.zeros((N_ROWS, N_COLS, 1, 3), dtype=np.uint8)
    data[..., 2] = 70
    data[2, 2, 0] = (70, 0, 0)
    for r, c in SIX[shift]:
        data[r, c, 0] = (0, 70, 0)
    return data


@pytest.mark.parametrize("lazy", [False, True], ids=["in memory", "lazy"])
@pytest.mark.parametrize("shift", [0.5, -0.5])
def test_batch_hex_frame_average_six_neighbours(run, shift, lazy):
    """Mean of the point and its six neighbours: (70*1 + 0*6, 0*1 + 70*6, 0) / 7 = (10, 60, 0). Rescaled to the
    range of uint8 (kikuchipy's and ours): (v - min) / (max - min) * 255 with min 0, max 60 = (42.5, 255, 0).
    The two points of kikuchipy's 3 x 3 window that are not neighbours hold (0, 0, 70): a rectangular window
    would give (70/9, 420/9, 140/9) = (7.8, 46.7, 15.6), which rescales to (0, 255, 51)."""
    data = _marked_scan(shift)
    signal = _signal(data, _hex(shift), lazy=lazy)
    mgr, batch_id, indexer = run(
        {"h.h5oina": signal}, {"frame_averaging": True, "frame_averaging_window": 3})
    (job,) = mgr.get_jobs(batch_id)
    assert job["status"] == "done", job
    got = indexer.calls[0]["data"]
    assert got.dtype == np.uint8 and got.shape == data.shape
    assert np.abs(got[2, 2, 0].astype(float) - np.array([42.5, 255.0, 0.0])).max() <= 1.0, got[2, 2, 0]


def test_batch_hex_window_of_one_changes_nothing(run):
    data = _marked_scan(0.5)
    mgr, batch_id, indexer = run(
        {"h.h5oina": _signal(data, _hex())}, {"frame_averaging": True, "frame_averaging_window": 1})
    assert [j["status"] for j in mgr.get_jobs(batch_id)] == ["done"]
    np.testing.assert_array_equal(indexer.calls[0]["data"], data)


# ---------------------------------------------------------------- a refused averaging fails that file alone

def _refuse_even(monkeypatch, data):
    return {"frame_averaging": True, "frame_averaging_window": 4}


def _refuse_large(monkeypatch, data):
    return {"frame_averaging": True, "frame_averaging_window": frame_average.HEX_MAX_WINDOW + 2}


def _refuse_memory(monkeypatch, data):
    monkeypatch.setattr(frame_average, "HEX_MAX_BYTES", data.nbytes - 1)
    return {"frame_averaging": True, "frame_averaging_window": 3}


def _fail_inside(monkeypatch, data):
    def broken(*args, **kwargs):
        raise RuntimeError("the averaging broke")
    monkeypatch.setattr(frame_average, "average_neighbour_patterns", broken)
    return {"frame_averaging": True, "frame_averaging_window": 3}


@pytest.mark.parametrize("setup", [_refuse_even, _refuse_large, _refuse_memory, _fail_inside],
                         ids=["even window", "window over the limit", "patterns over the memory ceiling",
                              "failure inside the averaging"])
def test_batch_hex_frame_average_refusal_fails_that_file_with_its_code(run, monkeypatch, setup):
    """The hexagonal file fails with ``hexFrameAverageRefused`` and is not indexed unaveraged; the square file
    after it in the same batch is indexed."""
    hex_data = _marked_scan(0.5)
    sq_data = np.zeros((3, 3, 1, 3), dtype=np.uint8)
    config = setup(monkeypatch, hex_data)
    signals = {"a_hex.h5oina": _signal(hex_data, _hex()),
               "b_sq.h5oina": _signal(sq_data, ScanGeometry.square(3, 3, 0.5), mirror=False)}
    mgr, batch_id, indexer = run(signals, config)
    jobs = {Path(j["file_path"]).name: j for j in mgr.get_jobs(batch_id)}
    assert jobs["a_hex.h5oina"]["status"] == "failed", jobs
    assert jobs["a_hex.h5oina"]["error_code"] == "hexFrameAverageRefused"
    assert jobs["b_sq.h5oina"]["status"] == "done", jobs["b_sq.h5oina"]
    assert [c["file"].endswith("b_sq.h5oina") for c in indexer.calls] == [True]


# ---------------------------------------------------------------- a square file keeps the old behaviour

def test_batch_square_frame_average_failure_is_noted_and_the_run_goes_on():
    signal = _signal(np.zeros((3, 3, 1, 3), dtype=np.uint8), ScanGeometry.square(3, 3, 0.5), mirror=False)

    def broken(*args, **kwargs):
        raise RuntimeError("kikuchipy broke")

    signal.average_neighbour_patterns = broken
    _, applied = M._apply_preprocessing(
        signal, {"frame_averaging": True, "frame_averaging_window": 3}, ScanGeometry.square(3, 3, 0.5))
    assert applied["frame_averaging"].startswith("FAILED: kikuchipy broke")


def test_batch_square_frame_average_uses_kikuchipys_window():
    """3 x 3 on a square scan is kikuchipy's rectangular window. The point (1, 1) averages the marked point (2, 2)
    (90, 0, 0) with eight points of zeros: (90/9, 0, 0) = (10, 0, 0), which fills the range as (255, 0, 0)."""
    data = np.zeros((5, 5, 1, 3), dtype=np.uint8)
    data[2, 2, 0] = (90, 0, 0)
    signal = _signal(data, ScanGeometry.square(5, 5, 0.5), mirror=False)
    _, applied = M._apply_preprocessing(
        signal, {"frame_averaging": True, "frame_averaging_window": 3}, ScanGeometry.square(5, 5, 0.5))
    assert applied["frame_averaging"] == "3x3"
    assert np.asarray(signal.data)[1, 1, 0].astype(int).tolist() == [255, 0, 0]


# ---------------------------------------------------------------- what the checkpoint records about the averaging

@pytest.mark.parametrize("window, label", [
    (3, "hex, 1 ring"),                                  # (3 - 1) / 2 = 1 ring: the point and its 6 neighbours
    (5, "hex, 2 rings"),                                 # (5 - 1) / 2 = 2 rings: 19 points
    (7, "hex, 3 rings"),
    (1, "hex, 0 rings (nothing averaged)"),              # a window of one averages nothing
])
def test_a_hexagonal_file_records_the_rings_it_averaged_not_a_rectangular_window(window, label):
    signal = _signal(np.zeros((N_ROWS, N_COLS, 1, 3), dtype=np.uint8), _hex())
    _, applied = M._apply_preprocessing(
        signal, {"frame_averaging": True, "frame_averaging_window": window}, _hex())
    assert applied["frame_averaging"] == label


# ---------------------------------------------------------------- the step of a hexagonal file is the geometry's

def test_a_hexagonal_file_takes_its_step_from_the_geometry_before_any_other_source(run, tmp_path):
    """The navigation axes carry no usable scale (0: nothing `_extract_step_size_um` reads gives a step) and the
    file has no header. A square file in this state is refused ("Could not determine step size"), a hexagonal
    one states its step in its geometry: 0.5 um."""
    signal = _signal(np.zeros((N_ROWS, N_COLS, 1, 3), dtype=np.uint8), _hex())
    for axis in signal.axes_manager.navigation_axes:
        axis.scale = 0.0
    assert M._extract_step_size_um(signal, str(tmp_path / "nofile.h5oina")) == 0.0          # premise
    mgr, batch_id, indexer = run({"hexscan.h5oina": signal})
    (job,) = mgr.get_jobs(batch_id)
    assert job["status"] == "done", job
    import h5py
    with h5py.File(_checkpoint(tmp_path, "hexscan"), "r") as f:
        assert f["metadata"].attrs["step_size_um"] == HEX_STEP


# ---------------------------------------------------------------- the cleanup of the batch works on the lattice

#: The diagonal pair (1, 1), (2, 2) of the 5 x 5 scan with odd rows shifted +0.5: row 2 holds x = col, row 1 holds
#: x = col + 0.5, so the points of row 1 half a step either side of x = 2 are the columns 1 and 2 and (1, 1) is a
#: neighbour of (2, 2). On the index grid they are two points that touch at a corner only.
PAIR = ((1, 1), (2, 2))


def _two_phase_ci():
    """Phase A wins everywhere (CI 0.9) except at the pair, where phase B wins (CI 0.9 against 0.1)."""
    a = np.full((N_ROWS, N_COLS), 0.9, dtype=np.float32)
    b = np.full((N_ROWS, N_COLS), 0.1, dtype=np.float32)
    for r, c in PAIR:
        a[r, c], b[r, c] = 0.1, 0.9
    return {"A": a, "B": b}


def _assignment(path):
    import h5py
    with h5py.File(path, "r") as f:
        aa = f["auto_assignment"]
        return np.array(aa["best_phase_id"]), np.array(aa["cleaned_phase_id"])


def test_batch_cleanup_hex_neighbours(run, tmp_path):
    """Minimum cluster size 2: B at (1, 1) and (2, 2) is one cluster of two on the hexagonal lattice and stays."""
    mgr, batch_id, _ = run(
        {"hexscan.h5oina": _signal(np.zeros((N_ROWS, N_COLS, 1, 3), dtype=np.uint8), _hex())},
        postprocessing={"min_cluster_size": 2}, phases=("A", "B"), indexer=_Indexer(_two_phase_ci()))
    assert [j["status"] for j in mgr.get_jobs(batch_id)] == ["done", "done"]
    best, cleaned = _assignment(_checkpoint(tmp_path, "hexscan"))
    assert best[1, 1] == best[2, 2] == 1 and best.sum() == 2                        # premise: B is only the pair
    assert cleaned[1, 1] == cleaned[2, 2] == 1


def test_batch_cleanup_square_control(run, tmp_path):
    """The same maps on a square scan: two clusters of one, each below 2, taken by the surrounding phase A."""
    sq = ScanGeometry.square(N_ROWS, N_COLS, HEX_STEP)
    mgr, batch_id, _ = run(
        {"sqscan.h5oina": _signal(np.zeros((N_ROWS, N_COLS, 1, 3), dtype=np.uint8), sq, mirror=False)},
        postprocessing={"min_cluster_size": 2}, phases=("A", "B"), indexer=_Indexer(_two_phase_ci()))
    assert [j["status"] for j in mgr.get_jobs(batch_id)] == ["done", "done"]
    best, cleaned = _assignment(_checkpoint(tmp_path, "sqscan"))
    assert best[1, 1] == best[2, 2] == 1                                            # premise
    assert cleaned[1, 1] == cleaned[2, 2] == 0


# ---------------------------------------------------------------- a resumed checkpoint of another pairing fails that file

def _old_checkpoint(tmp_path, name, shape, geometry, apex=None):
    from backend.api.services.checkpoint_writer import CheckpointWriter
    CheckpointWriter(str(tmp_path / f"{name}.h5oina")).init_metadata(
        shape, "earlier-batch", "hough", geometry=geometry, apex=apex)


def test_checkpoint_pairing_changed_fails_that_file_and_the_batch_goes_on(run, tmp_path):
    """File ``a_hex`` is hexagonal (odd rows +0.5), the checkpoint on disk was written for odd rows -0.5: that file
    fails with its code and nothing is indexed for it; ``b_sq`` after it runs."""
    _old_checkpoint(tmp_path, "a_hex", (N_ROWS, N_COLS), _hex(shift=-0.5))
    signals = {"a_hex.h5oina": _signal(np.zeros((N_ROWS, N_COLS, 1, 3), dtype=np.uint8), _hex()),
               "b_sq.h5oina": _signal(np.zeros((3, 3, 1, 3), dtype=np.uint8), ScanGeometry.square(3, 3, 0.5),
                                      mirror=False)}
    mgr, batch_id, indexer = run(signals)
    jobs = {Path(j["file_path"]).name: j for j in mgr.get_jobs(batch_id)}
    assert jobs["a_hex.h5oina"]["status"] == "failed", jobs
    assert jobs["a_hex.h5oina"]["error_code"] == "checkpointPairingChanged"
    assert "delete the checkpoint" in jobs["a_hex.h5oina"]["error_msg"]
    assert jobs["b_sq.h5oina"]["status"] == "done", jobs["b_sq.h5oina"]
    assert [c["file"].endswith("b_sq.h5oina") for c in indexer.calls] == [True]
    assert read_checkpoint_geometry(_checkpoint(tmp_path, "a_hex")) == _hex(shift=-0.5)      # left as it was


def test_a_square_file_with_a_stamped_checkpoint_of_another_step_fails_the_same_way(run, tmp_path):
    _old_checkpoint(tmp_path, "a_sq", (4, 3), ScanGeometry.square(4, 3, 0.25))
    signals = {"a_sq.h5oina": _signal(np.zeros((4, 3, 1, 3), dtype=np.uint8), ScanGeometry.square(4, 3, 0.5),
                                      mirror=False),
               "b_sq.h5oina": _signal(np.zeros((3, 3, 1, 3), dtype=np.uint8), ScanGeometry.square(3, 3, 0.5),
                                      mirror=False)}
    mgr, batch_id, indexer = run(signals)
    jobs = {Path(j["file_path"]).name: j for j in mgr.get_jobs(batch_id)}
    assert (jobs["a_sq.h5oina"]["status"], jobs["a_sq.h5oina"]["error_code"]) == ("failed", "checkpointPairingChanged")
    assert jobs["b_sq.h5oina"]["status"] == "done"
    assert len(indexer.calls) == 1


def test_a_square_file_with_a_placeholder_step_does_not_resume_a_hexagonal_checkpoint(run, tmp_path, monkeypatch):
    """The axes carry no scale (placeholder 1.0) and the step is found elsewhere (0.5), so the square file is not
    stamped and has no geometry to compare. The checkpoint of the same name and grid shape is hexagonal: the kind
    is still compared, the file fails with its code and is not cleaned later on six neighbours."""
    import kikuchipy as kp
    monkeypatch.setattr(M, "_extract_step_size_um", lambda sig, path: 0.5)
    _old_checkpoint(tmp_path, "bare", (2, 3), ScanGeometry.hex(2, 3, 0.5))
    mgr, batch_id, indexer = run({"bare.h5oina": kp.signals.EBSD(np.zeros((2, 3, 1, 3), dtype=np.uint8))})
    (job,) = mgr.get_jobs(batch_id)
    assert (job["status"], job["error_code"]) == ("failed", "checkpointPairingChanged"), job
    assert indexer.calls == []
    assert read_checkpoint_geometry(_checkpoint(tmp_path, "bare")) == ScanGeometry.hex(2, 3, 0.5)


def _delete_attr(name):
    def damage(path):
        import h5py
        with h5py.File(path, "a") as f:
            del f["metadata"].attrs[name]
    return damage


def _no_metadata_group(path):
    import h5py
    with h5py.File(path, "w") as f:
        f.create_group("phases")


def _not_an_hdf5_file(path):
    Path(path).write_bytes(b"this is not an HDF5 file")


@pytest.mark.parametrize("damage, code", [
    (_delete_attr("odd_row_shift"), "checkpointUnreadable"),     # a hex stamp that lacks its shift
    (_delete_attr("grid_shape"), "checkpointUnreadable"),        # a stamp without the grid it describes
    (_no_metadata_group, "checkpointUnreadable"),
    (_not_an_hdf5_file, "checkpointUnreadable"),
    (_delete_attr("grid_type"), "checkpointPairingChanged"),     # no grid_type reads as a legacy SQUARE checkpoint
], ids=["no odd_row_shift", "no grid_shape", "no /metadata", "not HDF5", "no grid_type"])
def test_a_checkpoint_that_cannot_be_read_fails_that_file_and_the_batch_ends_finished(
        run, tmp_path, damage, code, caplog):
    """Nothing in the checkpoint block may end the batch thread: the first file fails with its text and code, the
    second (healthy) file is indexed, and the batch is ``completed`` (not ``running``, no file left ``pending``).
    The log carries the traceback of the refusal, not only its message."""
    _old_checkpoint(tmp_path, "a_hex", (N_ROWS, N_COLS), _hex())
    damage(_checkpoint(tmp_path, "a_hex"))
    signals = {"a_hex.h5oina": _signal(np.zeros((N_ROWS, N_COLS, 1, 3), dtype=np.uint8), _hex()),
               "b_sq.h5oina": _signal(np.zeros((3, 3, 1, 3), dtype=np.uint8), ScanGeometry.square(3, 3, 0.5),
                                      mirror=False)}
    mgr, batch_id, indexer = run(signals)
    jobs = {Path(j["file_path"]).name: j for j in mgr.get_jobs(batch_id)}
    assert jobs["a_hex.h5oina"]["status"] == "failed", jobs
    assert jobs["a_hex.h5oina"]["error_code"] == code
    if code == "checkpointUnreadable":
        assert "checkpoint of a_hex.h5oina cannot be read" in jobs["a_hex.h5oina"]["error_msg"]
    assert jobs["b_sq.h5oina"]["status"] == "done", jobs["b_sq.h5oina"]
    assert [c["file"].endswith("b_sq.h5oina") for c in indexer.calls] == [True]
    assert mgr.queue.get_batch_status(batch_id)["status"] == "completed"
    refused = [r for r in caplog.records if r.getMessage().startswith("Checkpoint of a_hex")]
    assert refused and all(r.exc_info for r in refused), "the refusal is logged without its traceback"


def test_a_step_size_that_cannot_be_found_fails_that_file_with_its_text_and_the_batch_goes_on(run):
    """A square file whose axes carry no scale (0) and that has no header: the refusal is a per-file failure with
    its old text and no code; the next file runs and the batch is ``completed``."""
    bad = _signal(np.zeros((3, 3, 1, 3), dtype=np.uint8), ScanGeometry.square(3, 3, 0.5), mirror=False)
    for axis in bad.axes_manager.navigation_axes:
        axis.scale = 0.0
    signals = {"a_bad.h5oina": bad,
               "b_sq.h5oina": _signal(np.zeros((3, 3, 1, 3), dtype=np.uint8), ScanGeometry.square(3, 3, 0.5),
                                      mirror=False)}
    mgr, batch_id, indexer = run(signals)
    jobs = {Path(j["file_path"]).name: j for j in mgr.get_jobs(batch_id)}
    assert jobs["a_bad.h5oina"]["status"] == "failed", jobs
    assert jobs["a_bad.h5oina"]["error_code"] is None
    message = jobs["a_bad.h5oina"]["error_msg"]
    assert "Could not determine step size for a_bad.h5oina" in message
    # the text says what happens: this file is not indexed and the batch goes on (it does not claim to stop)
    assert "This file is not indexed" in message and "the batch goes on with the other files" in message
    assert "Refusing to start batch" not in message
    assert jobs["b_sq.h5oina"]["status"] == "done"
    assert mgr.queue.get_batch_status(batch_id)["status"] == "completed"


def _apex_signal(measurement_id, name):
    signal = _signal(np.zeros((N_ROWS, N_COLS, 1, 3), dtype=np.uint8), _hex())
    signal.metadata.set_item("Signal.apex", {"h5_path": "/a/folder/session.h5", "measurement_id": measurement_id,
                                             "measurement_name": name, "pairing": "guid"})
    return signal


def test_the_pairing_of_an_up_file_is_stored_and_another_measurement_fails_that_file(run, tmp_path):
    import h5py
    mgr, batch_id, _ = run({"u.h5oina": _apex_signal("GUID-A", "Area 1 / Map 1")})
    assert [j["status"] for j in mgr.get_jobs(batch_id)] == ["done"]
    with h5py.File(_checkpoint(tmp_path, "u"), "r") as f:
        attrs = f["metadata"].attrs
        assert (attrs["apex_h5_name"], attrs["apex_measurement_id"], attrs["apex_measurement_name"]) == (
            "session.h5", "GUID-A", "Area 1 / Map 1")
    mgr, batch_id, indexer = run({"u.h5oina": _apex_signal("GUID-B", "Area 1 / Map 2")})
    (job,) = mgr.get_jobs(batch_id)
    assert (job["status"], job["error_code"]) == ("failed", "checkpointPairingChanged"), job
    assert indexer.calls == []


# ---------------------------------------------------------------- EMSphInx refuses a hexagonal file by code

def test_a_hexagonal_file_fails_alone_in_the_emsphinx_engine_with_its_code(run, tmp_path, monkeypatch):
    """The real ``run_single_indexing_job`` of the spherical engine with EMSphInx, on the stamp of the file: the
    refusal is raised before anything is written or run, the job stores its code."""
    import indexing_controller
    from backend.api.routes import indexing as routes_indexing
    monkeypatch.setattr(M, "_get_detector_for_file", lambda file_path, signal: None)
    monkeypatch.setattr(routes_indexing, "build_spherical_det_params", lambda signal, detector, path: {})
    monkeypatch.setattr(indexing_controller, "_make_processed_temp_h5",
                        lambda *a, **k: pytest.fail("EMSphInx must not run on a hexagonal scan"))
    mgr, batch_id, _ = run(
        {"hexscan.h5oina": _signal(np.zeros((N_ROWS, N_COLS, 1, 3), dtype=np.uint8), _hex())},
        indexer=False, method="spherical")
    (job,) = mgr.get_jobs(batch_id)
    assert (job["status"], job["error_code"]) == ("failed", "emsphinxSourceRefused"), job
    assert "hexagonal scan" in job["error_msg"]
