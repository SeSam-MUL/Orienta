"""NORDIF and Oxford .ebsp: calibration patterns as a scan, an honest pattern centre, the beam energy."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import safe_loader  # noqa: E402
from backend.api.services.calibration_store import CalibrationStore  # noqa: E402


def _public(*parts) -> Path:
    """A file of the kikuchipy-paper reference data (Ånes et al., doi:10.5281/zenodo.20290635, CC-BY 4.0)."""
    root = os.environ.get("ORIENTA_PUBLIC_DATA")
    if not root:
        pytest.skip("ORIENTA_PUBLIC_DATA is not set (folder with the kikuchipy-paper reference data)")
    path = Path(root, *parts)
    if not path.exists():
        pytest.skip(f"{path} not found")
    return path


# ------------------------------------------------------------------- synthetic

def test_calibration_patterns_become_a_scan_of_one_row():
    data = np.arange(5 * 6 * 7, dtype=np.uint8).reshape(5, 6, 7)
    scan = {"data": data, "metadata": {"General": {}}, "detector": "kept",
            "axes": [{"size": 5, "index_in_array": 0, "name": "x", "scale": 1.0, "offset": 0, "units": "um"}]}
    out = safe_loader._calibration_scan_to_grid(scan)
    assert out["data"].shape == (1, 5, 6, 7)
    assert np.array_equal(out["data"][0, 3], data[3])        # pattern 3 is at (row 0, col 3)
    assert [a["name"] for a in out["axes"]] == ["y", "x", "dy", "dx"]
    assert [a["size"] for a in out["axes"]] == [1, 5, 6, 7]
    assert "units" not in out["axes"][0] and "units" not in out["axes"][1]   # the steps stay undefined
    assert out["detector"] == "kept" and scan["data"].shape == (5, 6, 7)     # the input is untouched


def test_calibration_patterns_of_the_wrong_shape_are_refused():
    with pytest.raises(ValueError):
        safe_loader._calibration_scan_to_grid({"data": np.zeros((6, 7))})


def test_the_beam_energy_falls_back_to_the_signals_metadata(tmp_path):
    from backend.api.routes.ebsd_viewer import _extract_beam_energy
    sig = SimpleNamespace(metadata=SimpleNamespace(get_item=lambda key, default=None: (
        20.0 if key == "Acquisition_instrument.SEM.beam_energy" else default)))
    absent = str(tmp_path / "Pattern.dat")                 # no HDF5 header to read
    assert _extract_beam_energy(absent, sig) == 20.0
    assert _extract_beam_energy(absent) is None
    zero = SimpleNamespace(metadata=SimpleNamespace(get_item=lambda key, default=None: 0.0))
    assert _extract_beam_energy(absent, zero) is None


def test_a_text_file_that_is_not_a_nordif_setting_is_refused_by_name(tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("remember to order tape\nmeeting on friday\n")
    with pytest.raises(ValueError, match=r"notes\.txt is not a NORDIF Setting\.txt") as caught:
        safe_loader.load_ebsd_safe(str(notes), verbose=False)
    assert caught.value.__cause__ is not None                       # kikuchipy's own error stays attached


def test_a_missing_text_file_says_it_does_not_exist(tmp_path):
    missing = tmp_path / "Setting.txt"
    with pytest.raises(ValueError, match=r"Setting\.txt: the file does not exist"):
        safe_loader.load_ebsd_safe(str(missing), verbose=False)


def test_the_crystal_hint_reads_the_signals_beam_energy(monkeypatch, tmp_path):
    """The hint's lattice estimate needs the voltage; a file without an HDF5 header has it only on the signal."""
    from types import SimpleNamespace
    from backend.api.routes import ebsd_viewer
    from backend.api.routes.crystal_hint import _gather_detector_geom
    det = SimpleNamespace(pc=np.array([0.5, 0.5, 0.7]))
    metadata = SimpleNamespace(get_item=lambda key, default=None: (
        25.0 if key == "Acquisition_instrument.SEM.beam_energy" else default))   # not the 20 kV default
    sig = SimpleNamespace(axes_manager=SimpleNamespace(signal_shape=(64, 64)), detector=det, metadata=metadata)
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda: sig)
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "ds")
    monkeypatch.setattr(ebsd_viewer, "_ebsd_file_path", str(tmp_path / "Pattern.dat"))
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store.get_detector",
                        lambda name: det)
    assert _gather_detector_geom()["voltage_kv"] == 25.0


def test_the_metadata_route_hands_the_signal_to_the_beam_energy(monkeypatch, tmp_path):
    """/metadata must pass the loaded signal on, or a file without an HDF5 header shows no energy."""
    import asyncio
    import inspect
    import kikuchipy as kp
    from backend.api.routes import ebsd_viewer
    sig = kp.signals.EBSD(np.zeros((1, 2, 8, 8), dtype=np.uint8))
    sig.metadata.set_item("Acquisition_instrument.SEM.beam_energy", 20.0)
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda: sig)
    # Without a dataset of its own, /metadata asks the calibration store for the detector of whatever an
    # earlier test left active; kikuchipy 0.13 refuses that entry's empty detector shape (order-dependent).
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "nordif-metadata-test")
    monkeypatch.setattr(ebsd_viewer, "_ebsd_file_path", str(tmp_path / "Pattern.dat"))
    meta = ebsd_viewer.get_metadata()
    if inspect.isawaitable(meta):
        meta = asyncio.run(meta)
    assert meta["beam_energy"] == 20.0


# ------------------------------------------------------------------ public data

def test_the_calibration_patterns_of_a_nordif_measurement_form_a_scan_of_one_row():
    path = _public("sdss", "sdss_nordif", "Setting.txt")
    sig = safe_loader.load_ebsd_safe(str(path), verbose=False)
    assert sig.axes_manager.navigation_shape == (5, 1)            # (cols, rows): five patterns in one row
    assert sig.axes_manager.signal_shape == (160, 160)            # "Resolution 160x160 px" (Setting.txt, line 54)
    assert np.asarray(sig.data[0, 3]).shape == (160, 160)         # the PC page reads data[row, col]
    # Pattern k of the scan is the k-th image listed under [Calibration patterns] in Setting.txt (lines 88-93,
    # first column = file name without .bmp). The order is read from the text here and the pixels from the
    # BMP files with Pillow: neither goes through the loader. The order is not the file-name order.
    from PIL import Image
    lines = [ln.strip() for ln in path.read_text(encoding="latin-1").splitlines()]
    start = lines.index("[Calibration patterns]") + 1
    names = []
    for line in lines[start:]:
        if not line.strip():
            break
        names.append(line.split("\t")[0])
    assert len(names) == 5 and names != sorted(names)
    for k, name in enumerate(names):
        bmp = np.asarray(Image.open(path.parent / f"{name}.bmp"))
        assert np.array_equal(np.asarray(sig.data[0, k]), bmp), f"pattern {k} is not {name}.bmp"
    store = CalibrationStore()
    assert store.register("cal", sig).pc_source == "missing"
    assert sig.metadata.get_item("Acquisition_instrument.SEM.beam_energy") == 20.0   # "Accelerating voltage 20 kV"
    from backend.api.routes.ebsd_viewer import _extract_step_size
    assert _extract_step_size(sig) is None                        # no scale bar for patterns that are not a map


def test_a_nordif_scan_has_no_pc_and_says_so():
    path = _public("sdss", "sdss_nordif", "Pattern.dat")
    sig = safe_loader.load_ebsd_safe(str(path), verbose=False)
    n_cols, n_rows = sig.axes_manager.navigation_shape
    assert n_rows * n_cols == 784 * 1121                          # "Number of samples 784x1121" (Setting.txt, line 80)
    store = CalibrationStore()
    entry = store.register("sdss", sig)
    assert (entry.pc_source, entry.pc_origin) == ("missing", "missing")
    assert store.get_pc("sdss") is None
    from backend.api.routes.ebsd_viewer import _extract_beam_energy
    assert _extract_beam_energy(str(path), sig) == 20.0


def test_an_oxford_ebsp_scan_has_no_pc_and_says_so():
    folder = _public("al10si", "al10si_oxford")
    files = sorted(folder.glob("*.ebsp"))
    if not files:
        pytest.skip(f"no .ebsp in {folder}")
    sig = safe_loader.load_ebsd_safe(str(files[0]), verbose=False)
    n_cols, n_rows = sig.axes_manager.navigation_shape
    # The dataset's README gives no map size. Independent of kikuchipy: an Oxford .ebsp (version 4) starts
    # with the version (int64, -4), one unknown byte, then a table of N int64 byte positions, one per
    # pattern; the first pattern follows the table, so the smallest position is 9 + 8 N. Read with numpy.
    head = np.fromfile(files[0], dtype="<i8", count=100_000, offset=9)
    first_pattern = int(head[head > 0].min())
    n_patterns = (first_pattern - 9) // 8
    table = np.fromfile(files[0], dtype="<i8", count=n_patterns, offset=9)
    assert table.min() == first_pattern and table.max() < files[0].stat().st_size
    assert n_patterns == 301 * 251                                # the table holds 75 551 positions
    assert n_rows * n_cols == n_patterns
    entry = CalibrationStore().register("ebsp", sig)
    assert (entry.pc_source, entry.pc_origin) == ("missing", "missing")


# ---------------------------------- the chain: load route, then a run is refused for want of a PC

def _load_then_start_hough(path, tmp_path):
    """POST /api/ebsd/load then POST /api/indexing/start (Hough, a minimal Ni CIF) through the real app.
    Returns (load response, start response). The loaded files are forgotten afterwards."""
    from fastapi.testclient import TestClient
    from backend.api.main import app
    from tests.apex_helpers import wipe_backend
    from tests.test_result_keeps_its_run_pc import _ni_cif

    client = TestClient(app)
    wipe_backend()
    try:
        loaded = client.post("/api/ebsd/load", json={"path": str(path)})
        started = client.post("/api/indexing/start", json={"method": "hough", "cif_paths": [_ni_cif(tmp_path)]})
        return loaded, started
    finally:
        wipe_backend()


def _assert_refused_for_want_of_a_pc(loaded, started):
    from backend.api.services.pc_guard import PC_MISSING_CODE
    assert loaded.status_code == 200, loaded.text
    assert loaded.json()["pc_origin"] == "missing", loaded.json()
    assert started.status_code == 400, started.text
    assert started.headers["x-orienta-code"] == PC_MISSING_CODE == "pc_missing"


def test_a_nordif_measurement_without_a_pc_is_refused_at_the_start_of_a_run(tmp_path):
    # Setting.txt is the NORDIF measurement's own file: five calibration patterns as a scan of one row
    # (the 12.6 GB Pattern.dat of the same folder goes through the same registration, see above).
    path = _public("sdss", "sdss_nordif", "Setting.txt")
    _assert_refused_for_want_of_a_pc(*_load_then_start_hough(path, tmp_path))


def test_an_oxford_ebsp_scan_without_a_pc_is_refused_at_the_start_of_a_run(tmp_path):
    folder = _public("al10si", "al10si_oxford")
    files = sorted(folder.glob("*.ebsp"))
    if not files:
        pytest.skip(f"no .ebsp in {folder}")
    _assert_refused_for_want_of_a_pc(*_load_then_start_hough(files[0], tmp_path))
