"""A native hexagonal Orienta ``.h5`` loads as the index rectangle, with its geometry attached.

The file is the one ``orienta_hex_h5.write_fresh_file`` writes: ``n_rows * n_cols`` patterns in index
order, a ``HexGrid`` header with Orienta's own keys. ``safe_loader.load_ebsd_safe`` sends it to
``orienta_hex_h5.load_native_hex`` (not to kikuchipy, which refuses a hexagonal header, and not to the
OIM resampler of ``edax_hex``), and then settles the pattern centre as for any Orienta rich file.

Hand scan used throughout: 3 rows x 4 columns, step_x 2.0 um, odd rows shifted by +0.5 step,
step_y = 2 * sqrt(3) / 2 = 1.7320508075688772 um. Pattern [r, c] is the file's pattern r * 4 + c and
every pixel of pattern k holds k, so a value says by itself which source row it came from.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import orienta_hex_h5 as oh  # noqa: E402
from backend.api.services.scan_geometry import ScanGeometry, geometry_of_signal  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

N_ROWS, N_COLS = 3, 4
SHAPE = (60, 40)                    # rows, cols of a pattern: not square, taller than wide
PC = (0.5241, 0.4133, 0.6317)       # Bruker, as the detector holds it
TILT = 7.5
SAMPLE_TILT = 71.0
TILTS = {"tilt": TILT, "sample_tilt": SAMPLE_TILT}
SQRT3 = 1.7320508075688772          # 2 * sqrt(3) / 2, by hand


def _geometry(shift=0.5, confirmed=True):
    return ScanGeometry.hex(N_ROWS, N_COLS, 2.0, odd_row_shift=shift, shift_confirmed=confirmed)


def _patterns():
    n = N_ROWS * N_COLS
    return np.broadcast_to(np.arange(n, dtype=np.uint8)[:, None, None], (n,) + SHAPE).copy()


def _native(path, geometry=None, pc=PC, tilts=TILTS):
    oh.write_fresh_file(path, _patterns(), geometry or _geometry(), pc_bruker=pc, tilts=tilts)
    return path


def _load(path, **kwargs):
    from safe_loader import load_ebsd_safe
    return load_ebsd_safe(str(path), verbose=False, **kwargs)


def _pc(sig):
    return np.asarray(sig.detector.pc).ravel()


# ------------------------------------------------------------------ the rectangle and its geometry

@pytest.mark.parametrize("shift, confirmed", [(0.5, True), (-0.5, True), (0.5, False)])
def test_loads_as_rectangle_with_geometry(tmp_path, shift, confirmed):
    geometry = _geometry(shift, confirmed)
    sig = _load(_native(tmp_path / "hex.h5", geometry))

    # the index rectangle: (rows, cols, pattern rows, pattern cols), nothing padded or dropped
    assert sig.data.shape == (N_ROWS, N_COLS) + SHAPE
    assert sig.axes_manager.navigation_shape == (N_COLS, N_ROWS)          # hyperspy: (x, y)
    ax = {a.name: a for a in sig.axes_manager.navigation_axes}
    assert ax["x"].scale == 2.0 and ax["y"].scale == SQRT3                # steps, by hand
    # pattern [r, c] is source row r * n_cols + c
    for r in range(N_ROWS):
        for c in range(N_COLS):
            assert np.all(np.asarray(sig.data[r, c]) == r * N_COLS + c), (r, c)

    # the geometry rides on the signal as the loader's mirror, with the shift and its confirmation
    mirror = sig.metadata.get_item("Signal.scan_geometry").as_dictionary()
    assert mirror["kind"] == "hex" and mirror["grid_shape"] == [3, 4]
    assert mirror["odd_row_shift"] == shift and mirror["shift_confirmed"] is confirmed
    assert mirror["step_x"] == 2.0 and mirror["step_y"] == SQRT3
    assert geometry_of_signal(sig) == geometry
    # it is NOT a resampled map
    assert sig.metadata.get_item("Signal.hex_resampled", None) is None


def test_kikuchipy_not_called(tmp_path, monkeypatch):
    """The native branch needs neither kikuchipy's reader (it refuses HexGrid) nor its eager fallback."""
    import safe_loader

    def refuse(*args, **kwargs):
        raise AssertionError("kikuchipy's loader was called for a native hexagonal file")

    monkeypatch.setattr(safe_loader, "_kp", refuse)
    sig = _load(_native(tmp_path / "hex.h5"))
    assert sig.data.shape == (N_ROWS, N_COLS) + SHAPE


def test_lazy_equals_eager(tmp_path, monkeypatch):
    path = _native(tmp_path / "hex.h5")
    eager = _load(path)
    assert not hasattr(eager.data, "dask")
    monkeypatch.setenv("KIKUCHIPY_LAZY_THRESHOLD_BYTES", "0")           # every file is "large"
    lazy = _load(path)
    assert hasattr(lazy.data, "dask")
    np.testing.assert_array_equal(np.asarray(lazy.data), np.asarray(eager.data))
    assert geometry_of_signal(lazy) == geometry_of_signal(eager) == _geometry()
    np.testing.assert_allclose(_pc(lazy), _pc(eager), atol=0)
    assert lazy.detector.tilt == eager.detector.tilt
    del lazy


# ------------------------------------------------------------- pattern centre, as any rich file

def test_header_pc_and_tilts_are_the_detector(tmp_path):
    path = _native(tmp_path / "hex.h5")
    # the header, by hand: x* = PCx, y* = 1 - PCy, z* = PCz * rows / min(rows, cols)
    with h5py.File(path, "r") as f:
        g = f["Scan1/EBSD/Header/Pattern Center Calibration"]
        assert g["x-star"][0] == pytest.approx(PC[0])
        assert g["y-star"][0] == pytest.approx(1 - PC[1])
        assert g["z-star"][0] == pytest.approx(PC[2] * 60 / 40)
    sig = _load(path)
    np.testing.assert_allclose(_pc(sig), PC, atol=1e-6)
    assert tuple(sig.detector.shape) == SHAPE
    assert sig.detector.tilt == pytest.approx(TILT)
    assert sig.detector.sample_tilt == pytest.approx(SAMPLE_TILT)
    assert sig.metadata.get_item("Signal.pc_source", None) is None          # the header states it


def test_no_pc_is_invented_when_none_is_known(tmp_path):
    from backend.api.services.calibration_store import CalibrationStore
    path = _native(tmp_path / "hex.h5", pc=None, tilts=None)
    with h5py.File(path, "r") as f:
        assert "Pattern Center Calibration" not in f["Scan1/EBSD/Header"]
    sig = _load(path)
    assert sig.metadata.get_item("Signal.pc_source") == "default"
    assert CalibrationStore().register("hex", sig).pc_source == "missing"
    assert sig.detector.sample_tilt == pytest.approx(70.0)                   # the writer's default


def _without_header_pc(tmp_path, detector_pc=PC, detector_shape=SHAPE):
    """What an export without a header PC looks like: /Detector/pc present."""
    path = _native(tmp_path / "old.h5", pc=None)
    with h5py.File(path, "a") as f:
        det = f.create_group("Detector")
        det.create_dataset("pc", data=np.asarray(detector_pc, dtype=float))
        det.attrs["sample_tilt"] = SAMPLE_TILT
        det.attrs["shape"] = list(detector_shape)
    return path


def test_detector_group_is_the_fallback_when_the_header_has_no_pc(tmp_path):
    sig = _load(_without_header_pc(tmp_path))
    np.testing.assert_allclose(_pc(sig), PC, atol=1e-6)
    assert sig.metadata.get_item("Signal.pc_source") == "orienta_export"
    assert tuple(sig.detector.shape) == SHAPE


def test_detector_pc_of_another_pattern_size_is_not_applied(tmp_path):
    sig = _load(_without_header_pc(tmp_path, detector_shape=(60, 80)))
    assert sig.metadata.get_item("Signal.pc_source") == "default"


def test_header_pc_wins_over_the_detector_group(tmp_path):
    path = _native(tmp_path / "both.h5")
    with h5py.File(path, "a") as f:
        f.create_group("Detector").create_dataset("pc", data=np.array([0.1, 0.2, 0.3]))
    sig = _load(path)
    np.testing.assert_allclose(_pc(sig), PC, atol=1e-6)
    assert sig.metadata.get_item("Signal.export_pc_ignored") == pytest.approx([0.1, 0.2, 0.3])


# ------------------------------------------------------------------------------- refusals

@pytest.mark.parametrize("key, named", [("Odd Row Shift", "odd_row_shift"),
                                        ("Shift Confirmed", "shift_confirmed"),
                                        ("Step Y", "step_y")])
def test_a_damaged_stamp_is_refused_never_read_as_square(tmp_path, key, named):
    path = _native(tmp_path / "damaged.h5")
    with h5py.File(path, "a") as f:
        del f["Scan1/EBSD/Header"][key]
    assert oh.is_native_hex_file(path)                      # still Orienta's hexagonal file
    with pytest.raises(ValueError, match=f"HexGrid stamp lacks {named}"):
        _load(path)


def test_a_damaged_stamp_reaches_the_user_as_a_400(tmp_path):
    from fastapi.testclient import TestClient
    from backend.api.main import app
    from tests.apex_helpers import wipe_backend

    wipe_backend()
    try:
        path = _native(tmp_path / "damaged.h5")
        with h5py.File(path, "a") as f:
            del f["Scan1/EBSD/Header/Odd Row Shift"]
        resp = TestClient(app).post("/api/ebsd/load", json={"path": str(path)})
        assert resp.status_code == 400, resp.text
        assert "HexGrid stamp lacks odd_row_shift" in resp.json()["detail"]
    finally:
        wipe_backend()


@pytest.mark.parametrize("refusal", ["no_pattern", "wrong_shape", "header_read"])
def test_a_refused_lazy_load_leaves_the_file_unlocked(tmp_path, monkeypatch, refusal):
    """A lazy load keeps the file open for the signal it returns; when it refuses instead, nothing
    holds the file. On Windows an open HDF5 file can be neither renamed nor opened for writing."""
    path = _native(tmp_path / "hex.h5")
    if refusal == "no_pattern":
        with h5py.File(path, "a") as f:
            del f[oh.PATTERN_PATH]
    elif refusal == "wrong_shape":
        with h5py.File(path, "a") as f:
            del f[oh.PATTERN_PATH]
            # 12 = 3 * 4 rows, so the header read accepts it, but a pattern stack is 3-D
            f.create_dataset(oh.PATTERN_PATH, data=np.zeros((12, SHAPE[0] * SHAPE[1]), np.uint8))
    else:
        import kikuchipy.io.plugins._h5ebsd as h5ebsd

        def unreadable(*args, **kwargs):
            raise ValueError("header unreadable")

        monkeypatch.setattr(h5ebsd, "_hdf5group2dict", unreadable)
    # ``refused`` keeps the traceback, and with it the frame of the loader, alive: a file the loader
    # left open stays open until this test ends, as it would for a caller that logs the error
    with pytest.raises(ValueError) as refused:
        oh.load_native_hex(path, lazy=True, verbose=False)
    moved = path.rename(tmp_path / "moved.h5")                  # fails while the file is still open
    with h5py.File(moved, "w") as f:                            # and so does opening it for writing
        f.create_dataset("x", data=np.zeros(1))
    assert refused.value is not None


def test_a_lazy_load_that_succeeds_keeps_the_file_for_the_signal(tmp_path):
    path = _native(tmp_path / "hex.h5")
    sig = oh.load_native_hex(path, lazy=True, verbose=False)
    assert hasattr(sig.data, "dask")
    np.testing.assert_array_equal(np.asarray(sig.data[1, 2]), np.full(SHAPE, 1 * N_COLS + 2, np.uint8))
    del sig


def test_the_load_route_answers_a_native_file_as_hexagonal(tmp_path):
    from fastapi.testclient import TestClient
    from backend.api.main import app
    from tests.apex_helpers import wipe_backend

    wipe_backend()
    try:
        resp = TestClient(app).post("/api/ebsd/load", json={"path": str(_native(tmp_path / "hex.h5"))})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["scan_geometry"]["kind"] == "hex"
        assert body["scan_geometry"]["grid_shape"] == [3, 4]
        assert body["hex_resampled"] is False
    finally:
        wipe_backend()


def test_a_variant_the_file_does_not_hold_is_refused(tmp_path):
    from backend.api.services.load_choices import LoadChoiceError, LoadVariant
    path = _native(tmp_path / "hex.h5")
    with pytest.raises(LoadChoiceError):
        _load(path, variant=LoadVariant("Scan7", "single"))
    sig = _load(path, variant=LoadVariant("Scan1", "single"))                # the one variant
    assert sig.metadata.get_item("Orienta.load_variant").as_dictionary() == {
        "measurement": "Scan1", "pattern_source": "single"}


# ------------------------------------------------------------ the OIM file keeps its own path

def _oim_hex_file(path, n_cols=4, n_rows=4, step_x=1.0):
    """An OIM-form hexagonal file: padded X/Y positions (-1111111 marks a slot without a
    measurement), patterns for the real points only, OIM's own ``Version``."""
    step_y = step_x * np.sqrt(3) / 2
    x = np.full(n_rows * n_cols, -1111111.0, np.float32)
    y = np.full(n_rows * n_cols, -1111111.0, np.float32)
    for r in range(n_rows):
        n = n_cols if r % 2 == 0 else n_cols - 1
        x0 = 0.0 if r % 2 == 0 else step_x / 2
        x[r * n_cols:r * n_cols + n] = x0 + np.arange(n) * step_x
        y[r * n_cols:r * n_cols + n] = r * step_y
    n_valid = int((x > -1.0e5).sum())                         # 4 + 3 + 4 + 3 = 14
    pats = np.broadcast_to(np.arange(n_valid, dtype=np.uint8)[:, None, None], (n_valid, 6, 5)).copy()
    with h5py.File(path, "w") as f:
        f.create_dataset("Manufacturer", data=np.bytes_("EDAX"))
        f.create_dataset("Version", data=np.bytes_("OIM Analysis 8.6.0"))
        # the scan group is named as Orienta's own file names it: only Version tells the two apart
        hdr = f.create_group("Scan1/EBSD/Header")
        hdr.create_dataset("Grid Type", data=np.bytes_("HexGrid"))
        hdr.create_dataset("nColumns", data=np.array([n_cols], np.int32))
        hdr.create_dataset("nRows", data=np.array([n_rows], np.int32))
        hdr.create_dataset("Step X", data=np.array([step_x], np.float32))
        hdr.create_dataset("Step Y", data=np.array([step_y], np.float32))
        hdr.create_dataset("Pattern Height", data=np.array([6], np.int32))
        hdr.create_dataset("Pattern Width", data=np.array([5], np.int32))
        data = f.create_group("Scan1/EBSD/Data")
        data.create_dataset("X Position", data=x)
        data.create_dataset("Y Position", data=y)
        data.create_dataset("Pattern", data=pats)
    return path


def test_oim_hex_still_resamples(tmp_path):
    path = _oim_hex_file(tmp_path / "oim_hex.h5")
    assert not oh.is_native_hex_file(path)
    sig = _load(path)
    assert sig.metadata.get_item("Signal.hex_resampled") is True
    assert sig.metadata.get_item("Signal.scan_geometry", None) is None       # a square map
    assert sig.data.shape == (4, 4, 6, 5)
    # Square point (r, c) sits at (c, r) um; the hex points are at x = 0..3 (rows 0, 2) and
    # x = 0.5, 1.5, 2.5 (rows 1, 3) with y = r * 0.866. Patterns 0-3 are row 0, 4-6 row 1, 7-10 row 2,
    # 11-13 row 3. By hand: (0, 0) -> 0, (0, 3) -> 3 (row 0, x = 3), (3, 0) -> x = 0.5 of row 3 = 11,
    # (3, 3) -> x = 2.5 of row 3 = 13.
    got = lambda r, c: int(np.asarray(sig.data[r, c])[0, 0])     # noqa: E731
    assert (got(0, 0), got(0, 3), got(3, 0), got(3, 3)) == (0, 3, 11, 13)


# ---------------------------------------------------------- the other readers of a hex file

def test_callers_fail_soft(tmp_path):
    """``edax_hex.is_edax_hex_file`` stays True for a native file (it keys on the grid type), so its
    other callers reach for ``X Position`` / ``Y Position``, which a native file does not have: they
    must answer None, not raise. The crop writer still refuses."""
    import edax_hex
    from backend.api.services import orientation_ground_truth, pattern_quality
    from backend.api.services.crop_export import write_cropped_h5oina, _looks_like_hex
    from backend.api.services.crop_window import CropWindow

    path = _native(tmp_path / "hex.h5")
    assert edax_hex.is_edax_hex_file(str(path)) is True
    assert pattern_quality.remap_hex_channel(str(path), np.arange(N_ROWS * N_COLS, dtype=float)) is None
    assert orientation_ground_truth._hex_square_map(str(path)) is None

    assert _looks_like_hex(str(path)) is True
    out = tmp_path / "crop.h5"
    window = CropWindow("hex.h5", row0=0, col0=0, rows=2, cols=2, original_shape=(N_ROWS, N_COLS))
    with pytest.raises(ValueError, match="hex"):
        write_cropped_h5oina(str(path), str(out), window)
    assert not out.exists()
