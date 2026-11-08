"""EDAX UP files loaded with the metadata of their APEX measurement (safe_loader + edax_apex)."""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import safe_loader  # noqa: E402
from backend.api.services.scan_geometry import geometry_of_signal, plain  # noqa: E402
from edax_apex import ApexError  # noqa: E402
from edax_up1 import EdaxHexUpError  # noqa: E402
from tests.apex_helpers import (  # noqa: E402
    GUID_A, GUID_A_BYTES, GUID_B, GUID_B_BYTES, make_apex_h5, make_osc, make_up, measurement_spec,
)

N_ROWS, N_COLS = 3, 5


def _hex_folder(folder, **spec_over):
    """session.h5 with one hexagonal 3 x 5 measurement and scan.up1 holding its 15 patterns.

    The UP header is deliberately wrong about the grid (it says 45 rows, as a scan that ended
    early would): the metadata file is the authority."""
    h5 = make_apex_h5(folder / "session.h5", [measurement_spec(
        n_rows=N_ROWS, n_cols=N_COLS, step_x=0.5, **spec_over)])
    up = make_up(folder / "scan.up1", n=N_ROWS * N_COLS, nx=N_COLS, ny=45)
    return h5, up


@pytest.mark.parametrize("lazy", [False, True])
def test_a_hex_scan_is_read_row_by_row_into_its_own_grid(tmp_path, monkeypatch, lazy):
    _, up = _hex_folder(tmp_path)
    if lazy:
        monkeypatch.setenv("KIKUCHIPY_LAZY_THRESHOLD_BYTES", "0")
    sig = safe_loader.load_ebsd_safe(str(up), verbose=False)
    assert sig.data.shape == (N_ROWS, N_COLS, 8, 8)
    assert bool(sig._lazy) is lazy
    data = np.asarray(sig.data)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            # Pattern i is a constant image of 7*i mod 256, and acquisition order is row after
            # row, so the pattern at (row r, col c) is number i = r*5 + c (by hand).
            assert data[r, c, 0, 0] == (7 * (r * N_COLS + c)) % 256


def test_the_h5_grid_wins_over_the_up_header(tmp_path):
    _, up = _hex_folder(tmp_path)
    assert struct.unpack_from("<2I", up.read_bytes(), 17) == (N_COLS, 45)    # what the header claims ...
    sig = safe_loader.load_ebsd_safe(str(up), verbose=False)
    assert sig.axes_manager.navigation_shape == (N_COLS, N_ROWS)    # ... is not what loads


def test_the_axes_carry_both_steps_of_the_hex_lattice(tmp_path):
    _, up = _hex_folder(tmp_path)
    nav = safe_loader.load_ebsd_safe(str(up), verbose=False).axes_manager.navigation_axes
    assert (nav[0].name, nav[0].size) == ("x", N_COLS)
    assert (nav[1].name, nav[1].size) == ("y", N_ROWS)
    assert nav[0].scale == pytest.approx(0.5, rel=1e-6)
    assert nav[1].scale == pytest.approx(0.5 * math.sqrt(3) / 2, rel=1e-6)       # rows sqrt(3)/2 apart
    assert str(nav[0].units) == "um"


def test_the_pc_is_converted_from_tsl_to_bruker(tmp_path):
    _, up = _hex_folder(tmp_path)
    det = safe_loader.load_ebsd_safe(str(up), verbose=False).detector
    # By hand: the file states TSL (x*, y*, z*) = (0.5, 0.4, 0.6); on a square detector Bruker's
    # is (x*, 1 - y*, z*) = (0.5, 0.6, 0.6) (kikuchipy's _pc_tsl2bruker).
    assert np.allclose(np.asarray(det.pc).reshape(-1), [0.5, 0.6, 0.6], atol=1e-6)
    assert det.tilt == pytest.approx(5.0)               # camera elevation of the measurement
    assert det.sample_tilt == pytest.approx(70.0)
    assert det.azimuthal == 0.0


def _ramp_patterns(up, n, sx, sy):
    """Replace the body of ``up`` (header of 58 bytes, UP1) by ``n`` identical, non-constant
    patterns: the pixel at (row y, column x) holds (3 * y + x) % 256."""
    y, x = np.mgrid[0:sy, 0:sx]
    one = ((3 * y + x) % 256).astype(np.uint8)
    head = Path(up).read_bytes()[:58]
    Path(up).write_bytes(head + np.tile(one, (n, 1, 1)).tobytes())


def test_the_camera_angles_of_the_measurement_reach_the_detector(tmp_path):
    """Fixture angles that are not zero, so that dropping either mapping cannot stay green."""
    _, up = _hex_folder(tmp_path, elevation=8.5, azimuth=12.0)
    det = safe_loader.load_ebsd_safe(str(up), verbose=False).detector
    assert det.tilt == pytest.approx(8.5)               # camera elevation written into the file
    assert det.azimuthal == pytest.approx(12.0)         # camera azimuth written into the file


@pytest.mark.parametrize("cut", [5, 64], ids=["a-few-bytes", "one-whole-pattern"])
@pytest.mark.parametrize("hex_scan", [False, True], ids=["square", "hex"])
def test_a_pattern_file_shorter_than_its_grid_is_refused_with_a_code(tmp_path, cut, hex_scan):
    """The header (a square one states nx * ny) and the measurement agree, the body is short."""
    grid = "HexGrid" if hex_scan else "SqrGrid"
    make_apex_h5(tmp_path / "s.h5", [measurement_spec(grid=grid, n_rows=N_ROWS, n_cols=N_COLS)])
    up = make_up(tmp_path / "p.up1", n=N_ROWS * N_COLS, nx=N_COLS, ny=N_ROWS, hex_flag=int(hex_scan))
    full = up.read_bytes()
    assert len(full) == 58 + N_ROWS * N_COLS * 64        # 8 x 8 UP1 patterns, by hand
    up.write_bytes(full[:-cut])
    with pytest.raises(ApexError) as ei:
        safe_loader.load_ebsd_safe(str(up), verbose=False)
    assert ei.value.code == "apexPatternCountMismatch"


def test_the_pairing_of_a_short_pattern_file_says_how_many_patterns_it_holds(tmp_path):
    make_apex_h5(tmp_path / "s.h5", [measurement_spec(grid="SqrGrid", n_rows=N_ROWS, n_cols=N_COLS)])
    up = make_up(tmp_path / "p.up1", n=N_ROWS * N_COLS, nx=N_COLS, ny=N_ROWS, hex_flag=0)
    up.write_bytes(up.read_bytes()[:-64 - 5])             # one pattern and 5 bytes short: 13 whole ones
    with pytest.raises(ApexError, match=r"holds 13 patterns.*= 15"):
        safe_loader.load_ebsd_safe(str(up), verbose=False)


@pytest.mark.parametrize("cut, fits", [(5, 14), (64 + 5, 13)])
def test_the_loader_itself_refuses_a_body_shorter_than_the_grid_it_is_given(tmp_path, cut, fits):
    """The size check of ``load_edax_up_with_apex`` stands on its own: the pairing counts the file
    too, but the loader is also called with a measurement the pairing has not seen."""
    import edax_apex
    from edax_up1 import read_up1_header
    h5 = make_apex_h5(tmp_path / "s.h5", [measurement_spec(grid="SqrGrid", n_rows=N_ROWS, n_cols=N_COLS)])
    up = make_up(tmp_path / "p.up1", n=N_ROWS * N_COLS, nx=N_COLS, ny=N_ROWS, hex_flag=0)
    up.write_bytes(up.read_bytes()[:-cut])
    m = edax_apex.read_measurement(h5, GUID_A)
    with pytest.raises(ApexError, match=rf"room for {fits} patterns.*needs 15") as ei:
        edax_apex.load_edax_up_with_apex(str(up), read_up1_header(str(up)), m, "guid", False)
    assert ei.value.code == "apexPatternCountMismatch"


@pytest.mark.parametrize("sx, sy, z_factor, at, expected", [
    # (row, col) chosen by hand; expected = (3 * row + col) % 256
    (8, 8, 1.0, (6, 1), 19),            # square detector: only y* flips; 3*6+1
    (60, 80, 0.75, (78, 1), 235),       # 80 rows x 60 columns: z* rescaled by min(60, 80) / 80; 3*78+1
    (80, 60, 1.0, (58, 1), 175),        # 60 rows x 80 columns: min(80, 60) / 60 = 1; 3*58+1
])
def test_the_pc_conversion_follows_kikuchipys_documented_formula_on_a_non_square_detector(
        tmp_path, sx, sy, z_factor, at, expected):
    """Expected by hand from the table in the kikuchipy documentation of ``EBSDDetector`` (EDAX TSL
    to Bruker): x_B = x_T, y_B = 1 - y_T, z_B = min(N_x, N_y) / N_y * z_T, with N_x the number of
    columns and N_y the number of rows. The file states (0.5, 0.4, 0.6). "1 - y*" alone is the
    square-detector special case; both installed kikuchipy versions apply the z factor."""
    make_apex_h5(tmp_path / "s.h5", [measurement_spec(n_rows=N_ROWS, n_cols=N_COLS)])
    up = make_up(tmp_path / "p.up1", n=N_ROWS * N_COLS, nx=N_COLS, ny=N_ROWS, sx=sx, sy=sy)
    _ramp_patterns(up, N_ROWS * N_COLS, sx, sy)
    sig = safe_loader.load_ebsd_safe(str(up), verbose=False)
    det = sig.detector
    assert det.shape == (sy, sx)
    # the data, not only the detector: sy rows of sx columns, and a pixel where it must be
    assert sig.data.shape == (N_ROWS, N_COLS, sy, sx)
    assert sig.data.shape[-2:] == (sy, sx)
    data = np.asarray(sig.data)
    assert data[1, 2][at] == expected
    assert np.allclose(np.asarray(det.pc).reshape(-1), [0.5, 0.6, 0.6 * z_factor], atol=1e-6)


def test_a_version_4_header_with_a_guid_but_no_metadata_file_beside_it_loads_as_a_square_scan_without_a_pc(
        tmp_path):
    """The case of a pattern file copied away from its measurement: the header names a measurement,
    nobody in the folder can say anything about it. It is a plain square scan whose pattern centre
    is missing, not an error and not a guess."""
    up = make_up(tmp_path / "p.up1", version=4, guid_bytes=GUID_A_BYTES, hex_flag=0, n=15, nx=5, ny=3)
    sig = safe_loader.load_ebsd_safe(str(up), verbose=False)
    assert sig.metadata.get_item("Signal.pc_source") == "default"
    g = geometry_of_signal(sig)
    assert (g.kind, g.n_rows, g.n_cols) == ("square", 3, 5)
    assert sig.metadata.get_item("Signal.apex", None) is None


def test_the_signal_carries_what_the_route_needs(tmp_path):
    h5, up = _hex_folder(tmp_path)
    sig = safe_loader.load_ebsd_safe(str(up), verbose=False)
    md = sig.metadata
    assert md.get_item("Signal.pc_source") == "apex"
    assert md.get_item("Acquisition_instrument.SEM.beam_energy") == 20.0
    g = geometry_of_signal(sig)
    # an APEX measurement carries the odd-row offset as confirmed: the direction (+1/2) was read
    # from the patterns of EDAX hex scans
    assert (g.kind, g.n_rows, g.n_cols, g.shift_confirmed) == ("hex", N_ROWS, N_COLS, True)
    apex = plain(md.get_item("Signal.apex"))
    assert apex["measurement_id"] == GUID_A and apex["pairing"] == "guid"
    assert Path(apex["h5_path"]).name == "session.h5"


def test_a_square_measurement_loads_square(tmp_path):
    make_apex_h5(tmp_path / "s.h5", [measurement_spec(grid="SqrGrid", n_rows=2, n_cols=4, step_x=1.0)])
    up = make_up(tmp_path / "q.up1", n=8, nx=4, ny=2, hex_flag=0)
    g = geometry_of_signal(safe_loader.load_ebsd_safe(str(up), verbose=False))
    assert (g.kind, g.n_rows, g.n_cols, g.step_x, g.step_y) == ("square", 2, 4, 1.0, 1.0)


def test_the_wrong_measurement_is_refused(tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [
        measurement_spec(id=GUID_A, n_rows=N_ROWS, n_cols=N_COLS),
        measurement_spec(id=GUID_B, area="Area 2", n_rows=N_ROWS, n_cols=N_COLS)])
    up = make_up(tmp_path / "p.up1", n=N_ROWS * N_COLS, nx=N_COLS, guid_bytes=GUID_B_BYTES)
    with pytest.raises(ApexError) as ei:
        safe_loader.load_ebsd_safe(str(up), verbose=False,
                                   apex={"h5_path": str(h5), "measurement_id": GUID_A})
    assert ei.value.code == "apexGuidMismatch"


def test_a_pattern_count_that_is_not_the_grid_is_refused(tmp_path):
    make_apex_h5(tmp_path / "s.h5", [measurement_spec(n_rows=N_ROWS, n_cols=N_COLS)])
    up = make_up(tmp_path / "p.up1", n=N_ROWS * N_COLS - 1, nx=N_COLS)
    with pytest.raises(ApexError) as ei:
        safe_loader.load_ebsd_safe(str(up), verbose=False)
    assert ei.value.code == "apexPatternCountMismatch"


def test_a_header_whose_hex_flag_disagrees_with_the_measurement_is_refused(tmp_path):
    make_apex_h5(tmp_path / "s.h5", [measurement_spec(n_rows=N_ROWS, n_cols=N_COLS)])       # HexGrid
    up = make_up(tmp_path / "p.up1", n=N_ROWS * N_COLS, nx=N_COLS, ny=N_ROWS, hex_flag=0)
    with pytest.raises(ApexError) as ei:
        safe_loader.load_ebsd_safe(str(up), verbose=False)
    assert ei.value.code == "apexGridMismatch"


def test_a_result_table_that_does_not_fit_the_grid_is_refused(tmp_path):
    make_apex_h5(tmp_path / "s.h5", [measurement_spec(n_rows=N_ROWS, n_cols=N_COLS, n_table=N_ROWS * N_COLS - 1)])
    up = make_up(tmp_path / "p.up1", n=N_ROWS * N_COLS, nx=N_COLS)
    with pytest.raises(ApexError) as ei:
        safe_loader.load_ebsd_safe(str(up), verbose=False)
    assert ei.value.code == "apexGridMismatch"


def test_a_lattice_that_is_not_hexagonal_is_refused(tmp_path):
    make_apex_h5(tmp_path / "s.h5", [measurement_spec(n_rows=N_ROWS, n_cols=N_COLS, step_y=0.5)])
    up = make_up(tmp_path / "p.up1", n=N_ROWS * N_COLS, nx=N_COLS)
    with pytest.raises(ApexError) as ei:
        safe_loader.load_ebsd_safe(str(up), verbose=False)
    assert ei.value.code == "apexUnexpectedLattice"


def test_a_hex_pattern_file_without_metadata_is_still_refused(tmp_path):
    up = make_up(tmp_path / "p.up1", n=N_ROWS * N_COLS, nx=N_COLS)
    with pytest.raises(EdaxHexUpError) as ei:
        safe_loader.load_ebsd_safe(str(up), verbose=False)
    assert "metadata file" in str(ei.value)


def test_a_metadata_file_in_another_folder_is_not_found_by_itself_but_can_be_named(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    h5 = make_apex_h5(elsewhere / "s.h5", [measurement_spec(n_rows=N_ROWS, n_cols=N_COLS)])
    here = tmp_path / "here"
    here.mkdir()
    up = make_up(here / "p.up1", n=N_ROWS * N_COLS, nx=N_COLS)
    with pytest.raises(EdaxHexUpError):
        safe_loader.load_ebsd_safe(str(up), verbose=False)
    sig = safe_loader.load_ebsd_safe(
        str(up), verbose=False, apex={"h5_path": str(h5), "measurement_id": GUID_A})
    assert geometry_of_signal(sig).kind == "hex"


def test_a_square_up_file_without_any_metadata_loads_as_before(tmp_path):
    up = make_up(tmp_path / "p.up1", guid_bytes=None, version=3, hex_flag=0, n=15, nx=5, ny=3)
    sig = safe_loader.load_ebsd_safe(str(up), verbose=False)
    assert sig.axes_manager.navigation_shape == (5, 3)
    assert sig.metadata.get_item("Signal.pc_source") == "default"


def test_the_osc_detector_keeps_the_tilt_pixel_size_and_binning_the_reader_set(tmp_path, monkeypatch):
    """The detector was rebuilt from the shape and the sample tilt alone: whatever tilt, pixel
    size and binning the reader had put on the signal's detector was dropped."""
    import kikuchipy as kp
    from kikuchipy.detectors import EBSDDetector

    sig = kp.data.nickel_ebsd_small()
    sig.detector = EBSDDetector(shape=sig.detector.shape, pc=(0.5, 0.5, 0.5), sample_tilt=70.0,
                                tilt=3.0, px_size=70.0, binning=2)

    class _Kp:
        @staticmethod
        def load(path, **kwargs):
            return sig

    monkeypatch.setattr(safe_loader, "_kp", lambda: _Kp)
    up = make_up(tmp_path / "s.up1", guid_bytes=None, version=3, hex_flag=0, n=9, nx=3, ny=3)
    make_osc(tmp_path / "s.osc", ncols=3, nrows=3, xstep=0.5, ystep=0.5, pc=(0.55, 0.40, 0.70))
    out = safe_loader.load_ebsd_safe(str(up), verbose=False)
    det = out.detector
    assert (det.tilt, det.px_size, det.binning) == (3.0, 70.0, 2)
    # TSL (0.55, 0.40, 0.70) -> Bruker (0.55, 1 - 0.40, 0.70) on a square detector, by hand
    assert np.allclose(np.asarray(det.pc).reshape(-1), [0.55, 0.60, 0.70], atol=1e-6)
    assert out.metadata.get_item("Signal.pc_source") == "osc"


def test_a_folder_with_both_an_osc_and_an_apex_file_loads_through_apex(tmp_path):
    """Controller ruling: the APEX measurement is looked for first. A folder that holds an .osc
    sidecar as well as the metadata .h5 of the same pattern file loads through the .h5."""
    _, up = _hex_folder(tmp_path)
    make_osc(tmp_path / "scan.osc", ncols=N_COLS, nrows=N_ROWS, xstep=0.5, ystep=0.5,
             pc=(0.55, 0.40, 0.70))
    sig = safe_loader.load_ebsd_safe(str(up), verbose=False)
    assert sig.metadata.get_item("Signal.pc_source") == "apex"
    apex = plain(sig.metadata.get_item("Signal.apex"))
    assert apex["pairing"] == "guid"                  # the "how" of the pairing
    assert geometry_of_signal(sig).kind == "hex"
    # the pattern centre is the h5's (0.5, 0.4, 0.6) -> Bruker (0.5, 0.6, 0.6), not the .osc's
    assert np.allclose(np.asarray(sig.detector.pc).reshape(-1), [0.5, 0.6, 0.6], atol=1e-6)


def test_a_complete_square_file_whose_header_states_another_grid_is_paired_by_its_body(tmp_path):
    """The metadata file is the authority. The header says 15 x 15 (225 patterns); the body holds
    exactly the 15 patterns of the 3 x 5 measurement. Counting the header refused this file with
    'holds 225 patterns'."""
    make_apex_h5(tmp_path / "s.h5", [measurement_spec(grid="SqrGrid", n_rows=N_ROWS, n_cols=N_COLS,
                                                      step_x=1.0)])
    up = make_up(tmp_path / "p.up1", n=N_ROWS * N_COLS, nx=15, ny=15, hex_flag=0)
    assert struct.unpack_from("<2I", up.read_bytes(), 17) == (15, 15)      # what the header claims
    sig = safe_loader.load_ebsd_safe(str(up), verbose=False)
    assert sig.axes_manager.navigation_shape == (N_COLS, N_ROWS)
    assert sig.metadata.get_item("Signal.pc_source") == "apex"
    data = np.asarray(sig.data)
    assert data[2, 4, 0, 0] == (7 * (2 * N_COLS + 4)) % 256                # pattern 14, by hand: 98


def test_a_body_that_really_holds_fewer_patterns_is_refused_with_the_true_count(tmp_path):
    make_apex_h5(tmp_path / "s.h5", [measurement_spec(grid="SqrGrid", n_rows=N_ROWS, n_cols=N_COLS)])
    up = make_up(tmp_path / "p.up1", n=14, nx=15, ny=15, hex_flag=0)
    with pytest.raises(ApexError, match=r"holds 14 patterns \(its header states 15 x 15\).*= 15") as ei:
        safe_loader.load_ebsd_safe(str(up), verbose=False)
    assert ei.value.code == "apexPatternCountMismatch"


def test_an_unreadable_metadata_file_is_named_without_its_folder(tmp_path):
    folder = tmp_path / "private_folder_name"
    folder.mkdir()
    not_apex = folder / "plain.h5"
    not_apex.write_bytes(b"not hdf5")
    up = make_up(tmp_path / "p.up1", n=N_ROWS * N_COLS, nx=N_COLS)
    with pytest.raises(ApexError) as ei:
        safe_loader.load_ebsd_safe(str(up), verbose=False,
                                   apex={"h5_path": str(not_apex), "measurement_id": GUID_A})
    assert ei.value.code == "apexUnreadable"
    msg = str(ei.value)
    assert "plain.h5" in msg and "could not be read as an APEX metadata file" in msg
    assert "private_folder_name" not in msg and str(tmp_path) not in msg


@pytest.mark.parametrize("chunk_bytes, chunks", [
    # 8 x 8 UP1 patterns are 64 bytes; a row of 5 is 320 bytes (by hand).
    (130, ((1, 1, 1), (2, 2, 1), (8,), (8,))),     # a row is bigger than 130: 130 // 64 = 2 columns
    (700, ((2, 1), (5,), (8,), (8,))),             # 700 // 320 = 2 rows; whole rows as before
    (320, ((1, 1, 1), (5,), (8,), (8,))),          # exactly one row: one row, all 5 columns
])
def test_a_lazy_scan_splits_a_row_that_is_bigger_than_a_chunk(tmp_path, monkeypatch, chunk_bytes, chunks):
    import edax_apex
    _, up = _hex_folder(tmp_path)
    monkeypatch.setattr(edax_apex, "_UP_CHUNK_BYTES", chunk_bytes)
    eager = np.asarray(safe_loader.load_ebsd_safe(str(up), verbose=False).data)
    monkeypatch.setenv("KIKUCHIPY_LAZY_THRESHOLD_BYTES", "0")
    lazy = safe_loader.load_ebsd_safe(str(up), verbose=False).data
    assert lazy.chunks == chunks
    assert np.array_equal(np.asarray(lazy), eager)


def _lazy_load(up, monkeypatch):
    monkeypatch.setenv("KIKUCHIPY_LAZY_THRESHOLD_BYTES", "0")
    return safe_loader.load_ebsd_safe(str(up), verbose=False)


def test_two_loads_of_one_lazy_file_get_two_different_dask_names(tmp_path, monkeypatch):
    # dask names an array after a hash of its content unless it is told not to, and that hash
    # would have to read the file. A fresh name per load costs nothing.
    _, up = _hex_folder(tmp_path)
    first = _lazy_load(up, monkeypatch).data
    second = _lazy_load(up, monkeypatch).data
    assert first.name != second.name
    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_loading_a_lazy_file_does_not_hash_the_memory_map(tmp_path, monkeypatch):
    import dask.array.core as dask_core
    seen = []
    real = dask_core.tokenize

    def watching(*args, **kwargs):
        seen.extend(a for a in args if isinstance(a, np.ndarray))
        return real(*args, **kwargs)

    monkeypatch.setattr(dask_core, "tokenize", watching)
    _, up = _hex_folder(tmp_path)
    sig = _lazy_load(up, monkeypatch)
    assert bool(sig._lazy)
    assert seen == []       # nothing of the file's pattern array went through the content hash


def test_a_lazy_load_never_copies_the_memory_map(tmp_path, monkeypatch):
    # dask.array.from_array calls x.copy() on anything array-like, whatever name it is given: a
    # 3 GB measurement sat in memory after a "lazy" load, once more after every switch-file.
    class NoCopy(np.memmap):
        def copy(self, *args, **kwargs):
            raise AssertionError("the whole memory map was copied into memory")

    _, up = _hex_folder(tmp_path)
    monkeypatch.setattr(np, "memmap", NoCopy)
    sig = _lazy_load(up, monkeypatch)
    assert bool(sig._lazy)
    assert sig.data.compute().shape == sig.data.shape


def test_the_lazy_patterns_equal_the_eager_ones_pattern_by_pattern_and_in_slices(tmp_path, monkeypatch):
    _, up = _hex_folder(tmp_path)
    eager = np.asarray(safe_loader.load_ebsd_safe(str(up), verbose=False).data)
    lazy = _lazy_load(up, monkeypatch).data
    assert lazy.dtype == eager.dtype and lazy.shape == eager.shape
    assert np.array_equal(lazy[1, 2].compute(), eager[1, 2])
    assert np.array_equal(lazy[:, 1:3, ::2, 3].compute(), eager[:, 1:3, ::2, 3])
