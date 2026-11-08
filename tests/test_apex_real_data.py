"""EDAX APEX measurements of real files: loaded grid, steps, pattern centre and the pattern/table order.

The folder comes from ``ORIENTA_APEX_DATA`` (EDAX APEX measurements that are not part of the
repository; nothing of it is copied into the tree). Without the variable these tests skip and say so. Nothing in this file names a
file, a sample or a value of those data: every expectation is read from the files themselves with
h5py and numpy, not with the code under test.

What the loader relies on, and what is therefore checked against the raw bytes:

1. the grid, the two steps, the camera elevation and the pattern centre are the ones the ``.h5``
   states. The pattern centre is stored in EDAX/TSL convention and arrives in Bruker's by the
   formula in kikuchipy's documentation, ``(x*, 1 - y*, min(Nx, Ny) / Ny * z*)``. "1 - y*" alone is
   the case of a SQUARE detector: the z factor needs the detector shape, which this test reads from
   the pattern file's own header (``Nx`` columns, ``Ny`` rows), so the claim holds for any shape;
2. the pattern at (row, col) is pattern number ``row * n_cols + col`` of the file, byte for byte;
3. pattern number ``i`` is the pattern at scan position ``i`` of a row-major scan (table row ``i``):
   neighbouring patterns look alike, and the patterns across a row boundary (the last of a row and
   the first of the next) do not, which is what row-major means and what a serpentine scan (every
   second row reversed) would not show. The check does not use the table's image quality: the
   similarity of background-corrected, downsampled patterns is compared for both orders.

``order_evidence`` and ``order_verdict`` are themselves tested on synthetic files below (one in
row-major order, one in serpentine order, one without any spatial structure), and their thresholds
come from those cases only. A measurement for which neither order is clearly better is reported as
"undecided" (the test skips it with that reason): that is a question about the data, not a failure.
The old image-quality correlation is printed for information and asserts nothing.
"""
from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import safe_loader  # noqa: E402
import edax_apex  # noqa: E402
from backend.api.services.scan_geometry import geometry_of_signal  # noqa: E402
from tests.apex_helpers import make_up  # noqa: E402

_DTYPE = {".up1": np.dtype("u1"), ".up2": np.dtype("<u2")}
_WINDOW_ROWS = 16                    # rows of the scan whose patterns are read for the order check
_TARGET_PX = 24                      # patterns are block-averaged to about this many pixels per side
_SIGNAL_MIN = 0.05                   # neighbours must be this much more alike than distant patterns
_FRACTION = 0.5                      # share of that difference the order-specific pairs must show


def _folder() -> Path:
    root = os.environ.get("ORIENTA_APEX_DATA")
    if not root:
        pytest.skip("ORIENTA_APEX_DATA is not set (a folder with EDAX APEX measurements)")
    folder = Path(root)
    if not folder.is_dir():
        pytest.skip(f"{folder} is not a folder")
    return folder


# ------------------------------------------------------------- raw readers (no code under test)

def _scalar(group, name):
    value = group[name][()]
    value = value[0] if getattr(value, "ndim", 0) else value
    return value.decode() if isinstance(value, bytes) else value


def _raw_measurements(h5_path):
    """(group path, GUID, dict of the raw facts) of every 'OIM Map' group, read with h5py only."""
    out = []
    with h5py.File(h5_path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Group) and "Sample" in obj and "EBSD" in obj and "Id" in obj.attrs:
                guid = obj.attrs["Id"]
                guid = guid[0] if getattr(guid, "ndim", 0) else guid
                guid = guid.decode() if isinstance(guid, bytes) else str(guid)
                s = obj["Sample"]
                pc = obj["EBSD/ANG/HEADER/Pattern Center Calibration"]
                table = obj["EBSD/ANG/DATA/DATA"]
                out.append((name, guid.strip("{}").lower(), dict(
                    grid=_scalar(s, "Grid Type"),
                    n_cols=int(_scalar(s, "Number Of Columns")), n_rows=int(_scalar(s, "Number Of Rows")),
                    step_x=float(_scalar(s, "Step X")), step_y=float(_scalar(s, "Step Y")),
                    elevation=float(_scalar(obj["Camera"], "Elevation Angle")),
                    pc=tuple(float(_scalar(pc, k)) for k in ("X-Star", "Y-Star", "Z-Star")),
                    iq=np.asarray(table["IQ"], dtype=np.float64))))
        f.visititems(visit)
    return out


def _up_header(path):
    with open(path, "rb") as fh:
        version, sx, sy, offset = struct.unpack("<4I", fh.read(16))
    return version, sx, sy, offset


def _raw_pattern(path, index):
    version, sx, sy, offset = _up_header(path)
    dtype = _DTYPE[Path(path).suffix.lower()]
    with open(path, "rb") as fh:
        fh.seek(offset + index * sx * sy * dtype.itemsize)
        return np.fromfile(fh, dtype=dtype, count=sx * sy).reshape(sy, sx)


def _raw_patterns(path, count):
    """The first ``count`` patterns of the file, in file order, as float64 (count, sy, sx)."""
    version, sx, sy, offset = _up_header(path)
    dtype = _DTYPE[Path(path).suffix.lower()]
    with open(path, "rb") as fh:
        fh.seek(offset)
        data = np.fromfile(fh, dtype=dtype, count=count * sx * sy)
    return data.reshape(count, sy, sx).astype(np.float64)


def _iq_correlations(path, iq, n_rows, n_cols):
    """INFORMATION ONLY (nothing asserts it): Pearson correlation of the pattern contrast (standard
    deviation) with the table's image quality over the first rows, for table order and for a
    serpentine order. It turned out not to separate the orders on real files (the image quality
    comes from the Hough transform, the raw standard deviation from the background)."""
    rows = min(_WINDOW_ROWS, n_rows)
    idx = np.arange(rows * n_cols).reshape(rows, n_cols)
    contrast = _raw_patterns(path, rows * n_cols).reshape(rows * n_cols, -1).std(axis=1)
    serp = idx.copy()
    serp[1::2] = serp[1::2][:, ::-1]
    r = lambda a, b: float(np.corrcoef(a, b)[0, 1])          # noqa: E731
    return r(contrast, iq[idx.ravel()]), r(contrast, iq[serp.ravel()])


def _prepared(patterns):
    """Block-averaged, background-corrected, zero-mean unit-norm patterns, shape (n, d).

    Each pattern loses its own slowly varying background (a heavy Gaussian blur of itself), then
    the window loses its common mean pattern, so what is left is what changes from point to point."""
    from scipy.ndimage import gaussian_filter
    n, sy, sx = patterns.shape
    f = max(1, min(sy, sx) // _TARGET_PX)
    small = patterns[:, : sy // f * f, : sx // f * f]
    small = small.reshape(n, sy // f, f, sx // f, f).mean(axis=(2, 4))
    small = small - np.stack([gaussian_filter(p, sigma=max(small.shape[1:]) / 6.0) for p in small])
    small = small - small.mean(axis=0)
    flat = small.reshape(n, -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    return flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)


def order_evidence(path, n_rows, n_cols, window_rows=_WINDOW_ROWS):
    """Mean similarity (normalised cross-correlation) of selected pairs of patterns of the first
    rows of the file, taking the file as ``rows x n_cols`` in file order:

    within    the file neighbours inside a row (the same in both orders),
    boundary  the last pattern of a row and the first of the next (neighbours only in serpentine),
    vert_rm   the pattern and the one a row below it (neighbours in row-major),
    vert_sp   the pattern and the mirrored one a row below it (neighbours in serpentine),
    far       patterns half the window apart (what unrelated patterns look like)."""
    rows = min(window_rows, n_rows)
    x = _prepared(_raw_patterns(path, rows * n_cols)).reshape(rows, n_cols, -1)
    ncc = lambda a, b: float(np.mean(np.sum(a * b, axis=-1)))          # noqa: E731
    half = rows // 2
    return dict(
        within=ncc(x[:, :-1], x[:, 1:]),
        boundary=ncc(x[:-1, -1], x[1:, 0]),
        vert_rm=ncc(x[:-1], x[1:]),
        vert_sp=ncc(x[:-1], x[1:, ::-1]),
        far=ncc(x[:half], np.roll(x[half:2 * half], n_cols // 2, axis=1)),
    )


def order_verdict(ev):
    """``"row-major"``, ``"serpentine"`` or ``"undecided"`` from ``order_evidence``.

    Neighbours must be clearly more alike than distant patterns at all (``within - far``); then, as a
    share of that difference, row-major needs the row boundary to look like a jump and the row below
    to be the neighbour, serpentine needs the opposite. Anything in between is undecided."""
    signal = ev["within"] - ev["far"]
    if signal < _SIGNAL_MIN:
        return "undecided"
    jump = (ev["within"] - ev["boundary"]) / signal           # 1: the boundary is a jump, 0: it is not
    below = (ev["vert_rm"] - ev["vert_sp"]) / signal          # 1: row-major rows line up, -1: mirrored ones do
    if jump > _FRACTION and below > _FRACTION:
        return "row-major"
    if jump < _FRACTION and below < -_FRACTION:
        return "serpentine"
    return "undecided"


def _pairs(folder):
    """(metadata file, measurement group, GUID, raw facts, pattern file) for every measurement of the
    folder that has a pattern file naming it by GUID in its own folder."""
    found = []
    for h5_path in sorted(folder.rglob("*.h5")):
        try:
            if not edax_apex.is_apex_file(h5_path):
                continue
            measurements = _raw_measurements(h5_path)
        except OSError:
            continue
        for group, guid, facts in measurements:
            for up in sorted(h5_path.parent.iterdir()):
                if up.suffix.lower() in _DTYPE and edax_apex.up_guid(up) == guid:
                    found.append((h5_path, group, guid, facts, up))
    return found


# ------------------------------------------------------------------------- the real files

def test_a_measurement_loads_with_the_grid_steps_and_pattern_centre_the_file_states():
    pairs = _pairs(_folder())
    if not pairs:
        pytest.skip("no APEX measurement with a pattern file naming it was found in the folder")
    for h5_path, _group, guid, raw, up in pairs:
        sig = safe_loader.load_ebsd_safe(str(up), verbose=False)
        g = geometry_of_signal(sig)
        assert g.kind == ("hex" if raw["grid"] == "HexGrid" else "square")
        assert (g.n_rows, g.n_cols) == (raw["n_rows"], raw["n_cols"])
        assert sig.axes_manager.navigation_shape == (raw["n_cols"], raw["n_rows"])
        assert (g.step_x, g.step_y) == (pytest.approx(raw["step_x"]), pytest.approx(raw["step_y"]))
        assert sig.metadata.get_item("Signal.pc_source") == "apex"
        x, y, z = raw["pc"]
        _version, n_cols_px, n_rows_px, _offset = _up_header(up)           # Nx, Ny of the pattern file
        z_bruker = min(n_cols_px, n_rows_px) / n_rows_px * z               # TSL -> Bruker, by the formula
        assert np.allclose(sig.detector.pc.reshape(-1)[:3], (x, 1.0 - y, z_bruker), atol=1e-6)
        assert sig.detector.tilt == pytest.approx(raw["elevation"])


def test_the_pattern_at_a_point_is_the_pattern_of_that_index_in_the_file():
    pairs = _pairs(_folder())
    if not pairs:
        pytest.skip("no APEX measurement with a pattern file naming it was found in the folder")
    h5_path, _group, guid, raw, up = pairs[0]
    sig = safe_loader.load_ebsd_safe(str(up), verbose=False)
    for row, col in ((0, 0), (0, raw["n_cols"] - 1), (1, 0), (raw["n_rows"] // 2, raw["n_cols"] // 3),
                     (raw["n_rows"] - 1, raw["n_cols"] - 1)):
        expected = _raw_pattern(up, row * raw["n_cols"] + col)
        assert np.array_equal(np.asarray(sig.data[row, col]), expected), (row, col)


def test_the_patterns_of_a_file_are_in_row_major_scan_order():
    pairs = _pairs(_folder())
    if not pairs:
        pytest.skip("no APEX measurement with a pattern file naming it was found in the folder")
    undecided = []
    for number, (h5_path, _group, guid, raw, up) in enumerate(pairs):
        ev = order_evidence(up, raw["n_rows"], raw["n_cols"])
        verdict = order_verdict(ev)
        iq_file, iq_serpentine = _iq_correlations(up, raw["iq"], raw["n_rows"], raw["n_cols"])
        print(f"measurement {number}: {verdict}; " + ", ".join(f"{k} {v:.3f}" for k, v in ev.items())
              + f"; (information only) contrast vs IQ: file order {iq_file:.3f}, serpentine {iq_serpentine:.3f}")
        if verdict == "undecided":
            undecided.append(number)
            continue
        # The loader reads the file as a row-major scan. A serpentine verdict means that assumption is
        # wrong for this file: report it, do not lower the thresholds (they come from the synthetic
        # files below, not from these).
        assert verdict == "row-major", f"measurement {number} reads as {verdict}: {ev}"
    if undecided:
        pytest.skip(f"order undecided for measurement(s) {undecided}: neither row-major nor serpentine is "
                    "clearly better (an open question about the data, not a failure); the others passed")


# ----------------------------------------- the check itself, on synthetic files with a known answer

_SYN_ROWS, _SYN_COLS, _SYN_PX, _SYN_BANDS = 24, 20, 32, 12


def _synthetic_up(tmp_path, order, noise, structured=True, hexagonal=True, seed=3):
    """A 24 x 20 scan of 32 x 32 pixel UP2 patterns whose neighbours look alike.

    Every pattern is a common background (a strong gradient plus a bright centre, which every real
    pattern also has), plus six fixed band-like images whose weights vary smoothly with the scan
    position (a few cells per period), plus noise of the given size. ``hexagonal`` shifts the odd
    rows by half a step and spaces the rows by 0.866, as a hexagonal scan does. ``order`` is
    ``"row-major"`` (file index = row * n_cols + col) or ``"serpentine"`` (every second row of the
    file runs backwards). ``structured=False`` leaves out the position dependence entirely."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:_SYN_PX, 0:_SYN_PX] / _SYN_PX
    background = 3000.0 + 2000.0 * xx + 1500.0 * yy + 2500.0 * np.exp(-((xx - .5) ** 2 + (yy - .5) ** 2) / .1)
    from scipy.ndimage import gaussian_filter
    bands = np.stack([gaussian_filter(rng.normal(size=(_SYN_PX, _SYN_PX)), 2.0) for _ in range(_SYN_BANDS)])
    bands *= 600.0 / bands.std()
    angle = rng.uniform(0, 2 * np.pi, _SYN_BANDS)
    k = rng.uniform(0.05, 0.18, _SYN_BANDS)[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    phase = rng.uniform(0, 2 * np.pi, _SYN_BANDS)
    scan = np.arange(_SYN_ROWS * _SYN_COLS).reshape(_SYN_ROWS, _SYN_COLS)
    if order == "serpentine":
        scan[1::2] = scan[1::2][:, ::-1]               # file position -> scan position, row by row
    patterns = np.empty((_SYN_ROWS * _SYN_COLS, _SYN_PX, _SYN_PX))
    for row in range(_SYN_ROWS):
        for file_col in range(_SYN_COLS):
            col = scan[row, file_col] % _SYN_COLS      # the scan column this file position holds
            x = col + (0.5 * (row % 2) if hexagonal else 0.0)
            y = row * (0.866 if hexagonal else 1.0)
            weights = (np.cos(2 * np.pi * (k[:, 0] * x + k[:, 1] * y) + phase) if structured
                       else np.zeros(_SYN_BANDS))
            patterns[row * _SYN_COLS + file_col] = (background + np.tensordot(weights, bands, axes=1)
                                                    + rng.normal(0.0, noise, (_SYN_PX, _SYN_PX)))
    up = make_up(tmp_path / "p.up2", n=_SYN_ROWS * _SYN_COLS, nx=_SYN_COLS, ny=_SYN_ROWS,
                 sx=_SYN_PX, sy=_SYN_PX, dtype_bytes=2)
    _version, _sx, _sy, offset = _up_header(up)
    with open(up, "r+b") as fh:
        fh.seek(offset)
        fh.write(patterns.clip(0, 65535).astype("<u2").tobytes())
    return up


def test_the_order_check_passes_a_file_in_row_major_order(tmp_path):
    ev = order_evidence(_synthetic_up(tmp_path, "row-major", noise=300.0), _SYN_ROWS, _SYN_COLS)
    assert order_verdict(ev) == "row-major", ev


def test_the_order_check_sees_a_file_written_in_serpentine_order(tmp_path):
    ev = order_evidence(_synthetic_up(tmp_path, "serpentine", noise=300.0), _SYN_ROWS, _SYN_COLS)
    assert order_verdict(ev) == "serpentine", ev


@pytest.mark.parametrize("seed", range(3, 11))
def test_the_order_check_says_undecided_when_the_noise_drowns_the_structure(tmp_path, seed):
    # Several seeds: for some of them the two ratios alone would look like an order, and only the
    # requirement that neighbours be clearly more alike than distant patterns keeps them undecided.
    ev = order_evidence(_synthetic_up(tmp_path, "row-major", noise=8000.0, seed=seed), _SYN_ROWS, _SYN_COLS)
    assert order_verdict(ev) == "undecided", ev


@pytest.mark.parametrize("seed", range(3, 11))
def test_the_order_check_says_undecided_for_patterns_without_any_spatial_structure(tmp_path, seed):
    ev = order_evidence(_synthetic_up(tmp_path, "row-major", noise=300.0, structured=False, seed=seed),
                        _SYN_ROWS, _SYN_COLS)
    assert order_verdict(ev) == "undecided", ev
