"""The positions of a hexagonal scan against OIM's own numbers.

OIM writes ``X Position`` and ``Y Position`` for every point of a hexagonal scan it has measured. This
test opens three such files and compares them with ``ScanGeometry.hex(...).positions()``, the grid
Orienta writes into a HexGrid ``.ang``. Nothing here is derived from the code under test: the
expectation is what OIM wrote.

What OIM writes (read from the files, not assumed):

* ``EBSD/Header``: ``nRows``, ``nColumns``, ``Step X`` and ``Step Y`` (= ``Step X`` * sqrt(3) / 2).
* ``EBSD/Data``: ``X Position`` and ``Y Position`` as flat float32 arrays of ``nRows * nColumns``
  entries, read here as ``(nRows, nColumns)``. The odd rows are half a step to the right of the even
  ones; in every odd row the last column is not a point of the scan and holds the sentinel -1111111
  (a number far below any position). Our rectangle keeps that cell, so it has ``nRows // 2`` points
  more than OIM measured.

The files are read through ``ORIENTA_HEX_REFERENCE``, a folder with ``prim_phase.h5``, ``Scan3.h5`` and
``scan2.h5``. The test skips by naming that variable when it is not set or a file is not there.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from backend.api.services.scan_geometry import ScanGeometry

ENV = "ORIENTA_HEX_REFERENCE"
FILES = ["prim_phase.h5", "Scan3.h5", "scan2.h5"]
#: OIM's marker for a cell without a point; every real position is far above it.
SENTINEL_LIMIT = -1e5


def _reference_file(name: str) -> Path:
    folder = os.environ.get(ENV)
    if not folder:
        pytest.skip(f"{ENV} is not set: OIM's hexagonal reference files are not available")
    path = Path(folder) / name
    if not path.is_file():
        pytest.skip(f"{ENV} is set, but {name} is not in that folder")
    return path


def _read_scan(path: Path):
    """``(n_rows, n_cols, step_x, x, y)`` of the only scan of an OIM ``.h5``; x and y are float32
    ``(n_rows, n_cols)`` arrays as OIM wrote them."""
    import h5py
    with h5py.File(path, "r") as f:
        scans = [k for k in f if isinstance(f[k], h5py.Group) and "EBSD" in f[k]]
        assert len(scans) == 1, f"{path.name}: expected one scan group, found {scans}"
        ebsd = f[scans[0]]["EBSD"]
        header = ebsd["Header"]
        assert header["Grid Type"][0] in (b"HexGrid", "HexGrid")
        n_rows, n_cols = int(header["nRows"][0]), int(header["nColumns"][0])
        step_x = float(header["Step X"][0])
        data = ebsd["Data"]
        x, y = data["X Position"][()], data["Y Position"][()]
    assert x.size == y.size == n_rows * n_cols
    return n_rows, n_cols, step_x, x.reshape(n_rows, n_cols), y.reshape(n_rows, n_cols)


@pytest.mark.parametrize("name", FILES)
def test_the_hex_positions_are_those_oim_wrote(name):
    n_rows, n_cols, step_x, x, y = _read_scan(_reference_file(name))
    valid = x > SENTINEL_LIMIT                      # the sentinel, not measure_hex_layout
    assert np.array_equal(valid, y > SENTINEL_LIMIT)

    # which cells are not points: exactly the last column of the odd rows
    expected_invalid = np.zeros((n_rows, n_cols), dtype=bool)
    expected_invalid[1::2, n_cols - 1] = True
    assert np.array_equal(~valid, expected_invalid), (
        f"{name}: the cells without a point are not 'last column of the odd rows'")
    assert int((~valid).sum()) == n_rows // 2       # the points our rectangle has more than OIM

    # ScanGeometry.hex derives step_y = step_x * sqrt(3) / 2; the files are float32 (rtol 1e-6)
    geometry = ScanGeometry.hex(n_rows, n_cols, step_x)
    gx, gy = geometry.positions()
    np.testing.assert_allclose(gx[valid], x[valid], rtol=1e-6, err_msg=f"{name}: x positions")
    np.testing.assert_allclose(gy[valid], y[valid], rtol=1e-6, err_msg=f"{name}: y positions")


def test_a_missing_reference_skips_by_naming_the_variable(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    with pytest.raises(pytest.skip.Exception) as skipped:
        _reference_file(FILES[0])
    assert ENV in str(skipped.value)
