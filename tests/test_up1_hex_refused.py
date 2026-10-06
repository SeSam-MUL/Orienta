"""EDAX UP1/UP2 scans on a hexagonal grid fail loudly instead of loading wrong.

A hexagonal scan stores its patterns row by row with alternate rows shifted by
half a step. Orienta resamples hexagonal EDAX scans onto a square grid for the
H5 format only. For UP1/UP2 nothing resampled or refused them: a version-3 file
with the hex flag went to kikuchipy, which returns a single navigation row (with
a warning), and a version-1 file beside a hexagonal `.osc` was reshaped as a
square map whenever its pattern count happened to be a perfect square. Both give
a map whose pixels are at the wrong positions.

The hex flag is in the v3 header; a v1 header has no grid at all, so the `.osc`
sidecar decides (its point records carry x/y in micrometres: on a hexagonal grid
the rows start at different x).
"""
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edax_up1 import (  # noqa: E402
    EdaxHexUpError,
    read_osc_metadata,
    resolve_up1_geometry,
)

OSC_MAGIC = bytes([0xB9, 0x0B, 0xEF, 0xFF, 0x02, 0x00, 0x00, 0x00])
SQUARE = Path(__file__).resolve().parents[1] / "Test_data" / "new test" / "Scan5.up1"


def _up1_v1(path, sx, sy, npat):
    with open(path, "wb") as f:
        f.write(struct.pack("<4I", 1, sx, sy, 16))
        f.write(b"\x00" * (sx * sy * npat))


def _up1_v3(path, sx, sy, nx, ny, is_hex):
    with open(path, "wb") as f:
        f.write(struct.pack("<4I", 3, sx, sy, 42))
        f.write(b"\x00")
        f.write(struct.pack("<2I", nx, ny))
        f.write(struct.pack("<B", is_hex))
        f.write(struct.pack("<2d", 1.0, 1.0))
        f.write(b"\x00" * (sx * sy * nx * ny))


def _osc(path, ncols, nrows, xstep, ystep, hex_grid):
    """An .osc with real point records: 14 float32 per point, x and y in um.

    On a hexagonal grid odd rows start half a step further right and the rows
    are sqrt(3)/2 of a step apart.
    """
    buf = bytearray(64)
    struct.pack_into("<3I", buf, 16, ncols - 1, nrows - 1, ncols * nrows)
    buf += b"\x00" * 3                           # off the 4-byte alignment, as real files
    buf += OSC_MAGIC + struct.pack("<2I", 0, 0) + struct.pack("<2f", xstep, ystep)
    rec = np.zeros((nrows * ncols, 14), dtype="<f4")
    for r in range(nrows):
        for c in range(ncols):
            rec[r * ncols + c, 3] = c * xstep + (xstep / 2 if hex_grid and r % 2 else 0.0)
            rec[r * ncols + c, 4] = r * ystep
    buf += rec.tobytes()
    Path(path).write_bytes(bytes(buf))


# ------------------------------------------------------------------ refused

def test_a_v3_file_with_the_hex_flag_is_refused(tmp_path):
    up = tmp_path / "hex.up1"
    _up1_v3(up, 4, 4, 5, 4, is_hex=1)
    with pytest.raises(EdaxHexUpError) as ei:
        resolve_up1_geometry(str(up))
    assert ei.value.code == "edaxHexUpUnsupported"
    msg = str(ei.value)
    assert "hex.up1" in msg
    assert "hexagonal grid" in msg
    assert "EDAX H5" in msg and "export the scan as H5 from OIM" in msg


def test_a_v1_file_beside_a_hexagonal_osc_is_refused_even_if_the_count_is_square(tmp_path):
    up = tmp_path / "scan.up1"
    _up1_v1(up, 4, 4, 36)                         # 36 = 6*6: would be read as a square map
    _osc(tmp_path / "scan.osc", 6, 6, 1.0, 0.866, hex_grid=True)
    with pytest.raises(EdaxHexUpError):
        resolve_up1_geometry(str(up))


def test_a_v1_file_beside_a_hexagonal_osc_is_refused_when_the_count_is_not_square(tmp_path):
    up = tmp_path / "scan.up1"
    _up1_v1(up, 4, 4, 40)                         # 8 x 5
    _osc(tmp_path / "scan.osc", 8, 5, 1.0, 0.866, hex_grid=True)
    with pytest.raises(EdaxHexUpError):
        resolve_up1_geometry(str(up))


def test_the_loader_entry_raises_it_for_both_kinds(tmp_path):
    from safe_loader import load_ebsd_safe

    up3 = tmp_path / "a.up2"
    _up1_v3(up3, 4, 4, 5, 4, is_hex=1)
    up1 = tmp_path / "b.up1"
    _up1_v1(up1, 4, 4, 36)
    _osc(tmp_path / "b.osc", 6, 6, 1.0, 0.866, hex_grid=True)
    for path in (up3, up1):
        with pytest.raises(EdaxHexUpError):
            load_ebsd_safe(str(path), verbose=False)


def test_the_load_route_answers_400_with_the_code_and_the_prose(tmp_path, monkeypatch):
    monkeypatch.setenv("ORIENTA_ALLOWED_HOSTS", "testserver")
    from fastapi.testclient import TestClient

    from backend.api.main import app
    from backend.api.problem import CODE_HEADER

    up = tmp_path / "hex.up1"
    _up1_v3(up, 4, 4, 5, 4, is_hex=1)
    client = TestClient(app, raise_server_exceptions=False)
    r = client.post("/api/ebsd/load", json={"path": str(up)})
    assert r.status_code == 400
    assert r.headers.get(CODE_HEADER.lower()) == "edaxHexUpUnsupported"
    assert "hexagonal grid" in r.json()["detail"]


def test_only_the_hexagonal_refusal_sets_the_code_header(tmp_path, monkeypatch):
    """Any exception with a `code` attribute used to be turned into a coded
    response; an unrelated error that happens to carry one must not be."""
    monkeypatch.setenv("ORIENTA_ALLOWED_HOSTS", "testserver")
    from fastapi.testclient import TestClient

    from backend.api.main import app
    from backend.api.problem import CODE_HEADER
    from backend.api.routes import ebsd_viewer

    class _Other(Exception):
        code = "somethingElse"

    def boom(*a, **k):
        raise _Other("unrelated failure")

    monkeypatch.setattr(ebsd_viewer, "_load_ebsd_blocking", boom)
    r = TestClient(app, raise_server_exceptions=False).post(
        "/api/ebsd/load", json={"path": str(tmp_path / "x.up1")})
    assert r.status_code == 400
    assert CODE_HEADER.lower() not in r.headers
    assert r.json()["detail"] == "unrelated failure"


# --------------------------------------------------- square files as before

def test_a_square_v1_file_with_real_records_still_resolves(tmp_path):
    up = tmp_path / "sq.up1"
    _up1_v1(up, 4, 4, 36)
    _osc(tmp_path / "sq.osc", 6, 6, 0.5, 0.5, hex_grid=False)
    g = resolve_up1_geometry(str(up))
    assert g.nav_shape == (6, 6)
    assert g.step_yx == pytest.approx((0.5, 0.5))


def test_a_square_v3_file_still_resolves(tmp_path):
    up = tmp_path / "sq3.up1"
    _up1_v3(up, 4, 4, 5, 4, is_hex=0)
    _osc(tmp_path / "sq3.osc", 5, 4, 0.7, 0.7, hex_grid=False)
    g = resolve_up1_geometry(str(up))
    assert g.version == 3 and g.nav_shape is None
    assert g.step_yx == pytest.approx((0.7, 0.7))


def test_a_layout_that_does_not_match_the_stated_step_is_not_trusted(tmp_path):
    """Staggered row starts only mean "hexagonal" if the records are laid out as
    assumed: x spacing along a row equal to the step the file states. Otherwise
    the decode is not trusted and the file is treated as before (square)."""
    buf = bytearray(64)
    struct.pack_into("<3I", buf, 16, 5, 5, 36)
    buf += b"\x00" * 3 + OSC_MAGIC + struct.pack("<2I", 0, 0) + struct.pack("<2f", 1.0, 0.866)
    rec = np.zeros((36, 14), dtype="<f4")
    for r in range(6):
        for c in range(6):
            rec[r * 6 + c, 3] = c * 7.0 + (3.5 if r % 2 else 0.0)   # spacing 7, step says 1
            rec[r * 6 + c, 4] = r * 0.866
    buf += rec.tobytes()
    p = tmp_path / "odd.osc"
    p.write_bytes(bytes(buf))
    assert read_osc_metadata(str(p)).is_hex is False


def test_an_osc_whose_first_row_has_one_point_cannot_be_told(tmp_path):
    buf = bytearray(64)
    struct.pack_into("<3I", buf, 16, 0, 5, 6)
    buf += b"\x00" * 3 + OSC_MAGIC + struct.pack("<2I", 0, 0) + struct.pack("<2f", 1.0, 0.866)
    rec = np.zeros((6, 14), dtype="<f4")
    rec[:, 3] = [0, 0.5, 0, 0.5, 0, 0.5]
    rec[:, 4] = np.arange(6) * 0.866
    buf += rec.tobytes()
    p = tmp_path / "one.osc"
    p.write_bytes(bytes(buf))
    assert read_osc_metadata(str(p)).is_hex is False


def test_the_osc_reader_says_square_for_a_square_grid_and_hex_for_a_staggered_one(tmp_path):
    _osc(tmp_path / "s.osc", 6, 6, 1.0, 1.0, hex_grid=False)
    _osc(tmp_path / "h.osc", 6, 6, 1.0, 0.866, hex_grid=True)
    assert read_osc_metadata(str(tmp_path / "s.osc")).is_hex is False
    assert read_osc_metadata(str(tmp_path / "h.osc")).is_hex is True


@pytest.mark.skipif(not SQUARE.exists(), reason="Scan5.up1 not present")
def test_the_real_square_scans_are_not_taken_for_hexagonal():
    d = SQUARE.parent
    for stem in ("Scan5", "Scan57", "map20260610152042893_cropped"):
        assert read_osc_metadata(str(d / f"{stem}.osc")).is_hex is False
        g = resolve_up1_geometry(str(d / f"{stem}.up1"))
        assert g.n_patterns > 0
