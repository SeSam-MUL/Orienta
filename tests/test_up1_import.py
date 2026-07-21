"""Tests for EDAX UP1/UP2 + OSC import (edax_up1 + safe_loader path).

Two layers:
  * Synthetic byte-level unit tests (always run) — craft tiny UP1/UP2/OSC files
    with known values and assert the parser reads them exactly, including the
    consistency guards and the odd-alignment OSC step block.
  * Real-file integration tests (skipif) — load the actual EDAX files in
    ``Test_data/new test/`` end-to-end and assert the 2-D grid, dtype and step.
"""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edax_up1 import (  # noqa: E402
    is_edax_up_file,
    read_up1_header,
    read_osc_metadata,
    resolve_up1_geometry,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "Test_data" / "new test"
SCAN5 = DATA_DIR / "Scan5.up1"           # v1, 79x79, 151x151
MAP_V3 = DATA_DIR / "map20260610152042893_cropped.up1"  # v3, 114x114, 123x91

OSC_MAGIC = bytes([0xB9, 0x0B, 0xEF, 0xFF, 0x02, 0x00, 0x00, 0x00])


# --------------------------------------------------------------------------
# Synthetic builders
# --------------------------------------------------------------------------
def _make_up1_v1(path, sx, sy, npat, dtype_bytes=1):
    with open(path, "wb") as f:
        f.write(struct.pack("<4I", 1, sx, sy, 16))          # version, sx, sy, offset
        f.write(b"\x00" * (sx * sy * dtype_bytes * npat))    # pattern data


def _make_up1_v3(path, sx, sy, nx, ny, is_hex=0, dx=0.5, dy=0.5, dtype_bytes=1):
    with open(path, "wb") as f:
        f.write(struct.pack("<4I", 3, sx, sy, 42))           # version, sx, sy, offset=42
        f.write(b"\x00")                                     # @16 skip byte
        f.write(struct.pack("<2I", nx, ny))                  # @17 nx, @21 ny
        f.write(struct.pack("<B", is_hex))                   # @25 is_hex
        f.write(struct.pack("<2d", dx, dy))                  # @26/@34 dx, dy (float64)
        f.write(b"\x00" * (sx * sy * dtype_bytes * nx * ny))


def _make_osc(path, ncols, nrows, npoints, xstep, ystep, with_magic=True, pc=None):
    # The PC lives at fixed offset 1860 (3x float32), so the buffer must reach
    # at least there when a PC is written.
    buf = bytearray(max(64, 1872 if pc is not None else 64))
    struct.pack_into("<3I", buf, 16, ncols - 1, nrows - 1, npoints)
    if pc is not None:
        struct.pack_into("<3f", buf, 1860, *pc)
    if with_magic:
        # Place the magic at an intentionally odd (non-4-aligned) offset to
        # exercise the same alignment situation as the real files.
        buf += b"\x00" * 3                     # push magic off 4-byte alignment
        buf += OSC_MAGIC
        buf += struct.pack("<2I", 0, 0)        # k+8 ptr, k+12
        buf += struct.pack("<2f", xstep, ystep)  # k+16 Xstep, k+20 Ystep
        buf += b"\x00" * 16                     # a couple of point floats
    Path(path).write_bytes(bytes(buf))


# --------------------------------------------------------------------------
# Synthetic unit tests (always run)
# --------------------------------------------------------------------------
def test_is_edax_up_file():
    assert is_edax_up_file("a.up1")
    assert is_edax_up_file("a.UP2")
    assert not is_edax_up_file("a.h5oina")
    assert not is_edax_up_file("a.h5")


def test_v1_header_flat(tmp_path):
    p = tmp_path / "s.up1"
    _make_up1_v1(p, 8, 8, 12)
    h = read_up1_header(str(p))
    assert h.version == 1
    assert (h.sx, h.sy) == (8, 8)
    assert h.n_patterns == 12
    assert h.nx is None and h.ny is None
    assert h.is_hex is False


def test_v3_header_has_grid(tmp_path):
    p = tmp_path / "m.up1"
    _make_up1_v3(p, 6, 6, nx=5, ny=4)
    h = read_up1_header(str(p))
    assert h.version == 3
    assert (h.nx, h.ny) == (5, 4)
    assert h.n_patterns == 20
    assert h.is_hex is False


def test_up2_is_uint16_sized(tmp_path):
    # A UP2 v1 file: pattern count must divide by 2 bytes/px, not 1.
    p = tmp_path / "s.up2"
    with open(p, "wb") as f:
        f.write(struct.pack("<4I", 1, 4, 4, 16))
        f.write(b"\x00" * (4 * 4 * 2 * 9))   # 9 patterns, 16-bit
    h = read_up1_header(str(p))
    assert h.n_patterns == 9


def test_version_2_rejected(tmp_path):
    p = tmp_path / "bad.up1"
    with open(p, "wb") as f:
        f.write(struct.pack("<4I", 2, 4, 4, 16))
    with pytest.raises(ValueError):
        read_up1_header(str(p))


def test_osc_grid_and_step(tmp_path):
    p = tmp_path / "s.osc"
    _make_osc(p, ncols=10, nrows=7, npoints=70, xstep=0.25, ystep=0.25)
    m = read_osc_metadata(str(p))
    assert (m.ncols, m.nrows, m.npoints) == (10, 7, 70)
    assert m.xstep == pytest.approx(0.25)
    assert m.ystep == pytest.approx(0.25)


def test_osc_inconsistent_grid_is_rejected(tmp_path):
    # ncols*nrows != npoints -> grid must come back None (never a guessed value)
    p = tmp_path / "bad.osc"
    _make_osc(p, ncols=10, nrows=7, npoints=999, xstep=1.0, ystep=1.0)
    m = read_osc_metadata(str(p))
    assert m.ncols is None and m.nrows is None and m.npoints is None
    # step is independent of the grid and should still be read
    assert m.xstep == pytest.approx(1.0)


def test_osc_missing_magic_gives_no_step(tmp_path):
    p = tmp_path / "nomag.osc"
    _make_osc(p, ncols=5, nrows=5, npoints=25, xstep=1.0, ystep=1.0, with_magic=False)
    m = read_osc_metadata(str(p))
    assert (m.ncols, m.nrows) == (5, 5)
    assert m.xstep is None and m.ystep is None


def test_osc_reads_pattern_centre(tmp_path):
    p = tmp_path / "pc.osc"
    _make_osc(p, ncols=10, nrows=7, npoints=70, xstep=1.0, ystep=1.0,
              pc=(0.55, 0.50, 0.70))
    m = read_osc_metadata(str(p))
    assert m.pc is not None
    assert m.pc[0] == pytest.approx(0.55)
    assert m.pc[1] == pytest.approx(0.50)
    assert m.pc[2] == pytest.approx(0.70)


def test_osc_no_pc_slot_is_none(tmp_path):
    p = tmp_path / "nopc.osc"
    _make_osc(p, ncols=5, nrows=5, npoints=25, xstep=1.0, ystep=1.0)  # no pc
    assert read_osc_metadata(str(p)).pc is None


def test_osc_implausible_pc_rejected(tmp_path):
    # All-zero / out-of-band values must not be read as a PC.
    p = tmp_path / "badpc.osc"
    _make_osc(p, ncols=5, nrows=5, npoints=25, xstep=1.0, ystep=1.0,
              pc=(0.0, 0.0, 0.0))
    assert read_osc_metadata(str(p)).pc is None


def test_resolve_v1_uses_osc_grid(tmp_path):
    up = tmp_path / "scan.up1"
    _make_up1_v1(up, 8, 8, 70)
    _make_osc(tmp_path / "scan.osc", ncols=10, nrows=7, npoints=70, xstep=0.3, ystep=0.3)
    g = resolve_up1_geometry(str(up))
    assert g.version == 1
    assert g.nav_shape == (7, 10)          # (nrows, ncols)
    assert g.step_yx == pytest.approx((0.3, 0.3))


def test_resolve_v1_square_fallback_without_osc(tmp_path):
    up = tmp_path / "sq.up1"
    _make_up1_v1(up, 4, 4, 25)             # 25 = 5*5, perfect square, no .osc
    g = resolve_up1_geometry(str(up))
    assert g.nav_shape == (5, 5)
    assert g.step_yx is None


def test_resolve_v1_non_square_without_osc_is_none(tmp_path):
    up = tmp_path / "ns.up1"
    _make_up1_v1(up, 4, 4, 30)             # 30 not a perfect square, no .osc
    g = resolve_up1_geometry(str(up))
    assert g.nav_shape is None             # caller must fail loud


def test_resolve_v3_keeps_header_grid_but_takes_osc_step(tmp_path):
    up = tmp_path / "m.up1"
    _make_up1_v3(up, 6, 6, nx=5, ny=4)
    _make_osc(tmp_path / "m.osc", ncols=5, nrows=4, npoints=20, xstep=0.7, ystep=0.7)
    g = resolve_up1_geometry(str(up))
    assert g.version == 3
    assert g.nav_shape is None             # kikuchipy header is authoritative for v3
    assert g.step_yx == pytest.approx((0.7, 0.7))


def test_resolve_v1_osc_pointcount_mismatch_ignored(tmp_path):
    up = tmp_path / "x.up1"
    _make_up1_v1(up, 4, 4, 20)             # 20 patterns
    _make_osc(tmp_path / "x.osc", ncols=6, nrows=4, npoints=24, xstep=1.0, ystep=1.0)
    g = resolve_up1_geometry(str(up))
    # osc npoints (24) != up count (20): don't reshape from that osc; 20 isn't
    # square either, so nav_shape stays None.
    assert g.nav_shape is None


# --------------------------------------------------------------------------
# Real-file integration tests
# --------------------------------------------------------------------------
@pytest.mark.skipif(not SCAN5.exists(), reason="Scan5.up1 not present")
def test_real_v1_scan5_geometry():
    g = resolve_up1_geometry(str(SCAN5))
    assert g.version == 1
    assert g.n_patterns == 22801
    assert g.nav_shape == (151, 151)
    assert g.step_yx == pytest.approx((1.0, 1.0))
    # PC recovered from the .osc (verified correct via refinement, see docs).
    assert g.pc_edax is not None
    assert g.pc_edax[0] == pytest.approx(0.5499, abs=1e-3)
    assert g.pc_edax[1] == pytest.approx(0.5026, abs=1e-3)
    assert g.pc_edax[2] == pytest.approx(0.7020, abs=1e-3)


@pytest.mark.skipif(not SCAN5.exists(), reason="Scan5.up1 not present")
def test_real_scan5_applies_osc_pc_to_detector():
    from safe_loader import load_ebsd_safe
    sig = load_ebsd_safe(str(SCAN5), verbose=False)
    # NOT the kikuchipy placeholder (0.5, 0.5, 0.5)
    assert not np.allclose(sig.detector.pc[0], [0.5, 0.5, 0.5])
    # pc_tsl round-trips to the .osc xstar/ystar/zstar
    pc_tsl = sig.detector.pc_tsl()[0]
    assert pc_tsl[0] == pytest.approx(0.5499, abs=1e-3)
    assert pc_tsl[2] == pytest.approx(0.7020, abs=1e-3)
    assert sig.metadata.get_item("Signal.pc_source") == "osc"


def test_default_pc_flag_when_no_osc(tmp_path):
    """A UP file with no .osc → default PC + pc_source='default' (the case the
    UI warns about)."""
    from safe_loader import load_ebsd_safe
    up = tmp_path / "sq.up1"
    _make_up1_v1(up, 6, 6, 25)  # 5x5, no .osc sidecar
    sig = load_ebsd_safe(str(up), verbose=False)
    assert np.allclose(sig.detector.pc[0], [0.5, 0.5, 0.5])
    assert sig.metadata.get_item("Signal.pc_source") == "default"


@pytest.mark.skipif(not MAP_V3.exists(), reason="map v3 up1 not present")
def test_real_v3_map_geometry():
    g = resolve_up1_geometry(str(MAP_V3))
    assert g.version == 3
    assert g.n_patterns == 11193
    assert g.nav_shape is None             # header carries 123x91
    assert g.step_yx == pytest.approx((1.0, 1.0))


@pytest.mark.skipif(not SCAN5.exists(), reason="Scan5.up1 not present")
def test_real_scan5_loads_2d_via_safe_loader():
    from safe_loader import load_ebsd_safe
    sig = load_ebsd_safe(str(SCAN5), verbose=False)
    assert sig.axes_manager.navigation_shape == (151, 151)
    assert sig.axes_manager.signal_shape == (79, 79)
    assert sig.data.dtype == np.uint8
    # detector present with a sensible default PC (calibrated later by the user)
    assert sig.detector.shape == (79, 79)
    # navigation axes carry the .osc step in µm
    for ax in sig.axes_manager.navigation_axes:
        assert ax.units == "um"
        assert ax.scale == pytest.approx(1.0)


@pytest.mark.skipif(not MAP_V3.exists(), reason="map v3 up1 not present")
def test_real_map_loads_2d_via_safe_loader():
    from safe_loader import load_ebsd_safe
    sig = load_ebsd_safe(str(MAP_V3), verbose=False)
    assert sig.axes_manager.navigation_shape == (123, 91)
    assert sig.axes_manager.signal_shape == (114, 114)
    assert sig.data.dtype == np.uint8


@pytest.mark.skipif(not SCAN5.exists(), reason="Scan5.up1 not present")
def test_up1_source_vendor_is_edax_for_indexing():
    """Regression: a .up1 is not HDF5, so the old h5py-only vendor probe
    returned 'unknown', which made the spherical reference-frame correction
    raise "unknown vendor". It must resolve to 'edax' (verified adapter)."""
    from safe_loader import load_ebsd_safe
    from backend.api.routes.indexing import build_spherical_det_params
    from backend.api.services.orientation_frame import apply_reference_frame_correction
    from orix.quaternion import Rotation

    sig = load_ebsd_safe(str(SCAN5), verbose=False)
    det_params = build_spherical_det_params(sig, sig.detector, str(SCAN5))
    assert det_params["source_vendor"] == "edax"
    # The exact call that crashed must now succeed for a UP file.
    out = apply_reference_frame_correction(
        Rotation.identity(), vendor=det_params["source_vendor"],
        header_meta={}, source="spherical_gpu",
    )
    assert out.size == 1
