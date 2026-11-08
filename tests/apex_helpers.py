"""Synthetic EDAX APEX files for tests: a metadata .h5 and the .up1 pattern files that belong to it.

The trees and header layouts are the real ones (group names, dataset names and types, byte
offsets); every value is made up. Nothing here comes from a measurement.
"""
from __future__ import annotations

import struct
from pathlib import Path

import h5py
import numpy as np

#: Two GUIDs written out by hand, with the 16 header bytes a little-endian field order gives
#: (the first three fields are byte-reversed): the independent expectation of the pairing tests.
GUID_A = "00112233-4455-6677-8899-aabbccddeeff"
GUID_A_BYTES = bytes.fromhex("33221100" "5544" "7766" "8899aabbccddeeff")
GUID_B = "01234567-89ab-cdef-0123-456789abcdef"
GUID_B_BYTES = bytes.fromhex("67452301" "ab89" "efcd" "0123456789abcdef")

_DATA_DTYPE = np.dtype([
    ("Orientations", "<f8", (9,)), ("IQ", "<f4"), ("CI", "<f4"), ("Phase", "<i4"),
    ("SEM Signal", "<i4"), ("Fit", "<f4"), ("Valid", "<i4"),
])

OSC_MAGIC = bytes([0xB9, 0x0B, 0xEF, 0xFF, 0x02, 0x00, 0x00, 0x00])


def _vlen(group, name, value):
    group.create_dataset(name, data=np.array([value], dtype=object), dtype=h5py.string_dtype())


def _f32(group, name, value):
    group.create_dataset(name, data=np.array([value], dtype=np.float32))


def _i32(group, name, value):
    group.create_dataset(name, data=np.array([value], dtype=np.int32))


def measurement_spec(**over) -> dict:
    """One measurement: a 3 x 5 hexagonal scan by default (step_y follows from step_x)."""
    spec = dict(
        id=GUID_A, area="Area 1", name="OIM Map 1", grid="HexGrid", n_rows=3, n_cols=5,
        step_x=0.5, step_y=None, pc=(0.5, 0.4, 0.6), elevation=5.0, azimuth=0.0, tilt=70.0,
        kv=20.0, n_table=None, phases=[("Phase A", "Phase A", 62), ("Phase B", "Phase A", 43)], sem=True,
        sem_data=None,                      # an explicit overview image; default a 40 x 50 ramp
        scan_end=None,                      # the raw value of the map's ScanEndTime attribute (APEX: int64[1] ticks)
        pattern_size=None,                  # (width, height) in pixels as the metadata records it; None: not recorded
    )
    spec.update(over)
    if spec["step_y"] is None:
        spec["step_y"] = spec["step_x"] * (3 ** 0.5 / 2 if spec["grid"] == "HexGrid" else 1.0)
    return spec


def make_apex_h5(path, measurements) -> Path:
    """The tree of an APEX file below ``TOP/Sample/<area>/<name>``. IQ of point i is 10 * i."""
    path = Path(path)
    with h5py.File(path, "w") as f:
        for spec in measurements:
            m = f.create_group(f"TOP/Sample/{spec['area']}/{spec['name']}")
            m.attrs.create("Id", np.array([spec["id"]], dtype=object), dtype=h5py.string_dtype())
            m.attrs.create("Name", np.array([spec["name"]], dtype=object), dtype=h5py.string_dtype())
            m.attrs.create("HexGrid", np.array([0], dtype=np.int16))       # unreliable in real files
            if spec["scan_end"] is not None:
                m.attrs.create("ScanEndTime", spec["scan_end"])
            sample = m.create_group("Sample")
            _vlen(sample, "Grid Type", spec["grid"].encode())
            _i32(sample, "Number Of Columns", spec["n_cols"])
            _i32(sample, "Number Of Rows", spec["n_rows"])
            _f32(sample, "Step X", spec["step_x"])
            _f32(sample, "Step Y", spec["step_y"])
            _f32(sample, "Sample Tilt", spec["tilt"])
            cam = m.create_group("Camera")
            _f32(cam, "Elevation Angle", spec["elevation"])
            _f32(cam, "Azimuthal Angle", spec["azimuth"])
            _f32(m.create_group("Column"), "SemKV", spec["kv"])
            hdr = m.create_group("EBSD/ANG/HEADER")
            pc = hdr.create_group("Pattern Center Calibration")
            for key, val in zip(("X-Star", "Y-Star", "Z-Star"), spec["pc"]):
                _f32(pc, key, val)
            if spec["pattern_size"] is not None:
                _i32(hdr, "Pattern Width", spec["pattern_size"][0])
                _i32(hdr, "Pattern Height", spec["pattern_size"][1])
            for i, (name, formula, sym) in enumerate(spec["phases"], start=1):
                p = hdr.create_group(f"Phase/{i}")
                _vlen(p, "Material Name", name.encode())
                _vlen(p, "Formula", formula.encode())
                _vlen(p, "Laue Group", b"6/mmm")
                _vlen(p, "Point Group", b"6/mmm")
                _i32(p, "Symmetry", sym)
            n = spec["n_table"] or spec["n_rows"] * spec["n_cols"]     # n_table: a table that does not fit the grid
            table = np.zeros(n, dtype=_DATA_DTYPE)
            table["IQ"] = np.arange(n, dtype=np.float32) * 10.0
            table["CI"] = 0.5
            table["Phase"] = 1
            table["Valid"] = 1
            m.create_dataset("EBSD/ANG/DATA/DATA", data=table)
            if spec["sem"]:
                sem = spec["sem_data"]
                if sem is None:
                    sem = (np.arange(40 * 50).reshape(40, 50) % 256).astype(np.uint8)
                m.create_dataset("SEM-PRIAS Images/DATA/SEM", data=sem)
    return path


def make_up(path, *, sx=8, sy=8, n=15, guid_bytes=GUID_A_BYTES, hex_flag=1, nx=5, ny=4,
            dx=0.5, dy=0.4330127, version=4, dtype_bytes=1, first=0) -> Path:
    """An UP file. Pattern ``i`` is a constant image of value ``(7 * i) % 256`` (UP1), except
    pattern 0, whose value is ``first`` (0 by default: all-zero bytes right behind a 42-byte header
    look like a nil GUID, which hides a reader that forgets to check where the header ends).

    Header: version, sx, sy, pattern offset (``<4I``), one flag byte at 16, ``nx, ny`` at 17
    (``<2I``), the hex flag at 25, ``dx, dy`` at 26 (``<2d``), the measurement GUID at 42 when the
    header is 58 bytes long (a header without it is 42 bytes)."""
    path = Path(path)
    offset = 58 if guid_bytes is not None else 42
    head = bytearray(struct.pack("<4I", version, sx, sy, offset))
    head += b"\x01" if hex_flag else b"\x00"
    head += struct.pack("<2I", nx, ny)
    head += struct.pack("<B", 1 if hex_flag else 0)
    head += struct.pack("<2d", dx, dy)
    assert len(head) == 42
    if guid_bytes is not None:
        head += guid_bytes
    body = bytearray()
    for i in range(n):
        body += bytes([first if i == 0 else (7 * i) % 256]) * (sx * sy * dtype_bytes)
    path.write_bytes(bytes(head) + bytes(body))
    return path


def wipe_backend() -> None:
    """Forget every loaded file. The viewer's registries are module-level, and re-loading a file
    that was visited before restores what its stash holds, so each test starts from nothing."""
    from backend.api.routes import ebsd_viewer
    from backend.api.services import crop_window, scan_geometry
    from backend.api.services.calibration_store import calibration_store

    ebsd_viewer._raw_signals.clear()
    ebsd_viewer._positions.clear()
    ebsd_viewer._dirty_datasets.clear()
    ebsd_viewer._signal_masks.clear()
    ebsd_viewer._overview_cache.clear()
    ebsd_viewer._registry_by_file.clear()
    ebsd_viewer._loaded_files.clear()
    ebsd_viewer._active_dataset = ""
    ebsd_viewer._ebsd_signal = None
    ebsd_viewer._ebsd_file_path = None
    ebsd_viewer._ebsd_variant_key = None
    ebsd_viewer._ebsd_apex = None
    crop_window.clear_all()
    scan_geometry.clear_all()
    calibration_store.clear()


def make_osc(path, *, ncols, nrows, xstep, ystep, pc=None) -> Path:
    """An .osc sidecar with grid, steps and (optionally) the PC at its fixed offset 1860."""
    buf = bytearray(max(64, 1872 if pc is not None else 64))
    struct.pack_into("<3I", buf, 16, ncols - 1, nrows - 1, ncols * nrows)
    if pc is not None:
        struct.pack_into("<3f", buf, 1860, *pc)
    buf += b"\x00" * 3                                    # push the magic off 4-byte alignment
    buf += OSC_MAGIC + struct.pack("<2I", 0, 0) + struct.pack("<2f", xstep, ystep)
    buf += b"\x00" * 16
    Path(path).write_bytes(bytes(buf))
    return Path(path)
