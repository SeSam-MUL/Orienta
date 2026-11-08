"""A synthetic file with two measurements on the SAME grid, for the tests of the readers that follow
the loaded measurement.

Not a test module: the fixtures (``two``, ``viewer``, ``session``) and the tables are imported by
the ``test_measurement_*`` modules.

A reader of per-measurement data reads the measurement that was loaded.

A file can offer several measurements (groups ``/1``, ``/2``, ...). Loading picks one; every reader
that later opens the file (Band Contrast and the image-quality channel, the acquisition header, the
EDS maps, the electron images, Euler and Phase, the exports) has to read THAT measurement. Before,
each of them took the first group that had its channel: for a second measurement with the same grid
nothing complained, and measurement 1's map was drawn or exported as measurement 2's.

The synthetic file below has two measurements on the SAME 2x3 grid whose every channel differs from
the other's on every pixel (hand-written values in the tables), so a read of the wrong group, of the
first group, or a fall-through gives a different number.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import variant_registry  # noqa: E402
from backend.api.services.load_choices import LoadChoiceError, LoadVariant  # noqa: E402

NY, NX, SY, SX = 2, 3, 4, 5
N = NY * NX

V1 = LoadVariant("1", "processed")
V2 = LoadVariant("2", "processed")
V2_RAW = LoadVariant("2", "unprocessed")
V7 = LoadVariant("7", "processed")          # a measurement the file does not hold

# ---- the two measurements, value by value (row-major pixels)
BC = {1: [11, 12, 13, 14, 15, 16], 2: [90, 20, 60, 30, 80, 50]}
IQ = {1: [0.10, 0.11, 0.12, 0.13, 0.14, 0.15], 2: [0.9, 0.2, 0.6, 0.3, 0.8, 0.5]}
BANDS = {1: 5, 2: 9}
PHASE = {1: 1, 2: 2}
EULER = {1: (0.1, 0.2, 0.3), 2: (1.1, 1.2, 1.3)}
PC = {1: (0.41, 0.61, 0.81), 2: (0.42, 0.62, 0.82)}          # X, Y, detector distance
KV = {1: 20.0, 2: 15.0}
TILT_DEG = {1: 70.0, 2: 60.0}
WD = {1: 10.0, 2: 12.0}
MAG = {1: 300.0, 2: 500.0}
STEP = {1: 0.5, 2: 0.25}
FE = {1: [100, 101, 102, 103, 104, 105], 2: [200, 201, 202, 203, 204, 205]}
SE = {1: [[100, 101, 102], [103, 104, 105]], 2: [[200, 201, 202], [203, 204, 205]]}
PHASE_NAME = {1: "Al", 2: "Si"}          # Aztec's phase of each measurement


def processed(g: int) -> np.ndarray:
    return np.full((N, SY, SX), 10 * g, np.uint8)


def unprocessed(g: int) -> np.ndarray:
    return (np.arange(N * SY * SX) * 3 + 300 + 1000 * (g - 1)).astype(np.int16).reshape(N, SY, SX)


def _scalar(value):
    return np.array([value], dtype=np.float64)


def _group(f, g: int):
    hdr = f.create_group(f"{g}/EBSD/Header")
    data = f.create_group(f"{g}/EBSD/Data")
    hdr.create_dataset("X Cells", data=NX)
    hdr.create_dataset("Y Cells", data=NY)
    hdr.create_dataset("X Step", data=_scalar(STEP[g]))
    hdr.create_dataset("Y Step", data=_scalar(STEP[g]))
    hdr.create_dataset("Pattern Height", data=SY)
    hdr.create_dataset("Pattern Width", data=SX)
    hdr.create_dataset("Beam Voltage", data=_scalar(KV[g]))
    hdr.create_dataset("Tilt Angle", data=_scalar(np.radians(TILT_DEG[g])))
    hdr.create_dataset("Working Distance", data=_scalar(WD[g]))
    hdr.create_dataset("Magnification", data=_scalar(MAG[g]))
    hdr.create_dataset("Scanning Rotation Angle", data=_scalar(0.0))
    hdr.create_dataset("Processed Static Background", data=np.full((SY, SX), 3 + g, np.uint8))
    hdr.create_dataset(f"Phases/{g}/Phase Name", data=np.bytes_(PHASE_NAME[g]))
    data.create_dataset("Band Contrast", data=np.array(BC[g], np.uint8))
    data.create_dataset("IQ", data=np.array(IQ[g], np.float32))
    data.create_dataset("Bands", data=np.full(N, BANDS[g], np.uint8))
    data.create_dataset("Phase", data=np.full(N, PHASE[g], np.uint8))
    data.create_dataset("Euler", data=np.tile(np.array(EULER[g], np.float32), (N, 1)))
    for name, value in zip(("Pattern Center X", "Pattern Center Y", "Detector Distance"), PC[g]):
        data.create_dataset(name, data=np.full(N, value, np.float32))
    data.create_dataset("Processed Patterns", data=processed(g))
    data.create_dataset("Unprocessed Patterns", data=unprocessed(g))
    eds_hdr = f.create_group(f"{g}/EDS/Header")
    eds_hdr.create_dataset("X Cells", data=NX)
    eds_hdr.create_dataset("Y Cells", data=NY)
    eds_hdr.create_dataset("Beam Voltage", data=_scalar(KV[g]))
    eds_hdr.create_dataset("Working Distance", data=_scalar(WD[g]))
    fe = f.create_group(f"{g}/EDS/Data/Window Integral/Fe")
    fe.create_dataset("Counts", data=np.array(FE[g], np.uint32))
    ei_hdr = f.create_group(f"{g}/Electron Image/Header")
    ei_hdr.create_dataset("X Cells", data=NX)
    ei_hdr.create_dataset("Y Cells", data=NY)
    f.create_dataset(f"{g}/Electron Image/Data/SE/Image 1", data=np.array(SE[g], np.uint16))


@pytest.fixture
def two(tmp_path):
    """An Oxford file with measurements 1 and 2 on the same 2x3 grid."""
    path = tmp_path / "two.h5oina"
    with h5py.File(path, "w") as f:
        f.create_dataset("Manufacturer", data=np.bytes_("Oxford Instruments"))
        f.create_dataset("Format Version", data=np.bytes_("5.0"))
        _group(f, 1)
        _group(f, 2)
    return str(path)


def _grid(values, dtype=np.float64):
    return np.asarray(values, dtype=dtype).reshape(NY, NX)


# Each reader is asked three ways: without a variant (what it always did: the first measurement),
# for measurement 1 named, and for measurement 2.
BOTH = [(None, 1), (V1, 1), (V2, 2)]
BOTH_IDS = ["no-variant", "measurement-1", "measurement-2"]


@pytest.fixture
def session(two):
    from backend.api.services import h5_session
    h5_session.open_file(two)
    yield h5_session
    h5_session.close_file()


@pytest.fixture
def viewer(two, monkeypatch):
    """``hold(variant)``: the EBSD viewer's active dataset is a load of ``two`` as that variant, and
    the HDF5 session has the file open (the state after a load)."""
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.services import h5_session

    def hold(variant):
        monkeypatch.setattr(ev, "_ebsd_file_path", two)
        monkeypatch.setattr(ev, "_active_dataset", "ds-measurements")
        if variant is None:
            variant_registry.drop("ds-measurements")
        else:
            variant_registry.register("ds-measurements", variant)
        h5_session.open_file(two)
        return h5_session

    yield hold
    from backend.api.services import h5_session as _hs
    _hs.close_file()
    variant_registry.drop("ds-measurements")


def _xmap():
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    xs, ys = np.meshgrid(np.arange(NX) * 0.5, np.arange(NY) * 0.5)
    return CrystalMap(rotations=Rotation.identity(N), phase_id=np.zeros(N, int),
                      x=xs.ravel(), y=ys.ravel(),
                      phase_list=PhaseList(Phase(name="Al", point_group="m-3m")))
