# backend/api/services/orientation_ground_truth.py
"""Load a vendor's own indexing solution from an EBSD file, as ground truth."""
from __future__ import annotations

from dataclasses import dataclass, field

import h5py
import numpy as np


@dataclass
class VendorOrientations:
    vendor: str                       # "oxford" | "edax"
    euler: np.ndarray                 # (N, 3) radians, Bunge ZXZ
    phase_id: np.ndarray              # (N,) int
    grid_shape: tuple[int, int]       # (n_rows, n_cols)
    header_meta: dict = field(default_factory=dict)


def _first_scan_group(h: h5py.File):
    for key in h:
        g = h[key]
        if isinstance(g, h5py.Group) and "EBSD" in g:
            return key, g
    raise ValueError("no EBSD scan group found")


def load_vendor_orientations(path: str) -> VendorOrientations:
    with h5py.File(path, "r") as h:
        manufacturer = ""
        if "Manufacturer" in h:
            raw = h["Manufacturer"][()]
            manufacturer = (raw[0] if np.ndim(raw) else raw)
            manufacturer = manufacturer.decode() if isinstance(manufacturer, bytes) else str(manufacturer)

        key, scan = _first_scan_group(h)
        data = scan["EBSD"]["Data"]
        header = scan["EBSD"]["Header"]

        if "Euler" in data:  # Oxford H5OINA
            euler = np.asarray(data["Euler"][:], dtype=np.float64)
            phase_id = np.asarray(data["Phase"][:], dtype=np.int64)
            n_cols = int(np.asarray(header["X Cells"])[0])
            n_rows = int(np.asarray(header["Y Cells"])[0])
            soe = np.asarray(header["Specimen Orientation Euler"][:], dtype=np.float64).reshape(-1)
            meta = {
                "specimen_orientation_euler": soe.tolist(),
                "scanning_rotation_angle": float(np.asarray(header["Scanning Rotation Angle"])[0]),
                "tilt_angle": float(np.asarray(header["Tilt Angle"])[0]),
            }
            return VendorOrientations("oxford", euler, phase_id, (n_rows, n_cols), meta)

        if all(k in data for k in ("Phi1", "Phi", "Phi2")):  # EDAX OIM h5
            euler = np.stack(
                [np.asarray(data["Phi1"][:]), np.asarray(data["Phi"][:]), np.asarray(data["Phi2"][:])],
                axis=-1,
            ).astype(np.float64)
            phase_id = np.asarray(data["Phase"][:], dtype=np.int64)
            n_cols = int(np.asarray(header["nColumns"])[0])
            n_rows = int(np.asarray(header["nRows"])[0])
            cs_id = None
            if "Coordinate System" in header and "ID" in header["Coordinate System"]:
                cs_id = int(np.asarray(header["Coordinate System"]["ID"])[0])
            meta = {
                "coordinate_system_id": cs_id,
                "sample_tilt": float(np.asarray(header["Sample Tilt"])[0]),
            }
            return VendorOrientations("edax", euler, phase_id, (n_rows, n_cols), meta)

    raise ValueError(f"{path}: no recognised Oxford/EDAX orientation data")
