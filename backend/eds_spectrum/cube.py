"""Reading the per-pixel EDS spectrum cube of an Oxford H5OINA file.

Layout (``root`` is ``"1"``): ``EDS/Data/Spectrum`` ``(n_px, n_channels)`` int,
``EDS/Data/Live Time`` ``(n_px,)`` seconds, ``EDS/Header`` with the channel grid
in eV.

The cube is chunked PER CHANNEL (chunks ``(n_px, 1)``, lzf): reading one pixel
row decompresses every channel chunk.  Everything here therefore reads channel
blocks ``ds[:, c0:c1]`` for all pixels.
"""
from __future__ import annotations

import contextlib
import os
from typing import Callable, Iterator, Optional, Tuple, Union

import h5py
import numpy as np

from .model import SpectrumMeta

H5Like = Union[str, os.PathLike, h5py.Group]  # h5py.File is a Group
Progress = Optional[Callable[[float], None]]

DEFAULT_BLOCK = 64


@contextlib.contextmanager
def _open(h5file: H5Like) -> Iterator[h5py.Group]:
    """Accept a path (opened and closed here) or an already open file/group."""
    if isinstance(h5file, h5py.Group):
        yield h5file
    else:
        with h5py.File(h5file, "r") as f:
            yield f


def _scalar(ds) -> float:
    return float(np.ravel(ds[()])[0])


def _text(ds) -> str:
    v = np.ravel(ds[()])[0]
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)


def has_spectrum_cube(h5file: H5Like, root: str = "1") -> bool:
    """True when the file carries a per-pixel spectrum dataset and its header."""
    with _open(h5file) as f:
        data, header = f.get(f"{root}/EDS/Data"), f.get(f"{root}/EDS/Header")
        if not isinstance(data, h5py.Group) or not isinstance(header, h5py.Group):
            return False
        spec = data.get("Spectrum")
        return isinstance(spec, h5py.Dataset) and spec.ndim == 2


def _spectrum(f: h5py.Group, root: str) -> h5py.Dataset:
    ds = f.get(f"{root}/EDS/Data/Spectrum")
    if not isinstance(ds, h5py.Dataset) or ds.ndim != 2:
        raise KeyError(f"no per-pixel spectrum cube under {root}/EDS/Data/Spectrum")
    return ds


def read_meta(h5file: H5Like, root: str = "1") -> SpectrumMeta:
    """Channel grid (keV) and acquisition conditions from ``EDS/Header``."""
    with _open(h5file) as f:
        h = f[f"{root}/EDS/Header"]
        n_ch = int(round(_scalar(h["Number Channels"])))
        n_cube = _spectrum(f, root).shape[1]
        if n_ch != n_cube:
            raise ValueError(f"header says {n_ch} channels, the spectrum cube "
                             f"has {n_cube}")
        tilt = h.get("Tilt Angle")
        det = h.get("Detector Serial Number")

        def _deg(name):
            ds = h.get(name)
            return float(np.degrees(_scalar(ds))) if ds is not None else None

        return SpectrumMeta(
            elevation_deg=_deg("Detector Elevation"),
            azimuth_deg=_deg("Detector Azimuth"),
            tilt_axis_deg=_deg("Tilt Axis"),
            start_kev=_scalar(h["Start Channel"]) / 1e3,
            width_kev=_scalar(h["Channel Width"]) / 1e3,
            n_channels=n_ch,
            beam_kv=_scalar(h["Beam Voltage"]),
            detector=_text(det) if det is not None else "",
            tilt_deg=float(np.degrees(_scalar(tilt))) if tilt is not None else None,
        )


def live_time(h5file: H5Like, root: str = "1") -> np.ndarray:
    """Live time per pixel in seconds, float64, flat in file order."""
    with _open(h5file) as f:
        return np.asarray(f[f"{root}/EDS/Data/Live Time"][()], dtype=float).ravel()


def channel_blocks(n_channels: int, block: int,
                   lo: int = 0, hi: Optional[int] = None) -> Iterator[Tuple[int, int]]:
    """``(c0, c1)`` pairs of at most ``block`` channels covering ``[lo, hi)``."""
    if block < 1:
        raise ValueError("block must be >= 1")
    hi = n_channels if hi is None else hi
    for c0 in range(lo, hi, block):
        yield c0, min(c0 + block, hi)


def sum_spectrum(h5file: H5Like, root: str = "1", block: int = DEFAULT_BLOCK,
                 progress: Progress = None) -> np.ndarray:
    """Counts per channel summed over all pixels (float64, ``n_channels``)."""
    with _open(h5file) as f:
        ds = _spectrum(f, root)
        out = np.zeros(ds.shape[1])
        for c0, c1 in channel_blocks(ds.shape[1], block):
            out[c0:c1] = ds[:, c0:c1].sum(axis=0, dtype=np.float64)
            if progress is not None:
                progress(c1 / ds.shape[1])
        return out
