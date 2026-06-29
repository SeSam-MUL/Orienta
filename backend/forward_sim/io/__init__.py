"""SP3 — serialisation of forward-sim masters into external file formats.

Exposes:

* :func:`write_master_h5` — writes our GPU-computed master pattern into an
  EMsoft-format ``EBSDmaster`` HDF5 file so it is a drop-in for
  ``kikuchipy.load(path, projection=...)`` and the kikuchipy dictionary-indexing
  pipeline.
* :func:`write_sht` — writes our master into an EMSphInx ``.sht``
  spherical-harmonics file, loadable by
  :mod:`backend.spherical_gpu.pipeline.sht_io` and the EMSphInx C++ reader.
* :func:`write_mc_h5` — writes our GPU Monte-Carlo output into an
  EMsoft-MCOpenCL-format ``.h5`` so ``load_mc`` reads it back and
  ``scan_missing`` / ``crystal_picker`` detect the GPU run's MC file.
"""
from __future__ import annotations

from .master_h5 import write_master_h5
from .mc_h5 import write_mc_h5
from .sht_writer import write_sht

__all__ = ["write_master_h5", "write_sht", "write_mc_h5"]
