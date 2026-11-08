"""Patterns rendered at a pattern centre the TEST chose, and an indexer to search them with.

Used by the tests of the pattern-centre search (``test_pc_cost``, ``test_pc_search``): a search
that finds the PC the test rendered the patterns at has been checked against a number that
does not come from the code under test and does not need a measured file.

* The patterns come from kikuchipy's bundled nickel master pattern (``kp.data``, shipped inside
  the package, no download) through ``EBSD master pattern.get_patterns``, with the detector
  handed over the way every projection in Orienta is (``detector_convention``).
* The indexer is built the way the application builds one (``ebsd_utils.create_indexer``) from
  ``Test_data/Ni.cif`` (tracked in the repository, present in every checkout).
* PyEBSDIndex's band detection runs on the CPU, as in the Windows installer (which ships no
  pyopencl). A development environment that has pyopencl would otherwise use its GPU for the
  band detection and, with many indexers in one process, run out of host memory (E18);
  ``cpu_band_detection`` forces the CPU for the length of a ``with`` block and puts everything back.
"""
from __future__ import annotations

import contextlib
import functools
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NI_CIF = ROOT / "Test_data" / "Ni.cif"


@contextlib.contextmanager
def cpu_band_detection():
    """PyEBSDIndex's CPU band detector for the length of the block; restored afterwards."""
    import pyebsdindex
    import pyebsdindex_mode
    had = pyebsdindex._pyopencl_installed
    saved = {name: getattr(sys.modules[name], "band_detect", None)
             for name in pyebsdindex_mode._BINDING_MODULES if name in sys.modules}
    pyebsdindex_mode.force_cpu_band_detection(platform="darwin")
    try:
        yield
    finally:
        pyebsdindex._pyopencl_installed = had
        for name, bound in saved.items():
            if bound is not None:
                setattr(sys.modules[name], "band_detect", bound)


def ni_phase_list():
    """The nickel phase as the Hough indexer is given it (``Test_data/Ni.cif``)."""
    if not NI_CIF.is_file():
        pytest.skip(f"{NI_CIF.name} is not in this checkout's Test_data/")
    from orix.crystal_map import Phase, PhaseList
    from ebsd_utils import sanitize_cif
    return PhaseList(Phase.from_cif(sanitize_cif(str(NI_CIF))))


@functools.lru_cache(maxsize=16)
def _render(shape, pc, n, seed, sample_tilt, tilt):
    import kikuchipy as kp
    from kikuchipy.detectors import EBSDDetector
    from orix.quaternion import Rotation
    from backend.api.services import detector_convention as dc
    master = kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")
    rng = np.random.default_rng(seed)
    eulers = np.column_stack([rng.uniform(0, 360, n), rng.uniform(5, 85, n), rng.uniform(0, 360, n)])
    detector = EBSDDetector(shape=shape, pc=pc, sample_tilt=sample_tilt, tilt=tilt, convention="bruker")
    patterns = master.get_patterns(
        rotations=Rotation.from_euler(np.deg2rad(eulers)),
        detector=dc.for_kikuchipy_projection(detector), energy=20, compute=True)
    out = np.asarray(patterns.data, dtype=np.float32).reshape(n, *shape)
    out.setflags(write=False)
    return out


def render_patterns(shape=(80, 80), pc=(0.52, 0.31, 0.62), n=9, seed=7,
                    sample_tilt=70.0, tilt=0.0):
    """``n`` patterns of nickel in random orientations, rendered at the Bruker ``pc``.

    Returns a writable ``(n, rows, cols)`` float32 array."""
    return np.array(_render(tuple(int(v) for v in shape), tuple(float(v) for v in pc), int(n),
                            int(seed), float(sample_tilt), float(tilt)))


def make_indexer(shape=(80, 80), sample_tilt=70.0, tilt=0.0, phase_list=None, n_bands=12):
    """A Hough indexer the way the application builds one, at a placeholder PC (0.5, 0.5, 0.5).

    Call it inside ``cpu_band_detection()``: the band detector is chosen when the indexer is built."""
    from kikuchipy.detectors import EBSDDetector
    from ebsd_utils import create_indexer, prepare_reflectors
    plist = phase_list if phase_list is not None else ni_phase_list()
    reflectors = prepare_reflectors(plist, min_d=1.0, f_threshold=0.1, max_reflectors=65)
    detector = EBSDDetector(shape=tuple(shape), pc=(0.5, 0.5, 0.5), sample_tilt=sample_tilt,
                            tilt=tilt, convention="bruker")
    return create_indexer(detector, plist, reflectors, nBands=n_bands)
