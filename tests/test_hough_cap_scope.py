"""The triplet-library cap must not be visible to anything but the library build.

`_capped_triplet_library` used to swap `pyebsdindex.tripletvote.np` — the MODULE
global — for a proxy. Numba reads a jitted function's globals when it compiles,
and it compiles `_tripvote_numba` lazily, on the first Hough index in a process
with a cold on-disk cache. The PC Refinement page fires "Index Pattern" and the
forward-sim preview together; the preview builds an indexer under the cap while
the index compiles, numba meets the proxy ("Untyped global name 'np': Cannot
determine Numba type of _CappedNumpy") and the failure sticks for the life of
the backend. Measured 2026-09-21 on a fresh v0.4.1 install: every "Index
Pattern" answered 500 until restart. A development machine never showed it
because its numba cache was warm.

So the cap now lives on a copy of `BandIndexer.build_trip_lib` alone, and the
module's `np` is numpy at every moment.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ebsd_utils import _capped_triplet_library, create_indexer  # noqa: E402

tv = pytest.importorskip("pyebsdindex.tripletvote")


def _nickel():
    from diffpy.structure import Atom, Lattice, Structure
    from diffsims.crystallography import ReciprocalLatticeVector
    from orix.crystal_map import Phase, PhaseList
    ph = Phase(name="Ni", space_group=225, structure=Structure(
        lattice=Lattice(3.524, 3.524, 3.524, 90, 90, 90), atoms=[Atom("Ni", [0, 0, 0])]))
    ref = ReciprocalLatticeVector(ph, hkl=[[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1]])
    return PhaseList(ph), ref


def _detector():
    import kikuchipy as kp
    return kp.detectors.EBSDDetector(shape=(60, 60), pc=(0.5, 0.5, 0.5), sample_tilt=70.0)


def test_the_module_numpy_is_numpy_while_the_cap_is_active():
    with _capped_triplet_library(10):
        assert tv.np is np, "numba compiling anything in tripletvote now would see a proxy"
    assert tv.np is np


def test_the_build_is_restored_afterwards():
    original = tv.BandIndexer.build_trip_lib
    with _capped_triplet_library(10):
        assert tv.BandIndexer.build_trip_lib is not original
    assert tv.BandIndexer.build_trip_lib is original


def test_the_cap_still_refuses_an_oversized_library(monkeypatch):
    # 1 KiB: smaller than any real library, so the build must be refused.
    monkeypatch.setenv("ORIENTA_HOUGH_LIBRARY_BUDGET_MB", "0.001")
    phases, ref = _nickel()
    with pytest.raises(MemoryError):
        create_indexer(_detector(), phases, ref)
    assert tv.BandIndexer.build_trip_lib.__globals__["np"] is np


def test_an_affordable_library_still_builds():
    phases, ref = _nickel()
    assert create_indexer(_detector(), phases, ref) is not None


def test_two_overlapping_caps_leave_the_pristine_build_behind():
    """Review finding: `original` was read BEFORE the lock. A second thread
    entering while the first held the cap read the first one's capped copy as
    its "original" and restored THAT on exit — the class kept a cap for the
    life of the backend, and every later Hough build was refused. The old
    module-`np` swap had the same hole."""
    import threading
    original = tv.BandIndexer.build_trip_lib
    a_inside, b_started = threading.Event(), threading.Event()

    def first():
        with _capped_triplet_library(5):
            a_inside.set()
            b_started.wait(5)
            import time
            time.sleep(0.2)          # B is now blocked on the lock

    def second():
        a_inside.wait(5)
        b_started.set()
        with _capped_triplet_library(10**9):
            pass

    ta, tb = threading.Thread(target=first), threading.Thread(target=second)
    ta.start(); tb.start(); ta.join(10); tb.join(10)
    assert tv.BandIndexer.build_trip_lib is original
    assert tv.BandIndexer.build_trip_lib.__globals__["np"] is np
