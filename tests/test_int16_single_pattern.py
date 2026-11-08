"""A single pattern an indexer is fed sees the values the map run sees for the same pixel.

Unprocessed AZtec patterns are int16 (about 600..6500). The map run rescales them to 0..255 by the
min/max of the WHOLE scan. Every single-pattern path (pattern-match variants, re-index, phase check,
phase test, PC refinement, quick test) used to hand the indexer the raw values, and the spherical
indexer answers 33-55 degrees differently for them (the Oxford reference scan, 6 of 9 patterns). So they read the pixel
through ``load_choices.signal_pattern``: the scan's range, not the pattern's own range (the indexer is
sensitive to the range), computed once per signal and dropped when the signal changes.

The value-range pass over a 27 GB scan takes minutes: it polls the cancel check, reports progress and
runs once per run.

Hand-written numbers: the fake scan below has min 100 and max 5100, the pattern at (0, 0) lies in
1000..1100, so the scan rescale gives 46 for 1000 (= 900 * 255 / 5000 = 45.9) and 51 for 1100, where
the pattern's own range would give 0 and 255.
"""
from __future__ import annotations

import asyncio
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import load_choices as lc  # noqa: E402
from backend.api.services.load_choices import to_uint8  # noqa: E402

H, W = 4, 5
NR, NC = 3, 4


def _scan(dtype=np.int16):
    data = np.full((NR, NC, H, W), 1000, dtype)
    data[0, 0, 0, 0] = 1100               # the pattern (0, 0) spans 1000..1100
    data[2, 3, 1, 1] = 100                # the scan's minimum, in another pattern
    data[2, 3, 2, 2] = 5100               # and its maximum
    return data


def _signal(dtype=np.int16):
    return types.SimpleNamespace(data=_scan(dtype))


def _expected_00():
    out = np.full((H, W), 46, np.uint8)   # 900 * 255 / 5000 = 45.9
    out[0, 0] = 51                        # 1000 * 255 / 5000
    return out


@pytest.fixture
def pass_counter(monkeypatch):
    """Counts the whole-array passes ``signed_pattern_range`` makes."""
    class Calls(list):
        kwargs: list

    calls = Calls()
    calls.kwargs = []
    real = lc.signed_pattern_range

    def counting(source, *a, **k):
        calls.append(tuple(source.shape))
        calls.kwargs.append(k)
        return real(source, *a, **k)

    monkeypatch.setattr(lc, "signed_pattern_range", counting)
    return calls


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #

def test_signal_pattern_is_rescaled_by_the_scan_not_by_its_own_range():
    got = lc.signal_pattern(_signal(), 0, 0)
    assert got.dtype == np.uint8
    assert np.array_equal(got, _expected_00()), got
    own = to_uint8(_scan()[0, 0])
    assert own.min() == 0 and own.max() == 255 and not np.array_equal(own, got)   # the premise


def test_signal_pattern_is_what_to_uint8_of_the_whole_scan_gives():
    sig = _signal()
    whole = to_uint8(sig.data)
    for r in range(NR):
        for c in range(NC):
            assert np.array_equal(lc.signal_pattern(sig, r, c), whole[r, c]), (r, c)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_signal_pattern_leaves_the_other_dtypes_byte_identical(dtype):
    data = (_scan() // 25).astype(dtype)
    got = lc.signal_pattern(types.SimpleNamespace(data=data), 1, 2)
    assert got.dtype == np.dtype(dtype) and got.tobytes() == data[1, 2].tobytes()


def test_the_scan_range_is_read_once_per_signal(pass_counter):
    sig = _signal()
    for r, c in ((0, 0), (1, 1), (2, 3), (0, 0)):
        lc.signal_pattern(sig, r, c)
    assert len(pass_counter) == 1, pass_counter
    lc.scan_value_range(sig)
    assert len(pass_counter) == 1


def test_a_replaced_array_a_crop_and_a_second_signal_are_never_served_a_stale_range(pass_counter):
    sig = _signal()
    assert lc.scan_value_range(sig) == (100.0, 5100.0)
    sig.data = sig.data * 2                               # NLPAR swaps the array
    assert lc.scan_value_range(sig) == (200.0, 10200.0)
    sig.data = sig.data[:2]                               # a crop without the rows holding the extremes
    assert lc.scan_value_range(sig) == (2000.0, 2200.0)
    other = _signal()
    other.data[...] = 7
    assert lc.scan_value_range(other) == (7.0, 7.0)       # its own, not the first one's
    assert len(pass_counter) == 4


def test_an_edit_in_place_is_seen_after_the_range_was_forgotten_and_not_before(pass_counter):
    sig = _signal()
    assert lc.scan_value_range(sig) == (100.0, 5100.0)
    sig.data[0, 0, 0, 0] = 9000                           # in place: the array object is the same
    assert lc.scan_value_range(sig) == (100.0, 5100.0), "premise: nothing can see an in-place edit"
    lc.forget_scan_range(sig)
    assert lc.scan_value_range(sig) == (100.0, 9000.0)


def test_marking_the_dataset_dirty_forgets_the_range(monkeypatch):
    """Every in-place edit of the viewer (background removal, averaging, autocontrast, CLAHE, NLPAR)
    ends in ``_mark_active_dirty``; that is where the range kept on the signal is dropped."""
    from backend.api.routes import ebsd_viewer as ev

    sig = _signal()
    monkeypatch.setattr(ev, "_active_dataset", "ds")
    monkeypatch.setitem(ev._raw_signals, "ds", sig)
    monkeypatch.setattr(ev, "_dirty_datasets", set())
    assert lc.scan_value_range(sig) == (100.0, 5100.0)
    sig.data[0, 0, 0, 0] = 9000
    ev._mark_active_dirty()
    assert lc.scan_value_range(sig) == (100.0, 9000.0)
    assert "ds" in ev._dirty_datasets


def test_marking_the_dataset_dirty_forgets_the_range_of_every_signal_that_shares_the_buffer(monkeypatch):
    """A crop of an eager signal is a view of its parent's array: background removal in place on the
    crop writes into the parent too, and the parent's kept range is as stale as the crop's."""
    from backend.api.routes import ebsd_viewer as ev

    parent = _signal()
    crop = types.SimpleNamespace(data=parent.data[:, :2])        # nav columns 0..1: neither extreme is in it
    monkeypatch.setattr(ev, "_active_dataset", "crop")
    monkeypatch.setattr(ev, "_raw_signals", {"parent": parent, "crop": crop})
    monkeypatch.setattr(ev, "_dirty_datasets", set())
    assert lc.scan_value_range(parent) == (100.0, 5100.0)
    assert lc.scan_value_range(crop) == (1000.0, 1100.0)
    crop.data[0, 0, 0, 0] = 9000                                 # the edit on the crop reaches the parent
    ev._mark_active_dirty()
    assert lc.scan_value_range(crop) == (1000.0, 9000.0)
    assert lc.scan_value_range(parent) == (100.0, 9000.0), "the parent kept the range from before the edit"


class _EditsThenFails:
    """A signal whose in-place operations write one value and then fail half way (an eager int16
    array is written in place chunk by chunk: a MemoryError or a cancelled worker leaves it half
    edited)."""

    def __init__(self):
        self.data = _scan()
        self.axes_manager = types.SimpleNamespace(navigation_shape=(NC, NR))

    def _half_edit(self, *a, **k):
        self.data[0, 0, 0, 0] = 9000
        raise MemoryError("out of memory half way")

    average_neighbour_patterns = remove_dynamic_background = rescale_intensity = _half_edit
    adaptive_histogram_equalization = _half_edit


@pytest.mark.parametrize("route,body", [
    ("frame-average", {"window_size": 3}),
    ("background-removal", {"method": "dynamic"}),
    ("autocontrast", None),
    ("clahe", {"kernel_size": 8}),
])
def test_a_failed_edit_in_place_still_forgets_the_range_and_marks_the_dataset_dirty(route, body, monkeypatch):
    """The four in-place routes mark the dataset dirty AFTER the operation, inside the try: an exception
    skipped it, and the half-edited array kept the range from before the edit, in a dataset not flagged
    as modified (indexing then streams the file instead of the viewer's patterns)."""
    from fastapi.testclient import TestClient

    from backend.api.main import app
    from backend.api.routes import ebsd_viewer as ev

    sig = _EditsThenFails()
    monkeypatch.setattr(ev, "_active_dataset", "ds")
    monkeypatch.setattr(ev, "_raw_signals", {"ds": sig})
    monkeypatch.setattr(ev, "_dirty_datasets", set())
    monkeypatch.setattr(ev, "_compute_include_mask", lambda *a, **k: None)
    assert lc.scan_value_range(sig) == (100.0, 5100.0)             # kept on the signal before the edit
    r = TestClient(app).post(f"/api/ebsd/{route}", json=body) if body is not None \
        else TestClient(app).post(f"/api/ebsd/{route}")
    assert r.status_code == 500, r.text
    assert sig.data[0, 0, 0, 0] == 9000, "premise: the edit half-wrote the array"
    assert "ds" in ev._dirty_datasets
    assert lc.scan_value_range(sig) == (100.0, 9000.0), "the range from before the failed edit was kept"


def test_a_pass_that_overlaps_an_edit_in_place_cannot_store_its_range(monkeypatch):
    """A click whose pass started before the operation ended has read half-edited values; the forget
    at the end of the operation must keep that result from being stored."""
    sig = _signal()
    real = lc.signed_pattern_range

    def overlapped(source, *a, **k):
        seen = real(source, *a, **k)                             # the pass has read the old values ...
        source[0, 0, 0, 0] = 9000                                # ... the operation writes on ...
        lc.forget_scan_range(sig)                                # ... and ends (what _mark_active_dirty does)
        return seen

    monkeypatch.setattr(lc, "signed_pattern_range", overlapped)
    assert lc.scan_value_range(sig) == (100.0, 5100.0)           # this click did see the old values
    monkeypatch.setattr(lc, "signed_pattern_range", real)
    assert lc.scan_value_range(sig) == (100.0, 9000.0), "the half-edited range was stored"


def test_a_pass_that_overlaps_nothing_is_stored(pass_counter):
    sig = _signal()
    lc.forget_scan_range(sig)                                    # an earlier edit, long finished
    lc.scan_value_range(sig)
    lc.scan_value_range(sig)
    assert len(pass_counter) == 1


# --------------------------------------------------------------------------- #
# The pass over a big scan: cancel and progress
# --------------------------------------------------------------------------- #

class _Counting:
    """An int16 'dataset' that counts the blocks read from it."""

    def __init__(self, data):
        self.data = data
        self.dtype = data.dtype
        self.shape = data.shape
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        return self.data[key]


class _Cancelled(Exception):
    pass


def test_the_range_pass_polls_cancel_before_every_block_and_reports_progress():
    src = _Counting(np.arange(10 * 3 * 2, dtype=np.int16).reshape(10, 3, 2))
    seen = []
    assert lc.signed_pattern_range(src, chunk=3, progress=lambda d, n: seen.append((d, n))) == (0.0, 59.0)
    assert seen == [(3, 10), (6, 10), (9, 10), (10, 10)]

    src = _Counting(src.data)
    polls = []

    def check():
        polls.append(len(src.reads))
        if len(polls) == 3:
            raise _Cancelled()

    with pytest.raises(_Cancelled):
        lc.signed_pattern_range(src, chunk=3, check=check)
    assert polls == [0, 1, 2] and len(src.reads) == 2, "cancel must stop the read, not wait for the end"


def test_the_range_pass_reads_blocks_of_bounded_size(monkeypatch):
    monkeypatch.setattr(lc, "_RANGE_BLOCK_BYTES", 2 * 3 * 2 * 2)       # two patterns of int16 3x2
    src = _Counting(np.zeros((9, 3, 2), np.int16))
    lc.signed_pattern_range(src, chunk=256)
    assert len(src.reads) == 5                                          # ceil(9 / 2)


RAW = lc.LoadVariant("1", "unprocessed")


def _oxford(tmp_path, data):
    """The fake scan as an Oxford file, one group, both pattern sources."""
    import h5py

    path = tmp_path / "scan.h5oina"
    flat = data.reshape(-1, H, W)
    with h5py.File(path, "w") as f:
        f.create_dataset("Manufacturer", data=np.bytes_("Oxford Instruments"))
        f.create_dataset("Format Version", data=np.bytes_("5.0"))
        hdr = f.create_group("1/EBSD/Header")
        for key, value in (("X Cells", NC), ("Y Cells", NR), ("Pattern Height", H), ("Pattern Width", W)):
            hdr.create_dataset(key, data=value)
        patterns = f.create_group("1/EBSD/Data")
        patterns.create_dataset("Processed Patterns", data=np.zeros_like(flat, np.uint8))
        patterns.create_dataset("Unprocessed Patterns", data=flat)
    return str(path)


def _lazy(data):
    da = pytest.importorskip("dask.array")
    return da.from_array(data, chunks=(1, 1, -1, -1))


def _unreadable_lazy():
    """A lazy array that fails when it is computed: a pass over it shows."""
    dask = pytest.importorskip("dask")
    import dask.array as da

    def read():
        raise AssertionError("the lazy array was read")

    return da.from_delayed(dask.delayed(read)(), shape=(NR, NC, H, W), dtype=np.int16)


def test_a_lazy_signal_takes_its_range_from_the_file(tmp_path):
    """One task per pattern: 970 patterns per second against 34000 from the file (485415 patterns of
    a real scan, 500 s against 14 s, past the client's 5 minutes). The numbers are the same."""
    path = _oxford(tmp_path, _scan())
    signal = types.SimpleNamespace(data=_unreadable_lazy())
    lc.note_file_source(signal, path, RAW)
    assert lc.scan_value_range(signal) == (100.0, 5100.0)


def test_the_file_is_read_with_cancel_and_progress(tmp_path):
    path = _oxford(tmp_path, _scan())
    signal = types.SimpleNamespace(data=_unreadable_lazy())
    lc.note_file_source(signal, path, RAW)
    seen = []

    def stop():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        lc.scan_value_range(signal, check=stop)
    lc.scan_value_range(signal, progress=lambda done, total: seen.append((done, total)))
    assert seen[-1] == (NR * NC, NR * NC)


def test_an_edit_or_a_new_array_keeps_the_signal_off_the_file(tmp_path):
    edited = _scan() + 7
    path = _oxford(tmp_path, _scan())
    signal = types.SimpleNamespace(data=_lazy(edited))
    lc.note_file_source(signal, path, RAW)
    assert lc.scan_value_range(signal) == (100.0, 5100.0)     # the file: nothing said the array changed
    lc.forget_scan_range(signal)
    assert lc.scan_value_range(signal) == (107.0, 5107.0)
    replaced = types.SimpleNamespace(data=_lazy(_scan()))
    lc.note_file_source(replaced, path, RAW)
    replaced.data = _lazy(edited)
    assert lc.scan_value_range(replaced) == (107.0, 5107.0)


@pytest.mark.parametrize("variant,kept", [(lc.LoadVariant("1", "processed"), slice(None)), (RAW, slice(0, 2))],
                         ids=["another-dtype", "another-size"])
def test_a_dataset_that_is_not_this_array_is_not_asked(tmp_path, variant, kept):
    path = _oxford(tmp_path, _scan())
    held = (_scan() + 7)[kept]
    signal = types.SimpleNamespace(data=_lazy(held))
    lc.note_file_source(signal, path, variant)
    assert lc.scan_value_range(signal) == (float(held.min()), float(held.max()))


def test_a_file_that_is_gone_is_not_a_failure(tmp_path):
    path = _oxford(tmp_path, _scan())
    signal = types.SimpleNamespace(data=_lazy(_scan() + 7))
    lc.note_file_source(signal, path, RAW)
    Path(path).unlink()
    assert lc.scan_value_range(signal) == (107.0, 5107.0)


def test_only_a_lazy_signed_signal_is_noted(tmp_path):
    path = _oxford(tmp_path, _scan())
    for data in (_scan(), _lazy(_scan().astype(np.uint16))):
        signal = types.SimpleNamespace(data=data)
        lc.note_file_source(signal, path, RAW)
        assert not hasattr(signal, "_orienta_file_source")
    signal = types.SimpleNamespace(data=_lazy(_scan()))
    lc.note_file_source(signal, str(tmp_path / "not-a-scan.up1"), None)      # no pattern dataset to name
    assert not hasattr(signal, "_orienta_file_source")


def test_without_a_variant_only_a_file_of_one_measurement_is_noted(tmp_path):
    """The reader takes its first group in HDF5 order, ``inspect_file`` in natural order: groups 2 and
    10 are not the same first group, and the range of the wrong one clips every pattern."""
    import h5py

    path = _oxford(tmp_path, _scan())
    signal = types.SimpleNamespace(data=_lazy(_scan()))
    lc.note_file_source(signal, path, None)                               # one measurement: the same group
    assert hasattr(signal, "_orienta_file_source")
    with h5py.File(path, "a") as f:
        f.copy("1", "2")
    signal = types.SimpleNamespace(data=_lazy(_scan()))
    lc.note_file_source(signal, path, None)
    assert not hasattr(signal, "_orienta_file_source")


def test_an_edit_on_one_dataset_leaves_the_file_source_of_the_others(tmp_path, monkeypatch):
    """The edit forgets every range, in case the buffer is shared. A lazy array is not shared, and
    without its source the next click takes the slow pass."""
    from backend.api.routes import ebsd_viewer as ev

    path = _oxford(tmp_path, _scan())
    edited, kept = types.SimpleNamespace(data=_lazy(_scan())), types.SimpleNamespace(data=_lazy(_scan()))
    lc.note_file_source(edited, path, RAW)
    lc.note_file_source(kept, path, RAW)
    monkeypatch.setattr(ev, "_raw_signals", {"edited": edited, "kept": kept})
    monkeypatch.setattr(ev, "_active_dataset", "edited")
    monkeypatch.setattr(ev, "_get_active_signal", lambda *a, **k: edited)
    monkeypatch.setattr(ev, "_dirty_datasets", set())
    ev._mark_active_dirty()
    assert getattr(edited, "_orienta_file_source", None) is None
    assert getattr(kept, "_orienta_file_source", None) is not None


def test_a_file_the_listing_chokes_on_is_read_through_the_array(tmp_path, monkeypatch):
    """A damaged group makes ``inspect_file`` raise more than ValueError. The load goes on without it."""
    def choke(path):
        raise TypeError("a dataset where a group belongs")

    monkeypatch.setattr(lc, "inspect_file", choke)
    signal = types.SimpleNamespace(data=_lazy(_scan()))
    lc.note_file_source(signal, str(tmp_path / "scan.h5oina"), None)
    assert not hasattr(signal, "_orienta_file_source")


def test_the_loader_notes_the_file(tmp_path, monkeypatch):
    import safe_loader

    path = _oxford(tmp_path, _scan())
    signal = types.SimpleNamespace(data=_unreadable_lazy())
    monkeypatch.setattr(safe_loader, "_load_ebsd_safe_impl", lambda *a, **k: signal)
    monkeypatch.setattr(safe_loader, "_tag_pc_less_formats", lambda *a, **k: None)
    loaded = safe_loader.load_ebsd_safe(path, verbose=False, variant=RAW)
    assert lc.scan_value_range(loaded) == (100.0, 5100.0)


def _dataset_in(tmp_path, data, name="1/EBSD/Data/Unprocessed Patterns"):
    import h5py

    path = tmp_path / "scan.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset(name, data=data)
    return path, name


def test_the_range_of_a_file_is_remembered_by_file_size_and_time_and_dataset(tmp_path, monkeypatch):
    import os

    import h5py

    lc._DATASET_RANGES.clear()
    path, name = _dataset_in(tmp_path, np.arange(24, dtype=np.int16).reshape(6, 2, 2))
    calls = []
    real = lc.signed_pattern_range
    monkeypatch.setattr(lc, "signed_pattern_range",
                        lambda s, *a, **k: calls.append(1) or real(s, *a, **k))
    with h5py.File(path, "r") as f:
        assert lc.dataset_pattern_range(f[name]) == (0.0, 23.0)
        assert lc.dataset_pattern_range(f[name]) == (0.0, 23.0)
    assert len(calls) == 1
    with h5py.File(path, "r+") as f:                                     # the file changes on disk
        f[name][0, 0, 0] = 500
    os.utime(path, ns=(1, 1_000_000_000_000))
    with h5py.File(path, "r") as f:
        assert lc.dataset_pattern_range(f[name]) == (1.0, 500.0)
    assert len(calls) == 2


class _LockedOnly(dict):
    """A dict that fails every access (lookup, eviction, store, size) made without ``lc._DATASET_RANGES_LOCK``."""

    def _held(self):
        assert lc._DATASET_RANGES_LOCK.locked(), "the range memory was touched without its lock"

    def __contains__(self, key):
        self._held()
        return super().__contains__(key)

    def __getitem__(self, key):
        self._held()
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        self._held()
        super().__setitem__(key, value)

    def __len__(self):
        self._held()
        return super().__len__()

    def __iter__(self):
        self._held()
        return super().__iter__()

    def pop(self, *a):
        self._held()
        return super().pop(*a)


def test_the_memory_of_file_ranges_is_only_touched_under_its_lock_and_the_pass_runs_outside_it(tmp_path, monkeypatch):
    """Two runs on different files can evict from and store into the memory at the same moment;
    the pass over a 27 GB file takes minutes and must not hold the lock the other run waits on."""
    import h5py

    monkeypatch.setattr(lc, "_DATASET_RANGES", _LockedOnly())
    monkeypatch.setattr(lc, "_DATASET_RANGES_MAX", 2)
    held_during_pass = []
    real = lc.signed_pattern_range
    monkeypatch.setattr(lc, "signed_pattern_range",
                        lambda s, *a, **k: held_during_pass.append(lc._DATASET_RANGES_LOCK.locked()) or real(s, *a, **k))
    paths = []
    for i in range(4):                                   # four files into a memory of two: two evictions
        path = tmp_path / f"scan{i}.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("d", data=np.full((3, 2, 2), i, np.int16))
        paths.append(path)
    for i, path in enumerate(paths):
        with h5py.File(path, "r") as f:
            assert lc.dataset_pattern_range(f["d"]) == (float(i), float(i))
    with h5py.File(paths[3], "r") as f:
        assert lc.dataset_pattern_range(f["d"]) == (3.0, 3.0)         # the newest is still remembered
    assert held_during_pass == [False] * 4, "the range pass ran under the lock, or ran again for a remembered file"
    assert dict.__len__(lc._DATASET_RANGES) == 2


# --------------------------------------------------------------------------- #
# The backend: once per run, on every path, cancellable
# --------------------------------------------------------------------------- #

class _Stub:
    """A phase indexer that records what it is handed and answers zeros."""

    def __init__(self, circmask=0):
        self.circmask = circmask
        self.gausbckg_on, self.nregions_n, self.normed, self.bandwidth = True, 10, True, 68
        self.seen = []
        import torch

        self.device = torch.device("cpu")

    def _record(self, pat):
        self.seen.append(np.asarray(getattr(pat, "arr", pat)).copy())
        b = self.seen[-1].shape[0]
        import torch
        return torch.zeros(b, 3), torch.ones(b)

    def _index_batch(self, pat):
        return self._record(pat)

    def _run_preprocessing(self, pat):
        return pat

    def _index_prepped_batch(self, prep):
        return self._record(prep)

    def _open_pattern_dataset(self, path, dataset_path=None):
        from backend.spherical_gpu.pipeline.indexer import Tier1Indexer

        return Tier1Indexer._open_pattern_dataset(self, path, dataset_path)

    def index_h5oina(self, *a, **k):              # the real streaming loop, on this stub
        from backend.spherical_gpu.pipeline.indexer import Tier1Indexer

        return Tier1Indexer.index_h5oina(self, *a, **k)


class _Tensorish:
    def __init__(self, arr):
        self.arr = arr

    def to(self, device):
        return self


def _backend(n_phases, monkeypatch, *, batch=2, cuda=False):
    import torch

    from backend.spherical_gpu import backend as be

    cfg = be.BackendConfig(phases=[be.PhaseConfig(sht_file=f"p{i}.sht", name=f"P{i}") for i in range(n_phases)],
                           device=torch.device("cpu"), batch_size=batch)
    backend = be.SphericalGPUBackend(cfg)
    backend._indexers = [_Stub(circmask=i) for i in range(n_phases)]       # differing -> per-phase loop
    if cuda:
        backend._device = types.SimpleNamespace(type="cuda")
        monkeypatch.setattr(torch, "from_numpy", lambda a: _Tensorish(a))
    return backend, be


@pytest.fixture
def range_passes(monkeypatch):
    from backend.spherical_gpu import backend as be

    calls = []
    real = be.signed_pattern_range

    def counting(source, *a, **k):
        calls.append(1)
        return real(source, *a, **k)

    monkeypatch.setattr(be, "signed_pattern_range", counting)
    return calls


def _scan_stack():
    return _scan().reshape(NR * NC, H, W)


@pytest.mark.parametrize("n_phases,same_preproc", [(1, True), (2, True), (2, False)],
                         ids=["single", "interleaved", "per_phase_fallback"])
def test_the_range_is_read_once_per_run_on_every_path(n_phases, same_preproc, monkeypatch, range_passes):
    backend, _ = _backend(n_phases, monkeypatch)
    if same_preproc:
        for ix in backend._indexers:
            ix.circmask = 0
    stack = _scan_stack()
    backend._index_with_phases("array", stack, None)
    assert len(range_passes) == 1, "the range pass ran more than once in a run"
    whole = to_uint8(stack).astype(np.float32)
    for ix in backend._indexers:
        assert np.array_equal(np.concatenate(ix.seen), whole), "a block was rescaled by its own range"


def test_an_oom_retry_does_not_read_the_range_again(monkeypatch, range_passes):
    from backend.spherical_gpu import backend as be

    backend, _ = _backend(1, monkeypatch, cuda=True)
    attempts = []

    def retry_once(fn, batch_size, **k):
        attempts.append(batch_size)
        fn(batch_size)                                   # the attempt that "ran out of memory"
        return fn(max(1, batch_size // 2))

    monkeypatch.setattr(be, "run_with_oom_retry", retry_once)
    backend._index_with_phases("array", _scan_stack(), None)
    assert len(attempts) == 1 and len(range_passes) == 1


def test_a_cancel_during_the_range_pass_ends_the_run_the_normal_way_before_any_batch(monkeypatch):
    from indexing_controller import CancelledIndexingError

    monkeypatch.setattr(lc, "_RANGE_BLOCK_BYTES", 2 * H * W * 2)       # two patterns a block: six blocks
    backend, _ = _backend(2, monkeypatch)
    src = _Counting(_scan_stack())
    backend.set_cancel_check(lambda: len(src.reads) >= 1)
    with pytest.raises(CancelledIndexingError):
        backend._index_with_phases("array", src, None)
    assert len(src.reads) == 1, "the pass went on after the cancel"
    assert all(not ix.seen for ix in backend._indexers), "a batch was indexed after the cancel"


def test_the_range_pass_reports_progress_through_the_run_callback(monkeypatch):
    monkeypatch.setattr(lc, "_RANGE_BLOCK_BYTES", 2 * H * W * 2)       # two patterns a block: six blocks
    backend, _ = _backend(1, monkeypatch)
    messages = []
    backend._index_with_phases("array", _scan_stack(), lambda msg, frac: messages.append((msg, frac)))
    reading = [m for m, _ in messages if "value range" in m]
    n = NR * NC
    assert reading[0].endswith("..."), reading
    assert [m.rsplit("(", 1)[-1] for m in reading[1:]] == [f"{k}/{n})" for k in (2, 4, 6, 8, 10, 12)], reading
    first_batch = next(i for i, (m, _) in enumerate(messages) if m.startswith("Indexed "))
    assert all("value range" in m for m, _ in messages[:first_batch])
    # The bar belongs to the indexing: the reading reports 0.0 (a bar that jumped to done/total of the
    # reading would run to the end, fall back to 0 for the first batch and climb again), and the
    # fractions never fall afterwards.
    assert [f for m, f in messages if "value range" in m] == [0.0] * len(reading)
    fractions = [f for _, f in messages]
    assert fractions == sorted(fractions), fractions
    assert fractions[first_batch:] == pytest.approx([k / n for k in (2, 4, 6, 8, 10, 12)])


@pytest.mark.parametrize("kind,name", [("h5_dataset", "1/EBSD/Data/Unprocessed Patterns"), ("h5", "1/EBSD/Data/Patterns")])
def test_a_file_range_that_is_remembered_posts_no_reading_message(kind, name, tmp_path, monkeypatch):
    """The 'Reading the value range...' line belongs to a pass that really runs: a second run on the
    same file answers from the memory in an instant and must not announce a read."""
    lc._DATASET_RANGES.clear()
    path, _ = _dataset_in(tmp_path, _scan_stack(), name=name)
    payload = (str(path), name) if kind == "h5_dataset" else str(path)
    backend, _ = _backend(1, monkeypatch, batch=4)
    for _run in range(2):
        backend._indexers[0].seen.clear()
        said = []
        backend._index_with_phases(kind, payload, lambda msg, frac: said.append(msg))
        reading = [m for m in said if "value range" in m]
        if _run == 0:
            assert reading[0].endswith("...") and reading[-1].endswith("(12/12)"), said
        else:
            assert reading == [], said
        assert any(m.startswith("Indexed ") for m in said), "premise: the run itself went on"
    assert len(lc._DATASET_RANGES) == 1


def test_a_streamed_file_cancels_during_the_pass_and_reuses_it_on_the_next_run(tmp_path, monkeypatch, pass_counter):
    from indexing_controller import CancelledIndexingError

    lc._DATASET_RANGES.clear()
    path, name = _dataset_in(tmp_path, _scan_stack())
    backend, _ = _backend(1, monkeypatch, batch=4)
    backend.set_cancel_check(lambda: True)
    with pytest.raises(CancelledIndexingError):
        backend._index_with_phases("h5_dataset", (str(path), name), None)
    assert not backend._indexers[0].seen
    (kw,) = pass_counter.kwargs
    assert callable(kw["check"]) and callable(kw["progress"]), "the streamed pass must poll cancel and report progress"
    with pytest.raises(CancelledIndexingError):
        kw["check"]()

    backend.set_cancel_check(None)
    backend._index_with_phases("h5_dataset", (str(path), name), None)
    whole = to_uint8(_scan_stack()).astype(np.float32)
    assert np.array_equal(np.concatenate(backend._indexers[0].seen), whole)
    assert len(lc._DATASET_RANGES) == 1, "the streamed pass is remembered per file"


@pytest.mark.parametrize("n_phases", [1, 2], ids=["single", "interleaved"])
def test_a_whole_file_run_reads_the_range_once_and_a_cancel_stops_the_pass(
        n_phases, tmp_path, monkeypatch, pass_counter):
    from indexing_controller import CancelledIndexingError

    lc._DATASET_RANGES.clear()
    path, _ = _dataset_in(tmp_path, _scan_stack(), name="1/EBSD/Data/Patterns")
    backend, _ = _backend(n_phases, monkeypatch, batch=4)
    for ix in backend._indexers:
        ix.circmask = 0
    backend.set_cancel_check(lambda: True)
    with pytest.raises(CancelledIndexingError):
        backend._index_with_phases("h5", str(path), None)
    assert all(not ix.seen for ix in backend._indexers)
    (kw,) = pass_counter.kwargs
    assert callable(kw["check"]) and callable(kw["progress"]), "the pass over the file must poll cancel and report progress"

    lc._DATASET_RANGES.clear()
    pass_counter.clear()
    backend.set_cancel_check(None)
    backend._index_with_phases("h5", str(path), None)
    assert len(pass_counter) == 1
    whole = to_uint8(_scan_stack()).astype(np.float32)
    for ix in backend._indexers:
        assert np.array_equal(np.concatenate(ix.seen), whole)


def test_the_tier1_indexer_cancels_its_own_range_pass_and_takes_a_range_it_is_given(tmp_path, pass_counter):
    import torch

    from backend.spherical_gpu.pipeline.indexer import Tier1Indexer
    from indexing_controller import CancelledIndexingError

    lc._DATASET_RANGES.clear()
    path, _ = _dataset_in(tmp_path, _scan_stack(), name="1/EBSD/Data/Patterns")
    seen = []

    def index_batch(pat):
        seen.append(np.asarray(pat).copy())
        return torch.zeros(pat.shape[0], 3), torch.ones(pat.shape[0])

    me = types.SimpleNamespace(
        _open_pattern_dataset=lambda p, dataset_path=None: Tier1Indexer._open_pattern_dataset(
            None, p, dataset_path),
        device=torch.device("cpu"), _index_batch=index_batch)
    with pytest.raises(CancelledIndexingError):
        Tier1Indexer.index_h5oina(me, str(path), batch_size=4, cancel_check=lambda: True)
    assert not seen
    (kw,) = pass_counter.kwargs
    assert callable(kw["check"]), "the pass of the streaming loop must poll the cancel check"
    with pytest.raises(CancelledIndexingError):
        kw["check"]()

    pass_counter.clear()
    Tier1Indexer.index_h5oina(me, str(path), batch_size=4, value_range=(100.0, 5100.0))
    assert len(pass_counter) == 0, "a range that was handed in must not be read again"
    assert np.array_equal(np.concatenate(seen), to_uint8(_scan_stack()).astype(np.float32))


# --------------------------------------------------------------------------- #
# The single-pattern readers
# --------------------------------------------------------------------------- #

def _result_with(sig):
    return types.SimpleNamespace(metadata={"signal": sig})


def test_get_experimental_pattern_for_the_indexer_is_the_scan_rescale():
    from tools.pattern_comparison import get_experimental_pattern

    res = _result_with(_signal())
    raw = get_experimental_pattern(res, 0, 0)
    assert raw.dtype == np.float32 and np.array_equal(raw, _scan()[0, 0].astype(np.float32)), "the default is unchanged"
    idx = get_experimental_pattern(res, 0, 0, for_indexer=True)
    assert idx.dtype == np.float32 and np.array_equal(idx, _expected_00().astype(np.float32))


def test_get_experimental_pattern_for_the_indexer_on_a_signal_read_through_inav():
    """The fallback for signals whose ``data[row, col]`` fails reads ``inav[col, row]``: also rescaled."""
    from tools.pattern_comparison import get_experimental_pattern

    full = _scan()

    class _Data:                         # `data[row, col]` fails, a slice of rows works (the range pass)
        shape, dtype, ndim = full.shape, full.dtype, full.ndim

        def __getitem__(self, key):
            if isinstance(key, tuple):
                raise IndexError("no")
            return full[key]

    class _Inav:
        def __getitem__(self, key):
            col, row = key
            return types.SimpleNamespace(data=full[row, col])

    sig = types.SimpleNamespace(data=_Data(), inav=_Inav())
    idx = get_experimental_pattern(_result_with(sig), 0, 0, for_indexer=True)
    assert np.array_equal(idx, _expected_00().astype(np.float32))


# --- the quick test and the selected-pixel Hough ---------------------------------------------------

def _nickel_int16():
    from tests.test_int16_patterns import _int16_nickel

    return _int16_nickel()


class _SeenHough(Exception):
    pass


@pytest.fixture
def hough_input(monkeypatch):
    """The array every ``EBSD.hough_indexing`` is handed, recorded (the call then stops)."""
    import kikuchipy.signals as ks

    seen = []

    def rec(self, *a, **k):
        seen.append(np.asarray(self.data).copy())
        raise _SeenHough()

    monkeypatch.setattr(ks.EBSD, "hough_indexing", rec)
    return seen


def _scan_rescale(sig, r, c):
    own = to_uint8(np.asarray(sig.data[r, c]))
    whole = to_uint8(np.asarray(sig.data))[r, c]
    assert not np.array_equal(own, whole), "premise: this pattern's range differs from the scan's"
    return whole


def test_the_quick_test_indexes_the_pattern_as_the_map_run_sees_it(monkeypatch, hough_input):
    import ebsd_utils
    import indexing_controller as ic

    sig = _nickel_int16()
    monkeypatch.setattr(ebsd_utils, "prepare_reflectors", lambda *a, **k: None)
    monkeypatch.setattr(ebsd_utils, "create_indexer", lambda *a, **k: None)
    with pytest.raises(_SeenHough):
        ic.quick_test_single_pattern(sig, sig.detector, object(), 1, 2)
    assert np.array_equal(np.squeeze(hough_input[0]), _scan_rescale(sig, 1, 2))


def test_the_selected_pixel_hough_indexes_the_pattern_as_the_map_run_sees_it(hough_input):
    from orix.crystal_map import PhaseList

    import indexing_controller as ic

    sig = _nickel_int16()
    ic._index_selected_hough(sig, [(1, 2)], PhaseList(), None, sig.detector)   # the failed pixel is skipped
    assert np.array_equal(np.squeeze(hough_input[0]), _scan_rescale(sig, 1, 2))


def test_the_pc_calibration_points_are_optimised_on_the_pattern_as_the_map_run_sees_it(tmp_path, monkeypatch):
    import pc_controller
    from pc_controller import PCController
    from tests.test_pc_per_pixel_single_pattern import _ni_cif

    sig = _nickel_int16()
    ctrl = PCController()
    ctrl.load_phase(_ni_cif(tmp_path))
    ctrl.attach_detector(sig.detector.deepcopy())
    handed = []

    def fake_optimize(detector, indexer, pattern, **k):
        handed.append(np.asarray(pattern).copy())
        return (0.5, 0.5, 0.5), None

    monkeypatch.setattr(pc_controller, "optimize_pc", fake_optimize)
    ctrl.refine_pc_at_points(sig, [(1, 2)])
    assert np.array_equal(handed[0], _scan_rescale(sig, 1, 2))


# --- the PC refinement routes ------------------------------------------------------------------------

@pytest.fixture
def pc_routes(monkeypatch):
    import kikuchipy as kp

    from backend.api.routes import ebsd_viewer as ev
    from backend.api.routes import pcrefinement as pcr
    from pc_controller import PCController

    sig = _nickel_int16()
    ctrl = PCController()
    ctrl.detector = object()
    ctrl.phase = object()
    monkeypatch.setattr(ev, "_get_active_signal", lambda: sig)
    monkeypatch.setattr(pcr, "_get_controller", lambda: ctrl)
    monkeypatch.setattr(pcr, "_auto_attach_detector", lambda c: None)
    assert kp is not None
    return types.SimpleNamespace(sig=sig, ctrl=ctrl, pcr=pcr)


def test_a_calibration_pattern_is_stored_as_the_map_run_sees_it(pc_routes):
    asyncio.run(pc_routes.pcr.add_pattern(pc_routes.pcr.AddPatternRequest(row=1, col=2)))
    stored = pc_routes.ctrl.patterns[0][1]
    assert stored.dtype == np.uint8
    assert np.array_equal(stored, _scan_rescale(pc_routes.sig, 1, 2))


def test_indexing_a_pattern_outside_the_calibration_set_uses_the_map_run_values(pc_routes, monkeypatch):
    seen = []
    monkeypatch.setattr(pc_routes.pcr, "_index_and_simulate",
                        lambda ctrl, idx, *a, **k: seen.append(np.asarray(ctrl.patterns[idx][1]).copy()) or {})
    asyncio.run(pc_routes.pcr.index_pattern(pc_routes.pcr.IndexPatternRequest(row=1, col=2)))
    assert np.array_equal(seen[0], _scan_rescale(pc_routes.sig, 1, 2))


def test_the_first_index_click_reads_the_scan_range_outside_the_state_lock(pc_routes, monkeypatch):
    """On a big lazy scan the first click spends minutes in the range pass; every other PC route
    waits for ``_state_lock`` meanwhile if the pass runs under it."""
    held = []
    real = lc.signed_pattern_range
    monkeypatch.setattr(lc, "signed_pattern_range",
                        lambda s, *a, **k: held.append(pc_routes.pcr._state_lock.locked()) or real(s, *a, **k))
    seen = []
    monkeypatch.setattr(pc_routes.pcr, "_index_and_simulate",
                        lambda ctrl, idx, *a, **k: seen.append(np.asarray(ctrl.patterns[idx][1]).copy()) or {})
    asyncio.run(pc_routes.pcr.index_pattern(pc_routes.pcr.IndexPatternRequest(row=1, col=2)))
    assert held == [False], "the range pass ran under the state lock (or not exactly once)"
    assert np.array_equal(seen[0], _scan_rescale(pc_routes.sig, 1, 2)), "premise: the click got the map run values"


def test_the_optimisation_gets_the_map_run_values(pc_routes, monkeypatch):
    # The stub below never runs the worker that takes the flag down again.
    monkeypatch.setattr(pc_routes.pcr, "_optimization_active", False)
    # A refine is checked against the PC the page holds before a job exists: it needs a real one.
    pc_routes.ctrl.detector = pc_routes.sig.detector.deepcopy()
    started = []
    tasks = types.SimpleNamespace(add_task=lambda fn, *a, **k: started.append(a))
    n_cols = pc_routes.sig.data.shape[1]
    asyncio.run(pc_routes.pcr.optimize_pc(pc_routes.pcr.OptimizePCRequest(patterns=[1 * n_cols + 2]), tasks))
    patterns_data = started[0][1]
    assert np.array_equal(patterns_data[0], _scan_rescale(pc_routes.sig, 1, 2))


# --- the phase test -------------------------------------------------------------------------------

class _Entry:
    def __init__(self, key, sht):
        self.key = key
        self.formula = self.display_formula = key
        self.elements = ()
        self.space_group = ""
        self.crystal_system = "cubic"
        self.sht_path = sht


@pytest.fixture
def phase_test(monkeypatch, tmp_path):
    import backend.api.routes.indexing as idx

    sht = tmp_path / "Al.sht"
    sht.write_bytes(b"x")
    sig = _signal()
    seen = []
    monkeypatch.setattr(idx, "_get_active_signal_for_phase_test", lambda: sig, raising=False)
    monkeypatch.setattr(idx, "_get_detector_for_phase_test", lambda *a, **k: (
        types.SimpleNamespace(pc_source="header"),
        {"pat_height": H, "pat_width": W, "n_cols": NC, "n_rows": NR, "pc_x": 0.5, "pc_y": 0.5, "pc_z": 0.6,
         "pixel_size": 50.0, "binning": 1, "vendor": "Bruker", "sample_tilt": 70.0, "tilt": 10.0}), raising=False)
    monkeypatch.setattr(idx, "get_pixel_at_pct_for_phase_test", lambda i: {"Al": 100.0}, raising=False)
    monkeypatch.setattr(idx, "_phase_test_library_entries", lambda keys: ([_Entry("Al", sht)], []), raising=False)

    def fake_compare(**kw):
        seen.append(np.asarray(kw["exp_pattern"]).copy())
        return [{"phase_id": 0, "phase_name": "Al", "r_score": 0.6, "ncc_score": 0.7, "euler_deg": [1, 2, 3],
                 "simulated_b64": "S", "ncc_diff_b64": "D", "rank": 1}]

    monkeypatch.setattr(idx, "_compute_phase_compare_results", fake_compare)
    return seen


def test_the_single_pixel_phase_test_indexes_the_pattern_as_the_map_run_sees_it(phase_test):
    from fastapi.testclient import TestClient

    from backend.api.main import app

    r = TestClient(app).post("/api/indexing/single-pixel-phase-test",
                             json={"pixel_index": 0, "bg_remove": False, "eds_weighting": "off"})
    assert r.status_code == 200, r.text
    assert np.array_equal(phase_test[0], _expected_00().astype(np.float32))


def test_the_phase_test_job_indexes_the_pattern_as_the_map_run_sees_it(phase_test):
    """The route starts the job with ``asyncio.create_task``; a ``TestClient`` request closes its event
    loop when the response is out, and a job that has not finished by then is dropped (status stays
    "running", whatever the machine's load). So the handler is called inside one loop that also
    waits for the job."""
    import backend.api.routes.indexing as idx

    async def drive():
        started = await idx.single_pixel_phase_test_start(
            idx.SinglePixelPhaseTestRequest(pixel_index=0, bg_remove=False, eds_weighting="off"))
        job = idx._phase_test_jobs[started["job_id"]]
        deadline = time.monotonic() + 120.0
        while job["status"] == "running" and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        return job

    job = asyncio.run(drive())
    assert job["status"] == "done", job
    assert np.array_equal(phase_test[0], _expected_00().astype(np.float32))


# --- the pattern-match dialog: variants, re-index, compare phases ---------------------------------

def _two_grain_result_with_scan():
    from tests.test_grain_flip import _fake_two_grain_result

    result, _qa, _qb = _fake_two_grain_result()
    data = np.full((8, 8, 8, 8), 1000, np.int16)
    data[0, 0, 0, 0] = 1100
    data[7, 7, 1, 1] = 100
    data[7, 7, 2, 2] = 5100
    result.metadata["signal"] = types.SimpleNamespace(data=data)
    return result, data


def _scan_00(data):
    return to_uint8(data)[0, 0].astype(np.float32)


@pytest.fixture
def match_seams(monkeypatch):
    import base64
    import io

    from PIL import Image

    import backend.api.routes.indexing as ix
    import backend.api.services.sht_pattern_renderer as shtr
    import indexing_controller as ic

    png = io.BytesIO()
    Image.fromarray((np.random.default_rng(3).random((8, 8)) * 255).astype(np.uint8)).save(png, format="PNG")
    monkeypatch.setattr(shtr, "render_pattern_to_png_b64", lambda **k: base64.b64encode(png.getvalue()).decode())
    monkeypatch.setattr(ic, "_resolve_cif_for_sht", lambda *a, **k: "x.cif")
    seen = {"hough": [], "compare": []}
    monkeypatch.setattr(ix, "_hough_euler_for_phase", lambda exp, *a, **k: seen["hough"].append(np.asarray(exp).copy()))
    monkeypatch.setattr(ix, "_compute_phase_compare_results",
                        lambda **k: seen["compare"].append(np.asarray(k["exp_pattern"]).copy()) or [])
    return ix, seen


def test_the_variants_hough_candidate_is_indexed_on_the_map_run_values(match_seams, monkeypatch):
    ix, seen = match_seams
    result, data = _two_grain_result_with_scan()
    monkeypatch.setattr(ix, "_get_result", lambda *a, **k: result)
    asyncio.run(ix.get_pattern_match_variants(row=0, col=0))
    assert len(seen["hough"]) == 1
    assert np.array_equal(seen["hough"][0], _scan_00(data))


def test_the_variants_reindex_candidate_is_indexed_on_the_map_run_values(match_seams, monkeypatch):
    ix, seen = match_seams
    result, data = _two_grain_result_with_scan()
    monkeypatch.setattr(ix, "_get_result", lambda *a, **k: result)
    asyncio.run(ix.get_pattern_match_variants(row=0, col=0, reindex=True))
    assert len(seen["compare"]) == 1
    assert np.array_equal(seen["compare"][0], _scan_00(data))


def test_the_compare_phases_reindex_of_the_dialog_is_indexed_on_the_map_run_values(match_seams, monkeypatch):
    ix, seen = match_seams
    result, data = _two_grain_result_with_scan()
    monkeypatch.setattr(ix, "_get_result", lambda *a, **k: result)
    asyncio.run(ix.get_pattern_match(row=0, col=0, compare_phases=True))
    assert len(seen["compare"]) == 1
    assert np.array_equal(seen["compare"][0], _scan_00(data))


def test_the_dialog_still_compares_the_raw_pattern(match_seams, monkeypatch):
    """Only what goes into an indexer is rescaled: the NCC and the picture keep the raw values."""
    ix, _ = match_seams
    result, data = _two_grain_result_with_scan()
    monkeypatch.setattr(ix, "_get_result", lambda *a, **k: result)
    from tools import pattern_comparison as pcmp

    got = []
    real = pcmp.get_experimental_pattern

    def spy(res, r, c, **k):
        out = real(res, r, c, **k)
        got.append((bool(k.get("for_indexer")), out))
        return out

    monkeypatch.setattr(pcmp, "get_experimental_pattern", spy)
    asyncio.run(ix.get_pattern_match(row=0, col=0, compare_phases=True))
    raw = [o for flag, o in got if not flag]
    assert raw and np.array_equal(raw[0], data[0, 0].astype(np.float32))


# --- the phase check ----------------------------------------------------------------------------

def test_the_phase_check_hough_candidate_is_indexed_on_the_map_run_values(monkeypatch):
    import backend.api.routes.indexing as ix
    import indexing_controller as ic
    from backend.spherical_gpu.pipeline import variant_unification as vu

    result, data = _two_grain_result_with_scan()
    result.metadata["indexing_method"] = "spherical"
    seen = []
    monkeypatch.setattr(ic, "_resolve_cif_for_sht", lambda *a, **k: "x.cif")
    monkeypatch.setattr(vu, "build_render_score_fn", lambda *a, **k: (lambda flats, quats: np.zeros(len(flats))))
    monkeypatch.setattr(ix, "_hough_quats_for_phase_batch",
                        lambda pats, cif, det, err_out=None: seen.append(np.asarray(pats).copy())
                        or np.tile([1.0, 0, 0, 0], (len(pats), 1)))
    ctx = ix._phase_check_ctx(result, build_scorers=False)
    ctx["hough_quats_fn"](0, [0, 1])
    assert len(seen) == 1
    assert np.array_equal(seen[0][0], _scan_00(data))
    assert np.array_equal(seen[0][1], to_uint8(data)[0, 1].astype(np.float32))


# --- the viewer's circular mask on an int16 map run ----------------------------------------------------------

from tests.test_hex_indexing_paths import fake  # noqa: E402,F401  (fixture: a backend that records what it is handed)


def _masked_map_run(fake, monkeypatch, tmp_path, selection=None):
    """The patterns ``index_array`` is handed by a map run on the int16 scan of ``_scan`` with the
    viewer mask on: one pixel, (0, 0) of every pattern, is outside it."""
    from backend.api.routes import ebsd_viewer
    from tests import test_hex_indexing_paths as hx

    hx._active_signal(monkeypatch, _scan())
    include = np.ones((H, W), bool)
    include[0, 0] = False
    monkeypatch.setattr(ebsd_viewer, "get_active_include_mask", lambda: include)
    with pytest.raises(hx._Stop):
        hx._run(hx._up_file(tmp_path), selection)
    handed = fake.instances[0].arrays[0]
    # what the real backend then makes of them (`_range_of` / `patterns_to_float32`)
    return lc.patterns_to_float32(handed, lc.signed_pattern_range(handed)), include


def _hand_00():
    """Pattern (0, 0) as the map run must see it: the scan rescale of ``_expected_00`` (1000 -> 46,
    1100 -> 51, by the scan's 100..5100) with the pixel outside the mask zeroed."""
    out = _expected_00().astype(np.float32)
    out[0, 0] = 0.0
    return out


def test_the_viewer_mask_does_not_move_the_range_a_map_run_rescales_an_int16_scan_by(fake, monkeypatch, tmp_path):
    """Zeroing the pixels outside the mask BEFORE the range is read would make the scan's minimum 0
    (100 -> 0): every value the indexer sees moves (1000 -> 50 instead of 46)."""
    seen, include = _masked_map_run(fake, monkeypatch, tmp_path)
    assert seen.shape == (NR * NC, H, W)
    assert np.array_equal(seen[0], _hand_00()), seen[0]
    last = np.full((H, W), 46, np.float32)                  # pattern (2, 3): 1000 except the two extremes
    last[1, 1], last[2, 2], last[0, 0] = 0.0, 255.0, 0.0      # 100 -> 0, 5100 -> 255, the pixel outside the mask
    assert np.array_equal(seen[NR * NC - 1], last), seen[NR * NC - 1]


def test_a_roi_run_with_the_viewer_mask_sees_the_values_a_click_sees(fake, monkeypatch, tmp_path):
    """The click path (``signal_pattern``) has no mask; inside the mask the two must agree exactly."""
    selection = np.zeros((NR, NC), bool)
    selection[0, 0] = selection[2, 3] = True                 # the ROI holds both extremes of the scan
    seen, include = _masked_map_run(fake, monkeypatch, tmp_path, selection)
    assert seen.shape == (2, H, W)
    assert np.array_equal(seen[0], _hand_00())
    click = lc.signal_pattern(_signal(), 0, 0).astype(np.float32)
    assert np.array_equal(seen[0][include], click[include]), "the map run and the click disagree inside the mask"


# --- a run on a selection rescales by the range of the WHOLE scan, as a click does -----------------------------
#
# ``_scan()`` holds its minimum (100) and its maximum (5100) in pixel (2, 3). The selection below is two
# pixels that hold neither: (0, 0) spans 1000..1100 and (1, 1) is all 1000. Rescaled by its own range the
# selection would read 0..255 (1000 -> 0, 1100 -> 255); by the scan's it reads 46 and 51 (hand-counted above).

SELECTION = [(0, 0), (1, 1)]                       # flat pattern numbers 0 and 1 * 4 + 1 = 5


def _selection_mask():
    mask = np.zeros((NR, NC), bool)
    for r, c in SELECTION:
        mask[r, c] = True
    return mask


def _all_46():
    return np.full((H, W), 46, np.float32)


@pytest.fixture
def fresh_file_ranges():
    lc._DATASET_RANGES.clear()
    yield
    lc._DATASET_RANGES.clear()


def _roi_file(tmp_path, scan):
    import h5py

    path = tmp_path / "scan.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("1/EBSD/Data/Patterns", data=scan.reshape(NR * NC, H, W))
    return path


def _roi_run(fake, monkeypatch, tmp_path, source, *, scan=None, mask=False, expect=None, **extra):
    """A Spherical run on the selection ``SELECTION`` of ``scan`` (default: the int16 ``_scan``), with the
    backend replaced by the recorder. ``source`` is "memory" (an .up1 file h5py cannot open: the active
    signal is the pattern store) or "file" (an HDF5 file the selection is read from). Returns the stack
    ``index_array`` was handed and its keyword arguments."""
    import indexing_controller as ic
    from backend.api.routes import ebsd_viewer
    from tests import test_hex_indexing_paths as hx

    scan = _scan() if scan is None else scan
    hx._active_signal(monkeypatch, scan)
    if mask:
        include = np.ones((H, W), bool)
        include[0, 0] = False
        monkeypatch.setattr(ebsd_viewer, "get_active_include_mask", lambda: include)
    path = hx._up_file(tmp_path) if source == "memory" else _roi_file(tmp_path, scan)
    cfg = types.SimpleNamespace(sht_file="master.sht", bandwidth=88, normed=True, refine=False,
                                circmask=0, gausbckg=False, nregions=10)
    with pytest.raises(expect or hx._Stop):
        ic.spherical_gpu_index_patterns(str(path), cfg, {"n_rows": NR, "n_cols": NC},
                                        selection_mask=_selection_mask(), sht_paths=["master.sht"], **extra)
    if expect is not None:
        return None, None
    return fake.instances[0].arrays[0], fake.instances[0].kwargs[0]


def _as_the_backend_sees(handed, kwargs):
    """What ``SphericalGPUBackend`` makes of the stack: it rescales by the ``value_range`` it is given, and
    by the range of the stack when it is given none (``_index_with_phases``)."""
    rng = kwargs["value_range"] if "value_range" in kwargs else lc.signed_pattern_range(handed)
    return lc.patterns_to_float32(handed, rng)


@pytest.mark.parametrize("source", ["memory", "file"])
def test_a_spherical_roi_run_rescales_an_int16_scan_by_the_range_of_the_whole_scan(
        source, fake, monkeypatch, tmp_path, fresh_file_ranges):
    handed, kwargs = _roi_run(fake, monkeypatch, tmp_path, source)
    assert handed.shape == (2, H, W)
    assert lc.signed_pattern_range(handed) == (1000.0, 1100.0), "premise: the selection holds no extreme of the scan"
    seen = _as_the_backend_sees(handed, kwargs)
    assert np.array_equal(seen[0], _expected_00().astype(np.float32)), seen[0]
    assert np.array_equal(seen[1], _all_46()), seen[1]
    assert np.array_equal(seen[0], lc.signal_pattern(_signal(), 0, 0).astype(np.float32)), "the click disagrees"


@pytest.mark.parametrize("source", ["memory", "file"])
def test_a_spherical_roi_run_with_the_viewer_mask_rescales_by_the_whole_scan_then_masks(
        source, fake, monkeypatch, tmp_path, fresh_file_ranges):
    handed, kwargs = _roi_run(fake, monkeypatch, tmp_path, source, mask=True)
    seen = _as_the_backend_sees(handed, kwargs)
    hand_1 = _all_46()
    hand_1[0, 0] = 0.0                                          # the pixel outside the mask
    assert np.array_equal(seen[0], _hand_00()), seen[0]
    assert np.array_equal(seen[1], hand_1), seen[1]


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("source", ["memory", "file"])
def test_a_spherical_roi_run_on_another_dtype_reads_no_range_and_changes_nothing(
        source, dtype, fake, monkeypatch, tmp_path, pass_counter, fresh_file_ranges):
    scan = (_scan() // 25).astype(dtype)
    handed, kwargs = _roi_run(fake, monkeypatch, tmp_path, source, scan=scan)
    assert pass_counter == [] and "value_range" not in kwargs, "a scan without signed integers gets no range pass"
    assert handed.dtype == np.dtype(dtype)
    assert np.array_equal(handed, scan.reshape(NR * NC, H, W)[[0, 5]])


@pytest.mark.parametrize("source,shape", [("memory", (NR, NC, H, W)), ("file", (NR * NC, H, W))])
def test_a_spherical_roi_run_reads_the_range_of_the_whole_scan_once(
        source, shape, fake, monkeypatch, tmp_path, pass_counter, fresh_file_ranges):
    _roi_run(fake, monkeypatch, tmp_path, source)
    assert pass_counter == [shape], pass_counter


def test_the_range_pass_of_a_spherical_roi_run_reports_progress(fake, monkeypatch, tmp_path, fresh_file_ranges):
    monkeypatch.setattr(lc, "_RANGE_BLOCK_BYTES", 2 * H * W * 2)       # two patterns a block: six blocks
    lines = []
    _roi_run(fake, monkeypatch, tmp_path, "file", progress_callback=lambda msg, pct=None: lines.append(msg))
    reading = [m for m in lines if "value range" in m]
    assert [m.rsplit("(", 1)[-1] for m in reading] == [f"{k}/12)" for k in (2, 4, 6, 8, 10, 12)], reading


def test_a_cancel_stops_the_range_pass_of_a_spherical_roi_run(fake, monkeypatch, tmp_path, fresh_file_ranges):
    import indexing_controller as ic

    monkeypatch.setattr(lc, "_RANGE_BLOCK_BYTES", 2 * H * W * 2)       # two patterns a block: six blocks
    polls, lines = [], []

    def cancel():
        polls.append(1)
        return len(polls) >= 3                                         # the third poll: before block three

    _roi_run(fake, monkeypatch, tmp_path, "file", expect=ic.CancelledIndexingError,
             cancel_check=cancel, progress_callback=lambda msg, pct=None: lines.append(msg))
    assert len(polls) == 3, "the pass went on after the cancel"
    assert len([m for m in lines if "value range" in m]) == 2, lines
    assert fake.instances[0].arrays == [], "the run went on to index after the cancel"


def test_the_backend_takes_the_range_it_is_handed_and_reads_none(monkeypatch, range_passes):
    backend, _ = _backend(1, monkeypatch)
    monkeypatch.setattr(backend, "_ensure_built", lambda detector_params: None)
    stack = _scan_stack()[[0, 5]]
    backend.index_array(stack, {}, value_range=(100.0, 5100.0))
    assert range_passes == [], "a range that was handed over is read again"
    assert np.array_equal(np.concatenate(backend._indexers[0].seen)[0], _expected_00().astype(np.float32))
    backend.index_array(stack, {})                                     # left out: the stack's own range, as before
    assert len(range_passes) == 1


# --- Hough on a selection, and the EMSphInx export of a cropped region ------------------------------------------

def _hough_scan(dtype=np.int16):
    """3 x 3 patterns of the detector of kikuchipy's nickel set (60 x 60), the same values as ``_scan``:
    1000 everywhere, 1100 at [0, 0] of pattern (0, 0), and the extremes 100 / 5100 in pattern (2, 2)."""
    import kikuchipy as kp
    from tests.test_int16_patterns import _int16_nickel

    base = _int16_nickel()
    data = np.full((3, 3, 60, 60), 1000, np.int16)
    data[0, 0, 0, 0] = 1100
    data[2, 2, 1, 1], data[2, 2, 2, 2] = 100, 5100
    if np.dtype(dtype) != np.int16:
        data = (data // 25).astype(dtype)
    return kp.signals.EBSD(data, detector=base.detector)


def _hough_input(monkeypatch, sig, tmp_path, mask, **extra):
    """The stack ``hough_index_patterns`` hands PyEBSDIndex (the call then stops); [] when it stops before."""
    import indexing_controller as ic
    from pyebsdindex.ebsd_index import EBSDIndexer
    from tests.test_int16_patterns import _Seen, _ni_controller

    seen = []

    def rec(self, patsin=None, *a, **k):
        seen.append(np.asarray(patsin).copy())
        raise _Seen()

    monkeypatch.setattr(EBSDIndexer, "index_pats", rec)
    ctrl, _ = _ni_controller(tmp_path)
    cfg = ic.IndexingConfig(method=ic.IndexingMethod.HOUGH, n_bands=9)
    with pytest.raises((_Seen, ic.IndexingCancelled)):
        ic.hough_index_patterns(signal=sig, phase_list=ctrl.phase_list, detector=sig.detector, config=cfg,
                                selection_mask=mask, **extra)
    return seen


def _hough_selection():
    mask = np.zeros((3, 3), bool)
    mask[0, 0] = mask[1, 1] = True                    # flat patterns 0 and 4: neither holds an extreme
    return mask


def _hand_hough_00():
    out = np.full((60, 60), 46, np.uint8)
    out[0, 0] = 51
    return out


def test_hough_on_a_selection_rescales_an_int16_scan_by_the_range_of_the_whole_scan(monkeypatch, tmp_path):
    seen = _hough_input(monkeypatch, _hough_scan(), tmp_path, _hough_selection())
    assert len(seen) == 1 and seen[0].dtype == np.uint8 and seen[0].shape == (2, 60, 60)
    assert np.array_equal(seen[0][0], _hand_hough_00()), seen[0][0]
    assert np.array_equal(seen[0][1], np.full((60, 60), 46, np.uint8)), seen[0][1]


def test_hough_on_the_full_grid_reads_no_extra_range_pass(monkeypatch, tmp_path, pass_counter):
    seen = _hough_input(monkeypatch, _hough_scan(), tmp_path, None)
    assert seen[0].shape == (9, 60, 60) and np.array_equal(seen[0][0], _hand_hough_00())
    assert pass_counter == [], "the full grid rescales by its own range, which is the scan's"


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_hough_on_a_selection_of_another_dtype_reads_no_range_and_changes_nothing(
        dtype, monkeypatch, tmp_path, pass_counter):
    sig = _hough_scan(dtype)
    seen = _hough_input(monkeypatch, sig, tmp_path, _hough_selection())
    assert pass_counter == []
    assert seen[0].dtype == np.dtype(dtype)
    assert np.array_equal(seen[0], np.asarray(sig.data).reshape(9, 60, 60)[[0, 4]])


def test_hough_on_a_selection_reads_the_range_once_and_reports_progress(monkeypatch, tmp_path, pass_counter):
    monkeypatch.setattr(lc, "_RANGE_BLOCK_BYTES", 60 * 60 * 2)         # less than a row of patterns: one row a block
    lines = []
    _hough_input(monkeypatch, _hough_scan(), tmp_path, _hough_selection(),
                 progress_callback=lambda msg, pct=None: lines.append(msg))
    assert pass_counter == [(3, 3, 60, 60)], "once, over the whole scan"
    reading = [m for m in lines if "value range" in m]
    assert [m.rsplit("(", 1)[-1] for m in reading] == ["1/3)", "2/3)", "3/3)"], reading


def test_a_cancel_stops_the_range_pass_of_hough_on_a_selection(monkeypatch, tmp_path, pass_counter):
    monkeypatch.setattr(lc, "_RANGE_BLOCK_BYTES", 60 * 60 * 2)
    seen = _hough_input(monkeypatch, _hough_scan(), tmp_path, _hough_selection(),
                        cancel_check=lambda: len(pass_counter) >= 1)    # true as soon as the pass has started
    assert seen == [], "Hough went on to index after the cancel"
    assert pass_counter == [(3, 3, 60, 60)]


def test_the_emsphinx_export_of_a_cropped_region_rescales_by_the_whole_scan(tmp_path, pass_counter, fresh_file_ranges):
    import h5py

    import indexing_controller as ic

    src, dst = _roi_file(tmp_path, _scan()), tmp_path / "dst.h5"
    # rows 0..1 x columns 0..2 of the 3 x 4 grid: flat patterns 0 and 1, neither holds an extreme
    ic._export_patterns_for_indexebsd(str(src), str(dst), "1/EBSD/Data/Patterns",
                                      crop_region=(0, 1, 0, 2), n_cols_full=NC)
    with h5py.File(dst, "r") as f:
        out = f["1/EBSD/Data/Patterns"][:]
    assert out.dtype == np.uint8 and out.shape == (2, H, W)
    assert np.array_equal(out[0], _expected_00()), out[0]
    assert np.array_equal(out[1], _all_46().astype(np.uint8)), out[1]
    assert pass_counter == [(NR * NC, H, W)]


def test_the_emsphinx_export_of_the_whole_dataset_reads_no_extra_range_pass(tmp_path, pass_counter, fresh_file_ranges):
    import h5py

    import indexing_controller as ic

    src, dst = _roi_file(tmp_path, _scan()), tmp_path / "dst.h5"
    ic._export_patterns_for_indexebsd(str(src), str(dst), "1/EBSD/Data/Patterns")
    with h5py.File(dst, "r") as f:
        out = f["1/EBSD/Data/Patterns"][:]
    assert np.array_equal(out, to_uint8(_scan().reshape(NR * NC, H, W)))
    assert pass_counter == []


def test_an_edit_on_a_crop_flags_the_parent_dirty_too_and_an_unrelated_dataset_not(monkeypatch):
    """The crop of an eager signal is a view of its parent's array: the edit lands in the parent, so
    a ROI run on the parent must not read the (unedited) file. A dataset with its own array is not
    flagged."""
    from backend.api.routes import ebsd_viewer as ev

    parent = _signal()
    crop = types.SimpleNamespace(data=parent.data[:, :2])
    stranger = _signal()
    monkeypatch.setattr(ev, "_active_dataset", "crop")
    monkeypatch.setattr(ev, "_raw_signals", {"parent": parent, "crop": crop, "stranger": stranger})
    monkeypatch.setattr(ev, "_dirty_datasets", set())
    crop.data[0, 0, 0, 0] = 9000
    ev._mark_active_dirty()
    assert ev._dirty_datasets == {"crop", "parent"}


class _LazyFailsBeforeBuilding:
    """A lazy signal whose in-place calls fail before they replace the graph (a MemoryError in the
    percentile pass of the contrast stretch, a bad parameter): its data object stays the very same."""

    def __init__(self):
        import dask.array as da
        self.data = da.from_array(_scan(), chunks=(1, 2, H, W))
        self.axes_manager = types.SimpleNamespace(navigation_shape=(NC, NR))

    def _fail(self, *a, **k):
        raise MemoryError("out of memory before anything was built")

    average_neighbour_patterns = remove_dynamic_background = rescale_intensity = _fail
    adaptive_histogram_equalization = _fail


@pytest.mark.parametrize("route,body", [
    ("frame-average", {"window_size": 3}),
    ("background-removal", {"method": "dynamic"}),
    ("autocontrast", None),
    ("clahe", {"kernel_size": 8}),
])
def test_a_failed_edit_of_a_lazy_signal_that_changed_nothing_leaves_the_dataset_clean(route, body, monkeypatch):
    """A lazy signal's data is replaced by the new graph only when the call succeeds, so after a
    failure it is the file's data still. Flagged dirty, a lazy scan above the memory ceiling would be
    refused by Spherical indexing until the file is loaded again."""
    from fastapi.testclient import TestClient

    from backend.api.main import app
    from backend.api.routes import ebsd_viewer as ev

    sig = _LazyFailsBeforeBuilding()
    before = sig.data
    monkeypatch.setattr(ev, "_active_dataset", "ds")
    monkeypatch.setattr(ev, "_raw_signals", {"ds": sig})
    monkeypatch.setattr(ev, "_dirty_datasets", set())
    monkeypatch.setattr(ev, "_compute_include_mask", lambda *a, **k: None)
    r = TestClient(app).post(f"/api/ebsd/{route}", json=body) if body is not None \
        else TestClient(app).post(f"/api/ebsd/{route}")
    assert r.status_code == 500, r.text
    assert sig.data is before, "premise: the failed call changed nothing"
    assert ev._dirty_datasets == set()
