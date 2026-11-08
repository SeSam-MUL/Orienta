"""Where loading a pattern variant and reading the unprocessed int16 patterns meet.

A file can be loaded as another measurement and as its unprocessed (int16) patterns. Everything that
re-reads the file reads THAT dataset; a signed-integer scan reaches an indexer as
its 8 bit rescale by the range of the whole scan. These tests hold the two together on
the synthetic Oxford file of ``test_load_variant``: group 1 unprocessed is 500..1333, group 2 is
1500..2333 (``7 * k - 500 + 1000 * g``, k = 0..119), so a read of the wrong dataset, a range taken
from the wrong dataset and a wrap modulo 256 each give other numbers.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import safe_loader  # noqa: E402
from backend.api.routes import indexing as ind_routes  # noqa: E402
from backend.api.services import load_choices as lc  # noqa: E402
from backend.api.services.load_choices import LoadVariant  # noqa: E402
from tests.test_load_variant import NX, NY, SX, SY, multi  # noqa: E402,F401  (``multi`` is a fixture)

N = NX * NY
RAW1 = LoadVariant("1", "unprocessed")
RAW2 = LoadVariant("2", "unprocessed")
RAW1_PATH = "1/EBSD/Data/Unprocessed Patterns"
RAW2_PATH = "2/EBSD/Data/Unprocessed Patterns"


def _counts(g: int) -> np.ndarray:
    """The unprocessed counts of group ``g`` as written by the fixture, spelled out again."""
    return np.array([7 * k - 500 + 1000 * g for k in range(N * SY * SX)], dtype=np.int64)


def _rescale(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """The 8 bit rescale by hand: (v - lo) / (hi - lo) * 255, rounded. (No value falls on .5 here:
    the step is 7 / 833 * 255 = 15 / 7 grey levels.)"""
    return np.rint((values.astype(np.float64) - lo) * 255.0 / (hi - lo)).astype(np.uint8)


@pytest.fixture
def range_passes(monkeypatch):
    """Every pass over a signed-integer source: (dataset name or None, shape, dtype)."""
    calls = []
    real = lc.signed_pattern_range

    def counting(source, *a, **k):
        calls.append((getattr(source, "name", None), tuple(source.shape), str(source.dtype)))
        return real(source, *a, **k)

    monkeypatch.setattr(lc, "signed_pattern_range", counting)
    monkeypatch.setattr(lc, "_DATASET_RANGES", {})
    return calls


# ----------------------------------------------------------- 1. a streamed run on the variant

def _backend_streaming_into(seen):
    """A ``SphericalGPUBackend`` whose indexer opens datasets for real and records the stream call."""
    from backend.spherical_gpu.backend import SphericalGPUBackend
    from backend.spherical_gpu.pipeline.indexer import Tier1Indexer

    class _Indexer:
        circmask, gausbckg_on, nregions_n, normed, bandwidth = 0, False, 10, True, 88

        def _open_pattern_dataset(self, path, dataset_path=None):
            return Tier1Indexer._open_pattern_dataset(None, path, dataset_path)

        def index_h5oina(self, path, **kwargs):
            seen.append((path, kwargs))
            return SimpleNamespace(euler_xyz=None, score=None, phase_id=None, runtime_seconds=0.0)

    backend = object.__new__(SphericalGPUBackend)
    backend.config = SimpleNamespace(batch_size=4)
    backend._device = SimpleNamespace(type="cpu")
    backend._ensure_built = lambda det: None
    backend._auto_batch_size = lambda ix: 4
    backend._raise_if_cancelled = lambda: None
    backend._cancel_check = None
    backend._indexers = [_Indexer()]
    return backend


@pytest.mark.parametrize("variant,path,lo,hi", [(RAW1, RAW1_PATH, 500.0, 1333.0),
                                                (RAW2, RAW2_PATH, 1500.0, 2333.0)],
                         ids=["unprocessed", "measurement-2-unprocessed"])
def test_a_streamed_run_on_an_unprocessed_variant_reads_the_range_of_that_dataset_once(
        multi, range_passes, variant, path, lo, hi):
    seen = []
    backend = _backend_streaming_into(seen)
    backend.index_h5(multi, {}, dataset_path=path)
    # one pass, over the variant's dataset (an int16 one of the whole scan), not over a dataset the
    # search of the indexer would have found first
    assert range_passes == [("/" + path, (N, SY, SX), "int16")]
    (opened, kwargs), = seen
    assert opened == multi
    assert kwargs["dataset_path"] == path
    assert kwargs["value_range"] == (lo, hi)                # handed on: the stream does not read it again


def test_a_streamed_run_on_the_default_variant_reads_no_range(multi, range_passes):
    seen = []
    _backend_streaming_into(seen).index_h5(multi, {})
    assert range_passes == []                               # the default is uint8: nothing to read
    (_, kwargs), = seen
    assert kwargs["dataset_path"] is None and kwargs["value_range"] is None


def test_the_stream_of_the_indexer_itself_rescales_by_the_range_of_the_named_dataset(multi, range_passes):
    """``Tier1Indexer.index_h5oina`` without a range from its caller reads it on the dataset it streams."""
    import torch

    from backend.spherical_gpu.pipeline.indexer import Tier1Indexer

    blocks = []

    def index_batch(pat):
        blocks.append(np.asarray(pat).copy())
        return torch.zeros(pat.shape[0], 3), torch.ones(pat.shape[0])

    indexer = object.__new__(Tier1Indexer)
    indexer.device = torch.device("cpu")
    indexer._index_batch = index_batch
    indexer.index_h5oina(multi, batch_size=4, dataset_path=RAW2_PATH)
    assert range_passes == [("/" + RAW2_PATH, (N, SY, SX), "int16")]
    streamed = np.concatenate(blocks).reshape(-1)
    assert np.array_equal(streamed, _rescale(_counts(2), 1500.0, 2333.0))


# ----------------------------------------------- 2. the rich export and the two ways to read a signal

def _signal(data):
    return SimpleNamespace(axes_manager=SimpleNamespace(navigation_shape=(NX, NY), signal_shape=(SX, SY)),
                           data=data)


def _stack_of(counts: np.ndarray) -> np.ndarray:
    return counts.reshape(NY, NX, SY, SX)


def test_the_default_path_rescales_a_signed_stack_and_the_variant_path_keeps_it_native():
    counts = _counts(1)
    signal = _signal(_stack_of(counts.astype(np.int16)))
    default, shape = ind_routes._signal_patterns_3d(signal)
    assert shape == (NY, NX, SY, SX)
    assert default.dtype == np.uint8
    assert np.array_equal(default.reshape(-1), _rescale(counts, 500.0, 1333.0))
    native, _ = ind_routes._signal_patterns_native(signal)
    assert native.dtype == np.int16 and np.array_equal(native.reshape(-1), counts)


def test_the_default_path_of_the_rich_export_does_not_wrap_a_float_stack(multi, tmp_path):
    """A background-corrected float stack has negative values; ``astype(np.uint8)`` wraps them modulo 256.
    The export without a variant writes the rescale of the whole stack."""
    out = tmp_path / "export.h5oina"
    shutil.copy2(multi, out)
    k = np.arange(N * SY * SX)
    floats = -37.5 + k * 3.75                               # -37.5 .. 408.75, negative start
    assert ind_routes._overwrite_patterns_with_processed(str(out), _signal(_stack_of(floats))) is True
    with h5py.File(out, "r") as f:
        stored = f["1/EBSD/Data/Processed Patterns"][...]
    assert stored.dtype == np.uint8
    expected = np.rint(k * 255.0 / (N * SY * SX - 1)).astype(np.uint8)
    assert np.array_equal(stored.reshape(-1), expected)
    wrapped = floats.astype(np.int64) % 256                 # what the plain cast would have stored
    assert not np.array_equal(stored.reshape(-1), wrapped)


def test_the_variant_path_of_the_rich_export_keeps_the_int16_counts(multi, tmp_path):
    out = tmp_path / "export.h5oina"
    shutil.copy2(multi, out)
    counts = _counts(1) + 11                                # an edit of the unprocessed counts
    assert ind_routes._overwrite_patterns_with_processed(
        str(out), _signal(_stack_of(counts.astype(np.int16))), RAW1) is True
    with h5py.File(out, "r") as f:
        stored = f[RAW1_PATH][...]
        untouched = f["1/EBSD/Data/Processed Patterns"][...]
    assert stored.dtype == np.int16 and np.array_equal(stored.reshape(-1), counts)
    assert untouched.dtype == np.uint8 and untouched.max() < 200      # the other source is as it was


# ---------------------------------------- 3. a single pattern of a signal loaded as an int16 variant

def test_a_single_pattern_of_an_unprocessed_variant_is_what_the_map_run_sees(multi):
    from tools.pattern_comparison import get_experimental_pattern

    signal = safe_loader.load_ebsd_safe(multi, verbose=False, variant=RAW2)
    assert np.asarray(signal.data).dtype == np.int16
    result = SimpleNamespace(metadata={"signal": signal})
    row, col = 1, 2
    flat = row * NX + col
    counts = _counts(2).reshape(N, SY, SX)[flat]
    # the map run: the rescale by the range of the whole dataset (its stream: ``patterns_to_float32``)
    expected = _rescale(_counts(2), 1500.0, 2333.0).reshape(N, SY, SX)[flat]
    got = get_experimental_pattern(result, row, col, for_indexer=True)
    assert got.dtype == np.float32 and np.array_equal(got, expected.astype(np.float32))
    with h5py.File(multi, "r") as f:                         # the second formula, the stream's own
        dset = f[RAW2_PATH]
        streamed = lc.patterns_to_float32(dset[flat:flat + 1], lc.signed_pattern_range(dset))[0]
    assert np.array_equal(got, streamed)
    # a display or a comparison keeps the counts
    raw = get_experimental_pattern(result, row, col)
    assert np.array_equal(raw, counts.astype(np.float32))
