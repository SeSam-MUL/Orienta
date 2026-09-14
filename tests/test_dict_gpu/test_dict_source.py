"""The block readers the streamed indexer pulls its tiles from, and the
arithmetic that decides how big a tile may be.

These are the pieces that can be wrong quietly. A source that silently drops
the last short block, or a tile size computed without the score matrix in it,
produces a run that looks fine and is either incomplete or out of memory.
"""
import numpy as np
import pytest

from backend.dict_gpu.pipeline.dict_source import (
    ArrayDictSource,
    H5DictSource,
    find_pattern_dataset,
)
from backend.dict_gpu.pipeline.tiling import (
    MIN_STREAM_TILE,
    RESIDENT_PEAK_FACTOR,
    compute_stream_tile_size,
    iter_tiles,
    resident_peak_bytes,
    should_stream,
    stream_bytes_per_entry,
)


# --------------------------------------------------------------- sources ----

def test_array_source_hands_back_exactly_the_entries_asked_for():
    a = np.arange(7 * 3 * 4, dtype=np.float32).reshape(7, 3, 4)
    src = ArrayDictSource(a)
    assert src.shape == (7, 3, 4)
    np.testing.assert_array_equal(src.read_block(2, 5), a[2:5])
    # Every block, concatenated, must be the whole dictionary in order.
    blocks = [src.read_block(s.start, s.stop) for s in iter_tiles(7, 3)]
    np.testing.assert_array_equal(np.concatenate(blocks), a)


def test_array_source_drops_a_leading_singleton_navigation_axis():
    """kikuchipy hands dictionaries over as (1, n, h, w) often enough."""
    a = np.zeros((1, 5, 3, 4), dtype=np.float32)
    assert ArrayDictSource(a).shape == (5, 3, 4)


def test_array_source_returns_float32_contiguous_for_torch():
    a = np.arange(4 * 2 * 2, dtype=np.float64).reshape(4, 2, 2)
    block = ArrayDictSource(a).read_block(0, 4)
    assert block.dtype == np.float32 and block.flags["C_CONTIGUOUS"]


def test_array_source_rejects_something_that_is_not_a_dictionary():
    with pytest.raises(ValueError):
        ArrayDictSource(np.zeros((4, 4), dtype=np.float32))


@pytest.fixture
def h5_dict(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "dict.h5"
    data = np.arange(20 * 3 * 4, dtype=np.float32).reshape(20, 3, 4)
    with h5py.File(path, "w") as f:
        f.create_dataset("Scan 1/EBSD/Data/patterns", data=data, chunks=(4, 3, 4))
    return path, data


def test_h5_source_reads_blocks_without_loading_the_file(h5_dict):
    path, data = h5_dict
    with H5DictSource(path, "Scan 1/EBSD/Data/patterns") as src:
        assert src.shape == (20, 3, 4)
        assert src.chunk_entries == 4
        np.testing.assert_array_equal(src.read_block(5, 9), data[5:9])
        blocks = [src.read_block(s.start, s.stop) for s in iter_tiles(20, 7)]
        np.testing.assert_array_equal(np.concatenate(blocks), data)


def test_h5_source_close_is_idempotent(h5_dict):
    path, _ = h5_dict
    src = H5DictSource(path, "Scan 1/EBSD/Data/patterns")
    src.close()
    src.close()


def test_find_pattern_dataset_locates_the_nested_one(h5_dict):
    path, _ = h5_dict
    assert find_pattern_dataset(path) == "Scan 1/EBSD/Data/patterns"


def test_find_pattern_dataset_refuses_to_guess_between_two(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "two.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("a/patterns", data=np.zeros((2, 2, 2), dtype=np.float32))
        f.create_dataset("b/patterns", data=np.zeros((2, 2, 2), dtype=np.float32))
    assert find_pattern_dataset(path) is None


def test_find_pattern_dataset_on_a_file_that_is_not_hdf5(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("not an h5 file")
    assert find_pattern_dataset(p) is None


# ---------------------------------------------------------------- sizing ----

def test_bytes_per_entry_counts_the_score_column():
    """At a full map the score matrix costs more per entry than the entry."""
    d = 128 * 156
    per_entry = stream_bytes_per_entry(d, d, n_query=28086)
    assert per_entry == d * 4 + d * 2 + 28086 * 6
    # and dropping the query count drops the cost
    assert stream_bytes_per_entry(d, d, n_query=1) < per_entry


def test_bytes_per_entry_adds_the_masked_copy_only_when_masked():
    d = 100
    assert stream_bytes_per_entry(d, d, 10) == d * 4 + d * 2 + 60
    assert stream_bytes_per_entry(d, 60, 10) == d * 4 + 60 * 4 + 60 * 2 + 60


def test_stream_tile_fits_the_budget_it_was_given():
    d, n_query, budget = 19968, 1000, int(8e9)
    tile = compute_stream_tile_size(300_000, d, d, n_query, budget)
    assert 0 < tile < 300_000
    assert tile * stream_bytes_per_entry(d, d, n_query) <= budget * 0.6 + 1


def test_stream_tile_is_rounded_to_whole_hdf5_chunks():
    tile = compute_stream_tile_size(
        10_000, 100, 100, 10, int(1e7), chunk_entries=12)
    assert tile % 12 == 0


def test_stream_tile_is_zero_when_nothing_fits():
    """Zero is the caller's cue to say 'this does not fit' with numbers,
    rather than to start a run that will die on the first allocation."""
    one_mb = 1_000_000
    assert compute_stream_tile_size(1_000_000, 19968, 19968, 28086, one_mb) == 0


def test_stream_tile_never_exceeds_the_dictionary():
    assert compute_stream_tile_size(50, 100, 100, 10, int(1e9)) == 50


# -------------------------------------------------------------- decision ----

def test_resident_estimate_matches_the_measured_factor():
    """3.76x was measured through the real pipeline (m4 audit script)."""
    n, d = 100_347, 128 * 156
    assert resident_peak_bytes(n, d, d) == int(n * d * 4 * RESIDENT_PEAK_FACTOR)
    # A detector mask copies the dictionary once more.
    assert resident_peak_bytes(n, d, d // 2) > resident_peak_bytes(n, d, d)


@pytest.mark.parametrize("n_dict,expect", [
    (100_347, True),    # the user's 8 GB Al dictionary
    (301_055, True),    # 24 GB Al7FeCu2
    (2_000, False),     # a small one that has always fitted
])
def test_the_rule_streams_the_big_ones_and_leaves_the_small_ones(n_dict, expect):
    d = 128 * 156
    budget = int(9.6e9)  # what this project's 12.88 GB card reports free - 2 GB
    assert should_stream(n_dict, d, d, budget) is expect


def test_no_budget_means_stream():
    assert should_stream(1000, 100, 100, 0) is True


def test_min_stream_tile_is_a_real_floor():
    assert MIN_STREAM_TILE >= 1
