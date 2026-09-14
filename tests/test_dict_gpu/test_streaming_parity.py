"""Streaming the dictionary past the GPU must give the same answer as holding
it there.

The GPU indexer used to upload the whole dictionary and then slice views of the
resident tensor, so the tiling bounded the GEMM's working set but not the
dictionary's footprint (measured peak: 3.76x the dictionary). Streaming reads
one tile at a time from the host array — or from the file — normalises it on
the device and throws it away again, which is the only way a 24 GB dictionary
fits on a 12 GB card.

That is a rewrite of the hot loop, so what these tests guard is not "does it
run" but "does it return the same thing": same winning entry, same ranking for
keep_n > 1, same score, with and without a detector mask, and with a tile size
deliberately chosen so the last tile is short (n_dict is not a multiple of the
tile). A run that silently dropped the tail of the dictionary, or lost the
index offset of a tile, would still produce a plausible map — these compare it
against the resident-tensor path entry by entry.
"""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="the GPU indexer requires CUDA")

N_DICT = 20003       # prime-ish: no tile size divides it evenly
N_TILES = 7
H = W = 12


def _fixture(n_rows=6, n_cols=7, n_dict=N_DICT, h=H, w=W, seed=0):
    """A dictionary big enough to tile, with a known answer per pixel."""
    import kikuchipy as kp
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation

    rng = np.random.default_rng(seed)
    rots = Rotation(rng.normal(size=(n_dict, 4)))
    dict_data = rng.normal(120, 30, (n_dict, h, w)).astype(np.float32)

    truth = rng.integers(0, n_dict, n_rows * n_cols)
    exp = (dict_data[truth] + rng.normal(0, 8, (n_rows * n_cols, h, w))
           ).astype(np.float32).reshape(n_rows, n_cols, h, w)

    phase = Phase(name="synthetic", point_group="m-3m")
    dic = kp.signals.EBSD(dict_data)
    dic.xmap = CrystalMap(rotations=rots, phase_list=PhaseList(phase))

    sig = kp.signals.EBSD(exp)
    return sig, dic, truth.reshape(n_rows, n_cols)


def _run(sig, dic, **kw):
    from backend.dict_gpu.api import gpu_dictionary_index_patterns

    class _Det:
        shape = sig.data.shape[-2:]
        pcx = pcy = (0.5,)
        pcz = (0.6,)
        tilt = 0.0
        azimuthal = 0.0
        sample_tilt = 70.0

    kw.setdefault("use_pca", False)
    return gpu_dictionary_index_patterns(
        experimental_signal=sig, master_pattern_or_path=dic, detector=_Det(),
        **kw)


def _scores_and_indices(result):
    scores = np.asarray(result.xmap.prop["scores"], dtype=np.float64)
    idx = np.asarray(result.xmap.prop["simulation_indices"], dtype=np.int64)
    if scores.ndim == 1:
        scores = scores[:, None]
        idx = idx[:, None]
    return scores, idx


def _disc_mask(h=H, w=W):
    """kikuchipy convention: True = EXCLUDE (the dark corners)."""
    yy, xx = np.mgrid[0:h, 0:w]
    r = min(h, w) / 2.0
    return ((yy - (h - 1) / 2.0) ** 2 + (xx - (w - 1) / 2.0) ** 2) > r ** 2


@pytest.mark.parametrize("keep_n", [1, 3])
@pytest.mark.parametrize("masked", [False, True], ids=["nomask", "mask"])
def test_streamed_and_resident_agree(keep_n, masked):
    sig, dic, truth = _fixture()
    kw = {"keep_n": keep_n}
    if masked:
        kw["signal_mask"] = _disc_mask()

    resident = _run(sig, dic, stream_dictionary=False, **kw)
    streamed = _run(sig, dic, stream_dictionary=True,
                    stream_tile_entries=-(-N_DICT // N_TILES), **kw)

    s_res, i_res = _scores_and_indices(resident)
    s_str, i_str = _scores_and_indices(streamed)

    assert i_res.shape == i_str.shape == (truth.size, keep_n)
    # The index equality is the criterion — it is what the map is made of.
    # The scores are compared as a sanity check with a tolerance rather than
    # exactly: the GEMM runs in fp16, so a different tile split reassociates
    # the reduction. Measured on real data at feat_dim 3600 the largest
    # difference that produces is 2.4e-4, and it moved the ranking on no pixel
    # of 2990 compared across three runs. This fixture is 144 features, where
    # the reduction is short enough to land on 1e-6; a wider detector would
    # want 1e-4 here.
    np.testing.assert_array_equal(i_str, i_res)
    np.testing.assert_allclose(s_str, s_res, rtol=1e-6, atol=1e-6)

    # Both must actually be right, or "they agree" means nothing.
    np.testing.assert_array_equal(i_res[:, 0], truth.ravel())


def test_streaming_reaches_the_last_short_tile():
    """The tail of the dictionary must still be searched.

    A tile size that does not divide n_dict leaves a short final tile, and the
    entry the pixels below match lives inside it. A loop that stopped at the
    last whole tile would return a wrong-but-plausible winner here.
    """
    sig, dic, _ = _fixture(n_rows=2, n_cols=2)
    tile = -(-N_DICT // N_TILES)
    last_tile_start = tile * (N_TILES - 1)
    assert last_tile_start < N_DICT, "fixture must have a short final tile"

    # Put every experimental pattern on an entry inside the final short tile.
    target = N_DICT - 3
    exp = np.repeat(np.asarray(dic.data)[target][None], 4, axis=0)
    exp = exp.reshape(2, 2, H, W).copy()
    sig.data[...] = exp

    res = _run(sig, dic, stream_dictionary=True, stream_tile_entries=tile,
               keep_n=1)
    _, idx = _scores_and_indices(res)
    assert set(idx[:, 0].tolist()) == {target}


def test_streaming_peak_vram_is_a_fraction_of_the_dictionary():
    """The point of the exercise: the dictionary is never wholly resident.

    Measured against a baseline taken after a warm-up run, because the first
    GEMM in a process makes cuBLAS take a workspace (8.5 MB on this card) out
    of the same allocator — a fixed cost that has nothing to do with the
    dictionary but would swamp a toy one.
    """
    sig, dic, _ = _fixture()
    dict_bytes = np.asarray(dic.data).nbytes

    _run(sig, dic, stream_dictionary=True, keep_n=1)  # warm up cuBLAS
    import gc
    gc.collect()
    baseline = torch.cuda.memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    _run(sig, dic, stream_dictionary=True,
         stream_tile_entries=-(-N_DICT // N_TILES), keep_n=1)
    streamed_peak = torch.cuda.max_memory_allocated() - baseline

    torch.cuda.reset_peak_memory_stats()
    _run(sig, dic, stream_dictionary=False, keep_n=1)
    resident_peak = torch.cuda.max_memory_allocated() - baseline

    assert streamed_peak < dict_bytes / 2, (
        f"streamed peak {streamed_peak/1e6:.2f} MB is not well below the "
        f"dictionary itself ({dict_bytes/1e6:.2f} MB)")
    assert streamed_peak < resident_peak / 2, (
        f"streamed {streamed_peak/1e6:.2f} MB vs resident "
        f"{resident_peak/1e6:.2f} MB")


def test_a_dictionary_given_as_a_path_streams_from_the_file():
    """Handed a path, the indexer must read tiles from the file rather than
    pull the whole dictionary through host RAM on its way to the card.

    Same answer as the in-memory run, and the result says which route it took.
    """
    import tempfile
    from pathlib import Path
    from backend.dict_gpu.pipeline.master_loader import open_streamed_dict
    from backend.dict_gpu.pipeline.dict_source import H5DictSource

    sig, dic, truth = _fixture(n_dict=3000)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "dict.h5"
        dic.save(str(path))

        payload = open_streamed_dict(path)
        assert payload is not None, "the file should be streamable"
        assert payload.dict_patterns is None, "patterns must stay on disk"
        assert isinstance(payload.dict_source, H5DictSource)
        assert payload.dict_rotations.size == 3000
        payload.dict_source.close()

        from_file = _run(sig, str(path), stream_dictionary=True,
                         stream_tile_entries=439, keep_n=1)

    in_memory = _run(sig, dic, stream_dictionary=False, keep_n=1)

    assert from_file.metadata["streamed"] is True
    assert from_file.metadata["stream_tile_entries"] == 439
    assert in_memory.metadata["streamed"] is False

    s_f, i_f = _scores_and_indices(from_file)
    s_m, i_m = _scores_and_indices(in_memory)
    np.testing.assert_array_equal(i_f, i_m)
    np.testing.assert_allclose(s_f, s_m, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(i_f[:, 0], truth.ravel())


def test_stop_is_honoured_between_tiles(monkeypatch):
    """A streamed run is where Stop matters most — the tile boundary is the
    only place it can take effect, and a 24 GB dictionary is a lot of them.

    Counting the matched tiles, not just the exception: a cancel that fired
    before the first tile would pass a bare `raises`, and so would one that
    only took effect after the last.
    """
    from indexing_controller import CancelledIndexingError
    import backend.dict_gpu.pipeline.indexer as indexer

    n_dict, tile = 4000, 500
    n_tiles = -(-n_dict // tile)
    sig, dic, _ = _fixture(n_dict=n_dict)

    matched = {"tiles": 0}
    real_gemm = indexer.gemm_topk_ncc

    def counting_gemm(*a, **kw):
        matched["tiles"] += 1
        return real_gemm(*a, **kw)

    monkeypatch.setattr(indexer, "gemm_topk_ncc", counting_gemm)

    seen = {"calls": 0}

    def cancel_after_the_first_tile():
        seen["calls"] += 1
        return seen["calls"] > 2

    with pytest.raises(CancelledIndexingError):
        _run(sig, dic, stream_dictionary=True, stream_tile_entries=tile,
             keep_n=1, cancel_check=cancel_after_the_first_tile)

    assert 1 <= matched["tiles"] < n_tiles, (
        f"{matched['tiles']} of {n_tiles} tiles matched before the cancel "
        f"took effect")


def test_a_budget_too_small_for_one_tile_says_so_with_the_numbers():
    """When nothing fits, the user needs the sizes, not a CUDA allocation
    message about whichever tensor happened to be last.

    The size of the smallest tile is checked against the arithmetic the tile
    was actually refused by, not merely for being present: the first version
    of this message printed ``budget // MIN_STREAM_TILE``, which is neither
    bytes per entry nor the cost of a tile — 0.60 GB where the truth was
    0.0046 GB, a factor of 130, and a test that only looked for the words
    could not see it.
    """
    import re
    from backend.dict_gpu.exceptions import GpuDictError
    from backend.dict_gpu.pipeline.tiling import (
        MIN_STREAM_TILE, stream_bytes_per_entry)

    h = w = 40                      # big enough that the figure is not 0.0 MiB
    n_rows, n_cols = 6, 7
    sig, dic, _ = _fixture(n_dict=500, h=h, w=w, n_rows=n_rows, n_cols=n_cols)
    with pytest.raises(GpuDictError) as exc:
        _run(sig, dic, stream_dictionary=True, keep_n=1,
             vram_budget_gb=1e-6)   # one kilobyte of budget
    msg = str(exc.value)
    assert "does not fit" in msg
    for expected in ("Dictionary", "experimental patterns", "free VRAM",
                     "smallest usable tile"):
        assert expected in msg, f"missing {expected!r} from:\n{msg}"

    m = re.search(r"smallest usable tile (\d+) entries \(([\d.]+) MiB\)", msg)
    assert m, f"no machine-readable tile size in:\n{msg}"
    assert int(m.group(1)) == MIN_STREAM_TILE
    expected_mib = MIN_STREAM_TILE * stream_bytes_per_entry(
        h * w, h * w, n_rows * n_cols) / (1 << 20)
    assert abs(float(m.group(2)) - expected_mib) < 0.05, (
        f"message says {m.group(2)} MiB, the tile costs {expected_mib:.3f} MiB")


def test_a_failed_run_still_closes_the_dictionary_file(monkeypatch):
    """A raise in the middle of a match must not leave the file open.

    Waiting for the garbage collector is not good enough: the frame holding
    the handle is held by the traceback (this project measured exactly that
    for the dictionary tensors), and on Windows an open handle keeps the file
    locked against regenerating it. Checked by writing to the file afterwards,
    which is what the lock would prevent.
    """
    import tempfile
    from pathlib import Path
    import backend.dict_gpu.pipeline.indexer as indexer

    sig, dic, _ = _fixture(n_dict=600)

    def exploding_gemm(*a, **kw):
        raise RuntimeError("boom, halfway through the match")

    monkeypatch.setattr(indexer, "gemm_topk_ncc", exploding_gemm)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "dict.h5"
        dic.save(str(path))
        with pytest.raises(RuntimeError, match="boom"):
            _run(sig, str(path), stream_dictionary=True,
                 stream_tile_entries=100, keep_n=1)
        # Opening for append would fail with the handle still held.
        import h5py
        with h5py.File(str(path), "a") as f:
            f.attrs["closed_properly"] = True


def test_pca_with_streaming_fails_loud_instead_of_guessing():
    """PCA needs the whole dictionary to fit its basis; streaming exists
    because it does not fit. Saying so beats fitting on a silent sample."""
    from backend.dict_gpu.exceptions import GpuDictError

    sig, dic, _ = _fixture(n_dict=2000)
    with pytest.raises(GpuDictError) as exc:
        _run(sig, dic, stream_dictionary=True, keep_n=1,
             use_pca=True, pca_components=8)
    msg = str(exc.value).lower()
    assert "pca" in msg and "stream" in msg
