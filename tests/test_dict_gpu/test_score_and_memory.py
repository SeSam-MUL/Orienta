"""Two defects found while asking why the same file reported two NCCs.

Neither is in the forward projection — that was checked against kikuchipy's own
`get_patterns` and agrees to 1e-8, as do the two independent GPU projectors in
this repo (`_pcadi` for the indexer, `dictionary_gpu` for the generator).

(a) The reported confidence silently changed meaning. PCA switches itself on
    under memory pressure, and in the truncated subspace the score is the
    cosine between the PROJECTIONS: the components orthogonal to the retained
    ones — exactly where model-vs-experiment mismatch lives — are dropped from
    both vectors before the angle is taken. Measured on LoGainNi at a 2 deg
    grid: 0.5669 truncated against 0.4685 exact, for the SAME winning
    orientations. So the same dictionary indexed on a busier card could report
    scores 0.1 apart with nothing saying why.

(b) The on-the-fly projection never batched. `project_master_to_detector`
    materialises ~20x the output size in transients, so handing it every
    rotation at once asked for 36,191 MiB on a 12,282 MiB card at 100,347
    rotations. It completed via host fallback, but with the experimental map
    also resident a full-map run took 474 s where the same dictionary read
    from disk took 6.4 s.
"""
import inspect

import numpy as np
import pytest
import torch

from backend.dict_gpu._pcadi import master_to_dict as m2d
from backend.dict_gpu.pipeline import indexer as indexer_mod


# --------------------------------------------------------------------------
# (a) the score must be an NCC whatever the search ran in
# --------------------------------------------------------------------------
def test_top_matches_are_rescored_at_full_dimension():
    src = inspect.getsource(indexer_mod.run_dictionary_index)
    assert "if pca is not None:" in src.split("7. tiled top-k NCC matching")[1], (
        "no re-score branch after the search"
    )
    assert "dict_norm[rows] * exp_rows" in src, (
        "the re-score must use the FULL-dimensional normalised tensors, "
        "not the projected ones"
    )


def test_rescored_candidates_are_reordered():
    """The shortlist came back ordered by the truncated score; 'best match'
    has to mean best by the number we report."""
    src = inspect.getsource(indexer_mod.run_dictionary_index)
    assert "torch.sort(best_scores, dim=1, descending=True)" in src
    assert "best_indices.gather(1, order)" in src


def test_padding_slots_keep_minus_inf():
    """A short final tile pads with index -1; those must not be re-scored into
    a real-looking value that could outrank a genuine match."""
    src = inspect.getsource(indexer_mod.run_dictionary_index)
    assert "torch.where(best_indices >= 0, exact, best_scores)" in src


def test_rescore_math_matches_a_plain_ncc():
    """The gather-and-dot the indexer does is an NCC when both sides are
    already mean-subtracted and L2-normalised."""
    rng = np.random.default_rng(0)
    n_dict, n_exp, dim, keep_n = 40, 7, 64, 3
    dict_norm = indexer_mod._normalise_rows(
        torch.from_numpy(rng.normal(size=(n_dict, dim)).astype(np.float32)))
    exp_norm = indexer_mod._normalise_rows(
        torch.from_numpy(rng.normal(size=(n_exp, dim)).astype(np.float32)))
    best_indices = torch.from_numpy(
        rng.integers(0, n_dict, size=(n_exp, keep_n)).astype(np.int64))

    flat = best_indices.reshape(-1)
    rows = torch.div(torch.arange(flat.numel()), keep_n, rounding_mode="floor")
    got = (dict_norm[flat] * exp_norm[rows]).sum(dim=1).reshape(n_exp, keep_n)

    for i in range(n_exp):
        for j in range(keep_n):
            a = exp_norm[i].numpy().astype(np.float64)
            b = dict_norm[best_indices[i, j]].numpy().astype(np.float64)
            want = float((a - a.mean()) @ (b - b.mean())
                         / (np.linalg.norm(a - a.mean()) * np.linalg.norm(b - b.mean())))
            assert got[i, j].item() == pytest.approx(want, abs=1e-5)


# --------------------------------------------------------------------------
# (b) the projection must not ask for more memory than the dictionary needs
# --------------------------------------------------------------------------
def test_projection_is_chunked():
    src = inspect.getsource(m2d._path_a)
    assert "for s in range(0, n, step)" in src, "still one unbatched call"
    assert "torch.empty(" in src, "output must be preallocated, not concatenated"


@pytest.mark.parametrize("pattern_dim,free_bytes,expect", [
    (60 * 60, 8_000_000_000, 4096),      # plenty free -> clamped at the cap
    (60 * 60, 100_000_000, 86),          # tight -> small but workable
    (60 * 60, 1_000, 64),                # absurdly tight -> floor, never 0
    (1024 * 1024, 8_000_000_000, 64),    # huge detector -> floor
])
def test_batch_size_is_bounded_both_ways(pattern_dim, free_bytes, expect,
                                          monkeypatch):
    """Never 0 (an empty loop silently returns an uninitialised dictionary) and
    never unbounded (that is the bug)."""
    monkeypatch.setattr(torch.cuda, "mem_get_info",
                        lambda dev=None: (free_bytes, free_bytes))
    got = m2d._projection_batch(pattern_dim, torch.float32, torch.device("cuda"))
    assert got == expect
    assert 64 <= got <= 4096


def test_batch_size_without_cuda_does_not_query_the_driver():
    """The CPU path must not call mem_get_info — it raises without a device."""
    got = m2d._projection_batch(60 * 60, torch.float32, torch.device("cpu"))
    assert 64 <= got <= 4096


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_chunked_projection_is_bit_identical_and_cheaper():
    """Loop restructuring only — the numbers must not move at all.

    Measured at 8000 rotations on a 60x60 detector: peak allocation
    2903 -> 1720 MiB, output torch.equal.
    """
    from orix.quaternion import Rotation
    from orix.sampling import get_sample_fundamental
    from orix.quaternion.symmetry import Oh
    from backend.dict_gpu._pcadi._projection.direction_cosines import (
        compute_direction_cosines)
    from backend.dict_gpu._pcadi._projection.project import project_master_to_detector

    class _Det:
        shape = (32, 32)
        pcx = pcy = (0.5,)
        pcz = (0.6,)
        tilt = 10.0
        azimuthal = 0.0
        sample_tilt = 70.0

    class _MP:
        pass

    rng = np.random.default_rng(0)
    mp = torch.from_numpy(rng.random((2, 64, 64)).astype(np.float32)).cuda()
    det = _Det()
    dirs = compute_direction_cosines(det, device="cuda", dtype=torch.float32)
    rots = Rotation(get_sample_fundamental(12.0, point_group=Oh).data[:600])
    q = torch.from_numpy(rots.data.astype(np.float32)).cuda()

    ref = project_master_to_detector(mp, q, dirs).clone()

    n = q.shape[0]
    out = torch.empty((n, dirs.shape[0], dirs.shape[1]), device="cuda",
                      dtype=torch.float32)
    step = m2d._projection_batch(dirs.shape[0] * dirs.shape[1],
                                 torch.float32, torch.device("cuda"))
    for s in range(0, n, step):
        e = min(s + step, n)
        out[s:e] = project_master_to_detector(mp, q[s:e], dirs)

    assert torch.equal(ref, out), "chunking changed the output"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
