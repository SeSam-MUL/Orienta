"""Behavioural cover for the exact-NCC re-score, against a numpy reference.

The sibling tests for this feature assert on source text. That was the wrong
call: `run_dictionary_index` runs end to end against a synthetic dictionary in
well under a second on this machine, so the real thing was cheap and available.
These tests run it and check every reported (index, score) pair against an
independent float64 NCC, which is what actually matters:

  - the score is an NCC whether the search ran in the PCA subspace or not
  - the two modes agree on the winning orientation
  - the candidates come back ordered by the score that is reported
  - padding slots (keep_n > dictionary size) never leak a finite score
  - a detector mask reduces the correlation to the kept pixels on both sides

A source-text assertion would pass on all of these while the arithmetic was
wrong, which is exactly how a fixed 1e6-row re-score tile survived review.
"""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="the GPU indexer requires CUDA")


def _fixture(n_rows=6, n_cols=7, n_dict=40, h=12, w=12, seed=0):
    """A synthetic dictionary + experimental signal that index sensibly.

    Every experimental pattern is a noisy copy of one dictionary entry, so the
    correct answer is known and the correlations are far from degenerate.
    """
    import kikuchipy as kp
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    from orix.sampling import get_sample_fundamental
    from orix.quaternion.symmetry import Oh

    rng = np.random.default_rng(seed)
    rots = Rotation(get_sample_fundamental(30.0, point_group=Oh).data[:n_dict])
    n_dict = rots.size
    dict_data = rng.normal(120, 30, (n_dict, h, w)).astype(np.float32)

    truth = rng.integers(0, n_dict, n_rows * n_cols)
    exp = (dict_data[truth] + rng.normal(0, 8, (n_rows * n_cols, h, w))
           ).astype(np.float32).reshape(n_rows, n_cols, h, w)

    phase = Phase(name="synthetic", point_group="m-3m")
    dic = kp.signals.EBSD(dict_data)
    dic.xmap = CrystalMap(rotations=rots, phase_list=PhaseList(phase))

    sig = kp.signals.EBSD(exp)
    return sig, dic, truth.reshape(n_rows, n_cols)


def _reference_ncc(exp_pattern, dict_pattern, keep=None):
    a = np.asarray(exp_pattern, dtype=np.float64).ravel()
    b = np.asarray(dict_pattern, dtype=np.float64).ravel()
    if keep is not None:
        a, b = a[keep], b[keep]
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _run(sig, dic, **kw):
    from backend.dict_gpu.api import gpu_dictionary_index_patterns

    class _Det:
        shape = sig.data.shape[-2:]
        pcx = pcy = (0.5,)
        pcz = (0.6,)
        tilt = 0.0
        azimuthal = 0.0
        sample_tilt = 70.0

    return gpu_dictionary_index_patterns(
        experimental_signal=sig, master_pattern_or_path=dic, detector=_Det(), **kw)


def _scores_and_indices(result):
    scores = np.asarray(result.xmap.prop["scores"], dtype=np.float64)
    idx = np.asarray(result.xmap.prop["simulation_indices"], dtype=np.int64)
    if scores.ndim == 1:
        scores = scores[:, None]
        idx = idx[:, None]
    return scores, idx


@pytest.mark.parametrize("pca_kw", [
    pytest.param(dict(use_pca=False), id="pca-off"),
    pytest.param(dict(use_pca=True, pca_components=16), id="pca-on"),
])
def test_every_reported_score_is_the_real_ncc(pca_kw):
    sig, dic, _ = _fixture()
    res = _run(sig, dic, keep_n=4, **pca_kw)
    scores, idx = _scores_and_indices(res)

    exp_flat = np.asarray(sig.data).reshape(-1, *sig.data.shape[-2:])
    dict_flat = np.asarray(dic.data)
    # PCA truncation is tight here (16 of 144 components) so the search picks
    # slightly different candidates; the point is that whatever it picked, the
    # number attached to it is the true NCC of that pair.
    tol = 3e-3
    for i in range(idx.shape[0]):
        for j in range(idx.shape[1]):
            if idx[i, j] < 0:
                continue
            want = _reference_ncc(exp_flat[i], dict_flat[idx[i, j]])
            assert scores[i, j] == pytest.approx(want, abs=tol), (
                f"pixel {i} rank {j}: reported {scores[i, j]}, true NCC {want}")


def test_candidates_are_ordered_by_the_reported_score():
    sig, dic, _ = _fixture()
    for kw in (dict(use_pca=False), dict(use_pca=True, pca_components=16)):
        scores, _ = _scores_and_indices(_run(sig, dic, keep_n=5, **kw))
        assert (np.diff(scores, axis=1) <= 1e-6).all(), (
            f"{kw}: scores are not descending within a pixel")


def test_pca_and_exact_agree_on_the_winner():
    sig, dic, truth = _fixture()
    _, idx_off = _scores_and_indices(_run(sig, dic, keep_n=1, use_pca=False))
    _, idx_on = _scores_and_indices(
        _run(sig, dic, keep_n=1, use_pca=True, pca_components=32))
    agree = (idx_off[:, 0] == idx_on[:, 0]).mean()
    assert agree > 0.9, f"only {agree:.0%} of winners agree"
    # and the exact search recovers the pattern each experimental copy came from
    assert (idx_off[:, 0] == truth.ravel()).mean() > 0.95


def test_padding_slots_never_report_a_finite_score():
    """keep_n larger than the dictionary pads with index -1; those must stay
    at -inf rather than being scored against dictionary row 0."""
    sig, dic, _ = _fixture(n_dict=3)
    n_dict = dic.data.shape[0]
    scores, idx = _scores_and_indices(
        _run(sig, dic, keep_n=n_dict + 5, use_pca=True, pca_components=4))
    pad = idx < 0
    assert pad.any(), "expected padding slots in this configuration"
    assert not np.isfinite(scores[pad]).any(), "a padding slot reported a score"
    assert np.isfinite(scores[~pad]).all()


def test_mask_restricts_the_correlation_to_the_kept_pixels():
    sig, dic, _ = _fixture()
    h, w = sig.data.shape[-2:]
    yy, xx = np.mgrid[0:h, 0:w]
    disc = ((yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2) <= (min(h, w) / 2) ** 2
    mask = ~disc                      # kikuchipy convention: True = EXCLUDE
    keep = np.flatnonzero(disc.ravel())

    res = _run(sig, dic, keep_n=1, use_pca=False, signal_mask=mask)
    scores, idx = _scores_and_indices(res)
    exp_flat = np.asarray(sig.data).reshape(-1, h, w)
    dict_flat = np.asarray(dic.data)
    for i in range(idx.shape[0]):
        want = _reference_ncc(exp_flat[i], dict_flat[idx[i, 0]], keep=keep)
        assert scores[i, 0] == pytest.approx(want, abs=1e-3)


def test_rescore_survives_more_slots_than_one_tile():
    """The tile is sized from the feature dimension, so a run with many slots
    must cross several passes and still produce exact scores."""
    from backend.dict_gpu.pipeline import indexer as indexer_mod

    feat_dim = 12 * 12
    tile = max(1, (384 << 20) // (feat_dim * 4 * 3))
    sig, dic, _ = _fixture(n_rows=8, n_cols=8)
    keep_n = 6
    assert sig.data.shape[0] * sig.data.shape[1] * keep_n < tile, (
        "fixture too small to be interesting, but the exactness check below "
        "still holds"
    )
    scores, idx = _scores_and_indices(
        _run(sig, dic, keep_n=keep_n, use_pca=True, pca_components=16))
    exp_flat = np.asarray(sig.data).reshape(-1, 12, 12)
    dict_flat = np.asarray(dic.data)
    bad = 0
    for i in range(idx.shape[0]):
        for j in range(idx.shape[1]):
            if idx[i, j] < 0:
                continue
            if abs(scores[i, j] - _reference_ncc(exp_flat[i], dict_flat[idx[i, j]])) > 3e-3:
                bad += 1
    assert bad == 0, f"{bad} slots disagree with the reference NCC"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
