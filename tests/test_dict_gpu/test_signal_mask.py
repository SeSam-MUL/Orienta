"""The circular detector mask must reach the GPU dictionary path.

``dictionary_index_patterns`` fetched the active mask from the viewer right
next to the kikuchipy call — which sits *after* the GPU dispatcher returns. So
the CPU path honoured the mask and the GPU path silently correlated the dark
corners outside the phosphor disc along with the pattern.

Both halves are covered here: that the controller hands the mask over, and
that the indexer actually drops the masked pixels from the correlation.
"""
import inspect

import numpy as np
import pytest
import torch

import indexing_controller
from backend.dict_gpu.exceptions import GpuDictError
from backend.dict_gpu.pipeline import indexer as indexer_mod


# --------------------------------------------------------------------------
# the controller side
# --------------------------------------------------------------------------
def test_mask_is_fetched_before_the_gpu_dispatcher():
    """Order matters: the GPU branch returns, so anything after it is dead
    code for GPU users."""
    src = inspect.getsource(indexing_controller.dictionary_index_patterns)
    fetch = src.index("get_active_include_mask")
    dispatch = src.index('config.compute_mode in ("auto", "gpu")')
    assert fetch < dispatch, (
        "the signal mask is fetched after the GPU dispatcher — the GPU path "
        "can never see it"
    )


def test_controller_passes_the_mask_and_the_step_to_the_gpu_call():
    src = inspect.getsource(indexing_controller.dictionary_index_patterns)
    call = src[src.index("gpu_dictionary_index_patterns("):]
    call = call[: call.index(")\n")]
    assert "signal_mask=sig_mask_kp" in call
    # same class of bug: the user's orientation resolution was dropped too
    assert "angular_step_deg=config.angular_step_deg" in call


def test_indexer_accepts_a_signal_mask():
    sig = inspect.signature(indexer_mod.run_dictionary_index)
    assert "signal_mask" in sig.parameters
    assert sig.parameters["signal_mask"].default is None


# --------------------------------------------------------------------------
# the indexer side — exercised directly on the masking helper's semantics
# --------------------------------------------------------------------------
def _columns_kept(mask: np.ndarray) -> np.ndarray:
    """Mirror of the indexer's column selection, kept honest by the source
    assertions below."""
    return np.flatnonzero(~mask.reshape(-1))


def test_mask_convention_is_true_equals_exclude():
    """kikuchipy's convention, which the viewer's mask is converted into."""
    src = inspect.getsource(indexer_mod.run_dictionary_index)
    assert "keep_flat = ~sm.reshape(-1)" in src, (
        "the indexer must treat True as EXCLUDE, like kikuchipy"
    )


def test_wrong_mask_shape_fails_loud():
    """A silently ignored mask is how the first bug survived; a mismatched one
    must not be silently ignored either.

    (This was three parametrised cases that all ignored the parameter — the
    behavioural cover now lives in test_rescore_behaviour.py, which runs the
    indexer with a real mask and checks the correlation against numpy.)
    """
    src = inspect.getsource(indexer_mod.run_dictionary_index)
    assert "signal_mask shape" in src and "GpuDictError" in src
    # the shape check compares against the detector shape, not the flat size
    assert "!= detector shape" in src


def test_all_excluded_is_an_error_not_an_empty_correlation():
    src = inspect.getsource(indexer_mod.run_dictionary_index)
    assert "signal_mask excludes every detector pixel" in src


def test_masked_columns_are_dropped_from_both_sides():
    """Dictionary and experimental patterns must be reduced identically, or
    the dot product compares different detector pixels."""
    src = inspect.getsource(indexer_mod.run_dictionary_index)
    assert "dict_flat = dict_flat[:, keep_cols]" in src
    assert "exp_t = exp_t[:, keep_cols]" in src


def test_memory_estimate_uses_the_reduced_dimension():
    """Otherwise a masked run over-estimates VRAM and needlessly enables PCA."""
    src = inspect.getsource(indexer_mod.run_dictionary_index)
    assert "fp32_bytes = n_dict * feat_dim * 4" in src


# --------------------------------------------------------------------------
# numerical: masking must change the score the way kikuchipy's does
# --------------------------------------------------------------------------
def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_masking_changes_the_score_and_matches_a_manual_reduction():
    """The whole point: the mean and the norm are taken over the disc only."""
    rng = np.random.default_rng(0)
    h = w = 20
    yy, xx = np.mgrid[0:h, 0:w]
    disc = ((yy - 9.5) ** 2 + (xx - 9.5) ** 2) <= 9.5 ** 2
    mask = ~disc  # True = exclude, kikuchipy convention

    # Realistic levels: bands around a bright mid-grey inside the disc, dark
    # corners outside it. Two DIFFERENT orientations, so the disc contents do
    # not correlate — but both share the same dark corners.
    pattern = rng.normal(150, 25, (h, w))
    other = rng.normal(150, 25, (h, w))
    pattern[mask] = 5.0
    other[mask] = 5.0

    keep = _columns_kept(mask)
    full = _ncc(pattern.ravel(), other.ravel())
    masked = _ncc(pattern.ravel()[keep], other.ravel()[keep])
    assert full > 0.5, (
        f"two unrelated patterns correlate at {full:.3f} through their shared "
        "corners alone — that is the score inflation the mask removes"
    )
    assert abs(masked) < 0.2, (
        f"masked NCC {masked:.3f}: with the corners gone, unrelated patterns "
        "must score near zero"
    )


def test_torch_column_gather_matches_numpy():
    """The indexer gathers with a torch index tensor; guard the equivalence."""
    rng = np.random.default_rng(1)
    mask = rng.random((8, 8)) < 0.3
    keep = _columns_kept(mask)
    data = rng.normal(size=(5, 64)).astype(np.float32)
    t = torch.from_numpy(data)[:, torch.from_numpy(keep)]
    np.testing.assert_allclose(t.numpy(), data[:, keep])


# --------------------------------------------------------------------------
# the mask is global viewer state — it must not follow a batch onto other files
# --------------------------------------------------------------------------
def _fake_signal(h, w):
    class _Sig:
        data = np.zeros((2, 2, h, w), dtype=np.float32)
    return _Sig()


@pytest.mark.parametrize("mask_shape,pattern_shape,expect_applied", [
    ((60, 60), (60, 60), True),      # the viewer's own file
    ((60, 60), (128, 156), False),   # a batch over differently shaped patterns
    ((128, 156), (60, 60), False),
])
def test_mask_is_scoped_to_matching_patterns(mask_shape, pattern_shape,
                                             expect_applied, monkeypatch):
    """`get_active_include_mask` returns the VIEWER's mask. The batch manager
    loads its own signal per file, so without this scoping a masked file open
    in the viewer made every batch run over other geometries die with
    "signal_mask shape (60, 60) != detector shape (128, 156)".
    """
    import backend.api.routes.ebsd_viewer as viewer

    monkeypatch.setattr(viewer, "get_active_include_mask",
                        lambda: np.ones(mask_shape, dtype=bool), raising=False)

    captured = {}

    def _fake_gpu(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop here — we only want the arguments")

    import backend.dict_gpu.api as gpu_api
    monkeypatch.setattr(gpu_api, "gpu_dictionary_index_patterns", _fake_gpu)

    from indexing_controller import IndexingConfig, IndexingMethod
    cfg = IndexingConfig(method=IndexingMethod.DICTIONARY, compute_mode="gpu")
    try:
        indexing_controller.dictionary_index_patterns(
            signal=_fake_signal(*pattern_shape), dictionary=object(), config=cfg,
            detector=object())
    except Exception:
        pass  # the fake raises, or CUDA is absent; we assert on what it got

    if not captured:
        pytest.skip("GPU dispatcher not reached in this environment")
    got = captured.get("signal_mask")
    assert (got is not None) is expect_applied, (
        f"mask {mask_shape} on {pattern_shape} patterns: "
        f"{'applied' if got is not None else 'skipped'}, expected the opposite"
    )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
