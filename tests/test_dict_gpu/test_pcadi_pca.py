import pytest, torch
import numpy as np
from backend.dict_gpu._pcadi.pca import GpuPCA

@pytest.mark.gpu
def test_pca_fit_then_transform_reduces_dim():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    rng = np.random.default_rng(0)
    X = torch.from_numpy(rng.standard_normal((1000, 200)).astype(np.float32)).cuda()
    pca = GpuPCA(n_components=32).fit(X)
    Y = pca.transform(X)
    assert Y.shape == (1000, 32)
    assert torch.isfinite(Y).all()

@pytest.mark.gpu
def test_pca_reconstruction_error_low_for_low_rank_data():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    rng = np.random.default_rng(0)
    # Data that lives in a 5-dim subspace
    basis = rng.standard_normal((5, 50)).astype(np.float32)
    coeffs = rng.standard_normal((300, 5)).astype(np.float32)
    X = torch.from_numpy(coeffs @ basis).cuda()
    pca = GpuPCA(n_components=5).fit(X)
    Y = pca.transform(X)
    X_rec = pca.inverse_transform(Y)
    err = (X - X_rec).norm() / X.norm()
    assert err.item() < 1e-3, f"reconstruction error {err.item()}"
