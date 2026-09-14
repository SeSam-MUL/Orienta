import pytest, torch
import numpy as np
from backend.dict_gpu._pcadi.quantize import quantize_int8, dequantize_int8


@pytest.mark.gpu
def test_int8_roundtrip_per_row_unit_norm_input():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    rng = np.random.default_rng(0)
    # Per-row L2-normalised data — typical for our NCC use case
    X = torch.from_numpy(rng.standard_normal((200, 500)).astype(np.float32)).cuda()
    X = X / X.norm(dim=1, keepdim=True)

    q, scale, zp = quantize_int8(X)
    assert q.dtype == torch.int8
    assert q.shape == X.shape
    assert scale.shape == (X.shape[0],)         # per-row scale
    assert zp.shape == (X.shape[0],)            # per-row zero point

    X_rec = dequantize_int8(q, scale, zp)
    assert X_rec.shape == X.shape
    assert X_rec.dtype == torch.float32

    # Per-row max abs error must stay below ~1/127 of the row's range
    err_per_row = (X - X_rec).abs().max(dim=1).values
    row_range = X.max(dim=1).values - X.min(dim=1).values
    rel_err = err_per_row / row_range
    assert (rel_err < 0.02).all(), (
        f"max relative error per row exceeded 2% — got max {rel_err.max().item():.4f}"
    )


@pytest.mark.gpu
def test_int8_handles_constant_row():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    # An all-zeros row would naively trip a divide-by-zero in scale calc
    X = torch.zeros(3, 50, device="cuda", dtype=torch.float32)
    X[1] = 0.5  # one constant non-zero row
    q, scale, zp = quantize_int8(X)
    X_rec = dequantize_int8(q, scale, zp)
    assert torch.allclose(X_rec, X, atol=1e-5), (
        f"constant rows must round-trip exactly; got max abs diff "
        f"{(X - X_rec).abs().max().item()}"
    )
