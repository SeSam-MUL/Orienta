import pytest
import numpy as np
import torch


def _random_unit_quats(n, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, 4)).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q


def test_build_crystal_map_full_mask_30x30():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA — keeps test fast and consistent with rest of suite")
    from backend.dict_gpu.pipeline.output import build_crystal_map
    from orix.quaternion import Rotation
    from orix.crystal_map import PhaseList

    n_rows, n_cols = 30, 30
    n = n_rows * n_cols
    n_dict = 500
    rng = np.random.default_rng(0)

    rotations_dict = Rotation(_random_unit_quats(n_dict, seed=1))
    indices_top1 = rng.integers(0, n_dict, size=n)
    scores_top1 = rng.uniform(0.3, 0.95, size=n).astype(np.float32)

    top_indices = torch.from_numpy(indices_top1.reshape(-1, 1)).cuda()
    top_scores = torch.from_numpy(scores_top1.reshape(-1, 1)).cuda()
    selection_mask = np.ones((n_rows, n_cols), dtype=bool)
    phase_list = PhaseList()  # default single-phase setup

    xmap = build_crystal_map(
        top_indices, top_scores, rotations_dict,
        phase_list, selection_mask, (n_rows, n_cols)
    )

    assert xmap.size == n_rows * n_cols
    assert xmap.scan_unit == "px"
    # Coordinates cover the full original grid
    assert sorted(np.unique(xmap.x)) == list(range(n_cols))
    assert sorted(np.unique(xmap.y)) == list(range(n_rows))
    # Best-match rotations correspond to the chosen dict indices
    expected_rot = rotations_dict[indices_top1]
    actual_rot = xmap.rotations
    np.testing.assert_allclose(actual_rot.data, expected_rot.data, atol=1e-6)
    assert "scores" in xmap.prop
    np.testing.assert_allclose(xmap.prop["scores"], scores_top1.reshape(-1, 1), atol=1e-6)


def test_build_crystal_map_partial_mask_back_maps_to_original_coords():
    """Half-masked pixels: xmap coords must match np.argwhere(mask) order."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from backend.dict_gpu.pipeline.output import build_crystal_map
    from orix.quaternion import Rotation
    from orix.crystal_map import PhaseList

    n_rows, n_cols = 30, 30
    rng = np.random.default_rng(0)
    selection_mask = rng.random((n_rows, n_cols)) > 0.5  # ~50% selected
    n_selected = int(selection_mask.sum())

    n_dict = 100
    rotations_dict = Rotation(_random_unit_quats(n_dict, seed=2))
    indices = rng.integers(0, n_dict, size=n_selected)
    scores = rng.uniform(0.3, 0.9, size=n_selected).astype(np.float32)

    top_indices = torch.from_numpy(indices.reshape(-1, 1)).cuda()
    top_scores = torch.from_numpy(scores.reshape(-1, 1)).cuda()

    xmap = build_crystal_map(
        top_indices, top_scores, rotations_dict,
        PhaseList(), selection_mask, (n_rows, n_cols)
    )

    assert xmap.size == n_selected
    expected_rc = np.argwhere(selection_mask)  # (n_selected, 2) of (row, col)
    np.testing.assert_array_equal(xmap.x.astype(int), expected_rc[:, 1])
    np.testing.assert_array_equal(xmap.y.astype(int), expected_rc[:, 0])


def test_build_crystal_map_keeps_top_k_in_props():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from backend.dict_gpu.pipeline.output import build_crystal_map
    from orix.quaternion import Rotation
    from orix.crystal_map import PhaseList

    n_rows, n_cols = 10, 10
    n = n_rows * n_cols
    n_dict = 50
    k = 5
    rng = np.random.default_rng(0)
    rotations_dict = Rotation(_random_unit_quats(n_dict, seed=3))
    indices = np.argsort(-rng.random((n, n_dict)))[:, :k]
    scores = -np.sort(-rng.uniform(0.3, 0.95, size=(n, n_dict)))[:, :k].astype(np.float32)

    top_indices = torch.from_numpy(indices).cuda()
    top_scores = torch.from_numpy(scores).cuda()
    selection_mask = np.ones((n_rows, n_cols), dtype=bool)

    xmap = build_crystal_map(
        top_indices, top_scores, rotations_dict,
        PhaseList(), selection_mask, (n_rows, n_cols)
    )
    assert xmap.prop["scores"].shape == (n, k)
    # Best-match rotation = rotations_dict[indices[:, 0]]
    np.testing.assert_allclose(
        xmap.rotations.data, rotations_dict[indices[:, 0]].data, atol=1e-6
    )
