"""A CrystalMap must carry ONE orientation per point.

User-reported, CPU dictionary path, full LoGainNi map:

    ipf-z: shape mismatch: value array of shape (561720,3) could not be
    broadcast to indexing result of shape (28086,3)

561720 = 28086 x 20, the default ``keep_n``. kikuchipy's
``dictionary_indexing`` stores every kept candidate as a map rotation, so
``xmap.rotations`` comes back ``(n, keep_n)``. The GPU path already returned
the best match only (``build_crystal_map`` keeps rank 0 and puts the rest in
``prop``), so the two paths handed downstream code structurally different maps
and only the GPU one could be drawn.

Fixed in two places, deliberately:
  - at the source, so new results are correct and the paths agree;
  - in the IPF colouring, so results SAVED before the fix still render instead
    of failing the whole layer.
"""
import numpy as np
import pytest
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Rotation

from indexing_controller import _best_match_only


def _keep_n_map(n_rows=3, n_cols=4, keep_n=5, seed=0):
    """A map shaped the way kikuchipy returns one.

    A real 2-D grid on purpose: with a single row orix drops ``y`` to None,
    which hides whether the collapse carries the coordinates through.
    """
    n = n_rows * n_cols
    rng = np.random.default_rng(seed)
    rot = Rotation(rng.normal(size=(n, keep_n, 4)))
    rot = Rotation(rot.data / np.linalg.norm(rot.data, axis=-1, keepdims=True))
    scores = np.sort(rng.random((n, keep_n)), axis=1)[:, ::-1].astype(np.float32)
    return CrystalMap(
        rotations=rot,
        phase_id=np.ones(n, dtype=int),
        x=np.tile(np.arange(n_cols, dtype=float), n_rows),
        y=np.repeat(np.arange(n_rows, dtype=float), n_cols),
        phase_list=PhaseList(Phase(name="Ni", point_group="m-3m")),
        prop={"scores": scores,
              "simulation_indices": rng.integers(0, 100, (n, keep_n))},
        scan_unit="px",
    )


def test_collapse_keeps_rank_zero():
    xmap = _keep_n_map()
    before = xmap.rotations.data.copy()
    out = _best_match_only(xmap)
    assert len(out.rotations.shape) == 1
    assert out.size == xmap.size
    np.testing.assert_allclose(out.rotations.data, before[:, 0])


def test_collapse_preserves_the_other_candidates_in_prop():
    """The pattern-match dialog reads simulation_indices by rank — dropping
    them would break it."""
    xmap = _keep_n_map()
    out = _best_match_only(xmap)
    assert out.prop["scores"].shape == (xmap.size, 5)
    assert out.prop["simulation_indices"].shape == (xmap.size, 5)
    np.testing.assert_array_equal(out.prop["scores"], xmap.prop["scores"])


def test_collapse_preserves_coordinates_and_phases():
    xmap = _keep_n_map()
    out = _best_match_only(xmap)
    np.testing.assert_allclose(out.x, xmap.x)
    np.testing.assert_allclose(out.y, xmap.y)
    np.testing.assert_array_equal(out.phase_id, xmap.phase_id)
    assert out.phases[1].point_group.name == "m-3m"
    assert out.scan_unit == xmap.scan_unit


def test_a_single_rotation_map_is_untouched():
    """The GPU path already returns this shape; it must pass through."""
    rng = np.random.default_rng(1)
    rot = Rotation(rng.normal(size=(9, 4)))
    xmap = CrystalMap(
        rotations=Rotation(rot.data / np.linalg.norm(rot.data, axis=-1, keepdims=True)),
        phase_id=np.ones(9, dtype=int),
        x=np.tile(np.arange(3, dtype=float), 3),
        y=np.repeat(np.arange(3, dtype=float), 3),
        phase_list=PhaseList(Phase(name="Ni", point_group="m-3m")), scan_unit="px")
    out = _best_match_only(xmap)
    assert out is xmap


# --------------------------------------------------------------------------
# the renderer half: an already-saved keep_n-deep result must still draw
# --------------------------------------------------------------------------
def test_ipf_colours_a_keep_n_deep_map_without_raising():
    """This is the exact failure the user hit — reproduced through the real
    colouring function, not a source-text assertion."""
    from tools.phase_map_generator import compute_ipf_colors

    n_rows, n_cols, keep_n = 4, 5, 20
    n = n_rows * n_cols
    rng = np.random.default_rng(2)
    q = rng.normal(size=(n, keep_n, 4))
    rot = Rotation(q / np.linalg.norm(q, axis=-1, keepdims=True))
    xmap = CrystalMap(
        rotations=rot,
        phase_id=np.ones(n, dtype=int),
        x=np.tile(np.arange(n_cols, dtype=float), n_rows),
        y=np.repeat(np.arange(n_rows, dtype=float), n_cols),
        phase_list=PhaseList(Phase(name="Ni", point_group="m-3m")),
        scan_unit="px",
    )
    rgb = compute_ipf_colors(xmap, "Z")
    assert rgb.shape == (n_rows, n_cols, 3), (
        f"got {rgb.shape}; before the fix this raised "
        f"'value array of shape ({n * keep_n},3) could not be broadcast to "
        f"indexing result of shape ({n},3)'"
    )
    assert np.isfinite(rgb).all()


def test_ipf_of_a_collapsed_map_equals_the_rank_zero_colours():
    """Collapsing must not change what the user sees — the colours have to be
    the ones rank 0 would have produced on its own."""
    from tools.phase_map_generator import compute_ipf_colors

    n_rows, n_cols, keep_n = 3, 4, 6
    n = n_rows * n_cols
    rng = np.random.default_rng(3)
    q = rng.normal(size=(n, keep_n, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)

    common = dict(
        phase_id=np.ones(n, dtype=int),
        x=np.tile(np.arange(n_cols, dtype=float), n_rows),
        y=np.repeat(np.arange(n_rows, dtype=float), n_cols),
        phase_list=PhaseList(Phase(name="Ni", point_group="m-3m")),
        scan_unit="px",
    )
    deep = CrystalMap(rotations=Rotation(q), **common)
    flat = CrystalMap(rotations=Rotation(q[:, 0]), **common)

    np.testing.assert_allclose(compute_ipf_colors(deep, "Z"),
                               compute_ipf_colors(flat, "Z"))


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
