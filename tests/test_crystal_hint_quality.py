"""Unit tests for crystal_hint_quality (Mode 3)."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from backend.api.services.crystal_hint_quality import (
    QualityCheckResult,
    _is_compatible,
    _system_from_phase,
    _system_from_sg_number,
    quality_check_region,
)


def test_system_from_sg_number_boundaries():
    assert _system_from_sg_number(225) == "cubic"
    assert _system_from_sg_number(194) == "hexagonal"
    assert _system_from_sg_number(166) == "trigonal"
    assert _system_from_sg_number(139) == "tetragonal"
    assert _system_from_sg_number(63) == "orthorhombic"
    assert _system_from_sg_number(14) == "monoclinic"


def test_is_compatible_exact():
    assert _is_compatible(["cubic"], "cubic") is True
    assert _is_compatible(["cubic", "tetragonal"], "cubic") is True


def test_is_compatible_trigonal_hexagonal_degenerate():
    """Detected hexagonal symmetry is compatible with indexed trigonal (n=3
    is sub-symmetry of n=6)."""
    assert _is_compatible(["hexagonal"], "trigonal") is True
    assert _is_compatible(["trigonal"], "hexagonal") is True


def test_is_compatible_mismatch():
    assert _is_compatible(["cubic"], "hexagonal") is False
    assert _is_compatible(["tetragonal"], "cubic") is False


def test_is_compatible_empty_input_neutral():
    """When no symmetry was detected, we don't flag — return True (neutral)."""
    assert _is_compatible([], "cubic") is True
    assert _is_compatible(["cubic"], "") is True


def test_system_from_phase_with_attribute_string():
    """Mock a phase with a clean string return — should be returned lowered."""
    class _PG: pass
    class _Phase:
        @property
        def point_group(self):
            pg = _PG()
            pg.system = "Cubic"
            return pg
    sys = _system_from_phase(_Phase())
    assert sys == "cubic"


def test_system_from_phase_falls_back_to_sg_number():
    """If point_group.system raises, _system_from_phase should try the
    space_group.number fallback."""
    class _SG:
        number = 225
    class _Phase:
        @property
        def point_group(self):
            raise AttributeError("no point_group")
        @property
        def space_group(self):
            return _SG()
    sys = _system_from_phase(_Phase())
    assert sys == "cubic"


def _synthetic_pattern_4fold(size=64):
    rng = np.random.RandomState(0)
    img = rng.rand(size, size).astype(np.float32) * 0.05
    cy, cx = (size - 1) / 2, (size - 1) / 2
    y, x = np.indices((size, size))
    yy = y - cy; xx = x - cx
    r2 = np.sqrt(yy * yy + xx * xx)
    theta = np.arctan2(yy, xx)
    falloff = np.exp(-r2 / (size * 0.20))
    bands = np.zeros_like(img)
    for k in range(4):
        angle = 2 * np.pi * k / 4
        d_angle = np.abs(np.angle(np.exp(1j * (theta - angle))))
        bands += np.exp(-(d_angle / (np.pi / 12)) ** 2)
    img += (bands * falloff) * 2.0
    img += np.exp(-r2 ** 2 / (2 * (size * 0.025) ** 2)) * 8.0
    return img


def test_quality_check_no_xmap_returns_zeros():
    """With xmap=None we can't compare — should return zero counts + warn."""
    def get_pat(r, c): return _synthetic_pattern_4fold()
    result = quality_check_region(
        get_pattern=get_pat, xmap=None,
        n_rows=4, n_cols=4, max_pixels=100,
    )
    assert result.n_pixels_indexed == 0
    assert result.n_pixels_mismatch == 0
    assert len(result.warnings) >= 1
    assert "no active" in result.warnings[0].lower()


def _build_mock_xmap_cubic(n_rows, n_cols, system="cubic", name="Al cF4",
                           space_group_number=225):
    """Fake xmap mimicking the modern orix PhaseList API (`.ids` + `[id]`)
    with all pixels indexed as one phase.

    ``system`` is a parameter because the mismatch tests need a phase whose
    crystal system contradicts the pattern's symmetry; the default is the
    cubic Al the match tests use.
    """
    class _PointGroup:
        pass

    class _SpaceGroup:
        pass

    class _Phase:
        pass

    _PointGroup.system = system
    _SpaceGroup.number = space_group_number
    _Phase.name = name
    _Phase.point_group = _PointGroup()
    # Also expose space_group for the fallback path
    _Phase.space_group = _SpaceGroup()

    class _PhaseList:
        ids = [1]
        _phase = _Phase()

        def __getitem__(self, pid):
            return self._phase

    xmap = MagicMock()
    xmap.phase_id = np.ones(n_rows * n_cols, dtype=int)
    xmap.phases = _PhaseList()
    return xmap


def test_quality_check_all_match_cubic():
    """4-fold patterns indexed as cubic phase: all should MATCH."""
    n_rows, n_cols = 4, 4
    xmap = _build_mock_xmap_cubic(n_rows, n_cols)

    def get_pat(r, c): return _synthetic_pattern_4fold()
    result = quality_check_region(
        get_pattern=get_pat, xmap=xmap,
        n_rows=n_rows, n_cols=n_cols, max_pixels=100,
    )
    assert result.n_pixels_indexed == 16
    # Most synthetic 4-fold patterns should match cubic
    assert result.n_pixels_match >= 10
    assert result.n_pixels_mismatch + result.n_pixels_match + result.n_pixels_no_detection \
           == result.n_pixels_indexed


def _synthetic_pattern_nfold(n, seed=0, size=64):
    """A star of ``n`` bands around the zone axis, plus noise and a bright core."""
    rng = np.random.RandomState(seed)
    img = rng.rand(size, size).astype(np.float32) * 0.05
    cy = cx = (size - 1) / 2
    y, x = np.indices((size, size))
    yy = y - cy
    xx = x - cx
    r2 = np.sqrt(yy * yy + xx * xx)
    theta = np.arctan2(yy, xx)
    falloff = np.exp(-r2 / (size * 0.20))
    bands = np.zeros_like(img)
    for k in range(n):
        angle = 2 * np.pi * k / n
        d_angle = np.abs(np.angle(np.exp(1j * (theta - angle))))
        bands += np.exp(-(d_angle / (np.pi / 18)) ** 2)
    img += (bands * falloff) * 2.0
    img += np.exp(-r2 ** 2 / (2 * (size * 0.025) ** 2)) * 8.0
    return img


def test_quality_check_threefold_indexed_as_tetragonal_mismatch():
    """A pattern whose symmetry no fold shares with the indexed system is flagged.

    3-fold is compatible with cubic / hexagonal / trigonal, never tetragonal.
    The synthetic 3-fold scores (measured): 3-fold 0.87, 6-fold 0.11, 2-fold
    0.08, 4-fold 0.04 — only the 3-fold clears CONF_LOW (0.15), so the union
    below is {cubic, hexagonal, trigonal} and tetragonal is genuinely absent.
    """
    n_rows, n_cols = 3, 3
    xmap = _build_mock_xmap_cubic(n_rows, n_cols, system="tetragonal",
                                  name="Al7FeCu2", space_group_number=128)

    def get_pat(r, c):
        return _synthetic_pattern_nfold(3, seed=r * 10 + c)

    result = quality_check_region(
        get_pattern=get_pat, xmap=xmap,
        n_rows=n_rows, n_cols=n_cols, max_pixels=100,
    )
    judged = result.n_pixels_indexed - result.n_pixels_no_detection
    assert judged > 0, "the detector found no symmetry at all — nothing was judged"
    # Every judged pixel, not "at least half": measured 9 of 9, and the
    # inherited `>= judged // 2` is satisfied by zero as soon as judged is 1.
    assert result.n_pixels_mismatch == judged


def test_quality_check_sixfold_indexed_as_cubic_is_compatible():
    """The counter-case, and why it is NOT a mismatch.

    This was written the other way round — "6-fold indexed as cubic must
    mismatch" — and passed until `3cfec53b` (2026-05-28), which changed
    `compatible_systems` from the best fold's systems to the UNION over every
    fold above CONF_LOW. That was a real fix: a user's tetragonal Al7FeCu2
    pixel scored 4-fold 0.35 and was being flagged off-system because the
    3-fold at 0.44 won the tie-break alone.

    Under the union rule this case cannot mismatch, and should not: a perfect
    6-fold star also carries the 3-fold and 2-fold of its own axis (measured
    here: 6-fold 0.93, 3-fold 0.85, 2-fold 0.80), and a cubic <111> zone axis
    genuinely shows 3-fold.

    What this test pins is "the union rule is in force" — NOT "cubic arrives
    via the 3-fold". Measured, the union here is all six systems, because
    `compatible_systems_for_fold(2)` is all six and the 2-fold scores 0.80 on
    its own. A reverted 3cfec53b is still caught (best-fold-only gives
    ['hexagonal', 'trigonal'], cubic absent, 9 mismatches), which is the
    regression that matters; a narrower claim than that would need a pattern
    whose 2-fold stays under CONF_LOW, and a 6-fold star cannot be one.

    `n_pixels_match == n_pixels_indexed` is the load-bearing half: after
    `indexed += 1` the code has exactly three outcomes, so it forces both
    mismatch and no_detection to zero.
    """
    n_rows, n_cols = 3, 3
    xmap = _build_mock_xmap_cubic(n_rows, n_cols)

    def get_pat(r, c):
        return _synthetic_pattern_nfold(6, seed=r * 10 + c)

    result = quality_check_region(
        get_pattern=get_pat, xmap=xmap,
        n_rows=n_rows, n_cols=n_cols, max_pixels=100,
    )
    assert result.n_pixels_indexed > 0, "nothing was indexed — 0 == 0 proves nothing"
    assert result.n_pixels_mismatch == 0
    assert result.n_pixels_match == result.n_pixels_indexed


def test_quality_check_unindexed_pixels_counted_separately():
    """phase_id < 0 (orix sentinel for unindexed) should land in unindexed."""
    n_rows, n_cols = 3, 3
    xmap = _build_mock_xmap_cubic(n_rows, n_cols)
    # Mark some pixels as unindexed (phase_id = -1, orix convention)
    xmap.phase_id = np.array([-1, 1, -1, 1, 1, -1, 1, 1, 1])

    def get_pat(r, c): return _synthetic_pattern_4fold()
    result = quality_check_region(
        get_pattern=get_pat, xmap=xmap,
        n_rows=n_rows, n_cols=n_cols, max_pixels=100,
    )
    assert result.n_pixels_unindexed == 3
    assert result.n_pixels_indexed == 6


def test_quality_check_handles_orix_ids_iteration():
    """Regression for code-review B3: real orix PhaseList exposes .ids not
    a tuple-yielding iterator. The service must enumerate phases via the
    .ids property and access each phase via [id]."""
    n_rows, n_cols = 2, 2
    xmap = _build_mock_xmap_cubic(n_rows, n_cols)  # already uses .ids style

    def get_pat(r, c): return _synthetic_pattern_4fold()
    result = quality_check_region(
        get_pattern=get_pat, xmap=xmap,
        n_rows=n_rows, n_cols=n_cols, max_pixels=100,
    )
    # If the .ids enumeration was broken, phase_system_cache would be empty,
    # _is_compatible would always return True (default), and 0 mismatches
    # would ever be reported. Instead we should see the cube/4-fold match
    # registering as MATCHES (not mismatch).
    assert result.n_pixels_indexed > 0
    assert result.n_pixels_match >= result.n_pixels_indexed - result.n_pixels_no_detection - 1


def test_quality_check_phase_id_zero_is_indexed_not_unindexed():
    """Regression for code-review B2: orix uses -1 for unindexed, NOT 0."""
    n_rows, n_cols = 2, 2
    xmap = _build_mock_xmap_cubic(n_rows, n_cols)
    # All pixels are phase_id = 0 (valid first-phase id under EDAX convention)
    xmap.phase_id = np.zeros(n_rows * n_cols, dtype=int)
    # Phase list still maps via id 0 → phase. Use a custom list:
    class _PG: system = "cubic"
    class _Phase: name = "Al"; point_group = _PG()
    class _List:
        ids = [0]
        def __getitem__(self, pid): return _Phase()
    xmap.phases = _List()

    def get_pat(r, c): return _synthetic_pattern_4fold()
    result = quality_check_region(
        get_pattern=get_pat, xmap=xmap,
        n_rows=n_rows, n_cols=n_cols, max_pixels=100,
    )
    # phase_id == 0 must count as indexed, NOT unindexed
    assert result.n_pixels_unindexed == 0
    assert result.n_pixels_indexed == 4


def test_quality_check_phase_id_negative_one_is_unindexed():
    """Negative phase_id (orix sentinel) → unindexed bucket."""
    n_rows, n_cols = 2, 2
    xmap = _build_mock_xmap_cubic(n_rows, n_cols)
    xmap.phase_id = np.full(n_rows * n_cols, -1, dtype=int)

    def get_pat(r, c): return _synthetic_pattern_4fold()
    result = quality_check_region(
        get_pattern=get_pat, xmap=xmap,
        n_rows=n_rows, n_cols=n_cols, max_pixels=100,
    )
    assert result.n_pixels_unindexed == 4
    assert result.n_pixels_indexed == 0


def test_quality_check_serializable():
    import json

    n_rows, n_cols = 2, 2
    xmap = _build_mock_xmap_cubic(n_rows, n_cols)

    def get_pat(r, c): return _synthetic_pattern_4fold()
    result = quality_check_region(
        get_pattern=get_pat, xmap=xmap,
        n_rows=n_rows, n_cols=n_cols, max_pixels=100,
    )
    s = json.dumps(result.to_dict())
    assert "n_pixels_match" in s
