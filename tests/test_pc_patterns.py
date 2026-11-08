"""Choosing the calibration patterns of "Find PC": the grid, the 3 x 3 blocks, the background."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import pc_patterns as pp  # noqa: E402


def _stack_reader(n_rows, n_cols, shape=(8, 8), value=lambda r, c: 0.0):
    """A scan whose pattern at (r, c) is the constant image ``value(r, c)`` + a fixed texture."""
    texture = np.random.default_rng(0).normal(size=shape).astype(np.float32)

    def read(rows, cols):
        return np.stack([texture + np.float32(value(int(r), int(c))) for r, c in zip(rows, cols)])
    return read


# ------------------------------------------------------------------ the grid (hand calculations)

def test_a_scan_of_225_patterns_or_fewer_is_looked_at_completely():
    assert pp.stride_for(15, 15) == 1            # 225 is not more than 225
    assert pp.stride_for(10, 10) == 1


def test_stride_follows_the_pattern_count():
    # 16 x 16 = 256 > 225: sqrt(256 / 225) = 1.067 -> 2, so rows 0, 2, ..., 14 (8) x 8 = 64 positions
    assert pp.stride_for(16, 16) == 2
    rows, cols, k = pp.sample_positions(16, 16)
    assert k == 2 and rows.size == cols.size == 64
    # 151 x 186 = 28086: sqrt(28086 / 225) = 11.17 -> 12, 13 rows x 16 columns = 208 <= 225
    rows, cols, k = pp.sample_positions(151, 186)
    assert k == 12 and rows.size == 13 * 16 == 208 and rows.max() == 144 and cols.max() == 180


def test_the_sample_of_a_normal_scan_follows_the_measurement_rule_by_hand():
    # 90 x 120 = 10800: sqrt(10800 / 225) = 6.93 -> 7; ceil(90 / 7) = 13 rows, ceil(120 / 7) = 18 columns
    rows, cols, k = pp.sample_positions(90, 120)
    assert k == 7 and rows.size == 13 * 18 == 234
    # 784 x 1121: sqrt(878864 / 225) = 62.5 -> 63; 13 x 18 = 234
    assert pp.sample_positions(784, 1121)[2] == 63


def test_no_scan_is_read_at_more_than_the_hard_cap_whatever_its_shape():
    """A one-row scan of 5000 patterns: the square-root rule gives stride 5 and 1000 positions;
    the stride is raised until at most 300 are read (17 -> ceil(5000 / 17) = 295)."""
    rows, _, k = pp.sample_positions(1, 5000)
    assert k == 17 and rows.size == 295
    for shape in [(5000, 1), (301, 251), (17, 14000), (2, 100000)]:
        assert pp.sample_positions(*shape)[0].size <= pp.HARD_CAP, shape


def test_the_hard_cap_holds_when_max_sample_is_larger_than_the_scan():
    """30 x 30 = 900 patterns with max_sample 1000: 900 <= 1000 would mean stride 1, but 900 > 300.
    Stride 1 is 900 positions, stride 2 is 15 x 15 = 225 <= 300, by hand."""
    assert pp.stride_for(30, 30, 1000) == 2
    assert pp.sample_positions(30, 30, 1000)[0].size <= pp.HARD_CAP
    assert pp.sample_positions(30, 30, 1000)[0].size == 225


def test_a_scan_of_nine_patterns_or_fewer_uses_every_pattern():
    """NORDIF's five calibration patterns (a 1 x 5 scan): all five are calibration patterns."""
    out = pp.pick(_stack_reader(1, 5), 1, 5, prepare=False, quality=lambda s: np.arange(len(s), dtype=float))
    assert out.coords == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    assert out.n_sampled == 5 and out.stride == 1 and out.patterns.shape == (5, 8, 8)


# ------------------------------------------------------------------ one per block

def _position_quality(n_rows, n_cols):
    """Quality 100 * row + column, in the order ``pick`` reads a scan that is read completely."""
    rr, cc = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    return lambda stack: (rr * 100 + cc).ravel().astype(float)      # noqa: E731


def test_one_pattern_per_block_the_best_of_its_block():
    """10 x 10 scan (100 patterns, read completely), quality = 100 * row + column, so the best
    pattern of a block is its bottom right corner. The blocks cut the scan at 10 / 3 and 20 / 3:
    rows 0-3 (0 <= r < 3.33), 4-6 (3.33 <= r < 6.67) and 7-9, by hand."""
    out = pp.pick(_stack_reader(10, 10), 10, 10, prepare=False, quality=_position_quality(10, 10))
    assert sorted(out.coords) == [(3, 3), (3, 6), (3, 9), (6, 3), (6, 6), (6, 9), (9, 3), (9, 6), (9, 9)]
    assert out.quality == [pytest.approx(r * 100 + c) for r, c in out.coords]


def test_the_blocks_are_blocks_of_the_scan_not_of_the_sample():
    """A 30 x 30 scan has 900 patterns -> stride 2 (sqrt(900 / 225) = 2), sampled rows 0, 2, ... 28.
    Block boundaries are at 10 and 20 of the SCAN: rows 0-9, 10-19, 20-29. With quality increasing in
    row and column the best sampled pattern of the first block is (8, 8) (the largest even index below 10)."""
    rows, cols, k = pp.sample_positions(30, 30)
    assert k == 2
    quality = lambda s: (rows * 100 + cols).astype(float)       # noqa: E731
    out = pp.pick(_stack_reader(30, 30), 30, 30, prepare=False, quality=quality)
    assert (8, 8) in out.coords and (28, 28) in out.coords and len(out.coords) == 9
    assert out.n_sampled == 225


def test_a_nan_quality_is_the_worst_value_and_a_block_of_only_nan_is_skipped_with_a_note():
    """10 x 10 scan read completely, quality = 100 * row + column; blocks are rows/columns 0-3, 4-6, 7-9.
    (3, 3) is NaN: the best of its block is then (3, 2) = 302 (against (2, 3) = 203).
    Every pattern of the block rows 7-9 x columns 7-9 is NaN: that block gives no pattern."""
    rr, cc = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
    scores = (rr * 100 + cc).astype(float)
    scores[(rr == 3) & (cc == 3)] = np.nan
    scores[(rr >= 7) & (cc >= 7)] = np.nan
    out = pp.pick(_stack_reader(10, 10), 10, 10, prepare=False, quality=lambda s: scores.ravel())
    assert sorted(out.coords) == [(3, 2), (3, 6), (3, 9), (6, 3), (6, 6), (6, 9), (9, 3), (9, 6)]
    assert len(out.quality) == 8 and not np.isnan(out.quality).any()
    assert out.patterns.shape == (8, 8, 8)
    assert out.notes == ["blockWithoutQuality"]


def test_a_non_finite_quality_is_none_in_both_branches_so_the_result_is_valid_json():
    """A scan read completely (1 x 5, nothing is chosen) keeps NaN and infinity out of ``quality``:
    both become None, a finite value stays. The result then passes json.dumps with allow_nan=False.
    The same holds for a chosen pattern of a 10 x 10 scan whose best value is infinite."""
    import json
    scores = np.array([1.0, np.nan, 3.0, np.inf, -np.inf])
    out = pp.pick(_stack_reader(1, 5), 1, 5, prepare=False, quality=lambda s: scores)
    assert out.quality == [1.0, None, 3.0, None, None]
    json.dumps(out.quality, allow_nan=False)

    rr, cc = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
    blocks = (rr * 100 + cc).astype(float)
    blocks[(rr == 3) & (cc == 3)] = np.inf          # the best of its block, not a number JSON holds
    out = pp.pick(_stack_reader(10, 10), 10, 10, prepare=False, quality=lambda s: blocks.ravel())
    assert (3, 3) in out.coords and out.quality[out.coords.index((3, 3))] is None
    assert sum(q is None for q in out.quality) == 1 and len(out.quality) == 9
    json.dumps(out.quality, allow_nan=False)


def test_no_note_when_every_block_has_a_quality():
    out = pp.pick(_stack_reader(10, 10), 10, 10, prepare=False, quality=_position_quality(10, 10))
    assert out.notes == []


def test_the_reader_is_asked_for_the_sampled_positions_only():
    asked = []

    def read(rows, cols):
        asked.append((rows.copy(), cols.copy()))
        return np.zeros((rows.size, 4, 4), np.float32)

    pp.pick(read, 100, 100, prepare=False, quality=lambda s: np.zeros(len(s)))
    # 100 x 100 = 10000: sqrt(10000 / 225) = 6.67 -> 7: ceil(100 / 7) = 15, so 15 x 15 = 225 positions, once
    assert len(asked) == 1 and asked[0][0].size == asked[0][1].size == 225
    assert asked[0][0].max() == 98 and asked[0][1].max() == 98          # 14 * 7


# ------------------------------------------------------------------ the background

def test_the_files_own_static_background_wins_when_it_fits():
    stack = np.ones((4, 6, 6), np.float32)
    kind, bg = pp.choose_background(stack, np.full((6, 6), 3.0))
    assert kind == "file" and float(bg.mean()) == 3.0
    kind, bg = pp.choose_background(stack, np.full((5, 5), 3.0))        # the wrong shape is not used
    assert kind == "mean" and float(bg.mean()) == 1.0
    kind, _ = pp.choose_background(stack, None)
    assert kind == "mean"


def test_preparing_removes_a_background_common_to_every_pattern_and_says_so():
    rng = np.random.default_rng(1)
    common = rng.normal(size=(24, 24)).astype(np.float32) * 5.0         # the same in every pattern
    own = rng.normal(size=(12, 24, 24)).astype(np.float32)               # different in every pattern

    def read(rows, cols):
        return np.stack([common + own[(int(r) * 4 + int(c)) % 12] for r, c in zip(rows, cols)])

    kept = pp.pick(read, 4, 4, prepare=False, quality=lambda s: np.zeros(len(s)))
    done = pp.pick(read, 4, 4, prepare=True, quality=lambda s: np.zeros(len(s)))
    assert kept.prepared is False and kept.background == "none"
    assert done.prepared is True and done.background == "mean"
    # what is shared by all patterns is gone: the pattern-to-pattern mean is far smaller than before
    assert np.abs(done.patterns.mean(axis=0)).mean() < 0.25 * np.abs(kept.patterns.mean(axis=0)).mean()
    assert done.patterns.dtype == np.float32 and np.isfinite(done.patterns).all()


def _random_reader(shape=(8, 8)):
    """Every position has its own random pattern (so a mean over them is not any one of them)."""
    def read(rows, cols):
        return np.stack([np.random.default_rng(int(r) * 1000 + int(c)).normal(size=shape).astype(np.float32)
                         for r, c in zip(rows, cols)])
    return read


def test_pick_hands_the_files_background_on_and_only_when_its_shape_fits():
    """16 patterns of 8 x 8 (stride 1): an 8 x 8 background is the file's, a 5 x 5 one is not used."""
    read = _random_reader()
    fits = pp.pick(read, 4, 4, prepare=True, file_background=np.full((8, 8), 100.0),
                   quality=lambda s: np.zeros(len(s)))
    assert fits.background == "file" and fits.prepared is True and fits.notes == []
    wrong = pp.pick(read, 4, 4, prepare=True, file_background=np.full((5, 5), 100.0),
                    quality=lambda s: np.zeros(len(s)))
    assert wrong.background == "mean" and wrong.notes == []


def test_a_single_pattern_gets_no_mean_background_and_a_note_instead_of_a_crash():
    """1 x 1 scan: the mean of the sampled patterns is the pattern itself (kikuchipy then divides by
    zero), 2 patterns would lose half of themselves. Below MIN_PATTERNS_FOR_MEAN = 3 there is none."""
    assert pp.MIN_PATTERNS_FOR_MEAN == 3
    for shape in [(1, 1), (1, 2), (2, 1)]:
        out = pp.pick(_random_reader(), *shape, prepare=True, quality=lambda s: np.zeros(len(s)))
        assert out.background == "none", shape
        assert out.notes == ["meanBackgroundSkipped"], shape
        assert out.patterns.shape == (shape[0] * shape[1], 8, 8) and np.isfinite(out.patterns).all()


def test_the_files_background_is_used_even_for_a_single_pattern_and_needs_no_note():
    out = pp.pick(_random_reader(), 1, 1, prepare=True, file_background=np.full((8, 8), 100.0),
                  quality=lambda s: np.zeros(len(s)))
    assert out.background == "file" and out.notes == []


def test_three_to_nine_patterns_get_the_mean_background_with_a_note_that_it_takes_part_of_each():
    """3 patterns: the mean holds a third of each; 9: a ninth. From 10 patterns on there is no note."""
    for shape in [(1, 3), (1, 5), (3, 3)]:
        out = pp.pick(_random_reader(), *shape, prepare=True, quality=lambda s: np.zeros(len(s)))
        assert out.background == "mean" and out.notes == ["meanFromFewPatterns"], shape
    out = pp.pick(_random_reader(), 2, 5, prepare=True, quality=lambda s: np.zeros(len(s)))
    assert out.background == "mean" and out.notes == []


def test_an_unprepared_scan_has_no_background_and_no_note():
    out = pp.pick(_random_reader(), 1, 1, prepare=False, quality=lambda s: np.zeros(len(s)))
    assert out.background == "none" and out.prepared is False and out.notes == []


def test_an_empty_scan_is_refused():
    with pytest.raises(ValueError):
        pp.pick(lambda r, c: np.zeros((0, 4, 4)), 0, 5)


def test_a_reader_that_returns_the_wrong_shape_is_refused():
    with pytest.raises(ValueError):
        pp.pick(lambda r, c: np.zeros((3, 4, 4)), 20, 20, prepare=False, quality=lambda s: np.zeros(len(s)))


# ------------------------------------------------------------------ other patterns: exclude

def test_excluded_positions_are_not_chosen_and_the_next_best_of_each_block_comes():
    """10 x 10 scan, quality 100 * row + column, blocks 0-3, 4-6, 7-9 in rows and columns (by hand). The nine
    bottom right corners are the first pick; with those excluded the best of each block is the position one
    column to the left (100 * r + c - 1 beats the row above, which is 100 lower)."""
    first = [(3, 3), (3, 6), (3, 9), (6, 3), (6, 6), (6, 9), (9, 3), (9, 6), (9, 9)]
    out = pp.pick(_stack_reader(10, 10), 10, 10, prepare=False, quality=_position_quality(10, 10), exclude=first)
    assert sorted(out.coords) == [(3, 2), (3, 5), (3, 8), (6, 2), (6, 5), (6, 8), (9, 2), (9, 5), (9, 8)]
    assert out.quality == [pytest.approx(r * 100 + c) for r, c in out.coords]


def test_a_block_with_nothing_left_gives_no_pattern():
    every_position_of_the_last_block = [(r, c) for r in range(7, 10) for c in range(7, 10)]
    out = pp.pick(_stack_reader(10, 10), 10, 10, prepare=False, quality=_position_quality(10, 10),
                  exclude=every_position_of_the_last_block)
    assert len(out.coords) == 8 and not any(r >= 7 and c >= 7 for r, c in out.coords)


def test_a_position_that_is_not_sampled_changes_nothing():
    plain = pp.pick(_stack_reader(10, 10), 10, 10, prepare=False, quality=_position_quality(10, 10))
    other = pp.pick(_stack_reader(10, 10), 10, 10, prepare=False, quality=_position_quality(10, 10),
                    exclude=[(50, 50), (-1, 2)])
    assert other.coords == plain.coords


def test_a_scan_that_is_used_completely_leaves_out_the_excluded_patterns():
    out = pp.pick(_stack_reader(1, 5), 1, 5, prepare=False, quality=lambda s: np.arange(len(s), dtype=float),
                  exclude=[(0, 1), (0, 3)])
    assert out.coords == [(0, 0), (0, 2), (0, 4)] and out.patterns.shape == (3, 8, 8)
    assert out.quality == [0.0, 2.0, 4.0]
