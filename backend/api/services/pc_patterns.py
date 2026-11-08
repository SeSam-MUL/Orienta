"""Choose the calibration patterns for "Find PC" from a whole scan.

The search of ``pc_search`` needs a handful of good patterns spread over the scan. The
measurement the search was developed on (five files from three makers) chose them like this, and so
does this module:

1. Look at a regular grid of positions, at most ``MAX_SAMPLE`` of them (every ``stride``-th row
   and column of the scan). A scan of ``MAX_SAMPLE`` patterns or fewer is looked at completely.
2. Remove the background from those patterns: the file's own static background when the signal
   has one, else the mean of the sampled patterns, then kikuchipy's dynamic background. (Not when
   the user already processed the signal in the viewer: then the patterns are taken as they are.)
3. Measure the image quality of each (kikuchipy's FFT image quality).
4. Split the SCAN into 3 x 3 blocks and take the best-quality sampled pattern of each block.

A scan of nine patterns or fewer (the five calibration patterns of a NORDIF ``Setting.txt``, say)
is not sampled: every pattern is a calibration pattern.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

#: Positions looked at (the measurement's value); see ``stride_for`` for the one case it may exceed.
MAX_SAMPLE = 225
#: No scan is read at more positions than this, whatever its shape.
HARD_CAP = 300
#: The scan is split into BLOCKS x BLOCKS blocks, one calibration pattern from each.
BLOCKS = 3
#: Fewer sampled patterns than this get no mean background: the mean of one pattern is the pattern
#: itself (kikuchipy then divides by zero), the mean of two takes half of each away.
MIN_PATTERNS_FOR_MEAN = 3


@dataclass
class PickResult:
    coords: List[Tuple[int, int]]
    patterns: np.ndarray                 # (n, rows, cols) float32, prepared or as read
    quality: List[Optional[float]]       # None where the value is not finite (NaN is not valid JSON)
    prepared: bool                       # background removed here
    background: str                     # "file" | "mean" | "none"
    n_sampled: int
    stride: int
    notes: List[str] = field(default_factory=list)


def _count(n_rows: int, n_cols: int, k: int) -> int:
    return int(math.ceil(n_rows / k)) * int(math.ceil(n_cols / k))


def stride_for(n_rows: int, n_cols: int, max_sample: int = MAX_SAMPLE, hard_cap: int = HARD_CAP) -> int:
    """Every ``stride``-th row and column is looked at.

    1 when the scan has ``max_sample`` patterns or fewer; otherwise the measurement's rule,
    ``ceil(sqrt(patterns / max_sample))`` (which gives 234 positions for a 90 x 120 scan), raised
    until at most ``hard_cap`` positions are read: a one-row scan of 5000 patterns would otherwise
    be read in full."""
    total = int(n_rows) * int(n_cols)
    if total <= min(max_sample, hard_cap):
        return 1
    k = int(math.ceil(math.sqrt(total / float(max_sample))))
    while _count(int(n_rows), int(n_cols), k) > hard_cap:
        k += 1
    return k


def sample_positions(n_rows: int, n_cols: int, max_sample: int = MAX_SAMPLE):
    """Rows, columns (flat arrays) of the sampled positions and the stride."""
    k = stride_for(n_rows, n_cols, max_sample)
    rr, cc = np.meshgrid(np.arange(0, n_rows, k), np.arange(0, n_cols, k), indexing="ij")
    return rr.ravel(), cc.ravel(), k


def _kikuchipy_quality(stack: np.ndarray) -> np.ndarray:
    import kikuchipy as kp
    signal = kp.signals.EBSD(np.asarray(stack, dtype=np.float32)[:, None])
    return np.asarray(signal.get_image_quality(show_progressbar=False), dtype=float).ravel()


def _prepare(stack: np.ndarray, background: Optional[np.ndarray]) -> np.ndarray:
    """Static background removed with ``background`` (not when it is None), then the dynamic background."""
    import kikuchipy as kp
    signal = kp.signals.EBSD(np.asarray(stack, dtype=np.float32)[:, None].copy())
    if background is not None:
        signal.remove_static_background(static_bg=np.asarray(background, dtype=np.float32),
                                        show_progressbar=False)
    signal.remove_dynamic_background(show_progressbar=False)
    return np.asarray(signal.data[:, 0], dtype=np.float32)


def choose_background(stack: np.ndarray, file_background=None):
    """``("file", bg)`` when the signal carries a static background of the patterns' shape, else
    ``("mean", mean of the sampled patterns)``."""
    if file_background is not None:
        bg = np.asarray(file_background)
        if bg.shape == tuple(stack.shape[1:]):
            return "file", bg.astype(np.float32)
    return "mean", np.asarray(stack, dtype=np.float32).mean(axis=0)


def _excluded_set(exclude) -> set:
    """``{(row, col), ...}`` of the positions that must not be chosen."""
    return {(int(p[0]), int(p[1])) for p in (exclude or ())}


def pick(read: Callable[[np.ndarray, np.ndarray], np.ndarray], n_rows: int, n_cols: int, *,
         prepare: bool = True, file_background=None, max_sample: int = MAX_SAMPLE,
         quality: Optional[Callable[[np.ndarray], np.ndarray]] = None,
         exclude=()) -> PickResult:
    """Choose calibration patterns from a scan of ``n_rows`` x ``n_cols`` patterns.

    ``read(rows, cols)`` returns the patterns at those positions as ``(m, rows, cols)``; it is the
    only thing that touches the data, so a lazy 27 GB scan is read for the sampled positions only.
    ``quality`` maps a stack to one number per pattern (default: kikuchipy's image quality).

    ``exclude`` are ``(row, col)`` positions that must not be chosen ("other patterns"): in each block
    the best position that is not excluded is taken. The sample, the background and the quality values
    do not depend on it, so the next best pattern of a block is the one a pick without ``exclude`` would
    have taken after the excluded ones. A block with nothing left gives no pattern.
    """
    banned = _excluded_set(exclude)
    n_rows, n_cols = int(n_rows), int(n_cols)
    if n_rows < 1 or n_cols < 1:
        raise ValueError("the scan has no patterns")
    quality = quality or _kikuchipy_quality
    if n_rows * n_cols <= BLOCKS * BLOCKS:
        rr, cc = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
        rr, cc = rr.ravel(), cc.ravel()
        stride = 1
        everything = True
    else:
        rr, cc, stride = sample_positions(n_rows, n_cols, max_sample)
        everything = False
    stack = np.asarray(read(rr, cc), dtype=np.float32)
    if stack.ndim != 3 or stack.shape[0] != rr.size:
        raise ValueError(f"read() must return ({rr.size}, rows, cols), got {stack.shape}")

    notes: List[str] = []
    background = "none"
    if prepare:
        background, bg = choose_background(stack, file_background)
        if background == "mean" and stack.shape[0] < MIN_PATTERNS_FOR_MEAN:
            background, bg = "none", None
            notes.append("meanBackgroundSkipped")
        elif background == "mean" and stack.shape[0] <= BLOCKS * BLOCKS:
            notes.append("meanFromFewPatterns")     # the mean holds 1/n of each pattern, n <= 9
        stack = _prepare(stack, bg)
    scores = np.asarray(quality(stack), dtype=float).ravel()

    allowed = np.array([(int(r), int(c)) not in banned for r, c in zip(rr, cc)], dtype=bool)
    if everything:
        chosen = [int(j) for j in np.flatnonzero(allowed)]
    else:
        chosen = []
        for bi in range(BLOCKS):
            for bj in range(BLOCKS):
                inside = ((rr >= bi * n_rows / BLOCKS) & (rr < (bi + 1) * n_rows / BLOCKS)
                          & (cc >= bj * n_cols / BLOCKS) & (cc < (bj + 1) * n_cols / BLOCKS)
                          & allowed)
                if inside.any():
                    candidates = np.flatnonzero(inside)
                    block = scores[candidates]
                    if np.isnan(block).all():
                        if "blockWithoutQuality" not in notes:
                            notes.append("blockWithoutQuality")
                        continue
                    # a NaN quality is the worst value, never the best
                    chosen.append(int(candidates[np.argmax(np.where(np.isnan(block), -np.inf, block))]))
    return PickResult(
        coords=[(int(rr[j]), int(cc[j])) for j in chosen],
        patterns=np.ascontiguousarray(stack[chosen]),
        quality=[float(scores[j]) if np.isfinite(scores[j]) else None for j in chosen],
        prepared=bool(prepare), background=background,
        n_sampled=int(rr.size), stride=int(stride), notes=notes)
