"""NLPAR pattern denoising for the EBSD Viewer (CPU, numba).

Non-local pattern averaging with re-indexing (Brewick, Wright, Rowenhorst,
Ultramicroscopy 200, 2019) as implemented in PyEBSDIndex. PyEBSDIndex's own
``NLPAR`` class works on FILES (file in, new file out) and only on raw 8-bit
patterns, which is the wrong shape for a viewer that holds patterns in memory
(possibly lazily, possibly after earlier processing). Its two numba kernels,
``NLPAR.sigma_numba`` and ``NLPAR.nlpar_nb``, work on in-memory float32 arrays,
so this module drives those kernels on strips of whole scan rows:

* pass 1 (sigma): the noise level of every pattern and the normalised pairwise
  distances used to choose lambda,
* lambda: the same optimisation PyEBSDIndex's ``opt_lambda`` runs
  (Nelder-Mead on three target weights, the middle optimum is used, as in
  ``auto_nlpar``),
* pass 2 (filter): the weighted average over the search window.

A strip carries ``radius`` extra rows on each side so that every pattern whose
result is kept sees exactly the neighbours it would see in a whole-scan call.
The saturation threshold of both kernels is derived from ``max(data)`` of the
array they are given; to make it the maximum of the WHOLE scan (what PyEBSDIndex
sees when the scan fits one chunk) and not of one strip, one extra, otherwise
unreferenced row holding the global maximum is appended to every strip. The
result is therefore independent of the strip height (tested bit for bit).

The output is a private array in the input dtype; the input is never written.
Cancelling or failing therefore leaves the caller's data untouched.

Pure functions: no FastAPI, no viewer state. The route lives in
``backend/api/routes/nlpar.py``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: What the viewer offers. Radius 3 is what the Ni gain series was measured with.
MIN_RADIUS = 1
MAX_RADIUS = 3
DEFAULT_RADIUS = 3

#: Target mean weights PyEBSDIndex's ``opt_lambda`` optimises for; ``auto_nlpar``
#: uses the middle one (index 1).
TARGET_WEIGHTS = (0.5, 0.34, 0.25)
LAMBDA_INDEX = 1

#: Float32 working memory per strip (input + output copy), in bytes.
STRIP_BUDGET_BYTES = 256 * 1024 ** 2

#: Rows of overlap the sigma pass needs: the kernel looks one pattern around
#: each pattern, and the lambda distances are normalised with the neighbour's
#: sigma, which in turn needs ITS neighbours.
_SIGMA_OVERLAP = 2

#: Scans with at least this many patterns are sub-sampled by 2 for lambda
#: (the same rule as PyEBSDIndex).
_LAMBDA_STRIDE_THRESHOLD = 1e6


class NlparCancelled(Exception):
    """Raised between strips when the cancel event is set."""


@dataclass
class NlparResult:
    data: np.ndarray                 # same shape and dtype as the input
    lam: float                       # lambda used
    lam_candidates: List[float]      # optimum for each target weight
    radius: int                      # effective search radius
    sigma_median: float
    n_patterns: int
    seconds: float
    n_strips: int
    pyebsdindex_version: str = ""


ProgressFn = Callable[[str, float, int, int], None]


# --------------------------------------------------------------------------- #
# kernels
# --------------------------------------------------------------------------- #

def _nlpar_class():
    """The numba (CPU) NLPAR class, regardless of whether PyEBSDIndex would
    pick an OpenCL class on this machine. Imported lazily: importing numba
    kernels costs seconds and must not slow down the backend start."""
    from pyebsdindex.nlpar_cpu import NLPAR
    return NLPAR


def pyebsdindex_version() -> str:
    try:
        import pyebsdindex
        return str(getattr(pyebsdindex, "__version__", ""))
    except Exception:
        return ""


_warm = False
_warm_lock = threading.Lock()


def warm_up() -> float:
    """Compile (or load from numba's cache) both kernels on a tiny array and
    return the seconds it took. The first call in a process can take tens of
    seconds; running it as its own stage keeps that from looking like a hang."""
    global _warm
    with _warm_lock:
        if _warm:
            return 0.0
        t0 = time.perf_counter()
        _run_tiny_pass()
        _warm = True
        return time.perf_counter() - t0


def _run_tiny_pass() -> None:
    cls = _nlpar_class()
    rows, cols, ph, pw = 4, 4, 6, 6
    rng = np.random.default_rng(0)
    buf = np.zeros((rows * cols + 1, ph * pw), dtype=np.float32)
    buf[: rows * cols] = rng.integers(0, 200, size=(rows * cols, ph * pw))
    indices = np.arange(ph * pw, dtype=np.int64)
    sigma, _, _ = cls.sigma_numba(
        buf, 1, rows, cols,
        np.array([0, rows], dtype=np.int64), np.array([0, cols], dtype=np.int64),
        indices, True)
    cls.nlpar_nb(
        buf, np.float32(1.0), 1, np.float32(0.0), np.asarray(sigma, dtype=np.float32),
        rows, cols,
        calclim=np.array([0, 0, cols, rows], dtype=np.int64),
        indices_in=indices, saturation_protect=True, diff_offset=np.float32(0.0))


# --------------------------------------------------------------------------- #
# geometry helpers (pure)
# --------------------------------------------------------------------------- #

def effective_radius(radius: int, n_rows: int, n_cols: int) -> int:
    """Largest usable radius <= ``radius``: the search window (2r+1)^2 must fit
    into the scan, otherwise the kernel's window arithmetic walks off the array.
    Returns 0 when not even radius 1 fits (scan smaller than 3x3)."""
    fit = (min(n_rows, n_cols) - 1) // 2
    return int(max(0, min(int(radius), fit)))


def window_start(j: int, radius: int, n_rows: int) -> int:
    """First row of the (2r+1)-row window the kernel uses for row ``j`` (it
    shifts the window instead of shrinking it at the scan border)."""
    return int(min(max(j - radius, 0), n_rows - 2 * radius - 1))


def filter_extent(a: int, b: int, radius: int, n_rows: int) -> Tuple[int, int]:
    """Rows to load so that core rows ``[a, b)`` see their full window."""
    return window_start(a, radius, n_rows), window_start(b - 1, radius, n_rows) + 2 * radius + 1


def sigma_extent(a: int, b: int, n_rows: int) -> Tuple[int, int]:
    return max(0, a - _SIGMA_OVERLAP), min(n_rows, b + _SIGMA_OVERLAP)


def strip_height(n_rows: int, n_cols: int, pattern_px: int, radius: int,
                 budget_bytes: int = STRIP_BUDGET_BYTES) -> int:
    """Core rows per strip so that the float32 working set (input + output of
    one strip, overlap included) stays near ``budget_bytes``. Never below
    ``6 * radius`` rows (the overlap would dominate) and never above the scan."""
    per_row = max(1, n_cols * pattern_px * 4 * 2)
    rows = budget_bytes // per_row - 2 * max(radius, _SIGMA_OVERLAP)
    rows = max(int(rows), 6 * radius, 1)
    return int(min(rows, n_rows))


def strip_working_bytes(n_rows: int, n_cols: int, pattern_px: int, radius: int,
                        budget_bytes: int = STRIP_BUDGET_BYTES) -> int:
    """Transient memory one filter strip costs: the float32 strip buffer plus
    the kernel's float32 result of the same size (both over the extended rows).

    Usually about ``budget_bytes``; but ``strip_height`` never goes below
    ``6 * radius`` core rows, so on wide scans of large patterns the floor wins
    and a strip is several times the budget (measured: 1.56 GiB for 500 columns
    of 128 x 156 at radius 3).
    """
    sr = effective_radius(radius, n_rows, n_cols)
    h = strip_height(n_rows, n_cols, pattern_px, sr, budget_bytes)
    e0, e1 = filter_extent(0, h, sr, n_rows)
    return int(((e1 - e0) * n_cols + 1) * pattern_px * 4 * 2)


def capacity_problem(n_patterns: int, pattern_px: int, itemsize: int,
                     available_bytes: int, max_bytes: int,
                     strip_bytes: int = 0) -> Optional[str]:
    """None when the denoised copy fits, else a user-facing reason.

    The result is an eager array of the input's size. It is refused when it
    would exceed ``max_bytes`` (the viewer's stash limit) or when less than
    2.5x its size, plus the strip working set ``strip_bytes``, is free in RAM,
    because the input stays alive next to it while the strips run.
    """
    out_bytes = int(n_patterns) * int(pattern_px) * int(itemsize)
    limit = min(int(max_bytes), int((int(available_bytes) - int(strip_bytes)) / 2.5))
    if out_bytes <= limit:
        return None
    max_patterns = max(0, limit // max(1, pattern_px * itemsize))
    return (f"NLPAR needs room for a second copy of the patterns "
            f"({out_bytes / 1024 ** 3:.1f} GB). Crop the scan first; "
            f"at most about {max_patterns} patterns fit here.")


# --------------------------------------------------------------------------- #
# lambda
# --------------------------------------------------------------------------- #

def optimise_lambda(d2: np.ndarray, dthresh: float = 0.0,
                    n_patterns: Optional[int] = None) -> List[float]:
    """The three lambda optima PyEBSDIndex's ``opt_lambda`` finds, one per
    target weight in ``TARGET_WEIGHTS``.

    ``d2`` is the normalised squared-distance array (rows, cols, 9) of the
    sigma pass. Same objective, same optimiser, same stride rule as upstream.
    """
    import scipy.optimize as opt

    dthresh32 = np.float32(dthresh)

    def loptfunc(lam, d2_, tw, dth):
        temp = np.maximum(d2_, dth)
        dw = np.exp(-temp / lam ** 2)
        w = np.sum(dw, axis=2) + 1e-12
        return np.mean(np.abs(tw - 1.0 / w))

    n = d2.shape[0] * d2.shape[1] if n_patterns is None else n_patterns
    stride = 1 if n < _LAMBDA_STRIDE_THRESHOLD else 2
    sample = d2[0::stride, 0::stride, :]
    out: List[float] = []
    for tw in TARGET_WEIGHTS:
        res = opt.minimize(loptfunc, 1.0, args=(sample, tw, dthresh32),
                           method="Nelder-Mead", bounds=[[0.001, 10.0]],
                           options={"fatol": 0.0001})
        out.append(float(np.atleast_1d(res["x"])[0]))
    return out


# --------------------------------------------------------------------------- #
# strip I/O
# --------------------------------------------------------------------------- #

def _read_rows(data, r0: int, r1: int) -> np.ndarray:
    """Rows ``[r0, r1)`` of an (rows, cols, H, W) array-like as a numpy array
    in the original dtype. Works for numpy and dask arrays."""
    return np.asarray(data[r0:r1])


def _global_max(data, strips: List[Tuple[int, int]], cancel) -> float:
    if isinstance(data, np.ndarray):
        return float(data.max())
    if hasattr(data, "chunks"):                    # dask: let it schedule the reads
        return float(data.max().compute())
    m = -np.inf
    for a, b in strips:
        _check_cancel(cancel)
        m = max(m, float(_read_rows(data, a, b).max()))
    return m


def _check_cancel(cancel: Optional[threading.Event]) -> None:
    if cancel is not None and cancel.is_set():
        raise NlparCancelled()


def _strip_buffer(block: np.ndarray, gmax: float) -> np.ndarray:
    """float32 (n_patterns + 1, H*W) copy of ``block`` plus the sentinel row
    that makes the kernels' ``max(data)`` the global maximum (see module
    docstring). The kernels never index the sentinel row."""
    n = block.shape[0] * block.shape[1]
    px = block.shape[2] * block.shape[3]
    buf = np.zeros((n + 1, px), dtype=np.float32)
    buf[:n] = block.reshape(n, px)
    buf[n, 0] = np.float32(gmax)
    return buf


def cast_like_input(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """float32 result -> input dtype the way PyEBSDIndex writes it back
    (``pat_flt2int(..., 'clip')``: clip to the type's range, truncate)."""
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return arr.astype(dtype)
    info = np.iinfo(dtype)
    return arr.clip(info.min, info.max).astype(dtype)


# --------------------------------------------------------------------------- #
# the denoiser
# --------------------------------------------------------------------------- #

def denoise(data, *, radius: int = DEFAULT_RADIUS, lam: Optional[float] = None,
            progress: Optional[ProgressFn] = None,
            cancel: Optional[threading.Event] = None,
            strip_rows: Optional[int] = None,
            budget_bytes: int = STRIP_BUDGET_BYTES) -> NlparResult:
    """Denoise an (rows, cols, H, W) pattern array. Returns a NEW array.

    ``data`` may be numpy or dask. ``lam`` fixes lambda (tests); by default it
    is estimated from the scan itself. ``strip_rows`` forces the strip height
    (tests); otherwise it follows ``budget_bytes``.
    """
    if getattr(data, "ndim", 0) != 4:
        raise ValueError("NLPAR needs a 2-D scan of patterns (rows x columns x H x W).")
    n_rows, n_cols, ph, pw = (int(s) for s in data.shape)
    px = ph * pw
    sr = effective_radius(radius, n_rows, n_cols)
    if sr < 1:
        raise ValueError("The scan is too small for NLPAR (at least 3 x 3 patterns are needed).")
    dtype = np.dtype(data.dtype)
    n_patterns = n_rows * n_cols

    def report(stage: str, frac: float, done: int = 0, total: int = 0) -> None:
        if progress is not None:
            progress(stage, float(min(1.0, max(0.0, frac))), done, total)

    t0 = time.perf_counter()
    cls = _nlpar_class()

    # stage weights: the sigma pass compares 8 neighbours per pattern, the
    # filter pass about half the window (pair distances are shared)
    pair_cost = ((2 * sr + 1) ** 2 - 1) / 2.0
    w_sigma = 8.0 / (8.0 + pair_cost)
    base_compile, base_max, base_sigma = 0.0, 0.03, 0.06
    span_sigma = 0.90 * w_sigma
    base_lambda = base_sigma + span_sigma
    base_filter = base_lambda + 0.02
    span_filter = 1.0 - base_filter

    report("compile", 0.0)
    _check_cancel(cancel)          # before the first compile, which can take tens of seconds
    warm_up()
    _check_cancel(cancel)

    h = strip_rows if strip_rows is not None else strip_height(n_rows, n_cols, px, sr, budget_bytes)
    h = int(max(1, min(h, n_rows)))
    strips = [(a, min(a + h, n_rows)) for a in range(0, n_rows, h)]
    n_strips = len(strips)

    report("max", base_max)
    gmax = _global_max(data, strips, cancel)

    mask = cls.makeautomask(ph, pw)
    indices = np.asarray(mask.flatten().nonzero()[0], dtype=np.int64)

    # ---- pass 1: sigma and normalised distances -----------------------------
    sigma = np.zeros((n_rows, n_cols), dtype=np.float32)
    d2 = np.zeros((n_rows, n_cols, 9), dtype=np.float32)
    for k, (a, b) in enumerate(strips):
        _check_cancel(cancel)
        e0, e1 = sigma_extent(a, b, n_rows)
        block = _read_rows(data, e0, e1)
        buf = _strip_buffer(block, gmax)
        n_ext = e1 - e0
        sig, dout, _ = cls.sigma_numba(
            buf, 1, n_ext, n_cols,
            np.array([0, n_ext], dtype=np.int64), np.array([0, n_cols], dtype=np.int64),
            indices, True)
        lo, hi = a - e0, b - e0
        sigma[a:b] = sig[lo:hi]
        d2[a:b] = dout[lo:hi]
        del block, buf, sig, dout
        report("sigma", base_sigma + span_sigma * (k + 1) / n_strips, k + 1, n_strips)

    sigma_median = float(np.median(sigma))

    # ---- lambda --------------------------------------------------------------
    report("lambda", base_lambda)
    _check_cancel(cancel)
    if lam is None:
        candidates = optimise_lambda(d2, 0.0, n_patterns)
        lam_used = candidates[LAMBDA_INDEX]
        if not np.isfinite(lam_used) or lam_used <= 0:
            raise ValueError("Could not estimate the NLPAR strength from this scan "
                             "(the patterns are too uniform or too few).")
    else:
        lam_used = float(lam)
        candidates = [lam_used]
    del d2

    # ---- pass 2: filter ------------------------------------------------------
    out = np.empty(data.shape, dtype=dtype)
    lam32 = np.float32(lam_used)
    dthresh32 = np.float32(0.0)
    for k, (a, b) in enumerate(strips):
        _check_cancel(cancel)
        e0, e1 = filter_extent(a, b, sr, n_rows)
        block = _read_rows(data, e0, e1)
        buf = _strip_buffer(block, gmax)
        n_ext = e1 - e0
        calclim = np.array([0, a - e0, n_cols, b - a], dtype=np.int64)
        res = cls.nlpar_nb(
            buf, lam32, sr, dthresh32, np.ascontiguousarray(sigma[e0:e1]),
            n_ext, n_cols, calclim=calclim, indices_in=indices,
            saturation_protect=True, diff_offset=np.float32(0.0))
        core = res[: n_ext * n_cols].reshape(n_ext, n_cols, ph, pw)[a - e0: b - e0]
        out[a:b] = cast_like_input(core, dtype)
        del block, buf, res, core
        report("filter", base_filter + span_filter * (k + 1) / n_strips, k + 1, n_strips)

    return NlparResult(
        data=out, lam=float(lam_used), lam_candidates=[float(c) for c in candidates],
        radius=int(sr), sigma_median=sigma_median, n_patterns=n_patterns,
        seconds=time.perf_counter() - t0, n_strips=n_strips,
        pyebsdindex_version=pyebsdindex_version())


# --------------------------------------------------------------------------- #
# advice: will NLPAR help on this scan, or only blur its fine structure?
# --------------------------------------------------------------------------- #
#
# NLPAR chooses its own strength lambda so that, on average, a pattern keeps a weight
# of 0.34 against its neighbours. Neighbours that differ from each other only by
# noise (very noisy scan, large grains) need a lambda of about 1; neighbours that
# differ by far more than the noise (clean scan, or structure finer than the search
# window) need a large one, because otherwise they get no weight. So lambda, which the
# run estimates anyway, says how much of the difference between neighbours is
# noise. It is read here from a sample of small blocks. Calibration against measured
# indexing results.

#: Estimated lambda at or above which NLPAR is NOT recommended.
LAMBDA_NOT_RECOMMENDED_FROM = 1.7

#: ~ number of sampled patterns the estimate is built from, and the block edge.
ADVICE_TARGET_PATTERNS = 1000
ADVICE_BLOCK = 12


@dataclass
class NoiseEstimate:
    lam: float                  # the lambda a run would choose, from the sample
    sigma: float                # noise level in counts (NLPAR's own sigma, median)
    n_patterns: int             # patterns the estimate rests on
    seconds: float


def _disc_mask(ph: int, pw: int) -> np.ndarray:
    """The mask NLPAR itself uses for its distances."""
    return _nlpar_class().makeautomask(ph, pw).astype(bool)


_OFFSETS = [(dj, di) for dj in (-1, 0, 1) for di in (-1, 0, 1) if (dj, di) != (0, 0)]


def _block_stats(flat: np.ndarray, ok: np.ndarray):
    """NLPAR's first pass on one small block, for the patterns that have full neighbourhoods.

    ``flat`` (b, b, K) float32 pattern pixels inside the disc mask, ``ok`` the same shape,
    False for saturated pixels. Returns

    * ``sigma`` (b-2, b-2): per pattern, the smallest mean squared difference to one of its
      8 neighbours, halved, square root (``sigma_numba``'s noise level); NaN where no
      neighbour differs;
    * ``dist`` (b-4, b-4, 9): the normalised squared distances ``opt_lambda`` optimises on
      (self entry -1, i.e. weight 1).

    Numpy only (asking for advice never waits for the numba compile); equality with the
    kernel is tested.
    """
    b0, b1 = flat.shape[:2]
    c, cok = flat[1:-1, 1:-1], ok[1:-1, 1:-1]
    d2s, n2s = [], []
    best = np.full(c.shape[:2], np.inf)
    for dj, di in _OFFSETS:
        nb = flat[1 + dj: b0 - 1 + dj, 1 + di: b1 - 1 + di]
        v = cok & ok[1 + dj: b0 - 1 + dj, 1 + di: b1 - 1 + di]
        diff = c - nb
        diff *= diff
        diff *= v
        d2 = diff.sum(-1)
        n2 = v.sum(-1) + 1e-12
        d2s.append(d2)
        n2s.append(n2)
        best = np.minimum(best, np.where(d2 >= 1e-3, d2 / (n2 * 2.0), np.inf))
    best[~np.isfinite(best)] = np.nan
    sigma = np.sqrt(best)
    s2_c = sigma[1:-1, 1:-1] ** 2
    dist = np.full(s2_c.shape + (9,), -1.0)
    for k, (dj, di) in enumerate(_OFFSETS):
        d2 = d2s[k][1:-1, 1:-1].astype(np.float64)
        n2 = n2s[k][1:-1, 1:-1]
        s2 = s2_c + sigma[1 + dj: b0 - 3 + dj, 1 + di: b1 - 3 + di] ** 2
        dist[..., k] = (d2 - n2 * s2) / (s2 * np.sqrt(2.0 * n2))
    return sigma, dist


def estimate_noise(data, *, target_patterns: int = ADVICE_TARGET_PATTERNS,
                   block: int = ADVICE_BLOCK, seed: int = 0) -> NoiseEstimate:
    """The lambda (and noise level) a run would find, from a sample of small blocks.

    ``data`` is (rows, cols, H, W), numpy or dask; only the blocks are read, so a lazy
    27 GB scan costs a few thousand patterns, not the whole file. Numpy only: asking
    for advice never waits for the numba compile.
    """
    t0 = time.perf_counter()
    if getattr(data, "ndim", 0) != 4:
        raise ValueError("needs a 2-D scan of patterns (rows x columns x H x W)")
    n_rows, n_cols, ph, pw = (int(v) for v in data.shape)
    if n_rows < 5 or n_cols < 5:
        raise ValueError("the scan is too small to estimate its noise (at least 5 x 5 patterns)")
    bh, bw = min(block, n_rows), min(block, n_cols)
    n_blocks = max(1, int(np.ceil(target_patterns / ((bh - 4) * (bw - 4)))))
    rng = np.random.default_rng(seed)
    r0s = rng.integers(0, n_rows - bh + 1, n_blocks)
    c0s = rng.integers(0, n_cols - bw + 1, n_blocks)
    blocks = [np.asarray(data[r:r + bh, c:c + bw]) for r, c in zip(r0s, c0s)]
    mxval = 0.9961 * float(max(b.max() for b in blocks))
    mask = _disc_mask(ph, pw).ravel()

    sig, d2s = [], []
    for blk in blocks:
        flat = blk.reshape(bh, bw, -1)[..., mask].astype(np.float32)
        ok = flat < np.float32(mxval)
        sigma, dist = _block_stats(flat, ok)
        sig.append(sigma[1:-1, 1:-1].ravel())
        d2s.append(dist.reshape(-1, 9))
    sig = np.concatenate(sig)
    d2 = np.concatenate(d2s)
    keep = np.isfinite(sig) & (sig > 0) & np.isfinite(d2).all(axis=1)
    if keep.sum() < 100:
        raise ValueError("the patterns are too uniform to estimate their noise")
    lam = optimise_lambda(d2[keep].astype(np.float32).reshape(-1, 1, 9), 0.0)[LAMBDA_INDEX]
    return NoiseEstimate(lam=float(lam), sigma=float(np.median(sig[keep])),
                         n_patterns=int(keep.sum()), seconds=time.perf_counter() - t0)


def advice_for(estimate: NoiseEstimate) -> str:
    """'recommended' or 'not_recommended' for a noise estimate."""
    return "not_recommended" if estimate.lam >= LAMBDA_NOT_RECOMMENDED_FROM else "recommended"


# --------------------------------------------------------------------------- #
# signal helpers (hyperspy objects, still no FastAPI)
# --------------------------------------------------------------------------- #

METADATA_KEY = "Orienta.nlpar"


def applied_info(signal) -> Optional[dict]:
    """The provenance mark of an earlier run, or None."""
    try:
        node = signal.metadata.get_item(METADATA_KEY, None)
    except Exception:
        return None
    if node is None:
        return None
    return node.as_dictionary() if hasattr(node, "as_dictionary") else dict(node)


def mark_applied(signal, result: NlparResult) -> None:
    signal.metadata.set_item(METADATA_KEY, {
        "radius": int(result.radius),
        "lam": float(result.lam),
        "sigma_median": float(result.sigma_median),
        "pyebsdindex": result.pyebsdindex_version,
    })


def swap_data(signal, new_data: np.ndarray) -> None:
    """Replace the pattern array of ``signal`` by an eager numpy array.

    A lazy signal (dask, possibly backed by an open file) becomes eager: the
    data is assigned, the lazy flag cleared and the class re-assigned exactly
    as hyperspy's own ``compute()`` does it.
    """
    lazy = bool(getattr(signal, "_lazy", False))
    signal.data = new_data
    if lazy:
        signal._lazy = False
        signal._assign_subclass()
