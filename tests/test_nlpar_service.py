"""NLPAR service: the in-memory strip driver against PyEBSDIndex's public file API.

The service re-uses PyEBSDIndex's two numba kernels (``NLPAR.sigma_numba`` and
``NLPAR.nlpar_nb``), which are internal API. The contract tests below pin the
service to the PUBLIC file-based ``nlpar_cpu.NLPAR`` on the same data: if a
PyEBSDIndex update changes what the kernels compute (or their signature), these
fail, not the user. Run them in every environment that ships a PyEBSDIndex.
"""

import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("pyebsdindex")
pytest.importorskip("numba")

from backend.api.services import nlpar as N  # noqa: E402


# --------------------------------------------------------------------------- #
# data: real patterns (kikuchipy's nickel set), tiled into grains, plus noise
# --------------------------------------------------------------------------- #

def make_scan(rows=12, cols=14, sigma=25.0, seed=1, with_clean=False):
    import kikuchipy as kp

    base = np.asarray(kp.data.nickel_ebsd_small(lazy=False).data)   # (3, 3, 60, 60) uint8
    clean = np.empty((rows, cols) + base.shape[2:], dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            clean[r, c] = base[(r // 4) % 3, (c // 5) % 3]
    rng = np.random.default_rng(seed)
    noisy = np.clip(clean + rng.normal(0, sigma, clean.shape), 0, 255).astype(np.uint8)
    # One saturated detector corner pixel (outside the NLPAR disc mask, so it does
    # not enter any distance). It makes the scan maximum 255 in EVERY chunk, which
    # keeps the PyEBSDIndex file API independent of how it splits a scan into
    # chunks (it derives the saturation threshold from each chunk's own maximum;
    # 0.3.10.1 always splits into at least two chunks, 0.3.9.1 does not).
    noisy[..., 0, 0] = 255
    return (noisy, clean) if with_clean else noisy


@pytest.fixture(scope="module")
def scan():
    return make_scan()


def write_up1(path, arr_u8):
    """EDAX .up1 (version 1): 4 x uint32 header [1, w, h, 16], then uint8 patterns."""
    nr, nc, h, w = arr_u8.shape
    with open(path, "wb") as f:
        f.write(np.array([1, w, h, 16], dtype=np.uint32).tobytes())
        f.write(np.ascontiguousarray(arr_u8.reshape(nr * nc, h, w), dtype=np.uint8).tobytes())


def read_up1(path, nr, nc):
    hdr = np.fromfile(path, dtype=np.uint32, count=4)
    w, h, off = int(hdr[1]), int(hdr[2]), int(hdr[3])
    return np.fromfile(path, dtype=np.uint8, offset=off).reshape(nr, nc, h, w)


def file_api(arr, tmp_path, radius, lam=None):
    """PyEBSDIndex's own CPU NLPAR on a file, exactly as ``auto_nlpar`` drives it."""
    from pyebsdindex.nlpar_cpu import NLPAR

    nr, nc = arr.shape[:2]
    fin, fout = tmp_path / "in.up1", tmp_path / "out.up1"
    write_up1(fin, arr)
    obj = NLPAR(str(fin), lam=0.7, searchradius=radius)
    obj.nrows, obj.ncols = nr, nc
    lams = None
    if lam is None:
        lams = [float(x) for x in np.atleast_1d(obj.opt_lambda(autoupdate=True, verbose=0))]
        lam = lams[1]
    obj.calcnlpar(fileout=str(fout), searchradius=radius, lam=lam, verbose=0)
    return read_up1(fout, nr, nc), lams, lam


# --------------------------------------------------------------------------- #
# (a) the public file API is the reference
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("radius", [1, 2, 3])
def test_service_equals_pyebsdindex_file_api(scan, tmp_path, radius):
    ref, lams, lam = file_api(scan, tmp_path, radius)
    res = N.denoise(scan, radius=radius)
    assert res.radius == radius
    # lambda: same optimiser on the same normalised distances. Within 0.02 (the
    # acceptance bound): PyEBSDIndex 0.3.10.1 estimates it from
    # chunks whose edge rows see incomplete neighbourhoods; 0.3.9.1 is exact.
    assert res.lam_candidates == pytest.approx(lams, abs=0.02)
    assert res.lam == pytest.approx(lam, abs=0.02)
    # pixels: with the SAME lambda the filter is identical
    res = N.denoise(scan, radius=radius, lam=lam)
    diff = np.abs(res.data.astype(int) - ref.astype(int))
    assert diff.max() <= 1, f"{(diff > 0).sum()} values differ, max {diff.max()}"


def test_service_with_fixed_lambda_equals_file_api(scan, tmp_path):
    ref, _, _ = file_api(scan, tmp_path, 3, lam=0.9)
    res = N.denoise(scan, radius=3, lam=0.9)
    assert np.abs(res.data.astype(int) - ref.astype(int)).max() <= 1


def test_real_ni_gain_crop_equals_file_api(tmp_path):
    """Real 8-bit NORDIF patterns (public Ni gain series, 24 dB scan), 20 x 20 crop.
    ``ORIENTA_TEST_DATA`` may point at a Test_data folder that is not this checkout's."""
    import kikuchipy as kp
    from tests.data_deps import TEST_DATA_DIR, need

    data_dir = Path(os.environ.get("ORIENTA_TEST_DATA") or TEST_DATA_DIR)
    scan_dir = need(data_dir / "ni_gain_anes2023" / "scan10_gain24db")
    path = next(scan_dir.glob("**/Pattern.dat"))
    sig = kp.load(str(path), lazy=True)
    crop = np.asarray(sig.data[:20, :20]).astype(np.uint8)
    ref, lams, lam = file_api(crop, tmp_path, 3)
    res = N.denoise(crop, radius=3)
    assert res.lam_candidates == pytest.approx(lams, abs=0.02)
    assert np.abs(N.denoise(crop, radius=3, lam=lam).data.astype(int) - ref.astype(int)).max() <= 1


# --------------------------------------------------------------------------- #
# (b) the strip height must not matter, bit for bit
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("strip_rows", [1, 3, 4, 5, 11])
def test_strip_height_does_not_change_the_result(scan, strip_rows):
    whole = N.denoise(scan, radius=3, strip_rows=scan.shape[0])
    strips = N.denoise(scan, radius=3, strip_rows=strip_rows)
    assert whole.n_strips == 1 and strips.n_strips > 1
    assert strips.lam == whole.lam
    assert strips.sigma_median == whole.sigma_median
    assert np.array_equal(strips.data, whole.data)


def test_saturated_pixels_are_judged_on_the_whole_scan_not_per_strip():
    """A strip without the brightest pixel must not move the saturation threshold.

    Only the first pattern reaches 255; everywhere else the scan tops out near
    200, so each later strip, judged alone, would call a different set of pixels
    saturated than the scan as a whole does.
    """
    scan = make_scan(rows=12, cols=14)
    scan = (scan.astype(np.float32) * 0.8).astype(np.uint8)
    scan[0, 0, 30, 30] = 255
    assert scan[2:].max() < 230
    whole = N.denoise(scan, radius=2, strip_rows=12)
    strips = N.denoise(scan, radius=2, strip_rows=2)
    assert strips.n_strips == 6
    assert np.array_equal(strips.data, whole.data)


# --------------------------------------------------------------------------- #
# (c) lazy input equals eager input
# --------------------------------------------------------------------------- #

def test_lazy_dask_input_equals_eager(scan):
    da = pytest.importorskip("dask.array")
    lazy = da.from_array(scan, chunks=(3, 5, 60, 60))
    a = N.denoise(scan, radius=2, strip_rows=4)
    b = N.denoise(lazy, radius=2, strip_rows=4)
    assert np.array_equal(a.data, b.data)
    assert isinstance(b.data, np.ndarray)


def test_swap_data_makes_a_lazy_signal_eager(scan):
    import kikuchipy as kp

    sig = kp.signals.EBSD(scan.copy()).as_lazy()
    assert sig._lazy
    res = N.denoise(scan, radius=1)
    N.swap_data(sig, res.data)
    assert not sig._lazy
    assert isinstance(sig, kp.signals.EBSD) and not isinstance(sig, kp.signals.LazyEBSD)
    assert isinstance(sig.data, np.ndarray)
    assert np.array_equal(sig.data, res.data)


def test_swap_data_on_an_eager_signal(scan):
    import kikuchipy as kp

    sig = kp.signals.EBSD(scan.copy())
    res = N.denoise(scan, radius=1)
    N.swap_data(sig, res.data)
    assert not sig._lazy and np.array_equal(sig.data, res.data)


# --------------------------------------------------------------------------- #
# (d) cancel and errors leave the input untouched
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("stop_at", ["sigma", "filter"])
def test_cancel_stops_between_strips_and_leaves_the_input_bit_identical(scan, stop_at):
    original = scan.copy()
    ev = threading.Event()
    seen = []

    def progress(stage, frac, done, total):
        seen.append(stage)
        if stage == stop_at and done == 1:
            ev.set()

    with pytest.raises(N.NlparCancelled):
        N.denoise(scan, radius=3, strip_rows=3, progress=progress, cancel=ev)
    assert stop_at in seen
    assert np.array_equal(scan, original)
    if stop_at == "sigma":
        assert "filter" not in seen            # it really stopped, not just raised at the end


def test_cancel_set_before_the_start_does_no_work(scan):
    ev = threading.Event()
    ev.set()
    with pytest.raises(N.NlparCancelled):
        N.denoise(scan, radius=3, cancel=ev)


def test_input_is_never_written(scan):
    original = scan.copy()
    N.denoise(scan, radius=3, strip_rows=2)
    assert np.array_equal(scan, original)


# --------------------------------------------------------------------------- #
# (e) memory guard
# --------------------------------------------------------------------------- #

def test_capacity_ok_when_the_copy_fits():
    assert N.capacity_problem(10_000, 3600, 1, available_bytes=8 * 1024 ** 3,
                              max_bytes=4 * 1024 ** 3) is None


def test_capacity_refuses_above_the_stash_limit_and_names_a_pattern_count():
    msg = N.capacity_problem(5_000_000, 3600, 1, available_bytes=64 * 1024 ** 3,
                             max_bytes=4 * 1024 ** 3)
    assert msg and "Crop" in msg
    assert str(4 * 1024 ** 3 // 3600) in msg


def test_capacity_refuses_when_ram_is_short():
    # 1 GB output needs 2.5 GB free
    out = 1024 ** 3
    assert N.capacity_problem(out // 3600, 3600, 1, available_bytes=int(2.4 * 1024 ** 3),
                              max_bytes=4 * 1024 ** 3)
    assert N.capacity_problem(out // 3600, 3600, 1, available_bytes=int(2.6 * 1024 ** 3),
                              max_bytes=4 * 1024 ** 3) is None


def test_capacity_counts_the_strip_working_set():
    """The strip floor (6 * radius core rows) can cost several times the budget on
    wide scans of large patterns; a run that fits only without it is refused."""
    out = 1024 ** 3
    strip = 512 * 1024 ** 2
    avail = int(2.6 * 1024 ** 3)                    # enough for 2.5 x out alone ...
    assert N.capacity_problem(out // 3600, 3600, 1, available_bytes=avail,
                              max_bytes=4 * 1024 ** 3, strip_bytes=strip)    # ... not with the strips
    assert N.capacity_problem(out // 3600, 3600, 1, available_bytes=avail + strip,
                              max_bytes=4 * 1024 ** 3, strip_bytes=strip) is None


def test_strip_working_bytes_is_the_buffer_and_result_of_the_largest_strip():
    """500 columns of 128 x 156 at radius 3 (the measured crop): the floor gives
    18 core rows, 21 extended, two float32 copies."""
    px = 128 * 156
    h = N.strip_height(120, 500, px, 3)
    assert h == 18                                   # the floor, not the 256 MiB budget
    e0, e1 = N.filter_extent(0, h, 3, 120)
    expected = ((e1 - e0) * 500 + 1) * px * 4 * 2
    got = N.strip_working_bytes(120, 500, px, 3)
    assert got == expected and got > 6 * N.STRIP_BUDGET_BYTES


# --------------------------------------------------------------------------- #
# behaviour around the edges
# --------------------------------------------------------------------------- #

def test_it_denoises():
    noisy, clean = make_scan(sigma=30.0, with_clean=True)
    res = N.denoise(noisy, radius=3)
    before = np.abs(noisy.astype(float) - clean).mean()
    after = np.abs(res.data.astype(float) - clean).mean()
    assert after < 0.7 * before


def test_dtype_and_shape_are_preserved(scan):
    res = N.denoise(scan, radius=1)
    assert res.data.dtype == np.uint8 and res.data.shape == scan.shape
    wide = scan.astype(np.uint16) * 200          # 16-bit detector counts
    res16 = N.denoise(wide, radius=1)
    assert res16.data.dtype == np.uint16
    assert res16.data.max() <= 65535
    res_f = N.denoise(scan.astype(np.float32), radius=1)
    assert res_f.data.dtype == np.float32


def test_radius_is_clamped_to_what_fits_the_scan():
    assert N.effective_radius(3, 12, 14) == 3
    assert N.effective_radius(3, 5, 40) == 2
    assert N.effective_radius(3, 3, 3) == 1
    assert N.effective_radius(3, 2, 40) == 0
    small = make_scan(rows=3, cols=3)
    res = N.denoise(small, radius=3)
    assert res.radius == 1 and res.data.shape == small.shape


def test_scan_smaller_than_three_by_three_is_refused():
    with pytest.raises(ValueError, match="too small"):
        N.denoise(make_scan(rows=2, cols=9), radius=1)


def test_non_scan_input_is_refused():
    with pytest.raises(ValueError, match="2-D scan"):
        N.denoise(np.zeros((10, 60, 60), dtype=np.uint8))


def test_filter_window_covers_what_the_kernel_reads():
    # whole-scan window of every row lies inside the loaded extent
    for n_rows in (7, 12):
        for sr in (1, 2, 3):
            if n_rows < 2 * sr + 1:
                continue
            for a in range(n_rows):
                for b in range(a + 1, n_rows + 1):
                    e0, e1 = N.filter_extent(a, b, sr, n_rows)
                    assert 0 <= e0 and e1 <= n_rows and e1 - e0 >= 2 * sr + 1
                    for j in range(a, b):
                        w0 = N.window_start(j, sr, n_rows)
                        assert e0 <= w0 and w0 + 2 * sr + 1 <= e1


def test_provenance_mark_survives_a_deepcopy(scan):
    import copy
    import kikuchipy as kp

    sig = kp.signals.EBSD(scan.copy())
    assert N.applied_info(sig) is None
    res = N.denoise(scan, radius=2)
    N.mark_applied(sig, res)
    info = N.applied_info(sig)
    assert info["radius"] == 2 and info["lam"] == pytest.approx(res.lam)
    assert N.applied_info(copy.deepcopy(sig))["radius"] == 2


# --------------------------------------------------------------------------- #
# the kernels do not bounds-check: no user-reachable shape may get them outside an array
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kind", ["random", "constant", "zeros", "saturated"])
def test_no_scan_shape_radius_or_content_reaches_a_kernel_out_of_bounds(kind):
    """A numba kernel that reads outside its array does not raise, it kills the process
    (an access violation took pytest down when a mutation moved the window by one row).
    Every combination here either raises ValueError up front or returns a result that does
    not depend on the strip height. A crash anywhere fails the whole test run, loudly."""
    import itertools

    rng = np.random.default_rng(7)
    for rows, cols in itertools.product(range(1, 8), repeat=2):
        for ph, pw in ((1, 1), (3, 5), (8, 8)):
            for radius in (-1, 0, 1, 2, 3, 9):
                shape = (rows, cols, ph, pw)
                if kind == "random":
                    d = rng.integers(0, 256, shape).astype(np.uint8)
                elif kind == "constant":
                    d = np.full(shape, 100, np.uint8)
                elif kind == "zeros":
                    d = np.zeros(shape, np.uint8)
                else:
                    d = np.full(shape, 255, np.uint8)
                try:
                    whole = N.denoise(d, radius=radius)
                except ValueError:
                    continue
                assert 1 <= whole.radius <= min(radius, (min(rows, cols) - 1) // 2)
                strips = N.denoise(d, radius=radius, strip_rows=1, lam=whole.lam)
                same = N.denoise(d, radius=radius, lam=whole.lam)
                assert np.array_equal(strips.data, same.data), shape


# --------------------------------------------------------------------------- #
# advice: lambda from a sample of blocks
# --------------------------------------------------------------------------- #

def _big_scan(sigma, rows=40, cols=40, seed=3):
    return make_scan(rows=rows, cols=cols, sigma=sigma, seed=seed)


def test_block_statistics_equal_the_numba_kernel():
    """The advice re-implements the first pass in numpy; it must be the kernel's numbers."""
    cls = N._nlpar_class()
    scan = make_scan(rows=12, cols=12, sigma=20.0, seed=5)
    b = scan.shape[0]
    mask = N._disc_mask(60, 60).ravel()
    flat = scan.reshape(b, b, -1)[..., mask].astype(np.float32)
    mx = 0.9961 * float(scan.max())
    sigma, dist = N._block_stats(flat, flat < np.float32(mx))

    full = np.zeros((b * b + 1, 60 * 60), dtype=np.float32)
    full[: b * b] = scan.reshape(b * b, -1)
    full[b * b, 0] = scan.max()
    idx = np.asarray(mask.nonzero()[0], dtype=np.int64)
    sig_k, dout, _ = cls.sigma_numba(full, 1, b, b, np.array([0, b], dtype=np.int64),
                                     np.array([0, b], dtype=np.int64), idx, True)
    assert np.allclose(sigma, sig_k[1:-1, 1:-1], rtol=1e-4)
    # kernel: 9 entries per pattern incl. the (very negative) self entry; ours: 8 + self -1
    ours = np.sort(dist[..., :8], axis=-1)
    theirs = np.sort(dout[2:-2, 2:-2], axis=-1)[..., 1:]
    assert np.allclose(ours, theirs, rtol=1e-3, atol=1e-3)


def test_sampled_lambda_is_the_lambda_a_run_finds():
    scan = _big_scan(25.0)
    est = N.estimate_noise(scan)
    full = N.denoise(scan, radius=1)
    assert est.lam == pytest.approx(full.lam, abs=0.1)
    assert est.sigma == pytest.approx(full.sigma_median, rel=0.1)
    assert 900 <= est.n_patterns <= 1100


def _drifting_scan(sigma, rows=40, cols=40, seed=3):
    """Real nickel patterns, but the orientation drifts from pattern to pattern (as in a real
    clean map: deformation, sub-grains) instead of being constant inside blocks. Neighbours then
    differ by more than a little noise explains."""
    import kikuchipy as kp

    base = np.asarray(kp.data.nickel_ebsd_small(lazy=False).data).astype(np.float32)
    clean = np.empty((rows, cols) + base.shape[2:], dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            tr, tc = (r % 3) / 3.0, (c % 3) / 3.0
            ir, ic = (r // 3) % 3, (c // 3) % 3
            jr, jc = (ir + 1) % 3, (ic + 1) % 3
            top = (1 - tc) * base[ir, ic] + tc * base[ir, jc]
            bot = (1 - tc) * base[jr, ic] + tc * base[jr, jc]
            clean[r, c] = (1 - tr) * top + tr * bot
    rng = np.random.default_rng(seed)
    out = np.clip(clean + rng.normal(0, sigma, clean.shape), 0, 255).astype(np.uint8)
    out[..., 0, 0] = 255
    return out


def test_noisy_scan_is_recommended_and_clean_scan_is_not():
    noisy = N.estimate_noise(_drifting_scan(30.0))
    clean = N.estimate_noise(_drifting_scan(1.0))
    assert N.advice_for(noisy) == "recommended", noisy
    assert N.advice_for(clean) == "not_recommended", clean
    assert clean.lam > noisy.lam
    assert N.advice_for(N.estimate_noise(_big_scan(30.0))) == "recommended"


def test_estimate_is_deterministic_and_lazy_equals_eager():
    da = pytest.importorskip("dask.array")
    scan = _big_scan(25.0)
    a = N.estimate_noise(scan)
    assert N.estimate_noise(scan).lam == a.lam
    assert N.estimate_noise(da.from_array(scan, chunks=(7, 9, 60, 60))).lam == a.lam


def test_estimate_reads_blocks_not_the_scan():
    class Spy:
        ndim = 4
        dtype = np.dtype(np.uint8)

        def __init__(self, arr):
            self.arr, self.shape, self.read = arr, arr.shape, 0

        def __getitem__(self, key):
            out = self.arr[key]
            self.read += out.shape[0] * out.shape[1]
            return out

    spy = Spy(_big_scan(25.0, rows=80, cols=80))
    N.estimate_noise(spy)
    assert spy.read <= 0.5 * 80 * 80, spy.read       # a few thousand of 6400 patterns at most


def test_estimate_refuses_what_it_cannot_judge():
    with pytest.raises(ValueError, match="too small"):
        N.estimate_noise(make_scan(rows=4, cols=30))
    with pytest.raises(ValueError, match="uniform"):
        N.estimate_noise(np.full((20, 20, 8, 8), 100, np.uint8))


# --------------------------------------------------------------------------- #
# cancel is honoured between the long steps, not only between strips
# --------------------------------------------------------------------------- #

def test_a_cancel_set_before_the_start_stops_before_the_first_compile(monkeypatch):
    ev = threading.Event()
    ev.set()
    calls = []
    monkeypatch.setattr(N, "warm_up", lambda: calls.append("warm_up") or 0.0)
    with pytest.raises(N.NlparCancelled):
        N.denoise(make_scan(), radius=1, cancel=ev)
    assert calls == [], "the numba compile (up to tens of seconds) ran although the run was cancelled"


def test_a_cancel_that_arrives_during_the_dask_maximum_stops_before_any_strip_is_read():
    class SlowDask:
        """Looks like a lazy array whose maximum takes long; the cancel arrives meanwhile."""
        chunks = ((1,), (1,))
        ndim = 4
        dtype = np.dtype(np.uint8)

        def __init__(self, arr, event):
            self.arr, self.shape, self.event, self.reads = arr, arr.shape, event, 0

        def max(self):
            outer = self

            class _Pending:
                def compute(self):
                    outer.event.set()                 # the user pressed Cancel during the compute
                    return float(outer.arr.max())
            return _Pending()

        def __getitem__(self, key):
            self.reads += 1
            return self.arr[key]

    ev = threading.Event()
    data = SlowDask(make_scan(), ev)
    with pytest.raises(N.NlparCancelled):
        N.denoise(data, radius=1, cancel=ev)
    assert data.reads == 0
