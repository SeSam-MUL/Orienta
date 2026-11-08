"""Cube reader, streamed per-pixel projection, detection and fit quality."""
from dataclasses import replace

import h5py
import numpy as np
import pytest

from backend.eds_spectrum.cube import (has_spectrum_cube, live_time, read_meta,
                                       sum_spectrum)
from backend.eds_spectrum.fit import (Detection, QualityWarning, detect,
                                      fit_sum_spectrum, forward, net_areas,
                                      project_cube, quality, significance)
from backend.eds_spectrum.model import NonlinearParams
from tests.test_eds_spectrum_fit import AREAS, FAMS, META, truth

NX, NY = 8, 5
N_PX = NX * NY


def write_cube(path, spectra, live, meta=META, nx=NX, ny=NY, root="1"):
    """An H5OINA-shaped EDS group: chunked per channel, lzf, full header."""
    with h5py.File(path, "w") as f:
        d = f.create_group(f"{root}/EDS/Data")
        d.create_dataset("Spectrum", data=spectra.astype(np.int32),
                         chunks=(spectra.shape[0], 1), compression="lzf")
        d.create_dataset("Live Time", data=np.asarray(live, dtype=np.float32))
        h = f.create_group(f"{root}/EDS/Header")
        h["Start Channel"] = np.array([meta.start_kev * 1e3], dtype=np.float32)
        h["Channel Width"] = np.array([meta.width_kev * 1e3], dtype=np.float32)
        h["Number Channels"] = np.array([meta.n_channels], dtype=np.int32)
        h["Beam Voltage"] = np.array([meta.beam_kv], dtype=np.float32)
        h["Detector Serial Number"] = np.array([b"TEST-0001"], dtype=object)
        h["Tilt Angle"] = np.array([np.deg2rad(70.0)], dtype=np.float32)
        h["X Cells"] = np.array([nx], dtype=np.int32)
        h["Y Cells"] = np.array([ny], dtype=np.int32)


def make_cube(path, ff_truth, per_px_scale, seed=0, dose=1.0):
    """Poisson pixels around ``forward(ff_truth, areas)``, areas varying per pixel."""
    rng = np.random.default_rng(seed)
    spectra = np.zeros((N_PX, META.n_channels))
    scales = {k: np.zeros(N_PX) for k in AREAS}
    for i in range(N_PX):
        areas = {}
        for k, v in AREAS.items():
            scales[k][i] = rng.uniform(0.5, 1.5)
            areas[k] = v * per_px_scale * scales[k][i]
        spectra[i] = rng.poisson(dose * forward(ff_truth, areas))
    live = 3e-4 * rng.uniform(0.5, 1.5, N_PX)
    write_cube(path, spectra, live)
    return scales


@pytest.fixture(scope="module")
def ff_truth():
    return truth(META, FAMS, AREAS, NonlinearParams(0.0, 1.0, 0.13, 3e-4))


@pytest.fixture(scope="module")
def cube(tmp_path_factory, ff_truth):
    path = tmp_path_factory.mktemp("cube") / "cube.h5oina"
    scales = make_cube(path, ff_truth, per_px_scale=0.25)
    ff = fit_sum_spectrum(sum_spectrum(path), read_meta(path), FAMS)
    return path, ff, scales


def test_reader_meta_and_shapes(cube):
    path, _, _ = cube
    assert has_spectrum_cube(path)
    meta = read_meta(path)
    assert meta.start_kev == pytest.approx(META.start_kev, abs=1e-6)
    assert meta.width_kev == pytest.approx(0.01, abs=1e-8)
    assert (meta.n_channels, meta.beam_kv, meta.detector) == (2048, 20.0, "TEST-0001")
    assert meta.tilt_deg == pytest.approx(70.0, abs=1e-4)
    assert sum_spectrum(path).shape == (2048,)
    assert live_time(path).shape == (N_PX,)


def test_sum_spectrum_equals_direct_sum_and_ignores_block(cube):
    path, _, _ = cube
    with h5py.File(path, "r") as f:
        direct = f["1/EDS/Data/Spectrum"][()].sum(axis=0).astype(float)
    assert np.array_equal(sum_spectrum(path), direct)
    assert np.array_equal(sum_spectrum(path, block=7), direct)


def test_has_spectrum_cube_false_without_cube(tmp_path):
    p = tmp_path / "plain.h5"
    with h5py.File(p, "w") as f:
        f.create_group("1/EBSD")
    assert not has_spectrum_cube(p)


def test_channel_count_mismatch_is_loud(tmp_path):
    p = tmp_path / "bad.h5"
    write_cube(p, np.ones((N_PX, 100)), np.ones(N_PX))  # header says 2048
    with pytest.raises(ValueError, match="channels"):
        read_meta(p)


def test_projection_sums_to_sum_fit(cube):
    f, ff, _ = cube
    maps = project_cube(f, ff)
    lt = live_time(f)
    total = {k: float((m.astype(float) * lt).sum()) for k, m in maps.items()}
    checked = 0
    for k, v in net_areas(ff).items():
        if v > 1e3:
            assert total[k] == pytest.approx(v, rel=1e-3), k
            checked += 1
    assert checked >= 6


def test_block_size_does_not_change_result(cube):
    f, ff, _ = cube
    a = project_cube(f, ff, block=7)
    b = project_cube(f, ff, block=64)
    assert a.keys() == b.keys()
    assert all(np.allclose(a[k], b[k], rtol=1e-5, atol=1e-3) for k in a)


def test_maps_are_finite_float32_one_per_family_and_sum_to_the_sum_fit(cube):
    """Unclipped maps: the live-time weighted sum over ALL pixels equals the
    sum-fit area of EVERY family, weak ones included."""
    f, ff, _ = cube
    maps = project_cube(f, ff)
    lt = live_time(f)
    areas = net_areas(ff)
    assert set(maps) == set(areas)
    for k, m in maps.items():
        assert m.dtype == np.float32 and m.shape == (N_PX,)
        assert np.isfinite(m).all(), k
        assert float((m.astype(float) * lt).sum()) == pytest.approx(
            areas[k], rel=1e-3, abs=1.0), k
    assert any((m < 0).any() for m in maps.values())  # weak families do go negative


def test_projection_follows_the_pixels_not_just_the_total(cube):
    """Per-pixel counts-per-live-second track the per-pixel truth."""
    f, ff, scales = cube
    maps = project_cube(f, ff)
    for k in ("Al K", "Fe K", "Cu K"):
        got = maps[k].astype(float) * live_time(f)  # counts in the main line
        assert np.corrcoef(got, scales[k])[0, 1] > 0.95, k
        assert got.mean() == pytest.approx(AREAS[k] * 0.25, rel=0.1), k


def test_zero_live_time_pixel_gives_zero_not_nan(cube, tmp_path):
    f, ff, _ = cube
    with h5py.File(f, "r") as src:
        spectra = src["1/EDS/Data/Spectrum"][()]
        lt = src["1/EDS/Data/Live Time"][()]
    lt[3] = 0.0
    p = tmp_path / "zero.h5"
    write_cube(p, spectra, lt)
    for m in project_cube(p, ff).values():
        assert np.isfinite(m).all() and m[3] == 0.0


def test_progress_reaches_one(cube):
    f, ff, _ = cube
    seen = []
    project_cube(f, ff, block=100, progress=seen.append)
    assert seen and seen[-1] == pytest.approx(1.0) and seen == sorted(seen)


# --------------------------------------------------------------------------- #
# detection and quality
# --------------------------------------------------------------------------- #
LTI_FAMS = FAMS + [("Ti", "K"), ("Ti", "L"), ("Zr", "L")]


@pytest.fixture(scope="module")
def ff_misfit():
    """Spectrum with Ti L (0.45 keV) and Zr L but no Ti K: the soft L feature is
    fitted with a large z and must still not be reported as titanium."""
    areas = dict(AREAS, **{"Ti L": 3e5, "Zr L": 2e5})
    t = truth(META, LTI_FAMS, areas, NonlinearParams(0.0, 1.0, 0.13, 3e-4))
    y = np.random.default_rng(3).poisson(forward(t, areas)).astype(float)
    return fit_sum_spectrum(y, META, LTI_FAMS), y


def test_detection_rule_ignores_l_only_below_1kev(ff_misfit):
    ff, _ = ff_misfit
    assert net_areas(ff)["Ti L"] > 1e5  # the bait is really there
    assert not any(d.element == "Ti" for d in detect(ff))


def test_detection_counts_l_only_for_elements_without_excited_k(ff_misfit):
    ff, _ = ff_misfit
    zr = [d for d in detect(ff) if d.element == "Zr"]
    assert len(zr) == 1 and zr[0].family == "L" and zr[0].z >= 5


def test_detection_finds_the_real_elements_and_nothing_absent(cube):
    _, ff, _ = cube
    found = {d.element for d in detect(ff)}
    assert {"Al", "Si", "Fe", "Mn", "Cu"} <= found
    d = next(d for d in detect(ff) if d.element == "Fe" and d.family == "K")
    assert isinstance(d, Detection)
    assert d.area == pytest.approx(net_areas(ff)["Fe K"])
    assert d.z == pytest.approx(d.area / d.sigma) and d.z >= 5


def test_sigma_is_larger_than_the_counting_floor(cube):
    """sigma >= sqrt(area): the covariance cannot beat Poisson on the line itself."""
    _, ff, _ = cube
    for d in detect(ff):
        if d.family == "K":
            assert d.sigma >= 0.9 * np.sqrt(d.area), d


def test_quality_silent_on_a_well_fitted_spectrum(ff_misfit):
    ff, y = ff_misfit
    assert quality(ff, y) == []


def test_quality_flags_an_unmodelled_peak(ff_misfit):
    ff, y = ff_misfit
    bad = y.copy()
    E = ff.meta.energies()
    bad[np.abs(E - 6.404) < 0.1] += 3000  # a Fe K-sized excess on top of the Fe K line
    warns = quality(ff, bad)
    w = next(w for w in warns if w.family == "Fe K")
    assert isinstance(w, QualityWarning) and w.residual_fraction > 0.05


def test_quality_ignores_tiny_families(ff_misfit):
    ff, y = ff_misfit
    bad = y.copy()
    bad[np.abs(ff.meta.energies() - 1.012) < 0.05] += 2000  # >5 % of Zn L, but Zn L < 1 % of Al K
    labels = [w.family for w in quality(ff, bad)]
    assert "Zn L" not in labels


def test_l_family_of_an_element_with_excited_k_is_not_evidence(cube):
    """Zn L (1.01 keV, z in the hundreds) is ignored because Zn K is excited at
    20 kV and fitted at ~0: the K line is the arbiter."""
    _, ff, _ = cube
    zn_l = next(d for d in significance(ff) if (d.element, d.family) == ("Zn", "L"))
    assert zn_l.z > 100
    assert not any(d.element == "Zn" for d in detect(ff))


def test_sigma_grows_with_the_misfit(cube):
    _, ff, _ = cube
    worse = replace(ff, chi2_red=4.0 * max(ff.chi2_red, 1.0))
    fe = lambda f: next(d for d in significance(f) if (d.element, d.family) == ("Fe", "K"))
    assert fe(worse).sigma == pytest.approx(2.0 * fe(ff).sigma, rel=1e-6)


def test_pixel_sum_identity_holds_on_a_sparse_sum(tmp_path, ff_truth):
    """Few counts: many channels of the sum are 0 or 1.  The projection weights
    must be the sum fit's own (1/max(sum, 1)), or the pixel sum drifts."""
    path = tmp_path / "sparse.h5oina"
    make_cube(path, ff_truth, per_px_scale=0.25, dose=3e-3)
    y = sum_spectrum(path)
    ff = fit_sum_spectrum(y, read_meta(path), FAMS)
    assert (y[ff.mask] < 1).sum() > 100  # the case that broke the old weights
    lt = live_time(path)
    areas = net_areas(ff)
    assert sum(a > 100 for a in areas.values()) >= 4
    for k, m in project_cube(path, ff).items():
        assert float((m.astype(float) * lt).sum()) == pytest.approx(
            areas[k], rel=1e-3, abs=1.0), k
