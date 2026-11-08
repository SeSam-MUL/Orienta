"""Sum fit, streamed projection and detection on a real AZtec H5OINA scan.

The Oxford reference scan: 90 x 120 px, ~180 counts/px, 2048 channels, 20 kV.
``ORIENTA_TEST_DATA`` may point at a ``Test_data`` folder that is not this
checkout's; the test skips with the missing path when the file is absent.
"""
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

from backend.eds_spectrum.cube import live_time, read_meta, sum_spectrum
from backend.eds_spectrum.fit import (detect, fit_sum_spectrum, net_areas,
                                      project_cube, quality, significance)
from backend.eds_spectrum.model import default_families
from tests.data_deps import TEST_DATA_DIR, need

SAMPLE_B = ("EBSD_SampleB_extrusion_withPattern Sample_B "
            "Arbeitsbereich 1 Elementverteilungsdaten 1.h5oina")


@pytest.fixture(scope="module")
def scan():
    data_dir = Path(os.environ.get("ORIENTA_TEST_DATA") or TEST_DATA_DIR)
    path = need(data_dir / SAMPLE_B)
    meta = read_meta(path)
    with h5py.File(path, "r") as f:
        exported = list(f["1/EDS/Data/Window Integral"].keys())
    fams = default_families(exported, meta)
    t0 = time.perf_counter()
    y = sum_spectrum(path)
    t_sum = time.perf_counter() - t0
    t0 = time.perf_counter()
    ff = fit_sum_spectrum(y, meta, fams)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    maps = project_cube(path, ff, y_sum=y)
    t_proj = time.perf_counter() - t0
    return dict(path=path, meta=meta, exported=exported, y=y, ff=ff, maps=maps,
                t_sum=t_sum, t_fit=t_fit, t_proj=t_proj)


def test_header_and_cube_shapes(scan):
    meta = scan["meta"]
    assert (meta.n_channels, meta.beam_kv) == (2048, 20.0)
    assert meta.start_kev == pytest.approx(-0.1973, abs=1e-4)
    assert scan["y"].shape == (2048,) and scan["y"].sum() > 1e6
    assert all(m.shape == (90 * 120,) for m in scan["maps"].values())


def test_chi2_and_speed(scan):
    # measured: chi2_red 5.18 (fit 1.7 s, sum read 0.3 s, projection 0.3 s); bound = 5.18 * 1.5 -> 8
    assert scan["ff"].chi2_red < 8
    assert scan["t_fit"] + scan["t_proj"] <= 10.0, (scan["t_fit"], scan["t_proj"])


def test_projection_total_matches_sum_fit_for_every_family(scan):
    """Unclipped maps: pixels add up to the sum fit, weak families included."""
    lt = live_time(scan["path"])
    areas = net_areas(scan["ff"])
    for k, m in scan["maps"].items():
        assert np.isfinite(m).all(), k
        total = float((m.astype(float) * lt).sum())
        assert total == pytest.approx(areas[k], rel=1e-3, abs=1.0), (k, total, areas[k])


def test_exported_elements_are_detected(scan):
    found = {d.element for d in detect(scan["ff"])}
    assert {"Al", "Si", "Fe", "Mn", "Cu", "Zn"} <= found


def test_detected_but_not_exported_elements_have_a_k_line_at_5_sigma(scan):
    exported = {w.split()[0] for w in scan["exported"]}
    sig = {(d.element, d.family): d for d in significance(scan["ff"])}
    extra = [d for d in detect(scan["ff"]) if d.element not in exported]
    for d in extra:
        k = sig.get((d.element, "K"))
        if k is not None:  # an element without excited K is detected through L
            assert k.z >= 5, (d.element, k.z)


def test_no_quality_warning_for_the_strong_lines(scan):
    warned = {w.family for w in quality(scan["ff"], scan["y"])}
    assert not warned & {"Al K", "Si K", "Fe K", "Mn K"}, warned


@pytest.mark.parametrize("n_px", [60, 400])
def test_pixel_sum_identity_on_a_sparse_subset_of_the_real_scan(scan, tmp_path, n_px):
    """The first ``n_px`` pixels, refit on their own sum: many empty channels."""
    sub = tmp_path / f"sub{n_px}.h5oina"
    with h5py.File(scan["path"], "r") as src, h5py.File(sub, "w") as dst:
        d = dst.create_group("1/EDS/Data")
        d.create_dataset("Spectrum", data=src["1/EDS/Data/Spectrum"][:n_px],
                         chunks=(n_px, 1), compression="lzf")
        d.create_dataset("Live Time", data=src["1/EDS/Data/Live Time"][:n_px])
        src.copy("1/EDS/Header", dst["1/EDS"])
    meta = read_meta(sub)
    y = sum_spectrum(sub)
    ff = fit_sum_spectrum(y, meta, scan["ff"].families)
    assert (y[ff.mask] < 1).any()
    lt = live_time(sub)
    areas = net_areas(ff)
    for k, m in project_cube(sub, ff).items():
        assert np.isfinite(m).all(), k
        total = float((m.astype(float) * lt).sum())
        assert total == pytest.approx(areas[k], rel=1e-3, abs=1.0), (k, total, areas[k])
