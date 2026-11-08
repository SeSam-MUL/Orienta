"""The RX panel shows the numbers of the analysis (the workbook's), and pixels without a grain are never RX.

Probe geometry: two 20 x 20 px grains side by side, a 2 x 2 hole inside the first that alpha 3.1 closes, and a
strip of not-indexed pixels on the right that belongs to no grain. The second grain turns by 1 degree per
column, so its GOS is about 5 degrees: it is substructured, the first grain is recrystallized.
"""
import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from analysis.ebsd_dataset import EBSDDataset
from analysis.grain_analysis import reconstruct_grains
from analysis.rx_analysis import analyze_recrystallization, classify_rx
from tests.synthetic_grain_map import _axis_angle, _qmul, random_quaternions, to_xmap

NY, NX, STRIP = 20, 50, 10


def _probe(with_bc=True):
    labels = np.zeros((NY, NX), int)
    labels[:, :20] = 1
    labels[:, 20:40] = 2
    labels[9:11, 9:11] = 0                                    # the closed hole
    rng = np.random.default_rng(5)
    base = random_quaternions(rng, 3)
    q = np.empty((NY, NX, 4))
    q[:] = np.array([1.0, 0.0, 0.0, 0.0])
    q[labels == 1] = base[1]
    cols = np.broadcast_to(np.arange(NX), (NY, NX))
    turn = _axis_angle(np.array([0.0, 0.0, 1.0]), np.deg2rad(1.0 * (cols - 20)))
    q2 = _qmul(turn, np.broadcast_to(base[2], (NY, NX, 4)))
    q[labels == 2] = q2[labels == 2]
    m = {"quats": q, "phase_id": np.where(labels > 0, 0, -1).astype(np.int64),
         "phase_names": np.array(["Al", "Al7FeCu2"]), "point_groups": np.array(["m-3m", "4/mmm"]), "step": 1.0}
    xmap = to_xmap(m)
    if with_bc:
        xmap.prop["bc"] = np.where(labels.ravel() > 0, 150.0, 0.0)
    ds = EBSDDataset(xmap, step_size=1.0)
    ds.grains = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=1, alpha=3.1, fill=False)
    return ds


@pytest.fixture
def client_probe():
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    saved = ar._dataset, dict(ar._grain_stats_settings)
    ar._dataset = _probe()
    ar._reset_grain_stats_settings()
    try:
        yield TestClient(app), ar
    finally:
        ar._dataset = saved[0]
        ar._grain_stats_settings.clear()
        ar._grain_stats_settings.update(saved[1])


def _analysis(ds):
    from analysis.deformation_analysis import calculate_kam
    kam = calculate_kam(ds, grains=ds.grains)
    return analyze_recrystallization(ds, ds.grains, kam, gos_threshold_deg=1.25, gbc_min_fraction=0.7,
                                     gkam_threshold_deg=0.55, min_grain_radius_um=0.4)


def _png(body):
    return np.array(Image.open(io.BytesIO(base64.b64decode(body["image"]))).convert("RGB"))


def test_the_probe_map_is_what_the_other_tests_assume():
    ds = _probe()
    assert ds.grains.n_grains == 2
    assert ds.grains.area[0] == 400.0 and list(ds.grains.grain_size) == [396, 400]
    assert (ds.grains.grain_ids[:, 40:] == 0).all() and (ds.grains.grain_ids[9:11, 9:11] == 1).all()


def test_classify_rx_gives_pixels_without_a_grain_their_own_class():
    ds = _probe()
    data = classify_rx(ds, ds.grains)
    assert (data[:, 40:] == -1).all()                         # the not-indexed strip: no grain, not RX
    assert (data[:, :20] == 0).all()                          # the first grain, hole included: recrystallized
    assert (data[:, 20:40] == 1).all()                        # the turning grain: substructured
    assert (data == 0).sum() == 400


def test_classify_rx_without_grains_is_all_no_grain():
    ds = _probe()
    ds.grains.gos = None
    assert (classify_rx(ds, ds.grains) == -1).all()


def test_the_panel_rx_fraction_equals_the_analysis(client_probe):
    client, ar = client_probe
    want = _analysis(ar._dataset)
    body = client.get("/api/analysis/map/rx").json()
    assert want.rx_fraction == pytest.approx(396 / 796 * 100)           # measured pixels of the RX grain
    assert body["rx_fraction"] == pytest.approx(want.rx_fraction)
    assert body["high_bc_fraction"] == pytest.approx(want.high_bc_fraction)
    assert body["rx_grain_count"] == len(want.rx_grain_ids) == 1
    assert body["total_grains"] == 2
    assert body["low_gkam_fraction"] == pytest.approx(want.low_gkam_fraction)
    assert body["low_kam_fraction"] == pytest.approx(want.low_kam_fraction)


RED = (215, 48, 39)
LIGHT_BLUE, YELLOW, BROWN, GREY = (166, 206, 227), (254, 224, 139), (140, 86, 40), (128, 128, 128)


def test_the_rx_map_draws_pixels_without_a_grain_neutral(client_probe):
    client, _ = client_probe
    rgb = _png(client.get("/api/analysis/map/rx").json())
    strip = rgb[:, 40:].reshape(-1, 3)
    assert (strip == strip[0]).all() and tuple(strip[0]) == GREY                    # one neutral grey
    assert tuple(rgb[0, 0]) != GREY and tuple(rgb[0, 25]) != GREY


def test_the_rx_map_is_red_for_the_grains_of_the_analysis_and_the_other_grains_keep_their_gos_band(client_probe):
    client, ar = client_probe
    want = _analysis(ar._dataset)
    assert list(want.rx_grain_ids) == [1]
    body = client.get("/api/analysis/map/rx").json()
    rgb = _png(body)
    assert (rgb[:, :20] == RED).all()                          # the RX grain, its closed hole included
    assert (rgb[:, 20:40] == YELLOW).all()                     # the turning grain: substructured, not RX
    assert body["rx_drawn"] is True and body["rx_needs_bc"] is False


def test_the_rx_colours_do_not_depend_on_the_colormap(client_probe):
    client, _ = client_probe
    reference = _png(client.get("/api/analysis/map/rx", params={"cmap": "gray"}).json())
    for cmap in ("viridis", "RdYlBu_r", "jet"):
        assert (_png(client.get("/api/analysis/map/rx", params={"cmap": cmap}).json()) == reference).all()


def test_a_grain_with_low_gos_that_the_analysis_does_not_call_rx_is_not_red(client_probe):
    client, ar = client_probe
    ds = ar._dataset
    ds.xmap.prop["bc"] = np.where(ds.grains.grain_ids.ravel() == 1, 60.0, 150.0)       # the low-GOS grain is dark
    body = client.get("/api/analysis/map/rx").json()
    rgb = _png(body)
    assert len(_analysis(ds).rx_grain_ids) == 0 and body["rx_grain_count"] == 0
    assert (rgb[:, :20] == LIGHT_BLUE).all()                   # GOS < 1.5 degrees, but gBC fails: a band, not RX
    assert not (rgb == RED).any() and body["rx_drawn"] is True


def test_without_native_band_contrast_there_is_no_rx_number_and_no_red(client_probe):
    client, ar = client_probe
    del ar._dataset.xmap.prop["bc"]
    body = client.get("/api/analysis/map/rx").json()
    assert "high_bc_fraction" not in body                      # not 0.0
    assert "low_gkam_fraction" not in body and "low_kam_fraction" not in body
    assert "rx_fraction" not in body and "rx_grain_count" not in body       # a GOS-only number is not the analysis'
    assert body["total_grains"] == 2 and body["rx_needs_bc"] is True and body["rx_drawn"] is False
    rgb = _png(body)
    assert not (rgb == RED).any()                              # GOS bands only
    assert (rgb[:, :20] == LIGHT_BLUE).all() and (rgb[:, 20:40] == YELLOW).all() and (rgb[:, 40:] == GREY).all()


def test_the_rx_map_is_red_exactly_where_the_analysis_has_its_rx_grains_on_a_real_map():
    """A real 602 x 805 scan (native band contrast): the red pixels of /map/rx are the pixels of the grains
    ``analyze_recrystallization`` returns, no more and no fewer. Needs Test_data/ (ORIENTA_TEST_DATA)."""
    import os
    from pathlib import Path
    from analysis.xmap_loader import load_xmap_file
    from tests.data_deps import TEST_DATA_DIR, first_missing
    path = Path(os.environ.get("ORIENTA_TEST_DATA", TEST_DATA_DIR)) / "ground_truth" / "G1_14.h5oina"
    if first_missing(path) is not None:
        pytest.skip(f"needs {path} - set ORIENTA_TEST_DATA to a Test_data directory that has it")
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    xmap, step, _ = load_xmap_file(str(path))
    ds = EBSDDataset(xmap, step_size=step, source=str(path))
    assert ds.has_native_bc
    ds.grains = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=3, alpha=3.1, fill=False)
    saved = ar._dataset, dict(ar._grain_stats_settings)
    ar._dataset = ds
    ar._reset_grain_stats_settings()
    try:
        body = TestClient(app).get("/api/analysis/map/rx").json()
    finally:
        ar._dataset = saved[0]
        ar._grain_stats_settings.clear()
        ar._grain_stats_settings.update(saved[1])
    want = _analysis(ds)
    assert len(want.rx_grain_ids) > 0 and body["rx_grain_count"] == len(want.rx_grain_ids)
    red = (_png(body) == RED).all(axis=-1)
    assert (red == np.isin(ds.grains.grain_ids, want.rx_grain_ids)).all()
    assert body["rx_fraction"] == pytest.approx(want.rx_fraction)
