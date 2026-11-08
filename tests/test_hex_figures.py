"""The /render and /export figures of a hexagonal result are drawn on the display raster."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.routes import phase_map  # noqa: E402
from backend.api.services.scan_geometry import ScanGeometry  # noqa: E402
from tests import hex_raster_oracle as oracle  # noqa: E402
from tests.test_phase_map_layer_endpoint import _make_fake_result  # noqa: E402

N_ROWS, N_COLS, STEP = 6, 5, 0.5
G_DICT = oracle.geometry_dict(N_ROWS, N_COLS, STEP, 0.5)
SPEC = oracle.spec(G_DICT, 4)
OWNER = oracle.brute_cell_index(G_DICT, SPEC)
INSIDE = OWNER >= 0

DEFAULTS = dict(
    direction="phase", show_legend=False, title="", step_x=STEP, step_y=STEP, scalebar_enabled=True,
    scalebar_location="lower right", scalebar_length=0.25, scalebar_font_size=10,
    scalebar_bar_color="#ffffff", scalebar_box_color="#000000", scalebar_box_alpha=0.6,
    confidence_overlay=False, confidence_alpha=0.5, confidence_cmap="RdYlGn", ci_threshold=0.0,
    uncertainty_threshold=0.0, min_cluster_size=0, fill_unindexed=False, modal_filter_size=0,
    color_overrides=None, show_ipf_keys=False, bc_modulate=False, dpi=50,
)


def _result(stamp):
    fake = _make_fake_result(n_rows=N_ROWS, n_cols=N_COLS)
    fake.metadata = {"scan_geometry": stamp} if stamp is not None else {}
    scores = np.linspace(0.0, 1.0, N_ROWS * N_COLS, dtype=np.float32)
    fake.confidence_scores = scores
    fake.xmap.scores = scores                      # the confidence overlay reads xmap.scores
    return fake


def _figure(monkeypatch, result, dataset=None, **over):
    import matplotlib.pyplot as plt
    monkeypatch.setattr("backend.api.routes.indexing.get_last_indexing_result", lambda: result)
    monkeypatch.setattr("backend.api.routes.analysis.get_analysis_dataset", lambda: dataset)
    fig, meta = phase_map._render_to_figure(**{**DEFAULTS, **over})
    images = [np.ma.filled(np.ma.asarray(im.get_array()), np.nan) for im in fig.axes[0].images]
    plt.close(fig)
    return images, meta


HEX = ScanGeometry.hex(N_ROWS, N_COLS, STEP).to_dict()


def test_the_phase_map_is_drawn_on_the_raster_with_the_scale_of_one_display_pixel(monkeypatch):
    (img,), meta = _figure(monkeypatch, _result(HEX))
    (legacy,), legacy_meta = _figure(monkeypatch, _result(None))                # (6, 5, 3), as before
    assert legacy.shape == (N_ROWS, N_COLS, 3) and "display" not in legacy_meta
    assert img.shape == (SPEC["height"], SPEC["width"], 4)
    np.testing.assert_array_equal(img[..., 3] > 0, INSIDE)
    np.testing.assert_allclose(img[..., :3][INSIDE], legacy.reshape(-1, 3)[OWNER[INSIDE]])
    assert meta["step_size"] == SPEC["step_um"] and meta["shape"] == [N_ROWS, N_COLS]
    assert meta["display"]["kind"] == "hex_raster" and meta["display"]["factor"] == 4


def test_the_confidence_overlay_is_drawn_on_the_raster_too(monkeypatch):
    images, _ = _figure(monkeypatch, _result(HEX), confidence_overlay=True)
    assert [im.shape[:2] for im in images] == [(SPEC["height"], SPEC["width"])] * 2
    assert int(np.isnan(images[1]).sum()) == int((~INSIDE).sum())


def test_the_ci_map_has_grey_underneath_and_nothing_in_the_corners(monkeypatch):
    images, meta = _figure(monkeypatch, _result(HEX), direction="ci")
    under, shown = images
    assert under.shape == (SPEC["height"], SPEC["width"], 4) and shown.shape == (SPEC["height"], SPEC["width"])
    np.testing.assert_array_equal(under[..., 3] == 1.0, INSIDE)
    np.testing.assert_allclose(under[..., :3][INSIDE][0], phase_map.UNINDEXED_COLOR, atol=1e-6)
    scores = np.linspace(0.0, 1.0, N_ROWS * N_COLS, dtype=np.float32)
    np.testing.assert_allclose(shown[INSIDE], scores[OWNER[INSIDE]])
    assert int(np.isnan(shown).sum()) == int((~INSIDE).sum())
    assert meta["step_size"] == SPEC["step_um"] and meta["display"]["factor"] == 4


@pytest.mark.parametrize("direction", ["ci", "bc"])
def test_a_corner_no_hexagon_covers_is_transparent_even_though_the_square_map_paints_nan_grey(
        monkeypatch, direction):
    # the square CI / BC maps paint NaN in the unindexed grey (cmap.set_bad); on the raster NaN also stands for
    # a corner, which must stay empty, so the colormap of the drawn image has to give NaN alpha 0
    import matplotlib.pyplot as plt
    monkeypatch.setattr("backend.api.routes.indexing.get_last_indexing_result", lambda: _result(HEX))
    monkeypatch.setattr("backend.api.routes.analysis.get_analysis_dataset", lambda: None)
    fig, _ = phase_map._render_to_figure(**{**DEFAULTS, "direction": direction})
    bad = fig.axes[0].images[-1].get_cmap()(np.nan)
    plt.close(fig)
    assert bad[3] == 0.0


def test_the_band_contrast_map_is_drawn_on_the_raster(monkeypatch):
    images, meta = _figure(monkeypatch, _result(HEX), direction="bc")
    bc = np.tile(np.linspace(0, 255, N_COLS, dtype=np.float32), (N_ROWS, 1)).ravel()    # the fake's prop['bc']
    np.testing.assert_allclose(images[1][INSIDE], bc[OWNER[INSIDE]])
    assert meta["display"]["shape"] == [SPEC["height"], SPEC["width"]]


def test_a_hexagonal_analysis_dataset_is_drawn_on_the_raster_when_there_is_no_result(monkeypatch):
    # the route draws the dataset's crystal map when there is no result: its stamp decides, not the result's
    fake = _result(None)
    dataset = SimpleNamespace(xmap=fake.xmap, shape=(N_ROWS, N_COLS), scan_geometry=HEX, per_phase_data={})
    (img,), meta = _figure(monkeypatch, None, dataset)
    (legacy,), _ = _figure(monkeypatch, None, SimpleNamespace(**{**vars(dataset), "scan_geometry": None}))
    assert legacy.shape == (N_ROWS, N_COLS, 3)
    assert img.shape == (SPEC["height"], SPEC["width"], 4)
    np.testing.assert_allclose(img[..., :3][INSIDE], legacy.reshape(-1, 3)[OWNER[INSIDE]])
    assert meta["step_size"] == SPEC["step_um"] and meta["display"]["factor"] == 4


def test_a_square_result_next_to_a_hexagonal_dataset_is_drawn_as_before(monkeypatch):
    # the result has a crystal map, so the figure is the result's; the dataset's stamp must not reach it
    fake = _result(ScanGeometry.square(N_ROWS, N_COLS, STEP).to_dict())
    dataset = SimpleNamespace(xmap=fake.xmap, shape=(N_ROWS, N_COLS), scan_geometry=HEX, per_phase_data={})
    (img,), meta = _figure(monkeypatch, fake, dataset)
    assert img.shape == (N_ROWS, N_COLS, 3) and "display" not in meta


@pytest.mark.parametrize("stamp", [ScanGeometry.square(N_ROWS, N_COLS, STEP).to_dict(), None],
                         ids=["square", "unknown"])
@pytest.mark.parametrize("direction", ["phase", "ci", "bc"])
def test_a_square_or_unknown_result_is_drawn_as_before(monkeypatch, direction, stamp):
    images, meta = _figure(monkeypatch, _result(stamp), direction=direction)
    assert len(images) == 1 and images[0].shape[:2] == (N_ROWS, N_COLS)
    assert "display" not in meta and meta["step_size"] == STEP


def _arm_routes(monkeypatch, result):
    monkeypatch.setattr("backend.api.routes.indexing.get_last_indexing_result", lambda: result)
    monkeypatch.setattr("backend.api.routes.analysis.get_analysis_dataset", lambda: None)
    monkeypatch.setattr("backend.api.routes.phase_map.get_last_indexing_result", lambda: result)
    monkeypatch.setattr("backend.api.routes.phase_map.get_analysis_dataset", lambda: None)


def test_the_render_and_export_routes_report_the_raster_they_drew(monkeypatch, tmp_path):
    import base64
    import io

    from fastapi.testclient import TestClient
    from PIL import Image

    from backend.api.main import app

    client = TestClient(app, raise_server_exceptions=False)
    _arm_routes(monkeypatch, _result(HEX))
    params = {"direction": "phase", "step_x": STEP, "step_y": STEP, "show_legend": "false"}

    resp = client.get("/api/phasemap/render", params=params)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["shape"] == [N_ROWS, N_COLS]
    assert body["display"]["shape"] == [SPEC["height"], SPEC["width"]] and body["display"]["factor"] == 4
    assert body["step_size"] == SPEC["step_um"]
    assert Image.open(io.BytesIO(base64.b64decode(body["image"]))).size[0] > 0

    out = tmp_path / "hex_figure.png"
    resp = client.post("/api/phasemap/export", json={**params, "output_path": str(out), "format": "png", "dpi": 50})
    assert resp.status_code == 200, resp.text
    assert resp.json()["display"]["factor"] == 4 and resp.json()["step_size"] == SPEC["step_um"]
    assert out.stat().st_size > 0
