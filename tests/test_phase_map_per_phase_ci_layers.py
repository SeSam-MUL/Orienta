"""The per-phase CI and uncertainty layers read every multi-phase result.

``per_phase_data[name]`` is ``{'euler', 'ci', 'ci_mean'}`` as the indexing writers
store it (competitive multi-phase since long, phase-map routed since the routing work) and
as an imported ``.h5`` carries it; the layer reader used to ``np.asarray`` the
whole dict (size 1) and answered 400 "size 1, expected N" for every such result,
while the CI figure read ``['ci']``. One reader, ``_per_phase_ci_grid``, serves
both now. The values below are written out by hand on a 2 x 3 grid.
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from backend.api.main import app  # noqa: E402

client = TestClient(app)

FE = np.array([[0.9, 0.8, 0.7], [0.1, 0.2, 0.3]])
AL = np.array([[0.5, 0.4, 0.9], [0.6, 0.2, 0.1]])
NAN = np.nan


def _cmap(value):
    """RGB the layer paints for a CI of ``value`` (fixed 0..1 axis, RdYlGn)."""
    import matplotlib

    return (np.array(matplotlib.colormaps["RdYlGn"](value)[:3]) * 255).astype(np.uint8)


def _fake_result(per_phase_data, mask=None):
    """A multi-phase result on a 2 x 3 grid with every pixel indexed."""
    result = MagicMock()
    result.original_shape = (2, 3)
    result.selection_mask = mask
    xmap = MagicMock()
    xmap.phase_id = np.zeros(6, dtype=np.int16)
    xmap.shape = (2, 3)
    xmap.prop = {}
    phase = MagicMock()
    phase.name = "Fe"
    xmap.phases = {0: phase}
    result.xmap = xmap
    result.confidence_scores = None
    result.metadata = {"per_phase_data": per_phase_data}
    return result


def _layer(result, kind, dataset=None):
    with patch("backend.api.routes.phase_map.get_last_indexing_result", return_value=result), \
         patch("backend.api.routes.phase_map.get_analysis_dataset", return_value=dataset):
        r = client.get("/api/phasemap/layer", params={"kind": kind})
    return r


def _rgba(response) -> np.ndarray:
    from PIL import Image

    assert response.status_code == 200, response.text
    png = base64.b64decode(response.json()["image"])
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))


def _entry(ci):
    return {"euler": None, "ci": np.asarray(ci, dtype=np.float32), "ci_mean": 0.5}


# --- a competitive multi-phase result (full grids) --------------------------------

def test_the_per_phase_ci_layer_paints_that_phases_grid():
    r = _layer(_fake_result({"Fe": _entry(FE), "Al": _entry(AL)}), "ci_Fe")
    img = _rgba(r)
    assert (img[..., 3] == 255).all()                       # every pixel has a CI
    np.testing.assert_array_equal(img[0, 0, :3], _cmap(0.9))
    np.testing.assert_array_equal(img[1, 0, :3], _cmap(0.1))
    np.testing.assert_array_equal(img[0, 2, :3], _cmap(0.7))


def test_the_uncertainty_layer_is_best_minus_second_best():
    # best - second per pixel: .9-.5  .8-.4  .9-.7 / .6-.1  .2-.2  .3-.1
    expected = np.array([[0.4, 0.4, 0.2], [0.5, 0.0, 0.2]])
    img = _rgba(_layer(_fake_result({"Fe": _entry(FE), "Al": _entry(AL)}), "uncertainty"))
    assert (img[..., 3] == 255).all()
    for (row, col), value in np.ndenumerate(expected):
        np.testing.assert_array_equal(img[row, col, :3], _cmap(value))


def test_a_phase_without_a_ci_at_a_pixel_does_not_blank_the_uncertainty_there():
    # Zn was not evaluated at pixel (0, 0): the other two phases still decide it
    zn = np.full((2, 3), 0.0)
    zn[0, 0] = NAN
    img = _rgba(_layer(_fake_result(
        {"Fe": _entry(FE), "Al": _entry(AL), "Zn": _entry(zn)}), "uncertainty"))
    assert (img[..., 3] == 255).all()
    np.testing.assert_array_equal(img[0, 0, :3], _cmap(0.4))        # .9 - .5
    np.testing.assert_array_equal(img[0, 2, :3], _cmap(0.2))        # .9 - .7


def test_an_unknown_phase_is_a_400_naming_it():
    r = _layer(_fake_result({"Fe": _entry(FE), "Al": _entry(AL)}), "ci_Zn")
    assert r.status_code == 400 and "Zn" in r.text


def test_one_phase_has_no_uncertainty():
    img = _rgba(_layer(_fake_result({"Fe": _entry(FE)}), "uncertainty"))
    assert (img[..., 3] == 0).all()


# --- one CI value per SELECTED pixel (a mask / ROI) -------------------------------

def test_a_ci_per_selected_pixel_is_placed_on_the_grid_through_the_mask():
    mask = np.array([[True, True, False], [False, True, True]])
    result = _fake_result({"Fe": _entry([0.9, 0.8, 0.2, 0.3]),
                           "Al": _entry([0.5, 0.4, 0.2, 0.1])}, mask=mask)
    img = _rgba(_layer(result, "ci_Fe"))
    assert (img[..., 3] == 255 * mask).all()                # only the selection is painted
    np.testing.assert_array_equal(img[0, 0, :3], _cmap(0.9))
    np.testing.assert_array_equal(img[0, 1, :3], _cmap(0.8))
    np.testing.assert_array_equal(img[1, 1, :3], _cmap(0.2))
    np.testing.assert_array_equal(img[1, 2, :3], _cmap(0.3))
    unc = _rgba(_layer(result, "uncertainty"))
    np.testing.assert_array_equal(unc[0, 0, :3], _cmap(0.4))       # .9 - .5
    np.testing.assert_array_equal(unc[1, 2, :3], _cmap(0.2))       # .3 - .1
    assert (unc[..., 3] == 255 * mask).all()


def test_a_ci_of_the_wrong_size_is_a_400_naming_the_sizes():
    r = _layer(_fake_result({"Fe": _entry([0.1, 0.2, 0.3, 0.4, 0.5])}), "ci_Fe")
    assert r.status_code == 400
    assert "size 5" in r.text and "expected 6" in r.text


# --- an analysis dataset ----------------------------------------------------------

def test_an_imported_dataset_with_dict_entries_serves_the_layers():
    dataset = SimpleNamespace(xmap=_fake_result({}).xmap, shape=(2, 3),
                              per_phase_data={"Fe": _entry(FE), "Al": _entry(AL)})
    img = _rgba(_layer(None, "ci_Al", dataset=dataset))
    np.testing.assert_array_equal(img[0, 2, :3], _cmap(0.9))
    np.testing.assert_array_equal(img[1, 0, :3], _cmap(0.6))
    unc = _rgba(_layer(None, "uncertainty", dataset=dataset))
    np.testing.assert_array_equal(unc[1, 0, :3], _cmap(0.5))


def test_the_bare_array_form_is_read_too():
    dataset = SimpleNamespace(xmap=_fake_result({}).xmap, shape=(2, 3),
                              per_phase_data={"Fe": FE.ravel().astype(np.float32),
                                              "Al": AL.ravel().astype(np.float32)})
    img = _rgba(_layer(None, "ci_Fe", dataset=dataset))
    np.testing.assert_array_equal(img[0, 0, :3], _cmap(0.9))
    unc = _rgba(_layer(None, "uncertainty", dataset=dataset))
    np.testing.assert_array_equal(unc[0, 0, :3], _cmap(0.4))


# --- the CI figure reads the same shapes, through the same reader -------------------

def _figure_data(result, direction):
    from backend.api.routes.phase_map import _render_ci_figure

    mask = result.selection_mask
    fig, _meta = _render_ci_figure(
        direction, result, None, 2, 3, mask, 1.0, False, "lower left",
        10, 12, "k", "w", 0.5, False, None, "viridis", 50)
    return np.asarray(fig.axes[0].images[0].get_array(), dtype=float)


def test_the_figure_shows_the_same_grids_as_the_layer():
    result = _fake_result({"Fe": _entry(FE), "Al": _entry(AL)})
    np.testing.assert_allclose(_figure_data(result, "ci_Fe"), FE, atol=1e-6)
    np.testing.assert_allclose(_figure_data(result, "uncertainty"),
                               [[0.4, 0.4, 0.2], [0.5, 0.0, 0.2]], atol=1e-6)


def test_the_figure_places_a_per_selected_pixel_ci_through_the_mask():
    mask = np.array([[True, True, False], [False, True, True]])
    result = _fake_result({"Fe": _entry([0.9, 0.8, 0.2, 0.3])}, mask=mask)
    np.testing.assert_allclose(_figure_data(result, "ci_Fe"),
                               [[0.9, 0.8, NAN], [NAN, 0.2, 0.3]], atol=1e-6)


def test_the_figures_uncertainty_places_per_selected_pixel_cis_through_the_mask():
    mask = np.array([[True, True, False], [False, True, True]])
    result = _fake_result({"Fe": _entry([0.9, 0.8, 0.2, 0.3]),
                           "Al": _entry([0.5, 0.4, 0.2, 0.1])}, mask=mask)
    np.testing.assert_allclose(_figure_data(result, "uncertainty"),
                               [[0.4, 0.4, NAN], [NAN, 0.0, 0.2]], atol=1e-6)


# --- a phase-map ROUTED result -----------------------------------------------------

def _routed(monkeypatch, tmp_path):
    from tests.test_phase_map_routing_run_handling import _routed_dictionary_result
    from tests.test_phase_map_routing_methods import _grid

    return _routed_dictionary_result(
        monkeypatch, tmp_path, ["PhaseA.cif", "PhaseB.cif"], _grid([2, 4]),
        ["PhaseA", "PhaseB"])[0]


def test_a_routed_result_serves_the_per_phase_ci_layer(monkeypatch, tmp_path):
    result = _routed(monkeypatch, tmp_path)
    name_a = str(result.xmap.phases[0].name)
    img = _rgba(_layer(result, f"ci_{name_a}"))
    # PhaseA ran on the first two pixels of the 2 x 3 grid, with CI 0.25
    assert (img[..., 3].ravel() == [255, 255, 0, 0, 0, 0]).all()
    np.testing.assert_array_equal(img[0, 0, :3], _cmap(0.25))
    name_b = str(result.xmap.phases[1].name)
    img_b = _rgba(_layer(result, f"ci_{name_b}"))
    assert (img_b[..., 3].ravel() == [0, 0, 255, 255, 255, 255]).all()
    np.testing.assert_array_equal(img_b[1, 2, :3], _cmap(0.75))


def test_a_routed_result_has_no_uncertainty_because_its_phases_never_competed(
        monkeypatch, tmp_path):
    img = _rgba(_layer(_routed(monkeypatch, tmp_path), "uncertainty"))
    assert (img[..., 3] == 0).all()            # each pixel has ONE phase's CI: no second-best
