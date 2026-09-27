"""An add-on's map, painted by the server, as a layer the page can stack.

The page already knows how to draw one of these: ``fetchLayerImage`` consumes
``{image, scale}`` from ``phaseMapApi.layer``. So this endpoint answers in that
shape rather than a new one, and these tests assert the SHAPE as much as the
picture -- a layer that returned the right pixels under different key names
would be just as unusable.

Two things are easy to get wrong here and are therefore tested directly:

* **the store has to keep more than the numbers.** ``_remember_map`` kept the
  ndarray alone, so ``unit``, ``vmin`` and ``vmax`` -- everything a legend is
  -- were discarded the moment ``_serialise`` had written the JSON. A painter
  written over that store cannot report a scale, however carefully it paints;
* **non-finite pixels must not drag the normalisation.** ``_diag_rgba``, the
  painter this one follows, masks with ``~np.isnan``. Copied unchanged, one
  ``+inf`` pixel flattens the whole layer to a single colour -- and a test
  with only a NaN in it would never notice.
"""
import base64
import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.api.main import app
from backend.api.routes import addons as addons_route
from backend.api.services.addons.outputs import MapOutput

FIXTURES = Path(__file__).parent / "fixtures"
client = TestClient(app)

#: The real fixture add-on, because both output routes check that the add-on
#: is installed and enabled before they serve a byte. Under an invented name
#: every test here would 404 for a reason that has nothing to do with
#: painting. The maps are placed in the store directly: what is under test is
#: the painter, and a map with a chosen unit, chosen bounds and an infinity in
#: it is not something the fixture add-on can be asked to produce.
NAME, RESULT_ID, KEY_PREFIX = "runnable", "res-paint", "addon.mean_quality"


@pytest.fixture(autouse=True)
def _installed_and_enabled(monkeypatch):
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(FIXTURES / "runnable_addon"))
    client.post(f"/api/addons/{NAME}/enabled", json={"enabled": True})
    yield


@pytest.fixture(autouse=True)
def _clear_map_store():
    addons_route._MAP_VALUES.clear()
    yield
    addons_route._MAP_VALUES.clear()


def _store(values, key="m", unit=None, vmin=None, vmax=None):
    """Put one map in the store the way a run does, and return its URL."""
    addons_route._remember_map(
        NAME, RESULT_ID, KEY_PREFIX,
        MapOutput(key=key, label="Painted", values=np.asarray(values),
                  unit=unit, vmin=vmin, vmax=vmax))
    return f"/api/addons/{NAME}/outputs/{RESULT_ID}/{KEY_PREFIX}/{key}"


def _image(url):
    r = client.get(url + "/image")
    assert r.status_code == 200, r.text
    body = r.json()
    img = Image.open(io.BytesIO(base64.b64decode(body["image"])))
    return body, np.asarray(img.convert("RGBA"))


def test_a_stored_map_is_painted_at_its_own_pixel_size():
    body, rgba = _image(_store(np.arange(12, dtype=float).reshape(3, 4)))
    assert rgba.shape == (3, 4, 4)
    assert body["shape"] == [3, 4]


def _hex_to_rgb(text):
    return tuple(int(text[i:i + 2], 16) for i in (1, 3, 5))


def test_the_pixels_are_the_colours_the_legend_promises_where_it_promises_them():
    """One assertion, three defects -- all of them invisible to the rest.

    A reviewer replaced the painter with broken variants and re-ran this file:
    painting in magma while the legend said viridis passed, mirroring the image
    left-right passed, and emitting BGRA instead of RGBA passed. Nine green
    tests, and the only way to find out was to look at the page.

    Reading the ACTUAL extreme pixels against the legend's own end stops closes
    all three: a different colormap moves both, a mirror moves which pixel is
    which, and a channel swap turns viridis' dark end (68, 1, 84) into
    (84, 1, 68). It also makes the claim in CHANGELOG-ADDON-API.md -- that the
    legend cannot disagree with the image -- something the suite checks rather
    than something the file asserts about itself.

    Tolerance of 1, and it is not slack: ``_scale_info`` ROUNDS its stops while
    the painter TRUNCATES, so the bright end is (253, 231, 36) against
    ``#fde725`` = (253, 231, 37). One count, invisible on screen, and an exact
    comparison would fail on correct code.
    """
    values = np.arange(6, dtype=float).reshape(2, 3)     # min top-left, max bottom-right
    body, rgba = _image(_store(values))
    lowest, highest = _hex_to_rgb(body["scale"]["stops"][0]), \
        _hex_to_rgb(body["scale"]["stops"][-1])
    assert np.all(np.abs(rgba[0, 0, :3].astype(int) - lowest) <= 1), \
        f"the smallest value must paint the legend's first stop: {rgba[0, 0]}"
    assert np.all(np.abs(rgba[1, 2, :3].astype(int) - highest) <= 1), \
        f"the largest value must paint the legend's last stop: {rgba[1, 2]}"


def test_the_scale_reports_the_unit_and_the_bounds_actually_painted():
    """The test that pins the store change.

    ``unit``, ``vmin`` and ``vmax`` live on the MapOutput and were thrown away
    after _serialise. Drop them from the stored record again and this is the
    test that reddens.
    """
    body, _ = _image(_store(np.arange(12, dtype=float).reshape(3, 4),
                            unit="counts", vmin=-5.0, vmax=100.0))
    assert body["scale"]["unit"] == "counts"
    assert body["scale"]["min"] == -5.0
    assert body["scale"]["max"] == 100.0
    assert body["scale"]["stops"], "a legend without colours is not a legend"


def test_declared_bounds_are_what_the_pixels_are_normalised_against():
    """Not just reported -- used.

    A painter that reported the declared bounds and normalised over the data's
    own range would pass the test above while drawing a different picture, so
    this one reads the pixels: with vmax far above the data, everything must
    land in the bottom of the ramp.
    """
    values = np.linspace(0.0, 1.0, 12).reshape(3, 4)
    _, painted_own = _image(_store(values, key="own"))
    _, painted_wide = _image(_store(values, key="wide", vmin=0.0, vmax=1000.0))
    assert painted_own[..., :3].max() > painted_wide[..., :3].max()


def test_bounds_left_open_normalise_over_the_maps_own_finite_range():
    values = np.array([[10.0, 20.0], [30.0, 40.0]])
    body, _ = _image(_store(values))
    assert body["scale"]["min"] == 10.0
    assert body["scale"]["max"] == 40.0


def test_a_map_that_is_all_nan_is_fully_transparent_and_not_an_error():
    """The output contract permits one, so the painter may not 500 on it."""
    body, rgba = _image(_store(np.full((2, 3), np.nan)))
    assert rgba[..., 3].max() == 0
    assert body["scale"] is not None, "a legend is still owed, even if empty"


def test_an_infinity_is_transparent_and_does_not_flatten_the_layer():
    """The finding this test exists for.

    Masking with ``~np.isnan`` -- what _diag_rgba does -- leaves the +inf in
    the normalisation, vmax becomes inf, and every finite pixel collapses to
    the bottom colour. A test carrying only a NaN would pass over that.
    """
    # Laid out so the transparent pixels are NOT left-right symmetric: with
    # both of them on row 0 this test also passed against a mirrored image,
    # measured. One infinity, one NaN, two finite values, no two alike.
    values = np.array([[np.nan, 1.0], [np.inf, 9.0]])
    body, rgba = _image(_store(values))
    assert body["scale"]["min"] == 1.0
    assert body["scale"]["max"] == 9.0
    assert rgba[0, 0, 3] == 0 and rgba[1, 0, 3] == 0
    assert rgba[0, 1, 3] == 255 and rgba[1, 1, 3] == 255
    assert tuple(rgba[0, 1, :3]) != tuple(rgba[1, 1, :3]), \
        "1 and 9 must not paint the same colour"


def test_a_map_without_a_unit_has_an_empty_unit_not_the_word_none():
    body, _ = _image(_store(np.arange(4.0).reshape(2, 2), unit=None))
    assert body["scale"]["unit"] == ""


def test_an_evicted_map_is_404_with_a_reason_not_a_blank_image():
    url = _store(np.arange(4.0).reshape(2, 2))
    addons_route._MAP_VALUES.clear()
    r = client.get(url + "/image")
    assert r.status_code == 404
    assert r.json()["reason"] == addons_route.REASON_MAP_NOT_STORED


def test_equal_bounds_are_legal_because_one_label_is_a_real_answer():
    """The refusal below must not catch this, and once it did.

    A label map declares vmin=0, vmax=n-1 -- the pattern the add-on changelog
    holds up as canonical -- so a single label means vmin == vmax == 0. An
    earlier version of the guard refused that, the shipped reference add-on
    produces it at ``n_components=1`` (a value its own manifest declares
    legal), and three such runs hit CRASH_LIMIT and switched the add-on off
    with a message blaming its author. Measured end to end.
    """
    from backend.api.services.addons.outputs import validate_outputs

    out = validate_outputs([MapOutput(key="component_map", label="Components",
                                      values=np.zeros((2, 2)),
                                      vmin=0.0, vmax=0.0)], shape=(2, 2))
    assert out[0].vmin == 0.0 and out[0].vmax == 0.0

    body, rgba = _image(_store(np.zeros((2, 2)), key="single",
                               vmin=0.0, vmax=0.0))
    assert rgba[..., 3].min() == 255, "one label is still a drawable map"
    assert body["scale"]["min"] == 0.0


def test_the_legend_names_the_colormap_the_changelog_promises():
    """API 0 promises viridis in writing; nothing pinned it.

    A colormap swapped for another -- including a reversed one, which the
    extremes test cannot see because image and legend reverse together --
    would silently break that promise.
    """
    body, _ = _image(_store(np.arange(4.0).reshape(2, 2)))
    assert body["scale"]["cmap"] == "viridis"


def test_an_inverted_declared_range_is_refused_at_the_source():
    """Refused where the numbers can be named, not repaired where they cannot.

    Each bound was checked alone, so ``vmin=1, vmax=0`` reached the painter,
    which found vmax <= vmin, widened it by 1e-6, and produced ONE colour over
    the whole map with a legend reading "1.0 to 1.000001" -- measured. That
    reads as a uniform field, which is a claim about the sample, from an add-on
    that merely wrote its two numbers in the wrong order.
    """
    from backend.api.services.addons.outputs import OutputError, validate_outputs

    with pytest.raises(OutputError) as excinfo:
        validate_outputs([MapOutput(key="m", label="Inverted",
                                    values=np.zeros((2, 2)),
                                    vmin=1.0, vmax=0.0)], shape=(2, 2))
    assert "vmin" in str(excinfo.value) and "vmax" in str(excinfo.value)


def test_a_single_bound_the_data_does_not_straddle_is_a_known_limit():
    """Pinned as behaviour, and named as a limit rather than left implied.

    Refusing this would be wrong: declaring only vmin is how an add-on fixes
    a scale so two scans can be compared, and a scan whose data all sit below
    it is a real measurement, not a mistake. But the legend then reads
    "10.0 to 10.000001", which says nothing useful about a map running 0..5 --
    so the earlier claim that the widening now only covers degenerate DATA was
    wrong, and this is the case it missed. Honouring the declaration is the
    lesser evil; inventing the other end would replace the add-on's statement
    with a guess.
    """
    body, _ = _image(_store(np.linspace(0.0, 5.0, 6).reshape(2, 3),
                            key="halfbound", vmin=10.0))
    assert body["scale"]["min"] == 10.0
    assert body["scale"]["max"] > 10.0


def test_the_raw_bytes_route_still_answers_exactly_as_before():
    """The store changed shape underneath it; the bytes may not.

    Asserted against the array that was stored, so a store that started
    handing out its record -- or a copy in another order -- is caught here
    rather than in the browser.
    """
    values = np.arange(12, dtype=np.float64).reshape(3, 4) * 1.5
    url = _store(values)
    r = client.get(url)
    assert r.status_code == 200, r.text
    assert r.content == np.ascontiguousarray(values).tobytes()
