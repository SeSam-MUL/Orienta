"""The stored calibration pattern is served at its own pixel size.

The Kikuchi lines of the PC page are in detector pixels (x along the columns, y
along the rows of the pattern) and the page draws them on the pattern image
with no scaling. So the image must have exactly the pixel size of the pattern
the lines were simulated for; a re-rendered, enlarged picture leaves the lines
in its top-left corner.
"""
import base64
import io

import numpy as np
import pytest

PAT_SHAPE = (60, 60)


@pytest.fixture
def client(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import backend.api.routes.pcrefinement as pcr
    monkeypatch.setattr(pcr, "_sessions", {})
    app = FastAPI()
    app.include_router(pcr.router, prefix="/api/pc")
    return TestClient(app), pcr


def _png_size(b64: str):
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(b64))).size   # (width, height)


@pytest.mark.parametrize("shape", [(60, 60), (128, 156), (96, 80)])
def test_stored_pattern_image_has_the_pixel_size_of_the_pattern(client, shape):
    c, pcr = client
    ctrl = pcr._get_controller()
    rng = np.random.default_rng(0)
    pat = rng.integers(0, 255, size=shape).astype(np.uint8)
    ctrl.add_pattern((3, 4), pat)

    body = c.get("/api/pc/pattern/0/image").json()

    assert body["success"] is True
    assert _png_size(body["image"]) == (shape[1], shape[0])
    assert (body["row"], body["col"]) == (3, 4)
    assert body["shape"] == list(shape)


def test_float_pattern_image_is_also_at_pattern_size(client):
    c, pcr = client
    ctrl = pcr._get_controller()
    pat = np.random.default_rng(1).random((64, 80)).astype(np.float32)
    ctrl.add_pattern((0, 0), pat)
    body = c.get("/api/pc/pattern/0/image").json()
    assert _png_size(body["image"]) == (80, 64)


def test_a_low_contrast_pattern_is_stretched_to_the_full_grey_range(client):
    from PIL import Image
    c, pcr = client
    ctrl = pcr._get_controller()
    pat = np.random.default_rng(2).integers(100, 140, size=(40, 50)).astype(np.uint8)
    ctrl.add_pattern((0, 0), pat)
    body = c.get("/api/pc/pattern/0/image").json()
    arr = np.asarray(Image.open(io.BytesIO(base64.b64decode(body["image"]))))
    assert arr.shape == (40, 50)
    assert arr.min() == 0 and arr.max() == 255
