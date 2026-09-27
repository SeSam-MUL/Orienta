"""The report pictures of an EDS export: magnified, with a scale bar.

The M5 tester opened ``phase_map.png`` from an export and found 21 x 22 px
without a scale — right for a raster, useless in a report. The 1:1 file stays
byte-identical to the renderer; a ``_figure.png`` sits beside it.
"""
from __future__ import annotations

import io

import numpy as np
import pytest

from backend.api.services import eds_export as X


def _png(w: int, h: int) -> tuple[bytes, np.ndarray]:
    from PIL import Image
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue(), rgb


def _decode(blob: bytes) -> np.ndarray:
    from PIL import Image
    with Image.open(io.BytesIO(blob)) as im:
        return np.asarray(im.convert("RGBA"))


@pytest.mark.parametrize("long_side, factor", [
    (21, 16),    # the tester's crop: 16x is the cap, 336 px
    (99, 16),    # 8x would give 792 < 800
    (100, 8),    # exactly 800
    (120, 8),    # an EBSD overview -> 960
    (199, 8),
    (200, 4),    # no longer "small": the old rule, 800 px
    (1000, 4),   # 4000 px, the ceiling
    (1001, 2),
    (3000, 1),
    (5000, 1),   # never shrunk
])
def test_figure_factor_matches_the_dialog_rule(long_side, factor):
    assert X._figure_factor(long_side) == factor


def test_png_size_reads_the_header():
    blob, _ = _png(22, 21)
    assert X._png_size(blob) == (22, 21)
    assert X._png_pixel_count(blob) == 462
    assert X._png_size(b"not a png") == (None, None)


def test_figure_without_step_is_a_pure_nearest_upscale():
    blob, rgb = _png(22, 21)
    out, info = X._figure_png(blob, None)
    img = _decode(out)
    assert img.shape == (21 * 16, 22 * 16, 4)
    expected = np.repeat(np.repeat(rgb, 16, axis=0), 16, axis=1)
    assert np.array_equal(img[..., :3], expected)
    assert np.all(img[..., 3] == 255)
    assert info["scale_factor"] == 16 and info["scalebar_um"] is None
    assert info["source_width"] == 22 and info["width"] == 352


def test_figure_with_step_carries_a_nice_scale_bar():
    blob, rgb = _png(22, 21)
    out, info = X._figure_png(blob, 0.5)
    img = _decode(out)
    assert info["scalebar_um"] == 2.0            # 1/2/5 rule: 0.2 * 22 * 0.5 = 2.2 -> 2
    assert info["scalebar_px"] == 64             # 2 um / 0.5 um * 16
    assert info["source_pixel_um"] == 0.5
    assert info["figure_pixel_um"] == pytest.approx(0.5 / 16)
    # The bar is white pixels near the bottom-left; the top-right corner is
    # untouched map.
    h, w = img.shape[:2]
    white = np.all(img[..., :3] == 255, axis=-1)
    assert white[int(h * 0.8):, : int(w * 0.5)].sum() >= info["scalebar_px"] * 4
    expected = np.repeat(np.repeat(rgb, 16, axis=0), 16, axis=1)
    assert np.array_equal(img[: h // 2, w // 2:, :3], expected[: h // 2, w // 2:])


@pytest.mark.parametrize("target, bar", [
    (2.2, 2.0), (0.9, 0.5), (11, 10), (49, 20), (50, 50), (0.03, 0.02), (123, 100),
])
def test_nice_bar_lengths(target, bar):
    assert X._nice_bar_um(target) == pytest.approx(bar)


def test_no_figure_when_it_would_duplicate_the_map():
    """A map over 2000 px with no scan step: factor 1 and no bar -> nothing
    to add, so no byte-different duplicate is written."""
    small, _ = _png(22, 21)
    mid, _ = _png(1200, 40)        # 2x still fits 4000 px -> magnified
    big, _ = _png(2500, 40)        # 2x would exceed 4000 px -> factor 1
    assert X._figure_factor(2500) == 1
    assert X._figure_worthwhile(small, None) is True     # magnified
    assert X._figure_worthwhile(mid, None) is True       # magnified 2x
    assert X._figure_worthwhile(big, None) is False      # neither
    assert X._figure_worthwhile(big, 0.5) is True        # a bar
    assert X._figure_worthwhile(b"junk", 0.5) is False


def test_um_labels_have_no_trailing_zeros():
    assert X._format_um(2.0) == "2 µm"
    assert X._format_um(0.5) == "0.5 µm"
    assert X._format_um(20.0) == "20 µm"
