"""Grain-boundary overlay: where the lines land, and which class they get."""

import numpy as np
import pytest

from backend.api.services.grain_boundaries import (
    SS,
    default_bands,
    parse_bands,
    render_boundaries,
)


def _angles(n_rows, n_cols, fill_h=np.nan, fill_v=np.nan):
    return {
        "h": np.full((n_rows, n_cols - 1), fill_h, dtype=np.float32),
        "v": np.full((n_rows - 1, n_cols), fill_v, dtype=np.float32),
    }


def test_overlay_is_transparent_without_boundaries():
    out = render_boundaries(_angles(6, 6), 6, 6, default_bands())
    assert out.shape == (6 * SS, 6 * SS, 4)
    assert out[..., 3].max() == 0, "no angles means nothing to draw"


def test_line_sits_on_the_interface_not_on_the_pixel():
    # A single vertical interface between column 2 and column 3. Drawn at
    # overlay column 3*SS: on the boundary, not through the middle of a pixel.
    ang = _angles(4, 6)
    ang["h"][:, 2] = 30.0
    bands = [{"id": "hagb", "min": 15, "max": None, "color": "#ffffff", "width": 1, "on": True}]
    out = render_boundaries(ang, 4, 6, bands)

    painted_cols = np.unique(np.nonzero(out[..., 3])[1])
    assert painted_cols.min() >= 3 * SS - 1 and painted_cols.max() <= 3 * SS + 1
    # …and it spans the full height of those pixel rows.
    assert (out[:, 3 * SS, 3] > 0).all()


def test_each_angle_lands_in_exactly_one_class():
    # 2°, 9° and 40° — one per class. Each must be drawn in its own colour and
    # nowhere else, or the map would double-count boundaries.
    ang = _angles(2, 4)
    ang["h"][0, 0] = 2.0
    ang["h"][0, 1] = 9.0
    ang["h"][0, 2] = 40.0
    out = render_boundaries(ang, 2, 4, default_bands())

    rgb = out[..., :3][out[..., 3] > 0]
    seen = {tuple(c) for c in np.unique(rgb, axis=0)}
    assert seen == {(0x8b, 0xe9, 0xfd), (0xf1, 0xfa, 0x8c), (0xff, 0xff, 0xff)}


def test_angles_below_the_first_class_are_not_drawn():
    ang = _angles(3, 3)
    ang["h"][:, :] = 0.2          # noise, under the 0.5° floor
    out = render_boundaries(ang, 3, 3, default_bands())
    assert out[..., 3].max() == 0


def test_a_switched_off_class_disappears():
    ang = _angles(2, 3)
    ang["h"][0, 0] = 30.0
    bands = default_bands()
    on = render_boundaries(ang, 2, 3, bands)
    bands[2]["on"] = False
    off = render_boundaries(ang, 2, 3, bands)
    assert on[..., 3].sum() > 0
    assert off[..., 3].sum() == 0


def test_width_widens_the_line():
    ang = _angles(4, 4)
    ang["v"][1, :] = 30.0
    thin = render_boundaries(ang, 4, 4, [{"min": 15, "max": None, "color": "#fff", "width": 1, "on": True}])
    thick = render_boundaries(ang, 4, 4, [{"min": 15, "max": None, "color": "#fff", "width": 4, "on": True}])
    assert (thick[..., 3] > 0).sum() > (thin[..., 3] > 0).sum()


def test_nan_edges_stay_blank():
    # NaN marks a pair that cannot be compared — an unindexed pixel, or two
    # different phases. Drawing a line there would claim an angle we never had.
    ang = _angles(3, 3)
    ang["h"][:, :] = np.nan
    ang["v"][:, :] = np.nan
    out = render_boundaries(ang, 3, 3, default_bands())
    assert out[..., 3].max() == 0


@pytest.mark.parametrize("raw", ["", "not json", "{}", "[]", "null"])
def test_unreadable_bands_fall_back_to_the_defaults(raw):
    assert parse_bands(raw) == default_bands()


def test_bands_come_through_as_written():
    got = parse_bands('[{"min": 1, "max": 4, "color": "#ff0000", "width": 2, "on": true}]')
    assert got == [{"min": 1, "max": 4, "color": "#ff0000", "width": 2, "on": True}]
