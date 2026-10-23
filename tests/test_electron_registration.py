"""An electron image laid over the EBSD/EDS map must be the part of the specimen
the map covers.

The EDS page background and the phase-map ``se:`` layer stretch whatever image
they are given over the whole map box. In an Aztec file the map is usually
acquired INSIDE a wider SE field, so the image used to be squeezed 1.1-5.7x
(measured: particles 12-97 EBSD pixels off). The header says where the map sits
in the image -- ``Relative Offset`` / ``Relative Size`` of the EBSD area, as
fractions of the electron image, whose own are [0, 0] / [1, 1] -- and the
backend now cuts the image to that extent, in native pixels.

The header values below are the ones read from the real files with h5py
(``Test_data``: an Oxford reference scan and aluminium-alloy scans), not invented ones.
"""
import base64
import io

import h5py
import numpy as np
import pytest

from backend.api.services import h5_session
from backend.api.services.crop_window import CropWindow
from backend.api.services.cropped_extractor import CroppedExtractor

IMG_W, IMG_H = 1024, 768
NAME = "SE/Bild 1"

# (label, cells (x, y), scan step, offset, size, image step) -- as in the files
SAMPLE_B = ("SampleB AB1", (120, 90), 0.5, [0.0, 0.0], [1.0, 1.0],
            0.05897275358438492)
AB1 = ("7050 AB1", (136, 39), 0.10000000149011612,
       [0.06955012679100037, 0.25501713156700134],
       [0.7882347702980042, 0.3013838827610016], 0.01684935949742794)
AB2 = ("7050 AB2", (166, 60), 0.20000000298023224,
       [0.0382525697350502, 0.40074118971824646],
       [0.9071323275566101, 0.44023242592811584], 0.035741064697504044)
AB3_7 = ("7050 AB3-7", (58, 49), 0.10000000149011612,
         [0.6855655312538147, 0.31352755427360535],
         [0.19209082424640656, 0.21637816727161407], 0.02948637679219246)
AB3_9 = ("7050 AB3-9", (56, 40), 0.10000000149011612,
         [0.09935732185840607, 0.6403027176856995],
         [0.18546700477600098, 0.1766352355480194], 0.02948637679219246)
REAL = [SAMPLE_B, AB1, AB2, AB3_7, AB3_9]


def write_file(path, cells, step, offset, size, img_step, img_offset=(0.0, 0.0),
               img_size=(1.0, 1.0), drop_scan_offset=False):
    """An Oxford-shaped file whose electron image encodes its own column index
    (value == column), so the cut a reader gets back can be read off exactly."""
    nx, ny = cells
    f32 = np.float32
    with h5py.File(path, "w") as f:
        for area in ("EBSD", "EDS"):
            h = f.create_group(f"1/{area}/Header")
            h.create_dataset("X Cells", data=np.array([nx], dtype=np.int32))
            h.create_dataset("Y Cells", data=np.array([ny], dtype=np.int32))
            h.create_dataset("X Step", data=np.array([step], dtype=f32))
            h.create_dataset("Y Step", data=np.array([step], dtype=f32))
            if not drop_scan_offset:
                h.create_dataset("Relative Offset", data=np.array(offset, dtype=f32))
            h.create_dataset("Relative Size", data=np.array(size, dtype=f32))
        f.create_dataset("1/EBSD/Data/Band Contrast",
                         data=np.arange(nx * ny, dtype=np.uint8))
        eds = f.create_group("1/EDS/Data/Window Integral/Al")
        eds.create_dataset("Counts", data=np.arange(nx * ny, dtype=f32))
        ih = f.create_group("1/Electron Image/Header")
        ih.create_dataset("X Cells", data=np.array([IMG_W], dtype=np.int32))
        ih.create_dataset("Y Cells", data=np.array([IMG_H], dtype=np.int32))
        ih.create_dataset("X Step", data=np.array([img_step], dtype=f32))
        ih.create_dataset("Y Step", data=np.array([img_step], dtype=f32))
        ih.create_dataset("Relative Offset", data=np.array(img_offset, dtype=f32))
        ih.create_dataset("Relative Size", data=np.array(img_size, dtype=f32))
        ramp = np.tile(np.arange(IMG_W, dtype=np.int16), (IMG_H, 1))
        f.create_dataset(f"1/Electron Image/Data/{NAME}", data=ramp.ravel())


@pytest.fixture
def opened():
    """Open a file in the h5 session; closed again afterwards."""
    def _open(path):
        h5_session.open_file(str(path))
        return h5_session.get_extractor()
    yield _open
    h5_session.close_file()


def full_window(ext):
    rows, cols = ext.get_grid_dimensions()
    return CropWindow("s", 0, 0, rows, cols, (rows, cols))


def overlay(ext, window=None):
    view = CroppedExtractor(ext, window or full_window(ext))
    return view, view.get_electron_image(NAME)


@pytest.mark.parametrize("case", REAL, ids=[c[0] for c in REAL])
def test_the_returned_image_covers_exactly_the_map(tmp_path, opened, case):
    """The physical size of the returned image equals the map's (step x cells)
    -- the property that makes 'stretch it over the map box' correct, and the
    scale bar (image step, native pixels) right."""
    label, cells, step, off, size, img_step = case
    p = tmp_path / "f.h5oina"
    write_file(p, cells, step, off, size, img_step)
    ext = opened(p)
    view, img = overlay(ext)
    h, w = img.shape
    assert w * img_step == pytest.approx(cells[0] * step, rel=0.012), label
    assert h * img_step == pytest.approx(cells[1] * step, rel=0.012), label
    assert view.last_crop_status(NAME)["registration"] == "registered"
    # native pixels: every value is an untouched source value (a column index)
    assert np.array_equal(np.unique(img[0]), np.arange(img[0, 0], img[0, 0] + w))
    # the scale-bar rule: the area's pixel size is still the file's image step
    assert ext.get_pixel_sizes()["electron_image"]["x"] == pytest.approx(img_step)


def test_7050_ab1_is_cut_at_the_header_offset(tmp_path, opened):
    """0.06955 x 1024 = 71.2 -> column 71;  (0.06955 + 0.78823) x 1024 = 878.4
    -> 878;  rows 0.25502 x 768 = 195.9 -> 196,  0.55640 x 768 = 427.3 -> 427.
    Hard numbers on purpose: the expectation must not come from the code."""
    label, cells, step, off, size, img_step = AB1
    p = tmp_path / "f.h5oina"
    write_file(p, cells, step, off, size, img_step)
    _, img = overlay(opened(p))
    assert img.shape == (427 - 196, 878 - 71)
    assert img[0, 0] == 71 and img[0, -1] == 877
    # 807 px x 0.016849 um = 13.60 um = 136 cells x 0.1 um
    assert img.shape[1] * img_step == pytest.approx(13.6, rel=0.005)


def test_7050_ab3_7_offset_is_far_from_the_origin(tmp_path, opened):
    """The 5.2x case: map at 68.6 % of the image, 19.2 % wide."""
    label, cells, step, off, size, img_step = AB3_7
    p = tmp_path / "f.h5oina"
    write_file(p, cells, step, off, size, img_step)
    _, img = overlay(opened(p))
    # 0.68557 x 1024 = 702.0;  0.87766 x 1024 = 898.7 -> 899
    assert img[0, 0] == 702 and img.shape[1] == 899 - 702


def test_sampleb_style_header_leaves_the_image_whole(tmp_path, opened):
    """Offset 0, size 1: header-consistent, so nothing changes."""
    label, cells, step, off, size, img_step = SAMPLE_B
    p = tmp_path / "f.h5oina"
    write_file(p, cells, step, off, size, img_step)
    view, img = overlay(opened(p))
    assert img.shape == (IMG_H, IMG_W)
    assert img[0, 0] == 0 and img[0, -1] == IMG_W - 1
    assert view.last_crop_status(NAME) == {
        "cropped": True, "exact": True, "reason": None,
        "registration": "registered"}


def test_the_images_own_offset_is_the_frame_not_a_disagreement(tmp_path, opened):
    """Scan offset [0.25, ...] against an image whose own offset is 0.25 is
    the scan at the image's left edge."""
    label, cells, step, off, size, img_step = AB1
    p = tmp_path / "f.h5oina"
    write_file(p, cells, step, [0.25, 0.25], size, img_step,
               img_offset=(0.25, 0.25))
    _, img = overlay(opened(p))
    assert img[0, 0] == 0


# --- when the file cannot place the image: deliver as before, and flag -------

def test_no_scan_offset_means_the_whole_image_flagged_unverified(tmp_path, opened):
    label, cells, step, off, size, img_step = AB1
    p = tmp_path / "f.h5oina"
    write_file(p, cells, step, off, size, img_step, drop_scan_offset=True)
    view, img = overlay(opened(p))
    assert img.shape == (IMG_H, IMG_W)
    st = view.last_crop_status(NAME)
    assert st["registration"] == "unverified"
    assert st["cropped"] is True      # nothing to crop; only the alignment is a guess
    assert st["reason"]


def test_a_relative_size_that_contradicts_the_step_size_is_unverified(tmp_path,
                                                                      opened):
    """Size says 50 % of the image (8.6 um); 136 cells x 0.1 um say 13.6 um."""
    label, cells, step, off, size, img_step = AB1
    p = tmp_path / "f.h5oina"
    write_file(p, cells, step, off, [0.5, 0.5], img_step)
    view, img = overlay(opened(p))
    assert img.shape == (IMG_H, IMG_W)
    assert view.last_crop_status(NAME)["registration"] == "unverified"


def test_a_map_before_the_image_is_padded_transparent(tmp_path, opened):
    """Offset -0.1: the first 10 % of the map has no image. The picture keeps
    the map's shape (NaN marks the gap), so every feature keeps its place."""
    label, cells, step, off, size, img_step = AB1
    p = tmp_path / "f.h5oina"
    write_file(p, cells, step, [-0.1, off[1]], size, img_step)
    ext = opened(p)
    view, img = overlay(ext)
    gap = int(round(0.1 * IMG_W))                   # 102 columns
    assert img.dtype == np.float32
    assert np.isnan(img[:, :gap]).all()
    assert img[0, gap] == 0                         # the image's own column 0
    assert view.last_crop_status(NAME)["exact"] is True


def test_padding_reaches_the_browser_as_alpha(tmp_path, opened):
    from PIL import Image
    from backend.api.services.image_utils import array_to_base64_raw

    arr = np.full((4, 6), np.nan, dtype=np.float32)
    arr[:, 3:] = np.arange(12, dtype=np.float32).reshape(4, 3)
    png = Image.open(io.BytesIO(base64.b64decode(array_to_base64_raw(arr))))
    assert png.mode == "LA"
    grey, alpha = np.array(png)[..., 0], np.array(png)[..., 1]
    assert (alpha[:, :3] == 0).all() and (alpha[:, 3:] == 255).all()
    assert grey[:, 3:].min() == 0 and grey[:, 3:].max() == 255   # scaled on real pixels


# --- the navigation crop uses the same projection ----------------------------

def test_a_crop_window_is_the_same_projection_as_the_overlay(tmp_path, opened):
    """Cropped to scan columns 34..94 of an alloy scan, the electron image must be the
    sub-box of the full overlay at the same place: 71.22 + 34 x 5.9348 = 273.0,
    71.22 + 94 x 5.9348 = 629.1."""
    label, cells, step, off, size, img_step = AB1
    p = tmp_path / "f.h5oina"
    write_file(p, cells, step, off, size, img_step)
    ext = opened(p)
    window = CropWindow("s", 5, 34, 20, 60, (39, 136))
    _, full = overlay(ext)
    _, cut = overlay(ext, window)
    assert cut[0, 0] == 273 and cut[0, -1] == 628
    # same pixels as the overlay, at the offset between the two boxes
    assert np.array_equal(cut[:, 0], full[cut.shape[0] * 0:cut.shape[0], 273 - 71])
    # y: 195.85 + 5 x 5.9348 = 225.5 -> 226 (round half to even is not used: 225.52)
    assert cut.shape[0] == round(195.85 + 25 * 5.9348) - round(195.85 + 5 * 5.9348)


# --- the route: what the EDS page and the phase map actually fetch -------------

@pytest.fixture
def client():
    from backend.api.main import app
    from fastapi.testclient import TestClient
    return TestClient(app)


def test_dataset_scope_is_registered_file_scope_is_the_stored_image(
        tmp_path, opened, client):
    label, cells, step, off, size, img_step = AB1
    p = tmp_path / "f.h5oina"
    write_file(p, cells, step, off, size, img_step)
    opened(p)
    url = f"/api/h5/electron/{NAME}"

    reg = client.get(url, params={"scope": "dataset"}).json()
    assert reg["shape"] == [427 - 196, 878 - 71]
    assert reg["crop"]["registration"] == "registered"

    # the cockpit / standalone views are untouched
    raw = client.get(url).json()
    assert raw["shape"] == [IMG_H, IMG_W]
    assert "registration" not in raw["crop"]


def test_the_route_flags_an_unverifiable_image(tmp_path, opened, client):
    label, cells, step, off, size, img_step = AB1
    p = tmp_path / "f.h5oina"
    write_file(p, cells, step, off, size, img_step, drop_scan_offset=True)
    opened(p)
    r = client.get(f"/api/h5/electron/{NAME}", params={"scope": "dataset"}).json()
    assert r["shape"] == [IMG_H, IMG_W]
    assert r["crop"]["registration"] == "unverified"
