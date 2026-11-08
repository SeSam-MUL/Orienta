"""Cropping the measurement that was loaded, and reloading a crop as the variant it was cut from.

The crop writer walks the whole file, but three things are the loaded measurement's alone: which
header the window is checked against, which electron-image grid places it, and which pattern
dataset the supplied (edited) patterns replace. They used to be the first measurement's: cropping a
second measurement with another grid failed with a plain ``ValueError``, and with the same grid the
first measurement's patterns were written into the second one's dataset as well.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import crop_export  # noqa: E402
from backend.api.services.crop_window import CropWindow  # noqa: E402
from backend.api.services.load_choices import LoadVariant, inspect_file  # noqa: E402

SY, SX = 4, 5
V1 = LoadVariant("1", "processed")
V2 = LoadVariant("2", "processed")
V2_RAW = LoadVariant("2", "unprocessed")


def _processed(g, n):
    return np.full((n, SY, SX), 10 * g, np.uint8)


def _unprocessed(g, n):
    return (np.arange(n * SY * SX) * 3 + 300 + 1000 * (g - 1)).astype(np.int16).reshape(n, SY, SX)


def _file(tmp_path, grids, name="multi.h5oina"):
    """An Oxford file; ``grids`` is ``{group: (rows, cols)}``. Measurement g has its own grid, its own
    patterns, a per-pixel Band Contrast (100 g + pixel index) and an electron image on its own grid."""
    path = tmp_path / name
    with h5py.File(path, "w") as f:
        f.create_dataset("Manufacturer", data=np.bytes_("Oxford Instruments"))
        f.create_dataset("Format Version", data=np.bytes_("5.0"))
        for g, (ny, nx) in grids.items():
            n = ny * nx
            hdr = f.create_group(f"{g}/EBSD/Header")
            data = f.create_group(f"{g}/EBSD/Data")
            for key, value in (("X Cells", nx), ("Y Cells", ny), ("X Step", 0.5), ("Y Step", 0.5),
                               ("Pattern Height", SY), ("Pattern Width", SX)):
                hdr.create_dataset(key, data=value)
            data.create_dataset("Processed Patterns", data=_processed(g, n))
            data.create_dataset("Unprocessed Patterns", data=_unprocessed(g, n))
            data.create_dataset("Band Contrast", data=(100 * g + np.arange(n)).astype(np.uint8))
            ei = f.create_group(f"{g}/Electron Image/Header")
            ei.create_dataset("X Cells", data=4 * nx)
            ei.create_dataset("Y Cells", data=4 * ny)
            f.create_dataset(f"{g}/Electron Image/Data/SE/Image 1",
                             data=np.zeros((4 * ny, 4 * nx), np.uint8))
    return str(path)


def _window(shape, row0=0, col0=1, rows=2, cols=2):
    return CropWindow("multi.h5oina", row0=row0, col0=col0, rows=rows, cols=cols, original_shape=shape)


def _idx(shape, window):
    return [(window.row0 + r) * shape[1] + window.col0 + c
            for r in range(window.rows) for c in range(window.cols)]


# ============================================================ B: the window is placed in ITS measurement

def test_cropping_a_measurement_with_another_grid_places_the_window_against_its_header(tmp_path):
    src = _file(tmp_path, {1: (2, 3), 2: (3, 4)})
    window = _window((3, 4), row0=1, col0=1, rows=2, cols=2)
    supplied = np.full((4, SY, SX), 77, np.uint8)
    out = tmp_path / "crop.h5oina"
    crop_export.write_cropped_h5oina(src, str(out), window, patterns=supplied, variant=V2)
    with h5py.File(out, "r") as f:
        assert np.array_equal(f["2/EBSD/Data/Processed Patterns"][...], supplied)
        assert f["2/EBSD/Data/Band Contrast"][...].tolist() == [200 + i for i in _idx((3, 4), window)]
        assert (int(f["2/EBSD/Header/X Cells"][()]), int(f["2/EBSD/Header/Y Cells"][()])) == (2, 2)
        # measurement 1 has another grid: nothing in it is the window's, it is copied as it was
        assert f["1/EBSD/Data/Band Contrast"][...].tolist() == [100 + i for i in range(6)]
        assert int(f["1/EBSD/Header/X Cells"][()]) == 3
        assert f["CropProvenance"].attrs["measurement"] == "2"


def test_a_window_from_another_grid_than_the_measurements_is_still_refused(tmp_path):
    src = _file(tmp_path, {1: (2, 3), 2: (3, 4)})
    with pytest.raises(ValueError, match=r"cut from a \(2, 3\) grid but multi.h5oina states \(3, 4\)"):
        crop_export.write_cropped_h5oina(src, str(tmp_path / "crop.h5oina"), _window((2, 3)),
                                         patterns=np.zeros((4, SY, SX), np.uint8), variant=V2)
    assert not (tmp_path / "crop.h5oina").exists()


def test_the_headers_and_the_electron_grid_are_the_named_measurements(tmp_path):
    src = _file(tmp_path, {1: (2, 3), 2: (3, 4)})
    with h5py.File(src, "r") as f:
        assert crop_export._declared_cells(crop_export._header_group(f, "EBSD", "2")) == (3, 4)
        assert crop_export._declared_cells(crop_export._header_group(f, "EBSD", "1")) == (2, 3)
        assert crop_export._declared_cells(crop_export._header_group(f, "EBSD")) == (2, 3)    # no name: the first
        assert crop_export._electron_grid(f, "2") == (12, 16)
        assert crop_export._electron_grid(f, "1") == (8, 12)
        assert crop_export._electron_grid(f) == (8, 12)
        assert crop_export._area_geom(f, ("EBSD",), "2")["x"] == 0.5


def test_the_electron_image_of_the_named_measurement_places_the_window(tmp_path):
    """Header-placed cut: the electron image of measurement 2 (12x16) is cut to the window's part, and
    measurement 1's image (another grid) is kept whole."""
    src = _file(tmp_path, {1: (2, 3), 2: (3, 4)})
    with h5py.File(src, "a") as f:
        for g in ("1", "2"):
            h = f[f"{g}/Electron Image/Header"]
            step = 0.125 if g == "2" else 0.1        # each measurement's image has its own pixel size
            h.create_dataset("X Step", data=step)
            h.create_dataset("Y Step", data=step)
            h.create_dataset("Relative Offset", data=np.array([0.0, 0.0]))
            f[f"{g}/EBSD/Header"].create_dataset("Relative Offset", data=np.array([0.0, 0.0]))
        f["2/Electron Image/Data/SE/Image 1"][...] = np.arange(12 * 16, dtype=np.uint8).reshape(12, 16)
    with h5py.File(src, "a") as f:
        # measurement 2 was scanned at half the step of measurement 1: a window on ITS cells is
        # half as big on the image (2 image pixels per cell, not 4), which only its own header says
        f["2/EBSD/Header/X Step"][()] = 0.25
        f["2/EBSD/Header/Y Step"][()] = 0.25
    window = _window((3, 4), row0=1, col0=1, rows=2, cols=2)
    out = tmp_path / "crop.h5oina"
    report = crop_export.write_cropped_h5oina(src, str(out), window, variant=V2)
    assert report["electron_images_cut"] == 1
    with h5py.File(out, "r") as f:
        img = f["2/Electron Image/Data/SE/Image 1"][...]
        assert img.shape == (4, 4)                  # 2x2 scan cells of 2x2 image pixels
        assert np.array_equal(img, np.arange(12 * 16, dtype=np.uint8).reshape(12, 16)[2:6, 2:6])
        assert f["1/Electron Image/Data/SE/Image 1"].shape == (8, 12)


def test_the_electron_grid_without_a_header_is_the_named_measurements_image(tmp_path):
    src = _file(tmp_path, {1: (2, 3), 2: (3, 4)})
    with h5py.File(src, "a") as f:
        for g in ("1", "2"):
            del f[f"{g}/Electron Image/Header"]            # only the image itself says how big it is
    with h5py.File(src, "r") as f:
        assert crop_export._electron_grid(f, "2") == (12, 16)
        assert crop_export._electron_grid(f, "1") == (8, 12)
        assert crop_export._electron_grid(f) == (8, 12)


# ============================================ C: a crop of the default variant touches the default only

def test_a_crop_without_a_choice_replaces_the_first_measurements_patterns_only(tmp_path):
    """Two measurements on the same grid, loaded without a choice: the supplied patterns are the first
    measurement's, and the second one's dataset is cut from the file like any other channel."""
    grid = (2, 3)
    src = _file(tmp_path, {1: grid, 2: grid})
    window = _window(grid)
    supplied = np.full((4, SY, SX), 77, np.uint8)
    out = tmp_path / "crop.h5oina"
    crop_export.write_cropped_h5oina(src, str(out), window, patterns=supplied)
    cut = _idx(grid, window)
    with h5py.File(out, "r") as f:
        assert np.array_equal(f["1/EBSD/Data/Processed Patterns"][...], supplied)
        assert np.array_equal(f["2/EBSD/Data/Processed Patterns"][...], _processed(2, 6)[cut])
        assert np.array_equal(f["1/EBSD/Data/Unprocessed Patterns"][...], _unprocessed(1, 6)[cut])
        assert "pattern_source" not in f["CropProvenance"].attrs        # no variant was chosen: nothing stated


def test_a_crop_of_the_second_measurement_leaves_the_first_ones_patterns_alone(tmp_path):
    grid = (2, 3)
    src = _file(tmp_path, {1: grid, 2: grid})
    window = _window(grid)
    supplied = np.full((4, SY, SX), 77, np.uint8)
    out = tmp_path / "crop.h5oina"
    crop_export.write_cropped_h5oina(src, str(out), window, patterns=supplied, variant=V2)
    cut = _idx(grid, window)
    with h5py.File(out, "r") as f:
        assert np.array_equal(f["2/EBSD/Data/Processed Patterns"][...], supplied)
        assert np.array_equal(f["1/EBSD/Data/Processed Patterns"][...], _processed(1, 6)[cut])


def test_an_edax_layout_crop_writes_into_its_one_pattern_dataset(tmp_path):
    path = tmp_path / "odd.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("Manufacturer", data=np.bytes_("Somebody Else"))
        hdr = f.create_group("S/EBSD/Header")
        hdr.create_dataset("nColumns", data=3)
        hdr.create_dataset("nRows", data=2)
        f.create_dataset("S/EBSD/Data/Pattern", data=np.zeros((6, SY, SX), np.uint8))
    assert inspect_file(str(path)).format == "edax_h5ebsd"
    supplied = np.full((4, SY, SX), 9, np.uint8)
    out = tmp_path / "crop.h5"
    crop_export.write_cropped_h5oina(str(path), str(out), CropWindow("odd.h5", 0, 1, 2, 2, (2, 3)),
                                     patterns=supplied)
    with h5py.File(out, "r") as f:
        assert np.array_equal(f["S/EBSD/Data/Pattern"][...], supplied)


# =========================================== reloading a crop: it is about the variant it was cut from

def test_a_crop_of_the_second_measurement_says_so_when_inspected_and_loads_as_it(tmp_path):
    import safe_loader
    from backend.api.routes import ebsd_viewer as ev
    grid = (2, 3)
    src = _file(tmp_path, {1: grid, 2: grid})
    supplied = (np.arange(4 * SY * SX) * 5 + 700).astype(np.int16).reshape(4, SY, SX)
    out = tmp_path / "crop.h5oina"
    crop_export.write_cropped_h5oina(src, str(out), _window(grid), patterns=supplied, variant=V2_RAW)
    choices = inspect_file(str(out))
    assert choices.suggested == V2_RAW
    assert choices.to_dict()["suggested"] == {"measurement": "2", "pattern_source": "unprocessed"}
    # a load without a choice takes it; the loader then reads exactly that measurement and source
    requested = ev._requested_variant(None, choices)
    assert requested == V2_RAW
    back = safe_loader.load_ebsd_safe(str(out), verbose=False, variant=requested)
    assert np.asarray(back.data).dtype == np.int16
    assert np.array_equal(np.asarray(back.data), supplied.reshape(2, 2, SY, SX))
    # an explicit choice is still the user's
    assert ev._requested_variant({"measurement": "1", "pattern_source": "processed"}, choices) == V1


def test_a_request_that_names_only_a_pattern_source_takes_the_crops_measurement(tmp_path):
    """A request with a source and no measurement is completed with the file's own: for a crop that is
    the measurement it was cut from (what a load without a choice takes), for any other file the
    default. The crop reload itself sends no variant, so it does not depend on this."""
    from backend.api.routes import ebsd_viewer as ev
    grid = (2, 3)
    src = _file(tmp_path, {1: grid, 2: grid})
    out = tmp_path / "crop.h5oina"
    crop_export.write_cropped_h5oina(src, str(out), _window(grid),
                                     patterns=np.zeros((4, SY, SX), np.int16), variant=V2_RAW)
    only_source = {"measurement": None, "pattern_source": "processed"}
    assert ev._requested_variant(only_source, inspect_file(str(out))) == V2
    assert ev._requested_variant(only_source, inspect_file(src)) == V1


def test_a_file_that_is_not_a_crop_and_a_crop_of_the_default_load_as_they_always_did(tmp_path):
    from backend.api.routes import ebsd_viewer as ev
    grid = (2, 3)
    src = _file(tmp_path, {1: grid, 2: grid})
    assert inspect_file(src).suggested is None
    assert "suggested" not in inspect_file(src).to_dict()
    assert ev._requested_variant(None, inspect_file(src)) is None
    out = tmp_path / "crop.h5oina"
    crop_export.write_cropped_h5oina(src, str(out), _window(grid), patterns=np.zeros((4, SY, SX), np.uint8))
    assert inspect_file(str(out)).suggested is None            # nothing was chosen, nothing is suggested
    assert ev._requested_variant(None, inspect_file(str(out))) is None


def test_a_crop_of_a_rich_export_goes_by_its_own_provenance(tmp_path):
    """The crop writer copies dataset attributes, so a crop of a marked rich export inherits the mark
    of the variant the export held. The crop's own record decides: here it records none."""
    from backend.api.routes.indexing import _mark_pattern_source
    grid = (2, 3)
    src = _file(tmp_path, {1: grid})
    _mark_pattern_source(src, LoadVariant("1", "unprocessed"), False)
    assert inspect_file(src).suggested == LoadVariant("1", "unprocessed")
    out = tmp_path / "crop.h5oina"
    crop_export.write_cropped_h5oina(src, str(out), _window(grid), patterns=np.zeros((4, SY, SX), np.uint8))
    assert inspect_file(str(out)).suggested is None


def test_a_rich_export_of_a_crop_loads_as_it_marks(tmp_path):
    """The export is a copy of the crop, CropProvenance included, and it ran on another source than the
    crop was cut from."""
    import shutil
    from backend.api.routes.indexing import _mark_pattern_source
    grid = (2, 3)
    src = _file(tmp_path, {1: grid, 2: grid})
    crop = tmp_path / "crop.h5oina"
    crop_export.write_cropped_h5oina(src, str(crop), _window(grid), patterns=np.zeros((4, SY, SX), np.int16),
                                     variant=V2_RAW)
    assert inspect_file(str(crop)).suggested == V2_RAW
    export = tmp_path / "rich.h5"
    shutil.copy2(crop, export)
    _mark_pattern_source(str(export), V2, False)
    assert inspect_file(str(export)).suggested == V2


def test_a_rich_export_of_a_default_result_on_a_crop_loads_as_the_default(tmp_path):
    """The crop was cut from the unprocessed patterns, the result ran on the default ones. The copy
    keeps CropProvenance, so the export has to say what it holds, or a reimport takes the crop's word."""
    import shutil
    from backend.api.routes.indexing import _mark_pattern_source
    raw = LoadVariant("1", "unprocessed")
    grid = (2, 3)
    src = _file(tmp_path, {1: grid})
    crop = tmp_path / "crop.h5oina"
    crop_export.write_cropped_h5oina(src, str(crop), _window(grid), patterns=np.zeros((4, SY, SX), np.int16),
                                     variant=raw)
    assert inspect_file(str(crop)).suggested == raw
    export = tmp_path / "rich.h5"
    shutil.copy2(crop, export)
    _mark_pattern_source(str(export), None, False)
    choices = inspect_file(str(export))
    assert choices.suggested == V1 and choices.suggested.is_default(choices)


@pytest.mark.parametrize("cut_from", [None, V1], ids=["no-record", "records-the-default"])
def test_a_rich_export_of_a_crop_of_the_default_stays_unmarked(tmp_path, cut_from):
    import shutil
    from backend.api.routes.indexing import _mark_pattern_source
    grid = (2, 3)
    src = _file(tmp_path, {1: grid})
    crop = tmp_path / "crop.h5oina"
    crop_export.write_cropped_h5oina(src, str(crop), _window(grid), patterns=np.zeros((4, SY, SX), np.uint8),
                                     variant=cut_from)
    export = tmp_path / "rich.h5"
    shutil.copy2(crop, export)
    _mark_pattern_source(str(export), None, False)
    with h5py.File(export, "r") as f:
        assert "pattern_source" not in f["1/EBSD/Data/Processed Patterns"].attrs


def test_a_suggestion_the_file_cannot_honour_is_dropped(tmp_path):
    grid = (2, 3)
    src = _file(tmp_path, {1: grid, 2: grid})
    out = tmp_path / "crop.h5oina"
    crop_export.write_cropped_h5oina(src, str(out), _window(grid), patterns=np.zeros((4, SY, SX), np.uint8),
                                     variant=V2_RAW)
    with h5py.File(out, "a") as f:
        f["CropProvenance"].attrs["measurement"] = "9"          # a measurement the file does not hold
    assert inspect_file(str(out)).suggested is None
