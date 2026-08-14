"""A scale bar must use the pixel size of the IMAGE it is drawn on.

In an Aztec h5oina the areas do not share a pixel size. Measured on the user's
"Arbeitsbereich 6" file:

    EBSD / EDS      12 x 9      X Step 0.6579 um   field  7.89 um
    Electron Image  1024 x 768  X Step 0.0621 um   field 63.57 um

a factor of 10.6, on a different field of view entirely. The EDS page fed ONE
global step size to the export dialog, so a scale bar burnt into an exported SE
image was wrong by that factor — silently, in a figure meant for publication.

EDS-only acquisitions have no EBSD header at all, so the pixel size has to come
from the EDS / Electron Image headers there.
"""
from pathlib import Path

import pytest

EDS_ONLY = Path(
    r"E:/Masterarbeit_Schlegl_2026_07/h5oina/"
    r"Masterarbeit_Schlegl_2026_07 6m_EBSD_TEST Arbeitsbereich 12 "
    r"Elementverteilungsdaten 25.h5oina"
)
WITH_EBSD = Path(
    r"E:/Masterarbeit_Schlegl_2026_07/h5oina/"
    r"Masterarbeit_Schlegl_2026_07 6m_EBSD_TEST Arbeitsbereich 6 "
    r"Elementverteilungsdaten 21.h5oina"
)


def _extractor(path):
    from backend.api.services.h5_session import open_file, get_extractor
    open_file(str(path))
    return get_extractor()


def _synthetic(tmp_path, areas):
    """areas: {name: (x_cells, y_cells, x_step)}"""
    import h5py
    import numpy as np

    p = tmp_path / "s.h5oina"
    with h5py.File(p, "w") as f:
        for name, (xc, yc, step) in areas.items():
            h = f.create_group(f"1/{name}/Header")
            h.create_dataset("X Cells", data=np.array([xc], dtype=np.int32))
            h.create_dataset("Y Cells", data=np.array([yc], dtype=np.int32))
            if step is not None:
                h.create_dataset("X Step", data=np.array([step], dtype=np.float32))
                h.create_dataset("Y Step", data=np.array([step], dtype=np.float32))
            f.create_group(f"1/{name}/Data")
    return p


def test_each_area_reports_its_own_pixel_size(tmp_path):
    from tools.h5_viewer_backend import H5OINADataExtractor
    import h5py

    p = _synthetic(tmp_path, {
        "EBSD": (12, 9, 0.6578947),
        "EDS": (12, 9, 0.6578947),
        "Electron Image": (1024, 768, 0.06207658),
    })
    with h5py.File(p, "r") as f:
        sizes = H5OINADataExtractor(f, "Oxford").get_pixel_sizes()

    assert sizes["ebsd"]["x"] == pytest.approx(0.6578947, rel=1e-5)
    assert sizes["electron_image"]["x"] == pytest.approx(0.06207658, rel=1e-5)
    # The whole point: they must NOT be the same number.
    assert sizes["ebsd"]["x"] / sizes["electron_image"]["x"] == pytest.approx(10.6, rel=0.05)


def test_missing_area_is_none_not_a_guess(tmp_path):
    """A scale bar drawn from a guessed pixel size is worse than none."""
    from tools.h5_viewer_backend import H5OINADataExtractor
    import h5py

    p = _synthetic(tmp_path, {"EDS": (64, 48, 0.25)})
    with h5py.File(p, "r") as f:
        sizes = H5OINADataExtractor(f, "Oxford").get_pixel_sizes()
    assert sizes["eds"]["x"] == pytest.approx(0.25)
    assert sizes["ebsd"] is None
    assert sizes["electron_image"] is None


def test_falls_back_to_bounding_box_when_no_step(tmp_path):
    """Older exports carry the field of view but no explicit step."""
    import h5py
    import numpy as np
    from tools.h5_viewer_backend import H5OINADataExtractor

    p = tmp_path / "bb.h5oina"
    with h5py.File(p, "w") as f:
        h = f.create_group("1/EDS/Header")
        h.create_dataset("X Cells", data=np.array([100], dtype=np.int32))
        h.create_dataset("Y Cells", data=np.array([50], dtype=np.int32))
        h.create_dataset("Bounding Box Size", data=np.array([20.0, 10.0], dtype=np.float32))
        f.create_group("1/EDS/Data")
    with h5py.File(p, "r") as f:
        sizes = H5OINADataExtractor(f, "Oxford").get_pixel_sizes()
    assert sizes["eds"]["x"] == pytest.approx(0.2)
    assert sizes["eds"]["y"] == pytest.approx(0.2)
    assert sizes["eds"]["source"] == "bounding_box"


@pytest.mark.skipif(not WITH_EBSD.is_file(), reason="reference file not present")
def test_real_file_electron_image_differs_from_the_scan_step():
    from backend.api.services.h5_session import close_file

    try:
        sizes = _extractor(WITH_EBSD).get_pixel_sizes()
        assert sizes["ebsd"]["x"] == pytest.approx(0.6578947, rel=1e-4)
        assert sizes["electron_image"]["x"] == pytest.approx(0.06207658, rel=1e-4)
    finally:
        close_file()


@pytest.mark.skipif(not EDS_ONLY.is_file(), reason="user's EDS-only file not present")
def test_eds_only_file_still_has_a_pixel_size():
    """No EBSD header here — the scale bar has to come from EDS / Electron Image."""
    from backend.api.services.h5_session import close_file

    try:
        sizes = _extractor(EDS_ONLY).get_pixel_sizes()
        assert sizes["ebsd"] is None
        assert sizes["eds"]["x"] == pytest.approx(0.1814546, rel=1e-4)
        assert sizes["electron_image"]["x"] == pytest.approx(0.1814546, rel=1e-4)
        assert sizes["eds"]["units"] == "um"
    finally:
        close_file()


@pytest.mark.skipif(not EDS_ONLY.is_file(), reason="user's EDS-only file not present")
def test_metadata_endpoint_serves_a_scale_for_an_eds_only_file():
    """The export dialog greys the scale bar out when step_size is null."""
    from backend.api.routes.ebsd_viewer import _load_ebsd_blocking, get_metadata
    import asyncio
    import inspect

    _load_ebsd_blocking(str(EDS_ONLY))
    # asyncio.run, not get_event_loop: an earlier test in the suite may have
    # closed the loop, and that is a test artefact, not a product failure.
    meta = get_metadata()
    if inspect.isawaitable(meta):
        meta = asyncio.run(meta)
    assert meta.get("step_size"), "no step size -> scale bar stays disabled"
    assert meta["step_size"]["x"] == pytest.approx(0.1814546, rel=1e-4)
    assert meta.get("pixel_sizes", {}).get("electron_image")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
