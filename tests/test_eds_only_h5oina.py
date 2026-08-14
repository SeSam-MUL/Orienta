"""Load an h5oina that has EDS + electron images but NO diffraction patterns.

Aztec writes "Elementverteilungsdaten" acquisitions with no /1/EBSD group at
all. Both loaders refuse those outright:

    kikuchipy : "... is not a supported h5ebsd file, as no top groups with
                 subgroup name 'EBSD' ... were found"
    unified   : "Could not find EBSD data group in file"

so the file could not be opened even though everything the EDS viewer needs
(Window Integral per element, SE/FSE images) is present. The EDS routes read
those through h5_session and never touch the EBSD signal, so an EDS-only load
is a legitimate mode — it just has to say plainly that there are no patterns.
"""
from pathlib import Path

import pytest

from safe_loader import probe_ebsd_content

EDS_ONLY = Path(
    r"E:/Masterarbeit_Schlegl_2026_07/h5oina/"
    r"Masterarbeit_Schlegl_2026_07 6m_EBSD_TEST Arbeitsbereich 12 "
    r"Elementverteilungsdaten 25.h5oina"
)
WITH_PATTERNS = Path(
    r"E:/Masterarbeit_Schlegl_2026_07/h5oina/"
    r"Masterarbeit_Schlegl_2026_07 6m_EBSD_TEST Arbeitsbereich 6 "
    r"Elementverteilungsdaten 21.h5oina"
)


# ---------------------------------------------------------------------------
# Probe — works on any HDF5, no vendor loader involved
# ---------------------------------------------------------------------------

def test_probe_reports_no_patterns_but_eds(tmp_path):
    """Synthetic stand-in so the rule is pinned without the 170 MB file."""
    import h5py
    import numpy as np

    p = tmp_path / "eds_only.h5oina"
    with h5py.File(p, "w") as f:
        g = f.create_group("1")
        eds = g.create_group("EDS/Data/Window Integral")
        eds.create_dataset("Al Ka1", data=np.zeros(12, dtype=np.float32))
        eds.create_dataset("Si Ka1", data=np.zeros(12, dtype=np.float32))
        g.create_group("Electron Image/Data").create_dataset(
            "SE", data=np.zeros((4, 4), dtype=np.uint8)
        )

    info = probe_ebsd_content(p)
    assert info["has_patterns"] is False
    assert info["has_eds"] is True
    assert info["has_electron_images"] is True
    assert info["eds_only"] is True


def test_probe_reports_patterns_when_present(tmp_path):
    import h5py
    import numpy as np

    p = tmp_path / "with_patterns.h5oina"
    with h5py.File(p, "w") as f:
        d = f.create_group("1/EBSD/Data")
        d.create_dataset("Processed Patterns", data=np.zeros((6, 4, 4), dtype=np.uint8))
        f.create_group("1/EDS/Data/Window Integral").create_dataset(
            "Al Ka1", data=np.zeros(6, dtype=np.float32)
        )

    info = probe_ebsd_content(p)
    assert info["has_patterns"] is True
    assert info["eds_only"] is False


def test_probe_is_quiet_on_a_non_hdf5_file(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("not hdf5", encoding="utf-8")
    info = probe_ebsd_content(p)
    assert info["has_patterns"] is False
    assert info["eds_only"] is False        # nothing to offer -> not an EDS file


def test_probe_finds_an_empty_pattern_dataset_as_no_patterns(tmp_path):
    """A zero-length patterns dataset is not usable EBSD data."""
    import h5py
    import numpy as np

    p = tmp_path / "empty.h5oina"
    with h5py.File(p, "w") as f:
        f.create_group("1/EBSD/Data").create_dataset(
            "Processed Patterns", data=np.zeros((0, 4, 4), dtype=np.uint8)
        )
        f.create_group("1/EDS/Data/Window Integral").create_dataset(
            "Al Ka1", data=np.zeros(3, dtype=np.float32)
        )
    info = probe_ebsd_content(p)
    assert info["has_patterns"] is False
    assert info["eds_only"] is True


# ---------------------------------------------------------------------------
# The real files
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not EDS_ONLY.is_file(), reason="user's EDS-only file not present")
def test_real_eds_only_file_is_classified_correctly():
    info = probe_ebsd_content(EDS_ONLY)
    assert info["has_patterns"] is False
    assert info["has_eds"] is True
    assert info["eds_only"] is True
    assert len(info["eds_elements"]) > 5
    assert "SE" in info["electron_images"] or "FSE" in info["electron_images"]


@pytest.mark.skipif(not WITH_PATTERNS.is_file(), reason="reference file not present")
def test_real_file_with_patterns_is_not_treated_as_eds_only():
    info = probe_ebsd_content(WITH_PATTERNS)
    assert info["has_patterns"] is True
    assert info["eds_only"] is False


@pytest.mark.skipif(not EDS_ONLY.is_file(), reason="user's EDS-only file not present")
def test_loading_the_eds_only_file_succeeds_and_says_there_are_no_patterns():
    from backend.api.routes.ebsd_viewer import _load_ebsd_blocking

    resp = _load_ebsd_blocking(str(EDS_ONLY))
    assert resp["success"] is True
    assert resp["has_patterns"] is False
    assert resp["has_eds"] is True
    assert resp["eds_elements"], "EDS viewer needs the element list"
    assert resp["electron_images"], "SE/FSE images should be offered"
    # The UI keys off this to explain the missing pattern views.
    assert resp.get("content_mode") == "eds_only"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# The extractor the EDS page actually queries
# ---------------------------------------------------------------------------

def _extractor(path):
    from backend.api.services.h5_session import open_file, get_extractor
    open_file(str(path))
    return get_extractor()


@pytest.mark.skipif(not EDS_ONLY.is_file(), reason="user's EDS-only file not present")
def test_extractor_serves_elements_and_images_without_an_ebsd_group():
    """_find_root_key required an EBSD group; without it the extractor
    reported no elements and no images, so the EDS page stayed empty."""
    from backend.api.services.h5_session import close_file

    try:
        ext = _extractor(EDS_ONLY)
        assert ext.root_key is not None, "no acquisition slot found"
        feats = ext.detect_available_features()
        assert feats["has_patterns"] is False
        assert feats["has_eds"] is True
        assert len(feats["eds_elements"]) >= 10
        assert len(feats["electron_images"]) >= 2
    finally:
        close_file()


@pytest.mark.skipif(not EDS_ONLY.is_file(), reason="user's EDS-only file not present")
def test_element_maps_come_back_on_the_real_grid():
    """The grid used to fall back to 1x1 (EBSD header missing), so every map
    was a flat 1-D array that could not be displayed."""
    import numpy as np
    from backend.api.services.h5_session import close_file

    try:
        ext = _extractor(EDS_ONLY)
        n_rows, n_cols = ext.get_grid_dimensions()
        assert (n_rows, n_cols) != (1, 1)
        el = ext.get_available_elements()[0]
        m = np.asarray(ext.get_element_map_2d(el))
        assert m.shape == (n_rows, n_cols)
        assert m.max() > 0, "element map is empty"
    finally:
        close_file()


@pytest.mark.skipif(not WITH_PATTERNS.is_file(), reason="reference file not present")
def test_normal_file_grid_is_unchanged():
    """The EDS grid fall-back must only fire when there is no EBSD header."""
    from backend.api.services.h5_session import close_file

    try:
        ext = _extractor(WITH_PATTERNS)
        feats = ext.detect_available_features()
        assert feats["has_patterns"] is True
        assert feats["grid_shape"] == (9, 12)   # from the EBSD header
    finally:
        close_file()
