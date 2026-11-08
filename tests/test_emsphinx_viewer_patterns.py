"""EMSphInx indexes the viewer's processed patterns only for the viewer's own file.

When the viewer's dataset was processed (background removal, averaging),
``_make_processed_temp_h5`` copies the source H5 and writes the processed
patterns into the copy, so the external binary indexes those. It never asked
WHICH file it was handed: a batch over file X while the viewer showed a
processed file Y wrote Y's patterns into X's copy whenever both had the same
pattern count (a series of equally sized scans), and the batch reported Y's
orientations as X's without a word.
"""
import h5py
import numpy as np
import pytest

import indexing_controller
from backend.api.routes import ebsd_viewer


#: The viewer's processed patterns: four different, non-constant patterns, so a
#: copy of the raw file and the written patterns cannot be mistaken for each other.
PROCESSED = np.arange(4 * 8 * 8, dtype=np.float64).reshape(2, 2, 8, 8)


class _Signal:
    def __init__(self, value=None):
        self.data = PROCESSED.copy()


def _h5(path, value):
    with h5py.File(path, "w") as f:
        f.create_dataset("1/EBSD/Data/Pattern", data=np.full((4, 8, 8), value, dtype=np.uint8))
    return str(path)


@pytest.fixture
def viewer(tmp_path, monkeypatch):
    """The viewer shows file Y, processed in memory (every pattern now 200)."""
    y = _h5(tmp_path / "Y.h5", 10)
    monkeypatch.setattr(ebsd_viewer, "is_active_signal_dirty", lambda: True)
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda: _Signal(200))
    monkeypatch.setattr(ebsd_viewer, "_ebsd_file_path", y)
    monkeypatch.setattr(indexing_controller, "_get_emsoft_data_path", lambda: tmp_path / "emsoft")
    return y


def test_another_file_keeps_its_own_patterns(viewer, tmp_path):
    x = _h5(tmp_path / "X.h5", 50)                      # same pattern count as Y
    assert indexing_controller._make_processed_temp_h5(x) is None
    with h5py.File(x, "r") as f:                        # and the file itself is untouched
        assert int(f["1/EBSD/Data/Pattern"][0, 0, 0]) == 50


def test_the_viewers_own_file_gets_its_processed_patterns(viewer):
    temp = indexing_controller._make_processed_temp_h5(viewer)
    try:
        assert temp is not None
        # rescaled globally into the uint8 range of the file, in the file's order
        flat = PROCESSED.reshape(-1, 8, 8)
        expected = ((flat - flat.min()) / (flat.max() - flat.min()) * 255).astype(np.uint8)
        with h5py.File(temp, "r") as f:
            np.testing.assert_array_equal(f["1/EBSD/Data/Pattern"][...], expected)
        with h5py.File(viewer, "r") as f:               # the source is never written
            assert np.all(f["1/EBSD/Data/Pattern"][...] == 10)
    finally:
        if temp:
            from pathlib import Path
            Path(temp).unlink(missing_ok=True)


def test_a_batch_never_takes_the_viewers_patterns(viewer):
    """Not even for the file the viewer shows: a batch indexes its file as it is."""
    assert indexing_controller._make_processed_temp_h5(viewer, use_viewer_patterns=False) is None


def test_spherical_index_patterns_hands_the_switch_on(viewer, monkeypatch):
    seen = []
    monkeypatch.setattr(indexing_controller, "_make_processed_temp_h5",
                        lambda h5, cb=None, use_viewer_patterns=True, variant=None: seen.append(use_viewer_patterns))
    monkeypatch.setattr(indexing_controller, "_spherical_index_patterns_impl", lambda **k: "r")
    monkeypatch.setattr(indexing_controller, "_active_scan_is_hex", lambda: False)
    from indexing_controller import IndexingConfig, IndexingMethod
    indexing_controller.spherical_index_patterns(
        h5_path=viewer, config=IndexingConfig(method=IndexingMethod.SPHERICAL),
        detector_params={}, use_viewer_patterns=False)
    assert seen == [False]

def test_an_unprocessed_viewer_changes_nothing(viewer, monkeypatch):
    monkeypatch.setattr(ebsd_viewer, "is_active_signal_dirty", lambda: False)
    assert indexing_controller._make_processed_temp_h5(viewer) is None
