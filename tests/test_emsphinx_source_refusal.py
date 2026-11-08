"""EMSphInx (CPU) refuses a source it cannot read, with a message the user can act on.

Indexing via SHT with data that were not pre-processed gave an error message. EMSphInx reads the patterns from the FILE. For a
source that is not HDF5 (an EDAX .up1/.up2) that died in the export step with a
bare h5py message, and for a hexagonal scan the file's pattern order does not match
the square grid EMSphInx is handed. The
spherical GPU backend, Hough and Dictionary index the patterns in memory, so the
refusal names them.
"""
import h5py
import numpy as np
import pytest

import indexing_controller
from indexing_controller import (
    IndexingConfig, IndexingMethod, _emsphinx_source_refusal, spherical_index_patterns,
)


@pytest.fixture
def square_scan(monkeypatch):
    monkeypatch.setattr(indexing_controller, "_active_scan_is_hex", lambda: False)


@pytest.fixture
def up1(tmp_path):
    p = tmp_path / "Scan.up1"
    p.write_bytes(b"\x04\x00\x00\x00" + bytes(200))   # an UP1 header is not HDF5
    return str(p)


@pytest.fixture
def h5(tmp_path):
    p = tmp_path / "Scan.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("1/EBSD/Data/Pattern", data=np.zeros((4, 8, 8), dtype=np.uint8))
    return str(p)


def test_refuses_a_source_that_is_not_hdf5(square_scan, up1):
    msg = _emsphinx_source_refusal(up1)
    assert msg is not None
    assert ".up1" in msg and "not HDF5" in msg
    for alternative in ("GPU", "Hough", "Dictionary"):
        assert alternative in msg


def test_refuses_a_hexagonal_scan(monkeypatch, h5):
    monkeypatch.setattr(indexing_controller, "_active_scan_is_hex", lambda: True)
    msg = _emsphinx_source_refusal(h5)
    assert msg is not None and "hexagonal" in msg


def test_the_hexagonal_text_does_not_claim_a_resampling(h5):
    """A hexagonal scan is loaded on its own lattice, nothing is resampled onto a square grid; the reason is that
    EMSphInx takes the file's patterns as points of a square grid."""
    msg = _emsphinx_source_refusal(h5, scan_geometry={"kind": "hex"})
    assert "resampled" not in msg and "square grid" in msg
    assert "hexagonal scan" in msg and "pattern order" in msg
    for alternative in ("GPU", "Hough", "Dictionary"):
        assert alternative in msg


def test_emsphinx_refusal_code(up1, h5, square_scan):
    """The refusal of a source is a ``ValueError`` that carries the code the batch stores and the interface
    translates; the crop refusal and every other ``ValueError`` stay without it."""
    from indexing_controller import EmsphinxSourceRefused
    for path, stamp in ((up1, {"kind": "square"}), (h5, {"kind": "hex"})):
        with pytest.raises(ValueError, match="cannot index this dataset") as raised:
            spherical_index_patterns(
                h5_path=path, config=IndexingConfig(method=IndexingMethod.SPHERICAL),
                detector_params={}, scan_geometry=stamp)
        assert isinstance(raised.value, EmsphinxSourceRefused)
        assert raised.value.code == "emsphinxSourceRefused"


def test_the_crop_refusal_has_no_source_code(monkeypatch, h5):
    monkeypatch.setattr(indexing_controller, "_emsphinx_crop_refusal", lambda *a, **k: "a crop")
    with pytest.raises(ValueError, match="a crop") as raised:
        spherical_index_patterns(
            h5_path=h5, config=IndexingConfig(method=IndexingMethod.SPHERICAL),
            detector_params={}, scan_geometry={"kind": "square"})
    assert getattr(raised.value, "code", None) is None


def test_lets_an_hdf5_square_scan_through(square_scan, h5):
    assert _emsphinx_source_refusal(h5) is None


def test_a_missing_path_is_not_refused_here(square_scan, tmp_path):
    # Whoever opens it reports what is wrong; this guard only knows "not HDF5".
    assert _emsphinx_source_refusal(str(tmp_path / "gone.h5")) is None


def test_a_stamp_decides_instead_of_the_viewer(monkeypatch, h5):
    """A batch run indexes its own file, never the viewer's active dataset. A square
    batch file was refused while the viewer showed a hexagonal scan."""
    monkeypatch.setattr(indexing_controller, "_active_scan_is_hex",
                        lambda: pytest.fail("a run with a stamp must not ask the viewer"))
    assert _emsphinx_source_refusal(h5, scan_geometry={"kind": "square"}) is None
    assert _emsphinx_source_refusal(h5, scan_geometry=None) is None          # unknown is not hex
    msg = _emsphinx_source_refusal(h5, scan_geometry={"kind": "hex"})
    assert msg is not None and "hexagonal" in msg


def test_spherical_index_patterns_passes_its_stamp_on(monkeypatch, h5):
    monkeypatch.setattr(indexing_controller, "_active_scan_is_hex", lambda: True)   # viewer: hex
    monkeypatch.setattr(indexing_controller, "_make_processed_temp_h5", lambda *a, **k: None)
    ran = []
    monkeypatch.setattr(indexing_controller, "_spherical_index_patterns_impl",
                        lambda **k: ran.append(k["h5_path"]) or "result")
    out = spherical_index_patterns(
        h5_path=h5, config=IndexingConfig(method=IndexingMethod.SPHERICAL),
        detector_params={}, scan_geometry={"kind": "square"},
    )
    assert out == "result" and ran == [h5]


def test_the_batch_hands_emsphinx_its_own_files_geometry():
    """The batch EMSphInx call carries the job's stamp, like its GPU sibling."""
    import inspect
    from backend.api.services import batch_manager
    src = inspect.getsource(batch_manager)
    call = src[src.index("result = spherical_index_patterns("):]
    call = call[:call.index(")\n")]
    assert "scan_geometry=_scan_stamp" in call
    assert "use_viewer_patterns=False" in call


@pytest.mark.parametrize("stamp", ['_run_scan_stamp["scan_geometry"]', '_item_scan_stamp["scan_geometry"]'])
def test_the_routes_hand_emsphinx_their_runs_stamp(stamp):
    """The interactive run and batch v2 pass the stamp taken at run start, like
    their GPU siblings: a file switch during the run must not change the verdict."""
    import inspect
    from backend.api.routes import indexing
    src = inspect.getsource(indexing)
    calls = src.split("result = spherical_index_patterns(")[1:]
    assert any(f"scan_geometry={stamp}" in c[:c.index(")\n")] for c in calls)


def test_spherical_index_patterns_refuses_before_touching_the_file(square_scan, up1, monkeypatch):
    monkeypatch.setattr(indexing_controller, "_spherical_index_patterns_impl",
                        lambda **k: pytest.fail("EMSphInx must not run on a non-HDF5 source"))
    monkeypatch.setattr(indexing_controller, "_make_processed_temp_h5",
                        lambda *a, **k: pytest.fail("must refuse before writing a temp H5"))
    with pytest.raises(ValueError, match="cannot index this dataset"):
        spherical_index_patterns(
            h5_path=up1, config=IndexingConfig(method=IndexingMethod.SPHERICAL),
            detector_params={"n_rows": 2, "n_cols": 2},
            selection_mask=np.ones((2, 2), dtype=bool),
        )
