import numpy as np
import h5py
import pytest
from backend.api.services import pattern_quality as pq


def _write_oxford_bc(path, n_rows, n_cols, values):
    with h5py.File(path, "w") as f:
        g = f.create_group("1")
        g.create_dataset("EBSD/Data/Band Contrast", data=np.asarray(values).ravel())
        h = g.create_group("EBSD/Header")
        h.create_dataset("X Cells", data=np.array(n_cols))
        h.create_dataset("Y Cells", data=np.array(n_rows))


class _FakeSignal:
    def __init__(self, iq):
        self._iq = np.asarray(iq, dtype=np.float64)
    def get_image_quality(self):
        return self._iq


def test_native_first_reads_oxford_bc(tmp_path):
    p = tmp_path / "s.h5oina"
    _write_oxford_bc(p, 2, 3, np.arange(6) * 40)  # 0..200
    qm = pq.get_quality_map(2, 3, source_file=str(p))
    assert qm.source == "native"
    assert qm.metric == "band_contrast"
    assert qm.label == pq.LABEL_NATIVE
    assert qm.array.shape == (2, 3)
    assert qm.value_range == (0, 255)


def test_falls_back_to_fft_when_no_native(tmp_path):
    iq = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    qm = pq.get_quality_map(2, 3, source_file=None, signal=_FakeSignal(iq))
    assert qm.source == "computed"
    assert qm.metric == "image_quality"
    assert qm.label == pq.LABEL_COMPUTED
    assert qm.array.shape == (2, 3)
    assert qm.value_range == (0, 1)


def test_xmap_prop_bc_treated_as_native(tmp_path):
    class _XM:
        prop = {"bc": np.array([10, 20, 30, 40, 50, 60], dtype=float)}
    qm = pq.get_quality_map(2, 3, source_file=None, xmap=_XM())
    assert qm.source == "native"
    assert qm.metric == "band_contrast"
    assert qm.array.shape == (2, 3)


def test_returns_none_when_nothing_available():
    assert pq.get_quality_map(2, 3, source_file=None, signal=None, xmap=None) is None


def test_no_cov_std_mean_path_exists():
    # The retired fake must not sneak back: the module exposes no CoV helper.
    assert not hasattr(pq, "_coefficient_of_variation")
    assert not any("std" in name and "mean" in name for name in dir(pq))


def test_native_read_does_not_mutate_shared_session(tmp_path, monkeypatch):
    # read_native_band_contrast on an explicit path must open its OWN handle.
    p = tmp_path / "s.h5oina"
    _write_oxford_bc(p, 2, 2, [1, 2, 3, 4])
    import backend.api.services.h5_session as h5s
    sentinel = object()
    monkeypatch.setattr(h5s, "_open_file", sentinel, raising=False)
    arr = pq.read_native_band_contrast(str(p), 2, 2)
    assert arr is not None
    assert getattr(h5s, "_open_file", None) is sentinel  # untouched
