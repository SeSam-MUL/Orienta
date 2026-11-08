"""The rich export computes the processed patterns BEFORE it removes the source's dataset.

``_overwrite_patterns_with_processed`` used to delete the dataset first and fit the new values after:
an exception in between left the export without its patterns (later reported as a missing dataset) in
a half-written file. A failure after the removal now takes the incomplete file away and raises.
"""
from __future__ import annotations

import shutil

import h5py
import numpy as np
import pytest

from tests.test_variant_rereads import (  # noqa: F401  (two_sources: fixture)
    RAW2, _edited_unprocessed, _kp, two_sources, unprocessed)


def _copy(two_sources, tmp_path):
    out = tmp_path / "out.h5"
    shutil.copy2(two_sources, out)
    return out


def _unprocessed_of(path, group):
    with h5py.File(path, "r") as f:
        return f[f"{group}/EBSD/Data/Unprocessed Patterns"][...]


def test_a_failure_while_fitting_the_new_patterns_leaves_the_source_dataset_in_place(
        two_sources, tmp_path, monkeypatch):
    from backend.api.routes.indexing import _overwrite_patterns_with_processed
    from backend.api.services import load_choices
    out = _copy(two_sources, tmp_path)

    def broken(*a, **k):
        raise RuntimeError("cannot fit")

    monkeypatch.setattr(load_choices, "fit_to_dtype", broken)
    assert _overwrite_patterns_with_processed(str(out), _kp(_edited_unprocessed(2)), RAW2) is False
    assert out.exists()
    assert np.array_equal(_unprocessed_of(out, 2), unprocessed(2))       # the dataset is still the file's own
    assert np.array_equal(_unprocessed_of(out, 1), unprocessed(1))


def test_a_failure_after_the_source_dataset_was_removed_removes_the_incomplete_file(
        two_sources, tmp_path, monkeypatch):
    from backend.api.routes.indexing import _overwrite_patterns_with_processed
    out = _copy(two_sources, tmp_path)
    real = h5py.Group.create_dataset

    def failing(self, name, *a, **k):
        if str(name).endswith("Unprocessed Patterns"):
            raise OSError("disk full")
        return real(self, name, *a, **k)

    monkeypatch.setattr(h5py.Group, "create_dataset", failing)
    with pytest.raises(OSError, match="disk full"):
        _overwrite_patterns_with_processed(str(out), _kp(_edited_unprocessed(2)), RAW2)
    assert not out.exists()                  # not a file to hand out: it would look like one without patterns


def test_the_successful_overwrite_is_unchanged(two_sources, tmp_path):
    from backend.api.routes.indexing import _overwrite_patterns_with_processed
    out = _copy(two_sources, tmp_path)
    assert _overwrite_patterns_with_processed(str(out), _kp(_edited_unprocessed(2)), RAW2) is True
    assert np.array_equal(_unprocessed_of(out, 2), _edited_unprocessed(2).reshape(-1, *unprocessed(2).shape[1:]))
    assert np.array_equal(_unprocessed_of(out, 1), unprocessed(1))
