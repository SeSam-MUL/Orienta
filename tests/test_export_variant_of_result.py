"""A full .h5 export states the variant of the RESULT, as the light export does.

The full export used the variant of the dataset the viewer holds, the light export the result's stamp.
The viewer can hold another measurement than the one a result was indexed on (the result stays in the
gallery while the viewer is switched), and the export then marked, and wrote the viewer's processed
patterns into, the wrong measurement.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.services import variant_registry
from tests.test_imported_result_variant import _result
from tests.test_indexing_export_light import restore_indexing_registry  # noqa: F401  (autouse)
from tests.two_measurements import *  # noqa: F401,F403  (constants and fixtures)


@pytest.fixture
def export(two, tmp_path, monkeypatch):
    """``do(stamp, viewer_variant, dirty=False)``: POST a full export of a result stamped ``stamp`` while
    the viewer holds ``viewer_variant`` of ``two``; returns the exported file."""
    import backend.api.routes.indexing as ix
    from backend.api.main import app
    from backend.api.routes import ebsd_viewer as ev
    count = []

    def do(stamp, viewer_variant, dirty=False):
        monkeypatch.setattr(ev, "_ebsd_file_path", two, raising=False)
        monkeypatch.setattr(ev, "_active_dataset", "ds-export")
        if viewer_variant is None:
            variant_registry.drop("ds-export")
        else:
            variant_registry.register("ds-export", viewer_variant)
        monkeypatch.setattr(ev, "is_active_signal_dirty", lambda: dirty)
        ix._result_registry.clear()
        ix._result_registry["r"] = _result(stamp, two)
        ix._active_result_id = "r"
        count.append(1)
        out = tmp_path / f"full-{len(count)}.h5"
        resp = TestClient(app).post("/api/indexing/export", json={"format": "h5", "filename": str(out)})
        assert resp.status_code == 200, resp.text
        return out

    yield do
    variant_registry.drop("ds-export")


def _dataset_attrs(path, group, name):
    with h5py.File(path, "r") as f:
        return dict(f[f"{group}/EBSD/Data/{name}"].attrs)


def test_the_export_marks_the_results_dataset_not_the_viewers(export):
    """The result ran on measurement 2's unprocessed patterns; the viewer holds the default."""
    out = export(V2_RAW.to_dict(), None)
    attrs = _dataset_attrs(out, "2", "Unprocessed Patterns")
    assert attrs["pattern_source"] == "unprocessed" and attrs["measurement"] == "2"
    assert attrs["patterns_processed"] == False                      # noqa: E712  (h5py bool)
    with h5py.File(out, "r") as f:
        assert f["Indexing"].attrs["measurement"] == "2"             # and so does the result group
        assert "pattern_source" not in f["1/EBSD/Data/Processed Patterns"].attrs


def test_a_default_result_is_not_marked_for_the_viewers_other_measurement(export):
    """The converse: the result is the default variant, the viewer holds measurement 2."""
    out = export(None, V2_RAW)
    with h5py.File(out, "r") as f:
        for group, name in (("1", "Processed Patterns"), ("2", "Unprocessed Patterns"),
                            ("2", "Processed Patterns")):
            assert "pattern_source" not in f[f"{group}/EBSD/Data/{name}"].attrs, (group, name)


def test_a_default_result_of_a_marked_file_does_not_inherit_the_mark(export, two):
    """The source is a rich export of measurement 2's unprocessed patterns; the result is the default
    variant. The copy would say measurement 2 and reload as a variant the result never saw."""
    from backend.api.routes.indexing import _mark_pattern_source
    from backend.api.services.load_choices import inspect_file
    _mark_pattern_source(two, V2_RAW, False)
    assert inspect_file(two).suggested == V2_RAW
    assert inspect_file(str(export(None, None))).suggested is None


def test_the_mark_step_leaves_a_file_without_a_pattern_dataset_alone(two, tmp_path):
    """An EDS-only scan group: nothing to mark, and a default variant to ask for is not there either."""
    from backend.api.routes.indexing import _mark_pattern_source
    path = tmp_path / "eds_scan.h5oina"
    with h5py.File(two, "r") as src, h5py.File(path, "w") as dst:
        for key in ("Manufacturer", "Format Version"):
            src.copy(key, dst)
        src.copy("1", dst)
        del dst["1/EBSD/Data/Processed Patterns"]
        del dst["1/EBSD/Data/Unprocessed Patterns"]
    _mark_pattern_source(str(path), None, False)


def test_the_viewers_edits_are_written_only_into_the_dataset_they_belong_to(export, monkeypatch):
    """Dirty viewer on measurement 2, result on the default: the edits are measurement 2's, so they are
    not written into the result's dataset. Same variant on both sides: they are."""
    import backend.api.routes.indexing as ix
    written = []
    monkeypatch.setattr(ix, "_overwrite_patterns_with_processed",
                        lambda out, sig, variant: written.append(variant) or True)
    from backend.api.routes import ebsd_viewer as ev
    monkeypatch.setattr(ev, "_get_active_signal", lambda *a, **k: object())
    monkeypatch.setattr(ix, "_get_active_signal", lambda *a, **k: object(), raising=False)
    export(None, V2, dirty=True)
    assert written == []
    export(V2.to_dict(), V2, dirty=True)
    assert written == [V2]


def test_no_choice_on_one_side_and_the_files_default_on_the_other_are_the_same_variant(export, monkeypatch):
    """The viewer loaded without a choice, the result stamped with the file's default spelled out:
    the same dataset, so the edits are written."""
    import backend.api.routes.indexing as ix
    written = []
    monkeypatch.setattr(ix, "_overwrite_patterns_with_processed",
                        lambda out, sig, variant: written.append(variant) or True)
    from backend.api.routes import ebsd_viewer as ev
    monkeypatch.setattr(ev, "_get_active_signal", lambda *a, **k: object())
    export(V1.to_dict(), None, dirty=True)
    assert len(written) == 1
