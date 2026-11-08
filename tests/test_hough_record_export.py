"""The record of which reflector list each phase was indexed with travels with the result.

``result.metadata["hough_reflectors"]`` is written to ``/Indexing/Citations`` by both
"Save result" branches, read back on re-import, and carried over by the sites that build
a fresh result from per-phase sub-results (like the citation steps).
"""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hough_reflectors as hr  # noqa: E402
from indexing_controller import (  # noqa: E402
    ComparisonConfig, IndexingMethod, run_per_phase_indexing)
from tests.citations.test_citations_export import (  # noqa: E402
    _fake_active_result, _patch_interactive_export_seams)
from tests.test_per_phase_indexing import (  # noqa: E402
    _make_stub_pmr, _stub_phase_config, _stub_signal)

RECORD = {
    "Al": {"mode": "default", "rule": {"min_d": 1.0, "f_threshold": 0.1, "max_rows": 70},
           "source": "default construction with min d 1 A and |F| above 0.1 of the strongest "
                     "reflection",
           "families_used": ["{111}", "{200}", "{220}", "{311}"], "n_rows": 64,
           "predicted_library_bytes": None, "budget_bytes": 4 << 30, "spec": None},
    "Al13Fe4": {"mode": "auto", "rule": {"min_d": 1.0, "f_threshold": 0.1, "max_rows": 60},
                "source": "the 60 strongest rows of the default construction ...",
                "families_used": ["{200}", "{110}"], "n_rows": 60,
                "predicted_library_bytes": 2 << 30, "budget_bytes": 4 << 30,
                "spec": {"mode": "auto", "rule": {"min_d": 1.0, "f_threshold": 0.1,
                                                  "max_rows": 60}},
                "full_library_bytes": 8 << 30},
}


@pytest.mark.parametrize("fmt", ["h5", "h5_light"])
def test_the_record_is_exported_and_restored_on_reimport(monkeypatch, tmp_path, fmt):
    from backend.api.services.citations.provenance import PROVENANCE_SCHEMA
    steps = [{"key": "indexing.hough", "params": {"orienta_version": "0.4.8"}}]
    result = _fake_active_result(steps)
    result.metadata["hough_reflectors"] = RECORD
    ix = _patch_interactive_export_seams(monkeypatch, result)
    out = tmp_path / f"record_{fmt}.h5"
    asyncio.run(ix.export_indexing_result(ix.ExportRequest(
        format=fmt, filename=str(out), include_eds=False, include_detector=False)))
    with h5py.File(out, "r") as f:
        written = json.loads(f["Indexing/Citations"].attrs["hough_reflectors"])
    assert written == RECORD

    rid = None
    try:
        resp = asyncio.run(ix.import_h5_result(ix.ImportH5Request(path=str(out))))
        rid = resp["result_id"]
        assert ix._result_registry[rid].metadata["hough_reflectors"] == RECORD
    finally:
        if rid is not None:
            ix._result_registry.pop(rid, None)


def test_a_result_without_a_record_writes_none(monkeypatch, tmp_path):
    result = _fake_active_result([{"key": "indexing.hough", "params": {}}])
    ix = _patch_interactive_export_seams(monkeypatch, result)
    out = tmp_path / "none.h5"
    asyncio.run(ix.export_indexing_result(ix.ExportRequest(
        format="h5_light", filename=str(out), include_eds=False, include_detector=False)))
    with h5py.File(out, "r") as f:
        assert "hough_reflectors" not in f["Indexing/Citations"].attrs


def test_a_malformed_record_in_a_file_is_ignored(tmp_path):
    from backend.api.routes import indexing as ix
    p = tmp_path / "bad.h5"
    with h5py.File(p, "w") as f:
        f.create_group("Indexing/Citations").attrs["hough_reflectors"] = json.dumps(["x"])
    assert ix._read_hough_reflectors(p) is None
    with h5py.File(p, "w") as f:
        f.create_group("Indexing/Citations").attrs["hough_reflectors"] = json.dumps({"Al": 3})
    assert ix._read_hough_reflectors(p) is None


def test_merge_build_records_unions_the_phases():
    class R:
        def __init__(self, md):
            self.metadata = md
    target = R({})
    hr.merge_build_records(target, [R({"hough_reflectors": {"Al": RECORD["Al"]}}),
                                    R({"hough_reflectors": {"Al13Fe4": RECORD["Al13Fe4"]}}),
                                    R({}), None])
    assert target.metadata["hough_reflectors"] == RECORD
    empty = R({})
    hr.merge_build_records(empty, [R({})])
    assert "hough_reflectors" not in empty.metadata


def test_the_per_phase_merge_carries_the_record():
    """``run_per_phase_indexing`` (EDS-routed per-phase runs) builds a fresh result from
    the per-phase sub-results; their records must not be lost there."""
    sig = _stub_signal()
    pc_a, pc_b = _stub_phase_config("A"), _stub_phase_config("B")
    mask_a = np.zeros((4, 4), dtype=bool); mask_a[0, :] = True
    mask_b = np.zeros((4, 4), dtype=bool); mask_b[3, :] = True
    consensus = np.full((4, 4), -1, dtype=np.int32)
    consensus[0, :], consensus[3, :] = 0, 1
    fake_a = _make_stub_pmr("A", 4, 4, mask_a)
    fake_b = _make_stub_pmr("B", 4, 4, mask_b)
    fake_a.indexing_result.metadata["hough_reflectors"] = {"A": RECORD["Al"]}
    fake_b.indexing_result.metadata["hough_reflectors"] = {"B": RECORD["Al13Fe4"]}

    def side_effect(*args, **kwargs):
        return fake_a if kwargs["phase_config"].name == "A" else fake_b

    with patch("indexing_controller.run_single_phase_method", side_effect=side_effect), \
         patch("indexing_controller.build_consensus_xmap", return_value=MagicMock()):
        result = run_per_phase_indexing(
            signal=sig, detector=None, method=IndexingMethod.HOUGH,
            config=ComparisonConfig(phases=[pc_a, pc_b], methods=[IndexingMethod.HOUGH]),
            phase_configs=[pc_a, pc_b], masks_per_phase=[mask_a, mask_b],
            consensus_map=consensus)
    assert result.metadata["hough_reflectors"] == {"A": RECORD["Al"], "B": RECORD["Al13Fe4"]}
