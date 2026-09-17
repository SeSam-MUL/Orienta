"""``/Indexing/Citations`` in an exported file, and reading it back on import.

``_write_citations`` mirrors ``_write_scan_provenance`` deliberately: ``None``
and an empty list write nothing, so a file exported before this feature
existed is byte-identical to one exported after, as long as no steps are
passed. ``_read_citations`` (routes/indexing.py) reads the trail back on
re-import so a collaborator who only has the exported ``.h5`` still gets the
right bibliography.
"""
from __future__ import annotations

import asyncio
import json

import h5py
import numpy as np
import pytest

from backend.api.services.result_exporter import _write_citations


def test_none_writes_nothing(tmp_path):
    """Byte-identical to a file exported before this existed."""
    path = tmp_path / "a.h5"
    with h5py.File(path, "w") as f:
        _write_citations(f.create_group("Indexing"), None)
    with h5py.File(path, "r") as f:
        assert "Citations" not in f["Indexing"]


def test_empty_list_writes_nothing(tmp_path):
    path = tmp_path / "b.h5"
    with h5py.File(path, "w") as f:
        _write_citations(f.create_group("Indexing"), [])
    with h5py.File(path, "r") as f:
        assert "Citations" not in f["Indexing"]


def test_steps_and_entries_round_trip(tmp_path):
    steps = [{"key": "indexing.hough", "params": {"orienta_version": "0.3.0"}}]
    path = tmp_path / "c.h5"
    with h5py.File(path, "w") as f:
        _write_citations(f.create_group("Indexing"), steps)
    with h5py.File(path, "r") as f:
        grp = f["Indexing/Citations"]
        assert json.loads(grp.attrs["steps"]) == steps
        entries = json.loads(grp.attrs["entries"])
        assert any(e["id"] == "orienta" for e in entries)
        assert grp.attrs["schema"] == 1


def test_unicode_survives(tmp_path):
    steps = [{"key": "indexing.hough", "params": {"note": "Weißensteiner"}}]
    path = tmp_path / "d.h5"
    with h5py.File(path, "w") as f:
        _write_citations(f.create_group("Indexing"), steps)
    with h5py.File(path, "r") as f:
        back = json.loads(f["Indexing/Citations"].attrs["steps"])
    assert back[0]["params"]["note"] == "Weißensteiner"


# ---------------------------------------------------------------------------
# Round trip through the real export/import pipeline
# ---------------------------------------------------------------------------

def test_export_and_reimport_round_trips_the_citation_list(tmp_path):
    """Real ``export_result_h5_light`` -> real ``import-h5`` -> same citations.

    Builds a genuine batch checkpoint via ``CheckpointWriter`` (the same class
    the real batch pipeline uses), exports it through
    ``export_result_h5_light`` with a citation trail attached, re-imports the
    resulting file through the real ``/api/indexing/import-h5`` handler, and
    asserts ``GET /api/citations/result/{id}`` returns the identical citation
    list for the freshly-registered "before" result and the re-imported
    "after" one.
    """
    from backend.api.routes import citations as citations_routes
    from backend.api.routes import indexing as indexing_routes
    from backend.api.services.checkpoint_writer import CheckpointWriter
    from backend.api.services.citations.provenance import PROVENANCE_SCHEMA
    from backend.api.services.result_exporter import export_result_h5_light

    n_rows, n_cols = 4, 3
    grid_shape = (n_rows, n_cols)

    # A minimal stand-in source file. export_result_h5_light reads the
    # vendor + quality fields from it defensively (missing groups -> skip),
    # so an empty file is a legitimate "no extra metadata available" source.
    source_h5 = tmp_path / "source.h5"
    with h5py.File(source_h5, "w"):
        pass

    writer = CheckpointWriter(str(source_h5))
    writer.init_metadata(
        grid_shape, batch_id="citation-roundtrip", method="hough",
        step_size_um=0.5,
    )
    ci_map = np.full(grid_shape, 0.8, dtype=np.float32)
    orientation_map = np.zeros((*grid_shape, 3), dtype=np.float32)
    writer.write_phase_result(
        "Al", ci_map, orientation_map,
        {
            "phase_file": "Al.cif", "ci_mean": 0.8, "ci_median": 0.8,
            "duration_sec": 1.0, "space_group": 225, "point_group": "m-3m",
        },
    )
    writer.compute_auto_assignment(confidence_threshold=0.3)

    steps = [{"key": "indexing.hough", "params": {"orienta_version": "0.3.0"}}]

    class _StubResult:
        """Enough of an IndexingResult for the citations endpoint."""

        def __init__(self, steps):
            self.metadata = {
                "provenance": {"schema": PROVENANCE_SCHEMA, "steps": steps}
            }

    before_id = "citation_roundtrip_before"
    indexing_routes._result_registry[before_id] = _StubResult(steps)
    try:
        before = citations_routes.citations_for_result(before_id)
    finally:
        indexing_routes._result_registry.pop(before_id, None)

    out_path = export_result_h5_light(
        source_h5_path=str(source_h5),
        checkpoint_path=writer.checkpoint_path,
        output_dir=str(tmp_path),
        include_eds=False,
        citation_steps=steps,
    )

    with h5py.File(out_path, "r") as f:
        assert "Indexing/Citations" in f, (
            "export_result_h5_light must have written /Indexing/Citations "
            "when citation_steps was passed"
        )

    result_id = None
    try:
        resp = asyncio.run(
            indexing_routes.import_h5_result(
                indexing_routes.ImportH5Request(path=str(out_path))
            )
        )
        result_id = resp["result_id"]
        after = citations_routes.citations_for_result(result_id)
    finally:
        if result_id is not None:
            indexing_routes._result_registry.pop(result_id, None)

    assert after["steps"] == before["steps"] == steps
    assert after["bibtex"] == before["bibtex"]
    assert after["methods"] == before["methods"]
    assert after["plain"] == before["plain"]
