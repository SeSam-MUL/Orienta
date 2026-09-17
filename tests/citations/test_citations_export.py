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
from pathlib import Path

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


# ---------------------------------------------------------------------------
# The user-facing exporters: POST /api/indexing/export (rich + light), and
# the batch auto-export. These duplicate the /Indexing writer inline in
# routes/indexing.py rather than calling result_exporter's exporters, and
# were the actual gap the round-trip test above did not cover: it proved the
# module functions round-trip, not that the routes a user actually clicks
# call them. Fixed by reusing _write_citations at each site, fed from
# get_steps(<the real result object at that site>).
# ---------------------------------------------------------------------------

def _fake_active_result(steps):
    """A minimal but real CrystalMap-backed result, shaped like what
    ``_get_result()`` returns to ``export_indexing_result`` — same
    construction as ``tests/test_export_roi.py``'s ``_fake_roi_result``,
    trimmed to a full (non-ROI) 2x2 grid since ROI placement isn't what
    this is checking."""
    from types import SimpleNamespace

    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation

    from backend.api.services.citations.provenance import PROVENANCE_SCHEMA

    n_rows, n_cols = 2, 2
    n_pix = n_rows * n_cols
    rng = np.random.default_rng(7)
    eulers = rng.uniform(0, 2 * np.pi, (n_pix, 3))
    phase_list = PhaseList(
        phases=[Phase(name="Al", space_group=225, point_group="m-3m")]
    )
    xx, yy = np.meshgrid(
        np.arange(n_cols, dtype=float), np.arange(n_rows, dtype=float)
    )
    xmap = CrystalMap(
        rotations=Rotation.from_euler(eulers),
        phase_id=np.zeros(n_pix, dtype=int),
        x=xx.ravel(), y=yy.ravel(),
        phase_list=phase_list,
    )
    return SimpleNamespace(
        xmap=xmap,
        original_shape=(n_rows, n_cols),
        selection_mask=np.ones((n_rows, n_cols), dtype=bool),
        confidence_scores=np.full(n_pix, 0.5, dtype=np.float32),
        method=SimpleNamespace(value="hough"),
        metadata={"provenance": {"schema": PROVENANCE_SCHEMA, "steps": steps}},
    )


def _patch_interactive_export_seams(monkeypatch, result):
    """Same route seams ``tests/test_export_roi.py`` patches, duplicated
    locally so this file has no cross-test-module import."""
    import backend.api.routes.ebsd_viewer as ev
    import backend.api.routes.indexing as ix
    import backend.api.services.reference_frame_state as rfs

    monkeypatch.setattr(ix, "_get_result", lambda *_a, **_k: result)
    monkeypatch.setattr(ix, "_resolve_source_vendor", lambda *_a, **_k: "unknown")
    monkeypatch.setattr(ix, "_resolve_step_size_um", lambda *_a, **_k: 0.25)
    monkeypatch.setattr(ev, "_ebsd_file_path", None, raising=False)
    monkeypatch.setattr(rfs, "get_frame", lambda *_a, **_k: {"apply_to_export": False})
    return ix


@pytest.mark.parametrize("fmt", ["h5", "h5_light"])
def test_interactive_export_writes_citations(monkeypatch, tmp_path, fmt):
    """POST /api/indexing/export, both branches (rich :9635, light :9929 as
    of the finding this fixes) — each must write /Indexing/Citations from
    the real active result's recorded steps."""
    steps = [{"key": "indexing.hough", "params": {"orienta_version": "0.3.0"}}]
    result = _fake_active_result(steps)
    ix = _patch_interactive_export_seams(monkeypatch, result)
    out = tmp_path / f"interactive_{fmt}.h5"

    asyncio.run(ix.export_indexing_result(ix.ExportRequest(
        format=fmt, filename=str(out), include_eds=False, include_detector=False,
    )))

    assert out.is_file()
    with h5py.File(out, "r") as f:
        assert "Indexing/Citations" in f, (
            f"format={fmt}: /Indexing/Citations missing from the interactive "
            "export — the inline writer at this branch never called "
            "_write_citations"
        )
        written_steps = json.loads(f["Indexing/Citations"].attrs["steps"])
    assert written_steps == steps


@pytest.mark.integration
def test_batch_auto_export_writes_citations(h5oina_path, al_cif_path, tmp_path):
    """The batch path (``start_batch_indexing``'s auto-export block, :10653
    as of the finding this fixes) — real Hough run on a small ROI of the
    real SampleB file, real background-thread batch, real auto-export.

    Not mocked: this is the smallest real run that reaches the code path
    under test. Mirrors the small-ROI real-Hough pattern already used by
    ``tests/citations/conftest.py::indexed_result_hough`` and by
    ``tests/test_batch_route_spherical_reachable.py`` for the batch route
    itself (same polling idiom, same generous timeout because the batch
    loads the full ~521 MB source file before it ever gets to the small
    ROI).
    """
    pytest.importorskip("pyebsdindex")
    import time

    from fastapi.testclient import TestClient

    from backend.api.main import app
    from backend.api.routes import ebsd_viewer
    from backend.api.services import crop_window as cw
    from backend.api.services import h5_session

    def _wipe():
        """Same reset ``fresh_backend`` autouse fixtures elsewhere in this
        suite apply on both sides of a test (e.g.
        ``tests/test_crop_provenance.py``, ``tests/test_batch_route_
        spherical_reachable.py``). This test is the ONE in this file that
        drives a real ``load_ebsd_file`` on a real 521 MB source through the
        batch route, so it is the one that must not leave that signal (and
        its live h5py handle) sitting in the module-level registries for a
        later, unrelated test to inherit — that leak is exactly what made
        ``tests/test_export_roi.py::test_import_h5_restores_roi_result``
        fail with ``OSError: file is already open for read-only`` when this
        test ran first in the same session (first found by running this
        file back-to-back with that one)."""
        ebsd_viewer._raw_signals.clear()
        ebsd_viewer._positions.clear()
        ebsd_viewer._dirty_datasets.clear()
        ebsd_viewer._signal_masks.clear()
        ebsd_viewer._overview_cache.clear()
        ebsd_viewer._registry_by_file.clear()
        ebsd_viewer._active_dataset = ""
        ebsd_viewer._ebsd_signal = None
        ebsd_viewer._ebsd_file_path = None
        cw.clear_all()
        if h5_session.is_open():
            h5_session.close_file()

    _wipe()
    try:
        client = TestClient(app)
        dataset = {
            "file_path": str(h5oina_path),
            "method": "hough",
            "cif_paths": [str(al_cif_path)],
            "selection_mode": "region",
            # Same 12x12 window as conftest.indexed_result_hough — inside the
            # single Al grain, small enough to stay fast.
            "row_start": 18, "row_end": 30, "col_start": 2, "col_end": 14,
        }
        resp = client.post("/api/indexing/batch/start", json={
            "datasets": [dataset],
            "auto_export": True,
            "export_dir": str(tmp_path),
            "cleanup_after_export": False,
        })
        assert resp.status_code == 200, resp.text

        deadline = time.monotonic() + 300.0
        status = {}
        while time.monotonic() < deadline:
            status = client.get("/api/indexing/batch/status").json()
            if not status.get("running"):
                break
            time.sleep(0.2)
        else:
            pytest.fail(f"batch still running after 300s: {status}")

        results = status.get("results", [])
        assert results, status
        entry = results[0]
        assert entry["error"] is None, entry
        export_path = entry.get("export_path")
        assert export_path and Path(export_path).is_file(), entry

        with h5py.File(export_path, "r") as f:
            assert "Indexing/Citations" in f, (
                "batch auto-export never wrote /Indexing/Citations"
            )
            steps = json.loads(f["Indexing/Citations"].attrs["steps"])
        assert any(s["key"] == "indexing.hough" for s in steps), steps
    finally:
        _wipe()
