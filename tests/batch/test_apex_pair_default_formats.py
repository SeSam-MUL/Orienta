"""A hexagonal APEX pair through the real batch with its default settings, end to end.

The real ``BatchManager`` loads a fake APEX pair (``tests/apex_helpers.py``: a metadata ``.h5`` and the
``.up1`` file that names its measurement) with the real loader, writes the checkpoint, takes the
auto-assignment and exports with the default formats (light ``.h5``, ``.ang``, ``.ctf``) and the
default ``include_eds`` (on). Only the indexer is replaced: a stand-in hands back the confidence
index and the symmetry of each phase.

Three phases of different symmetry; the one in the middle (by name) wins no point:

    Al  m-3m   space group 225   TSL # Symmetry 43   wins points 0-6
    Mg  6/mmm  space group 194   TSL # Symmetry 62   wins nothing
    Sn  4/mmm  space group 141   TSL # Symmetry 42   wins points 7-14

The codes are the TSL codes of the Laue groups (OIM's table), by hand. orix leaves Mg out of the map and
numbers the file Al = 1, Sn = 2, so a table numbered by the full list would state Sn as 62.
"""
from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pytest

from backend.api.services import batch_manager as M
from tests.apex_helpers import GUID_A, GUID_A_BYTES, make_apex_h5, make_up, measurement_spec
from tests.test_batch_hex_run import _Indexer

N_ROWS, N_COLS = 3, 5
N = N_ROWS * N_COLS

#: name -> (space group, point group, lattice, the points it wins)
PHASES = {
    "Al": (225, "m-3m", [4.05, 4.05, 4.05, 90.0, 90.0, 90.0], range(0, 7)),
    "Mg": (194, "6/mmm", [3.21, 3.21, 5.21, 90.0, 90.0, 120.0], range(0, 0)),
    "Sn": (141, "4/mmm", [5.83, 5.83, 3.18, 90.0, 90.0, 90.0], range(7, 15)),
}
HAND_SYMMETRY = {"Al": "43", "Sn": "42"}                     # by name, Mg is not in the file


class _PhaseIndexer(_Indexer):
    """The stand-in indexer of the hexagonal tests, with the symmetry and the winning points of each phase."""

    def __call__(self, signal, file_path, phase_name, phase_path, method, grid_shape, job_config=None,
                 progress_callback=None, scan_geometry=None):
        super().__call__(signal, file_path, phase_name, phase_path, method, grid_shape,
                         job_config=job_config, progress_callback=progress_callback,
                         scan_geometry=scan_geometry)
        sg, pg, lattice, wins = PHASES[phase_name]
        ci = np.full(N, 0.1, dtype=np.float32)
        ci[list(wins)] = 0.9
        euler = np.zeros((N_ROWS, N_COLS, 3), dtype=np.float32)
        euler[..., 0] = np.arange(N).reshape(N_ROWS, N_COLS) * 0.05
        return (ci.reshape(N_ROWS, N_COLS), euler,
                {"ci_mean": float(ci.mean()), "ci_median": float(np.median(ci)), "duration_sec": 0.1,
                 "space_group": sg, "point_group": pg, "lattice_constants": lattice})


@pytest.fixture
def pair(tmp_path):
    folder = tmp_path / "pair"
    folder.mkdir()
    make_apex_h5(folder / "session.h5", [measurement_spec(
        id=GUID_A, name="OIM Map 1", n_rows=N_ROWS, n_cols=N_COLS, step_x=0.5, pc=(0.5, 0.4, 0.6))])
    up = make_up(folder / "Scan.up1", n=N, nx=N_COLS, ny=45, guid_bytes=GUID_A_BYTES)
    return up


@pytest.fixture
def run(monkeypatch, tmp_path):
    """``run(up, **config)`` -> (manager, batch id, output folder); only the default settings that the page
    sends are left in the config unless given."""
    monkeypatch.setattr(M, "_batch_names", {})
    monkeypatch.setattr(M, "_register_signal_calibration", lambda signal, file_path: None)
    monkeypatch.setattr(M, "run_single_indexing_job", _PhaseIndexer())

    def go(up, **config):
        mgr = M.BatchManager(db_path=str(tmp_path / "jobs.db"))
        out = tmp_path / "out"
        base = {"auto_export": True, "export_dir": str(out)}
        base.update(config)
        cifs = []
        for name in PHASES:
            cif = tmp_path / f"{name}.cif"
            cif.write_text(f"data_{name}\n", encoding="utf-8")
            cifs.append({"name": name, "path": str(cif), "method": "hough"})
        batch_id = mgr.queue.create_batch([str(up)], cifs, base)
        mgr._run_batch_sync_impl(batch_id)
        return mgr, batch_id, out

    return go


def _symmetry_by_name(path):
    blocks, cur = [], None
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if not ln.startswith("#"):
            break
        if re.match(r"^#\s*Phase\s+\d+\s*$", ln):
            cur = {}
            blocks.append(cur)
        m = re.match(r"^#\s*(MaterialName|Symmetry)\s+(.*)$", ln)
        if m and cur is not None:
            cur[m.group(1)] = m.group(2).strip()
    return {b["MaterialName"]: b["Symmetry"] for b in blocks}


def _record(mgr, batch_id, up):
    return mgr.exports[batch_id][str(up)]


# ------------------------------------------------------------------------------ the default settings

def test_default_formats_of_an_apex_pair_state_the_symmetry_of_every_written_phase(run, pair):
    mgr, batch_id, out = run(pair)
    assert sorted(j["status"] for j in mgr.get_jobs(batch_id)) == ["done"] * 3
    record = _record(mgr, batch_id, pair)
    outputs = record["outputs"]
    assert record["grid_kind"] == "hex"
    assert outputs["ang"] and outputs["ctf"] and outputs["h5_light"], record
    assert outputs["h5_rich"] is None and outputs["ang_square"] is None      # not asked for

    assert _symmetry_by_name(outputs["ang"]) == HAND_SYMMETRY

    # the .ctf lists the two phases that are in the map, numbered 1 and 2, with the Laue class of the
    # Oxford table (m-3m 11, 4/mmm 5): Al first, Sn second
    ctf = Path(outputs["ctf"]).read_text(encoding="utf-8").splitlines()
    i = next(k for k, ln in enumerate(ctf) if ln.startswith("Phases\t"))
    assert ctf[i].split("\t")[1] == "2"
    assert [ctf[i + 1 + k].split("\t")[2:4] for k in range(2)] == [["Al", "11"], ["Sn", "5"]]
    numbers = [int(ln.split("\t")[0]) for ln in ctf[i + 4:] if ln.strip()]
    assert len(numbers) == N and set(numbers) == {1, 2}


def test_default_formats_of_an_apex_pair_write_the_light_h5_without_eds_and_say_so(run, pair):
    """include_eds is on by default; a .up1 file holds no EDS. The light .h5 is written without it
    and the record says so with a code (it used to end in "file signature not found")."""
    mgr, batch_id, out = run(pair)
    record = _record(mgr, batch_id, pair)
    assert record["errors"] is None and record["error_items"] is None, record["errors"]
    light = record["outputs"]["h5_light"]
    with h5py.File(light, "r") as f:
        assert "EDS" not in f
        assert bool(f["Documentation"].attrs["eds_included"]) is False
        assert f["Indexing"].attrs["source_vendor"] in ("edax", b"edax")      # a .up1 is an EDAX file
        # the light file lists every phase that was tested (by name) and numbers its points by that
        # table: Al points 0-6 are phase 1, Sn points 7-14 are phase 3 (Mg is 2 and wins nowhere)
        names = [f[f"Indexing/Phases/{k}"].attrs["name"] for k in ("1", "2", "3")]
        assert [n.decode() if isinstance(n, bytes) else n for n in names] == ["Al", "Mg", "Sn"]
        assert f["Indexing/Assignment/phase_id"][()].ravel().tolist() == [1] * 7 + [3] * 8
    items = [(n["code"], n["params"]) for n in record["note_items"]]
    assert ("edsNotCopied", {}) in items, items
    assert any("EDS" in n for n in record["notes"])


def test_the_batch_ang_of_an_apex_pair_states_the_edax_frame(run, pair):
    mgr, batch_id, out = run(pair)
    ang = _record(mgr, batch_id, pair)["outputs"]["ang"]
    frame = " ".join(ln for ln in Path(ang).read_text(encoding="utf-8").splitlines()
                     if ln.startswith("# ORIENTA_EULER_FRAME"))
    assert "EDAX system, whose own frame this is" in frame


def test_the_edax_off_switch_writes_no_note(run, pair):
    """include_eds off: nothing was asked for, nothing is said."""
    mgr, batch_id, out = run(pair, include_eds=False)
    record = _record(mgr, batch_id, pair)
    assert record["outputs"]["h5_light"]
    assert "edsNotCopied" not in [n["code"] for n in record["note_items"]]


# ------------------------------------------------------------------------------ the rich .h5

def test_a_rich_h5_of_an_apex_pair_is_refused_with_a_code_and_the_other_files_are_written(run, pair):
    mgr, batch_id, out = run(pair, export_formats=["h5_rich", "h5_light", "ang", "ctf"])
    record = _record(mgr, batch_id, pair)
    outputs = record["outputs"]
    assert outputs["h5_rich"] is None
    assert outputs["h5_light"] and outputs["ang"] and outputs["ctf"], record
    item = record["error_items"]["h5_rich"]
    assert item["code"] == "richNeedsHdf5Source"
    assert "another file format" in item["text"] and "light" in item["text"]
    assert set(record["errors"]) == {"h5_rich"}                         # it stays an error
    # nothing was copied before the refusal: no .up1 under a .h5 name, no part file
    assert not [p for p in out.iterdir() if p.name.startswith("result_Scan") and p.suffix == ".h5"
                and p.name != "result_Scan_light.h5"]
    assert not list(out.glob("*.part"))


@pytest.mark.parametrize("name", ["Scan.up1", "Scan.UP2", "Scan.ebsp", "Scan.dat"])
def test_the_texts_for_a_pattern_file_that_is_not_hdf5_name_no_file_type(tmp_path, name):
    """Any pattern file that is not HDF5 gets the sentence, not only .up1/.up2: it must not name a
    file type that is not the user's."""
    from backend.api.services.result_exporter import RichNeedsHdf5Source, export_result_h5
    source = tmp_path / name
    source.write_bytes(b"not an hdf5 file")
    with pytest.raises(RichNeedsHdf5Source) as raised:
        export_result_h5(str(source), str(tmp_path / "c.h5"), str(tmp_path / "o"))
    notes = M._eds_notes(str(source), ["h5_light"], True)
    assert len(notes) == 1 and notes[0].code == "edsNotCopied"
    for text in (str(raised.value), str(notes[0])):
        assert "another file format" in text
        assert ".up1" not in text and ".up2" not in text and ".ebsp" not in text


# ------------------------------------------------------------------------------ the checkpoint

def test_a_checkpoint_that_refuses_with_a_type_error_fails_that_file_and_not_the_thread(
        run, pair, monkeypatch):
    """A scalar ``grid_shape`` makes ``tuple(int(v) for v in grid_shape)`` raise ``TypeError`` ("'int' object
    is not iterable"); only ValueError and OSError were caught, so the exception ended the batch thread with
    the status still "running"."""
    from backend.api.services.checkpoint_writer import CheckpointWriter

    def scalar_grid(self, grid_shape, *a, **k):
        tuple(int(v) for v in 15)                    # what a scalar grid_shape does in init_metadata

    monkeypatch.setattr(CheckpointWriter, "init_metadata", scalar_grid)
    mgr, batch_id, out = run(pair)                   # must return: the thread is not ended by the exception
    jobs = mgr.get_jobs(batch_id)
    assert {j["status"] for j in jobs} == {"failed"}
    assert all("not iterable" in j["error_msg"] for j in jobs)
    assert all(j["error_code"] == "checkpointUnreadable" for j in jobs)


# ------------------------------------------------------------------------------ the cleanup

def test_a_failed_cleanup_is_in_the_record_with_a_code(run, pair, monkeypatch):
    """The export carries the map before the cleanup: the record says so."""
    from backend.api.services.checkpoint_writer import CheckpointWriter

    def broken(self, **kw):
        raise RuntimeError("cleanup went wrong")

    monkeypatch.setattr(CheckpointWriter, "apply_cleanup", broken)
    mgr, batch_id, out = run(pair, postprocessing={"ci_threshold": 0.3})
    record = _record(mgr, batch_id, pair)
    assert record["outputs"]["ang"]                                      # still exported
    items = {n["code"]: n for n in record["note_items"]}
    assert items["cleanupFailed"]["params"] == {"detail": "cleanup went wrong"}
    assert "cleanup went wrong" in items["cleanupFailed"]["text"]


def test_a_failed_auto_assignment_is_in_the_record_with_a_code(run, pair, monkeypatch):
    from backend.api.services.checkpoint_writer import CheckpointWriter
    real = CheckpointWriter.compute_auto_assignment
    calls = []

    def broken(self, *a, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("assignment went wrong")
        return real(self, *a, **kw)

    monkeypatch.setattr(CheckpointWriter, "compute_auto_assignment", broken)
    mgr, batch_id, out = run(pair)
    record = _record(mgr, batch_id, pair)
    items = {n["code"]: n for n in record["note_items"]}
    assert items["autoAssignmentFailed"]["params"] == {"detail": "assignment went wrong"}


def test_a_clean_run_has_no_cleanup_note(run, pair):
    mgr, batch_id, out = run(pair, postprocessing={"ci_threshold": 0.3})
    codes = [n["code"] for n in _record(mgr, batch_id, pair)["note_items"]]
    assert "cleanupFailed" not in codes and "autoAssignmentFailed" not in codes
