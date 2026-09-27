"""The HTTP surface of the EDS export and the presets.

Spec section 8 is frozen and the frontend builds against it, so these pin the
shapes rather than the arithmetic — the numbers themselves are checked in
``test_eds_export.py`` against a hand-countable map.

What matters here is the refusals. An export that writes into an existing
folder, or writes German decimals into a comma-separated file, or exports a
phase map that belongs to a different scan than the one that is open, all
produce a file that looks right and is not. Each of those is a status code
below and none of them is a warning.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.main import app
from backend.api.routes import eds as eds_routes
from backend.api.routes import eds_export as export_routes
from backend.api.services import eds_presets as presets_svc
from backend.api.services.cif_phase_library import CifPhaseEntry
from backend.api.services.phase_map_store import PhaseMapState

N_ROWS = N_COLS = 8
N_PX = N_ROWS * N_COLS
STEP = 0.25


def _entry(name, comp):
    return CifPhaseEntry(key=name, cif_filename=name, formula=name,
                         space_group="Fm-3m", space_group_number=225,
                         crystal_system="cubic", composition=comp,
                         elements=sorted(comp))


class _FakeStore:
    """Stands in for the process-wide store.

    A fake rather than the real one because ``set_classification`` autosaves a
    sidecar next to the file path and mutates process state that outlives the
    test — a test suite that leaves a phase map in the store makes the next
    test's failure depend on the order it ran in.
    """

    def __init__(self, state):
        self._state = state

    def get_state(self):
        return self._state


@pytest.fixture
def loaded(monkeypatch):
    """An 8x8 Al matrix with a 2x2 Si particle, and a store holding its map."""
    region = np.zeros((N_ROWS, N_COLS), dtype=np.int32)
    region[3:5, 3:5] = 1
    flat = region.ravel()
    al = np.full(N_PX, 94.0)
    si = np.full(N_PX, 6.0)
    al[flat == 1], si[flat == 1] = 35.0, 65.0
    at_maps = {"Al": al, "Si": si}

    state = PhaseMapState(
        phase_grid=np.where(region > 0, 1, 0).astype(np.int32),
        score_grid=np.full((N_ROWS, N_COLS), 0.75, dtype=np.float32),
        phase_entries=[_entry("Al.cif", {"Al": 100.0}),
                       _entry("Si.cif", {"Si": 100.0})],
        n_rows=N_ROWS, n_cols=N_COLS, tolerance=15.0, min_score=0.3,
        region_grid=region, region_phase=[0, 1],
    )

    monkeypatch.setattr(
        eds_routes, "_build_at_pct_maps_for_loaded_file",
        lambda: (at_maps, N_ROWS, N_COLS, "scanB.h5oina"))
    monkeypatch.setattr(eds_routes, "_eds_step_um", lambda: (STEP, STEP))
    monkeypatch.setattr(export_routes, "get_phase_map_store",
                        lambda: _FakeStore(state))
    return TestClient(app), state, at_maps


@pytest.fixture
def no_map(monkeypatch):
    monkeypatch.setattr(export_routes, "get_phase_map_store",
                        lambda: _FakeStore(None))
    return TestClient(app)


def _export(client, tmp_path, **body):
    payload = {"dest_dir": str(tmp_path)}
    payload.update(body)
    return client.post("/api/eds/export", json=payload)


# ---------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------

def test_preview_answers_the_question_the_dialog_asks(loaded):
    client, _state, _maps = loaded
    r = client.get("/api/eds/export/preview")
    assert r.status_code == 200
    b = r.json()
    assert b["ok"] is True
    assert b["n_total_px"] == N_PX
    assert b["n_classified_px"] == N_PX
    assert b["n_regions"] == 2
    assert b["n_particles"] == 2        # the matrix and the 2x2 particle
    assert b["n_phases"] == 2
    assert b["step_x_um"] == STEP
    assert b["area_available"] is True
    assert b["connectivity"] == 8
    assert b["particle_basis"] == "region"
    assert b["warnings"] == []


def test_preview_reports_the_source_size_so_the_dialog_can_warn_about_hashing(
        loaded, tmp_path, monkeypatch):
    client, _state, _maps = loaded
    src = tmp_path / "scanB.h5oina"
    src.write_bytes(b"0123456789")
    maps = eds_routes._build_at_pct_maps_for_loaded_file()
    monkeypatch.setattr(
        eds_routes, "_build_at_pct_maps_for_loaded_file",
        lambda: (maps[0], maps[1], maps[2], str(src)))
    b = client.get("/api/eds/export/preview").json()
    assert b["source_bytes"] == 10


def test_preview_says_no_step_size_rather_than_inventing_one(loaded, monkeypatch):
    client, _state, _maps = loaded
    monkeypatch.setattr(eds_routes, "_eds_step_um", lambda: (None, None))
    b = client.get("/api/eds/export/preview").json()
    assert b["area_available"] is False
    assert b["step_x_um"] is None
    assert [w["code"] for w in b["warnings"]] == ["no_step_size"]


def test_preview_refuses_when_there_is_nothing_classified(no_map):
    r = no_map.get("/api/eds/export/preview")
    assert r.status_code == 400
    assert "classify" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def test_export_writes_the_folder_and_reports_what_it_wrote(loaded, tmp_path):
    client, _state, _maps = loaded
    r = _export(client, tmp_path, hash_source=False)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["ok"] is True
    folder = Path(b["folder"])
    assert folder.is_dir()
    assert folder.name.startswith("scanB_EDS_")

    names = {f["name"] for f in b["files"]}
    # phase_map.png and region_map.png are in the DEFAULT set: the reader who
    # asked for them had already missed a wrong headline number for want of a
    # picture, so an artefact she would have had to go and switch on would
    # have been the same wall with an extra checkbox.
    assert names == {"phases.csv", "regions.csv", "particles.csv",
                     "definitions.csv", "provenance.json", "caption.txt",
                     "summary.xlsx", "phase_map.png", "region_map.png",
                     # the report versions: magnified, with a scale bar
                     "phase_map_figure.png", "region_map_figure.png"}
    # A picture reports its size in pixels, not "rows" of anything.
    by_name = {f["name"]: f for f in b["files"]}
    assert by_name["phase_map.png"]["width"] and by_name["phase_map.png"]["height"]
    assert by_name["phase_map_figure.png"]["width"] >= 8 * by_name["phase_map.png"]["width"]
    assert by_name["phases.csv"]["width"] is None
    # and the run record says how the figure relates to the 1:1 raster
    fig = b["provenance"]["map_figures"]["phase_map_figure.png"]
    assert fig["scale_factor"] >= 8 and fig["width"] == by_name["phase_map_figure.png"]["width"]
    assert fig["source_width"] == by_name["phase_map.png"]["width"]
    for f in b["files"]:
        assert f["bytes"] > 0
        assert (folder / f["name"]).exists()

    # The record travels back with the response as well as sitting in the
    # folder, so the UI can show it without reading the disk it just wrote.
    assert b["provenance"]["counts"]["n_px_total"] == N_PX
    assert b["provenance"]["grid"]["n_rows"] == N_ROWS


def test_the_provenance_in_the_response_is_the_one_in_the_file(loaded, tmp_path):
    client, _state, _maps = loaded
    b = _export(client, tmp_path, hash_source=False).json()
    on_disk = json.loads(
        (Path(b["folder"]) / "provenance.json").read_text(encoding="utf-8"))
    assert on_disk["counts"] == b["provenance"]["counts"]
    assert on_disk["exported_at"] == b["provenance"]["exported_at"]


def test_the_opt_in_artefacts_are_written_when_asked_for(loaded, tmp_path):
    client, _state, _maps = loaded
    r = _export(client, tmp_path, hash_source=False,
                artefacts={"phases": True, "regions": True, "particles": True,
                           "definitions": True, "xlsx": True,
                           "labels": True, "pixels": True})
    b = r.json()
    names = {f["name"]: f for f in b["files"]}
    assert "labels.npz" in names
    assert names["pixels.csv"]["rows"] == N_PX


def test_turning_a_table_off_leaves_it_out(loaded, tmp_path):
    client, _state, _maps = loaded
    b = _export(client, tmp_path, hash_source=False,
                artefacts={"phases": True, "regions": False,
                           "particles": False, "definitions": False,
                           "xlsx": False, "map_png": False}).json()
    assert {f["name"] for f in b["files"]} == {"phases.csv", "provenance.json",
                                               "caption.txt"}


def test_provenance_json_is_never_optional(loaded, tmp_path):
    """It is the artefact the whole consultation converged on. An export
    without it is a screenshot."""
    client, _state, _maps = loaded
    b = _export(client, tmp_path, hash_source=False,
                artefacts={"phases": False, "regions": False,
                           "particles": False, "definitions": False,
                           "xlsx": False, "map_png": False}).json()
    # caption.txt is unconditional for the same reason and by the same
    # argument; the pair is the floor of an export.
    assert {f["name"] for f in b["files"]} == {"provenance.json", "caption.txt"}


def test_min_particle_px_reaches_the_core_and_only_flags(loaded, tmp_path):
    client, _state, _maps = loaded
    b = _export(client, tmp_path, hash_source=False, min_particle_px=100).json()
    particles = [f for f in b["files"] if f["name"] == "particles.csv"][0]
    assert particles["rows"] == 2                 # nothing was deleted
    assert b["provenance"]["particles"]["min_particle_px"] == 100
    assert any(w["code"] == "below_size_limit" for w in b["warnings"])


def test_a_german_locale_combination_that_would_shift_columns_is_refused(
        loaded, tmp_path):
    client, _state, _maps = loaded
    r = _export(client, tmp_path, decimal=",", delimiter=",")
    assert r.status_code == 400
    assert "decimal comma" in r.json()["detail"]


def test_a_workable_german_locale_is_accepted(loaded, tmp_path):
    client, _state, _maps = loaded
    r = _export(client, tmp_path, hash_source=False, decimal=",", delimiter=";")
    assert r.status_code == 200
    text = (Path(r.json()["folder"]) / "regions.csv").read_text(
        encoding="utf-8-sig")
    assert ";" in text.splitlines()[0]
    assert r.json()["provenance"]["locale"] == {"decimal": ",", "delimiter": ";"}


def test_a_missing_destination_is_refused_before_anything_is_computed(
        loaded, tmp_path):
    client, _state, _maps = loaded
    r = _export(client, tmp_path / "nope")
    assert r.status_code == 400
    assert "does not exist" in r.json()["detail"]


def test_exporting_twice_never_overwrites(loaded, tmp_path):
    client, _state, _maps = loaded
    a = _export(client, tmp_path, hash_source=False).json()["folder"]
    b = _export(client, tmp_path, hash_source=False).json()["folder"]
    assert a != b
    assert Path(a).is_dir() and Path(b).is_dir()


def test_on_existing_fail_gives_a_conflict_rather_than_a_second_folder(
        loaded, tmp_path, monkeypatch):
    client, _state, _maps = loaded
    # Pin the clock so both attempts resolve to the same folder name; without
    # this the test would only pass inside one wall-clock minute.
    from backend.api.services import eds_export as svc
    fixed = svc.datetime(2026, 8, 27, 12, 0)

    class _Clock(svc.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr(export_routes, "datetime", _Clock)
    assert _export(client, tmp_path, hash_source=False).status_code == 200
    r = _export(client, tmp_path, hash_source=False, on_existing="fail")
    assert r.status_code == 409


def test_a_phase_map_from_a_different_scan_is_refused(loaded, monkeypatch,
                                                     tmp_path):
    """The store holds one map and the session holds one file, and they can
    disagree. Exporting that mix would put numbers from two datasets in the
    same row and nothing in the file would say so."""
    client, _state, _maps = loaded
    monkeypatch.setattr(
        eds_routes, "_build_at_pct_maps_for_loaded_file",
        lambda: ({"Al": np.full(100, 90.0)}, 10, 10, "other.h5oina"))
    r = _export(client, tmp_path)
    assert r.status_code == 400
    assert "Re-classify" in r.json()["detail"]
    assert client.get("/api/eds/export/preview").status_code == 400


def test_export_refuses_when_no_map_is_classified(no_map, tmp_path):
    r = no_map.post("/api/eds/export", json={"dest_dir": str(tmp_path)})
    assert r.status_code == 400


def test_a_stated_smoothing_width_reaches_the_smoothed_columns(loaded, tmp_path):
    client, _state, _maps = loaded
    b = _export(client, tmp_path, hash_source=False, scale=3).json()
    sm = b["provenance"]["classification"]["smoothing"]
    assert sm["scale_px"] == 3
    assert sm["report"]["scale_source"] == "pixels"
    header = (Path(b["folder"]) / "regions.csv").read_text(
        encoding="utf-8-sig").splitlines()[0]
    assert "smoothed_mean_at_pct_Si" in header


def test_no_stated_width_omits_the_smoothed_columns(loaded, tmp_path):
    client, _state, _maps = loaded
    b = _export(client, tmp_path, hash_source=False).json()
    assert b["provenance"]["classification"]["smoothing"]["scale_px"] is None
    header = (Path(b["folder"]) / "regions.csv").read_text(
        encoding="utf-8-sig").splitlines()[0]
    assert "smoothed_" not in header


def test_requested_values_are_echoed_beside_the_resolved_ones(loaded, tmp_path):
    client, _state, _maps = loaded
    b = _export(client, tmp_path, hash_source=False,
                requested={"mode": "cluster", "n_clusters": 8}).json()
    c = b["provenance"]["classification"]
    assert c["mode"] == "cluster"
    assert c["n_clusters_requested"] == 8
    assert c["n_clusters_resolved"] == 2
    assert c["matrix_element_resolved"] == "Al"


# ---------------------------------------------------------------------------
# the attestation: recomputed here, never taken from the caller
# ---------------------------------------------------------------------------
#
# A service-lab tester posted a compatibility report she had written by hand,
# for a different scan, with a key no validator would have thought to reject,
# and the export wrote it into provenance.json verbatim. These pin the shape
# that closes it: the record is what THIS server computed against the loaded
# scan, and the caller's object is either absent, or present under a name that
# cannot be mistaken for a verdict.


def _cu_preset(client):
    """A preset the loaded 8x8 Al/Si scan cannot satisfy: it needs Cu."""
    r = client.post("/api/eds/presets", json={
        "name": "cu", "settings": {
            "region_defs": [{"name": "Cu rich", "elements": [
                {"element": "Cu", "min_at_pct": 5.0}]}]}})
    assert r.status_code == 200, r.text
    return r.json()["preset"]


def test_a_forged_compatibility_report_never_reaches_the_file(
        loaded, preset_dir, tmp_path):
    """The attack, verbatim. HTTP 200 is fine; adopting her words is not.

    She forged ``ok`` AND invented ``_forged`` — the second is why this is
    fixed by recomputing rather than by validating: no validator rejects a key
    it has never heard of.
    """
    client, _state, _maps = loaded
    saved = _cu_preset(client)
    forged = {"ok": True, "preset_hash": "DEADBEEF",
              "_forged": "this report was computed against a DIFFERENT steel scan"}

    b = _export(client, tmp_path, hash_source=False,
                preset_name="cu", compatibility=forged).json()

    comp = b["provenance"]["preset"]["compatibility"]
    # The server's verdict, not hers: this scan measures Al and Si only.
    assert comp["computed_by"] == "server"
    assert comp["ok"] is False
    assert [x["code"] for x in comp["blockers"]] == ["missing_element"]
    assert comp["preset_hash"] == saved["content_hash"] != "DEADBEEF"
    assert b["provenance"]["preset"]["override"] is True

    # And the invented key is nowhere in the artefact — checked on the BYTES,
    # because a nested copy would still be a forged claim in the file.
    raw = (Path(b["folder"]) / "provenance.json").read_text(encoding="utf-8")
    assert "_forged" not in raw
    assert "steel" not in raw              # nor the sentence it carried
    on_disk = json.loads(raw)["preset"]["compatibility"]
    # Her hash survives ONLY as her claim, under a name that cannot be read as
    # a verdict, and exactly once.
    assert raw.count("DEADBEEF") == 1
    assert on_disk["compatibility_client_reported"]["preset_hash"] == "DEADBEEF"
    assert on_disk["ok"] is False
    assert on_disk["blockers"][0]["code"] == "missing_element"


def test_an_override_records_the_servers_objection_not_the_clients(
        loaded, preset_dir, tmp_path):
    """A record of the excuse is worth nothing; a record of the objection is
    the whole point of the file."""
    client, _state, _maps = loaded
    _cu_preset(client)
    # The user was refused and went on — but by their own account it was a
    # trivial warning about a step size.
    claimed = {"ok": False, "preset_hash": "deadbeef",
               "blockers": [{"code": "step_size_differs", "message": "close enough"}],
               "warnings": []}

    b = _export(client, tmp_path, hash_source=False,
                preset_name="cu", compatibility=claimed).json()
    comp = b["provenance"]["preset"]["compatibility"]

    assert [x["code"] for x in comp["blockers"]] == ["missing_element"]
    # Their INTENT to override survives; their account of what they overrode
    # does not become the record.
    assert comp["override_requested"] is True
    assert comp["client_report_agrees"] is False
    assert "compatibility_client_reported_note" in comp
    # Their claim, projected onto the fields a report is made of — enough to
    # read the disagreement, and no room for a key of their own naming.
    assert comp["compatibility_client_reported"] == {
        "ok": False, "preset_hash": "deadbeef", "warnings": [],
        "blockers": [{"code": "step_size_differs", "message": "close enough"}],
    }


def test_an_agreeing_client_report_is_not_copied_in_twice(
        loaded, preset_dir, tmp_path):
    client, _state, _maps = loaded
    _cu_preset(client)
    honest = client.post("/api/eds/presets/cu/check").json()

    b = _export(client, tmp_path, hash_source=False,
                preset_name="cu", compatibility=honest).json()
    comp = b["provenance"]["preset"]["compatibility"]

    assert comp["client_report_agrees"] is True
    assert "compatibility_client_reported" not in comp
    assert comp["blockers"] == honest["blockers"]


def test_a_report_about_a_preset_nobody_named_is_a_claim_and_not_a_verdict(
        loaded, preset_dir, tmp_path):
    """Nothing to recompute against — so nothing is asserted, in either
    direction. No ``ok`` key at all, and therefore no override claimed."""
    client, _state, _maps = loaded
    forged = {"ok": True, "preset_hash": "DEADBEEF", "_forged": "elsewhere"}

    b = _export(client, tmp_path, hash_source=False,
                preset_name="", compatibility=forged).json()
    comp = b["provenance"]["preset"]["compatibility"]

    assert comp["computed_by"] == "none"
    assert "ok" not in comp
    assert b["provenance"]["preset"]["override"] is False
    assert comp["compatibility_client_reported"] == {
        "ok": True, "preset_hash": "DEADBEEF", "undeclared_fields_dropped": 1}
    # Even here, where nothing could be checked, the caller does not get to
    # write a key of its own into the record.
    raw = (Path(b["folder"]) / "provenance.json").read_text(encoding="utf-8")
    assert "_forged" not in raw


def test_a_preset_name_that_no_longer_resolves_is_recorded_not_fatal(
        loaded, preset_dir, tmp_path):
    """The map is real. Refusing to write it because a preset was deleted
    would destroy work over bookkeeping — but the file has to say so."""
    client, _state, _maps = loaded
    r = _export(client, tmp_path, hash_source=False, preset_name="ghost")
    assert r.status_code == 200
    comp = r.json()["provenance"]["preset"]["compatibility"]
    assert comp["computed_by"] == "none"
    assert "ghost" in comp["computed_note"]


def test_the_recomputed_record_carries_the_preset_version(
        loaded, preset_dir, tmp_path):
    """The number a human quotes in an email. The content hash beside it is
    the stronger identifier; both is better."""
    client, _state, _maps = loaded
    client.post("/api/eds/presets", json={
        "name": "ok", "settings": _SETTINGS, "matrix_element": "Al"})
    # Saving over it bumps the version, so this is 2 and not a default 1.
    client.post("/api/eds/presets", json={
        "name": "ok", "settings": _SETTINGS, "matrix_element": "Al",
        "overwrite": True})

    b = _export(client, tmp_path, hash_source=False, preset_name="ok").json()
    comp = b["provenance"]["preset"]["compatibility"]
    assert comp["preset_version"] == 2
    assert comp["computed_by"] == "server"
    assert comp["ok"] is True


def test_a_clean_preset_is_recorded_even_when_nobody_overrode_anything(
        loaded, preset_dir, tmp_path):
    """A run that fitted is evidence too, and it is free to record."""
    client, _state, _maps = loaded
    client.post("/api/eds/presets", json={
        "name": "ok", "settings": _SETTINGS, "matrix_element": "Al"})
    b = _export(client, tmp_path, hash_source=False, preset_name="ok").json()
    comp = b["provenance"]["preset"]["compatibility"]
    assert comp["ok"] is True and comp["computed_by"] == "server"
    assert comp["override_requested"] is False
    assert b["provenance"]["preset"]["override"] is False


def test_no_preset_and_no_claim_still_records_nothing(loaded, tmp_path):
    """"No preset was involved" has to stay distinguishable from "one was
    applied and it fitted"."""
    client, _state, _maps = loaded
    b = _export(client, tmp_path, hash_source=False).json()
    assert b["provenance"]["preset"]["compatibility"] is None
    assert b["provenance"]["preset"]["override"] is False


# ---------------------------------------------------------------------------
# unknown fields are typos
# ---------------------------------------------------------------------------

def test_an_unknown_field_in_an_export_request_is_refused(loaded, tmp_path):
    """``dest`` is not ``dest_dir``. Accepting it wrote the folder somewhere
    else and said nothing."""
    client, _state, _maps = loaded
    r = _export(client, tmp_path, hash_source=False, flibbertigibbet=1)
    assert r.status_code == 422
    assert "flibbertigibbet" in r.text


def test_an_unknown_artefact_is_refused(loaded, tmp_path):
    client, _state, _maps = loaded
    r = _export(client, tmp_path, hash_source=False,
                artefacts={"phases": True, "map_pnj": True})
    assert r.status_code == 422
    assert "map_pnj" in r.text


def test_an_unknown_field_in_a_preset_save_is_refused(loaded, preset_dir):
    """``matrix`` is not ``matrix_element``; the preset would have been born
    with no pinned matrix and no complaint."""
    client, _state, _maps = loaded
    r = client.post("/api/eds/presets", json={
        "name": "typo", "settings": _SETTINGS, "matrix": "Al"})
    assert r.status_code == 422
    assert "matrix" in r.text


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------

@pytest.fixture
def preset_dir(tmp_path):
    presets_svc.set_preset_dir_for_test(tmp_path / "presets")
    presets_svc.set_builtin_dir_for_test(tmp_path / "builtin")
    (tmp_path / "presets").mkdir()
    (tmp_path / "builtin").mkdir()
    yield tmp_path / "presets"
    presets_svc.set_preset_dir_for_test(None)
    presets_svc.set_builtin_dir_for_test(None)


_SETTINGS = {
    "mode": "cluster",
    "min_score": 0.4,
    "region_defs": [{
        "name": "Si rich",
        "elements": [{"element": "Si", "min_at_pct": 40.0}],
    }],
}


def test_the_mailed_preset_is_byte_identical_to_the_saved_one(
        loaded, preset_dir):
    """A tester mailed a preset to a colleague and it would not diff against
    her saved copy — 81 of 81 lines differed with no content changed, because
    ``save_preset`` went through Python's text mode (CRLF on Windows) while the
    endpoint returned LF."""
    client, _state, _maps = loaded
    client.post("/api/eds/presets", json={
        "name": "mailme", "settings": _SETTINGS, "matrix_element": "Al"})

    body = client.get("/api/eds/presets/mailme/export").json()
    mailed = body["json_text"].encode("utf-8")
    on_disk = (preset_dir / "mailme.json").read_bytes()

    assert mailed == on_disk
    assert b"\r\n" not in mailed
    assert body["newline"] == "\n"


def test_a_preset_saves_lists_loads_and_deletes(loaded, preset_dir):
    client, _state, _maps = loaded

    assert client.get("/api/eds/presets").json()["presets"] == []

    r = client.post("/api/eds/presets", json={
        "name": "6xxx extrusion", "settings": _SETTINGS,
        "author": "SL", "notes": "for AlMgSi",
        "authored_elements": ["Al", "Si"], "matrix_element": "Al"})
    assert r.status_code == 200, r.text
    saved = r.json()["preset"]
    assert saved["name"] == "6xxx extrusion"
    assert len(saved["content_hash"]) == 16

    listed = client.get("/api/eds/presets").json()["presets"]
    assert [p["name"] for p in listed] == ["6xxx extrusion"]

    got = client.get("/api/eds/presets/6xxx extrusion").json()["preset"]
    assert got["settings"]["min_score"] == 0.4
    assert got["content_hash"] == saved["content_hash"]

    assert client.delete("/api/eds/presets/6xxx extrusion").status_code == 200
    assert client.get("/api/eds/presets/6xxx extrusion").status_code == 404


def test_saving_over_a_name_without_overwrite_is_a_conflict(loaded, preset_dir):
    client, _state, _maps = loaded
    body = {"name": "p", "settings": _SETTINGS}
    assert client.post("/api/eds/presets", json=body).status_code == 200
    assert client.post("/api/eds/presets", json=body).status_code == 409
    body["overwrite"] = True
    assert client.post("/api/eds/presets", json=body).status_code == 200


def test_a_preset_that_constrains_nothing_is_refused(loaded, preset_dir):
    """It would mean "run the defaults", which is not a recipe."""
    client, _state, _maps = loaded
    r = client.post("/api/eds/presets", json={"name": "empty", "settings": {}})
    assert r.status_code == 400


def test_deleting_something_that_is_not_there_is_a_404(loaded, preset_dir):
    client, _state, _maps = loaded
    assert client.delete("/api/eds/presets/ghost").status_code == 404


def test_a_preset_round_trips_through_export_and_import(loaded, preset_dir):
    client, _state, _maps = loaded
    client.post("/api/eds/presets", json={"name": "p", "settings": _SETTINGS,
                                          "matrix_element": "Al"})
    exported = client.get("/api/eds/presets/p/export").json()
    assert exported["filename"] == "p.json"
    original = json.loads(exported["json_text"])

    client.delete("/api/eds/presets/p")
    r = client.post("/api/eds/presets/import",
                    json={"json_text": exported["json_text"], "save": True})
    assert r.status_code == 200, r.text
    back = r.json()["preset"]
    # The hash answers "is this the same analysis?" and must survive the trip.
    assert back["content_hash"] == original["content_hash"]
    assert back["settings"] == original["settings"]
    assert r.json()["path"] is not None


def test_importing_without_saving_does_not_save(loaded, preset_dir):
    client, _state, _maps = loaded
    client.post("/api/eds/presets", json={"name": "p", "settings": _SETTINGS})
    text = client.get("/api/eds/presets/p/export").json()["json_text"]
    client.delete("/api/eds/presets/p")
    r = client.post("/api/eds/presets/import", json={"json_text": text})
    assert r.status_code == 200
    assert r.json()["path"] is None
    assert client.get("/api/eds/presets").json()["presets"] == []


def test_importing_rubbish_is_a_400_with_a_reason(loaded, preset_dir):
    client, _state, _maps = loaded
    r = client.post("/api/eds/presets/import", json={"json_text": "{nope"})
    assert r.status_code == 400
    assert "JSON" in r.json()["detail"]


# ---------------------------------------------------------------------------
# the compatibility check — the feature's real value
# ---------------------------------------------------------------------------

def test_a_preset_naming_an_unmeasured_element_is_refused_and_names_it(
        loaded, preset_dir):
    """Acceptance criterion 5. This scan measured Al and Si only."""
    client, _state, _maps = loaded
    client.post("/api/eds/presets", json={
        "name": "cu", "settings": {
            "region_defs": [{"name": "Cu rich", "elements": [
                {"element": "Cu", "min_at_pct": 5.0}]}]}})
    r = client.post("/api/eds/presets/cu/check")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    codes = [b["code"] for b in body["blockers"]]
    assert "missing_element" in codes
    assert "Cu" in body["blockers"][0]["message"]
    assert body["preset_name"] == "cu"


def test_a_preset_pinning_the_wrong_matrix_is_refused(loaded, preset_dir):
    """"Getting it wrong inverts the whole measure." This scan is 94 at% Al."""
    client, _state, _maps = loaded
    client.post("/api/eds/presets", json={
        "name": "mg", "settings": _SETTINGS, "matrix_element": "Si"})
    body = client.post("/api/eds/presets/mg/check").json()
    assert body["ok"] is False
    assert "matrix_mismatch" in [b["code"] for b in body["blockers"]]


def test_a_matching_preset_passes(loaded, preset_dir):
    client, _state, _maps = loaded
    client.post("/api/eds/presets", json={
        "name": "ok", "settings": _SETTINGS, "matrix_element": "Al",
        "authored_elements": ["Al", "Si"]})
    body = client.post("/api/eds/presets/ok/check").json()
    assert body["ok"] is True
    assert body["blockers"] == []


def test_a_different_element_set_warns_rather_than_refusing(loaded, preset_dir):
    """at% is renormalised over the measured elements, so a superset means a
    different composition in every window — a warning, not a refusal."""
    client, _state, _maps = loaded
    client.post("/api/eds/presets", json={
        "name": "wider", "settings": _SETTINGS, "matrix_element": "Al",
        "authored_elements": ["Al", "Si", "Mg", "Fe"]})
    body = client.post("/api/eds/presets/wider/check").json()
    codes = [w["code"] for w in body["warnings"]]
    assert "element_set_differs" in codes
    # Mg and Fe are only in the AUTHORED list, not in a clause, so nothing is
    # undecidable and nothing blocks.
    assert body["ok"] is True


def test_a_check_on_a_preset_that_does_not_exist_is_a_404(loaded, preset_dir):
    client, _state, _maps = loaded
    assert client.post("/api/eds/presets/ghost/check").status_code == 404


def test_the_check_report_is_the_shape_that_goes_into_the_export(
        loaded, preset_dir, tmp_path):
    """``ok: false`` -> the UI refuses -> an explicit override comes back as
    ``compatibility`` -> the refusal lands in provenance.json. This walks that
    whole path so the two halves cannot drift apart."""
    client, _state, _maps = loaded
    client.post("/api/eds/presets", json={
        "name": "cu", "settings": {
            "region_defs": [{"name": "Cu rich", "elements": [
                {"element": "Cu", "min_at_pct": 5.0}]}]}})
    report = client.post("/api/eds/presets/cu/check").json()
    assert set(report) == {"ok", "preset_name", "preset_hash",
                           "blockers", "warnings"}

    b = _export(client, tmp_path, hash_source=False,
                preset_name="cu", compatibility=report).json()
    p = b["provenance"]["preset"]
    assert p["override"] is True
    assert p["compatibility"]["blockers"][0]["code"] == "missing_element"


# ---------------------------------------------------------------------------
# the portability record, filled in server-side
# ---------------------------------------------------------------------------
#
# ``authored_elements`` and ``authored_step_um`` are the entire input to
# ``element_set_differs`` and ``step_size_differs``. The UI sends both; a
# preset written through the raw API, a script or curl would otherwise be born
# with no guards at all — and a preset with no guards is the one that travels
# furthest before anybody notices. Filled from the loaded scan, and OMITTED
# rather than guessed when there is no scan to read.

def test_a_preset_authored_through_the_raw_api_is_not_born_without_guards(
        loaded, preset_dir):
    client, _state, _maps = loaded
    r = client.post("/api/eds/presets", json={"name": "bare",
                                              "settings": _SETTINGS})
    assert r.status_code == 200, r.text
    p = r.json()["preset"]
    assert p["authored_elements"] == ["Al", "Si"]      # the loaded scan
    assert p["authored_step_um"] == STEP
    # And it survives to the file, which is where the next check reads it.
    assert client.get("/api/eds/presets/bare").json()["preset"][
        "authored_step_um"] == STEP


def test_what_the_caller_states_wins_over_the_loaded_scan(loaded, preset_dir):
    """The UI sends what the map was actually classified against; the backfill
    is a fallback, never an override."""
    client, _state, _maps = loaded
    p = client.post("/api/eds/presets", json={
        "name": "stated", "settings": _SETTINGS,
        "authored_elements": ["Mg", "Zn"], "authored_step_um": 1.25,
    }).json()["preset"]
    assert p["authored_elements"] == ["Mg", "Zn"]
    assert p["authored_step_um"] == 1.25


def test_with_no_file_open_the_record_is_omitted_rather_than_invented(
        loaded, preset_dir, monkeypatch):
    """A false "authored against" record is worse than none: it would be read
    months later as evidence. The blank is itself reported, by
    ``portability_not_checkable``."""
    client, _state, _maps = loaded

    def _no_file():
        raise HTTPException(status_code=400, detail="No file open")

    monkeypatch.setattr(eds_routes, "_build_at_pct_maps_for_loaded_file",
                        _no_file)
    monkeypatch.setattr(eds_routes, "_eds_step_um", _no_file)

    p = client.post("/api/eds/presets", json={"name": "orphan",
                                              "settings": _SETTINGS}
                    ).json()["preset"]
    assert p["authored_elements"] == []
    assert p["authored_step_um"] is None


def test_a_file_without_a_step_size_records_no_step(loaded, preset_dir,
                                                    monkeypatch):
    monkeypatch.setattr(eds_routes, "_eds_step_um", lambda: (None, None))
    client, _state, _maps = loaded
    p = client.post("/api/eds/presets", json={"name": "nostep",
                                              "settings": _SETTINGS}
                    ).json()["preset"]
    assert p["authored_elements"] == ["Al", "Si"]
    assert p["authored_step_um"] is None


def test_an_imported_preset_is_never_stamped_with_our_scan(loaded, preset_dir):
    """A colleague's preset was authored against THEIR scan. Backfilling ours
    onto it would forge exactly the provenance the field exists to carry."""
    client, _state, _maps = loaded
    text = json.dumps({
        "schema": presets_svc.PRESET_SCHEMA, "name": "from a colleague",
        "settings": _SETTINGS,
    })
    p = client.post("/api/eds/presets/import",
                    json={"json_text": text, "save": True}).json()["preset"]
    assert p["authored_elements"] == []
    assert p["authored_step_um"] is None


# ---------------------------------------------------------------------------
# the unpinned candidate list
# ---------------------------------------------------------------------------

def test_saving_an_unpinned_preset_offers_to_pin_it_rather_than_pinning_it(
        loaded, preset_dir):
    """Pinning here would save a recipe the user did not write — "the phases
    my library held on the day I pressed Save" is not "all of them". So the
    save reports the consequence and leaves the choice."""
    client, _state, _maps = loaded
    body = client.post("/api/eds/presets", json={"name": "loose",
                                                 "settings": _SETTINGS}).json()
    assert body["preset"]["phase_list_pinned"] is False
    assert body["preset"]["settings"]["phase_keys"] is None, \
        "the backend must not pin the list on the user's behalf"
    assert [h["code"] for h in body["hints"]] == ["phase_list_not_pinned"]


def test_a_pinned_preset_is_saved_without_the_hint(loaded, preset_dir):
    client, _state, _maps = loaded
    body = client.post("/api/eds/presets", json={
        "name": "pinned",
        "settings": {**_SETTINGS, "phase_keys": ["Al.cif", "Si.cif"]},
    }).json()
    assert body["preset"]["phase_list_pinned"] is True
    assert body["hints"] == []


def test_the_check_warns_about_an_unpinned_list_against_the_real_library(
        loaded, preset_dir, monkeypatch):
    """End to end: the warning the industrial-QA tester asked for reaches the
    endpoint the Apply button gates on, and does not block."""
    client, _state, _maps = loaded
    monkeypatch.setattr(
        "backend.api.services.cif_phase_library.load_cif_phase_library",
        lambda _p: {"Al.cif": _entry("Al.cif", {"Al": 100.0}),
                    "Si.cif": _entry("Si.cif", {"Si": 100.0})})
    monkeypatch.setattr("backend.api.routes.eds._crystal_db_path",
                        lambda: Path(__file__))          # any existing file

    client.post("/api/eds/presets", json={
        "name": "loose", "settings": _SETTINGS, "matrix_element": "Al"})
    body = client.post("/api/eds/presets/loose/check").json()
    w = next(w for w in body["warnings"]
             if w["code"] == "phase_list_not_pinned")
    assert w["detail"] == {"pinned": False, "n_available": 2,
                           "named_in_rules": []}
    assert body["ok"] is True, "not reproducible is not the same as not usable"


def test_the_check_says_when_a_portability_guard_could_not_run(
        loaded, preset_dir, monkeypatch):
    """A preset written before the portability fields existed, or by hand.
    The two warnings must not silently stay quiet — that reads exactly like
    "checked, fine"."""
    client, _state, _maps = loaded
    # Save it past the route so the backfill does not give it a record.
    presets_svc.save_preset(presets_svc.preset_from_settings(
        "handwritten", _SETTINGS, matrix_element="Al"))

    body = client.post("/api/eds/presets/handwritten/check").json()
    w = next(w for w in body["warnings"]
             if w["code"] == "portability_not_checkable")
    assert [g["missing"] for g in w["detail"]["guards"]] == [
        "authored_elements", "authored_step_um"]
    assert body["ok"] is True


def test_the_shipped_starter_presets_are_listed_by_the_endpoint(loaded,
                                                                tmp_path):
    """The shelf has to be stocked through the real endpoint, not only through
    the loader: ``GET /presets`` is what the UI shows on a fresh install."""
    presets_svc.set_preset_dir_for_test(tmp_path / "empty_user")
    presets_svc.set_builtin_dir_for_test(None)           # the real directory
    try:
        client, _state, _maps = loaded
        listed = client.get("/api/eds/presets").json()["presets"]
        assert listed, "a fresh install would show an empty preset list"
        assert all(p["builtin"] for p in listed)
        assert "Find the particles in an aluminium matrix" in [
            p["name"] for p in listed]
        # A built-in cannot be deleted from the UI either.
        assert client.delete(
            "/api/eds/presets/Find the particles in an aluminium matrix"
        ).status_code == 403
    finally:
        presets_svc.set_preset_dir_for_test(None)
        presets_svc.set_builtin_dir_for_test(None)


def test_a_shipped_starter_passes_the_check_on_this_scan(loaded, tmp_path,
                                                         monkeypatch):
    """It warns (unpinned list, no authoring record) but never blocks — a
    starter that refuses a normal aluminium scan would be worse than none."""
    presets_svc.set_preset_dir_for_test(tmp_path / "empty_user")
    presets_svc.set_builtin_dir_for_test(None)
    try:
        client, _state, _maps = loaded
        body = client.post(
            "/api/eds/presets/Find the particles in an aluminium matrix/check"
        ).json()
        assert body["ok"] is True
        assert body["blockers"] == []
        codes = [w["code"] for w in body["warnings"]]
        assert "phase_list_not_pinned" in codes
        assert "portability_not_checkable" in codes
    finally:
        presets_svc.set_preset_dir_for_test(None)
        presets_svc.set_builtin_dir_for_test(None)


# ---------------------------------------------------------------------------
# plausibility and the picture, over HTTP
# ---------------------------------------------------------------------------

@pytest.fixture
def implausible(monkeypatch):
    """The same 8x8 map with the silicon particle diluted to 15 at% Si.

    Si.cif then needs 100 at% silicon on pixels reading 15 - a factor of
    6.67, the same shape of failure a group leader found on a real scan where
    two Fe-intermetallics covering 61 % of the map were assigned to pixels
    carrying a third of their iron. 4 of 64 px is 6.25 %, over the naming
    floor, so it reaches ``warnings`` and not only the table.
    """
    region = np.zeros((N_ROWS, N_COLS), dtype=np.int32)
    region[3:5, 3:5] = 1
    flat = region.ravel()
    al = np.full(N_PX, 94.0)
    si = np.full(N_PX, 6.0)
    al[flat == 1], si[flat == 1] = 85.0, 15.0
    at_maps = {"Al": al, "Si": si}

    state = PhaseMapState(
        phase_grid=np.where(region > 0, 1, 0).astype(np.int32),
        score_grid=np.full((N_ROWS, N_COLS), 0.75, dtype=np.float32),
        phase_entries=[_entry("Al.cif", {"Al": 100.0}),
                       _entry("Si.cif", {"Si": 100.0})],
        n_rows=N_ROWS, n_cols=N_COLS, tolerance=15.0, min_score=0.3,
        region_grid=region, region_phase=[0, 1],
    )
    monkeypatch.setattr(
        eds_routes, "_build_at_pct_maps_for_loaded_file",
        lambda: (at_maps, N_ROWS, N_COLS, "scanB.h5oina"))
    monkeypatch.setattr(eds_routes, "_eds_step_um", lambda: (STEP, STEP))
    monkeypatch.setattr(export_routes, "get_phase_map_store",
                        lambda: _FakeStore(state))
    return TestClient(app), state, at_maps


def test_an_implausible_headline_reaches_the_client(implausible, tmp_path):
    """The whole point: the response says the number is wrong.

    "You already print both numbers next to each other. You are one
    subtraction away from saying so."
    """
    client, _state, _maps = implausible
    b = _export(client, tmp_path, hash_source=False).json()

    named = [w for w in b["warnings"] if w["code"] == "composition_implausible"]
    assert len(named) == 1
    for piece in ("Si.cif", "6.25 %", "15.00 at% Si", "100.00 at%", "6.67x",
                  "deficient"):
        assert piece in named[0]["message"], named[0]["message"]
    assert any(w["code"] == "composition_implausible_summary"
               for w in b["warnings"])

    p = b["provenance"]["plausibility"]
    assert p["n_rows_implausible"] == 1
    assert p["n_rows_named_in_warnings"] == 1
    assert p["frac_of_scan_implausible"] == pytest.approx(4 / N_PX)
    assert p["rows"][0]["element"] == "Si"


def test_the_structured_columns_are_in_the_written_tables(implausible, tmp_path):
    """Sortable, so a reader can rank the whole map by how badly it fits
    instead of reading a paragraph."""
    import csv as _csv
    client, _state, _maps = implausible
    b = _export(client, tmp_path, hash_source=False).json()
    with open(Path(b["folder"]) / "phases.csv", newline="",
              encoding="utf-8-sig") as fh:
        rows = {r["phase_name"]: r for r in _csv.DictReader(fh)}
    si = rows["Si.cif"]
    assert si["worst_element"] == "Si"
    assert float(si["worst_element_ratio"]) == pytest.approx(100 / 15)
    assert si["composition_implausible"] == "True"
    assert si["review_flag"] == "True"
    assert si["review_reasons"] == "implausible_composition"


def test_a_plausible_map_says_so_quietly(loaded, tmp_path):
    """The default fixture is a correct call at 1.54x. A check that fired here
    would be worse than no check."""
    client, _state, _maps = loaded
    b = _export(client, tmp_path, hash_source=False).json()
    assert not [w for w in b["warnings"]
                if w["code"].startswith("composition_implausible")]
    assert b["provenance"]["plausibility"]["n_rows_implausible"] == 0
    assert b["provenance"]["plausibility"]["n_rows_checkable"] == 1


def test_the_map_pngs_are_in_the_default_artefact_set(loaded, tmp_path):
    client, _state, _maps = loaded
    b = _export(client, tmp_path, hash_source=False).json()
    names = {f["name"]: f for f in b["files"]}
    assert "phase_map.png" in names and "region_map.png" in names
    assert names["phase_map.png"]["rows"] == N_PX

    from PIL import Image
    path = Path(b["folder"]) / "phase_map.png"
    with Image.open(path) as im:
        im.load()
        assert im.format == "PNG"
        assert im.size == (N_COLS, N_ROWS)
    assert path.stat().st_size == names["phase_map.png"]["bytes"]


def test_the_picture_can_be_turned_off(loaded, tmp_path):
    client, _state, _maps = loaded
    b = _export(client, tmp_path, hash_source=False,
                artefacts={"phases": True, "regions": True, "particles": True,
                           "definitions": True, "xlsx": True,
                           "map_png": False}).json()
    names = {f["name"] for f in b["files"]}
    assert "phase_map.png" not in names
    assert "phases.csv" in names


def test_the_exported_picture_uses_the_colours_the_page_is_showing(loaded,
                                                                  tmp_path,
                                                                  monkeypatch):
    """Read at call time, never bound at import.

    ``POST /api/eds/phase-map/colors`` REBINDS ``_COLOR_OVERRIDES``; a
    module-level ``from ... import`` would freeze whatever it held when this
    module was first imported and the exported picture would quietly use
    yesterday's colours.
    """
    import base64
    from backend.api.services import phase_map_store as pms

    client, state, _maps = loaded
    monkeypatch.setattr(eds_routes, "_COLOR_OVERRIDES", {"Si": "#ff00ff"})
    assert export_routes._live_colour_overrides() == {"Si": "#ff00ff"}

    b = _export(client, tmp_path, hash_source=False).json()
    on_disk = (Path(b["folder"]) / "phase_map.png").read_bytes()
    assert on_disk == base64.b64decode(
        pms.render_phase_map_to_base64(state, {"Si": "#ff00ff"}))

    # rebound after the import: the next export follows
    monkeypatch.setattr(eds_routes, "_COLOR_OVERRIDES", {"Si": "#00ff00"})
    b2 = _export(client, tmp_path, hash_source=False).json()
    assert (Path(b2["folder"]) / "phase_map.png").read_bytes() != on_disk


# ---------------------------------------------------------------------------
# what the scoring cannot see  (routes/eds.py)
# ---------------------------------------------------------------------------
#
# These live here rather than beside the rest of the auto-classify tests only
# because of file ownership on this change. They exercise the two pure helpers
# that build `POST /api/eds/auto-classify`'s new `warnings` list; the route
# wiring is one `response["warnings"] = ...` line above them.
#
# The failure they cover is silent by construction: C and O are dropped before
# any phase is scored, so on an oxide scan Al2O3 is scored on its aluminium
# alone and is indistinguishable from Al metal. The export folder says so
# afterwards. Nothing said so at the time.

def test_a_clean_metal_scan_raises_no_exclusion_warning():
    """Calibration, in the direction that matters: a warning that fires on
    every aluminium scan is one nobody reads by the second week.

    The numbers are SampleB's, from its own export record: the per-pixel at%
    sum excluding C and O runs 91.30-100.00, so C+O carries 1.56 at% on
    average and 8.70 at% in its worst pixel.
    """
    n = 10_800
    at_maps = {"Al": np.full(n, 94.0), "Si": np.full(n, 4.44),
               "O": np.full(n, 1.56)}
    at_maps["O"][:5] = 8.70          # the worst pixels, still under the floor
    assert eds_routes._excluded_signal_warning(at_maps) is None


def test_an_oxide_scan_says_what_the_scoring_could_not_see():
    n = 1_000
    at_maps = {"Al": np.full(n, 40.0), "O": np.full(n, 60.0)}
    w = eds_routes._excluded_signal_warning(at_maps)
    assert w is not None
    assert w["code"] == "scoring_ignores_c_and_o"
    assert w["detail"]["elements"] == ["O"]
    assert w["detail"]["mean_at_pct"] == 60.0
    # Names the consequence, not just the fact.
    assert "Al2O3 cannot be told from Al" in w["message"]


def test_a_small_carbide_particle_is_not_averaged_away():
    """A 2 at% map mean hides a 200 px carbide. The per-pixel arm catches it."""
    n = 10_000
    c = np.zeros(n)
    c[:200] = 25.0                    # Fe3C, the floor of the family
    at_maps = {"Fe": 100.0 - c, "C": c}
    w = eds_routes._excluded_signal_warning(at_maps)
    assert w is not None
    assert w["detail"]["mean_at_pct"] < 1.0
    assert w["detail"]["frac_px_above"] == 0.02


def test_an_oxide_and_its_metal_are_reported_as_indistinguishable():
    """The tester's case, proved rather than guessed: chemistry_fit is
    1 - 0.5*L1 on the C/O-stripped compositions, so two phases that are equal
    after the strip can never score differently on any pixel."""
    cands = [_entry("Al.cif", {"Al": 100.0}),
             _entry("Al2O3.cif", {"Al": 40.0, "O": 60.0}),
             _entry("Si.cif", {"Si": 100.0})]
    out = eds_routes._degenerate_candidate_warnings(cands)
    codes = [w["code"] for w in out]
    assert codes == ["indistinguishable_after_excluding_c_and_o"]
    pairs = out[0]["detail"]["pairs"]
    assert len(pairs) == 1
    assert {pairs[0]["a"], pairs[0]["b"]} == {"Al.cif", "Al2O3.cif"}
    assert pairs[0]["score_gap_bound"] == 0.0
    assert "tie-break" in out[0]["message"]


def test_a_phase_that_is_only_c_or_o_scores_neutral_and_says_so():
    """Graphite has nothing left after the strip, so it scores 1.0 against
    every pixel in the scan — a different and worse failure than a tie."""
    cands = [_entry("Fe.cif", {"Fe": 100.0}), _entry("C.cif", {"C": 100.0})]
    out = eds_routes._degenerate_candidate_warnings(cands)
    assert [w["code"] for w in out] == ["phase_has_no_scoreable_chemistry"]
    assert out[0]["detail"]["phases"] == ["C.cif"]


def test_a_library_with_no_oxide_or_carbide_says_nothing():
    """Additive means additive: the ordinary metal run gains no new noise."""
    cands = [_entry("Al.cif", {"Al": 100.0}), _entry("Si.cif", {"Si": 100.0}),
             _entry("Al3Fe.cif", {"Al": 75.0, "Fe": 25.0})]
    assert eds_routes._degenerate_candidate_warnings(cands) == []


def test_two_metals_close_but_separable_are_not_flagged():
    """The bound is the library's own tie tolerance, not a round number."""
    cands = [_entry("AlO.cif", {"Al": 50.0, "O": 50.0}),
             _entry("Al9Fe.cif", {"Al": 90.0, "Fe": 10.0})]
    assert eds_routes._degenerate_candidate_warnings(cands) == []


def test_every_eds_request_model_refuses_unknown_fields():
    """One assertion per model, so a model added later without the config is
    a failure here rather than a silent default in somebody's batch."""
    import inspect
    from pydantic import BaseModel

    missing = [name for name, obj in vars(eds_routes).items()
               if inspect.isclass(obj) and issubclass(obj, BaseModel)
               and obj is not BaseModel
               and obj.model_config.get("extra") != "forbid"]
    assert missing == []
