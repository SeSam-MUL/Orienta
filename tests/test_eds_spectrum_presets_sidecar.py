"""Presets and saved phase maps remember which numbers they were made on.

Orienta's EDS numbers come from one of two sources: AZtec's window integrals
("window") or net intensities fitted from the spectrum cube ("spectrum", the
default for every file with a cube). There is no permanent switch back, so a
recipe or a saved map made on one source is meeting the other the day a file is
opened with the spectrum fit. A threshold chosen on window counts does not mean
the same on net intensities, so:

* a preset carries ``quant_source``; one without the field was written before
  the spectrum source existed, i.e. on window numbers;
* checking it against a file served by the other source is a WARNING
  (``preset_made_on_other_source``), never a blocker, and its thresholds are
  never converted;
* a phase-map sidecar carries ``quant_source`` too (an optional key: the schema
  stays 4, nobody's map on disk is discarded) and the ``GET /phase-map``
  response says so in ``notices`` when it differs from the open file's source.

Real data: the Oxford reference scan (this checkout's ``Test_data``, or ``ORIENTA_TEST_DATA``) for
the route cases; the rest needs no file.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.services import eds_presets as presets_svc
from backend.api.services import eds_spectrum_service as svc
from backend.api.services import h5_session
from backend.api.services.cif_phase_library import CifPhaseEntry
from backend.api.services.eds_presets import (
    PRESET_SCHEMA, EdsPreset, check_compatibility, import_preset_json,
    preset_from_settings,
)
from backend.api.services.phase_map_store import (
    _SIDECAR_SCHEMA, PhaseMapStore, _sidecar_path_for, get_phase_map_store,
)
from tests.test_eds_spectrum_service import WAIT, sampleb  # noqa: F401  (fixture)

SETTINGS = {"mode": "cluster", "min_score": 0.4,
            "rules": None, "element_weights": {"Fe": 2.0}}


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_session(tmp_path):
    from backend.api.services import crop_window as crop_window_service

    def _reset():
        h5_session.close_file()
        svc.reset()
        crop_window_service.clear_all()
        get_phase_map_store().clear()

    _reset()
    presets_svc.set_preset_dir_for_test(tmp_path / "user")
    presets_svc.set_builtin_dir_for_test(tmp_path / "shipped")
    yield
    presets_svc.set_preset_dir_for_test(None)
    presets_svc.set_builtin_dir_for_test(None)
    _reset()


@pytest.fixture
def ready(sampleb):  # noqa: F811
    h5_session.open_file(str(sampleb))
    assert svc.get_result(str(sampleb), wait_s=WAIT) is not None
    return sampleb


def _save(client, name):
    r = client.post("/api/eds/presets", json={"name": name, "settings": SETTINGS})
    assert r.status_code == 200, r.text
    return r.json()


def _codes(report):
    return [w["code"] for w in report["warnings"]]


# ---------------------------------------------------------------------------
# presets: the field
# ---------------------------------------------------------------------------
def test_saving_a_preset_on_a_spectrum_file_records_the_spectrum_source(ready, client):
    body = _save(client, "on spectrum")
    assert body["preset"]["quant_source"] == "spectrum"
    # And the FILE says so, not only the response.
    written = json.loads(open(body["path"], encoding="utf-8").read())
    assert written["quant_source"] == "spectrum"
    assert written["schema"] == PRESET_SCHEMA == 1       # the schema did not move


def test_saving_a_preset_on_a_window_file_records_the_window_source(ready, client):
    with svc.force_window_source():
        body = _save(client, "on window")
    assert body["preset"]["quant_source"] == "window"


def test_a_preset_without_the_field_reads_as_window():
    """The truth about every preset written before the field existed."""
    old = {"schema": 1, "name": "old", "settings": dict(SETTINGS)}
    assert "quant_source" not in old
    assert EdsPreset.from_dict(old).quant_source == "window"
    assert import_preset_json(json.dumps(old)).quant_source == "window"


def test_the_field_survives_export_and_import():
    p = preset_from_settings("made on spectrum", SETTINGS, quant_source="spectrum")
    again = import_preset_json(presets_svc.export_preset_json(p))
    assert again.quant_source == "spectrum"
    # An imported preset keeps ITS source: stamping ours would forge provenance.
    assert again.to_dict()["quant_source"] == "spectrum"


def test_an_unknown_source_is_refused_not_guessed():
    with pytest.raises(ValueError, match="quant_source"):
        EdsPreset.from_dict({"schema": 1, "name": "x", "settings": dict(SETTINGS),
                             "quant_source": "magic"})
    with pytest.raises(ValueError, match="quant_source"):
        preset_from_settings("x", SETTINGS, quant_source="magic")


def test_the_source_is_not_part_of_the_content_hash():
    """Same recipe, same hash: the source says where it was made, and the hash
    answers "is this the same analysis?" -- the warning handles the rest."""
    a = preset_from_settings("a", SETTINGS, quant_source="window")
    b = preset_from_settings("a", SETTINGS, quant_source="spectrum")
    assert a.content_hash == b.content_hash


# ---------------------------------------------------------------------------
# presets: the check
# ---------------------------------------------------------------------------
def _check_args(**over):
    kw = dict(measured_elements=["Al", "Fe", "Si"], available_phase_keys=None)
    kw.update(over)
    return kw


def test_a_window_preset_on_a_spectrum_file_warns():
    p = preset_from_settings("w", SETTINGS, quant_source="window",
                             authored_elements=["Al", "Fe", "Si"],
                             authored_step_um=0.5)
    report = check_compatibility(p, **_check_args(step_um=0.5,
                                                  quant_source="spectrum"))
    hits = [w for w in report.warnings if w["code"] == "preset_made_on_other_source"]
    assert len(hits) == 1
    assert hits[0]["detail"] == {"preset": "window", "current": "spectrum"}
    assert report.ok is True            # a warning, never a blocker
    assert report.blockers == []


def test_a_spectrum_preset_on_a_window_file_warns_too():
    p = preset_from_settings("s", SETTINGS, quant_source="spectrum")
    report = check_compatibility(p, **_check_args(quant_source="window"))
    assert "preset_made_on_other_source" in [w["code"] for w in report.warnings]


def test_the_same_source_does_not_warn():
    for source in ("window", "spectrum"):
        p = preset_from_settings("p", SETTINGS, quant_source=source)
        report = check_compatibility(p, **_check_args(quant_source=source))
        assert "preset_made_on_other_source" not in [
            w["code"] for w in report.warnings], source


def test_the_check_defaults_to_the_window_source():
    """A caller that does not pass it is a window caller (the old signature)."""
    p = preset_from_settings("w", SETTINGS)            # window by default
    report = check_compatibility(p, **_check_args())
    assert "preset_made_on_other_source" not in [w["code"] for w in report.warnings]


def test_the_warning_never_touches_the_thresholds():
    p = preset_from_settings("w", SETTINGS, quant_source="window")
    before = json.dumps(p.settings, sort_keys=True)
    check_compatibility(p, **_check_args(quant_source="spectrum"))
    assert json.dumps(p.settings, sort_keys=True) == before


def test_the_check_route_names_the_source_of_the_open_file(ready, client):
    old = {"schema": 1, "name": "legacy", "settings": dict(SETTINGS)}
    presets_svc.save_preset(import_preset_json(json.dumps(old)))
    r = client.post("/api/eds/presets/legacy/check")
    assert r.status_code == 200, r.text
    hits = [w for w in r.json()["warnings"]
            if w["code"] == "preset_made_on_other_source"]
    assert len(hits) == 1
    assert hits[0]["detail"] == {"preset": "window", "current": "spectrum"}

    with svc.force_window_source():
        r = client.post("/api/eds/presets/legacy/check")
    assert "preset_made_on_other_source" not in _codes(r.json())


def test_a_preset_saved_here_is_clean_here(ready, client):
    _save(client, "mine")
    r = client.post("/api/eds/presets/mine/check")
    assert r.status_code == 200, r.text
    assert "preset_made_on_other_source" not in _codes(r.json())


# ---------------------------------------------------------------------------
# sidecars: the key
# ---------------------------------------------------------------------------
N_ROWS = N_COLS = 4
ENTRIES = [
    CifPhaseEntry(key=n, cif_filename=n, formula=n, space_group="Fm-3m",
                  space_group_number=225, crystal_system="cubic",
                  composition={n.split(".")[0]: 100.0},
                  elements=[n.split(".")[0]])
    for n in ("Al.cif", "Si.cif")
]


def _classify(store, *, file_path=None, **kw):
    store.set_classification(
        phase_grid=np.zeros((N_ROWS, N_COLS), dtype=np.int32),
        score_grid=np.full((N_ROWS, N_COLS), 0.8, dtype=np.float32),
        phase_entries=list(ENTRIES), tolerance=15.0, min_score=0.3,
        file_path=file_path, **kw)


@pytest.mark.parametrize("source", ["window", "spectrum"])
def test_the_sidecar_round_trips_the_source(tmp_path, source):
    f = tmp_path / "scan.h5oina"
    f.write_bytes(b"key")
    store = PhaseMapStore()
    _classify(store, file_path=str(f), quant_source=source)
    assert store.get_state().quant_source == source
    fresh = PhaseMapStore()
    assert fresh.load_from_disk(str(f)) is True
    assert fresh.get_state().quant_source == source
    assert _SIDECAR_SCHEMA == 4                          # no bump


def test_a_sidecar_without_the_key_still_loads_and_reads_as_window(tmp_path):
    """Built by REMOVING the key from a file the app just wrote."""
    f = tmp_path / "scan.h5oina"
    f.write_bytes(b"key")
    store = PhaseMapStore()
    _classify(store, file_path=str(f), quant_source="spectrum")
    sidecar = _sidecar_path_for(str(f))
    with np.load(str(sidecar), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    meta = json.loads(arrays["meta"].item())
    assert meta["schema"] == 4 and meta["quant_source"] == "spectrum"
    del meta["quant_source"]
    arrays["meta"] = np.array(json.dumps(meta), dtype=str)
    np.savez_compressed(str(sidecar), **arrays)

    fresh = PhaseMapStore()
    assert fresh.load_from_disk(str(f)) is True          # the map is not discarded
    state = fresh.get_state()
    assert state.quant_source == "window"
    assert state.phase_grid.shape == (N_ROWS, N_COLS)


def test_a_map_classified_without_a_source_claims_window():
    store = PhaseMapStore()
    _classify(store)
    assert store.get_state().quant_source == "window"


def test_undo_does_not_wipe_the_source():
    """``_snapshot`` rebuilds the state field by field: a field it forgets is
    reset by every undo and autosaved that way."""
    store = PhaseMapStore()
    _classify(store, quant_source="spectrum")
    store.assign_region(0, 1, 0, 1, 1)
    assert store.undo() is True
    assert store.get_state().quant_source == "spectrum"


def test_an_unknown_source_is_refused_at_classification():
    with pytest.raises(ValueError, match="quant_source"):
        _classify(PhaseMapStore(), quant_source="magic")


# ---------------------------------------------------------------------------
# sidecars: the notice
# ---------------------------------------------------------------------------
def test_a_map_made_on_window_numbers_gets_a_notice_on_a_spectrum_file(ready, client):
    _classify(get_phase_map_store(), file_path=None, quant_source="window")
    r = client.get("/api/eds/phase-map", params={"include_image": "false"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["quant_source"] == "window"
    notices = [n for n in body["notices"] if n["code"] == "sidecar_made_on_other_source"]
    assert len(notices) == 1
    assert notices[0]["detail"] == {"sidecar": "window", "current": "spectrum"}
    # The map itself is not touched by the notice.
    assert body["loaded"] is True and body["n_rows"] == N_ROWS


def test_a_map_made_on_the_same_source_gets_no_notice(ready, client):
    _classify(get_phase_map_store(), file_path=None, quant_source="spectrum")
    body = client.get("/api/eds/phase-map", params={"include_image": "false"}).json()
    assert body["quant_source"] == "spectrum"
    assert body["notices"] == []


def test_the_notice_follows_the_open_files_source(ready, client):
    _classify(get_phase_map_store(), file_path=None, quant_source="window")
    with svc.force_window_source():
        body = client.get("/api/eds/phase-map",
                          params={"include_image": "false"}).json()
    assert body["notices"] == []


def test_a_loaded_old_sidecar_is_noticed_not_discarded(ready, client, tmp_path):
    f = tmp_path / "scan.h5oina"
    f.write_bytes(b"key")
    store = PhaseMapStore()
    _classify(store, file_path=str(f), quant_source="spectrum")
    sidecar = _sidecar_path_for(str(f))
    with np.load(str(sidecar), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    meta = json.loads(arrays["meta"].item())
    del meta["quant_source"]
    arrays["meta"] = np.array(json.dumps(meta), dtype=str)
    np.savez_compressed(str(sidecar), **arrays)
    assert get_phase_map_store().load_from_disk(str(f)) is True

    body = client.get("/api/eds/phase-map", params={"include_image": "false"}).json()
    assert body["loaded"] is True
    assert [n["code"] for n in body["notices"]] == ["sidecar_made_on_other_source"]


def test_no_open_file_means_no_notice(client):
    _classify(get_phase_map_store(), file_path=None, quant_source="window")
    body = client.get("/api/eds/phase-map", params={"include_image": "false"}).json()
    assert body["notices"] == []


# ---------------------------------------------------------------------------
# the classification stamps the source of the maps it was made from
# ---------------------------------------------------------------------------
class _Maps(dict):
    """At% maps as the quantification returns them: a dict with a provenance."""

    def __init__(self, source):
        al = np.full(N_ROWS * N_COLS, 94.0)
        si = np.full(N_ROWS * N_COLS, 6.0)
        al[5:7], si[5:7] = 35.0, 65.0
        super().__init__({"Al": al, "Si": si})
        self.provenance = {"source": source}


@pytest.mark.parametrize("source", ["window", "spectrum"])
def test_auto_classify_stamps_the_source_of_its_maps(monkeypatch, tmp_path, source):
    """The call site, not only the store: delete the argument in
    ``auto_classify`` and this goes red."""
    from backend.api.routes import eds as eds_routes

    scan = tmp_path / "scan.h5oina"
    monkeypatch.setattr(
        eds_routes, "_build_at_pct_maps_for_loaded_file",
        lambda: (_Maps(source), N_ROWS, N_COLS, str(scan)))
    monkeypatch.setattr(
        eds_routes, "load_cif_phase_library_with_skips",
        lambda *a, **kw: ({e.key: e for e in ENTRIES}, []))
    monkeypatch.setattr(eds_routes, "_eds_step_um", lambda: (0.5, 0.5))
    from backend.api.main import app
    r = TestClient(app).post("/api/eds/auto-classify", json={})
    assert r.status_code == 200, r.text
    assert get_phase_map_store().get_state().quant_source == source
    # ...and the sidecar the autosave wrote says so, not only the process.
    with np.load(str(_sidecar_path_for(str(scan))), allow_pickle=False) as data:
        assert json.loads(data["meta"].item())["quant_source"] == source


def test_maps_without_provenance_are_window_maps():
    from backend.api.routes import eds as eds_routes
    assert eds_routes._maps_source({"Al": np.zeros(1)}) == "window"
    assert eds_routes._maps_source(None) == "window"
    assert eds_routes._maps_source(_Maps("spectrum")) == "spectrum"
