"""The EDS phase map belongs to one measurement of its file.

The store keeps one map and a sidecar ``<file>.phase_map.npz`` per file. Loading another measurement of
the SAME file does not reopen the HDF5 session (the path is the same), so the store was neither cleared
nor reloaded: the classification of measurement 1 stayed on screen after measurement 2 was loaded, the
EDS export paired it with measurement 2's at-% maps, and a first open of measurement 2 restored the
sidecar measurement 1 had left. The map is now stored, saved and restored per measurement; the default
measurement keeps today's sidecar name, so every existing sidecar loads as before.

The same rule holds for the analysis dataset stashed per file when another file is loaded.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.api.routes import ebsd_viewer as ev
from backend.api.services import h5_session, variant_registry
from backend.api.services.cif_phase_library import CifPhaseEntry
from backend.api.services.load_choices import LoadVariant
from backend.api.services.phase_map_store import _sidecar_path_for, get_phase_map_store
from tests.test_variant_provenance import client, multi  # noqa: F401  (fixtures)
from tests.test_load_variant_routes import _v

ENTRIES = [
    CifPhaseEntry(key="A.cif", cif_filename="A.cif", formula="Fe", space_group="", space_group_number=None,
                  crystal_system="", composition={"Fe": 100.0}, elements=["Fe"]),
    CifPhaseEntry(key="B.cif", cif_filename="B.cif", formula="O2", space_group="", space_group_number=None,
                  crystal_system="", composition={"O": 100.0}, elements=["O"]),
]
# the maps are told apart by their pixels: measurement 1 is phase A everywhere, measurement 2 phase B
GRID = {"default": np.zeros((4, 4), np.int32), "2": np.ones((4, 4), np.int32)}


def _classify(path, measurement):
    get_phase_map_store().set_classification(
        phase_grid=GRID["default" if measurement is None else measurement],
        score_grid=np.full((4, 4), 0.8, np.float32), phase_entries=ENTRIES, tolerance=1.0, min_score=0.5,
        file_path=path, measurement=measurement)


def _held():
    state = get_phase_map_store().get_state()
    return None if state is None else int(state.phase_grid[0, 0])


def _load(client, path, variant=None):
    body = {"path": path}
    if variant is not None:
        body["variant"] = variant
    resp = client.post("/api/ebsd/load", json=body)
    assert resp.status_code == 200, resp.text


@pytest.fixture(autouse=True)
def empty_store():
    get_phase_map_store().clear()
    yield
    get_phase_map_store().clear()


# ----------------------------------------------------------------------------------- the sidecar name

def test_the_default_measurement_keeps_todays_sidecar_name(tmp_path):
    p = str(tmp_path / "scan.h5oina")
    assert _sidecar_path_for(p).name == "scan.h5oina.phase_map.npz"
    assert _sidecar_path_for(p, None).name == "scan.h5oina.phase_map.npz"
    assert _sidecar_path_for(p, "2").name == "scan.h5oina.measurement-2.phase_map.npz"
    assert _sidecar_path_for(p, "Scan 1/a").name == "scan.h5oina.measurement-Scan_1_a.phase_map.npz"


def test_a_sidecar_written_before_measurements_loads_for_the_default_only(tmp_path):
    store = get_phase_map_store()
    path = str(tmp_path / "scan.h5oina")
    _classify(path, None)                                              # writes the old name, no key
    store.clear()
    # as if written by an older version: the meta has no "measurement" key at all
    import json
    sidecar = _sidecar_path_for(path)
    with np.load(sidecar, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    meta = json.loads(str(arrays["meta"]))
    meta.pop("measurement")
    arrays["meta"] = np.array(json.dumps(meta), dtype=str)
    np.savez_compressed(sidecar, **arrays)
    assert store.load_from_disk(path) and _held() == 0
    store.clear()
    # ... and copied to the name of another measurement it is not that measurement's
    other = _sidecar_path_for(path, "2")
    other.write_bytes(sidecar.read_bytes())
    assert not store.load_from_disk(path, "2") and _held() is None


# --------------------------------------------------------------------------- the store follows a load

def test_loading_another_measurement_of_the_same_file_drops_the_other_measurements_map(client, multi):
    _load(client, multi)
    _classify(multi, None)
    assert _held() == 0
    _load(client, multi, _v("2", "processed"))
    assert _held() is None                                             # was: measurement 1's map stayed
    assert get_phase_map_store().file_path() is None or get_phase_map_store().measurement() == "2"


def test_each_measurement_is_saved_to_its_own_sidecar_and_comes_back(client, multi):
    _load(client, multi)
    _classify(multi, None)
    _load(client, multi, _v("2", "processed"))
    _classify(multi, "2")
    assert _held() == 1
    assert _sidecar_path_for(multi).is_file() and _sidecar_path_for(multi, "2").is_file()
    _load(client, multi)                                               # back: measurement 1's map, from its sidecar
    assert _held() == 0
    _load(client, multi, _v("2", "processed"))
    assert _held() == 1
    _load(client, multi, _v("1", "unprocessed"))                       # the same measurement, another source
    assert _held() == 0


def test_a_first_open_of_measurement_2_does_not_restore_the_sidecar_measurement_1_left(client, multi):
    _load(client, multi)
    _classify(multi, None)
    get_phase_map_store().clear()
    h5_session.close_file()
    _load(client, multi, _v("2", "processed"))                         # a fresh session on measurement 2
    assert _held() is None
    _load(client, multi)
    assert _held() == 0


def test_a_map_classified_on_measurement_2_is_stamped_with_it(client, multi):
    _load(client, multi, _v("2", "processed"))
    assert h5_session.active_measurement() == "2"
    _load(client, multi)
    assert h5_session.active_measurement() is None
    _load(client, multi, _v("1", "processed"))
    assert h5_session.active_measurement() is None                     # "1" is the file's default slot


def test_the_export_does_not_pair_a_left_map_with_the_loaded_measurements_chemistry(client, multi):
    """The last place the two meet: even when the load did not do it, the export drops the other
    measurement's map (HTTP 400, nothing to export), instead of building tables from it."""
    from fastapi import HTTPException
    from backend.api.routes.eds_export import _state_or_400
    _load(client, multi)
    _classify(multi, None)
    variant_registry.register(ev._active_dataset, LoadVariant("2", "processed"))
    with pytest.raises(HTTPException) as e:
        _state_or_400()
    assert e.value.status_code == 400


def test_the_state_response_follows_the_loaded_measurement(client, multi):
    from backend.api.routes.eds import _state_to_response
    _load(client, multi)
    _classify(multi, None)
    assert _state_to_response()["loaded"] is True
    variant_registry.register(ev._active_dataset, LoadVariant("2", "processed"))
    assert _state_to_response() == {"loaded": False}


def test_the_classify_route_saves_the_map_for_the_measurement_it_read(client, multi, monkeypatch):
    """The real classify route, with the chemistry and the phase library stubbed: the map is stamped with
    (and the sidecar named for) the measurement whose at-% maps it was made from."""
    import asyncio
    from backend.api.routes import eds
    nx, ny = 3, 2
    at_maps = {"Fe": np.full(ny * nx, 100.0), "O": np.zeros(ny * nx)}
    monkeypatch.setattr(eds, "_build_at_pct_maps_for_loaded_file", lambda: (at_maps, ny, nx, multi))
    monkeypatch.setattr(eds, "load_cif_phase_library_with_skips", lambda path: ({e.key: e for e in ENTRIES}, []))
    monkeypatch.setattr(eds, "_crystal_db_path", lambda: "unused")
    for variant, expected in ((None, None), (_v("2", "processed"), "2")):
        _load(client, multi, variant)
        asyncio.run(eds.auto_classify(eds.AutoClassifyRequest(mode="pixel")))
        assert get_phase_map_store().measurement() == expected
        assert _sidecar_path_for(multi, expected).is_file()


def test_the_classify_route_labels_the_map_with_the_measurement_it_started_on(client, multi, monkeypatch):
    """The chemistry is read for the measurement that is active when the read starts (the extractor is
    chosen first). A viewer switch to measurement 2 during that read (seconds on a big EDS file) must not
    give measurement 1's chemistry the label of measurement 2: the map would be saved to measurement 2's
    sidecar and kept as its own."""
    import asyncio
    from backend.api.routes import eds
    nx, ny = 3, 2
    at_maps = {"Fe": np.full(ny * nx, 100.0), "O": np.zeros(ny * nx)}

    def read_while_the_viewer_switches():
        variant_registry.register(ev._active_dataset, LoadVariant("2", "processed"))
        return at_maps, ny, nx, multi

    monkeypatch.setattr(eds, "_build_at_pct_maps_for_loaded_file", read_while_the_viewer_switches)
    monkeypatch.setattr(eds, "load_cif_phase_library_with_skips", lambda path: ({e.key: e for e in ENTRIES}, []))
    monkeypatch.setattr(eds, "_crystal_db_path", lambda: "unused")
    _load(client, multi)                                               # the default measurement
    asyncio.run(eds.auto_classify(eds.AutoClassifyRequest(mode="pixel")))
    assert _sidecar_path_for(multi, None).is_file(), "the map of measurement 1 was not saved as measurement 1's"
    assert not _sidecar_path_for(multi, "2").exists(), "measurement 1's chemistry got measurement 2's label"


def test_a_map_the_user_cleared_stays_cleared_while_its_sidecar_exists(client, multi):
    """DELETE /phase-map starts from scratch. The sidecar stays on disk, and the check after a load must
    not read it back the next time the state is asked for."""
    from backend.api.routes.eds import _state_to_response, clear_phase_map
    import asyncio
    _load(client, multi)
    _classify(multi, None)
    assert _sidecar_path_for(multi).is_file()
    asyncio.run(clear_phase_map())
    assert _held() is None
    assert _state_to_response() == {"loaded": False}
    assert not h5_session.sync_phase_map_store()
    assert _held() is None
    _load(client, multi, _v("2", "processed"))                         # leaving and coming back is a load
    _load(client, multi)
    assert _held() == 0


def test_a_cleared_map_stays_cleared_after_a_plain_session_open(client, multi):
    """The HDF5 page opens the session without a load: the sidecar it restores counts as matched too."""
    _load(client, multi)
    _classify(multi, None)
    h5_session.close_file()
    h5_session.open_file(multi)
    assert _held() == 0
    get_phase_map_store().clear()
    assert not h5_session.sync_phase_map_store()
    assert _held() is None


def test_the_synchronous_loader_switches_the_map_too(client, multi):
    """``load_ebsd_file`` (the batch loader) loads the file the way the entry last loaded remembers it;
    another measurement of the open file must swap the map there as well."""
    _load(client, multi)
    _classify(multi, None)
    _load(client, multi, _v("2", "processed"))
    _classify(multi, "2")
    assert _held() == 1
    ev._loaded_entry(multi)["variant"] = None                        # the entry now remembers the default
    assert ev.load_ebsd_file(multi)
    assert _held() == 0
    assert get_phase_map_store().measurement() is None


def test_opening_the_session_restores_the_map_of_the_measurement_the_viewer_holds(client, multi):
    """``open_file`` also runs without a load (the HDF5 page, an import): the sidecar it restores is the
    viewer's measurement's, and nothing else is restored when that has none."""
    _load(client, multi)
    _classify(multi, None)                                             # only the default has a sidecar
    variant_registry.register(ev._active_dataset, LoadVariant("2", "processed"))
    h5_session.close_file()
    h5_session.open_file(multi)
    assert _held() is None
    variant_registry.drop(ev._active_dataset)
    h5_session.close_file()
    h5_session.open_file(multi)
    assert _held() == 0


# ----------------------------------------------------------------------- the analysis dataset stash

def test_the_analysis_dataset_is_stashed_per_variant_of_the_file():
    from backend.api.routes import analysis
    saved, saved_stash = analysis._dataset, dict(analysis._dataset_by_file)
    analysis._dataset_by_file.clear()
    try:
        first, second = object(), object()
        analysis._dataset = first
        analysis.stash_dataset_for_file("/d/a.h5", None)
        analysis._dataset = second
        analysis.stash_dataset_for_file("/d/a.h5", ("2", "processed"))
        analysis._dataset = None
        assert analysis.restore_dataset_for_file("/d/a.h5", ("2", "processed")) and analysis._dataset is second
        assert analysis.restore_dataset_for_file("/d/a.h5") and analysis._dataset is first
        assert not analysis.restore_dataset_for_file("/d/a.h5", ("1", "unprocessed")) and analysis._dataset is None
        assert "/d/a.h5" in analysis._dataset_by_file            # the default variant keeps the plain key
    finally:
        analysis._dataset = saved
        analysis._dataset_by_file.clear()
        analysis._dataset_by_file.update(saved_stash)


def test_loading_another_measurement_does_not_bring_the_other_ones_analysis_back(client, multi):
    from backend.api.routes import analysis
    saved, saved_stash = analysis._dataset, dict(analysis._dataset_by_file)
    analysis._dataset_by_file.clear()
    try:
        _load(client, multi)
        first = analysis._dataset = object()                           # an analysis of measurement 1
        _load(client, multi, _v("2", "processed"))
        assert analysis._dataset is None                               # was: the same object came back
        second = analysis._dataset = object()
        _load(client, multi)
        assert analysis._dataset is first                              # measurement 1's own, not 2's
        _load(client, multi, _v("2", "processed"))
        assert analysis._dataset is second
    finally:
        analysis._dataset = saved
        analysis._dataset_by_file.clear()
        analysis._dataset_by_file.update(saved_stash)


def test_an_eds_only_load_stashes_the_analysis_of_the_file_it_leaves_under_its_measurement(client, multi, tmp_path):
    """The EDS-only load path stashes the outgoing file's analysis dataset itself (it has its own copy of
    the stash and restore calls, apart from the ordinary load's). Left out, or keyed by the path alone,
    the analysis of measurement 2 is lost, or comes back under measurement 1."""
    import h5py
    from backend.api.routes import analysis
    eds_only = tmp_path / "eds_only.h5oina"
    with h5py.File(eds_only, "w") as f:
        g = f.create_group("1")
        eds = g.create_group("EDS/Data/Window Integral")
        eds.create_dataset("Al Ka1", data=np.zeros(12, dtype=np.float32))
        g.create_group("Electron Image/Data").create_dataset("SE", data=np.zeros((4, 4), dtype=np.uint8))
    saved, saved_stash = analysis._dataset, dict(analysis._dataset_by_file)
    analysis._dataset_by_file.clear()
    try:
        _load(client, multi, _v("2", "processed"))
        mine = analysis._dataset = object()                            # an analysis of measurement 2
        _load(client, str(eds_only))                                   # the EDS-only path
        assert analysis._dataset is None, "premise: the EDS-only file has no analysis of its own"
        _load(client, multi, _v("2", "processed"))
        assert analysis._dataset is mine, "the analysis of measurement 2 was lost on the EDS-only load"
        _load(client, str(eds_only))
        _load(client, multi)                                           # the default measurement of the file
        assert analysis._dataset is None, "measurement 2's analysis came back under measurement 1"
    finally:
        analysis._dataset = saved
        analysis._dataset_by_file.clear()
        analysis._dataset_by_file.update(saved_stash)
