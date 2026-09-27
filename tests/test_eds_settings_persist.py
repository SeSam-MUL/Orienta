"""A map has to be able to say how it was made — after a restart, too.

The export writes a ``provenance.json`` whose whole job is to answer "what did
you do?". On a real scan it used to answer:

    "classification": {"mode": null, "n_clusters_requested": null,
                       "smoothing": {"scale_px": null, ...}}

because ``PhaseMapState`` persisted the classification RESULT plus tolerance,
min_score, the region definitions and the element weights — and nothing about
the settings that produced it. A reloaded map could be looked at and could not
be defended.

Smoothing is the sharp end of that. Its own module calls it "the single most
important knob on the page", and a user put the consequence plainly: *"5 px at
a 0.66 um step is a 3.3 um averaging box, which is larger than most of the
particles I am counting. Nobody in a hearing knows what 'Scale 5' means;
'3.3 um averaging box' they understand."* So the width travels with the map,
the step travels with the width, and the export quotes the box rather than a
pixel count.

Three things this file pins that are easy to get wrong:

* the sidecar schema stays **4**. ``settings`` is an OPTIONAL key, and bumping
  would discard every map already on a user's disk, hand edits and all.
* ``_snapshot`` rebuilds the state field by field, so a new field that is not
  added there is silently reset to empty by every undo — and then autosaved.
  That has already happened once in this repo, to ``region_defs`` and
  ``element_weights``, which exist precisely to survive that.
* a map classified per pixel records ``scale = None``. That is a recorded fact
  ("nothing was smoothed"), not an unknown, so it must not fall through to
  whatever width the export caller happened to state.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Needs the maintainer's crystal library / measurement data, which a clone
# does not have — skip with the missing path named, never fail.
from tests.data_deps import CIF_LIBRARY, requires

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.main import app
from backend.api.routes import eds as eds_routes
from backend.api.services import eds_export as X
from backend.api.services.cif_phase_library import CifPhaseEntry
from backend.api.services.phase_map_store import (
    _SIDECAR_SCHEMA, PhaseMapState, PhaseMapStore, _sidecar_path_for,
    get_phase_map_store,
)

N_ROWS = N_COLS = 8
N_PX = N_ROWS * N_COLS
STEP = 0.6579          # the user's real step, so the 3.3 um box is the real one

SETTINGS = {
    "mode": "cluster",
    "scale": 5,                       # RESOLVED px
    "scale_um": 3.3,                  # requested
    "scale_px_requested": None,
    "scale_report": {"scale_px_used": 5, "scale_um_used": 3.2895,
                     "scale_source": "um", "step_x_um": STEP,
                     "step_y_um": STEP},
    "n_clusters": None,               # auto
    "k_used": 2,
    "min_score": 0.3,
    "cluster_remainder": True,
    "phase_keys": ["Al.cif", "Si.cif"],
    "rules": None,
    "step_x_um": STEP,
    "step_y_um": STEP,
}


def _entry(name, comp):
    return CifPhaseEntry(
        key=name, cif_filename=name, formula=name, space_group="Fm-3m",
        space_group_number=225, crystal_system="cubic", composition=comp,
        elements=sorted(comp),
    )


ENTRIES = [_entry("Al.cif", {"Al": 100.0}), _entry("Si.cif", {"Si": 100.0})]


def _classify(store, *, file_path=None, settings=SETTINGS):
    region = _region_grid()
    store.set_classification(
        phase_grid=np.where(region > 0, 1, 0).astype(np.int32),
        score_grid=np.full((N_ROWS, N_COLS), 0.8, dtype=np.float32),
        phase_entries=list(ENTRIES), tolerance=15.0, min_score=0.3,
        file_path=file_path,
        region_grid=region, region_phase=[0, 1],
        settings=settings,
    )


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------

def test_a_fresh_classification_carries_its_settings():
    store = PhaseMapStore()
    _classify(store)
    assert store.get_state().settings == SETTINGS


def test_a_map_made_without_them_claims_nothing():
    """Absence must read as "not recorded", never as a default."""
    store = PhaseMapStore()
    _classify(store, settings=None)
    assert store.get_state().settings == {}


def test_settings_survive_a_sidecar_round_trip(tmp_path):
    f = tmp_path / "scan.h5oina"
    f.write_bytes(b"not a real file, only a key for the sidecar")
    store = PhaseMapStore()
    _classify(store, file_path=str(f))
    assert store.save_to_disk(str(f)) is not None

    fresh = PhaseMapStore()
    assert fresh.load_from_disk(str(f)) is True
    restored = fresh.get_state().settings
    assert restored == SETTINGS
    # The two numbers the whole feature exists for, specifically.
    assert restored["scale"] == 5
    assert restored["scale_um"] == pytest.approx(3.3)
    assert restored["step_x_um"] == pytest.approx(STEP)


def test_the_sidecar_schema_was_not_bumped():
    """An OPTIONAL field does not justify invalidating everyone's maps.

    ``load_from_disk`` refuses any sidecar whose schema is not exactly this
    number, so a bump silently discards every phase map already on disk —
    including the hand edits and hand-written definitions that the sidecar
    exists to protect.
    """
    assert _SIDECAR_SCHEMA == 4


def test_a_sidecar_written_before_this_feature_still_loads(tmp_path):
    """The compatibility case, built by REMOVING the key from a real file.

    Hand-rolling an "old" npz would test a file this app never wrote. This
    takes one the app just wrote and deletes exactly the new key, which is what
    an older Orienta actually left behind.
    """
    f = tmp_path / "scan.h5oina"
    f.write_bytes(b"key")
    store = PhaseMapStore()
    _classify(store, file_path=str(f))
    sidecar = _sidecar_path_for(str(f))

    with np.load(str(sidecar), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    meta = json.loads(arrays["meta"].item())
    assert meta["schema"] == 4
    del meta["settings"]
    arrays["meta"] = np.array(json.dumps(meta), dtype=str)
    np.savez_compressed(str(sidecar), **arrays)

    fresh = PhaseMapStore()
    assert fresh.load_from_disk(str(f)) is True      # loads, does not raise
    state = fresh.get_state()
    assert state.settings == {}
    assert state.region_phase == [0, 1]              # and the map is all there
    assert state.phase_grid.shape == (N_ROWS, N_COLS)


def test_undo_does_not_wipe_the_settings():
    """The ``_snapshot`` trap, which this repo has already fallen into once.

    ``_snapshot`` rebuilds the state field by field. A field missing from it is
    reset to its default by every undo — and ``undo`` calls ``_autosave``, so
    the empty version is written to disk. Mutation-tested: delete
    ``settings=dict(st.settings or {})`` from ``_snapshot`` and this test goes
    red on the first assert.
    """
    store = PhaseMapStore()
    _classify(store)                                  # file_path None: no autosave
    store.assign_region(0, 1, 0, 1, 1)                # a hand edit -> snapshot
    assert store.undo() is True

    assert store.get_state().settings == SETTINGS
    assert store.get_state().region_defs == []        # the older sibling field
    # And the redo direction, which reads the other half of the swap.
    assert store.undo() is True
    assert store.get_state().settings == SETTINGS


def test_a_later_run_replaces_the_settings_rather_than_merging_them():
    """A re-classify at a different width must stop claiming the old one."""
    store = PhaseMapStore()
    _classify(store)
    _classify(store, settings={**SETTINGS, "scale": 2, "scale_um": None})
    assert store.get_state().settings["scale"] == 2
    assert store.get_state().settings["scale_um"] is None


# ---------------------------------------------------------------------------
# the endpoint
# ---------------------------------------------------------------------------

#: Where the silicon particle sits. ONE definition, used both to build the
#: composition maps and to lay out the region grid — a fixture whose particle
#: and whose region 1 are in different places measures nothing.
PARTICLE = (slice(3, 5), slice(3, 5))


def _region_grid():
    region = np.zeros((N_ROWS, N_COLS), dtype=np.int32)
    region[PARTICLE] = 1
    return region


def _at_maps():
    al = np.full((N_ROWS, N_COLS), 94.0)
    si = np.full((N_ROWS, N_COLS), 6.0)
    al[PARTICLE], si[PARTICLE] = 35.0, 65.0
    return {"Al": al.ravel(), "Si": si.ravel()}


def _library():
    return {"Al.cif": ENTRIES[0], "Si.cif": ENTRIES[1]}


@pytest.fixture
def client(monkeypatch, tmp_path):
    """A loaded file with a known step, and the REAL store behind it.

    The real one, not a fake: what is being checked is what
    ``set_classification`` actually stored. The source path points into
    ``tmp_path`` so the autosaved sidecar lands there and not in the repo.
    """
    at_maps = _at_maps()
    scan = tmp_path / "scan.h5oina"
    monkeypatch.setattr(
        eds_routes, "_build_at_pct_maps_for_loaded_file",
        lambda: (at_maps, N_ROWS, N_COLS, str(scan)))
    monkeypatch.setattr(eds_routes, "load_cif_phase_library",
                        lambda *a, **kw: _library())
    monkeypatch.setattr(eds_routes, "_eds_step_um", lambda: (STEP, STEP))
    get_phase_map_store().clear()
    try:
        yield TestClient(app)
    finally:
        get_phase_map_store().clear()


def _stored_settings():
    state = get_phase_map_store().get_state()
    assert state is not None
    return state.settings


@requires(CIF_LIBRARY)
def test_auto_classify_records_both_the_requested_um_and_the_resolved_px(client):
    """"Asked for 3.3 um, smoothed with 5 px" — both halves, or neither is
    checkable."""
    r = client.post("/api/eds/auto-classify", json={"scale_um": 3.3})
    assert r.status_code == 200, r.text
    body = r.json()

    s = _stored_settings()
    assert s["scale_um"] == pytest.approx(3.3)           # requested
    assert s["scale"] == body["scale_px_used"] == 5      # resolved
    assert s["scale_report"]["scale_source"] == "um"
    assert s["step_x_um"] == pytest.approx(STEP)
    assert s["step_y_um"] == pytest.approx(STEP)
    assert s["mode"] == "cluster"
    # None is the recorded answer for "let the criterion choose", and the
    # number it chose sits beside it.
    assert s["n_clusters"] is None
    assert s["k_used"] == body["k_used"]


@requires(CIF_LIBRARY)
def test_auto_classify_records_a_pinned_cluster_count_as_requested(client):
    r = client.post("/api/eds/auto-classify",
                    json={"n_clusters": 2, "min_score": 0.25,
                          "phase_keys": ["Al.cif", "Si.cif"],
                          "cluster_remainder": False})
    assert r.status_code == 200, r.text
    s = _stored_settings()
    assert s["n_clusters"] == 2
    assert s["min_score"] == pytest.approx(0.25)
    assert s["phase_keys"] == ["Al.cif", "Si.cif"]
    assert s["cluster_remainder"] is False


@requires(CIF_LIBRARY)
def test_pixel_mode_records_that_nothing_was_smoothed(client):
    """Not "unknown" — recorded. Per-pixel matching never smooths, and a width
    in this slot would be a false entry in the provenance."""
    r = client.post("/api/eds/auto-classify", json={"mode": "pixel",
                                                    "scale_um": 3.3})
    assert r.status_code == 200, r.text
    s = _stored_settings()
    assert s["mode"] == "pixel"
    assert "scale" in s and s["scale"] is None
    assert s["scale_report"]["scale_source"] == "not_applicable"
    # The response has always said this; the map now says it too.
    assert r.json()["scale_source"] == "not_applicable"


@requires(CIF_LIBRARY)
def test_the_settings_reach_the_sidecar_the_endpoint_wrote(client, tmp_path):
    """End to end: classify, then read the file on disk, not the process."""
    assert client.post("/api/eds/auto-classify",
                       json={"scale_um": 3.3}).status_code == 200
    fresh = PhaseMapStore()
    assert fresh.load_from_disk(str(tmp_path / "scan.h5oina")) is True
    assert fresh.get_state().settings["scale"] == 5


# ---------------------------------------------------------------------------
# the export
# ---------------------------------------------------------------------------

def _state(settings):
    region = _region_grid()
    return PhaseMapState(
        phase_grid=np.where(region > 0, 1, 0).astype(np.int32),
        score_grid=np.full((N_ROWS, N_COLS), 0.8, dtype=np.float32),
        phase_entries=list(ENTRIES), n_rows=N_ROWS, n_cols=N_COLS,
        tolerance=15.0, min_score=0.3,
        region_grid=region, region_phase=[0, 1],
        settings=dict(settings) if settings else {},
    )


def _tables(settings, *, step=STEP, **opt_kw):
    state = _state(settings)
    at_maps = _at_maps()
    geom = X.ExportGeometry(n_rows=N_ROWS, n_cols=N_COLS, step_x_um=step,
                            step_y_um=step, source_path="scan.h5oina")
    opts = X.ExportOptions(hash_source=False, **opt_kw)
    return X.build_tables(state, at_maps, geometry=geom, options=opts)


def test_provenance_quotes_the_physical_box_when_the_map_recorded_it():
    """The gap this whole change closes. "5" is not a length; 3.29 um is."""
    c = _tables(SETTINGS).provenance["classification"]
    sm = c["smoothing"]
    assert sm["scale_px"] == 5
    assert sm["source"] == "map"
    assert sm["box_um"]["x"] == pytest.approx(5 * STEP)      # 3.2895 um
    assert sm["box_um"]["y"] == pytest.approx(5 * STEP)
    assert "5 px" in sm["note"] and "3.2895 um box" in sm["note"]
    # And the rest of the record stops being null.
    assert c["mode"] == "cluster"
    assert c["n_clusters_requested"] is None
    assert c["n_clusters_at_classification"] == 2
    assert c["cluster_remainder"] is True
    assert c["phase_keys"] == ["Al.cif", "Si.cif"]
    assert c["settings_source"] == "map"
    assert "recorded WITH the map" in c["settings_note"]


def test_the_recorded_width_also_produces_the_smoothed_columns():
    """The columns claim to explain THESE regions, so they follow the recorded
    width — which is now known without the caller stating anything."""
    tables = _tables(SETTINGS)
    assert "smoothed_mean_at_pct_Si" in tables.regions.columns
    row = next(r for r in tables.regions.rows if r["region_id"] == 1)
    # Smoothing drags a small particle toward the matrix — the measured effect
    # that makes the basis flag necessary in the first place.
    assert row["smoothed_mean_at_pct_Si"] < row["mean_at_pct_Si"]


def test_the_recorded_width_beats_the_export_callers_and_says_so():
    sm = _tables(SETTINGS, smoothing_scale_px=2
                 ).provenance["classification"]["smoothing"]
    assert sm["scale_px"] == 5
    assert sm["source"] == "map"
    assert "asked for 2 px" in sm["note"]


def test_a_map_with_no_record_keeps_the_honest_note():
    c = _tables(None).provenance["classification"]
    sm = c["smoothing"]
    assert sm["scale_px"] is None
    assert sm["box_um"] is None
    assert sm["source"] is None
    assert "not persisted" in sm["note"]
    assert c["mode"] is None
    assert c["settings_source"] is None
    assert "carries no record" in c["settings_note"]


def test_a_map_with_no_record_still_takes_the_callers_width_as_the_callers():
    sm = _tables(None, smoothing_scale_px=2
                 ).provenance["classification"]["smoothing"]
    assert sm["scale_px"] == 2
    assert sm["source"] == "caller"
    assert sm["box_um"]["x"] == pytest.approx(2 * STEP)
    assert "supplied by the export caller" in sm["note"]


def test_a_pixel_mode_map_reports_that_nothing_was_smoothed():
    """Recorded, not unknown — so the caller's width must not fill the hole."""
    sm = _tables({**SETTINGS, "mode": "pixel", "scale": None},
                 smoothing_scale_px=2
                 ).provenance["classification"]["smoothing"]
    assert sm["scale_px"] is None
    assert sm["source"] == "map"
    assert "classified per pixel" in sm["note"]


def test_no_step_size_means_no_box_rather_than_a_bare_pixel_count():
    """A width in pixels is all this file can honestly support; inventing the
    micron edge would be exactly the number a reader would quote."""
    sm = _tables({**SETTINGS, "step_x_um": None, "step_y_um": None},
                 step=None).provenance["classification"]["smoothing"]
    assert sm["scale_px"] == 5
    assert sm["box_um"] is None
    assert "um box" not in sm["note"]


def test_a_recorded_none_beats_a_stale_echo_from_the_export_dialog():
    """"Nothing was pinned" is a statement. A dialog echo must not overwrite
    it with a number nobody chose."""
    c = _tables(SETTINGS, requested={"n_clusters": 8, "mode": "pixel"}
                ).provenance["classification"]
    assert c["n_clusters_requested"] is None
    assert c["mode"] == "cluster"


def test_the_export_can_state_the_box_after_a_backend_restart(tmp_path):
    """The whole point, end to end: classify, drop the process, reload from the
    sidecar, export — and the record is still complete."""
    f = tmp_path / "scan.h5oina"
    f.write_bytes(b"key")
    store = PhaseMapStore()
    _classify(store, file_path=str(f))
    store.save_to_disk(str(f))

    fresh = PhaseMapStore()                       # a "restarted backend"
    assert fresh.load_from_disk(str(f)) is True
    geom = X.ExportGeometry(n_rows=N_ROWS, n_cols=N_COLS, step_x_um=STEP,
                            step_y_um=STEP, source_path=str(f))
    prov = X.build_tables(fresh.get_state(), _at_maps(), geometry=geom,
                          options=X.ExportOptions(hash_source=False)).provenance
    sm = prov["classification"]["smoothing"]
    assert sm["scale_px"] == 5
    assert sm["box_um"]["x"] == pytest.approx(5 * STEP)
    assert prov["classification"]["mode"] == "cluster"
