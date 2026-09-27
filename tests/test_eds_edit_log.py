"""A phase map has to be able to say what was done to it by hand.

Six ways exist to put a thumb on an EDS phase map — paint pixels, name a
region, merge, split, grow, snap edges — and exactly one of them, painting,
used to leave a trace (``locked_mask``). The export therefore wrote:

    "hand_edits": {"pixels_painted": 321, "regions_named": null,
                   "merges": null, "splits": null, "boundary_edits": null}

which is honest about a hole. Three testers hit it independently. A failure
analyst could not write the one line his casework needs — *"142 particles;
7 phase assignments changed by hand; 3 regions merged; 0 boundaries edited"* —
and a group leader signing off on a student's map put the consequence plainly:
*"if the merges field is blank I cannot tell whether they merged nothing or
merged forty times until the answer looked right."*

What this file pins:

* every mutating method leaves an entry, because they all go through
  ``_snapshot`` and that is where the recording happens;
* the counters are **tallied as entries are recorded**, not by counting the
  retained list — the list is capped, and counting it would undercount
  exactly the heavily-edited map whose count matters most;
* **undo rolls the log back**. ``_snapshot`` rebuilds the state field by
  field, so a new field missing from it is silently reset by every undo — and
  ``undo`` autosaves, so the empty version reaches disk. This repo has already
  been bitten by that twice (``region_defs``/``element_weights``, then
  ``settings``). Mutation-tested: see the docstring on the undo test.
* the sidecar schema stays **4**. The log rides in the JSON meta as optional
  keys; a bump would discard every map already on a user's disk.
* a map from before this feature reports **null**, never 0. "Not recorded" and
  "none happened" are different answers, and collapsing them would rebuild the
  exact hole this feature fills.
* the region grid carries an "edited" flag, because region-derived particle
  ids are ordered by ``(region_id, -n_px, centroid)`` — a merge pops an id and
  renumbers everything above it, so ids burned onto a report figure before a
  boundary edit no longer point at the same particles.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.routes import eds as eds_routes
from backend.api.services.cif_phase_library import CifPhaseEntry
from backend.api.services.phase_map_store import (
    _EDIT_LOG_CAP, _SIDECAR_SCHEMA, PhaseMapState, PhaseMapStore,
    _sidecar_path_for, get_phase_map_store,
)

N_ROWS = N_COLS = 8


def _entry(name, comp):
    return CifPhaseEntry(
        key=name, cif_filename=name, formula=name, space_group="Fm-3m",
        space_group_number=225, crystal_system="cubic", composition=comp,
        elements=sorted(comp),
    )


AL = _entry("Al.cif", {"Al": 100.0})
SI = _entry("Si.cif", {"Si": 100.0})
ENTRIES = [AL, SI]

#: Three regions in vertical stripes, so a merge, a grow and a snap all have
#: somewhere to move to. Region 1 is the "particle".
def _region_grid():
    region = np.zeros((N_ROWS, N_COLS), dtype=np.int32)
    region[:, 3:5] = 1
    region[:, 5:] = 2
    return region


def _features():
    """Composition per pixel, matching the stripes, with a gradient inside
    region 1 so a split has two things to find."""
    region = _region_grid().ravel()
    si = np.where(region == 1, 60.0, np.where(region == 2, 20.0, 5.0))
    # Top half of the particle is richer — the two parts a split should find.
    rows = np.repeat(np.arange(N_ROWS), N_COLS)
    si = si + np.where((region == 1) & (rows < N_ROWS // 2), 25.0, 0.0)
    return np.stack([100.0 - si, si], axis=1)


def _store(file_path=None):
    store = PhaseMapStore()
    region = _region_grid()
    store.set_classification(
        phase_grid=np.where(region == 1, 1, 0).astype(np.int32),
        score_grid=np.full((N_ROWS, N_COLS), 0.8, dtype=np.float32),
        phase_entries=list(ENTRIES), tolerance=15.0, min_score=0.3,
        file_path=file_path,
        region_grid=region, region_phase=[0, 1, 0],
        region_features=_features(),
    )
    return store


def _ops(store):
    return [e["op"] for e in store.get_state().edit_log]


def _counts(store):
    return store.get_state().edit_summary()["counts"]


# ---------------------------------------------------------------------------
# the shape of the record
# ---------------------------------------------------------------------------

def test_a_fresh_classification_starts_an_empty_but_tracked_record():
    """Tracked with all-zero counts, which is a measurement — as opposed to
    the null-valued record an untracked map reports."""
    s = _store().get_state()
    assert s.edit_log == []
    assert s.edit_log_total == 0
    assert s.edits_tracked is True
    assert s.region_grid_edited is False
    summary = s.edit_summary()
    assert summary["tracked"] is True
    assert summary["n_edits"] == 0
    assert summary["counts"] == {
        "regions_named": 0, "merges": 0, "splits": 0, "grows": 0,
        "edge_snaps": 0, "paints": 0, "phase_replacements": 0,
    }
    assert summary["log_truncated"] is False
    assert summary["region_grid_edited"] is False


def test_naming_a_region_is_recorded_with_the_name_not_just_an_index():
    """An index is useless in a record — the list it points into is rebuilt by
    the next classification. The reader needs "region 1 named Si.cif"."""
    store = _store()
    n = store.assign_region_phase(1, 1)
    assert n > 0
    entry, = store.get_state().edit_log
    assert entry["op"] == "name_region"
    assert entry["detail"]["region_id"] == 1
    assert entry["detail"]["phase_index"] == 1
    assert entry["detail"]["phase"] == "Si.cif"
    assert entry["detail"]["n_px"] == n
    assert _counts(store)["regions_named"] == 1
    # Naming does not move a pixel between regions, so ids stay stable.
    assert store.get_state().region_grid_edited is False


def test_a_merge_records_which_regions_and_how_many_pixels_moved():
    store = _store()
    moved = store.merge_regions(0, 1)
    entry, = store.get_state().edit_log
    assert entry["op"] == "merge"
    assert entry["detail"]["keep_id"] == 0
    assert entry["detail"]["drop_id"] == 1
    assert entry["detail"]["pixels_moved"] == moved == N_ROWS * 2
    assert entry["detail"]["n_regions_after"] == 2
    assert _counts(store)["merges"] == 1


def test_a_merge_flags_that_region_ids_were_renumbered():
    """The detectability half of the particle-id problem. Fixing id stability
    belongs to the export; saying "the grid moved under you" belongs here."""
    store = _store()
    assert store.get_state().region_grid_edited is False
    store.merge_regions(0, 1)
    state = store.get_state()
    assert state.region_grid_edited is True
    assert state.edit_summary()["region_grid_edited"] is True
    # And the record says which ids the renumbering started at.
    assert state.edit_log[0]["detail"]["region_ids_renumbered_above"] == 1


def test_a_split_records_the_parts_it_created():
    store = _store()
    n_new = store.split_region(1, 2)
    assert n_new == 2
    entry, = store.get_state().edit_log
    assert entry["op"] == "split"
    assert entry["detail"]["region_id"] == 1
    assert entry["detail"]["n_parts"] == 2
    # The ids the caller will see afterwards, so the record is followable.
    assert entry["detail"]["new_region_ids"] == [1, 3]
    assert sorted(set(store.get_state().region_grid.ravel().tolist())) == [0, 1, 2, 3]
    assert _counts(store)["splits"] == 1
    assert store.get_state().region_grid_edited is True


def test_growing_a_boundary_records_the_direction_and_the_pixels_moved():
    store = _store()
    changed = store.grow_region(1, 1)
    entry, = store.get_state().edit_log
    assert entry["op"] == "grow"
    assert entry["detail"]["region_id"] == 1
    assert entry["detail"]["steps"] == 1
    assert entry["detail"]["direction"] == "grow"
    assert entry["detail"]["pixels_changed"] == changed > 0
    assert _counts(store)["grows"] == 1
    assert store.get_state().region_grid_edited is True

    store.grow_region(1, -1)
    assert store.get_state().edit_log[-1]["detail"]["direction"] == "shrink"
    assert _counts(store)["grows"] == 2


def _snap_store():
    """Two wide regions whose boundary sits one column off the chemical edge.

    Snapping is the only boundary tool that looks at the data, so a fixture
    whose boundary is already on the gradient measures nothing — it returns 0
    and never reaches the recording path.
    """
    store = PhaseMapStore()
    region = np.zeros((N_ROWS, N_COLS), dtype=np.int32)
    region[:, 4:] = 1                                 # regions meet at col 4
    cols = np.tile(np.arange(N_COLS), N_ROWS)
    si = np.where(cols >= 6, 80.0, 5.0)               # chemistry steps at col 6
    store.set_classification(
        phase_grid=region.copy(),
        score_grid=np.full((N_ROWS, N_COLS), 0.8, dtype=np.float32),
        phase_entries=list(ENTRIES), tolerance=15.0, min_score=0.3,
        region_grid=region, region_phase=[0, 1],
        region_features=np.stack([100.0 - si, si], axis=1),
    )
    return store


def test_snapping_edges_records_the_strength_it_was_given():
    pytest.importorskip("skimage")
    store = _snap_store()
    changed = store.snap_region_edges(1.0)
    assert changed > 0, "the fixture must actually move a boundary"
    entry, = store.get_state().edit_log
    assert entry["op"] == "snap_edges"
    assert entry["detail"]["strength"] == pytest.approx(1.0)
    assert entry["detail"]["radius_px"] == 1
    assert entry["detail"]["pixels_changed"] == changed
    assert _counts(store)["edge_snaps"] == 1
    assert store.get_state().region_grid_edited is True


def test_painting_is_recorded_too_by_rectangle_and_by_mask():
    """It was already countable from ``locked_mask``, but a pixel count is not
    a history: 40 px could be one lasso or forty clicks."""
    store = _store()
    store.assign_region(0, 1, 0, 1, 1)
    mask = np.zeros((N_ROWS, N_COLS), dtype=bool)
    mask[6, 6] = True
    store.assign_mask(mask, 0)

    rect, painted = store.get_state().edit_log
    assert rect["op"] == painted["op"] == "paint"
    assert rect["detail"] == {"shape": "rect", "phase_index": 1,
                              "phase": "Si.cif", "n_px": 4,
                              "bbox": [0, 0, 1, 1]}
    assert painted["detail"]["shape"] == "mask"
    assert painted["detail"]["n_px"] == 1
    assert _counts(store)["paints"] == 2
    # Painting does not touch the region grid.
    assert store.get_state().region_grid_edited is False


def test_a_map_wide_phase_replacement_is_recorded():
    store = _store()
    n = store.replace_phase(0, 1)
    entry, = store.get_state().edit_log
    assert entry["op"] == "replace_phase"
    assert entry["detail"] == {"from_phase_index": 0, "to_phase_index": 1,
                               "phase": "Si.cif", "n_px": n}
    assert _counts(store)["phase_replacements"] == 1


def test_clearing_a_pixel_records_the_decision_as_unclassified():
    """"-1" is a decision too, and reads as an index into a list otherwise."""
    store = _store()
    store.assign_region(0, 0, 0, 0, -1)
    assert store.get_state().edit_log[0]["detail"]["phase"] == "unclassified"


def test_the_analysts_summary_line_can_be_written_from_the_counts():
    """The line the failure analyst asked for, assembled end to end."""
    store = _store()
    store.assign_region_phase(1, 1)
    store.assign_region_phase(2, 0)
    store.merge_regions(0, 2)
    counts = _counts(store)
    line = (f"{counts['regions_named']} phase assignments changed by hand; "
            f"{counts['merges']} regions merged; "
            f"{counts['grows'] + counts['edge_snaps']} boundaries edited")
    assert line == ("2 phase assignments changed by hand; 1 regions merged; "
                    "0 boundaries edited")


def test_the_log_is_json_shaped():
    """It rides inside the sidecar's JSON meta blob — a detail that failed to
    serialise would lose the whole map, not just the log."""
    store = _store()
    store.assign_region_phase(1, 1)
    store.merge_regions(0, 1)
    store.grow_region(1, 1)
    round_tripped = json.loads(json.dumps(store.get_state().edit_log))
    assert round_tripped == store.get_state().edit_log


def test_a_refused_edit_leaves_no_trace():
    """A rejected call must not put a line in somebody's provenance."""
    store = _store()
    with pytest.raises(ValueError):
        store.assign_region_phase(99, 0)
    with pytest.raises(ValueError):
        store.assign_region(0, 1, 0, 1, 42)
    assert store.merge_regions(1, 1) == 0            # no-op, same id
    assert store.grow_region(1, 0) == 0              # no-op, zero steps
    assert store.get_state().edit_log == []
    assert store.get_state().edit_log_total == 0


# ---------------------------------------------------------------------------
# undo — the trap this repo has fallen into twice
# ---------------------------------------------------------------------------

def test_undo_rolls_the_log_back_to_exactly_what_came_before():
    """The mutation test for every one of the new fields.

    Deliberately TWO edits deep, and it checks the surviving entry rather than
    only the removed one: a snapshot that drops ``edit_log`` hands undo a fresh
    empty list, which looks like a correct rollback when only one edit ever
    happened. Verified by deleting each new line from ``_snapshot`` in turn —
    ``edit_log``, ``edit_log_total``, ``edit_counts``, ``edits_tracked`` — this
    test goes red for all four.
    """
    store = _store()                                  # no file_path: no autosave
    store.assign_region(0, 1, 0, 1, 1)                # edit 1: a paint
    store.assign_region_phase(1, 1)                   # edit 2: a name
    assert _ops(store) == ["paint", "name_region"]

    assert store.undo() is True
    state = store.get_state()
    assert _ops(store) == ["paint"], "undo did not restore the earlier record"
    assert state.edit_log_total == 1
    assert state.edit_counts.get("paints") == 1
    assert state.edit_counts.get("regions_named", 0) == 0
    assert state.edits_tracked is True                # the flag is not the log
    assert state.region_grid == pytest.approx(_region_grid())


def test_undo_rolls_back_the_region_edited_flag_as_well():
    """Also two deep, and in both directions: undoing a paint must not claim
    the earlier merge never moved the region grid."""
    store = _store()
    store.merge_regions(0, 1)
    store.assign_region(0, 0, 0, 0, 1)                # a paint on top
    assert store.undo() is True
    assert store.get_state().region_grid_edited is True, \
        "undo forgot that a merge had already renumbered the regions"
    # ...and taking the merge itself back does clear it.
    store2 = _store()
    store2.merge_regions(0, 1)
    assert store2.get_state().region_grid_edited is True
    assert store2.undo() is True
    assert store2.get_state().region_grid_edited is False


def test_redo_puts_the_record_back():
    """Undo of an undo is a redo — the map comes back, so the record must."""
    store = _store()
    store.assign_region_phase(1, 1)
    assert store.undo() is True
    assert store.undo() is True
    assert _ops(store) == ["name_region"]
    assert _counts(store)["regions_named"] == 1


def test_undo_after_several_edits_takes_back_exactly_one():
    store = _store()
    store.assign_region_phase(1, 1)
    store.merge_regions(0, 1)
    assert _ops(store) == ["name_region", "merge"]
    assert store.undo() is True
    assert _ops(store) == ["name_region"]
    assert store.get_state().edit_log_total == 1


def test_a_reclassify_is_not_a_hand_edit_and_starts_a_fresh_record():
    """It replaces the map, so counting it would put a false entry in the
    provenance — and the counts are "since this classification"."""
    store = _store()
    store.assign_region_phase(1, 1)
    region = _region_grid()
    store.set_classification(
        phase_grid=np.zeros((N_ROWS, N_COLS), dtype=np.int32),
        score_grid=np.full((N_ROWS, N_COLS), 0.9, dtype=np.float32),
        phase_entries=list(ENTRIES), tolerance=15.0, min_score=0.3,
        region_grid=region, region_phase=[0, 1, 0],
    )
    state = store.get_state()
    assert state.edit_log == []
    assert state.edit_log_total == 0
    assert state.edits_tracked is True
    assert state.region_grid_edited is False
    # And the previous record is still behind the undo, not destroyed.
    assert store.undo() is True
    assert _ops(store) == ["name_region"]


# ---------------------------------------------------------------------------
# the cap
# ---------------------------------------------------------------------------

def test_the_log_is_capped_but_says_so_and_keeps_counting():
    """A truncated log that looks complete is the failure this whole feature
    exists to prevent, so the total is kept beside the retained entries."""
    store = _store()
    n = _EDIT_LOG_CAP + 25
    for i in range(n):
        store.assign_region(0, 0, 0, 0, i % 2)
    state = store.get_state()
    assert len(state.edit_log) == _EDIT_LOG_CAP
    assert state.edit_log_total == n
    assert state.edit_log_truncated is True

    summary = state.edit_summary()
    assert summary["log_truncated"] is True
    assert summary["log_retained"] == _EDIT_LOG_CAP
    assert summary["n_edits"] == n
    # The counter is a tally, not a count of what survived — this is the whole
    # reason it is not derived from the retained list.
    assert summary["counts"]["paints"] == n
    assert f"of {n}" in summary["note"]


def test_the_cap_keeps_the_most_recent_edits():
    store = _store()
    for _ in range(_EDIT_LOG_CAP):
        store.assign_region(0, 0, 0, 0, 0)
    store.assign_region_phase(1, 1)                  # the newest one
    log = store.get_state().edit_log
    assert len(log) == _EDIT_LOG_CAP
    assert log[-1]["op"] == "name_region"


# ---------------------------------------------------------------------------
# the sidecar
# ---------------------------------------------------------------------------

def test_the_sidecar_schema_was_not_bumped():
    """``load_from_disk`` refuses any sidecar whose schema is not exactly this
    number, so a bump silently discards every phase map already on disk — hand
    edits and hand-written definitions included. The log rides as optional
    keys precisely so it does not cost anybody their map."""
    assert _SIDECAR_SCHEMA == 4


def test_the_record_survives_a_sidecar_round_trip(tmp_path):
    f = tmp_path / "scan.h5oina"
    f.write_bytes(b"not a real file, only a key for the sidecar")
    store = _store(file_path=str(f))
    store.assign_region_phase(1, 1)
    store.merge_regions(0, 1)
    assert store.save_to_disk(str(f)) is not None

    fresh = PhaseMapStore()
    assert fresh.load_from_disk(str(f)) is True
    state = fresh.get_state()
    assert [e["op"] for e in state.edit_log] == ["name_region", "merge"]
    assert state.edit_log[0]["detail"]["phase"] == "Si.cif"
    assert state.edit_log_total == 2
    assert state.edits_tracked is True
    assert state.region_grid_edited is True
    assert state.edit_summary()["counts"]["merges"] == 1
    assert state.edit_summary()["counts"]["regions_named"] == 1


def test_a_sidecar_written_before_this_feature_reports_null_not_zero(tmp_path):
    """Built by REMOVING the keys from a real file — hand-rolling an "old" npz
    would test a file this app never wrote.

    The distinction is the point of the feature: 0 merges means "I looked and
    there were none"; null means "nobody was watching". Collapsing them would
    rebuild the hole.
    """
    f = tmp_path / "scan.h5oina"
    f.write_bytes(b"key")
    store = _store(file_path=str(f))
    store.merge_regions(0, 1)
    sidecar = _sidecar_path_for(str(f))

    with np.load(str(sidecar), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    meta = json.loads(arrays["meta"].item())
    assert meta["schema"] == 4
    for key in ("edit_tracking", "edit_log", "edit_log_total", "edit_counts",
                "region_grid_edited"):
        del meta[key]
    arrays["meta"] = np.array(json.dumps(meta), dtype=str)
    np.savez_compressed(str(sidecar), **arrays)

    fresh = PhaseMapStore()
    assert fresh.load_from_disk(str(f)) is True      # loads, does not raise
    state = fresh.get_state()
    assert state.edit_log == []
    assert state.edits_tracked is False
    summary = state.edit_summary()
    assert summary["tracked"] is False
    assert set(summary["counts"].values()) == {None}
    assert summary["n_edits"] is None
    assert summary["region_grid_edited"] is None
    assert "would be a claim" in summary["note"]
    # And the map itself is all there.
    assert state.region_phase and state.phase_grid.shape == (N_ROWS, N_COLS)


def test_an_untracked_map_stays_untracked_when_it_is_saved_again(tmp_path):
    """Re-saving must not upgrade "nobody was watching" into "watched from the
    start" — every edit before the reload would then be claimed as absent."""
    f = tmp_path / "scan.h5oina"
    f.write_bytes(b"key")
    store = PhaseMapStore()
    region = _region_grid()
    store._state = PhaseMapState(                    # a map from before tracking
        phase_grid=np.zeros((N_ROWS, N_COLS), dtype=np.int32),
        score_grid=np.full((N_ROWS, N_COLS), 0.8, dtype=np.float32),
        phase_entries=list(ENTRIES), n_rows=N_ROWS, n_cols=N_COLS,
        tolerance=15.0, min_score=0.3,
        region_grid=region, region_phase=[0, 1, 0],
    )
    store._file_path = str(f)
    store.save_to_disk(str(f))

    fresh = PhaseMapStore()
    assert fresh.load_from_disk(str(f)) is True
    assert fresh.get_state().edits_tracked is False


def test_a_hand_built_state_claims_nothing_by_default():
    """The export is handed states built elsewhere (its own tests do). The
    default must be "not watched", not "watched and clean"."""
    state = PhaseMapState(
        phase_grid=np.zeros((2, 2), dtype=np.int32),
        score_grid=np.zeros((2, 2), dtype=np.float32),
        phase_entries=list(ENTRIES), n_rows=2, n_cols=2,
        tolerance=15.0, min_score=0.3,
    )
    assert state.edits_tracked is False
    assert set(state.edit_summary()["counts"].values()) == {None}


# ---------------------------------------------------------------------------
# the route
# ---------------------------------------------------------------------------

def test_the_state_response_carries_the_counts():
    """The export reads them off the state; the frontend reads them off this."""
    store = get_phase_map_store()
    store.clear()
    try:
        region = _region_grid()
        store.set_classification(
            phase_grid=np.where(region == 1, 1, 0).astype(np.int32),
            score_grid=np.full((N_ROWS, N_COLS), 0.8, dtype=np.float32),
            phase_entries=list(ENTRIES), tolerance=15.0, min_score=0.3,
            region_grid=region, region_phase=[0, 1, 0],
        )
        store.assign_region_phase(1, 1)
        store.merge_regions(0, 2)

        body = eds_routes._state_to_response(include_image=False)
        edits = body["hand_edits"]
        assert edits["tracked"] is True
        assert edits["counts"]["regions_named"] == 1
        assert edits["counts"]["merges"] == 1
        assert edits["counts"]["splits"] == 0
        assert edits["region_grid_edited"] is True
        assert [e["op"] for e in edits["log"]] == ["name_region", "merge"]
        # It has to survive the JSON encoder the route hands it to.
        assert json.loads(json.dumps(edits)) == edits
    finally:
        store.clear()
