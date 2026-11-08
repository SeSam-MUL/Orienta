"""Applying a calibration is always asked for, per named dataset, and checked against the settings."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import pc_calibration_apply as use  # noqa: E402
from backend.api.services import pc_calibrations as cal  # noqa: E402
from backend.api.services.calibration_store import CalibrationStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CAL_PC = (0.5026, 0.3272, 0.8458)
#: what ``followers`` says of scan_b when it holds one PC and no map
B_ONLY = [{"dataset": "scan_b", "per_pixel": False, "map_kind": None}]


@pytest.fixture(autouse=True)
def clean(tmp_path):
    cal.set_calibrations_dir_for_test(tmp_path / "Calibrations")
    use.forget_all()
    yield
    use.forget_all()
    cal.set_calibrations_dir_for_test(None)


def _signal(wd=15.0, shape=(128, 156), tilt=4.32, pc=(0.5, 0.4, 0.6)):
    import kikuchipy as kp
    from kikuchipy.detectors import EBSDDetector
    det = EBSDDetector(shape=shape, pc=pc, sample_tilt=70.0, tilt=tilt)
    sig = kp.signals.EBSD(np.zeros((2, 2) + shape, np.uint8), detector=det)
    if wd is not None:
        sig.metadata.set_item("Acquisition_instrument.SEM.working_distance", wd)
    return sig


@pytest.fixture
def world(tmp_path):
    """A store holding two files with the same settings, one with another working distance, and a calibration."""
    store = CalibrationStore()
    files = {}
    for name, wd in (("scan_a", 15.0), ("scan_b", 15.05), ("scan_far", 20.0)):
        path = tmp_path / f"{name}.h5oina"
        path.write_bytes(name.encode() * 50)
        files[name] = path
        store.register(name, _signal(wd=wd), file_key=str(path))
    made = cal.create(
        name="REM 2026-10-14", pc=CAL_PC, origin={"method": "find", "refinement": {"n_patterns": 9, "method": "PSO"}},
        features=cal.features_of_signal(_signal(wd=15.0)))
    return store, made, files


def test_an_applied_pc_reaches_the_dataset_is_remembered_and_makes_the_file_a_member(world):
    store, made, files = world
    out = use.apply(made["id"], ["scan_a"], store=store)
    assert out == [{"dataset": "scan_a", "status": "applied", "pc": list(CAL_PC), "version": 1,
                    "mismatch_accepted": False, "pc_map_action": "none", "pc_map_actions": {"scan_a": "single"},
                    "previous": {"pc": [0.5, 0.4, 0.6], "had_pc": True}}]    # the header PC of _signal()
    np.testing.assert_allclose(store.get_pc("scan_a"), CAL_PC, atol=1e-12)
    assert store.get_entry("scan_a").pc_source == "calibration"
    assert "calibration" in store.USER_MADE_SOURCES         # so that quick-loading the file never replaces it
    assert use.current_ref("scan_a", store) == {"id": made["id"], "name": "REM 2026-10-14", "version": 1}
    members = cal.get(made["id"])["members"]
    assert [(m["name"], m["version"]) for m in members] == [("scan_a.h5oina", 1)]
    # the refinement that made the PC is credited on the file that took it
    assert store.get_entry("scan_a").pc_refinement == {"n_patterns": 9, "method": "PSO"}


def test_applying_a_calibration_to_a_file_without_a_pc_lifts_the_refusal_to_index(world, tmp_path):
    from backend.api.services import pc_guard
    store, made, _ = world
    path = tmp_path / "nopc.h5oina"
    path.write_bytes(b"nopc" * 50)
    sig = _signal(wd=15.0)
    sig.metadata.set_item("Signal.pc_source", "default")            # what a loader without a PC sets
    store.register("nopc", sig, file_key=str(path))
    assert store.get_entry("nopc").pc_source == "missing"
    assert pc_guard.is_pc_missing("nopc", store) is True
    out = use.apply(made["id"], ["nopc"], store=store)
    assert out[0]["status"] == "applied"
    assert pc_guard.is_pc_missing("nopc", store) is False
    np.testing.assert_allclose(store.get_pc("nopc"), CAL_PC, atol=1e-12)


def test_only_the_named_datasets_change(world):
    store, made, _ = world
    before_b = store.get_pc("scan_b").copy()
    use.apply(made["id"], ["scan_a"], store=store)
    np.testing.assert_allclose(store.get_pc("scan_b"), before_b)
    assert use.current_ref("scan_b", store) is None


def test_nothing_is_applied_without_a_name(world):
    store, made, _ = world
    with pytest.raises(cal.CalibrationError) as exc:
        use.apply(made["id"], [], store=store)
    assert exc.value.code == "needsTargets"
    with pytest.raises(cal.CalibrationError) as exc:
        use.apply("no-such-abc123", ["scan_a"], store=store)
    assert exc.value.code == "unknownCalibration"


def test_a_dataset_with_other_settings_is_refused_with_what_differs_and_nothing_is_written(world):
    store, made, _ = world
    before = store.get_pc("scan_far").copy()
    out = use.apply(made["id"], ["scan_far"], store=store)
    assert out[0]["status"] == "mismatch" and out[0]["unverified"] is False
    assert [d["key"] for d in out[0]["differences"]] == ["working_distance_mm"]
    assert out[0]["differences"][0]["calibration"] == 15.0 and out[0]["differences"][0]["file"] == 20.0
    np.testing.assert_allclose(store.get_pc("scan_far"), before)
    assert cal.get(made["id"])["members"] == []


def test_naming_the_dataset_a_second_time_as_accepted_applies_it_anyway(world):
    store, made, _ = world
    out = use.apply(made["id"], ["scan_far", "scan_a"], accept_mismatch=["scan_far"], store=store)
    assert [(o["dataset"], o["status"], o["mismatch_accepted"]) for o in out] == [
        ("scan_far", "applied", True), ("scan_a", "applied", False)]
    np.testing.assert_allclose(store.get_pc("scan_far"), CAL_PC, atol=1e-12)


def test_acceptance_does_not_carry_over_to_another_dataset(world):
    store, made, _ = world
    out = use.apply(made["id"], ["scan_far"], accept_mismatch=["scan_a"], store=store)
    assert out[0]["status"] == "mismatch"


def test_an_entry_whose_settings_could_not_be_read_cannot_be_checked_and_needs_acceptance(world, tmp_path):
    store, made, _ = world
    store.get_entry("scan_a").features = None
    out = use.apply(made["id"], ["scan_a"], store=store)
    assert out[0]["status"] == "mismatch" and out[0]["unverified"] is True and out[0]["differences"] == []
    out = use.apply(made["id"], ["scan_a"], accept_mismatch=["scan_a"], store=store)
    assert out[0]["status"] == "applied"


def test_a_dataset_that_is_not_loaded_and_a_copy_are_reported_not_written(world):
    store, made, _ = world
    store.register_derived("scan_a_copy", "scan_a")
    out = use.apply(made["id"], ["nope", "scan_a_copy"], store=store)
    assert [(o["dataset"], o["status"]) for o in out] == [("nope", "notLoaded"), ("scan_a_copy", "isCopy")]


def test_the_copies_of_an_applied_file_follow_it(world):
    store, made, _ = world
    store.register_derived("scan_a_bg", "scan_a")
    use.apply(made["id"], ["scan_a"], store=store)
    np.testing.assert_allclose(store.get_pc("scan_a_bg"), CAL_PC, atol=1e-12)
    assert use.current_ref("scan_a_bg", store)["id"] == made["id"]


def test_a_pc_typed_or_refined_afterwards_ends_the_membership(world):
    store, made, _ = world
    use.apply(made["id"], ["scan_a"], store=store)
    store.update_pc("scan_a", (0.51, 0.33, 0.85), source="refined")
    assert use.current_ref("scan_a", store) is None
    assert use.usage_of("scan_a", store) == {"id": made["id"], "version": 1}      # it is remembered where it came from
    store.update_pc("scan_a", CAL_PC, source="manual")                              # typed back to the very same PC
    assert use.current_ref("scan_a", store)["version"] == 1


# ------------------------------------------------------------------ the PC page and the per-pixel map

# PC(row, col) of a 2 x 2 map: (0.50 + 0.01 col, 0.40 + 0.02 row, 0.60); its mean is (0.505, 0.41, 0.60)
def _plane():
    rows, cols = np.mgrid[0:2, 0:2]
    return np.stack([0.50 + 0.01 * cols, 0.40 + 0.02 * rows, 0.60 + 0 * rows], axis=-1)


PLANE_OFFSETS = _plane() - np.array([0.505, 0.41, 0.60])


def _page(monkeypatch, sessions):
    """The PC page's sessions: ``{dataset: detector pc}``; each is a real session holding a full sim cache."""
    from kikuchipy.detectors import EBSDDetector
    from backend.api.routes import pcrefinement as R
    from pc_controller import PCController
    made = {}
    for name, pc in sessions.items():
        ctrl = PCController()
        det = EBSDDetector(shape=(60, 60), pc=pc, sample_tilt=70.0)
        if np.asarray(pc).ndim == 3:
            ctrl.apply_pixelwise_pc(det)
        else:
            ctrl.attach_detector(det)
        made[name] = R._PCSession(ctrl=ctrl, sim_cache={"stale": 1})
    monkeypatch.setattr(R, "_sessions", made)
    return made


def test_the_named_datasets_own_pc_page_takes_the_pc_and_its_cache_is_emptied(world, monkeypatch):
    """A refine updates its own controller; a calibration does not, so the apply must: otherwise the page shows the
    old PC while the store, indexing and export use the new one, and the next refine starts from the old PC."""
    store, made, _ = world
    store.register_derived("scan_a_bg", "scan_a")
    sessions = _page(monkeypatch, {"scan_a": (0.5, 0.4, 0.6), "scan_a_bg": (0.5, 0.4, 0.6)})
    out = use.apply(made["id"], ["scan_a", "scan_b"], store=store)          # scan_b has no session at all
    assert [(o["dataset"], o["status"]) for o in out] == [("scan_a", "applied"), ("scan_b", "applied")]
    for name in ("scan_a", "scan_a_bg"):                                    # the named one, and its follower
        np.testing.assert_allclose(np.asarray(sessions[name].ctrl.detector.pc).ravel(), CAL_PC, atol=1e-12)
        assert sessions[name].sim_cache == {}
    np.testing.assert_allclose(store.get_pc("scan_b"), CAL_PC, atol=1e-12)


def test_a_header_map_is_shifted_to_the_calibrations_pc_kept_and_reported(world, tmp_path):
    store, made, _ = world
    path = tmp_path / "map.h5oina"
    path.write_bytes(b"map" * 50)
    store.register("scan_map", _signal(wd=15.0, pc=_plane()), file_key=str(path))
    assert store.get_entry("scan_map").pc_map is not None and store.get_entry("scan_map").pc_source == "header"
    out = use.apply(made["id"], ["scan_map"], store=store)
    assert out[0]["status"] == "applied"
    assert out[0]["pc_map_action"] == "shifted" and out[0]["pc_map_actions"] == {"scan_map": "shifted"}
    kept = store.get_entry("scan_map").pc_map
    assert kept is not None                                                  # the map is still there ...
    np.testing.assert_allclose(kept.reshape(-1, 3).mean(axis=0), CAL_PC, atol=1e-12)   # ... its mean is the PC
    np.testing.assert_allclose(kept - np.asarray(CAL_PC), PLANE_OFFSETS, atol=1e-12)    # ... and its shape too
    assert use.current_ref("scan_map", store)["id"] == made["id"]


def test_a_refined_map_is_replaced_and_the_answer_says_so_for_every_dataset_written(world, monkeypatch):
    """A grid calibration's work is gone after this apply, and the person is told."""
    store, made, _ = world
    store.update_pc_map("scan_a", _plane())                                  # source "refined_map"
    store.register_derived("scan_a_bg", "scan_a")
    sessions = _page(monkeypatch, {"scan_a": _plane()})
    out = use.apply(made["id"], ["scan_a"], store=store)
    assert out[0]["pc_map_action"] == "replaced"
    assert out[0]["pc_map_actions"] == {"scan_a": "replaced", "scan_a_bg": "shifted"}   # the copy's inherited map moves
    assert store.get_entry("scan_a").pc_map is None
    np.testing.assert_allclose(np.asarray(store.get_detector("scan_a").pc).ravel(), CAL_PC, atol=1e-12)
    # the PC page's controller lost its map too
    from pc_controller import pc_is_per_pixel
    assert not pc_is_per_pixel(sessions["scan_a"].ctrl.detector)
    np.testing.assert_allclose(np.asarray(sessions["scan_a"].ctrl.detector.pc).ravel(), CAL_PC, atol=1e-12)


def test_a_file_with_one_pc_reports_no_map_action(world):
    store, made, _ = world
    out = use.apply(made["id"], ["scan_b"], store=store)
    assert out[0]["pc_map_action"] == "none" and out[0]["pc_map_actions"] == {"scan_b": "single"}


# ------------------------------------------------------------------ refining at a member

def test_followers_are_the_other_datasets_on_exactly_the_same_version(world):
    store, made, _ = world
    use.apply(made["id"], ["scan_a", "scan_b"], store=store)
    assert use.followers("scan_a", store) == {
        "calibration": {"id": made["id"], "name": "REM 2026-10-14", "version": 1, "latest": 1},
        "diverged": False, "followers": B_ONLY}
    store.update_pc("scan_a", (0.51, 0.33, 0.85), source="refined")                 # a refine at scan_a
    got = use.followers("scan_a", store)
    assert got["diverged"] is True and got["followers"] == B_ONLY
    np.testing.assert_allclose(store.get_pc("scan_b"), CAL_PC, atol=1e-12)           # scan_b was not touched


def test_a_follower_says_whether_it_holds_a_map_and_of_which_kind(world, tmp_path):
    """A new version REPLACES a map from the grid calibration and MOVES one the file brought: the question has to say
    which of its files has which. Hand values: a file with one PC has no map; a header map is "header"; a map the
    grid calibration made (``refined_map``) is "grid"."""
    store, made, _ = world
    for name in ("scan_map", "scan_grid"):
        path = tmp_path / f"{name}.h5oina"
        path.write_bytes(name.encode() * 50)
        store.register(name, _signal(wd=15.0, pc=_plane()), file_key=str(path))
    use.apply(made["id"], ["scan_a", "scan_b", "scan_map", "scan_grid"], store=store)
    # scan_grid: a grid calibration after the apply that left the mean where the calibration's PC is
    store.update_pc_map("scan_grid", np.asarray(CAL_PC) + PLANE_OFFSETS)
    assert store.get_entry("scan_grid").pc_source == "refined_map"
    assert use.current_ref("scan_grid", store) is not None
    store.update_pc("scan_a", (0.51, 0.33, 0.85), source="refined")
    got = use.followers("scan_a", store)["followers"]
    assert got == [
        {"dataset": "scan_b", "per_pixel": False, "map_kind": None},
        {"dataset": "scan_grid", "per_pixel": True, "map_kind": "grid"},
        {"dataset": "scan_map", "per_pixel": True, "map_kind": "header"},
    ]


def test_a_dataset_that_never_used_a_calibration_has_no_followers(world):
    store, made, _ = world
    assert use.followers("scan_a", store) == {"calibration": None, "diverged": False, "followers": []}


def test_a_new_version_moves_the_datasets_that_are_applied_to_it(world):
    store, made, _ = world
    use.apply(made["id"], ["scan_a", "scan_b"], store=store)
    store.update_pc("scan_a", (0.51, 0.33, 0.85), source="refined")
    cal.add_version(made["id"], pc=(0.51, 0.33, 0.85), origin={"method": "refine", "dataset": "scan_a"})
    out = use.apply(made["id"], ["scan_a", "scan_b"], store=store)
    assert [o["version"] for o in out] == [2, 2]
    np.testing.assert_allclose(store.get_pc("scan_b"), (0.51, 0.33, 0.85), atol=1e-12)
    assert use.current_ref("scan_b", store)["version"] == 2
    assert cal.version_pc(cal.get(made["id"]), 1) == list(CAL_PC)                   # the old PC is still readable


# ------------------------------------------------------------------ after a restart

def test_after_a_restart_a_member_with_exactly_the_calibrations_pc_is_still_recognised(world):
    """``forget_all`` is what a restart does to the in-memory gifts; the calibration file still lists the files."""
    store, made, _ = world
    use.apply(made["id"], ["scan_a", "scan_b"], store=store)
    use.forget_all()
    assert use._USAGE == {}                                                 # the memory really is empty
    assert use.current_ref("scan_a", store) == {"id": made["id"], "name": "REM 2026-10-14", "version": 1}
    assert use.followers("scan_a", store)["followers"] == B_ONLY
    assert use.followers("scan_a", store)["diverged"] is False
    store.update_pc("scan_a", (0.51, 0.33, 0.85), source="refined")
    assert use.current_ref("scan_a", store) is None                         # another PC: no longer on the version


def test_after_a_restart_a_refine_at_a_member_still_finds_it_diverged_and_its_followers(world):
    store, made, _ = world
    use.apply(made["id"], ["scan_a", "scan_b"], store=store)
    use.forget_all()
    store.update_pc("scan_a", (0.51, 0.33, 0.85), source="refined")        # the refine, after the restart
    got = use.followers("scan_a", store)
    assert got["calibration"] == {"id": made["id"], "name": "REM 2026-10-14", "version": 1, "latest": 1}
    assert got["diverged"] is True and got["followers"] == B_ONLY
    np.testing.assert_allclose(store.get_pc("scan_b"), CAL_PC, atol=1e-12)


def test_the_same_pc_on_a_file_that_was_never_a_member_is_not_recognised(world):
    """Another file IS a member (so the calibration has a member list and a restart has something to read); the
    file whose PC was typed in is not one of them."""
    store, made, _ = world
    use.apply(made["id"], ["scan_a"], store=store)
    use.forget_all()
    store.update_pc("scan_b", CAL_PC, source="manual")        # typed in: exactly the calibration's PC, no membership
    assert use.current_ref("scan_b", store) is None
    assert use.usage_of("scan_b", store) is None
    assert use.followers("scan_b", store) == {"calibration": None, "diverged": False, "followers": []}


def test_a_file_that_states_none_of_its_settings_cannot_be_checked_and_needs_acceptance(world, tmp_path):
    """An .ebsp states no tilt and no binning (``pc_calibrations.settings_stated``): nothing but its pattern size
    can be compared with the calibration, so it needs the second naming like a file whose settings differ."""
    store, made, _ = world
    path = tmp_path / "pat.ebsp"
    path.write_bytes(b"pat" * 50)
    sig = _signal(wd=None)
    sig.metadata.set_item("General.original_filename", str(path))
    store.register("pat", sig, file_key=str(path))
    assert [store.get_entry("pat").features[k] for k in cal.SETTING_KEYS] == [None] * len(cal.SETTING_KEYS)
    out = use.apply(made["id"], ["pat"], store=store)
    assert out[0]["status"] == "mismatch" and out[0]["unverified"] is True and out[0]["differences"] == []
    got = {c["dataset"]: c for c in use.candidates(made["id"], store)}
    assert got["pat"]["match"] is False and got["pat"]["unverified"] is True
    assert got["pat"]["checked"] == ["pattern_shape"]            # all that could be compared
    assert use.apply(made["id"], ["pat"], accept_mismatch=["pat"], store=store)[0]["status"] == "applied"


def test_a_dataset_without_a_file_is_remembered_by_the_session_alone(world):
    """No path means no fingerprint, so no member list can hold it: only the memory of this process knows."""
    store, made, _ = world
    store.register("no_path", _signal(wd=15.0))                          # registered without a file_key
    use.apply(made["id"], ["no_path"], store=store)
    assert use.current_ref("no_path", store)["id"] == made["id"]
    assert cal.get(made["id"])["members"] == []
    use.forget_all()
    assert use.current_ref("no_path", store) is None                     # a restart cannot know it


# ------------------------------------------------------------------ the list to tick from

def test_candidates_mark_matching_files_and_leave_out_copies(world):
    store, made, _ = world
    store.register_derived("scan_a_copy", "scan_a")
    use.apply(made["id"], ["scan_a"], store=store)
    got = {c["dataset"]: c for c in use.candidates(made["id"], store)}
    assert set(got) == {"scan_a", "scan_b", "scan_far"}
    assert got["scan_a"]["match"] is True and got["scan_a"]["on_this_version"] is True
    assert got["scan_b"]["match"] is True and got["scan_b"]["on_this_version"] is False     # 15.05 vs 15.0: within 0.1
    assert got["scan_far"]["match"] is False
    assert [d["key"] for d in got["scan_far"]["differences"]] == ["working_distance_mm"]
    assert got["scan_far"]["has_pc"] is True
    # these signals state only a working distance (no file name: the unified-loader shape), plus the pattern shape
    assert got["scan_a"]["checked"] == got["scan_far"]["checked"] == ["pattern_shape", "working_distance_mm"]


def test_candidates_carry_what_the_apply_table_shows(world):
    """The table "file, working distance, old PC (version), new PC" reads these: the PC the file has now, the
    version it still runs on, and the settings it states."""
    store, made, _ = world
    use.apply(made["id"], ["scan_a"], store=store)
    got = {c["dataset"]: c for c in use.candidates(made["id"], store)}
    assert got["scan_a"]["pc"] == pytest.approx(CAL_PC)
    assert got["scan_a"]["ref"] == {"id": made["id"], "name": "REM 2026-10-14", "version": 1}
    # scan_b was never given the calibration: its own PC (the detector's 0.5, 0.4, 0.6), on no version
    assert got["scan_b"]["pc"] == pytest.approx([0.5, 0.4, 0.6]) and got["scan_b"]["ref"] is None
    assert got["scan_b"]["settings"]["file"]["working_distance_mm"] == 15.05
    assert got["scan_far"]["settings"]["file"]["working_distance_mm"] == 20.0
    assert got["scan_far"]["settings"]["file"]["pattern_shape"] == [128, 156]
    assert got["scan_far"]["settings"]["file"]["binning"] is None            # not stated by these signals
    # a file without a PC has none to show
    store.get_entry("scan_b").pc_source = "missing"
    assert {c["dataset"]: c for c in use.candidates(made["id"], store)}["scan_b"]["pc"] is None


# ------------------------------------------------------------------ a dataset made from a calibration

def test_a_calibration_made_on_a_dataset_is_that_datasets_version(world):
    store, made, files = world
    other = cal.create(name="From scan_a", pc=tuple(store.get_pc("scan_a")),
                       features=store.get_entry("scan_a").features, origin={"method": "typed"})
    use.note_dataset(other["id"], "scan_a", 1, store=store)
    assert use.current_ref("scan_a", store)["id"] == other["id"]
    assert [m["name"] for m in cal.get(other["id"])["members"]] == ["scan_a.h5oina"]


# ------------------------------------------------------------------ the rule, held by the source

def test_loading_a_file_never_consults_the_calibrations(world, tmp_path):
    """The load path registers a file in the store; a matching calibration exists; the PC stays the file's."""
    store, made, _ = world
    path = tmp_path / "fresh.h5oina"
    path.write_bytes(b"fresh" * 30)
    entry = store.register("fresh", _signal(wd=15.0, pc=(0.47, 0.31, 0.82)), file_key=str(path))
    np.testing.assert_allclose(entry.pc_single, (0.47, 0.31, 0.82))
    assert use.current_ref("fresh", store) is None


def _calls_apply(source: str) -> bool:
    """Does this module use ``pc_calibration_apply.apply``, however it got hold of it? An AST walk, not a regex:
    ``use.apply(...)``, ``from ...pc_calibration_apply import apply as give`` and ``give(...)``, the function taken
    without being called (``fn = use.apply``), ``import ...pc_calibration_apply as m``, the dotted name, and
    ``importlib.import_module("...pc_calibration_apply").apply``."""
    import ast
    tree = ast.parse(source)
    modules, functions = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[-1] == "pc_calibration_apply" and alias.asname:
                    modules.add(alias.asname)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "pc_calibration_apply":
                    modules.add(alias.asname or alias.name)
                elif (node.module or "").split(".")[-1] == "pc_calibration_apply" and alias.name == "apply":
                    functions.add(alias.asname or alias.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "apply":
            owner = node.value
            if isinstance(owner, ast.Name) and owner.id in modules:
                return True
            if isinstance(owner, ast.Attribute) and owner.attr == "pc_calibration_apply":
                return True
            if isinstance(owner, ast.Call) and any(
                    isinstance(c, ast.Constant) and isinstance(c.value, str) and "pc_calibration_apply" in c.value
                    for c in ast.walk(owner)):
                return True
        if isinstance(node, ast.Name) and node.id in functions and isinstance(node.ctx, ast.Load):
            return True
    return False


@pytest.mark.parametrize("source, expected", [
    ("from backend.api.services import pc_calibration_apply as use\nuse.apply('x', ['a'])", True),
    ("from backend.api.services.pc_calibration_apply import apply as give\ngive('x', ['a'])", True),
    ("from backend.api.services import pc_calibration_apply as use\nfn = use.apply", True),
    ("import backend.api.services.pc_calibration_apply as m\nm.apply('x', ['a'])", True),
    ("import backend.api.services.pc_calibration_apply\nbackend.api.services.pc_calibration_apply.apply('x', [])", True),
    ("import importlib\nimportlib.import_module('backend.api.services.pc_calibration_apply').apply('x', [])", True),
    ("from backend.api.services import pc_calibration_apply as use\nuse.current_ref('a')", False),
    ("def apply(x):\n    return x\napply(1)", False),
])
def test_the_scan_for_callers_of_apply_sees_through_aliases(source, expected):
    assert _calls_apply(source) is expected


def test_only_the_calibration_routes_apply_a_calibration():
    callers = []
    for base in (ROOT, ROOT / "backend", ROOT / "analysis"):
        for path in (base.glob("*.py") if base == ROOT else base.glob("**/*.py")):
            if "tests" in path.parts or path.name == "pc_calibration_apply.py":
                continue
            try:
                found = _calls_apply(path.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            if found:
                callers.append(path.relative_to(ROOT).as_posix())
    # the calibration routes are the only caller; before they exist there is none
    assert set(callers) <= {"backend/api/routes/pc_calibrations.py"}, callers
