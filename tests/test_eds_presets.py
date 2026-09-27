"""EDS analysis presets: the recipe as a file, and its refusal to travel blind.

The point of a preset is not that it saves typing — it is that a recipe applied
to the wrong dataset produces a plausible map from settings that cannot mean
here what they meant there. So most of what is tested below is the refusal, not
the round trip: which conditions block, which only warn, and whether the
message names the thing the user has to go and look at.

The one that is easiest to get wrong, and is therefore tested from both sides,
is ``element_set_differs`` on a SUPERSET. at% is renormalised over the measured
elements, so one extra measured element moves every other element's at% and
every absolute window silently means a different composition. "More data" is
not "compatible".
"""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.services import eds_presets as ep
from backend.api.services.eds_presets import (
    EdsPreset,
    PRESET_SCHEMA,
    check_compatibility,
    delete_preset,
    export_preset_json,
    import_preset_json,
    list_presets,
    load_preset,
    preset_from_settings,
    safe_filename,
    save_preset,
)


@pytest.fixture(autouse=True)
def preset_dirs(tmp_path):
    """Every test gets its own user + shipped directory, and the real
    %APPDATA% is never touched."""
    ep.set_preset_dir_for_test(tmp_path / "user")
    ep.set_builtin_dir_for_test(tmp_path / "shipped")
    yield tmp_path
    ep.set_preset_dir_for_test(None)
    ep.set_builtin_dir_for_test(None)


# --- helpers ----------------------------------------------------------------

def _settings(**over):
    """A realistic recipe: a weighted clustering, one hand-declared region and
    one phase rule, exactly the shape ``AutoClassifyRequest`` carries."""
    s = {
        "mode": "cluster",
        "min_score": 0.35,
        "scale_um": 1.5,
        "n_clusters": 8,
        "element_weights": {"Fe": 3.0, "Si": 3.0},
        "region_defs": [
            {"name": "Si particle",
             "elements": [{"element": "Si", "min_at_pct": 21.5}]},
        ],
        "rules": {
            "rules": [
                {"phase_key": "Al6Fe.cif", "phase_formula": "Al6Fe",
                 "enrichment": [{"element": "Fe", "min_factor": 2.0}]},
            ],
        },
        "phase_keys": ["Al6Fe.cif", "Si.cif"],
    }
    s.update(over)
    return s


def _preset(name="6xxx extrusion", **over):
    return preset_from_settings(
        name, _settings(**over.pop("settings", {})),
        author="seb", matrix_element="Al",
        authored_elements=["Al", "Fe", "Si", "Mn"],
        authored_step_um=0.65, **over,
    )


def _write_builtin(p: EdsPreset) -> Path:
    """Put a preset into the shipped directory, the way an app update would."""
    d = ep.builtin_preset_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{safe_filename(p.name)}.json"
    path.write_text(export_preset_json(p), encoding="utf-8")
    return path


ALL_MEASURED = ["Al", "Fe", "Si", "Mn"]
ALL_PHASES = ["Al6Fe.cif", "Si.cif"]


def _clean_check(p, **over):
    kw = dict(measured_elements=ALL_MEASURED, available_phase_keys=ALL_PHASES,
              dominant_element="Al", step_um=0.65)
    kw.update(over)
    return check_compatibility(p, **kw)


def _codes(items):
    return [i["code"] for i in items]


# --- round trip -------------------------------------------------------------

def test_save_load_round_trip_keeps_the_recipe():
    p = _preset()
    path = save_preset(p)
    assert path.exists()

    q = load_preset("6xxx extrusion")
    assert q.name == "6xxx extrusion"
    assert q.author == "seb"
    assert q.matrix_element == "Al"
    assert q.authored_elements == ["Al", "Fe", "Si", "Mn"]
    assert q.authored_step_um == 0.65
    assert q.settings == p.settings
    assert q.content_hash == p.content_hash
    # The clauses survive as clauses, not as an opaque blob.
    assert q.settings["region_defs"][0]["elements"][0]["min_at_pct"] == 21.5
    assert q.settings["rules"]["rules"][0]["enrichment"][0]["min_factor"] == 2.0


def test_json_export_is_readable_and_reimports():
    p = _preset()
    text = export_preset_json(p)
    # Mailable and diffable: indented, sorted, one preset per document.
    assert "\n" in text and '"settings"' in text
    assert json.loads(text)["name"] == p.name

    q = import_preset_json(text)
    assert q.settings == p.settings
    assert q.content_hash == p.content_hash
    # An imported preset is always the user's, whatever the file claimed.
    assert q.builtin is False


def test_pinned_matrix_reaches_the_payload_that_is_actually_run():
    """The matrix pin is a projection of ``rules.matrix_elements`` — if it only
    lived in the metadata, the classifier would never see it."""
    p = _preset()
    assert p.settings["rules"]["matrix_elements"] == ["Al"]

    # Declaring it inside the rules instead must give the same preset.
    s = _settings()
    s["rules"]["matrix_elements"] = ["Al"]
    q = preset_from_settings("via rules", s)
    assert q.matrix_element == "Al"
    assert q.content_hash == p.content_hash


def test_scan_bound_state_is_stripped():
    """``keep_manual_edits`` is about the hand edits of the map on screen, not
    about a recipe; carrying it would apply one scan's decision to another."""
    p = preset_from_settings("x", _settings(keep_manual_edits=False),
                             matrix_element="Al")
    assert "keep_manual_edits" not in p.settings


def test_half_written_region_window_is_not_stored():
    """A clause with neither bound set claims every pixel; it is what the
    editor's Add button produces mid-edit, and it must not enter a shared
    recipe."""
    s = _settings(region_defs=[
        {"name": "unfinished", "elements": [{"element": "Si"}]},
    ])
    p = preset_from_settings("x", s, matrix_element="Al")
    assert p.settings["region_defs"] is None


def test_element_line_labels_are_reduced_to_symbols():
    """A clause stored as `Fe Ka1` would never match: the backend works in
    symbols everywhere."""
    p = preset_from_settings(
        "x", _settings(element_weights={"Fe Ka1": 3.0}),
        matrix_element="Al Ka1", authored_elements=["Al Ka1", "Fe Ka1"],
    )
    assert p.settings["element_weights"] == {"Fe": 3.0}
    assert p.matrix_element == "Al"
    assert p.authored_elements == ["Al", "Fe"]


# --- content hash -----------------------------------------------------------

def test_hash_survives_rename_and_notes_edit_but_follows_the_settings():
    a = _preset("first")
    b = _preset("a completely different name")
    b.notes = "reviewed 2026-08-27"
    b.tags = ["draft"]
    b.author = "somebody else"
    assert a.content_hash == b.content_hash, "housekeeping must not move the hash"

    c = _preset("first", settings={"min_score": 0.36})
    assert c.content_hash != a.content_hash

    d = _preset("first", settings={
        "region_defs": [{"name": "Si particle",
                         "elements": [{"element": "Si", "min_at_pct": 30.0}]}]})
    assert d.content_hash != a.content_hash


def test_hash_names_the_recipe_not_the_spelling():
    """A setting left out and the same setting written at its default are the
    same analysis, so they must hash the same — otherwise the hash answers
    'was this typed the same way', which nobody asked."""
    a = preset_from_settings("a", {"min_score": 0.5}, matrix_element="Al")
    b = preset_from_settings("b", {"min_score": 0.5, "mode": "cluster",
                                   "tolerance": 15.0, "cluster_remainder": True},
                             matrix_element="Al")
    assert a.content_hash == b.content_hash


def test_defaults_are_written_out_so_a_future_default_cannot_move_a_preset():
    p = preset_from_settings("a", {"min_score": 0.5}, matrix_element="Al")
    for key in ("mode", "tolerance", "min_score", "cluster_remainder"):
        assert key in p.settings


def test_the_inert_tolerance_is_stored_but_not_hashed():
    """Spec section 9. ``tolerance`` reaches ``auto_classify_pixels`` and is
    never read there, so it cannot change a result — and a knob that cannot
    change a result cannot tell two recipes apart. Two presets differing only
    in it ARE the same analysis and must say so."""
    a = _preset("a", settings={"tolerance": 15.0})
    b = _preset("b", settings={"tolerance": 42.0})
    assert a.settings["tolerance"] != b.settings["tolerance"]
    assert a.content_hash == b.content_hash

    # It is excluded, not dropped: every other key still moves the hash.
    assert _preset("c", settings={"min_score": 0.9}).content_hash != a.content_hash


def test_the_excluded_tolerance_still_round_trips_losslessly():
    """A future build may reconnect the slider; the value has to survive to
    that day, in the file and through a save/load."""
    p = _preset("keeps its tolerance", settings={"tolerance": 42.5})
    assert json.loads(export_preset_json(p))["settings"]["tolerance"] == 42.5
    assert import_preset_json(export_preset_json(p)).settings["tolerance"] == 42.5

    save_preset(p)
    q = load_preset("keeps its tolerance")
    assert q.settings["tolerance"] == 42.5
    assert q.content_hash == p.content_hash


# --- blockers ---------------------------------------------------------------

def test_missing_element_blocks_and_names_every_affected_clause():
    s = _settings(
        region_defs=[{"name": "Mg zone",
                      "elements": [{"element": "Mg", "min_at_pct": 5.0}],
                      "ratios": [{"numerator": "Mg", "denominator": "Si",
                                  "min_ratio": 0.5}]}],
        element_weights={"Mg": 4.0},
    )
    p = preset_from_settings("mg recipe", s, matrix_element="Al",
                             authored_elements=ALL_MEASURED + ["Mg"])
    rep = _clean_check(p, measured_elements=ALL_MEASURED)

    assert rep.ok is False
    blocker = next(b for b in rep.blockers if b["code"] == "missing_element")
    assert blocker["detail"]["element"] == "Mg"
    assert "Mg" in blocker["message"]
    clauses = blocker["detail"]["clauses"]
    # content clause, ratio clause (both legs point at it), and the weight
    assert any("Mg >= 5 at%" in c for c in clauses)
    assert any("Mg:Si" in c for c in clauses)
    assert any("element weight" in c for c in clauses)
    for c in clauses:
        assert c in blocker["message"]


def test_missing_element_reports_one_blocker_per_element():
    s = _settings(element_weights={"Mg": 2.0, "Zn": 2.0})
    p = preset_from_settings("x", s, matrix_element="Al")
    rep = _clean_check(p)
    named = sorted(b["detail"]["element"] for b in rep.blockers
                   if b["code"] == "missing_element")
    assert named == ["Mg", "Zn"]


def test_matrix_mismatch_blocks_and_names_both():
    p = _preset()
    rep = _clean_check(p, dominant_element="Fe")
    blocker = next(b for b in rep.blockers if b["code"] == "matrix_mismatch")
    assert blocker["detail"] == {"preset_matrix": "Al", "scan_dominant": "Fe"}
    assert "Al" in blocker["message"] and "Fe" in blocker["message"]
    assert rep.ok is False


def test_matrix_check_is_skipped_and_said_out_loud_when_dominant_unknown():
    p = _preset()
    rep = _clean_check(p, dominant_element="")
    assert rep.ok is True
    assert "matrix_not_checked" in _codes(rep.warnings)
    assert "matrix_mismatch" not in _codes(rep.blockers)


def test_missing_phase_blocks_and_names_them():
    p = _preset()
    rep = _clean_check(p, available_phase_keys=["Al6Fe.cif"])
    blocker = next(b for b in rep.blockers if b["code"] == "missing_phase")
    assert blocker["detail"]["phase_keys"] == ["Si.cif"]
    assert "Si.cif" in blocker["message"]
    assert rep.ok is False


def test_a_rule_on_an_absent_phase_also_blocks():
    """A rule participates in the run as much as a ticked phase does."""
    s = _settings(phase_keys=None)
    p = preset_from_settings("x", s, matrix_element="Al")
    rep = _clean_check(p, available_phase_keys=["Si.cif"])
    blocker = next(b for b in rep.blockers if b["code"] == "missing_phase")
    assert blocker["detail"]["phase_keys"] == ["Al6Fe.cif"]


def test_unknown_library_warns_instead_of_blocking():
    p = _preset()
    rep = _clean_check(p, available_phase_keys=None)
    assert "missing_phase" not in _codes(rep.blockers)
    assert "phases_not_checked" in _codes(rep.warnings)


# --- warnings ---------------------------------------------------------------

def test_clean_match_is_ok_and_silent():
    """No absolute windows, same elements, same step, matrix agrees."""
    s = _settings(region_defs=None)
    p = preset_from_settings("clean", s, matrix_element="Al",
                             authored_elements=ALL_MEASURED,
                             authored_step_um=0.65)
    rep = _clean_check(p)
    assert rep.ok is True
    assert rep.blockers == []
    assert rep.warnings == []


def test_a_superset_of_elements_still_warns():
    """The trap: "more data" is not "compatible". at% is renormalised over the
    measured elements, so an extra element moves every other element's at%."""
    p = _preset()
    rep = _clean_check(p, measured_elements=ALL_MEASURED + ["Cu"])
    w = next(w for w in rep.warnings if w["code"] == "element_set_differs")
    assert w["detail"]["added"] == ["Cu"]
    assert w["detail"]["removed"] == []
    assert "renormalised" in w["message"]
    assert rep.ok is True, "a different element set warns, it does not refuse"


def test_element_set_identical_does_not_warn():
    p = _preset()
    rep = _clean_check(p)
    assert "element_set_differs" not in _codes(rep.warnings)


def test_step_size_difference_warns_but_a_rounding_difference_does_not():
    p = _preset()
    assert "step_size_differs" in _codes(_clean_check(p, step_um=1.3).warnings)
    # 0.6501 vs 0.65 is the same pixel box; warning on it would train the user
    # to ignore the warning.
    assert "step_size_differs" not in _codes(_clean_check(p, step_um=0.6501).warnings)


def test_absolute_windows_warn_and_point_at_enrichment():
    p = _preset()
    w = next(w for w in _clean_check(p).warnings
             if w["code"] == "absolute_window_not_portable")
    assert "Si >= 21.5 at%" in w["message"]
    assert "Enrichment" in w["message"]
    assert w["detail"]["clauses"] == ["region 'Si particle': Si >= 21.5 at%"]


def test_enrichment_only_preset_raises_no_portability_warning():
    s = _settings(region_defs=[
        {"name": "Fe rich",
         "enrichment": [{"element": "Fe", "min_factor": 2.0}]}])
    p = preset_from_settings("enriched", s, matrix_element="Al",
                             authored_elements=ALL_MEASURED,
                             authored_step_um=0.65)
    assert "absolute_window_not_portable" not in _codes(_clean_check(p).warnings)


def test_report_as_dict_records_which_preset_was_overridden():
    """A refusal the user overrides has to be stampable into the export."""
    p = _preset()
    d = _clean_check(p, dominant_element="Fe").as_dict()
    assert d["ok"] is False
    assert d["preset_name"] == "6xxx extrusion"
    assert d["preset_hash"] == p.content_hash
    assert d["blockers"][0]["code"] == "matrix_mismatch"
    json.dumps(d)  # must survive the trip into provenance.json


def test_every_finding_carries_a_stable_code():
    """The UI translates on the code; keying on the English prose would break
    a preset the moment the app is used in another language."""
    p = _preset()
    rep = _clean_check(p, measured_elements=["Al", "Fe", "Si"],
                       dominant_element="Fe", available_phase_keys=[],
                       step_um=2.0)
    for item in rep.blockers + rep.warnings:
        assert set(item) == {"code", "message", "detail"}
        assert item["code"] and item["code"].islower()


# --- files, builtins, versions ----------------------------------------------

def test_empty_preset_is_rejected():
    p = preset_from_settings("nothing", {})
    with pytest.raises(ValueError, match="constrains nothing"):
        save_preset(p)
    # Defaults spelled out explicitly are still nothing.
    q = preset_from_settings("also nothing",
                             {"mode": "cluster", "tolerance": 15.0,
                              "keep_manual_edits": False})
    with pytest.raises(ValueError):
        save_preset(q)


# --- the refusal has to fire for the button it protects ---------------------
#
# It did not. ``_settings_are_empty`` answers "not empty" on the first key it
# does not recognise -- correct in itself, because an unknown key may be real
# content from a newer build -- and the save dialog ALWAYS sends
# ``n_clusters_pinned``, which was not a known setting. So every save from the
# app short-circuited before judging a single setting, and the advertised
# refusal could only ever fire for a bare ``{}`` posted by a script. Found by a
# documentation audit, 2026-08-27.
#
# ``n_clusters_pinned`` is not content: it is a second copy of what
# ``n_clusters`` already says. ``PresetBar.jsx`` writes it as
# ``handle.nClusters != null`` and reads it back as
# ``s.n_clusters_pinned !== false && s.n_clusters != null`` -- with the key
# absent that is exactly ``n_clusters != null`` again, so dropping it preserves
# the app's own semantics.

#: The literal payload ``presetSettingsFrom(handle)`` builds with every control
#: at its default (``PhaseMapPanel.jsx`` useState defaults: mode 'cluster',
#: scale null, scaleUnit 'px', nClusters null, minScore 0.3, elementWeights {},
#: regionDefs [], rules null, selectedPhaseKeys empty Set, clusterRemainder
#: true). Written out rather than built from a helper: the point of the test is
#: that THESE BYTES are refused, and a helper could drift away from the dialog.
DIALOG_ALL_DEFAULTS = {
    "mode": "cluster",
    "scale": None,
    "scale_um": None,
    "n_clusters": None,
    "n_clusters_pinned": False,
    "min_score": 0.3,
    "element_weights": {},
    "region_defs": [],
    "rules": None,
    "phase_keys": [],
    "cluster_remainder": True,
}


def test_the_dialogs_own_all_default_payload_is_refused():
    """The path the refusal exists for. Not a synthetic ``{}``: this is what
    ``POST /api/eds/presets`` receives when the user names a preset without
    having changed anything."""
    p = preset_from_settings("default analysis wearing a name",
                             dict(DIALOG_ALL_DEFAULTS))
    with pytest.raises(ValueError, match="constrains nothing"):
        save_preset(p)


def test_the_ui_only_pinned_flag_never_reaches_the_file():
    """It is a duplicate, and a stored duplicate is the copy that goes stale."""
    p = preset_from_settings("pinned at eight",
                             dict(DIALOG_ALL_DEFAULTS, n_clusters=8,
                                  n_clusters_pinned=True))
    assert "n_clusters_pinned" not in p.settings
    # ...and the fact itself survives, in the one place that carries it.
    assert p.settings["n_clusters"] == 8
    assert json.loads(export_preset_json(p))["settings"]["n_clusters"] == 8


def test_dropping_the_flag_does_not_move_the_hash_off_the_spelling():
    """A hand-written preset and the dialog's own save of the same recipe are
    the same analysis and must hash alike. Putting the flag in the defaults
    instead would have made them differ."""
    hand = preset_from_settings("hand", {"n_clusters": 8})
    app = preset_from_settings("app", {"n_clusters": 8,
                                       "n_clusters_pinned": True})
    assert hand.content_hash == app.content_hash


def test_mutation_one_real_setting_makes_the_same_payload_saveable():
    """The mutation half: the refusal has to be about the SETTINGS being
    default, not about the payload's shape. Otherwise it would refuse every
    save from the dialog, which is the opposite failure."""
    for field, value in [
        ("n_clusters", 8),
        ("min_score", 0.55),
        ("scale_um", 1.5),
        ("cluster_remainder", False),
        ("element_weights", {"Fe": 3.0}),
        ("phase_keys", ["Al6Fe.cif"]),
        ("region_defs", [{"name": "Si particle",
                          "elements": [{"element": "Si", "min_at_pct": 21.5}]}]),
    ]:
        p = preset_from_settings(f"changed {field}",
                                 dict(DIALOG_ALL_DEFAULTS, **{field: value}))
        assert save_preset(p).exists(), f"{field} must count as content"


def test_every_key_the_dialog_sends_is_known_or_excluded():
    """The invariant that keeps this bug from coming back in another key.

    ``_settings_are_empty`` short-circuits on any key it does not recognise,
    which is right for content from a newer build and fatal for a field the UI
    itself always sends. Sourced from ``PresetBar.jsx`` rather than restated,
    so adding a field there and not here fails HERE instead of silently
    disarming the guard.
    """
    src = (ROOT / "frontend" / "src" / "components" / "EDS"
           / "PresetBar.jsx").read_text(encoding="utf-8")
    body = src.split("export function presetSettingsFrom", 1)[1].split("\n}", 1)[0]
    sent = set(re.findall(r"^\s{4}(\w+):", body, re.M))
    assert "n_clusters_pinned" in sent, "the dialog stopped sending the flag"
    assert len(sent) >= 10, f"only found {sorted(sent)} — parser drifted"

    unjudgeable = sent - set(ep._SETTING_DEFAULTS) - set(ep._EXCLUDED_SETTING_KEYS)
    assert not unjudgeable, (
        f"the save dialog sends {sorted(unjudgeable)}, which _settings_are_empty "
        f"cannot judge — it will short-circuit and the 'constrains nothing' "
        f"refusal will never fire for a save made through the app. Add each to "
        f"_SETTING_DEFAULTS (a real setting) or to _EXCLUDED_SETTING_KEYS (UI "
        f"bookkeeping)."
    )


def test_saving_over_an_existing_preset_needs_overwrite_and_bumps_version():
    p = _preset()
    save_preset(p)
    assert load_preset(p.name).version == 1

    again = _preset(settings={"min_score": 0.4})
    with pytest.raises(FileExistsError):
        save_preset(again)

    save_preset(again, overwrite=True)
    stored = load_preset(p.name)
    assert stored.version == 2
    assert stored.settings["min_score"] == 0.4
    assert len(list_presets()) == 1, "a new version replaces, it does not pile up"


def test_a_builtin_is_never_mutated():
    shipped = _preset("Al matrix starter")
    shipped_path = _write_builtin(shipped)
    original_bytes = shipped_path.read_bytes()

    assert load_preset("Al matrix starter").builtin is True

    fork = _preset("Al matrix starter", settings={"min_score": 0.9})
    with pytest.raises(FileExistsError):
        save_preset(fork)

    path = save_preset(fork, overwrite=True)
    assert path.parent == ep.preset_dir(), "a fork lands in the user directory"
    assert shipped_path.read_bytes() == original_bytes

    # The fork shadows the shipped copy, and is a user preset whatever it says.
    loaded = load_preset("Al matrix starter")
    assert loaded.builtin is False
    assert loaded.settings["min_score"] == 0.9
    assert loaded.version == 2
    assert [p.name for p in list_presets()] == ["Al matrix starter"]


def test_a_file_claiming_to_be_builtin_is_not_one():
    """Location decides, not content — otherwise a mailed JSON could declare
    itself read-only and become unwritable."""
    p = _preset("mailed")
    p.builtin = True
    save_preset(p)
    assert load_preset("mailed").builtin is False


def test_builtins_are_listed_first_then_user_presets():
    _write_builtin(_preset("zz shipped"))
    save_preset(_preset("aa mine"))
    assert [p.name for p in list_presets()] == ["zz shipped", "aa mine"]


def test_delete_removes_a_user_preset_and_refuses_a_builtin():
    _write_builtin(_preset("Al matrix starter"))
    assert delete_preset("does not exist") is False
    with pytest.raises(PermissionError):
        delete_preset("Al matrix starter")

    save_preset(_preset("Al matrix starter"), overwrite=True)
    assert delete_preset("Al matrix starter") is True
    # Deleting the fork re-exposes the shipped original.
    assert load_preset("Al matrix starter").builtin is True


def test_a_junk_file_does_not_take_the_list_with_it():
    save_preset(_preset("good one"))
    d = ep.preset_dir()
    (d / "broken.json").write_text("{not json at all", encoding="utf-8")
    (d / "empty.json").write_text("", encoding="utf-8")
    (d / "from_the_future.json").write_text(
        json.dumps({"schema": 99, "name": "future", "settings": {}}),
        encoding="utf-8")
    (d / "notes.txt").write_text("ignore me", encoding="utf-8")

    names = [p.name for p in list_presets()]
    assert names == ["good one"]
    assert load_preset("good one").name == "good one"


def test_unsafe_names_become_safe_filenames_and_still_load_by_true_name():
    name = "AL/Fe 6xxx: draft"
    path = save_preset(_preset(name))
    assert path.name == "AL_Fe_6xxx_draft.json"
    assert "/" not in path.name and ":" not in path.name
    assert load_preset(name).name == name, "the true name lives inside the JSON"


def test_reserved_and_degenerate_names_are_handled():
    assert safe_filename("CON").lower().startswith("con_preset")
    assert safe_filename("***") == "preset"
    assert safe_filename("  ..spaced.. ") == "spaced"
    assert len(safe_filename("x" * 300)) <= 80

    save_preset(_preset("CON"))
    assert load_preset("CON").name == "CON"


def test_two_names_that_sanitise_alike_get_separate_files():
    a = save_preset(_preset("a/b"))
    b = save_preset(_preset("a:b"))
    assert a != b
    assert load_preset("a/b").name == "a/b"
    assert load_preset("a:b").name == "a:b"


def test_load_preset_raises_keyerror_when_absent():
    with pytest.raises(KeyError):
        load_preset("never saved")


# --- import validation ------------------------------------------------------

@pytest.mark.parametrize("text", [
    "{not json",
    "[]",
    '"a string"',
    json.dumps({"name": "x", "settings": {}}),                     # no schema
    json.dumps({"schema": "1", "name": "x", "settings": {}}),      # schema as text
    json.dumps({"schema": 99, "name": "x", "settings": {}}),       # newer schema
    json.dumps({"schema": PRESET_SCHEMA, "settings": {}}),         # no name
    json.dumps({"schema": PRESET_SCHEMA, "name": "x"}),            # no settings
    json.dumps({"schema": PRESET_SCHEMA, "name": "x", "settings": []}),
])
def test_malformed_presets_are_refused(text):
    with pytest.raises(ValueError):
        import_preset_json(text)


def test_import_recomputes_the_hash_rather_than_trusting_it():
    p = _preset()
    payload = json.loads(export_preset_json(p))
    payload["content_hash"] = "deadbeefdeadbeef"
    q = import_preset_json(json.dumps(payload))
    assert q.content_hash == p.content_hash


def test_unknown_metadata_keyword_raises():
    with pytest.raises(TypeError):
        preset_from_settings("x", _settings(), autor="typo")


# --- the phase list, pinned or not ------------------------------------------
#
# The asymmetry that makes this worth its own section: the guard that already
# existed stops a stale UI silently NARROWING a run (``routes/eds.py``: an
# empty or full selection means "use everything"), while WIDENING was
# unguarded — and widening is the direction that moves an audited lab's
# numbers. Add a CIF in week 30 and the same preset, at the same content hash,
# runs a different analysis; the only way to find out afterwards is to diff two
# provenance files.

def _unpinned(**over):
    """The same recipe as ``_preset``, minus the pinned candidate list."""
    s = _settings(phase_keys=None, **over)
    s["rules"].pop("phase_keys", None)
    return preset_from_settings("unpinned", s, matrix_element="Al",
                                authored_elements=ALL_MEASURED,
                                authored_step_um=0.65)


def test_an_unpinned_phase_list_warns_and_a_pinned_one_does_not():
    """Mutation test: one field decides, and nothing else moves.

    The preset is otherwise identical in both halves, so a warning that
    appeared for some other reason would show up in both.
    """
    pinned = _preset()                       # phase_keys = [Al6Fe.cif, Si.cif]
    assert pinned.phase_list_pinned is True
    assert "phase_list_not_pinned" not in _codes(_clean_check(pinned).warnings)

    loose = _unpinned()
    assert loose.phase_list_pinned is False
    w = next(w for w in _clean_check(loose).warnings
             if w["code"] == "phase_list_not_pinned")
    assert w["detail"] == {"pinned": False, "n_available": 2,
                           "named_in_rules": ["Al6Fe.cif"]}
    # It has to say what goes wrong, not merely that something might.
    assert "content hash" in w["message"]
    # And it has to head off the obvious objection: this preset DOES name a
    # phase, in a rule — which gates who may compete and narrows nothing.
    assert "do not narrow the list" in w["message"]


def test_naming_a_phase_in_a_rule_is_not_pinning_the_list():
    """The trap this warning has to survive: a preset with rules looks pinned.
    A rule decides which candidates may compete for a region; every other phase
    in the library still competes."""
    s = _settings(phase_keys=None)
    s["rules"].pop("phase_keys", None)
    p = preset_from_settings("ruled", s, matrix_element="Al")
    assert p.settings["rules"]["rules"][0]["phase_key"] == "Al6Fe.cif"
    assert p.phase_list_pinned is False
    assert "phase_list_not_pinned" in _codes(_clean_check(p).warnings)


def test_pinning_the_list_inside_the_rules_counts_as_pinned():
    """``rules.phase_keys`` narrows the library at classification time exactly
    as ``settings['phase_keys']`` does, so it has to count here too."""
    s = _settings(phase_keys=None)
    s["rules"]["phase_keys"] = ["Al6Fe.cif"]
    p = preset_from_settings("via rules", s, matrix_element="Al")
    assert p.phase_list_pinned is True
    assert "phase_list_not_pinned" not in _codes(_clean_check(p).warnings)


def test_an_unpinned_list_warns_rather_than_blocking():
    """An unpinned preset is still usable — it is just not reproducible. A
    refusal would make the shipped starter presets, which cannot know a
    stranger's library, unusable on the day they are most needed."""
    rep = _clean_check(_unpinned())
    assert rep.ok is True
    assert "phase_list_not_pinned" not in _codes(rep.blockers)


def test_the_unpinned_warning_fires_even_when_the_library_is_unknown():
    """Unpinnedness is a property of the preset, not of this machine's
    database, so an unreadable library must not suppress it."""
    w = next(w for w in _clean_check(_unpinned(),
                                     available_phase_keys=None).warnings
             if w["code"] == "phase_list_not_pinned")
    assert w["detail"]["n_available"] is None


def test_whether_the_list_was_pinned_is_recorded_in_the_saved_file():
    save_preset(_preset())
    save_preset(_unpinned())
    assert json.loads(export_preset_json(load_preset("6xxx extrusion"))
                      )["phase_list_pinned"] is True
    assert json.loads(export_preset_json(load_preset("unpinned"))
                      )["phase_list_pinned"] is False


def test_the_pinned_record_is_derived_and_never_trusted():
    """A stored flag and the settings it describes are two copies of one fact,
    and the stored one is the copy that goes stale."""
    payload = json.loads(export_preset_json(_unpinned()))
    payload["phase_list_pinned"] = True          # a lie, or an old build
    assert import_preset_json(json.dumps(payload)).phase_list_pinned is False


# --- a guard that cannot run is not a clean result --------------------------

def _no_record(**meta):
    return preset_from_settings("bare", _settings(region_defs=None),
                                matrix_element="Al", **meta)


def test_a_preset_without_an_authored_element_list_says_the_guard_is_off():
    """``element_set_differs`` is gated on ``authored_elements``. A preset made
    through the raw API, by hand or by a colleague has none — and then the
    warning silently never fires, which reads exactly like "checked, fine"."""
    rep = _clean_check(_no_record(authored_step_um=0.65))
    w = next(w for w in rep.warnings if w["code"] == "portability_not_checkable")
    guards = w["detail"]["guards"]
    assert [g["guard"] for g in guards] == ["element_set_differs"]
    assert guards[0]["missing"] == "authored_elements"
    assert "renormalised" in guards[0]["why"]
    assert "element_set_differs" in w["message"]
    assert rep.ok is True, "an unchecked guard warns; it does not refuse"


def test_a_preset_without_an_authored_step_says_the_step_guard_is_off():
    rep = _clean_check(_no_record(authored_elements=ALL_MEASURED))
    w = next(w for w in rep.warnings if w["code"] == "portability_not_checkable")
    assert [g["missing"] for g in w["detail"]["guards"]] == ["authored_step_um"]


def test_a_scan_without_a_step_disables_the_step_guard_just_as_thoroughly():
    """The gate has two sides. A file with no step size leaves the same check
    unrun, and the report has to say so from that side too."""
    p = _no_record(authored_elements=ALL_MEASURED, authored_step_um=0.65)
    w = next(w for w in _clean_check(p, step_um=None).warnings
             if w["code"] == "portability_not_checkable")
    assert [g["missing"] for g in w["detail"]["guards"]] == ["step_um"]


def test_both_missing_are_reported_together_and_counted():
    w = next(w for w in _clean_check(_no_record()).warnings
             if w["code"] == "portability_not_checkable")
    assert [g["guard"] for g in w["detail"]["guards"]] == [
        "element_set_differs", "step_size_differs"]
    assert w["message"].startswith("2 portability check(s) could not run")


def test_a_preset_carrying_both_records_raises_no_such_warning():
    """The mutation half: with the record present the guards run, so the
    warning must be absent — otherwise it would be noise on every preset."""
    p = _no_record(authored_elements=ALL_MEASURED, authored_step_um=0.65)
    rep = _clean_check(p)
    assert "portability_not_checkable" not in _codes(rep.warnings)
    assert rep.warnings == [], "a fully recorded, matching preset is silent"


# --- the shipped starter presets --------------------------------------------
#
# "You built the shelf and put nothing on it." These load through the real
# loader from the real directory — a built-in that only works inside a fixture
# is the same empty shelf with a test standing in front of it.

@pytest.fixture
def shipped(tmp_path):
    """The REAL shipped directory, with the user directory still redirected."""
    ep.set_builtin_dir_for_test(None)
    yield ep.builtin_preset_dir()
    ep.set_builtin_dir_for_test(tmp_path / "shipped")


# A realistic aluminium scan: the element set actually measured on the SampleB
# test file (90x120 px, 0.5 um step). A starter preset that blocks on this
# would be worse than no starter preset at all.
AL_SCAN = ["Al", "C", "Cu", "Fe", "Mn", "O", "Si", "Zn"]


def test_the_shelf_is_not_empty(shipped):
    names = [p.name for p in list_presets()]
    assert "Find the particles in an aluminium matrix" in names
    assert len(names) >= 2
    assert all(p.builtin for p in list_presets())


def test_every_shipped_preset_loads_through_the_real_loader(shipped):
    files = sorted(shipped.glob("*.json"))
    assert files, "the built-in directory ships no presets"
    loaded = list_presets()
    assert len(loaded) == len(files)
    for p in loaded:
        assert p.author == "Orienta"
        assert p.builtin is True
        assert p.version == 1
        assert load_preset(p.name).content_hash == p.content_hash


def test_no_shipped_preset_blocks_a_normal_aluminium_scan(shipped):
    """The one failure mode that would make built-ins worse than none."""
    for p in list_presets():
        rep = check_compatibility(
            p, measured_elements=AL_SCAN,
            available_phase_keys=["Al6Fe.cif", "Si.cif", "Al.cif"],
            dominant_element="Al", step_um=0.5)
        assert rep.blockers == [], f"{p.name} blocks: {rep.blockers}"
        assert rep.ok is True


def test_shipped_presets_declare_their_smoothing_as_a_length(shipped):
    """A pixel count is not a length: 5 px at a 0.2 um step and at a 2 um step
    are a 1 um and a 10 um box. A built-in has to mean the same analysis on a
    stranger's step size, and only ``scale_um`` can promise that."""
    for p in list_presets():
        assert p.settings["scale_um"], f"{p.name} has no physical width"
        assert p.settings["scale"] is None, f"{p.name} pins a pixel count"


def test_shipped_presets_are_honest_about_not_pinning_a_phase_list(shipped):
    """A starter cannot know the user's library, so it does not pin one — and
    then Gap 1's warning has to fire for it too, not be waived because it
    ships with the app."""
    for p in list_presets():
        assert p.phase_list_pinned is False
        rep = check_compatibility(p, measured_elements=AL_SCAN,
                                  available_phase_keys=["Al.cif"],
                                  dominant_element="Al", step_um=0.5)
        assert "phase_list_not_pinned" in _codes(rep.warnings)
        assert "library" in p.notes.lower(), \
            f"{p.name} must say the candidate list is whatever the user holds"


def test_every_shipped_preset_says_what_it_is_not_for(shipped):
    """The tester was explicit: the "not for" line is what makes the first line
    trustworthy, and a starter must promise a starting point you can then
    correct, never the correct answer."""
    for p in list_presets():
        low = p.notes.lower()
        assert "not for" in low, f"{p.name} does not say what it is not for"
        assert "starting point" in low, f"{p.name} promises too much"
        assert len(p.notes) > 200


def test_the_aluminium_starter_pins_aluminium_and_refuses_a_steel_scan(shipped):
    """It is named after the question the user walked in with, so it has to
    refuse the sample that question was not about — enrichment is measured
    against the matrix, and getting it wrong inverts the measure."""
    p = load_preset("Find the particles in an aluminium matrix")
    assert p.matrix_element == "Al"
    rep = check_compatibility(p, measured_elements=["Fe", "Cr", "C"],
                              available_phase_keys=["Fe.cif"],
                              dominant_element="Fe", step_um=0.5)
    assert rep.ok is False
    assert "matrix_mismatch" in _codes(rep.blockers)


def test_a_shipped_preset_cannot_be_mutated_in_place(shipped):
    """An edit forks into the user directory; the file that ships with the app
    is replaced by an update and never by a user."""
    stem = "Find_the_particles_in_an_aluminium_matrix.json"
    name = "Find the particles in an aluminium matrix"
    original = (shipped / stem).read_bytes()

    fork = load_preset(name)
    fork.settings = dict(fork.settings, scale_um=9.0)
    path = save_preset(fork, overwrite=True)

    assert path.parent == ep.preset_dir()
    assert (shipped / stem).read_bytes() == original
    assert load_preset(name).builtin is False
    assert load_preset(name).settings["scale_um"] == 9.0


def test_a_missing_builtin_directory_is_not_an_error(tmp_path):
    """A slimmed deployment may ship no starter presets at all; the user's own
    must still list."""
    ep.set_builtin_dir_for_test(tmp_path / "does" / "not" / "exist")
    save_preset(_preset("mine"))
    assert [p.name for p in list_presets()] == ["mine"]


# --- the bytes that get mailed ----------------------------------------------
#
# A tester mails presets to colleagues. ``save_preset`` went through Python's
# text mode, so the file landed CRLF on Windows, while ``export_preset_json``
# -- what ``GET /presets/{name}/export`` returns and what the UI writes out --
# is LF. She diffed a mailed copy against her saved one and measured 81 of 81
# lines differing without a single character of content changing, which is
# exactly the diff that makes somebody believe the preset travelled wrong.

def test_the_file_on_disk_is_byte_identical_to_what_gets_mailed():
    path = save_preset(_preset())
    mailed = export_preset_json(load_preset("6xxx extrusion")).encode("utf-8")
    assert path.read_bytes() == mailed


def test_a_saved_preset_is_lf_like_the_shipped_ones():
    """One convention across saved, shipped and mailed. The built-ins are LF
    because they are repository files; the other two now follow them."""
    raw = save_preset(_preset()).read_bytes()
    assert bytes([13, 10]) not in raw           # no CRLF anywhere
    assert raw.endswith(bytes([125, 10]))       # ...and it ends "}" + LF


def test_a_preset_survives_the_round_trip_through_its_own_bytes():
    """The point of the byte equality rather than the equality itself: what
    the colleague opens has to import back as the same recipe."""
    path = save_preset(_preset())
    original = load_preset("6xxx extrusion")
    reimported = import_preset_json(path.read_text(encoding="utf-8"))
    assert reimported.content_hash == original.content_hash
    assert reimported.settings == original.settings
