"""The HTTP surface: what is installed, and what the user allows to run.

Two properties are worth more than the rest of this file and are tested
directly rather than implied:

* the listing IMPORTS NOTHING. Discovery reads manifests off disk, and the
  moment a catalogue call starts importing add-on code it lands in the same
  start-up path as kikuchipy, orix and torch. The failure mode is silent and
  only shows up as start-up time in production, so it gets its own test;
* a decision the user made is either recorded or reported. An add-on that
  fails its activation probe is written down as DISABLED, not merely left
  alone -- those two look identical for an add-on nobody had enabled yet, and
  differ completely for one that was.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.routes import addons as addons_route
from backend.api.routes import indexing as indexing_routes
from backend.api.routes.addons import ADDON_FAILED_STATUS, API_PREFIX
from backend.api.services.addons.context import build_context
from backend.api.services.addons.outputs import MapOutput
from backend.api.services.addons.trust import TrustStore
from backend.api.services.pattern_quality import LABEL_NATIVE

FIXTURES = Path(__file__).parent / "fixtures"
client = TestClient(app)


@pytest.fixture(autouse=True)
def _addon_dirs(monkeypatch):
    """Point discovery at the fixtures. The user directory and the trust file
    are already redirected to tmp_path by tests/addons/conftest.py -- without
    that, the second run of this module fails on a file the first left in the
    developer's real home directory.

    Note what is NOT here: no syspath_prepend. The runner puts the add-on's
    own folder on sys.path, and if it stops doing so these tests must fail.
    """
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(FIXTURES / "runnable_addon"))


def _entry(name="runnable"):
    body = client.get("/api/addons").json()
    return next(a for a in body["addons"] if a["name"] == name)


def test_listing_shows_the_addon_without_running_it():
    body = client.get("/api/addons").json()
    assert "runnable" in [a["name"] for a in body["addons"]]
    assert body["api_version"] == 0


def test_the_listing_never_imports_the_addons_module():
    """The invariant the whole package is built around, measured directly.

    ``monkeypatch.delitem`` restores sys.modules afterwards, so this test
    cannot disturb the runner tests that legitimately import the same module.
    A listing that imported anything -- to read a docstring, to count
    something, to "validate" -- turns this red.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.delitem(sys.modules, "runnable_addon_module", raising=False)
        assert client.get("/api/addons").status_code == 200
        assert "runnable_addon_module" not in sys.modules


def test_the_listing_survives_an_addon_that_cannot_be_imported(monkeypatch):
    """The same invariant from the other side, in behaviour rather than
    bookkeeping: this add-on raises at import. If the catalogue imported it,
    the listing would either fail or carry an error it has no business
    knowing yet -- importing is what ENABLING is for."""
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(FIXTURES / "exploding_addon"))
    r = client.get("/api/addons")
    assert r.status_code == 200
    entry = _entry("exploding")
    assert entry["error"] == ""
    assert entry["display_name"] == "Exploding test add-on"


def test_the_listing_says_where_the_addon_came_from_on_disk():
    """The spec's trust dialog shows name, version, authors, DOI and where it
    came from on disk. The last one had no consumer at all."""
    entry = _entry()
    assert entry["source_path"].endswith("orienta-addon.toml")
    assert "runnable_addon" in entry["source_path"]


def test_the_listing_carries_what_the_trust_dialog_has_to_show():
    """Name, version, authors, DOI -- the dialog asks the user to trust a
    person, and a dialog missing the person is a dialog about nothing."""
    entry = _entry()
    assert entry["display_name"] == "Runnable test add-on"
    assert entry["version"] == "0.1.0"
    assert entry["authors"] == ["Test <t@example.org>"]
    assert entry["doi"] == "10.5281/zenodo.7777777"
    assert entry["origin"] == "folder"
    assert [a["key"] for a in entry["analyses"]] == ["addon.mean_quality"]
    assert "scale" in entry["analyses"][0]["params_schema"]["properties"]


def test_a_new_addon_is_listed_as_not_yet_known():
    entry = _entry()
    assert entry["known"] is False and entry["enabled"] is False


def test_enabling_it_is_remembered():
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    entry = _entry()
    assert entry["enabled"] is True and entry["known"] is True


def test_enabling_answers_with_the_same_entry_the_listing_shows():
    """The response is the updated row, not a second opinion about it."""
    r = client.post("/api/addons/runnable/enabled", json={"enabled": True})
    assert r.status_code == 200
    assert r.json() == _entry()


def test_disabling_a_broken_addon_still_works(monkeypatch):
    """Turning something OFF must never depend on being able to import it --
    otherwise the one add-on a user most needs to switch off, the one that
    breaks, is the one they cannot."""
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(FIXTURES / "exploding_addon"))
    r = client.post("/api/addons/exploding/enabled", json={"enabled": False})
    assert r.status_code == 200
    entry = _entry("exploding")
    assert entry["enabled"] is False and entry["known"] is True


def test_a_broken_manifest_is_listed_with_its_error(monkeypatch):
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(FIXTURES / "broken_addon"))
    body = client.get("/api/addons").json()
    assert any(a["error"] for a in body["addons"])


def test_a_rejected_manifest_still_has_every_field_the_ui_reads(monkeypatch):
    """Shown as rejected, not dropped -- and shown in the shape the table
    renders, so a rejected row cannot take the page down with it."""
    good = _entry()
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(FIXTURES / "broken_addon"))
    bad = next(a for a in client.get("/api/addons").json()["addons"]
               if a["error"])
    assert set(bad) == set(good)
    assert bad["source_path"].endswith("orienta-addon.toml")
    assert bad["enabled"] is False and bad["compatible"] is False


def test_an_addon_that_raises_on_activation_is_disabled_and_says_so(monkeypatch):
    """The spec's sentence, which nothing implemented: only the crash counter
    existed, and that is a different requirement."""
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(FIXTURES / "exploding_addon"))
    r = client.post("/api/addons/exploding/enabled", json={"enabled": True})
    assert r.status_code == ADDON_FAILED_STATUS
    assert "exploding" in r.json()["detail"]
    assert "explodes on import" in r.json()["detail"]
    assert _entry("exploding")["enabled"] is False


def test_a_failed_probe_turns_an_already_enabled_addon_off(monkeypatch):
    """The half of "is disabled" that "was never enabled" hides.

    An add-on enabled yesterday and broken by today's update is the real
    case, and for it "leave the store alone" and "write disabled" are
    different outcomes.
    """
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(FIXTURES / "exploding_addon"))
    TrustStore().set_enabled("exploding", True, version="0.1.0", doi="")
    assert _entry("exploding")["enabled"] is True

    r = client.post("/api/addons/exploding/enabled", json={"enabled": True})
    assert r.status_code == ADDON_FAILED_STATUS
    entry = _entry("exploding")
    assert entry["enabled"] is False and entry["known"] is True


def test_an_incompatible_addon_cannot_be_enabled(monkeypatch):
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(FIXTURES / "future_addon"))
    r = client.post("/api/addons/future/enabled", json={"enabled": True})
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert ">=99" in detail                       # what it asked for
    entry = _entry("future")
    assert entry["compatible"] is False
    assert entry["compatibility"]                 # a sentence, not just a flag


def test_the_version_gate_fires_before_anything_is_imported(monkeypatch):
    """409, not 422. The fixture's module does not exist, so an
    implementation that probed first would answer with an import error
    naming ``never_imported_module`` -- which is exactly why it is named
    that. Refusing an add-on is not a reason to run it."""
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(FIXTURES / "future_addon"))
    r = client.post("/api/addons/future/enabled", json={"enabled": True})
    assert r.status_code == 409
    assert "never_imported_module" not in r.json()["detail"]
    assert _entry("future")["known"] is False     # nothing was recorded either


def test_an_addon_that_asks_for_nothing_is_compatible():
    entry = _entry()
    assert entry["compatible"] is True and entry["compatibility"] == ""


def test_the_listing_names_the_build_it_is_judging_against():
    """``orienta_version`` is what every compatibility sentence compares to;
    without it in the payload the user is told "incompatible" and cannot see
    with what. None is an honest answer -- a source checkout has no release
    tag -- but the key is not optional."""
    body = client.get("/api/addons").json()
    assert "orienta_version" in body


def test_an_unknown_addon_is_404():
    r = client.post("/api/addons/nobody/enabled", json={"enabled": True})
    assert r.status_code == 404


def test_an_unwritable_trust_store_is_reported_not_a_bare_500(monkeypatch):
    """The obligation that the runner's guard starts and this layer finishes:
    no failure reaches the user as an unattributed traceback."""
    def _boom(*a, **k):
        raise OSError("read-only file system")

    monkeypatch.setattr(TrustStore, "set_enabled", _boom)
    r = client.post("/api/addons/runnable/enabled", json={"enabled": True})
    assert r.status_code == 500
    detail = r.json()["detail"]
    assert "runnable" in detail
    assert "read-only file system" in detail


def test_a_failed_probe_still_names_the_addon_when_the_store_is_unwritable(
        monkeypatch):
    """Two failures at once, and the add-on's own is the one that must be
    readable first -- the store's problem is Orienta's, and it is appended
    rather than allowed to replace the answer."""
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(FIXTURES / "exploding_addon"))

    def _boom(*a, **k):
        raise OSError("read-only file system")

    monkeypatch.setattr(TrustStore, "set_enabled", _boom)
    r = client.post("/api/addons/exploding/enabled", json={"enabled": True})
    assert r.status_code == ADDON_FAILED_STATUS
    detail = r.json()["detail"]
    assert detail.startswith("exploding")
    assert "explodes on import" in detail
    assert "not recorded" in detail


def test_a_json_serialisation_failure_is_reported_not_a_bare_500(monkeypatch):
    """The measured escape, and the reason this guard is not an enumeration.

    An ``OSError``-only ``except`` let ``json.dump``'s ``TypeError`` through
    as an unattributed 500. The next one will not be a TypeError either, so
    what is pinned here is the behaviour -- named, attributed, non-fatal --
    for a class the guard was never told about.
    """
    class _Odd(Exception):
        pass

    def _boom(*a, **k):
        raise _Odd("something the guard was never told about")

    monkeypatch.setattr(TrustStore, "set_enabled", _boom)
    r = client.post("/api/addons/runnable/enabled", json={"enabled": True})
    assert r.status_code == 500
    detail = r.json()["detail"]
    assert "runnable" in detail
    assert "_Odd" in detail and "never told about" in detail


# --- the add-on's failure has its own status code ---------------------------

def test_a_probe_failure_is_not_confused_with_a_malformed_request(monkeypatch):
    """Two different things must not answer with the same number.

    422 is what FastAPI returns when the REQUEST is wrong. If an add-on
    blowing up on import answered 422 too, a caller -- or a log reader --
    could not tell "you sent nonsense" from "their code raised", and the
    author of the add-on never hears about it.
    """
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(FIXTURES / "exploding_addon"))
    broken_addon = client.post("/api/addons/exploding/enabled",
                               json={"enabled": True})
    broken_request = client.post("/api/addons/exploding/enabled",
                                 json={"enabled": "yes please"})
    assert broken_addon.status_code == ADDON_FAILED_STATUS
    assert broken_request.status_code == 422
    assert broken_addon.status_code != broken_request.status_code


def test_an_unknown_body_field_is_refused_rather_than_ignored():
    """A caller who sends a field we do not understand is told so. Accepting
    it with 200 tells them it worked."""
    r = client.post("/api/addons/runnable/enabled",
                    json={"enabled": True, "force": True})
    assert r.status_code == 422
    assert _entry()["known"] is False        # and nothing was decided


# --- two add-ons cannot share one name --------------------------------------

def _two_addons_named(tmp_path, name, *, dirs=("one", "two")):
    """Two installations claiming the same add-on name, as a user gets them:
    one from a package, one from a zip that happened to pick the same name."""
    for folder in dirs:
        d = tmp_path / folder
        d.mkdir()
        (d / "orienta-addon.toml").write_bytes(
            f'api_version = 0\n'
            f'name = "{name}"\n'
            f'display_name = "From {folder}"\n'
            f'version = "0.1.0"\n'
            f'authors = ["{folder} <{folder}@example.org>"]\n'
            f'doi = ""\n\n'
            f'[[analyses]]\n'
            f'key = "addon.{folder}"\n'
            f'label = "{folder}"\n'
            f'python_name = "{folder}_module:analyse"\n'
            f'sentence = "It ran."\n'
            f'citations = ["orienta"]\n'.encode("utf-8"))
    return tmp_path


def test_a_duplicated_name_is_shown_as_a_conflict(tmp_path, monkeypatch):
    monkeypatch.setenv("ORIENTA_ADDON_DIRS",
                       str(_two_addons_named(tmp_path, "twins")))
    rows = [a for a in client.get("/api/addons").json()["addons"]
            if a["name"] == "twins"]
    assert len(rows) == 2                  # both shown, neither hidden
    for row in rows:
        assert row["conflict"]
        assert "one" in row["conflict"] and "two" in row["conflict"]


def test_a_duplicated_name_cannot_be_enabled_at_all(tmp_path, monkeypatch):
    """Refused, not resolved by picking. Consent is keyed by name, so
    enabling either would enable both -- one author's decision applied to
    another author's code, which is the runner's module collision one level
    up."""
    monkeypatch.setenv("ORIENTA_ADDON_DIRS",
                       str(_two_addons_named(tmp_path, "twins")))
    r = client.post("/api/addons/twins/enabled", json={"enabled": True})
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert "one" in detail and "two" in detail      # BOTH locations named
    rows = [a for a in client.get("/api/addons").json()["addons"]
            if a["name"] == "twins"]
    assert [row["enabled"] for row in rows] == [False, False]


def test_an_addon_with_a_unique_name_reports_no_conflict():
    assert _entry()["conflict"] == ""


# --- two add-ons cannot share one analysis key ------------------------------

_SHARED_KEY_MANIFEST = """api_version = 0
name = "{name}"
display_name = "Analysis by {name}"
version = "0.1.0"
authors = ["{name} <{name}@example.org>"]
doi = "10.5281/zenodo.{doi}"

[[analyses]]
key = "{key}"
label = "{name} label"
python_name = "{name}_module:analyse"
sentence = "{name} did it."
"""


def _two_addons_sharing_a_key(tmp_path, key="addon.shared",
                              names=("alice", "bob")):
    """Two DIFFERENT add-ons declaring one analysis key.

    The realistic arrival: the documented way to start an add-on is to fork
    the shipped example, and a fork that renames the package and keeps the
    key is the first thing anyone produces.
    """
    for i, name in enumerate(names):
        d = tmp_path / name
        d.mkdir()
        (d / "orienta-addon.toml").write_bytes(
            _SHARED_KEY_MANIFEST.format(
                name=name, key=key, doi=1000000 + i).encode("utf-8"))
    return tmp_path


def test_a_shared_analysis_key_is_shown_as_a_conflict_on_both_rows(
        tmp_path, monkeypatch):
    """The key is the unit of credit: it carries the DOI and the methods
    sentence. Two add-ons on one key means one author's work is cited for the
    other's numbers, so both rows say so before anything runs."""
    monkeypatch.setenv("ORIENTA_ADDON_DIRS",
                       str(_two_addons_sharing_a_key(tmp_path)))
    rows = {a["name"]: a for a in client.get("/api/addons").json()["addons"]}
    for name in ("alice", "bob"):
        assert "addon.shared" in rows[name]["conflict"], rows[name]
        assert "alice" in rows[name]["conflict"]
        assert "bob" in rows[name]["conflict"]


def test_an_addon_whose_key_another_claims_cannot_be_enabled(
        tmp_path, monkeypatch):
    monkeypatch.setenv("ORIENTA_ADDON_DIRS",
                       str(_two_addons_sharing_a_key(tmp_path)))
    r = client.post("/api/addons/alice/enabled", json={"enabled": True})
    assert r.status_code == 409
    assert r.json()["reason"] == addons_route.REASON_ANALYSIS_KEY_CONFLICT
    assert "addon.shared" in r.json()["detail"]
    rows = {a["name"]: a for a in client.get("/api/addons").json()["addons"]}
    assert rows["alice"]["enabled"] is False
    assert rows["bob"]["enabled"] is False


def test_a_shared_key_can_still_be_switched_off(tmp_path, monkeypatch):
    """Disabling must never depend on the add-on being in good standing, or
    the one a user most needs to switch off is the one they cannot."""
    monkeypatch.setenv("ORIENTA_ADDON_DIRS",
                       str(_two_addons_sharing_a_key(tmp_path)))
    r = client.post("/api/addons/alice/enabled", json={"enabled": False})
    assert r.status_code == 200


def test_a_run_of_a_shared_key_is_refused_before_the_addon_is_imported(
        tmp_path, monkeypatch, registered_result):
    """Enabled on a machine where the twin was installed afterwards. The
    refusal comes BEFORE the import: the fixtures' modules do not exist, so a
    gate that fired later would name one instead."""
    trust_file = Path(os.environ["ORIENTA_ADDON_TRUST_FILE"])
    trust_file.write_text(json.dumps(
        {"alice": {"enabled": True, "version": "0.1.0", "doi": "",
                   "crashes": 0}}), encoding="utf-8")
    monkeypatch.setenv("ORIENTA_ADDON_DIRS",
                       str(_two_addons_sharing_a_key(tmp_path)))
    r = client.post("/api/addons/alice/run", json={
        "analysis_key": "addon.shared", "result_id": registered_result,
        "params": {}})
    assert r.status_code == 409
    assert r.json()["reason"] == addons_route.REASON_ANALYSIS_KEY_CONFLICT
    assert "bob" in r.json()["detail"]
    assert "alice_module" not in r.json()["detail"]


# --- a manifest cannot take the trust store down with it --------------------

def test_a_manifest_with_a_toml_date_cannot_wipe_the_trust_store(
        tmp_path, monkeypatch):
    """The whole measured chain, end to end.

    ``version = 2026-01-01`` is a ``datetime.date`` to tomllib, travelled
    untyped into the trust store, and ``json.dump`` raised AFTER the open()
    had truncated the file -- erasing every consent decision on the machine,
    silently, because ``_load`` reads a truncated file as "nothing decided".
    Three fixes stand between here and there; this pins the outcome.
    """
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    assert _entry()["enabled"] is True

    bad = tmp_path / "dated"
    bad.mkdir()
    (bad / "orienta-addon.toml").write_bytes(
        b'api_version = 0\n'
        b'name = "dated"\n'
        b'display_name = "Unquoted version"\n'
        b'version = 2026-01-01\n'
        b'authors = ["T <t@example.org>"]\n'
        b'doi = ""\n\n'
        b'[[analyses]]\n'
        b'key = "addon.dated"\n'
        b'label = "Dated"\n'
        b'python_name = "dated_module:analyse"\n'
        b'sentence = "It ran."\n'
        b'citations = ["orienta"]\n')
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", os.pathsep.join(
        [str(FIXTURES / "runnable_addon"), str(tmp_path)]))

    rejected = next(a for a in client.get("/api/addons").json()["addons"]
                    if a["error"])
    assert "version" in rejected["error"]
    assert "date" in rejected["error"]
    # Shown as rejected, and identifiable: a row with no name still says
    # which file on disk it is.
    assert "dated" in rejected["display_name"]

    assert client.post("/api/addons/dated/enabled",
                       json={"enabled": True}).status_code == 404
    # And the decision the user made earlier is still there.
    assert _entry()["enabled"] is True


def test_two_rejected_manifests_are_told_apart(tmp_path, monkeypatch):
    for folder in ("first", "second"):
        d = tmp_path / folder
        d.mkdir()
        (d / "orienta-addon.toml").write_bytes(b"api_version = 99\n")
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(tmp_path))
    rows = [a for a in client.get("/api/addons").json()["addons"] if a["error"]]
    assert len(rows) == 2
    shown = sorted(row["display_name"] for row in rows)
    assert shown[0] != shown[1]
    assert "first" in shown[0] and "second" in shown[1]


# --- the mount point --------------------------------------------------------

def test_the_mount_prefix_is_the_constant_other_code_builds_urls_from():
    """``API_PREFIX`` is what Task 8b builds a map's ``values_url`` from.
    Until that test exists, nothing measured that the router is actually
    mounted there -- and a docstring claiming a test that does not exist is
    worse than no claim at all."""
    mounted = [r for r in app.routes if getattr(r, "path", None) == API_PREFIX]
    assert mounted, f"nothing is mounted at {API_PREFIX}"
    assert "GET" in mounted[0].methods
    assert client.get(API_PREFIX).status_code == 200


# ---------------------------------------------------------------------------
# Running an add-on, and a map's bytes
# ---------------------------------------------------------------------------

class _XMapWithBandContrast:
    """Just enough of a CrystalMap for pattern_quality._prop_bc to find BC.

    This is the path a light-h5 re-import takes, so it exercises the real
    get_quality_map precedence without needing a 500 MB file on disk.
    """

    phases = ()
    phase_id = None
    rotations = None

    def __init__(self, values):
        self.prop = {"bc": np.asarray(values, dtype=float).ravel()}


class _Result:
    def __init__(self, xmap=None):
        self.original_shape = (3, 4)
        self.metadata = {}
        self.confidence_scores = None
        self.xmap = xmap


@pytest.fixture(autouse=True)
def _clear_stored_maps():
    """The map store is process-level, exactly as ``_result_registry`` is.

    Cleared around every test so one test's map cannot answer another test's
    URL -- which would make
    ``test_fetching_a_map_that_was_never_produced_is_404`` pass or fail on
    collection order rather than on behaviour.
    """
    addons_route._MAP_VALUES.clear()
    yield
    addons_route._MAP_VALUES.clear()


@pytest.fixture
def registered_result():
    indexing_routes._result_registry["addon-test"] = _Result()
    yield "addon-test"
    indexing_routes._result_registry.pop("addon-test", None)


@pytest.fixture
def result_with_quality():
    values = np.arange(12, dtype=float).reshape(3, 4)
    indexing_routes._result_registry["addon-bc"] = _Result(
        _XMapWithBandContrast(values))
    yield "addon-bc", values
    indexing_routes._result_registry.pop("addon-bc", None)


def _refuse_python_only_json(token):
    raise AssertionError(
        f"the response body contains the non-JSON token {token!r}")


def test_running_a_disabled_addon_is_refused_with_a_reason(registered_result):
    client.post("/api/addons/runnable/enabled", json={"enabled": False})
    r = client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": registered_result,
        "params": {}})
    assert r.status_code == 409
    assert "enable" in r.json()["detail"].lower()
    assert r.json()["reason"] == addons_route.REASON_ADDON_NOT_ENABLED


def test_running_an_enabled_addon_returns_its_outputs(registered_result):
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    r = client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": registered_result,
        "params": {}})
    assert r.status_code == 200
    assert "mean" in {o["key"] for o in r.json()["outputs"]}


def test_the_route_hands_the_addon_a_real_quality_map(result_with_quality):
    """build_context(result) alone left quality None, and the example add-on's
    first statement raises on exactly that -- so the acceptance run could only
    ever answer 422."""
    result_id, values = result_with_quality
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    r = client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": result_id,
        "params": {"scale": 2.0}})
    assert r.status_code == 200, r.text
    by_key = {o["key"]: o for o in r.json()["outputs"]}
    assert by_key["mean"]["value"] == pytest.approx(float(values.mean()) * 2.0)


def test_the_quality_source_names_the_metric_not_just_the_provenance(
        result_with_quality):
    """"native" says where a number came from and not WHAT it is, and Oxford
    band contrast and EDAX image quality are different measurements an add-on
    may legitimately branch on. The label carries both."""
    result_id, _ = result_with_quality
    result = indexing_routes._result_registry[result_id]
    context = build_context(result, **addons_route._context_extras(result))
    assert context.quality_source == LABEL_NATIVE
    assert context.quality is not None


def test_the_route_resolves_a_step_size_the_metadata_does_not_carry(monkeypatch):
    """Nothing in product code ever writes metadata['step_size_um'], so
    reading only metadata makes this field None for every real result. The
    resolver is monkeypatched because what it returns depends on a loaded
    signal; what is under test is that the route ASKS it and passes the
    answer on.
    """
    result = _Result()
    assert "step_size_um" not in result.metadata
    monkeypatch.setattr("backend.api.routes.indexing._resolve_step_size_um",
                        lambda r: 0.25)
    context = build_context(result, **addons_route._context_extras(result))
    assert context.step_size_um == 0.25


def test_an_unresolvable_step_size_is_none_not_a_guess():
    result = _Result()
    context = build_context(result, **addons_route._context_extras(result))
    assert context.step_size_um is None


def test_a_resolver_that_raises_costs_the_field_and_not_the_run(monkeypatch):
    """_resolve_step_size_um reaches into the loaded-signal machinery, which
    can fail for reasons that have nothing to do with the add-on. Losing the
    field is honest; losing the run is not."""
    def _boom(_result):
        raise RuntimeError("no signal machinery here")

    monkeypatch.setattr("backend.api.routes.indexing._resolve_step_size_um",
                        _boom)
    result = _Result()
    context = build_context(result, **addons_route._context_extras(result))
    assert context.step_size_um is None


def test_a_map_output_travels_as_bytes_not_as_json(result_with_quality):
    """M4, asserted against the RAW bytes the server sent.

    The previous form re-encoded an ALREADY-PARSED body with allow_nan=True,
    so it could not see the one thing it exists to see.
    """
    result_id, values = result_with_quality
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    response = client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": result_id,
        "params": {"scale": 2.0}})
    assert response.status_code == 200, response.text

    body = json.loads(response.content.decode("utf-8"),
                      parse_constant=_refuse_python_only_json)
    scaled = next(o for o in body["outputs"] if o["key"] == "scaled")
    assert scaled["kind"] == "map"
    assert "values" not in scaled
    assert scaled["shape"] == [3, 4]

    # Built from API_PREFIX; if that ever stops matching the mount in main.py
    # this is a 404 rather than a silently wrong link.
    assert scaled["values_url"].startswith(API_PREFIX)
    raw = client.get(scaled["values_url"])
    assert raw.status_code == 200
    assert raw.headers["content-type"].startswith("application/octet-stream")
    back = np.frombuffer(raw.content,
                         dtype=np.dtype(scaled["dtype"])).reshape(scaled["shape"])
    assert np.array_equal(back, values * 2.0)


def test_the_stored_bytes_are_the_runs_own_and_not_a_later_mutation(
        result_with_quality):
    """validate_outputs freezes a VIEW of the add-on's array, and view.base
    stays writable -- documented in outputs.py and pinned there. An add-on
    that keeps a reference could therefore change what this endpoint serves
    seconds after the run answered, under a URL the run minted. The store
    holds its own copy, so the bytes are the ones the run reported."""
    result_id, values = result_with_quality
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    body = client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": result_id,
        "params": {}}).json()
    scaled = next(o for o in body["outputs"] if o["key"] == "scaled")

    # The store keeps a StoredMap (values plus what a legend needs) since the
    # layer painter; the copy this test is about is still ``.values``.
    stored = addons_route._MAP_VALUES[
        ("runnable", result_id, "addon.mean_quality", "scaled")].values
    assert stored.flags["OWNDATA"], (
        "the store holds a view of the add-on's own buffer; the add-on can "
        "still write through view.base and change what this URL serves")
    back = np.frombuffer(client.get(scaled["values_url"]).content,
                         dtype=np.dtype(scaled["dtype"])).reshape(3, 4)
    assert np.array_equal(back, values)


def test_fetching_a_map_that_was_never_produced_is_404(registered_result):
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    r = client.get(
        f"/api/addons/runnable/outputs/{registered_result}/"
        "addon.mean_quality/nope")
    assert r.status_code == 404
    assert r.json()["reason"] == addons_route.REASON_MAP_NOT_STORED


def test_an_evicted_map_says_so_rather_than_serving_a_zero_array(
        result_with_quality, monkeypatch):
    """The store is capped. An eviction that answered with an empty body, or
    with zeros, would hand a client a map that looks measured and is not."""
    result_id, _ = result_with_quality
    monkeypatch.setattr(addons_route, "MAX_STORED_MAPS", 1)
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    body = client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": result_id,
        "params": {}}).json()
    url = next(o for o in body["outputs"] if o["key"] == "scaled")["values_url"]
    assert client.get(url).status_code == 200

    addons_route._remember_map(
        "other", result_id, "addon.other",
        MapOutput(key="k", label="Other", values=np.zeros((3, 4))))
    evicted = client.get(url)
    assert evicted.status_code == 404
    assert "run the analysis again" in evicted.json()["detail"]


def test_an_undeclared_parameter_is_reported_back(registered_result):
    """Reported, not silently dropped: a run that quietly ignored a setting
    the user made is the other half of the same dishonesty."""
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    body = client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": registered_result,
        "params": {"leak": "C:/Users/someone/scan.h5oina"}}).json()
    assert body["ignored_params"] == ["leak"]


def test_an_undeclared_path_param_never_reaches_the_trail(
        registered_result, monkeypatch):
    """The trail goes into the methods paragraph and every exported .h5, and
    `params` is an unauthenticated JSON body.

    STRICT IS TURNED OFF ON PURPOSE. record_step raises under pytest and only
    warns in production, so with the leak re-introduced this test PASSED under
    plain pytest -- the run 422'd before anything was recorded, which is not
    what its own docstring describes. "0" is production's mode, and production
    is where the leak would happen.
    """
    from backend.api.services.citations.provenance import get_steps

    monkeypatch.setenv("ORIENTA_CITATIONS_STRICT", "0")
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": registered_result,
        "params": {"leak": "C:/Users/someone/secret/scan.h5oina"}})
    result = indexing_routes._result_registry[registered_result]
    assert "someone" not in repr(get_steps(result))


def test_a_declared_path_param_reaches_the_addon_and_not_the_trail(
        registered_result, monkeypatch):
    """``report_dir`` is declared with format="path": the add-on is GIVEN it
    and the trail must not be. Measured on the trail with strict off, for the
    same reason as above -- under pytest's default the run would 422 before
    recording anything, and a status-code assertion would pass with the
    exclusion removed."""
    from backend.api.services.citations.provenance import get_steps

    monkeypatch.setenv("ORIENTA_CITATIONS_STRICT", "0")
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    r = client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": registered_result,
        "params": {"report_dir": "C:/Users/someone/reports"}})
    assert r.status_code == 200, r.text
    assert r.json()["ignored_params"] == []     # declared, so not ignored
    result = indexing_routes._result_registry[registered_result]
    assert "someone" not in repr(get_steps(result))


def test_an_unknown_result_id_is_404(registered_result):
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    r = client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": "nope", "params": {}})
    assert r.status_code == 404


def test_the_two_404s_a_run_can_answer_are_told_apart(registered_result):
    """Same number, two different things to do about it. Without a code a
    client has to match English to know whether to re-run the indexing or
    install the add-on."""
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    no_addon = client.post("/api/addons/nobody/run", json={
        "analysis_key": "addon.mean_quality", "result_id": registered_result,
        "params": {}})
    no_result = client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": "nope", "params": {}})
    assert no_addon.status_code == no_result.status_code == 404
    assert no_addon.json()["reason"] == addons_route.REASON_ADDON_NOT_FOUND
    assert no_result.json()["reason"] == addons_route.REASON_RESULT_NOT_FOUND
    assert no_addon.json()["reason"] != no_result.json()["reason"]


def test_the_two_409s_on_enable_are_told_apart_without_reading_english(
        tmp_path, monkeypatch):
    """Both refusals are 409 and differed only by a sentence, so a client had
    to string-match to tell "needs a newer Orienta" from "two add-ons claim
    this name" -- and the sentence is the part most likely to be reworded or
    translated."""
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(FIXTURES / "future_addon"))
    incompatible = client.post("/api/addons/future/enabled",
                               json={"enabled": True})
    monkeypatch.setenv("ORIENTA_ADDON_DIRS",
                       str(_two_addons_named(tmp_path, "twins")))
    conflict = client.post("/api/addons/twins/enabled", json={"enabled": True})
    assert incompatible.status_code == conflict.status_code == 409
    assert incompatible.json()["reason"] == addons_route.REASON_ADDON_INCOMPATIBLE
    assert conflict.json()["reason"] == addons_route.REASON_ADDON_NAME_CONFLICT
    assert incompatible.json()["reason"] != conflict.json()["reason"]


def test_every_refusal_still_carries_the_sentence_a_person_reads():
    """The code is FOR a machine and does not replace the sentence. A body
    that carried only a code would move the explanation out of the one place
    a user and a bug report can both see it."""
    r = client.post("/api/addons/nobody/enabled", json={"enabled": True})
    assert r.status_code == 404
    assert isinstance(r.json()["detail"], str) and r.json()["detail"]
    assert r.json()["reason"] == addons_route.REASON_ADDON_NOT_FOUND


def test_an_addon_failure_is_a_4xx_naming_the_addon(registered_result):
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    r = client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": registered_result,
        "params": {"fail": True}})
    assert r.status_code == ADDON_FAILED_STATUS
    assert "runnable" in r.json()["detail"]
    assert r.json()["reason"] == addons_route.REASON_ADDON_FAILED


def test_a_run_failure_is_not_confused_with_a_malformed_request(
        registered_result):
    """The run half of what the enable half already pins.

    ADDON_FAILED_STATUS exists because a caller that cannot tell "your
    request was malformed" from "their code raised" reports the second as the
    first. A request-validation answer carries no ``reason`` at all (main.py
    normalises it to a sentence plus ``errors``), so a client that branched on
    the status alone had to read English to tell them apart -- which is what
    the run path forced until this test existed.
    """
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    failed = client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": registered_result,
        "params": {"fail": True}})
    malformed = client.post("/api/addons/runnable/run",
                            json={"analysis_key": "addon.mean_quality"})
    assert failed.status_code == ADDON_FAILED_STATUS
    assert malformed.status_code == 422
    assert failed.status_code != malformed.status_code
    assert failed.json()["reason"] == addons_route.REASON_ADDON_FAILED
    assert "reason" not in malformed.json()      # a validation answer has none
    assert "errors" in malformed.json()          # main.py's own shape


def test_a_result_the_context_cannot_use_answers_like_a_failed_dependency(
        monkeypatch):
    """Not 422 either, and for the same reason: the REQUEST was fine. What
    could not be produced is the context, out of a result whose scan shape is
    unusable -- and ``set_enabled`` already answers this class with 424, so
    422 here left the two routes disagreeing with each other."""
    indexing_routes._result_registry["addon-shapeless"] = _Result()
    indexing_routes._result_registry["addon-shapeless"].original_shape = ()
    try:
        client.post("/api/addons/runnable/enabled", json={"enabled": True})
        r = client.post("/api/addons/runnable/run", json={
            "analysis_key": "addon.mean_quality",
            "result_id": "addon-shapeless", "params": {}})
    finally:
        indexing_routes._result_registry.pop("addon-shapeless", None)
    assert r.status_code == ADDON_FAILED_STATUS
    assert r.json()["reason"] == addons_route.REASON_RESULT_UNUSABLE
    assert isinstance(r.json()["detail"], str) and r.json()["detail"]


def test_a_trust_store_that_fails_while_recording_a_crash_still_names_the_addon(
        registered_result, monkeypatch):
    """The bookkeeping inside the guard must not beat the guard.

    ``record_crash`` sits in every except clause of ``run_analysis``.
    Unguarded, a read-only trust store there replaces "runnable /
    addon.mean_quality: this add-on chose to fail" with an unattributed 500 --
    the exact outcome the guard exists to prevent, arriving only on the path
    where something has already gone wrong.
    """
    def _boom(*a, **k):
        raise OSError("read-only file system")

    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    monkeypatch.setattr(TrustStore, "record_crash", _boom)
    r = client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": registered_result,
        "params": {"fail": True}})
    assert r.status_code == ADDON_FAILED_STATUS
    detail = r.json()["detail"]
    assert detail.startswith("runnable")
    assert "chose to fail" in detail


def test_a_trust_store_that_fails_while_clearing_crashes_does_not_lose_the_run(
        registered_result, monkeypatch):
    """The same unguarded call on the SUCCESS path: an add-on that ran and
    produced its outputs must not be reported as failed because Orienta could
    not write its own file."""
    def _boom(*a, **k):
        raise OSError("read-only file system")

    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    monkeypatch.setattr(TrustStore, "clear_crashes", _boom)
    r = client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": registered_result,
        "params": {}})
    assert r.status_code == 200, r.text
    assert "mean" in {o["key"] for o in r.json()["outputs"]}


def test_a_nan_colour_bound_is_refused_before_it_can_reach_the_wire(
        registered_result, monkeypatch):
    """The other half of the raw-bytes assertion, and the half that bites.

    ``outputs.py`` refuses a non-finite colour bound; this measures what
    happens through the whole HTTP path if it ever stops. Starlette's
    JSONResponse encodes with ``allow_nan=False``, so a NaN that gets past the
    guard does not spoil one field -- it takes the WHOLE response down as an
    unattributed error. The bytes are read BEFORE the status code, because
    ``json.loads(json.dumps(body))`` over an already-parsed body re-encodes
    with allow_nan=True and cannot see a ``NaN`` token.
    """
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(FIXTURES / "nan_bound_addon"))
    client.post("/api/addons/nanbound/enabled", json={"enabled": True})
    r = client.post("/api/addons/nanbound/run", json={
        "analysis_key": "addon.nan_bound", "result_id": registered_result,
        "params": {}})
    json.loads(r.content.decode("utf-8"),
               parse_constant=_refuse_python_only_json)
    assert r.status_code == ADDON_FAILED_STATUS
    assert "nanbound" in r.json()["detail"]
    assert "vmin" in r.json()["detail"]


def test_an_addon_enabled_on_an_older_build_is_refused_at_run(
        monkeypatch, registered_result):
    """The realistic case the run-time gate exists for: the decision was
    recorded when this build did satisfy it, and a later Orienta does not."""
    trust_file = Path(os.environ["ORIENTA_ADDON_TRUST_FILE"])
    trust_file.write_text(json.dumps(
        {"future": {"enabled": True, "version": "0.1.0", "doi": "",
                    "crashes": 0}}), encoding="utf-8")
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(FIXTURES / "future_addon"))
    r = client.post("/api/addons/future/run", json={
        "analysis_key": "addon.future", "result_id": registered_result,
        "params": {}})
    assert r.status_code == 409
    assert ">=99" in r.json()["detail"]
    assert r.json()["reason"] == addons_route.REASON_ADDON_INCOMPATIBLE
    # And nothing was imported to find that out: the fixture's module does
    # not exist, so a gate that fired after the import would say so.
    assert "never_imported_module" not in r.json()["detail"]


def _to_thread_spy(monkeypatch):
    """Record the NAME of everything offloaded, not merely that something was.

    "Some call went to a thread" is satisfied by any one of them, so it stays
    green while the expensive one moves back onto the loop -- a guard the
    correct and the broken implementation both pass.
    """
    seen = []
    real = addons_route.asyncio.to_thread

    async def _spy(fn, *a, **kw):
        seen.append(getattr(fn, "__name__", repr(fn)))
        return await real(fn, *a, **kw)

    monkeypatch.setattr(addons_route.asyncio, "to_thread", _spy)
    return seen


def test_the_run_does_not_block_the_event_loop(registered_result, monkeypatch):
    """An add-on of unknown duration must not stall every other request --
    and neither may the disk reads around it."""
    seen = _to_thread_spy(monkeypatch)
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": registered_result,
        "params": {}})
    assert {"_find", "_trust_store", "_context_for", "run_analysis"} <= set(seen), seen


def test_the_citation_panel_sees_the_addon_step(registered_result):
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": registered_result,
        "params": {}})
    body = client.get(f"/api/citations/result/{registered_result}").json()
    assert "Mean pattern quality" in body["methods"]


def test_the_addons_doi_reaches_the_bibliography(registered_result):
    """End to end. routes/citations.py filters ids with `cid in library`, so
    an entry that never joins load_library() reaches neither BibTeX nor the
    plain list nor the exported .h5 -- only the sentence shows up."""
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": registered_result,
        "params": {}})
    body = client.get(f"/api/citations/result/{registered_result}").json()
    assert "10.5281/zenodo.7777777" in body["bibtex"]
    assert "10.5281/zenodo.7777777" in body["plain"]
    assert body["undeclared"] == []


def test_the_addons_doi_reaches_an_exported_h5(registered_result, tmp_path):
    """The OTHER half of the spec's sentence: "the citation panel AND the
    exported .h5 treat an add-on's works exactly like Orienta's own".

    Driven through the real result_exporter._write_citations, not asserted in
    prose. It filters with ``cid in library`` exactly as the panel does, so
    without Task 6's library registration the add-on's DOI is dropped here
    while the panel test above still passes.
    """
    import h5py

    from backend.api.services.citations.provenance import get_steps
    from backend.api.services.result_exporter import _write_citations

    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": registered_result,
        "params": {}})

    result = indexing_routes._result_registry[registered_result]
    out = tmp_path / "exported.h5"
    with h5py.File(out, "w") as fh:
        _write_citations(fh.create_group("Indexing"), get_steps(result))

    with h5py.File(out, "r") as fh:
        group = fh["/Indexing/Citations"]
        ids = [e["id"] for e in json.loads(group.attrs["entries"])]
        steps = json.loads(group.attrs["steps"])

    assert "doi:10.5281/zenodo.7777777" in ids
    assert [s["key"] for s in steps] == ["addon.mean_quality"]


# --- the map store tells two analyses apart, and cannot eat its own output ---

def _map_url_of(body, key="m"):
    return next(o for o in body["outputs"] if o["key"] == key)["values_url"]


def test_two_analyses_of_one_addon_do_not_overwrite_each_others_maps(
        registered_result, monkeypatch):
    """The DESIGNED path, not a hypothetical one.

    The spec's plan for a map is that it becomes a layer in the existing stack,
    which means its URL is held across runs and refreshed -- not fetched once.
    So: run A, keep its layer, run B of the same add-on, refresh, and A's layer
    draws B's numbers under A's label. Shape and dtype match, so nothing looks
    wrong. This package has shipped two defects of exactly this shape already,
    both of them "just a key".
    """
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(FIXTURES / "two_maps_addon"))
    client.post("/api/addons/twomaps/enabled", json={"enabled": True})

    first = client.post("/api/addons/twomaps/run", json={
        "analysis_key": "addon.first", "result_id": registered_result,
        "params": {}})
    assert first.status_code == 200, first.text
    url_a = _map_url_of(first.json())

    second = client.post("/api/addons/twomaps/run", json={
        "analysis_key": "addon.second", "result_id": registered_result,
        "params": {}})
    assert second.status_code == 200, second.text
    url_b = _map_url_of(second.json())

    assert url_a != url_b, "one URL for two analyses -- the second wins"
    back_a = np.frombuffer(client.get(url_a).content, dtype=np.float64)
    back_b = np.frombuffer(client.get(url_b).content, dtype=np.float64)
    assert np.all(back_a == 1.0), "analysis A's URL now serves B's numbers"
    assert np.all(back_b == 2.0)


def test_a_run_cannot_evict_its_own_maps(registered_result, monkeypatch):
    """Measured before the fix: an add-on returning more maps than the cap
    evicted its first DURING its own serialisation, and the 200 then advertised
    a values_url that 404s with "run the analysis again" -- a remedy that
    cannot ever work, because running it again evicts it again."""
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(FIXTURES / "two_maps_addon"))
    monkeypatch.setattr(addons_route, "MAX_STORED_MAPS", 2)
    client.post("/api/addons/twomaps/enabled", json={"enabled": True})
    body = client.post("/api/addons/twomaps/run", json={
        "analysis_key": "addon.many", "result_id": registered_result,
        "params": {}}).json()

    urls = [o["values_url"] for o in body["outputs"] if o["kind"] == "map"]
    assert len(urls) == 3           # more maps than the cap, in one run
    for i, url in enumerate(urls):
        r = client.get(url)
        assert r.status_code == 200, f"the 200 advertised a dead URL: {url}"
        assert np.all(np.frombuffer(r.content, dtype=np.float64) == float(i))


def test_a_later_run_still_displaces_an_older_one(registered_result,
                                                  result_with_quality,
                                                  monkeypatch):
    """The cap still does its job. "A run cannot evict ITSELF" must not quietly
    become "nothing is ever evicted", which would make the store a leak."""
    other_id, _ = result_with_quality
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(FIXTURES / "two_maps_addon"))
    monkeypatch.setattr(addons_route, "MAX_STORED_MAPS", 1)
    client.post("/api/addons/twomaps/enabled", json={"enabled": True})

    old = client.post("/api/addons/twomaps/run", json={
        "analysis_key": "addon.first", "result_id": registered_result,
        "params": {}}).json()
    url_old = _map_url_of(old)
    assert client.get(url_old).status_code == 200

    client.post("/api/addons/twomaps/run", json={
        "analysis_key": "addon.first", "result_id": other_id, "params": {}})
    assert len(addons_route._MAP_VALUES) == 1
    assert client.get(url_old).status_code == 404


# --- a map's bytes answer to the same gates the run does --------------------

def test_a_disabled_addons_stored_map_stops_serving(result_with_quality):
    """A decision to switch an add-on off should stop its data being served.
    Not a security boundary -- an add-on runs in this process with the user's
    permissions -- but a store that outlives an uninstall is a surprise."""
    result_id, _ = result_with_quality
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    body = client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": result_id,
        "params": {}}).json()
    url = _map_url_of(body, "scaled")
    assert client.get(url).status_code == 200

    client.post("/api/addons/runnable/enabled", json={"enabled": False})
    r = client.get(url)
    assert r.status_code == 409
    assert r.json()["reason"] == addons_route.REASON_ADDON_NOT_ENABLED


def test_a_map_url_for_an_addon_that_is_not_installed_is_404(registered_result):
    r = client.get(
        f"/api/addons/nobody/outputs/{registered_result}/addon.x/m")
    assert r.status_code == 404
    assert r.json()["reason"] == addons_route.REASON_ADDON_NOT_FOUND


# --- the run body says what the context could and could not carry -----------

def test_the_run_body_shows_the_context_fields_it_could_not_fill(
        registered_result):
    """The global constraint, verbatim: what cannot be produced is SHOWN as
    missing, never silently dropped. ``_context_extras`` degrades both fields
    at logger.debug -- off in every real deployment -- so an add-on computing
    lengths in micrometres against ``step_size_um = None`` hands back numbers
    the caller cannot tell apart from resolved ones."""
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    body = client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": registered_result,
        "params": {}}).json()
    assert body["context"] == {"step_size_um": None, "quality_source": None}


def test_the_run_body_reports_the_context_fields_it_could_fill(
        result_with_quality, monkeypatch):
    """And the other half: a null must mean "absent", not "we never say".

    A block that were always null would pass the test above and tell the caller
    nothing, so the filled case is pinned too -- and against the same LABEL the
    context carries, not against "native".
    """
    result_id, _ = result_with_quality
    monkeypatch.setattr("backend.api.routes.indexing._resolve_step_size_um",
                        lambda r: 0.25)
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    body = client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": result_id,
        "params": {}}).json()
    assert body["context"]["step_size_um"] == 0.25
    assert body["context"]["quality_source"] == LABEL_NATIVE


# --- nothing that touches a disk runs on the event loop ---------------------

def _on_the_event_loop():
    """True when called on a thread that is running an asyncio loop.

    ``threading.main_thread()`` is the wrong test here: TestClient drives the
    app through an anyio portal, so the loop runs on a worker thread and every
    check against "main" answers False whether or not anything blocks. A
    ``to_thread`` worker has no running loop; the loop's own thread does.
    """
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def test_nothing_that_touches_a_disk_runs_on_the_event_loop(
        registered_result, monkeypatch):
    """``_find`` reads every manifest under $ORIENTA_ADDON_DIRS -- which may be
    a network mount -- ``TrustStore()`` reads a JSON file, and
    ``_context_extras`` opens the source file looking for native band contrast.
    All three sat before the call this task was written to protect."""
    seen = {}
    real_find, real_store = addons_route._find, addons_route._trust_store
    real_extras = addons_route._context_extras

    def _find_spy(name):
        seen["find"] = _on_the_event_loop()
        return real_find(name)

    def _store_spy():
        seen["store"] = _on_the_event_loop()
        return real_store()

    def _extras_spy(result):
        seen["extras"] = _on_the_event_loop()
        return real_extras(result)

    monkeypatch.setattr(addons_route, "_find", _find_spy)
    monkeypatch.setattr(addons_route, "_trust_store", _store_spy)
    monkeypatch.setattr(addons_route, "_context_extras", _extras_spy)
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    r = client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": registered_result,
        "params": {}})
    assert r.status_code == 200, r.text
    assert seen.get("find") is False, "discovery ran on the event loop"
    assert seen.get("store") is False, "the trust store ran on the event loop"
    assert seen.get("extras") is False, "the context build ran on the event loop"


# --- a refusal points at something that exists ------------------------------

def test_the_refusals_name_the_page_that_enables(registered_result):
    """The remedy has to be a thing that EXISTS, and which thing that is has
    changed once already.

    The earlier form of this test asserted the opposite: it required the
    refusal to quote ``POST /api/addons/<name>/enabled`` and forbade the words
    "in settings", on the stated ground that "API 0 ships no front end" and
    naming a screen nobody had written would send the user hunting for it.
    That ground is gone -- this build ships the Add-ons page -- so the rule
    inverted with it: an HTTP call is now the instruction nobody can follow,
    since the person reading this sentence is reading it inside the app, not
    in a terminal.

    The assertion is therefore two-sided. Naming the page alone would stay
    green if the curl instruction were left beside it.
    """
    client.post("/api/addons/runnable/enabled", json={"enabled": False})
    run = client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": registered_result,
        "params": {}})
    assert run.status_code == 409
    detail = run.json()["detail"]
    assert "Add-ons" in detail, detail
    assert "/api/addons/" not in detail, detail
    assert "POST" not in detail, detail


def test_the_map_refusal_names_the_page_too():
    client.post("/api/addons/runnable/enabled", json={"enabled": False})
    r = client.get(f"{API_PREFIX}/runnable/outputs/x/addon.mean_quality/m")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert "Add-ons" in detail, detail
    assert "/api/addons/" not in detail, detail


def test_serving_a_maps_bytes_does_not_block_the_event_loop(monkeypatch):
    """``_find`` re-reads every manifest under $ORIENTA_ADDON_DIRS -- measured
    at ~37 ms for five add-ons on a local disk, and the directories may be a
    network mount -- and the trust store is another file read. On every byte
    fetch, on the event loop, which is the thing that has made this app look
    dead before.

    Both calls are named. "Something was offloaded" would stay green with the
    manifest scan -- the expensive half -- back on the loop.
    """
    seen = _to_thread_spy(monkeypatch)
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    client.get(f"{API_PREFIX}/runnable/outputs/x/addon.mean_quality/m")
    assert {"_find", "_trust_store"} <= set(seen), seen


# --- the gate asks the same question the bridge will ------------------------

_TWO_ANALYSES_MANIFEST = """api_version = 0
name = "{name}"
display_name = "Analysis by {name}"
version = "0.1.0"
authors = ["{name} <{name}@example.org>"]
doi = ""

[[analyses]]
key = "{first}"
label = "{name} first"
python_name = "{name}_module:analyse"
sentence = "{name} did it."

[[analyses]]
key = "{second}"
label = "{name} second"
python_name = "{name}_module:analyse"
sentence = "{name} did it again."
"""


def _carol_and_bob(tmp_path):
    """Carol declares a clean key AND one that bob already declares."""
    (tmp_path / "bob").mkdir()
    (tmp_path / "bob" / "orienta-addon.toml").write_bytes(
        _SHARED_KEY_MANIFEST.format(
            name="bob", key="addon.k2", doi=1000001).encode("utf-8"))
    (tmp_path / "carol").mkdir()
    (tmp_path / "carol" / "orienta-addon.toml").write_bytes(
        _TWO_ANALYSES_MANIFEST.format(
            name="carol", first="addon.k3",
            second="addon.k2").encode("utf-8"))
    return tmp_path


def test_a_clean_analysis_beside_a_conflicting_one_is_refused_before_import(
        tmp_path, monkeypatch, registered_result):
    """The route gate asked about ONE key; ``register_manifest_citations``
    loops over EVERY analysis in the manifest. So a run of carol's perfectly
    clean ``addon.k3`` passed the gate, imported, RAN, and was then refused by
    the bridge -- the work discarded after it succeeded, and the caller told
    ``addon_failed``, which blames the add-on's code for what its manifest
    did. The gate has to ask the question the bridge will ask.
    """
    trust_file = Path(os.environ["ORIENTA_ADDON_TRUST_FILE"])
    trust_file.write_text(json.dumps(
        {"carol": {"enabled": True, "version": "0.1.0", "doi": "",
                   "crashes": 0}}), encoding="utf-8")
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(_carol_and_bob(tmp_path)))
    r = client.post("/api/addons/carol/run", json={
        "analysis_key": "addon.k3", "result_id": registered_result,
        "params": {}})
    assert r.status_code == 409, r.text
    assert r.json()["reason"] == addons_route.REASON_ANALYSIS_KEY_CONFLICT
    assert "addon.k2" in r.json()["detail"]
    # Before the import: carol_module does not exist, so a gate that fired
    # afterwards would name it.
    assert "carol_module" not in r.json()["detail"]
