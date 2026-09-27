"""Why an add-on is switched off, which the listing could not say.

``CRASH_LIMIT`` disables an add-on automatically, and the row the page renders
carried only ``enabled``. So an interface could show the switch and nothing
else: not the count, not the limit, and -- the part that matters -- not whether
the user turned it off or the runtime did. A page that cannot tell those apart
is reporting a state it does not understand, and the user's own decision is the
one it overwrites in the telling.

What this file does NOT test, because API 0 cannot do it: separating an add-on
that REFUSED (validated its input and said no) from one that crashed. There is
nothing between returning and raising in this contract, so an honest refusal is
a ``ValueError`` and so is a bug. Guessing `except ValueError = refusal` would
invert the guard -- a real crash that happens to raise ValueError would stop
counting towards the disable -- so the counter is exposed and the
classification is left to the contract. ``CHANGELOG-ADDON-API.md`` carries it
as a named limit of API 0.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.routes import addons as addons_route
from backend.api.services.addons.trust import CRASH_LIMIT

FIXTURES = Path(__file__).parent / "fixtures"
client = TestClient(app)


@pytest.fixture(autouse=True)
def _addon_dirs(monkeypatch):
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(FIXTURES / "runnable_addon"))


class _XMap:
    def __init__(self, values):
        self.prop = {"bc": values.ravel()}


class _Result:
    def __init__(self):
        self.original_shape = (3, 4)
        self.metadata = {}
        self.xmap = _XMap(np.arange(12, dtype=float))


@pytest.fixture
def registered_result():
    from backend.api.routes import indexing as indexing_routes

    indexing_routes._result_registry["addon-crash-test"] = _Result()
    yield "addon-crash-test"
    indexing_routes._result_registry.pop("addon-crash-test", None)


def _row(name="runnable"):
    body = client.get("/api/addons").json()
    return next(a for a in body["addons"] if a["name"] == name)


def _fail_once(result_id):
    return client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": result_id,
        "params": {"fail": True}})


def test_the_listing_says_how_many_times_it_failed_and_what_the_limit_is():
    """Both numbers, because one without the other says nothing.

    "2 crashes" is only alarming if you know the add-on is switched off at 3,
    and the limit is a constant of the runtime, not of the add-on -- a page
    that hard-codes it is a second copy that goes stale silently.
    """
    row = _row()
    assert row["crashes"] == 0
    assert row["crash_limit"] == CRASH_LIMIT


def test_a_failing_run_is_counted_and_the_count_is_visible(registered_result):
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    _fail_once(registered_result)
    assert _row()["crashes"] == 1


def test_a_successful_run_clears_the_count(registered_result):
    """Already true; pinned here so this task cannot quietly break it.

    A counter that only rises would disable every add-on that is used often
    enough, which is the opposite of what it is for: three failures IN A ROW
    mean broken, three failures across a month of work mean a hard dataset.
    """
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    _fail_once(registered_result)
    assert _row()["crashes"] == 1
    ok = client.post("/api/addons/runnable/run", json={
        "analysis_key": "addon.mean_quality", "result_id": registered_result,
        "params": {}})
    assert ok.status_code == 200, ok.text
    assert _row()["crashes"] == 0


def test_the_runtime_disabling_an_addon_is_distinguishable_from_the_user_doing_it(
        registered_result):
    """Two states, not one boolean -- and the page must be able to tell them
    apart to say anything true about the switch.

    Both read ``enabled: false``. What separates them is the counter, and it
    separates them cleanly because ``set_enabled`` ZEROES it: a user-disabled
    add-on is (false, 0), a runtime-disabled one is (false, CRASH_LIMIT).
    """
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    for _ in range(CRASH_LIMIT):
        _fail_once(registered_result)
    by_runtime = _row()
    assert by_runtime["enabled"] is False
    assert by_runtime["crashes"] >= CRASH_LIMIT
    assert by_runtime["disabled_by"] == "runtime"

    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    client.post("/api/addons/runnable/enabled", json={"enabled": False})
    by_user = _row()
    assert by_user["enabled"] is False
    assert by_user["crashes"] == 0
    assert by_user["disabled_by"] == "user"


def test_an_enabled_addon_is_not_described_as_disabled_by_anyone():
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    row = _row()
    assert row["enabled"] is True
    assert row["disabled_by"] is None


def test_an_addon_nobody_has_decided_about_is_not_called_user_disabled():
    """A never-enabled add-on reads ``enabled: false`` too.

    Calling that "the user switched it off" would put words in the user's
    mouth about a decision they never made -- and it is the first thing every
    freshly installed add-on would say about itself.
    """
    trust_file = Path(os.environ["ORIENTA_ADDON_TRUST_FILE"])
    if trust_file.exists():
        trust_file.unlink()
    row = _row()
    assert row["known"] is False
    assert row["enabled"] is False
    assert row["disabled_by"] is None


def test_an_addon_that_fails_to_activate_is_not_blamed_on_the_user(monkeypatch):
    """The user pressed ENABLE. They must not be told they switched it off.

    Orienta writes the trust row itself when an add-on raises on activation --
    deliberately, so an add-on that WAS working and broke overnight reads as
    disabled rather than untouched. The row it leaves,
    ``(enabled=false, known=true, crashes=0)``, is byte for byte the row a
    user's own switch-off leaves, so inferring from the state produced exactly
    the sentence the feature exists to avoid. Measured on this fixture before
    the fix: ``disabled_by == "user"``.
    """
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(FIXTURES / "exploding_addon"))
    r = client.post("/api/addons/exploding/enabled", json={"enabled": True})
    assert r.status_code >= 400, r.text          # the import really did fail

    row = _row("exploding")
    assert row["enabled"] is False
    assert row["crashes"] == 0                   # it never ran, so it never crashed
    assert row["disabled_by"] == "runtime"


def test_a_later_success_does_not_rewrite_who_switched_it_off(registered_result):
    """A cleared counter must not turn a runtime decision into the user's.

    Two runs in flight make this reachable: the third failure disables, a
    fourth that was already running succeeds and zeroes the count, and the
    row becomes (false, 0) -- which, inferred, reads as the user's doing.
    """
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    for _ in range(CRASH_LIMIT):
        _fail_once(registered_result)
    assert _row()["disabled_by"] == "runtime"

    from backend.api.services.addons.trust import TrustStore

    TrustStore().clear_crashes("runnable")
    row = _row()
    assert row["crashes"] == 0
    assert row["disabled_by"] == "runtime", (
        "the counter is not the record -- the decision is")


def test_switching_it_back_on_clears_the_record(registered_result):
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    for _ in range(CRASH_LIMIT):
        _fail_once(registered_result)
    assert _row()["disabled_by"] == "runtime"
    client.post("/api/addons/runnable/enabled", json={"enabled": True})
    assert _row()["disabled_by"] is None


def test_a_trust_file_from_before_this_field_still_reads_correctly():
    """Migration on read: an old row has no ``disabled_by`` at all.

    Those files predate the failed-activation path writing rows of its own,
    so inferring from (enabled, known, crashes) is right for them -- which is
    why the inference stays as a fallback rather than being deleted.
    """
    trust_file = Path(os.environ["ORIENTA_ADDON_TRUST_FILE"])
    trust_file.write_text(json.dumps({
        "runnable": {"enabled": False, "version": "0.1.0", "doi": "",
                     "crashes": 0},
    }), encoding="utf-8")
    assert _row()["disabled_by"] == "user"

    trust_file.write_text(json.dumps({
        "runnable": {"enabled": False, "version": "0.1.0", "doi": "",
                     "crashes": CRASH_LIMIT},
    }), encoding="utf-8")
    assert _row()["disabled_by"] == "runtime"


def test_a_rejected_manifest_still_carries_the_same_keys(monkeypatch):
    """The row that most needs to be seen must not take the table down.

    A rejected manifest is rendered with the same keys as a good one -- the
    listing code says so -- so the two new fields belong there as well.
    """
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(FIXTURES / "broken_addon"))
    body = client.get("/api/addons").json()
    assert body["addons"], "the broken fixture must still be listed"
    # Asserted, so this cannot silently move to the good-row path if the
    # fixture ever starts parsing: a rejected row is the one with no name.
    assert any(row["name"] == "" and row["error"] for row in body["addons"]), \
        "this test is only meaningful against a REJECTED manifest"
    for row in body["addons"]:
        assert "crashes" in row
        assert "crash_limit" in row
        assert "disabled_by" in row
