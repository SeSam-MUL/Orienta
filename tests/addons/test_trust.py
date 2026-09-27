import json
import threading
from datetime import date

import pytest

from backend.api.services.addons.trust import (
    CRASH_LIMIT,
    TrustStore,
    orienta_version_satisfies,
    running_orienta_version,
    trust_store_path,
)
from backend.api.services.app_version import get_version_info


@pytest.mark.parametrize("requires,version,ok", [
    (">=0.4,<0.6", "0.4.0", True),
    (">=0.4,<0.6", "0.5.9", True),
    (">=0.4,<0.6", "0.6.0", False),
    (">=0.4,<0.6", "0.3.9", False),
    ("", "0.5.0", True),                 # no constraint = any version
    (">=0.4", "0.4", True),
    ("==0.5.1", "0.5.1", True),
    ("==0.5.1", "0.5.2", False),
    # git tags in this project are "v0.3.0", and `release` returns them verbatim.
    (">=0.3", "v0.3.0", True),
    ("<0.6", "v0.5.1", True),
    (">=0.4,<0.6", "v0.4.0", True),
    (">=v0.4", "0.5", True),
    ("~=0.4", "0.5", False),             # unrecognised comparator: refuse
])
def test_version_gate(requires, version, ok):
    assert orienta_version_satisfies(requires, version) is ok


def test_an_unparseable_running_version_satisfies_nothing():
    """A dev build with no tag must not silently pass a floor it may not meet."""
    assert orienta_version_satisfies(">=0.4", "unknown") is False
    assert orienta_version_satisfies(">=0.4", None) is False


@pytest.mark.parametrize("version", ["unknown", None])
@pytest.mark.parametrize("requires", ["<0.6", "<=0.6", "==0.0"])
def test_an_unparseable_running_version_also_fails_upper_bounds_and_equality(
    requires, version,
):
    """A lower bound (">=0.4") is not the only shape a constraint can take.

    A guard that zero-pads instead of refusing outright would make an
    unparseable version read as the smallest possible version — which
    correctly fails ">=0.4" but WRONGLY satisfies "<0.6", "<=0.6" and
    "==0.0". All three must still refuse, the same as the lower-bound case
    above.
    """
    assert orienta_version_satisfies(requires, version) is False


def test_no_constraint_passes_even_an_unparseable_version():
    assert orienta_version_satisfies("", "unknown") is True
    assert orienta_version_satisfies("", None) is True


def test_the_display_string_is_not_a_version():
    """The exact trap this helper exists to avoid: app_version's `version` is
    a DISPLAY string of the form '<date> (<short hash>)' on an untagged
    checkout, which parses as (<year>,) — satisfying '>=0.4' for entirely the
    wrong reason and the spec's own '>=0.4,<0.6' never.

    The literal below is SHAPE, not this checkout's hash: the hash moves with
    every commit, and a test that quoted the real one would rot by the next.
    """
    display = "2026-09-17 (0000000)"
    assert orienta_version_satisfies(">=0.4", display) is True
    assert orienta_version_satisfies(">=0.4,<0.6", display) is False


def test_the_running_version_is_the_release_tag_from_real_version_info():
    """Against the REAL get_version_info() shape, whatever this checkout is."""
    info = get_version_info()
    assert "release" in info and "version" in info
    assert running_orienta_version() == info["release"]
    if info["release"] is None:
        # An untagged checkout cannot say which release it is, so it satisfies
        # nothing that asks for one — and everything that asks for nothing.
        assert orienta_version_satisfies(">=0.4", running_orienta_version()) is False
        assert orienta_version_satisfies("", running_orienta_version()) is True
    else:
        assert orienta_version_satisfies("", running_orienta_version()) is True


def test_unknown_addon_is_neither_enabled_nor_known(tmp_path):
    s = TrustStore(tmp_path / "t.json")
    assert s.is_known("x") is False
    assert s.is_enabled("x") is False


def test_a_decision_survives_a_reload(tmp_path):
    p = tmp_path / "t.json"
    TrustStore(p).set_enabled("x", True, version="1.0", doi="10.5281/zenodo.1")
    assert TrustStore(p).is_enabled("x") is True
    assert TrustStore(p).is_known("x") is True


def test_disabling_is_remembered_too(tmp_path):
    p = tmp_path / "t.json"
    TrustStore(p).set_enabled("x", False, version="1.0", doi="")
    s = TrustStore(p)
    assert s.is_known("x") is True and s.is_enabled("x") is False


def test_repeated_crashes_disable_it(tmp_path):
    p = tmp_path / "t.json"
    s = TrustStore(p)
    s.set_enabled("x", True, version="1.0", doi="")
    for _ in range(CRASH_LIMIT):
        s.record_crash("x")
    assert TrustStore(p).is_enabled("x") is False


def test_a_success_clears_the_crash_count(tmp_path):
    p = tmp_path / "t.json"
    s = TrustStore(p)
    s.set_enabled("x", True, version="1.0", doi="")
    s.record_crash("x")
    s.clear_crashes("x")
    for _ in range(CRASH_LIMIT - 1):
        s.record_crash("x")
    assert s.is_enabled("x") is True


def test_a_corrupt_store_does_not_take_the_app_down(tmp_path):
    p = tmp_path / "t.json"
    p.write_text("{not json", encoding="utf-8")
    s = TrustStore(p)
    assert s.is_enabled("x") is False
    s.set_enabled("x", True, version="1.0", doi="")
    assert json.loads(p.read_text(encoding="utf-8"))["x"]["enabled"] is True


def test_the_recorded_version_and_doi_are_kept_for_the_dialog(tmp_path):
    p = tmp_path / "t.json"
    TrustStore(p).set_enabled("x", True, version="2.1", doi="10.5281/zenodo.9")
    entry = json.loads(p.read_text(encoding="utf-8"))["x"]
    assert entry["version"] == "2.1" and entry["doi"] == "10.5281/zenodo.9"


def test_the_trust_file_location_is_overridable(tmp_path, monkeypatch):
    """Without this the suite writes into the developer's real home directory,
    and the SECOND run of the API tests fails on a file the first left there."""
    monkeypatch.setenv("ORIENTA_ADDON_TRUST_FILE", str(tmp_path / "custom.json"))
    assert trust_store_path() == tmp_path / "custom.json"
    monkeypatch.delenv("ORIENTA_ADDON_TRUST_FILE", raising=False)
    assert ".orienta" in str(trust_store_path())


# --- a failed write must change nothing -------------------------------------

def test_a_failed_save_leaves_the_previous_store_intact(tmp_path):
    """The measured data loss, pinned.

    The old ``_save`` opened the real file for writing -- which TRUNCATES --
    and only then serialised. A value ``json`` cannot encode therefore left a
    truncated file, and ``_load`` reads a truncated file as "nothing decided
    yet": every consent decision on the machine, gone, with no message. The
    unserialisable value is not exotic. ``version = 2026-01-01`` in a
    manifest is a ``datetime.date`` to tomllib, and that is how it was found.
    """
    path = tmp_path / "trust.json"
    store = TrustStore(path)
    store.set_enabled("good", True, version="1.0", doi="10.5281/zenodo.1")
    before = path.read_bytes()

    with pytest.raises(TypeError):
        store.set_enabled("bad", True, version=date(2026, 1, 1), doi="")

    assert path.read_bytes() == before          # byte for byte
    reopened = TrustStore(path)
    assert reopened.is_enabled("good") is True  # and still readable
    assert reopened.is_known("bad") is False
    # The in-memory picture was rolled back too: a store that kept a decision
    # it could not write would answer for something the file cannot back up.
    assert store.is_known("bad") is False
    # And no half-written temp file left lying beside it.
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []


def test_concurrent_decisions_do_not_overwrite_each_other(tmp_path):
    """Every mutator is a read-modify-write, and FastAPI runs sync routes in a
    thread pool, so two enables really do overlap. Each store here reads the
    file BEFORE any of them writes -- exactly the shape of two requests that
    arrive together -- so without a re-read before the modify, the last writer
    wins and the other decisions vanish silently."""
    path = tmp_path / "trust.json"
    stores = [TrustStore(path) for _ in range(20)]      # all read an empty file
    threads = [threading.Thread(
        target=s.set_enabled, args=(f"addon{i}", True),
        kwargs={"version": "1.0", "doi": ""}) for i, s in enumerate(stores)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = TrustStore(path)
    missing = [i for i in range(20) if not final.is_enabled(f"addon{i}")]
    assert missing == []
