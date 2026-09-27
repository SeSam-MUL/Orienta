"""Server mode is off until somebody turns it on.

A fresh macOS installation greeted the M5 tester on 2026-09-25 with "Server
mode" switched ON and a database root that was empty, showing its Windows
example path `Z:\\SharedDrive\\EBSD_Database` as ghost text. Nothing was
shared, nothing was reachable, and nothing had been configured -- the switch
was simply on by default.

The risk is not that it breaks something (an empty root means the page never
even tests a connection). It is that a switch which is on and does nothing
teaches people that the switches on this page do not mean anything.
"""

import json

import pytest

from backend.api.services import user_config_manager as uc


def test_a_fresh_install_has_server_mode_off():
    assert uc._DEFAULT_CONFIG["server"]["enabled"] is False


def test_a_fresh_install_has_no_database_root():
    assert uc._DEFAULT_CONFIG["server"]["database_root"] == ""


@pytest.fixture
def _no_cache(monkeypatch):
    """load_config() memoises in a module global.

    Without clearing it, the first of these tests would decide the answer for
    the second -- and for every other test file that touches the config in the
    same session.
    """
    monkeypatch.setattr(uc, "_CONFIG_CACHE", None, raising=False)
    monkeypatch.setattr(uc, "_MIGRATION_DONE", True, raising=False)
    yield
    uc._CONFIG_CACHE = None


def _load_from(tmp_path, monkeypatch, payload: dict) -> dict:
    """Load a config we wrote, and ONLY that one.

    `load_config` reads `_resolve_path()`, not the public `config_path()`.
    Patching the public name left the real file in play, so the first run of
    this test read the developer's own config -- where server mode is on --
    and reported it as the default. It failed loudly, which is the only
    reason it was noticed.
    """
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(uc, "_resolve_path", lambda: cfg)
    return uc.load_config(run_migration=False)


def test_an_existing_decision_survives(tmp_path, monkeypatch, _no_cache):
    """The default only moves installations that never decided.

    Someone who switched server mode on and saved it must keep it -- this is
    a default, not a migration.
    """
    loaded = _load_from(tmp_path, monkeypatch, {
        "schema_version": 1,
        "server": {"enabled": True, "database_root": "Z:\\Share\\EBSD"},
    })
    assert loaded["server"]["enabled"] is True
    assert loaded["server"]["database_root"] == "Z:\\Share\\EBSD"


def test_a_config_without_a_server_section_gets_the_new_default(tmp_path, monkeypatch, _no_cache):
    loaded = _load_from(tmp_path, monkeypatch, {"schema_version": 1, "api_keys": {}})
    assert loaded["server"]["enabled"] is False


@pytest.mark.parametrize("key", ["manual_paths", "api_keys"])
def test_the_other_sections_are_untouched(key):
    # Positive control: if someone later replaces the whole default block,
    # these keep the rest of the schema honest.
    assert key in uc._DEFAULT_CONFIG
