"""Unit tests for user_config_manager.

Critical behaviours under test:
  - Config path resolves to a per-machine location (NOT inside the project tree).
  - One-shot legacy migration moves data/server_config.json → user-home config
    AND deletes the legacy files (so they don't get committed/transferred).
  - Default config schema always has the api_keys / server / manual_paths
    sections, even on a fresh first run.
  - API key redaction never returns the full key.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.api.services import user_config_manager as uc


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Redirect every test to its own tempdir config + clear caches."""
    target = tmp_path / "user_home" / "config.json"
    uc.set_config_path_for_test(target)
    # Also stop the legacy-migration probe from hitting the real project tree
    # by pointing both legacy paths inside tmp_path so they don't exist.
    monkeypatch.setattr(uc, "_legacy_server_path", lambda: tmp_path / "absent_server.json")
    monkeypatch.setattr(uc, "_legacy_manual_paths_path", lambda: tmp_path / "absent_paths.json")
    yield target
    uc.set_config_path_for_test(None)
    uc.reset_cache_for_test()


# --- Path resolution -----------------------------------------------------

def test_user_config_dir_is_outside_project_tree():
    """Real per-machine dir must NEVER live inside the project tree."""
    real_dir = uc._user_config_dir()
    project_root = Path(__file__).resolve().parents[1]
    real_dir_resolved = real_dir.resolve()
    # The user-config dir must not be under the project root.
    try:
        real_dir_resolved.relative_to(project_root)
        pytest.fail(
            f"user config dir {real_dir_resolved} is INSIDE project tree "
            f"{project_root} — would travel with project files!"
        )
    except ValueError:
        pass  # not under project_root — good


def test_config_path_uses_os_appdata_on_windows():
    """On Windows, must resolve under %APPDATA%."""
    if os.name != "nt":
        pytest.skip("Windows-only behaviour")
    appdata = os.environ.get("APPDATA")
    if not appdata:
        pytest.skip("APPDATA not set in this environment")
    real_dir = uc._user_config_dir()
    assert str(real_dir).startswith(appdata), (
        f"expected dir under APPDATA={appdata}, got {real_dir}"
    )


# --- Default config ------------------------------------------------------

def test_load_config_returns_defaults_on_fresh_install():
    """No config file exists yet → return merged defaults."""
    cfg = uc.load_config()
    assert cfg["schema_version"] == 1
    assert "api_keys" in cfg
    assert "materials_project" in cfg["api_keys"]
    assert cfg["api_keys"]["materials_project"] == ""
    # Server mode is OFF on a fresh install as of 2026-09-25. It used to be on,
    # with an empty database_root, so a new installation came up advertising a
    # share that did not exist -- see tests/test_server_mode_default.py.
    assert cfg["server"]["enabled"] is False
    assert cfg["manual_paths"]["emsoft_bin_dir"] == ""


def test_save_then_load_round_trips(isolated_config):
    """Save → load returns the saved values."""
    uc.save_config({
        "server": {"enabled": False, "database_root": "/mnt/srv", "offline_mode": True},
        "api_keys": {"materials_project": "mp_secret_xyz"},
    })
    uc.reset_cache_for_test()
    cfg = uc.load_config()
    assert cfg["server"]["enabled"] is False
    assert cfg["server"]["database_root"] == "/mnt/srv"
    assert cfg["api_keys"]["materials_project"] == "mp_secret_xyz"


def test_update_section_merges_not_replaces(isolated_config):
    """update_section keeps other keys in the same section."""
    uc.save_config({
        "server": {"enabled": True, "database_root": "/a", "offline_mode": False},
    })
    uc.update_section("server", {"offline_mode": True})
    cfg = uc.load_config()
    assert cfg["server"]["enabled"] is True              # preserved
    assert cfg["server"]["database_root"] == "/a"        # preserved
    assert cfg["server"]["offline_mode"] is True         # updated


def test_set_api_key_persists():
    uc.set_api_key("materials_project", "mp_abc123")
    uc.reset_cache_for_test()
    assert uc.get_api_key("materials_project") == "mp_abc123"
    assert uc.has_api_key("materials_project") is True


def test_get_api_key_empty_when_not_set():
    assert uc.get_api_key("materials_project") == ""
    assert uc.has_api_key("materials_project") is False


def test_get_api_key_handles_unknown_name():
    assert uc.get_api_key("openai") == ""
    assert uc.has_api_key("openai") is False


# --- Redaction -----------------------------------------------------------

def test_redact_key_hides_all_but_last_4():
    # "mp_abcd1234efgh" is 15 chars → 11 stars + "efgh"
    assert uc.redact_key("mp_abcd1234efgh") == "***********efgh"
    # 5-char "short" — shorter than visible_tail (4) so check the all-stars branch
    assert uc.redact_key("short") == "*****"
    assert uc.redact_key("") == ""


def test_redact_does_not_leak_full_key():
    full = "mp-very-secret-api-key"
    redacted = uc.redact_key(full)
    assert full not in redacted


# --- Migration -----------------------------------------------------------

def test_migration_imports_and_deletes_legacy(tmp_path, monkeypatch):
    """End-to-end migration: legacy files exist → config gets their values →
    legacy files are deleted."""
    target = tmp_path / "userhome" / "config.json"
    uc.set_config_path_for_test(target)
    uc.reset_cache_for_test()
    uc._MIGRATION_DONE = False

    # Set up legacy files in a project-tree-shaped tempdir
    legacy_dir = tmp_path / "project" / "data"
    legacy_dir.mkdir(parents=True)
    legacy_server = legacy_dir / "server_config.json"
    legacy_paths = legacy_dir / "manual_paths.json"
    legacy_server.write_text(json.dumps({
        "enabled": False, "database_root": "/srv/old", "offline_mode": True,
    }))
    legacy_paths.write_text(json.dumps({
        "emsoft_bin_dir": "/opt/old_emsoft", "emsphinx_dir": "/opt/old_sphinx",
    }))
    monkeypatch.setattr(uc, "_legacy_server_path", lambda: legacy_server)
    monkeypatch.setattr(uc, "_legacy_manual_paths_path", lambda: legacy_paths)

    cfg = uc.load_config()

    # 1. Values migrated
    assert cfg["server"]["database_root"] == "/srv/old"
    assert cfg["server"]["enabled"] is False
    assert cfg["server"]["offline_mode"] is True
    assert cfg["manual_paths"]["emsoft_bin_dir"] == "/opt/old_emsoft"

    # 2. Legacy files deleted
    assert not legacy_server.exists(), "legacy server file should be deleted"
    assert not legacy_paths.exists(), "legacy paths file should be deleted"

    # 3. New per-machine file written
    assert target.exists()

    uc.set_config_path_for_test(None)


def test_migration_only_runs_once(tmp_path, monkeypatch):
    """A second load_config call should NOT re-migrate (idempotent)."""
    target = tmp_path / "uh" / "config.json"
    uc.set_config_path_for_test(target)
    uc.reset_cache_for_test()
    uc._MIGRATION_DONE = False

    legacy_dir = tmp_path / "project" / "data"
    legacy_dir.mkdir(parents=True)
    legacy_server = legacy_dir / "server_config.json"
    legacy_server.write_text(json.dumps({"database_root": "/srv1"}))
    monkeypatch.setattr(uc, "_legacy_server_path", lambda: legacy_server)
    monkeypatch.setattr(uc, "_legacy_manual_paths_path", lambda: tmp_path / "nope.json")

    uc.load_config()
    # First call deletes legacy + migrates
    assert not legacy_server.exists()

    # Re-create the legacy file as if someone copied a stale copy back
    legacy_server.write_text(json.dumps({"database_root": "/srv2_STALE"}))
    uc.reset_cache_for_test()
    # Migration should NOT clobber the existing per-machine config with stale data
    cfg2 = uc.load_config()
    # The stale legacy got deleted (cleanup), but the saved per-machine value wins
    assert cfg2["server"]["database_root"] == "/srv1"
    assert not legacy_server.exists()


def test_migration_preserves_corrupt_legacy_as_bak(tmp_path, monkeypatch):
    """Regression for code-review Issue 1: a corrupt legacy file must NOT be
    deleted (would lose user data). Should be renamed to .corrupt.bak so the
    user can recover."""
    target = tmp_path / "userhome" / "config.json"
    uc.set_config_path_for_test(target)
    uc.reset_cache_for_test()
    uc._MIGRATION_DONE = False

    legacy_dir = tmp_path / "project" / "data"
    legacy_dir.mkdir(parents=True)
    legacy_server = legacy_dir / "server_config.json"
    # Write garbage that pymatgen/json can't parse
    legacy_server.write_text("{this is not valid JSON")
    monkeypatch.setattr(uc, "_legacy_server_path", lambda: legacy_server)
    monkeypatch.setattr(uc, "_legacy_manual_paths_path", lambda: tmp_path / "nope.json")

    uc.load_config()

    # The corrupt original must NOT have been silently deleted
    assert not legacy_server.exists()
    backup = legacy_server.with_suffix(legacy_server.suffix + ".corrupt.bak")
    assert backup.exists(), (
        "Corrupt legacy file should be renamed to .corrupt.bak for manual recovery"
    )
    # Bak should contain the original garbage so the user can fix it manually
    assert backup.read_text() == "{this is not valid JSON"


def test_migration_no_op_when_legacy_absent(tmp_path, monkeypatch):
    """No legacy files → fresh defaults, no errors."""
    target = tmp_path / "x" / "config.json"
    uc.set_config_path_for_test(target)
    uc.reset_cache_for_test()
    uc._MIGRATION_DONE = False
    monkeypatch.setattr(uc, "_legacy_server_path", lambda: tmp_path / "nope1.json")
    monkeypatch.setattr(uc, "_legacy_manual_paths_path", lambda: tmp_path / "nope2.json")

    cfg = uc.load_config()
    assert cfg["server"]["database_root"] == ""
    # File should not have been written (no migration happened)
    assert not target.exists()


# --- Atomic write --------------------------------------------------------

def test_atomic_write_does_not_corrupt_on_simulated_error(tmp_path, monkeypatch):
    """If the write fails midway, the original file must remain intact."""
    target = tmp_path / "uh" / "config.json"
    uc.set_config_path_for_test(target)
    uc.reset_cache_for_test()
    uc._MIGRATION_DONE = True
    monkeypatch.setattr(uc, "_legacy_server_path", lambda: tmp_path / "nope1.json")
    monkeypatch.setattr(uc, "_legacy_manual_paths_path", lambda: tmp_path / "nope2.json")

    # Write a valid initial config
    uc.save_config({"server": {"database_root": "/intact"}})
    original = target.read_text()

    # Simulate a write failure inside _write_atomic by patching os.replace
    with patch("backend.api.services.user_config_manager.os.replace",
               side_effect=OSError("boom")):
        with pytest.raises(OSError):
            uc.save_config({"server": {"database_root": "/should_not_apply"}})

    # File must still hold the original content
    assert target.read_text() == original


def test_atomic_write_cleans_up_tempfile_on_error(tmp_path, monkeypatch):
    """Failed save should not leave stray .json.tmp files behind."""
    target = tmp_path / "uh" / "config.json"
    uc.set_config_path_for_test(target)
    uc.reset_cache_for_test()
    uc._MIGRATION_DONE = True
    monkeypatch.setattr(uc, "_legacy_server_path", lambda: tmp_path / "nope1.json")
    monkeypatch.setattr(uc, "_legacy_manual_paths_path", lambda: tmp_path / "nope2.json")

    target.parent.mkdir(parents=True, exist_ok=True)
    with patch("backend.api.services.user_config_manager.os.replace",
               side_effect=OSError("boom")):
        with pytest.raises(OSError):
            uc.save_config({"server": {"database_root": "/x"}})

    tmps = list(target.parent.glob(".config_*.json.tmp"))
    assert tmps == [], f"orphan tempfile(s) left behind: {tmps}"
