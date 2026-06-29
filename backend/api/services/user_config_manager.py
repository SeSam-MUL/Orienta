"""Per-machine user configuration manager.

Stores user settings (API keys, server config, manual paths) in a JSON
file under the OS-conventional user-config directory:

  Windows: %APPDATA%\\Kikuchipy\\config.json
  Linux:   $XDG_CONFIG_HOME/Kikuchipy/config.json  (or ~/.config/...)
  macOS:   ~/Library/Application Support/Kikuchipy/config.json

This is **per-machine** by design: when the project tree is transferred
to another PC, the user's API keys and server config STAY BEHIND on the
original machine. The new machine starts with a clean config.

Migration: on first load, if the legacy project-tree files
`<project_root>/data/server_config.json` and `<project_root>/data/manual_paths.json`
exist, their contents are imported into the per-machine config and the
legacy files are deleted (so they don't get committed or transferred).
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _user_config_dir() -> Path:
    """Return the OS-conventional per-user config dir for Kikuchipy.

    Never returns a path inside the project tree.
    """
    import sys
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "Kikuchipy"
        return Path.home() / "AppData" / "Roaming" / "Kikuchipy"
    # macOS-specific (sys.platform is "darwin" on macOS, no os.uname dependency)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Kikuchipy"
    # Linux / other Unix
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "Kikuchipy"
    return Path.home() / ".config" / "Kikuchipy"


def _user_config_path() -> Path:
    return _user_config_dir() / "config.json"


def _project_root() -> Path:
    # backend/api/services/user_config_manager.py → up 3
    return Path(__file__).resolve().parents[3]


def _legacy_server_path() -> Path:
    return _project_root() / "data" / "server_config.json"


def _legacy_manual_paths_path() -> Path:
    return _project_root() / "data" / "manual_paths.json"


# --- Default schema -------------------------------------------------------

_DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "server": {
        "enabled": True,
        "database_root": "",
        "offline_mode": False,
    },
    "manual_paths": {
        "emsoft_bin_dir": "",
        "emsphinx_dir": "",
    },
    "api_keys": {
        # Materials Project API key (free signup at https://materialsproject.org)
        "materials_project": "",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Returns a new dict."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# --- Module-level config (lazy-loaded) ------------------------------------

_CONFIG_CACHE: Optional[dict] = None
_CONFIG_PATH_OVERRIDE: Optional[Path] = None       # tests inject this
_MIGRATION_DONE = False


def set_config_path_for_test(path: Optional[Path]) -> None:
    """Test-only: override the config file path (also disables real migration)."""
    global _CONFIG_PATH_OVERRIDE, _CONFIG_CACHE, _MIGRATION_DONE
    _CONFIG_PATH_OVERRIDE = path
    _CONFIG_CACHE = None
    _MIGRATION_DONE = True   # tests opt-in to migration explicitly


def _resolve_path() -> Path:
    return _CONFIG_PATH_OVERRIDE or _user_config_path()


def _read_raw(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
        return json.loads(text) if text.strip() else {}
    except Exception as exc:
        logger.warning("user_config: could not read %s: %s — falling back to defaults", path, exc)
        return {}


def _write_atomic(path: Path, data: dict) -> None:
    """Atomic write: write to a tempfile in the same dir, then rename. On
    Windows os.replace handles the cross-process atomicity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".config_", suffix=".json.tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp_name, path)
    except Exception:
        # Clean up the tempfile if we never managed to rename it
        try: os.unlink(tmp_name)
        except OSError: pass
        raise


def _safe_read_legacy(path: Path) -> tuple[dict, bool]:
    """Read a legacy file. Returns (parsed_dict, was_readable).

    `was_readable` is False if the file exists but can't be parsed — in that
    case we MUST NOT delete the original (data loss). Only delete cleanly-
    parsed files; rename corrupt ones to .corrupt.bak instead.
    """
    if not path.exists():
        return {}, True
    try:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return {}, True   # empty file = OK to delete
        return json.loads(text), True
    except Exception as exc:
        logger.warning("legacy file %s unreadable: %s — keeping as .corrupt.bak",
                       path, exc)
        return {}, False


def _migrate_legacy_if_needed(target_path: Path) -> dict:
    """One-shot migration: if the legacy project-tree files exist AND the new
    per-machine config file doesn't, copy their content into the new config
    and DELETE the legacy files.

    NEVER deletes a legacy file we couldn't parse — that would lose user
    data. Such files are renamed to `<file>.corrupt.bak` so the user can
    recover manually. Note: the existing user-home config (if present) is
    *only* overwritten when the legacy file has parseable content;
    documented in the design spec.

    Returns the merged-from-legacy config (or empty if nothing to migrate).
    """
    legacy_server = _legacy_server_path()
    legacy_paths = _legacy_manual_paths_path()

    has_server = legacy_server.exists()
    has_paths = legacy_paths.exists()
    if not (has_server or has_paths):
        return {}

    def _retire(p: Path, was_readable: bool) -> None:
        """Delete cleanly-read files; rename corrupt ones to .corrupt.bak."""
        if not p.exists():
            return
        if was_readable:
            try:
                p.unlink()
            except Exception as exc:
                logger.warning("could not delete legacy %s: %s", p, exc)
        else:
            backup = p.with_suffix(p.suffix + ".corrupt.bak")
            try:
                # Use os.replace so .bak gets overwritten if it already exists
                os.replace(p, backup)
                logger.warning(
                    "user_config: legacy %s was corrupt — preserved as %s "
                    "for manual recovery.", p, backup,
                )
            except Exception as exc:
                logger.warning("could not rename corrupt legacy %s: %s", p, exc)

    # If the new config already has real content, don't clobber it.
    # (Design choice: GUI-saved values on the new machine win over project-tree
    # leftovers. Documented in the spec.)
    if target_path.exists():
        existing = _read_raw(target_path)
        if existing.get("server") or existing.get("manual_paths"):
            logger.info(
                "user_config: user-home config already populated; "
                "retiring legacy project-tree files without merging."
            )
            for p in (legacy_server, legacy_paths):
                # Treat as 'readable' for retirement purposes since we're not
                # merging — there's no data loss possible.
                _retire(p, was_readable=True)
            return {}

    server_data, server_ok = _safe_read_legacy(legacy_server) if has_server else ({}, True)
    paths_data, paths_ok = _safe_read_legacy(legacy_paths) if has_paths else ({}, True)

    migrated: dict = {}
    if has_server and server_ok and server_data:
        migrated["server"] = server_data
    if has_paths and paths_ok and paths_data:
        migrated["manual_paths"] = paths_data

    if migrated:
        logger.info(
            "user_config: migrating legacy project-tree config to per-machine "
            "location %s (then retiring the originals).",
            target_path,
        )

    # Retire each file based on whether it was actually parseable
    _retire(legacy_server, server_ok)
    _retire(legacy_paths, paths_ok)

    return migrated


def load_config(*, run_migration: bool = True) -> dict:
    """Read the per-machine config, performing one-shot legacy migration on
    first call. Subsequent calls return a cached copy.

    The cache holds the in-memory representation merged with defaults so
    callers never see missing keys.
    """
    global _CONFIG_CACHE, _MIGRATION_DONE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    target = _resolve_path()
    on_disk = _read_raw(target)

    if run_migration and not _MIGRATION_DONE:
        migrated = _migrate_legacy_if_needed(target)
        if migrated:
            # Persist migrated config so subsequent boots don't re-migrate
            merged = _deep_merge(on_disk, migrated)
            try:
                _write_atomic(target, merged)
                on_disk = merged
            except Exception as exc:
                logger.warning(
                    "user_config: could not persist migrated config to %s: %s",
                    target, exc,
                )
                # Fall back: use the merged result in-memory even if write failed
                on_disk = merged
        _MIGRATION_DONE = True

    merged_with_defaults = _deep_merge(_DEFAULT_CONFIG, on_disk)
    _CONFIG_CACHE = merged_with_defaults
    return _CONFIG_CACHE


def save_config(data: dict) -> dict:
    """Replace the entire config with `data` (deep-merged onto defaults so
    missing top-level keys keep their defaults). Persists to disk + returns
    the resulting cached config."""
    global _CONFIG_CACHE
    merged = _deep_merge(_DEFAULT_CONFIG, data)
    target = _resolve_path()
    _write_atomic(target, merged)
    _CONFIG_CACHE = merged
    return _CONFIG_CACHE


def update_section(section: str, values: dict) -> dict:
    """Atomically update one top-level section (e.g. 'server', 'api_keys').

    Returns the resulting config (full structure).
    """
    cfg = load_config()
    new_cfg = dict(cfg)
    new_section = dict(new_cfg.get(section, {}))
    new_section.update(values)
    new_cfg[section] = new_section
    return save_config(new_cfg)


def get_section(section: str) -> dict:
    """Return a copy of one top-level section."""
    return dict(load_config().get(section, {}))


def get_api_key(name: str) -> str:
    """Get a single API key by name. Empty string if not configured."""
    return get_section("api_keys").get(name, "") or ""


def set_api_key(name: str, value: str) -> dict:
    """Persist a single API key. Empty string clears it."""
    return update_section("api_keys", {name: value})


def has_api_key(name: str) -> bool:
    """True iff the named key is configured + non-empty."""
    return bool(get_api_key(name).strip())


def redact_key(key: str, visible_tail: int = 4) -> str:
    """Mask all but the last `visible_tail` characters of an API key for
    safe display in API responses + logs.

    Short keys (≤ 2 * visible_tail chars) are fully masked — showing the
    last 4 chars of an 8-char key would leak half of it.
    """
    if not key:
        return ""
    if len(key) <= 2 * visible_tail:
        return "*" * len(key)
    return "*" * (len(key) - visible_tail) + key[-visible_tail:]


def config_path() -> Path:
    """Return the resolved config file path (for UI hint + diagnostics)."""
    return _resolve_path()


def reset_cache_for_test() -> None:
    """Test-only: drop the in-memory cache so the next load() reads fresh."""
    global _CONFIG_CACHE, _MIGRATION_DONE
    _CONFIG_CACHE = None
    _MIGRATION_DONE = False
