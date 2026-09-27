"""An install without .git still has an identity: the VERSION file.

Without it a bundle install reports version "unknown", and the update check has
nothing to compare the newest release against — so it can never say "up to date"
and can never say "there is a newer one" either.

git stays authoritative where it exists: a developer checkout must not start
reading a stale VERSION file left over from testing.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def fresh_module(monkeypatch, tmp_path):
    import backend.api.services.app_version as av

    monkeypatch.setattr(av, "PROJECT_ROOT", tmp_path)
    av.get_version_info.cache_clear()
    yield av
    av.get_version_info.cache_clear()


def _no_git(module, monkeypatch):
    monkeypatch.setattr(module, "_run_git", lambda *a, **k: None)
    monkeypatch.setattr(module, "_read_git_files", lambda *a, **k: (None, None))


def test_version_file_supplies_the_release_when_there_is_no_git(fresh_module, tmp_path, monkeypatch):
    _no_git(fresh_module, monkeypatch)
    (tmp_path / "VERSION").write_text("v0.4.0\n", encoding="utf-8", newline="")

    info = fresh_module.get_version_info()

    assert info["release"] == "v0.4.0"
    assert info["version"] == "v0.4.0", "the About page must not show 'v0.4.0+None (None)'"
    assert info["source"] == "version-file"
    assert info["commits_since_release"] == 0


def test_a_blank_or_junk_version_file_is_ignored(fresh_module, tmp_path, monkeypatch):
    """A control: passes before the change too."""
    _no_git(fresh_module, monkeypatch)
    (tmp_path / "VERSION").write_text("   \n", encoding="utf-8", newline="")

    info = fresh_module.get_version_info()

    assert info["release"] is None
    assert info["source"] == "unknown"


def test_git_still_wins_over_a_leftover_version_file(fresh_module, tmp_path, monkeypatch):
    """A control: passes before the change too."""
    (tmp_path / "VERSION").write_text("v0.0.1\n", encoding="utf-8", newline="")
    monkeypatch.setattr(
        fresh_module,
        "_run_git",
        lambda args, *a, **k: "abc1234|2026-09-17" if args[0] == "log" else "main",
    )

    info = fresh_module.get_version_info()

    assert info["source"] == "git"
    assert info["release"] != "v0.0.1"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("v0.4.0", "v0.4.0"),
        ("0.4.0", "0.4.0"),
        ("v0.4.0-rc1", "v0.4.0-rc1"),
        ("v1.0.0+build.5", "v1.0.0+build.5"),
        ("  v0.4.0  \n", "v0.4.0"),
        ("", None),
        ("not-a-tag", None),
        ("v0.4", None),
        ("<!DOCTYPE html>", None),
    ],
)
def test_read_version_file_accepts_release_tags_and_nothing_else(
    fresh_module, tmp_path, text, expected
):
    """Named in the task's Interfaces block, so it gets a test of its own.

    Anything that is not a release tag is ignored rather than shown: a
    half-written file must not become the version the update check compares
    against.
    """
    (tmp_path / "VERSION").write_text(text, encoding="utf-8", newline="")
    assert fresh_module._read_version_file(tmp_path) == expected


def test_an_absent_version_file_is_not_an_error(fresh_module, tmp_path):
    assert fresh_module._read_version_file(tmp_path) is None
