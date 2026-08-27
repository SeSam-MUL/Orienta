"""Tests for backend/api/services/app_version.py."""

import re
import subprocess

import pytest

from backend.api.services import app_version


@pytest.fixture(autouse=True)
def _clear_cache():
    app_version.get_version_info.cache_clear()
    yield
    app_version.get_version_info.cache_clear()


def test_real_repo_resolves_via_git():
    """This working copy IS a git repo — the happy path must yield a commit."""
    info = app_version.get_version_info()
    assert info["app"] == "Orienta"
    assert info["source"] in ("git", "git-files")
    assert info["commit"], "expected a commit hash in a git checkout"
    assert info["version"] != "unknown"
    if info["source"] == "git":
        # git path also carries the commit date (YYYY-MM-DD)
        assert info["commit_date"] and len(info["commit_date"]) == 10


def test_git_binary_missing_falls_back_to_git_files(monkeypatch):
    def boom(*a, **k):
        raise OSError("git not installed")

    monkeypatch.setattr(subprocess, "run", boom)
    info = app_version.get_version_info()
    # .git exists in this repo, so the manual parse must still find the hash
    assert info["source"] == "git-files"
    assert info["commit"]
    assert info["commit_date"] is None
    assert info["version"] == f"({info['commit']})"


def test_no_git_at_all_reports_unknown(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise OSError("git not installed")

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(app_version, "PROJECT_ROOT", tmp_path)
    info = app_version.get_version_info()
    assert info == {
        "app": "Orienta",
        "version": "unknown",
        "release": None,
        "commits_since_release": None,
        "commit": None,
        "commit_date": None,
        "branch": None,
        "source": "unknown",
    }


@pytest.mark.parametrize(
    "described,expected",
    [
        ("v0.2.0", ("v0.2.0", 0)),
        ("v0.1.0-164-g5faee319", ("v0.1.0", 164)),
        ("v1.0.0-rc1-3-gabcdef12", ("v1.0.0-rc1", 3)),  # hyphen inside the tag
        ("", (None, None)),
    ],
)
def test_parse_describe(described, expected):
    assert app_version._parse_describe(described) == expected


def test_version_prefers_the_release_lineage(monkeypatch):
    """A tagged checkout must say which release it is, not just a date."""
    real = app_version._run_git

    def fake(args, cwd):
        if args[0] == "describe":
            return "v0.1.0-164-g5faee319"
        return real(args, cwd)

    monkeypatch.setattr(app_version, "_run_git", fake)
    info = app_version.get_version_info()
    assert info["release"] == "v0.1.0"
    assert info["commits_since_release"] == 164
    assert info["version"].startswith("v0.1.0+164 (")


def test_version_on_an_exact_tag_is_just_the_tag(monkeypatch):
    real = app_version._run_git
    monkeypatch.setattr(
        app_version,
        "_run_git",
        lambda args, cwd: "v0.2.0" if args[0] == "describe" else real(args, cwd),
    )
    info = app_version.get_version_info()
    assert info["version"] == "v0.2.0"
    assert info["commits_since_release"] == 0


def test_repo_without_tags_falls_back_to_date_and_hash(monkeypatch):
    real = app_version._run_git
    monkeypatch.setattr(
        app_version,
        "_run_git",
        lambda args, cwd: None if args[0] == "describe" else real(args, cwd),
    )
    info = app_version.get_version_info()
    assert info["release"] is None
    assert re.match(r"^\d{4}-\d{2}-\d{2} \([0-9a-f]+\)$", info["version"])


def test_version_line_contains_app_and_version():
    line = app_version.version_line()
    assert line.startswith("Orienta ")
    info = app_version.get_version_info()
    assert info["version"] in line


def test_git_files_parse_matches_git(monkeypatch):
    """The fallback parser must agree with git about the current commit."""
    via_git = app_version.get_version_info()
    if via_git["source"] != "git":
        pytest.skip("git binary unavailable")
    app_version.get_version_info.cache_clear()

    def boom(*a, **k):
        raise OSError("git not installed")

    monkeypatch.setattr(subprocess, "run", boom)
    via_files = app_version.get_version_info()
    assert via_files["commit"][:7] == via_git["commit"][:7]
    assert via_files["branch"] == via_git["branch"]
