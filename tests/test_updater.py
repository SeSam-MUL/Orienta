"""Tests for backend/api/services/updater.py.

The update path is the one feature that can leave the app unusable, so the
failure branches get more attention here than the happy path.
"""

import subprocess
from pathlib import Path

import pytest

from backend.api.services import updater


@pytest.fixture(autouse=True)
def _clean_state():
    updater._check_cache.clear()
    updater._progress.clear()
    updater._progress["state"] = "idle"
    yield
    updater._check_cache.clear()


# --------------------------------------------------------------------------
# version comparison
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tag", ["v0.2.0", "0.2.0", "v10.4.7"])
def test_parse_version_accepts_release_tags(tag):
    assert updater.parse_version(tag) is not None


@pytest.mark.parametrize("tag", ["main", "nightly", "", "v1.2", "release-x"])
def test_parse_version_rejects_non_releases(tag):
    assert updater.parse_version(tag) is None


def test_version_ordering():
    p = updater.parse_version
    assert p("v0.2.0") > p("v0.1.0")
    assert p("v0.10.0") > p("v0.9.0")      # not string ordering
    assert p("v1.0.0") > p("v0.99.99")
    assert p("v1.0.1") > p("v1.0.0")


def test_prerelease_sorts_below_its_release():
    p = updater.parse_version
    assert p("v1.0.0") > p("v1.0.0-rc1")


# --------------------------------------------------------------------------
# install kind
# --------------------------------------------------------------------------

def test_install_kind_zip_without_git_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(updater, "PROJECT_ROOT", tmp_path)
    assert updater.install_kind() == "zip"


def test_install_kind_zip_without_remote(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(updater, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(updater, "_git_out", lambda *a, **k: None)
    assert updater.install_kind() == "zip"


def test_install_kind_git(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(updater, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(updater, "_git_out", lambda *a, **k: "https://example/x.git")
    assert updater.install_kind() == "git"


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------

def _fake_check(monkeypatch, *, kind="git", tags=(), current="v0.1.0"):
    monkeypatch.setattr(updater, "install_kind", lambda: kind)
    monkeypatch.setattr(updater, "_git", lambda *a, **k: None)
    monkeypatch.setattr(
        updater, "probe_remote",
        lambda: (list(tags), "" if tags else "remote_unreachable"),
    )
    monkeypatch.setattr(updater, "_changelog_for", lambda t: f"## {t}\nnotes")
    monkeypatch.setattr(
        updater, "get_version_info",
        lambda: {"version": current or "x", "release": current},
    )


def test_check_offers_a_newer_release(monkeypatch):
    _fake_check(monkeypatch, tags=["v0.3.0", "v0.2.0", "v0.1.0"], current="v0.1.0")
    r = updater.check_for_update(force=True)
    assert r["available"] is True
    assert r["latest"] == "v0.3.0"
    assert "notes" in r and r["notes"]


def test_check_is_quiet_when_up_to_date(monkeypatch):
    _fake_check(monkeypatch, tags=["v0.2.0"], current="v0.2.0")
    r = updater.check_for_update(force=True)
    assert r["available"] is False
    assert r["reason"] == "up_to_date"


def test_check_never_offers_a_downgrade(monkeypatch):
    _fake_check(monkeypatch, tags=["v0.1.0"], current="v0.2.0")
    r = updater.check_for_update(force=True)
    assert r["available"] is False


def test_remote_tags_keep_only_releases_newest_first():
    assert updater._parse_ls_remote(
        "aaa\trefs/tags/nightly\n"
        "bbb\trefs/tags/v0.4.0\n"
        "ccc\trefs/tags/v0.10.0\n"
        "ddd\trefs/tags/some-branch-tag\n"
    ) == ["v0.10.0", "v0.4.0"]


def test_remote_tags_empty_when_git_fails(monkeypatch):
    monkeypatch.setattr(
        updater, "_git",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "boom"),
    )
    assert updater._remote_release_tags() == []


def _completed(code, stdout="", stderr=""):
    return subprocess.CompletedProcess([], code, stdout, stderr)


def test_probe_reports_missing_credentials_separately(monkeypatch):
    """A private repo answers 'not found' without credentials — telling the
    user 'server unreachable' would send them chasing the wrong problem."""
    for err in (
        "fatal: could not read Username for 'https://github.com'",
        "remote: Repository not found.",
        "fatal: Authentication failed for 'https://github.com/x.git'",
    ):
        monkeypatch.setattr(updater, "_git", lambda *a, **k: _completed(128, "", err))
        assert updater.probe_remote() == ([], "auth_required"), err


def test_probe_reports_a_real_network_failure_as_unreachable(monkeypatch):
    monkeypatch.setattr(
        updater, "_git",
        lambda *a, **k: _completed(128, "", "fatal: unable to access ...: Could not resolve host"),
    )
    assert updater.probe_remote() == ([], "remote_unreachable")


def test_probe_reports_a_timeout_as_unreachable(monkeypatch):
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("git", 30)

    monkeypatch.setattr(updater, "_git", timeout)
    assert updater.probe_remote() == ([], "remote_unreachable")


def test_check_surfaces_auth_required(monkeypatch):
    monkeypatch.setattr(updater, "install_kind", lambda: "git")
    monkeypatch.setattr(updater, "_git", lambda *a, **k: _completed(128, "", "Repository not found"))
    monkeypatch.setattr(
        updater, "get_version_info", lambda: {"version": "v0.1.0", "release": "v0.1.0"}
    )
    r = updater.check_for_update(force=True)
    assert r["available"] is False
    assert r["reason"] == "auth_required"


def test_git_calls_never_prompt():
    """A background check must not pop a credential window at the user."""
    env = updater._git_env()
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GCM_INTERACTIVE"] == "never"


def test_check_reports_a_zip_install_instead_of_failing(monkeypatch):
    _fake_check(monkeypatch, kind="zip")
    r = updater.check_for_update(force=True)
    assert r["available"] is False
    assert r["install_kind"] == "zip"
    assert r["reason"] == "not_a_git_install"


def test_check_survives_an_unreachable_remote(monkeypatch):
    _fake_check(monkeypatch, tags=[], current="v0.1.0")
    r = updater.check_for_update(force=True)
    assert r["available"] is False
    assert r["reason"] == "remote_unreachable"


def test_check_result_is_cached(monkeypatch):
    calls = []
    _fake_check(monkeypatch, tags=["v0.9.0"], current="v0.1.0")
    monkeypatch.setattr(
        updater, "probe_remote", lambda: (calls.append(1), (["v0.9.0"], ""))[1]
    )
    updater.check_for_update(force=True)
    updater.check_for_update()          # cached
    assert len(calls) == 1
    updater.check_for_update(force=True)
    assert len(calls) == 2


# --------------------------------------------------------------------------
# update — the branches that must never break the app
# --------------------------------------------------------------------------

def test_update_refuses_a_dirty_working_tree(monkeypatch):
    monkeypatch.setattr(
        updater, "_git_out",
        lambda *a, **k: "HEADSHA" if a[0] == "rev-parse" else " M some/file.py",
    )
    result = updater.perform_update("v0.9.0")
    assert result["state"] == "failed"
    assert "local changes" in result["error"]


def test_dirty_check_ignores_untracked_files(monkeypatch, tmp_path):
    """A dataset dropped in the folder must not block updates forever.

    git leaves untracked files alone during a checkout; only edits to tracked
    files are at risk, so only those may refuse the update.
    """
    (tmp_path / "frontend").mkdir()
    monkeypatch.setattr(updater, "PROJECT_ROOT", tmp_path)
    seen = {}

    def git_out(*a, **k):
        if a[0] == "status":
            seen["args"] = a
            return ""          # clean once untracked files are excluded
        if a[0] == "rev-parse":
            return "SHA"
        if a[0] == "diff":
            return ""
        return None

    monkeypatch.setattr(updater, "_git_out", git_out)
    monkeypatch.setattr(updater, "_git", lambda *a, **k: None)
    monkeypatch.setattr(updater, "_run_step", lambda *a, **k: True)

    result = updater.perform_update("v0.9.0")
    assert result["state"] == "done"
    assert "--untracked-files=no" in seen["args"]


def test_update_refuses_an_unknown_tag(monkeypatch):
    def git_out(*a, **k):
        if a[0] == "rev-parse" and a[1] == "HEAD":
            return "HEADSHA"
        if a[0] == "status":
            return ""
        return None  # tag never resolves

    monkeypatch.setattr(updater, "_git_out", git_out)
    monkeypatch.setattr(updater, "_run_step", lambda *a, **k: True)
    result = updater.perform_update("v9.9.9")
    assert result["state"] == "failed"
    assert "not found" in result["error"]


def test_failed_build_rolls_back_head_and_dist(monkeypatch, tmp_path):
    """The decisive test: a broken build must not leave a broken app."""
    frontend = tmp_path / "frontend"
    dist = frontend / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("OLD BUILD", encoding="utf-8")
    monkeypatch.setattr(updater, "PROJECT_ROOT", tmp_path)

    def git_out(*a, **k):
        if a[0] == "rev-parse" and a[1] == "HEAD":
            return "OLDSHA"
        if a[0] == "status":
            return ""
        if a[0] == "rev-parse":
            return "NEWSHA"
        if a[0] == "diff":
            return ""          # nothing changed → skip pip/npm install
        return None

    checkouts = []
    monkeypatch.setattr(updater, "_git_out", git_out)
    monkeypatch.setattr(
        updater, "_git",
        lambda *a, **k: checkouts.append(a) or subprocess.CompletedProcess(a, 0, "", ""),
    )
    # every step succeeds except the build
    monkeypatch.setattr(
        updater, "_run_step",
        lambda name, cmd, cwd: name != "build",
    )

    result = updater.perform_update("v0.9.0")

    assert result["state"] == "failed"
    assert "Building the interface failed" in result["error"]
    # HEAD was put back
    assert any("checkout" in a and "OLDSHA" in a for a in checkouts)
    # and the old built interface is in place, not a half-written one
    assert (dist / "index.html").read_text(encoding="utf-8") == "OLD BUILD"
    assert not (frontend / "dist.update-backup").exists()


def test_dependencies_are_skipped_when_unchanged(monkeypatch, tmp_path):
    (tmp_path / "frontend").mkdir()
    monkeypatch.setattr(updater, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(updater, "_git", lambda *a, **k: None)
    monkeypatch.setattr(
        updater, "_git_out",
        lambda *a, **k: {"rev-parse": "SHA", "status": "", "diff": ""}.get(a[0]),
    )
    ran = []
    monkeypatch.setattr(
        updater, "_run_step", lambda name, cmd, cwd: ran.append(name) or True
    )
    result = updater.perform_update("v0.9.0")
    assert result["state"] == "done"
    assert "dependencies" not in ran
    assert "node_modules" not in ran
    assert "build" in ran        # the interface is always rebuilt


def test_changed_dependencies_are_installed(monkeypatch, tmp_path):
    (tmp_path / "frontend").mkdir()
    monkeypatch.setattr(updater, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(updater, "_git", lambda *a, **k: None)
    monkeypatch.setattr(
        updater, "_git_out",
        lambda *a, **k: {"rev-parse": "SHA", "status": "",
                         "diff": "requirements.txt"}.get(a[0]),
    )
    ran = []
    monkeypatch.setattr(
        updater, "_run_step", lambda name, cmd, cwd: ran.append(name) or True
    )
    updater.perform_update("v0.9.0")
    assert "dependencies" in ran and "node_modules" in ran


def test_unknown_diff_result_installs_rather_than_skips(monkeypatch):
    """When git cannot answer, reinstalling is cheap; skipping breaks the app."""
    monkeypatch.setattr(updater, "_git_out", lambda *a, **k: None)
    assert updater._changed_between("a", "b", "requirements.txt") is True


def test_start_update_refuses_a_second_run(monkeypatch):
    monkeypatch.setattr(updater, "perform_update", lambda tag: None)
    assert updater.start_update("v0.9.0") is True
    updater._progress["state"] = "running"
    assert updater.start_update("v0.9.0") is False
