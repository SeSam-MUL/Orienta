"""Tests for backend/api/services/app_version.py."""

import re
import subprocess
from pathlib import Path

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


# --------------------------------------------------------------------------
# `.git` is not always a directory
#
# In a git worktree and in a submodule it is a FILE holding `gitdir: <path>`:
# HEAD lives there, the refs live where its `commondir` points. Reading
# `<root>/.git/HEAD` raises NotADirectoryError, which this module caught as
# OSError and reported as "no git information found" -- so a copy started from a
# worktree without a git binary showed its version as unknown, and two tests
# above failed in every worktree anyone ever used.
#
# The layouts are built by hand rather than with `git worktree add`: these tests
# have to fail on the shape of the files, not on whether git is installed.
# --------------------------------------------------------------------------

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _clone(root, branch="main", commit=COMMIT):
    """An ordinary checkout: .git is a directory."""
    git = root / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text(f"ref: refs/heads/{branch}\n", encoding="utf-8")
    (git / "refs" / "heads" / branch).write_text(commit + "\n", encoding="utf-8")
    return root


def _worktree(root, gitdir, branch="feature/x", commit=COMMIT, common=None):
    """A worktree: .git is a file, HEAD is private, refs are shared."""
    common = common or (gitdir.parent.parent)
    gitdir.mkdir(parents=True)
    root.mkdir(parents=True, exist_ok=True)
    (root / ".git").write_text(f"gitdir: {gitdir.as_posix()}\n", encoding="utf-8")
    (gitdir / "HEAD").write_text(f"ref: refs/heads/{branch}\n", encoding="utf-8")
    import os
    rel = os.path.relpath(common, gitdir)
    (gitdir / "commondir").write_text(rel.replace("\\", "/") + "\n", encoding="utf-8")
    (common / "refs" / "heads" / Path(branch).parent.as_posix()).mkdir(
        parents=True, exist_ok=True)
    (common / "refs" / "heads" / branch).write_text(commit + "\n", encoding="utf-8")
    return root


def test_an_ordinary_checkout_still_reads(tmp_path):
    root = _clone(tmp_path / "repo")
    assert app_version._read_git_files(root) == (COMMIT[:9], "main")


def test_a_worktree_is_read_through_its_gitdir_pointer(tmp_path):
    """The case that reported "unknown" for months."""
    common = tmp_path / "main" / ".git"
    (common / "refs" / "heads").mkdir(parents=True)
    root = _worktree(tmp_path / "wt", common / "worktrees" / "wt")
    assert app_version._read_git_files(root) == (COMMIT[:9], "feature/x")


def test_a_branch_with_slashes_survives_the_worktree_path(tmp_path):
    """`golive/unified-phase-identity` is the normal case in this project."""
    common = tmp_path / "main" / ".git"
    (common / "refs" / "heads").mkdir(parents=True)
    root = _worktree(tmp_path / "wt", common / "worktrees" / "wt",
                     branch="golive/unified-phase-identity")
    commit, branch = app_version._read_git_files(root)
    assert (commit, branch) == (COMMIT[:9], "golive/unified-phase-identity")


def test_a_packed_ref_is_found_in_the_COMMON_directory(tmp_path):
    """After `git pack-refs` the loose file is gone -- and it was never in the
    worktree's own directory to begin with."""
    common = tmp_path / "main" / ".git"
    (common / "refs" / "heads").mkdir(parents=True)
    root = _worktree(tmp_path / "wt", common / "worktrees" / "wt", branch="main")
    (common / "refs" / "heads" / "main").unlink()
    (common / "packed-refs").write_text(
        f"# pack-refs with: peeled fully-peeled sorted \n{COMMIT} refs/heads/main\n",
        encoding="utf-8")
    assert app_version._read_git_files(root) == (COMMIT[:9], "main")


def test_a_submodule_style_relative_gitdir_is_resolved(tmp_path):
    """git writes a RELATIVE gitdir for a submodule, and no commondir."""
    root = tmp_path / "sub"
    real = tmp_path / "parent" / ".git" / "modules" / "sub"
    (real / "refs" / "heads").mkdir(parents=True)
    root.mkdir()
    (root / ".git").write_text("gitdir: ../parent/.git/modules/sub\n", encoding="utf-8")
    (real / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (real / "refs" / "heads" / "main").write_text(COMMIT + "\n", encoding="utf-8")
    assert app_version._read_git_files(root) == (COMMIT[:9], "main")


def test_a_detached_head_in_a_worktree_reports_the_hash(tmp_path):
    common = tmp_path / "main" / ".git"
    (common / "refs" / "heads").mkdir(parents=True)
    root = _worktree(tmp_path / "wt", common / "worktrees" / "wt")
    gitdir = common / "worktrees" / "wt"
    (gitdir / "HEAD").write_text(COMMIT + "\n", encoding="utf-8")
    assert app_version._read_git_files(root) == (COMMIT[:9], None)


@pytest.mark.parametrize("contents", ["", "not a pointer\n", "gitdir:\n",
                                      "gitdir: /nowhere/at/all\n"])
def test_an_unreadable_pointer_answers_unknown_without_raising(tmp_path, contents):
    """Whatever the file says, this must not take the launch down: the caller
    reports "unknown", which is a sentence a user can live with."""
    root = tmp_path / "odd"
    root.mkdir()
    (root / ".git").write_text(contents, encoding="utf-8")
    assert app_version._read_git_files(root) == (None, None)


def test_the_fallback_agrees_with_git_in_this_checkout():
    """The measurement, wherever this runs -- a clone or a worktree.

    Skips when git cannot answer rather than asserting something weaker: the
    point is agreement between two readers of the same repository.
    """
    from pathlib import Path as _P
    root = _P(app_version.__file__).resolve().parents[3]
    try:
        real = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        ref = subprocess.run(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("no git binary")
    if real.returncode or ref.returncode:
        pytest.skip("not a git checkout")
    commit, branch = app_version._read_git_files(root)
    assert commit == real.stdout.strip()[:9]
    assert branch == ref.stdout.strip()
