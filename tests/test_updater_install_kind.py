"""Three kinds of install, three different truths about updating.

  git    -- a checkout whose top level IS this project: fetch + checkout
  bundle -- no git here, but a VERSION file: download a runtime package
  zip    -- neither: cannot update itself and must be told so

Two things this file exists to pin, both of which were wrong at first:

* **`.git` is not always a directory.** A `git worktree`, a submodule and
  `clone --separate-git-dir` all write a `.git` FILE. Testing `.is_dir()` told a
  perfectly updatable worktree that it cannot update itself — and this project
  uses worktrees routinely.
* **`git -C <dir>` searches UPWARD.** A bundle unpacked anywhere inside someone
  else's repository would be reported as a git install of *that* project, and
  the update would compare against their tags.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def at(monkeypatch, tmp_path):
    import backend.api.services.app_version as av
    import backend.api.services.updater as up

    monkeypatch.setattr(up, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(av, "PROJECT_ROOT", tmp_path)
    av.get_version_info.cache_clear()
    yield up, tmp_path
    av.get_version_info.cache_clear()


def _git_says(monkeypatch, up, *, toplevel=None, remote=None):
    """Stub `_git_out` per subcommand, not wholesale.

    A stub that answers every git question with the same string made the
    anchoring check meaningless — `--show-toplevel` would return a remote URL
    and still be compared against the project root.
    """
    def fake(*args, **kwargs):
        if args[:2] == ("rev-parse", "--show-toplevel"):
            return toplevel
        if args[:2] == ("remote", "get-url"):
            return remote
        return None

    monkeypatch.setattr(up, "_git_out", fake)


# --------------------------------------------------------------------------
# git
# --------------------------------------------------------------------------

def test_a_checkout_whose_toplevel_is_this_project_is_git(at, monkeypatch):
    up, root = at
    _git_says(monkeypatch, up, toplevel=str(root), remote="https://example.invalid/x.git")
    assert up.install_kind() == "git"


def test_a_worktree_is_git_even_though_dot_git_is_a_file(at, monkeypatch):
    """The motivating case: `.git` is a FILE here, and `.is_dir()` said no."""
    up, root = at
    (root / ".git").write_text("gitdir: /elsewhere/.git/worktrees/x\n",
                               encoding="utf-8", newline="")
    _git_says(monkeypatch, up, toplevel=str(root), remote="https://example.invalid/x.git")
    assert up.install_kind() == "git"


def test_a_checkout_without_a_remote_is_a_zip(at, monkeypatch):
    up, root = at
    _git_says(monkeypatch, up, toplevel=str(root), remote=None)
    assert up.install_kind() == "zip"


def test_a_bundle_inside_someone_elses_repository_is_still_a_bundle(at, monkeypatch):
    """`git -C` searches upward. Without anchoring, an install unpacked into
    any git working tree would report that project's identity as its own."""
    up, root = at
    (root / "VERSION").write_text("v0.4.0\n", encoding="utf-8", newline="")
    _git_says(monkeypatch, up,
              toplevel=str(root.parent),                 # a DIFFERENT repository
              remote="https://example.invalid/not-ours.git")
    assert up.install_kind() == "bundle"


# --------------------------------------------------------------------------
# bundle / zip
# --------------------------------------------------------------------------

def test_no_git_but_a_version_file_is_a_bundle(at, monkeypatch):
    up, root = at
    _git_says(monkeypatch, up)
    (root / "VERSION").write_text("v0.4.0\n", encoding="utf-8", newline="")
    assert up.install_kind() == "bundle"


def test_neither_is_a_zip(at, monkeypatch):
    up, _ = at
    _git_says(monkeypatch, up)
    assert up.install_kind() == "zip"


def test_a_version_file_that_cannot_be_read_is_still_a_bundle(at, monkeypatch):
    """The file is THERE — we simply may not open it.

    An earlier version guarded this with `Path.stat()`, which does not read the
    file at all: an ACL or an exclusive lock passes the stat and fails on the
    open, so the guard was a no-op for its own scenario and the answer was still
    "this installation cannot update itself".
    """
    up, root = at
    _git_says(monkeypatch, up)
    version = root / "VERSION"
    version.write_text("v0.4.0\n", encoding="utf-8", newline="")

    real_read = Path.read_text

    def refuse(self, *a, **k):
        if self == version:
            raise PermissionError("locked by another process")
        return real_read(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", refuse)
    assert up.install_kind() == "bundle"


def test_a_half_written_version_file_is_still_a_bundle(at, monkeypatch):
    """A crashed update can leave junk in VERSION. That is a REPAIRABLE
    installed copy, not a zip with no way back."""
    up, root = at
    _git_says(monkeypatch, up)
    (root / "VERSION").write_text("v0.4", encoding="utf-8", newline="")
    assert up.install_kind() == "bundle"


def test_a_version_file_with_a_byte_order_mark_is_read(at, monkeypatch):
    """PowerShell writes a BOM by default and `str.strip()` does not remove one,
    so a VERSION regenerated on Windows parsed as junk."""
    up, root = at
    _git_says(monkeypatch, up)
    (root / "VERSION").write_bytes("﻿v0.4.0\n".encode("utf-8"))
    assert up.install_kind() == "bundle"

    import backend.api.services.app_version as av
    assert av.read_version_state(root) == ("tag", "v0.4.0")


# --------------------------------------------------------------------------
# one definition of "release tag"
# --------------------------------------------------------------------------

def test_parse_version_and_the_tag_pattern_are_the_same_rule(at):
    """They are literally the same object now. This test is what stops someone
    reintroducing a second spelling with capture groups — which is how they
    drifted before, one accepting "v1.2.3-" that the other refused."""
    from backend.api.services.app_version import _TAG_LINE_RE

    up, _ = at
    assert up._TAG_RE is _TAG_LINE_RE


@pytest.mark.parametrize("tag", [
    "v0.4.0", "0.4.0", "v1.2.3-rc1", "v1.0.0+build.5", "v10.20.30",
    # Hyphenated pre-releases are REAL and a stricter suffix class silently
    # dropped them from the available-release list, with no message: an
    # unparseable tag is discarded by `_parse_ls_remote`.
    "v1.2.3-beta-1", "v1.2.3-rc-2", "v1.0.0+build-5",
])
def test_real_release_tags_parse(at, tag):
    up, _ = at
    assert up.parse_version(tag) is not None, tag


@pytest.mark.parametrize("junk", ["", "v0.4", "not-a-tag", "<!DOCTYPE html>", "v1.2.3-"])
def test_non_releases_do_not_parse(at, junk):
    up, _ = at
    assert up.parse_version(junk) is None, junk


def test_an_unparseable_local_tag_does_not_crash_the_check(at, monkeypatch):
    """`git describe` returns the nearest reachable tag of ANY shape, and this
    project creates tags like "wip-uncommitted-2026-09-14". Comparing a tuple
    with None raises TypeError, which routes/system.py swallows into
    reason="check_failed": a dead dialog with no explanation.
    """
    up, root = at
    _git_says(monkeypatch, up, toplevel=str(root), remote="https://example.invalid/x.git")
    monkeypatch.setattr(up, "_git", lambda *a, **k: None)
    monkeypatch.setattr(up, "probe_remote", lambda: (["v9.9.9"], ""))
    monkeypatch.setattr(up, "_changelog_for", lambda t: "notes")
    monkeypatch.setattr(
        up, "get_version_info",
        lambda: {"release": "wip-uncommitted-2026-09-14", "version": "x",
                 "commit": "abc", "branch": "main"},
    )
    up._check_cache.clear()

    result = up.check_for_update(force=True)

    assert result["available"] is True
    assert result["reason"] == "no_local_release"
