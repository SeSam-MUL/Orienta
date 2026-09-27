"""`repo_url` must not be null just because there is no git remote.

`GET /api/system/version` reports `repo_url`, and the "Open a GitHub issue"
button in the problem-report dialog appears only when it is set. It was derived
from `git remote get-url origin` alone, which answers nothing in **either** kind
of installation:

  * the development tree has no remote -- measured on this checkout,
    `git remote get-url origin` -> "error: No such remote 'origin'";
  * an installed copy has no `.git` at all, so there is no remote to ask.

So the button has never been shown to anybody. These tests pin the fallback and,
just as importantly, that a real remote still wins -- a fork must keep sending
people to its own tracker, which was the reason the value was read from the
checkout in the first place.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from backend.api.routes import system
from backend.api.services import github_releases

CANONICAL = "https://github.com/SeSam-MUL/Orienta"


@pytest.fixture
def no_remote(monkeypatch):
    """Neither a remote nor git: `_git_out` returns None for both."""
    from backend.api.services import updater
    monkeypatch.setattr(updater, "_git_out", lambda *a, **k: None)


def _with_remote(monkeypatch, value):
    from backend.api.services import updater
    monkeypatch.setattr(updater, "_git_out", lambda *a, **k: value)


# --- the fallback ----------------------------------------------------------

def test_without_a_remote_it_still_names_the_project(no_remote):
    assert system._repo_web_url() == CANONICAL


def test_it_never_answers_none(no_remote):
    """The contract the frontend gates the button on."""
    assert system._repo_web_url() is not None


def test_the_fallback_is_not_a_second_hardcoded_string():
    """It comes from the module that already owns the repository's identity.

    If someone re-hardcodes the URL in system.py this fails, which is the point:
    the update check and the issue link must not be able to disagree about which
    repository this is.
    """
    import inspect
    src = inspect.getsource(system._repo_web_url)
    assert "github_releases.repo_page_url()" in src
    assert "github.com/SeSam-MUL" not in src, (
        "the canonical URL is hardcoded in system.py again")


# --- a real remote still wins ---------------------------------------------

@pytest.mark.parametrize("remote,expected", [
    ("https://github.com/someone/fork", "https://github.com/someone/fork"),
    ("https://github.com/someone/fork.git", "https://github.com/someone/fork"),
    ("git@github.com:someone/fork.git", "https://github.com/someone/fork"),
    ("  https://gitlab.com/o/r  ", "https://gitlab.com/o/r"),
])
def test_a_remote_wins_over_the_fallback(monkeypatch, remote, expected):
    _with_remote(monkeypatch, remote)
    assert system._repo_web_url() == expected


@pytest.mark.parametrize("remote", [
    "/home/someone/mirror.git",        # a local clone source
    "C:\\Users\\someone\\mirror",
    "",
])
def test_a_remote_that_is_not_a_web_url_falls_back(monkeypatch, remote):
    """A filesystem remote is not somewhere a browser can go."""
    _with_remote(monkeypatch, remote)
    assert system._repo_web_url() == CANONICAL


# --- the env override the update check already honours ---------------------

def test_the_update_repo_env_var_moves_the_fallback_too(no_remote, monkeypatch):
    monkeypatch.setenv(github_releases.REPO_SLUG_ENV, "fork-owner/fork-name")
    assert system._repo_web_url() == "https://github.com/fork-owner/fork-name"


# --- the two canonical sources must not drift -----------------------------

def test_it_agrees_with_citation_cff(monkeypatch):
    """`CITATION.cff` is the other place the project states its own address.

    Read with a regex rather than a YAML parser: this is a one-line scalar and
    the test must not depend on pyyaml being installed.
    """
    monkeypatch.delenv(github_releases.REPO_SLUG_ENV, raising=False)
    cff = Path(__file__).resolve().parents[1] / "CITATION.cff"
    if not cff.is_file():
        pytest.skip("CITATION.cff is not in this checkout")
    m = re.search(r'^repository-code:\s*"?([^"\s]+)"?\s*$',
                  cff.read_text(encoding="utf-8"), re.MULTILINE)
    assert m, "CITATION.cff has no repository-code line"
    assert m.group(1).rstrip("/") == github_releases.repo_page_url(), (
        "CITATION.cff and github_releases disagree about which repository this "
        "is; a citation and an issue link pointing at different projects is "
        "worse than either being wrong alone")


def test_the_releases_url_shares_the_same_slug(monkeypatch):
    monkeypatch.delenv(github_releases.REPO_SLUG_ENV, raising=False)
    monkeypatch.delenv(github_releases.RELEASES_URL_ENV, raising=False)
    assert github_releases.releases_page_url() == (
        github_releases.repo_page_url() + "/releases")


# --- through the endpoint -------------------------------------------------

def test_the_version_endpoint_reports_it(no_remote, monkeypatch):
    monkeypatch.setenv("ORIENTA_ALLOWED_HOSTS", "testserver")
    monkeypatch.setenv("ORIENTA_NO_FILE_LOG", "1")
    from fastapi.testclient import TestClient

    from backend.api.main import app

    r = TestClient(app, raise_server_exceptions=False).get("/api/system/version")
    assert r.status_code == 200, r.text
    assert r.json().get("repo_url") == CANONICAL, (
        "the endpoint the dialog reads still has no repo_url")
