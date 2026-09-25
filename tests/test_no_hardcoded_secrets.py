"""No credential may live in the source tree — and the CIF builder's key resolver.

A Materials Project API key sat in the CIF database builder as a literal, shipped
in every release package and pushed to the public repository (found 2026-09-22
while building the release). The key was revoked; these tests keep the class of
mistake from coming back and pin where the key comes from now.

The leaked key itself is deliberately NOT written here: a test that carries the
secret is the same leak in a different file.

Two lessons from the review of the first draft are built in:
* the name pattern must match a BARE ``API_KEY = "…"``, not only a prefixed
  ``MP_API_KEY`` — the draft would have missed the canonical spelling of the
  very mistake it exists to catch;
* the value test must not demand mixed character classes. A lowercase hex token,
  an all-digit key and an AWS secret (which contains ``/``) are all real keys and
  all slipped through. Length plus entropy is the property; identifier shape is
  only a demotion.
"""
import collections
import importlib.util
import math
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPO_ROOT / "crystal-structures-for-ebsd-main" / "calculationxtal" / "cif_database_builder.py"

#: `API_KEY = "…"`, `MP_API_KEY = "…"`, `token: "…"`, `password = "…"`. The part
#: before the keyword is optional on purpose (see the module docstring).
SECRET_ASSIGNMENT = re.compile(
    r"""(?P<name>[A-Za-z0-9_]*"""
    r"""(?:API_?KEY|APIKEY|ACCESS_?KEY|SECRET_?KEY|PRIVATE_?KEY|AUTH_?TOKEN|"""
    r"""ACCESS_?TOKEN|API_?TOKEN|TOKEN|SECRET|PASSWORD|PASSWD|PWD)"""
    r"""[A-Za-z0-9_]*)\s*[:=]\s*["'](?P<value>[^"'\n]{12,})["']""",
    re.IGNORECASE,
)
PLACEHOLDER = re.compile(
    r"(your|ihren|example|placeholder|xxx|dummy|changeme|redacted|<|\$\{|%\(|"
    r"\{\{|_here|\bhere\b|\.\.\.)",
    re.IGNORECASE,
)
#: Words that mark a VALUE as invented. Deliberately not applied to names: a
#: name like API_SECRET contains "secret" and must stay covered.
VALUE_PLACEHOLDER = re.compile(
    r"(secret|fake|sample|fixture|test|super|invalid|notakey)", re.IGNORECASE)
#: Names that end in KEY but mean a storage slot, an i18n string or a dict key.
BENIGN_NAME = re.compile(
    r"(storage|i18n|label|tip|cache|splitter|panel|sort|theme|locale|column|"
    r"prop|field|query|group|map)_?key", re.IGNORECASE)
#: Binary and data formats — everything else that decodes as text is scanned.
BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".pdf", ".zip", ".gz",
    ".h5", ".hdf5", ".sht", ".xtal", ".npy", ".npz", ".xlsx", ".xls", ".docx",
    ".pptx", ".woff", ".woff2", ".ttf", ".eot", ".exe", ".dll", ".so", ".dylib",
    ".pyc", ".pyo", ".mp4", ".webm", ".bin", ".dat", ".h5oina", ".ang", ".ctf",
}


def _shannon(s: str) -> float:
    counts = collections.Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def _looks_like_a_credential(value: str) -> bool:
    """Length and entropy, not character classes.

    A real key may be lowercase hex, all digits, or contain '/' (AWS). What it
    is not: a sentence, a path with spaces, or a plain dotted identifier.
    """
    if PLACEHOLDER.search(value) or VALUE_PLACEHOLDER.search(value) or " " in value:
        return False
    if len(value) < 12 or _shannon(value) < 2.6:
        return False
    if re.fullmatch(r"[a-z][a-z0-9_.\-]*", value) and _shannon(value) < 3.6:
        return False
    return not value.startswith(("http://", "https://", "/", "./", "../"))


def _tracked_text_files():
    out = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files"],
                         capture_output=True, text=True, check=True).stdout
    for rel in out.splitlines():
        if not rel or rel.startswith("frontend/node_modules"):
            continue
        # This file carries invented keys on purpose, to prove the guard bites.
        if rel.endswith("tests/test_no_hardcoded_secrets.py"):
            continue
        p = REPO_ROOT / rel
        if p.suffix.lower() in BINARY_SUFFIXES or rel.endswith("package-lock.json"):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeDecodeError):
            continue          # not text
        if len(text) > 4_000_000:
            continue
        yield rel, text


def _offenders(text: str):
    for m in SECRET_ASSIGNMENT.finditer(text):
        name, value = m.group("name"), m.group("value")
        if PLACEHOLDER.search(name) or BENIGN_NAME.search(name):
            continue
        if not _looks_like_a_credential(value):
            continue
        yield m, name, value


def test_the_cif_builder_carries_no_key_literal():
    text = BUILDER.read_text(encoding="utf-8")
    hits = [f"{n} = {v[:4]}..." for _, n, v in _offenders(text)]
    assert not hits, f"credential-looking literal in the CIF builder: {hits}"


def test_the_builder_passes_only_the_resolved_key_to_the_api():
    """A literal could come back under a name the pattern does not know; the
    call site cannot lie."""
    text = BUILDER.read_text(encoding="utf-8")
    args = re.findall(r"MPRester\(([^)]*)\)", text)
    assert args, "no MPRester call found — did the file move?"
    assert all(a.strip() == "_mp_key" for a in args), f"MPRester called with {args}"


def test_no_tracked_file_assigns_a_credential_literal():
    """The whole tree, not just the file the leak was found in."""
    found = []
    for rel, text in _tracked_text_files():
        for m, name, _ in _offenders(text):
            found.append(f"{rel}:{text[:m.start()].count(chr(10)) + 1} {name}")
    assert not found, "credential-looking literals:\n  " + "\n  ".join(found)


@pytest.mark.parametrize("spelling", [
    'API_KEY = "Ab3Cd9Ef2Gh7Jk1Lm"',              # the bare form the first draft missed
    'MP_API_KEY = "Ab3Cd9Ef2Gh7Jk1Lm"',           # the spelling that leaked
    'SECRET_KEY = "abcdef0123456789abcdef01"',   # lowercase hex
    'password = "1234567890123456"',             # digits only
    'aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEX"',  # contains /
    'apiToken: "Ab3Cd9Ef2Gh7Jk1Lm"',              # JS object form
])
def test_the_guard_catches_the_shapes_it_claims(spelling):
    assert list(_offenders(spelling)), f"guard misses {spelling}"


@pytest.mark.parametrize("benign", [
    'STORAGE_KEY = "orienta.theme.v2"',
    'SPLITTER_KEY = "cockpit-splitter-pos"',
    'i18nKey = "settings:apiKeys.providers.materialsProject.label"',
    'API_KEY = ""',
    'API_KEY = "YOUR_KEY_HERE"',
    'password = "enter your password"',
])
def test_the_guard_stays_quiet_on_ordinary_code(benign):
    assert not list(_offenders(benign)), f"guard false-positives on {benign}"


# --------------------------------------------------------------------------
# where the key comes from now
@pytest.fixture()
def builder(monkeypatch):
    """Load the builder by path, under a private name.

    Not `import cif_database_builder`: that would leave `calculationxtal/` on
    sys.path for the rest of the session (it also holds Xtal_Generator_GUI.py).
    """
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    pytest.importorskip("pymatgen")
    spec = importlib.util.spec_from_file_location("_cif_builder_under_test", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "_MP_KEY_WARNED", False, raising=False)
    return mod


def test_key_comes_from_the_environment_first(builder, monkeypatch):
    monkeypatch.setenv("MP_API_KEY", "  env-key  ")
    assert builder.mp_api_key() == "env-key"


def test_key_falls_back_to_the_app_settings(builder, monkeypatch):
    monkeypatch.delenv("MP_API_KEY", raising=False)
    from backend.api.services import user_config_manager as uc
    monkeypatch.setattr(uc, "get_api_key",
                        lambda name: "settings-key" if name == "materials_project" else "")
    assert builder.mp_api_key() == "settings-key"


def test_whitespace_only_counts_as_no_key(builder, monkeypatch):
    monkeypatch.setenv("MP_API_KEY", "   ")
    from backend.api.services import user_config_manager as uc
    monkeypatch.setattr(uc, "get_api_key", lambda name: "")
    assert builder.mp_api_key() == ""


def test_no_key_configured_is_an_empty_string_not_a_crash(builder, monkeypatch):
    monkeypatch.delenv("MP_API_KEY", raising=False)
    from backend.api.services import user_config_manager as uc
    monkeypatch.setattr(uc, "get_api_key", lambda name: "")
    assert builder.mp_api_key() == ""


def test_a_missing_key_says_so_once_per_build(builder, monkeypatch, caplog):
    """Silence would leave the DOIs simply absent with no reason given — and a
    process-global flag would silence every build after the first."""
    import logging
    monkeypatch.delenv("MP_API_KEY", raising=False)
    from backend.api.services import user_config_manager as uc
    monkeypatch.setattr(uc, "get_api_key", lambda name: "")
    with caplog.at_level(logging.WARNING):
        assert builder._mp_key_or_warn() == ""
        assert builder._mp_key_or_warn() == ""          # same build: quiet
        builder.reset_mp_key_warning()                  # next build starts over
        assert builder._mp_key_or_warn() == ""
    warnings = [r for r in caplog.records if "Materials Project API key" in r.getMessage()]
    assert len(warnings) == 2
    assert "Settings" in warnings[0].getMessage()
