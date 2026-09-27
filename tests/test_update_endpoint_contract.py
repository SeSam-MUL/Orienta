r"""Everything the update path computes in two languages, held level by running
both.

Revision 1 of this work died on exactly this: the build wrote
`orienta-runtime-v0.3.1.zip.sha256` and the updater looked up
`orienta-runtime-sha256`. A later revision repeated the class one layer down,
with a JavaScript digest parser that refused the checksum file this project
publishes.

So: no assertion here is written against a literal. Each value is produced by
BOTH implementations and compared.

The inputs in the digest section are not decorative. Each was a MEASURED
divergence between Python and JavaScript, found by a differential run rather
than by reading either side:

  * "\r\n"    — Python's `$` under MULTILINE matches only before "\n";
                JavaScript's under /m matches before "\r" too. A .sha256 written
                on Windows, the ORDINARY case, parsed there and was refused here.
  * "﻿"  — JS's \s matches the BOM and Python's does not. PowerShell writes
                one by default.
  * "﻿﻿" — Python's lstrip removes a RUN, a single-BOM strip does not.
  * U+2028/9  — line terminators to JavaScript, ordinary characters to Python.
  * "\x1c"    — a separator to Python's str.split(), not to JavaScript's \s.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from backend.api.services import github_releases as gr

REPO_ROOT = Path(__file__).resolve().parents[1]
JS = REPO_ROOT / "electron" / "update_endpoints.js"

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(
    NODE is None,
    reason="node is not installed; the JavaScript half of these contracts cannot run",
)

#: Cleared for the "by default" comparisons. Both sides read these, so a
#: developer with one exported would otherwise compare override against
#: override — and the test would pass while the two DEFAULTS disagreed, which
#: is the drift it is named after.
OVERRIDES = (gr.REPO_SLUG_ENV, gr.API_ROOT_ENV, gr.RELEASES_URL_ENV)


def node(expression: str, env: dict | None = None, scrub: bool = False) -> str:
    """Evaluate an expression against the real JavaScript module."""
    child = {k: v for k, v in os.environ.items()
             if not (scrub and k in OVERRIDES)}
    child.update(env or {})
    result = subprocess.run(
        [NODE, "-e",
         f"const m=require({json.dumps(str(JS))});"
         f"process.stdout.write(String({expression}))"],
        capture_output=True, text=True, encoding="utf-8", env=child,
    )
    assert result.returncode == 0, f"node failed: {result.stderr}"
    return result.stdout


@pytest.fixture
def no_overrides(monkeypatch):
    for name in OVERRIDES:
        monkeypatch.delenv(name, raising=False)


def test_the_javascript_module_loads_at_all():
    """A control. If this fails, every other test here reports vacuously on a
    module that could not be required."""
    assert node("typeof m.runtimeAssetNames") == "function"


# --------------------------------------------------------------------------
# names and urls
# --------------------------------------------------------------------------

def test_the_asset_names_agree():
    tag = "v9.8.7-rc1"
    package, checksum = gr.runtime_asset_names(tag)
    out = node(f"JSON.stringify(m.runtimeAssetNames({json.dumps(tag)}))")
    assert json.loads(out) == {"package": package, "checksum": checksum}


def test_the_api_root_and_repo_slug_agree_by_default(no_overrides):
    assert node("m.githubApiRoot()", scrub=True) == gr.api_root()
    assert node("m.repoSlug()", scrub=True) == gr.repo_slug()


def test_the_releases_page_agrees_by_default(no_overrides):
    assert node("m.releasesPageUrl()", scrub=True) == gr.releases_page_url()


def test_the_release_by_tag_url_agrees_by_default(no_overrides):
    expected = f"{gr.api_root()}/repos/{gr.repo_slug()}/releases/tags/v1.2.3"
    assert node('m.releaseByTagUrl("v1.2.3")', scrub=True) == expected


@pytest.mark.parametrize("const, js_expr, py_getter, value", [
    ("REPO_SLUG_ENV", "m.repoSlug()", "repo_slug", "someone/else"),
    ("API_ROOT_ENV", "m.githubApiRoot()", "api_root", "https://mirror.invalid/api"),
    ("RELEASES_URL_ENV", "m.releasesPageUrl()", "releases_page_url",
     "https://mirror.invalid/x/releases"),
])
def test_both_sides_honour_the_same_environment_variable(
    monkeypatch, const, js_expr, py_getter, value
):
    """One variable, two readers. If only one side honours it, a user pointed at
    a mirror gets half an installation from somewhere else."""
    name = getattr(gr, const)
    monkeypatch.setenv(name, value)
    assert getattr(gr, py_getter)() == value
    assert node(js_expr, env={name: value}) == value


def test_the_environment_variable_names_agree():
    """Read the names from both modules, rather than retyping them here."""
    for const in ("REPO_SLUG_ENV", "API_ROOT_ENV", "RELEASES_URL_ENV"):
        assert node(f"m.{const}") == getattr(gr, const)


# --------------------------------------------------------------------------
# the digest parser: every input below was a measured divergence
# --------------------------------------------------------------------------

_D = "a" * 64
_E = "b" * 64

AGREE_CASES = [
    _D, _D + "\n", _D + "\r\n", _D + "\r",
    "  " + _D, "\t" + _D,
    _D + "  x.zip", _D + " *x.zip", _D + "\t x.zip", _D + "  name with spaces.zip",
    _D.upper(), "0" * 64,
    "﻿" + _D, "﻿" + _D + "\r\n", "﻿﻿" + _D,
    "\x1c" + _D, "\x85" + _D, " " + _D, " " + _D, " " + _D,
    "\v" + _D, "\f" + _D,
    "", "   ", "﻿", "\n", "\r\n",
    "not-a-digest  f.zip", "<!DOCTYPE html>", "<html>\n<body>login</body>\n</html>",
    "a" * 63, "a" * 65, "g" * 64, _D + "x", "x" + _D,
    "\n\n" + _D + "\n\n", "# comment\n" + _D,
    f"{_D}  one.zip\n{_E}  two.zip\n",
    f"{_D}  one.zip\r\n{_E}  two.zip\r\n",
    f"{_D}\n{_E}\n",
]


def _python_verdict(text, filename=None) -> dict:
    try:
        return {"ok": gr.parse_sha256_document(text, filename)}
    except ValueError:
        return {"err": True}


def _node_verdict(text, filename=None) -> dict:
    args = f"{json.dumps(text)}, {json.dumps(filename)}"
    expr = (
        "(() => { try { return JSON.stringify({ ok: m.parseSha256Document("
        + args
        + ") }); } catch (e) { return JSON.stringify({ err: true }); } })()"
    )
    return json.loads(node(expr))


@pytest.mark.parametrize("document", AGREE_CASES)
def test_both_digest_parsers_reach_the_same_verdict(document):
    assert _python_verdict(document) == _node_verdict(document)


@pytest.mark.parametrize("document, filename", [
    (f"{_D}  one.zip\n{_E}  two.zip\n", "two.zip"),
    (f"{_D}  one.zip\r\n{_E}  two.zip\r\n", "two.zip"),
    (f"{_D}  one.zip\n{_E}  two.zip\n", "one.zip"),
    (f"{_D}  one.zip\n{_E}  two.zip\n", "missing.zip"),
    (f"{_D}  sub/dir/two.zip\n", "two.zip"),
    (f"{_D}  sub\\dir\\two.zip\n", "two.zip"),
    (_D + "\n", "anything.zip"),
    (f"{_D} *bin.tar.gz\n", "bin.tar.gz"),
])
def test_both_digest_parsers_agree_when_asked_for_one_file(document, filename):
    """python-build-standalone publishes an aggregate SHA256SUMS as well as
    per-asset files. Answering an aggregate with its FIRST line would return the
    digest of a different file and fail every install with a mismatch that
    blames the download."""
    assert _python_verdict(document, filename) == _node_verdict(document, filename)


def test_an_aggregate_document_is_refused_rather_than_guessed():
    aggregate = f"{_D}  one.zip\n{_E}  two.zip\n"
    assert _python_verdict(aggregate) == {"err": True}
    assert _node_verdict(aggregate) == {"err": True}
    assert _python_verdict(aggregate, "two.zip") == {"ok": _E}


def test_the_digest_parser_accepts_what_the_release_driver_writes(tmp_path):
    """Read the producer, not a guess at the producer."""
    digest = "c" * 64
    path = gr.write_checksum_file(tmp_path / "pkg.zip.sha256", digest, "pkg.zip")
    text = path.read_text(encoding="utf-8")
    assert gr.parse_sha256_document(text) == digest
    assert node(f"m.parseSha256Document({json.dumps(text)})") == digest


# --------------------------------------------------------------------------
# asset lookup
# --------------------------------------------------------------------------

def test_the_asset_lookup_agrees_and_is_exact():
    """A substring match returns the CHECKSUM as the package, which is how
    revision 1 of this work failed."""
    package, checksum = gr.runtime_asset_names("v1.2.3")
    release = {"assets": [
        {"name": checksum, "browser_download_url": "https://x/sum"},
        {"name": package, "browser_download_url": "https://x/pkg"},
    ]}
    blob = json.dumps(release)
    assert node(f"m.assetUrl({blob}, {json.dumps(package)})") == gr.asset_url(release, package)
    assert node(f"m.assetUrl({blob}, {json.dumps(package)})") == "https://x/pkg"
    assert node(f"m.assetUrl({blob}, {json.dumps(checksum)})") == "https://x/sum"
    assert node(f'String(m.assetUrl({blob}, "orienta-runtime"))') == "null"


@pytest.mark.parametrize("release", [
    {}, {"assets": None}, {"assets": []}, {"assets": [None]},
    {"assets": ["orienta-runtime-v1.0.0.zip"]}, {"assets": [{"name": 7}]},
    {"assets": [{"name": "x.zip"}]},          # found, but no url
    [], "not-a-release",
])
def test_a_malformed_release_body_yields_nothing_on_both_sides(release):
    """The module is built to run against an arbitrary ORIENTA_GITHUB_API — a
    mirror or GitHub Enterprise — so a differently shaped body is in scope by
    design and must not raise out of a function whose contract is "url or None".
    """
    assert gr.asset_url(release, "x.zip") is None
    assert node(f"String(m.assetUrl({json.dumps(release)}, \"x.zip\"))") == "null"


def test_neither_side_hardcodes_the_other_half_of_a_name():
    assert JS.read_text(encoding="utf-8").count("orienta-runtime") == 1
    py = (REPO_ROOT / "backend" / "api" / "services" / "github_releases.py").read_text(
        encoding="utf-8")
    assert py.count("orienta-runtime") == 1
