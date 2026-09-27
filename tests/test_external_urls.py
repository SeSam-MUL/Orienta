"""Every URL the installer depends on must actually resolve, and its checksum
document must actually contain our file.

An earlier revision of this work shipped an asset name that could never match
and called it verified; a name asserted only against itself proves nothing. So
every URL here is read OUT of the module rather than rebuilt, and the checksum
document is parsed with the real parser rather than eyeballed.

MEASURED 2026-09-20, and the reason this file exists: the pinned interpreter
release publishes 871 assets, exactly ZERO of them `.sha256`, and one aggregate
`SHA256SUMS`. The derived `<archive>.sha256` URL returned 404 — and because the
setup refuses to install without a digest, that would have made every install
impossible, discovered only after a 48 MB download on a user's machine.

Marked `network`; `pytest.ini` deselects it. Run it before touching the setup,
and again whenever the pinned interpreter release changes:

    pytest tests/test_external_urls.py -m network
"""
from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.network

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALLER_JS = REPO_ROOT / "electron" / "setup" / "installer.js"

_UA = {"User-Agent": "Orienta-Test"}


def _from_node(expression: str) -> str:
    """Ask the module itself, so a changed default cannot slip past this test."""
    result = subprocess.run(
        ["node", "-e",
         f"const m=require({json.dumps(str(INSTALLER_JS))});"
         f"process.stdout.write(String({expression}))"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, f"node failed: {result.stderr}"
    return result.stdout.strip()


def _head(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers=_UA)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (urllib.error.URLError, OSError) as exc:
        pytest.fail(f"{url} is unreachable from this machine: {exc}")


def _get(url: str) -> str:
    with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=60) as r:
        return r.read().decode("utf-8")


def test_the_pinned_interpreter_still_exists():
    url = _from_node("m.pythonUrl()")
    assert url.startswith("https://"), url
    assert _head(url) == 200, f"{url} is gone; every new install would fail"


def test_the_checksum_document_exists():
    url = _from_node("m.pythonSha256Url()")
    assert _head(url) == 200, (
        f"{url} does not exist. The setup refuses to install without a digest, so no "
        "install could complete. Find what the release actually publishes and point "
        "pythonSha256Url() at it (ORIENTA_PYTHON_SHA256_URL overrides it)."
    )


def test_the_checksum_document_contains_our_archive():
    """A 200 proves a response, not a digest — and this document names hundreds
    of files, so taking its first line would give the digest of a different one
    and fail every install with a mismatch that blames the download."""
    from backend.api.services.github_releases import parse_sha256_document

    document = _get(_from_node("m.pythonSha256Url()"))
    archive = _from_node("m.pythonArchiveName()")

    digest = parse_sha256_document(document, filename=archive)
    assert len(digest) == 64

    # And the JavaScript twin must agree, because the INSTALLER is the side that
    # uses it — the first install runs before any Python exists.
    import os
    import tempfile
    tmp = Path(tempfile.gettempdir()) / "orienta-sha256sums-test"
    tmp.write_text(document, encoding="utf-8", newline="")
    try:
        js = subprocess.run(
            ["node", "-e",
             'const fs=require("node:fs");'
             f'const m=require({json.dumps(str(REPO_ROOT / "electron" / "update_endpoints.js"))});'
             'process.stdout.write(m.parseSha256Document('
             'fs.readFileSync(process.argv[1],"utf8"), process.argv[2]))',
             str(tmp), archive],
            capture_output=True, text=True, encoding="utf-8",
        )
        assert js.returncode == 0, js.stderr
        assert js.stdout == digest, "the two digest parsers disagree on the real document"
    finally:
        os.unlink(tmp)


def test_an_aggregate_is_refused_without_a_filename():
    """The safety property, checked against the REAL document rather than a
    fixture: answering it with its first line is the failure mode."""
    from backend.api.services.github_releases import parse_sha256_document

    document = _get(_from_node("m.pythonSha256Url()"))
    if len(document.strip().splitlines()) <= 1:
        pytest.skip("the release now publishes a single-entry document")
    with pytest.raises(ValueError):
        parse_sha256_document(document)


def test_the_runtime_package_of_the_current_release_is_downloadable():
    """Once a release carries the installer assets, they must actually resolve.

    Skipped until one does — no release has been published with them yet, and a
    test that failed for that reason would be noise rather than a finding.

    MEASURED 2026-09-20 on SeSam-MUL/Orienta, and worth knowing before the first
    publish: the newest RELEASE object is v0.2.6 (2026-09-08), while the newest
    TAG is v0.3.0. v0.2.4, v0.2.5 and v0.3.0 are tags with no release. The
    bundle update check reads `releases/latest`, which sees only published,
    non-prerelease Release objects — so those three versions are invisible to
    it, and a bundle install would compare itself against v0.2.6.

    The consequence for the release process: a version is only installable and
    only offered as an update if it has a RELEASE with the runtime package
    attached. Tagging is not enough.
    """
    from backend.api.services import github_releases as gr

    release, reason = gr.latest_release()
    if release is None:
        pytest.skip(f"no release to check yet ({reason})")
    package, checksum = gr.runtime_asset_names(release["tag_name"])
    if not gr.asset_url(release, package):
        pytest.skip(
            f"release {release['tag_name']} predates the installer and carries no "
            f"{package}; run this again after the first publish"
        )
    for name in (package, checksum):
        url = gr.asset_url(release, name)
        assert url, f"release {release['tag_name']} has no asset named {name}"
        assert _head(url) in (200, 302), f"{url} is not downloadable"
