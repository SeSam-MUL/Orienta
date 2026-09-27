"""Read the project's GitHub releases over plain HTTPS.

No third-party dependency: this runs during an update, possibly before the
environment has been repaired, so it uses only the standard library.

The unauthenticated API allows 60 requests per hour per IP. A university NAT
shares that IP across a building, so the caller must cache aggressively and must
treat a refusal as "unknown", never as "no update available".

Every external location here is overridable by an environment variable. A
literal URL the project does not control is a single point of failure for every
future install — a repo that is renamed, made private, or mirrored onto GitHub
Enterprise must be reachable without a code change.

`electron/update_endpoints.js` is the JavaScript twin of the pure functions
below; the FIRST INSTALL runs before any backend exists and so cannot ask Python
for them. `tests/test_update_endpoint_contract.py` runs both and compares.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

logger = logging.getLogger(__name__)

REPO_SLUG_ENV = "ORIENTA_UPDATE_REPO"
API_ROOT_ENV = "ORIENTA_GITHUB_API"
RELEASES_URL_ENV = "ORIENTA_RELEASES_URL"

DEFAULT_REPO = "SeSam-MUL/Orienta"
DEFAULT_API_ROOT = "https://api.github.com"
DEFAULT_WEB_ROOT = "https://github.com"

#: `\A…\Z` and not `^…$`: in Python `$` also matches BEFORE a trailing newline,
#: so a digest with a stray "\n" passed validation and `write_checksum_file`
#: then produced a two-line document that `sha256sum -c` rejects.
_SHA256_RE = re.compile(r"\A[0-9a-fA-F]{64}\Z")

#: One entry of a checksum document: a digest, optionally followed by a file
#: name (the "*" marks binary mode in the sha256sum convention).
#:
#: The whitespace class is written out as [ \t] rather than using \s or
#: str.split(), because Python and JavaScript do not agree on what whitespace
#: is and this function has a JavaScript twin. Measured: JS's \s matches U+FEFF
#: (the BOM) and Python's does not, while Python's str.split() treats U+001C..1F
#: and U+0085 as separators and JS's \s does not. A BOM is exactly what
#: PowerShell writes, so that divergence was reachable by regenerating a
#: .sha256 file on Windows.
_ENTRY_RE = re.compile(
    r"^[ \t]*(?P<digest>[0-9a-fA-F]{64})(?:[ \t]+\*?(?P<name>[^\r\n]*?))?[ \t]*$",
    re.MULTILINE,
)

_BOM = "\ufeff"

#: Line terminators that EITHER language recognises, all folded to "\n" before
#: matching. Python's re.MULTILINE honours only "\n"; JavaScript's /m also
#: honours "\r", U+2028 and U+2029. Each of those was a MEASURED divergence —
#: a .sha256 written on Windows (CRLF) parsed on the JavaScript side and was
#: refused on this one.
_LINE_BREAKS = ("\r\n", "\r", "\u2028", "\u2029")


def _normalise(text: str) -> str:
    """Strip leading BOMs and fold every line terminator to "\n".

    `lstrip` removes a RUN of BOMs, so the JavaScript twin must too; a
    single-BOM strip there was itself a divergence.
    """
    text = text.lstrip(_BOM)
    for terminator in _LINE_BREAKS:
        text = text.replace(terminator, "\n")
    return text

_TIMEOUT_S = 15.0


def repo_slug() -> str:
    return os.environ.get(REPO_SLUG_ENV, "").strip() or DEFAULT_REPO


def api_root() -> str:
    return (os.environ.get(API_ROOT_ENV, "").strip() or DEFAULT_API_ROOT).rstrip("/")


def releases_page_url() -> str:
    """Where to send a user in a BROWSER when the app cannot help itself.

    Deliberately not derived from `api_root()`: that is the API host and would
    show them JSON.
    """
    override = os.environ.get(RELEASES_URL_ENV, "").strip()
    if override:
        return override.rstrip("/")
    return f"{DEFAULT_WEB_ROOT}/{repo_slug()}/releases"


def runtime_asset_names(tag: str) -> tuple[str, str]:
    """(package, checksum) for a tag. The single Python source of these names.

    `electron/update_endpoints.js` holds the JavaScript twin, and
    `tests/test_update_endpoint_contract.py` keeps the two level by running both.
    Do not rebuild either name anywhere else: revision 1 of this work shipped an
    updater that searched for a name the build never wrote.
    """
    package = f"orienta-runtime-{tag}.zip"
    return package, package + ".sha256"


def _open(url: str, timeout: float):
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "Orienta-Updater"},
    )
    return urllib.request.urlopen(request, timeout=timeout)


def _get(url: str, timeout: float) -> tuple[dict | None, str]:
    try:
        with _open(url, timeout) as response:
            return json.loads(response.read().decode("utf-8")), ""
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 429):
            logger.info("Update check rate-limited by GitHub (%s)", exc.code)
            return None, "rate_limited"
        if exc.code == 404:
            return None, "not_found"
        return None, "unreachable"
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        # ValueError covers a proxy or captive portal answering 200 with an HTML
        # login page: that is unreachable, not a release.
        logger.info("Update check could not reach GitHub: %s", exc)
        return None, "unreachable"


def latest_release(timeout: float = _TIMEOUT_S) -> tuple[dict | None, str]:
    return _get(f"{api_root()}/repos/{repo_slug()}/releases/latest", timeout)


def release_by_tag(tag: str, timeout: float = _TIMEOUT_S) -> tuple[dict | None, str]:
    """The release for one tag, so that what gets installed is the version whose
    notes the user was actually shown.

    The check is cached for hours; resolving `latest` again at install time could
    install something they never saw.
    """
    return _get(f"{api_root()}/repos/{repo_slug()}/releases/tags/{tag}", timeout)


def asset_url(release: dict, exact_name: str) -> str | None:
    """The download URL of one asset, matched EXACTLY.

    Never a substring match: the checksum asset's name contains the package's,
    so `any(name in a["name"])` returns the wrong one — which is precisely how
    revision 1 of this work failed.
    """
    if not isinstance(release, dict):
        return None
    assets = release.get("assets")
    if not isinstance(assets, list):
        return None
    for asset in assets:
        # A mirror or GitHub Enterprise instance may answer with a differently
        # shaped body; `[None]` or a list of bare strings must not raise
        # AttributeError out of a function whose contract is "url or None".
        if isinstance(asset, dict) and asset.get("name") == exact_name:
            url = asset.get("browser_download_url")
            return url if isinstance(url, str) and url else None
    return None


def parse_sha256_document(text, filename: str | None = None) -> str:
    """The digest out of a `.sha256` (or `SHA256SUMS`) document.

    Three producers exist and they disagree. `write_checksum_file` below writes
    the sha256sum form "<hex>  <filename>"; python-build-standalone publishes
    per-asset files containing bare hex **and** an aggregate `SHA256SUMS` listing
    every asset. Pass `filename` whenever the document might be an aggregate:
    without it, a multi-entry document is REFUSED rather than silently answered
    with its first line — which would be the digest of a different file and would
    fail every install with a mismatch that blames the download.

    Raising is the point. A proxy or captive portal serving an HTML login page in
    place of a digest must stop the install, not become a digest.

    Accepts `bytes` as well as `str`: the input is a network download, and a
    caller that forgot to decode would otherwise get `TypeError` out of a
    function documented to raise `ValueError`.
    """
    if isinstance(text, (bytes, bytearray)):
        text = bytes(text).decode("utf-8", errors="replace")
    text = _normalise(text or "")

    entries = [
        (m.group("digest").lower(), (m.group("name") or "").strip())
        for m in _ENTRY_RE.finditer(text)
    ]
    if not entries:
        raise ValueError(
            f"not a SHA-256 digest document (starts with {text[:40]!r})"
        )

    if filename is not None:
        want = PurePosixPath(filename.replace("\\", "/")).name
        for digest, name in entries:
            if name and PurePosixPath(name.replace("\\", "/")).name == want:
                return digest
        raise ValueError(
            f"the checksum document lists {len(entries)} file(s), none of them {want!r}"
        )

    if len(entries) > 1:
        raise ValueError(
            f"the checksum document lists {len(entries)} files; say which one is "
            "wanted by passing filename="
        )
    return entries[0][0]


def write_checksum_file(path: Path, digest: str, name: str) -> Path:
    """Write a `.sha256` in the sha256sum convention: digest, two spaces, name.

    The producer of the format `parse_sha256_document` reads, kept beside it so
    the two cannot drift. `newline=""` with an explicit "\\n" because this file
    is verified by `sha256sum -c` on other platforms, where CRLF fails.
    """
    if not _SHA256_RE.match(digest or ""):
        raise ValueError(f"not a SHA-256 digest: {digest!r}")
    path.write_text(f"{digest.lower()}  {name}\n", encoding="utf-8", newline="")
    return path
