"""The releases API, its non-answers, and exact asset naming.

Rate limiting matters more than it looks: the unauthenticated limit is 60
requests an hour PER IP, and a university NAT shares one IP across a building.
A rate-limited check must read "unknown", never "you are up to date".
"""
from __future__ import annotations

import json
import urllib.error

import pytest

from backend.api.services import github_releases as gr


class _Resp:
    def __init__(self, status, payload=None):
        self.status, self._payload = status, payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_a_release_is_returned_with_no_reason(monkeypatch):
    monkeypatch.setattr(
        gr, "_open",
        lambda url, timeout: _Resp(200, {"tag_name": "v0.4.0", "body": "notes", "assets": []}),
    )
    release, reason = gr.latest_release()
    assert reason == ""
    assert release["tag_name"] == "v0.4.0"


@pytest.mark.parametrize(
    "status, expected",
    [(403, "rate_limited"), (429, "rate_limited"), (404, "not_found"), (500, "unreachable")],
)
def test_error_statuses_map_to_distinguishable_reasons(monkeypatch, status, expected):
    def boom(url, timeout):
        raise urllib.error.HTTPError(url, status, "no", {}, None)

    monkeypatch.setattr(gr, "_open", boom)
    assert gr.latest_release() == (None, expected)


def test_a_network_failure_is_unreachable_not_a_crash(monkeypatch):
    monkeypatch.setattr(
        gr, "_open", lambda url, timeout: (_ for _ in ()).throw(OSError("no route")))
    assert gr.latest_release() == (None, "unreachable")


def test_a_body_that_is_not_json_is_unreachable_not_a_crash(monkeypatch):
    class _Junk(_Resp):
        def read(self):
            return b"<!DOCTYPE html><html>proxy login</html>"

    monkeypatch.setattr(gr, "_open", lambda url, timeout: _Junk(200))
    assert gr.latest_release() == (None, "unreachable")


def test_release_by_tag_asks_for_that_tag(monkeypatch):
    seen = {}

    def capture(url, timeout):
        seen["url"] = url
        return _Resp(200, {"tag_name": "v0.4.0", "body": "", "assets": []})

    monkeypatch.setattr(gr, "_open", capture)
    gr.release_by_tag("v0.4.0")
    assert seen["url"].endswith("/releases/tags/v0.4.0")


def test_the_names_the_build_writes_are_the_names_the_updater_looks_up():
    package, checksum = gr.runtime_asset_names("v0.4.0")
    assert package == "orienta-runtime-v0.4.0.zip"
    assert checksum == "orienta-runtime-v0.4.0.zip.sha256"

    # The checksum is listed FIRST on purpose: a substring match returns it as
    # the package. That was revision 1 of this plan's defect.
    release = {"assets": [
        {"name": checksum, "browser_download_url": "https://x/sum"},
        {"name": package, "browser_download_url": "https://x/pkg"},
    ]}
    assert gr.asset_url(release, package) == "https://x/pkg"
    assert gr.asset_url(release, checksum) == "https://x/sum"
    assert gr.asset_url(release, "orienta-runtime") is None, "matching must be exact"


def test_asset_url_survives_a_release_with_no_assets_key():
    assert gr.asset_url({}, "anything") is None
    assert gr.asset_url({"assets": None}, "anything") is None


@pytest.mark.parametrize("const, getter, expected", [
    ("REPO_SLUG_ENV", "repo_slug", "someone/else"),
    ("API_ROOT_ENV", "api_root", "https://github.example.invalid/api/v3"),
    ("RELEASES_URL_ENV", "releases_page_url", "https://git.example.invalid/x/y/releases"),
])
def test_every_external_location_is_overridable(monkeypatch, const, getter, expected):
    """A repo that goes private, moves, or lives on GitHub Enterprise must be
    reachable without a code change.

    The variable NAME is read from the module, not retyped here: a test that
    hardcodes it keeps passing after the constant is renamed and nothing reads
    the variable any more.
    """
    monkeypatch.setenv(getattr(gr, const), expected)
    assert getattr(gr, getter)() == expected


def test_a_trailing_slash_on_the_api_root_does_not_double_up(monkeypatch):
    monkeypatch.setenv(gr.API_ROOT_ENV, "https://example.invalid/api/v3/")
    assert gr.api_root() == "https://example.invalid/api/v3"


def test_the_releases_page_is_a_web_url_not_the_api_root():
    """The failure dialog sends the user here in a browser. `api_root()` is the
    API host and would show them JSON."""
    assert "api." not in gr.releases_page_url()
    assert gr.releases_page_url().endswith("/releases")
    assert gr.repo_slug() in gr.releases_page_url()


# --------------------------------------------------------------------------
# the checksum document: one producer, one parser
# --------------------------------------------------------------------------

def test_the_digest_parser_accepts_both_formats_in_the_wild():
    """Two producers, two shapes, one parser.

    scripts/build_release.py writes "<hex>  <filename>" (the sha256sum
    convention). python-build-standalone publishes bare hex. A parser that
    accepted only one would kill every install of the other kind after the
    download had already happened.
    """
    digest = "a" * 64
    assert gr.parse_sha256_document(digest) == digest
    assert gr.parse_sha256_document(digest + "\n") == digest
    assert gr.parse_sha256_document(f"{digest}  orienta-runtime-v0.4.0.zip\n") == digest
    assert gr.parse_sha256_document(f"{digest} *orienta-runtime-v0.4.0.zip\n") == digest
    assert gr.parse_sha256_document(digest.upper()) == digest, "normalised to lower case"


@pytest.mark.parametrize("junk", [
    "", "   ", "not-a-digest  file.zip", "<!DOCTYPE html>", "a" * 63, "a" * 65, "g" * 64,
])
def test_the_digest_parser_refuses_anything_else(junk):
    """A captive portal serving an HTML login page instead of a digest must stop
    the install, not become a digest that then "mismatches" and blames the
    download."""
    with pytest.raises(ValueError):
        gr.parse_sha256_document(junk)


def test_write_checksum_file_round_trips_through_the_parser(tmp_path):
    digest = "b" * 64
    path = gr.write_checksum_file(tmp_path / "x.zip.sha256", digest, "x.zip")
    text = path.read_text(encoding="utf-8")
    assert text == f"{digest}  x.zip\n"
    assert gr.parse_sha256_document(text) == digest


def test_write_checksum_file_refuses_a_non_digest(tmp_path):
    with pytest.raises(ValueError):
        gr.write_checksum_file(tmp_path / "x.sha256", "not-a-digest", "x.zip")


def test_write_checksum_file_writes_lf_not_crlf(tmp_path):
    """Every write to a repo file in this project specifies its line endings;
    this one is also consumed by `sha256sum -c` on other platforms."""
    path = gr.write_checksum_file(tmp_path / "x.sha256", "c" * 64, "x.zip")
    assert b"\r\n" not in path.read_bytes()
