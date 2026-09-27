"""The entrypoint must never be served from a cache after an update.

Measured on Sebastian's machine on 2026-09-27, after the runtime update landed:
Settings -> About reported **v0.4.6** and the Indexing toolbar showed **no**
collection button. Both were true. The backend was the new one and answered
`/api/system/version` correctly; the renderer was showing the **0.4.4**
`index.html` and its hashed chunks out of its own HTTP cache. `Ctrl+Shift+R`
brought the feature back, which is the proof that nothing was wrong with the
build.

The cause was one spelling. `_NoCacheIndexStaticFiles` asked
`path in ("", "/", "index.html")`, but `path` is not the URL -- it is what
Starlette's `get_path()` computed, and for `GET /` that is **`"."`**. So the
root, the one address Electron actually loads, went out with no `Cache-Control`
at all: only an ETag and a Last-Modified. With no `Cache-Control`, Chromium
falls back to its freshness HEURISTIC (a tenth of the document's age) and does
not even ask.

These tests mount the REAL class over a temporary directory rather than
requiring `frontend/dist`. A first draft used the built app and skipped when
the frontend was absent -- which is every worktree and every fresh clone, so
the guard would have been green everywhere it mattered and asleep. One test
still goes through the built app where it exists, to prove the mount is wired
with this class and not merely that the class is correct.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.main import _NoCacheIndexStaticFiles, _serves_entrypoint

# What the browser must be told. `no-store` is the load-bearing token: it
# forbids writing to the cache at all, so the heuristic never gets a document
# to be clever about.
REQUIRED = "no-store"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A real mount of the real class over a minimal build."""
    dist = tmp_path_factory.mktemp("dist")
    (dist / "index.html").write_text(
        "<!doctype html><div id=\"root\"></div>", encoding="utf-8")
    assets = dist / "assets"
    assets.mkdir()
    (assets / "index-BcKkMjTo.js").write_text("export default 1;\n", encoding="utf-8")
    (dist / "favicon.svg").write_text("<svg/>", encoding="utf-8")

    app = FastAPI()
    app.mount("/", _NoCacheIndexStaticFiles(directory=str(dist), html=True), name="frontend")
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("url", [
    "/",                            # what Electron loads
    "/?lang=de",                    # what Electron loads with a language: the real case
    "/?lang=ja&view=polefigure",
    "/index.html",                  # what a hard reload asks for
])
def test_the_entrypoint_is_never_cached(client, url):
    r = client.get(url)
    assert r.status_code == 200, f"{url} answered {r.status_code}"
    cc = r.headers.get("cache-control", "")
    assert REQUIRED in cc, (
        f"{url} was served with Cache-Control={cc!r}. Without no-store, Chromium "
        "caches it heuristically and an updated app keeps showing the old one."
    )


def test_a_query_string_does_not_change_the_answer(client):
    """The header must not depend on the query, because the URL does.

    `main.js` loads `http://127.0.0.1:<port>/` plus whatever
    `startLanguage.queryFor()` returns, so the address differs between
    installations and languages. A rule that held for one of them would fail for
    the others on someone else's machine.
    """
    plain = client.get("/").headers.get("cache-control", "")
    with_query = client.get("/?lang=zh").headers.get("cache-control", "")
    assert plain == with_query and REQUIRED in plain


def test_the_hashed_assets_keep_their_caching(client):
    """The chunks are content-addressed, so caching them is the point.

    Not a style preference: a cached chunk can only be reached by an index that
    names it, and the index is now uncacheable. Sending `no-store` here would
    re-download several megabytes on every start for no benefit.
    """
    r = client.get("/assets/index-BcKkMjTo.js")
    assert r.status_code == 200
    assert REQUIRED not in r.headers.get("cache-control", "")


def test_the_predicate_covers_the_spelling_starlette_actually_uses(tmp_path):
    """Asserted against the real `get_path`, because the defect was a belief
    about what that function returns."""
    from starlette.staticfiles import StaticFiles

    (tmp_path / "index.html").write_text("x", encoding="utf-8")
    sf = StaticFiles(directory=str(tmp_path), html=True)
    computed = {
        url: sf.get_path({"type": "http", "path": url, "method": "GET", "headers": []})
        for url in ("/", "/index.html", "/assets/x.js")
    }
    # The measurement that explains the bug: the root is neither "" nor "/".
    assert computed["/"] == "."
    assert computed["/index.html"] == "index.html"

    assert _serves_entrypoint(computed["/"]) is True
    assert _serves_entrypoint(computed["/index.html"]) is True
    assert _serves_entrypoint(computed["/assets/x.js"]) is False
    # Spellings that do not occur today but would cost another silent morning.
    for spelling in ("", "/", "./", "index.html", "./index.html"):
        assert _serves_entrypoint(spelling) is True, spelling
    for other in ("assets/index-abc123.js", "favicon.svg", "index.html.map"):
        assert _serves_entrypoint(other) is False, other


def test_the_shipped_app_mounts_this_class():
    """The class being right is worth nothing if the app mounts another one.

    Skips where the frontend is not built, and says so: this is the one check
    that cannot be made without it, which is why the rest of the file does not
    depend on it.
    """
    from backend.api import main as main_module

    if not main_module.FRONTEND_DIST.exists():
        pytest.skip("frontend/dist is not built in this tree; the mount does not exist")

    mounts = [r for r in main_module.app.routes if getattr(r, "name", None) == "frontend"]
    assert mounts, "the frontend mount is gone"
    assert isinstance(mounts[0].app, _NoCacheIndexStaticFiles)

    with TestClient(main_module.app) as c:
        r = c.get("/")
        assert r.status_code == 200
        assert REQUIRED in r.headers.get("cache-control", "")
