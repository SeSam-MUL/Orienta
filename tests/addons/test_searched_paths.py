"""The listing says where it looked, because an empty one otherwise says nothing.

A user meeting this feature for the first time meets an empty list. Before
this, the app contained no statement anywhere of where an add-on has to be put
— the answer lived in a README outside it.
"""
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.services.addons.discovery import (
    addon_search_dirs,
    discover_addons,
    user_addon_dir,
)

FIXTURES = Path(__file__).parent / "fixtures"
client = TestClient(app)


def test_the_listing_names_the_user_directory(monkeypatch):
    monkeypatch.delenv("ORIENTA_ADDON_DIRS", raising=False)
    body = client.get("/api/addons").json()
    assert body["searched_paths"] == [str(user_addon_dir())]


def test_extra_directories_are_named_too_and_in_order(monkeypatch):
    a, b = FIXTURES / "runnable_addon", FIXTURES / "two_maps_addon"
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", f"{a}{os.pathsep}{b}")
    paths = client.get("/api/addons").json()["searched_paths"]
    assert paths == [str(user_addon_dir()), str(a), str(b)]


def test_the_paths_reported_are_the_paths_READ(monkeypatch):
    """The point of putting this in discovery rather than the route.

    Composing the list a second time in the route would be "where add-ons
    live" stated twice — the defect shape this package keeps meeting, and one
    that would show the user a folder nothing actually reads.

    Asserted through behaviour, not by comparing two call sites: an add-on
    placed in a reported directory must be found.
    """
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(FIXTURES / "runnable_addon"))
    body = client.get("/api/addons").json()
    reported = body["searched_paths"]
    assert str(FIXTURES / "runnable_addon") in reported
    assert "runnable" in [a["name"] for a in body["addons"]]

    # And the other direction: a directory that is NOT reported is not read.
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(FIXTURES / "two_maps_addon"))
    body = client.get("/api/addons").json()
    assert str(FIXTURES / "runnable_addon") not in body["searched_paths"]
    assert "runnable" not in [a["name"] for a in body["addons"]]


def test_the_route_reports_the_SAME_list_the_search_uses(monkeypatch):
    """The guard against the two lists drifting apart.

    An earlier version of this test planted a manifest in a directory it had
    passed in and asserted it was found — which any implementation that reads
    its own argument passes, re-composed or not. It could only ever catch
    DIVERGENT drift, and the defect shape here is an identical duplication
    that drifts later. So the route's answer is compared to the composer's,
    for a configuration with more than one entry.
    """
    a, b = FIXTURES / "runnable_addon", FIXTURES / "two_maps_addon"
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", f"{a}{os.pathsep}{b}")
    reported = client.get("/api/addons").json()["searched_paths"]
    composed = [str(p) for p in addon_search_dirs([a, b])]
    assert reported == composed


def test_an_entry_point_addon_is_listed_without_a_directory(monkeypatch):
    """So the field is FOLDERS searched, not "everywhere we looked".

    Asserted through a real entry-point add-on rather than by restating the
    three-line composer: it must be in the listing, and its location must NOT
    be among the searched paths — that is what makes the empty state's
    wording true instead of merely careful.
    """
    monkeypatch.delenv("ORIENTA_ADDON_DIRS", raising=False)
    import backend.api.services.addons.discovery as disc

    fake = disc.DiscoveredAddon(
        manifest=None, origin="entry_point", error="from an installed package",
        source_path=None)
    monkeypatch.setattr(disc, "_from_entry_points", lambda: [fake])
    body = client.get("/api/addons").json()
    assert any(a["origin"] == "entry_point" for a in body["addons"])
    assert body["searched_paths"] == [str(user_addon_dir())]


def test_the_user_directory_is_created_at_startup(monkeypatch, tmp_path):
    """The empty state names it; being told to create it by hand is a step
    this app can take itself.

    At STARTUP, through the real lifespan, not on a listing: ``security.py``
    leaves GET requests ungated on the stated ground that "they change
    nothing", and a listing that created a directory would have made that
    false -- a cross-origin simple GET would have had a side effect even
    though CORS discarded the answer.
    """
    target = tmp_path / "home" / ".orienta" / "addons"
    monkeypatch.setenv("ORIENTA_ADDON_USER_DIR", str(target))
    assert not target.exists()
    with TestClient(app):                    # runs the lifespan
        pass
    assert target.is_dir()


def test_listing_the_addons_writes_NOTHING(monkeypatch, tmp_path):
    """A GET stays a GET. This is the invariant the security layer rests on."""
    target = tmp_path / "never" / "created"
    monkeypatch.setenv("ORIENTA_ADDON_USER_DIR", str(target))
    r = client.get("/api/addons")
    assert r.status_code == 200
    assert str(target) in r.json()["searched_paths"]
    assert not target.exists()


def test_a_configured_directory_is_NOT_created(monkeypatch, tmp_path):
    """A typo in $ORIENTA_ADDON_DIRS must stay visible.

    Creating it would turn the mistake into a real, permanently empty folder
    and the user would look for their add-on in a directory that only exists
    because they misspelled it.
    """
    from backend.api.services.addons.discovery import ensure_user_addon_dir

    typo = tmp_path / "addnos"
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(typo))
    ensure_user_addon_dir()
    client.get("/api/addons")
    assert not typo.exists()


def test_a_user_directory_that_cannot_be_created_does_not_break_the_listing(
        monkeypatch, tmp_path):
    """A read-only home is not a reason for the list to fail.

    Discovery reads a missing directory as empty, so the only thing lost is
    the convenience -- and the listing still names the path.
    """
    from backend.api.services.addons.discovery import ensure_user_addon_dir

    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    monkeypatch.setenv("ORIENTA_ADDON_USER_DIR", str(blocker / "addons"))
    assert ensure_user_addon_dir() is None     # and did not raise
    r = client.get("/api/addons")
    assert r.status_code == 200, r.text
    assert str(blocker / "addons") in r.json()["searched_paths"]


def test_discovery_reads_exactly_the_directories_it_reports(monkeypatch, tmp_path):
    """A guard against the two lists drifting apart later.

    ``discover_addons`` calls ``addon_search_dirs``; if someone re-composes
    the directory list inside it, this is what notices. Measured by planting
    a manifest in the LAST reported directory: it has to be found.
    """
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(tmp_path))
    (tmp_path / "orienta-addon.toml").write_text(
        'api_version = 0\nname = "planted"\ndisplay_name = "Planted"\n'
        'version = "0.1.0"\nauthors = ["T <t@example.org>"]\ndoi = ""\n'
        # A manifest declaring no analyses is REJECTED, so it would appear as
        # an error row rather than as a found add-on -- and this test would
        # then pass or fail for a reason that has nothing to do with paths.
        '\n[[analyses]]\nkey = "addon.planted"\nlabel = "Planted"\n'
        'python_name = "planted_module:analyse"\nsentence = "It ran."\n',
        encoding="utf-8")
    reported = addon_search_dirs([tmp_path])
    assert reported[-1] == tmp_path
    found = discover_addons(extra_dirs=[tmp_path])
    assert "planted" in [d.manifest.name for d in found if d.manifest]
