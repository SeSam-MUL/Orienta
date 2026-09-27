import importlib
import shutil
import sys
from pathlib import Path

from backend.api.services.addons import discovery
from backend.api.services.addons.discovery import (
    ENTRY_POINT_GROUP,
    discover_addons,
    user_addon_dir,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _folders(found):
    """Only what came from a directory — entry points are the machine's."""
    return [d for d in found if d.origin == "folder"]


def _entry_points(found):
    return [d for d in found if d.origin == "entry-point"]


def _register_distribution(tmp_path, dist_name, entry_name, entry_value):
    """A REAL installed distribution, discoverable by importlib.metadata.

    Not a monkeypatch of ``entry_points()`` itself: this puts an actual
    ``*.dist-info`` directory with an actual ``entry_points.txt`` on
    ``sys.path`` and lets the stdlib find it, so the test exercises the same
    code path a real add-on package would go through.
    """
    dist_info = tmp_path / f"{dist_name}-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {dist_name}\nVersion: 1.0.0\n",
        encoding="utf-8")
    (dist_info / "entry_points.txt").write_text(
        f"[{ENTRY_POINT_GROUP}]\n{entry_name} = {entry_value}\n",
        encoding="utf-8")


def test_finds_a_manifest_in_a_folder():
    found = discover_addons(extra_dirs=[FIXTURES / "folder_addon"],
                            include_user_dir=False)
    names = [d.manifest.name for d in found if d.manifest]
    assert "folder-addon" in names


def test_a_broken_manifest_is_reported_not_dropped():
    """An add-on that cannot be read must be visible, with its reason."""
    found = _folders(discover_addons(extra_dirs=[FIXTURES / "broken_addon"],
                                     include_user_dir=False))
    assert len(found) == 1
    assert found[0].manifest is None
    assert "api_version" in found[0].error


def test_a_broken_manifest_still_says_where_it_is():
    """The trust dialog shows where an add-on came from on disk; an add-on
    that failed to parse is exactly the one the user needs to go and find."""
    found = _folders(discover_addons(extra_dirs=[FIXTURES / "broken_addon"],
                                     include_user_dir=False))
    assert found[0].source_path == FIXTURES / "broken_addon" / "orienta-addon.toml"


def test_one_broken_manifest_does_not_hide_a_good_one():
    found = discover_addons(extra_dirs=[FIXTURES / "broken_addon",
                                        FIXTURES / "folder_addon"],
                            include_user_dir=False)
    assert any(d.manifest for d in found)
    assert any(d.error for d in found)


def test_discovery_imports_no_addon_code():
    """THE invariant. Discovery reads files; it must not execute the add-on.

    Named against a module that EXISTS and imports only the stdlib, so the
    assertion can fail. Popped first, so a previous test that imported it
    cannot make this one pass by accident — and so the "before" state is the
    same whatever order the suite runs in.
    """
    sys.modules.pop("discovery_probe_module", None)
    assert "discovery_probe_module" not in sys.modules
    discover_addons(extra_dirs=[FIXTURES / "importable_addon"],
                    include_user_dir=False)
    assert "discovery_probe_module" not in sys.modules, (
        "discovery imported the add-on")


def test_a_missing_directory_contributes_nothing():
    """Stated as a difference, not as `== []`: entry points are also scanned,
    and a machine with one installed would fail the absolute form for a
    reason that has nothing to do with the missing directory."""
    base = discover_addons(extra_dirs=[], include_user_dir=False)
    assert discover_addons(extra_dirs=[FIXTURES / "does_not_exist"],
                           include_user_dir=False) == base


def test_the_user_directory_is_read_when_included_and_not_when_excluded():
    """Both directions, against a manifest this test puts there itself.

    `assert len(without) <= len(with_user)` was satisfied by "always include"
    AND by "never include" — it could not fail. `user_addon_dir()` here is the
    tmp_path the conftest points `$ORIENTA_ADDON_USER_DIR` at, so this writes
    nothing into the developer's real home directory.
    """
    target = user_addon_dir() / "folder_addon"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "folder_addon" / "orienta-addon.toml",
                target / "orienta-addon.toml")

    included = discover_addons(extra_dirs=[], include_user_dir=True)
    excluded = discover_addons(extra_dirs=[], include_user_dir=False)
    assert "folder-addon" in [d.manifest.name for d in included if d.manifest]
    assert "folder-addon" not in [d.manifest.name for d in excluded if d.manifest]


def test_user_addon_dir_is_under_the_home_directory(monkeypatch):
    monkeypatch.delenv("ORIENTA_ADDON_USER_DIR", raising=False)
    d = user_addon_dir()
    assert d.name == "addons"
    assert ".orienta" in str(d)


def test_user_addon_dir_honours_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ORIENTA_ADDON_USER_DIR", str(tmp_path / "elsewhere"))
    assert user_addon_dir() == tmp_path / "elsewhere"


def test_entry_point_group_is_the_documented_one():
    assert ENTRY_POINT_GROUP == "orienta.addons"


def test_results_are_sorted_by_name_for_a_stable_catalogue():
    """A literal expected order, not a re-derivation of the function's own
    sort key. The zzz_addon directory sorts LAST and declares the name that
    sorts FIRST, so `list(reversed(found))` — or no sort at all — fails."""
    found = _folders(discover_addons(
        extra_dirs=[FIXTURES / "zzz_addon", FIXTURES / "folder_addon"],
        include_user_dir=False))
    assert [d.manifest.name for d in found] == ["aaa-addon", "folder-addon"]


def test_entry_point_resolves_a_real_manifest_without_importing_the_package(
        tmp_path, monkeypatch):
    """Success branch, over a REAL distribution.

    ``find_spec`` must LOCATE the entry point's package, not import it — the
    whole reason ``EntryPoint.load()`` is not used. Proven directly: after
    discovery finds the manifest, the package itself is still absent from
    ``sys.modules``.
    """
    _register_distribution(tmp_path, "epaddon", "epaddon",
                           "epaddon_pkg:orienta-addon.toml")
    pkg_dir = tmp_path / "epaddon_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy(FIXTURES / "folder_addon" / "orienta-addon.toml",
                pkg_dir / "orienta-addon.toml")

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    sys.modules.pop("epaddon_pkg", None)

    found = _entry_points(discover_addons(extra_dirs=[], include_user_dir=False))
    names = [d.manifest.name for d in found if d.manifest]
    assert "folder-addon" in names
    assert "epaddon_pkg" not in sys.modules


def test_entry_point_naming_a_missing_package_is_reported(tmp_path, monkeypatch):
    """Failure branch: the entry point's own module cannot be located.

    A real distribution, deliberately pointed at a package that was never
    created — ``find_spec`` returns ``None`` rather than raising, and that
    must surface as a visible, explained row, not a silent drop.
    """
    _register_distribution(tmp_path, "epmissing", "epmissing",
                           "epaddon_missing_pkg:orienta-addon.toml")

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    found = _entry_points(discover_addons(extra_dirs=[], include_user_dir=False))
    matches = [d for d in found
              if d.manifest is None and "epaddon_missing_pkg" in d.error]
    assert matches, f"expected a 'cannot locate' row, got: {found}"


def test_entry_point_with_dotted_module_name_is_refused_before_any_import(
        tmp_path, monkeypatch):
    """THE dotted-name hazard, over a real distribution.

    ``find_spec`` on a dotted name ("epaddon_dotted.sub") imports the PARENT
    package as a side effect of locating the child — measured separately:
    with the guard removed, ``epaddon_dotted`` lands in ``sys.modules`` even
    though the child was never found. The guard must refuse before that call
    is ever made, so the parent package here exists and is real, and the
    proof is that it never gets imported.
    """
    _register_distribution(tmp_path, "epdotted", "epdotted",
                           "epaddon_dotted.sub:orienta-addon.toml")
    pkg_dir = tmp_path / "epaddon_dotted"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    sys.modules.pop("epaddon_dotted", None)

    found = _entry_points(discover_addons(extra_dirs=[], include_user_dir=False))
    matches = [d for d in found
              if d.manifest is None and "dotted" in d.error]
    assert matches, f"expected a 'dotted' refusal row, got: {found}"
    assert "epaddon_dotted" not in sys.modules, (
        "resolving the dotted entry point imported its parent package")


def test_a_failure_of_entry_point_discovery_itself_is_shown_not_swallowed(
        monkeypatch):
    """Not one add-on failing — the whole SOURCE failing.

    Measured cause, and the reason this is not hypothetical: importing the
    ``importlib_metadata`` BACKPORT deletes the stdlib
    ``importlib.machinery.PathFinder.find_distributions`` and appends its own
    stricter ``MetadataPathFinder`` to ``sys.meta_path``. Nothing in Orienta
    imports it by name — ``orix.crystal_map`` is a lazy module, so touching
    ``Phase``/``PhaseList`` performs the real import, which reaches the
    backport through ``dask._compatibility``. (``import orix.crystal_map``
    alone does NOT do it, nor does ``import diffpy.structure``: measured, the
    attribute touch is the trigger.) In the running app a CrystalMap exists
    long before discovery, so that stricter parser answers the call, and it
    raises EAGERLY on a single malformed ``entry_points.txt`` anywhere in
    site-packages. Swallowing that returns an empty list, which the user
    cannot tell apart from "nothing installed" — so the catalogue carries a
    row saying what was lost instead.

    The raise is injected at the module attribute rather than by planting a
    corrupt distribution, because which parser answers depends on what else
    the session has imported: the hostile condition would be present or
    absent depending on collection order, which is exactly the flakiness the
    conftest fixture exists to remove.
    """
    class _ExplodingMetadata:
        @staticmethod
        def entry_points(**kwargs):
            raise ValueError("Unable to parse entry_points.txt of broken-dist")

    monkeypatch.setattr(discovery, "importlib_metadata", _ExplodingMetadata)

    found = discover_addons(extra_dirs=[FIXTURES / "folder_addon"],
                            include_user_dir=False)
    rows = _entry_points(found)
    assert len(rows) == 1, rows
    assert rows[0].manifest is None
    assert "entry-point discovery itself failed" in rows[0].error
    assert "broken-dist" in rows[0].error
    # and the sort at the end of discover_addons keys on d.error for exactly
    # this kind of row, so it must not be empty and must not have crashed
    assert rows[0].error
    assert "folder-addon" in [d.manifest.name for d in found if d.manifest]
