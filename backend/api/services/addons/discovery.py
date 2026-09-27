"""Finding add-ons, and finding them without running them.

Two sources, one parser:

* an entry point in the group ``orienta.addons`` whose value is the manifest's
  path INSIDE the distribution, e.g. ``my_addon:orienta-addon.toml``;
* any directory under ``~/.orienta/addons/`` holding an ``orienta-addon.toml``
  (or under ``$ORIENTA_ADDON_USER_DIR``, which overrides it).

The entry point is resolved with ``importlib.util.find_spec``, which locates a
package without executing it. ``EntryPoint.load()`` is deliberately NOT used —
that imports.

One caveat, because it is easy to get wrong later: ``find_spec`` on a DOTTED
name ("my_pkg.sub") imports the parent package to find the child. The
documented entry-point form names a single top-level module, which imports
nothing, and a dotted one is refused here rather than quietly breaking the
invariant this module exists to keep.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from importlib import util as importlib_util
from pathlib import Path
from typing import List, Optional

from .manifest import AddonManifest, ManifestError, parse_manifest

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "orienta.addons"
MANIFEST_NAME = "orienta-addon.toml"
USER_DIR_ENV = "ORIENTA_ADDON_USER_DIR"
#: Extra folders to scan, ``os.pathsep``-separated. Read HERE and not in the
#: route that happens to have needed it first: this module already owns
#: ``$ORIENTA_ADDON_USER_DIR``, and two readers of "where add-ons live" is one
#: fact held twice -- the defect shape this package keeps meeting.
EXTRA_DIRS_ENV = "ORIENTA_ADDON_DIRS"


@dataclass(frozen=True)
class DiscoveredAddon:
    manifest: Optional[AddonManifest]
    origin: str
    error: str = ""
    source_path: Optional[Path] = None


def user_addon_dir() -> Path:
    """Where folder-installed add-ons live.

    ``$ORIENTA_ADDON_USER_DIR`` wins, so a test never reads the developer's
    own installed add-ons and a deployment can put them on a shared drive.
    """
    override = os.environ.get(USER_DIR_ENV)
    if override:
        return Path(override)
    return Path.home() / ".orienta" / "addons"


def extra_addon_dirs() -> List[Path]:
    """The folders ``$ORIENTA_ADDON_DIRS`` names, in order."""
    raw = os.environ.get(EXTRA_DIRS_ENV, "")
    return [Path(p) for p in raw.split(os.pathsep) if p]


def _read(path: Path, origin: str) -> DiscoveredAddon:
    try:
        return DiscoveredAddon(manifest=parse_manifest(path), origin=origin,
                               source_path=path)
    except ManifestError as exc:
        # Shown, not dropped: an add-on the user installed and that does not
        # appear is indistinguishable from one that was never installed.
        logger.warning("add-on manifest rejected: %s", exc)
        return DiscoveredAddon(manifest=None, origin=origin, error=str(exc),
                               source_path=path)


def _from_entry_points() -> List[DiscoveredAddon]:
    out: List[DiscoveredAddon] = []
    try:
        entries = importlib_metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception as exc:
        # Shown, not dropped — the same rule as _read below, and here it
        # costs more: this is not one add-on failing but the whole SOURCE
        # failing, so an empty list would read as "none installed".
        #
        # It is reachable, and the chain is measured. Importing the
        # ``importlib_metadata`` BACKPORT (8.7.0 here) deletes the stdlib
        # ``importlib.machinery.PathFinder.find_distributions`` and appends
        # its own stricter ``MetadataPathFinder`` to ``sys.meta_path``.
        # Orienta reaches that import without ever naming it: ``orix``
        # exposes ``orix.crystal_map`` lazily, so touching ``Phase`` or
        # ``PhaseList`` performs the real import, which pulls in dask, whose
        # ``dask._compatibility`` imports the backport. (Measured: ``import
        # orix.crystal_map`` on its own leaves ``find_distributions``
        # intact, and so does ``import diffpy.structure``; the first
        # attribute touch is what does it.) In the running app a CrystalMap
        # exists long before add-on discovery, so the stricter parser
        # answers this call, and it raises eagerly on ONE malformed
        # ``entry_points.txt`` anywhere in site-packages. Every entry-point
        # add-on on the machine would vanish for a reason the user cannot
        # see. Exception, not BaseException, for the reason written into the
        # per-entry handler below.
        logger.exception("could not read entry points for %s", ENTRY_POINT_GROUP)
        out.append(DiscoveredAddon(
            None, "entry-point",
            f"entry-point discovery itself failed, so no add-on installed as "
            f"a package could be listed: {type(exc).__name__}: {exc}"))
        return out
    for entry in entries:
        try:
            module_name, _, relative = entry.value.partition(":")
            if "." in module_name:
                out.append(DiscoveredAddon(
                    None, "entry-point",
                    f"{entry.name}: entry-point module {module_name!r} is "
                    "dotted; name a top-level package, because locating a "
                    "dotted one would import its parent"))
                continue
            spec = importlib_util.find_spec(module_name)
            locations = list(getattr(spec, "submodule_search_locations", None) or [])
            if spec is None or not locations:
                out.append(DiscoveredAddon(
                    None, "entry-point",
                    f"{entry.name}: cannot locate package {module_name!r}"))
                continue
            out.append(_read(Path(locations[0]) / (relative or MANIFEST_NAME),
                             "entry-point"))
        except Exception as exc:    # one bad entry must not hide the rest
            # Exception, not BaseException: a KeyboardInterrupt or a
            # SystemExit during discovery is the operator stopping the
            # process, and swallowing it into a catalogue row is wrong.
            logger.exception("entry point %s failed", getattr(entry, "name", "?"))
            out.append(DiscoveredAddon(None, "entry-point", f"{entry!r}: {exc}"))
    return out


def _from_dirs(dirs: List[Path]) -> List[DiscoveredAddon]:
    out: List[DiscoveredAddon] = []
    for base in dirs:
        base = Path(base)
        if not base.is_dir():
            continue
        candidates = ([base / MANIFEST_NAME] if (base / MANIFEST_NAME).is_file()
                      else sorted(base.glob(f"*/{MANIFEST_NAME}")))
        for path in candidates:
            out.append(_read(path, "folder"))
    return out


def addon_search_dirs(extra_dirs: Optional[List[Path]] = None, *,
                      include_user_dir: bool = True) -> List[Path]:
    """The folders a discovery run reads, in the order it reads them.

    Exists so the empty listing can NAME them. A user meeting this feature for
    the first time meets an empty list, and one that cannot say where it
    looked sends them to a README outside the app to find out that
    ``~/.orienta/addons`` is the answer.

    Composed here rather than in the route, and then USED by
    ``discover_addons``, so the list shown and the list read are the same
    object -- "where add-ons live" stated twice is the defect this package
    keeps meeting.

    Entry-point add-ons have no directory and are deliberately absent: the
    caller says "folders searched", not "everywhere we looked".
    """
    dirs: List[Path] = []
    if include_user_dir:
        dirs.append(user_addon_dir())
    dirs.extend(extra_dirs or [])
    return dirs


def ensure_user_addon_dir() -> Optional[Path]:
    """Create ``~/.orienta/addons`` if it is missing. Returns it, or None.

    So that the folder the empty listing NAMES is a folder that exists: being
    told to put an add-on in a directory one then has to create by hand is a
    step this app can take itself.

    ONLY the user directory. The folders ``$ORIENTA_ADDON_DIRS`` names are
    explicitly configured, and creating one would turn a typo in that variable
    into a real, permanently empty directory -- hiding the mistake instead of
    leaving it visible as "nothing was found there".

    Never fatal. A read-only home, a roaming profile that is not mounted yet,
    a permissions rule -- none of those is a reason for the add-on LIST to
    fail, and discovery reads a missing directory as empty anyway.

    CALLED AT STARTUP, not from ``discover_addons``. Two reasons, both found by
    review. ``backend/api/security.py`` leaves GET requests ungated on the
    stated ground that "they change nothing", and a listing that created a
    directory made that false -- a cross-origin simple GET would have had a
    side effect even though CORS discarded the answer. And discovery has four
    callers, one of which runs while REFUSING an analysis-key conflict; a
    convenience meant for one screen would have been applied by all of them,
    once per call.
    """
    path = user_addon_dir()
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError:
        logger.warning("could not create the add-on directory %s; add-ons "
                       "installed there will still be found if it is created "
                       "by hand", path, exc_info=True)
        return None


def discover_addons(extra_dirs: Optional[List[Path]] = None, *,
                    include_user_dir: bool = True) -> List[DiscoveredAddon]:
    """Every add-on this installation can see. Imports none of them."""
    dirs = addon_search_dirs(extra_dirs, include_user_dir=include_user_dir)
    found = _from_entry_points() + _from_dirs(dirs)
    return sorted(found, key=lambda d: d.manifest.name if d.manifest else d.error)
