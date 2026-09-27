"""Which add-ons this build can run, which the user has agreed to, and which
have earned a pause.

This is not security. An add-on runs in Orienta's own process with the user's
permissions, and no comparable project sandboxes — napari, QGIS, ImageJ,
Slicer and HyperSpy all rely on a named author and a public repository. What
this module provides is informed consent: the decision is asked once, recorded,
and revisited automatically when an add-on keeps crashing.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

CRASH_LIMIT = 3
TRUST_FILE_ENV = "ORIENTA_ADDON_TRUST_FILE"

#: Serialises read-modify-write on the store WITHIN this process.
#:
#: Every mutator is a read-modify-write, and FastAPI runs synchronous routes
#: in a thread pool, so two requests enabling two different add-ons really do
#: overlap: both read the same file, each adds its own key, the second write
#: wins and the first user's decision is gone with no message. Electron drives
#: concurrent requests, so this is narrow, not hypothetical.
#:
#: Module level and re-entrant: the lock protects the FILE, which every
#: instance shares, so a per-instance lock would protect nothing -- each
#: request builds its own TrustStore.
#:
#: WHAT REMAINS RACY, plainly: two PROCESSES. A second backend, or a script
#: run beside the app, still reads and writes without coordination, and the
#: reload below narrows that window to the microseconds between reload and
#: os.replace rather than closing it. A lock file would close it and was
#: rejected: a crashed holder leaves a stale lock, which needs a timeout,
#: which needs a policy for breaking someone else's lock -- a larger failure
#: surface than the one it removes, for a file holding a handful of booleans
#: the user can set again.
_STORE_LOCK = threading.RLock()

_COMPARATORS = ("==", ">=", "<=", ">", "<")
# A leading "v" is optional on BOTH sides: app_version's `release` is a git
# tag and this project's tags are "v0.3.0", while a manifest is more likely
# to write ">=0.4". Neither spelling may be the one that fails.
_NUMERIC = re.compile(r"^v?(\d+(?:\.\d+)*)")


def trust_store_path() -> Path:
    """Where the enable/disable decisions live.

    ``$ORIENTA_ADDON_TRUST_FILE`` wins, so a test never writes into the
    developer's real home directory — which is not hypothetical tidiness:
    without it the first run of the API tests leaves a file behind and the
    second run fails on "a new add-on is listed as not yet known".
    """
    override = os.environ.get(TRUST_FILE_ENV)
    if override:
        return Path(override)
    return Path.home() / ".orienta" / "addons-trust.json"


def running_orienta_version() -> Optional[str]:
    """This build's RELEASE, or None when it has no tag to name.

    Deliberately not ``get_version_info()["version"]``: that is a display
    string ("<date> (<short hash>)" on an untagged checkout, "v0.1.0+164
    (<short hash>)" past a tag), and comparing it numerically reads the YEAR
    as a major version. A build that cannot say which release it is says None, and
    every constraint that asks for one then fails — see
    :func:`orienta_version_satisfies`.
    """
    from ..app_version import get_version_info

    try:
        return get_version_info().get("release")
    except Exception:                       # pragma: no cover - defensive
        logger.exception("could not resolve the running Orienta version")
        return None


def _numeric_parts(version):
    """Leading dotted numbers of a version, or None when there are none."""
    if version is None:
        return None
    match = _NUMERIC.match(str(version).strip())
    if not match:
        return None
    return tuple(int(p) for p in match.group(1).split("."))


def _pad(a, b):
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)), b + (0,) * (n - len(b))


def orienta_version_satisfies(requires: str, version: Optional[str]) -> bool:
    """Does ``version`` meet every comparator in ``requires``?

    ``requires`` is the comma-separated form the manifest uses, e.g.
    ">=0.4,<0.6". An empty constraint accepts anything, including a version
    that cannot be parsed — the add-on asked for nothing.
    """
    clauses = [c.strip() for c in (requires or "").split(",") if c.strip()]
    if not clauses:
        return True
    running = _numeric_parts(version)
    if running is None:
        return False
    for clause in clauses:
        for op in _COMPARATORS:
            if clause.startswith(op):
                wanted = _numeric_parts(clause[len(op):])
                if wanted is None:
                    return False
                a, b = _pad(running, wanted)
                ok = {
                    "==": a == b, ">=": a >= b, "<=": a <= b,
                    ">": a > b, "<": a < b,
                }[op]
                if not ok:
                    return False
                break
        else:
            logger.warning("unrecognised version clause %r", clause)
            return False
    return True


class TrustStore:
    """The user's enable/disable decisions, as a small JSON file."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else trust_store_path()
        self._data: Dict[str, dict] = self._load()

    def _load(self) -> Dict[str, dict]:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except Exception:
            # A corrupt store must not stop the app; it means "nothing decided
            # yet", and the user is asked again.
            logger.exception("add-on trust store unreadable at %s", self.path)
            return {}

    def _save(self) -> None:
        """Write the store so that a FAILED write changes nothing.

        The previous form opened the real file for writing -- which truncates
        before a single byte is serialised -- and only then called
        ``json.dump``. Anything that raised mid-serialisation therefore left a
        truncated file, which ``_load`` swallows as "nothing decided yet": one
        add-on with an unserialisable manifest field erased every consent
        decision on the machine, silently. Measured, with a TOML date.

        So: serialise into a temp file beside the target, then ``os.replace``,
        which is atomic on POSIX and on Windows. The target is either the old
        content or the new one, never a half of either. The temp file must
        share the directory, or the replace crosses a filesystem and stops
        being atomic.

        Honest limit: ``fsync`` on the temp file makes its CONTENT durable
        before the swap, but the directory entry is not synced (Windows offers
        no portable way), so a power loss immediately after ``os.replace`` can
        still leave the old file on some filesystems. That is a different
        failure from the one measured here, and the one measured here is now
        impossible.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        finally:
            try:
                tmp.unlink()            # a no-op once os.replace consumed it
            except OSError:
                pass

    def _commit(self, snapshot: Dict[str, dict]) -> None:
        """Save, or put the in-memory picture back and re-raise.

        A store that kept a decision in memory after failing to write it would
        answer ``is_enabled`` with something the file cannot back up, which is
        exactly the "silently dropped" this package refuses everywhere else.
        The caller is told by the exception; nobody has to guess.
        """
        try:
            self._save()
        except Exception:
            self._data.clear()
            self._data.update(snapshot)
            raise

    def _snapshot(self) -> Dict[str, dict]:
        return {name: dict(entry) for name, entry in self._data.items()}

    def _reload(self) -> None:
        """Re-read the file immediately before modifying it.

        ``__init__`` may have run a whole request ago. Without this, two
        concurrent enables both write the state they read at construction and
        the first decision vanishes. See ``_STORE_LOCK`` for what this does
        and does not close.
        """
        self._data = self._load()

    def is_known(self, name: str) -> bool:
        return name in self._data

    def known_version(self, name: str) -> str:
        """The version the recorded decision was made ABOUT.

        ``is_known`` answers "a decision exists under this name", and a name is
        not an identity: delete ``grain-stats`` and drop a colleague's
        ``grain-stats`` in its place, and the old row still says known. Only
        one is installed, so the duplicate guard sees no conflict, and consent
        given for one author's code would silently cover another's. These two
        readers are what let a caller notice that the thing in front of it is
        not the thing that was agreed to.
        """
        return str(self._data.get(name, {}).get("version", "") or "")

    def known_doi(self, name: str) -> str:
        return str(self._data.get(name, {}).get("doi", "") or "")

    def is_enabled(self, name: str) -> bool:
        return bool(self._data.get(name, {}).get("enabled", False))

    def crashes(self, name: str) -> int:
        return int(self._data.get(name, {}).get("crashes", 0))

    def set_enabled(self, name: str, enabled: bool, *, version: str,
                    doi: str, by: str = "user") -> None:
        """Record a decision, and WHOSE it was.

        ``by`` is written down rather than inferred later, because the two
        cases are indistinguishable from the stored state. Orienta itself
        calls this with ``enabled=False`` when an add-on raises on
        activation -- the user pressed ENABLE -- and the row it leaves,
        ``(enabled=false, known=true, crashes=0)``, is byte for byte the row a
        user's own switch-off leaves. Measured: the page then told the user
        they had switched off the add-on they had just tried to switch on.
        """
        with _STORE_LOCK:
            self._reload()
            snapshot = self._snapshot()
            entry = self._data.setdefault(name, {})
            entry.update({"enabled": bool(enabled), "version": version,
                          "doi": doi, "crashes": 0})
            if enabled:
                entry.pop("disabled_by", None)
            else:
                entry["disabled_by"] = str(by)
            self._commit(snapshot)

    def disabled_by(self, name: str) -> Optional[str]:
        """Who switched it off, as recorded; None if nothing was recorded.

        None also covers every trust file written before this field existed.
        The caller falls back to inferring, which is right for those: they
        predate the failed-activation path writing rows of its own.
        """
        value = self._data.get(name, {}).get("disabled_by")
        return str(value) if value else None

    def record_crash(self, name: str) -> int:
        with _STORE_LOCK:
            self._reload()
            snapshot = self._snapshot()
            entry = self._data.setdefault(name, {"enabled": True, "version": "",
                                                 "doi": "", "crashes": 0})
            entry["crashes"] = int(entry.get("crashes", 0)) + 1
            if entry["crashes"] >= CRASH_LIMIT:
                entry["enabled"] = False
                # Recorded, so a later success zeroing the counter cannot
                # turn this into "the user switched it off". Two runs in
                # flight make that reachable: the third failure disables, the
                # fourth succeeds and clears the count.
                entry["disabled_by"] = "runtime"
                logger.warning(
                    "add-on %s disabled after %d failures; switch it on "
                    "again under Add-ons (or POST /api/addons/%s/enabled) "
                    "once its author has fixed it",
                    name, entry["crashes"], name)
            self._commit(snapshot)
            return entry["crashes"]

    def clear_crashes(self, name: str) -> None:
        with _STORE_LOCK:
            self._reload()
            if name in self._data and self._data[name].get("crashes"):
                snapshot = self._snapshot()
                self._data[name]["crashes"] = 0
                self._commit(snapshot)
