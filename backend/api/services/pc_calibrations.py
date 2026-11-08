"""Named pattern-centre calibrations: a PC that outlives a file, with the settings it is valid for.

A calibration is what a person means by "the PC of the SEM on that day": one pattern centre
(Bruker convention), the acquisition settings it was found for (the ``features``), where it came
from (``origin``), which files use it (``members``) and every earlier PC it has had
(``versions``: a new refine makes a new version, the old one stays readable, so a result
indexed at an older PC can still say which PC that was).

THE RULE OF THIS MODULE: NOTHING IS APPLIED WITHOUT BEING ASKED. ``suggest`` only reads.
``apply`` changes a dataset only for a dataset the caller names, and refuses a dataset whose
settings differ from the calibration's unless the caller names that dataset again as accepted.
Loading a file never calls ``apply``.

Where it lives: one JSON file per calibration in ``Database/Calibrations/`` beside the phase
library (it travels with it, like ``Database/Collections``): schema version, LF line endings,
atomic writes, ONE write lock for the folder, no absolute path anywhere (a file is recognised by
name, size and a header id, see ``fingerprint_of``). The folder can be moved with the
environment variable ``ORIENTA_CALIBRATIONS_DIR`` (a live check against a read-only library, a
test); ``set_calibrations_dir_for_test`` does the same for a test.

"Same settings" means: pattern shape and binning equal; camera tilt and sample tilt within ``TOL_TILT_DEG``; working
distance and detector insertion within ``TOL_WD_MM``, each when both headers have one. Date, instrument, beam
energy and magnification are only shown next to a proposal (``hints``), never compared: none of the file formats
probed states a microscope or a detector model, so there is nothing to compare. A setting one side does not state
is ``unknown``, never a match. A calibration file whose ``features`` hold a key not listed in ``FEATURE_KEYS``
loads; the key is ignored by every comparison and left as it is in the file.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import logging
import math
import numbers
import os
import re
import tempfile
import threading
import time
import uuid
from datetime import date as _calendar_date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA = 1
_DIRNAME = "Calibrations"
ENV_DIR = "ORIENTA_CALIBRATIONS_DIR"
TOL_TILT_DEG = 0.1
TOL_WD_MM = 0.1
#: Longest name and author, in characters (after whitespace is normalised for the name).
MAX_NAME_CHARS = 200
MAX_AUTHOR_CHARS = 200
#: The "header id" of a file: a hash of its first 64 KiB (see ``fingerprint_of``).
HEADER_BYTES = 64 * 1024
FEATURE_KEYS = ("pattern_shape", "binning", "camera_tilt", "sample_tilt", "working_distance_mm",
                "detector_insertion_mm")
#: The settings that can be unknown (the pattern shape always is known: it is the shape of the data). A
#: calibration file written before ``detector_insertion_mm`` existed lacks it and reads it as ``None``.
SETTING_KEYS = ("binning", "camera_tilt", "sample_tilt", "working_distance_mm", "detector_insertion_mm")
#: The key of the file's header that states the detector insertion, in mm (Oxford ``.h5oina``: ``Detector
#: Insertion Distance``). Read from the ORIGINAL metadata; no other format of this project states one.
INSERTION_KEY = "Detector Insertion Distance"

#: Settings a file FORMAT does not carry. For such a file kikuchipy's detector holds its own defaults
#: (camera tilt 0, sample tilt 70, binning 1), which nobody measured; they must not be compared as if
#: they had been. Read from the loaders (re-read them if one changes):
#:   "ebsp"   Oxford binary patterns: kikuchipy's ``oxford_binary`` reader sets no detector geometry.
#:   "up"     EDAX UP1/UP2: ``kikuchipy.edax_binary`` sets none, and ``safe_loader._load_edax_up`` builds a new
#:            detector from the .osc's PC and the default tilts (``safe_loader.py:592-604``), so the sidecar's
#:            tilts do not arrive either.
#:   "nordif" any ``.dat``: ``kikuchipy.nordif`` reads the tilt angle and the elevation from ``Setting.txt``
#:            (``nordif/_api.py:283-284``) and no binning. No other loader of this project takes a ``.dat``.
#:   "edax_h5", "bruker_h5"  EDAX and Bruker h5ebsd: the readers (``edax_h5ebsd/_api.py``, ``bruker_h5ebsd/_api.py``,
#:            ``edax_hex.py``) read tilts but no binning, so the detector's 1 is kikuchipy's default.
#:   "unified_fallback"  the signal ``ebsd_adapter`` builds from ``unified_loader``'s ``DetectorData`` (safe_loader's
#:            second strategy): it records neither a file name nor a manufacturer. ``DetectorData.binning`` is never
#:            read from the file (the dataclass default 1) and the tilts are read with defaults (the Oxford reference scan: binning 8
#:            and camera tilt 4.32 in the file, 1 and 0 in the signal). Told by exactly that absence.
#: A format not listed here counts as stating all three (a loader that silently falls back to a default for a
#: header that lacks a key is not caught by this table). Oxford is deliberately not listed: with kikuchipy
#: 0.11.3 (the test environment) an Aztec >= 7 file with a Camera Mode keeps binning 1 although the camera may have
#: binned; ``safe_loader`` patches the reader for the header key it does carry, and the shipped env
#: (the shipped runtime) is newer.
#:
#: ``detector_insertion_mm`` has no kikuchipy default: it is read from the header and is ``None`` when it has
#: none. The formats whose loader keeps no header at all (``ebsp``, ``up``, ``nordif``, ``unified_fallback``)
#: are listed for it too, so that a file of such a format can never state it.
_UNSTATED = {
    "ebsp": ("binning", "camera_tilt", "sample_tilt", "detector_insertion_mm"),
    "up": ("binning", "camera_tilt", "sample_tilt", "detector_insertion_mm"),
    "nordif": ("binning", "detector_insertion_mm"),
    "edax_h5": ("binning",),
    "bruker_h5": ("binning",),
    "unified_fallback": ("binning", "camera_tilt", "sample_tilt", "detector_insertion_mm"),
}

_DIR_OVERRIDE: Optional[Path] = None


class CalibrationError(ValueError):
    """A calibration request that cannot be served. ``code`` is what the interface translates on."""

    def __init__(self, code: str, message: str, **params):
        super().__init__(message)
        self.code = code
        self.params = params


# ------------------------------------------------------------------ where it lives

def calibrations_dir() -> Path:
    if _DIR_OVERRIDE is not None:
        return _DIR_OVERRIDE
    env = os.environ.get(ENV_DIR)
    if env:
        return Path(env)
    from path_utils import get_local_database_path
    return get_local_database_path() / _DIRNAME


def set_calibrations_dir_for_test(path: Optional[Path]) -> None:
    """Test-only: redirect the folder. ``None`` restores. ``tests/conftest.py`` fails any test that
    writes under ``Database/``, and rightly: a test must not add calibrations to a user's library."""
    global _DIR_OVERRIDE
    _DIR_OVERRIDE = Path(path) if path is not None else None


#: On Windows a file that is open cannot be replaced, and one that is being replaced cannot be opened
#: (PermissionError, WinError 5). Both are over in milliseconds: try again shortly, then give up.
_RETRIES = 10
_RETRY_PAUSE_S = 0.02


def _retrying(action):
    """``action()``, repeated up to ``_RETRIES`` times while it raises PermissionError."""
    for attempt in range(_RETRIES):
        try:
            return action()
        except PermissionError:
            if attempt == _RETRIES - 1:
                raise
            time.sleep(_RETRY_PAUSE_S)


def _write_atomic(path: Path, text: str) -> None:
    """Temp file beside the target, then replace. ``newline="\\n"``: without it Python writes CRLF on
    Windows and a library copied between machines differs on every line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".calibration_", suffix=".json.tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        _retrying(lambda: os.replace(tmp, path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


#: One lock for the whole folder (every write is read-modify-write; see phase_collections).
_WRITE_LOCK = threading.RLock()


def _serialised(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _WRITE_LOCK:
            return fn(*args, **kwargs)
    wrapper._serialised = True          # noqa: SLF001 - read by the guard test
    return wrapper


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ reading and validating

def _valid_pc(value) -> bool:
    """Three finite numbers in a list, a tuple or an array. A text ("123" would be read as three digits), a record
    or a list of texts or truth values is no PC."""
    if isinstance(value, (str, bytes, dict)):
        return False
    try:
        items = list(value)
        if len(items) != 3 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real) for v in items):
            return False
        return all(math.isfinite(float(v)) for v in items)
    except (TypeError, ValueError, OverflowError):          # OverflowError: a whole number beyond a float
        return False


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value) -> bool:
    """A number that is neither nan, an infinity (``1e400`` in a JSON file is one) nor a whole number too large
    for a float (``math.isfinite(10**400)`` raises)."""
    try:
        return _number(value) and math.isfinite(value)
    except OverflowError:
        return False


def _validate(doc: Any) -> dict:
    """The document if it is a calibration this build can use, else ``CalibrationError``: the code
    ``newerSchema`` / ``notACalibration`` for what is missing, ``unreadable`` for a key whose content has
    the wrong type (the keys are there, so a later code would otherwise stumble over it)."""
    if not isinstance(doc, dict):
        raise CalibrationError("notACalibration", "not a calibration file")
    try:
        newer = int(doc.get("schema", 0)) > SCHEMA
    except (TypeError, ValueError, OverflowError) as exc:
        raise CalibrationError("unreadable", f"'schema' is not a number: {doc.get('schema')!r}") from exc
    if newer:
        raise CalibrationError("newerSchema", "written by a newer Orienta; left untouched",
                               schema=doc.get("schema"))
    for key in ("id", "name", "versions", "features", "members"):
        if key not in doc:
            raise CalibrationError("notACalibration", f"missing {key!r}")
    versions = doc["versions"]
    if not isinstance(versions, list) or any(not isinstance(v, dict) for v in versions):
        raise CalibrationError("unreadable", "'versions' are a list of versions")
    if not versions or any(not _valid_pc(v.get("pc")) for v in versions):
        raise CalibrationError("notACalibration", "the versions are not numbered 1..n with a PC each")
    try:
        numbers = [int(v.get("version", 0)) for v in versions]
    except (TypeError, ValueError, OverflowError) as exc:
        raise CalibrationError("unreadable", "a version number is not a number") from exc
    if numbers != list(range(1, len(versions) + 1)):
        raise CalibrationError("notACalibration", "the versions are not numbered 1..n with a PC each")
    _validate_types(doc)
    return doc


def _validate_types(doc: dict) -> None:
    """The rest of what the readers rely on: ``unreadable`` if a key holds the wrong type."""
    features, members = doc["features"], doc["members"]
    shape = features.get("pattern_shape") if isinstance(features, dict) else None
    problems = []
    if not isinstance(doc["id"], str) or not isinstance(doc["name"], str):
        problems.append("'id' and 'name' are text")
    if any(not (v.get("origin") is None or isinstance(v.get("origin"), dict)) for v in doc["versions"]):
        problems.append("'origin' of a version is a record or nothing")
    elif any(isinstance(v.get("origin"), dict) and not (v["origin"].get("refinement") is None
                                                        or isinstance(v["origin"].get("refinement"), dict))
             for v in doc["versions"]):
        problems.append("'refinement' of an origin is a record or nothing")
    if not (doc.get("created") is None or isinstance(doc["created"], str)):
        problems.append("'created' is text")
    if (not isinstance(shape, list) or len(shape) != 2
            or not all(_finite(v) for v in shape)
            or any(features.get(k) is not None and not _finite(features[k]) for k in SETTING_KEYS)):
        problems.append("'features' hold a pattern shape and numbers or nothing")
    if (not isinstance(members, list)
            or any(not isinstance(m, dict) or not isinstance(m.get("name"), str)
                   or not isinstance(m.get("header_id"), str)
                   or not _finite(m.get("size")) or not _finite(m.get("version")) for m in members)):
        problems.append("'members' are files with a name, size, header id and version")
    if problems:
        raise CalibrationError("unreadable", "unexpected content: " + "; ".join(problems))


def _path_of(cal_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,60}", str(cal_id)):
        raise CalibrationError("unknownCalibration", "no such calibration", id=str(cal_id))
    return calibrations_dir() / f"{cal_id}.json"


def load_all() -> Tuple[List[dict], List[dict]]:
    """``(calibrations, problems)``: every readable calibration, name order; a file that cannot be
    used (broken, or from a newer Orienta) is reported, never skipped silently and never rewritten."""
    found: List[dict] = []
    problems: List[dict] = []
    folder = calibrations_dir()
    try:
        files = sorted(p for p in folder.glob("*.json"))
    except OSError:
        files = []
    for path in files:
        try:
            found.append(_read_file(path))
        except FileNotFoundError:
            continue                       # deleted between the listing and the read: nothing to report
        except CalibrationError as exc:
            problems.append({"file": path.name, "code": exc.code, "message": str(exc)})
    found.sort(key=lambda d: (str(d["name"]).casefold(), d["id"]))
    return found, problems


def _read_file(path: Path) -> dict:
    """The validated calibration in ``path``. ``FileNotFoundError`` if there is none, else ``CalibrationError``
    with the code of what is wrong with the file. Under the folder lock, so that in one process a read never
    meets the replace of the same file; the retry is for another process."""
    with _WRITE_LOCK:
        try:
            text = _retrying(lambda: path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise
        except (OSError, ValueError) as exc:               # ValueError: not UTF-8
            raise CalibrationError("unreadable", f"cannot read {path.name}: {exc}") from exc
        try:
            doc = json.loads(text)
        except (ValueError, RecursionError) as exc:        # RecursionError: nested deeper than the parser goes
            raise CalibrationError("unreadable", f"{path.name} is not JSON: {exc}") from exc
        doc = _validate(doc)
        if doc["id"] != path.stem:
            # an Explorer copy ("<id> - Copy.json") or a hand-renamed file: writes go to <id>.json, so it
            # would be listed twice and edited under the other file's name
            raise CalibrationError("idMismatch", f"the id {doc['id']!r} is not the file name {path.stem!r}",
                                   id=doc["id"])
        return doc


def get(cal_id: str) -> Optional[dict]:
    """The calibration, ``None`` if there is none; a file that cannot be used raises ``CalibrationError``
    with the code ``load_all`` reports for it."""
    try:
        return _read_file(_path_of(cal_id))
    except FileNotFoundError:
        return None


def _require(cal_id: str) -> dict:
    cal = get(cal_id)
    if cal is None:
        raise CalibrationError("unknownCalibration", "no such calibration", id=str(cal_id))
    return cal


def latest(cal: dict) -> dict:
    return cal["versions"][-1]


def version_pc(cal: dict, version: int) -> List[float]:
    for entry in cal["versions"]:
        if int(entry["version"]) == int(version):
            return [float(v) for v in entry["pc"]]
    raise CalibrationError("unknownVersion", "no such version", version=int(version))


def version_facts(entry: dict) -> dict:
    """What an answer says about the version it shows: when that version was made (``version_created``; the
    calibration's own ``created`` is the date of version 1), the NAME of the file it was made on
    (``source_name``) and how good the search result it came from was (``quality``, see
    ``pc_search.quality_of``). A fact the version does not hold (one saved before it was recorded, or from a typed
    PC) is ``None``."""
    origin = entry.get("origin") or {}
    return {"version_created": entry.get("created"), "source_name": origin.get("source_name"),
            "quality": origin.get("quality")}


def quality_of_version(calibration: Optional[dict], version: int) -> Optional[dict]:
    """The quality stored with ``version`` of ``calibration``, or ``None``."""
    if calibration is None:
        return None
    for entry in calibration["versions"]:
        if int(entry["version"]) == int(version):
            return version_facts(entry)["quality"]
    return None


def summary(cal: dict) -> dict:
    """What the interface lists: no history of members' fingerprints, just their number. ``version`` and ``pc`` are
    the latest version's; ``version_created``, ``source_name`` and ``quality`` are that version's too."""
    top = latest(cal)
    return {"id": cal["id"], "name": cal["name"], "created": cal.get("created"),
            "author": cal.get("author"), "version": int(top["version"]), "pc": list(top["pc"]),
            "features": cal["features"], "origin": top.get("origin"), **version_facts(top),
            "n_versions": len(cal["versions"]), "n_members": len(cal["members"])}


# ------------------------------------------------------------------ writing

def _save(cal: dict) -> None:
    _write_atomic(_path_of(cal["id"]), json.dumps(cal, indent=2, ensure_ascii=False) + "\n")


def _mint_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(name).casefold()).strip("-")[:24].strip("-") or "pc"
    taken = {p.stem for p in calibrations_dir().glob("*.json")} if calibrations_dir().is_dir() else set()
    while True:
        cal_id = f"{slug}-{uuid.uuid4().hex[:6]}"
        if cal_id not in taken:
            return cal_id


def _check_text(text: str, what: str, limit: int) -> None:
    """``name`` / ``author`` land in a UTF-8 file: a lone surrogate (``"\\ud800"`` in a JSON request) cannot be
    written, and a text of megabytes would be listed with every calibration."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise CalibrationError("badName", f"the {what} holds a character that cannot be saved", field=what) from None
    if len(text) > limit:
        raise CalibrationError("nameTooLong", f"the {what} is longer than {limit} characters",
                               field=what, limit=limit)


def _check_name(name: str, *, except_id: Optional[str] = None) -> str:
    text = " ".join(str(name or "").split())
    if not text:
        raise CalibrationError("needsName", "a calibration needs a name")
    _check_text(text, "name", MAX_NAME_CHARS)
    others, _ = load_all()
    if any(c["name"].casefold() == text.casefold() and c["id"] != except_id for c in others):
        raise CalibrationError("nameTaken", "a calibration with this name exists", name=text)
    return text


def _stated(value):
    """``value`` if it is a finite number, else ``None``: a nan, an infinity or a text is no setting. (A nan
    would count as 'within tolerance' in a comparison, ``abs(nan) > tol`` being false, and is no JSON.)"""
    try:
        return value if math.isfinite(float(value)) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _base_name(name) -> str:
    """The file name of ``name``: no folder, whatever separator it came with (a path must not land in a file
    that travels with the library)."""
    return re.split(r"[\\/]", str(name).rstrip("\\/"))[-1]


def real_file_name(file_key, *given) -> Optional[str]:
    """The NAME of the file ``file_key`` stands for, as it is written on disk.

    A file key is a canonical path and on Windows it is lower case, so its last part does not read like the
    file. The name is taken from the folder listing (the entry that matches ignoring case; an exact match
    wins on a system that tells two apart); if the folder cannot be read, from the first of ``given`` (paths
    or names the caller holds, e.g. the one the file was loaded with) that names the same file; else from the
    key itself. ``None`` for no key."""
    if not file_key:
        return None
    base = _base_name(file_key)
    try:
        names = os.listdir(os.path.dirname(str(file_key)) or ".")
    except OSError:
        names = []
    if base in names:
        return base
    for n in names:
        if n.casefold() == base.casefold():
            return n
    for g in given:
        if g and _base_name(g).casefold() == base.casefold():
            return _base_name(g)
    return base


def _clean_features(features: dict) -> dict:
    out = {key: features.get(key) for key in FEATURE_KEYS}
    for key in SETTING_KEYS:
        out[key] = _stated(out[key])
    shape = out["pattern_shape"]
    try:                                   # a nan, an infinity or a text in the shape is no shape
        whole = ([int(v) for v in shape] if isinstance(shape, (list, tuple)) and len(shape) == 2 else None)
    except (TypeError, ValueError, OverflowError):
        whole = None
    if whole is None or any(v < 1 for v in whole):
        raise CalibrationError("needsFeatures", "a calibration needs the pattern shape of its files")
    out["pattern_shape"] = whole
    out["hints"] = dict(features.get("hints") or {})
    return out


def _version_entry(number: int, pc, origin: Optional[dict]) -> dict:
    if not _valid_pc(pc):
        raise CalibrationError("badPc", "a pattern centre is three finite numbers")
    origin = dict(origin or {})
    if origin.get("dataset") is not None:
        origin["dataset"] = _base_name(origin["dataset"])
    # the file the version was made on: its name, never a path (the file travels with the library)
    origin["source_name"] = _base_name(origin["source_name"]) if origin.get("source_name") else None
    return {"version": int(number), "created": _now(), "pc": [float(v) for v in pc], "origin": origin}


@_serialised
def create(*, name: str, pc, features: dict, origin: Optional[dict] = None, author: str = "",
           member: Optional[dict] = None) -> dict:
    """A new calibration at version 1. ``member`` is the fingerprint of the file it was made on."""
    text = _check_name(name)
    who = str(author).strip() or "unknown"
    _check_text(who, "author", MAX_AUTHOR_CHARS)
    cal = {
        "schema": SCHEMA, "id": _mint_id(text), "name": text, "created": _now(),
        "author": who,
        "features": _clean_features(features),
        "versions": [_version_entry(1, pc, origin)],
        "members": [],
    }
    if member:
        cal["members"].append(_member_entry(member, 1))
    _save(cal)
    return cal


@_serialised
def add_version(cal_id: str, *, pc, origin: Optional[dict] = None) -> dict:
    """The next PC of a calibration. The earlier versions are kept as they were."""
    cal = _require(cal_id)
    cal["versions"].append(_version_entry(len(cal["versions"]) + 1, pc, origin))
    _save(cal)
    return cal


def _member_entry(fingerprint: dict, version: int) -> dict:
    return {"name": _base_name(fingerprint["name"]), "size": int(fingerprint["size"]),
            "header_id": str(fingerprint["header_id"]), "version": int(version), "applied": _now()}


def same_file(a: dict, b: dict) -> bool:
    """Name (without any folder), size and header id all equal."""
    return (_base_name(a.get("name")).casefold() == _base_name(b.get("name")).casefold()
            and int(a.get("size", -1)) == int(b.get("size", -2))
            and a.get("header_id") == b.get("header_id"))


@_serialised
def set_member(cal_id: str, fingerprint: dict, version: int) -> dict:
    """File ``fingerprint`` uses ``version`` of the calibration (a file is a member once)."""
    cal = _require(cal_id)
    cal["members"] = [m for m in cal["members"] if not same_file(m, fingerprint)]
    cal["members"].append(_member_entry(fingerprint, version))
    _save(cal)
    return cal


@_serialised
def rename(cal_id: str, name: str) -> dict:
    cal = _require(cal_id)
    cal["name"] = _check_name(name, except_id=cal_id)
    _save(cal)
    return cal


@_serialised
def delete(cal_id: str) -> bool:
    """Removes the file; a file of a newer Orienta or one that cannot be read is refused with its code
    (``newerSchema`` / ``unreadable``), as every other writer does: it is 'left untouched'."""
    path = _path_of(cal_id)
    try:
        _read_file(path)
    except FileNotFoundError:
        return False
    except CalibrationError as exc:
        if exc.code in ("newerSchema", "unreadable"):
            raise
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


# ------------------------------------------------------------------ files and their settings

def fingerprint_of(path) -> dict:
    """How a file is recognised later, without its path: name, size and a header id (SHA-256 of
    its first 64 KiB). Two files with the same name and size whose first 64 KiB are identical are
    taken for one; such files have the same settings in practice."""
    p = Path(path)
    try:
        size = p.stat().st_size
        with open(p, "rb") as handle:
            head = handle.read(HEADER_BYTES)
    except OSError as exc:
        raise CalibrationError("fileUnreadable", f"cannot read {p.name}: {exc.strerror or exc}") from exc
    return {"name": p.name, "size": int(size), "header_id": hashlib.sha256(head).hexdigest()}


def _float_or_none(value) -> Optional[float]:
    try:
        out = float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None
    return out if math.isfinite(out) else None


def _meta_of(signal, path, tree="metadata"):
    try:
        return getattr(signal, tree).get_item(path, None)
    except Exception:  # noqa: BLE001 - a stand-in signal without a metadata tree
        return None


def _format_of(signal) -> Optional[str]:
    """The key of ``_UNSTATED`` that names the file format of ``signal``, or ``None``.

    By the file name the reader recorded (it may carry the folder, with either separator) for the formats
    that have an extension of their own. The h5ebsd readers of EDAX and Bruker record the name WITHOUT its
    extension (``_h5ebsd.get_metadata_filename_title``), so those are told by ``manufacturer`` in the
    original metadata, which the readers write (kikuchipy lower-cases it: ``edax``, ``bruker nano``; the
    hexagonal-grid EDAX loader writes ``EDAX``)."""
    name = _meta_of(signal, "General.original_filename")
    if isinstance(name, str):
        base = re.split(r"[\\/]", name)[-1].lower()
        if base.endswith(".ebsp"):
            return "ebsp"
        if base.endswith((".up1", ".up2")):
            return "up"
        if base.endswith(".dat"):              # the only reader that takes a .dat is the NORDIF one
            return "nordif"
    manufacturer = _meta_of(signal, "manufacturer", "original_metadata")
    if isinstance(manufacturer, str):
        manufacturer = manufacturer.casefold()
        if manufacturer.startswith("edax"):
            return "edax_h5"
        if manufacturer.startswith("bruker"):
            return "bruker_h5"
    if not (isinstance(name, str) and name) and not (isinstance(manufacturer, str) and manufacturer):
        return "unified_fallback"             # no reader recorded where this signal came from
    return None


#: File extensions whose files state their PC in a vendor's convention, for the loaders that record no format
#: key (Oxford ``.h5oina`` is read by kikuchipy with the name WITHOUT its extension, the unified-loader fallback
#: records no name at all): ``pc_convention.CONVENTIONS``.
_CONVENTION_BY_EXTENSION = {".h5oina": "oxford", ".ebsp": "oxford",
                            ".ang": "tsl", ".osc": "tsl", ".up1": "tsl", ".up2": "tsl"}
#: The format key (see ``_format_of``) -> the convention of its numbers. NORDIF has no entry: its PC is
#: kikuchipy's own (Bruker's), but the file does not say so.
_CONVENTION_BY_FORMAT = {"ebsp": "oxford", "up": "tsl", "edax_h5": "tsl", "bruker_h5": "bruker"}


def convention_of(signal, file_key=None) -> Tuple[str, bool]:
    """``(convention, known)``: the convention the file's own vendor states a PC in, as a name of
    ``pc_convention.CONVENTIONS``, and whether the format says so. ``("bruker", False)`` when it does not (the
    fallback is Orienta's own convention, not a statement about the file).

    By the format key ``_format_of`` knows where there is one, then by the maker the reader recorded (Oxford
    ``.h5oina`` has no key of its own: kikuchipy writes ``manufacturer = "oxford instruments"``; a file written by
    kikuchipy's h5ebsd writer holds Bruker numbers), then by the extension of the file's name or path."""
    fmt = _format_of(signal)
    if fmt in _CONVENTION_BY_FORMAT:
        return _CONVENTION_BY_FORMAT[fmt], True
    maker = _meta_of(signal, "manufacturer", "original_metadata")
    maker = maker.casefold() if isinstance(maker, str) else ""
    if maker.startswith("oxford"):
        return "oxford", True
    if maker.startswith("kikuchipy"):
        return "bruker", True
    for name in (_meta_of(signal, "General.original_filename"), file_key):
        if isinstance(name, str):
            known = _CONVENTION_BY_EXTENSION.get(Path(re.split(r"[\\/]", name)[-1]).suffix.lower())
            if known:
                return known, True
    return "bruker", False


def settings_stated(signal) -> set:
    """The settings of ``SETTING_KEYS`` that the file's FORMAT can carry (``_UNSTATED`` lists those it cannot).
    ``working_distance_mm`` is read from the header like the two new ones and is not in ``_UNSTATED``; it is
    left out here as it always was."""
    return (set(SETTING_KEYS) - {"working_distance_mm"}) - set(_UNSTATED.get(_format_of(signal), ()))


def stated_settings(features: dict) -> List[str]:
    """The settings of ``SETTING_KEYS`` that ``features`` states (those that are not ``None``)."""
    return [key for key in SETTING_KEYS if (features or {}).get(key) is not None]


_DATE_PART = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})(?:[T ]|\s*$)")


def _oxford_acquisition_date(signal) -> Optional[str]:
    """The day an Oxford ``.h5oina`` was acquired (``YYYY-MM-DD``), or ``None``.

    kikuchipy's Oxford reader leaves the header's "Acquisition Date" (``2026-01-19T09:04:03``) in the original
    metadata and sets no ``General.date``. Only the date part is kept. Other makers state none there (EDAX
    files carry no acquisition date), and a value that is not a date, or not a day of the calendar
    ("2026-13-45"), is not guessed at."""
    maker = _meta_of(signal, "manufacturer", "original_metadata")
    if not (isinstance(maker, str) and maker.casefold().startswith("oxford")):
        return None
    stated = _meta_of(signal, "Acquisition Date", "original_metadata")
    found = _DATE_PART.match(stated) if isinstance(stated, str) else None
    if not found:
        return None
    try:
        _calendar_date.fromisoformat(found.group(1))
    except ValueError:
        return None
    return found.group(1)


def features_of_signal(signal) -> dict:
    """The acquisition settings of a loaded signal (``FEATURE_KEYS``) and the ``hints`` shown next to
    a proposal. A setting the file does not state is ``None``: one the header lacks (a working distance),
    and one the file's format does not carry at all (``settings_stated``), for which kikuchipy's detector
    holds a default that nobody measured."""
    det = signal.detector
    stated = settings_stated(signal)

    def meta(path):
        return _meta_of(signal, path)

    hints = {}
    for key, path in (("date", "General.date"), ("time", "General.time"),
                      ("microscope", "Acquisition_instrument.SEM.microscope"),
                      ("beam_energy_kv", "Acquisition_instrument.SEM.beam_energy"),
                      ("magnification", "Acquisition_instrument.SEM.magnification")):
        value = meta(path)
        if value is not None:
            hints[key] = value if isinstance(value, str) else _float_or_none(value)
    if "date" not in hints:
        date = _oxford_acquisition_date(signal)
        if date is not None:
            hints["date"] = date

    def setting(key, value, kind):
        value = _stated(value) if key in stated else None
        return None if value is None else kind(value)

    return {
        "pattern_shape": [int(v) for v in det.shape],
        "binning": setting("binning", det.binning, int),
        "camera_tilt": setting("camera_tilt", det.tilt, float),
        "sample_tilt": setting("sample_tilt", det.sample_tilt, float),
        "working_distance_mm": _float_or_none(meta("Acquisition_instrument.SEM.working_distance")),
        # stated in the header or not stated; no default stands in (Oxford states the insertion, in mm)
        "detector_insertion_mm": (_float_or_none(_meta_of(signal, INSERTION_KEY, "original_metadata"))
                                  if "detector_insertion_mm" in stated else None),
        "hints": {k: v for k, v in hints.items() if v is not None},
    }


def compare_features(calibration: dict, other: dict) -> dict:
    """Do two sets of settings count as the same? ``{match, differences, unknown, checked, unverified}``.

    A difference is a setting both sides state and that differs beyond its tolerance. A setting
    one side does not state (a header without a working distance, a format without tilts) is
    ``unknown``: it is shown, it does not make the files differ and it is not a silent match.
    ``checked`` lists every setting that was compared: the pattern shape (always) and the settings that
    both sides state; ``unverified`` is true when the pattern shape is all that could be compared."""
    differences: List[dict] = []
    unknown: List[str] = []
    checked: List[str] = ["pattern_shape"]

    def add(key, mine, theirs, tolerance=None):
        differences.append({"key": key, "calibration": mine, "file": theirs, "tolerance": tolerance})

    if list(calibration.get("pattern_shape") or []) != list(other.get("pattern_shape") or []):
        add("pattern_shape", calibration.get("pattern_shape"), other.get("pattern_shape"))
    for key, tol in (("binning", None), ("camera_tilt", TOL_TILT_DEG), ("sample_tilt", TOL_TILT_DEG),
                     ("working_distance_mm", TOL_WD_MM), ("detector_insertion_mm", TOL_WD_MM)):
        mine, theirs = _stated(calibration.get(key)), _stated(other.get(key))
        if mine is None or theirs is None:
            unknown.append(key)
            continue
        checked.append(key)
        if tol is None:
            if int(mine) != int(theirs):
                add(key, mine, theirs)
        elif abs(float(mine) - float(theirs)) > tol + 1e-9:
            add(key, float(mine), float(theirs), tol)
    return {"match": not differences, "differences": differences, "unknown": unknown,
            "checked": checked, "unverified": checked == ["pattern_shape"]}


def settings_of(features: Optional[dict]) -> dict:
    """The settings of ``features`` as an answer shows them side by side: the pattern shape and every key of
    ``SETTING_KEYS``, each ``None`` where the file does not state it (the page writes "not stated")."""
    features = features or {}
    shape = features.get("pattern_shape")
    out = {"pattern_shape": list(shape) if isinstance(shape, (list, tuple)) and len(shape) == 2 else None}
    for key in SETTING_KEYS:
        out[key] = _stated(features.get(key))
    return out


def suggest(features: dict, fingerprint: Optional[dict] = None, replaces: Optional[dict] = None,
            only_fitting: bool = True) -> List[dict]:
    """The calibrations that fit a file, best first: those the file was a member of, then those whose
    settings match. Read-only; the caller decides what to do with the answer.

    ``only_fitting=False`` lists every calibration with its comparison (the "Take PC from..." window shows the ones
    that do not fit too, marked as such). Each answer carries ``settings`` (``{"file": ..., "calibration": ...}``,
    see ``settings_of``): the values the settings table shows, also the ones that agree.

    Every offer says what taking it would change: ``replaces`` is ``{"pc", "kind"}`` of the PC the file uses now
    (``pc_in_use.pc_in_use_of``; ``None`` when the caller gave none), ``file_date`` the date of the file
    (``hints.date``) and ``version_created`` the date of the offered version, with its ``source_name`` and
    ``quality``. The size of the change in pixels is for the page to work out."""
    out: List[dict] = []
    calibrations, _ = load_all()
    for cal in calibrations:
        compared = compare_features(cal["features"], features)
        member = next((m for m in cal["members"] if fingerprint and same_file(m, fingerprint)), None)
        if only_fitting and not member and (not compared["match"] or compared["unverified"]):
            continue                       # a fit on the pattern shape alone says nothing: not suggested
        top = latest(cal)
        out.append({
            "id": cal["id"], "name": cal["name"], "version": int(top["version"]),
            "pc": list(top["pc"]), "via_member": member is not None,
            "member_version": None if member is None else int(member["version"]),
            "match": compared["match"] and not compared["unverified"],
            "differences": compared["differences"], "checked": compared["checked"],
            "unknown": compared["unknown"], "unverified": compared["unverified"],
            "settings": {"file": settings_of(features), "calibration": settings_of(cal["features"])},
            "hints": {"calibration": cal["features"].get("hints", {}),
                      "file": features.get("hints", {})},
            "created": cal.get("created"),
            **version_facts(top),
            "file_date": (features.get("hints") or {}).get("date"),
            "replaces": None if replaces is None else {"pc": replaces.get("pc"), "kind": replaces.get("kind")},
        })
    out.sort(key=lambda s: s["created"] or "", reverse=True)                    # newest first ...
    out.sort(key=lambda s: (not s["via_member"], not s["match"]))               # ... members, then matches
    return out
