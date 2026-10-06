"""Describe, and check, one phase file the user points at by path.

The Indexing page lists the phases of the local library. A phase that lives
elsewhere is added by typing or pasting its path (the only way in a plain
browser, where there is no native file dialog) or by browsing to it (Electron).
This module is the server-side check behind both: it says what is wrong with a
path in terms the user can act on, and for a good one returns the same record
the library listing produces, so the page treats it like any library phase.

Each failure is a :class:`PhasePathError` with a stable ``code`` and its
``params``, so the page can word it in the user's language; ``message`` is the
English sentence for the log and as a fallback.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Dict

#: Which file a phase brings to each indexing method.
METHOD_EXTENSIONS = {
    "hough": (".cif",),
    "dictionary": (".h5", ".hdf5"),
    "spherical": (".sht",),
}

_FILE_TYPE = {"hough": "cif", "dictionary": "master", "spherical": "sht"}

#: A .sht header is 40 bytes plus the first block; anything shorter is cut off.
_SHT_MIN_BYTES = 44


class PhasePathError(ValueError):
    """A path that cannot be used as a phase for the chosen method."""

    def __init__(self, code: str, message: str, **params):
        super().__init__(message)
        self.code = code
        self.message = message
        self.params = params

    def as_detail(self) -> dict:
        return {"code": self.code, "message": self.message, "params": self.params}


def _clean(raw) -> str:
    """The path as typed: no surrounding spaces or quotes ("Copy as path")."""
    text = str(raw or "").strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def _check_cif(path: Path) -> None:
    from ebsd_utils import sanitize_cif
    from orix.crystal_map import Phase

    try:
        phase = Phase.from_cif(sanitize_cif(str(path)))
    except Exception as exc:  # diffpy raises several unrelated types
        raise PhasePathError(
            "unreadable",
            f"{path.name} could not be read as a crystal structure: {exc}",
            path=str(path), reason=str(exc)[:300],
        ) from exc
    if len(phase.structure) == 0:
        raise PhasePathError(
            "unreadable",
            f"{path.name} has no atom sites, so it is not a usable phase.",
            path=str(path), reason="no atom sites",
        )


def _check_sht(path: Path) -> None:
    try:
        with open(path, "rb") as fh:
            head = fh.read(_SHT_MIN_BYTES)
    except OSError as exc:
        raise PhasePathError(
            "unreadable", f"{path.name} could not be opened: {exc}",
            path=str(path), reason=str(exc)[:300]) from exc
    if len(head) < _SHT_MIN_BYTES:
        raise PhasePathError(
            "unreadable", f"{path.name} is too short to be an EMsoft .sht file.",
            path=str(path), reason="truncated")
    if head[:4] not in (b"*sht", b"*SHT"):
        raise PhasePathError(
            "unreadable", f"{path.name} is not an EMsoft .sht file (wrong header).",
            path=str(path), reason="bad header")
    endian = "<" if head[:4] == b"*sht" else ">"
    version = struct.unpack_from(f"{endian}bb", head, 4)
    if version != (1, 1):
        raise PhasePathError(
            "unreadable",
            f"{path.name} is .sht version {version[0]}.{version[1]}; only 1.1 is supported.",
            path=str(path), reason="version")


def _check_master(path: Path) -> None:
    import h5py

    try:
        if not h5py.is_hdf5(str(path)):
            raise PhasePathError(
                "unreadable", f"{path.name} is not an HDF5 file.",
                path=str(path), reason="not hdf5")
        with h5py.File(str(path), "r") as fh:
            is_master = "EMData/EBSDmaster" in fh
    except PhasePathError:
        raise
    except Exception as exc:
        raise PhasePathError(
            "unreadable", f"{path.name} could not be opened: {exc}",
            path=str(path), reason=str(exc)[:300]) from exc
    if not is_master:
        raise PhasePathError(
            "not_a_phase",
            f"{path.name} is an HDF5 file but not an EMsoft master pattern "
            "(no EMData/EBSDmaster group).",
            path=str(path))


_CHECKS = {"hough": _check_cif, "spherical": _check_sht, "dictionary": _check_master}


def inspect_phase_path(method: str, raw_path: str) -> Dict:
    """Validate ``raw_path`` as a phase file for ``method``; return its record.

    Raises :class:`PhasePathError`. The record has the keys of a library
    listing entry plus ``user_added`` and ``in_library``.
    """
    method = str(method or "").lower()
    if method not in METHOD_EXTENSIONS:
        raise PhasePathError("unknown_method", f"Unknown indexing method: {method!r}",
                             method=method)

    text = _clean(raw_path)
    if not text:
        raise PhasePathError("empty", "Enter the path of a phase file.")
    path = Path(os.path.expanduser(text))
    if not path.is_absolute():
        raise PhasePathError(
            "not_absolute",
            f"{text!r} is not a full path. Give the complete path, starting at the "
            "drive or root folder.",
            path=text)
    if not path.exists():
        raise PhasePathError("not_found", f"No such file: {path}", path=str(path))
    if not path.is_file():
        raise PhasePathError("not_a_file", f"{path} is a folder, not a file.",
                             path=str(path))

    allowed = METHOD_EXTENSIONS[method]
    if path.suffix.lower() not in allowed:
        raise PhasePathError(
            "wrong_extension",
            f"{path.name} has the extension {path.suffix or '(none)'}; "
            f"{method} indexing needs a {allowed[0]} file.",
            path=str(path), got=path.suffix, expected=allowed[0], method=method)

    _CHECKS[method](path)

    # `path` is a Path while the metadata is read (as in library discovery,
    # which turns it into a string only at the very end).
    entry: Dict = {"path": path, "filename": path.name, "material": "-",
                   "file_type": _FILE_TYPE[method]}
    from indexing_controller import enrich_phase_entries
    from path_utils import get_local_database_path

    db_root = get_local_database_path()
    enrich_phase_entries([entry], db_root)
    entry["path"] = str(path)
    entry["user_added"] = True
    try:
        entry["in_library"] = Path(db_root).resolve() in path.resolve().parents
    except OSError:
        entry["in_library"] = False
    return entry
