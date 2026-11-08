"""EDAX APEX metadata files (.h5) and the pattern files (.up1/.up2) they belong to.

APEX writes one ``.h5`` for a session. Below ``<top>/<sample>/Area <n>/`` every
measurement is a group ``OIM Map <m>``; it holds the scan grid, the step sizes,
the pattern centre, the camera angles, the phases and the per-point results
(orientation, IQ, CI, phase), but NO patterns. The patterns of one measurement
are in a ``.up1``/``.up2`` file whose header carries the measurement's ``Id`` as
a 16-byte GUID at byte 42 (little-endian field order). That GUID, not the file
name, is what ties the two together.

Layout read here (names only; see tests/apex_helpers.py for a synthetic file of
the same shape)::

    <measurement>.attrs['Id']                 text GUID
    <measurement>.attrs['Name']               'OIM Map 1'
    <measurement>.attrs['ScanEndTime']        int64[1], .NET ticks (100 ns since 0001-01-01), the end
                                              of the scan; no time zone is stored
    <measurement>/Sample/{Grid Type, Number Of Columns, Number Of Rows, Step X, Step Y,
                          Sample Tilt}
    <measurement>/Camera/{Elevation Angle, Azimuthal Angle}
    <measurement>/Column/SemKV
    <measurement>/EBSD/ANG/HEADER/{Pattern Width, Pattern Height}   optional: the pattern size in pixels
    <measurement>/EBSD/ANG/HEADER/Pattern Center Calibration/{X-Star, Y-Star, Z-Star}
    <measurement>/EBSD/ANG/HEADER/Phase/<n>/{Material Name, Formula, Symmetry,
                                              Laue Group, Point Group}
    <measurement>/EBSD/ANG/DATA/DATA          compound, one record per scan point
    <measurement>/SEM-PRIAS Images/DATA/SEM  overview image
"""
from __future__ import annotations

import io
import logging
import re
import struct
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_MEASUREMENT = re.compile(r"(?:^|/)Area [^/]+/OIM Map [^/]+$")
_UP_SUFFIXES = (".up1", ".up2")
#: The GUID sits at byte 42 of an UP header whose pattern data starts at byte 58 or later.
_GUID_AT, _GUID_END = 42, 58

#: Every code the APEX routes, listings and loaders can emit (``ApexError.code``, a ``problem`` or
#: ``measurement_problem`` of the listing, the reason a stored pairing was dropped). The interface
#: has one translation per code; ``tests/test_apex_dialog_backend.py`` holds this tuple against the
#: ``"apex..."`` literals of the source, so a code cannot be added without being listed here.
APEX_CODES = (
    "apexBadId",                  # a measurement's Id attribute is no GUID
    "apexBadPath",                # a path with a NUL character
    "apexFileNotFound",           # /inspect: the .h5/.hdf5 named does not exist
    "apexFileUnreadable",         # /inspect: an HDF5/.h5 file that cannot be opened (e.g. held by APEX)
    "apexFolderUnreadable",       # a folder that cannot be listed (missing, a file, empty, no permission)
    "apexGridMismatch",           # table != grid, or a UP header's hexagonal flag contradicts the grid type
    "apexGuidMismatch",           # the pattern file names another measurement
    "apexInvalidGrid",            # a square grid with a step that is not positive
    "apexMeasurementNotFound",    # the file has no measurement with that id
    "apexNeedsMeasurement",       # an APEX metadata file was loaded as if it held patterns
    "apexNoMeasurement",          # the file holds no OIM Map
    "apexNoPatternFile",          # no pattern file names the measurement
    "apexNoPreview",              # the preview asked for does not exist
    "apexPairingLost",            # a result's pairing names no metadata file any more, and none is remembered
    "apexPatternCountMismatch",   # the pattern file holds another number of patterns than the grid
    "apexPatternFileUnreadable",  # the pattern file is no pattern file, or cannot be read
    "apexUnexpectedLattice",      # a "HexGrid" whose rows are not sqrt(3)/2 apart
    "apexUnknownGrid",            # a grid type that is neither HexGrid nor SqrGrid
    "apexUnreadable",             # the metadata file, or one of its groups, cannot be read
)


def safe_text(text) -> str:
    """``text`` as a string that is always valid UTF-8: a lone surrogate (what a file name that is
    not valid UTF-8 becomes on Linux) is replaced by ``?``. JSON answers carry file names, and one
    unencodable name must not turn the whole answer into a 500. Such a name is shown, not round-tripped."""
    return str(text).encode("utf-8", "replace").decode("utf-8")


class ApexError(ValueError):
    """A refusal the interface can word. ``code`` is what it translates on."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ApexPhase:
    index: int
    name: str
    formula: str
    symmetry: int
    laue_group: str
    point_group: str


@dataclass(frozen=True)
class ApexMeasurement:
    h5_path: str
    group: str                       # path of the 'OIM Map n' group inside the file
    area: str                        # 'Area 1'
    name: str                        # 'OIM Map 1'
    id: str                          # normalised GUID text, lower case
    grid_type: str                   # 'HexGrid' | 'SqrGrid'
    n_rows: int
    n_cols: int
    step_x: float                    # um
    step_y: float                    # um
    sample_tilt: float               # deg
    elevation: float                 # deg, camera elevation
    azimuth: float                   # deg, camera azimuth
    pc_tsl: Tuple[float, float, float]
    beam_kv: Optional[float]
    phases: Tuple[ApexPhase, ...]
    has_sem: bool
    has_iq: bool
    n_points: int                    # records in the per-point table
    #: The end of the scan as ISO 8601 ("2000-01-02T03:04:05"), from the map's ``ScanEndTime``
    #: attribute, or None when the file has none. APEX writes it as .NET ticks without saying
    #: whether they are UTC or local time, so the string carries no zone designator.
    scan_end: Optional[str] = None
    #: The pattern size the metadata records, ``(rows, cols)`` as a pattern file's ``pattern_shape``
    #: is, or None when it does not (the names are those of the OIM ``.h5`` export,
    #: ``edax_hex.py``; not confirmed on an APEX metadata file, so None is the normal case until
    #: it is: nothing is guessed from other fields).
    pattern_shape: Optional[Tuple[int, int]] = None

    @property
    def is_hex(self) -> bool:
        return self.grid_type == "HexGrid"

    @property
    def expected_patterns(self) -> int:
        return int(self.n_rows) * int(self.n_cols)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "area": self.area, "group": self.group,
            "grid_type": self.grid_type, "hex": self.is_hex,
            "n_rows": self.n_rows, "n_cols": self.n_cols, "n_points": self.n_points,
            "step_x": self.step_x, "step_y": self.step_y,
            "sample_tilt": self.sample_tilt, "elevation": self.elevation,
            "azimuth": self.azimuth, "pc": list(self.pc_tsl), "pc_convention": "tsl",
            "beam_kv": self.beam_kv, "has_sem": self.has_sem, "has_iq": self.has_iq,
            "phases": [{"index": p.index, "name": p.name, "formula": p.formula,
                        "symmetry": p.symmetry, "laue_group": p.laue_group,
                        "point_group": p.point_group} for p in self.phases],
            "pattern_shape": list(self.pattern_shape) if self.pattern_shape else None,
        }


# --------------------------------------------------------------------- helpers

def normalise_guid(text) -> str:
    """Canonical lower-case GUID text; ``ApexError`` when it is not a GUID."""
    try:
        return str(uuid.UUID(str(text).strip()))
    except ValueError as exc:
        raise ApexError("apexBadId", f"{text!r} is not a GUID") from exc


def _resolved(p: Path) -> Path:
    try:
        return p.resolve()
    except (OSError, ValueError):              # a path Windows cannot resolve (an embedded NUL)
        return p


def without_paths(text, *files) -> str:
    """``text`` with the folders of the given files removed, so that only file names are left.

    A library's exception text can carry the full local path (h5py: "unable to open file: name =
    '<folder>/x.h5'"; an ``OSError`` for a missing bitmap). What reaches the user names the file, never the folders above
    it. A folder is recognised in every spelling it appears in: as given, resolved, with ``/``, and
    with doubled backslashes (the way ``repr`` writes a path)."""
    out = str(text)
    for file in files:
        p = Path(str(file))
        folders = set()
        for q in (p.parent, _resolved(p).parent):
            folders.update((str(q), q.as_posix()))
        folders.update(s.replace("\\", "\\\\") for s in list(folders))
        folders.update(s.replace("/", "\\") for s in list(folders))
        for s in sorted(folders, key=len, reverse=True):
            if s and s not in (".", "\\", "/"):
                for sep in ("\\\\", "\\", "/"):
                    out = out.replace(s + sep, "")
    return out


def _text(value) -> str:
    v = np.ravel(value)[0] if np.ndim(value) else value
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)


def _num(value) -> float:
    return float(np.ravel(value)[0])


_TICKS_PER_SECOND = 10_000_000


def dotnet_ticks_to_iso(value) -> Optional[str]:
    """ISO 8601 text (seconds, no zone) of a .NET ``DateTime`` tick count, or None.

    A tick is 100 ns counted from 0001-01-01. APEX stores the end of a scan like that (an int64
    attribute of length 1) and does not say whether the clock was UTC or local, so no zone is
    attached. Anything that is not an integer, or lies outside the years 2000..2100, is not a
    time stamp this function vouches for: None."""
    try:
        arr = np.asarray(value)
        if arr.dtype.kind not in "iu" or arr.size < 1:
            return None
        seconds = int(np.ravel(arr)[0]) // _TICKS_PER_SECOND
        moment = datetime(1, 1, 1) + timedelta(seconds=seconds)
    except (OverflowError, ValueError, TypeError):
        return None
    return moment.isoformat(timespec="seconds") if 2000 <= moment.year <= 2100 else None


def _pattern_shape(group) -> Optional[Tuple[int, int]]:
    """``(rows, cols)`` from ``Pattern Height`` / ``Pattern Width`` below ``EBSD/ANG/HEADER`` when
    both are there and positive; anything else is "not recorded", never a reason to refuse."""
    try:
        header = group["EBSD/ANG/HEADER"]
        width, height = int(_num(header["Pattern Width"][()])), int(_num(header["Pattern Height"][()]))
    except Exception:  # noqa: BLE001 - absent, odd type or shape: not recorded
        return None
    return (height, width) if width > 0 and height > 0 else None


def _scan_end_iso(group) -> Optional[str]:
    """The ``ScanEndTime`` attribute of a measurement group as ISO text; a missing or odd
    attribute is no reason to refuse the measurement."""
    try:
        raw = group.attrs.get("ScanEndTime")
    except Exception:  # noqa: BLE001 - an attribute h5py cannot decode
        return None
    return None if raw is None else dotnet_ticks_to_iso(raw)


def _measurement_groups(h5) -> List[str]:
    found: List[str] = []

    def visit(name, obj):
        if _MEASUREMENT.search(name) and hasattr(obj, "keys") and "Sample" in obj and "EBSD" in obj:
            found.append(name)

    h5.visititems(visit)
    return found


def is_apex_file(path) -> bool:
    """True when ``path`` is an HDF5 file with at least one APEX measurement."""
    path = str(path)
    if Path(path).suffix.lower() not in (".h5", ".hdf5"):
        return False
    try:
        import h5py
        if not h5py.is_hdf5(path):
            return False
    except Exception:  # noqa: BLE001 - not even recognisable as HDF5: not ours
        return False
    try:
        with h5py.File(path, "r") as f:
            return bool(_measurement_groups(f))
    except Exception as exc:  # noqa: BLE001 - HDF5 that will not open, e.g. a session APEX holds open
        logger.warning("%s is an HDF5 file but could not be opened to look for APEX measurements "
                       "(%s: %s)", Path(path).name, type(exc).__name__, exc)
        return False


def probe_file(path) -> Tuple[str, Optional[str]]:
    """What an ``.h5``/``.hdf5`` file that exists is, for the load dialog: ``("apex", None)`` when
    it holds at least one APEX measurement, ``("plain", None)`` for an HDF5 file that does not, and
    ``("unreadable", reason)`` when it cannot be opened as HDF5 at all (not HDF5 under that name, or
    held open by the program that is still writing it). ``reason`` is an English phrase with the
    file name and no folders; the OS/HDF5 text is logged with its traceback.

    ``is_apex_file`` says False for all three and cannot tell them apart."""
    import h5py
    path = str(path)
    name = Path(path).name
    try:
        if not h5py.is_hdf5(path):
            logger.warning("%s has an HDF5 name but is not an HDF5 file (or cannot be opened)", name)
            return "unreadable", f"{name} is not an HDF5 file, or cannot be opened"
        with h5py.File(path, "r") as f:
            return ("apex" if _measurement_groups(f) else "plain"), None
    except Exception as exc:  # noqa: BLE001 - e.g. an HDF5 file the writing program still holds
        logger.warning("%s could not be opened to look for APEX measurements", name, exc_info=True)
        return "unreadable", f"{name} cannot be opened ({type(exc).__name__}: {without_paths(exc, path)})"


def _read_group(h5_path: str, h5, name: str) -> ApexMeasurement:
    g = h5[name]
    sample, cam = g["Sample"], g["Camera"]
    ang = g["EBSD/ANG"]
    pc = ang["HEADER/Pattern Center Calibration"]
    phases = []
    if "HEADER/Phase" in ang:
        for key in sorted(ang["HEADER/Phase"].keys(), key=lambda k: (len(k), k)):
            p = ang["HEADER/Phase"][key]
            phases.append(ApexPhase(
                index=int(key) if key.isdigit() else len(phases) + 1,
                name=_text(p["Material Name"][()]) if "Material Name" in p else "",
                formula=_text(p["Formula"][()]) if "Formula" in p else "",
                symmetry=int(_num(p["Symmetry"][()])) if "Symmetry" in p else 0,
                laue_group=_text(p["Laue Group"][()]) if "Laue Group" in p else "",
                point_group=_text(p["Point Group"][()]) if "Point Group" in p else ""))
    table = ang["DATA/DATA"]
    kv = None
    if "Column" in g and "SemKV" in g["Column"]:
        kv = _num(g["Column/SemKV"][()])
        kv = kv if kv > 0 else None
    area = name.split("/")[-2]
    return ApexMeasurement(
        h5_path=h5_path, group=name, area=area, name=name.split("/")[-1],
        id=normalise_guid(_text(g.attrs["Id"])),
        grid_type=_text(sample["Grid Type"][()]),
        n_rows=int(_num(sample["Number Of Rows"][()])),
        n_cols=int(_num(sample["Number Of Columns"][()])),
        step_x=_num(sample["Step X"][()]), step_y=_num(sample["Step Y"][()]),
        sample_tilt=_num(sample["Sample Tilt"][()]),
        elevation=_num(cam["Elevation Angle"][()]),
        azimuth=_num(cam["Azimuthal Angle"][()]),
        pc_tsl=(_num(pc["X-Star"][()]), _num(pc["Y-Star"][()]), _num(pc["Z-Star"][()])),
        beam_kv=kv,
        phases=tuple(phases),
        has_sem="SEM-PRIAS Images/DATA/SEM" in g,
        has_iq="IQ" in (table.dtype.names or ()),
        n_points=int(table.shape[0]),
        scan_end=_scan_end_iso(g),
        pattern_shape=_pattern_shape(g),
    )


def _natural(text: str) -> list:
    """Sort key that reads digit runs as numbers: "Area 2" comes before "Area 10"."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(text))]


@dataclass(frozen=True)
class ApexEntry:
    """One measurement group of an APEX file: read (``measurement``), or why it could not be
    (``error``). ``id`` is the group's GUID when that much could be read, else None."""
    group: str
    area: str
    name: str
    id: Optional[str]
    measurement: Optional[ApexMeasurement] = None
    error: Optional[ApexError] = None

    @property
    def problem_detail(self) -> Optional[str]:
        """The English sentence that goes with ``error.code`` (file and group names, no folders)."""
        return None if self.error is None else safe_text(self.error)


def _group_id(g) -> str:
    """The normalised ``Id`` attribute of a measurement group (``ApexError`` when it is missing or
    not a GUID)."""
    if "Id" not in g.attrs:
        raise ApexError("apexUnreadable", "the measurement group has no Id attribute")
    return normalise_guid(_text(g.attrs["Id"]))


def read_entries(path) -> List[ApexEntry]:
    """Every measurement group of an APEX file, each read on its own: one group that cannot be
    read (a missing dataset, an Id that is no GUID) is an entry with an ``error``, the others are
    still listed. Ordered by area then name, numbers in them by value (a plain text sort puts
    "Area 10" before "Area 2").

    The file itself unreadable is ``ApexError('apexUnreadable')``, a file without any measurement
    ``ApexError('apexNoMeasurement')``."""
    import h5py
    path = str(path)
    fname = Path(path).name
    out: List[ApexEntry] = []
    try:
        with h5py.File(path, "r") as f:
            for name in _measurement_groups(f):
                area, leaf = name.split("/")[-2], name.split("/")[-1]
                gid: Optional[str] = None
                try:
                    gid = _group_id(f[name])
                    out.append(ApexEntry(name, area, leaf, gid, _read_group(path, f, name)))
                except ApexError as exc:
                    out.append(ApexEntry(name, area, leaf, gid, error=exc))
                except Exception as exc:  # noqa: BLE001 - a damaged group is that group's problem
                    # Only the code reaches the interface (plus ``problem_detail``, the same text
                    # as the error's message); the traceback is for whoever has to find the bug.
                    logger.warning("%s: group %s / %s could not be read", fname, area, leaf,
                                   exc_info=True)
                    out.append(ApexEntry(name, area, leaf, gid, error=ApexError(
                        "apexUnreadable",
                        f"{fname}: {area} / {leaf}: {type(exc).__name__}: {without_paths(exc, path)}")))
    except ApexError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ApexError("apexUnreadable",
                        f"{fname}: {type(exc).__name__}: {without_paths(exc, path)}") from exc
    if not out:
        raise ApexError("apexNoMeasurement", f"{fname} holds no OIM Map measurement")
    out.sort(key=lambda e: (_natural(e.area), _natural(e.name)))
    return out


def list_measurements(path) -> List[ApexMeasurement]:
    """Every measurement of an APEX file, ordered by area then name, numbers in them by value.
    Strict: one measurement that cannot be read refuses the whole file (``read_entries`` lists
    the others anyway)."""
    entries = read_entries(path)
    for entry in entries:
        if entry.error is not None:
            raise entry.error
    return [e.measurement for e in entries]


def read_measurement(path, measurement_id) -> ApexMeasurement:
    """The measurement with this id. A damaged sibling group does not stop it; the wanted group
    itself being damaged does (with that group's code). When no group carries the id but some
    group's own Id could not be read, that group's error is raised: it may be the one asked for."""
    wanted = normalise_guid(measurement_id)
    entries = read_entries(path)
    for entry in entries:
        if entry.id == wanted:
            if entry.error is not None:
                raise entry.error
            return entry.measurement
    for entry in entries:                # a group whose Id could not be read might be the one asked for
        if entry.id is None and entry.error is not None:
            raise entry.error
    raise ApexError("apexMeasurementNotFound",
                    f"{Path(str(path)).name} has no measurement {wanted}")


# ----------------------------------------------------------- pattern file side

def up_guid(path) -> Optional[str]:
    """The measurement GUID in the header of an UP file, or None when it has none
    (version 1 and 3 headers end before byte 58)."""
    try:
        with open(path, "rb") as f:
            head = f.read(_GUID_END)
    except (OSError, ValueError):      # unreadable, or a name the OS cannot encode
        return None
    if len(head) < _GUID_END:
        return None
    _version, _sx, _sy, offset = struct.unpack_from("<4I", head, 0)
    if offset < _GUID_END:
        return None
    guid = uuid.UUID(bytes_le=head[_GUID_AT:_GUID_END])
    if guid.int == 0:                  # 16 zero bytes: the field is there but names no measurement
        return None
    return str(guid)


def find_apex_sidecar(pattern_path) -> Optional[ApexMeasurement]:
    """The measurement an UP file belongs to, looked up among the ``.h5`` files of
    its own folder; None when it has no GUID or no metadata file beside it names it.

    Known limit: every ``.h5`` of the folder is opened to look (``is_apex_file``). That is cheap
    for the handful of files of a measurement folder and slow in a folder of hundreds of large
    HDF5 files; the load dialog, which already knows the metadata file, never takes this path."""
    guid = up_guid(pattern_path)
    if guid is None:
        return None
    for p in sorted(Path(pattern_path).parent.iterdir()):
        if p.suffix.lower() in (".h5", ".hdf5") and p.is_file() and is_apex_file(p):
            try:
                listed = read_entries(p)
            except ApexError as exc:
                # an unrelated, unreadable session (e.g. an aborted one) must not stop the search
                logger.warning("%s: skipped while looking for the measurement of %s (%s: %s)",
                               p.name, Path(str(pattern_path)).name, exc.code, exc)
                continue
            for entry in listed:
                if entry.error is not None:
                    logger.warning("%s: %s skipped while looking for the measurement of %s (%s: %s)",
                                   p.name, entry.name, Path(str(pattern_path)).name,
                                   entry.error.code, entry.error)
                elif entry.id == guid:
                    return entry.measurement
    return None


def require_up_suffix(pattern_path) -> None:
    """``ApexError('apexPatternFileUnreadable')`` unless the path names a ``.up1``/``.up2`` file."""
    p = Path(str(pattern_path))
    if p.suffix.lower() not in _UP_SUFFIXES:
        raise ApexError("apexPatternFileUnreadable",
                        f"{p.name} is not an EDAX pattern file (.up1 or .up2)")


def read_up_header(pattern_path):
    """The header of an UP file; a header that cannot be read (version 2, truncated, zero size, the
    file gone) is an ``ApexError('apexPatternFileUnreadable')`` that names the file, not its folder."""
    from edax_up1 import read_up1_header
    require_up_suffix(pattern_path)
    try:
        return read_up1_header(str(pattern_path))
    except (ValueError, OSError) as exc:
        raise ApexError("apexPatternFileUnreadable",
                        f"{Path(str(pattern_path)).name} cannot be read as an EDAX pattern file "
                        f"({type(exc).__name__}: {without_paths(exc, pattern_path)})") from exc


def check_pairing(measurement: ApexMeasurement, pattern_path) -> str:
    """Prove that ``pattern_path`` holds the patterns of ``measurement``.

    Returns ``'guid'`` when the header names the measurement, ``'count'`` when the
    header has no GUID but the pattern count equals the grid. Raises ``ApexError``
    when the header names another measurement or the count is wrong.

    The patterns are counted in the body of the file, for every header version: the metadata
    file is the authority on the grid, and a header that states another grid (a scan that ended
    early, a hexagonal grid written as a square one) must not make a complete file look wrong."""
    # Only pattern files are opened: the path comes from the request, and a header read of any other
    # file would put 16 of its bytes into the message below.
    require_up_suffix(pattern_path)
    guid = up_guid(pattern_path)
    name = Path(str(pattern_path)).name
    if guid is not None and guid != measurement.id:
        raise ApexError("apexGuidMismatch",
                        f"{name} belongs to measurement {guid}, not to {measurement.id}")
    header = read_up_header(pattern_path)
    n = pattern_count(pattern_path, header)
    if n != measurement.expected_patterns:
        stated = ""
        if header.nx is not None and header.ny is not None and header.nx * header.ny != n:
            stated = f" (its header states {header.nx} x {header.ny})"
        raise ApexError("apexPatternCountMismatch",
                        f"{name} holds {n} patterns{stated}, the measurement "
                        f"is {measurement.n_rows} x {measurement.n_cols} = "
                        f"{measurement.expected_patterns}")
    return "guid" if guid is not None else "count"


def pattern_count(pattern_path, header) -> int:
    """How many whole patterns the body of an UP file holds (counted from the file size, for every
    header version; a pattern cut in the middle does not count). A file that cannot be measured any
    more (it vanished since its header was read) is ``ApexError('apexPatternFileUnreadable')``."""
    per_pattern = header.sx * header.sy * _UP_NBYTES[Path(str(pattern_path)).suffix.lower()]
    try:
        size = Path(str(pattern_path)).stat().st_size
    except OSError as exc:
        raise ApexError("apexPatternFileUnreadable",
                        f"{Path(str(pattern_path)).name} cannot be read any more "
                        f"({type(exc).__name__})") from exc
    return max(0, (size - header.pattern_offset) // per_pattern)


def check_header_grid(header, measurement: ApexMeasurement, pattern_path) -> None:
    """``ApexError('apexGridMismatch')`` when a v3+ UP header's hexagonal flag disagrees with the
    measurement's grid type (older headers carry no flag). The load makes this check."""
    if header.version >= 3 and bool(header.is_hex) != measurement.is_hex:
        raise ApexError("apexGridMismatch",
                        f"{Path(str(pattern_path)).name} says hexagonal={bool(header.is_hex)}, "
                        f"the measurement says {measurement.grid_type}")


def check_pattern_file(measurement: ApexMeasurement, pattern_path) -> str:
    """Everything about a pattern file that can be known before the load, in the load's terms:
    ``check_pairing`` (GUID or count) and the header's hexagonal flag against the grid type.
    Returns ``'guid'`` or ``'count'``; raises the ``ApexError`` the load would."""
    how = check_pairing(measurement, pattern_path)
    check_header_grid(read_up_header(pattern_path), measurement, pattern_path)
    return how


def describe_pattern_file(pattern_path) -> dict:
    """What the load dialog shows of a pattern file, from its header and its size alone:
    ``{"path", "name", "n_patterns", "pattern_shape": [rows, cols], "problem"}``. A file whose
    header cannot be read has ``None`` for the two numbers and ``problem`` =
    ``apexPatternFileUnreadable``. ``path`` and ``name`` are always valid UTF-8 (``safe_text``): a
    file name that is not shows with ``?`` for the bytes that are not text."""
    path = str(pattern_path)
    out = {"path": safe_text(path), "name": safe_text(Path(path).name), "n_patterns": None,
           "pattern_shape": None, "problem": None}
    try:
        header = read_up_header(path)
        out["n_patterns"] = pattern_count(path, header)
        out["pattern_shape"] = [int(header.sy), int(header.sx)]
    except ApexError as exc:
        out["problem"] = exc.code
    except OSError:                                  # the file went away between the listing and now
        out["problem"] = "apexPatternFileUnreadable"
    return out


def count_relation(n_patterns: Optional[int], measurement: ApexMeasurement) -> Optional[str]:
    """``'fits'``, ``'too_few'`` or ``'too_many'``: the pattern count against the measurement's
    grid (so that "incomplete" is said only of a file that has too few). None when unknown."""
    if n_patterns is None:
        return None
    expected = measurement.expected_patterns
    return "fits" if n_patterns == expected else ("too_few" if n_patterns < expected else "too_many")


def pattern_file_entry(measurement: ApexMeasurement, pattern_path,
                       groups_by_id: Optional[dict] = None) -> dict:
    """One pattern file of a measurement, as ``/inspect`` lists it: ``describe_pattern_file`` plus
    ``pairing`` (``'guid'`` when the header names a measurement, else ``'count'``), ``count``
    (``count_relation``), ``matches_grid`` and ``problem`` (the code ``check_pattern_file`` raises,
    or None). A file that does not read as a pattern file has ``pairing`` and ``count`` None.

    ``names_measurement`` is the GUID the header names (None for a header without one) and
    ``names_group`` the group key of that measurement when it is one of the file's own
    (``groups_by_id``: GUID -> group key of every measurement of the metadata file; without it only
    the measurement asked about is known), else None: the dialog says "belongs to OIM Map 2"."""
    entry = describe_pattern_file(pattern_path)
    entry["pairing"] = entry["count"] = entry["names_measurement"] = entry["names_group"] = None
    if entry["problem"] is None:             # only a file that reads as a pattern file is looked into
        guid = up_guid(pattern_path)
        entry["pairing"] = "guid" if guid is not None else "count"
        entry["names_measurement"] = guid
        known = groups_by_id if groups_by_id is not None else {measurement.id: measurement.group}
        entry["names_group"] = known.get(guid) if guid is not None else None
        entry["count"] = count_relation(entry["n_patterns"], measurement)
        try:
            check_pattern_file(measurement, pattern_path)
        except ApexError as exc:
            entry["problem"] = exc.code
    entry["matches_grid"] = entry["problem"] is None
    return entry


def pattern_files_in(folder) -> List[Tuple[str, Optional[str]]]:
    """``(path, GUID or None)`` of every ``.up1``/``.up2`` file of a folder, in name order.

    A folder that cannot be listed is ``ApexError('apexFolderUnreadable')``: "no file" would be a
    wrong thing to say about a folder nobody could look into."""
    if not str(folder).strip():                 # Path("") is the working directory: not a folder anybody named
        raise ApexError("apexFolderUnreadable", "no folder was named")
    folder = Path(str(folder))
    try:
        entries = sorted(folder.iterdir())
    except (OSError, ValueError) as exc:
        raise ApexError("apexFolderUnreadable",
                        f"{folder.name or 'the folder'} cannot be listed ({type(exc).__name__})") from exc
    out = []
    for p in entries:
        try:
            if p.suffix.lower() in _UP_SUFFIXES and p.is_file():
                out.append((str(p), up_guid(p)))
        except OSError:
            continue
    return out


def measurement_problem(measurement: ApexMeasurement) -> Optional[str]:
    """The code of what makes a measurement impossible to open whatever pattern file it gets, from
    the metadata file alone and in the load's order: the per-point table against the grid
    (``apexGridMismatch``), then the grid itself (``apexUnknownGrid``, ``apexUnexpectedLattice``,
    ``apexInvalidGrid``). None when nothing is wrong."""
    if measurement.n_points != measurement.expected_patterns:
        return "apexGridMismatch"
    try:
        geometry_from_apex(measurement)
    except ApexError as exc:
        return exc.code
    return None


def load_state(measurement: ApexMeasurement, files: List[dict],
               folder_problem: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[str]]:
    """``(loadable, problem, measurement_problem)`` of a measurement given its listed pattern files
    (``pattern_file_entry``). Loadable: nothing wrong with the measurement itself and at least one
    usable file. ``problem`` is the reason it is not: the measurement's own code, else
    ``folder_problem`` when the folder could not be listed (no file was looked for, so "no pattern
    file" would be false), else ``apexNoPatternFile`` when no file names it, else the first file's
    code. ``measurement_problem`` is the first kind alone, so the dialog can tell "choose another
    file" from "cannot be opened at all"."""
    own = measurement_problem(measurement)
    if own is not None:
        return False, own, own
    if any(f["matches_grid"] for f in files):
        return True, None, None
    if folder_problem is not None:
        return False, folder_problem, None
    if not files:
        return False, "apexNoPatternFile", None
    return False, files[0]["problem"], None


def scan_pattern_files(folder, measurements: List[ApexMeasurement],
                       groups_by_id: Optional[dict] = None) -> Tuple[dict, List[dict]]:
    """Look into ``folder`` once: ``({measurement id: [file entries]}, unmatched)``.

    A file whose header names a measurement of the list goes to that measurement
    (``pattern_file_entry``; ``groups_by_id`` as there); a file naming a measurement that is not in
    the list is left out of both. ``unmatched`` are the files with no GUID (or an all-zero one):
    ``describe_pattern_file`` plus ``names_measurement`` / ``names_group`` (both None) and
    ``fits``, the ids of the measurements the file could be chosen for without Open refusing it:
    the pattern count equals the grid, the measurement itself has no problem apart from its pattern
    file (``measurement_problem``) and a v3+ header's hexagonal flag agrees with its grid type.
    Nothing is paired from that."""
    by_id = {m.id: m for m in measurements}
    listed: dict = {m.id: [] for m in measurements}
    unmatched: List[dict] = []
    for path, guid in pattern_files_in(folder):
        if guid is None:
            info = describe_pattern_file(path)
            info["names_measurement"] = info["names_group"] = None
            info["fits"] = []
            header = None
            if info["n_patterns"] is not None:
                try:
                    header = read_up_header(path)      # read a moment ago by describe_pattern_file
                except ApexError:                      # ... and gone since: it fits nothing
                    info["problem"] = "apexPatternFileUnreadable"
            if header is not None:
                for m in measurements:
                    if m.expected_patterns != info["n_patterns"] or measurement_problem(m) is not None:
                        continue
                    try:
                        check_header_grid(header, m, path)
                    except ApexError:
                        continue
                    info["fits"].append(m.id)
            unmatched.append(info)
        elif guid in by_id:
            listed[guid].append(pattern_file_entry(by_id[guid], path, groups_by_id))
    return listed, unmatched


# --------------------------------------------------------------------- preview

def point_map(measurement: ApexMeasurement, field: str) -> np.ndarray:
    """One column of the per-point table as ``(n_rows, n_cols)``, in index order."""
    import h5py
    with h5py.File(measurement.h5_path, "r") as f:
        table = f[measurement.group + "/EBSD/ANG/DATA/DATA"]
        if field not in (table.dtype.names or ()):
            raise ApexError("apexNoPreview", f"this measurement has no {field} column")
        values = np.asarray(table[field])
    if values.size != measurement.expected_patterns:
        raise ApexError("apexPatternCountMismatch",
                        f"{field}: {values.size} values for a "
                        f"{measurement.n_rows} x {measurement.n_cols} grid")
    return values.reshape(measurement.n_rows, measurement.n_cols)


def sem_image(measurement: ApexMeasurement) -> Optional[np.ndarray]:
    import h5py
    with h5py.File(measurement.h5_path, "r") as f:
        key = measurement.group + "/SEM-PRIAS Images/DATA/SEM"
        return np.asarray(f[key]) if key in f else None


def _png(arr: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def preview_png(measurement: ApexMeasurement, kind: str, max_side: int = 320) -> bytes:
    """A small grey PNG: the overview image (``sem``) or the IQ map (``iq``)."""
    if kind == "sem":
        img = sem_image(measurement)
        if img is None:
            raise ApexError("apexNoPreview", "this measurement has no overview image")
    elif kind == "iq":
        img = point_map(measurement, "IQ")
    else:
        raise ApexError("apexNoPreview", f"unknown preview kind {kind!r}")
    img = np.asarray(img, dtype=np.float64)
    lo, hi = np.percentile(img, [2, 98])
    scaled = np.clip((img - lo) / (hi - lo) if hi > lo else np.zeros_like(img), 0, 1)
    out = (scaled * 255).astype(np.uint8)
    step = max(1, int(np.ceil(max(out.shape) / max_side)))
    return _png(out[::step, ::step])


# ------------------------------------------------------------------- loading

_UP_NBYTES = {".up1": 1, ".up2": 2}
_UP_DTYPE = {".up1": np.uint8, ".up2": np.uint16}
#: A native UP scan opened lazily is cut into chunks of about this many bytes.
_UP_CHUNK_BYTES = 32 * 2 ** 20


class _MappedPatterns:
    """A memory map as ``dask.array.from_array`` gets to see it: shape, dtype and reading, no ``copy``.

    ``from_array`` calls ``copy()`` on anything array-like, which for a memory map reads the whole
    file into memory. An object that is not array-like is read block by block when a block is asked for."""

    def __init__(self, mapped: np.ndarray) -> None:
        self._mapped = mapped
        self.shape = mapped.shape
        self.dtype = mapped.dtype
        self.ndim = mapped.ndim

    def __getitem__(self, key):
        return np.array(self._mapped[key])


def geometry_from_apex(m: ApexMeasurement):
    """The ``ScanGeometry`` an APEX measurement describes.

    The APEX file does not state which way the odd rows of a hexagonal scan are offset
    (``HexGrid`` and ``IsScanSuccessful`` in the file are both unreliable). The direction was
    measured from the patterns of EDAX hexagonal scans: +1/2 step, the convention OIM's own
    hex files follow (``scan_geometry.OIM_ODD_ROW_SHIFT``). An APEX measurement therefore
    carries it as confirmed (``shift_confirmed=True``)."""
    from backend.api.services.scan_geometry import OIM_ODD_ROW_SHIFT, ScanGeometry
    if m.grid_type == "HexGrid":
        try:
            return ScanGeometry.hex(m.n_rows, m.n_cols, m.step_x, m.step_y,
                                    odd_row_shift=OIM_ODD_ROW_SHIFT, shift_confirmed=True)
        except ValueError as exc:
            raise ApexError("apexUnexpectedLattice", str(exc)) from exc
    if m.grid_type == "SqrGrid":
        try:
            return ScanGeometry.square(m.n_rows, m.n_cols, m.step_x, m.step_y)
        except ValueError as exc:             # a step that is not positive, a grid without points
            raise ApexError("apexInvalidGrid", f"{m.area} / {m.name}: {exc}") from exc
    raise ApexError("apexUnknownGrid", f"unknown grid type {m.grid_type!r}")


def measurement_for_pattern_file(file_path, explicit: Optional[dict] = None):
    """The APEX measurement a pattern file is loaded with, and how the pairing was proven.

    ``explicit`` is the user's choice (``h5_path``, ``measurement_id``); it is checked, never
    trusted. Without it the folder of the file is searched for a metadata file that names the
    file's GUID. Returns ``(None, None)`` when there is none."""
    if explicit:
        h5_path = explicit.get("h5_path")
        if not h5_path or not is_apex_file(h5_path):
            raise ApexError("apexUnreadable",
                            f"{Path(str(h5_path)).name!r} could not be read as an APEX metadata file")
        m = read_measurement(h5_path, explicit.get("measurement_id"))
    else:
        m = find_apex_sidecar(file_path)
        if m is None:
            return None, None
    return m, check_pairing(m, file_path)


def load_edax_up_with_apex(file_path, header, m: ApexMeasurement, how: str, use_lazy: bool,
                           verbose: bool = True):
    """Open an UP scan in the grid its APEX measurement states.

    The patterns are stored in acquisition order, row after row, every row ``n_cols`` long (also
    on a hexagonal scan); nothing is resampled and no point is dropped. kikuchipy cannot do this
    itself: for a hexagonal header it returns a one-dimensional navigation and ignores
    ``nav_shape``. The signal is the index-space rectangle; the geometry that says where each
    point lies is attached as ``Signal.scan_geometry`` for the route to register."""
    from kikuchipy.io._io import _dict2signal
    from edax_up1 import tsl_detector

    suffix = Path(file_path).suffix.lower()
    check_header_grid(header, m, file_path)
    if m.n_points != m.expected_patterns:
        raise ApexError("apexGridMismatch",
                        f"the measurement states {m.n_rows} x {m.n_cols} points but its "
                        f"table holds {m.n_points}")
    geometry = geometry_from_apex(m)
    n_rows, n_cols, sy, sx = m.n_rows, m.n_cols, int(header.sy), int(header.sx)

    # A header and a measurement that agree do not prove the body is all there (a copy that was
    # cut short): np.memmap would answer with a bare numpy/OS error.
    per_pattern = sy * sx * _UP_NBYTES[suffix]
    needed = int(header.pattern_offset) + n_rows * n_cols * per_pattern
    try:
        size = Path(file_path).stat().st_size
    except OSError as exc:
        raise ApexError("apexPatternFileUnreadable",
                        f"{Path(str(file_path)).name} cannot be read any more "
                        f"({type(exc).__name__})") from exc
    if size < needed:
        fits = max(0, (size - int(header.pattern_offset)) // per_pattern)
        raise ApexError("apexPatternCountMismatch",
                        f"{Path(str(file_path)).name} is shorter than its header and the "
                        f"measurement say: it has room for {fits} patterns, the {n_rows} x "
                        f"{n_cols} grid needs {n_rows * n_cols}")

    mm = np.memmap(file_path, dtype=_UP_DTYPE[suffix], mode="c",
                   offset=int(header.pattern_offset), shape=(n_rows * n_cols, sy, sx))
    data = mm.reshape(n_rows, n_cols, sy, sx)
    if use_lazy:
        import dask.array as da
        per_row = n_cols * sy * sx * _UP_NBYTES[suffix]
        rows_per_chunk = max(1, min(n_rows, _UP_CHUNK_BYTES // max(per_row, 1)))
        cols_per_chunk = n_cols
        if rows_per_chunk == 1:
            # one row alone may exceed the chunk size (a very wide scan of large patterns)
            per_pattern = sy * sx * _UP_NBYTES[suffix]
            cols_per_chunk = max(1, min(n_cols, _UP_CHUNK_BYTES // max(per_pattern, 1)))
        # name=False: by default dask names the array after a hash of its content, and there is
        # no content to hash without reading the file.
        data = da.from_array(_MappedPatterns(data), chunks=(rows_per_chunk, cols_per_chunk, sy, sx),
                             name=False)

    axes = [
        {"size": n_rows, "index_in_array": 0, "name": "y", "scale": geometry.step_y,
         "offset": 0.0, "units": "um"},
        {"size": n_cols, "index_in_array": 1, "name": "x", "scale": geometry.step_x,
         "offset": 0.0, "units": "um"},
        {"size": sy, "index_in_array": 2, "name": "dy", "scale": 1.0, "offset": 0.0, "units": "um"},
        {"size": sx, "index_in_array": 3, "name": "dx", "scale": 1.0, "offset": 0.0, "units": "um"},
    ]
    metadata = {
        "General": {"original_filename": str(file_path), "title": Path(str(file_path)).stem},
        "Signal": {"signal_type": "EBSD", "record_by": "image"},
    }
    if m.beam_kv:
        metadata["Acquisition_instrument"] = {"SEM": {"beam_energy": m.beam_kv}}
    scan = {
        "data": data,
        "axes": axes,
        "metadata": metadata,
        "original_metadata": {"manufacturer": "EDAX",
                              "apex": {"area": m.area, "measurement": m.name, "pairing": how}},
        # The PC is the file's own (EDAX/TSL convention); kikuchipy stores it in Bruker's.
        "detector": tsl_detector((sy, sx), m.pc_tsl, sample_tilt=m.sample_tilt,
                                 tilt=m.elevation, azimuthal=m.azimuth),
    }
    signal = _dict2signal(scan, lazy=use_lazy)
    signal.metadata.set_item("Signal.pc_source", "apex")
    signal.metadata.set_item("Signal.scan_geometry", geometry.to_dict())
    # The record the API hands on (``/load`` answer ``apex``, loaded-files entry): the metadata file
    # (absolute), the measurement id and name, and how the pairing was proven ("guid" | "count").
    signal.metadata.set_item("Signal.apex", {
        # absolute: the pairing is remembered, and a path relative to the folder the backend was
        # started in must still find its file after the working directory has moved
        "h5_path": str(Path(m.h5_path).resolve()), "measurement_id": m.id,
        "measurement_name": f"{m.area} / {m.name}", "pairing": how})
    if verbose:
        logger.info("Loaded %s with APEX measurement %s / %s (%s, %d x %d)",
                    Path(str(file_path)).name, m.area, m.name, m.grid_type, n_rows, n_cols)
    return signal
