"""EDAX APEX metadata files: what the load dialog needs before anything is loaded.

``GET /inspect``                  is a picked file an APEX file? ``kind`` is ``"apex"`` (then: its
                                  measurements, whether each can be opened and why not, the pattern
                                  files found for each - matched by GUID, same folder only - and the
                                  pattern files of the folder that name no measurement),
                                  ``"choices"`` (any other HDF5 EBSD file with more than one
                                  measurement x pattern source: see ``load_choices``), ``"plain"``
                                  (one variant), or ``"unreadable"`` (an HDF5 name that cannot be
                                  opened as HDF5, e.g. one APEX still holds); a missing file or a
                                  folder is a coded ``400``, as is a damaged group.
``GET /inspect/preview``          a PNG of the middle pattern of one pattern source of an HDF5 file.
``GET /apex/pattern-files``       the same pattern-file listing for another folder.
``GET /apex/preview``             a small PNG of one measurement: its overview image or its IQ map.
``POST /apex/check-pairing``      is this pattern file the one that belongs to that measurement?
                                  (``200`` or a coded ``400``.)
``POST /apex/describe-pattern-file``  the same check as data: the file's entry as ``/inspect``
                                  lists it, with ``problem`` in the body and ``200`` either way.

Every code these routes can emit is in ``edax_apex.APEX_CODES``.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Response
from pydantic import BaseModel

import edax_apex as apex
from backend.api.problem import problem
from backend.api.services import load_choices

logger = logging.getLogger(__name__)
router = APIRouter()


class PairingRequest(BaseModel):
    h5_path: str
    measurement_id: str
    pattern_path: str


def _refusal(exc: apex.ApexError):
    return problem(400, exc.code, apex.safe_text(exc))


def _reject_nul(*paths: str) -> None:
    """A path with an embedded NUL cannot name a file; the OS calls raise ``ValueError`` on it."""
    if any("\x00" in p for p in paths):
        raise problem(400, "apexBadPath", "a file path contains a NUL character")


def _h5_state(path: str):
    """``apex.probe_file`` for a path that is asked about: ``("apex" | "plain" | "unreadable",
    reason)``, after the checks that are coded refusals instead. A name that does not exist is
    ``apexFileNotFound``; a folder, or a path the OS will not let us look at (no permission), is
    ``apexFileUnreadable`` (with the sentence that says which). Shared by ``/inspect`` and the
    other-folder route, so the same file gets the same code on both."""
    name = Path(path).name
    try:
        p = Path(path)
        is_dir, is_file = p.is_dir(), p.is_file()
    except (OSError, ValueError) as exc:
        raise apex.ApexError("apexFileUnreadable",
                             f"{name} cannot be accessed ({type(exc).__name__})") from exc
    if is_dir:
        raise apex.ApexError("apexFileUnreadable", f"{name} is a folder, not a file")
    if not is_file:
        raise apex.ApexError("apexFileNotFound", f"{name} does not exist")
    return apex.probe_file(path)


def _require_apex_file(path: str) -> None:
    """The metadata file of a request must be an APEX file: ``apexFileNotFound`` /
    ``apexFileUnreadable`` as ``/inspect`` says them, ``apexNoMeasurement`` for an HDF5 file that
    holds no OIM Map."""
    state, reason = _h5_state(path)
    if state == "unreadable":
        raise apex.ApexError("apexFileUnreadable", reason)
    if state == "plain":
        raise apex.ApexError("apexNoMeasurement", f"{Path(path).name} holds no OIM Map measurement")


# ------------------------------------------------------------ the answer of /inspect

def _blank_measurement() -> dict:
    """The keys of a measurement entry with nothing in them: what a measurement that could not be
    read is listed as, so that the dialog finds the same keys on every entry."""
    probe = apex.ApexMeasurement(
        h5_path="", group="", area="", name="", id="", grid_type="", n_rows=0, n_cols=0,
        step_x=0.0, step_y=0.0, sample_tilt=0.0, elevation=0.0, azimuth=0.0, pc_tsl=(0.0, 0.0, 0.0),
        beam_kv=None, phases=(), has_sem=False, has_iq=False, n_points=0)
    blank = {key: None for key in probe.to_dict()}
    blank.update(has_sem=False, has_iq=False, phases=[])
    return blank


def _measured_at(scan_end: Optional[str]) -> dict:
    """When the scan ended, from the map group's ``ScanEndTime`` attribute.

    What the evidence on a real APEX file showed (no value is recorded here): ``ScanEndTime`` is an
    int64 array of length 1, .NET ticks (100 ns since 0001-01-01), and is by name the end of the
    scan. The map's own ``TimeStamp`` attribute lies within milliseconds of it, so it is the same
    moment (the end, not the start). The ``TimeStamp`` of the enclosing area is not the start of the
    scan either: its distance to the map's does not grow with the number of points. No scan start
    and no time zone are stored anywhere in the file.

    So: ``measured_at`` (ISO 8601, no zone designator), which moment it is (``measured_at_kind``,
    always ``"scan_end"``) and ``measured_at_timezone`` (always ``"unknown"``). All three are None
    when the file has no usable ``ScanEndTime``."""
    if scan_end is None:
        return {"measured_at": None, "measured_at_kind": None, "measured_at_timezone": None}
    return {"measured_at": scan_end, "measured_at_kind": "scan_end", "measured_at_timezone": "unknown"}


def _measurement_entry(entry: apex.ApexEntry, files: List[dict],
                       folder_problem: Optional[str] = None) -> dict:
    """One measurement of the answer: ``ApexMeasurement.to_dict()`` plus ``consistent`` (the table
    holds one record per grid point), ``pattern_files``, ``loadable``, ``problem`` /
    ``measurement_problem`` (``edax_apex.load_state``), ``problem_detail`` and the ``measured_at*``
    fields. A group that could not be read has the same keys with nothing in them, ``loadable``
    False, its error code as ``problem`` and ``measurement_problem`` and the English sentence of its
    error as ``problem_detail`` (None on a group that was read). ``folder_problem`` is the code the
    folder searched could not be listed with: it becomes the ``problem`` of a measurement that
    would otherwise say "no pattern file", which nobody looked for."""
    if entry.measurement is None:
        out = _blank_measurement()
        out.update(id=entry.id, name=entry.name, area=entry.area, group=entry.group,
                   consistent=None, pattern_files=[], loadable=False,
                   problem=entry.error.code, measurement_problem=entry.error.code,
                   problem_detail=entry.problem_detail, **_measured_at(None))
        return out
    m = entry.measurement
    loadable, why_not, own = apex.load_state(m, files, folder_problem)
    out = m.to_dict()
    out.update(consistent=m.n_points == m.expected_patterns, pattern_files=files, loadable=loadable,
               problem=why_not, measurement_problem=own, problem_detail=None,
               **_measured_at(m.scan_end))
    return out


def _listing(h5_path: str, folder: str) -> dict:
    """The part of the answer that depends on the folder searched: ``measurements`` (each with
    its ``pattern_files`` from ``folder``) and ``unmatched_pattern_files`` (see
    ``edax_apex.scan_pattern_files``)."""
    entries = apex.read_entries(h5_path)
    groups = {e.id: e.group for e in entries if e.id}
    listed, unmatched = apex.scan_pattern_files(
        folder, [e.measurement for e in entries if e.measurement], groups)
    return {"measurements": [_measurement_entry(e, listed.get(e.id, []) if e.measurement else [])
                             for e in entries],
            "unmatched_pattern_files": unmatched}


#: the names ``/inspect`` is asked about: the HDF5 extensions of EBSD files (APEX uses ``.h5``)
_INSPECTED_SUFFIXES = (".h5", ".hdf5", ".h5oina", ".h5ebsd")


def _list_choices(path: str):
    """``load_choices.inspect_file`` for a request: a group that cannot be read (a broken link, a
    header that is not a group, ... ``inspect_file`` itself only answers for ``OSError``) is the
    coded ``loadInspectFailed`` with the file's name in the sentence, never an unhandled error."""
    try:
        return load_choices.inspect_file(path)
    except load_choices.LoadChoiceError:
        raise
    except Exception as exc:  # noqa: BLE001 - whatever a damaged group raises
        logger.warning("The measurements of %s could not be listed", Path(path).name, exc_info=True)
        raise load_choices.LoadChoiceError(
            load_choices.INSPECT_FAILED,
            f"{Path(path).name}: a group of the file could not be read ({type(exc).__name__})") from exc


def _choices_or_plain(path: str) -> dict:
    """The answer for an HDF5 file that is not APEX: ``kind: "choices"`` when it offers more than
    one variant (measurement x pattern source), else ``{"kind": "plain"}``."""
    choices = _list_choices(path)
    if choices is None or choices.n_variants() <= 1:
        return {"kind": "plain"}
    body = choices.to_dict()
    body["default_variant"] = body.pop("default")
    return {"kind": "choices", "path": path, **body}


def _inspect(path: str) -> dict:
    """The answer of ``GET /inspect``.

    ``{"kind": "plain"}`` for a file that is neither an APEX metadata file nor one that offers a
    choice: any name that does not end in ``.h5``/``.hdf5``/``.h5oina``/``.h5ebsd``, an HDF5 file with
    no APEX measurement and at most one loadable variant. A name with one of those endings that does
    not exist is a coded ``400`` (``apexFileNotFound``); so is a folder
    with such a name or a path the OS will not let us look at (``apexFileUnreadable``, the sentence
    says which). One that exists but cannot be opened as HDF5 (not HDF5 under that name, or held
    open by the program still writing it) is ``{"kind": "unreadable", "h5_path", "problem":
    "apexFileUnreadable", "problem_detail"}`` (the OS/HDF5 reason is in the log). **Changed:** a
    non-HDF5 file with an ``.h5`` name used to be ``plain``, and so did an unopenable HDF5 file and
    a missing one; the dialog never appeared for them.

    An HDF5 file with more than one variant is ``{"kind": "choices", "path", "format", "n_variants",
    "measurements", "default_variant"[, "suggested"]}`` (``load_choices.FileChoices.to_dict``; its
    ``default`` is called ``default_variant`` here); a file whose groups cannot be listed is a coded
    ``400`` (``loadInspectFailed``).

    An APEX file is::

        {"kind": "apex", "h5_path": path, "folder": <its folder>, "folder_problem": None | code,
         "measurements": [...], "unmatched_pattern_files": [...]}

    Each measurement is ``ApexMeasurement.to_dict()`` (``id``, ``name``, ``area``, ``group``,
    ``grid_type``, ``hex``, ``n_rows``, ``n_cols``, ``n_points``, ``step_x``, ``step_y`` in um,
    ``sample_tilt``, ``elevation``, ``azimuth``, ``pc`` + ``pc_convention`` ("tsl"), ``beam_kv``,
    ``has_sem``, ``has_iq``, ``phases``, ``pattern_shape`` ([rows, cols] of one pattern as the metadata
    records it, None when it does not; a pattern file's own ``pattern_shape`` is in the same order)) plus

    ``consistent``            the table holds one record per grid point;
    ``measured_at``           when the scan ended, ISO 8601 without a zone, or None; with
                              ``measured_at_kind`` ("scan_end") and ``measured_at_timezone``
                              ("unknown": APEX stores .NET ticks and no zone), see ``_measured_at``;
    ``loadable``              True when nothing stands in the way of opening it, no pattern data
                              needed to know: the grid type is known, the lattice is one Orienta
                              reads, the table matches the grid, and a pattern file listed below
                              passes every check the load would make;
    ``problem``               why not (None when loadable): the measurement's own code
                              (``apexGridMismatch``, ``apexUnknownGrid``, ``apexUnexpectedLattice``,
                              ``apexInvalidGrid``, or for a group that could not be read
                              ``apexUnreadable`` / ``apexBadId``), else ``folder_problem`` when the
                              folder could not be listed, else ``apexNoPatternFile``, else the
                              code of the first listed file;
    ``measurement_problem``   the first kind alone (None when the measurement itself is fine and
                              only a pattern file is missing or wrong);
    ``problem_detail``        the English sentence behind ``problem`` of a group that could not be
                              read (file and group names, no folders); None otherwise;
    ``pattern_files``         the ``.up1``/``.up2`` files of the folder whose header names the
                              measurement, each ``{"path", "name", "n_patterns", "pattern_shape"
                              [rows, cols], "pairing", "count" ("fits" | "too_few" | "too_many" |
                              None), "names_measurement", "names_group", "matches_grid",
                              "problem"}``. Here ``pairing`` is always ``"guid"`` (a file is listed
                              under a measurement only because its header names it);
                              ``"count"`` occurs only in the answer of ``describe-pattern-file``,
                              for a file the user chose whose header names no measurement.
                              ``names_measurement`` is the GUID the header names, ``names_group``
                              the group key of that measurement when it is one of this file's own
                              (so the dialog can say "belongs to OIM Map 2"), else None.
                              ``problem`` is the code the load would refuse the file with: the
                              pairing checks of ``check_pairing`` or a hexagonal flag in the UP
                              header that contradicts the grid type, or ``None``.

    Rows are keyed on ``group`` (unique within the file), never on ``id``: a measurement whose
    group could not be read is still listed, with the same keys, nothing in them (``id`` is None
    when even its Id attribute could not be read) and its error as ``problem``.

    ``unmatched_pattern_files`` are the pattern files of the folder whose header names no
    measurement: ``{"path", "name", "n_patterns", "pattern_shape", "problem", "names_measurement"
    (None), "names_group" (None), "fits"}`` where ``fits`` are the ids of the measurements that
    file could be chosen for without Open refusing it (the pattern count equals the grid, the
    measurement has no problem apart from its pattern file, a v3+ header's hexagonal flag agrees
    with its grid type). Nothing is paired automatically. ``folder_problem`` is
    ``apexFolderUnreadable`` when the folder could not be listed (then both lists of files are
    empty and no measurement says "no pattern file")."""
    if Path(path).suffix.lower() not in _INSPECTED_SUFFIXES:
        return {"kind": "plain"}
    state, reason = _h5_state(path)
    if state == "plain":
        return _choices_or_plain(path)
    if state == "unreadable":
        return {"kind": "unreadable", "h5_path": path, "problem": "apexFileUnreadable",
                "problem_detail": reason}
    folder = str(Path(path).parent)
    try:
        listing = _listing(path, folder)
        folder_problem = None
    except apex.ApexError as exc:
        if exc.code != "apexFolderUnreadable":
            raise
        entries = apex.read_entries(path)
        listing = {"measurements": [_measurement_entry(e, [], exc.code) for e in entries],
                   "unmatched_pattern_files": []}
        folder_problem = exc.code
    return {"kind": "apex", "h5_path": path, "folder": folder, "folder_problem": folder_problem,
            **listing}


@router.get("/inspect")
async def inspect_file(path: str):
    """Tell an APEX metadata file from any other file, and list the variants of an HDF5 file that
    offers a choice, without loading anything."""
    _reject_nul(path)
    try:
        return await asyncio.to_thread(_inspect, path)
    except apex.ApexError as exc:
        raise _refusal(exc)
    except load_choices.LoadChoiceError as exc:
        raise problem(400, exc.code, str(exc))


def _preview_png(path: str, measurement: str, source: str) -> bytes:
    # The names /inspect asks about, before the file system: other paths get one answer,
    # whether they exist or not.
    if Path(path).suffix.lower() not in _INSPECTED_SUFFIXES:
        raise load_choices.LoadChoiceError("loadNoPatterns", f"{Path(path).name} is not an HDF5 file")
    state, reason = _h5_state(path)
    if state == "unreadable":
        raise apex.ApexError("apexFileUnreadable", reason)
    if state == "apex":
        raise load_choices.LoadChoiceError(
            "loadNoPatterns", f"{Path(path).name} is an APEX metadata file and holds no patterns")
    try:
        return load_choices.preview_png(path, measurement, source)
    except load_choices.LoadChoiceError:
        raise
    except Exception as exc:  # noqa: BLE001 - a damaged group or dataset
        logger.warning("The preview pattern of %s could not be read", Path(path).name, exc_info=True)
        raise load_choices.LoadChoiceError(
            load_choices.INSPECT_FAILED,
            f"{Path(path).name}: the pattern could not be read ({type(exc).__name__})") from exc


@router.get("/inspect/preview")
async def inspect_preview(path: str, measurement: str, source: str):
    """A PNG of the middle pattern of one pattern source (``source``: ``processed``,
    ``unprocessed`` or ``single``) of one measurement of an HDF5 file, stretched to 8 bit by the
    value range of that one pattern. Only that pattern is read; nothing is loaded. The path is
    checked as ``/inspect`` checks it; a measurement or source the file lacks is a coded ``400``
    (``loadVariantUnknown``), a file without patterns ``loadNoPatterns``, a damaged group
    ``loadInspectFailed``."""
    _reject_nul(path)
    try:
        png = await asyncio.to_thread(_preview_png, path, measurement, source)
    except apex.ApexError as exc:
        raise _refusal(exc)
    except load_choices.LoadChoiceError as exc:
        raise problem(400, exc.code, str(exc))
    return Response(content=png, media_type="image/png")


@router.get("/apex/pattern-files")
async def apex_pattern_files(h5_path: str, folder: str):
    """The pattern files of another folder, listed as ``/inspect`` lists those of the metadata
    file's own: ``{"h5_path", "folder", "measurements", "unmatched_pattern_files"}`` with the
    measurement entries of ``/inspect``. A folder that cannot be listed (missing, a file, empty or
    blank, no permission) is a coded ``400`` (``apexFolderUnreadable``), not an empty list."""
    def _work() -> dict:
        _require_apex_file(h5_path)
        return {"h5_path": h5_path, "folder": folder, **_listing(h5_path, folder)}
    _reject_nul(h5_path, folder)
    try:
        return await asyncio.to_thread(_work)
    except apex.ApexError as exc:
        raise _refusal(exc)


@router.get("/apex/preview")
async def apex_preview(path: str, measurement: str, kind: str = "sem"):
    """A small grey PNG of one measurement: ``kind`` is ``sem`` (overview image) or ``iq``."""
    def _work() -> bytes:
        return apex.preview_png(apex.read_measurement(path, measurement), kind)
    _reject_nul(path)
    try:
        png = await asyncio.to_thread(_work)
    except apex.ApexError as exc:
        raise _refusal(exc)
    except Exception as exc:  # noqa: BLE001 - the file changed under us, a damaged table: still a coded answer
        # The answer is the same for every cause, so a bug in the PNG code would hide behind it:
        # leave the traceback (the file's name only, as the answer does).
        logger.exception("APEX preview of %s failed", Path(path).name)
        raise problem(400, "apexUnreadable",
                      f"{Path(path).name}: {type(exc).__name__}: {apex.without_paths(exc, path)}")
    return Response(content=png, media_type="image/png")


@router.post("/apex/check-pairing")
async def apex_check_pairing(req: PairingRequest):
    """Prove that a pattern file the user chose belongs to a measurement (GUID, else count) and
    that the load would not refuse it for its header (a hexagonal flag that contradicts the grid
    type). ``{"ok", "pairing", "name", "n_patterns" (of the measurement), "pattern_shape"
    [rows, cols] (of the file)}``; a refusal is a coded ``400``."""
    def _work() -> dict:
        m = apex.read_measurement(req.h5_path, req.measurement_id)
        how = apex.check_pattern_file(m, req.pattern_path)
        return {"ok": True, "pairing": how, "name": apex.safe_text(Path(req.pattern_path).name),
                "n_patterns": m.expected_patterns,
                "pattern_shape": apex.describe_pattern_file(req.pattern_path)["pattern_shape"]}
    _reject_nul(req.h5_path, req.pattern_path)
    try:
        return await asyncio.to_thread(_work)
    except apex.ApexError as exc:
        raise _refusal(exc)


@router.post("/apex/describe-pattern-file")
async def apex_describe_pattern_file(req: PairingRequest):
    """The pattern file the user chose, as data: the entry ``/inspect`` would list for it
    (``n_patterns``, ``pattern_shape``, ``pairing``, ``count`` = ``fits`` | ``too_few`` |
    ``too_many``, ``names_measurement``, ``names_group`` (a measurement of the same metadata file,
    possibly another than the one asked about), ``matches_grid``, ``problem``). A file that does not
    fit answers ``200`` with its ``problem``, so that the dialog can say "too few" or "too many"
    with the numbers; an unreadable metadata file or an unknown measurement is a coded ``400`` as
    everywhere."""
    def _work() -> dict:
        m = apex.read_measurement(req.h5_path, req.measurement_id)
        groups = {e.id: e.group for e in apex.read_entries(req.h5_path) if e.id}
        return apex.pattern_file_entry(m, req.pattern_path, groups)
    _reject_nul(req.h5_path, req.pattern_path)
    try:
        return await asyncio.to_thread(_work)
    except apex.ApexError as exc:
        raise _refusal(exc)
