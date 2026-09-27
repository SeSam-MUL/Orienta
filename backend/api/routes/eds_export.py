"""HTTP for the EDS export and the analysis presets.

Thin on purpose. Everything that decides a number lives in
``services/eds_export.py`` and ``services/eds_presets.py``, which take data
rather than requests; this file only turns the loaded file into that data and
turns exceptions into status codes. The split is what lets a future batch
driver walk a folder without going through FastAPI, and it is the reason the
batch feature could be deferred honestly instead of silently omitted
(spec section 2).

WHY IT REUSES ``routes/eds.py``. ``_build_at_pct_maps_for_loaded_file`` and
``_eds_step_um`` are imported, not reimplemented. The dominant defect class in
this repo is derived state cached in two places with only one of them
maintained; a second step-size reader that
drifted from the first would put one length in the phase map and a different
one in the export of the same phase map, and nothing would ever fail.

Contract: docs/superpowers/specs/2026-08-27-eds-presets-and-export-design.md
          section 8, frozen — the frontend builds against it.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from backend.api.services import eds_presets as presets_svc
from backend.api.services.eds_export import (
    ExportGeometry,
    ExportOptions,
    build_tables,
    write_export,
)
from backend.api.services.eds_particles import CONNECTIVITY, find_particles
from backend.api.services.phase_map_store import get_phase_map_store

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# shared plumbing
# ---------------------------------------------------------------------------

def _state_or_400():
    state = get_phase_map_store().get_state()
    if state is None:
        raise HTTPException(
            status_code=400,
            detail="No phase map to export — classify the EDS map first.")
    return state


def _at_maps():
    """The At.% maps for the loaded file, through the one existing builder."""
    from backend.api.routes.eds import _build_at_pct_maps_for_loaded_file
    return _build_at_pct_maps_for_loaded_file()


def _step_um():
    from backend.api.routes.eds import _eds_step_um
    return _eds_step_um()


def _geometry(state, file_path: Optional[str]) -> ExportGeometry:
    step_x, step_y = _step_um()
    return ExportGeometry(
        n_rows=int(state.n_rows), n_cols=int(state.n_cols),
        step_x_um=step_x, step_y_um=step_y,
        source_path=file_path,
    )


def _live_colour_overrides() -> Dict[str, str]:
    """The phase colours the page is showing RIGHT NOW.

    Read off the attribute at call time, never bound at import: ``POST
    /api/eds/phase-map/colors`` REBINDS that module global, so
    ``from ... import _COLOR_OVERRIDES`` would freeze whatever it held when
    this module was first imported and the exported picture would quietly use
    yesterday's colours. ``{}`` on any failure, which is the same palette the
    page shows when nothing has been overridden.
    """
    try:
        from backend.api.routes import eds as eds_routes
        return dict(getattr(eds_routes, "_COLOR_OVERRIDES", None) or {})
    except Exception:
        logger.debug("eds export: colour overrides unavailable", exc_info=True)
        return {}


def _guard_grid(state, n_rows: int, n_cols: int) -> None:
    """Refuse to export a map that does not belong to the loaded file.

    The store holds one map and the session holds one file, and they can
    disagree — a file switch clears the map, but a map restored from a sidecar
    next to a differently-shaped scan would not. Exporting that would produce
    a table of areas and compositions in which every row is a mix of two
    datasets, and nothing about the file would say so.
    """
    if int(state.n_rows) != int(n_rows) or int(state.n_cols) != int(n_cols):
        raise HTTPException(
            status_code=400,
            detail=(f"The stored phase map is {state.n_rows}x{state.n_cols} "
                    f"but the loaded file is {n_rows}x{n_cols}. Re-classify "
                    f"before exporting — mixing the two would put numbers "
                    f"from two datasets in the same row."))


# ---------------------------------------------------------------------------
# compatibility: recomputed here, never adopted from the caller
# ---------------------------------------------------------------------------
#
# WHY IT IS RECOMPUTED AND NOT VALIDATED. A service-lab tester posted
#
#     {"ok": true, "preset_hash": "DEADBEEF",
#      "_forged": "this report was computed against a DIFFERENT steel scan"}
#
# as the ``compatibility`` field of ``POST /api/eds/export`` and got a 200,
# with that object written verbatim into ``provenance.json`` -- junk key and
# all -- beside a ``source.path`` naming another scan. The frontend guard was
# the only defence and the backend never looked, although it knew
# ``preset_name`` and had every input the check needs sitting in
# :func:`check_preset` a few lines below.
#
# A VALIDATION would have to get every field right INCLUDING THE ONES THAT DO
# NOT EXIST YET: she did not only forge ``ok``, she invented a key no validator
# would have thought to reject. A RECOMPUTATION has to be right once, and the
# whole class of invented keys stops mattering, because the posted object stops
# being an input to the record at all.
#
# For a file whose entire purpose is defensibility, an unvalidated attestation
# is the wrong shape.

#: Attached to a client-supplied report that survived into the record.
_CLIENT_REPORT_NOTE = (
    "The compatibility report the CLIENT posted with this export. It is kept "
    "because it disagrees with the one recomputed here, and it is a record of "
    "what the caller believed -- never the verdict. The fields beside it (ok, "
    "blockers, warnings) were recomputed on this server against the preset "
    "named above and the scan named in source.path. Both are shown on purpose: "
    "quietly reconciling them would hide the disagreement, which is the one "
    "thing worth knowing when there is one. Projected onto the fields a "
    "compatibility report actually has: a caller cannot write a key of its "
    "own naming into this file, and 'undeclared_fields_dropped' counts what "
    "was left out."
)

_SERVER_REPORT_NOTE = (
    "Recomputed on the server at export time, against the preset named here "
    "and the scan in source.path -- its measured elements, its dominant "
    "element, the phase keys this machine's CIF library actually holds and its "
    "step size. Whatever the caller posted as 'compatibility' was discarded "
    "for this record. 'override' in the block above is derived from the 'ok' "
    "below, so an override documents the objection this server raised rather "
    "than the caller's account of it; 'override_requested' says whether the "
    "caller itself acknowledged a refusal."
)


def _compat_inputs() -> Dict[str, Any]:
    """The facts about the CURRENTLY LOADED scan a compatibility check needs.

    ONE reader for both callers -- ``POST /presets/{name}/check`` and the
    export. A second one that drifted would let the dialog show a verdict the
    export then contradicts, which is the dominant defect class in this repo:
    derived state cached twice, with only one copy maintained.
    """
    at_maps, _n_rows, _n_cols, _fp = _at_maps()
    step_x, _step_y = _step_um()

    dominant = ""
    try:
        from backend.api.services.chemistry_score import infer_matrix_element
        dominant = infer_matrix_element(at_maps) or ""
    except Exception:
        logger.debug("compatibility: matrix inference unavailable", exc_info=True)

    # ``None`` means "the library could not be read", which the compatibility
    # check must treat differently from "the library has no such phase" --
    # refusing every preset because the database has not been built yet would
    # be a refusal about the wrong thing.
    phase_keys = None
    try:
        from backend.api.routes.eds import _crystal_db_path
        from backend.api.services.cif_phase_library import load_cif_phase_library
        db = _crystal_db_path()
        if db.is_file():
            phase_keys = list(load_cif_phase_library(db).keys())
    except Exception:
        logger.debug("compatibility: CIF library unavailable", exc_info=True)

    return {
        "measured_elements": list(at_maps.keys()),
        "available_phase_keys": phase_keys,
        "dominant_element": dominant,
        "step_um": step_x,
    }


def _verdict(report: Optional[dict]) -> tuple:
    """The decision-bearing part of a report, for comparing two of them.

    Codes and the hash, not the prose: messages are English sentences this same
    module wrote, and a wording change between two builds is not a
    disagreement. A different ``ok``, a different blocker or a report about
    different preset content is.
    """
    r = report or {}
    return (
        bool(r.get("ok", True)),
        str(r.get("preset_hash") or ""),
        tuple(sorted(str(b.get("code", "")) for b in (r.get("blockers") or []))),
        tuple(sorted(str(w.get("code", "")) for w in (r.get("warnings") or []))),
    )


#: The fields a compatibility report is made of. Anything else a caller
#: sends is not part of the format.
_REPORT_KEYS = ("ok", "preset_name", "preset_hash", "blockers", "warnings")


def _client_claim(blob: dict) -> dict:
    """The caller's report, projected onto the shape a report actually has.

    Undeclared keys are DROPPED and only their NUMBER survives. The forged
    object carried ``_forged``, a key no validator would have rejected because
    none would think to expect it; copying it into ``provenance.json`` would
    let a caller write a sentence of its own choosing into the evidence, under
    a name of its own choosing — the same defect one level down, in the very
    field added to contain it.

    The count is the part that is a fact about the request ("the caller sent 2
    fields this format does not define"). Their contents are a fact about
    nothing. Blockers and warnings keep ``code`` and ``message``, which is what
    makes a disagreement legible; ``detail`` is free-form by design and is not
    needed to show one.
    """
    out: Dict[str, Any] = {}
    for k in _REPORT_KEYS:
        if k not in blob:
            continue
        v = blob[k]
        if k in ("blockers", "warnings"):
            items = v if isinstance(v, (list, tuple)) else []
            v = [{"code": str((i or {}).get("code", "")),
                  "message": str((i or {}).get("message", ""))}
                 for i in items if isinstance(i, dict)]
        out[k] = v
    dropped = sum(1 for k in blob if k not in _REPORT_KEYS)
    if dropped:
        out["undeclared_fields_dropped"] = dropped
    return out


def _unverifiable(reason: str, client: Optional[dict],
                  override_requested: bool) -> dict:
    """A record for the case where nothing could be recomputed.

    Deliberately carries NO ``ok`` key. An unverifiable claim is not a pass and
    not a failure, and the export's ``override`` flag reads ``ok`` with a
    default of ``True`` -- so an absent key says "no refusal was established
    here" rather than inventing either verdict.
    """
    out: Dict[str, Any] = {
        "computed_by": "none",
        "computed_note": reason,
        "override_requested": bool(override_requested),
    }
    if client is not None:
        out["compatibility_client_reported"] = _client_claim(client)
        out["compatibility_client_reported_note"] = _CLIENT_REPORT_NOTE
    return out


def _compatibility_for_provenance(
    preset_name: str, client: Optional[dict],
) -> Optional[dict]:
    """What ``provenance.preset.compatibility`` says about this run.

    ``None`` when no preset was named and none was claimed -- "no preset was
    involved" has to stay distinguishable from "one was applied and it fitted".

    WHERE THE EXTRA KEYS LIVE. ``provenance["preset"]`` is assembled by
    ``services/eds_export.py`` as ``{name, compatibility, override, note}``, and
    this dict is the one part of it written verbatim. So everything this route
    knows and that block has no slot for rides inside it:
    ``provenance.preset.compatibility.preset_version`` (the number a human
    quotes in an email; the content hash beside it is the stronger identifier)
    and ``.compatibility_client_reported``.
    """
    client_blob = dict(client) if isinstance(client, dict) else None
    # A real fact about what the user CHOSE, and the only part of the posted
    # object worth keeping: they were shown a refusal and went on anyway.
    # Their account of WHAT they overrode is not kept -- that is the server's
    # to state.
    override_requested = bool(
        client_blob is not None and not client_blob.get("ok", True))

    name = str(preset_name or "").strip()
    if not name:
        if client_blob is None:
            return None
        return _unverifiable(
            "No preset was named in the export request, so there was nothing "
            "to recompute. The caller sent a compatibility report anyway; it "
            "is recorded here as a claim, not as a verdict.",
            client_blob, override_requested)

    try:
        preset = presets_svc.load_preset(name)
    except Exception as exc:
        # NOT an error for the export: the map is real, and refusing to write
        # it because a preset name no longer resolves would destroy work over a
        # bookkeeping problem. It is recorded, and loudly.
        logger.warning("eds export: preset %r could not be resolved for the "
                       "compatibility record: %s", name, exc)
        return _unverifiable(
            f"No preset named {name!r} could be read on this machine "
            f"({exc}), so its compatibility with this scan could not be "
            f"recomputed. Nothing here was checked.",
            client_blob, override_requested)

    try:
        report = presets_svc.check_compatibility(preset, **_compat_inputs())
    except Exception as exc:                                   # pragma: no cover
        logger.warning("eds export: compatibility recomputation failed: %s", exc)
        return _unverifiable(
            f"The compatibility check could not be run against the loaded "
            f"scan ({exc}). Nothing here was checked.",
            client_blob, override_requested)

    record: Dict[str, Any] = dict(report.as_dict())
    server_verdict = _verdict(record)
    record["computed_by"] = "server"
    record["computed_note"] = _SERVER_REPORT_NOTE
    record["preset_version"] = int(preset.version)
    record["override_requested"] = override_requested
    if client_blob is not None:
        agrees = _verdict(client_blob) == server_verdict
        record["client_report_agrees"] = agrees
        if not agrees:
            # Only when it differs. An identical copy would be noise, and the
            # flag above already says the two agreed.
            record["compatibility_client_reported"] = _client_claim(client_blob)
            record["compatibility_client_reported_note"] = _CLIENT_REPORT_NOTE
    return record


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

class ExportArtefacts(BaseModel):
    #: A field this model does not declare is a TYPO, not an extension. A PhD
    #: tester sent ``scale_px: 3`` to another EDS endpoint, got a 200, and ran
    #: 60 scans smoothed at the default width he meant to change. Silence is
    #: the failure mode: a rejected request costs him a minute, an accepted one
    #: is a wrong analysis he never sees.
    model_config = ConfigDict(extra="forbid")

    phases: bool = True
    regions: bool = True
    particles: bool = True
    definitions: bool = True
    xlsx: bool = True
    #: ``phase_map.png`` and, when the map has regions, ``region_map.png``.
    #: ON by default, and additive to the frozen contract — a client that
    #: sends the old artefact set without this key gets the pictures, which
    #: is the point. A reader who had to find a wrong headline number by
    #: reading a CSV said a picture would have caught it in four seconds, and
    #: the alternative on offer was a right-click on another page into another
    #: dialog with five decisions she had no opinion about.
    map_png: bool = True
    labels: bool = False
    pixels: bool = False


class ExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dest_dir: str
    artefacts: ExportArtefacts = ExportArtefacts()
    decimal: str = "."
    delimiter: str = ","
    connectivity: int = CONNECTIVITY
    min_particle_px: int = 0
    preset_name: str = ""
    compatibility: Optional[dict] = None
    # --- additive, optional; a client that ignores these gets the frozen
    # contract of spec section 8 unchanged ---------------------------------
    #: Smoothing width the map was classified with. Omitted -> the smoothed
    #: composition columns are omitted, rather than recomputed at a default
    #: width that may not be the one the regions were formed on.
    scale: Optional[int] = None
    scale_um: Optional[float] = None
    #: Echo of what the UI asked for, so "requested" can sit beside "resolved"
    #: in the provenance for values the store does not persist.
    requested: Dict[str, Any] = {}
    #: sha256 of the source costs a full read — tens of seconds on a 27 GB
    #: scan. Turning it off is recorded in the provenance, never silent.
    hash_source: bool = True
    on_existing: str = "suffix"


@router.get("/export/preview")
async def export_preview():
    """A cheap dry run, so the dialog can say "1 842 particles" before writing.

    Deliberately runs the real particle finder rather than estimating: the
    particle count is the number the dialog exists to show, and an estimate
    that differs from the file it is previewing is worse than no preview.
    """
    state = _state_or_400()
    at_maps, n_rows, n_cols, file_path = _at_maps()
    _guard_grid(state, n_rows, n_cols)

    step_x, step_y = _step_um()
    warnings: List[Dict[str, str]] = []

    if state.region_grid is not None:
        grid = np.asarray(state.region_grid, dtype=np.int32)
        basis = "region"
    else:
        grid = np.asarray(state.phase_grid, dtype=np.int32)
        basis = "phase"
        warnings.append({
            "code": "particles_from_phases",
            "message": ("This map has no regions, so particles will be "
                        "connected components of the phase grid."),
        })

    particles = find_particles(grid, connectivity=CONNECTIVITY)
    n_total = int(state.n_rows) * int(state.n_cols)
    n_classified = int((np.asarray(state.phase_grid) >= 0).sum())
    area_available = bool(step_x and step_y)
    if not area_available:
        warnings.append({
            "code": "no_step_size",
            "message": ("This file carries no step size, so every column in "
                        "micrometres will be omitted."),
        })

    source_bytes = None
    if file_path:
        try:
            source_bytes = int(Path(file_path).stat().st_size)
        except OSError:
            source_bytes = None

    return {
        "ok": True,
        "n_phases": int(len({int(v) for v in
                             np.unique(np.asarray(state.phase_grid))
                             if int(v) >= 0})),
        "n_regions": int(len(state.region_phase or [])),
        "n_particles": len(particles),
        "n_classified_px": n_classified,
        "n_total_px": n_total,
        "step_x_um": step_x,
        "step_y_um": step_y,
        "area_available": area_available,
        "connectivity": CONNECTIVITY,
        "particle_basis": basis,
        # So the dialog can warn before asking for a sha256 of a 27 GB file.
        "source_path": file_path,
        "source_bytes": source_bytes,
        "warnings": warnings,
    }


@router.post("/export")
async def export(req: ExportRequest):
    """Write the export folder. Never into one that already exists."""
    state = _state_or_400()
    at_maps, n_rows, n_cols, file_path = _at_maps()
    _guard_grid(state, n_rows, n_cols)

    dest = Path(req.dest_dir).expanduser()
    if not dest.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Destination folder does not exist: {req.dest_dir}")

    # Smoothing: resolved through the SAME helper the classification uses, so
    # "5 px" here means what it meant there. Asked for only when the caller
    # states a width — `_resolve_scale(None, None)` would happily return the
    # default, and a column labelled "smoothed" that was smoothed at a width
    # nobody chose is worse than no column.
    scale_px = None
    scale_report = None
    if req.scale is not None or req.scale_um is not None:
        from backend.api.routes.eds import _resolve_scale
        scale_px, scale_report = _resolve_scale(req.scale, req.scale_um)

    opts = ExportOptions(
        connectivity=int(req.connectivity),
        min_particle_px=int(req.min_particle_px),
        decimal=req.decimal,
        delimiter=req.delimiter,
        artefacts=req.artefacts.model_dump() if hasattr(req.artefacts, "model_dump")
        else req.artefacts.dict(),
        on_existing=req.on_existing,
        hash_source=bool(req.hash_source),
        preset_name=req.preset_name or "",
        colour_overrides=_live_colour_overrides(),
        # RECOMPUTED, never echoed: `req.compatibility` is the caller's account
        # and is discarded for the record. See _compatibility_for_provenance.
        compatibility=_compatibility_for_provenance(
            req.preset_name or "", req.compatibility),
        smoothing_scale_px=scale_px,
        scale_report=scale_report,
        requested=dict(req.requested or {}),
        timestamp=datetime.now(),
    )
    try:
        opts.validate()
    except ValueError as exc:
        # Locale combinations land here. A German Excel reading "3.14" as 314
        # is a wrong number in a report, so this refuses rather than warns.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        tables = build_tables(state, at_maps, geometry=_geometry(state, file_path),
                              options=opts)
        result = write_export(tables, dest, options=opts, source_path=file_path)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:                                   # pragma: no cover
        logger.exception("EDS export failed")
        raise HTTPException(status_code=500,
                            detail=f"Export failed: {exc}") from exc

    return {
        "ok": True,
        "folder": result.folder,
        "files": [{"name": f.name, "rows": f.rows, "bytes": f.bytes,
                   "width": f.width, "height": f.height}
                  for f in result.files],
        "warnings": result.warnings,
        "provenance": result.provenance,
    }


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------

class PresetSaveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    settings: dict
    overwrite: bool = False
    author: str = ""
    notes: str = ""
    material_class: str = ""
    tags: List[str] = []
    authored_elements: List[str] = []
    authored_step_um: Optional[float] = None
    matrix_element: str = ""


class PresetImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    json_text: str
    save: bool = False
    overwrite: bool = False


def _preset_meta(p: presets_svc.EdsPreset) -> dict:
    d = p.to_dict()
    d["content_hash"] = p.content_hash
    return d


def _authored_against_loaded_file(
    elements: List[str], step_um: Optional[float],
) -> tuple[List[str], Optional[float]]:
    """Fill the portability record from the loaded scan when the caller omits it.

    The UI sends both, but a preset authored through the raw API, a script or
    curl would otherwise be born with no ``authored_elements`` and no
    ``authored_step_um`` — and those two fields are the entire input to
    ``element_set_differs`` and ``step_size_differs``. A preset with no guards
    is the one that travels furthest before anybody notices.

    OMITTED, NEVER GUESSED. If no file is open, or the file carries no step
    size, the field stays empty: "authored against Al, Si at 0.65 um" is a
    factual claim that will be read months later as evidence, and inventing it
    is worse than leaving it blank — the blank is itself reported, by
    ``portability_not_checkable``.

    Deliberately not applied on import: a colleague's preset was authored
    against THEIR scan, and stamping ours onto it would forge exactly the
    provenance the field exists to carry.
    """
    if not elements:
        try:
            at_maps, _n_rows, _n_cols, _fp = _at_maps()
            elements = [str(k) for k in at_maps.keys()]
        except Exception:
            # No file open (HTTPException 400) or no EDS data — leave it empty.
            logger.debug("preset save: no loaded scan to record elements from",
                         exc_info=True)
    if step_um is None:
        try:
            step_x, _step_y = _step_um()
            step_um = step_x
        except Exception:
            logger.debug("preset save: no loaded scan to record a step from",
                         exc_info=True)
    return list(elements), step_um


@router.get("/presets")
async def list_presets():
    return {"presets": [_preset_meta(p) for p in presets_svc.list_presets()]}


@router.get("/presets/{name}")
async def get_preset(name: str):
    try:
        return {"preset": _preset_meta(presets_svc.load_preset(name))}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/presets")
async def save_preset(req: PresetSaveRequest):
    """Write a preset to the user directory.

    ``hints`` is additive and optional — a client that ignores it gets the
    frozen contract of spec section 8 unchanged. It carries the offer to pin
    the phase list, which is deliberately an OFFER: pinning here would save a
    recipe the user did not write ("the phases my library held on the day I
    pressed Save" is not "all of them"), so the choice stays with them and only
    the consequence of not choosing is made visible.
    """
    authored_elements, authored_step_um = _authored_against_loaded_file(
        list(req.authored_elements or []), req.authored_step_um)
    try:
        p = presets_svc.preset_from_settings(
            req.name, req.settings,
            author=req.author, notes=req.notes,
            material_class=req.material_class, tags=req.tags,
            authored_elements=authored_elements,
            authored_step_um=authored_step_um,
            matrix_element=req.matrix_element,
        )
        path = presets_svc.save_preset(p, overwrite=req.overwrite)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    hints: List[dict] = []
    if not p.phase_list_pinned:
        hints.append({
            "code": "phase_list_not_pinned",
            "message": (
                "This preset does not name the phases it may choose from, so "
                "it will run against whatever the CIF library holds at the "
                "time — a CIF added next month changes its results without "
                "changing its content hash. Save it again with the phase list "
                "pinned to make it reproducible."
            ),
            "detail": {"pinned": False},
        })
    return {"ok": True, "path": str(path), "preset": _preset_meta(p),
            "hints": hints}


@router.delete("/presets/{name}")
async def delete_preset(name: str):
    try:
        deleted = presets_svc.delete_preset(name)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail=f"no EDS preset named {name!r}")
    return {"ok": True}


@router.post("/presets/{name}/check")
async def check_preset(name: str):
    """Can this recipe mean here what it meant where it was written?

    Evaluated against the CURRENTLY LOADED FILE — its measured elements, its
    dominant element, the phase keys the library actually has, its step size.
    This is what the Apply button gates on: ``ok: false`` means the UI must
    refuse and show ``blockers[].message``, and proceeding requires an
    explicit override that is then passed back into ``POST /api/eds/export``
    as ``compatibility`` so the refusal lands in ``provenance.json``.

    A warning that only appears on screen is clicked away in half a second;
    that is the whole reason the report is shaped to be storable.
    """
    try:
        preset = presets_svc.load_preset(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # The SAME inputs the export recomputes with, through the same reader, so
    # the verdict the dialog shows and the verdict the file records cannot be
    # two different answers to one question.
    report = presets_svc.check_compatibility(preset, **_compat_inputs())
    return report.as_dict()


@router.post("/presets/import")
async def import_preset(req: PresetImportRequest):
    try:
        p = presets_svc.import_preset_json(req.json_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    saved_path = None
    if req.save:
        try:
            saved_path = str(presets_svc.save_preset(p, overwrite=req.overwrite))
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "preset": _preset_meta(p), "path": saved_path}


@router.get("/presets/{name}/export")
async def export_preset(name: str):
    """The bytes to mail to a colleague.

    BYTE-IDENTICAL to the file on disk for a preset this build wrote. It was
    not: ``save_preset`` went through Python's text mode and landed CRLF on
    Windows while this returned LF, so a tester who diffed a mailed copy
    against her saved one saw all 81 lines differ with no content changed.
    Both are LF now (``_write_atomic``), and a test pins the equality so the
    two cannot drift apart again.

    ``json_text`` is the CANONICAL serialisation of the preset as loaded, not a
    copy of the file's bytes. For anything this build saved the two are the
    same string; a file hand-edited into a different key order, or written by
    an older build that did not yet record a derived field, is normalised on
    the way out. That is deliberate — the recipe is what travels, not the
    formatting — and ``newline`` below says what separator it carries so a
    reader never has to guess at a diff.
    """
    try:
        p = presets_svc.load_preset(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "json_text": presets_svc.export_preset_json(p),
        "filename": f"{presets_svc.safe_filename(p.name)}.json",
        # Additive, and stated rather than implied: the whole bug was two
        # halves of one feature disagreeing about this single character.
        "newline": "\n",
        "note": ("UTF-8, LF line endings, keys sorted — byte-identical to the "
                 "file this build saved for the same preset."),
    }
