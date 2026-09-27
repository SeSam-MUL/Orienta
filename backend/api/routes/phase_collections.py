"""HTTP surface for phase collections.

NO PATH PARAMETERS, anywhere. Two reasons, both measured on the first draft:
`PUT /state` was declared after `PUT /{name}` and was therefore unreachable —
every call landed in the update handler and answered "no collection named
'state'", which silently killed the whole active/hidden mechanism. And a
collection may legitimately be called "Al / Si", which no `{name}` segment can
carry. The name travels in the body of every mutating call and in the query
string of the one read that needs it.

`resolve` is the only endpoint that needs more than the service: turning a
collection key into a Dictionary master is a fuzzy filename search that lives
in routes/indexing.py, and duplicating it would give two answers to one
question. Imported inside the handler to keep the module import cheap and
cycle-free.
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.api.services import phase_collections as pc

logger = logging.getLogger(__name__)
router = APIRouter()


class CreateRequest(BaseModel):
    name: str
    parent: Optional[str] = None
    exclusive: bool = True
    description: str = ""


class UpdateRequest(BaseModel):
    name: str
    parent: Optional[str] = None
    clear_parent: bool = False
    description: Optional[str] = None
    member_keys: Optional[list[str]] = Field(
        default=None,
        description="Full replacement, in the order they should be stored.",
    )


class RenameRequest(BaseModel):
    name: str
    new_name: str


class NameRequest(BaseModel):
    name: str


class MembersRequest(BaseModel):
    name: str
    keys: list[str]
    position: Optional[int] = None


class StateRequest(BaseModel):
    active: Optional[str] = None
    hidden: list[str] = Field(default_factory=list)


class SuggestApplyRequest(BaseModel):
    accepted: list[str] = Field(
        default_factory=list,
        description="Names from POST /suggest that the user wants created.",
    )


def _view(c: pc.PhaseCollection, hidden: set, cols: dict,
          idx: dict, masters: dict) -> dict:
    eff = pc.effective_members(c.name, cols)
    members = pc.annotate(c.members, idx)
    for m in members:
        m["has_master"] = bool(masters.get(m["key"]))
    return {
        "name": c.name,
        "parent": c.parent,
        "exclusive": c.exclusive,
        "description": c.description,
        "hidden": c.name in hidden or (c.parent or "") in hidden,
        "member_count": len(c.members),
        "effective_member_count": len(eff),
        "members": members,
    }


def _master_map(idx: dict) -> dict:
    """key -> master .h5 path, or None. One directory scan for the whole call.

    Best effort: if the Dictionary library cannot be read, every phase simply
    reports no master, which disables the Dictionary button rather than
    claiming an availability we did not verify.
    """
    try:
        from backend.api.routes.indexing import (
            _dictionary_library_files, _master_h5_for_phase,
        )
        files = _dictionary_library_files()
    except Exception:
        logger.debug("phase_collections: dictionary scan failed", exc_info=True)
        return {}
    out = {}
    for key, entry in idx.items():
        try:
            out[key] = _master_h5_for_phase(entry, files)
        except Exception:
            out[key] = None
    return out


@router.get("/")
def list_collections():
    """Everything every picker needs, in one call.

    Loads the collections once and the library index once, then hands both
    down. The first draft re-read every file per collection and rebuilt the
    index per collection.
    """
    cols, problems = pc.load_all()
    by_name = {c.name: c for c in cols}
    idx = pc._index()
    masters = _master_map(idx)
    state = pc.load_state()
    hidden = set(state.get("hidden") or [])
    unassigned = pc.annotate(
        [pc.PhaseMember(key=k) for k in pc.unassigned_keys(idx, cols)], idx)
    for m in unassigned:
        m["has_master"] = bool(masters.get(m["key"]))
    return {
        "collections": [_view(c, hidden, by_name, idx, masters) for c in
                        sorted(cols, key=lambda c: (c.parent or c.name).lower())],
        "unassigned": unassigned,
        "problems": problems + [
            {"kind": "missing_phase", **p}
            for p in pc.propose_repairs(idx, cols)
        ],
        "state": state,
    }


@router.post("/")
def create(req: CreateRequest):
    """Create an empty collection. Refuses a name that is already taken.

    The refusal is the point: without it, creating a collection that already
    exists wrote over its file and its members were gone with no message.
    """
    taken = {c.name.casefold() for c in pc.load_all()[0]}
    if req.name.strip().casefold() in taken:
        raise HTTPException(
            status_code=409,
            detail=f"a collection named {req.name!r} already exists",
        )
    try:
        pc.save(pc.PhaseCollection(
            name=req.name, parent=req.parent, exclusive=req.exclusive,
            description=req.description, members=[],
        ))
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"success": True, "name": req.name}


@router.put("/update")
def update(req: UpdateRequest):
    cols = {c.name: c for c in pc.load_all()[0]}
    if req.name not in cols:
        raise HTTPException(status_code=404,
                            detail=f"no collection named {req.name!r}")
    c = cols[req.name]
    if req.description is not None:
        c.description = req.description
    # `parent: null` means "leave it alone"; promoting a child to the top
    # level needs its own flag, because null cannot say both things.
    if req.clear_parent:
        c.parent = None
    elif req.parent is not None:
        c.parent = req.parent or None
    if req.member_keys is not None:
        have = {m.key: m for m in c.members}
        c.members = [have.get(k) or pc.PhaseMember(key=k) for k in req.member_keys]
    try:
        pc.save(c)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"success": True}


@router.patch("/rename")
def rename(req: RenameRequest):
    try:
        pc.rename(req.name, req.new_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"success": True, "name": req.new_name}


@router.delete("/")
def remove(req: NameRequest):
    """Delete a collection. Removes NO crystal file."""
    if not pc.delete(req.name):
        raise HTTPException(status_code=404,
                            detail=f"no collection named {req.name!r}")
    return {"success": True}


@router.post("/members")
def add_members(req: MembersRequest):
    try:
        pc.assign(req.keys, req.name, req.position)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"success": True}


@router.delete("/members")
def remove_members(req: MembersRequest):
    """Take phases out of THIS collection. Removes nothing from disk, and
    nothing from any other collection — the first draft called the un-file
    everywhere helper here, so removing a phase from A also removed it from B.
    """
    try:
        pc.unassign_from(req.name, req.keys)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"success": True}


@router.put("/state")
def put_state(req: StateRequest):
    pc.save_state({"active": req.active, "hidden": req.hidden})
    return {"success": True, "state": pc.load_state()}


@router.get("/resolve")
def resolve(name: str = Query(...),
            method: Literal["hough", "spherical", "dictionary"] = Query(...)):
    """Collection keys to file paths for one indexing method, in order.

    Both lists matter. A caller that only reads ``paths`` would silently run
    on a subset of the collection the user chose.
    """
    try:
        members = pc.effective_members(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    from backend.api.services.crystal_hint_local_library import get_index
    from backend.api.routes.indexing import (
        _dictionary_library_files, _master_h5_for_phase,
    )
    idx = get_index()
    dict_files = _dictionary_library_files() if method == "dictionary" else []
    paths: list[str] = []
    missing: list[dict] = []
    for m in members:
        e = idx.get(m.key)
        if e is None:
            missing.append({"key": m.key, "reason": "not_in_library"})
            continue
        if method == "hough":
            # Hough computes its reflectors FROM THIS CIF, so a file whose
            # structure could not be read uniquely must not be handed over.
            # Measured on sd_1816951 (MgCu2), whose blocks give pymatgen two
            # compositions: `orix Phase.from_cif(sanitize_cif(...))` is diffpy,
            # not pymatgen, and it SUCCEEDS -- returning 40 sites,
            # {'Mg': 32, 'Cu': 8} = Mg4Cu, while the .xtal EMsoft simulated the
            # master from says Cu 16c + Mg 8b = MgCu2. That is the inverted
            # stoichiometry `cif_phase_library.one_structure` was written to
            # refuse, and the structure factors would be computed from it in
            # silence. `parse_error` is the only thing that knows, so it has to
            # be consulted here rather than only displayed.
            if e.parse_error:
                missing.append({"key": m.key, "reason": "structure_unreadable",
                                "detail": e.parse_error})
                continue
            path, why = e.cif_path, "no_cif"
        elif method == "spherical":
            path, why = e.sht_path, "no_sht"
        else:
            path, why = _master_h5_for_phase(e, dict_files), "no_master"
        if path:
            paths.append(str(path))
        else:
            missing.append({"key": m.key, "reason": why})
    return {"name": name, "method": method, "paths": paths, "missing": missing}


@router.post("/suggest")
def suggest():
    """A proposal. Nothing is written until /suggest/apply."""
    return {"suggestions": pc.suggest()}


@router.post("/suggest/apply")
def suggest_apply(req: SuggestApplyRequest):
    wanted = set(req.accepted)
    taken = {c.name.casefold() for c in pc.load_all()[0]}
    created, skipped = [], []
    for sug in pc.suggest():
        if sug["name"] not in wanted:
            continue
        if sug["name"].casefold() in taken:
            skipped.append(sug["name"])
            continue
        pc.save(pc.PhaseCollection(
            name=sug["name"],
            members=[pc.PhaseMember(key=k) for k in sug["keys"]],
        ))
        created.append(sug["name"])
    return {"success": True, "created": created, "skipped": skipped}
