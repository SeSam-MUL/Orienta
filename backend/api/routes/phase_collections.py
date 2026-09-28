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
    #: Who made this group. Optional, and never invented server-side: this app
    #: has no user identity, so an author is only ever what a caller states.
    #: It matters because the folder is copied and mailed between machines
    #: (spec §2.5.3) -- on the other end, "who was that" has no other answer.
    author: str = ""


class UpdateRequest(BaseModel):
    name: str
    parent: Optional[str] = None
    clear_parent: bool = False
    description: Optional[str] = None
    #: `None` leaves the stored author alone -- the same convention `description`
    #: uses. An empty string is a deliberate clearing.
    author: Optional[str] = None
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


class AddMembersRequest(MembersRequest):
    """`POST /members` only.

    `move` lives here rather than on `MembersRequest`, which `DELETE /members`
    also uses: a field that a second endpoint accepts and ignores is the same
    defect as a button that silently does nothing, and this project has shipped
    that twice (the dead IPF-X/Y buttons, the wirelesss `tolerance` slider).
    """

    #: False adds -- membership is a tag, so filing `Al2Cu` into "Cu systems"
    #: leaves it in "Al systems". True is the named action "move here (remove
    #: from the others)": the ⋯ menu entry and Alt-drag. The decision belongs to
    #: the CALLER; until schema 2 it was derived from the target's `exclusive`
    #: flag, so a drag could not express which of the two it meant.
    move: bool = False


class StateRequest(BaseModel):
    active: Optional[str] = None
    hidden: list[str] = Field(default_factory=list)


class SuggestApplyRequest(BaseModel):
    accepted: list[str] = Field(
        default_factory=list,
        description="Names from POST /suggest that the user wants created.",
    )


def _view(c: pc.PhaseCollection, hidden: set, cols: dict,
          idx: dict, masters: dict, name_of: dict | None = None) -> dict:
    eff = pc.effective_members(c.name, cols)
    members = pc.annotate(c.members, idx)
    for m in members:
        m["has_master"] = bool(masters.get(m["key"]))
    # `hidden` is a set of IDENTITIES, already resolved by the caller: the state
    # file may hold names (written before schema 2) while `c.parent` is always an
    # id, so comparing raw strings here would un-hide a group whose parent the
    # user hid, and say nothing about it.
    #
    # `parent` GOES OUT AS A NAME. Identities are a property of the data at rest,
    # where they solve the real problem: a rename does not break a stored
    # reference. On the wire they solved nothing and broke the client -- every
    # consumer compares `c.parent === c.name` (collectionFilter.js:25,
    # CollectionManager.jsx:295, CollectionPicker.jsx:110), so an id made
    # `activeKeySet` take its "renamed or deleted: filter nothing" branch and the
    # phase filter turned OFF while the toolbar still named a collection. Within
    # one response a name is unambiguous, and the client refetches after every
    # change, so there is nothing for an id to protect here. `id` is sent
    # alongside for whoever wants to store a reference across a rename.
    return {
        "id": c.id,
        "name": c.name,
        "parent": (name_of or {}).get(c.parent, c.parent),
        "exclusive": c.exclusive,
        "description": c.description,
        "hidden": c.id in hidden or (c.parent or "") in hidden,
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
        from backend.api.services.phase_library import dictionary_map, master_map
        # TWO passes over the library, not two per phase: the freshness check
        # stats every candidate `.h5`, so a per-key loop cost 1.45 s for 36
        # phases where the batch costs 0.04 s.
        masters, dicts = master_map(), dictionary_map()
    except Exception:
        logger.debug("phase_collections: master scan failed", exc_info=True)
        return {}
    out = {}
    for key in idx:
        p = masters.get(key) or dicts.get(key)
        out[key] = str(p) if p is not None else None
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
    # Resolved to identities for the comparisons below, then turned back into
    # names for the client (see `_view`). `name_of` is the one translation table.
    state = pc.resolve_state(pc.load_state(), cols)
    hidden = set(state.get("hidden") or [])
    name_of = {c.id: c.name for c in cols if c.id}
    wire_state = {
        **state,
        "active": name_of.get(state.get("active"), state.get("active")),
        "hidden": [name_of.get(h, h) for h in (state.get("hidden") or [])],
    }
    unassigned = pc.annotate(
        [pc.PhaseMember(key=k) for k in pc.unassigned_keys(idx, cols)], idx)
    for m in unassigned:
        m["has_master"] = bool(masters.get(m["key"]))
    return {
        # Grouped by the parent's NAME, not its id: `databaseGrouping.js:98-101`
        # tracks no parents of its own and relies on a child following its
        # parent, and a key mixing ids with names breaks that adjacency
        # (measured ['Al systems', 'Al then', 'Al-Fe phases'] against
        # ['Al systems', 'Al-Fe phases', 'Al then']).
        #
        # The second term is a PRE-EXISTING defect this made visible: a parent
        # and its child produce the SAME first key -- the parent's name -- so the
        # order fell to the stable sort, i.e. to filename order, and
        # `Al-Fe_phases.json` sorts before `Al_systems.json`. The child came
        # first and the adjacency the client depends on was never guaranteed.
        # `0` for a parent, `1` for a child settles it; the third term keeps
        # siblings in a stable, readable order rather than a filesystem one.
        "collections": [_view(c, hidden, by_name, idx, masters, name_of) for c in
                        sorted(cols, key=lambda c: (
                            (name_of.get(c.parent, c.parent) or c.name).lower(),
                            1 if c.parent else 0,
                            c.name.lower()))],
        "unassigned": unassigned,
        "problems": problems + [
            {"kind": "missing_phase", **p}
            for p in pc.propose_repairs(idx, cols)
        ],
        "state": wire_state,
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
            description=req.description, author=req.author, members=[],
        ))
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"success": True, "name": req.name}


@router.put("/update")
def update(req: UpdateRequest):
    # Through `resolve_ref` like the other four write routes. This was the ONE
    # that still did an exact, case-sensitive dict lookup -- not even casefold --
    # so it answered 404 for an identity AND for a name whose casing differed,
    # while `PATCH /rename` on the same collection accepted both. Nesting is the
    # thing this route does, so the mismatch fell on the one action f7 had just
    # built. Found by f7's closing review.
    cols = pc.load_all()[0]
    c = pc.resolve_ref(req.name, cols)
    if c is None:
        raise HTTPException(status_code=404,
                            detail=f"no collection named {req.name!r}")
    if req.description is not None:
        c.description = req.description
    if req.author is not None:
        c.author = req.author
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
def add_members(req: AddMembersRequest):
    try:
        pc.assign(req.keys, req.name, req.position, move=req.move)
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
    """Set the active collection and the hidden ones.

    The caller may send an identity or a name -- the frontend has both in hand,
    and a name is what a hand-written request or an older build will send. It is
    stored as the identity, so a later rename does not break it.
    """
    cols = pc.load_all()[0]
    state = pc.resolve_state({"active": req.active, "hidden": req.hidden}, cols)
    pc.save_state(state)
    # Stored as identities, answered as NAMES and in the same shape `GET /`
    # uses -- including `schema`, which this endpoint used to drop because
    # `resolve_state`'s `{**state, ...}` only carries what it was handed. A
    # client that read one response and wrote the other would have seen two
    # different shapes for one thing.
    name_of = {c.id: c.name for c in cols if c.id}
    return {"success": True, "state": {
        "schema": pc.COLLECTION_SCHEMA,
        "active": name_of.get(state.get("active"), state.get("active")),
        "hidden": [name_of.get(h, h) for h in (state.get("hidden") or [])],
    }}


@router.get("/resolve")
def resolve(name: str = Query(...),
            method: Literal["hough", "spherical", "dictionary"] = Query(...)):
    """Collection keys to file paths for one indexing method, in order.

    Both lists matter. A caller that only reads ``paths`` would silently run
    on a subset of the collection the user chose.
    """
    # `name` is a reference: an identity, or a name for one more version. An
    # indexing run started from a saved identity must not fail because somebody
    # renamed the collection in between -- that is the whole reason ids exist.
    cols, _ = pc.load_all()
    target = pc.resolve_ref(name, cols)
    if target is None:
        raise HTTPException(status_code=404,
                            detail=f"no collection named {name!r}")
    members = pc.effective_members(target.name, {c.name: c for c in cols})
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
