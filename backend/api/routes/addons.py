"""The add-on surface: what is installed, what the user allows, and running one.

This file is written in two halves, matching the two tasks that built it:
listing and enabling first, then running and map bytes. Each half imports only
what it uses -- an import that names something the file does not touch is how a
reader learns to stop trusting the import block.

ONE NUMBER, ONE MEANING
-----------------------
Every refusal carries a machine-readable ``reason`` beside its sentence. Two
409s that differ only by an English sentence force a client to string-match to
tell "this add-on needs a newer Orienta" from "two add-ons claim this name",
and the sentence is the part most likely to be reworded or translated; the same
is true of the two different 404s a run can answer. Inventing private status
numbers would answer that by lying to every HTTP intermediary about what the
number means, so the code goes in the BODY, beside the sentence it explains.
The sentence stays -- it is what a user and a bug report can both read.

THE LISTING IMPORTS NOTHING
---------------------------
``discover_addons`` reads manifests off disk and resolves entry points with
``find_spec``, which locates a package without executing it. Nothing in
:func:`list_addons` may change that. Orienta's start-up already imports
kikuchipy, orix, diffsims and torch; an add-on whose code runs merely to
answer "what is installed?" joins that path, and every user pays for it
whether or not they ever enable the add-on. The failure mode is silent --
nothing breaks, the app just gets slower and arbitrary code runs earlier --
which is why ``test_the_listing_never_imports_the_addons_module`` measures
``sys.modules`` rather than trusting this paragraph.

Importing happens in exactly two places: here, on an explicit enable, and in
the runner, on an explicit run.

ONE NAME, ONE ADD-ON
--------------------
Consent is recorded by name, so two installations claiming one name share one
row in the trust store and a decision about either governs both. That is the
runner's module collision a level up -- one author's decision applied to
another author's code -- and it gets the same answer: the listing SHOWS the
conflict on both rows, and enabling is refused with both locations named.
Orienta does not choose which of them the user meant.

ONE ANALYSIS KEY, ONE ADD-ON
----------------------------
The same shape a third time, and the one that reaches a manuscript. An
analysis key is what a ``StepCitation`` is looked up by, so it carries the
DOIs and the methods sentence: two add-ons declaring one key means whichever
registered first lends its citations and its prose to the other's numbers, in
the BibTeX, in the methods paragraph and in the exported .h5.
``citations_bridge`` refuses that registration outright; this file shows the
same collision STATICALLY, on every row that declares the key, so a user meets
it in the listing rather than through a failed run. Refused rather than picked,
for the reason a duplicated name is: choosing would be choosing whose work gets
cited.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from backend.api.services.addons.context import ContextError, build_context
from backend.api.services.addons.discovery import (
    addon_search_dirs,
    discover_addons,
    extra_addon_dirs,
)
from backend.api.services.addons import jobs
from backend.api.services.addons.manifest import API_VERSION
from backend.api.services.addons.outputs import MapOutput, TableOutput
from backend.api.services.addons.runner import (
    AddonError,
    probe_import,
    run_analysis,
)
from backend.api.services.addons.trust import (
    CRASH_LIMIT,
    TrustStore,
    orienta_version_satisfies,
    running_orienta_version,
)
from backend.api.services.pattern_quality import get_quality_map

logger = logging.getLogger(__name__)
router = APIRouter()

#: Where main.py mounts this router, and what a map output's ``values_url`` is
#: built from -- so a drift between the two would mint links that 404.
#:
#: ``test_the_mount_prefix_is_the_constant_other_code_builds_urls_from`` walks
#: ``app.routes`` and pins the listing route's real path to this constant, so
#: the two cannot drift. The earlier note here claimed a test that did not
#: exist yet (8b's), which is a claim with nothing behind it.
API_PREFIX = "/api/addons"

#: Not 422. FastAPI answers its OWN request-validation failures with 422, and
#: a caller that cannot tell "your request was malformed" from "the add-on's
#: code raised while being imported" will report the second as the first. 424
#: Failed Dependency says what happened: the request was fine, the thing it
#: depends on was not.
#:
#: EVERY such answer uses it, not only the probe. The run route raised 422 for
#: an add-on that raised and for a result the context could not use, so the two
#: routes disagreed with each other over one class of outcome -- and the
#: collision is not cosmetic: a request-validation answer carries NO ``reason``
#: at all (``main.py`` normalises the body to a sentence plus ``errors``), so a
#: client sharing the number with it has nothing to branch on but English.
ADDON_FAILED_STATUS = 424

#: Machine-readable reasons. Stable strings, not numbers: they are read by a
#: front end and printed in a bug report, and a number would have to be looked
#: up somewhere else to mean anything.
REASON_ADDON_NOT_FOUND = "addon_not_found"
REASON_ADDON_NAME_CONFLICT = "addon_name_conflict"
REASON_ANALYSIS_KEY_CONFLICT = "analysis_key_conflict"
REASON_ADDON_INCOMPATIBLE = "addon_incompatible"
REASON_ADDON_NOT_ENABLED = "addon_not_enabled"
REASON_ADDON_IMPORT_FAILED = "addon_import_failed"
REASON_ADDON_FAILED = "addon_failed"
REASON_RESULT_NOT_FOUND = "result_not_found"
REASON_RESULT_UNUSABLE = "result_unusable"
REASON_MAP_NOT_STORED = "map_not_stored"
REASON_TRUST_STORE_UNWRITABLE = "trust_store_unwritable"
REASON_JOB_NOT_FOUND = "job_not_found"
REASON_JOB_UNFINISHED = "job_unfinished"


class AddonRefusal(HTTPException):
    """An HTTPException that also says, in one stable token, WHICH refusal.

    Still an ``HTTPException``: if one ever escapes a route that forgot the
    decorator below, FastAPI answers with the same status and the same
    ``detail`` as before, minus the code. Degrading to the old shape is the
    right failure here -- the alternative, a bare 500, would hide a refusal the
    user could have acted on.
    """

    def __init__(self, status_code: int, reason: str, detail: str):
        super().__init__(status_code=status_code, detail=detail)
        self.reason = reason


def _refusal_body(exc: AddonRefusal) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code,
                        content={"detail": exc.detail, "reason": exc.reason})


def _with_reason_code(fn):
    """Answer an :class:`AddonRefusal` with a body that carries its reason.

    A decorator, and not a change to ``detail``: making ``detail`` a dict would
    move the sentence behind a key, and every existing caller and test reads
    ``r.json()["detail"]`` as the text to show a person. An app-level exception
    handler would do the same job, but it lives in ``main.py``, which this
    router has no business editing for its own error shape.

    ``functools.wraps`` sets ``__wrapped__``, which ``inspect.signature`` --
    and therefore FastAPI dependency analysis -- follows, so the wrapper is
    typed exactly as the route it wraps.
    """
    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def _async(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except AddonRefusal as exc:
                return _refusal_body(exc)
        return _async

    @functools.wraps(fn)
    def _sync(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except AddonRefusal as exc:
            return _refusal_body(exc)
    return _sync


class EnabledRequest(BaseModel):
    # An unknown field must not read as success. Without this, a caller that
    # sends {"enabled": true, "force": true} gets 200 and believes "force"
    # did something; so does one that misspells a field it will add later.
    # Accepting silently is the same dishonesty as dropping an add-on from
    # the listing -- the caller cannot tell it was ignored.
    model_config = ConfigDict(extra="forbid")

    enabled: bool


def _extra_dirs() -> List[Path]:
    # Delegated: ``discovery`` already owns ``$ORIENTA_ADDON_USER_DIR``, and
    # "where add-ons live" read in two places is one fact held twice.
    # ``citations_bridge`` needs the same answer and must not import a route.
    return extra_addon_dirs()


def _trust_store() -> TrustStore:
    # No argument: TrustStore defaults to trust_store_path(), which honours
    # $ORIENTA_ADDON_TRUST_FILE. Hardcoding Path.home() here made the test
    # suite write into the developer's real home directory.
    return TrustStore()


def _compatibility(manifest) -> Tuple[bool, str]:
    """(satisfied, a sentence a person can act on)."""
    running = running_orienta_version()
    if not manifest.requires_orienta:
        return True, ""
    if orienta_version_satisfies(manifest.requires_orienta, running):
        return True, ""
    if running is None:
        return False, (
            f"{manifest.name} requires Orienta {manifest.requires_orienta}, and "
            "this build carries no release tag to compare against (it is a "
            "source checkout). Install a tagged release, or ask the author to "
            "drop the requirement."
        )
    return False, (
        f"{manifest.name} requires Orienta {manifest.requires_orienta}; this "
        f"build is {running}."
    )


def _duplicate_sources(found) -> Dict[str, List[str]]:
    """name -> every source path claiming it, for names claimed more than once.

    Consent is keyed by NAME. Two add-ons installed under one name therefore
    share one row in the trust store: a user who agreed to a known-good
    ``grain-stats`` from Zenodo has, without being asked, also agreed to a
    ``grain-stats`` that arrived in a zip. That is the runner's module
    collision one level up -- one author's decision applied to another
    author's code -- and it is answered the same way: named, not guessed at.
    """
    by_name: Dict[str, List[str]] = {}
    for d in found:
        if d.manifest is not None:
            by_name.setdefault(d.manifest.name, []).append(
                str(d.source_path) if d.source_path else "<unknown location>")
    return {name: paths for name, paths in by_name.items() if len(paths) > 1}


def _duplicate_analysis_keys(found) -> Dict[str, List[str]]:
    """analysis key -> every add-on NAME declaring it, for keys claimed twice.

    By name, and de-duplicated by name, because two installations under one
    name are already reported as a name conflict and would otherwise be
    reported twice for the same fact.

    An analysis key is the unit of academic credit: ``render_methods`` looks a
    ``StepCitation`` up by key at render time, so the key decides which DOI and
    which sentence describe a recorded run. ``citations_bridge`` refuses the
    registration when a key is already another add-on's; this is the static
    half of the same rule, and it is the half a user can act on -- it needs
    nothing to have run yet.
    """
    by_key: Dict[str, List[str]] = {}
    for d in found:
        if d.manifest is None:
            continue
        for analysis in d.manifest.analyses:
            names = by_key.setdefault(analysis.key, [])
            if d.manifest.name not in names:
                names.append(d.manifest.name)
    return {key: sorted(names) for key, names in by_key.items()
            if len(names) > 1}


def _key_conflict_sentence(key: str, names: List[str]) -> str:
    return (
        f"the analysis key {key!r} is declared by {len(names)} different "
        "add-ons: " + ", ".join(names)
        + ". A key carries the citations and the methods sentence, so "
        "whichever ran first would lend its DOI and its wording to the "
        "other's results. Ask one of the authors to choose a different key, "
        "or remove one add-on; Orienta will not choose."
    )


def _key_conflicts_for(manifest, key_conflicts: Dict[str, List[str]]
                       ) -> List[str]:
    """One sentence per declared key that another add-on also claims."""
    return [_key_conflict_sentence(a.key, key_conflicts[a.key])
            for a in manifest.analyses if a.key in key_conflicts]


def _conflict_sentence(name: str, paths: List[str]) -> str:
    return (
        f"{len(paths)} add-ons are installed under the name {name!r}: "
        + "; ".join(sorted(paths))
        + ". Consent is recorded by name, so enabling one would enable the "
        "other as well -- one author's decision applied to another author's "
        "code. Remove or rename one of them; Orienta will not choose."
    )


def _disabled_by(name: str, trust) -> Optional[str]:
    """Who switched this add-on off: ``"user"``, ``"runtime"``, or nobody.

    NOT a third boolean. Three states exist and two of them share
    ``enabled: false``, so a field that can be None is the honest shape.

    An add-on nobody has decided about is ``None``, not ``"user"``: a freshly
    installed one reads ``enabled: false`` as well, and calling that the user's
    decision would be the first thing every new add-on said about itself, in
    the user's name.

    ``crashes >= CRASH_LIMIT`` is read rather than ``==``: the counter is a
    read-modify-write on a file that two processes can share, and a row that
    stopped saying "runtime" because the count overshot by one would send the
    user looking for a decision they never made.
    """
    if trust.is_enabled(name):
        return None
    recorded = trust.disabled_by(name)
    if recorded:
        return recorded
    # Only for trust files written before the field existed. Inferring is what
    # this used to do always, and it got the failed-activation case wrong:
    # Orienta writes a row itself when an add-on raises on activation, and
    # that row is byte for byte a user's own switch-off.
    if trust.crashes(name) >= CRASH_LIMIT:
        return "runtime"
    return "user" if trust.is_known(name) else None


def _describe(discovered, trust, duplicates: Dict[str, List[str]],
              key_conflicts: Dict[str, List[str]]) -> dict:
    m = discovered.manifest
    source = str(discovered.source_path) if discovered.source_path else ""
    if m is None:
        # Shown as rejected, never dropped: an add-on the user installed and
        # that simply does not appear is indistinguishable from one that was
        # never installed. Same keys as a good row, so the table that renders
        # it cannot be taken down by the row that most needs to be seen.
        #
        # display_name falls back to the path because a rejected manifest has
        # no name to show, and two rejected rows both reading "" are
        # indistinguishable in the one place the user has to act: the list.
        # Show what there is.
        return {"name": "", "display_name": source or "(unknown location)",
                "version": "", "authors": [],
                "doi": "", "origin": discovered.origin, "source_path": source,
                "enabled": False, "known": False, "compatible": False,
                "compatibility": discovered.error, "conflict": "",
                "error": discovered.error, "analyses": [],
                "known_version": "", "known_doi": "",
                # Same keys as a good row, these three included: a table that
                # reads them unconditionally must not be taken down by the one
                # row the user has to act on. It was never run, so it never
                # crashed, and nobody switched it off.
                "crashes": 0, "crash_limit": CRASH_LIMIT, "disabled_by": None}
    compatible, reason = _compatibility(m)
    paths = duplicates.get(m.name) or []
    # ONE field, both kinds. A row says everything that stands between this
    # add-on and a run, and a second field would let a reader who knows about
    # the first one miss the second.
    conflicts = ([_conflict_sentence(m.name, paths)] if paths else [])
    conflicts += _key_conflicts_for(m, key_conflicts)
    return {
        "name": m.name,
        "display_name": m.display_name,
        "version": m.version,
        "authors": list(m.authors),
        "doi": m.doi,
        "origin": discovered.origin,
        # The spec's trust dialog shows where it came from ON DISK.
        "source_path": source,
        "enabled": trust.is_enabled(m.name),
        "known": trust.is_known(m.name),
        # WHAT the recorded decision was about. A name is not an identity:
        # delete an add-on and drop a different one with the same name in its
        # place, and ``known`` still says yes while nothing else notices --
        # only one is installed, so the duplicate guard is silent. Shipping
        # both lets the page compare them against the manifest it is showing
        # and ask again when they differ.
        "known_version": trust.known_version(m.name),
        "known_doi": trust.known_doi(m.name),
        # WHY it is switched off, which "enabled: false" cannot say. Three
        # failures disable an add-on automatically, so the same false means
        # either "you decided" or "we decided" -- and a page that shows only
        # the switch reports the second as the first, putting a decision in
        # the user's mouth that they never made.
        #
        # The counter is what separates them, and it separates them cleanly
        # because ``set_enabled`` zeroes it: user-disabled is (false, 0),
        # runtime-disabled is (false, CRASH_LIMIT).
        #
        # The limit travels WITH the count: "2 crashes" means nothing without
        # knowing where the cut is, and a front end that hard-codes 3 is a
        # second copy of a runtime constant that goes stale in silence.
        "crashes": trust.crashes(m.name),
        "crash_limit": CRASH_LIMIT,
        "disabled_by": _disabled_by(m.name, trust),
        "compatible": compatible,
        "compatibility": reason,
        # Non-empty when another installation claims this same name, or
        # another add-on claims one of these analysis keys. The row is still
        # shown -- hiding the conflict would hide the thing the user has to
        # resolve.
        "conflict": " ".join(conflicts),
        "error": "",
        "analyses": [{"key": a.key, "label": a.label,
                      "params_schema": a.params_schema} for a in m.analyses],
    }


@router.get("")
def list_addons():
    trust = _trust_store()
    found = discover_addons(extra_dirs=_extra_dirs())
    duplicates = _duplicate_sources(found)
    key_conflicts = _duplicate_analysis_keys(found)
    return {"api_version": API_VERSION,
            "orienta_version": running_orienta_version(),
            # WHERE IT LOOKED. An empty list is the first thing a user meets,
            # and one that cannot name a folder sends them outside the app to
            # find out that ~/.orienta/addons is the answer. From discovery,
            # which composes the same list it reads.
            "searched_paths": [str(p) for p in
                               addon_search_dirs(_extra_dirs())],
            "addons": [_describe(d, trust, duplicates, key_conflicts)
                       for d in found]}


def _find(name: str) -> Tuple[object, Dict[str, List[str]],
                             Dict[str, List[str]]]:
    """The one add-on with this name, and both conflict maps of the listing.

    Refuses rather than picks when two claim the name: picking one would
    write a consent decision that also governs the other, which is the whole
    defect. Both source paths are named, the way the runner's identity check
    names both locations.
    """
    found = discover_addons(extra_dirs=_extra_dirs())
    duplicates = _duplicate_sources(found)
    matches = [d for d in found
               if d.manifest is not None and d.manifest.name == name]
    if not matches:
        raise AddonRefusal(404, REASON_ADDON_NOT_FOUND,
                           f"No add-on named {name!r}.")
    if len(matches) > 1:
        raise AddonRefusal(
            409, REASON_ADDON_NAME_CONFLICT,
            _conflict_sentence(name, duplicates.get(name) or []))
    return matches[0], duplicates, _duplicate_analysis_keys(found)


def _require_compatible(manifest) -> None:
    compatible, reason = _compatibility(manifest)
    if not compatible:
        raise AddonRefusal(409, REASON_ADDON_INCOMPATIBLE, reason)


def _require_no_key_conflict(manifest, key_conflicts: Dict[str, List[str]]
                             ) -> None:
    """Refuse while ANY key this manifest declares is also another add-on's.

    EVERY key, on the run path too, and an earlier round had it otherwise --
    the gate asked about the key being run while
    ``register_manifest_citations`` loops over the whole manifest. So an
    add-on declaring one clean key beside one conflicting key passed the gate,
    was imported, RAN, and was then refused by the bridge: the work discarded
    after it had succeeded, and the caller told ``addon_failed``, which blames
    the add-on's code for something its manifest did. A gate that asks a
    narrower question than the thing it guards is not a gate.

    On the enable path this sits beside ``_require_compatible`` and inside the
    ``if body.enabled`` branch, for that branch's reason: switching an add-on
    OFF must never depend on it being in good standing, or the one a user most
    needs to switch off is the one they cannot.
    """
    sentences = _key_conflicts_for(manifest, key_conflicts)
    if sentences:
        raise AddonRefusal(409, REASON_ANALYSIS_KEY_CONFLICT,
                           " ".join(sentences))


def _store_failure(name: str, enabled: bool, trust: TrustStore,
                   exc: Exception) -> str:
    """One sentence about a decision that could not be written down.

    The runner's rule -- no failure leaves this system as a raw exception --
    does not stop at the runner. A trust store on a read-only directory, a
    full disk or a file another process holds open would otherwise surface as
    a 500 with an empty body and a traceback in the log that names neither the
    add-on nor the file.
    """
    verb = "enabled" if enabled else "disabled"
    return (f"{name} could not be {verb}: the add-on trust store at "
            f"{trust.path} could not be written "
            f"({type(exc).__name__}: {exc}). The decision was not recorded; "
            f"decisions already in the file are unchanged.")


def _record(name: str, enabled: bool, manifest, trust: TrustStore) -> None:
    """Write the decision down, or say plainly that it was not written.

    ``Exception``, not an enumeration. The first round caught ``OSError``
    only and argued a ``TypeError`` from ``json.dump`` "cannot occur today";
    it could, via an unquoted TOML date, and it escaped as an unattributed
    500 -- so the enumeration was wrong on its first contact with reality.
    The suggested ``(OSError, TypeError, ValueError)`` is the same kind of
    bet one item longer: a surrogate in a path raises ``UnicodeEncodeError``,
    a recursive structure raises ``RecursionError``, neither is in it.
    Attribution does not depend on knowing which one happened, and the
    traceback is still logged, so nothing is hidden by catching broadly.

    ``BaseException`` is deliberately NOT caught, for ``runner.py``'s reason:
    ``KeyboardInterrupt`` is the operator stopping the process. Unlike the
    runner, ``SystemExit`` is not caught either -- the code running here is
    Orienta's own, so an exit is the operator too, not a third party's
    script-shaped mistake.
    """
    try:
        trust.set_enabled(name, enabled, version=manifest.version,
                          doi=manifest.doi)
    except Exception as exc:
        logger.exception("could not record the add-on decision for %s", name)
        raise AddonRefusal(
            500, REASON_TRUST_STORE_UNWRITABLE,
            _store_failure(name, enabled, trust, exc)) from exc


@router.post("/{name}/enabled")
@_with_reason_code
def set_enabled(name: str, body: EnabledRequest):
    discovered, duplicates, key_conflicts = _find(name)
    manifest = discovered.manifest
    trust = _trust_store()
    if body.enabled:
        # Checked BEFORE the probe, so an incompatible add-on is never
        # imported to find out it is incompatible.
        _require_compatible(manifest)
        _require_no_key_conflict(manifest, key_conflicts)
        try:
            probe_import(manifest)
        except AddonError as exc:
            # "An add-on that raises on activation is disabled and says so."
            # Written down as disabled, not merely left alone: for an add-on
            # that WAS enabled -- yesterday's install, broken by today's
            # update -- those are different outcomes, and only one of them is
            # what the sentence promises.
            detail = str(exc)           # the add-on's own name comes first
            try:
                # by="runtime": the user pressed ENABLE. Recording this as
                # their decision is how the listing came to tell them they
                # had switched off the add-on they were switching on.
                trust.set_enabled(name, False, version=manifest.version,
                                  doi=manifest.doi, by="runtime")
            except Exception as store_exc:      # see _record for the width
                # Two failures at once. The add-on's is the one the user can
                # act on, so ours is appended rather than allowed to replace
                # it -- a 500 here would hide the import error entirely.
                logger.exception(
                    "could not record the failed activation of %s", name)
                detail = f"{detail} {_store_failure(name, False, trust, store_exc)}"
            raise AddonRefusal(ADDON_FAILED_STATUS,
                               REASON_ADDON_IMPORT_FAILED, detail) from exc
    # Disabling deliberately does NOT probe: switching an add-on off must
    # never depend on being able to import it, or the one a user most needs
    # to switch off is the one they cannot.
    _record(name, body.enabled, manifest, trust)
    return _describe(discovered, trust, duplicates, key_conflicts)


# ---------------------------------------------------------------------------
# Running an add-on, and serving a map's bytes
# ---------------------------------------------------------------------------

#: Map values from recent runs, keyed (addon, result_id, ANALYSIS key, output
#: key). Same idea and same lifetime as ``indexing._result_registry``: in
#: memory, capped, oldest evicted. A map does not travel in the JSON response
#: (see the serialisation note on :func:`map_values`), so it has to wait here
#: for the client to fetch it.
#:
#: THE ANALYSIS KEY IS PART OF THE KEY, and that is not tidiness. A map is
#: meant to become a LAYER in the existing stack, so its URL is held across
#: runs and refreshed -- not fetched once and forgotten. Without the analysis
#: key, running analysis B of the same add-on on the same result replaces
#: analysis A's slot, and A's layer then draws B's numbers under A's label,
#: with matching shape and dtype so that nothing looks wrong. This package has
#: already shipped two defects of exactly that shape -- a module name
#: collision that ran the wrong add-on's code, and a name collision that let
#: one add-on inherit another's consent -- and both were "just a key".
class StoredMap(NamedTuple):
    """A stored map AND what a legend needs to explain it.

    The store kept the ndarray alone, so ``unit``, ``vmin`` and ``vmax`` were
    discarded the moment ``_serialise`` had written them into the JSON. The
    numbers survived in the response; what they MEAN did not, and the layer
    painter is a second reader that needs them. Kept beside the values rather
    than re-derived: bounds an add-on declared are a statement about its own
    output, and re-measuring the array would quietly replace them.
    """
    values: np.ndarray
    unit: Optional[str]
    vmin: Optional[float]
    vmax: Optional[float]


_MAP_VALUES: "OrderedDict[Tuple[str, str, str, str], StoredMap]" = OrderedDict()

#: Guards every WRITE to the store above. It had one writer -- the event loop --
#: until the job route put ``_serialise`` in a worker thread. See
#: ``_remember_map`` for the measurement.
_MAP_LOCK = threading.Lock()

#: How many maps the store holds before the oldest of ANOTHER run is dropped.
MAX_STORED_MAPS = 24


class RunRequest(BaseModel):
    # Same reason as EnabledRequest: a field we do not understand must not
    # read as success. ``params`` is the add-on's own dict and stays open --
    # what an analysis declares is the manifest's business, and the runner
    # reports back everything it did not declare.
    model_config = ConfigDict(extra="forbid")

    analysis_key: str
    result_id: str
    params: Dict[str, Any] = {}


def _map_url(name: str, result_id: str, analysis_key: str, key: str) -> str:
    return f"{API_PREFIX}/{name}/outputs/{result_id}/{analysis_key}/{key}"


def _evict_other_runs(current: Tuple[str, str, str]) -> None:
    """Make room, taking only from runs OTHER than the one being stored.

    The plain "drop the oldest" form could eat its own output: an add-on
    returning more than ``MAX_STORED_MAPS`` maps evicted its first one DURING
    its own serialisation, and the 200 then advertised a ``values_url`` that
    404s with "run the analysis again" -- a remedy that cannot ever work,
    because running it again evicts it again. Handing a caller an instruction
    guaranteed to fail is worse than any storage cost.

    The alternative considered and rejected: cap the number of maps per run and
    refuse the rest in ``validate_outputs``. That turns a storage detail into a
    rule an add-on author cannot discover until they hit it, in a package whose
    whole point is that outsiders write against it -- and it would refuse a
    legitimate result (one map per phase is an ordinary shape) for the
    convenience of this dict. An eviction policy that cannot eat its own output
    costs nothing anybody has to learn.

    HONEST LIMIT: a single run of more than ``MAX_STORED_MAPS`` maps therefore
    leaves the store OVER its cap until another run displaces it. The cap is a
    bound on how many old maps are kept, not a hard ceiling on memory; a run
    big enough to matter was already holding those arrays a moment earlier, in
    the add-on.
    """
    while len(_MAP_VALUES) > MAX_STORED_MAPS:
        # OrderedDict iterates oldest-first, so this is the oldest entry that
        # does not belong to the run now being stored.
        victim = next((k for k in _MAP_VALUES if k[:3] != current), None)
        if victim is None:
            return
        del _MAP_VALUES[victim]


def _remember_map(name: str, result_id: str, analysis_key: str,
                  output: MapOutput) -> np.ndarray:
    """Keep a map's values until the client fetches them; return what is kept.

    A COPY, not the array the add-on returned. ``validate_outputs`` freezes a
    read-only VIEW, and ``view.base`` is the add-on's own buffer and stays
    writable -- outputs.py says so and pins it. Between the run answering and
    the client following the URL, an add-on that kept a reference could
    therefore change the bytes served under a link this run minted, and
    nothing would say so. The values have to be held in memory either way, so
    the copy costs one transient duplicate of one map and buys the guarantee
    that the bytes are the run's own.

    C-contiguous, because ``shape`` + ``dtype`` + a flat byte stream is the
    whole contract the client reads it back with.

    UNDER A LOCK, since the job route: this was a read-modify-write sequence
    (insert, move_to_end, then scan and delete) that had exactly one writer for
    as long as ``_serialise`` only ever ran on the event loop. The job worker is
    a second one. Measured with two inserters and two scanners against this
    dict: ``RuntimeError: OrderedDict mutated during iteration``, raised out of
    ``_evict_other_runs``'s generator -- which on the job path would be recorded
    as the ADD-ON's failure, and on the synchronous path is a bare 500, because
    ``_with_reason_code`` catches only ``AddonRefusal``. The array copy stays
    outside the lock; only the dict work is serialised.
    """
    key = (name, result_id, analysis_key, output.key)
    stored = np.array(output.values, order="C")
    with _MAP_LOCK:
        _MAP_VALUES[key] = StoredMap(stored, output.unit,
                                     output.vmin, output.vmax)
        _MAP_VALUES.move_to_end(key)
        _evict_other_runs(key[:3])
    return stored


def _serialise(output, *, name: str, result_id: str,
               analysis_key: str) -> dict:
    """One output as JSON the browser can parse, plus a URL for a map's bytes.

    Nothing here coerces or guards: ``validate_outputs`` has already made every
    field a plain Python object that ``json.dumps(..., allow_nan=False)``
    accepts, and re-checking here would be a second opinion that can disagree
    with the first. A map's ``values`` are the one field that never appears.
    """
    if isinstance(output, MapOutput):
        values = _remember_map(name, result_id, analysis_key, output)
        return {"kind": "map", "key": output.key, "label": output.label,
                "unit": output.unit, "vmin": output.vmin, "vmax": output.vmax,
                "shape": [int(d) for d in values.shape],
                "dtype": values.dtype.str,
                "n_bytes": int(values.nbytes),
                "values_url": _map_url(name, result_id, analysis_key,
                                       output.key)}
    if isinstance(output, TableOutput):
        return {"kind": "table", "key": output.key, "label": output.label,
                "columns": list(output.columns),
                "rows": [list(r) for r in output.rows]}
    return {"kind": "scalar", "key": output.key, "label": output.label,
            "value": float(output.value), "unit": output.unit}


def _resolve_step_size_um(result):
    """The repo's own resolver, imported lazily.

    routes/indexing is heavy and this module is imported at start-up, so the
    import happens per call -- the same reason ``_result_registry`` is fetched
    inside :func:`run_addon`. A resolver that raises costs the FIELD, not the
    run: it reaches into the loaded-signal machinery, which can fail for
    reasons that have nothing to do with the add-on, and ``None`` is an answer
    the contract already has a meaning for.
    """
    from backend.api.routes.indexing import _resolve_step_size_um as _resolve

    try:
        return _resolve(result)
    except Exception:
        logger.debug("step size could not be resolved for this result",
                     exc_info=True)
        return None


def _context_extras(result) -> Dict[str, Any]:
    """The optional data a context can carry, resolved from a stored result.

    Quality comes from ``pattern_quality.get_quality_map``, the repo's
    documented single source, with ``allow_compute=False``: computing FFT image
    quality needs the loaded signal and would turn an add-on run into minutes
    of silent EBSD work the caller never asked for.

    ``quality_source`` carries the map's LABEL, not its ``source``: "Band
    Contrast (native)" names the metric AND the provenance, where "native"
    names only the provenance -- and Oxford band contrast and EDAX image
    quality are different measurements an add-on may want to distinguish.

    ``step_size_um`` is RESOLVED, not read from metadata: nothing in product
    code writes ``metadata["step_size_um"]``, so reading only there would make
    the field None on every real result. None stays None -- a wrong step
    silently mis-scales every length an add-on computes, so it is shown as
    missing rather than defaulted to 1.

    ``eds_at_pct`` stays empty. The per-pixel EDS quantification lives behind
    the EDS service and is not wired here; the field exists, is exercised by
    the context tests, and an add-on reading it on a real run today gets {} --
    said here rather than implied.

    ``report`` is NOT in here, and the difference is which route is running.
    It is not context data resolved from the result; it is a sink bound to one
    run, so it is threaded through ``_context_for(result, report=...)``.
    :func:`start_addon_job` passes one that writes into the job registry --
    the fixture add-on's ``context.report("scaling", 0.5)`` reaches the poll,
    and ``test_the_addons_progress_reaches_the_poll`` pins it.
    :func:`run_addon` passes none, so ``build_context`` installs
    ``_noop_report`` and a synchronous run's progress is still discarded: that
    route answers only when the work is over, so there is nobody to tell.
    """
    extras: Dict[str, Any] = {}
    step = _resolve_step_size_um(result)
    if step is not None:
        extras["step_size_um"] = step

    shape = tuple(int(d) for d in (getattr(result, "original_shape", ()) or ()))
    metadata = getattr(result, "metadata", None) or {}
    if len(shape) != 2:
        return extras
    try:
        quality = get_quality_map(
            shape[0], shape[1],
            source_file=metadata.get("source_file"),
            signal=None,
            xmap=getattr(result, "xmap", None),
            allow_compute=False,
        )
    except Exception:
        logger.debug("no quality map for this result", exc_info=True)
        quality = None
    # Shape-checked, as every other caller in this repo does: the native
    # branch returns a FULL-SCAN array for a cropped result.
    if quality is not None and np.asarray(quality.array).shape == shape:
        extras["quality"] = np.asarray(quality.array, dtype=np.float64)
        extras["quality_source"] = quality.label
    return extras


@router.get("/{name}/outputs/{result_id}/{analysis_key}/{key}")
@_with_reason_code
async def map_values(name: str, result_id: str, analysis_key: str, key: str):
    """The raw bytes of a map output, C-contiguous, in the dtype the run
    reported.

    BYTES RATHER THAN JSON, for three reasons in order of weight. (a) An
    all-NaN map -- which ``validate_outputs`` allows on purpose, an unindexed
    pixel being genuinely absent -- has no valid JSON representation at all:
    Python writes the bare token ``NaN`` and ``JSON.parse`` rejects it, so
    nested lists would make the honest output the one that breaks the client.
    (b) A 300x400 float64 map is ~2.4 MB of JSON text built one Python float at
    a time, for data the front end turns straight back into a typed array.
    (c) A map is meant to become a layer in the existing stack, and holding the
    values in a small process-level store is what lets a later plan hand them
    to that stack without changing this contract.

    ``validate_outputs`` refuses a non-numeric map dtype, which is what keeps
    this endpoint from serving the heap addresses of an object array's
    contents -- measured once as HTTP 200 over eight-byte pointers.

    The add-on must still be installed and still be enabled. Not a security
    boundary -- an add-on runs in this process with the user's permissions and
    the CORS allowlist is three localhost origins -- but a decision to switch
    an add-on off should stop its data being served, and the alternative is a
    store that quietly outlives an uninstall.
    """
    stored = await _stored_map_or_refuse(name, result_id, analysis_key, key)
    return Response(content=stored.values.tobytes(),
                    media_type="application/octet-stream")


async def _stored_map_or_refuse(name: str, result_id: str, analysis_key: str,
                                key: str) -> StoredMap:
    """The gate both map routes pass, and the lookup behind it.

    ONE copy, so the bytes and the painted layer cannot disagree about which
    add-on is allowed to serve which map, or about what an evicted map says.

    The add-on must still be installed and still be enabled. Not a security
    boundary -- an add-on runs in this process with the user's permissions and
    the CORS allowlist is three localhost origins -- but a decision to switch
    an add-on off should stop its data being served, and the alternative is a
    store that quietly outlives an uninstall.

    OFFLOADED, for :func:`run_addon`'s reason and with a number behind it:
    ``_find`` re-reads every manifest under $ORIENTA_ADDON_DIRS -- measured at
    37 ms for five add-ons on a local disk, and those directories may be a
    network mount -- and ``TrustStore()`` reads a JSON file. On the event loop,
    on every byte fetch, that is the health check stalling behind a layer
    refresh.

    NOT memoised, deliberately. A cached catalogue is a second copy of a fact
    that lives on disk, and this branch's commonest defect is exactly that:
    state held twice with one copy updated. The stale directions are both wrong
    in a way nobody could debug -- an add-on installed mid-session invisible, or
    an uninstalled one still serving its bytes.
    """
    await asyncio.to_thread(_find, name)    # installed, and not one of two
    trust = await asyncio.to_thread(_trust_store)
    if not trust.is_enabled(name):
        raise AddonRefusal(
            409, REASON_ADDON_NOT_ENABLED,
            f"The add-on {name!r} is not enabled, so its stored outputs are "
            "no longer served. Switch it on again under Add-ons and fetch "
            "them.")
    stored = _MAP_VALUES.get((name, result_id, analysis_key, key))
    if stored is None:
        raise AddonRefusal(
            404, REASON_MAP_NOT_STORED,
            f"No stored map {key!r} from {analysis_key!r} of {name!r} on "
            f"result {result_id!r}. Map values are kept in memory until "
            "another run displaces them; run the analysis again.")
    return stored


#: The colormap every add-on map is painted with. An add-on declares none, and
#: viridis is perceptually uniform and matplotlib's sequential default. Stated
#: as a choice, and as a limit, in CHANGELOG-ADDON-API.md -- including the case
#: where it is wrong: a LABEL map (the reference add-on's component rank) reads
#: as a gradient that is not there.
ADDON_MAP_CMAP = "viridis"


def _paint_map(stored: StoredMap) -> Tuple[str, dict]:
    """A stored map as a transparent-background PNG, plus its legend.

    Modelled on ``phase_map._diag_rgba``, the LAYER painter, and deliberately
    not on the other renderer in that module, which bakes in a colorbar, a
    title and a background -- it would look almost right and could never be
    stacked.

    Two changes from the model, both load-bearing:

    * masked with ``np.isfinite``, not ``~np.isnan``. An add-on may legitimately
      return an infinity, and under the model's mask it would enter the
      normalisation, make ``vmax`` infinite and flatten every finite pixel to
      one colour -- a layer that looks like data and is not.
    * ``unit or ""``, because ``MapOutput.unit`` is optional and the reference
      add-on sets none: the legend would otherwise read "None".

    Declared bounds are USED, not only reported. An add-on that says a map runs
    0..1 is describing its own output, and re-measuring the array would quietly
    replace that statement with what this particular scan happened to contain.
    """
    import base64
    import io

    import matplotlib
    from PIL import Image

    # Lazily, and from the layer painter's own module, so the legend is built
    # by the SAME function that builds every other layer's -- a second sampler
    # of the same colormap is a second thing that can drift.
    from backend.api.routes.phase_map import _scale_info

    arr = np.asarray(stored.values, dtype=float)
    finite = np.isfinite(arr)
    if finite.any():
        lo, hi = float(arr[finite].min()), float(arr[finite].max())
    else:
        # Permitted by the output contract (every pixel unindexed), so it
        # renders fully transparent rather than failing. The legend still gets
        # numbers; an empty one would be a second thing to explain.
        lo, hi = 0.0, 1.0
    vmin = lo if stored.vmin is None else float(stored.vmin)
    vmax = hi if stored.vmax is None else float(stored.vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        # Defensive, and it repairs BOTH: an earlier version widened vmax and
        # left vmin, so a -inf bound produced scale {min: -inf, max: -inf} and
        # Starlette's json.dumps(allow_nan=False) turned the whole response
        # into a 500. A guard that cannot do its job is worse than none.
        # ``_as_json_number`` already refuses a non-finite bound on the way in,
        # so this should be unreachable; it is here because "should be" is not
        # a property of a stored value read back later.
        vmin, vmax = lo, hi
    if vmax <= vmin:
        # Two cases reach here, and the earlier comment claimed only the
        # first. (a) Degenerate DATA, or a label map with one label, where
        # painting one colour is the truth. (b) A SINGLE declared bound the
        # data does not straddle -- vmin=10 on a map running 0..5 -- where
        # the legend then reads "10.0 to 10.000001" and says nothing useful.
        # ``_validate_map`` cannot catch (b): it compares the two bounds, and
        # here one of them is ours. Refusing it would be wrong anyway, since
        # fixing one end is how an add-on makes two scans comparable, and
        # inventing the other end would replace its statement with a guess.
        # Named as a limit, and pinned by a test, rather than left implied.
        vmax = vmin + 1e-6

    # Non-finite pixels are painted (as ``norm`` 0, i.e. the colormap's darkest
    # colour) and THEN made transparent, where the model leaves them NaN and
    # lets matplotlib's "bad" colour paint them transparent black. Identical
    # under the canvas compositing the page does, since alpha is 0 either way.
    # It differs only for a reader that ignores alpha and keys on black -- the
    # front end has such a path (``useLayerStack``'s blackToAlpha), which no
    # add-on layer uses today. Named so that a future one does not discover it.
    norm = np.zeros(arr.shape, dtype=float)
    np.divide(arr - vmin, vmax - vmin, out=norm, where=finite)
    rgba = (matplotlib.colormaps[ADDON_MAP_CMAP](np.clip(norm, 0.0, 1.0))
            * 255).astype(np.uint8)
    rgba[..., 3] = np.where(finite, 255, 0).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG", optimize=False)
    scale = _scale_info(ADDON_MAP_CMAP, vmin, vmax, unit=stored.unit or "")
    return base64.b64encode(buf.getvalue()).decode("utf-8"), scale


@router.get("/{name}/outputs/{result_id}/{analysis_key}/{key}/image")
@_with_reason_code
async def get_addon_map_image(name: str, result_id: str, analysis_key: str,
                              key: str):
    """An add-on's map as a layer the page can already draw.

    ``{image, scale}``, the shape ``phaseMapApi.layer`` answers in, because
    ``fetchLayerImage`` consumes exactly that -- a new shape here would mean a
    second decoder in the front end for the same picture.

    The bytes route stays: it is the honest numbers, and a caller that wants to
    compute with a map rather than look at it should not have to decode a PNG.
    """
    stored = await _stored_map_or_refuse(name, result_id, analysis_key, key)

    def _work():
        # The same lock the layer route holds. matplotlib is not thread-safe,
        # and this backend paints layers from more than one thread.
        from backend.api.routes.phase_map import _mpl_lock

        with _mpl_lock:
            return _paint_map(stored)

    image, scale = await asyncio.to_thread(_work)
    return {"image": image,
            "shape": [int(d) for d in np.asarray(stored.values).shape],
            "scale": scale}


def _context_for(result, report=None):
    """``build_context`` plus the extras, as one call something can offload.

    ``report`` is the progress sink the add-on calls. It is threaded here
    rather than into ``run_analysis`` because the add-on reaches it as
    ``context.report`` -- ``build_context`` is the only seam it has.
    """
    return build_context(result, report=report, **_context_extras(result))


async def _gates_for_run(name: str, body: RunRequest):
    """The five gates every run passes, and the three things it needs after.

    ONE copy, shared by the synchronous route and the job route. Each gate
    raises the refusal it owns, so both routes answer a disabled, incompatible,
    conflicting, unknown or unusable case identically -- and, for the job
    route, answer it SYNCHRONOUSLY: a refusal is not a failed job. A job whose
    add-on was never even reached would have to be polled to learn that
    nothing is going to happen.

    Order matters and is not alphabetical: compatibility and key conflicts are
    checked BEFORE the result lookup and before any import, so an add-on this
    build refuses is never imported to find out that it is refused, and an
    analysis whose key another add-on also declares never does work that could
    not be recorded honestly.

    Everything that touches a disk is offloaded -- ``_find`` reads every
    manifest under ``$ORIENTA_ADDON_DIRS``, which may be a network mount, and
    ``TrustStore()`` reads a JSON file. Both block the event loop, and
    therefore the health check, which has made this app look dead before.
    """
    from backend.api.routes.indexing import _result_registry

    discovered, _duplicates, key_conflicts = await asyncio.to_thread(
        _find, name)
    manifest = discovered.manifest
    trust = await asyncio.to_thread(_trust_store)
    if not trust.is_enabled(name):
        raise AddonRefusal(
            409, REASON_ADDON_NOT_ENABLED,
            f"The add-on {name!r} is not enabled. Switch it on under "
            "Add-ons, after checking who wrote it -- it runs with your "
            "permissions.")
    # Again here, and not only on enable: the realistic case is a decision
    # recorded when this build DID satisfy the requirement, on an Orienta that
    # no longer does.
    _require_compatible(manifest)
    # EVERY declared key, not only the one asked for: ``register_manifest_
    # citations`` registers the whole manifest, so a narrower gate here lets
    # the add-on do the work and lose it.
    _require_no_key_conflict(manifest, key_conflicts)

    result = _result_registry.get(body.result_id)
    if result is None:
        raise AddonRefusal(
            404, REASON_RESULT_NOT_FOUND,
            f"No result {body.result_id!r}. Results live in memory only and "
            "are lost on eviction or a backend restart.")
    return manifest, trust, result


@router.post("/{name}/run")
@_with_reason_code
async def run_addon(name: str, body: RunRequest):
    """Run one analysis of one add-on against one stored result, and wait.

    Kept beside ``/run-job``, which does the same work as a polled job, because
    a caller that can wait should not have to poll: this is the route a script,
    a test and a second add-on use. The page uses the job route, for the reason
    written there.

    EVERYTHING THAT TOUCHES A DISK IS OFFLOADED, not only the add-on call. The
    gates do their own (see ``_gates_for_run``), and ``_context_extras`` reads
    the source file looking for native band contrast. Each of those blocks the
    event loop, and therefore the health check, which has made this app look
    dead before.

    NO TIMEOUT, deliberately. An add-on that never returns holds a
    default-executor worker for the life of the process, and N of them exhaust
    the pool. A timeout would NOT fix that: Python cannot kill a thread, so it
    would return control to the caller while the add-on kept running and the
    worker stayed gone -- a promise that the run had stopped, which would be
    false. An honest absence beats a false guarantee. The real fix is the
    separate process that ``context.py``'s numpy-only, no-open-handles
    restriction exists to keep possible: there, a hung analysis is a PID that
    can be killed.
    """
    manifest, trust, result = await _gates_for_run(name, body)

    try:
        context = await asyncio.to_thread(_context_for, result)
    except ContextError as exc:
        # ADDON_FAILED_STATUS, not 422: the REQUEST was well formed and the
        # add-on was never reached. What failed is the stored result this run
        # depends on, which is the same class of answer ``set_enabled`` gives
        # for a failed probe -- and 422 is FastAPI's own number for a
        # malformed request, whose body carries no ``reason`` to tell the two
        # apart.
        raise AddonRefusal(ADDON_FAILED_STATUS, REASON_RESULT_UNUSABLE,
                           str(exc)) from exc

    try:
        # An add-on's analysis is arbitrary user code of unknown duration.
        run = await asyncio.to_thread(
            run_analysis, manifest, body.analysis_key, context,
            body.params, result=result, trust=trust)
    except AddonError as exc:
        # The add-on's own name is already the first thing in this message;
        # nothing is prefixed to it.
        raise AddonRefusal(ADDON_FAILED_STATUS, REASON_ADDON_FAILED,
                           str(exc)) from exc
    return {
        "outputs": [_serialise(o, name=name, result_id=body.result_id,
                               analysis_key=body.analysis_key)
                    for o in run.outputs],
        "ignored_params": list(run.ignored_params),
        # WHAT THE ADD-ON WAS AND WAS NOT GIVEN. Both fields degrade silently
        # inside _context_extras -- a missing quality channel and an
        # unresolvable step size are logged at DEBUG, which is off in every
        # real deployment. Without this block an add-on that computed lengths
        # in micrometres against step_size_um=None hands back numbers the
        # caller cannot tell apart from resolved ones. What cannot be produced
        # is SHOWN as missing; null here is an answer, not an omission.
        "context": {"step_size_um": context.step_size_um,
                    "quality_source": context.quality_source},
    }


def _run_job_worker(job_id: str, manifest, name: str, analysis_key: str,
                    params, result_id: str, result, trust) -> None:
    """One add-on run, in its own thread, reporting into the job registry.

    The whole sequence lives here, context included: ``_context_for`` reads the
    source file, and a route that did it before starting the thread would block
    on exactly the case this module exists for.

    Nothing propagates out of this function. A thread's exception goes to
    stderr and nowhere a user can see, so every exit lands in ``finish_job``:
    the named ones with the add-on's own message, everything else with its type
    and text, and a ``finally`` as the net under both. That ``finally`` is what
    licenses ``prune_jobs`` to have no stale-running cutoff -- see its
    docstring. It checks first only to keep the log quiet on the normal path;
    ``finish_job`` would refuse the second write anyway.
    """
    def report(message, fraction=None):
        # The add-on holds this. jobs.report_progress swallows everything by
        # design, so nothing here can be counted against the add-on by
        # ``_note_crash``.
        jobs.report_progress(job_id, message, fraction)

    try:
        try:
            context = _context_for(result, report=report)
        except ContextError as exc:
            # The add-on was never reached: what failed is the stored result
            # this run depends on. Same distinction the synchronous route
            # draws, same reason code.
            jobs.finish_job(job_id, error=str(exc),
                            reason=REASON_RESULT_UNUSABLE)
            return
        try:
            run = run_analysis(manifest, analysis_key, context, params,
                               result=result, trust=trust)
        except AddonError as exc:
            # The add-on's own name is already the first thing in this
            # message; nothing is prefixed to it.
            jobs.finish_job(job_id, error=str(exc),
                            reason=REASON_ADDON_FAILED)
            return
        jobs.finish_job(
            job_id,
            outputs=[_serialise(o, name=name, result_id=result_id,
                                analysis_key=analysis_key)
                     for o in run.outputs],
            ignored_params=tuple(run.ignored_params),
            context={"step_size_um": context.step_size_um,
                     "quality_source": context.quality_source})
    except Exception as exc:                        # noqa: BLE001
        # Not only the add-on's code: _serialise runs here too, and a crash in
        # it is ours. The name still goes first, so a reader knows which run
        # died before knowing whose fault it was.
        logger.exception("add-on job %s (%s / %s) crashed", job_id, name,
                         analysis_key)
        jobs.finish_job(
            job_id,
            error=f"{name} / {analysis_key}: {type(exc).__name__}: {exc}",
            reason=REASON_ADDON_FAILED)
    finally:
        job = jobs.get_job(job_id)
        if job is not None and job["state"] == "running":
            jobs.finish_job(
                job_id,
                error=f"{name} / {analysis_key}: the run ended without "
                      "reporting a result.",
                reason=REASON_ADDON_FAILED)


@router.post("/{name}/run-job", status_code=202)
@_with_reason_code
async def start_addon_job(name: str, body: RunRequest):
    """Start the same run as ``/run``, and answer with a job id instead.

    The page uses this one because axios gives up after 300 s
    (``services/api.js:18``) and a dictionary-sized analysis outlives that: the
    request would abort in the browser while the thread kept computing a result
    nobody would ever fetch.

    A refusal is still synchronous. Every gate in ``_gates_for_run`` answers
    before a job exists, so "the add-on is not enabled" arrives as a 409 with
    its reason, not as a job that has to be polled to discover that nothing is
    going to happen.

    A thread, not ``asyncio.to_thread``: that returns a coroutine, so awaiting
    it is what makes ``/run`` block, and not awaiting it means it never runs
    unless wrapped in a task the loop holds only weakly and may collect
    mid-run. ``threading.Thread(daemon=True)`` is what this backend already
    runs for a long computation with a polled id -- see
    ``routes/forward_diagnostics.py:114`` -- and the registry is guarded by its
    own lock.
    """
    manifest, trust, result = await _gates_for_run(name, body)
    job_id = jobs.start_job(name, body.analysis_key, body.result_id)
    try:
        threading.Thread(
            target=_run_job_worker,
            args=(job_id, manifest, name, body.analysis_key, body.params,
                  body.result_id, result, trust),
            name=f"addon-job-{job_id[:8]}", daemon=True).start()
    except Exception as exc:                        # noqa: BLE001
        # The job is registered as ``running`` one line above, so a thread that
        # never starts -- nothing caps how many of these run at once, and
        # ``can't start new thread`` is a real answer -- would leave it running
        # FOREVER. That is not one stuck row: it falsifies the invariant
        # ``prune_jobs`` is built on ("a job left running means a thread that
        # is still alive"), which is why there is no stale-running cutoff to
        # clean it up, and ``/result`` would answer "poll until it finishes"
        # for the life of the process.
        logger.exception("could not start a worker thread for add-on job %s",
                         job_id)
        jobs.finish_job(
            job_id,
            error=f"{name} / {body.analysis_key}: the run could not be "
                  f"started ({type(exc).__name__}: {exc}).",
            reason=REASON_ADDON_FAILED)
        raise AddonRefusal(
            ADDON_FAILED_STATUS, REASON_ADDON_FAILED,
            f"The run of {body.analysis_key!r} could not be started: "
            f"{type(exc).__name__}: {exc}") from exc
    return {"job_id": job_id}


@router.get("/jobs/{job_id}")
@_with_reason_code
async def get_addon_job(job_id: str):
    """What a poll needs: state, message, fraction, and the failure if any.

    Without ``outputs``, which is the whole point of polling separately: a
    finished map's outputs are fetched once, from ``/result``, not re-sent on
    every tick of a progress bar. Dropped inside ``get_job``, not here: this
    route used to copy the whole list and then discard the copy -- 70.8 ms of
    event loop, per tick, for a 50k-row table.
    """
    job = jobs.get_job(job_id, include_outputs=False)
    if job is None:
        raise AddonRefusal(
            404, REASON_JOB_NOT_FOUND,
            f"No add-on job {job_id!r}. Jobs live in memory only: they are "
            "lost on a backend restart, and a finished one is dropped an hour "
            "after it finished.")
    return job


@router.get("/jobs/{job_id}/result")
@_with_reason_code
async def get_addon_job_result(job_id: str):
    """The finished run, in the same three fields ``/run`` returns.

    A failed job is answered as the failure it was -- the add-on's own message
    and the reason the worker recorded -- and not as a 200 with an error field
    the caller has to remember to look at.
    """
    job = jobs.get_job(job_id)
    if job is None:
        raise AddonRefusal(
            404, REASON_JOB_NOT_FOUND,
            f"No add-on job {job_id!r}. Jobs live in memory only: they are "
            "lost on a backend restart, and a finished one is dropped an hour "
            "after it finished.")
    if job["state"] == "running":
        raise AddonRefusal(
            409, REASON_JOB_UNFINISHED,
            f"The add-on job {job_id!r} is still running. Poll "
            f"{API_PREFIX}/jobs/{job_id} until its state is no longer "
            "'running'.")
    if job["state"] == "failed":
        raise AddonRefusal(ADDON_FAILED_STATUS,
                           job["reason"] or REASON_ADDON_FAILED,
                           job["error"] or "The run failed.")
    return {
        "outputs": job["outputs"],
        "ignored_params": list(job["ignored_params"] or ()),
        "context": job["context"],
    }
