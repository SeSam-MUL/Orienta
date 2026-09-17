"""The trail: which steps actually ran on a result, with which parameters.

ONE typed subtree under metadata["provenance"], deliberately. The rest of
`IndexingResult.metadata` is a schemaless drawer of ~75 keys of which six
survive a save/load cycle; this claims one subtree and fixes persistence for
it alone rather than trying to fix the drawer.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

PROVENANCE_SCHEMA = 1

_JSON_SCALARS = (str, int, float, bool, type(None))

# A params dict ends up in an HDF5 attribute and must stay small. An array at
# or below this many elements is inlined as a list; above it, only shape and
# dtype are recorded. 32 comfortably covers the small fixed-size vectors that
# actually belong in a citation trail (a 3-vector PC, a 4x4 detector matrix,
# a handful of phase weights) while keeping a per-pixel map (hundreds of
# thousands of elements) from being dumped whole.
_MAX_INLINE_ARRAY_SIZE = 32


def _strict() -> bool:
    """Raise on an undeclared key unless explicitly told not to.

    An explicit ``ORIENTA_CITATIONS_STRICT`` wins in either direction: "0"
    forces lenient even under pytest, anything else forces strict even in
    production. With no override, strict only under pytest
    (``PYTEST_CURRENT_TEST`` is set per-test by pytest itself) — so the test
    suite still catches a mistyped key or a step added without a matching
    ``STEP_REGISTRY`` entry immediately, but a production indexing run that
    can take minutes to hours is never aborted by that same bookkeeping
    mistake. In production the honest alternative is used instead: log a
    warning and record the step anyway, visible in the citation list as
    "ran, no citation declared" rather than silently dropped.
    """
    override = os.environ.get("ORIENTA_CITATIONS_STRICT")
    if override is not None:
        return override != "0"
    return "PYTEST_CURRENT_TEST" in os.environ


def _jsonable(value: Any) -> Any:
    """Params must survive h5 and JSON. Stringify rather than drop.

    numpy scalars and arrays are pervasive in this codebase's step params
    (pixel counts, PC vectors, per-phase maps) and need explicit handling:
    a bare ``repr()`` fallback would write ``"np.int64(5)"`` into a citation
    trail, and eventually into a methods paragraph.
    """
    if isinstance(value, np.generic):
        # np.float64 subclasses float and would pass the _JSON_SCALARS check
        # below unchanged, but np.int64 and np.bool_ do not subclass their
        # Python equivalents — handle all numpy scalars uniformly via .item()
        # so every one becomes a real Python int/float/bool.
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size <= _MAX_INLINE_ARRAY_SIZE:
            return _jsonable(value.tolist())
        return f"ndarray shape={value.shape} dtype={value.dtype}"
    if isinstance(value, _JSON_SCALARS):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return repr(value)


# A Windows drive letter ("C:\\", "e:/") anywhere in a string.
_DRIVE_RE = re.compile(r"[A-Za-z]:[\\/]")


def _path_like(text: str) -> bool:
    """Does this string look like a local filesystem path?"""
    return "/" in text or "\\" in text or bool(_DRIVE_RE.search(text))


def _local_paths_in(value: Any, where: str = "") -> List[str]:
    """Every path-looking string inside ``value``, keys included."""
    found: List[str] = []
    if isinstance(value, str):
        if _path_like(value):
            found.append(f"{where}={value!r}" if where else repr(value))
    elif isinstance(value, dict):
        for k, v in value.items():
            found.extend(_local_paths_in(str(k), f"{where}.key" if where else "key"))
            found.extend(_local_paths_in(v, f"{where}.{k}" if where else str(k)))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            found.extend(_local_paths_in(v, f"{where}[{i}]"))
    return found


def _reject_local_paths(key: str, params: Dict[str, Any]) -> None:
    """Defence in depth: a recorded param must never be a local path.

    The methods paragraph is written to be pasted into a manuscript, and the
    whole trail is copied into every exported ``.h5``. A param keyed or
    valued by an absolute path therefore publishes the operator's username
    and directory layout — which is exactly what happened once, because the
    EDS strength dict is keyed by phase FILE PATH on the wire and was
    recorded verbatim (fixed at the recording site in routes/indexing.py by
    re-keying to phase names).

    This is the net BEHIND that fix, not a replacement for it: it cannot
    know what the right name would have been, so it must not rewrite the
    value — silently editing a recorded fact is the opposite of this
    module's contract. It raises under :func:`_strict` (on under pytest) so
    a future leak is a failing test, and only warns otherwise, so it can
    never abort a user's hours-long indexing run over bookkeeping.
    """
    offenders = _local_paths_in(params)
    if not offenders:
        return
    message = (
        f"citation step '{key}' records what looks like a local filesystem "
        f"path: {'; '.join(offenders[:5])}. Recorded params reach the methods "
        "paragraph and every exported .h5 — record a phase/file NAME "
        "(e.g. Path(p).stem or the name on the xmap), never a path."
    )
    if _strict():
        raise ValueError(message)
    logger.warning(message)


def ensure_provenance(result) -> Dict[str, Any]:
    """Return the provenance subtree, creating it if a path forgot."""
    if getattr(result, "metadata", None) is None:
        result.metadata = {}
    prov = result.metadata.get("provenance")
    if not isinstance(prov, dict):
        prov = {"schema": PROVENANCE_SCHEMA, "steps": []}
        result.metadata["provenance"] = prov
    prov.setdefault("schema", PROVENANCE_SCHEMA)
    prov.setdefault("steps", [])
    return prov


def record_step(result, key: str, params: Optional[dict] = None) -> None:
    """Record that ``key`` ran on ``result`` with ``params``.

    A step that did not run records nothing: absence IS the record that it
    was off.
    """
    from .steps import get_step

    if get_step(key) is None:
        message = (
            f"citation step '{key}' is not declared in STEP_REGISTRY; "
            "add a StepCitation for it in citations/steps.py"
        )
        if _strict():
            raise KeyError(message)
        logger.warning(message)
    jsonable = _jsonable(params or {})
    # After _jsonable, so it sees what will actually be written (a Path
    # object stringifies to a path here, not before).
    _reject_local_paths(key, jsonable)
    prov = ensure_provenance(result)
    prov["steps"].append({"key": key, "params": jsonable})


def get_steps(result) -> List[dict]:
    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, dict):
        return []
    prov = metadata.get("provenance")
    if not isinstance(prov, dict):
        return []
    steps = prov.get("steps")
    return steps if isinstance(steps, list) else []


def merge_provenance(target, sources: Iterable[Any]) -> None:
    """Union the recorded steps of ``sources`` onto ``target``.

    For the sites where a multi-phase run builds a *fresh* result from
    several per-phase sub-results (the "compare phases" merge and the
    per-phase-EDS-routing merge): each sub-result already recorded its own
    steps via :func:`record_step` while it was built, and those steps would
    otherwise be silently lost the moment the merge constructs a brand-new
    ``metadata`` dict for the combined result.

    ``sources`` is an iterable of result-like objects — anything
    :func:`get_steps` accepts (an object whose ``.metadata`` is a dict with
    a ``provenance.steps`` list). A source that carries no trail at all
    (``None``, a plain object, a result nothing was ever recorded on)
    contributes nothing and does not raise.

    Deduplicates by the JSON-serialised ``(key, params)`` pair, not by key
    alone: two phases indexed with genuinely different parameters (a
    different bandwidth, a different dictionary size) are two distinct
    facts and both are kept. First-seen order is preserved, including
    steps already on ``target`` before this call.
    """
    prov = ensure_provenance(target)
    seen = {json.dumps(step, sort_keys=True) for step in prov["steps"]}
    for source in sources:
        for step in get_steps(source):
            fingerprint = json.dumps(step, sort_keys=True)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            prov["steps"].append(
                {"key": step["key"], "params": dict(step.get("params") or {})}
            )
