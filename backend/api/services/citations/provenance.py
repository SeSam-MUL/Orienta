"""The trail: which steps actually ran on a result, with which parameters.

ONE typed subtree under metadata["provenance"], deliberately. The rest of
`IndexingResult.metadata` is a schemaless drawer of ~75 keys of which six
survive a save/load cycle; this claims one subtree and fixes persistence for
it alone rather than trying to fix the drawer.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

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
    prov = ensure_provenance(result)
    prov["steps"].append({"key": key, "params": _jsonable(params or {})})


def get_steps(result) -> List[dict]:
    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, dict):
        return []
    prov = metadata.get("provenance")
    if not isinstance(prov, dict):
        return []
    steps = prov.get("steps")
    return steps if isinstance(steps, list) else []
