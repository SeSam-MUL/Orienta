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

logger = logging.getLogger(__name__)

PROVENANCE_SCHEMA = 1

_JSON_SCALARS = (str, int, float, bool, type(None))


def _strict() -> bool:
    """Raise on an undeclared key unless explicitly told not to.

    Default strict: a key that no step declares is a bookkeeping bug, and the
    place to find it is the test run, not a user's citation list.
    """
    return os.environ.get("ORIENTA_CITATIONS_STRICT", "1") != "0"


def _jsonable(value: Any) -> Any:
    """Params must survive h5 and JSON. Stringify rather than drop."""
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
