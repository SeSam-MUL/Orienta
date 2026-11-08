"""A stand-in for the pattern-centre search, for the tests of the routes that run it.

``pc_search.run_global`` minimises PyEBSDIndex's cost over a stack of patterns; a route test does
not want that, it wants to know what the route does with an answer. ``stub_search`` replaces the
function: the joint PC it returns is the mean of the PCs the test hands it (one per calibration
pattern, in order), the "per pattern" values are those PCs, and every call is recorded.
"""
from __future__ import annotations

import numpy as np

from backend.api.services import pc_search


def clear_confidence(**extra) -> dict:
    return {"state": "clear", "reasons": [], "contrast": 0.95, "growth": 3.2, "typical_cost": 8.0,
            "excess_small": 0.15, "excess": 0.5, "min_excess": 0.1,
            "radius_small": 0.01, "radius": 0.02, "n_reference": 60, "best_point": [0.5, 0.5, 0.5],
            **extra}


def unclear_confidence(**extra) -> dict:
    return clear_confidence(**{"state": "unclear", "reasons": ["noContrast"], "contrast": 0.1,
                               "growth": 1.2, **extra})


def not_judged_confidence(**extra) -> dict:
    """What the search reports while it has no threshold (``pc_search.CONTRAST_MIN`` is None): the
    numbers as always, the state "notJudged", no reason."""
    return clear_confidence(**{"state": "notJudged", "reasons": [], **extra})


def stub_search(monkeypatch, pcs, *, on_call=None, confidence=None, on_edge=(), method="NM", joint=None,
                contrast_at_start=None, second_position=None, cost=0.1):
    """Replace ``pc_search.run_global``; returns the list of recorded calls. ``joint`` is the PC the
    search "finds" when it should differ from the mean of ``pcs``. ``second_position`` (``{"pc", "cost"}``)
    is what a find reports as its other position; ``cost`` is the cost of the result.

    Each call is ``{"indexer", "patterns", "mode", "method", "search_limit", "start", ...}`` (the
    keyword arguments of the real function). ``on_call(call)`` runs before the answer is built
    (a test that switches the active dataset in the middle of a run does it there)."""
    calls = []

    def fake(indexer, patterns, **kwargs):
        record = {"indexer": indexer, "patterns": patterns, **kwargs}
        calls.append(record)
        if on_call is not None:
            on_call(record)
        arr = np.asarray(pcs[:len(patterns)], dtype=float).reshape(-1, 3)
        q1, med, q3 = np.percentile(arr, [25, 50, 75], axis=0)
        return pc_search.SearchReport(
            pc=tuple(float(v) for v in (arr.mean(axis=0) if joint is None else joint)), cost=cost,
            start=kwargs.get("start"),
            mode=kwargs["mode"], method=method, n_patterns=len(arr), per_pattern=arr,
            spread={"median": med.tolist(), "iqr": (q3 - q1).tolist()},
            confidence=confidence if confidence is not None else clear_confidence(),
            on_edge=list(on_edge), contrast_at_start=contrast_at_start,
            second_position=second_position if kwargs["mode"] == "find" else None)

    monkeypatch.setattr(pc_search, "run_global", fake)
    return calls
