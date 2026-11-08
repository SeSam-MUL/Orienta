"""Refuse, by name, what a hexagonal scan cannot honestly do yet.

A hexagonal scan is stored and indexed as the index-space rectangle it is. The evaluations that
have been taught its six neighbours (``scan_neighbours``) and the maps drawn on its hexagonal
display raster (``hex_raster``) are not listed here. What is listed still reads the scan through
the rectangle: anything that looks at a point's NEIGHBOURS there sees four of the six (the
Analysis page's maps and grain tools, NLPAR, the crystal hint's neighbourhood average, the
grain-based phase assignment, the smoothness step of the R+PC refinement), and anything that
places a grid point on the index grid puts every second row half a step off (the pixel-wise
pattern-centre calibration). Saving a result is not one of them: a hexagonal scan is written
with its grid as measured (HexGrid ``.ang``, Orienta ``.h5``), or resampled with a note in the file.
A result from the listed endpoints would be plausible and wrong, which is worse than no result.
These endpoints answer ``400`` with a code the interface translates, and the layer picker shows
the layers as "not available" with the reason.

One table (``GUARDED``) and one FastAPI dependency (``hex_gate``, installed on the routers that hold
these endpoints in ``backend/api/main.py``) do it, so a route is not forgotten by editing its body.
``tests/test_hex_guard.py`` holds the table against the real route table: a renamed route, a route
without the dependency, or a guard that fires on a square scan turns a test red.

A result imported from an Orienta ``.h5`` (or an analysis dataset loaded from one) carries the
geometry the file states on ``/Indexing``, so a hexagonal file is refused here like a fresh
hexagonal result. An unknown geometry (a file without a stamp, a dataset nobody registered) is NOT
refused: unknown is not hexagonal, and the square path is the behaviour that existed before.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

from starlette.requests import HTTPConnection

from backend.api.problem import problem

NEIGHBOURHOOD = "hex_neighbourhood_unsupported"
PC_GRID = "hex_pc_grid_pending"
ANALYSIS_PAGE = "hex_analysis_page_pending"

#: The English sentence of each code; ``{what}`` is the entry's ``what``. The interface shows its
#: own translation of the code (``errors.<code>`` in the ``common`` locale namespace).
MESSAGES = {
    NEIGHBOURHOOD: (
        "{what} compares every point with its four neighbours on the index grid. On a hexagonal "
        "scan every point has six neighbours, so the result would be silently distorted. It is "
        "not available for hexagonal scans yet."),
    PC_GRID: (
        "{what} places its grid points on the index grid, which is half a step off on every "
        "second row of a hexagonal scan. Use a single global pattern centre for now."),
    ANALYSIS_PAGE: (
        "{what} cannot be drawn for a hexagonal scan yet: the page draws every map on the index "
        "grid, which puts every second row half a step off."),
}


class Ask(NamedTuple):
    """What a ``when`` predicate may look at."""
    query: dict
    body: Optional[dict]
    path: dict


#: When in the work a limit is met.
BEFORE_INDEXING = "before_indexing"
AFTER_INDEXING = "after_indexing"
STAGES = (BEFORE_INDEXING, AFTER_INDEXING)

#: The groups the interface sorts the limits into, within a stage. Every ``Guarded`` entry carries
#: one; ``tests/test_hex_limits.py`` fails a new entry without a valid one.
CATEGORIES = ("preprocessing", "grains_and_boundaries", "maps", "cleanup", "refinement")


@dataclass(frozen=True)
class Guarded:
    method: str
    path: str                 # the route's path as the app has it, prefix included
    basis: str                # "result" | "dataset" | "analysis": whose geometry decides
    code: str
    what: str
    when: Optional[Callable[[Ask], bool]] = None     # None: always
    stage: str = AFTER_INDEXING                       # one of STAGES: for the list the interface shows
    #: Stable snake_case id of the limit AS THE USER SEES IT: the interface translates its text per
    #: key (the English ``what`` is a fragment that only reads well inside the refusal sentence).
    #: Routes that are one limit to the user share a key (the three crystal-hint routes); their
    #: other fields then agree (``tests/test_hex_limits.py``).
    key: str = ""
    category: str = ""                                # one of CATEGORIES
    #: For a ``partial`` entry, what is refused, as data read from the gate's own constants (called
    #: when the answer is built, so a changed constant changes the answer): ``affects`` names the
    #: kinds of thing that are refused (no entry sets it today; the interface still reads it),
    #: ``parameters`` the request parameters that trigger the refusal.
    affects: Optional[Callable[[], list]] = None
    parameters: Optional[Callable[[], list]] = None


def is_hex(geometry) -> bool:
    return isinstance(geometry, dict) and geometry.get("kind") == "hex"


# ----------------------------------------------------------------- predicates

def _positive(value) -> bool:
    try:
        return float(value or 0) > 0
    except (TypeError, ValueError):
        return False


#: The crystal hint's neighbourhood average: its request parameter is the radius ``r`` of a
#: ``(2r + 1) x (2r + 1)`` window of patterns on the index grid. On a hexagonal scan every second row
#: sits half a step to one side, so that block reaches up to 1.8 steps from its centre and misses true
#: neighbours. The crystal hint does not average on the scan's neighbour table (``frame_average``
#: does), so a radius above 0 stays refused for a hexagonal scan; radius 0 is one pattern and works.
AVERAGING_PARAMETER = "avg_radius"


def averaging_asked(ask: Ask) -> bool:
    """Does the request ask for a neighbourhood average (a radius above 0)?"""
    src = {**(ask.query or {}), **(ask.body or {})}
    return _positive(src.get(AVERAGING_PARAMETER))


def _averaging_parameters() -> list:
    return [AVERAGING_PARAMETER]


# ---------------------------------------------------------------------- table

GUARDED = (
    Guarded("POST", "/api/ebsd/nlpar", "dataset", NEIGHBOURHOOD, "NLPAR denoising",
            stage=BEFORE_INDEXING, key="nlpar", category="preprocessing"),
    # The next three routes are one limit to the user (the crystal hint's neighbourhood average).
    Guarded("POST", "/api/crystal-hint/analyze-pixel", "dataset", NEIGHBOURHOOD,
            "The crystal hint's neighbourhood averaging", averaging_asked, stage=BEFORE_INDEXING,
            key="crystal_hint_averaging", category="preprocessing", parameters=_averaging_parameters),
    Guarded("POST", "/api/crystal-hint/analyze-region", "dataset", NEIGHBOURHOOD,
            "The crystal hint's neighbourhood averaging", averaging_asked, stage=BEFORE_INDEXING,
            key="crystal_hint_averaging", category="preprocessing", parameters=_averaging_parameters),
    Guarded("POST", "/api/crystal-hint/quality-check", "dataset", NEIGHBOURHOOD,
            "The crystal hint's neighbourhood averaging", averaging_asked, stage=BEFORE_INDEXING,
            key="crystal_hint_averaging", category="preprocessing", parameters=_averaging_parameters),
    Guarded("POST", "/api/pc/calibrate/grid", "dataset", PC_GRID,
            "The pixel-wise pattern-centre calibration", stage=BEFORE_INDEXING,
            key="pixelwise_pc_calibration", category="refinement"),
    Guarded("POST", "/api/indexing/grain-phase-assign", "result", NEIGHBOURHOOD,
            "The grain-based phase assignment",
            key="grain_phase_assignment", category="grains_and_boundaries"),
    Guarded("POST", "/api/analysis/grains/reconstruct", "analysis", NEIGHBOURHOOD,
            "The grain reconstruction", key="grain_reconstruction", category="grains_and_boundaries"),
    Guarded("POST", "/api/analysis/grain-size", "analysis", NEIGHBOURHOOD, "The grain-size analysis",
            key="grain_size", category="grains_and_boundaries"),
    Guarded("POST", "/api/analysis/deformation", "analysis", NEIGHBOURHOOD,
            "The deformation analysis (KAM, boundaries)",
            key="deformation_analysis", category="grains_and_boundaries"),
    # Every map of the Analysis page, the ones that compare nothing included (bc, ipf_*): the page
    # draws them all on the index grid. One entry with a code and a key of its own, so that it can be
    # deleted when analysis/ has been reworked and the page can use the six neighbours.
    Guarded("GET", "/api/analysis/map/{map_type}", "analysis", ANALYSIS_PAGE, "The maps of the Analysis page",
            key="analysis_map", category="maps"),
    Guarded("POST", "/api/refinement/compute", "result", NEIGHBOURHOOD,
            "The joint R+PC refinement (its smoothness step)",
            key="refinement_compute", category="refinement"),
    Guarded("POST", "/api/refinement/resmooth", "result", NEIGHBOURHOOD,
            "The smoothness step of the R+PC refinement",
            key="refinement_resmooth", category="refinement"),
)

_BY_ROUTE = {(g.method, g.path): g for g in GUARDED}


# ------------------------------------------------------------------- the gate

def _geometry(basis: str, ask: Ask) -> Optional[dict]:
    """The geometry (as a dict) the entry's basis names, or None when it is unknown."""
    if basis == "dataset":
        from backend.api.routes import ebsd_viewer
        from backend.api.services import scan_geometry
        geometry = scan_geometry.geometry_for(ebsd_viewer._active_dataset)
        return geometry.to_dict() if geometry is not None else None
    if basis == "analysis":
        from backend.api.routes import analysis
        return getattr(analysis.get_analysis_dataset(), "scan_geometry", None)
    # "result": the result named in the request, else the active one
    from backend.api.routes import indexing
    rid = ask.query.get("result_id") or (ask.body or {}).get("result_id")
    if rid and not isinstance(rid, str):
        return None             # a malformed id is the route's to reject (422), not a lookup key
    result = indexing._result_registry.get(rid) if rid else indexing.get_last_indexing_result()
    meta = getattr(result, "metadata", None)
    return meta.get("scan_geometry") if isinstance(meta, dict) else None


async def hex_gate(request: HTTPConnection) -> None:
    """Router dependency: answer 400 with the entry's code when a guarded endpoint is asked about
    a hexagonal scan. Everything else passes untouched.

    Typed as ``HTTPConnection`` so FastAPI can inject it into a websocket route of a gated router
    as well; a websocket has no method and nothing in the table, so it returns at once."""
    if request.scope.get("type") != "http":
        return
    method = request.scope.get("method")
    route = request.scope.get("route")
    entry = _BY_ROUTE.get((method, getattr(route, "path", None)))
    if entry is None:
        return
    body = None
    if method in ("POST", "PUT", "PATCH"):
        try:
            # for an http scope FastAPI hands over the route's own Request, whose body is cached
            parsed = await request.json()  # type: ignore[attr-defined]
            body = parsed if isinstance(parsed, dict) else None
        except Exception:  # noqa: BLE001 - no body, or not JSON: the route reports that itself
            body = None
    ask = Ask(dict(request.query_params), body, dict(request.path_params))
    if entry.when is not None and not entry.when(ask):
        return
    if is_hex(_geometry(entry.basis, ask)):
        raise problem(400, entry.code, MESSAGES[entry.code].format(what=entry.what))


# --------------------------------------------------- the list the interface shows

#: What a hexagonal scan can still do although it looks like it might not. The interface says so
#: next to the list of limits; the routes behind them are not in ``GUARDED``.
NOT_LIMITED = (
    ("image_export", "Exporting an image (PNG) of a phase map, a map in the viewer or a pattern"),
    ("file_export", "Saving a result as .ang or .h5, or resampled"),
    ("global_pc_refinement", "Refining one global pattern centre"),
)


def limits() -> dict:
    """What a hexagonal scan cannot do yet, built from ``GUARDED`` alone.

    ``{"groups": {"before_indexing": [...], "after_indexing": [...]},
    "categories": [...], "not_limited": [{"key", "what"}]}``. A table item is ``{"key",
    "category", "method", "path", "what", "code", "message", "partial", "affects", "parameters"}``:
    ``key`` is the stable id the interface translates the limit by (``what`` is an English fragment
    of the refusal sentence); routes that are one limit to the user share a key (the three
    crystal-hint routes) and agree in every field but ``method``/``path``. ``category`` is one
    of ``categories`` (the groups the limits are sorted into). ``partial`` is true when only some
    requests to the route are refused; then ``affects`` lists the kinds of thing that are refused
    (no entry sets it today) and ``parameters`` the request parameters that make
    a request refused, both read from the gate's own constants (empty lists when they do not
    apply, and for a route that is always refused). ``message`` is the English sentence; ``code``
    says which sentence shape it is."""
    groups = {stage: [] for stage in STAGES}
    for g in GUARDED:
        groups[g.stage].append({
            "key": g.key, "category": g.category,
            "method": g.method, "path": g.path, "what": g.what, "code": g.code,
            "message": MESSAGES[g.code].format(what=g.what), "partial": g.when is not None,
            "affects": g.affects() if g.affects else [],
            "parameters": g.parameters() if g.parameters else []})
    return {"groups": groups, "categories": list(CATEGORIES),
            "not_limited": [{"key": key, "what": what} for key, what in NOT_LIMITED]}
