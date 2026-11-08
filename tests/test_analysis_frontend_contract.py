"""The frontend must send exactly the field names the backend declares.

The vitest for ``analysisApi.reconstructGrains`` only checks the JS side against
itself; this test holds the JS call text against the pydantic request model, so a
renamed field on either side turns red.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from backend.api.routes.analysis import (
    BatchProcessRequest, GrainReconstructionRequest, GrainSizeRequest, RXCriteriaRequest,
)

API_JS = Path(__file__).resolve().parents[1] / "frontend" / "src" / "services" / "api.js"
FIELDS = ("alpha", "fill", "misorientation_threshold", "min_grain_size", "edge_rule", "ecd_basis", "stats_phase")


def _reconstruct_call(text: str) -> str:
    match = re.search(r"reconstructGrains:.*?\}\)", text, re.DOTALL)
    assert match, "reconstructGrains call not found in api.js"
    return match.group(0)


def _call(text: str, name: str) -> str:
    """The text of the property ``name: ...`` of ``analysisApi``, up to the comma that ends it, found by balancing
    brackets (a call can hold ``})`` inside, e.g. ``...(x ? { x } : {})``). ``//`` comments are removed first, so a
    word in a comment cannot stand for a field that is no longer sent."""
    block = text[text.index("export const analysisApi"):]       # other APIs have a `batch` too
    block = re.sub(r"//[^\n]*", "", block)
    match = re.search(rf"\b{name}:", block)
    assert match, f"{name} call not found in analysisApi"
    depth, quote = 0, None
    for i in range(match.end(), len(block)):
        ch = block[i]
        if quote:
            if ch == quote and block[i - 1] != "\\":
                quote = None
        elif ch in "'\"`":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                return block[match.start():i]                    # the object literal of analysisApi ends
            depth -= 1
        elif ch == "," and depth == 0:
            return block[match.start():i]
    raise AssertionError(f"{name} call in analysisApi does not end")


def _missing_fields(call: str, fields=FIELDS, model=GrainReconstructionRequest) -> list[str]:
    declared = model.model_fields
    missing = []
    for name in fields:
        assert name in declared, f"{name} is not a field of {model.__name__}"
        if not re.search(rf"\b{name}\b\s*(:|,|\}})", call):
            missing.append(name)
    return missing


def test_frontend_sends_the_same_field_names():
    call = _reconstruct_call(API_JS.read_text(encoding="utf-8"))
    assert _missing_fields(call) == []


def test_contract_check_notices_a_renamed_key():
    """Guard for the guard: a renamed key in a scratch copy must be reported."""
    call = _reconstruct_call(API_JS.read_text(encoding="utf-8"))
    broken = call.replace("min_grain_size:", "min_size:")
    assert broken != call
    assert _missing_fields(broken) == ["min_grain_size"]


def test_grain_size_call_sends_the_fields_the_model_declares():
    call = _call(API_JS.read_text(encoding="utf-8"), "grainSize")
    assert _missing_fields(call, ("edge_rule", "ecd_basis", "stats_phase"), GrainSizeRequest) == []


def test_batch_call_sends_the_fields_the_model_declares():
    call = _call(API_JS.read_text(encoding="utf-8"), "batch")
    assert _missing_fields(call, ("folder_path", "output_path", "grain_edge_rule", "grain_ecd_basis",
                                  "grain_stats_phase", "misorientation_threshold", "min_grain_size", "alpha", "fill",
                                  "rx_gos_threshold", "rx_gbc_fraction", "rx_gkam_threshold", "rx_min_radius_um"),
                           BatchProcessRequest) == []


RX_FIELD_NAMES = ("rx_gos_threshold", "rx_gbc_fraction", "rx_gkam_threshold", "rx_min_radius_um")


def test_rx_classify_call_sends_the_fields_the_model_declares():
    call = _call(API_JS.read_text(encoding="utf-8"), "rxClassify")
    assert _missing_fields(call, RX_FIELD_NAMES + ("phase",), RXCriteriaRequest) == []
    broken = call.replace("rx_gkam_threshold:", "rx_gkam:")
    assert broken != call and _missing_fields(broken, RX_FIELD_NAMES, RXCriteriaRequest) == ["rx_gkam_threshold"]


def test_the_rx_phase_is_part_of_the_text_the_check_reads():
    """The call is read to its end: ``phase`` is sent through a spread whose text holds ``})``, which a lazy match
    to the first ``})`` cut off. A call that no longer sends it must be reported."""
    call = _call(API_JS.read_text(encoding="utf-8"), "rxClassify")
    assert "rx_min_radius_um" in call and re.search(r"\bphase\b\s*\}", call), call
    without = re.sub(r"\s*\.\.\.\(phase !== undefined \? \{ phase \} : \{\}\),?", "", call)
    assert without != call and _missing_fields(without, ("phase",), RXCriteriaRequest) == ["phase"]
    # a word 'phase' in a comment is not a field that is sent, and the next property is not part of the call
    scratch = ("export const analysisApi = {\n  rxClassify: (c) => api.post('/x', {\n    // phase: sent\n"
               "    rx_gos_threshold: c.g,\n  }),\n  next: { phase: 1 },\n}")
    commented = _call(scratch, "rxClassify")
    assert _missing_fields(commented, ("phase",), RXCriteriaRequest) == ["phase"]
    assert "next" not in commented


def _rx_js() -> str:
    return (API_JS.parents[1] / "components" / "Analysis" / "rxCriteria.js").read_text(encoding="utf-8")


def test_the_page_holds_the_rx_criteria_the_backend_names_with_its_defaults_and_ranges():
    """rxCriteria.js (the wire names, the defaults, the ranges) against analysis.rx_analysis, which the routes,
    /status and the batch use: the page reads GET /status and the answer of POST /rx under these names."""
    from analysis.rx_analysis import DEFAULT_RX_CRITERIA, RX_CRITERIA_RANGES

    text = _rx_js()
    wire = dict(re.findall(r"(\w+): '(rx_\w+)'", re.search(r"RX_WIRE = Object\.freeze\(\{(.*?)\}\)", text, re.DOTALL).group(1)))
    assert sorted(wire.values()) == sorted(RX_FIELD_NAMES) == sorted(DEFAULT_RX_CRITERIA) == sorted(RX_CRITERIA_RANGES)
    defaults = dict(re.findall(r"(\w+): '([\d.]+)'", re.search(r"DEFAULT_RX_CRITERIA = Object\.freeze\(\{(.*?)\}\)", text, re.DOTALL).group(1)))
    ranges = {k: (float(a), float(b)) for k, a, b in
              re.findall(r"(\w+): \[([\d.]+), ([\d.]+)\]", re.search(r"RX_RANGES = Object\.freeze\(\{(.*?)\}\)", text, re.DOTALL).group(1))}
    assert sorted(defaults) == sorted(ranges) == sorted(wire)
    for page_name, wire_name in wire.items():
        assert float(defaults[page_name]) == DEFAULT_RX_CRITERIA[wire_name], page_name
        assert ranges[page_name] == RX_CRITERIA_RANGES[wire_name], page_name


def test_status_and_the_rx_answer_carry_the_names_the_page_reads():
    """The fields of GET /status and of the POST /rx answer are the wire names, as numbers."""
    import numpy as np
    from fastapi.testclient import TestClient

    from analysis.ebsd_dataset import EBSDDataset
    from backend.api.main import app
    from tests.synthetic_grain_map import make_map, to_xmap
    import backend.api.routes.analysis as ar

    m = make_map(shape=(20, 30), n_seeds=6)
    xmap = to_xmap(m)
    xmap.prop["bc"] = np.random.default_rng(0).uniform(20, 200, xmap.size)
    saved = ar._dataset, dict(ar._grain_stats_settings), dict(ar._rx_criteria)
    try:
        ar._dataset = EBSDDataset(xmap, step_size=m["step"])
        ar._reset_grain_stats_settings()
        client = TestClient(app)
        status = client.get("/api/analysis/status").json()
        assert {k: status[k] for k in RX_FIELD_NAMES} == ar._rx_criteria
        from analysis.grain_analysis import reconstruct_grains
        ar._dataset.grains = reconstruct_grains(ar._dataset, angle_threshold_deg=5.0, min_pixels=1, alpha=0.0, fill=False)
        answer = client.post("/api/analysis/rx", json={"rx_gos_threshold": 3}).json()
        assert answer["rx_gos_threshold"] == 3.0 and all(isinstance(answer[k], float) for k in RX_FIELD_NAMES)
    finally:
        ar._dataset = saved[0]
        ar._grain_stats_settings.clear()
        ar._grain_stats_settings.update(saved[1])
        ar._rx_criteria.clear()
        ar._rx_criteria.update(saved[2])


def test_new_fields_are_options_the_page_offers():
    """The values the page offers are the values the backend accepts."""
    from typing import get_args

    from backend.api.routes.analysis import EcdBasis, EdgeRule

    stats_js = (API_JS.parents[1] / "components" / "Analysis" / "grainStats.js").read_text(encoding="utf-8")
    for name, literal in (("EDGE_RULES", EdgeRule), ("ECD_BASES", EcdBasis)):
        match = re.search(rf"export const {name} = \[(.*?)\]", stats_js, re.DOTALL)
        assert match, f"{name} not found in grainStats.js"
        offered = re.findall(r"'([a-z_]+)'", match.group(1))
        assert offered == list(get_args(literal)), f"{name}: page offers {offered}"


# ---------------------------------------------------------------------------------------------
# The page and the backend agree on which maps need grains, and on the safeguard numbers
# ---------------------------------------------------------------------------------------------

ROUTES_PY = Path(__file__).resolve().parents[1] / "backend" / "api" / "routes" / "analysis.py"
PAGE_JSX = API_JS.parents[1] / "components" / "Analysis" / "AnalysisPage.jsx"
LOCALES = API_JS.parents[1] / "locales"


GRAIN_MESSAGE = "Reconstruct grains first"


def _called_names(node) -> set[str]:
    import ast

    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            if isinstance(sub.func, ast.Name):
                names.add(sub.func.id)
            elif isinstance(sub.func, ast.Attribute):
                names.add(sub.func.attr)
    return names


def _grain_map_types_in_route(source: str | None = None) -> set[str]:
    """Map types whose branch in ``get_analysis_map`` needs grains, read from the source.

    A branch needs grains when it raises "Reconstruct grains first ..." itself or calls a module-level
    helper that does (directly or through other helpers), so moving the check into a helper is seen.
    """
    import ast

    text = ROUTES_PY.read_text(encoding="utf-8") if source is None else source
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:     # the file is being edited by someone else, or is broken
        raise AssertionError(f"{ROUTES_PY.name} does not parse: {exc}") from exc
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    route = next((f for f in functions if f.name == "get_analysis_map"), None)
    assert route is not None, (
        f"get_analysis_map not found in {ROUTES_PY.name}; if the map route was renamed or moved, "
        "update this contract test and GRAIN_LAYERS in AnalysisPage.jsx together")

    helpers = {f.name for f in functions if f is not route and GRAIN_MESSAGE in ast.unparse(f)}
    grew = True
    while grew:        # a helper that calls a helper
        grew = False
        for f in functions:
            if f is not route and f.name not in helpers and _called_names(f) & helpers:
                helpers.add(f.name)
                grew = True

    found: set[str] = set()
    for node in ast.walk(route):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "map_type"
                and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
                and isinstance(test.comparators[0], ast.Constant)):
            continue
        branch = ast.Module(body=node.body, type_ignores=[])
        if GRAIN_MESSAGE in ast.unparse(branch) or _called_names(branch) & helpers:
            found.add(test.comparators[0].value)
    return found


def _grain_layers_in_page() -> set[str]:
    match = re.search(r"const GRAIN_LAYERS = \[(.*?)\]", PAGE_JSX.read_text(encoding="utf-8"), re.DOTALL)
    assert match, "GRAIN_LAYERS not found in AnalysisPage.jsx"
    return set(re.findall(r"'([a-z_]+)'", match.group(1)))


def test_the_page_gates_exactly_the_maps_that_need_grains():
    in_route = _grain_map_types_in_route()
    assert in_route, "no map type in the route raises 'Reconstruct grains first'; the parser found nothing"
    assert _grain_layers_in_page() == in_route


def test_the_route_parser_sees_a_branch_that_needs_grains():
    """Guard for the guard: the four layers known today are found."""
    assert {"gos", "ecd", "rx", "grain_boundaries"} <= _grain_map_types_in_route()


def test_the_safeguard_numbers_in_the_texts_are_the_backends():
    import inspect
    import json

    from analysis.grain_analysis import EDGE_MIN_FRACTION, EDGE_MIN_GRAINS, analyze_grain_size

    import re

    tolerance = inspect.signature(analyze_grain_size).parameters["edge_tolerance_fraction"].default
    # a number as a whole token: "50" in "fewer than 50, so" but not in "150" or "500"
    min_grains = rf"(?<!\d){EDGE_MIN_GRAINS}(?!\d)"
    min_share = rf"(?<!\d){round(EDGE_MIN_FRACTION * 100)} %"
    max_share = rf"(?<!\d){round(tolerance * 100)} %"
    # the notice names the limit that fired: the count (Few), the share (Share) or the outer-row limit (Many)
    wanted = {
        ("edgeRuleTip", "matlab_script"): (min_grains, min_share),
        ("edgeRuleTip", "mtex"): (min_grains, min_share),
        ("edgeRuleTip", "outer_row"): (max_share,),
        ("statEdgeSafeguardFew", None): (min_grains,),
        ("statEdgeSafeguardFewPhase", None): (min_grains,),
        ("statEdgeSafeguardShare", None): (min_share,),
        ("statEdgeSafeguardSharePhase", None): (min_share,),
        ("statEdgeSafeguardMany", None): (max_share,),
        ("statEdgeSafeguardManyPhase", None): (max_share,),
    }
    for lang in ("en", "de", "ja", "zh"):
        grain = json.loads((LOCALES / lang / "analysis.json").read_text(encoding="utf-8"))["grain"]
        for (key, sub), numbers in wanted.items():
            text = grain[key] if sub is None else grain[key][sub]
            for number in numbers:
                assert re.search(number, text), f"{lang} grain.{key}{'.' + sub if sub else ''} lacks /{number}/: {text}"


_ROUTE_WITH_HELPER = """
def _need_grains(ds, what):
    if ds.grains is None:
        raise HTTPException(status_code=400, detail=f"Reconstruct grains first for {what}")

def get_analysis_map(map_type):
    if map_type == "bc":
        data = 1
    elif map_type == "gos":
        _need_grains(ds, "GOS map")
    elif map_type == "ecd":
        if ds.grains is None:
            raise HTTPException(status_code=400, detail="Reconstruct grains first for ECD map")
    elif map_type in ("ipf", "ipf_x"):
        data = 2
"""


def test_the_route_parser_sees_a_grain_check_that_moved_into_a_helper():
    assert _grain_map_types_in_route(_ROUTE_WITH_HELPER) == {"gos", "ecd"}


def test_the_route_parser_says_what_is_wrong_when_the_route_is_gone():
    import pytest

    with pytest.raises(AssertionError, match="get_analysis_map not found"):
        _grain_map_types_in_route("def something_else():" + chr(10) + "    pass" + chr(10))
    with pytest.raises(AssertionError, match="does not parse"):
        _grain_map_types_in_route("def broken(:" + chr(10))


# ---------------------------------------------------------------------------------------------
# What the page reads from the answers: the names are the backend's
# ---------------------------------------------------------------------------------------------
STATS_JS = API_JS.parents[1] / "components" / "Analysis" / "grainStats.js"
STORE_JS = API_JS.parents[1] / "stores" / "useResultStore.js"


def _page_text() -> str:
    return PAGE_JSX.read_text(encoding="utf-8")


def _slice(text: str, start: str, end: str) -> str:
    """The text from ``start`` to the next ``end`` after it."""
    first = text.index(start)
    return text[first:text.index(end, first)]


def _names_read(body: str, var: str) -> set[str]:
    return set(re.findall(rf"\b{var}\??\.([a-z_]+)", body))


@pytest.fixture
def real_status():
    """The real answer of GET /status for a map whose grains were reconstructed with a chosen phase."""
    import numpy as np
    from fastapi.testclient import TestClient

    import backend.api.routes.analysis as ar
    from backend.api.main import app
    from tests.test_grain_stats_phase import _tile_map

    ds, _, _, _ = _tile_map(12, 12, 8)
    ds.xmap.prop["bc"] = np.random.default_rng(0).uniform(20, 200, ds.xmap.size)
    saved = ar._dataset, dict(ar._grain_stats_settings)
    ar._dataset = ds
    ar._reset_grain_stats_settings()
    try:
        client = TestClient(app)
        body = {"misorientation_threshold": 5.0, "min_grain_size": 1, "alpha": 0.0, "fill": False, "stats_phase": 1}
        task = client.post("/api/analysis/grains/reconstruct", json=body).json()["task_id"]
        assert client.get(f"/api/analysis/task/{task}").json()["status"] == "completed"
        yield client.get("/api/analysis/status").json()
    finally:
        ar._dataset = saved[0]
        ar._grain_stats_settings.clear()
        ar._grain_stats_settings.update(saved[1])


def test_the_page_reads_the_status_fields_the_backend_writes(real_status):
    stats_body = _slice(STATS_JS.read_text(encoding="utf-8"), "export function settingsFromStatus", "\n}\n")
    store_body = _slice(STORE_JS.read_text(encoding="utf-8"), "applyBackendStatus: (d) =>", "\n  }),")
    read = _names_read(stats_body, "d") | _names_read(store_body, "d")
    assert {"has_grains", "edge_rule", "ecd_basis", "stats_phase", "recon_params", "shape", "phase_options",
            "loaded", "dataset_token", "reconstruct"} <= read
    assert read <= set(real_status), f"the page reads {sorted(read - set(real_status))}, which /status does not carry"
    recon = _names_read(stats_body, "r")
    assert recon == {"threshold", "min_pixels", "min_intercept", "alpha", "fill"}
    assert recon <= set(real_status["recon_params"])
    assert real_status["has_grains"] is True and real_status["stats_phase"] == 1
    assert isinstance(real_status["dataset_token"], str) and real_status["reconstruct"] is None
    # the phase list: the select reads id and name
    assert {"id", "name"} <= set(real_status["phase_options"][0])
    select_js = (STATS_JS.parent / "GrainStatsSettings.jsx").read_text(encoding="utf-8")
    page_js = select_js + STATS_JS.read_text(encoding="utf-8") + _page_text()
    assert set(re.findall(r"\bp\.(id|name)\b", page_js)) == {"id", "name"}


def test_the_page_reads_the_statistics_phase_fields_of_the_answers(real_status):
    """The names grainStatsFrom reads for the phase are the ones the reconstruct and /grain-size answers carry."""
    from backend.api.routes.analysis import _edge_fields
    from analysis.grain_analysis import GrainSizeResult

    carried = set(_edge_fields(GrainSizeResult(
        avg_maxDimX=0, std_maxDimX=0, avg_maxDimY=0, std_maxDimY=0, avg_ECD=0, std_ECD=0, avg_AR=0, std_AR=0,
        area_weighted_ECD=0, ecd_per_grain=[], maxDimX_per_grain=[], maxDimY_per_grain=[], ar_per_grain=[])))
    body = _slice(STATS_JS.read_text(encoding="utf-8"), "export function grainStatsFrom", "\n}\n")
    read = _names_read(body, "r")
    assert {"stats_phase", "stats_phase_name", "n_grains_total", "edge_rule", "ecd_basis"} <= read
    assert {"stats_phase", "stats_phase_name", "n_grains_total", "n_edge_grains", "n_grains_used",
            "edge_safeguard_applied", "edge_rule", "ecd_basis"} <= carried


def test_the_page_reads_the_batch_result_the_backend_writes():
    from analysis.batch_processor import BatchResult

    result = BatchResult(processed=["/d/a.h5"], failed=[{"file": "/d/b.h5", "reason": "x"}],
                         skipped=[{"file": "/d/c.h5", "reason": "y"}], n_files=3,
                         warnings=[{"file": "/d/a.h5", "reason": "rx: z"}]).to_dict()
    body = _slice(_page_text(), "function BatchResultLists", "export function BatchTab")
    read = _names_read(body, "result") | _names_read(body, "e")
    assert {"processed", "failed", "skipped", "warnings", "file", "reason"} <= read
    for name in read - {"file", "reason"}:
        assert name in result, f"the page reads result.{name}, which the batch result does not carry"
    for item in (result["failed"][0], result["skipped"][0], result["warnings"][0]):
        assert {"file", "reason"} <= set(item)


def test_the_page_reads_the_export_answers_the_backend_writes():
    page = _page_text()
    routes = ROUTES_PY.read_text(encoding="utf-8")
    # the export task result is {path, warnings}; the Electron path reads exactly these
    export_route = _slice(routes, "async def export_excel(", '@router.get("/export/excel-download")')
    result_dict = re.search(r'\["result"\] = \{([^}]*)\}', export_route)
    assert result_dict, "the export task sets no result"
    assert {"path", "requested_path", "warnings"} <= set(re.findall(r'"([a-z_]+)":', result_dict.group(1)))
    body = _slice(page, "// Poll the export task", "const handleExport")
    assert {"path", "requested_path", "warnings"} <= _names_read(body, "result")
    # the download header: the name the route sets, the page reads, and main.py exposes to the browser
    assert 'headers["X-Orienta-Warnings"]' in routes
    assert "x-orienta-warnings" in page
    main = (ROUTES_PY.parents[1] / "main.py").read_text(encoding="utf-8")
    assert "X-Orienta-Warnings" in main[main.index("expose_headers"):]


def test_a_running_reconstruction_is_listed_as_the_store_reads_it_and_a_second_start_answers_409():
    """The route's own dict for a run in flight, and the 409 of a second start, held against the page's code."""
    import numpy as np
    from fastapi.testclient import TestClient

    import backend.api.routes.analysis as ar
    from backend.api.main import app
    from tests.test_grain_stats_phase import _tile_map

    ds, _, _, _ = _tile_map(12, 12, 8)
    ds.xmap.prop["bc"] = np.random.default_rng(0).uniform(20, 200, ds.xmap.size)
    saved = ar._dataset, dict(ar._grain_stats_settings)
    ar._dataset = ds
    ar._reset_grain_stats_settings()
    task_id = "contract-run"
    ar._analysis_tasks[task_id] = {"status": "running", "progress": 0.25}
    with ar._reconstruct_lock:
        ar._reconstruct_runs[task_id] = ds
    try:
        client = TestClient(app)
        running = client.get("/api/analysis/status").json()["reconstruct"]
        assert running == {"task_id": task_id, "status": "running", "progress": 0.25}
        store_js = STORE_JS.read_text(encoding="utf-8")
        assert "d.reconstruct?.task_id" in store_js
        # the second start: 409 with the route's own text; the page branches on the status code
        body = {"misorientation_threshold": 5.0, "min_grain_size": 1, "alpha": 0.0, "fill": False}
        answer = client.post("/api/analysis/grains/reconstruct", json=body)
        assert answer.status_code == 409
        assert answer.json()["detail"] == ar.RECONSTRUCT_BUSY
        assert "err?.response?.status === 409" in _page_text()
    finally:
        with ar._reconstruct_lock:
            ar._reconstruct_runs.pop(task_id, None)
        ar._analysis_tasks.pop(task_id, None)
        ar._dataset = saved[0]
        ar._grain_stats_settings.clear()
        ar._grain_stats_settings.update(saved[1])
