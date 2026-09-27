"""The contract's proof: the shipped example add-on, exercised end to end.

If the add-on contract cannot express ``analysis/bc_analysis.py``, the contract
is wrong -- so this module measures the example against the real runner, the
real manifest parser, the real citation machinery and the real HTTP surface,
and never against a stand-in for any of them.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
No ``monkeypatch.syspath_prepend``. The runner is what puts a folder add-on's
own directory on ``sys.path``; if it stops doing that, a folder install is
broken for every external author and these tests must be the ones that say so.

No fixture copy of the example either. ``EXAMPLE`` is the directory that ships,
so a change to the shipped manifest or the shipped code is what these tests
read. A copy would let the example rot while its tests stayed green.
"""
from __future__ import annotations

import ast
import subprocess
import sys
import string
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.routes import addons as addons_route
from backend.api.routes import indexing as indexing_routes
from backend.api.services.addons.context import build_context
from backend.api.services.addons.manifest import declared_params, parse_manifest
from backend.api.services.addons.outputs import (
    MapOutput,
    ScalarOutput,
    TableOutput,
)
from backend.api.services.addons.runner import AddonError, run_analysis

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "orienta-addon-bc-gmm"
MANIFEST = EXAMPLE / "orienta-addon.toml"
SOURCE = EXAMPLE / "orienta_addon_bc_gmm" / "analysis.py"
README = EXAMPLE / "README.md"

client = TestClient(app)


class _Result:
    def __init__(self, shape):
        self.original_shape = shape
        self.metadata = {}
        self.confidence_scores = None
        self.xmap = None


def _bimodal_quality(shape=(40, 40), seed=0):
    """Two well-separated populations, so a mixture has something to find."""
    rng = np.random.default_rng(seed)
    a = rng.normal(60, 8, size=shape[0] * shape[1] // 2)
    b = rng.normal(160, 8, size=shape[0] * shape[1] - a.size)
    return np.concatenate([a, b]).reshape(shape)


def _run(params, quality=None, shape=(40, 40), quality_source=None):
    q = _bimodal_quality(shape) if quality is None else quality
    result = _Result(tuple(int(d) for d in np.asarray(q).shape))
    run = run_analysis(
        parse_manifest(MANIFEST), "addon.bc_gmm",
        build_context(result, quality=q, quality_source=quality_source),
        params, result=result)
    return run, result


def _by_key(run):
    return {o.key: o for o in run.outputs}


def _analyse_callable():
    """The function the runner would call, resolved the way the runner does.

    Not a bare ``import orienta_addon_bc_gmm``: the example is a folder
    add-on, and nothing in this process knows where its folder is until the
    runner puts it on ``sys.path``. Importing it directly passes or fails on
    whether some earlier test in the session happened to run the add-on
    first, which is a test that measures collection order. ``probe_import``
    is the public entry point that does exactly what enabling does.
    """
    import importlib

    from backend.api.services.addons.runner import probe_import

    manifest = parse_manifest(MANIFEST)
    probe_import(manifest)
    module_name, _, attribute = manifest.analyses[0].python_name.partition(":")
    return getattr(importlib.import_module(module_name), attribute)


# --- the manifest -----------------------------------------------------------

def test_the_manifest_is_valid():
    m = parse_manifest(MANIFEST)
    assert m.name == "bc-gmm"
    assert m.analyses[0].key == "addon.bc_gmm"


def test_the_addon_declares_a_citation():
    """Orienta's own library id, because this add-on is an Orienta artifact
    and has no DOI of its own. Nothing here invents one."""
    assert parse_manifest(MANIFEST).analyses[0].citations == ("orienta",)


def test_citing_a_core_id_does_not_mint_a_second_entry_over_it():
    """The half of the credit path this example CAN prove on its own.

    ``citations = ["orienta"]`` names the bibliography entry Orienta already
    resolves. If the bridge minted an entry for it anyway, the panel would
    show Orienta twice -- once real, once as a placeholder titled "[work ...;
    title not resolved offline]" -- for an add-on that cited it correctly.
    """
    from backend.api.services.addons.citations_bridge import (
        addon_library_entries,
    )

    assert addon_library_entries(parse_manifest(MANIFEST)) == {}


# --- manifest and code, which are two halves of one declaration -------------

def test_every_declared_parameter_is_one_the_function_accepts():
    """A declared name the function does not take is a dead control.

    The runner passes only declared names, by keyword. A name in the schema
    that the signature does not have raises ``TypeError`` inside the add-on
    the first time a user touches that control -- and the manifest and the
    code are each perfectly valid on their own, which is this branch's
    dominant defect shape.
    """
    import inspect

    signature = inspect.signature(_analyse_callable())
    for name in declared_params(parse_manifest(MANIFEST).analyses[0]):
        assert name in signature.parameters, name


def test_the_declared_defaults_are_the_defaults_the_code_uses():
    """The schema's default is what the form will show; the signature's is
    what an empty ``params`` actually runs with. Two places, one fact."""
    import inspect

    signature = inspect.signature(_analyse_callable())
    for name, schema in declared_params(
            parse_manifest(MANIFEST).analyses[0]).items():
        if "default" not in schema:
            continue
        assert signature.parameters[name].default == schema["default"], name


def _top_level_imports_of_the_example():
    """Every top-level module name the shipped package imports.

    EVERY .py the add-on ships, not just analysis.py. ``__init__.py`` runs
    first and is the file someone reaches for when they want an import to
    happen once, so a guard that skipped it had its hole in the likeliest
    place. Any file added to the package later is covered without editing
    the tests below.
    """
    sources = sorted(EXAMPLE.rglob("*.py"))
    assert len(sources) >= 2, sources
    imported = set()
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif (isinstance(node, ast.ImportFrom) and node.level == 0
                    and node.module):
                imported.add(node.module.split(".")[0])
    return imported


def test_the_addon_imports_exactly_numpy_sklearn_and_orienta():
    """The ABI claim the spec leans on, measured on the source -- and stated
    as an EQUALITY, because the previous form could not see the one import
    that matters.

    It intersected with a list of heavy libraries, so ``backend`` was
    collected into ``imported`` and never asserted on: the test named in the
    README as measuring its "numpy and scikit-learn" claim was structurally
    unable to falsify it. A test that measures a claim must be able to fail
    on it.

    ``backend`` is here because ``validate_outputs`` is strict ``isinstance``
    on ``MapOutput``/``TableOutput``/``ScalarOutput``, so EVERY add-on imports
    that module -- there is no duck-typed way in. It is a known limit of API 0
    and is written down in the README and in CHANGELOG-ADDON-API.md rather
    than left for each author to discover; what it costs is measured by the
    test below.
    """
    forbidden = {"orix", "kikuchipy", "diffsims", "torch", "hyperspy",
                 "h5py", "matplotlib"}
    imported = _top_level_imports_of_the_example()
    outside_stdlib = {m for m in imported if m not in sys.stdlib_module_names}
    assert outside_stdlib == {"numpy", "sklearn", "backend"}, outside_stdlib
    # Kept as its own assertion: the equality above would also fail if one of
    # these appeared, but this one names what the failure means.
    assert not (imported & forbidden), sorted(imported & forbidden)


def test_the_orienta_module_every_addon_must_import_is_a_leaf():
    """What that one import actually costs, measured rather than assumed.

    ``outputs.py`` imports ``math``, ``re``, ``dataclasses``, ``typing`` and
    ``numpy`` and nothing else, so importing it does NOT drag FastAPI, orix,
    kikuchipy or torch into an add-on's process. That is what makes the
    coupling a licensing and API-stability question rather than a dependency
    one, and it is the sentence the README now makes -- so it gets a guard.

    A subprocess, because this process has imported half of Orienta already
    and a ``sys.modules`` delta measured here would be meaningless.
    """
    code = (
        "import sys, numpy\n"
        "before = set(sys.modules)\n"
        "import backend.api.services.addons.outputs\n"
        "new = sorted(set(sys.modules) - before)\n"
        "print(len(new))\n"
        "print(' '.join(sorted({n.split('.')[0] for n in new})))\n"
        "print(' '.join(n for n in new if n.split('.')[0] == 'backend'))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=str(REPO))
    assert out.returncode == 0, out.stderr
    count, tops, backend_modules = out.stdout.splitlines()[:3]

    # numpy is already imported in the child, so the delta is what OUR
    # module adds on top of the dependency the add-on has anyway.
    # Measured on this machine: 19, all of them stdlib or empty packages.
    assert int(count) < 40, (count, tops)
    heavy = set(tops.split()) & {
        "fastapi", "starlette", "pydantic", "orix", "kikuchipy",
        "diffsims", "torch", "hyperspy", "h5py", "matplotlib", "scipy",
        "sklearn"}
    assert not heavy, heavy
    # And of Orienta itself: the module, plus the empty package __init__
    # files on the way to it. Nothing else of the backend is reachable.
    assert backend_modules.split() == [
        "backend", "backend.api", "backend.api.services",
        "backend.api.services.addons",
        "backend.api.services.addons.outputs"], backend_modules


# --- running it through the real runner -------------------------------------

def test_it_runs_through_the_real_runner_and_returns_a_histogram():
    """Note what is NOT here: no syspath_prepend. If the runner stops putting
    the example's own folder on sys.path, this fails -- which is the point."""
    run, _ = _run({"n_components": 2}, quality_source="Band Contrast (native)")
    assert "histogram" in {o.key for o in run.outputs}


def test_it_finds_the_two_populations_it_was_given():
    run, _ = _run({"n_components": 2})
    centres = sorted(o.value for o in run.outputs
                     if o.key.startswith("component_")
                     and o.key.endswith("_mean"))
    assert len(centres) == 2
    assert centres[0] == pytest.approx(60, abs=12)
    assert centres[1] == pytest.approx(160, abs=12)


@pytest.mark.parametrize("n_components", [2, 4])
def test_the_component_count_is_a_parameter_not_a_constant(n_components):
    """The generalisation the core version's own docstring asks for.

    NEVER 3, on purpose. The core fixes the count at three, so an add-on that
    quietly did the same would satisfy a test that asked for three and got
    three. These two ask for a count the constant cannot produce.
    """
    run, _ = _run({"n_components": n_components})
    assert sum(1 for o in run.outputs
               if o.key.endswith("_mean")) == n_components


def test_it_returns_all_three_output_kinds():
    """It is a template as much as a test: an author reading it should find
    every kind of output the contract has, produced honestly."""
    run, _ = _run({"n_components": 2})
    kinds = {type(o) for o in run.outputs}
    assert kinds == {MapOutput, TableOutput, ScalarOutput}


def test_the_component_map_covers_the_scan_grid():
    run, _ = _run({"n_components": 2})
    component_map = _by_key(run)["component_map"]
    assert component_map.values.shape == (40, 40)
    finite = component_map.values[np.isfinite(component_map.values)]
    assert set(np.unique(finite)) == {0.0, 1.0}


# --- the mutation this module is built around -------------------------------

def test_the_map_names_the_same_components_the_table_does():
    """THE guard of this add-on, and the reason it has a required mutation.

    Two halves, each defensible alone: the scalars are sorted by ascending
    mean so that rank 0 is the low-quality population, and the map carries
    sklearn's own component labels. sklearn's labels are in NO defined order,
    so unless the map is relabelled through the same permutation, a pixel
    marked 0 on the map belongs to the component the table calls 1 -- with
    the right shape, the right dtype, the right value range and the right
    label, so nothing about it looks wrong.

    That is exactly the defect this branch has already shipped twice (one
    add-on's code credited to another; a map URL serving another analysis's
    numbers), and on this add-on it would tell a user that the deformed
    pixels are the recrystallised ones. Measured directly: the quality values
    under each component of the MAP must have the mean the TABLE reports for
    that same rank.

    THREE COMPONENTS AND NOT TWO, and that is the whole difference between a
    test and a decoration. Measured on this fixture: at ``n_components=2``
    sklearn's own component order is ALREADY ascending (means 59.79, 160.07;
    argsort [0, 1]), so dropping the relabelling changes nothing and a
    two-component version of this test stays green against the broken code.
    At three it is a real permutation (means 59.79, 164.28, 154.69; argsort
    [0, 2, 1]), and the two upper components' assigned means are 152.8 and
    165.3 -- 12.5 apart, far outside the tolerance below.
    """
    q = _bimodal_quality()
    run, _ = _run({"n_components": 3}, quality=q)
    outputs = _by_key(run)
    labels = outputs["component_map"].values
    for rank in range(3):
        reported = outputs[f"component_{rank}_mean"].value
        measured = float(np.mean(q[labels == rank]))
        assert measured == pytest.approx(reported, abs=3.0), rank


def test_the_components_are_reported_in_ascending_order_of_mean():
    run, _ = _run({"n_components": 3})
    outputs = _by_key(run)
    means = [outputs[f"component_{i}_mean"].value for i in range(3)]
    assert means == sorted(means)


# --- refusals ---------------------------------------------------------------

def test_no_quality_map_is_refused_honestly():
    result = _Result((4, 4))
    with pytest.raises(AddonError, match="quality"):
        run_analysis(parse_manifest(MANIFEST), "addon.bc_gmm",
                     build_context(result), {}, result=result)


def test_a_nonsense_component_count_is_refused_by_the_addon():
    """The schema's ``minimum``/``maximum`` are documentation for a form that
    does not exist yet -- the runner filters parameters by NAME and validates
    nothing else. An add-on that leans on them hands its user sklearn's
    traceback instead of a sentence, so this one checks its own inputs.

    MATCHED ON OUR OWN WORDING, and that is the whole test. An earlier version
    matched ``"n_components"``, which passes with ``_positive_int`` deleted:
    scikit-learn's own refusal is "The \'n_components\' parameter of
    GaussianMixture must be an int in the range [1, inf). Got 0 instead." and
    already contains the name. A guard whose test the absent guard also
    satisfies is decoration.
    """
    with pytest.raises(AddonError, match="must be at least 1"):
        _run({"n_components": 0})


def test_a_non_numeric_component_count_is_refused_by_the_addon():
    """Same trap, other branch: sklearn says "must be an int in the range",
    which shares no phrase with "must be a whole number"."""
    with pytest.raises(AddonError, match="must be a whole number"):
        _run({"n_components": "three"})


def test_a_negative_bin_width_is_refused_by_the_addon():
    """Without ``_non_negative_float`` this still raises -- from ``_bin_edges``,
    saying "cannot divide a range of". Matching our own sentence is what tells
    the two apart."""
    with pytest.raises(AddonError, match="zero or more"):
        _run({"n_components": 2, "bin_width": -5})


def test_a_constant_quality_map_is_refused_rather_than_decomposed():
    """The worst defect this add-on could have had, and it is not a crash.

    A mixture asked for more components than the data has distinct values
    returns the extras with mean 0.0 and weight 0.0 and reports
    ``converged_ = True``. Measured with these defaults on a constant map:
    means [50. 0. 0.], weights [1. 0. 0.]. Those zeros would reach the
    components table, the scalars and the provenance trail as findings --
    two populations at band contrast 0 that no pixel is anywhere near.
    """
    q = np.full((40, 40), 50.0)
    with pytest.raises(AddonError, match="1 distinct pattern-quality"):
        _run({"n_components": 3}, quality=q)


def test_two_distinct_values_cannot_support_three_components():
    """One phantom instead of two, and just as invented. Measured:
    means [50. 90. 0.], weights [0.5 0.5 0.]."""
    q = np.repeat([50.0, 90.0], 800).reshape(40, 40)
    with pytest.raises(AddonError, match="2 distinct pattern-quality"):
        _run({"n_components": 3}, quality=q)


def test_exactly_as_many_components_as_distinct_values_is_allowed():
    """The other side of the guard: it must refuse the invented component and
    nothing else. Two values, two components, and the fit is exact."""
    q = np.repeat([50.0, 90.0], 800).reshape(40, 40)
    run, _ = _run({"n_components": 2}, quality=q)
    outputs = _by_key(run)
    assert outputs["component_0_mean"].value == pytest.approx(50.0, abs=0.1)
    assert outputs["component_1_mean"].value == pytest.approx(90.0, abs=0.1)
    assert outputs["component_0_area_fraction"].value == pytest.approx(
        0.5, abs=0.01)


def test_too_few_pixels_for_the_components_asked_for_is_refused():
    q = _bimodal_quality(shape=(4, 4))
    with pytest.raises(AddonError, match="too few"):
        _run({"n_components": 6}, quality=q)


def test_a_failure_names_the_addon_first():
    result = _Result((4, 4))
    with pytest.raises(AddonError) as excinfo:
        run_analysis(parse_manifest(MANIFEST), "addon.bc_gmm",
                     build_context(result), {}, result=result)
    assert str(excinfo.value).startswith("bc-gmm / addon.bc_gmm:")


# --- the sentence and the methods paragraph ---------------------------------

def test_the_sentence_names_what_the_analysis_actually_reports():
    """Its slots must be satisfiable: a declared parameter or a scalar output
    key. Otherwise the flagship sentence renders '[n_px not recorded]'."""
    analysis = parse_manifest(MANIFEST).analyses[0]
    slots = {f for _l, f, _s, _c in string.Formatter().parse(analysis.sentence)
             if f}
    run, _ = _run({"n_components": 2})
    available = set(declared_params(analysis)) | set(run.recorded_params)
    assert slots <= available, slots - available


def test_the_methods_paragraph_comes_out_complete():
    from backend.api.services.citations.provenance import get_steps
    from backend.api.services.citations.render import render_methods

    _, result = _run({"n_components": 2})
    sentence = render_methods(get_steps(result))
    assert "not recorded" not in sentence
    assert "1600" in sentence                 # the 40x40 pixels it used


def test_a_run_on_the_defaults_still_names_what_it_did():
    """The most ordinary call there is: press the button, type nothing.

    The runner records what the caller SENT, and a parameter left at its
    default is never sent -- so a sentence naming a declared parameter reads
    "[n_components not recorded] Gaussian components" on exactly the call
    every user makes first. Measured on this add-on before it also emitted
    ``n_components`` as a scalar output. The manifest default is 3, and 3 is
    what the paragraph has to say.
    """
    from backend.api.services.citations.provenance import get_steps
    from backend.api.services.citations.render import render_methods

    _, result = _run({})
    sentence = render_methods(get_steps(result))
    assert "not recorded" not in sentence, sentence
    assert "3 Gaussian components" in sentence, sentence


def test_the_sentence_does_not_claim_the_pixels_were_indexed():
    """The add-on never looks at ``phase_ids``: it counts pixels that carry a
    finite pattern-quality value, which is not the same set. A methods
    paragraph is written to be pasted into a manuscript."""
    assert "indexed" not in parse_manifest(MANIFEST).analyses[0].sentence


# --- the README, which is where an external author meets the contract -------

def _readme_paragraphs():
    return [p for p in README.read_text(encoding="utf-8").split("\n\n")
            if p.strip()]


def test_the_readme_warns_about_the_path_looking_string_parameter():
    """A declared string parameter whose value happens to look like a path
    RAISES under pytest and only warns in production -- same add-on, opposite
    outcomes. The README is the only place an external author meets that.

    Asserted as STRUCTURE, not as four substrings anywhere in the file: all of
    the asymmetry has to stand in the SAME paragraph, or a reader meets one
    half without the other and concludes their add-on is broken.
    """
    paragraphs = [p for p in _readme_paragraphs() if "asymmetry" in p]
    assert len(paragraphs) == 1, [p[:60] for p in paragraphs]
    paragraph = paragraphs[0]
    for required in ("pytest", "raises", "warns", 'format = "path"'):
        assert required in paragraph, (required, paragraph)


def test_the_readme_says_what_the_contract_could_not_carry():
    """Two things this rebuild could not express are named in the README
    rather than quietly left out: grain-average band contrast (the context
    carries no grains) and the quality source (only numbers reach the
    methods paragraph).

    Located in the section that is ABOUT that, so a stray "grain" in a
    sentence about something else cannot satisfy it.
    """
    text = README.read_text(encoding="utf-8")
    start = text.index("## What this rebuild could not express")
    section = text[start:text.index("\n## ", start + 1)]
    assert "grain_average_bc" in section, section
    assert "orix" in section, section
    assert "quality_source" in section, section


def test_the_readme_says_the_addon_imports_orienta_and_what_that_means():
    """The dependency claim an author reads before writing anything.

    "numpy and scikit-learn" was the whole of it, while line 42 of analysis.py
    imports ``backend.api.services.addons.outputs`` -- which every add-on must,
    because ``validate_outputs`` is a strict ``isinstance``. An author planning
    a licence, or a packager reading the requirements, was told the wrong
    thing by the only document written for them.
    """
    text = README.read_text(encoding="utf-8")
    start = text.index("## Requirements")
    section = text[start:]
    assert "backend.api.services.addons.outputs" in section, section
    assert "GPL" in section, section


# --- end to end, over HTTP, as a folder install -----------------------------

class _XMapWithBandContrast:
    """Just enough of a CrystalMap for pattern_quality._prop_bc to find BC.

    The path a light-h5 re-import takes, so the real ``get_quality_map``
    precedence runs without a 500 MB file on disk.
    """

    phases = ()
    phase_id = None
    rotations = None

    def __init__(self, values):
        self.prop = {"bc": np.asarray(values, dtype=float).ravel()}


class _ResultWithQuality:
    def __init__(self, values):
        self.original_shape = tuple(int(d) for d in np.asarray(values).shape)
        self.metadata = {}
        self.confidence_scores = None
        self.xmap = _XMapWithBandContrast(values)


@pytest.fixture
def installed_example(monkeypatch):
    """The example, discovered exactly as the README says it is installed.

    ``ORIENTA_ADDON_USER_DIR`` is what ``~/.orienta/addons`` resolves to, so
    pointing it at the directory that CONTAINS the example is the folder
    install of the README with the location redirected away from the
    developer's home. Nothing is copied: a copy would be discovered under a
    second path while the module is already imported from the first, which
    the runner's identity check refuses -- correctly, and it is why an
    add-on that is MOVED needs a backend restart.
    """
    monkeypatch.setenv("ORIENTA_ADDON_USER_DIR", str(EXAMPLE.parent))
    monkeypatch.delenv("ORIENTA_ADDON_DIRS", raising=False)
    yield


@pytest.fixture
def registered_bimodal_result():
    values = _bimodal_quality((40, 40))
    indexing_routes._result_registry["bc-gmm-e2e"] = _ResultWithQuality(values)
    yield "bc-gmm-e2e", values
    indexing_routes._result_registry.pop("bc-gmm-e2e", None)
    addons_route._MAP_VALUES.clear()


def test_the_example_is_listed_enabled_run_and_cited(installed_example,
                                                     registered_bimodal_result):
    """The run this whole branch exists for, through the HTTP surface only.

    Install as a folder, list it without importing it, enable it (which DOES
    import it), run it against a result that carries native band contrast,
    and read the analysis back out of the citation panel.

    ON THE BIBTEX ASSERTION, so nobody later reads it as more than it is:
    ``routes/citations._ALWAYS`` cites ``orienta`` on EVERY result, so the
    presence of Orienta's DOI does NOT prove the add-on's declaration
    carried it. It is checked because an add-on citing a core id must not
    BREAK it. The discriminating evidence is the methods sentence and the
    recorded step below -- nothing but this add-on produces either.
    """
    result_id, values = registered_bimodal_result

    listing = client.get("/api/addons")
    assert listing.status_code == 200
    entry = next(a for a in listing.json()["addons"] if a["name"] == "bc-gmm")
    assert entry["source_path"].endswith("orienta-addon.toml")
    assert entry["analyses"][0]["key"] == "addon.bc_gmm"

    enabled = client.post("/api/addons/bc-gmm/enabled", json={"enabled": True})
    assert enabled.status_code == 200, enabled.text
    assert enabled.json()["enabled"] is True

    run = client.post("/api/addons/bc-gmm/run", json={
        "analysis_key": "addon.bc_gmm", "result_id": result_id,
        "params": {"n_components": 2}})
    assert run.status_code == 200, run.text
    body = run.json()
    assert body["context"]["quality_source"] == "Band Contrast (native)"
    by_key = {o["key"]: o for o in body["outputs"]}
    assert by_key["histogram"]["kind"] == "table"
    assert by_key["component_map"]["kind"] == "map"
    assert by_key["n_px"]["value"] == float(values.size)

    # The map's bytes really are this run's numbers, over the URL it minted.
    # CONTENT, not size: this branch has already shipped a map URL serving
    # another analysis's numbers under the right label with a matching shape
    # and dtype, and a size assertion would have passed straight over it. The
    # bytes have to carry two component ranks, and the quality values under
    # each rank have to have the mean this run reported for it.
    raw = client.get(by_key["component_map"]["values_url"])
    assert raw.status_code == 200
    labels = np.frombuffer(
        raw.content, dtype=np.dtype(by_key["component_map"]["dtype"])
    ).reshape(tuple(by_key["component_map"]["shape"]))
    assert labels.shape == values.shape
    finite = labels[np.isfinite(labels)]
    assert set(np.unique(finite).tolist()) == {0.0, 1.0}
    for rank in (0, 1):
        reported = by_key[f"component_{rank}_mean"]["value"]
        assert float(np.mean(values[labels == rank])) == pytest.approx(
            reported, abs=3.0), rank

    cited = client.get(f"/api/citations/result/{result_id}")
    assert cited.status_code == 200
    payload = cited.json()
    assert "addon.bc_gmm" in [s["key"] for s in payload["steps"]]
    # Registration, and then CREDIT, which is the branch's flagship feature
    # and the thing ``undeclared == []`` does NOT prove: that only says the
    # step is declared, which is true of a step declaring no works at all.
    # This says the add-on's own manifest line reached the registry the
    # citation panel and the .h5 exporter both read.
    from backend.api.services.citations.steps import get_step

    assert get_step("addon.bc_gmm").citation_ids == ("orienta",)
    assert payload["undeclared"] == []
    assert "2 Gaussian components" in payload["methods"]
    assert "1600" in payload["methods"]
    assert "not recorded" not in payload["methods"]
    # Against CITATION.cff, not against a literal. This assertion is about the
    # add-on's declared work REACHING the bibliography; which DOI Orienta
    # publishes under is a different fact, and it moved under this test once
    # already (21021938 -> the concept DOI 22664080) when golive updated the
    # citation metadata. A copied number turns that release into a red test in
    # a file about add-ons.
    from backend.api.services.citations.self_entry import orienta_entry_from_cff

    own_doi = orienta_entry_from_cff().get("DOI")
    assert own_doi, "CITATION.cff must declare a DOI for this to mean anything"
    assert own_doi in payload["bibtex"]
