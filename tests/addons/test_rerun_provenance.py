"""Re-running one add-on on one result: the trail keeps ONE step for it.

Nothing else in this suite re-runs an analysis on the same result, which is
why the defect these tests pin survived every review on this branch and was
found only by running the application. Measured there, on a real imported
result (174x145, 25230 px), running the reference add-on three times --
``n_components=3``, ``3`` again, then ``2`` -- left three steps on the trail
and a methods paragraph that said both "decomposed into 3 Gaussian
components" and "decomposed into 2 Gaussian components" about the same data,
neither marked superseded, in the text a researcher pastes into a paper.

The trail describes what produced the outputs the user is HOLDING, and the
map store already decided that is the last run: it keys on ``(name,
result_id, analysis_key, key)``, so run 3 replaced run 1's map. Two parts of
one system must not disagree about which run is current.

A core step is a different animal and is pinned here too: a multi-phase
Dictionary run records one ``indexing.dictionary`` step PER PHASE with
genuinely different parameters, and both are facts.
"""
import sys

import numpy as np
import pytest

from backend.api.services.addons.context import build_context
from backend.api.services.addons.manifest import parse_manifest
from backend.api.services.addons.runner import AddonError, run_analysis
from backend.api.services.citations.provenance import get_steps, record_step
from backend.api.services.citations.render import render_methods

from .test_runner import FIXTURES, MANIFEST, _Result

TWO_MAPS = FIXTURES / "two_maps_addon" / "orienta-addon.toml"


@pytest.fixture(autouse=True)
def _forget_the_modules():
    names = ("runnable_addon_module", "two_maps_addon_module")
    for name in names:
        sys.modules.pop(name, None)
    yield
    for name in names:
        sys.modules.pop(name, None)


def _run(manifest, key, result, params=None):
    return run_analysis(
        manifest, key,
        build_context(result, quality=np.full((3, 4), 2.0),
                      quality_source="native"),
        params or {}, result=result)


def test_the_same_run_twice_leaves_one_step():
    result = _Result()
    manifest = parse_manifest(MANIFEST)
    _run(manifest, "addon.mean_quality", result, {"scale": 1.0})
    _run(manifest, "addon.mean_quality", result, {"scale": 1.0})
    assert [s["key"] for s in get_steps(result)] == ["addon.mean_quality"]


def test_a_retuned_parameter_replaces_the_earlier_step():
    """The ordinary act of an author: run it, change a number, run it again."""
    result = _Result()
    manifest = parse_manifest(MANIFEST)
    _run(manifest, "addon.mean_quality", result, {"scale": 1.0})
    _run(manifest, "addon.mean_quality", result, {"scale": 3.0})
    steps = get_steps(result)
    assert [s["key"] for s in steps] == ["addon.mean_quality"]
    assert steps[0]["params"]["scale"] == 3.0


def test_the_methods_paragraph_states_the_rerun_once():
    """The measured symptom, in the text that gets pasted into a paper.

    ``render_methods`` dedupes on the rendered SENTENCE, so two identical
    runs already collapsed there and two differing ones never could. This is
    the assertion that was false in the running application.
    """
    result = _Result()
    manifest = parse_manifest(MANIFEST)
    _run(manifest, "addon.mean_quality", result, {"scale": 1.0})
    _run(manifest, "addon.mean_quality", result, {"scale": 3.0})
    paragraph = render_methods(get_steps(result))
    assert paragraph.count("Mean pattern quality") == 1
    assert "at scale 3." in paragraph
    assert "at scale 1." not in paragraph


def test_two_analyses_of_one_addon_keep_their_own_steps():
    """Different keys, different outputs, two facts."""
    result = _Result()
    manifest = parse_manifest(TWO_MAPS)
    _run(manifest, "addon.first", result)
    _run(manifest, "addon.second", result)
    _run(manifest, "addon.first", result)
    assert [s["key"] for s in get_steps(result)] == ["addon.first",
                                                     "addon.second"]


def test_two_addons_keep_their_own_steps():
    """A re-run of one must not disturb the other's step, or its position.

    Position is kept where the analysis FIRST entered the pipeline: tuning a
    parameter must not reshuffle the methods paragraph.
    """
    result = _Result()
    runnable = parse_manifest(MANIFEST)
    two_maps = parse_manifest(TWO_MAPS)
    _run(runnable, "addon.mean_quality", result, {"scale": 1.0})
    _run(two_maps, "addon.first", result)
    _run(runnable, "addon.mean_quality", result, {"scale": 3.0})
    steps = get_steps(result)
    assert [s["key"] for s in steps] == ["addon.mean_quality", "addon.first"]
    assert steps[0]["params"]["scale"] == 3.0


def test_a_core_step_still_appends():
    """A multi-phase Dictionary run records one step PER PHASE, and the two
    dictionary sizes are two facts about one run. Nothing the add-on runner
    needs may turn that into one."""
    result = _Result()
    record_step(result, "indexing.dictionary", {"dict_size": 100347})
    record_step(result, "indexing.dictionary", {"dict_size": 42})
    steps = get_steps(result)
    assert [s["key"] for s in steps] == ["indexing.dictionary",
                                         "indexing.dictionary"]
    assert [s["params"]["dict_size"] for s in steps] == [100347, 42]


def test_a_refused_rerun_does_not_erase_the_run_that_produced_the_outputs():
    """Validate first, then replace.

    ``record_step`` raises under pytest on a path-looking param. If the
    replacement removed the old step before validating the new one, a refused
    re-run would leave the trail with NO record of the run whose outputs the
    user is still holding -- a silent deletion in the one subtree whose job
    is to be honest about what ran.
    """
    result = _Result()
    manifest = parse_manifest(MANIFEST)
    _run(manifest, "addon.mean_quality", result, {"scale": 1.0})
    with pytest.raises(ValueError):
        record_step(result, "addon.mean_quality",
                    {"leak": "C:/Users/someone/scan.h5oina"}, replace=True)
    steps = get_steps(result)
    assert [s["key"] for s in steps] == ["addon.mean_quality"]
    assert steps[0]["params"]["scale"] == 1.0
    assert "leak" not in steps[0]["params"]


def test_a_failed_rerun_leaves_the_earlier_step_standing():
    """The add-on itself raising is the same story one layer out."""
    result = _Result()
    manifest = parse_manifest(MANIFEST)
    _run(manifest, "addon.mean_quality", result, {"scale": 1.0})
    with pytest.raises(AddonError):
        _run(manifest, "addon.mean_quality", result,
             {"scale": 3.0, "fail": True})
    steps = get_steps(result)
    assert [s["key"] for s in steps] == ["addon.mean_quality"]
    assert steps[0]["params"]["scale"] == 1.0
