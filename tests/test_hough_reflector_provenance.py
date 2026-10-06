"""A result must say which reflector families produced it.

Hough indexing with a phase's own selection is a different experiment from the
default list; a reader of the result (its metadata, the methods paragraph, the
exported trail) has to be able to see that. A run without a selection stays
exactly as it was: no extra step, no extra metadata, the same sentence.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ebsd_utils as eu  # noqa: E402
import hough_reflectors as hr  # noqa: E402
from tests.test_hough_reflectors import NI_CIF, _phase, _plist  # noqa: E402

from backend.api.services.citations.provenance import get_steps  # noqa: E402
from backend.api.services.citations.render import render_methods  # noqa: E402


@pytest.fixture(autouse=True)
def clean_registry():
    hr.clear_specs()
    yield
    hr.clear_specs()


def _run(tmp_path):
    import kikuchipy as kp
    from indexing_controller import IndexingConfig, IndexingMethod, hough_index_patterns
    s = kp.data.nickel_ebsd_small()
    ni, _ = _phase(tmp_path, NI_CIF, "Ni")
    mask = np.ones(s.axes_manager.navigation_shape[::-1], dtype=bool)
    return hough_index_patterns(s, _plist(ni), s.detector.deepcopy(),
                                IndexingConfig(method=IndexingMethod.HOUGH), mask)


def test_a_run_without_a_selection_records_nothing_extra(tmp_path):
    result = _run(tmp_path)
    keys = [s["key"] for s in get_steps(result)]
    assert keys == ["indexing.hough"]
    assert "hough_reflectors" not in result.metadata
    text = render_methods(get_steps(result))
    assert "reflector" not in text.lower()


def test_a_custom_selection_is_in_the_metadata_and_the_methods_sentence(tmp_path):
    hr.set_spec("Ni", {"mode": "custom", "families": [[1, 1, 1], [2, 0, 0], [2, 2, 0]]})
    result = _run(tmp_path)
    steps = get_steps(result)
    assert [s["key"] for s in steps] == ["indexing.hough", "indexing.hough_reflectors"]
    step = steps[1]["params"]
    assert step["phase"] == "Ni"
    assert step["families"] == ["{111}", "{200}", "{220}"]
    assert step["source"] == "user selection"
    meta = result.metadata["hough_reflectors"]["Ni"]
    assert meta["mode"] == "custom" and meta["families_used"] == ["{111}", "{200}", "{220}"]
    assert meta["spec"]["families"] == [[1, 1, 1], [2, 0, 0], [2, 2, 0]]
    text = render_methods(steps)
    assert "used the reflector families {111}, {200} and {220} (user selection)" in text
    assert text.startswith("Orientations were determined by Hough/Radon indexing")


def test_only_the_families_pyebsdindex_kept_are_named(tmp_path):
    """{222} repeats the pole of {111}: ticked, but not what the library used."""
    hr.set_spec("Ni", {"mode": "custom",
                       "families": [[1, 1, 1], [2, 0, 0], [2, 2, 2]]})
    result = _run(tmp_path)
    step = get_steps(result)[1]["params"]
    assert step["families"] == ["{111}", "{200}"]
    assert result.metadata["hough_reflectors"]["Ni"]["families_ticked"] == \
        ["{111}", "{200}", "{222}"]


def test_a_changed_rule_says_so(tmp_path):
    hr.set_spec("Ni", {"mode": "auto", "rule": {"min_d": 1.2, "f_threshold": 0.4}})
    result = _run(tmp_path)
    step = get_steps(result)[1]["params"]
    assert step["source"] == "default construction with min d 1.2 A and |F| threshold 0.4"
    assert result.metadata["hough_reflectors"]["Ni"]["mode"] == "auto"


def test_the_step_is_declared_with_a_citation_and_survives_the_trail_round_trip():
    from backend.api.services.citations.steps import get_step
    step = get_step("indexing.hough_reflectors")
    assert step is not None and "pyebsdindex" in step.citation_ids
    text = render_methods([{"key": "indexing.hough_reflectors", "params": {
        "phase": "Al", "families": ["{200}"], "source": "user selection"}}])
    assert text == "Hough indexing of Al used the reflector families {200} (user selection)."


def _change_right_after_the_indexer_is_built(monkeypatch, spec):
    """The nastiest moment: the indexer exists, the run has not yet written down
    what it was built with."""
    real = eu.create_indexer

    def create_then_change(*a, **k):
        out = real(*a, **k)
        hr.set_spec("Ni", spec)
        return out

    monkeypatch.setattr(eu, "create_indexer", create_then_change)


def test_the_provenance_is_that_of_the_indexer_that_was_built_not_of_the_registry_now(
        tmp_path, monkeypatch):
    """A selection changed while a run is going must not change what the finished
    result says it was indexed with: the families come from the snapshot taken when
    the indexer was built."""
    import kikuchipy as kp
    from indexing_controller import IndexingConfig, IndexingMethod, hough_index_patterns
    s = kp.data.nickel_ebsd_small()
    ni, _ = _phase(tmp_path, NI_CIF, "Ni")
    mask = np.ones(s.axes_manager.navigation_shape[::-1], dtype=bool)
    hr.set_spec("Ni", {"mode": "custom", "families": [[1, 1, 1], [2, 0, 0], [2, 2, 0]]})
    _change_right_after_the_indexer_is_built(
        monkeypatch, {"mode": "custom", "families": [[1, 1, 1], [3, 1, 1]]})

    result = hough_index_patterns(s, _plist(ni), s.detector.deepcopy(),
                                  IndexingConfig(method=IndexingMethod.HOUGH), mask)
    assert hr.get_spec("Ni")["families"] == [[1, 1, 1], [3, 1, 1]]      # it did change
    step = get_steps(result)[1]["params"]
    assert step["families"] == ["{111}", "{200}", "{220}"]               # what was built
    meta = result.metadata["hough_reflectors"]["Ni"]
    assert meta["spec"]["families"] == [[1, 1, 1], [2, 0, 0], [2, 2, 0]]
    assert "{311}" not in render_methods(get_steps(result))


def test_a_selection_added_during_a_default_run_is_not_claimed_by_it(tmp_path, monkeypatch):
    import kikuchipy as kp
    from indexing_controller import IndexingConfig, IndexingMethod, hough_index_patterns
    s = kp.data.nickel_ebsd_small()
    ni, _ = _phase(tmp_path, NI_CIF, "Ni")
    mask = np.ones(s.axes_manager.navigation_shape[::-1], dtype=bool)
    _change_right_after_the_indexer_is_built(
        monkeypatch, {"mode": "custom", "families": [[1, 1, 1], [2, 0, 0]]})
    result = hough_index_patterns(s, _plist(ni), s.detector.deepcopy(),
                                  IndexingConfig(method=IndexingMethod.HOUGH), mask)
    assert [x["key"] for x in get_steps(result)] == ["indexing.hough"]
    assert "hough_reflectors" not in result.metadata
