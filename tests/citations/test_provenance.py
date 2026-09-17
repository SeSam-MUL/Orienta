import numpy as np
import pytest

from backend.api.services.citations.provenance import (
    PROVENANCE_SCHEMA,
    ensure_provenance,
    get_steps,
    record_step,
)


class FakeResult:
    def __init__(self, metadata=None):
        self.metadata = metadata


def test_record_step_creates_the_subtree():
    r = FakeResult()
    record_step(r, "indexing.hough", {"orienta_version": "0.3.0"})
    prov = r.metadata["provenance"]
    assert prov["schema"] == PROVENANCE_SCHEMA
    assert prov["steps"][0]["key"] == "indexing.hough"
    assert prov["steps"][0]["params"]["orienta_version"] == "0.3.0"


def test_record_step_appends_in_order():
    r = FakeResult()
    record_step(r, "preprocessing.background", {"background": "dynamic"})
    record_step(r, "indexing.hough", {})
    assert [s["key"] for s in get_steps(r)] == [
        "preprocessing.background", "indexing.hough"]


def test_unknown_key_raises_in_tests():
    """Drift is caught by running the pipeline, not by grepping source."""
    r = FakeResult()
    with pytest.raises(KeyError, match="not declared"):
        record_step(r, "totally.made.up", {})


def test_unknown_key_is_recorded_when_strict_is_off(monkeypatch):
    monkeypatch.setenv("ORIENTA_CITATIONS_STRICT", "0")
    r = FakeResult()
    record_step(r, "totally.made.up", {})
    assert get_steps(r)[0]["key"] == "totally.made.up"


def test_unknown_key_is_recorded_by_default_outside_pytest(monkeypatch):
    """Pins the production default: no env var, and (simulated) not under
    pytest, must record rather than raise — a mistyped citation key must
    never abort an indexing run that can take hours."""
    monkeypatch.delenv("ORIENTA_CITATIONS_STRICT", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    r = FakeResult()
    record_step(r, "totally.made.up", {})
    assert get_steps(r)[0]["key"] == "totally.made.up"


def test_params_must_be_json_safe():
    r = FakeResult()
    record_step(r, "indexing.hough", {"n": 1, "ok": True, "name": "x",
                                      "vals": [1, 2], "map": {"a": 1.5}})
    assert get_steps(r)[0]["params"]["map"]["a"] == 1.5


def test_non_json_param_is_stringified_not_dropped():
    class Weird:
        def __repr__(self):
            return "<weird>"

    r = FakeResult()
    record_step(r, "indexing.hough", {"thing": Weird()})
    assert get_steps(r)[0]["params"]["thing"] == "<weird>"


def test_get_steps_on_a_result_with_no_metadata_is_empty():
    assert get_steps(FakeResult()) == []
    assert get_steps(FakeResult(metadata={})) == []


def test_ensure_provenance_is_idempotent():
    r = FakeResult()
    a = ensure_provenance(r)
    a["steps"].append({"key": "x", "params": {}})
    b = ensure_provenance(r)
    assert b is a
    assert len(b["steps"]) == 1


def test_numpy_scalar_int_round_trips_as_python_int():
    r = FakeResult()
    record_step(r, "indexing.hough", {"n_patterns": np.int64(28086)})
    value = get_steps(r)[0]["params"]["n_patterns"]
    assert value == 28086
    assert type(value) is int


def test_numpy_scalar_bool_round_trips_as_python_bool():
    r = FakeResult()
    record_step(r, "indexing.hough", {"refine": np.bool_(True)})
    value = get_steps(r)[0]["params"]["refine"]
    assert value is True


def test_small_ndarray_becomes_a_list():
    r = FakeResult()
    record_step(r, "indexing.hough", {"pc": np.array([0.547, 0.465, 0.609])})
    value = get_steps(r)[0]["params"]["pc"]
    assert value == [0.547, 0.465, 0.609]
    assert isinstance(value, list)


def test_large_ndarray_is_summarised_not_dumped():
    r = FakeResult()
    big = np.zeros((301, 402), dtype=np.float32)
    record_step(r, "indexing.hough", {"map": big})
    value = get_steps(r)[0]["params"]["map"]
    assert isinstance(value, str)
    assert "301" in value and "402" in value
    assert "float32" in value
    assert len(value) < 100
