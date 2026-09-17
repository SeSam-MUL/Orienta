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
