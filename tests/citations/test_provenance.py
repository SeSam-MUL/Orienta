import numpy as np
import pytest

from backend.api.services.citations.provenance import (
    PROVENANCE_SCHEMA,
    ensure_provenance,
    get_steps,
    merge_provenance,
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


# --- merge_provenance -------------------------------------------------
#
# Real `IndexingResult` objects rather than `FakeResult`: the multi-phase
# merge sites (indexing_controller.py, backend/api/routes/indexing.py)
# call merge_provenance on real per-phase IndexingResult objects, and a
# dataclass with a plain `.metadata` dict is exactly as cheap to build as
# FakeResult — no reason to test the real merge logic against a stand-in
# when the real object is one import away.

def _real_result(metadata=None):
    from indexing_controller import IndexingMethod, IndexingResult

    return IndexingResult(
        xmap=None,
        selection_mask=np.zeros((1, 1), dtype=bool),
        original_shape=(1, 1),
        method=IndexingMethod.HOUGH,
        metadata=metadata if metadata is not None else {},
    )


def test_merge_provenance_dedupes_identical_key_and_params():
    sub_a = _real_result()
    sub_b = _real_result()
    record_step(sub_a, "indexing.spherical",
               {"orienta_version": "0.3.0", "bandwidth": 88})
    record_step(sub_b, "indexing.spherical",
               {"orienta_version": "0.3.0", "bandwidth": 88})

    merged = _real_result()
    merge_provenance(merged, [sub_a, sub_b])

    assert len(get_steps(merged)) == 1
    assert get_steps(merged)[0]["key"] == "indexing.spherical"


def test_merge_provenance_keeps_both_when_params_differ():
    sub_a = _real_result()
    sub_b = _real_result()
    record_step(sub_a, "indexing.spherical",
               {"orienta_version": "0.3.0", "bandwidth": 88})
    record_step(sub_b, "indexing.spherical",
               {"orienta_version": "0.3.0", "bandwidth": 128})

    merged = _real_result()
    merge_provenance(merged, [sub_a, sub_b])

    steps = get_steps(merged)
    assert len(steps) == 2
    bandwidths = {s["params"]["bandwidth"] for s in steps}
    assert bandwidths == {88, 128}


def test_merge_provenance_with_a_traceless_source_is_harmless():
    """A phase that failed / recorded nothing must not raise and must not
    add a step — same 'absence is the record' rule as record_step itself."""
    sub_with_trail = _real_result()
    record_step(sub_with_trail, "indexing.hough", {"orienta_version": "0.3.0"})
    sub_traceless = _real_result()  # never touched by record_step

    merged = _real_result()
    merge_provenance(merged, [sub_with_trail, sub_traceless, None])

    steps = get_steps(merged)
    assert len(steps) == 1
    assert steps[0]["key"] == "indexing.hough"


def test_merge_provenance_then_record_step_still_appends():
    """The merged result is a normal result afterwards — a later step
    recorded directly on it (e.g. eds.chemistry_prior on the merge) must
    append normally, not be swallowed by the merge machinery."""
    sub = _real_result()
    record_step(sub, "indexing.hough", {"orienta_version": "0.3.0"})

    merged = _real_result()
    merge_provenance(merged, [sub])
    record_step(merged, "eds.chemistry_prior",
               {"strength_by_phase": {"Al.cif": 0.5}, "n_adjusted": 12})

    keys = [s["key"] for s in get_steps(merged)]
    assert keys == ["indexing.hough", "eds.chemistry_prior"]


def test_merge_provenance_is_idempotent_on_repeated_merges():
    """Merging the same sources twice (e.g. a retry, or a second helper
    touching the same merge) must not duplicate."""
    sub = _real_result()
    record_step(sub, "indexing.hough", {"orienta_version": "0.3.0"})

    merged = _real_result()
    merge_provenance(merged, [sub])
    merge_provenance(merged, [sub])

    assert len(get_steps(merged)) == 1


def test_merge_provenance_preserves_first_seen_order():
    sub_a = _real_result()
    sub_b = _real_result()
    record_step(sub_a, "indexing.hough", {"orienta_version": "0.3.0"})
    record_step(sub_b, "refinement.orientation", {"n_refined": 4})

    merged = _real_result()
    record_step(merged, "eds.particle_rescue", {"n_changed": 3})
    merge_provenance(merged, [sub_a, sub_b])

    assert [s["key"] for s in get_steps(merged)] == [
        "eds.particle_rescue", "indexing.hough", "refinement.orientation",
    ]
