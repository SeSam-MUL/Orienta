"""The autouse fixture in ``tests/conftest.py`` gives every test the indexing result
registry as it found it.

Each pair below is ordered: the first test leaves state behind on purpose (what the
batch tests, ``_store_result`` callers and a few load tests did by accident), the
second asserts that the next test starts clean. The second one is the guard: with
the fixture removed or its restore dropped, it fails. It skips when run alone,
because then there is nothing for it to compare against.
"""
from collections import OrderedDict
from types import SimpleNamespace

import pytest

_seen: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _a_result_that_was_there_before_the_tests():
    """Module scope is set up BEFORE the per-test fixture of conftest, so the
    per-test snapshot sees a registry that is not empty and an active id. Without
    this, "clear it" and "put back what was there" are the same thing and a restore
    that forgets the items could not fail."""
    from backend.api.routes import indexing

    saved = (indexing._result_registry.copy(), indexing._active_result_id)
    indexing._result_registry["seed"] = SimpleNamespace(metadata={})
    indexing._active_result_id = "seed"
    yield
    indexing._result_registry.clear()
    indexing._result_registry.update(saved[0])
    indexing._active_result_id = saved[1]


def test_a_a_stored_result_is_left_active():
    from backend.api.routes import indexing

    _seen["registry"] = indexing._result_registry
    _seen["items"] = list(indexing._result_registry.items())
    _seen["active"] = indexing._active_result_id

    result_id = indexing._store_result(SimpleNamespace(metadata={}), "isolation")

    assert indexing._active_result_id == result_id          # premise: it IS left active
    assert result_id in indexing._result_registry


def test_b_the_next_test_finds_the_registry_as_it_was():
    from backend.api.routes import indexing

    if "items" not in _seen:
        pytest.skip("needs test_a to have run first")
    assert list(indexing._result_registry.items()) == _seen["items"]
    assert indexing._active_result_id == _seen["active"]


def test_c_the_batch_pattern_pops_entries_and_leaves_the_id_dangling():
    from backend.api.routes import indexing

    result_id = indexing._store_result(SimpleNamespace(metadata={}), "isolation")
    indexing._result_registry.pop(result_id)

    assert indexing._active_result_id == result_id          # premise: dangling id
    assert result_id not in indexing._result_registry


def test_d_a_dangling_active_id_does_not_reach_the_next_test():
    from backend.api.routes import indexing

    if "active" not in _seen:
        pytest.skip("needs test_a to have run first")
    assert indexing._active_result_id == _seen["active"]


def test_e_the_registry_object_is_swapped_for_another_and_not_swapped_back():
    from backend.api.routes import indexing

    indexing._result_registry = OrderedDict(swapped=SimpleNamespace(metadata={}))

    assert "swapped" in indexing._result_registry           # premise


def test_f_other_modules_bind_the_registry_by_name_so_it_must_be_the_same_object():
    from backend.api.routes import indexing

    if "registry" not in _seen:
        pytest.skip("needs test_a to have run first")
    assert indexing._result_registry is _seen["registry"]
