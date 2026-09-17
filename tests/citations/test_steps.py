import pytest

from backend.api.services.citations.render import load_library
from backend.api.services.citations.steps import (
    STEP_REGISTRY,
    StepCitation,
    get_step,
    register_step,
)


def test_registry_is_not_empty():
    assert len(STEP_REGISTRY) >= 5


def test_every_declared_citation_id_exists_in_the_library():
    """The two-lists-must-match failure class. This is the guard."""
    library = load_library()
    missing = {
        step.key: [cid for cid in step.citation_ids if cid not in library]
        for step in STEP_REGISTRY.values()
    }
    missing = {k: v for k, v in missing.items() if v}
    assert not missing, f"steps cite ids absent from library.json: {missing}"


def test_a_step_without_citations_states_a_reason():
    for step in STEP_REGISTRY.values():
        if not step.citation_ids:
            assert step.no_citation_reason, (
                f"{step.key} has no citations and no reason; silence is the "
                "one thing the honesty rules forbid"
            )


def test_every_key_matches_its_registry_slot():
    for key, step in STEP_REGISTRY.items():
        assert key == step.key


def test_sentence_slots_are_named_not_positional():
    for step in STEP_REGISTRY.values():
        assert "{}" not in step.sentence, (
            f"{step.key}: positional slots break when params are a dict"
        )


def test_register_step_adds_and_get_step_finds(monkeypatch):
    monkeypatch.setitem(
        STEP_REGISTRY, "test.addon",
        StepCitation(key="test.addon", label="Test", citation_ids=(),
                     sentence="Ran the test step.",
                     no_citation_reason="fixture"),
    )
    assert get_step("test.addon").label == "Test"


def test_get_step_returns_none_for_unknown():
    assert get_step("nope.not.a.step") is None
