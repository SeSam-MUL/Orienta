import pytest

from backend.api.services.citations.render import load_library
from backend.api.services.citations.steps import (
    STEP_REGISTRY,
    StepCitation,
    get_step,
    register_step,
)


#: Every key the shipped registry declares. Pinned EXACTLY, not with a ``>=``
#: floor: the registry is the add-on contract's public surface, and a step
#: that silently VANISHES (a refactor drops a StepCitation, a run stops being
#: citable) is the failure a floor cannot see. Update this set deliberately,
#: in the same commit that adds or removes a step.
EXPECTED_KEYS = {
    "indexing.hough",
    "indexing.dictionary",
    "indexing.hough_reflectors",
    "indexing.spherical",
    "indexing.spherical_emsphinx",
    "eds.chemistry_prior",
    "eds.particle_rescue",
    "pseudosym.resolver",
    "indexing.hough_anchor",
    "preprocessing.background",
    "refinement.orientation",
}


def test_registry_holds_exactly_the_declared_steps():
    assert set(STEP_REGISTRY) == EXPECTED_KEYS
    assert len(STEP_REGISTRY) == 11


def test_the_emsphinx_step_does_not_claim_an_orienta_reimplementation():
    """C2. The EMSphInx path shells out to the authors' own binary; its
    sentence must credit EMSphInx, not claim Orienta implemented it."""
    emsphinx = STEP_REGISTRY["indexing.spherical_emsphinx"].sentence
    assert "EMSphInx" in emsphinx
    assert "reimplementation" not in emsphinx.lower()
    # ...and the Orienta-GPU key must keep saying it IS one, so the two
    # sentences stay distinguishable.
    assert "reimplementation" in STEP_REGISTRY["indexing.spherical"].sentence


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
