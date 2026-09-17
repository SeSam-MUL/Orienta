import pytest

from backend.api.services.citations.render import render_methods
from backend.api.services.citations.steps import STEP_REGISTRY, StepCitation, register_step


def test_fills_named_slots_from_params():
    out = render_methods([
        {"key": "indexing.spherical",
         "params": {"bandwidth": 88, "orienta_version": "0.3.0"}},
    ])
    assert "bandwidth 88" in out
    assert "Orienta 0.3.0" in out


def test_missing_param_is_named_not_defaulted():
    """Honesty rule 2: a blank beats a guessed number."""
    out = render_methods([
        {"key": "indexing.spherical", "params": {"orienta_version": "0.3.0"}},
    ])
    assert "[bandwidth not recorded]" in out
    assert "bandwidth 88" not in out


def test_undeclared_step_is_shown_not_dropped():
    """Honesty rule 1."""
    out = render_methods([{"key": "some.addon.step", "params": {}}])
    assert "some.addon.step" in out
    assert "no citation declared" in out


def test_steps_are_joined_in_order():
    out = render_methods([
        {"key": "preprocessing.background", "params": {"background": "dynamic"}},
        {"key": "indexing.hough", "params": {"orienta_version": "0.3.0"}},
    ])
    assert out.index("preprocessed") < out.index("Hough")


def test_empty_returns_empty_string():
    assert render_methods([]) == ""


# --- Fix round 1: composite params must read as prose, not Python repr,
# and a missing key must survive a format spec without raising. -----------


@pytest.fixture
def temp_step():
    """Register a throwaway step for one test, then remove it again.

    The prose rules below (float trimming, list joining, bool words, None
    naming) are formatter behaviour that no *real* STEP_REGISTRY template
    happens to exercise yet. register_step is the same public hook a future
    add-on would use to declare its own step, so this fixture drives
    render_methods through exactly that path rather than reaching into the
    formatter class directly.
    """
    registered = []

    def _register(key: str, sentence: str) -> StepCitation:
        step = StepCitation(key=key, label=key, citation_ids=(), sentence=sentence)
        register_step(step)
        registered.append(key)
        return step

    yield _register

    for key in registered:
        STEP_REGISTRY.pop(key, None)


def test_dict_param_renders_as_prose_not_python_repr():
    """Finding 1's own measured example: strengths must not repr as a dict."""
    out = render_methods([
        {"key": "eds.chemistry_prior",
         "params": {"strength_by_phase": {"Al": 0.0, "Si": 0.75}, "n_adjusted": 12}},
    ])
    # Built independently of the implementation: the finding's own repr was
    # "{'Al': 0.0, 'Si': 0.75}" -- none of its punctuation may survive.
    assert "{" not in out
    assert "}" not in out
    assert "'" not in out
    assert ":" not in out
    assert "12 pixels" in out


def test_dict_value_renders_as_bare_float_not_percentage():
    """Round 2, finding 1: no [0,1]-implies-percentage heuristic.

    0.0/0.75 are the API's own scale -- "%" would be this formatter
    inventing a unit the data never claimed. Pinned against the real
    strength_by_phase shape so the heuristic cannot creep back in.
    """
    out = render_methods([
        {"key": "eds.chemistry_prior",
         "params": {"strength_by_phase": {"Al": 0.0, "Si": 0.75}, "n_adjusted": 12}},
    ])
    assert "Al 0, Si 0.75" in out
    assert "%" not in out


def test_missing_key_with_format_spec_names_it_and_does_not_raise(temp_step):
    """Finding 2: a spec applied to a sentinel must not raise ValueError."""
    temp_step("test.format_spec_missing", "Value: {missing_val:.1f} recorded.")
    # The bug this guards: str.format_map(_NamingDict(...)) raised here.
    out = render_methods([{"key": "test.format_spec_missing", "params": {}}])
    assert "[missing_val not recorded]" in out
    assert ".1f" not in out


def test_missing_key_with_str_conversion_is_named_not_a_repr(temp_step):
    """Round 2, finding 2: {x!s} calls str() before format_field ever runs."""
    temp_step("test.conversion_s", "Value: {missing_val!s} recorded.")
    out = render_methods([{"key": "test.conversion_s", "params": {}}])
    assert "[missing_val not recorded]" in out
    assert "_Missing object at" not in out


def test_missing_key_with_repr_conversion_is_named_not_a_repr(temp_step):
    """Round 2, finding 2: {x!r} calls repr() before format_field ever runs."""
    temp_step("test.conversion_r", "Value: {missing_val!r} recorded.")
    out = render_methods([{"key": "test.conversion_r", "params": {}}])
    assert "[missing_val not recorded]" in out
    assert "_Missing object at" not in out


def test_float_is_trimmed_not_full_precision(temp_step):
    temp_step("test.float_value", "NCC was {ncc}.")
    out = render_methods([
        {"key": "test.float_value", "params": {"ncc": 0.8234567}},
    ])
    assert "0.823" in out
    assert "0.8234567" not in out


def test_integral_float_has_no_trailing_dot_zero(temp_step):
    temp_step("test.integral_float", "Bandwidth: {bandwidth}.")
    out = render_methods([
        {"key": "test.integral_float", "params": {"bandwidth": 88.0}},
    ])
    assert "88.0" not in out
    assert "Bandwidth: 88." in out


def test_list_join_uses_and_before_last_item(temp_step):
    temp_step("test.list_three", "Phases: {phases}.")
    out = render_methods([
        {"key": "test.list_three", "params": {"phases": ["Al", "Si", "Fe"]}},
    ])
    assert "Al, Si and Fe" in out
    assert "Al, Si, and Fe" not in out


def test_two_item_list_joined_with_and_only(temp_step):
    # Two items is the edge of the "comma-separated, ' and ' before the
    # last" rule -- there is no earlier comma to place.
    temp_step("test.list_two", "Phases: {phases}.")
    out = render_methods([
        {"key": "test.list_two", "params": {"phases": ["Al", "Si"]}},
    ])
    assert "Phases: Al and Si." in out
    assert "Al, Si" not in out


def test_bool_renders_as_yes_no_not_python_bool(temp_step):
    temp_step("test.bool_value", "Refined: {refined}.")
    out_true = render_methods([
        {"key": "test.bool_value", "params": {"refined": True}},
    ])
    out_false = render_methods([
        {"key": "test.bool_value", "params": {"refined": False}},
    ])
    assert "Refined: yes." in out_true
    assert "Refined: no." in out_false
    assert "True" not in out_true
    assert "False" not in out_false


def test_none_value_is_named_not_printed_as_none(temp_step):
    temp_step("test.none_value", "Threshold: {threshold}.")
    out = render_methods([
        {"key": "test.none_value", "params": {"threshold": None}},
    ])
    assert "None" not in out
    assert "[threshold not recorded]" in out


def test_none_dict_value_uses_the_same_bracketed_form_as_top_level(temp_step):
    """Round 2, finding 3: one absence phrasing, not two.

    A missing/None top-level value already reads "[key not recorded]" (via
    _Missing). A None *inside* a dict must read identically, not the
    unbracketed "key not recorded" it used to.
    """
    temp_step("test.dict_with_none", "Composition: {comp}.")
    out = render_methods([
        {"key": "test.dict_with_none", "params": {"comp": {"Al": 0.5, "Si": None}}},
    ])
    assert "[Si not recorded]" in out
    assert "Al 0.5" in out
    # Bracketed and unbracketed "Si not recorded" have the same substring
    # count only if every occurrence is the bracketed form.
    assert out.count("Si not recorded") == out.count("[Si not recorded]")


# ---------------------------------------------------------------------------
# I2: one sentence per FACT. Deduplicated on the rendered sentence, not on
# the key and not on (key, params) -- both were tried and both were wrong.
# ---------------------------------------------------------------------------

def test_a_repeated_step_with_a_slot_free_sentence_prints_once():
    """POST /pseudosym/unify and refine_orientations mutate an ALREADY
    STORED result and append a step that may already be there with
    DIFFERENT params -- so merge_provenance's (key, params) fingerprint
    cannot dedupe them -- and pseudosym.resolver's template has no slots to
    tell the two apart, so it printed verbatim twice."""
    out = render_methods([
        {"key": "pseudosym.resolver", "params": {}},
        {"key": "pseudosym.resolver", "params": {"n_changed": 61}},
    ])
    assert out.count("Pseudo-symmetric orientation variants") == 1


def test_a_multi_phase_dictionary_run_keeps_every_phase():
    """The regression a key-dedupe introduced. A multi-phase Dictionary run
    records one indexing.dictionary step PER PHASE with genuinely different
    dict_size (different point groups sample different numbers of
    orientations). Collapsing them to one printed a single phase's numbers
    for the whole run -- understating how much dictionary was used, in text
    meant for peer review."""
    out = render_methods([
        {"key": "indexing.dictionary",
         "params": {"dict_size": 100347, "angular_step_deg": 2.0,
                    "orienta_version": "0.3.0"}},
        {"key": "indexing.dictionary",
         "params": {"dict_size": 333891, "angular_step_deg": 2.0,
                    "orienta_version": "0.3.0"}},
    ])
    assert "100347 simulated patterns" in out
    assert "333891 simulated patterns" in out
    assert out.count("Orientations were determined by dictionary indexing") == 2


def test_two_identical_dictionary_entries_still_print_once():
    """Same phase recorded twice (a merge that saw the same sub-result via
    two routes) is one fact, not two."""
    params = {"dict_size": 100347, "angular_step_deg": 2.0,
              "orienta_version": "0.3.0"}
    out = render_methods([
        {"key": "indexing.dictionary", "params": dict(params)},
        {"key": "indexing.dictionary", "params": dict(params)},
    ])
    assert out.count("Orientations were determined by dictionary indexing") == 1


def test_differing_params_on_a_slotted_template_are_two_facts():
    """The counterpart of the pseudosym case: when the template DOES have a
    slot for what differs, the two entries say different things and both
    must survive."""
    out = render_methods([
        {"key": "refinement.orientation", "params": {"n_refined": 512}},
        {"key": "refinement.orientation", "params": {"n_refined": 2048}},
    ])
    assert "512 pixels" in out
    assert "2048 pixels" in out


def test_first_seen_order_survives_the_dedupe():
    out = render_methods([
        {"key": "indexing.hough", "params": {"orienta_version": "0.3.0"}},
        {"key": "refinement.orientation", "params": {"n_refined": 1}},
        {"key": "indexing.hough", "params": {"orienta_version": "0.3.0"}},
    ])
    assert out.index("Hough") < out.index("refined")
    assert out.count("Hough/Radon indexing") == 1


def test_a_repeated_undeclared_step_also_prints_once():
    """Its sentence names only the key, so differing params cannot make it
    two facts."""
    out = render_methods([
        {"key": "some.addon.step", "params": {}},
        {"key": "some.addon.step", "params": {"x": 1}},
    ])
    assert out.count("some.addon.step") == 1


def test_two_different_undeclared_steps_both_print():
    out = render_methods([
        {"key": "some.addon.step", "params": {}},
        {"key": "other.addon.step", "params": {}},
    ])
    assert "some.addon.step" in out
    assert "other.addon.step" in out
