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
    assert "Al 0%" in out
    assert "Si 75%" in out
    assert "12 pixels" in out


def test_missing_key_with_format_spec_names_it_and_does_not_raise(temp_step):
    """Finding 2: a spec applied to a sentinel must not raise ValueError."""
    temp_step("test.format_spec_missing", "Value: {missing_val:.1f} recorded.")
    # The bug this guards: str.format_map(_NamingDict(...)) raised here.
    out = render_methods([{"key": "test.format_spec_missing", "params": {}}])
    assert "[missing_val not recorded]" in out
    assert ".1f" not in out


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
