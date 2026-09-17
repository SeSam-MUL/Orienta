from backend.api.services.citations.render import render_methods


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
