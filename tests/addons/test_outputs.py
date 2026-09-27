import json
from pathlib import Path

import numpy as np
import pytest

from backend.api.services.addons.outputs import (
    MapOutput,
    OutputError,
    ScalarOutput,
    TableOutput,
    validate_outputs,
)


def test_a_well_formed_set_passes():
    out = validate_outputs([
        MapOutput(key="kam", label="KAM", values=np.zeros((3, 4)), unit="deg"),
        TableOutput(key="t", label="T", columns=("a", "b"), rows=[(1, 2)]),
        ScalarOutput(key="s", label="S", value=1.5, unit="um"),
    ], shape=(3, 4))
    assert len(out) == 3


def test_a_map_must_match_the_scan_grid():
    with pytest.raises(OutputError, match="shape"):
        validate_outputs([MapOutput(key="k", label="K", values=np.zeros((2, 2)))],
                         shape=(3, 4))


def test_duplicate_keys_are_refused():
    with pytest.raises(OutputError, match="duplicate"):
        validate_outputs([
            ScalarOutput(key="s", label="A", value=1),
            ScalarOutput(key="s", label="B", value=2),
        ], shape=(3, 4))


def test_a_table_row_must_match_its_columns():
    with pytest.raises(OutputError, match="columns"):
        validate_outputs([TableOutput(key="t", label="T", columns=("a", "b"),
                                      rows=[(1,)])], shape=(3, 4))


def test_a_non_finite_scalar_is_refused_rather_than_printed():
    """NaN in a manuscript number is worse than an error here."""
    with pytest.raises(OutputError, match="finite"):
        validate_outputs([ScalarOutput(key="s", label="S", value=float("nan"))],
                         shape=(3, 4))


def test_a_non_finite_table_cell_is_refused_too():
    """Same argument, and JSON has no NaN token: Python writes one and
    JSON.parse rejects it. An add-on that means 'not measured' writes None."""
    with pytest.raises(OutputError, match="finite"):
        validate_outputs([TableOutput(key="t", label="T", columns=("a",),
                                      rows=[(float("inf"),)])], shape=(3, 4))


def test_a_table_cell_may_be_none_because_absence_is_expressible():
    out = validate_outputs([TableOutput(key="t", label="T", columns=("a", "b"),
                                        rows=[(None, "text")])], shape=(3, 4))
    assert out[0].rows[0] == (None, "text")


def test_a_map_of_all_nan_is_allowed_because_absence_is_data():
    out = validate_outputs([MapOutput(key="k", label="K",
                                      values=np.full((3, 4), np.nan))], shape=(3, 4))
    assert out[0].key == "k"


def test_an_unknown_output_type_is_refused_by_name():
    with pytest.raises(OutputError, match="unknown output"):
        validate_outputs(["not an output"], shape=(3, 4))


def test_returning_nothing_is_allowed():
    assert validate_outputs([], shape=(3, 4)) == ()


def test_map_values_are_returned_read_only():
    out = validate_outputs([MapOutput(key="k", label="K", values=np.zeros((3, 4)))],
                           shape=(3, 4))
    with pytest.raises(ValueError):
        out[0].values[0, 0] = 1.0


def test_a_map_that_is_not_numeric_is_refused_by_name_and_dtype():
    """An object map's bytes are the ADDRESSES of the Python objects it
    points at. Measured through the real endpoint before this check existed:
    200, dtype '|O', and eight-byte heap pointers in the body."""
    values = np.array([["a", "b", "c", "d"]] * 3, dtype=object)
    with pytest.raises(OutputError) as exc:
        validate_outputs([MapOutput(key="labels", label="L", values=values)],
                         shape=(3, 4))
    assert "labels" in str(exc.value) and "|O" in str(exc.value)


def test_a_string_map_is_refused_too():
    with pytest.raises(OutputError, match="numeric"):
        validate_outputs([MapOutput(key="s", label="S",
                                    values=np.full((3, 4), "x"))], shape=(3, 4))


def test_a_boolean_map_is_allowed():
    """A mask is a legitimate map and draws fine."""
    out = validate_outputs([MapOutput(key="m", label="Mask",
                                      values=np.zeros((3, 4), dtype=bool))],
                           shape=(3, 4))
    assert out[0].values.dtype == np.bool_


def test_a_non_finite_colour_bound_is_refused_on_the_map_the_plan_blesses():
    """The all-NaN map is allowed on purpose, and np.nanmin of one is NaN —
    so the natural colour bound for the blessed map is the one that 500s the
    whole response through JSONResponse's allow_nan=False."""
    values = np.full((3, 4), np.nan)
    with pytest.raises(OutputError, match="vmin"):
        validate_outputs([MapOutput(key="k", label="K", values=values,
                                    vmin=float("nan"))], shape=(3, 4))


def test_a_non_finite_vmax_is_refused_too():
    with pytest.raises(OutputError, match="vmax"):
        validate_outputs([MapOutput(key="k", label="K", values=np.zeros((3, 4)),
                                    vmax=float("inf"))], shape=(3, 4))


def test_colour_bounds_may_be_none():
    out = validate_outputs([MapOutput(key="k", label="K",
                                      values=np.zeros((3, 4)))], shape=(3, 4))
    assert out[0].vmin is None and out[0].vmax is None


def test_a_scalar_that_is_not_a_number_is_refused_here_not_in_the_route():
    """_is_finite passes strings by design (table cells carry text), so this
    reached the route's float(value) and raised OUTSIDE the runner's guard —
    a raw 500 naming no add-on."""
    with pytest.raises(OutputError, match="number"):
        validate_outputs([ScalarOutput(key="s", label="S", value="hello")],
                         shape=(3, 4))


def test_a_boolean_scalar_is_refused_because_bool_is_not_a_measurement():
    with pytest.raises(OutputError, match="number"):
        validate_outputs([ScalarOutput(key="s", label="S", value=True)],
                         shape=(3, 4))
# ---------------------------------------------------------------------------
# Fix round 1. Every test below is the same defect wearing a different hat:
# validate_outputs blessed a value that json.dumps then refused, which moves
# the raw exception out of the runner's guard instead of removing it. The
# closing test states the rule the individual ones are instances of.
# ---------------------------------------------------------------------------


def _json_safe_part(output):
    """Everything about an output that travels as JSON (a map's values do not)."""
    if isinstance(output, MapOutput):
        return {"key": output.key, "label": output.label, "unit": output.unit,
                "vmin": output.vmin, "vmax": output.vmax}
    if isinstance(output, TableOutput):
        return {"key": output.key, "label": output.label,
                "columns": list(output.columns),
                "rows": [list(r) for r in output.rows]}
    return {"key": output.key, "label": output.label, "value": output.value,
            "unit": output.unit}


#: Every field of every output kind that ``_json_safe_part`` serialises, and
#: for each a value that is LEGAL but not a plain Python object. The
#: post-condition test below fills all of them at once. Round 2 exists because
#: this fixture used to carry plain "M" and "deg" in the text fields, so the
#: test could not see that label and unit were unvalidated and uncoerced --
#: the test was sound, its inputs were too narrow. Adding a field to an output
#: without adding a row here is what this list is meant to make obvious.
_NON_PLAIN_BY_FIELD = {
    "key": "m",                        # keys are already required to be str
    "label": np.str_("Label"),
    "unit": np.str_("deg"),
    "vmin": np.float32(0),
    "vmax": np.int64(5),
    "value": np.float64(1.5),
    "columns": (np.str_("a"), np.str_("b"), np.str_("c"), np.str_("d")),
    "cells": (np.int64(1), np.float32(2.5), np.bool_(True), None),
}

_PLAIN_TYPES = (str, int, float, bool, type(None))


def _every_serialised_leaf(part):
    """Each scalar value in a _json_safe_part dict, flattened."""
    for value in part.values():
        if isinstance(value, list):
            for inner in value:
                if isinstance(inner, list):
                    yield from inner
                else:
                    yield inner
        else:
            yield value


def test_everything_that_leaves_validate_outputs_survives_json():
    """The general rule, not another instance of it.

    allow_nan=False is what Starlette's JSONResponse uses. Every field that
    _json_safe_part serialises is filled from _NON_PLAIN_BY_FIELD, so this
    holds for the whole surface rather than for the fields some finding
    happened to name. It asserts two things, and the second is the one that
    generalises: the encoder accepts the result, AND every serialised leaf is
    a plain Python object -- a numpy type that json happens to tolerate today
    would still fail here.
    """
    f = _NON_PLAIN_BY_FIELD
    out = validate_outputs([
        MapOutput(key="m", label=f["label"], values=np.zeros((3, 4), dtype=np.float32),
                  unit=f["unit"], vmin=f["vmin"], vmax=f["vmax"]),
        TableOutput(key="t", label=f["label"], columns=f["columns"],
                    rows=[f["cells"], (3, 4.5, False, "text")]),
        ScalarOutput(key="s", label=f["label"], value=f["value"], unit=f["unit"]),
    ], shape=(3, 4))

    parts = [_json_safe_part(o) for o in out]
    json.dumps(parts, allow_nan=False)
    for part in parts:
        for leaf in _every_serialised_leaf(part):
            assert type(leaf) in _PLAIN_TYPES, (
                f"{leaf!r} is a {type(leaf).__name__}, not a plain Python object")


def test_a_numpy_bound_is_a_plain_float_on_the_way_out():
    """np.float32 is what values.min() returns, so refusing it would make every
    add-on write float() by hand. It is coerced instead, and the COERCED value
    is what the output carries -- a check that left the original in place would
    have moved the TypeError to the encoder, which is finding 2 again."""
    out = validate_outputs([MapOutput(key="k", label="K",
                                      values=np.zeros((3, 4)),
                                      vmin=np.float32(0), vmax=np.int64(5))],
                           shape=(3, 4))
    assert type(out[0].vmin) is float and type(out[0].vmax) is float


def test_a_numpy_scalar_value_is_a_plain_float_on_the_way_out():
    out = validate_outputs([ScalarOutput(key="s", label="S",
                                         value=np.float64(2.5))], shape=(3, 4))
    assert type(out[0].value) is float and out[0].value == 2.5


def test_a_numpy_cell_is_a_plain_python_value_on_the_way_out():
    out = validate_outputs([TableOutput(key="t", label="T",
                                        columns=("a", "b", "c"),
                                        rows=[(np.int64(1), np.float32(2.5),
                                               np.bool_(True))])], shape=(3, 4))
    a, b, c = out[0].rows[0]
    assert type(a) is int and type(b) is float and type(c) is bool


def test_a_table_cell_that_is_not_plain_data_is_refused_with_its_coordinates():
    """Measured on the first draft: a Path cell passed validation and json.dumps
    then raised TypeError: Object of type WindowsPath is not JSON serializable."""
    with pytest.raises(OutputError) as exc:
        validate_outputs([TableOutput(key="t", label="T", columns=("a", "path"),
                                      rows=[(1, Path("x"))])], shape=(3, 4))
    message = str(exc.value)
    assert "'t'" in message and "row 0" in message and "'path'" in message


def test_a_callable_cell_is_refused_because_a_cell_is_data():
    with pytest.raises(OutputError, match="function"):
        validate_outputs([TableOutput(key="t", label="T", columns=("f",),
                                      rows=[(lambda: 1,)])], shape=(3, 4))


def test_a_non_finite_numpy_cell_is_refused_like_any_other():
    with pytest.raises(OutputError, match="finite"):
        validate_outputs([TableOutput(key="t", label="T", columns=("a",),
                                      rows=[(np.float32("nan"),)])], shape=(3, 4))


def test_a_numpy_bound_that_is_not_finite_is_refused_like_a_python_one():
    with pytest.raises(OutputError, match="vmin"):
        validate_outputs([MapOutput(key="k", label="K", values=np.zeros((3, 4)),
                                    vmin=np.float32("nan"))], shape=(3, 4))


def test_a_key_must_be_a_non_empty_string():
    """A key names a JSON field and, in Task 8b, a URL path segment."""
    with pytest.raises(OutputError, match="key"):
        validate_outputs([ScalarOutput(key=7, label="S", value=1)], shape=(3, 4))


def test_an_unhashable_key_is_an_output_error_not_a_type_error():
    """Measured: this raised TypeError: unhashable type: 'list' from inside
    validate_outputs itself -- the guard was the thing breaking the guarantee."""
    with pytest.raises(OutputError, match="key"):
        validate_outputs([ScalarOutput(key=["a"], label="S", value=1)],
                         shape=(3, 4))


def test_an_empty_key_is_refused():
    with pytest.raises(OutputError, match="key"):
        validate_outputs([ScalarOutput(key="", label="S", value=1)], shape=(3, 4))


def test_one_output_instead_of_a_list_is_told_so_rather_than_crashing():
    """The most ordinary add-on-author mistake there is. It used to arrive as
    TypeError: 'MapOutput' object is not iterable. It is NOT wrapped in a list
    for them -- that would be guessing at what they meant."""
    with pytest.raises(OutputError, match="sequence"):
        validate_outputs(MapOutput(key="k", label="K", values=np.zeros((3, 4))),
                         shape=(3, 4))


def test_something_that_is_not_a_sequence_at_all_is_refused_the_same_way():
    with pytest.raises(OutputError, match="sequence"):
        validate_outputs(42, shape=(3, 4))


def test_none_still_means_nothing_was_returned():
    assert validate_outputs(None, shape=(3, 4)) == ()


def test_duplicate_column_names_are_refused_like_duplicate_keys():
    """Two columns called 'a' make one of them unreadable in every consumer,
    for exactly the reason a duplicate output key is refused twenty lines up."""
    with pytest.raises(OutputError, match="duplicate"):
        validate_outputs([TableOutput(key="t", label="T", columns=("a", "a"),
                                      rows=[(1, 2)])], shape=(3, 4))


def test_a_column_name_must_be_a_string():
    with pytest.raises(OutputError, match="column"):
        validate_outputs([TableOutput(key="t", label="T", columns=("a", 2),
                                      rows=[(1, 2)])], shape=(3, 4))


def test_a_rebuilt_map_keeps_every_field_where_it_belongs():
    """validate_outputs reconstructs a MapOutput to swap in the frozen view.
    Nothing pinned the field order, so swapping unit for vmin in the rebuild
    kept all nineteen earlier tests green -- silently voiding "units live in
    field names", which is one of the three load-bearing rules of the contract.
    """
    values = np.arange(12, dtype=float).reshape(3, 4)
    out = validate_outputs([MapOutput(key="kam", label="KAM", values=values,
                                      unit="deg", vmin=0.0, vmax=5.0)],
                           shape=(3, 4))[0]
    assert out.key == "kam"
    assert out.label == "KAM"
    assert out.unit == "deg"
    assert out.vmin == 0.0
    assert out.vmax == 5.0
    np.testing.assert_array_equal(out.values, values)


def test_a_rebuilt_table_keeps_every_field_where_it_belongs():
    out = validate_outputs([TableOutput(key="t", label="Grains",
                                        columns=("ecd", "n"),
                                        rows=[(1.5, 2)])], shape=(3, 4))[0]
    assert out.key == "t" and out.label == "Grains"
    assert out.columns == ("ecd", "n") and out.rows == ((1.5, 2),)


def test_a_rebuilt_scalar_keeps_every_field_where_it_belongs():
    out = validate_outputs([ScalarOutput(key="s", label="Mean ECD",
                                         value=1.5, unit="um")], shape=(3, 4))[0]
    assert out.key == "s" and out.label == "Mean ECD"
    assert out.value == 1.5 and out.unit == "um"


def test_the_frozen_view_is_a_guard_against_accident_not_a_boundary():
    """Documented rather than fixed: the returned view is read-only, but its
    .base is the add-on's own array and stays writable. Copying every map to
    close that is a real cost, and the gap disappears once the runner is a
    separate process. This test exists so the docstring's claim is a measured
    one rather than a hope."""
    values = np.zeros((3, 4))
    out = validate_outputs([MapOutput(key="k", label="K", values=values)],
                           shape=(3, 4))[0]
    with pytest.raises(ValueError):
        out.values[0, 0] = 1.0
    values[0, 0] = 9.0          # the add-on still holds the writable original
    assert out.values[0, 0] == 9.0
# ---------------------------------------------------------------------------
# Fix round 2. Round 1 closed the number fields and stopped one field short:
# label and unit were the same defect, unvalidated. The tests below are
# organised by the property that is being kept, not by the field that was
# reported, which is the only arrangement that survives the next field.
# ---------------------------------------------------------------------------

_ALL_KINDS = ("map", "table", "scalar")


def _output_with(kind, **overrides):
    """One of each output kind, well formed unless a field is overridden."""
    if kind == "map":
        fields = dict(key="k", label="K", values=np.zeros((3, 4)), unit="deg")
        fields.update(overrides)
        return MapOutput(**fields)
    if kind == "table":
        fields = dict(key="t", label="T", columns=("a",), rows=[(1,)])
        fields.update({k: v for k, v in overrides.items() if k != "unit"})
        return TableOutput(**fields)
    fields = dict(key="s", label="S", value=1.0, unit="um")
    fields.update(overrides)
    return ScalarOutput(**fields)


@pytest.mark.parametrize("kind", _ALL_KINDS)
@pytest.mark.parametrize("bad", [Path("x"), np.int64(5), 7, None, ""])
def test_a_label_must_be_a_non_empty_string_on_every_kind(kind, bad):
    """Measured on the round-1 code: label=Path('x') validated, and json.dumps
    then raised TypeError: Object of type WindowsPath is not JSON serializable
    -- findings 1 and 2 of round 1, verbatim, one field along."""
    with pytest.raises(OutputError, match="label"):
        validate_outputs([_output_with(kind, label=bad)], shape=(3, 4))


@pytest.mark.parametrize("kind", ["map", "scalar"])
@pytest.mark.parametrize("bad", [np.float32(3), Path("x"), 7, ""])
def test_a_unit_must_be_a_non_empty_string_or_none(kind, bad):
    with pytest.raises(OutputError, match="unit"):
        validate_outputs([_output_with(kind, unit=bad)], shape=(3, 4))


def test_a_callable_unit_is_refused_like_any_other_non_string():
    with pytest.raises(OutputError, match="unit"):
        validate_outputs([_output_with("map", unit=lambda: "deg")], shape=(3, 4))


@pytest.mark.parametrize("kind", ["map", "scalar"])
def test_no_unit_is_still_how_a_bare_number_is_expressed(kind):
    """Units live in field names, and a count has no unit. None must stay legal
    or the contract would force an add-on to invent one."""
    out = validate_outputs([_output_with(kind, unit=None)], shape=(3, 4))
    assert out[0].unit is None


@pytest.mark.parametrize("kind", _ALL_KINDS)
def test_a_numpy_string_is_plain_on_the_way_out(kind):
    """np.str_ is a str subclass, so JSON tolerates it -- but the contract says
    plain Python objects, and a promise with an exception in it is not one."""
    out = validate_outputs([_output_with(kind, label=np.str_("Lbl"))],
                           shape=(3, 4))[0]
    assert type(out.label) is str
    if kind == "table":
        out2 = validate_outputs([TableOutput(key="t", label="T",
                                             columns=(np.str_("a"),),
                                             rows=[(1,)])], shape=(3, 4))[0]
        assert type(out2.columns[0]) is str
    else:
        out3 = validate_outputs([_output_with(kind, unit=np.str_("deg"))],
                                shape=(3, 4))[0]
        assert type(out3.unit) is str


# --- raw exceptions that still escaped -------------------------------------


def test_rows_that_are_not_a_sequence_are_an_output_error():
    """Measured: TypeError: 'int' object is not iterable."""
    with pytest.raises(OutputError, match="rows"):
        validate_outputs([TableOutput(key="t", label="T", columns=("a",),
                                      rows=5)], shape=(3, 4))


def test_a_row_that_is_not_a_sequence_is_an_output_error():
    with pytest.raises(OutputError, match="row 0"):
        validate_outputs([TableOutput(key="t", label="T", columns=("a",),
                                      rows=[5])], shape=(3, 4))


def test_columns_that_are_not_a_sequence_are_an_output_error():
    with pytest.raises(OutputError, match="columns"):
        validate_outputs([TableOutput(key="t", label="T", columns=5,
                                      rows=[(1,)])], shape=(3, 4))


def test_a_ragged_map_is_an_output_error_not_a_numpy_one():
    """Measured: ValueError out of np.asarray, straight through the guard."""
    with pytest.raises(OutputError, match="rectangular|array"):
        validate_outputs([MapOutput(key="k", label="K",
                                    values=[[1, 2], [3]])], shape=(3, 4))


def test_an_integer_too_big_for_isfinite_does_not_crash_the_guard():
    """Measured: TypeError: ufunc 'isfinite' not supported for the input types
    -- np.isfinite cannot take a Python int wider than 64 bits, so the
    finiteness guard was itself a raw exception."""
    with pytest.raises(OutputError):
        validate_outputs([ScalarOutput(key="s", label="S", value=2 ** 70)],
                         shape=(3, 4))


# --- coercion is exact, or it is refused ------------------------------------


@pytest.mark.parametrize("value", [2 ** 53, -(2 ** 53) - 1, 2 ** 70,
                                   np.int64(2 ** 60)])
def test_an_integer_javascript_cannot_read_back_is_refused(value):
    """JSON carries it; JavaScript rounds it. Number.MAX_SAFE_INTEGER is
    2**53 - 1, and this value is going to a browser."""
    with pytest.raises(OutputError, match="exact|safe"):
        validate_outputs([ScalarOutput(key="s", label="S", value=value)],
                         shape=(3, 4))


def test_the_same_integer_limit_applies_to_a_cell_and_to_a_colour_bound():
    with pytest.raises(OutputError, match="exact|safe"):
        validate_outputs([TableOutput(key="t", label="T", columns=("a",),
                                      rows=[(2 ** 60,)])], shape=(3, 4))
    with pytest.raises(OutputError, match="exact|safe"):
        validate_outputs([MapOutput(key="k", label="K", values=np.zeros((3, 4)),
                                    vmin=2 ** 60)], shape=(3, 4))


def test_an_integer_at_the_edge_is_still_carried():
    out = validate_outputs([ScalarOutput(key="s", label="S",
                                         value=2 ** 53 - 1)], shape=(3, 4))
    assert out[0].value == 2 ** 53 - 1


@pytest.mark.skipif(
    np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant,
    reason="on this platform np.longdouble IS float64, so float() loses nothing")
def test_a_wider_than_float64_value_is_refused_where_one_exists():
    with pytest.raises(OutputError, match="exact"):
        validate_outputs([ScalarOutput(key="s", label="S",
                                       value=np.longdouble("1.2345678901234567890"))],
                         shape=(3, 4))


@pytest.mark.skipif(
    np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant,
    reason="this pins the OTHER platform: longdouble wider than float64")
def test_a_longdouble_that_is_exactly_float64_is_carried_not_refused():
    """Refusing by TYPE would be wrong here: numpy on Windows gives longdouble
    a 52-bit mantissa, so float() is exact and a refusal would claim a loss
    that did not happen. The guard asks whether the conversion lost anything,
    which is the same question on every platform."""
    out = validate_outputs([ScalarOutput(key="s", label="S",
                                         value=np.longdouble("0.5"))],
                           shape=(3, 4))
    assert type(out[0].value) is float and out[0].value == 0.5


def test_ordinary_numpy_scalars_are_still_coerced_not_refused():
    """The line is 'converted exactly' vs 'quietly changed the number'.
    values.mean() is the most ordinary add-on line there is."""
    out = validate_outputs([ScalarOutput(key="s", label="S",
                                         value=np.zeros(4, dtype=np.float32).mean())],
                           shape=(3, 4))
    assert type(out[0].value) is float


# --- messages carry a remedy -----------------------------------------------


def test_the_non_finite_scalar_message_says_what_to_do_instead():
    """These are read by research groups with no access to our source. A
    message that names a problem and stops sends them to the source."""
    with pytest.raises(OutputError, match="do not emit"):
        validate_outputs([ScalarOutput(key="s", label="S", value=float("nan"))],
                         shape=(3, 4))


def test_the_non_finite_bound_message_says_what_to_do_instead():
    with pytest.raises(OutputError, match="None"):
        validate_outputs([MapOutput(key="k", label="K", values=np.zeros((3, 4)),
                                    vmin=float("nan"))], shape=(3, 4))


def test_the_shape_message_says_which_axis_is_rows():
    with pytest.raises(OutputError, match="row"):
        validate_outputs([MapOutput(key="k", label="K", values=np.zeros((4, 3)))],
                         shape=(3, 4))


# --- the class, not the instance -------------------------------------------


@pytest.mark.parametrize("kind,field", [
    ("map", "label"), ("map", "unit"), ("map", "vmin"), ("map", "vmax"),
    ("table", "label"),
    ("scalar", "label"), ("scalar", "unit"), ("scalar", "value"),
])
def test_every_serialised_field_refuses_what_json_cannot_carry(kind, field):
    """One row per field that reaches the encoder. A Path is refused by every
    one of them -- the property is 'this field is validated', not 'this
    particular finding was fixed'. A new field on an output is a new row."""
    with pytest.raises(OutputError):
        validate_outputs([_output_with(kind, **{field: Path("x")})], shape=(3, 4))


# --- a key is a URL path segment, so it has to be able to be one -----------

@pytest.mark.parametrize("key, offender", [
    ("a/b", "/"),
    ("a#b", "#"),
    ("a?b", "?"),
    ("a b", " "),
    ("../escape", "/"),
    ("kam\u00b0", "\u00b0"),
])
def test_a_key_that_cannot_be_a_url_segment_is_refused(key, offender):
    """``_check_key``'s own comment says a key "names a URL path segment" and
    then checked nothing of the sort. A map output is answered with a 200 that
    advertises ``values_url``, and a key carrying '/', '#' or '?' makes that a
    link which can never resolve -- the same defect as a map URL missing its
    analysis key, one layer out."""
    with pytest.raises(OutputError) as excinfo:
        validate_outputs([MapOutput(key=key, label="L",
                                    values=np.zeros((2, 2)))], shape=(2, 2))
    assert repr(offender) in str(excinfo.value) or offender in str(excinfo.value)
    assert repr(key) in str(excinfo.value)


def test_the_ordinary_keys_an_addon_writes_are_still_accepted():
    for key in ("histogram", "component_0_mean", "bc.hist", "kam-map", "x1"):
        out = validate_outputs(
            [MapOutput(key=key, label="L", values=np.zeros((2, 2)))],
            shape=(2, 2))
        assert out[0].key == key


@pytest.mark.parametrize("key", ["..", ".", "..."])
def test_a_key_of_nothing_but_dots_is_refused(key):
    """``..`` passes the character class and is collapsed by every URL
    resolver, so the link a 200 advertises resolves somewhere else entirely --
    the exact failure the class exists to prevent, walking through it."""
    with pytest.raises(OutputError) as excinfo:
        validate_outputs([MapOutput(key=key, label="L",
                                    values=np.zeros((2, 2)))], shape=(2, 2))
    assert repr(key) in str(excinfo.value)
