"""What an add-on returns, and what Orienta will draw from it.

An add-on declares its results; it does not render them. A map becomes a layer
in the existing stack and inherits the colour scale, the image export, the
annotations and the scale bar — none of which know any layer type, which is
exactly why they work for anything that serves the protocol. An add-on that
drew its own canvas would be cut off from all four.

Slicer's CLI modules make the same trade and name the same price: such a module
"cannot pass intermediate results, update views during execution, accept
steering input, or request data interactively". For "map in, numbers out" that
is the right constraint.

THE RULE THIS MODULE EXISTS TO KEEP
-----------------------------------
No failure path may leave the runner as a raw exception. That is stronger than
it first reads, and it was broken the same way three reviews running: a guard
that refuses ``float("nan")`` but waves through ``np.float32("nan")`` has not
honoured the rule, it has moved the raw exception from here into ``json.dumps``,
where nothing names the add-on. Then, one round later: guards on every NUMBER
field and none on ``label`` or ``unit``, which are serialised beside them.

So ``validate_outputs`` does not only CHECK; it NORMALISES, and it does so for
EVERY field that reaches the encoder rather than for the fields some finding
happened to name. Everything it returns is built from plain Python objects that
``json.dumps(..., allow_nan=False)`` accepts — that encoder, with that flag,
being what Starlette's ``JSONResponse`` runs. A map's ``values`` are the one
deliberate exception: they leave as an array and travel as raw bytes, never as
JSON, which is why their dtype is restricted instead of coerced.

COERCED, OR REFUSED — AND THE LINE BETWEEN THEM
------------------------------------------------
Numpy scalars are coerced rather than refused, because ``values.mean()`` returns
one and making every add-on write ``float(...)`` by hand would tax the most
ordinary line there is. That is only honest while the conversion is EXACT:
``float(np.float32)`` and ``int(np.int64)`` are, so nothing is guessed. Where a
conversion would LOSE something, the value is refused and told so:

* a float wider than ``float64`` — ``float()`` truncates it. The guard asks
  whether the conversion lost anything (``float(x) != x``) rather than refusing
  a type, because ``np.longdouble`` is 80-bit on Linux and MEASURED at 52-bit
  (i.e. exactly ``float64``) on Windows. Refusing it by type would claim a loss
  that did not happen on half the platforms we run on.
* an integer beyond ``2**53 - 1`` — JSON carries it and JavaScript, which is
  where it is going, silently rounds it. Exact in Python, wrong in the browser.

That is the line: converted exactly, or the number is quietly changed and we
say so instead.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import numpy as np


class OutputError(Exception):
    """An add-on returned something Orienta cannot draw. Names the offender."""


@dataclass(frozen=True)
class MapOutput:
    key: str
    label: str
    values: np.ndarray
    unit: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None


@dataclass(frozen=True)
class TableOutput:
    key: str
    label: str
    columns: Tuple[str, ...]
    rows: Sequence[Sequence[Any]]


@dataclass(frozen=True)
class ScalarOutput:
    key: str
    label: str
    value: float
    unit: Optional[str] = None


AddonOutput = Union[MapOutput, TableOutput, ScalarOutput]

#: Map dtype kinds Orienta can colour and serve as bytes: bool, signed int,
#: unsigned int, float. NOT object or str (their bytes are heap addresses or
#: padded code points, neither of which np.frombuffer can read back), and not
#: complex (a colour scale would have to invent a projection).
_MAP_DTYPE_KINDS = "biuf"

#: Returned by ``_as_json_value`` for anything outside the whitelist. A
#: sentinel and not ``None``, because ``None`` is itself a legal cell: it is
#: how an add-on says "not measured", and JSON ``null`` reads as absence.
_NOT_JSON = object()

#: JavaScript's ``Number.MAX_SAFE_INTEGER``. Above it, JSON round-trips through
#: a browser with the wrong value and no error anywhere.
_MAX_SAFE_INT = 2 ** 53 - 1

#: What may appear in an output key. Letters, digits, dot, underscore, hyphen:
#: the characters that are themselves in a URL path segment, in a JSON field
#: name and in an h5 name, with no encoding anywhere between. ``routes/addons``
#: mints ``values_url`` by interpolation and quotes nothing, so a key outside
#: this set produces a 200 that advertises a link which can never resolve --
#: '/' splits the segment, '#' truncates it, '?' turns the rest into a query.
#: Kept deliberately narrower than "what percent-encoding could carry": the
#: same string is also a dict key in the map store and a column heading, and a
#: key nobody can type is a key nobody can report a bug about.
_KEY_CHARS = re.compile(r"[A-Za-z0-9._-]+")


def _as_json_number(value: Any) -> Optional[Union[int, float]]:
    """A plain Python ``int``/``float`` for any real number, else ``None``.

    The one place in this module that decides what counts as a number, so a
    numpy scalar cannot be a number in one check and not in the next — which is
    precisely how a NaN ``np.float32`` bound once reached the encoder. ``bool``
    is excluded first and deliberately: it is an ``int`` subclass, and a
    ``True`` recorded into a methods paragraph renders as a measurement.
    """
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return None


def _number_problem(number: Union[int, float], original: Any) -> Optional[str]:
    """Why this number cannot travel, or ``None``.

    One definition, used by colour bounds, table cells and scalars alike, so
    the three cannot drift apart. It returns the PROBLEM and not the remedy:
    the remedy differs per field (a bound may be ``None``, a scalar may simply
    be left out) and belongs to the caller that knows which field it is.

    ``math.isfinite`` and not ``np.isfinite``: the numpy one raises
    ``TypeError: ufunc 'isfinite' not supported`` on a Python int wider than 64
    bits, so the finiteness guard was itself a raw exception. A Python int is
    finite by construction and needs no test at all.
    """
    if isinstance(number, float):
        if not math.isfinite(number):
            return f"is not finite ({original!r})"
        if float(original) != original:
            return (f"is {original!r}, which float64 cannot hold exactly; "
                    "converting it would quietly change the number")
        return None
    if abs(number) > _MAX_SAFE_INT:
        return (f"is {original!r}, which is beyond JavaScript's safe integer "
                f"range (+/-{_MAX_SAFE_INT}); JSON would carry it and the "
                "browser would read back a different number")
    return None


def _as_json_value(value: Any) -> Any:
    """A table cell as plain data, or ``_NOT_JSON``.

    A whitelist and not a blacklist: the first draft asked "is this a
    non-finite number?", which every non-number passed, so a ``Path`` and a
    lambda both validated and ``json.dumps`` then raised ``TypeError`` outside
    the runner's guard. Asking "is this one of the things a cell may be?"
    cannot fail that way for a type nobody thought of.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return str(value)                       # np.str_ is a str subclass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    number = _as_json_number(value)
    if number is None:
        return _NOT_JSON
    return number


def _check_text(value: Any, subject: str, *, optional: bool) -> Optional[str]:
    """A field that is shown to a human and serialised as JSON.

    ``label`` and ``unit`` were unvalidated for three rounds while every number
    beside them was guarded, so ``label=Path("x")`` validated and then raised
    ``TypeError`` in the encoder. ``str(value)`` normalises ``np.str_``, which
    JSON tolerates but which is not the plain Python object this contract
    promises.
    """
    if value is None:
        if optional:
            return None
        raise OutputError(
            f"{subject} is missing; give it a short name to show in the "
            "interface and in an export")
    if not isinstance(value, str) or not value:
        raise OutputError(
            f"{subject} is {value!r} ({type(value).__name__}); it must be a "
            "non-empty string"
            + (", or None if there is no unit" if optional else "")
            + ". It is shown to a human and written into JSON.")
    return str(value)


def _check_key(key: Any) -> str:
    """A key names a JSON field and a URL path segment. Both, actually.

    Checked BEFORE the duplicate lookup, because an unhashable key used to
    raise ``TypeError: unhashable type: 'list'`` out of the ``seen`` set — the
    guard itself becoming the raw exception it exists to prevent.

    The docstring said "a URL path segment" for three rounds while the check
    asked only "is it a non-empty string?", so ``key="a/b"`` validated, was
    served, and came back as a 200 advertising ``values_url`` with an extra
    slash in it — a link that can never resolve, on a response that says
    everything worked. A declared shape that nothing enforces is a comment;
    that is this module's own lesson about ``label`` and ``unit``, one field
    along.
    """
    if not isinstance(key, str) or not key:
        raise OutputError(
            f"output key {key!r} is not usable: a key must be a non-empty "
            "string. It names a JSON field and a URL path segment, so it "
            "cannot be a number, a list or blank.")
    if not _KEY_CHARS.fullmatch(key):
        bad = next(c for c in key if not _KEY_CHARS.fullmatch(c))
        raise OutputError(
            f"output key {key!r} contains {bad!r}, which cannot appear in a "
            "URL path segment. A key may use letters, digits, '.', '_' and "
            "'-' only: it names the path this output's values are served "
            "from, so a key with '/', '#', '?' or a space in it would be "
            "answered with a link nothing can fetch.")
    if set(key) == {"."}:
        # '.' and '..' pass the character class and are then COLLAPSED by
        # every URL resolver, so the link a 200 advertises points somewhere
        # else entirely -- the same "listed, and unfetchable" failure walking
        # straight through the guard that exists to stop it.
        raise OutputError(
            f"output key {key!r} is nothing but dots, which a URL resolver "
            "removes or reads as the directory above. Give the output a name.")
    return str(key)


def _tuple_or_error(value: Any, subject: str, remedy: str) -> Tuple[Any, ...]:
    """``tuple(value)``, or a readable refusal instead of a ``TypeError``.

    Used for the add-on's whole return, for ``columns`` and for each row — the
    same accident in three places, and it escaped as ``TypeError: 'int' object
    is not iterable`` from all three.
    """
    try:
        return tuple(value)
    except TypeError as exc:
        raise OutputError(
            f"{subject} is {value!r} ({type(value).__name__}), which is not a "
            f"sequence. {remedy}") from exc


def _as_sequence(outputs: Any) -> Tuple[Any, ...]:
    """The add-on's return value as a tuple of items, or a readable refusal.

    Returning one output instead of a one-element list is the most ordinary
    mistake an add-on author makes, and it used to arrive as ``TypeError:
    'MapOutput' object is not iterable``. It is refused rather than wrapped:
    wrapping would be guessing at what the author meant, and this contract
    shows what it cannot produce instead of guessing at it.
    """
    if outputs is None:
        return ()
    if isinstance(outputs, (MapOutput, TableOutput, ScalarOutput)):
        raise OutputError(
            f"an add-on returned a bare {type(outputs).__name__}; return a "
            "sequence of outputs, e.g. [output], even when there is only one")
    return _tuple_or_error(
        outputs, "an add-on's return value",
        "Return a list or tuple of outputs, or None for nothing.")


def _validate_map(item: MapOutput, shape: Tuple[int, int]) -> MapOutput:
    label = _check_text(item.label, f"output {item.key!r}: label", optional=False)
    unit = _check_text(item.unit, f"output {item.key!r}: unit", optional=True)

    try:
        values = np.asarray(item.values)
    except (ValueError, TypeError) as exc:
        # A ragged nested list reaches numpy and dies there; the add-on author
        # sees a numpy traceback about inhomogeneous shapes and no output name.
        raise OutputError(
            f"output {item.key!r}: values cannot be read as an array "
            f"({exc.__class__.__name__}: {exc}); a map must be rectangular, "
            "one value per scan pixel") from exc
    rows, cols = tuple(shape)
    if values.shape != (rows, cols):
        raise OutputError(
            f"output {item.key!r}: shape {values.shape} does not match the "
            f"scan grid {(rows, cols)}. A map is indexed [row, column], so it "
            f"must have {rows} rows and {cols} columns.")
    if values.dtype.kind not in _MAP_DTYPE_KINDS:
        raise OutputError(
            f"output {item.key!r}: values have dtype {values.dtype.str!r}; a "
            "map must be numeric or boolean. Its bytes are served raw over "
            "HTTP, and an object array would serve the addresses of the "
            f"Python objects it points at — which the client cannot read back "
            "and must not see.")

    bounds = {}
    for name, bound in (("vmin", item.vmin), ("vmax", item.vmax)):
        # JSONResponse uses allow_nan=False, so a NaN here is a 500 for the
        # WHOLE response, not one field. An all-NaN map is allowed on purpose,
        # and the bound an add-on naturally writes for one is np.nanmin -> NaN.
        if bound is None:
            bounds[name] = None
            continue
        number = _as_json_number(bound)
        if number is None:
            raise OutputError(
                f"output {item.key!r}: {name} is {bound!r} "
                f"({type(bound).__name__}); a colour bound must be a number. "
                "Pass None to let Orienta scale the map itself.")
        problem = _number_problem(number, bound)
        if problem is not None:
            raise OutputError(
                f"output {item.key!r}: {name} {problem}. Pass None to let "
                "Orienta scale the map itself.")
        bounds[name] = float(number)

    # BOTH bounds, together. Each was checked alone, and every other bad bound
    # is refused here by name -- but an inverted pair passed, and the layer
    # painter then finds vmax <= vmin, widens it by 1e-6 and produces a map of
    # ONE colour with a legend reading "1.0 to 1.000001". Measured. That does
    # not look like an error; it looks like a uniform field, which is a claim
    # about the sample. Orienta's own layers can only collapse a range when
    # the data are degenerate; these bounds come from an outsider, and writing
    # (vmax, vmin) in the wrong order is an ordinary slip.
    if bounds["vmin"] is not None and bounds["vmax"] is not None:
        # STRICTLY greater. An earlier version refused vmin >= vmax and
        # disabled the shipped reference add-on: its component_map declares
        # vmin=0, vmax=n-1, so n_components=1 -- a value its own manifest
        # declares legal -- produced vmin == vmax == 0, three refusals in a
        # row hit CRASH_LIMIT, and the add-on was switched off with a message
        # blaming its author. Measured end to end.
        #
        # vmin == vmax is not the slip this guard is for: it is what a label
        # map says when it has ONE label, and the changelog holds up
        # vmin=0/vmax=n-1 as the canonical pattern. The painter widens an
        # equal pair by 1e-6 and paints one colour, which for one label is
        # the truth. An INVERTED pair is the slip, and it stays refused.
        if bounds["vmin"] > bounds["vmax"]:
            raise OutputError(
                f"output {item.key!r}: vmin ({bounds['vmin']}) is above vmax "
                f"({bounds['vmax']}). A colour range has to run upwards; an "
                "inverted one paints every pixel the same colour and says so "
                "in the legend. Pass None for either bound to let Orienta "
                "scale the map itself.")

    frozen = values.view()
    frozen.setflags(write=False)
    # Keyword arguments, not positions: an earlier rebuild was positional, and
    # swapping unit for vmin in it left every test of the day green while
    # silently voiding "units live in field names".
    return MapOutput(key=item.key, label=label, values=frozen, unit=unit,
                     vmin=bounds["vmin"], vmax=bounds["vmax"])


def _validate_table(item: TableOutput) -> TableOutput:
    label = _check_text(item.label, f"output {item.key!r}: label", optional=False)
    raw_columns = _tuple_or_error(
        item.columns, f"output {item.key!r}: columns",
        "Give one column name per value in a row.")

    columns = []
    seen_columns = set()
    for column in raw_columns:
        name = _check_text(column, f"output {item.key!r}: column name {column!r}",
                           optional=False)
        if name in seen_columns:
            raise OutputError(
                f"output {item.key!r}: duplicate column name {name!r}; two "
                "columns with one name make one of them unreadable in every "
                "consumer, for the same reason a duplicate output key does")
        seen_columns.add(name)
        columns.append(name)
    columns = tuple(columns)

    width = len(columns)
    rows = []
    for i, raw_row in enumerate(_tuple_or_error(
            item.rows, f"output {item.key!r}: rows",
            "Give a sequence of rows, each a sequence of values.")):
        raw_row = _tuple_or_error(
            raw_row, f"output {item.key!r}: row {i}",
            f"Each row is a sequence of {width} values.")
        if len(raw_row) != width:
            raise OutputError(
                f"output {item.key!r}: row {i} has {len(raw_row)} values for "
                f"{width} columns")
        row = []
        for j, raw_cell in enumerate(raw_row):
            cell = _as_json_value(raw_cell)
            if cell is _NOT_JSON:
                raise OutputError(
                    f"output {item.key!r}: row {i} column {columns[j]!r} is a "
                    f"{type(raw_cell).__name__}, which is not data a table can "
                    "carry. A cell is None, a string, a bool or a number; "
                    "anything else cannot be serialised or shown.")
            number = _as_json_number(cell)
            if number is not None:
                # Same rules as a scalar, and for a second reason: JSON has no
                # NaN token. Python writes one; JSON.parse rejects it, so a NaN
                # here breaks the whole response, not one cell.
                problem = _number_problem(number, raw_cell)
                if problem is not None:
                    raise OutputError(
                        f"output {item.key!r}: row {i} column {columns[j]!r} "
                        f"{problem}; write None for a value that was not "
                        "measured")
            row.append(cell)
        rows.append(tuple(row))
    return TableOutput(key=item.key, label=label, columns=columns,
                       rows=tuple(rows))


def _validate_scalar(item: ScalarOutput) -> ScalarOutput:
    label = _check_text(item.label, f"output {item.key!r}: label", optional=False)
    unit = _check_text(item.unit, f"output {item.key!r}: unit", optional=True)

    # The route sends this number to JSON. A string passed the old finiteness
    # check by design (table cells carry text), reached the route's
    # float(value) and raised OUTSIDE the runner's guard — a raw 500 naming no
    # add-on, which is the one thing the Global Constraints forbid.
    number = _as_json_number(item.value)
    if number is None:
        raise OutputError(
            f"output {item.key!r}: value is {item.value!r} "
            f"({type(item.value).__name__}); a scalar must be a real number. "
            "For a value that was not measured, do not emit the scalar at all "
            f"— its absence is readable, a placeholder is not.")
    # A NaN map is data — unindexed pixels are genuinely absent. A NaN scalar is
    # a number someone would paste into a paper.
    problem = _number_problem(number, item.value)
    if problem is not None:
        raise OutputError(
            f"output {item.key!r}: value {problem}. For a value that was not "
            f"measured, do not emit the scalar at all — its absence is "
            "readable, a placeholder is not.")
    return ScalarOutput(key=item.key, label=label, value=number, unit=unit)


def validate_outputs(outputs: Iterable[Any], *,
                     shape: Tuple[int, int]) -> Tuple[AddonOutput, ...]:
    """Check an add-on's returns against the grid they claim to describe.

    Returns a NEW tuple of NEW output objects — this normalises as well as
    checks, and the caller must use what comes back rather than what it passed
    in. Two things change on the way through:

    * every field that is serialised — ``key``, ``label``, ``unit``, column
      names, ``vmin``/``vmax``, a scalar's ``value`` and every table cell —
      becomes a plain Python object, so nothing downstream can hand
      ``json.dumps`` a type it refuses. Numpy scalars and ``np.str_`` are
      coerced where that is exact and refused where it would lose something;
    * a map's ``values`` become a read-only VIEW of the add-on's array.

    That freeze is a guard against the accident, NOT a boundary. ``view.base``
    is the add-on's own array and stays writable, so an add-on that keeps a
    reference can still change the buffer after validation — measured, and
    pinned by a test. Copying every map to close that is a real cost on a
    per-pixel array, and the gap disappears once the runner moves into a
    separate process, which is what the numpy-only boundary exists to allow.
    ``context.py`` documents the identical mechanism on the way in.

    Raises:
        OutputError: for anything Orienta cannot draw or serialise, naming the
            output and, for a table, the row and column, and saying what to
            write instead. Every refusal in this module is an ``OutputError``;
            nothing here may reach the runner as a raw exception, since that is
            the failure this contract is written against. Add-on authors read
            these messages without access to this source, so a message that
            names a problem without a remedy sends them somewhere they cannot
            go.
    """
    checked = []
    seen = set()
    for item in _as_sequence(outputs):
        if not isinstance(item, (MapOutput, TableOutput, ScalarOutput)):
            raise OutputError(f"unknown output type {type(item).__name__}")
        key = _check_key(item.key)
        if key in seen:
            raise OutputError(f"duplicate output key {key!r}")
        seen.add(key)

        if isinstance(item, MapOutput):
            checked.append(_validate_map(item, shape))
        elif isinstance(item, TableOutput):
            checked.append(_validate_table(item))
        else:
            checked.append(_validate_scalar(item))
    return tuple(checked)
