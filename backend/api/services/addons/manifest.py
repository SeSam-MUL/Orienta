"""What an add-on declares about itself, and nothing else.

The manifest exists so Orienta can build its catalogue WITHOUT importing add-on
code. That indirection is the whole design: this application's start-up already
imports kikuchipy, orix, diffsims and torch, and an add-on whose code runs to
answer "what do you offer?" lands in that same path.

So this module reads one file and validates it. It imports nothing from the
add-on and nothing heavy from Orienta.
"""
from __future__ import annotations

import re
import string
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

API_VERSION = 0

_REQUIRED_TOP = ("name", "display_name", "version", "authors")
_REQUIRED_ANALYSIS = ("key", "label", "python_name", "sentence")
_KEY_PREFIX = "addon."
# Characters BibTeX cannot carry in an entry key. render_bibtex writes the
# citation id straight into "@misc{<id>," without escaping it, so a reference
# containing one of these produces a .bib file no reader can parse.
_BAD_IN_REFERENCE = set(",{}= \t\n\r")
_PATH_FORMAT = "path"
#: What ``name`` and an analysis ``key`` may contain. Both are interpolated
#: into URLs without quoting -- ``/api/addons/{name}/run`` and the
#: ``values_url`` of a map output, which carries the name AND the key -- and
#: both are dict keys in process-level stores. ``outputs._KEY_CHARS`` is the
#: same set for the same reason; the two are deliberately identical, so an
#: author meets one rule and not three.
_SEGMENT_CHARS = re.compile(r"[A-Za-z0-9._-]+")


class ManifestError(Exception):
    """A manifest could not be read or does not meet the contract.

    Always raised with the offending field named, because the author reading
    it has no debugger into Orienta.
    """


@dataclass(frozen=True)
class AnalysisSpec:
    key: str
    label: str
    python_name: str
    sentence: str
    citations: Tuple[str, ...]
    params_schema: Dict[str, Any]


@dataclass(frozen=True)
class AddonManifest:
    api_version: int
    name: str
    display_name: str
    version: str
    requires_orienta: str
    authors: Tuple[str, ...]
    doi: str
    analyses: Tuple[AnalysisSpec, ...]
    source_path: Path


def declared_params(spec: AnalysisSpec) -> Dict[str, dict]:
    """The parameters this analysis declares, as JSON Schema properties.

    An analysis with no ``params_schema`` declares NO parameters, and the
    runner will accept none — the manifest is the contract, and a run cannot
    pass something the manifest never promised to understand.
    """
    properties = (spec.params_schema or {}).get("properties")
    return dict(properties) if isinstance(properties, dict) else {}


def recordable_params(spec: AnalysisSpec) -> Tuple[str, ...]:
    """Declared parameters minus the ones that must never reach the trail.

    A ``format = "path"`` parameter is handed to the add-on and deliberately
    NOT recorded: recorded params are written into the methods paragraph and
    into every exported ``.h5``, and a local path there publishes the
    operator's username and directory layout (see
    ``citations.provenance._reject_local_paths``, which refuses one outright
    under pytest). Excluded by declaration, not by sniffing the value.
    """
    return tuple(
        name for name, schema in declared_params(spec).items()
        if (schema or {}).get("format") != _PATH_FORMAT
    )


#: What a ``default`` in a params_schema may be made of. TOML's own value
#: types minus the four date/time ones -- which are the whole point: a
#: ``default`` is RECORDED for a parameter the caller leaves alone, and
#: ``provenance._jsonable`` ends in ``repr()``, so an unquoted 2026-01-01
#: reaches a methods paragraph as "datetime.date(2026, 1, 1)".
_JSONABLE_DEFAULTS = (str, int, float, bool)


def _check_default(path: Path, field: str, value) -> None:
    """A recorded value must survive JSON and h5 as itself.

    The same lesson as ``_require_str``, one level down: that one guards the
    fields this module turns into a dataclass, and nothing guarded the values
    inside ``params_schema``, which are handed straight to the provenance
    trail. An unquoted TOML date is not a mistake TOML rejects.

    Containers are walked rather than refused: ``_jsonable`` handles lists and
    dicts, and an ``enum``-typed parameter with a list default is ordinary.
    """
    if isinstance(value, bool) or isinstance(value, _JSONABLE_DEFAULTS):
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _check_default(path, f"{field}[{i}]", item)
        return
    if isinstance(value, dict):
        for k, item in value.items():
            _check_default(path, f"{field}.{k}", item)
        return
    got = type(value).__name__
    hint = (" An unquoted TOML value like 2026-01-01 is a date, not text -- "
            "put it in quotes." if got in ("date", "datetime", "time") else "")
    raise ManifestError(
        f"{path}: {field} is a {got} ({value!r}), which cannot be written "
        f"into a citation trail as itself. A default may be a string, a "
        f"number, a boolean, or a list or table of those -- it is recorded "
        f"for a parameter the caller leaves alone, and it ends up in the "
        f"methods paragraph and in every exported .h5.{hint}")


def _check_schema_defaults(path: Path, field: str,
                           schema: Dict[str, Any]) -> None:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    for name, declared in properties.items():
        if isinstance(declared, dict) and "default" in declared:
            _check_default(path, f"{field}.properties.{name}.default",
                           declared["default"])


def _named_slots(template: str) -> Tuple[str, ...]:
    """The named slots of a str.format template, or raise on a positional one."""
    names = []
    for _literal, field, _spec, _conv in string.Formatter().parse(template):
        if field is None:
            continue
        if field == "" or field.isdigit():
            raise ManifestError(
                f"sentence uses a positional slot ({{{field}}}); slots must be "
                "named, because parameters arrive as a dict"
            )
        names.append(field.split(".")[0].split("[")[0])
    return tuple(names)


def _require_str(path: Path, field: str, value) -> str:
    """A field the dataclass annotates ``str`` must actually BE a str.

    Nothing enforced these annotations, and TOML is not a string format: it
    has integers, floats, booleans, arrays, tables, and -- the one that
    measurably did damage -- dates. ``version = 2026-01-01`` is not a
    mistake TOML rejects; ``tomllib`` returns a ``datetime.date``, which
    travelled untouched through this parser into the trust store, where
    ``json.dump`` raised ``TypeError`` AFTER truncating the file. One
    badly-written add-on erased every consent decision on the machine.

    So this is ``outputs.py``'s lesson one module along: a declared type that
    nothing enforces is a comment. The check belongs here, at the edge where
    foreign data enters, and not at each of the places that later assume it
    -- ``key.startswith``, ``name.count(":")`` and ``Formatter().parse``
    would each have raised a bare ``AttributeError``/``TypeError`` that
    ``discovery._read`` does not catch, taking the whole listing down rather
    than rejecting one manifest.
    """
    if not isinstance(value, str):
        got = type(value).__name__
        hint = ""
        if got in ("date", "datetime", "time", "int", "float", "bool"):
            hint = (" An unquoted TOML value like 2026-01-01 or 1.2 is a "
                    "date or a number, not text -- put it in quotes.")
        raise ManifestError(
            f"{path}: {field} must be a string, not {got} ({value!r}).{hint}")
    return value


def _require_nonempty_str(path: Path, field: str, value) -> str:
    """A string field that must also SAY something.

    For ``version`` this is a trust rule, not tidiness. The page asks for
    consent again when the add-on under a known name has changed, and it
    decides that by comparing the recorded version and DOI with the ones in
    front of it. An EMPTY recorded version is read as "nothing to compare" —
    it has to be, because trust files written before those fields existed
    carry none — so an add-on declaring ``version = ""`` would switch its own
    staleness check off permanently: delete the folder, drop different code in
    under the same name, and it is enabled with no dialog. ``doi`` is empty for
    most add-ons, including the shipped example, so the version is the only
    discriminator left.

    A guard whose input is controlled by the party it guards against is not a
    guard. The empty declaration is refused here instead.
    """
    text = _require_str(path, field, value)
    if not text.strip():
        raise ManifestError(
            f"{path}: {field} must not be empty. It is what tells a user that "
            "the add-on under this name has changed since they allowed it to "
            "run.")
    return text


def _require_str_list(path: Path, field: str, value) -> Tuple[str, ...]:
    """A list of strings, and NOT a bare string.

    ``tuple("Jane")`` is ``('J', 'a', 'n', 'e')``, so ``authors = "Jane"``
    used to produce four authors, one letter each, in the dialog that asks a
    user to trust a person.
    """
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ManifestError(
            f"{path}: {field} must be a list of strings, not "
            f"{type(value).__name__} ({value!r})")
    return tuple(_require_str(path, f"{field}[{i}]", v)
                 for i, v in enumerate(value))


def _require_table(path: Path, field: str, value) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(
            f"{path}: {field} must be a table, not "
            f"{type(value).__name__} ({value!r})")
    return dict(value)


def _check_segment(path: Path, field: str, value: str) -> str:
    """A field that ends up in a URL must be able to be one.

    ``name`` was validated as a string and nothing else, and an analysis
    ``key`` only for its ``addon.`` prefix -- which separates an add-on's keys
    from the CORE's, not from another add-on's and not from a router. Both are
    interpolated into paths with no quoting, so ``name = "grain/stats"`` or
    ``key = "addon.a?b"`` produces routes that answer 404 while the listing
    reports the add-on as installed and compatible: a failure with nothing in
    it that names a cause.

    Refused at the edge, with the offending character named, because the
    author reading it has no debugger into Orienta.
    """
    if not _SEGMENT_CHARS.fullmatch(value):
        bad = next((c for c in value
                    if not _SEGMENT_CHARS.fullmatch(c)), None)
        problem = (f"contains {bad!r}" if bad is not None else "is empty")
        raise ManifestError(
            f"{path}: {field} is {value!r}, which {problem}. It is used as a "
            "URL path segment, so it may contain letters, digits, '.', '_' "
            "and '-' only.")
    if set(value) == {"."}:
        # '.' and '..' pass the class above and are then collapsed by every
        # URL resolver: the add-on would be listed, compatible, and reachable
        # at no address at all.
        raise ManifestError(
            f"{path}: {field} is {value!r}, which is nothing but dots. A URL "
            "resolver removes it or reads it as the path above, so the add-on "
            "would have no address. Give it a name.")
    return value


def _check_reference(path: Path, reference: str) -> None:
    if not reference or set(reference) & _BAD_IN_REFERENCE:
        raise ManifestError(
            f"{path}: citation reference {reference!r} cannot be a BibTeX "
            "entry key; it must be non-empty and free of commas, braces, "
            "'=' and whitespace"
        )


def parse_manifest(path: Path) -> AddonManifest:
    """Read and validate one add-on manifest. Never imports the add-on."""
    path = Path(path)
    try:
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ManifestError(f"no manifest at {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ManifestError(f"{path} is not valid TOML: {exc}") from exc
    except OSError as exc:
        raise ManifestError(f"could not read {path}: {exc}") from exc

    # BEFORE the required-field loop, deliberately: an author who compiled
    # against a different API must read that, not a complaint about a field
    # whose meaning may have changed between the two versions.
    if "api_version" not in raw:
        raise ManifestError(
            f"{path}: missing required field 'api_version'; this Orienta "
            f"speaks {API_VERSION}"
        )
    declared = raw["api_version"]
    if (not isinstance(declared, int) or isinstance(declared, bool)
            or declared != API_VERSION):
        raise ManifestError(
            f"{path}: api_version is {declared!r}, this Orienta speaks "
            f"{API_VERSION}. The add-on was not loaded."
        )

    for field in _REQUIRED_TOP:
        if field not in raw:
            raise ManifestError(f"{path}: missing required field {field!r}")

    declared_analyses = raw.get("analyses") or []
    if isinstance(declared_analyses, dict) or not isinstance(
            declared_analyses, (list, tuple)):
        raise ManifestError(
            f"{path}: analyses must be a list of tables ([[analyses]]), not "
            f"{type(declared_analyses).__name__}")

    analyses = []
    declared_keys: Dict[str, int] = {}
    for index, entry in enumerate(declared_analyses):
        entry = _require_table(path, f"analyses[{index}]", entry)
        for field in _REQUIRED_ANALYSIS:
            if field not in entry:
                raise ManifestError(
                    f"{path}: analysis is missing required field {field!r}")
        key = _check_segment(
            path, f"analyses[{index}].key",
            _require_str(path, f"analyses[{index}].key", entry["key"]))
        if not key.startswith(_KEY_PREFIX):
            raise ManifestError(
                f"{path}: analysis key {key!r} must start with {_KEY_PREFIX!r} "
                "so it cannot collide with a core step"
            )
        if key in declared_keys:
            # The second one is UNREACHABLE: ``run_analysis`` resolves a key
            # with ``next(a for a in manifest.analyses if a.key == key)``, so
            # it always runs the first, and the map store is keyed by the
            # analysis key too -- the second analysis's outputs would be
            # served under the first's label. A listed analysis that can never
            # run is worse than a refused manifest: nothing anywhere says why
            # it does nothing.
            raise ManifestError(
                f"{path}: analyses[{index}] declares the key {key!r}, which "
                f"analyses[{declared_keys[key]}] already declares. One key "
                "names one analysis -- the second could never be run, and its "
                "outputs would be served under the first one's name.")
        declared_keys[key] = index
        name = _require_str(path, f"analyses[{index}].python_name",
                            entry["python_name"])
        if name.count(":") != 1 or not all(name.split(":")):
            raise ManifestError(
                f"{path}: python_name {name!r} must be 'module:attribute'")
        sentence = _require_str(path, f"analyses[{index}].sentence",
                                entry["sentence"])
        _named_slots(sentence)
        citations = _require_str_list(path, f"analyses[{index}].citations",
                                      entry.get("citations") or [])
        for reference in citations:
            _check_reference(path, reference)
        params_schema = _require_table(
            path, f"analyses[{index}].params_schema",
            entry.get("params_schema") or {})
        _check_schema_defaults(path, f"analyses[{index}].params_schema",
                               params_schema)
        analyses.append(AnalysisSpec(
            key=key,
            label=_require_str(path, f"analyses[{index}].label",
                               entry["label"]),
            python_name=name,
            sentence=sentence,
            citations=citations,
            params_schema=params_schema,
        ))

    if not analyses:
        raise ManifestError(f"{path}: declares no analyses")

    doi = _require_str(path, "doi", raw.get("doi") or "")
    if doi:
        _check_reference(path, f"doi:{doi}")

    return AddonManifest(
        api_version=declared,
        name=_check_segment(path, "name",
                            _require_str(path, "name", raw["name"])),
        display_name=_require_str(path, "display_name", raw["display_name"]),
        version=_require_nonempty_str(path, "version", raw["version"]),
        requires_orienta=_require_str(path, "requires_orienta",
                                      raw.get("requires_orienta") or ""),
        authors=_require_str_list(path, "authors", raw["authors"]),
        doi=doi,
        analyses=tuple(analyses),
        source_path=path,
    )
