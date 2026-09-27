"""Analysis presets for the EDS phase map: a recipe you can mail, diff and hash.

WHY THIS EXISTS. The recipe that produces a phase map — the smoothing width,
the region count, the element weights, and above all the windows and rules the
user wrote by hand — lived in one browser tab and in one sidecar next to one
scan. Carrying it to the next sample meant retyping it, and nobody could say
afterwards whether two maps had been made the same way.

WHY ONE JSON FILE PER PRESET, and not a database. A user asked for exactly
this and refused the opaque alternative: a preset has to be readable, diffable,
mailable and git-committable. A row in a binary store is none of those, and the
first question anyone asks about a classified map — "what did you do?" — is
answered by a file you can open, not by a name that points into a store on
somebody else's machine.

WHAT IS NEVER IN A PRESET. File paths, grid shape, crop window, hand-painted
pixels, region-to-phase assignments by id, measured background levels. Each is
bound to one scan, and carrying it into another one is the silent-wrong-answer
failure mode. ``keep_manual_edits`` is dropped for the same reason: it is a
statement about the hand edits of the map currently on screen, not about a
recipe. ``n_clusters_pinned`` is dropped as a duplicate rather than as scan
state — it re-states what ``n_clusters`` already says (``None`` = not pinned,
a number = pinned) and a second copy of a fact is the copy that goes stale.
What IS in a preset is everything that changes a number.

WHAT IS STORED BUT NOT HASHED: ``tolerance``. It is inert in this build — no
code path reads it — so it cannot tell two analyses apart, and hashing it
would make two identical recipes hash differently after a hand-edit. It still
round-trips losslessly, against the day the slider is reconnected. See
``_HASH_EXCLUDED_SETTING_KEYS`` and spec section 9.

THE COMPATIBILITY CHECK IS THE FEATURE. A preset that applies silently to a
dataset it does not fit is worse than no preset, because it produces a
plausible map from a recipe that cannot mean here what it meant there. Three
conditions refuse outright (an element the scan never measured, a pinned matrix
element that is not this scan's dominant element, a phase the library does not
have); the rest warn. The refusals can be overridden — that is the user's right
— but :meth:`CompatibilityReport.as_dict` is written so the override is
recordable in the export, because a warning that only appears on screen is
clicked away in half a second.

TWO OF THE WARNINGS ARE ABOUT THE CHECK ITSELF, and they exist because silence
is ambiguous. ``phase_list_not_pinned`` says the preset never named its
candidate phases, so it re-reads the library on every run: add one CIF and the
same preset, at the same content hash, quietly runs a different analysis — and
note the asymmetry, because the existing guard stops a stale UI NARROWING a run
while widening was unguarded, and widening is the direction that moves numbers.
``portability_not_checkable`` says a guard could not run at all — the preset
carries no ``authored_elements`` or no ``authored_step_um``, or this scan has no
step size — because a check that silently did not run reads exactly like a check
that ran and found nothing.

THE SUBTLE ONE IS ``element_set_differs``, and it fires even on a SUPERSET.
at% is renormalised over the measured elements, so measuring one extra element
shifts every other element's at% and therefore changes what every absolute at%
window means. A window that kept a particle on the authoring scan can keep half
of it here without a single number in the preset having changed.

RELATED: ``absolute_window_not_portable``. Measured in this repo on one scan,
"Si >= 2x background" kept 96.3 % of a real particle while "Si >= 40 at%" kept
46.3 %. Enrichment carries its bar per dataset and travels between samples; an
absolute window largely does not.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from backend.api.services.phase_rules import (
    RegionDefinition,
    RuleSet,
    region_defs_from_list,
    region_defs_to_list,
    rule_set_from_dict,
    rule_set_to_dict,
)
from backend.api.services.user_config_manager import config_path

logger = logging.getLogger(__name__)

#: Bumped only when the on-disk shape changes incompatibly. A file carrying any
#: other number is refused rather than guessed at: a preset read wrong is a
#: wrong analysis, and there is no half-understood preset worth running.
PRESET_SCHEMA = 1

# --- what a preset carries --------------------------------------------------
#
# The keys of ``AutoClassifyRequest`` that change a number, plus ``scale_um``
# (smoothing as a physical length, so one preset means one physical width
# across two step sizes).
#
# Every known key is written out explicitly, filled from these defaults when
# absent. Two reasons, both load-bearing:
#   * the content hash then names the RECIPE, not the spelling — a setting
#     omitted and a setting written at its default are the same analysis and
#     must hash the same;
#   * an old preset cannot be silently moved by a future change of a default.
_SETTING_DEFAULTS: dict[str, Any] = {
    "mode": "cluster",
    "tolerance": 15.0,
    "min_score": 0.3,
    "n_clusters": None,      # None = chosen automatically; a number = pinned
    "scale": None,           # smoothing box in pixels (legacy)
    "scale_um": None,        # smoothing box as a length; wins when both given
    "element_weights": None,
    "rules": None,
    "region_defs": None,
    "phase_keys": None,      # explicit, never "all"
    "cluster_remainder": True,
}

#: Keys the UI sends that are NOT part of a recipe, dropped on the way in.
#:
#: ``keep_manual_edits`` is bound to one scan's hand edits: it is a statement
#: about the map currently on screen, not about a recipe.
#:
#: ``n_clusters_pinned`` is a SECOND COPY of a fact ``n_clusters`` already
#: carries, and the second copy is the one that can go stale. The writer
#: derives it (``PresetBar.jsx``: ``n_clusters_pinned: handle.nClusters !=
#: null``) and the reader collapses back to exactly that when the key is
#: absent (``pinned = s.n_clusters_pinned !== false && s.n_clusters != null``,
#: and ``undefined !== false``), so dropping it round-trips the app's own
#: semantics unchanged: ``n_clusters = null`` IS "not pinned" and a number IS
#: "pinned". No backend code reads it — ``AutoClassifyRequest`` is
#: ``extra="forbid"`` and has no such field, so it can never reach the
#: classifier.
#:
#: KEEPING IT WOULD HAVE COST TWO THINGS. It cannot go in
#: :data:`_SETTING_DEFAULTS` with a static default: at ``False`` a hand-written
#: ``{"n_clusters": 8}`` and the app's ``{"n_clusters": 8,
#: "n_clusters_pinned": true}`` are the same recipe and would hash
#: differently — the hash would answer "was this typed the same way", which
#: nobody asked; at ``True`` the all-default save still differs from the
#: default and the emptiness guard below stays dead, which is the whole bug.
#: And leaving it unknown kept that guard dead outright: every save from the
#: dialog carries this key, so :func:`_settings_are_empty` short-circuited on
#: it before judging a single setting, and the refusal this module advertises
#: could only ever fire for a bare ``{}`` posted by a script.
_EXCLUDED_SETTING_KEYS = frozenset({"keep_manual_edits", "n_clusters_pinned"})

#: Stored in the file, but NOT hashed. Spec section 9.
#:
#: ``tolerance`` is inert in this build: ``auto_classify_pixels`` takes it and
#: never reads it in its body (it was superseded by the vetoed L1 scorer and
#: kept for API compatibility), and ``eds_clustering`` never mentions it. A
#: knob that cannot change a result cannot distinguish two recipes, so hashing
#: it would make two genuinely identical analyses hash differently after a
#: hand-edit of the JSON — and the hash exists to answer "is this the same
#: analysis?".
#:
#: It is still written to the file and still round-trips unchanged, because a
#: future build may reconnect the slider; on that day this set shrinks and the
#: hashes move once, deliberately.
_HASH_EXCLUDED_SETTING_KEYS = frozenset({"tolerance"})

_FLOAT_KEYS = frozenset({"tolerance", "min_score", "scale_um"})
_INT_KEYS = frozenset({"n_clusters", "scale"})

try:  # pragma: no cover - trivial fallback
    from eds_utils import parse_element_name as _parse_element_name
except Exception:  # repo-root module may be absent in slimmed deployments
    def _parse_element_name(raw: str) -> str:
        parts = str(raw).split()
        return parts[0].strip() if parts else ""


def _symbol(raw: str) -> str:
    """Element SYMBOL, not the Aztec line label.

    The backend works in symbols throughout (``parse_element_name`` strips the
    line on load), and a clause stored as ``Al Ka1`` would never match
    anything. Measured and authored elements have to be compared on the same
    footing or every check in this module is decorative — that exact mismatch
    already shipped once in the rule editor.

    Empty input is answered with empty rather than passed on: the repo-root
    ``parse_element_name`` indexes into ``split()`` unguarded, and "no matrix
    element declared" is a perfectly normal state here.
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    return _parse_element_name(text)


# --- storage location -------------------------------------------------------

_PRESET_DIR_OVERRIDE: Optional[Path] = None      # tests inject this
_BUILTIN_DIR_OVERRIDE: Optional[Path] = None     # tests inject this


def preset_dir() -> Path:
    """Where the user's own presets live.

    A subdirectory of the same per-machine config directory the rest of the app
    uses (``%APPDATA%/Kikuchipy`` on Windows), so presets stay behind when the
    project tree is copied to another PC — exactly like API keys and server
    config. Derived from :func:`user_config_manager.config_path` rather than
    re-deriving the OS convention, so there is one definition of "the user's
    Kikuchipy directory".
    """
    if _PRESET_DIR_OVERRIDE is not None:
        return _PRESET_DIR_OVERRIDE
    return config_path().parent / "eds_presets"


def builtin_preset_dir() -> Path:
    """Where the shipped, read-only starter presets live.

    Beside the app (next to this module), never in the user directory: they
    travel with the code, are replaced by an update, and are never mutated in
    place. The directory may legitimately not exist yet.
    """
    if _BUILTIN_DIR_OVERRIDE is not None:
        return _BUILTIN_DIR_OVERRIDE
    return Path(__file__).resolve().parent / "eds_presets_builtin"


def set_preset_dir_for_test(p: Optional[Path]) -> None:
    """Test-only: redirect the user preset directory. ``None`` restores."""
    global _PRESET_DIR_OVERRIDE
    _PRESET_DIR_OVERRIDE = Path(p) if p is not None else None


def set_builtin_dir_for_test(p: Optional[Path]) -> None:
    """Test-only: redirect the shipped preset directory. ``None`` restores."""
    global _BUILTIN_DIR_OVERRIDE
    _BUILTIN_DIR_OVERRIDE = Path(p) if p is not None else None


# --- settings normalisation -------------------------------------------------

def _coerce_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _coerce_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _clean_weights(v: Any) -> Optional[dict[str, float]]:
    if not isinstance(v, dict) or not v:
        return None
    out: dict[str, float] = {}
    for k, val in v.items():
        sym = _symbol(k)
        f = _coerce_float(val)
        if sym and f is not None:
            out[sym] = f
    return out or None


def _clean_keys(v: Any) -> Optional[list[str]]:
    if not isinstance(v, (list, tuple)):
        return None
    keys = [str(k) for k in v if str(k)]
    return keys or None


def normalise_settings(settings: Optional[dict],
                       matrix_element: str = "") -> dict:
    """Canonical wire form of a settings payload.

    Rules and region definitions go out through the very parsers that will read
    them back at classification time (``rule_set_from_dict`` /
    ``region_defs_from_list``), so what is stored is what will be evaluated — a
    preset cannot contain a clause the evaluator would silently drop.

    ``matrix_element`` is a projection of ``rules.matrix_elements``: the two are
    kept in sync here, in both directions, so pinning the matrix survives into
    the payload that is actually run AND is covered by the content hash, which
    is taken over the settings alone.
    """
    raw = dict(settings or {})
    for k in _EXCLUDED_SETTING_KEYS:
        raw.pop(k, None)

    out: dict[str, Any] = {}

    # Unknown keys are kept verbatim: a preset written by a newer build should
    # survive a round trip through an older one rather than be quietly thinned.
    for k, v in raw.items():
        if k not in _SETTING_DEFAULTS:
            out[k] = v

    for key, default in _SETTING_DEFAULTS.items():
        v = raw.get(key, default)
        if key in _FLOAT_KEYS:
            out[key] = default if v is None else _coerce_float(v)
        elif key in _INT_KEYS:
            out[key] = None if v is None else _coerce_int(v)
        elif key == "mode":
            out[key] = str(v or default)
        elif key == "cluster_remainder":
            out[key] = bool(default if v is None else v)
        elif key == "element_weights":
            out[key] = _clean_weights(v)
        elif key == "phase_keys":
            out[key] = _clean_keys(v)
        elif key == "region_defs":
            defs = region_defs_from_list(v) if v else []
            # A definition with no clause that binds claims nothing; storing it
            # would put a half-written window into a shared recipe.
            defs = [d for d in defs if not d.is_empty]
            out[key] = region_defs_to_list(defs) or None
        else:  # "rules" — resolved below, together with the matrix element
            out[key] = v

    rs = rule_set_from_dict(out.get("rules")) if out.get("rules") else None
    pinned = _symbol(matrix_element) if matrix_element else ""
    if not pinned and rs is not None and rs.matrix_elements:
        pinned = _symbol(rs.matrix_elements[0])
    if pinned:
        rs = RuleSet(
            name=rs.name if rs else "",
            phase_keys=rs.phase_keys if rs else None,
            matrix_elements=(pinned,),
            rules=rs.rules if rs else (),
        )
    out["rules"] = (rule_set_to_dict(rs)
                    if (rs is not None and not rs.is_empty) else None)
    return out


def matrix_element_of(settings: Optional[dict]) -> str:
    """The pinned matrix element, read back out of normalised settings."""
    rules = (settings or {}).get("rules")
    rs = rule_set_from_dict(rules) if rules else None
    if rs is not None and rs.matrix_elements:
        return _symbol(rs.matrix_elements[0])
    return ""


def phase_list_is_pinned(settings: Optional[dict]) -> bool:
    """Does this recipe name the phases it may choose from?

    Read from the two places that actually narrow the candidate library at
    classification time, and only those:

    * ``settings["phase_keys"]`` — ``routes/eds.py`` keeps a strict subset and
      treats an empty or absent list as "use everything" (line ~1036);
    * ``rules.phase_keys`` — the RuleSet-level participation filter applied a
      few lines further down.

    Why it is worth a function of its own: unpinned is not a neutral state. An
    unpinned preset re-reads the library on every run, so adding one CIF in
    week 30 makes the same preset, at the same content hash, run a different
    analysis — and nothing on screen or in the file says so. The one guard the
    codebase already had works the other way round (a stale UI may not silently
    NARROW a run); widening was unguarded, and widening is the direction that
    moves the numbers.
    """
    s = settings or {}
    if s.get("phase_keys"):
        return True
    rules = s.get("rules")
    rs = rule_set_from_dict(rules) if rules else None
    # ``is not None`` and not truthiness: an explicitly empty list is still a
    # decision the author wrote down, not an absent one.
    return bool(rs is not None and rs.phase_keys is not None)


def _settings_are_empty(settings: dict) -> bool:
    """True when the recipe constrains nothing at all.

    Compared against the defaults rather than against ``{}``, because
    normalisation writes every key: a preset that says only "mode=cluster,
    tolerance=15" is the default analysis wearing a name, and saving it would
    mean "whatever the defaults happen to be" on every future dataset.

    THE UNKNOWN-KEY SHORT-CIRCUIT IS LOAD-BEARING AND WAS ONCE FATAL. An
    unknown key may be real content from a newer build, and content this
    function cannot judge must not be judged as absent — so it answers "not
    empty". That is only safe while every key the SAVE DIALOG itself sends is
    either a known setting or explicitly excluded: ``n_clusters_pinned`` was
    neither, so every save from the app returned here on the first key and
    this guard could not fire for the one path it exists to protect. The
    invariant is pinned by ``test_every_key_the_dialog_sends_is_known_or_
    excluded``; a new UI-only field must join :data:`_EXCLUDED_SETTING_KEYS`.
    """
    for k, v in settings.items():
        if k not in _SETTING_DEFAULTS:
            return False           # an unknown key is content we cannot judge
        if v != _SETTING_DEFAULTS[k]:
            return False
    return True


# --- the preset -------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class EdsPreset:
    """One analysis recipe, plus who wrote it and what it was written against."""

    name: str
    settings: dict
    version: int = 1
    author: str = ""
    created: str = ""
    notes: str = ""
    #: Free text on purpose. A service lab will keep ~100 presets and will want
    #: to group them by whatever it actually works on ("6xxx extrusion",
    #: "customer 4711"); a closed vocabulary would be wrong within a week.
    material_class: str = ""
    tags: list[str] = field(default_factory=list)
    #: The element list the preset was authored against. Not a constraint — a
    #: record, and the input to the most valuable warning in the set.
    authored_elements: list[str] = field(default_factory=list)
    authored_step_um: Optional[float] = None
    #: PINNED, never "infer". The manual states that getting the matrix wrong
    #: inverts the whole measure, so a preset that re-infers it per scan can
    #: quietly become a different analysis on the next file.
    matrix_element: str = ""
    #: Shipped with the app, read-only. Set from WHERE the file was found, never
    #: from what the file claims — otherwise a mailed JSON could declare itself
    #: built-in and become unwritable.
    builtin: bool = False
    schema: int = PRESET_SCHEMA

    @property
    def content_hash(self) -> str:
        """sha256 over the EFFECTIVE settings, first 16 hex characters.

        Renaming a preset, fixing a typo in its notes or adding a tag must not
        change the hash: it answers "is this the same analysis?", and that
        answer has to survive housekeeping. Conversely any change to a number,
        a rule or a window changes it.

        "Effective" is the whole subtlety: keys in
        :data:`_HASH_EXCLUDED_SETTING_KEYS` are stored but not hashed, because
        they cannot change a result and therefore cannot tell two analyses
        apart.
        """
        hashed = {k: v for k, v in self.settings.items()
                  if k not in _HASH_EXCLUDED_SETTING_KEYS}
        payload = json.dumps(hashed, sort_keys=True,
                             separators=(",", ":"), ensure_ascii=True,
                             default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def phase_list_pinned(self) -> bool:
        """Does this preset name its candidate phases? See
        :func:`phase_list_is_pinned`.

        Derived, never stored as an independent truth: a stored flag and the
        settings it describes are two copies of one fact, and the stored one is
        the copy that goes stale. It is WRITTEN into the file (``to_dict``) for
        the human reading the diff, and recomputed on every read.
        """
        return phase_list_is_pinned(self.settings)

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "schema": int(self.schema),
            "name": self.name,
            "version": int(self.version),
            "author": self.author,
            "created": self.created,
            "notes": self.notes,
            "material_class": self.material_class,
            "tags": list(self.tags),
            "authored_elements": list(self.authored_elements),
            "authored_step_um": self.authored_step_um,
            "matrix_element": self.matrix_element,
            "builtin": bool(self.builtin),
            # Derived — written for the human reading the file and for a diff.
            # Recomputed on read, never trusted.
            "content_hash": self.content_hash,
            "phase_list_pinned": self.phase_list_pinned,
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, payload: Any, *, builtin: bool = False) -> "EdsPreset":
        """Parse a preset. Raises ``ValueError`` on anything unusable."""
        if not isinstance(payload, dict):
            raise ValueError("preset must be a JSON object")
        schema = payload.get("schema")
        if not isinstance(schema, int) or isinstance(schema, bool):
            raise ValueError(
                f"preset has no usable 'schema' field (got {schema!r}); "
                f"this build writes schema {PRESET_SCHEMA}"
            )
        if schema != PRESET_SCHEMA:
            raise ValueError(
                f"preset schema {schema} cannot be read by this build "
                f"(expected {PRESET_SCHEMA})"
            )
        name = str(payload.get("name") or "").strip()
        if not name:
            raise ValueError("preset has no name")
        settings = payload.get("settings")
        if not isinstance(settings, dict):
            raise ValueError(f"preset {name!r} has no settings object")

        matrix = _symbol(str(payload.get("matrix_element") or ""))
        norm = normalise_settings(settings, matrix)
        # Sync back: normalisation may have found the pin inside the rules.
        matrix = matrix or matrix_element_of(norm)

        preset = cls(
            name=name,
            settings=norm,
            version=max(1, _coerce_int(payload.get("version")) or 1),
            author=str(payload.get("author") or ""),
            created=str(payload.get("created") or ""),
            notes=str(payload.get("notes") or ""),
            material_class=str(payload.get("material_class") or ""),
            tags=[str(t) for t in (payload.get("tags") or []) if str(t)],
            authored_elements=[
                _symbol(str(e)) for e in (payload.get("authored_elements") or [])
                if _symbol(str(e))
            ],
            authored_step_um=_coerce_float(payload.get("authored_step_um")),
            matrix_element=matrix,
            builtin=bool(builtin),
            schema=PRESET_SCHEMA,
        )
        stated = payload.get("content_hash")
        if stated and stated != preset.content_hash:
            # Not fatal: normalisation can legitimately move the hash (an older
            # file, a dropped half-written clause). Worth saying out loud once.
            logger.info(
                "eds_presets: %r declared content hash %s but normalises to %s",
                name, stated, preset.content_hash,
            )
        return preset


#: Everything ``preset_from_settings`` accepts besides the settings themselves.
_META_KEYS = frozenset({
    "version", "author", "created", "notes", "material_class", "tags",
    "authored_elements", "authored_step_um", "matrix_element", "builtin",
})


def preset_from_settings(name: str, settings: dict, **meta: Any) -> EdsPreset:
    """Build a preset from a settings payload plus optional metadata.

    ``created`` defaults to now; ``matrix_element`` to whatever the settings
    already pin. An unknown keyword raises rather than being swallowed — the
    caller would otherwise get a preset that does not carry what they thought
    it carried, and would find out one dataset too late.
    """
    unknown = set(meta) - _META_KEYS
    if unknown:
        raise TypeError(f"unknown preset metadata: {', '.join(sorted(unknown))}")

    matrix = _symbol(str(meta.get("matrix_element") or ""))
    norm = normalise_settings(settings, matrix)
    matrix = matrix or matrix_element_of(norm)
    authored = [_symbol(str(e)) for e in (meta.get("authored_elements") or [])]

    return EdsPreset(
        name=str(name).strip(),
        settings=norm,
        version=max(1, _coerce_int(meta.get("version")) or 1),
        author=str(meta.get("author") or ""),
        created=str(meta.get("created") or "") or _now_iso(),
        notes=str(meta.get("notes") or ""),
        material_class=str(meta.get("material_class") or ""),
        tags=[str(t) for t in (meta.get("tags") or []) if str(t)],
        authored_elements=[e for e in authored if e],
        authored_step_um=_coerce_float(meta.get("authored_step_um")),
        matrix_element=matrix,
        builtin=bool(meta.get("builtin", False)),
    )


def export_preset_json(p: EdsPreset) -> str:
    """Pretty, stable-key-order JSON — the form that gets mailed and diffed."""
    return json.dumps(p.to_dict(), indent=2, sort_keys=True,
                      ensure_ascii=False) + "\n"


def import_preset_json(text: str) -> EdsPreset:
    """Parse a mailed/exported preset. Always lands as a USER preset."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON: {exc}") from exc
    return EdsPreset.from_dict(payload, builtin=False)


# --- files ------------------------------------------------------------------

# Windows reserves these stems whatever the extension.
_RESERVED_STEMS = frozenset({
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
})

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str) -> str:
    """A filesystem-safe stem for a name a user actually typed.

    Somebody WILL type ``AL/Fe 6xxx: draft``. The true name stays inside the
    JSON and is what every lookup matches on; this only decides where the bytes
    go.
    """
    stem = _UNSAFE.sub("_", str(name).strip())
    stem = re.sub(r"_+", "_", stem).strip("._-")
    if stem.lower() in _RESERVED_STEMS:
        stem = f"{stem}_preset"
    if not stem:
        stem = "preset"
    return stem[:80]


def _read_file(path: Path, *, builtin: bool) -> EdsPreset:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return EdsPreset.from_dict(payload, builtin=builtin)


def _scan_dir(d: Path, *, builtin: bool) -> list[tuple[Path, EdsPreset]]:
    """Every readable preset in a directory. Junk is skipped, never fatal."""
    out: list[tuple[Path, EdsPreset]] = []
    try:
        entries = sorted(d.glob("*.json"))
    except OSError as exc:
        logger.warning("eds_presets: cannot list %s: %s", d, exc)
        return out
    for f in entries:
        try:
            out.append((f, _read_file(f, builtin=builtin)))
        except Exception as exc:
            # One bad file must not take the whole list with it — the user
            # would lose access to every preset because of one bad paste.
            logger.warning("eds_presets: skipping %s: %s", f, exc)
    return out


def list_presets() -> list[EdsPreset]:
    """All presets: shipped ones first, then the user's, each sorted by name.

    A user preset SHADOWS a shipped one of the same name — that is what a fork
    is — so the shipped one is left out rather than listed twice under a name
    that could then mean either.
    """
    user = _scan_dir(preset_dir(), builtin=False)
    user_names = {p.name.casefold() for _f, p in user}
    builtins = [
        (f, p) for f, p in _scan_dir(builtin_preset_dir(), builtin=True)
        if p.name.casefold() not in user_names
    ]
    return (
        [p for _f, p in sorted(builtins, key=lambda t: t[1].name.lower())]
        + [p for _f, p in sorted(user, key=lambda t: t[1].name.lower())]
    )


def _find(name: str) -> tuple[Optional[tuple[Path, EdsPreset]],
                              Optional[tuple[Path, EdsPreset]]]:
    """``(user_hit, builtin_hit)`` for a name, matched on the name in the JSON.

    Matched on the stored name and not on the filename, because the filename is
    a sanitised approximation of it and two different names can share one.
    """
    key = str(name).strip().casefold()
    user = next((t for t in _scan_dir(preset_dir(), builtin=False)
                 if t[1].name.casefold() == key), None)
    shipped = next((t for t in _scan_dir(builtin_preset_dir(), builtin=True)
                    if t[1].name.casefold() == key), None)
    return user, shipped


def load_preset(name: str) -> EdsPreset:
    """Load by name. A user fork wins over a shipped preset of the same name."""
    user, shipped = _find(name)
    hit = user or shipped
    if hit is None:
        raise KeyError(f"no EDS preset named {name!r}")
    return hit[1]


def _target_path(name: str) -> Path:
    """Where a user preset of this name goes.

    Two different names can sanitise to the same stem (``a/b`` and ``a:b``), so
    a stem already taken by a DIFFERENT name gets a short hash suffix rather
    than silently overwriting somebody else's recipe.
    """
    d = preset_dir()
    stem = safe_filename(name)
    path = d / f"{stem}.json"
    if path.exists():
        try:
            existing = _read_file(path, builtin=False)
            same = existing.name.casefold() == str(name).strip().casefold()
        except Exception:
            # Unreadable file sitting on the stem we want: step aside rather
            # than clobber something we could not even parse.
            same = False
        if not same:
            suffix = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
            return d / f"{stem}_{suffix}.json"
    return path


def _write_atomic(path: Path, text: str) -> None:
    """Temp file in the same directory, then replace — same as the config
    manager. A half-written preset would be indistinguishable from a corrupt
    one, and it would be the one the user just made.

    ``newline="\n"`` IS THE POINT, not tidiness. Without it Python's text mode
    translates every ``\n`` to the platform separator, so on Windows the file
    on disk came out CRLF while ``GET /presets/{name}/export`` — the bytes that
    get mailed to a colleague — returned the same JSON with LF. A tester who
    diffed a mailed copy against her saved one measured 81 of 81 lines
    differing with not one character of content changed, which is exactly the
    diff that makes somebody believe the preset travelled wrong. The shipped
    built-in presets are LF too, so one convention now covers all three.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".preset_", suffix=".json.tmp",
                                    dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def save_preset(p: EdsPreset, *, overwrite: bool = False) -> Path:
    """Write a preset to the USER directory. Never touches a shipped file.

    Saving over an existing name bumps ``version``: an edited shared preset
    should be recognisable as a later take of the same recipe rather than as
    the same thing with different numbers. Deliberately just an integer — no
    version archive, because a silent pile of old copies is its own problem.
    """
    if not str(p.name).strip():
        raise ValueError("a preset needs a name")
    # Re-normalise on the way out: a caller may have edited `settings` directly.
    p.settings = normalise_settings(p.settings, p.matrix_element)
    p.matrix_element = p.matrix_element or matrix_element_of(p.settings)
    if _settings_are_empty(p.settings):
        raise ValueError(
            f"preset {p.name!r} constrains nothing — it would mean "
            f"'whatever the defaults are' on every future dataset"
        )

    user, shipped = _find(p.name)
    if (user or shipped) and not overwrite:
        what = "a built-in preset" if (shipped and not user) else "a preset"
        raise FileExistsError(
            f"{what} named {p.name!r} already exists; "
            f"pass overwrite=True to save a new version"
        )

    previous = user[1] if user else (shipped[1] if shipped else None)
    if previous is not None:
        p.version = max(int(previous.version), int(p.version)) + 1
    # A fork of a shipped preset is a user preset; only the location decides.
    p.builtin = False
    if not p.created:
        p.created = _now_iso()

    path = user[0] if user else _target_path(p.name)
    # Record whether the candidate list was pinned, at the moment it was saved.
    # NOT pinned here on the user's behalf: pinning would save a recipe the
    # user did not write, and "the phases my library happened to hold on the
    # day I pressed Save" is a different analysis from "all of them". The
    # offer to pin belongs at the call site (the route returns it as a hint);
    # the refusal to do it silently belongs here.
    if not p.phase_list_pinned:
        logger.info(
            "eds_presets: %r saved WITHOUT a pinned phase list — it will run "
            "against whatever the CIF library holds at the time, so a library "
            "change moves its results without moving its content hash (%s)",
            p.name, p.content_hash,
        )
    _write_atomic(path, export_preset_json(p))
    if shipped and not user:
        logger.info("eds_presets: %r forked from the built-in copy into %s",
                    p.name, path)
    return path


def delete_preset(name: str) -> bool:
    """Delete a user preset. ``True`` if something was deleted.

    A shipped preset cannot be deleted — raising beats returning ``False``,
    which a caller would read as "there was nothing there" and show nothing.
    Deleting a fork re-exposes the shipped original, which is the point.
    """
    user, shipped = _find(name)
    if user is None:
        if shipped is not None:
            raise PermissionError(
                f"{name!r} is a built-in preset and cannot be deleted"
            )
        return False
    try:
        user[0].unlink()
    except OSError as exc:
        logger.warning("eds_presets: could not delete %s: %s", user[0], exc)
        return False
    return True


# --- compatibility ----------------------------------------------------------

@dataclass
class CompatibilityReport:
    """What stands between this preset and this scan.

    ``as_dict`` is the form that goes into the export when the user overrides a
    refusal, so the file records what was overridden and against which preset.
    """

    ok: bool
    blockers: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    preset_name: str = ""
    preset_hash: str = ""

    def as_dict(self) -> dict:
        return {
            "ok": bool(self.ok),
            "preset_name": self.preset_name,
            "preset_hash": self.preset_hash,
            "blockers": [dict(b) for b in self.blockers],
            "warnings": [dict(w) for w in self.warnings],
        }


def _bounds(lo: Optional[float], hi: Optional[float], unit: str) -> str:
    if lo is not None and hi is not None:
        return f"{lo:g}-{hi:g}{unit}"
    if lo is not None:
        return f">= {lo:g}{unit}"
    if hi is not None:
        return f"<= {hi:g}{unit}"
    return "any"


def _clause_texts(holder: Any, where: str) -> list[tuple[str, str]]:
    """``(element, human-readable clause)`` for every clause of a rule/def.

    A ratio contributes BOTH legs: either leg missing makes the clause
    undecidable, and an undecidable clause blocks rather than passing.
    """
    out: list[tuple[str, str]] = []
    for e in getattr(holder, "elements", ()):
        out.append((_symbol(e.element),
                    f"{where}: {e.element} "
                    f"{_bounds(e.min_at_pct, e.max_at_pct, ' at%')}"))
    for r in getattr(holder, "ratios", ()):
        text = (f"{where}: {r.numerator}:{r.denominator} "
                f"{_bounds(r.min_ratio, r.max_ratio, '')}")
        out.append((_symbol(r.numerator), text))
        out.append((_symbol(r.denominator), text))
    for x in getattr(holder, "enrichment", ()):
        out.append((_symbol(x.element),
                    f"{where}: {x.element} "
                    f"{_bounds(x.min_factor, x.max_factor, 'x background')}"))
    return out


def _absolute_windows(holder: Any, where: str) -> list[str]:
    return [
        f"{where}: {e.element} {_bounds(e.min_at_pct, e.max_at_pct, ' at%')}"
        for e in getattr(holder, "elements", ())
        if e.min_at_pct is not None or e.max_at_pct is not None
    ]


def _rule_set_of(settings: dict) -> Optional[RuleSet]:
    payload = (settings or {}).get("rules")
    return rule_set_from_dict(payload) if payload else None


def _region_defs_of(settings: dict) -> list[RegionDefinition]:
    return region_defs_from_list((settings or {}).get("region_defs") or [])


def check_compatibility(
    p: EdsPreset,
    *,
    measured_elements: Iterable[str],
    available_phase_keys: Optional[Iterable[str]],
    dominant_element: str = "",
    step_um: Optional[float] = None,
) -> CompatibilityReport:
    """Can this recipe mean here what it meant where it was written?

    Blockers are refusals the caller may override explicitly; warnings proceed.
    Both are returned as ``{code, message, detail}`` so the UI can translate on
    ``code`` and never on the English prose — a preset written in the German UI
    has to behave identically in the English one, and nothing here is ever
    keyed on a translated string.
    """
    measured_set = {s for s in (_symbol(str(e)) for e in (measured_elements or []))
                    if s}

    blockers: list[dict] = []
    warnings: list[dict] = []

    rs = _rule_set_of(p.settings)
    defs = _region_defs_of(p.settings)
    rules = list(rs.rules) if rs else []

    # --- every element this preset names ------------------------------------
    refs: list[tuple[str, str]] = []
    for rule in rules:
        refs.extend(_clause_texts(
            rule, f"rule for {rule.phase_formula or rule.phase_key}"))
    for d in defs:
        refs.extend(_clause_texts(d, f"region '{d.name}'"))
    for el in (p.settings.get("element_weights") or {}):
        refs.append((_symbol(el), f"element weight on {el}"))
    if p.matrix_element:
        refs.append((p.matrix_element,
                     f"pinned matrix element {p.matrix_element}"))

    missing: dict[str, list[str]] = {}
    for element, text in refs:
        if not element or element in measured_set:
            continue
        clauses = missing.setdefault(element, [])
        if text not in clauses:
            clauses.append(text)

    for element in sorted(missing):
        clauses = missing[element]
        blockers.append({
            "code": "missing_element",
            "message": (
                f"{element} was not measured in this scan, but the preset uses "
                f"it in {len(clauses)} place(s): " + "; ".join(clauses)
                + ". A clause on an unmeasured element cannot be decided, so "
                  "it blocks rather than silently passing."
            ),
            "detail": {"element": element, "clauses": clauses},
        })

    # --- the matrix element -------------------------------------------------
    dominant = _symbol(str(dominant_element)) if dominant_element else ""
    if p.matrix_element and dominant and p.matrix_element != dominant:
        blockers.append({
            "code": "matrix_mismatch",
            "message": (
                f"The preset pins {p.matrix_element} as the matrix element, "
                f"but this scan's dominant element is {dominant}. Enrichment "
                f"is measured against the matrix, so getting it wrong inverts "
                f"the whole measure."
            ),
            "detail": {"preset_matrix": p.matrix_element,
                       "scan_dominant": dominant},
        })
    elif p.matrix_element and not dominant:
        warnings.append({
            "code": "matrix_not_checked",
            "message": (
                f"The preset pins {p.matrix_element} as the matrix element, "
                f"but this scan's dominant element is not known here, so the "
                f"check was skipped. Confirm it before trusting enrichment "
                f"clauses."
            ),
            "detail": {"preset_matrix": p.matrix_element},
        })

    # --- phases the library has to have -------------------------------------
    wanted: list[str] = []

    def _want(k: Any) -> None:
        key = str(k or "")
        if key and key not in wanted:
            wanted.append(key)

    for k in (p.settings.get("phase_keys") or []):
        _want(k)
    for k in ((rs.phase_keys if rs and rs.phase_keys else None) or []):
        _want(k)
    for rule in rules:
        _want(rule.phase_key)
    for d in defs:
        _want(d.phase_key)

    available_list = (None if available_phase_keys is None
                      else [str(k) for k in available_phase_keys])

    # --- a preset that does not pin its candidate list ----------------------
    #
    # A WARNING, not a blocker: an unpinned preset is perfectly usable, it is
    # just not reproducible, and refusing it would make the built-in starter
    # presets — which cannot know a stranger's library — unusable on the day
    # they are needed most.
    #
    # It fires whether or not the library could be read, because unpinnedness
    # is a property of the preset and not of this machine's database.
    if not p.phase_list_pinned:
        n_available = None if available_list is None else len(available_list)
        warnings.append({
            "code": "phase_list_not_pinned",
            "message": (
                "This preset does not name the phases it may choose from, so "
                "it runs against whatever the CIF library holds at the time"
                + (f" — {n_available} phase(s) right now" if n_available
                   is not None else "")
                + ". Adding one CIF to the library changes what this preset "
                  "does, while its name, its numbers and its content hash all "
                  "stay identical: two runs a month apart would only be "
                  "distinguishable by diffing their provenance afterwards."
                # Said explicitly, because a preset with rules LOOKS pinned:
                # a rule decides which candidates may compete, it does not
                # remove the rest of the library from the competition.
                + (f" The {len(wanted)} phase(s) it names in rules or regions "
                   f"do not narrow the list — they decide which candidates "
                   f"may compete, and every other phase in the library still "
                   f"does." if wanted else "")
                + " Pin the phase list to make the recipe reproducible."
            ),
            "detail": {"pinned": False, "n_available": n_available,
                       "named_in_rules": list(wanted)},
        })

    if available_list is None:
        if wanted:
            warnings.append({
                "code": "phases_not_checked",
                "message": ("The phase library was not supplied, so the "
                            "preset's phase list could not be checked."),
                "detail": {"phase_keys": wanted},
            })
    else:
        available = set(available_list)
        absent = [k for k in wanted if k not in available]
        if absent:
            blockers.append({
                "code": "missing_phase",
                "message": (
                    "The phase library does not contain: " + ", ".join(absent)
                    + ". The preset names them explicitly, so applying it "
                      "would run a different phase list than the one it was "
                      "written for."
                ),
                "detail": {"phase_keys": absent},
            })

    # --- the element set, even a superset -----------------------------------
    authored = {e for e in (p.authored_elements or []) if e}
    if authored and authored != measured_set:
        added = sorted(measured_set - authored)
        removed = sorted(authored - measured_set)
        warnings.append({
            "code": "element_set_differs",
            "message": (
                "This scan measured a different element set than the preset "
                "was written against"
                + (f" (extra here: {', '.join(added)})" if added else "")
                + (f" (absent here: {', '.join(removed)})" if removed else "")
                + ". at% is renormalised over the measured elements, so an "
                  "extra element shifts every other element's at% — every "
                  "absolute at% window therefore means a different composition "
                  "here than it did there, without a single number in the "
                  "preset having changed."
            ),
            "detail": {"authored": sorted(authored),
                       "measured": sorted(measured_set),
                       "added": added, "removed": removed},
        })

    # --- step size ----------------------------------------------------------
    authored_step = p.authored_step_um
    if authored_step and step_um:
        # Below ~1 % the smoothing box lands on the same pixel count; above it
        # the same setting covers a different physical width.
        if abs(float(step_um) - float(authored_step)) > 0.01 * abs(float(authored_step)):
            warnings.append({
                "code": "step_size_differs",
                "message": (
                    f"The preset was authored at {float(authored_step):g} um "
                    f"per pixel, this scan is {float(step_um):g}. Smoothing is "
                    f"a physical length, so the same setting covers a "
                    f"different area here."
                ),
                "detail": {"authored_step_um": float(authored_step),
                           "step_um": float(step_um)},
            })

    # --- guards that could not run at all -----------------------------------
    #
    # ``element_set_differs`` and ``step_size_differs`` are gated on what the
    # preset records about its own authoring, and a preset written through the
    # raw API, by hand, or by a colleague on an older build carries neither.
    # Those two warnings then never fire — and a check that silently does not
    # run reads exactly like a check that ran and found nothing. That is the
    # silent-adaptation failure this whole module exists to refuse, so the
    # absence of a guard is reported as loudly as its findings.
    #
    # The scan side counts too: a file with no step size disables the step
    # guard just as thoroughly as a preset with no ``authored_step_um``.
    inactive: list[dict] = []
    if not authored:
        inactive.append({
            "guard": "element_set_differs",
            "missing": "authored_elements",
            "why": ("without the element list the preset was written against, "
                    "nothing can tell whether this scan measured the same "
                    "elements — and at% is renormalised over them, so a "
                    "different set silently redefines every absolute window"),
        })
    if not authored_step:
        inactive.append({
            "guard": "step_size_differs",
            "missing": "authored_step_um",
            "why": ("without the step size the preset was written at, the "
                    "smoothing width cannot be compared — the same setting "
                    "covers a different physical area at a different step"),
        })
    elif not step_um:
        inactive.append({
            "guard": "step_size_differs",
            "missing": "step_um",
            "why": ("this scan carries no step size, so the preset's authoring "
                    "step has nothing to be compared against"),
        })
    if inactive:
        warnings.append({
            "code": "portability_not_checkable",
            "message": (
                f"{len(inactive)} portability check(s) could not run: "
                + "; ".join(f"{i['guard']} (no {i['missing']}: {i['why']})"
                            for i in inactive)
                + ". This is not a clean result — it is an unchecked one."
            ),
            "detail": {"guards": inactive},
        })

    # --- absolute windows are the least portable thing in a preset ----------
    absolute: list[str] = []
    for rule in rules:
        absolute.extend(_absolute_windows(
            rule, f"rule for {rule.phase_formula or rule.phase_key}"))
    for d in defs:
        absolute.extend(_absolute_windows(d, f"region '{d.name}'"))
    if absolute:
        warnings.append({
            "code": "absolute_window_not_portable",
            "message": (
                f"The preset contains {len(absolute)} absolute at% window(s): "
                + "; ".join(absolute)
                + ". Measured on one scan in this project, 'Si >= 2x "
                  "background' kept 96.3 % of a real particle while "
                  "'Si >= 40 at%' kept 46.3 %. Enrichment re-measures its bar "
                  "on each dataset and is the portable form; absolute windows "
                  "largely are not."
            ),
            "detail": {"clauses": absolute},
        })

    return CompatibilityReport(
        ok=not blockers,
        blockers=blockers,
        warnings=warnings,
        preset_name=p.name,
        preset_hash=p.content_hash,
    )
