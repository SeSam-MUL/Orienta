"""Every reason code the backend can send has a sentence, in all four languages.

The list is READ FROM ``backend/api/routes/addons.py``, never copied here. A
copied list passes forever while the backend grows a fourteenth code, and the
one thing this guard exists to catch is exactly that: a code that reaches a
user as a bare token because nobody remembered the locale files.

A pytest and not a vitest, deliberately. A vitest parsing Python source would
couple the frontend suite to the backend's file layout; this gives the same
guarantee -- it reads the same two things -- without that coupling. No other
pytest in this repo reads a locale file; this is the first.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

#: Repo root from this file: tests/addons/ -> tests/ -> root.
ROOT = Path(__file__).resolve().parents[2]
ROUTES = ROOT / "backend" / "api" / "routes" / "addons.py"
LOCALES = ROOT / "frontend" / "src" / "locales"
LANGUAGES = ("en", "de", "ja", "zh")

#: The eleven codes that existed when this guard was written, plus the two the
#: job routes added. A FLOOR, not the list: the point is to notice growth, and
#: a parse that silently found nothing would make every assertion below
#: vacuously true forever -- which is the failure this file is here to prevent,
#: arriving through the file itself.
MINIMUM_CODES = 11

_CONSTANT = re.compile(r'^REASON_[A-Z0-9_]+\s*=\s*"([a-z][a-z0-9_]*)"\s*$',
                       re.MULTILINE)


def _read(path: Path) -> str:
    """Read, or fail naming the path that was tried.

    A missing file must not read as "nothing to check". Both sides of this
    comparison have moved in this repo's history -- routes were split, locales
    were renamed -- and either move would otherwise turn this guard off
    without a word.
    """
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        pytest.fail(f"cannot read {path} ({exc}). This guard compares the "
                    "backend's REASON_* constants against the add-on locale "
                    "files; if either has moved, fix the path here rather "
                    "than letting the check pass on nothing.")


def backend_reason_codes() -> set[str]:
    codes = set(_CONSTANT.findall(_read(ROUTES)))
    assert len(codes) >= MINIMUM_CODES, (
        f"parsed only {len(codes)} REASON_* constants from {ROUTES}; expected "
        f"at least {MINIMUM_CODES}. Either the constants moved or their shape "
        "changed -- in both cases this guard is now checking nothing, so it "
        "fails here instead of passing quietly.")
    return codes


def test_the_backend_codes_can_be_read_at_all():
    codes = backend_reason_codes()
    assert "addon_not_enabled" in codes, sorted(codes)


@pytest.mark.parametrize("lng", LANGUAGES)
def test_every_backend_reason_has_a_sentence_in_this_language(lng):
    doc = json.loads(_read(LOCALES / lng / "addons.json"))
    have = set(doc.get("reason", {}))
    missing = sorted(backend_reason_codes() - have)
    assert not missing, (
        f"{lng}/addons.json has no sentence for: {missing}. A code without one "
        "reaches the user as a bare token like 'addon_not_enabled'.")


@pytest.mark.parametrize("lng", LANGUAGES)
def test_no_sentence_is_empty_or_left_as_english_placeholder(lng):
    doc = json.loads(_read(LOCALES / lng / "addons.json"))
    blank = sorted(k for k, v in doc.get("reason", {}).items()
                   if not isinstance(v, str) or not v.strip())
    assert not blank, f"{lng}/addons.json has empty sentences for: {blank}"
    for extra in ("unknownTitle", "noDetail"):
        value = doc.get("failure", {}).get(extra)
        assert isinstance(value, str) and value.strip(), (
            f"{lng}/addons.json is missing failure.{extra}, which is what the "
            "page falls back to when a failure carries no usable text at all.")


def test_the_four_locales_agree_on_their_keys():
    """Not just "each covers the backend" -- the same shape everywhere.

    A key that exists only in English is how a page ends up showing one
    language its sentence and another the raw key, and the per-language test
    above cannot see it: it only checks what the BACKEND names.
    """
    shapes = {}
    for lng in LANGUAGES:
        doc = json.loads(_read(LOCALES / lng / "addons.json"))
        shapes[lng] = ({f"reason.{k}" for k in doc.get("reason", {})}
                       | {f"failure.{k}" for k in doc.get("failure", {})})
    reference = shapes["en"]
    for lng in LANGUAGES[1:]:
        assert shapes[lng] == reference, (
            f"{lng} differs from en: missing {sorted(reference - shapes[lng])}, "
            f"extra {sorted(shapes[lng] - reference)}")


def test_no_two_languages_ship_the_same_string_for_a_code():
    """A sentence identical across en/de/ja/zh is an untranslated copy.

    Measured against the real files rather than assumed: every one of these
    is a full sentence, so a collision is not a coincidence of short words.
    """
    docs = {lng: json.loads(_read(LOCALES / lng / "addons.json"))["reason"]
            for lng in LANGUAGES}
    untranslated = [code for code in docs["en"]
                    if any(docs[lng].get(code) == docs["en"][code]
                           for lng in ("de", "ja", "zh"))]
    assert not untranslated, (
        f"these codes carry the English string in another language: "
        f"{untranslated}")
