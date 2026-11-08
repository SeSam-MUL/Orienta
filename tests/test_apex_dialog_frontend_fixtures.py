"""The two lists the APEX load dialog's locale tests read, held against the backend.

The dialog words every refusal code of ``edax_apex.APEX_CODES`` and every limit that
``hex_guard.limits()`` returns, one text per key in four languages. The frontend tests
(``frontend/src/components/EBSDViewer/apexLocales.test.js``) read these two fixtures; this file is the
other half: it compares the fixtures with the backend as it is now, so a code or a limit added on
this side turns a test red until the fixture - and with it the locale test and the texts - follows.
Regenerate with the commands in each test's message.
"""
from __future__ import annotations

import json
from pathlib import Path

import edax_apex
from backend.api.services import hex_guard

FIXTURES = (Path(__file__).resolve().parents[1] / "frontend" / "src" / "components" / "EBSDViewer"
            / "__fixtures__")

#: What the dialog does not read from a limit: the English sentences and the route it guards.
DROPPED = ("message", "method", "path")


def _read(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_the_apex_code_fixture_is_the_backends_list_in_the_backends_order():
    assert _read("apexCodes.json") == {"codes": list(edax_apex.APEX_CODES)}, (
        "edax_apex.APEX_CODES changed: regenerate apexCodes.json ({'codes': list(APEX_CODES)}) and "
        "add a text for each new code under errors.* of ebsdviewer.json in en/de/ja/zh")


def test_the_hex_limit_fixture_is_the_backends_answer_without_the_english_sentences():
    live = hex_guard.limits()

    def slim(item):
        return {k: v for k, v in item.items() if k not in DROPPED}

    expected = {
        "groups": {stage: [slim(i) for i in items] for stage, items in live["groups"].items()},
        "categories": live["categories"],
        "not_limited": live["not_limited"],
    }
    assert _read("hexLimits.json") == expected, (
        "hex_guard.limits() changed: regenerate hexLimits.json (the answer without message, method "
        "and path) and add a text for each new key, category, affected kind and parameter under "
        "hexLimits.* of ebsdviewer.json in en/de/ja/zh")


def test_the_fixture_has_the_fields_the_dialog_reads_on_every_item():
    fixture = _read("hexLimits.json")
    assert set(fixture["groups"]) == {"before_indexing", "after_indexing"}       # no batch group any more
    for stage, items in fixture["groups"].items():
        for item in items:
            assert item["key"] and item["category"] in fixture["categories"], (stage, item)
            assert isinstance(item["partial"], bool), item
            assert isinstance(item["affects"], list) and isinstance(item["parameters"], list), item
