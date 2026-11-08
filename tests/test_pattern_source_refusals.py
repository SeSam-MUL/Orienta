"""A result's own source file that cannot be read for a reason with a code says the code.

``tools.pattern_comparison.get_experimental_pattern`` reads a result's patterns from the result's own
file when another file is on screen. When that file is refused with a code (its metadata pairing is
lost, the measurement or pattern source is not in the file any more), the answer used to be an empty
panel: the code was swallowed and logged. The caller can now hand a list in (``refusals``) and gets
the code back, so the Pattern Match window can word it. Without the list nothing changes."""
import os
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("ORIENTA_NO_STARTUP_REAPER", "1")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.problem import problem  # noqa: E402
from backend.api.routes import ebsd_viewer as ev  # noqa: E402
from backend.api.services.load_choices import LoadChoiceError  # noqa: E402
from tools import pattern_comparison as pc  # noqa: E402


@pytest.fixture
def result(tmp_path, monkeypatch):
    source = tmp_path / "other.h5oina"
    source.write_bytes(b"x")                         # exists; its contents are never read here
    on_screen = tmp_path / "on_screen.h5oina"
    on_screen.write_bytes(b"x")
    monkeypatch.setattr(ev, "_ebsd_file_path", str(on_screen))
    monkeypatch.setattr(ev, "_get_active_signal", lambda: None)
    monkeypatch.setattr(pc, "_source_signal_cache", {})
    return types.SimpleNamespace(metadata={"source_file": str(source)})


def _refuse_pairing(*_a, **_k):
    raise problem(400, "apexPairingLost", "lost")


def test_a_lost_pairing_comes_back_as_its_code(result, monkeypatch):
    monkeypatch.setattr(ev, "remembered_apex_pairing", _refuse_pairing)
    refusals = []
    assert pc.get_experimental_pattern(result, 0, 0, refusals=refusals) is None
    assert refusals == ["apexPairingLost"]


def test_a_variant_the_file_does_not_hold_comes_back_as_its_code(result, monkeypatch):
    import safe_loader

    def refuse(*_a, **_k):
        raise LoadChoiceError("loadVariantUnknown", "no such source")

    monkeypatch.setattr(ev, "remembered_apex_pairing", lambda *a, **k: None)
    monkeypatch.setattr(safe_loader, "load_ebsd_safe", refuse)
    refusals = []
    assert pc.get_experimental_pattern(result, 0, 0, refusals=refusals) is None
    assert refusals == ["loadVariantUnknown"]


def test_an_error_without_a_code_says_nothing_and_the_answer_is_as_before(result, monkeypatch):
    import safe_loader

    def broken(*_a, **_k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(ev, "remembered_apex_pairing", lambda *a, **k: None)
    monkeypatch.setattr(safe_loader, "load_ebsd_safe", broken)
    refusals = []
    assert pc.get_experimental_pattern(result, 0, 0, refusals=refusals) is None
    assert refusals == []


def test_without_a_list_a_refused_source_is_still_none(result, monkeypatch):
    monkeypatch.setattr(ev, "remembered_apex_pairing", _refuse_pairing)
    assert pc.get_experimental_pattern(result, 0, 0) is None


def test_a_refused_source_is_not_cached_so_the_next_click_asks_again(result, monkeypatch):
    monkeypatch.setattr(ev, "remembered_apex_pairing", _refuse_pairing)
    first, second = [], []
    pc.get_experimental_pattern(result, 0, 0, refusals=first)
    pc.get_experimental_pattern(result, 0, 1, refusals=second)
    assert first == ["apexPairingLost"] and second == ["apexPairingLost"]
