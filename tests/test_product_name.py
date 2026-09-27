"""The product is called Orienta everywhere a user can read it.

A Mac tester reported "the About field still says something about KikuchiGUI".
The About section itself was already clean, and the first version of this file
guessed the tester had read the per-user config path shown on the same Settings
page. The M5 report then named the real one: the macOS MENU BAR, "About
kikuchipy-gui", from `app.name` — fixed in electron/main.js, covered by
tests/electron/appName.test.js.

The config path still says "Kikuchipy" and is still not renamed here: it holds
live user data and needs a migration. See tasks/mac-tester/naming-audit.md.

This guard covers the surfaces that CAN be renamed freely, because the old
name kept coming back in exactly those places (a stale doc claiming the
exported `software` attribute is "Kikuchipy GUI", a stale window-title
comment).

Two names must NOT be caught by it, and a blind search-and-replace would have
destroyed both:

  * "kikuchipy"  — the third-party library we depend on and cite,
  * "Kikuchi"    — the physical pattern (Kikuchi bands, Kikuchi lines).

So the guard looks for the PRODUCT spellings only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Product-name spellings that must not appear in shipped, user-readable text.
# Deliberately NOT r"kikuchipy" on its own: that is the library.
FORBIDDEN = re.compile(
    r"Kikuchipy[ _-]?GUI"      # "Kikuchipy GUI", "Kikuchipy_GUI", "kikuchipy-gui"
    r"|KikuchiGUI"
    r"|KikuchipyGUI"
    r"|Kikuchi[ _-]?GUI",
    re.IGNORECASE,
)


def _locale_files() -> list[Path]:
    return sorted((ROOT / "frontend" / "src" / "locales").rglob("*.json"))


def _iter_strings(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _iter_strings(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _iter_strings(v, f"{path}[{i}]")
    elif isinstance(node, str):
        yield path, node


class TestNothingTheUserReadsSaysTheOldName:
    def test_every_locale_string(self):
        offenders = []
        for f in _locale_files():
            data = json.loads(f.read_text(encoding="utf-8"))
            for key, text in _iter_strings(data):
                if FORBIDDEN.search(text):
                    offenders.append(f"{f.relative_to(ROOT)} :: {key} :: {text[:80]}")
        assert not offenders, "old product name in user-facing text:\n" + "\n".join(offenders)

    @pytest.mark.parametrize("rel", [
        "frontend/index.html",
        "docs/light_h5_format.md",
    ])
    def test_shipped_file(self, rel):
        text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        hits = [m.group(0) for m in FORBIDDEN.finditer(text)]
        assert not hits, f"{rel} still says {set(hits)}"

    def test_the_backend_app_name(self):
        from backend.api.services.app_version import APP_NAME

        assert APP_NAME == "Orienta"
        assert not FORBIDDEN.search(APP_NAME)

    def test_the_exported_software_attribute(self):
        """The name written INTO users' result files.

        docs/light_h5_format.md documented "Kikuchipy GUI" long after the code
        had moved to "Orienta"; the doc was the lie, not the code.
        """
        written = set()
        for rel in ("backend/api/services/result_exporter.py",
                    "backend/api/routes/indexing.py"):
            for m in re.finditer(r'attrs\["software"\]\s*=\s*"([^"]+)"',
                                 (ROOT / rel).read_text(encoding="utf-8", errors="replace")):
                written.add(m.group(1))
        assert written == {"Orienta"}, written

        doc = (ROOT / "docs" / "light_h5_format.md").read_text(encoding="utf-8")
        assert '`software` | str | always `"Orienta"`' in doc

    def test_the_packaged_app_is_named_orienta(self):
        """electron-builder names the bundle from build.productName.

        The dmg and the dock take this name.

        NOT the macOS app menu: that reads `app.name`, which Electron takes from
        package.json `name`, and an M5 tester found "About kikuchipy-gui" in the
        menu bar of the signed dmg. This docstring used to claim `name` is
        dev-only; that was wrong. electron/main.js calls app.setName('Orienta')
        and tests/electron/appName.test.js covers it.
        """
        pkg = json.loads((ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
        assert pkg["build"]["productName"] == "Orienta"

    def test_the_app_id_is_left_alone_on_purpose(self):
        """A canary, not an aspiration.

        `com.kikuchipy.gui` carries the update path and the uninstaller entry of
        every installed copy. Changing it orphans them, so it must stay until
        someone plans that migration. If this fails, read
        tasks/mac-tester/naming-audit.md before "fixing" it.
        """
        pkg = json.loads((ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
        assert pkg["build"]["appId"] == "com.kikuchipy.gui"
