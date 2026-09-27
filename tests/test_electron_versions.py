"""Electron must stay inside its support window.

Electron supports the latest three stable majors on an eight-week cycle. This
project sat on 36.9.5 (Chromium 136, Node 22.19) while 44.4.1 (Chromium 152,
Node 24.21) was current. An installer freezes its Electron into every machine it
is run on, so the floor is pinned here rather than remembered.

Raise MINIMUM_ELECTRON_MAJOR deliberately when bumping; never lower it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

MINIMUM_ELECTRON_MAJOR = 44
MINIMUM_ELECTRON_BUILDER_MAJOR = 26

PKG = Path(__file__).resolve().parents[1] / "frontend" / "package.json"


def _major(spec: str) -> int:
    m = re.search(r"(\d+)", spec)
    assert m, f"no version number in {spec!r}"
    return int(m.group(1))


def _dev_dependencies() -> dict:
    return json.loads(PKG.read_text(encoding="utf-8"))["devDependencies"]


def test_electron_is_within_its_support_window():
    dev = _dev_dependencies()
    assert _major(dev["electron"]) >= MINIMUM_ELECTRON_MAJOR, (
        f"electron {dev['electron']} is below {MINIMUM_ELECTRON_MAJOR}; "
        "Electron supports only the latest three majors"
    )


def test_electron_builder_can_package_that_electron():
    dev = _dev_dependencies()
    assert _major(dev["electron-builder"]) >= MINIMUM_ELECTRON_BUILDER_MAJOR, (
        f"electron-builder {dev['electron-builder']} predates electron "
        f"{MINIMUM_ELECTRON_MAJOR} and cannot package it"
    )


def test_the_app_version_is_a_release_version_not_the_scaffold_default():
    """`app.getVersion()` decides which release tag the setup asks GitHub for.

    While this reads "1.0.0" — the Vite scaffold default — the first-run wizard
    and the "Reinstall packages" action both query a tag that cannot exist. It is
    read at runtime, so no test downstream of the installer can catch it.
    """
    version = json.loads(PKG.read_text(encoding="utf-8"))["version"]
    assert version != "1.0.0", (
        "frontend/package.json still carries the scaffold version; set it to the "
        "release this build will be published as"
    )
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[-+.][\w.]+)?", version), (
        f"{version!r} is not a release version"
    )
