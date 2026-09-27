"""The electron-builder configuration, pinned at the parts that were broken.

Every assertion here corresponds to a failure MEASURED on 2026-09-20 by running
the packager and launching what it produced. None of it is stylistic.

Before these settings, `npx electron-builder --dir` ended with

    Application entry file "..\\electron\\main.js" in "...app.asar" is corrupted:
    "..\\electron\\main.js" was not found in this archive

and had it got past that, Electron 44 would have loaded a CommonJS main process
as ESM and died on `require` at the first line, because the packager copies
`frontend/package.json` — which declares `"type": "module"` for Vite — into the
package, where it becomes the nearest manifest above `main.js`.

These are exactly the defects that no amount of reading the config reveals and
that `npm test` cannot see: the frontend suite runs in jsdom and never loads
`electron/`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG = REPO_ROOT / "frontend" / "package.json"


def _pkg() -> dict:
    return json.loads(PKG.read_text(encoding="utf-8"))


def _build() -> dict:
    return _pkg()["build"]


def test_the_entry_point_is_inside_the_packaged_app():
    """`main` naming a path outside the app root is an install-time failure on
    the user's machine, not a build-time one."""
    for main in (_pkg()["main"], _build().get("extraMetadata", {}).get("main")):
        if main is not None:
            assert not str(main).startswith(".."), (
                f'"main": {main!r} points outside the computed app root'
            )


def test_the_packaged_manifest_overrides_the_vite_module_type():
    """frontend/package.json needs "type": "module" for Vite, and the packager
    copies it into the app. Without this override Electron treats the CommonJS
    main process as ESM: `ReferenceError: require is not defined in ES module
    scope`, on every machine, before any window exists."""
    assert _pkg().get("type") == "module", (
        "precondition changed — if Vite no longer needs this, the override below "
        "may be removable, but check by BUILDING, not by reading"
    )
    assert _build().get("extraMetadata", {}).get("type") == "commonjs"


def test_the_electron_sources_are_copied_in_by_an_explicit_mapping():
    """A bare "../electron/**/*" pattern matches NOTHING: electron-builder
    resolves file patterns against the app directory and does not climb out of
    it. The failure is silent — the files are simply absent."""
    mappings = [f for f in _build()["files"] if isinstance(f, dict)]
    assert any(m.get("from") == "../electron" and m.get("to") == "electron"
               for m in mappings), _build()["files"]
    assert not any(isinstance(f, str) and f.startswith("..") for f in _build()["files"]), (
        "a string pattern that climbs above the app directory silently matches nothing"
    )


def test_the_app_files_stay_unpacked():
    """The updater runs `electron/apply_update.py` with the installed Python.
    A path inside an asar archive is not something another process can open."""
    assert _build()["asar"] is False


def test_the_installer_is_per_user_and_needs_no_admin_rights():
    nsis = _build()["nsis"]
    assert nsis["perMachine"] is False
    assert nsis["oneClick"] is False
    # A user who picks C:\Program Files needs the elevation the design forbids.
    # Briefly enabled on 2026-09-21 and taken back the same day: in "only for
    # me" mode the installer runs unelevated and the directory page checks
    # nothing, so choosing Program Files ends mid-install in NSIS's "Error
    # opening file for writing — Abort / Retry / Ignore". Someone who wants
    # Program Files has a proper path already: the "for all users" page, which
    # asks for elevation. The folder that actually matters for disk space — the
    # 2.5–8 GB data folder — is chosen in the setup wizard instead.
    assert nsis["allowToChangeInstallationDirectory"] is False


def test_the_window_icon_is_packaged():
    """`main.js` builds the BrowserWindow icon from ../resources/icon.png.
    `win.icon` covers the executable and the installer, not the window."""
    mappings = [f for f in _build()["files"] if isinstance(f, dict)]
    assert any(m.get("from") == "../resources" for m in mappings)


def test_the_output_directory_is_gitignored():
    root = Path(__file__).resolve().parents[1]
    output = _build()["directories"]["output"]
    assert output.startswith("../release/"), output
    ignored = (root / ".gitignore").read_text(encoding="utf-8")
    assert "release/" in ignored or "/release" in ignored


def test_the_app_version_is_not_the_scaffold_default():
    """app.getVersion() decides which release tag the setup asks GitHub for."""
    assert _pkg()["version"] != "1.0.0"


@pytest.mark.parametrize("platform", ["mac", "linux"])
def test_every_platform_we_ship_for_has_a_target(platform):
    """Was `test_only_windows_is_targeted_for_now`, inverted with the port.

    Windows was deliberately first; the note said to re-add these with the
    port rather than ship a macOS target nobody had built. This is that."""
    assert platform in _build()


def test_the_mac_build_is_ad_hoc_signed():
    """Not cosmetic: Apple Silicon refuses to LOAD an unsigned arm64 binary
    at all, so "no signature" is not an option even without a developer
    account. `-` is the local ad-hoc identity and needs none."""
    assert _build()["mac"]["identity"] == "-"


def test_the_mac_build_does_not_claim_a_hardened_runtime_or_notarisation():
    """Both are only meaningful WITH notarisation, which needs a paid account
    we do not have. Claiming them produces a build that fails its own checks."""
    assert _build()["mac"]["hardenedRuntime"] is False
    assert _build()["mac"]["notarize"] is False


def test_the_mac_build_states_the_macos_floor():
    """platform.js refuses below macOS 14 with a sentence; the package must
    agree, or the two disagree about what we support."""
    assert _build()["mac"]["minimumSystemVersion"] == "14.0"


@pytest.mark.parametrize("platform,arch", [("mac", "arm64"), ("linux", "x64")])
def test_each_platform_builds_only_the_architecture_we_support(platform, arch):
    """An Intel Mac has no PyTorch build and we ship no ARM Linux. Building
    an architecture we refuse at install time wastes a runner and produces a
    file that cannot work."""
    targets = _build()[platform]["target"]
    assert [t["arch"] for t in targets] == [[arch]]


def test_the_mac_and_linux_icons_are_large_enough_for_an_icns():
    """electron-builder needs at least 512 px to generate a macOS icon set,
    and `resources/icon.png` is 256 -- the Windows icon, which would either
    fail the build or be upscaled into something soft on a Retina screen."""
    import struct
    for platform in ("mac", "linux"):
        rel = _build()[platform]["icon"].removeprefix("../")
        data = (REPO_ROOT / rel).read_bytes()[:33]
        assert rel.endswith(".png"), rel
        width, height = struct.unpack(">II", data[16:24])
        assert width >= 512 and height >= 512, f"{rel} is {width}x{height}"


def test_electron_builder_is_pinned_exactly():
    """26.15.3 ad-hoc signs a mac build by itself. A caret could take us to a
    version that does not -- silently, on a platform nobody here can check."""
    spec = _pkg()["devDependencies"]["electron-builder"]
    assert spec == "26.15.3", f"must be exact, not a range: {spec}"


def test_the_shell_does_not_carry_its_own_copy_of_the_frontend():
    """The UI ships in the RUNTIME package, which the backend serves.

    A second copy inside the installed .exe cannot be reached -- nothing loads
    it -- but it is updated by a different mechanism from the one the user
    actually sees, so after an update the two disagree. Either it is dead
    weight or it is a stale UI waiting for a fallback to find it.
    """
    build = _build()
    patterns = [f for f in build["files"] if isinstance(f, str)]
    assert not [p for p in patterns if p.startswith("dist")], patterns


def test_the_shell_carries_the_files_it_does_need():
    """The counterpart: dropping too much is the other way to break this."""
    build = _build()
    froms = {f["from"] for f in build["files"] if isinstance(f, dict)}
    assert "../electron" in froms, "the main process itself must be packaged"
