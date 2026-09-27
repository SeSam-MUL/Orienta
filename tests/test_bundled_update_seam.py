"""The seam: the shell parks the bundled package, the applier applies it.

Both halves are unit-tested on their own side -- `tests/electron/bundledUpdate.test.js`
for the decision and the parking, `tests/test_apply_update.py` for the unpack --
and the bug this closes lived in neither half. It lived in the JOIN: the applier
opens with "Apply a parked runtime package, before the backend starts", had 33
tests, and no line in `electron/` ever ran it outside the first-install wizard.
Two green suites, one disconnected wire, and a 0.4.6 shell serving 0.4.4 code.

So this test crosses the seam in the same order the shell does, with a fixture
that makes the two promises visible at once:

  * the version on disk moves v0.4.4 -> v0.4.6
  * a file the new MANIFEST drops is pruned, not left behind for ever
  * `runtime/Database` -- the crystal library -- is untouched

Node is required, because the parking half IS JavaScript; where there is no node
the test skips rather than asserting a weaker thing.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(
    NODE is None, reason="the parking half is JavaScript and needs node")


def _make_bundle(resources: Path, tag: str) -> Path:
    """A package that renames one file, so the prune has something to do."""
    resources.mkdir(parents=True, exist_ok=True)
    zip_path = resources / f"orienta-runtime-{tag}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("VERSION", f"{tag}\n")
        zf.writestr("MANIFEST", "VERSION\nMANIFEST\nbackend/kept.py\nbackend/added.py\n")
        zf.writestr("backend/kept.py", "# the new one\n")
        zf.writestr("backend/added.py", "# arrived with this release\n")
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    (resources / f"{zip_path.name}.sha256").write_text(
        f"{digest}  {zip_path.name}\n", encoding="utf-8", newline="")
    return zip_path


def _make_installation(home: Path) -> None:
    runtime = home / "runtime"
    (runtime / "backend").mkdir(parents=True)
    (runtime / "VERSION").write_text("v0.4.4\n", encoding="utf-8", newline="")
    (runtime / "MANIFEST").write_text(
        "VERSION\nMANIFEST\nbackend/kept.py\nbackend/dropped.py\n",
        encoding="utf-8", newline="")
    (runtime / "backend" / "kept.py").write_text("# the old one\n", encoding="utf-8")
    (runtime / "backend" / "dropped.py").write_text("# this release removes me\n",
                                                    encoding="utf-8")
    # The crystal library lives INSIDE runtime/, which is why the applier refuses
    # any member landing there and never deletes the directory wholesale.
    library = runtime / "Database" / "CIF_Library"
    library.mkdir(parents=True)
    (library / "Al.cif").write_text("data_Al\n_cell_length_a 4.05\n", encoding="utf-8")


def _park_via_the_shell(home: Path, resources: Path) -> dict:
    """Call the real `parkBundled`, the way main.js does."""
    script = (
        "const bu = require(process.argv[1]);"
        "const pkg = bu.bundledPackage(process.argv[3]);"
        "if (!pkg) { console.error('no bundled package found'); process.exit(2); }"
        "const decision = bu.updateDecision({"
        "  bundled: pkg,"
        "  installed: bu.installedTag(process.argv[2]),"
        "  parked: bu.alreadyParked(process.argv[2]),"
        "});"
        "if (decision.action !== 'park') {"
        "  console.error('decision was ' + decision.action + ': ' + decision.reason);"
        "  process.exit(3);"
        "}"
        "const rec = bu.parkBundled({ home: process.argv[2], bundled: pkg });"
        "process.stdout.write(JSON.stringify({ decision, rec }));"
    )
    out = subprocess.run(
        [NODE, "-e", script, str(REPO / "electron" / "bundled_update.js"),
         str(home), str(resources)],
        capture_output=True, text=True, check=True, cwd=REPO)
    return json.loads(out.stdout)


def _apply(home: Path) -> dict:
    verdict = home / "verdict.json"
    proc = subprocess.run(
        [sys.executable, str(REPO / "electron" / "apply_update.py"),
         "--home", str(home), "--result-json", str(verdict)],
        capture_output=True, text=True)
    assert proc.returncode == 0, f"the applier reported the runtime incomplete: {proc.stdout}\n{proc.stderr}"
    return json.loads(verdict.read_text(encoding="utf-8"))


def test_a_newer_bundle_replaces_the_program_files_and_spares_the_library(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    resources = tmp_path / "resources"
    _make_installation(home)
    _make_bundle(resources, "v0.4.6")

    parked = _park_via_the_shell(home, resources)
    assert parked["decision"]["action"] == "park"
    assert parked["rec"]["tag"] == "v0.4.6"
    assert (home / "pending.json").is_file()

    verdict = _apply(home)
    assert verdict["applied"] is True, verdict
    assert verdict["tag"] == "v0.4.6"

    runtime = home / "runtime"
    # The one symptom a user could see: About reads this file.
    assert (runtime / "VERSION").read_text(encoding="utf-8").strip() == "v0.4.6"
    assert (runtime / "backend" / "kept.py").read_text(encoding="utf-8") == "# the new one\n"
    assert (runtime / "backend" / "added.py").is_file()
    # Pruned, because the new MANIFEST does not list it. extract_only() could not
    # do this, which is why the parking route exists.
    assert not (runtime / "backend" / "dropped.py").exists()
    # Untouched, byte for byte.
    assert (runtime / "Database" / "CIF_Library" / "Al.cif").read_text(encoding="utf-8") \
        == "data_Al\n_cell_length_a 4.05\n"
    # Cleaned up after itself, so the next start does not apply it again.
    assert not (home / "pending.json").exists()


def test_the_same_bundle_twice_changes_nothing_the_second_time(tmp_path):
    """Idempotence across the seam: a reinstall of the same version is a no-op.

    Worth its own test because the shell runs this on EVERY start. A second
    application that pruned or rewrote anything would make every launch a
    rewrite of the program files.
    """
    home = tmp_path / "home"
    home.mkdir()
    resources = tmp_path / "resources"
    _make_installation(home)
    _make_bundle(resources, "v0.4.6")
    _park_via_the_shell(home, resources)
    _apply(home)

    runtime = home / "runtime"
    before = {p.relative_to(runtime).as_posix(): p.stat().st_mtime_ns
              for p in runtime.rglob("*") if p.is_file()}

    # Now the shell asks again, with the installation already at v0.4.6.
    script = (
        "const bu = require(process.argv[1]);"
        "const pkg = bu.bundledPackage(process.argv[3]);"
        "const d = bu.updateDecision({ bundled: pkg,"
        "  installed: bu.installedTag(process.argv[2]),"
        "  parked: bu.alreadyParked(process.argv[2]) });"
        "process.stdout.write(d.action);"
    )
    out = subprocess.run(
        [NODE, "-e", script, str(REPO / "electron" / "bundled_update.js"),
         str(home), str(resources)],
        capture_output=True, text=True, check=True, cwd=REPO)
    assert out.stdout.strip() == "none"

    after = {p.relative_to(runtime).as_posix(): p.stat().st_mtime_ns
             for p in runtime.rglob("*") if p.is_file()}
    assert after == before


def test_an_older_bundle_never_rolls_the_installation_back(tmp_path):
    """Installing an older .exe on purpose must not downgrade the program files."""
    home = tmp_path / "home"
    home.mkdir()
    resources = tmp_path / "resources"
    _make_installation(home)
    (home / "runtime" / "VERSION").write_text("v0.5.0\n", encoding="utf-8", newline="")
    _make_bundle(resources, "v0.4.6")

    script = (
        "const bu = require(process.argv[1]);"
        "const pkg = bu.bundledPackage(process.argv[3]);"
        "const d = bu.updateDecision({ bundled: pkg,"
        "  installed: bu.installedTag(process.argv[2]),"
        "  parked: bu.alreadyParked(process.argv[2]) });"
        "process.stdout.write(JSON.stringify(d));"
    )
    out = subprocess.run(
        [NODE, "-e", script, str(REPO / "electron" / "bundled_update.js"),
         str(home), str(resources)],
        capture_output=True, text=True, check=True, cwd=REPO)
    decision = json.loads(out.stdout)
    assert decision["action"] == "none"
    assert "newer than the bundled" in decision["reason"]
    assert not (home / "pending.json").exists()
    assert (home / "runtime" / "VERSION").read_text(encoding="utf-8").strip() == "v0.5.0"
