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


def _decide_and_park(home: Path, resources: Path) -> dict:
    """The decision as main.js now makes it: both digests read, then park if asked.

    Deliberately a second helper rather than a parameter on the one above: that
    one reproduces the call WITHOUT digests, which is what shipped in Build 3, and
    keeping both means the tests can still show what the old wiring decided.
    """
    script = (
        "const bu = require(process.argv[1]);"
        "const pkg = bu.bundledPackage(process.argv[3]);"
        "if (!pkg) { console.error('no bundled package found'); process.exit(2); }"
        "let bundledDigest = null;"
        "try { bundledDigest = bu.digestFrom(pkg.sum); } catch (e) { bundledDigest = null; }"
        "const decision = bu.updateDecision({"
        "  bundled: pkg,"
        "  installed: bu.installedTag(process.argv[2]),"
        "  parked: bu.alreadyParked(process.argv[2]),"
        "  bundledDigest,"
        "  installedDigest: bu.installedDigest(process.argv[2]),"
        "});"
        "let rec = null;"
        "if (decision.action === 'park') {"
        "  rec = bu.parkBundled({ home: process.argv[2], bundled: pkg });"
        "}"
        "process.stdout.write(JSON.stringify({ decision, rec }));"
    )
    out = subprocess.run(
        [NODE, "-e", script, str(REPO / "electron" / "bundled_update.js"),
         str(home), str(resources)],
        capture_output=True, text=True, check=True, cwd=REPO)
    return json.loads(out.stdout)


def _rebuild_bundle(resources: Path, tag: str, marker: str) -> Path:
    """Another package under the SAME tag, differing only in content."""
    for stale in resources.glob(f"orienta-runtime-{tag}.zip*"):
        stale.unlink()
    zip_path = resources / f"orienta-runtime-{tag}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("VERSION", f"{tag}\n")
        zf.writestr("MANIFEST", "VERSION\nMANIFEST\nbackend/kept.py\nbackend/added.py\n")
        zf.writestr("backend/kept.py", f"# {marker}\n")
        zf.writestr("backend/added.py", "# arrived with this release\n")
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    (resources / f"{zip_path.name}.sha256").write_text(
        f"{digest}  {zip_path.name}\n", encoding="utf-8", newline="")
    return zip_path


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


def test_the_same_version_rebuilt_still_replaces_the_program_files(tmp_path):
    """Build 3 -> Build 4 under ONE tag, through the real applier and the real JS.

    This is the case Sebastian's "the release stays 0.4.6" decision creates, and
    the one a tag comparison gets wrong: his machine holds Build 3 as v0.4.6, so
    "same version" would mean "nothing to do" and the program files would stay a
    build behind beneath a new shell. The whole lifecycle is asserted here because
    each step only makes sense given the next: apply, record, do not loop, and
    apply again when the content changes again.
    """
    home = tmp_path / "home"
    home.mkdir()
    resources = tmp_path / "resources"
    _make_installation(home)
    # What every installation from before this change looks like: the tag of the
    # release, and nothing saying which build produced it.
    (home / "runtime" / "VERSION").write_text("v0.4.6\n", encoding="utf-8", newline="")
    assert not (home / "runtime" / "SOURCE_SHA256").exists()
    build4 = _make_bundle(resources, "v0.4.6")

    # 1. It must apply, and the reason must say why -- not "newer", because it is
    #    not newer; because the runtime cannot say which build it is.
    first = _decide_and_park(home, resources)
    assert first["decision"]["action"] == "park", first["decision"]
    assert "does not say which build" in first["decision"]["reason"]

    verdict = _apply(home)
    assert verdict["applied"] is True, verdict
    runtime = home / "runtime"
    assert (runtime / "backend" / "kept.py").read_text(encoding="utf-8") == "# the new one\n"
    assert (runtime / "backend" / "added.py").is_file()
    assert not (runtime / "backend" / "dropped.py").exists()      # pruned by manifest diff
    # Untouched through all of it, which is the one thing that may never break.
    assert (runtime / "Database" / "CIF_Library" / "Al.cif").read_text(encoding="utf-8") \
        == "data_Al\n_cell_length_a 4.05\n"

    # 2. The applier recorded the identity the next decision needs, and it is the
    #    digest of the package that was actually applied.
    recorded = (runtime / "SOURCE_SHA256").read_text(encoding="utf-8").strip()
    assert recorded == hashlib.sha256(build4.read_bytes()).hexdigest()

    # 3. Next start: same build, so nothing happens. Without this the fix would
    #    unpack on every launch for ever, which is worse than the bug.
    second = _decide_and_park(home, resources)
    assert second["decision"]["action"] == "none", second["decision"]
    assert "same build" in second["decision"]["reason"]
    assert not (home / "pending.json").exists()

    # 4. Build 5, same tag again: applied, and the log names both builds so that
    #    "why did it update again" has an answer that is not "trust me".
    build5 = _rebuild_bundle(resources, "v0.4.6", "the fifth build")
    third = _decide_and_park(home, resources)
    assert third["decision"]["action"] == "park", third["decision"]
    assert "DIFFERENT build" in third["decision"]["reason"]
    assert recorded[:12] in third["decision"]["reason"]
    assert hashlib.sha256(build5.read_bytes()).hexdigest()[:12] in third["decision"]["reason"]

    assert _apply(home)["applied"] is True
    assert (runtime / "backend" / "kept.py").read_text(encoding="utf-8") == "# the fifth build\n"
    assert (runtime / "SOURCE_SHA256").read_text(encoding="utf-8").strip() \
        == hashlib.sha256(build5.read_bytes()).hexdigest()


def test_a_first_install_records_which_build_it_unpacked(tmp_path):
    """`extract_only` is the wizard's path, and it must leave the same identity.

    Without it a freshly installed copy would apply, on its very first start, the
    package it had just been installed from -- harmless but absurd, and it would
    make the log say "does not say which build it is" about a runtime nobody had
    yet had the chance to change.
    """
    home = tmp_path / "home"
    (home / "runtime").mkdir(parents=True)
    resources = tmp_path / "resources"
    archive = _make_bundle(resources, "v0.4.6")

    sys.path.insert(0, str(REPO / "electron"))
    try:
        import apply_update
        result = apply_update.extract_only(archive, home / "runtime", home)
    finally:
        sys.path.pop(0)

    assert result["ok"] is True, result
    assert (home / "runtime" / "VERSION").read_text(encoding="utf-8").strip() == "v0.4.6"
    assert (home / "runtime" / "SOURCE_SHA256").read_text(encoding="utf-8").strip() \
        == hashlib.sha256(archive.read_bytes()).hexdigest()
