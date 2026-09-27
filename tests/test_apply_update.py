"""Applying an update, and the two things that must never happen.

The operation is: prune what the OLD manifest listed and the new one does not,
then extract the package over `runtime/`. It is idempotent — running it twice is
running it once — which is the whole reason this design replaced three
generations of directory swapping, each of which found a new way to destroy the
user's crystal library.

NEVER: touch anything under `runtime/Database`.
NEVER: delete a path the old manifest did not list.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "electron"))
import apply_update as au  # noqa: E402


def _zip_bytes(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return buf.getvalue()


def _package(manifest, **files):
    base = {"VERSION": "v0.4.1\n", "MANIFEST": "\n".join(manifest) + "\n"}
    base.update(files)
    return base


@pytest.fixture
def home(tmp_path):
    runtime = tmp_path / "runtime"
    (runtime / "backend" / "api").mkdir(parents=True)
    (runtime / "backend" / "api" / "main.py").write_text("old\n", encoding="utf-8", newline="")
    (runtime / "backend" / "api" / "gone.py").write_text("old\n", encoding="utf-8", newline="")
    (runtime / "MANIFEST").write_text(
        "backend/api/main.py\nbackend/api/gone.py\n", encoding="utf-8", newline="")
    (runtime / "VERSION").write_text("v0.4.0\n", encoding="utf-8", newline="")
    (runtime / "Database" / "CIF_Library").mkdir(parents=True)
    (runtime / "Database" / "CIF_Library" / "mine.cif").write_text(
        "irreplaceable", encoding="utf-8", newline="")
    return tmp_path


def _park(home, payload, tag="v0.4.1"):
    pending = home / "pending"
    pending.mkdir(exist_ok=True)
    name = f"orienta-runtime-{tag}.zip"
    data = _zip_bytes(payload)
    (pending / name).write_bytes(data)
    (home / "pending.json").write_text(
        json.dumps({"tag": tag, "file": name, "sha256": hashlib.sha256(data).hexdigest()}),
        encoding="utf-8", newline="")


def library(home):
    return home / "runtime" / "Database" / "CIF_Library" / "mine.cif"


NOTHING_TO_DO = {"applied": False, "tag": None, "error": None,
                 "dirty": False, "terminal": False}


# --------------------------------------------------------------------------
# the ordinary path
# --------------------------------------------------------------------------

def test_nothing_to_do_without_a_pending_record(home):
    assert au.apply(home) == NOTHING_TO_DO


def test_the_package_is_extracted_over_the_runtime(home):
    _park(home, _package(["backend/api/main.py"], **{"backend/api/main.py": "new\n"}))
    result = au.apply(home)
    assert result["applied"] is True and result["tag"] == "v0.4.1"
    assert (home / "runtime" / "backend" / "api" / "main.py").read_text().strip() == "new"
    assert (home / "runtime" / "VERSION").read_text().strip() == "v0.4.1"
    assert not (home / "pending.json").exists()


def test_a_file_the_new_manifest_drops_is_removed(home):
    _park(home, _package(["backend/api/main.py"], **{"backend/api/main.py": "new\n"}))
    au.apply(home)
    assert not (home / "runtime" / "backend" / "api" / "gone.py").exists()


def test_a_file_no_manifest_ever_listed_is_left_alone(home):
    stray = home / "runtime" / "user_notes.txt"
    stray.write_text("mine", encoding="utf-8", newline="")
    _park(home, _package(["backend/api/main.py"], **{"backend/api/main.py": "new\n"}))
    au.apply(home)
    assert stray.is_file()


# --------------------------------------------------------------------------
# the crystal library
# --------------------------------------------------------------------------

def test_the_database_survives_even_if_a_manifest_names_it(home):
    """The rule does not depend on the manifest being sane."""
    (home / "runtime" / "MANIFEST").write_text(
        "backend/api/main.py\nDatabase/CIF_Library/mine.cif\n", encoding="utf-8", newline="")
    _park(home, _package(["backend/api/main.py"], **{"backend/api/main.py": "new\n"}))
    au.apply(home)
    assert library(home).read_text() == "irreplaceable"


def test_a_package_that_writes_into_database_is_refused_whole(home):
    _park(home, _package(["Database/x.cif"], **{"Database/x.cif": "no\n"}))
    result = au.apply(home)
    assert result["applied"] is False
    assert result["terminal"] is True
    assert library(home).read_text() == "irreplaceable"
    assert (home / "runtime" / "VERSION").read_text().strip() == "v0.4.0", "nothing was applied"


@pytest.mark.parametrize("member", [
    "Database/x.cif", "Database", "backend/Database/x",
    # Win32 is case-insensitive and strips trailing dots and spaces. Every one
    # of these opens the SAME directory as "Database".
    "database/x.cif", "DATABASE/x.cif", "DataBase/x.cif",
    "Database./x.cif", "Database /x.cif", "Database\\x.cif",
])
def test_protected_covers_every_spelling_windows_resolves_to_the_library(member):
    assert au.protected(member), f"{member!r} would reach the crystal library"


@pytest.mark.parametrize("member", [
    "backend/api/main.py", "backend/api/routes/database.py",
    "simulation/database_browser.py",
])
def test_protected_does_not_over_reach(member):
    """`database.py` is a FILE. Refusing it would drop real modules."""
    assert not au.protected(member)


# --------------------------------------------------------------------------
# escaping the runtime
# --------------------------------------------------------------------------

def test_a_member_that_escapes_the_runtime_is_refused_whole(home):
    _park(home, _package(["x"], **{"../evil.txt": "no\n"}))
    assert au.apply(home)["applied"] is False
    assert not (home / "evil.txt").exists()


def test_a_sibling_prefix_member_is_refused_too(home):
    """`str.startswith` passes '../runtime-evil/x'. is_relative_to does not."""
    _park(home, _package(["x"], **{"../runtime-evil/x": "no\n"}))
    assert au.apply(home)["applied"] is False
    assert not (home / "runtime-evil").exists()


# --------------------------------------------------------------------------
# idempotence — the property that replaced the phase machine
# --------------------------------------------------------------------------

def test_applying_twice_is_applying_once(home):
    payload = _package(["backend/api/main.py"], **{"backend/api/main.py": "new\n"})
    _park(home, payload)
    first = au.apply(home)
    _park(home, payload)          # as if the record had survived a kill
    second = au.apply(home)
    assert first["applied"] and second["applied"]
    assert (home / "runtime" / "backend" / "api" / "main.py").read_text().strip() == "new"
    assert library(home).read_text() == "irreplaceable"


def test_an_apply_interrupted_after_the_prune_still_removes_what_it_removed(home, monkeypatch):
    """The property prune-first actually buys.

    With extract-then-prune, a run killed after the extraction can NEVER prune:
    runtime/MANIFEST is already the new one, so the next start computes an empty
    difference and the removed file stays for ever.
    """
    payload = _package(["backend/api/main.py"], **{"backend/api/main.py": "new\n"})
    _park(home, payload)
    monkeypatch.setattr(au, "safe_extract",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("killed here")))

    first = au.apply(home)

    gone = home / "runtime" / "backend" / "api" / "gone.py"
    assert first["dirty"] is True
    assert not gone.exists(), "the prune must have completed before the extraction"
    assert (home / "runtime" / "MANIFEST").read_text().split() == [
        "backend/api/main.py", "backend/api/gone.py",
    ], "the old manifest must survive an interrupted run"

    monkeypatch.undo()
    second = au.apply(home)
    assert second["applied"] is True
    assert (home / "runtime" / "backend" / "api" / "main.py").read_text().strip() == "new"


def test_re_applying_a_completed_package_prunes_nothing_and_that_is_correct(home):
    """Stated so nobody "fixes" it back: after a clean apply the old manifest
    legitimately IS the new one."""
    payload = _package(["backend/api/main.py"], **{"backend/api/main.py": "new\n"})
    _park(home, payload)
    au.apply(home)
    restored = home / "runtime" / "backend" / "api" / "gone.py"
    restored.write_text("put back by the user\n", encoding="utf-8", newline="")

    _park(home, payload)
    assert au.apply(home)["applied"] is True
    assert restored.is_file()


def test_prune_runs_before_extract(home):
    """Pinned by CALL ORDER, not by reading the source.

    The earlier version of this test searched the source for "prune(runtime"
    and found `dry_run_prune(runtime, ...)` -- a different function, on an
    earlier line -- so deleting the real prune call entirely left it green.
    A test of an ordering has to observe the ordering.
    """
    calls = []
    real_prune, real_extract = au.prune, au.safe_extract

    def spy_prune(*a, **kw):
        calls.append("prune")
        return real_prune(*a, **kw)

    def spy_extract(*a, **kw):
        calls.append("extract")
        return real_extract(*a, **kw)

    _park(home, _package(["backend/api/main.py"], **{"backend/api/main.py": "new\n"}))
    (home / "runtime" / "MANIFEST").write_text("backend/api/old.py\n", encoding="utf-8")
    (home / "runtime" / "backend" / "api").mkdir(parents=True, exist_ok=True)
    (home / "runtime" / "backend" / "api" / "old.py").write_text("x\n", encoding="utf-8")

    with mock.patch.object(au, "prune", spy_prune), \
            mock.patch.object(au, "safe_extract", spy_extract):
        assert au.apply(home)["applied"] is True

    assert calls == ["prune", "extract"], calls


# --------------------------------------------------------------------------
# terminal vs dirty
# --------------------------------------------------------------------------

def test_a_worthless_package_is_discarded_rather_than_nagging_forever(home):
    _park(home, _package(["backend/api/main.py"], **{"backend/api/main.py": "new\n"}))
    (home / "pending" / "orienta-runtime-v0.4.1.zip").unlink()

    result = au.apply(home)

    assert result["terminal"] is True
    assert result["dirty"] is False
    assert not (home / "pending.json").exists()
    assert au.apply(home) == NOTHING_TO_DO


def test_a_corrupted_package_is_refused_before_anything_is_written(home):
    _park(home, _package(["backend/api/main.py"], **{"backend/api/main.py": "new\n"}))
    record = json.loads((home / "pending.json").read_text(encoding="utf-8"))
    record["sha256"] = "0" * 64
    (home / "pending.json").write_text(json.dumps(record), encoding="utf-8", newline="")

    result = au.apply(home)

    assert result["applied"] is False
    assert result["terminal"] is True
    assert (home / "runtime" / "backend" / "api" / "main.py").read_text().strip() == "old"


def test_an_unreadable_record_is_discarded_not_repeated(home):
    _park(home, _package(["x"], **{"x": "y\n"}))
    (home / "pending.json").write_text("{ not json", encoding="utf-8", newline="")
    assert au.apply(home)["terminal"] is True
    assert not (home / "pending.json").exists()


def test_a_failure_after_the_prune_began_is_reported_as_dirty(home, monkeypatch):
    _park(home, _package(["backend/api/main.py"], **{"backend/api/main.py": "new\n"}))
    monkeypatch.setattr(au, "safe_extract",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))

    result = au.apply(home)

    assert result["dirty"] is True and result["terminal"] is False
    assert (home / "pending.json").exists(), "a dirty tree must be repairable by re-running"
    assert (home / "runtime" / au.MARKER).exists()


def test_a_terminal_failure_after_a_dirty_run_is_reported_as_dirty(home, monkeypatch):
    """The worst outcome in the design, and it was reachable: run 1 killed
    mid-extraction, then the parked zip quarantined by antivirus. Classifying
    that terminal discards the package and tells the shell to start the backend
    against a tree that is half one release and half the other."""
    _park(home, _package(["backend/api/main.py"], **{"backend/api/main.py": "new\n"}))
    monkeypatch.setattr(au, "safe_extract",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("killed")))
    assert au.apply(home)["dirty"] is True
    monkeypatch.undo()

    (home / "pending" / "orienta-runtime-v0.4.1.zip").unlink()   # quarantined

    result = au.apply(home)

    assert result["dirty"] is True
    assert result["terminal"] is False
    assert (home / "pending.json").exists()


# --------------------------------------------------------------------------
# reparse points
# --------------------------------------------------------------------------

def test_a_junction_refusal_happens_before_any_file_is_deleted(home):
    if os.name != "nt":
        pytest.skip("junctions are a Windows concept")
    target = home / "elsewhere"
    (target / "dist").mkdir(parents=True)
    (target / "dist" / "old-chunk.js").write_text("mine", encoding="utf-8", newline="")
    os.system(f'mklink /J "{home / "runtime" / "frontend"}" "{target}" >nul 2>&1')
    (home / "runtime" / "MANIFEST").write_text(
        "backend/api/main.py\nbackend/api/gone.py\nfrontend/dist/old-chunk.js\n",
        encoding="utf-8", newline="")
    _park(home, _package(["backend/api/main.py"], **{"backend/api/main.py": "new\n"}))

    result = au.apply(home)

    assert result["applied"] is False
    assert result["terminal"] is True, "nothing was touched; do not strand the user"
    assert (home / "runtime" / "backend" / "api" / "gone.py").is_file(), (
        "the refusal must precede the prune"
    )
    assert (target / "dist" / "old-chunk.js").is_file()


def test_an_unreadable_old_manifest_is_refused_rather_than_silently_skipped(home, monkeypatch):
    """Returning [] on an unreadable MANIFEST makes the prune a permanent
    no-op — the outcome the ordering exists to prevent, by another route.

    The error is raised by the REAL `_read_manifest` against a real read
    failure. An earlier version monkeypatched `_read_manifest` itself with a
    function that raised the sentence the test then asserted on, so it passed
    with the production branch deleted.
    """
    _park(home, _package(["backend/api/main.py"], **{"backend/api/main.py": "new\n"}))
    manifest = home / "runtime" / "MANIFEST"
    manifest.write_text("backend/api/old.py\n", encoding="utf-8")

    real_read = Path.read_text

    def locked(self, *a, **kw):
        if self.name == "MANIFEST" and "pending" not in str(self):
            raise PermissionError(13, "being used by another process")
        return real_read(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", locked)
    result = au.apply(home)

    assert result["applied"] is False
    assert "MANIFEST" in result["error"]
    # And the classification, which is the half that decides whether the user
    # keeps their download: a read lock clears by itself.
    assert result["terminal"] is False


def test_a_locked_manifest_keeps_the_downloaded_package(home, monkeypatch):
    """A momentary read lock must not cost the user the download.

    Antivirus and backup agents hold files open for seconds at a time. Treating
    that as "this package is bad" deletes a perfectly good 40 MB download and
    makes the user fetch it again.
    """
    _park(home, _package(["backend/api/main.py"], **{"backend/api/main.py": "new\n"}))
    (home / "runtime" / "MANIFEST").write_text("backend/api/old.py\n", encoding="utf-8")
    parked = home / "pending" / "orienta-runtime-v0.4.1.zip"
    assert parked.is_file()

    real_read = Path.read_text

    def locked(self, *a, **kw):
        if self.name == "MANIFEST" and "pending" not in str(self):
            raise PermissionError(13, "being used by another process")
        return real_read(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", locked)
    result = au.apply(home)

    assert result["terminal"] is False
    assert result["dirty"] is False
    assert parked.is_file(), "the package was discarded over a transient read lock"
    assert (home / "pending.json").is_file()


# --------------------------------------------------------------------------
# the marker, and the names that reach it sideways
# --------------------------------------------------------------------------

def test_a_manifest_entry_with_a_trailing_space_cannot_delete_the_marker():
    """`.applying ` opens the marker on Windows while comparing unequal to it.

    A prune that deletes its own marker un-marks a dirty run, so the NEXT
    failure is classified terminal: the package is discarded and the shell is
    told it may start against a half-written tree.
    """
    for spelling in (".applying ", ".applying.", ".APPLYING", "runtime/.Applying "):
        assert au.names_marker(spelling), spelling
        assert spelling not in au.prune_candidates([spelling], [])


def test_the_marker_guard_matches_the_protected_guard():
    """Both name tests must normalise the same way, or one of them is a hole."""
    assert au.prune_candidates([".applying "], []) == []
    assert au.prune_candidates(["Database. /x.cif"], []) == []
    # ...and an ordinary file is still a candidate, so the filters are not
    # simply refusing everything.
    assert au.prune_candidates(["backend/api/old.py"], []) == ["backend/api/old.py"]


# --------------------------------------------------------------------------
# a conflict is named, not split on a space
# --------------------------------------------------------------------------

def test_a_packaged_path_containing_a_space_is_matched_whole(home, tmp_path):
    """Recovering the filename with `c.split(" ")[0]` yields `docs/user`.

    That is never in the prune set, so a conflict the prune WOULD have cleared
    is reported as surviving and the whole update is refused terminally.
    """
    # The realistic shape: the old release shipped `docs/user guide` as a FILE,
    # the new one ships it as a DIRECTORY containing index.md. The prune removes
    # the old file first, so the extraction has room -- provided the conflict is
    # matched against the prune set by its whole name.
    old_file = "docs/user guide"
    new_file = "docs/user guide/index.md"
    (home / "runtime" / "docs").mkdir(parents=True, exist_ok=True)
    (home / "runtime" / old_file).write_text("old\n", encoding="utf-8", newline="")
    (home / "runtime" / "MANIFEST").write_text(old_file + "\n", encoding="utf-8", newline="")
    # The explicit DIRECTORY member is what makes this a type conflict, and it
    # is named "docs/user guide/" — with a trailing slash a manifest entry never
    # carries. Without it the archive holds only the leaf, whose parent is not a
    # directory on disk, so `_type_conflicts` finds nothing and this test would
    # pass without ever reaching the code it is named after.
    _park(home, _package([new_file], **{old_file + "/": "", new_file: "new\n"}))

    result = au.apply(home)
    assert result["applied"] is True, result["error"]
    assert (home / "runtime" / new_file).is_file()


def test_a_conflict_the_prune_will_not_clear_is_refused_up_front(home):
    """The other half: the check has to actually refuse something.

    `extractall` raises a bare OSError part-way through a file/directory
    collision, having already overwritten an arbitrary prefix of the tree.
    Naming it before the marker is written is what keeps that from happening,
    so a test that only proves clearable conflicts pass would be satisfied by
    removing the check altogether.
    """
    clash = "backend/api/main.py"          # a FILE on disk and in both manifests
    (home / "runtime" / "MANIFEST").write_text(clash + "\n", encoding="utf-8", newline="")
    _park(home, _package([clash], **{clash + "/": ""}))   # a DIRECTORY in the release

    result = au.apply(home)
    assert result["applied"] is False
    assert clash in result["error"]
    assert result["terminal"] is True
    # and nothing was touched on the way to finding out
    assert (home / "runtime" / "VERSION").read_text().strip() == "v0.4.0"
    assert (home / "runtime" / clash).is_file()


def test_type_conflicts_returns_pairs_not_sentences(home, tmp_path):
    archive = tmp_path / "conflict.zip"
    archive.write_bytes(_zip_bytes(_package(["a b/c.txt"], **{"a b/c.txt": "x\n"})))
    # "a b/c.txt" is a DIRECTORY on disk, so it is a type conflict.
    (home / "runtime" / "a b" / "c.txt").mkdir(parents=True)

    conflicts = au._type_conflicts(archive, home / "runtime")
    assert conflicts, "the fixture should produce a conflict"
    for entry in conflicts:
        assert isinstance(entry, tuple) and len(entry) == 2
    assert any(name == "a b/c.txt" for name, _ in conflicts)


# --------------------------------------------------------------------------
# discarding must never be the thing that crashes
# --------------------------------------------------------------------------

def test_discarding_never_raises_even_when_it_cannot(home, monkeypatch):
    """`apply` promises it never raises, and `_discard_pending` runs inside its
    failure handlers — so a refusal here would leave the shell with an
    unhandled traceback and NO verdict, unable to say whether it may start."""
    _park(home, _package(["backend/api/main.py"], **{"backend/api/main.py": "new\n"}))

    def refuse(target):
        raise ValueError(f"{target} is a reparse point")

    monkeypatch.setattr(au, "assert_no_reparse_point", refuse)
    au._discard_pending(home)   # must not raise


def test_a_refusal_to_discard_still_produces_a_verdict(home, monkeypatch):
    """The whole point: the shell always gets a JSON answer."""
    _park(home, _package([], ))
    (home / "pending" / "orienta-runtime-v0.4.1.zip").unlink()

    monkeypatch.setattr(au, "assert_no_reparse_point",
                        lambda target: (_ for _ in ()).throw(ValueError("junction")))
    result = au.apply(home)
    assert result["applied"] is False
    assert isinstance(result["error"], str) and result["error"]


# --------------------------------------------------------------------------
# the first-install path
# --------------------------------------------------------------------------

def test_extract_only_refuses_a_destination_that_is_not_the_runtime(home):
    """protected() tests member NAMES; without this the guarantee "nothing ever
    writes under Database" is a property of one call site."""
    pkg = home / "pending" / "p.zip"
    pkg.parent.mkdir(exist_ok=True)
    pkg.write_bytes(_zip_bytes(_package(["x"], **{"x": "y\n"})))

    bad = au.extract_only(pkg, home / "runtime" / "Database", home)
    assert bad["ok"] is False and "refusing" in bad["error"]
    assert library(home).read_text() == "irreplaceable"

    assert au.extract_only(pkg, home / "runtime", home)["ok"] is True


def test_extract_only_applies_the_same_member_checks_as_an_update(home):
    pkg = home / "pending" / "p.zip"
    pkg.parent.mkdir(exist_ok=True)
    pkg.write_bytes(_zip_bytes(_package(["x"], **{"../evil.txt": "no\n"})))
    assert au.extract_only(pkg, home / "runtime", home)["ok"] is False
    assert not (home / "evil.txt").exists()


# --------------------------------------------------------------------------
# the CLI the shell drives
# --------------------------------------------------------------------------

def test_the_cli_writes_its_verdict_to_a_file(home, tmp_path):
    """stdout is not a reliable channel: on Windows `print` encodes with the
    locale code page, so one path under C:\\Users\\Müller\\ kills the process
    with UnicodeEncodeError and no output at all."""
    _park(home, _package(["backend/api/main.py"], **{"backend/api/main.py": "new\n"}))
    verdict = tmp_path / "verdict.json"
    code = au.main(["--home", str(home), "--result-json", str(verdict)])
    assert code == 0
    assert json.loads(verdict.read_text(encoding="utf-8"))["applied"] is True


def test_the_exit_code_never_contradicts_the_verdict(home, monkeypatch):
    """0 whenever the shell may proceed; 1 ONLY when the runtime is incomplete."""
    _park(home, _package(["backend/api/main.py"], **{"backend/api/main.py": "new\n"}))
    monkeypatch.setattr(au, "safe_extract",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("killed")))
    assert au.main(["--home", str(home)]) == 1        # dirty
    monkeypatch.undo()
    assert au.main(["--home", str(home)]) == 0        # repaired
    assert au.main(["--home", str(home)]) == 0        # nothing to do
