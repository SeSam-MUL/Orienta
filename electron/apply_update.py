"""Apply a parked runtime package, before the backend starts.

This file lives in the Electron package, not in `runtime/`: it is the code that
rewrites `runtime/`, so it must not be inside what it rewrites. It is run by the
interpreter the setup installed, under the Orienta home directory.

The operation is deliberately boring, and the ORDER is the design:

  1. read `pending.json`; nothing there -> nothing to do
  2. verify the parked zip against the digest recorded when it was downloaded
  3. refuse the whole package if any member would escape `runtime/` or land
     under `Database/`
  4. mark the tree as being worked on
  5. **prune first** -- delete paths the OLD manifest listed and the new one
     does not
  6. extract over `runtime/`
  7. unmark, and remove `pending.json`

**Prune before extract, not after.** Extracting writes the package's own
MANIFEST over `runtime/MANIFEST`. With extract-first, a run interrupted after
the extraction can never prune: the difference it computes next time is empty,
and every file that release removes stays for ever. With prune-first the prune
has always completed before any MANIFEST is overwritten, so the only
interruptible window leaves the OLD manifest in place and the next run
recomputes the same candidates.

Re-applying a package that has already been applied cleanly prunes nothing.
That is correct — the old manifest legitimately IS the new one — and any test
asserting otherwise is wrong about the design rather than about the code.

Three outcomes, which the shell must treat differently:

  applied           done
  terminal failure  the parked package is worthless (missing, corrupt,
                    refused). It is DISCARDED here, so the shell reports it
                    once and starts normally. Keeping it produced a modal error
                    box on every launch that nothing in the product could clear.
  dirty failure     pruning or extraction had begun. `pending.json` is kept so
                    the next start repairs it, and the shell must NOT start the
                    backend: uvicorn would import a tree that is half one
                    release and half the other.

It is idempotent: running it twice is running it once, and a process killed
anywhere in steps 4-7 is repaired by running it again. That property is why this
design replaced three generations of directory swapping, each of which found a
new way to destroy the user's crystal library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

PROTECTED_DIR = "database"          # compared case-folded; see `protected`
MARKER = ".applying"
CHUNK = 1 << 20


def _normalise_component(part: str) -> str:
    """One path component as WINDOWS will resolve it.

    Win32 is case-insensitive and strips trailing dots and spaces, so
    `database`, `DATABASE` and `Database.` all open the user's crystal library
    while a literal `== "Database"` lets every one of them through. This one
    function is what the whole Database rule rests on.
    """
    return part.rstrip(" .").casefold()


def names_marker(rel: str) -> bool:
    """Does this path name the in-progress marker, as WINDOWS will open it?

    Raw equality against ".applying" is not enough, for the same reason
    `protected` cannot use it: Win32 strips trailing dots and spaces, so a
    manifest entry ".applying " compares unequal to the marker and then opens
    it. Deleting the marker mid-prune un-marks a dirty run, and the NEXT
    failure is then classified terminal -- which discards the package and tells
    the shell it may start against a half-updated tree.
    """
    parts = PurePosixPath(rel.replace("\\", "/")).parts
    return any(_normalise_component(part) == MARKER for part in parts)


def protected(rel) -> bool:
    """Is this repo-relative path off limits, at any depth?"""
    parts = PurePosixPath(str(rel).replace("\\", "/")).parts
    return any(_normalise_component(part) == PROTECTED_DIR for part in parts)


def _lstat_or_none(target: Path):
    """None means absent. Any other error is raised: "cannot tell" is not
    "nothing here", and this guards a delete."""
    try:
        return target.lstat()
    except FileNotFoundError:
        return None


def assert_no_reparse_point(path: Path) -> None:
    """Refuse to delete through a junction or symlink."""
    stat = _lstat_or_none(path)
    if stat is None:
        return
    if path.is_symlink() or (os.name == "nt" and bool(getattr(stat, "st_reparse_tag", 0))):
        raise ValueError(f"{path} is a junction or symlink; refusing to delete through it")


def assert_inside(path: Path, root: Path) -> None:
    """Refuse a path that resolves outside `root`.

    Checking the victim alone is not enough: a user short of space on C: does
    `mklink /J runtime/frontend E:/orienta-frontend`, and then
    `frontend/dist/old.js` is a real file with no reparse tag of its own,
    sitting outside the runtime. Resolving catches the ancestor.
    """
    resolved_root = root.resolve()
    if not path.resolve().is_relative_to(resolved_root):
        raise ValueError(f"{path} resolves outside {resolved_root}; refusing to touch it")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_members(archive: Path, target: Path) -> None:
    """Refuse the whole package if any member escapes or is protected.

    The protected test is applied to the RESOLVED path as well as the member
    name. Win32 expands 8.3 short names during resolution, so `DATABA~1/x.cif`
    passes a name test and lands in the crystal library; the same is true of any
    junction the user has made inside the runtime. Judge where the write goes.
    """
    resolved = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            parts = PurePosixPath(member.replace("\\", "/")).parts
            if protected(member) or names_marker(member):
                raise ValueError(
                    f"package member {member!r} would write into a protected location")
            destination = (resolved / member).resolve()
            # is_relative_to, not str.startswith: a prefix test passes
            # '../runtime-evil/x'.
            if not destination.is_relative_to(resolved):
                raise ValueError(f"package member {member!r} escapes the runtime")
            if protected(destination.relative_to(resolved)):
                raise ValueError(
                    f"package member {member!r} resolves into a protected directory")


def _type_conflicts(archive: Path, target: Path) -> list[tuple[str, str]]:
    """Members whose kind (file vs directory) disagrees with what is on disk.

    `extractall` raises a bare OSError on these, part-way through. Naming them
    up front turns an unrecoverable loop into a sentence.

    Returns (name, description) PAIRS rather than one joined string: the caller
    has to match the name against the prune set, and recovering it by splitting
    the sentence on its first space silently fails for any packaged path
    containing one -- turning a conflict the prune would have cleared into a
    terminal refusal that discards the update.
    """
    conflicts = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            existing = target / info.filename
            if _lstat_or_none(existing) is None:
                continue
            if info.is_dir() and not existing.is_dir():
                conflicts.append((info.filename, "a directory in the release, a file here"))
            elif not info.is_dir() and existing.is_dir():
                conflicts.append((info.filename, "a file in the release, a directory here"))
    return conflicts


def safe_extract(archive: Path, target: Path) -> None:
    _check_members(archive, target)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)


class TransientError(Exception):
    """Something could not be read NOW, and may well be readable in a second.

    Separated from a permanent refusal because the two want opposite handling:
    a refusal means the package is bad and should be discarded, while losing a
    race with antivirus or a backup agent means the package is FINE and the
    next launch should simply try again. Collapsing them deletes a good
    download over a momentary read lock.
    """


def _read_manifest(path: Path) -> list[str]:
    """The previous package's file list.

    Absent is legitimate (a runtime the setup unpacked has no predecessor) and
    means "nothing to prune". **Unreadable is not the same answer**: silently
    returning [] would make the prune a permanent no-op, which is the defect the
    ordering above exists to prevent, reached by another route.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except (OSError, UnicodeDecodeError) as exc:
        raise TransientError(
            f"{path} exists but cannot be read ({exc}); another program may be "
            f"holding it open") from exc
    return [line.strip() for line in text.splitlines() if line.strip()]


def prune_candidates(old_manifest: list[str], new_manifest: list[str]) -> list[str]:
    """What `prune` would consider, with the never-touchables already removed."""
    return [
        rel for rel in sorted(set(old_manifest) - set(new_manifest))
        if not protected(rel) and not names_marker(rel)
    ]


def dry_run_prune(runtime: Path, old_manifest: list[str], new_manifest: list[str]) -> None:
    """Raise NOW for anything that would make `prune` fail later.

    Every refusal must happen BEFORE the marker is written. Leaving the junction
    and directory checks inside the prune meant a user with a junction in the
    runtime got: files deleted, then a refusal, then a dialog whose options
    repeat the identical failure or re-enter it on the next launch.
    """
    for rel in prune_candidates(old_manifest, new_manifest):
        victim = runtime / rel
        assert_no_reparse_point(victim)
        if _lstat_or_none(victim) is None:
            continue
        assert_inside(victim, runtime)


def prune(runtime: Path, old_manifest: list[str], new_manifest: list[str]) -> list[str]:
    """Delete what the old package shipped and the new one does not.

    Only paths the OLD manifest listed are candidates — a file the user put
    there is never one. Protected paths are skipped whatever the manifest says:
    a package cannot talk this function into deleting a Database by listing one.
    """
    removed = []
    for rel in prune_candidates(old_manifest, new_manifest):
        victim = runtime / rel
        assert_no_reparse_point(victim)
        if _lstat_or_none(victim) is None:
            continue
        assert_inside(victim, runtime)
        # Judge the RESOLVED path too: a junction inside the runtime, an 8.3
        # short name, or an alternate data stream all pass a test on the
        # manifest string. Candidates come from the PREVIOUS package's manifest,
        # so these strings are package-controlled.
        if protected(victim.resolve().relative_to(runtime.resolve())):
            continue
        if victim.is_dir():
            # A manifest lists files only. Leave it rather than removing a tree
            # nobody asked to remove.
            continue
        victim.unlink()
        removed.append(rel)
    return removed


def _discard_pending(home: Path) -> None:
    """Throw away a worthless parked package so it cannot nag for ever."""
    try:
        (home / "pending.json").unlink(missing_ok=True)
        pending = home / "pending"
        if not pending.exists():
            return
        # EVERY level, not just the first two. `shutil.rmtree` does not treat a
        # junction specially on the Python 3.11 the setup installs
        # (`os.path.islink` is False for one, and `DirEntry.is_junction` arrives
        # in 3.12), so a junction at depth 2 is recursed into and its TARGET's
        # contents are deleted -- silently, because ignore_errors swallows
        # whatever it meets on the way.
        for current, dirnames, filenames in os.walk(pending):
            here = Path(current)
            assert_no_reparse_point(here)
            for name in list(dirnames) + list(filenames):
                assert_no_reparse_point(here / name)
        shutil.rmtree(pending, ignore_errors=True)
    except Exception:  # noqa: BLE001
        # This runs inside the failure handlers, and `apply`'s docstring
        # promises it never raises. A junction planted at <home>/pending would
        # otherwise carry a ValueError out of apply(), out of main(), and reach
        # the shell as an unhandled traceback with NO verdict written -- so the
        # shell could not tell whether the runtime was safe to start. A discard
        # that fails costs disk space; a crash here costs the launch.
        pass


def _result(applied=False, tag=None, error=None, dirty=False, terminal=False) -> dict:
    return {"applied": applied, "tag": tag, "error": error,
            "dirty": dirty, "terminal": terminal}


def apply(home: Path) -> dict:
    """Apply the parked package, if there is one. Never raises."""
    runtime = home / "runtime"
    pending_json = home / "pending.json"
    marker = runtime / MARKER

    try:
        record = json.loads(pending_json.read_text(encoding="utf-8"))
    except FileNotFoundError:
        if _lstat_or_none(marker) is not None:
            return _result(
                error="a previous update was interrupted and there is nothing left to "
                      "finish it with. Reinstall Orienta to repair the installation — "
                      "your data and crystal library are not touched.",
                dirty=True)
        return _result()
    except (OSError, ValueError) as exc:
        _discard_pending(home)
        return _result(error=f"the pending-update record was damaged and has been "
                             f"discarded ({exc}).", terminal=True)

    tag = str(record.get("tag") or "")
    archive = home / "pending" / str(record.get("file") or "")

    # An earlier attempt got as far as mutating the tree. Whatever happens
    # below, this run may NOT be classified terminal: discarding the package
    # would leave a half-applied runtime with nothing to complete it, and the
    # shell would then start the backend against it.
    already_dirty = _lstat_or_none(marker) is not None

    def fail(exc) -> dict:
        if already_dirty:
            return _result(tag=tag or None, error=str(exc), dirty=True)
        _discard_pending(home)
        return _result(tag=tag or None,
                       error=f"{exc}. The update was discarded; Orienta will start normally.",
                       terminal=True)

    # --- EVERYTHING that can refuse refuses here, before anything is touched.
    # Nothing below may raise for a reason that will still be true next time, or
    # the product becomes unlaunchable with no way out.
    try:
        if not archive.is_file():
            raise ValueError(f"the parked package {archive.name} is missing")
        if sha256_of(archive).lower() != str(record.get("sha256") or "").lower():
            raise ValueError("the parked package is corrupt (checksum mismatch)")
        with zipfile.ZipFile(archive) as zf:
            if "MANIFEST" not in set(zf.namelist()):
                raise ValueError("the parked package has no MANIFEST")
            new_manifest = [line.strip() for line
                            in zf.read("MANIFEST").decode("utf-8").splitlines() if line.strip()]
        _check_members(archive, runtime)
        old_manifest = _read_manifest(runtime / "MANIFEST")
        dry_run_prune(runtime, old_manifest, new_manifest)

        # A conflict the prune will clear is not a conflict.
        clearable = set(old_manifest) - set(new_manifest)
        # rstrip("/"), because a zip DIRECTORY member is named with a trailing
        # slash while a manifest entry never has one. The directory-over-file
        # case is the only one the prune can actually clear, so without this the
        # match never fires for the case it exists to serve.
        surviving = [(name, why) for name, why in _type_conflicts(archive, runtime)
                     if name.rstrip("/") not in clearable]
        if surviving:
            raise ValueError("this release changes " + "; ".join(
                f"{name} ({why})" for name, why in surviving[:3]))
    except TransientError as exc:
        # Keep the package. Nothing has been touched yet, and the condition is
        # the kind that clears by itself.
        return _result(tag=tag or None,
                       error=f"{exc}. Orienta will try again next time it starts.")
    except Exception as exc:  # noqa: BLE001
        return fail(exc)

    # --- from here only genuine I/O can fail, and the repair is to repeat
    try:
        runtime.mkdir(parents=True, exist_ok=True)
        marker.write_text(tag + "\n", encoding="utf-8", newline="")
        prune(runtime, old_manifest, new_manifest)
        safe_extract(archive, runtime)
    except Exception as exc:  # noqa: BLE001
        return _result(tag=tag or None, error=str(exc), dirty=True)

    marker.unlink(missing_ok=True)
    pending_json.unlink(missing_ok=True)
    return _result(applied=True, tag=tag)


def extract_only(archive: Path, into: Path, home: Path) -> dict:
    """The FIRST-INSTALL path: unpack a package into a fresh runtime.

    Separate from `apply` because there is no old manifest and nothing to
    prune, but it goes through the SAME member checks — the setup and the
    updater must not hold two different ideas of what a package may contain.

    `into` is validated rather than trusted. `protected()` tests member NAMES
    and would pass every one of them into a destination that IS the crystal
    library, so without this line "nothing ever writes under Database" is a
    property of one call site rather than of this module.
    """
    expected = (home / "runtime").resolve()
    try:
        if into.resolve() != expected:
            return {"ok": False, "error": f"refusing to extract into {into}; expected {expected}"}
        safe_extract(archive, into)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "error": None}


def _emit(result: dict, result_json: str | None) -> None:
    r"""Write the verdict where the caller asked, in UTF-8.

    stdout is not a reliable channel here: on Windows `print` encodes with the
    locale code page, so one path under `C:\Users\Müller\` kills the process
    with UnicodeEncodeError and no output at all — which the shell then has to
    guess about. A freshly installed CPython may also put a warning on the same
    stream.
    """
    blob = json.dumps(result, ensure_ascii=False)
    if result_json:
        Path(result_json).write_text(blob, encoding="utf-8", newline="")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - older/odd streams
        pass
    print(blob)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--home", required=True)
    ap.add_argument("--result-json", help="write the verdict here as well as to stdout")
    ap.add_argument("--extract", help="a package to unpack (first install) instead of applying")
    ap.add_argument("--into", help="destination for --extract; must be <home>/runtime")
    args = ap.parse_args(argv)
    home = Path(args.home)

    if args.extract:
        if not args.into:
            ap.error("--extract requires --into")
        result = extract_only(Path(args.extract), Path(args.into), home)
        _emit(result, args.result_json)
        return 0 if result["ok"] else 1

    result = apply(home)
    _emit(result, args.result_json)
    # Exit 0 whenever the shell may proceed — applied, nothing to do, or a
    # terminal failure whose package has been discarded. Exit 1 ONLY when the
    # runtime is incomplete. The caller reads the verdict, not the code, but the
    # code must not contradict it.
    return 1 if result["dirty"] else 0


if __name__ == "__main__":
    sys.exit(main())
