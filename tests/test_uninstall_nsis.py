"""The uninstaller's data step, run for real.

Each test compiles a tiny installer that includes electron/nsis/uninstall.nsh
UNCHANGED, runs it silently against a fake data folder, and checks what is left.
The answers to the two questions are fixed at compile time; that, and the
redirected profile folders, are the only differences from the shipped
uninstaller.

Why this module is long: an UPDATE runs the uninstaller of the OLD version, so
whatever ships in uninstall.nsh keeps running for that version for as long as
anyone has it installed. A flaw found after release cannot be fixed by the next
release. Every case below is either a measurement the file relies on or a
finding from the review of its first version.

  * NSIS 3.0.4's `RMDir /r` FOLLOWS a junction and deletes what is behind it.
  * `rd` on an enumerated `Database.` deleted the REAL `Database` beside it.
  * A linked `runtime\\` was walked into and emptied.
  * One marker (.python_path) was enough to call a folder Orienta's; a
    repository root has one.
  * An unreachable folder deleted the only record of where it was.

Both profile folders are redirected at compile time, the bench runs in a
temporary working directory, and the REAL %LOCALAPPDATA%\\Orienta and
%APPDATA%\\Orienta are watched for deletions around the whole module.

Skipped where the NSIS compiler electron-builder downloads is not present.
"""
from __future__ import annotations

import hashlib
import json
import os
import string
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NSH = REPO / "electron" / "nsis" / "uninstall.nsh"
# electron/setup/package_sync.js: what the packages were brought to, an update in
# flight, the last failure of one. Written at the top of the home by the shell.
PACKAGE_SYNC_FILES = (".packages_lock.json", ".packages_sync.json", ".packages_sync_failed.json")
LONG = "\\\\?\\"


def _makensis() -> Path | None:
    cache = Path(os.environ.get("LOCALAPPDATA", "")) / "electron-builder" / "Cache"
    found = sorted(cache.glob("nsis-*/*/Bin/makensis.exe"))
    return found[-1] if found else None


MAKENSIS = _makensis()

pytestmark = [
    pytest.mark.skipif(os.name != "nt", reason="NSIS and junctions are Windows-only"),
    pytest.mark.skipif(MAKENSIS is None, reason="electron-builder's makensis is not cached here"),
]


# --------------------------------------------------------------------------
# the real folders must not lose anything
# --------------------------------------------------------------------------

def _what_must_not_disappear(root: Path) -> dict[str, str]:
    """Every path under python\\ and runtime\\, the top level, and the library
    byte for byte.

    Not a fingerprint of everything: an installed Orienta running on the same
    machine rewrites its logs and browser storage continuously, and a guard
    that fires on every write cannot be kept green. The risk here is DELETION.
    """
    if not root.exists():
        return {}
    seen = {f"top:{p.name}": "" for p in root.iterdir()}
    for sub in ("python", "runtime"):
        base = root / sub
        if base.is_dir() and not base.is_symlink():
            for p in base.rglob("*"):
                seen[f"{sub}:{p.relative_to(base)}"] = ""
    library = root / "runtime" / "Database"
    if library.is_dir():
        for p in library.rglob("*"):
            if p.is_file():
                seen[f"lib:{p.relative_to(library)}"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return seen


@pytest.fixture(scope="module", autouse=True)
def real_folders_untouched():
    real = [Path(os.environ["LOCALAPPDATA"]) / "Orienta", Path(os.environ["APPDATA"]) / "Orienta"]
    before = [_what_must_not_disappear(p) for p in real]
    yield
    after = [_what_must_not_disappear(p) for p in real]
    for folder, was, now in zip(real, before, after):
        missing = sorted(set(was) - set(now))[:10]
        changed = sorted(k for k in was if k.startswith("lib:") and k in now and was[k] != now[k])
        assert not missing and not changed, (
            f"{folder} lost {missing} / library files changed {changed} during the run")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _junction(link: Path, target: Path) -> None:
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                   check=True, capture_output=True)


def _home(home: Path, *, marker=True, legacy=True, user_file=True, lib_link=None) -> None:
    """A data folder shaped like a real one.

    marker: the .orienta-home file setup writes from now on.
    legacy: runtime\\VERSION + python\\python.exe + .python_path, the markers an
            install from before the marker existed has.
    """
    home.mkdir(parents=True, exist_ok=True)
    (home / "python" / "Lib").mkdir(parents=True)
    (home / "runtime").mkdir()
    if marker:
        (home / ".orienta-home").write_text("x")
    if legacy:
        (home / "python" / "python.exe").write_text("exe")
        (home / "runtime" / "VERSION").write_text("v0.4.0\n")
        (home / ".python_path").write_text("x")
    (home / "python" / "Lib" / "os.py").write_text("x")
    (home / "python.old-123").mkdir()
    (home / "python.old-123" / "x.txt").write_text("x")
    (home / "electron" / "userData").mkdir(parents=True)
    (home / "electron" / "userData" / "prefs").write_text("x")
    (home / "logs").mkdir()
    (home / "logs" / "orienta-setup.log").write_text("x")
    (home / "setup-tmp").mkdir()
    (home / "pending").mkdir()
    (home / "pending.json").write_text("{}")
    (home / ".install_mode").write_text("cpu")
    (home / ".install_incomplete").write_text("{}")
    for name in PACKAGE_SYNC_FILES:
        (home / name).write_text("{}")
    (home / "runtime" / "backend" / "api").mkdir(parents=True)
    (home / "runtime" / "backend" / "api" / "main.py").write_text("x")
    (home / "runtime" / "MANIFEST").write_text("x")
    if lib_link is None:
        (home / "runtime" / "Database" / "CIF_Library").mkdir(parents=True)
        (home / "runtime" / "Database" / "CIF_Library" / "mine.cif").write_text("irreplaceable")
    else:
        _junction(home / "runtime" / "Database", lib_link)
    if user_file:
        (home / "notes_the_user_put_here.txt").write_text("not ours")


def _library(home: Path) -> Path:
    return home / "runtime" / "Database" / "CIF_Library" / "mine.cif"


def _pointer(appdata: Path, text: str, *, utf16=True) -> Path:
    (appdata / "Orienta").mkdir(parents=True, exist_ok=True)
    file = appdata / "Orienta" / "home.txt"
    if utf16:
        file.write_bytes(b"\xff\xfe" + f"{text}\r\n".encode("utf-16-le"))
    else:
        file.write_text(text + "\n", encoding="utf-8")
    return file


@pytest.fixture
def box(tmp_path):
    (tmp_path / "appdata").mkdir()
    (tmp_path / "local").mkdir()
    victim = tmp_path / "outside" / "victim"
    victim.mkdir(parents=True)
    (victim / "keep.txt").write_text("behind a link")
    return tmp_path


def _run(box: Path, *, answers=None, env_home=None, language=None,
         via_hook=None, install_mode="CurrentUser") -> dict:
    """Compile and run. `via_hook` = True/False inserts customUnInstall with
    ${isUpdated} forced to that value; None inserts orientaRemoveData directly.
    Returns what the step decided, from its test-only dump."""
    dump = box / "decided.txt"
    defines = [f"/DORIENTA_TEST_APPDATA={box / 'appdata'}",
               f"/DORIENTA_TEST_LOCALAPPDATA={box / 'local'}",
               f"/DORIENTA_TEST_DUMP={dump}"]
    if answers is not None:
        defines += [f"/DORIENTA_TEST_ANSWER_COMPONENTS={answers[0]}",
                    f"/DORIENTA_TEST_ANSWER_LIBRARY={answers[1]}"]
    lines = ["Unicode true", "Var installMode"]
    if via_hook is not None:
        lines.append(f'!define isUpdated "1 == {1 if via_hook else 0}"')
    lines += [f'!include "{NSH}"', f'OutFile "{box / "bench.exe"}"',
              "RequestExecutionLevel user", "SilentInstall silent", "Section",
              f'  StrCpy $installMode "{install_mode}"']
    if language is not None:
        lines.append(f"  StrCpy $LANGUAGE {language}")
    lines.append("  !insertmacro customUnInstall" if via_hook is not None
                 else "  !insertmacro orientaRemoveData")
    lines.append("SectionEnd")
    script = box / "bench.nsi"
    script.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    build = subprocess.run([str(MAKENSIS), "/V1", "/WX", *defines, str(script)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert build.returncode == 0, build.stdout[-3000:] + build.stderr[-2000:]
    env = dict(os.environ)
    env.pop("ORIENTA_HOME", None)
    if env_home is not None:
        env["ORIENTA_HOME"] = str(env_home)
    # A temporary working directory: a relative path must never resolve
    # against the repository.
    work = box / "cwd"
    work.mkdir(exist_ok=True)
    subprocess.run([str(box / "bench.exe")], env=env, cwd=work, check=True, timeout=120)

    if not dump.exists():
        return {}
    raw = dump.read_bytes().decode("utf-16")
    decided = {"raw": raw}
    # The short fields are one line each. The two sentences contain line
    # breaks of their own, so they are cut out by position, not by line.
    head, _, sentences = raw.partition("ask=")
    for line in head.splitlines():
        key, _, value = line.partition("=")
        decided[key] = value
    decided["ask"], _, decided["lib"] = sentences.partition("\r\nlib=")
    return decided


# --------------------------------------------------------------------------
# the ordinary cases
# --------------------------------------------------------------------------

def test_no_to_everything_moves_nothing(box):
    home = box / "home"
    _home(home)
    _run(box, answers=("no", "no"), env_home=home)
    assert (home / "python" / "python.exe").exists()
    assert (home / "runtime" / "VERSION").exists()
    assert _library(home).exists()


def test_components_go_the_library_and_the_users_own_file_stay(box):
    home = box / "home"
    _home(home)
    _junction(home / "python" / "Lib" / "linkout", box / "outside" / "victim")
    decided = _run(box, answers=("yes", "no"), env_home=home)
    assert decided["verdict"] == "ours"
    for gone in ("python", "python.old-123", "electron", "logs", "setup-tmp", "pending",
                 "pending.json", ".python_path", ".install_mode", ".install_incomplete",
                 *PACKAGE_SYNC_FILES):
        assert not (home / gone).exists(), gone
    assert sorted(p.name for p in (home / "runtime").iterdir()) == ["Database"]
    assert _library(home).read_text() == "irreplaceable"
    assert (home / "notes_the_user_put_here.txt").exists()
    assert (box / "outside" / "victim" / "keep.txt").exists()
    # With the library kept, the folder must still be recognisable next time.
    assert (home / ".orienta-home").exists()
    # A clean run reports nothing left behind. Without this, a broken
    # "does it exist?" test reading every missing path as a file would show
    # "could not remove" on every uninstall and still pass every other check.
    assert decided["leftBehind"] == "no"


def test_both_yes_removes_the_folder_entirely_marker_included(box):
    home = box / "home"
    _home(home, user_file=False)
    decided = _run(box, answers=("yes", "yes"), env_home=home)
    assert not home.exists()
    assert decided["leftBehind"] == "no"


def test_a_percent_sign_is_left_alone_and_the_rest_still_cleaned(box):
    """cmd.exe expands %VAR% even inside quotes, which would hand `rd` a
    different folder — here %TEMP%, the system's temporary directory."""
    home = box / "home"
    _home(home, user_file=False)
    odd = home / "runtime" / "odd%TEMP%dir"
    odd.mkdir()
    (odd / "x.txt").write_text("x")
    decided = _run(box, answers=("yes", "no"), env_home=home)
    assert (odd / "x.txt").exists()
    assert decided["leftBehind"] == "yes"
    assert not (home / "python").exists()
    assert Path(os.environ["TEMP"]).exists()


def test_an_install_from_before_the_marker_is_still_recognised(box):
    home = box / "home"
    _home(home, marker=False)
    decided = _run(box, answers=("yes", "no"), env_home=home)
    assert decided["verdict"] == "ours"
    assert not (home / "python").exists()


def test_a_silent_uninstall_keeps_everything(box):
    """No answers compiled in: the dialogs' /SD defaults decide, both No."""
    home = box / "home"
    _home(home)
    _run(box, env_home=home)
    assert (home / "python" / "python.exe").exists()
    assert _library(home).exists()


def test_the_default_location_is_used_when_nothing_says_otherwise(box):
    home = box / "local" / "Orienta"
    _home(home, user_file=False)
    _run(box, answers=("yes", "yes"))
    assert not home.exists()


# --------------------------------------------------------------------------
# links, on every level
# --------------------------------------------------------------------------

def test_a_linked_library_loses_only_its_link_and_says_so(box):
    library = box / "outside" / "library"
    (library / "CIF_Library").mkdir(parents=True)
    (library / "CIF_Library" / "mine.cif").write_text("irreplaceable")
    home = box / "home"
    _home(home, user_file=False, lib_link=library)
    decided = _run(box, answers=("yes", "yes"), env_home=home)
    assert (library / "CIF_Library" / "mine.cif").read_text() == "irreplaceable"
    assert not (home / "runtime" / "Database").exists()
    # The LINK branch, not `rd` happening to leave the target alone too.
    assert decided["libWasLink"] == "yes"


def test_a_linked_runtime_is_not_walked_into(box):
    """Review finding: `runtime` as a junction to a developer checkout was
    enumerated and everything but `Database` deleted in the CHECKOUT."""
    checkout = box / "outside" / "checkout"
    for d in ("backend", ".git", "electron", "Database"):
        (checkout / d).mkdir(parents=True)
        (checkout / d / "keep.txt").write_text("the developer's work")
    home = box / "home"
    home.mkdir()
    (home / ".orienta-home").write_text("x")
    (home / "python").mkdir()
    _junction(home / "runtime", checkout)
    decided = _run(box, answers=("yes", "yes"), env_home=home)
    for d in ("backend", ".git", "electron", "Database"):
        assert (checkout / d / "keep.txt").exists(), d
    assert not (home / "runtime").exists()          # only the link went
    # The user asked for the library to go, and it did not: they are told.
    assert decided["libWasLink"] == "yes"


def test_a_symlinked_runtime_is_not_walked_into_either(box):
    """The symlink reparse tag (0xA000000C), not only the junction tag."""
    checkout = box / "outside" / "checkout"
    (checkout / "backend").mkdir(parents=True)
    (checkout / "backend" / "keep.txt").write_text("x")
    home = box / "home"
    home.mkdir()
    (home / ".orienta-home").write_text("x")
    try:
        os.symlink(checkout, home / "runtime", target_is_directory=True)
    except OSError:
        pytest.skip("creating a directory symlink needs Developer Mode or elevation")
    _run(box, answers=("yes", "yes"), env_home=home)
    assert (checkout / "backend" / "keep.txt").exists()


def test_a_linked_data_folder_keeps_its_link_and_its_library(box):
    """Review finding: removing the link of a linked data folder orphaned a
    kept library and silenced the message saying where it was."""
    real = box / "outside" / "realdata"
    _home(real, user_file=False)
    home = box / "home"
    _junction(home, real)
    _run(box, answers=("yes", "no"), env_home=home)
    assert home.exists(), "the link to the data folder was removed"
    assert _library(real).read_text() == "irreplaceable"
    assert not (real / "python").exists()


def test_a_trailing_dot_name_cannot_reach_the_library(box):
    """MEASURED: `rd` on an enumerated `Database.` deleted the real
    `Database` beside it, because Win32 strips the trailing dot."""
    home = box / "home"
    _home(home, user_file=False)
    os.mkdir(LONG + str(home / "runtime" / "Database."))
    decided = _run(box, answers=("yes", "no"), env_home=home)
    assert _library(home).read_text() == "irreplaceable"
    assert decided["leftBehind"] == "yes"


# --------------------------------------------------------------------------
# recognition
# --------------------------------------------------------------------------

def test_one_marker_is_not_enough(box):
    """Review finding: a repository root can have a .python_path."""
    home = box / "home"
    _home(home, marker=False, legacy=False)
    (home / ".python_path").write_text("x")
    decided = _run(box, answers=("yes", "yes"), env_home=home)
    assert decided["verdict"] == "unsure"
    assert (home / "python" / "Lib" / "os.py").exists()


def test_a_repository_is_never_a_data_folder(box):
    home = box / "home"
    _home(home)
    (home / ".git").mkdir()
    decided = _run(box, answers=("yes", "yes"), env_home=home)
    assert decided["verdict"] == "unsure"
    assert _library(home).exists() and (home / "python").exists()


def test_a_drive_root_is_refused_after_normalisation(box):
    """`X:\\.` passed a length check before normalisation. A real drive root
    with a complete data folder in it, made with `subst` so nothing real is at
    risk if the guard fails."""
    root = box / "fake_drive"
    _home(root, user_file=False)
    # Tried rather than predicted: a mapped network drive can hold a letter
    # that no listing of local disks shows.
    free = None
    for letter in reversed(string.ascii_uppercase[8:]):
        if os.path.exists(f"{letter}:\\"):
            continue
        if subprocess.run(["subst", f"{letter}:", str(root)], capture_output=True).returncode == 0:
            free = letter
            break
    if free is None:
        pytest.skip("no free drive letter for subst")
    try:
        decided = _run(box, answers=("yes", "yes"), env_home=f"{free}:\\.")
    finally:
        subprocess.run(["subst", f"{free}:", "/d"])
    assert decided["verdict"] == "unsure"
    assert _library(root).exists() and (root / "python").exists()


def test_a_long_path_prefix_is_refused(box):
    """GetFullPathNameW leaves \\\\?\\ paths untouched, so `\\\\?\\C:\\.` would
    reach the root checks un-normalised."""
    home = box / "home"
    _home(home)
    decided = _run(box, answers=("yes", "yes"), env_home=LONG + str(home))
    assert decided["verdict"] == "unsure"
    assert (home / "python").exists()


def test_a_drive_relative_path_is_refused(box):
    """`C:rel` is relative to C:'s current directory, not absolute. A complete
    data folder sits where it would resolve, so a failing guard shows."""
    drive = str(box)[:2]
    rel = box / "cwd" / "rel"
    _home(rel)
    decided = _run(box, answers=("yes", "yes"), env_home=f"{drive}rel")
    assert decided["verdict"] == "unsure"
    assert (rel / "python").exists()


def test_a_relative_orienta_home_is_refused(box):
    rel = box / "cwd" / "rel"
    _home(rel)
    decided = _run(box, answers=("yes", "yes"), env_home="rel")
    assert decided["verdict"] == "unsure"
    assert (rel / "python").exists()


# --------------------------------------------------------------------------
# the pointer
# --------------------------------------------------------------------------

def test_found_through_the_utf16_pointer_with_a_non_ascii_path(box):
    home = box / "OneDrive - Montanuniversität Leoben" / "Orienta"
    _home(home, user_file=False)
    pointer = _pointer(box / "appdata", str(home))
    decided = _run(box, answers=("yes", "no"))
    assert decided["source"] == "pointer"
    assert not (home / "python").exists()
    assert _library(home).exists()
    assert pointer.exists()                 # library kept, so the pointer too


def test_a_hand_written_utf8_pointer_with_padding_is_read(box):
    home = box / "plain" / "Orienta"
    _home(home, user_file=False)
    _pointer(box / "appdata", "   " + str(home) + "  ", utf16=False)
    decided = _run(box, answers=("yes", "no"))
    assert decided["home"] == str(home)
    assert not (home / "python").exists()


def test_an_unreachable_folder_changes_nothing_and_keeps_the_pointer(box):
    """Review finding: an unplugged drive deleted the only record of where
    the library was."""
    pointer = _pointer(box / "appdata", str(box / "unplugged" / "Orienta"))
    decided = _run(box, answers=("yes", "yes"))
    assert decided["verdict"] == "missing"
    assert pointer.exists()


def test_a_pointer_to_a_different_folder_survives_an_orienta_home_removal(box):
    home = box / "home"
    _home(home, user_file=False)
    other = box / "elsewhere" / "Orienta"
    other.mkdir(parents=True)
    pointer = _pointer(box / "appdata", str(other))
    _run(box, answers=("yes", "yes"), env_home=home)
    assert not home.exists()
    assert pointer.exists()


# --------------------------------------------------------------------------
# the real hook
# --------------------------------------------------------------------------

def test_through_the_hook_an_update_touches_nothing(box):
    """Executed, not grepped: ${isUpdated} forced true, both answers Yes."""
    home = box / "home"
    _home(home)
    _run(box, answers=("yes", "yes"), env_home=home, via_hook=True)
    assert (home / "python" / "python.exe").exists()
    assert _library(home).exists()


def test_through_the_hook_an_uninstall_does_its_work(box):
    home = box / "home"
    _home(home)
    _run(box, answers=("yes", "no"), env_home=home, via_hook=False)
    assert not (home / "python").exists()
    assert _library(home).exists()


def test_an_all_users_install_still_works_on_the_users_folder(box):
    home = box / "home"
    _home(home)
    _run(box, answers=("yes", "no"), env_home=home, install_mode="all")
    assert not (home / "python").exists()
    assert _library(home).exists()


# --------------------------------------------------------------------------
# the sentences
# --------------------------------------------------------------------------

@pytest.mark.parametrize("language,must_contain", [
    (1031, ["Kristallbibliothek", "LÖSCHEN", "rückgängig", "„Nein“"]),
    (1041, ["結晶ライブラリ", "元に戻せません"]),
    (2052, ["晶体库", "无法撤销"]),
    (1033, ["crystal library", "cannot be undone"]),
])
def test_the_questions_arrive_in_their_language_not_as_mojibake(box, language, must_contain):
    home = box / "home"
    _home(home)
    decided = _run(box, answers=("no", "no"), env_home=home, language=language)
    text = decided["ask"] + "\n" + decided["lib"]
    for fragment in must_contain:
        assert fragment in text, f"{fragment!r} missing from:\n{text}"
    assert str(home) in text


# --------------------------------------------------------------------------
# the file itself, and its wiring
# --------------------------------------------------------------------------

def _code() -> list[str]:
    return [line for line in NSH.read_text(encoding="utf-8-sig").splitlines()
            if not line.lstrip().startswith(";")]


def test_the_include_has_a_byte_order_mark():
    assert NSH.read_bytes()[:3] == b"\xef\xbb\xbf"


def test_nothing_in_it_uses_a_recursive_rmdir():
    assert not [l for l in _code() if "RMDir /r" in l or "RMDir /R" in l]


def test_cmd_runs_without_autorun_or_delayed_expansion():
    rd = [l for l in _code() if " rd " in l]
    assert rd and all("/d /v:off" in l for l in rd)


def test_the_library_question_defaults_to_no():
    line = next(l for l in _code() if "$oMsgLib" in l and "MessageBox" in l)
    assert "MB_DEFBUTTON2" in line and "/SD IDNO" in line


def test_setup_writes_the_marker_the_uninstaller_trusts():
    installer = (REPO / "electron" / "setup" / "installer.js").read_text(encoding="utf-8")
    assert "'.orienta-home'" in installer


def test_the_shipped_build_includes_it_and_defines_no_test_switches():
    build = json.loads((REPO / "frontend" / "package.json").read_text(encoding="utf-8"))["build"]
    assert build["nsis"]["include"] == "../electron/nsis/installer.nsh"
    wrapper = (REPO / "electron" / "nsis" / "installer.nsh").read_text(encoding="utf-8-sig")
    assert r'!include "${__FILEDIR__}\uninstall.nsh"' in wrapper
    assert "ORIENTA_TEST" not in json.dumps(build)
