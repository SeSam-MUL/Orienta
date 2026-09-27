"""The projection that decides what the build machines may see.

Most tests here work by BREAKING the tree and checking that `verify` says so.
A guard nobody has watched fail is a claim, not a guard -- and this particular
one stands between the maintainer's crystal library and a hosted git server.

One test is not a mutation but a control: an empty directory must NOT come back
clean, because a scanner that reads nothing and a scanner that finds nothing
print the same word.

The last group asks a different question -- not "what does verify reject" but
"what reaches the index", which is the set that actually gets pushed.
"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_build_repo import (  # noqa: E402
    EXCLUDE_DIRS,
    EXCLUDE_SUFFIXES,
    clear,
    keep,
    stage,
    verify,
)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A minimal projection that passes, so a test can make exactly one thing wrong."""
    out = tmp_path / "projection"
    (out / "backend" / "api").mkdir(parents=True)
    (out / "backend" / "api" / "main.py").write_text(
        "def main():\n    return 0\n", encoding="utf-8")
    (out / "README.md").write_text("Orienta\n", encoding="utf-8")
    return out


def test_the_clean_tree_is_clean(tree: Path) -> None:
    assert verify(tree) == []


# --------------------------------------------------------------------------
# the control -- without it, every assertion below is unfalsifiable
# --------------------------------------------------------------------------

def test_an_empty_tree_is_refused_not_called_clean(tmp_path: Path) -> None:
    """A scan of nothing must not read as a scan that found nothing.

    This is the failure mode that would matter most in practice: assemble()
    silently producing an empty or near-empty directory, verify() finding no
    forbidden files in it, and the result being published as 'clean'.
    """
    empty = tmp_path / "empty"
    empty.mkdir()
    problems = verify(empty)
    assert problems, "an empty tree came back clean"
    assert "control term" in problems[0]


# --------------------------------------------------------------------------
# the data guards
# --------------------------------------------------------------------------

@pytest.mark.parametrize("suffix", EXCLUDE_SUFFIXES)
def test_every_forbidden_suffix_is_caught(tree: Path, suffix: str) -> None:
    """Each one, not a representative sample.

    The list grew twice -- .h5 after a test fixture turned up in it, .npy/.npz
    after simulated master patterns were found wearing a different extension.
    A parametrised test is what keeps an addition to the list from being an
    addition that does nothing.
    """
    (tree / f"something{suffix}").write_bytes(b"\x00binary\x00")
    problems = verify(tree)
    assert problems, f"{suffix} was not caught"
    assert "forbidden suffix" in problems[0]


def test_the_suffix_check_ignores_case(tree: Path) -> None:
    """Windows writes SampleB.H5OINA as readily as sampleb.h5oina."""
    (tree / "Sample.H5OINA").write_bytes(b"\x00")
    assert any("forbidden suffix" in p for p in verify(tree))


@pytest.mark.parametrize("excluded", EXCLUDE_DIRS)
def test_every_excluded_directory_is_caught(tree: Path, excluded: str) -> None:
    (tree / excluded.rstrip("/")).mkdir(parents=True)
    assert any("excluded directories" in p for p in verify(tree))


def test_a_data_file_inside_an_excluded_directory_is_caught_twice(tree: Path) -> None:
    """Belt and braces, deliberately: the two guards overlap and should.

    The 547 MB h5oina lives under Test_data/, so both the directory rule and
    the suffix rule would stop it. Either one failing alone must not be enough
    to publish it.
    """
    data = tree / "Test_data"
    data.mkdir()
    (data / "scan.h5oina").write_bytes(b"\x00")
    problems = verify(tree)
    assert any("forbidden suffix" in p for p in problems)
    assert any("excluded directories" in p for p in problems)


# --------------------------------------------------------------------------
# the credential guard
# --------------------------------------------------------------------------

def test_a_github_token_is_caught(tree: Path) -> None:
    # Shaped like one, not one: 36 characters after the prefix.
    (tree / "config.py").write_text(
        'TOKEN = "ghp_' + "A1b2C3d4" * 4 + 'E5f6"\n', encoding="utf-8")
    assert any("credential-shaped" in p for p in verify(tree))


def test_a_private_key_block_is_caught(tree: Path) -> None:
    # Assembled at runtime rather than written out, so this file does not
    # itself contain the header. It did, in the first draft, and the guard
    # duly refused to publish the repository -- correctly, on its own terms.
    # The fix belongs here and not in the skip list: a test file is exactly
    # the sort of place a real key gets left behind, and a guard widened to
    # forgive its own test has been switched off by increments.
    header = "-----BEGIN " + "OPENSSH PRIVATE" + " KEY-----"
    (tree / "id_rsa").write_text(f"{header}\nabc\n", encoding="utf-8")
    assert any("credential-shaped" in p for p in verify(tree))


def test_the_guard_does_not_flag_its_own_description(tree: Path) -> None:
    """The script naming the shapes it hunts must not trip over itself.

    Left unhandled this is the kind of thing that gets a guard disabled rather
    than fixed, the first time someone hits it at the end of a long day.
    """
    scripts = tree / "scripts"
    scripts.mkdir()
    (scripts / "make_build_repo.py").write_text(
        'PATTERN = rb"gh[pousr]_[A-Za-z0-9]{36}"\n'
        'EXAMPLE = "ghp_' + "A1b2C3d4" * 4 + 'E5f6"\n', encoding="utf-8")
    assert verify(tree) == []


def test_binary_files_do_not_crash_the_scan(tree: Path) -> None:
    """An icon or a font must not end the projection with a decode error."""
    (tree / "icon.png").write_bytes(bytes(range(256)) * 8)
    assert verify(tree) == []


# --------------------------------------------------------------------------
# what `keep` decides, before anything is copied
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "backend/api/main.py",
    "frontend/src/App.jsx",
    "scripts/build_release.py",
    ".github/workflows/build-mac-linux.yml",
    "requirements-lock-linux-cpu.txt",
    "orienta-macos-lock.yml",
    "tests/test_runtime_package.py",
    "licenses/EMsoft-License.txt",
])
def test_the_build_needs_these(path: str) -> None:
    """Named one by one, because each was withheld by an earlier draft.

    `tasks/` had to be excluded for the .npy arrays inside it, and the first
    attempt at that reached `tests/` too. The lock files are what the wizard
    reads on the machine it is installing on; without them the package cannot
    finish an install it has already started.
    """
    assert keep(path), f"{path} would not reach the build machines"


@pytest.mark.parametrize("path", [
    "Test_data/SampleB.h5oina",
    "Database/CIF_Library/Al.cif",
    "Database/XTAL_Library/Si.xtal",
    "sample_data/Ni.sht",
    "tests/data/dict_gpu/realdata_sampleB_30x30.h5",
    "diagnostic_runs/run.sht",
    "tasks/forward_sim/iterations/master.npy",
    "matlab_testskripts/gKAM.m",
])
def test_none_of_this_may_leave_the_machine(path: str) -> None:
    assert not keep(path), f"{path} would be published"


# --------------------------------------------------------------------------
# what reaches the index -- the step that published 1854 of 1856 files once
# --------------------------------------------------------------------------

@pytest.fixture
def staged(tree: Path) -> tuple[Path, list[str]]:
    """A projection that carries a .gitignore matching one of its own files.

    This is the shape the source repository has: `IndexEBSD.nml` is tracked
    there AND matched by its .gitignore. It is the whole reason `stage` forces
    the add, so the fixture reproduces it rather than describing it.
    """
    (tree / ".gitignore").write_text("*.nml\nlogs/\n", encoding="utf-8")
    (tree / "IndexEBSD.nml").write_text("&EBSDIndexingNameListType\n/\n", encoding="utf-8")
    copied = ["README.md", ".gitignore", "IndexEBSD.nml", "backend/api/main.py"]
    problems = stage(tree, copied)
    assert problems == [], problems
    return tree, copied


def _index(repo: Path) -> set[str]:
    """Read THIS repository's index, whatever the environment says.

    The GIT_* scrub is not decoration here: without it the helper reads
    whichever file `GIT_INDEX_FILE` names, which made the test for exactly that
    pollution fail against correct code. A helper that trusts the environment
    cannot check code that does not.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    out = subprocess.run(["git", "-C", str(repo), "ls-files", "--cached", "-z"],
                         capture_output=True, text=True, encoding="utf-8",
                         check=True, env=env)
    return {entry for entry in out.stdout.split("\0") if entry}


def test_every_copied_file_is_staged_including_the_ignored_one(
        staged: tuple[Path, list[str]]) -> None:
    """The bite: with `git add -A` here, IndexEBSD.nml is simply absent.

    Verified by doing exactly that in a throwaway repository -- `add -A` stages
    three of these four files and exits 0.
    """
    repo, copied = staged
    assert _index(repo) == set(copied)
    assert "IndexEBSD.nml" in _index(repo), "the ignored-but-tracked file was dropped"


def test_a_directory_in_the_list_is_refused_not_expanded_quietly(tree: Path) -> None:
    """The comparison guard itself, with something for it to catch.

    A pathspec naming a directory adds the whole subtree, so one wrong entry in
    the list is how a withheld file would reach the index -- here a .cif beside
    a .py. Both halves of the report fire: the directory is not staged as a
    path, and two files are staged that nobody copied.

    Without this test the comparison was unguarded: replacing its condition
    with `if False` left all 54 tests passing, which is how a guard becomes
    decoration.
    """
    (tree / "pkg").mkdir()
    (tree / "pkg" / "a.py").write_text("x\n", encoding="utf-8")
    (tree / "pkg" / "secret.cif").write_text("x\n", encoding="utf-8")
    (tree / "top.py").write_text("x\n", encoding="utf-8")

    problems = stage(tree, ["top.py", "pkg"])

    assert problems, "a directory in the list was expanded and nobody noticed"
    assert "not staged: pkg" in problems[0]
    assert "secret.cif" in problems[0]


def test_a_second_run_can_clear_the_repository_of_the_first(tree: Path) -> None:
    """git writes loose objects read-only, and Windows will not unlink those.

    The first version of `stage` made the projection directory undeletable:
    `rmtree` took all the projected files and none of `.git`, and the next run
    died on `mkdir` with a bare FileExistsError. `clear` exists for this and
    nothing else.
    """
    assert stage(tree, ["README.md", "backend/api/main.py"]) == []
    assert (tree / ".git").is_dir()

    clear(tree)

    assert not tree.exists(), "an earlier run's repository survived the clear"


def test_clear_adds_permissions_and_never_takes_the_execute_bit(
        tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The mode handed to chmod is the guard, not the platform's reaction to it.

    `chmod(path, S_IWRITE | S_IREAD)` is 0o600: on Windows it only clears the
    read-only attribute, on POSIX it strips a directory's execute bit and
    `rmtree` then cannot enter it (`Permission denied: 'api'`). That broke the
    macOS and Linux builds of 0.4.6 and could not fail on the machine it was
    written on -- so what is checked here is the ARGUMENT, which is the same on
    every platform.
    """
    seen: list[tuple[Path, int]] = []
    real_chmod = os.chmod
    monkeypatch.setattr(os, "chmod", lambda p, m, **kw: (seen.append((Path(p), m)),
                                                        real_chmod(p, m, **kw))[1])
    assert stage(tree, ["README.md", "backend/api/main.py"]) == []

    clear(tree)

    dirs = [(p, m) for p, m in seen if p.name in ("backend", "api", ".git")]
    assert dirs, "premise: clear() reached the directories at all"
    for p, mode in dirs:
        assert mode & stat.S_IXUSR, f"{p.name} would become un-enterable (mode {oct(mode)})"
    for p, mode in seen:
        assert mode & stat.S_IWUSR, f"{p.name} was not made writable (mode {oct(mode)})"


def test_the_repository_is_configured_for_long_paths(
        staged: tuple[Path, list[str]]) -> None:
    """Git refuses a path this tree contains at 260 characters; Python does not.

    Pinned as configuration rather than by building a >260-character path,
    which would make the test pass or fail on the host's LongPathsEnabled
    setting instead of on our code. The failure itself was observed once, on
    the real projection, and is written down in the function.
    """
    repo, _ = staged
    out = subprocess.run(["git", "-C", str(repo), "config", "--local", "core.longpaths"],
                         capture_output=True, text=True, encoding="utf-8", check=True)
    assert out.stdout.strip() == "true"


def test_the_line_endings_are_pinned_not_inherited(
        staged: tuple[Path, list[str]]) -> None:
    """Two machines must project the same bytes into the index.

    git-for-windows sets `core.autocrlf=true` system-wide, and a fresh index
    has no prior entry to suppress normalisation: 91 of the 1856 blobs come out
    with their CRLF stripped here and with it kept on Linux. `input` is the
    behaviour the runners have already accepted -- normalise going in, never
    coming out -- and pinning it is what makes the projection reproducible
    rather than a property of whoever ran it.
    """
    repo, _ = staged
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    out = subprocess.run(["git", "-C", str(repo), "config", "--local", "core.autocrlf"],
                         capture_output=True, text=True, encoding="utf-8",
                         check=True, env=env)
    assert out.stdout.strip() == "input"


def test_the_branch_is_main_and_nothing_is_committed(
        staged: tuple[Path, list[str]]) -> None:
    """A commit is a person's step. The script must stop one short of it."""
    repo, _ = staged
    head = subprocess.run(["git", "-C", str(repo), "symbolic-ref", "--short", "HEAD"],
                          capture_output=True, text=True, encoding="utf-8", check=True)
    assert head.stdout.strip() == "main"
    log = subprocess.run(["git", "-C", str(repo), "log", "--oneline"],
                         capture_output=True, text=True, encoding="utf-8")
    assert log.returncode != 0, "the script committed, and it must not"


def test_a_name_in_the_list_that_is_not_on_disk_stops_the_projection(tree: Path) -> None:
    """A list that outruns the tree must not produce a staged repository.

    Written against what git actually does rather than what the count guard was
    built for: `add` with a pathspec matching nothing fails outright, so this
    case never reaches the comparison. The comparison is the net for the
    opposite drift -- a file present, added, and silently not staged -- which
    `test_every_copied_file_is_staged_including_the_ignored_one` is the live
    example of.
    """
    problems = stage(tree, ["README.md", "backend/api/main.py", "not_copied.py"])
    assert problems, "a missing file passed as staged"
    assert "not_copied.py" in problems[0]


def test_a_git_variable_in_the_environment_does_not_fool_the_comparison(
        tree: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The one way found so far for the comparison to pass on an empty index.

    With `GIT_INDEX_FILE` set, the add and the read that checks it both went to
    the same foreign file: they agreed with each other, `stage` reported
    success, and `out/.git/index` held nothing. A shell that has been in the
    middle of a rebase is enough to set it.
    """
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "elsewhere.index"))
    assert stage(tree, ["README.md", "backend/api/main.py"]) == []
    assert _index(tree) == {"README.md", "backend/api/main.py"}, \
        "the files went to an index outside the projection"


def test_an_empty_projection_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "nothing"
    empty.mkdir()
    assert stage(empty, []) == ["nothing was copied, so there is nothing to stage"]
