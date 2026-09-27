"""The macOS lock file, held against the reason it exists.

macOS is the one platform that does not install from a pip lock. The reason
is a bug report: on a Mac the pip wheels of torch, scikit-learn and faiss each
bring their own OpenMP runtime, and the second one to start aborts the backend
about thirty seconds after launch. conda-forge builds all of them against one
llvm-openmp.

So the property that matters is not "the file parses" — it is "exactly one
OpenMP runtime, and the versions we validated on Windows".
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCKFILE = REPO_ROOT / "orienta-macos-lock.yml"
ENVIRONMENT = REPO_ROOT / "environment-macos.yml"

pytestmark = pytest.mark.skipif(not LOCKFILE.exists(), reason="orienta-macos-lock.yml not generated")


def test_the_file_name_ends_in_lock_yml_because_micromamba_checks_that():
    """The name is load-bearing, and it looks arbitrary, so it is pinned here.

    micromamba decides whether a file is a lockfile by its NAME:

        is_conda_env_lockfile_name(f) = f.ends_with("-lock.yml")
                                     || f.ends_with("-lock.yaml")

    Anything else is parsed as a plain environment YAML instead. Measured with
    micromamba 2.9.0 on this repository's own lock, same bytes, two names:

        conda-lock-macos.yml   -> exit 0, no output, NOTHING installed
        orienta-macos-lock.yml -> exit 0, a transaction of 467 packages

    The first is the dangerous one: not an error, a silent success that builds
    an empty environment. It was the committed name until 2026-09-23.
    """
    assert LOCKFILE.name.endswith(("-lock.yml", "-lock.yaml")), LOCKFILE.name


def records() -> list[dict[str, str]]:
    """Every package entry, as its own record.

    Field by field rather than by searching the whole file: an assertion that
    greps the text passes as long as SOME package satisfies it, which is how
    the first version of `test_blas_is_apple_accelerate` survived swapping
    Accelerate for OpenBLAS (two unrelated build strings kept the word alive).
    """
    text = LOCKFILE.read_text(encoding="utf-8")
    out: list[dict[str, str]] = []
    for block in re.split(r"^- name: ", text, flags=re.M)[1:]:
        entry = {"name": block.splitlines()[0].strip()}
        for field in ("version", "manager", "platform", "url"):
            found = re.search(rf"^  {field}: (\S+)", block, re.M)
            if found:
                # conda-lock quotes some versions ('16.0'), so the quotes come off.
                entry[field] = found.group(1).strip("'\"")
        out.append(entry)
    return out


def names_and_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in records():
        out.setdefault(entry["name"], entry.get("version", ""))
    return out


def test_exactly_one_openmp_runtime():
    # The whole point. A second one is the GitHub issue, again.
    found = sorted(n for n in names_and_versions() if n in ("llvm-openmp", "intel-openmp", "libgomp"))
    assert found == ["llvm-openmp"], found


def test_it_is_for_apple_silicon_only():
    # Every package record, not the `platforms:` list in the header -- that
    # list is metadata and stays right while a package underneath it is wrong.
    # A package is either built for this architecture or architecture-free
    # (conda's `noarch`); anything else is a wheel for the wrong machine.
    wrong = [
        (e["name"], e.get("platform"), e.get("url"))
        for e in records()
        if e.get("platform") != "osx-arm64"
        or not re.search(r"/(osx-arm64|noarch)/", e.get("url", ""))
    ]
    assert wrong == [], wrong


def test_it_carries_the_versions_the_environment_file_pins():
    # The yml is what a human edits; the lock is what a machine installs. They
    # drift apart silently -- the project's own record calls that its dominant
    # defect class -- so they are compared here.
    pinned = dict(re.findall(r"^\s*- ([A-Za-z0-9_.-]+)=([0-9][^\s=]*)$",
                             ENVIRONMENT.read_text(encoding="utf-8"), re.M))
    locked = names_and_versions()
    mismatched = {
        name: (want, locked.get(name))
        for name, want in pinned.items()
        if name in locked and not locked[name].startswith(want)
    }
    assert mismatched == {}, mismatched
    missing = [name for name in pinned if name not in locked]
    assert missing == [], missing


def test_the_stack_the_backend_actually_imports_is_in_it():
    locked = names_and_versions()
    for package in ("pytorch", "scikit-learn", "numba", "kikuchipy", "orix",
                    "pyebsdindex", "hyperspy", "h5py", "fastapi", "ray-core"):
        assert package in locked, package


def test_the_lock_carries_what_the_environment_file_asks_for_including_pytest():
    # pytest is in environment-macos.yml and therefore in the lock, exactly as
    # it is in the Windows locks. The earlier name of this test claimed the
    # installer left it out; it does not, and the assertion never said so.
    # If it should be dropped, that is an edit to the yml, not to this test.
    locked = names_and_versions()
    assert "pytest" in locked, "the yml pins it, so the lock should have it"


def test_it_does_not_record_where_this_repository_lives():
    """No directory layout of whoever generated it.

    Two reasons, and the weaker one is the privacy: this file ships inside the
    runtime package every user downloads. The stronger one is that a recorded
    path makes the file depend on the checkout it was made from, so
    `--check` reports "out of date" for a solve that is identical -- and the
    CI job built on it (plan T10) would be red from its first run.
    """
    # Package urls are excluded: every one of them contains "https:", which a
    # drive-letter pattern matches. Only the lines naming a FILE are checked.
    text = LOCKFILE.read_text(encoding="utf-8")
    offenders = [
        line for line in text.splitlines()
        if not line.lstrip().startswith("url:")
        and re.search(r"(worktrees|\b[A-Za-z]:[\\/]|\.\.[\\/]|/home/|/Users/)", line)
    ]
    assert offenders == [], offenders


def test_blas_is_apple_accelerate():
    # Measured by the Mac session: Accelerate, not OpenBLAS. OpenBLAS would
    # bring its own threading and undo the one-runtime property.
    blas = [e for e in records() if e["name"] == "libblas"]
    assert len(blas) == 1, blas
    # The build string lives in the file name, so the url is what decides.
    assert "accelerate" in blas[0].get("url", "").lower(), blas[0]
    # And nothing may drag OpenBLAS in beside it.
    assert [e["name"] for e in records() if e["name"] == "libopenblas"] == []
