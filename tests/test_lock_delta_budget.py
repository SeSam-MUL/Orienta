"""What an update of a 0.4.6 installation is allowed to change.

The package sync of release 0.4.7 installs the new lock into the environment of
an existing installation. Whatever the lock names differently from the lock of
0.4.6 is what that update touches, so the difference between the two is the
whole blast radius, and it is small on purpose: the three libraries whose
behaviour was validated on real data (kikuchipy, orix, PyEBSDIndex) and nothing
that decides whether the GPU stack still loads.

This test holds the HEAD locks against the locks of v0.4.6
(tests/fixtures/locks/v0.4.6/, verbatim) and fails when

* a guarded name changes, is added or is removed: torch, torchvision, numpy,
  scipy, cupy, nvidia-*, triton (and, on macOS, the one OpenMP runtime);
* the delta of a pip lock is anything but the three libraries;
* the delta of the macOS conda lock is anything but the list written out below.

The expected values are written out, not derived, so that a future lock change
has to update this file knowingly. A lock whose delta is wider than the release
notes say is the failure this test exists for.

Pip locks list a handful of direct pins only on Windows (the Linux locks
resolve every package). Two names are therefore *added* to the Windows locks
without being new to an installation: threadpoolctl 3.7.0 (scikit-learn
requires it, every 0.4.6 install has 3.7.0) and packaging 26.3 (a requirement
of requirements.txt since the detector convention helper imports it). The test
requires both to equal what the v0.4.6 lock of the same flavour for Linux
resolved, which is what a fresh 0.4.6 install resolved too, so that the pin
does not change an installed version.

The guards are watched failing at the end of the file: the same checks run over
a deliberately damaged copy of each HEAD lock. Without a name the guard can
find in the lock, "no guarded name changed" would also be true of an empty
parse, so a positive anchor asserts that the guarded names are present.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "locks" / "v0.4.6"

#: sha256 of the fixture files, so that nobody "fixes" a fixture to make a delta pass.
FIXTURE_SHA256 = {
    "requirements-lock-cpu.txt": "50583924c48db56b3bf29ab6da03675c6e8c68710b6f50493bf1f142b5817914",
    "requirements-lock-gpu.txt": "2b5e527f83954c3df2177e10d9ae715f1f750835c8b68d9134f9a5e8c4a80b8f",
    "requirements-lock-linux-cpu.txt": "f65047f0b84dd737c23753ad0056770354ae7dc06bf58f30baa4cad4e0c1c9b1",
    "requirements-lock-linux-gpu.txt": "69986d56d44577f0e5c9606590ed9dac2c57f1fb72b0d6ec64f701e09eca6a4d",
    "orienta-macos-lock.yml": "75b4e22ad7fe465f9b828d57cd1206073a76804c9e435bcf79e872f686d0f9e8",
    "environment-macos.yml": "ad47a8262b1045769add331b318767921f6f0acfd6a7b8f721a222ec31000825",
}

#: Names an update must never touch. Matched against the normalised name.
GUARDED = re.compile(
    r"^(torch|torchvision|torchaudio|pytorch|libtorch|numpy|scipy|cupy(-.*)?"
    r"|nvidia-.*|triton(-.*)?|llvm-openmp)$"
)

#: The delta of every Windows and Linux pip lock, old -> new.
PIP_DELTA = {
    "kikuchipy": ("0.11.3", "0.13.1"),
    "orix": ("0.14.1", "0.15.0"),
    "pyebsdindex": ("0.3.9.1", "0.3.10.1"),
}

#: Names that appear in the Windows locks for the first time, with the version
#: they must have (checked against the Linux lock of 0.4.6, see the docstring).
WINDOWS_NEW_PINS = {"threadpoolctl": "3.7.0", "packaging": "26.3"}

PIP_LOCKS = {
    # lock file -> the v0.4.6 Linux lock of the same flavour, or None for Linux itself
    "requirements-lock-cpu.txt": "requirements-lock-linux-cpu.txt",
    "requirements-lock-gpu.txt": "requirements-lock-linux-gpu.txt",
    "requirements-lock-linux-cpu.txt": None,
    "requirements-lock-linux-gpu.txt": None,
}

#: The delta of the macOS conda lock: conda package -> (old version, new version),
#: None meaning "not in the lock of 0.4.6". Seven moved (the three libraries, their
#: -base outputs and openssl, which the solver took along), two arrived
#: (psygnal, a dependency of kikuchipy 0.13, and lazy_loader).
MACOS_DELTA = {
    "kikuchipy": ("0.11.3", "0.13.1"),
    "kikuchipy-base": ("0.11.3", "0.13.1"),
    "orix": ("0.14.1", "0.15.0"),
    "orix-base": ("0.14.1", "0.15.0"),
    "pyebsdindex": ("0.3.9.1", "0.3.10.1"),
    "pyebsdindex-base": ("0.3.9.1", "0.3.10.1"),
    "openssl": ("3.6.4", "3.6.5"),
    "psygnal": (None, "0.16.1"),
    "lazy_loader": (None, "0.5"),
}

#: environment-macos.yml pins (`- name=version`); packaging is new there.
MACOS_ENV_DELTA = {
    "kikuchipy": ("0.11.3", "0.13.1"),
    "orix": ("0.14.1", "0.15.0"),
    "pyebsdindex": ("0.3.9.1", "0.3.10.1"),
    "packaging": (None, "26.3"),
}


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def pip_pins(text: str) -> dict[str, set[str]]:
    """name -> {version, ...}. Windows locks pin torch twice (+cu126 and macOS)."""
    pins: dict[str, set[str]] = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==([^\s;]+)", line)
        if m:
            pins.setdefault(_norm(m.group(1)), set()).add(m.group(2))
    return pins


def conda_pins(text: str) -> dict[str, str]:
    """conda package -> 'version' (build string is part of the file name, not compared)."""
    pins: dict[str, str] = {}
    for block in re.split(r"^- name: ", text, flags=re.M)[1:]:
        name = block.splitlines()[0].strip()
        version = re.search(r"^  version: (\S+)", block, re.M)
        assert version, f"{name} has no version"
        assert name not in pins, f"{name} appears twice in the lock"
        pins[name] = version.group(1).strip("'\"")
    return pins


def env_pins(text: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^  - ([A-Za-z0-9._-]+)=([^=\s#]+)", line)
        if m:
            pins[_norm(m.group(1))] = m.group(2)
    return pins


def delta(old: dict, new: dict) -> dict[str, tuple]:
    """name -> (old, new) for every name whose value differs; absent side is None."""
    return {k: (old.get(k), new.get(k)) for k in set(old) | set(new)
            if old.get(k) != new.get(k)}


def guarded_in(names) -> list[str]:
    return sorted(n for n in names if GUARDED.match(n))


def one(values: set[str] | None) -> str | None:
    if values is None:
        return None
    assert len(values) == 1, values
    return next(iter(values))


def old_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def head_text(name: str) -> str:
    return (REPO_ROOT / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# the fixtures are what 0.4.6 shipped
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(FIXTURE_SHA256))
def test_the_fixture_is_the_file_of_v046_byte_for_byte(name: str) -> None:
    digest = hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()
    assert digest == FIXTURE_SHA256[name], (
        f"{name} differs from the file the v0.4.6 tag carries; the fixture is the "
        "reference for 'what changed since 0.4.6' and is never edited"
    )


def test_the_fixture_is_the_0_4_6_stack() -> None:
    """kikuchipy 0.11.3 is what a 0.4.6 install has: the thing this all compares against."""
    pins = pip_pins(old_text("requirements-lock-gpu.txt"))
    assert pins["kikuchipy"] == {"0.11.3"}


# --------------------------------------------------------------------------
# Windows and Linux pip locks
# --------------------------------------------------------------------------

@pytest.mark.parametrize("lock", sorted(PIP_LOCKS))
def test_the_guarded_names_are_present_so_the_guard_can_fail(lock: str) -> None:
    names = pip_pins(head_text(lock))
    expected = {"numpy", "scipy", "torch", "torchvision"}
    if lock.endswith("gpu.txt"):
        expected |= {"cupy-cuda12x"}
    assert expected <= set(guarded_in(names)), guarded_in(names)


@pytest.mark.parametrize("lock", sorted(PIP_LOCKS))
def test_no_guarded_package_changes_in_a_pip_lock(lock: str) -> None:
    changes = delta(pip_pins(old_text(lock)), pip_pins(head_text(lock)))
    assert guarded_in(changes) == [], {n: changes[n] for n in guarded_in(changes)}


@pytest.mark.parametrize("lock", sorted(PIP_LOCKS))
def test_the_pip_delta_is_the_three_libraries(lock: str) -> None:
    old, new = pip_pins(old_text(lock)), pip_pins(head_text(lock))
    changes = delta(old, new)
    added = {k: v for k, v in changes.items() if v[0] is None}
    removed = {k: v for k, v in changes.items() if v[1] is None}
    moved = {k: (one(v[0]), one(v[1])) for k, v in changes.items()
             if v[0] is not None and v[1] is not None}

    assert removed == {}, "an update must not drop a package"
    assert moved == PIP_DELTA, moved
    if PIP_LOCKS[lock] is None:  # Linux locks list every package: nothing may be new
        assert added == {}, added
    else:  # Windows locks list direct pins only: see the module docstring
        reference = pip_pins(old_text(PIP_LOCKS[lock]))
        assert {k: one(v[1]) for k, v in added.items()} == WINDOWS_NEW_PINS, added
        for name, version in WINDOWS_NEW_PINS.items():
            assert reference[name] == {version}, (
                f"{name} {version} is new to the Windows lock but 0.4.6 resolved "
                f"{reference.get(name)}: the pin would change an installed version")


# --------------------------------------------------------------------------
# macOS
# --------------------------------------------------------------------------

def test_macos_guarded_names_are_present_so_the_guard_can_fail() -> None:
    assert {"numpy", "scipy", "pytorch", "libtorch", "torchvision", "llvm-openmp"} <= set(
        guarded_in(conda_pins(head_text("orienta-macos-lock.yml"))))


def test_no_guarded_package_changes_in_the_macos_lock() -> None:
    changes = delta(conda_pins(old_text("orienta-macos-lock.yml")),
                    conda_pins(head_text("orienta-macos-lock.yml")))
    assert guarded_in(changes) == [], {n: changes[n] for n in guarded_in(changes)}


def test_the_macos_delta_is_exactly_the_listed_one() -> None:
    changes = delta(conda_pins(old_text("orienta-macos-lock.yml")),
                    conda_pins(head_text("orienta-macos-lock.yml")))
    assert changes == MACOS_DELTA, (
        "the macOS lock changed differently from what this test expects; if that "
        "is intended, update MACOS_DELTA and say why in the commit")


def test_the_macos_environment_file_changes_only_the_three_pins() -> None:
    changes = delta(env_pins(old_text("environment-macos.yml")),
                    env_pins(head_text("environment-macos.yml")))
    assert changes == MACOS_ENV_DELTA, changes


def test_the_macos_environment_file_and_the_lock_agree_on_the_three_libraries() -> None:
    env = env_pins(head_text("environment-macos.yml"))
    lock = conda_pins(head_text("orienta-macos-lock.yml"))
    for name in ("kikuchipy", "orix", "pyebsdindex", "packaging"):
        assert lock[name] == env[name], (name, lock[name], env[name])


# --------------------------------------------------------------------------
# the guards, watched failing
# --------------------------------------------------------------------------

def _bump(text: str, pattern: str, replacement: str) -> str:
    damaged, count = re.subn(pattern, replacement, text, count=1, flags=re.M)
    assert count == 1, f"the damage did not apply: {pattern}"
    return damaged


@pytest.mark.parametrize("lock, pattern, replacement, name", [
    ("requirements-lock-gpu.txt", r"^numpy==2\.3\.5", "numpy==2.3.6", "numpy"),
    ("requirements-lock-gpu.txt", r"^torch==2\.11\.0\+cu126", "torch==2.11.1+cu126", "torch"),
    ("requirements-lock-gpu.txt", r"^cupy-cuda12x==14\.0\.1", "cupy-cuda12x==14.0.2", "cupy-cuda12x"),
    ("requirements-lock-gpu.txt", r"^nvidia-cublas-cu12==12\.9\.2\.10", "nvidia-cublas-cu12==12.9.2.11",
     "nvidia-cublas-cu12"),
    ("requirements-lock-linux-cpu.txt", r"^scipy==1\.17\.1", "scipy==1.17.2", "scipy"),
    ("requirements-lock-linux-gpu.txt", r"^triton==", "triton==9.", "triton"),
])
def test_a_damaged_pip_lock_trips_the_guard(lock, pattern, replacement, name) -> None:
    damaged = _bump(head_text(lock), pattern, replacement)
    changes = delta(pip_pins(old_text(lock)), pip_pins(damaged))
    assert name in guarded_in(changes)


def test_a_damaged_macos_lock_trips_the_guard() -> None:
    text = head_text("orienta-macos-lock.yml")
    damaged = _bump(text, r"(^- name: pytorch\n  version: )\S+", r"\g<1>9.9.9")
    changes = delta(conda_pins(old_text("orienta-macos-lock.yml")), conda_pins(damaged))
    assert "pytorch" in guarded_in(changes)
    assert changes != MACOS_DELTA


def test_a_damaged_delta_is_not_the_three_libraries() -> None:
    """Moving a fourth package changes the delta, even when it is not guarded."""
    lock = "requirements-lock-linux-cpu.txt"
    damaged = _bump(head_text(lock), r"^pydantic==\S+", "pydantic==0.0.1")
    changes = delta(pip_pins(old_text(lock)), pip_pins(damaged))
    moved = {k: (one(v[0]), one(v[1])) for k, v in changes.items()}
    assert moved != PIP_DELTA
