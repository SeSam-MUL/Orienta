"""The torch requirement must name a CUDA build, not merely offer an index.

Measured on 2026-09-17, on a machine with an RTX 4070 and driver 591.86: a
fresh ``pip install -r requirements.txt`` produced

    torch          : 2.14.0+cpu
    cuda compiled  : None
    cuda available : False

``requirements.txt`` carried

    --extra-index-url https://download.pytorch.org/whl/cu121
    torch>=2.0

``--extra-index-url`` means "also look here". pip then takes the **highest**
version across all indexes it knows. The cu121 index stops at
``torch 2.5.1+cu121`` for cp311/Windows; PyPI is at ``2.14.0``; ``torch>=2.0``
constrains nothing downward. PyPI therefore wins every time, and the PyPI torch
wheel on Windows is CPU-only. The CUDA line had been inert since torch 2.6
appeared on PyPI.

It failed **silently**: cupy keeps working (its own runtime via the ``nvidia-*``
wheels), so the app runs and merely computes everything on the CPU. Measured
cost: fp32 matmul 11.2 TFLOP/s on the GPU against 0.71 TFLOP/s on the CPU.

The fix that holds is a **pin with a local version**: ``torch==2.11.0+cu126``
exists only in the PyTorch index, so there is no version contest for
``--extra-index-url`` to lose. Verified by installing the pinned set into a
fresh standalone interpreter: ``torch 2.11.0+cu126``,
``torch.cuda.is_available() == True``.

These tests pin that property in ``requirements.txt`` and in any lock file, so
the next person who relaxes the pin has to argue with a red test rather than
discover it on a tester's machine six months later.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO_ROOT / "requirements.txt"

# "torch", "torchvision" -- the distributions served by the PyTorch index.
_TORCH_NAMES = ("torch", "torchvision")
_INDEX_RE = re.compile(r"--extra-index-url\s+\S*download\.pytorch\.org/whl/(\w+)")
_LOCAL_VERSION_RE = re.compile(r"==\s*[\w.]+\+(\w+)")


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _requirement_lines(text: str) -> list[str]:
    return [s for s in (_strip_comment(ln) for ln in text.splitlines()) if s]


def _cuda_tag_of_index(lines: list[str]) -> str | None:
    """The cuXXX tag of the declared PyTorch index, if one is declared."""
    for line in lines:
        m = _INDEX_RE.search(line)
        if m:
            return m.group(1)
    return None


def _applies_where_cuda_exists(marker: str) -> bool:
    """Does this requirement apply on a platform that can have CUDA?

    No marker at all, or a marker that merely excludes macOS (which has no
    CUDA wheels), means Windows and Linux get this line.
    """
    if not marker:
        return True
    return 'Darwin' in marker and '!=' in marker


def _torch_specs(lines: list[str]) -> list[tuple[str, str, str]]:
    """(distribution, specifier-part, marker) for every torch/torchvision line."""
    out = []
    for line in lines:
        if line.startswith("--"):
            continue
        body, _, marker = line.partition(";")
        name = re.split(r"[<>=!\[]", body.strip(), maxsplit=1)[0].strip().lower()
        if name in _TORCH_NAMES:
            out.append((name, body.strip(), marker.strip()))
    return out


def _check_pinned_to_index(text: str, source: str) -> None:
    lines = _requirement_lines(text)
    tag = _cuda_tag_of_index(lines)
    specs = _torch_specs(lines)

    if not specs:
        pytest.skip(f"{source} requires no torch -- nothing to guard")

    cuda_platform_specs = [s for s in specs if _applies_where_cuda_exists(s[2])]
    if not cuda_platform_specs:
        # Only macOS-restricted torch lines: there is no build to name.
        return

    # Deleting the index line is the easiest way to "relax" the pin, because the
    # pin is only installable BECAUSE of the index. An unpinned torch with no
    # index is the original defect minus one line, so it must fail here rather
    # than skip -- a guard that steps aside for the edit that reintroduces the
    # bug is not a guard.
    if tag is None:
        for name, body, _ in cuda_platform_specs:
            assert _LOCAL_VERSION_RE.search(body), (
                f"{source}: '{body}' declares no PyTorch index and no +cuXXX/+cpu "
                f"build, so it resolves from PyPI -- which is CPU-only on Windows "
                f"and, on Linux, a CUDA build with a different major than the rest "
                f"of this file. Name the index and the build."
            )
        return

    for name, body, marker in cuda_platform_specs:
        local = _LOCAL_VERSION_RE.search(body)
        assert local, (
            f"{source}: '{body}' resolves from PyPI, not from the {tag} index.\n"
            f"An unpinned or plainly-pinned {name} loses to PyPI's newer wheel, "
            f"which is CPU-only on Windows. Pin the local version, e.g. "
            f"'{name}==<version>+{tag}'."
        )
        found = local.group(1)
        assert found == tag, (
            f"{source}: '{body}' is pinned to build '+{found}' while the declared "
            f"index serves '{tag}'. pip cannot satisfy that from the declared index."
        )


def test_requirements_pins_torch_to_the_declared_cuda_index():
    _check_pinned_to_index(REQUIREMENTS.read_text(encoding="utf-8"), "requirements.txt")


# Every lock file in the repository, found rather than listed: a new platform's
# lock that nobody remembered to add here would otherwise ship unguarded.
ALL_LOCKS = sorted(p.name for p in REPO_ROOT.glob("requirements-lock-*.txt"))


def test_the_guard_covers_every_lock_file_we_ship():
    assert "requirements-lock-linux-gpu.txt" in ALL_LOCKS, ALL_LOCKS
    assert len(ALL_LOCKS) >= 4, ALL_LOCKS


@pytest.mark.parametrize("lock_name", ALL_LOCKS)
def test_lock_files_pin_torch_to_their_declared_index(lock_name):
    """A lock file is what an installer executes -- the guard matters most here."""
    lock = REPO_ROOT / lock_name
    if not lock.exists():
        pytest.skip(f"{lock_name} does not exist yet")
    _check_pinned_to_index(lock.read_text(encoding="utf-8"), lock_name)


def test_guard_rejects_the_configuration_that_shipped_the_defect():
    """The exact text that produced torch 2.14.0+cpu must fail this guard.

    Without this, a guard that silently passes everything would look healthy.
    """
    broken = (
        "--extra-index-url https://download.pytorch.org/whl/cu121\n"
        "torch>=2.0\n"
        "torchvision>=0.15\n"
    )
    with pytest.raises(AssertionError, match="resolves from PyPI"):
        _check_pinned_to_index(broken, "<the 2026-09-17 defect>")


def test_guard_rejects_a_pin_that_does_not_match_its_index():
    mismatched = (
        "--extra-index-url https://download.pytorch.org/whl/cu126\n"
        "torch==2.5.1+cu121\n"
    )
    with pytest.raises(AssertionError, match="cannot satisfy"):
        _check_pinned_to_index(mismatched, "<mismatched index>")


def test_guard_accepts_a_macos_only_plain_torch_alongside_a_pinned_cuda_torch():
    """macOS has no CUDA wheels; a plain pin there is correct, not a defect."""
    ok = (
        "--extra-index-url https://download.pytorch.org/whl/cu126\n"
        'torch==2.11.0+cu126 ; platform_system != "Darwin"\n'
        'torch==2.11.0 ; platform_system == "Darwin"\n'
    )
    _check_pinned_to_index(ok, "<mixed platforms>")
