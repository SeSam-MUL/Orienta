"""The lock-file generator, pinned at the decisions that are easy to get wrong.

Every case below exists because an earlier version of the generator got it
wrong, and most were found by review rather than by use:

* **The CPU lock must name `+cpu` builds, not merely drop `+cu126`.** Measured:
  PyPI's `torch 2.11.0` on Linux IS the CUDA build -- its metadata requires
  `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, `nvidia-cusparselt-cu13`,
  `nvidia-nvshmem-cu13` and `triton`. A "CPU" lock that resolved from PyPI
  would have installed several GB of CUDA 13 on the machines least able to
  use it, next to CUDA 12 pins elsewhere in the same file.
* **Only torch's own distributions get a build rewrite.** The first version
  applied the Darwin-marker rule to every exact pin, so any macOS-only pin
  (`pyobjc-*`, `tensorflow-metal`) lost its marker and broke the lock on
  Windows and Linux.
* **Exact pins are checked against the environment too.** They were skipped --
  and the only exact pins in this project are the torch lines, i.e. exactly
  the ones this work exists to get right.
* **`-e`, `-r`, `-f` are options.** `_distribution` read `-e` as a package name,
  because a hyphen is legal inside one; those lines vanished from the lock and
  the file gained comments claiming a distribution named `-e` was missing.
* **`--index-url` counts as the PyTorch index too**, not only
  `--extra-index-url` -- PyTorch's own install page recommends the former.
* **A dotted local label must not be truncated.** `2.11.0+cu126.post1` became
  `2.11.0.post1`, a different release that may not exist.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.pin_requirements import CPU_INDEX, generate  # noqa: E402

SOURCE = """\
# a comment
kikuchipy>=0.11.3
dask[array]  # extras must survive
--extra-index-url https://download.pytorch.org/whl/cu126
torch==2.11.0+cu126 ; platform_system != "Darwin"
torchvision==0.26.0+cu126 ; platform_system != "Darwin"
torch==2.11.0 ; platform_system == "Darwin"
torchvision==0.26.0 ; platform_system == "Darwin"
cupy-cuda12x>=14.0 ; platform_system != "Darwin"
nvidia-cublas-cu12 ; platform_system != "Darwin"
triton-windows>=3.6 ; sys_platform == "win32"
definitely-not-installed-xyz
"""


def lines_of(cpu: bool, source: str = SOURCE) -> list[str]:
    return generate(source, cpu=cpu)[0]


# --------------------------------------------------------------------------
# GPU lock
# --------------------------------------------------------------------------

def test_gpu_lock_keeps_the_platform_split_verbatim():
    lines = lines_of(cpu=False)
    assert 'torch==2.11.0+cu126 ; platform_system != "Darwin"' in lines
    assert 'torch==2.11.0 ; platform_system == "Darwin"' in lines
    assert "--extra-index-url https://download.pytorch.org/whl/cu126" in lines


def test_gpu_lock_pins_loose_requirements_from_the_environment():
    kikuchipy = [ln for ln in lines_of(cpu=False) if ln.startswith("kikuchipy==")]
    assert len(kikuchipy) == 1, f"expected exactly one pinned kikuchipy, got {kikuchipy}"


def test_extras_are_preserved_when_pinning():
    assert any(ln.startswith("dask[array]==") for ln in lines_of(cpu=False))


# --------------------------------------------------------------------------
# CPU lock -- the flavour rewrite
# --------------------------------------------------------------------------

def test_cpu_lock_points_at_the_cpu_index_rather_than_dropping_it():
    """Dropping the index sends torch to PyPI, whose Linux wheel is CUDA."""
    lines = lines_of(cpu=True)
    assert f"--extra-index-url {CPU_INDEX}" in lines
    assert not any("whl/cu126" in ln for ln in lines)


def test_cpu_lock_names_the_cpu_build_explicitly():
    lines = lines_of(cpu=True)
    assert 'torch==2.11.0+cpu ; platform_system != "Darwin"' in lines
    assert 'torchvision==0.26.0+cpu ; platform_system != "Darwin"' in lines


def test_cpu_lock_leaves_macos_on_the_plain_build():
    """macOS has no CUDA and no +cpu wheel; the plain build is correct there."""
    assert 'torch==2.11.0 ; platform_system == "Darwin"' in lines_of(cpu=True)


def test_cpu_lock_drops_the_cuda_only_distributions():
    joined = "\n".join(lines_of(cpu=True))
    for forbidden in ("cupy-cuda", "nvidia-", "triton-windows"):
        assert forbidden not in joined, f"{forbidden!r} leaked into the CPU lock"


def test_only_torch_distributions_get_a_build_rewrite():
    """A macOS-only pin must keep its marker and its version."""
    source = 'pyobjc-core==10.3 ; platform_system == "Darwin"\n'
    assert lines_of(cpu=True, source=source) == ['pyobjc-core==10.3 ; platform_system == "Darwin"']


def test_index_url_counts_as_the_pytorch_index_too():
    source = (
        "--index-url https://download.pytorch.org/whl/cu126\n"
        'torch==2.11.0+cu126 ; platform_system != "Darwin"\n'
    )
    lines = lines_of(cpu=True, source=source)
    assert f"--index-url {CPU_INDEX}" in lines
    assert 'torch==2.11.0+cpu ; platform_system != "Darwin"' in lines


def test_a_dotted_local_label_is_not_truncated_into_another_release():
    source = "torch==2.11.0+cu126.post1\n"
    assert lines_of(cpu=True, source=source) == ["torch==2.11.0+cpu"]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def test_uninstalled_requirements_are_reported_not_silently_dropped():
    lines, missing, _ = generate(SOURCE, cpu=False)
    assert "definitely-not-installed-xyz" in missing
    assert not any("definitely-not-installed-xyz" in ln for ln in lines)


def test_an_exact_pin_that_disagrees_with_the_environment_is_reported():
    """The torch lines are the only exact pins here; they must not go unchecked."""
    source = "kikuchipy==0.0.1-not-what-is-installed\n"
    _, _, disagreeing = generate(source, cpu=False)
    assert any("kikuchipy" in d for d in disagreeing), disagreeing


def test_an_exact_pin_of_an_uninstalled_package_is_reported_as_missing():
    source = "definitely-not-installed-xyz==1.2.3\n"
    _, missing, _ = generate(source, cpu=False)
    assert "definitely-not-installed-xyz" in missing


@pytest.mark.parametrize("option", ["-e .", "-r other.txt", "-f https://example.com/wheels"])
def test_short_form_options_are_carried_through_not_read_as_packages(option):
    lines, missing, _ = generate(option + "\n", cpu=False)
    assert lines == [option]
    assert missing == []


@pytest.mark.parametrize("cpu", [False, True])
def test_comments_and_blank_lines_never_reach_the_lock(cpu):
    lines = lines_of(cpu=cpu)
    assert all(ln.strip() for ln in lines)
    assert not any(ln.lstrip().startswith("#") for ln in lines)


def test_the_lock_files_carry_no_machine_specific_path():
    """A lock file ships to every tester and is published on GitHub.

    The generator first wrote `# Pinned from: <sys.executable>` -- an absolute
    path with the maintainer's username, against the project's own rule. It also
    made the file's digest machine-specific, so regenerating it elsewhere would
    look like a dependency change to the update check even when no package moved.
    """
    import re
    for name in ("requirements-lock-cpu.txt", "requirements-lock-gpu.txt"):
        path = REPO_ROOT / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"[A-Za-z]:\\", text), f"{name} contains a Windows path"
        assert "/Users/" not in text and "/home/" not in text, f"{name} contains a home directory"
        assert "sys.executable" not in text


# --------------------------------------------------------------------------
# Ranges: requirements.txt caps kikuchipy/orix/pyebsdindex until validated
def _with_versions(monkeypatch, installed):
    import scripts.pin_requirements as pr

    def fake(dist):
        if dist in installed:
            return installed[dist]
        raise pr.PackageNotFoundError(dist)

    monkeypatch.setattr(pr, "version", fake)
    return pr


def test_a_version_outside_the_declared_range_is_reported(monkeypatch):
    """Measured: generate() from an env with kikuchipy 0.13.1 pinned it without
    a word, although requirements.txt says <0.12."""
    pr = _with_versions(monkeypatch, {"kikuchipy": "0.13.1", "orix": "0.14.1"})
    lines, _, disagreeing = pr.generate(
        "kikuchipy>=0.11.3,<0.12\norix>=0.14,<0.15\n", cpu=False)
    assert lines == ["kikuchipy==0.13.1", "orix==0.14.1"]
    assert [d for d in disagreeing if d.startswith(pr.OUT_OF_RANGE)] == [
        f"{pr.OUT_OF_RANGE}: kikuchipy 0.13.1 not in '>=0.11.3,<0.12'"]


def test_main_refuses_to_write_a_lock_outside_the_range(monkeypatch, tmp_path):
    pr = _with_versions(monkeypatch, {"kikuchipy": "0.13.1"})
    src = tmp_path / "req.txt"
    src.write_text("kikuchipy>=0.11.3,<0.12\n", encoding="utf-8")
    out = tmp_path / "lock.txt"
    assert pr.main(["--cpu", "-r", str(src), "-o", str(out)]) == 1
    assert not out.exists()


def test_extras_and_unbounded_lines_are_not_range_violations(monkeypatch):
    pr = _with_versions(monkeypatch, {"dask": "2026.8.0", "numpy": "2.4.6"})
    _, _, disagreeing = pr.generate("dask[array]>=2024\nnumpy\n", cpu=False)
    assert not [d for d in disagreeing if d.startswith(pr.OUT_OF_RANGE)]
