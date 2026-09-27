"""environment-macos.yml must carry every package requirements.txt installs on a Mac.

On macOS the environment is built and updated from environment-macos.yml alone
(pip would bring back a second OpenMP runtime, see updater.python_dependency_command).
A package added to requirements.txt but not to the yml would therefore never
reach a Mac, without any error.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: pip distribution name -> conda-forge package name, where they differ.
CONDA_NAME = {
    "matplotlib": "matplotlib-base",
    "pyqt5": "pyqt",
    "ray": "ray-core",
    "torch": "pytorch",
    "uvicorn": "uvicorn-standard",
}


def _darwin_requirements() -> set[str]:
    from packaging.markers import Marker

    darwin = {"platform_system": "Darwin", "sys_platform": "darwin"}
    names = set()
    for raw in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        body, _, marker = line.partition(";")
        if marker.strip() and not Marker(marker.strip()).evaluate(darwin):
            continue
        name = re.match(r"[A-Za-z0-9._-]+", body.strip()).group(0).lower()
        names.add(CONDA_NAME.get(name, name))
    return names


def _yml_packages() -> set[str]:
    text = (REPO_ROOT / "environment-macos.yml").read_text(encoding="utf-8")
    deps = text.split("dependencies:", 1)[1]
    return {
        re.match(r"[A-Za-z0-9._-]+", m.group(1)).group(0).lower()
        for m in re.finditer(r"^\s*-\s*(\S+)", deps, re.M)
    }


def test_every_darwin_requirement_is_in_the_macos_environment():
    missing = sorted(_darwin_requirements() - _yml_packages())
    assert not missing, (
        f"in requirements.txt for macOS but not in environment-macos.yml: {missing}"
    )


def test_no_cuda_packages_reach_the_macos_environment():
    yml = _yml_packages()
    assert not {p for p in yml if p.startswith(("cupy", "nvidia-", "triton"))}
