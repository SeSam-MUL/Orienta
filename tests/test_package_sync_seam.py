"""The seam: the shell's package sync against a real pip and a real interpreter.

`tests/electron/packageSync.test.js` proves every decision, every argument list
and the order of the steps with a pip that is a script. This test is the other
half: the same `electron/setup/package_sync.js`, run by Node with its real
runner against a real virtual environment holding the packages of a 0.4.6
installation, over the real network. What a scripted pip cannot say is whether
the command line we build is one pip accepts, whether the report it writes is
the one we parse, whether `importlib.metadata` agrees with what pip installed,
and what an environment looks like after pip is killed half-way.

OPT-IN, because it needs a network and builds a ~700 MB environment (3 to 5
minutes the first time):

    ORIENTA_PACKAGE_SYNC_SEAM=1 python -m pytest tests/test_package_sync_seam.py -v

  ORIENTA_PACKAGE_SYNC_HOME   an existing throwaway home whose python/ is a venv
                              at the OLD versions (reused across runs; this test
                              puts it back to them before every scenario)
  ORIENTA_PACKAGE_SYNC_PYTHON the interpreter to build the venv from (default:
                              the one running pytest)

It never touches the real installation: the home is a temporary folder (or the
one named above), and a path under %LOCALAPPDATA%\\Orienta is refused outright.

The lock is a small one -- the three libraries the 0.4.7 lock moves -- because
what is under test is the sync, not the other hundred packages of a full
environment, which pip would not touch anyway (it says so: that was measured).
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import venv
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
DRIVER = REPO / "tests" / "package_sync_seam_driver.js"

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("ORIENTA_PACKAGE_SYNC_SEAM"),
        reason="opt-in: needs a network and a ~700 MB virtual environment "
               "(set ORIENTA_PACKAGE_SYNC_SEAM=1)"),
    pytest.mark.skipif(NODE is None, reason="the sync is JavaScript and needs node"),
    pytest.mark.skipif(sys.platform == "darwin",
                       reason="macOS has its own sync (conda, not pip); this is the pip one"),
]

OLD = {"kikuchipy": "0.11.3", "orix": "0.14.1", "pyebsdindex": "0.3.9.1"}
NEW = {"kikuchipy": "0.13.1", "orix": "0.15.0", "pyebsdindex": "0.3.10.1"}
LOCK = (
    "kikuchipy==0.13.1\n"
    "orix==0.15.0\n"
    "pyebsdindex==0.3.10.1\n"
    "threadpoolctl==3.7.0\n"
)
LOCK_NAME = {"win32": "requirements-lock-cpu.txt"}.get(
    sys.platform, "requirements-lock-linux-cpu.txt")
OWN_FILES = (".packages_lock.json", ".packages_sync.json", ".packages_sync_failed.json")


def _python_in(home: Path) -> Path:
    sub = ("Scripts", "python.exe") if sys.platform == "win32" else ("bin", "python")
    return home / "python" / sub[0] / sub[1]


def _pip(home: Path, *args: str, timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(_python_in(home)), "-m", "pip", *args, "--disable-pip-version-check"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout)


def _versions(home: Path, names=("kikuchipy", "orix", "pyebsdindex")) -> dict:
    code = ("import json, sys; from importlib import metadata as m; "
            "out = {}\nfor n in sys.argv[1:]:\n"
            "    try: out[n] = m.version(n)\n"
            "    except m.PackageNotFoundError: out[n] = None\n"
            "print(json.dumps(out))")
    out = subprocess.run(
        [str(_python_in(home)), "-c", code, *names],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def _freeze(home: Path) -> str:
    out = _pip(home, "freeze")
    assert out.returncode == 0, out.stderr
    return out.stdout


def _online() -> bool:
    try:
        with socket.create_connection(("pypi.org", 443), timeout=8):
            return True
    except OSError:
        return False


def _specs(versions: dict) -> list[str]:
    return [f"{name}=={version}" for name, version in versions.items()]


@pytest.fixture(scope="module")
def home(tmp_path_factory):
    if not _online():
        pytest.skip("no route to pypi.org")
    given = os.environ.get("ORIENTA_PACKAGE_SYNC_HOME")
    if given:
        h = Path(given)
        assert (h / "python").is_dir(), f"{h}\\python is not a virtual environment"
    else:
        h = tmp_path_factory.mktemp("orienta-seam-home")
        base = os.environ.get("ORIENTA_PACKAGE_SYNC_PYTHON")
        if base:
            subprocess.run([base, "-m", "venv", str(h / "python")], check=True)
        else:
            venv.create(h / "python", with_pip=True)
        built = _pip(h, "install", "--only-binary", ":all:", "--no-cache-dir",
                     "--progress-bar", "off", *_specs(OLD), "threadpoolctl==3.7.0")
        assert built.returncode == 0, built.stderr[-2000:]

    # NEVER the real installation. A new build launched against it would sync it.
    real = os.environ.get("LOCALAPPDATA", "")
    if real:
        real_home = (Path(real) / "Orienta").resolve()
        resolved = h.resolve()
        assert resolved != real_home and real_home not in resolved.parents, (
            f"{resolved} is the real installation; this test must use a throwaway home")

    (h / "runtime").mkdir(exist_ok=True)
    (h / ".install_mode").write_text("cpu\n", encoding="utf-8", newline="")
    return h


@pytest.fixture()
def old_home(home):
    """The home, put back to a 0.4.6 installation: old packages, no records."""
    for name in OWN_FILES:
        (home / name).unlink(missing_ok=True)
    shutil.rmtree(home / "setup-tmp", ignore_errors=True)
    (home / "runtime" / LOCK_NAME).write_text(LOCK, encoding="utf-8", newline="")
    if _versions(home) != OLD:
        back = _pip(home, "install", "--force-reinstall", "--no-deps", "--only-binary", ":all:",
                    "--no-cache-dir", "--progress-bar", "off", *_specs(OLD))
        assert back.returncode == 0, back.stderr[-2000:]
    assert _versions(home) == OLD
    return home


def _start(home: Path, **env) -> dict:
    """One start of the sync, as a separate process, the way an app start is."""
    merged = {**os.environ, **env}
    out = subprocess.run(
        [NODE, str(DRIVER), str(REPO), str(home), str(_python_in(home))],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=merged, timeout=1200)
    assert out.returncode == 0, f"driver exited {out.returncode}\n{out.stderr}\n{out.stdout[-2000:]}"
    return json.loads(out.stdout)


def _record(home: Path, name: str) -> dict | None:
    f = home / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def _log(started: dict) -> str:
    return "\n".join(started["logs"])


OFFLINE = {
    "PIP_INDEX_URL": "http://127.0.0.1:9/simple",
    "ORIENTA_PYTORCH_INDEX_CPU": "http://127.0.0.1:9/",
}


def test_an_old_installation_is_brought_to_the_lock_and_the_second_start_starts_no_pip(old_home):
    first = _start(old_home)
    assert first["result"]["action"] == "synced", _log(first)
    assert _versions(old_home) == NEW
    assert first["phases"] == ["syncing"]
    assert first["notified"] == []
    rec = _record(old_home, ".packages_lock.json")
    assert rec["lock"] == LOCK_NAME and rec["mode"] == "cpu" and rec["how"] == "sync"
    import hashlib
    assert rec["sha256"] == hashlib.sha256(LOCK.encode()).hexdigest()
    assert not (old_home / ".packages_sync.json").exists()
    assert not (old_home / ".packages_sync_failed.json").exists()
    assert not (old_home / "setup-tmp" / "package-sync-report.json").exists()
    # pip was asked once, then it installed, then everything was checked
    kinds = [c["kind"] for c in first["calls"]]
    assert kinds[0] == "dry" and "install" in kinds
    assert "reinstall" not in kinds

    # the libraries really run, not merely carry the right metadata
    ok = subprocess.run([str(_python_in(old_home)), "-c", "import kikuchipy, orix, pyebsdindex"],
                        capture_output=True, text=True, timeout=180)
    assert ok.returncode == 0, ok.stderr[-1500:]

    # and the next start does not start a single process
    second = _start(old_home)
    assert second["result"] == {"ok": True, "action": "fastpath"}
    assert second["calls"] == []
    assert second["phases"] == []


def test_offline_the_environment_is_untouched_and_the_user_hears_it_once(old_home):
    before = _freeze(old_home)

    first = _start(old_home, **OFFLINE)
    assert first["result"]["action"] == "failed" and first["result"]["reason"] == "network", _log(first)
    assert _freeze(old_home) == before, "an unreachable index must leave pip freeze identical"
    assert len(first["notified"]) == 1
    failure = _record(old_home, ".packages_sync_failed.json")
    assert failure["reason"] == "network" and failure["count"] == 1 and failure["shown"] is True
    assert _record(old_home, ".packages_lock.json") is None
    assert not (old_home / ".packages_sync.json").exists()

    # the next start asks again (the network may be back) but says nothing new
    second = _start(old_home, **OFFLINE)
    assert second["result"]["reason"] == "network"
    assert second["notified"] == []
    assert [c["kind"] for c in second["calls"]] == ["dry"]
    assert _record(old_home, ".packages_sync_failed.json")["count"] == 2
    assert _freeze(old_home) == before

    # and when the network is back, it simply works and forgets the failure
    third = _start(old_home)
    assert third["result"]["action"] == "synced", _log(third)
    assert not (old_home / ".packages_sync_failed.json").exists()
    assert _versions(old_home) == NEW


def test_closing_the_window_during_the_install_ends_pip_and_the_next_start_repairs(old_home):
    first = _start(old_home, SEAM_KILL="close")
    assert first["killed"] is True, "the install never reached the point the test kills at\n" + _log(first)
    assert first["result"] == {"cancelled": True}
    # pip is not left writing, nobody is told, nothing is recorded -- but the
    # marker says an install was in flight
    assert (old_home / ".packages_sync.json").exists()
    assert _record(old_home, ".packages_lock.json") is None
    assert _record(old_home, ".packages_sync_failed.json") is None
    assert first["notified"] == []
    marker = _record(old_home, ".packages_sync.json")
    assert marker["from"] == OLD and marker["to"] == NEW

    second = _start(old_home)
    assert second["result"]["action"] == "synced", _log(second)
    assert second["result"]["how"] == "repair"
    assert "reinstall" in [c["kind"] for c in second["calls"]]
    assert _versions(old_home) == NEW
    assert not (old_home / ".packages_sync.json").exists()
    assert _record(old_home, ".packages_lock.json")["how"] == "repair"

    third = _start(old_home)
    assert third["result"]["action"] == "fastpath" and third["calls"] == []


def test_pip_killed_after_it_removed_a_package_is_put_back_and_verified(old_home):
    first = _start(old_home, SEAM_KILL="pip")
    assert first["killed"] is True, "pip never got as far as removing a package\n" + _log(first)
    # The app was alive. The environment was missing a library for a moment; by
    # the end of the same start it is whole, verified and recorded.
    assert first["result"]["action"] == "synced", _log(first)
    assert first["result"]["how"] == "repair"
    kinds = [c["kind"] for c in first["calls"]]
    assert "reinstall" in kinds and kinds.index("reinstall") > kinds.index("install")
    assert _versions(old_home) == NEW
    ok = subprocess.run([str(_python_in(old_home)), "-c", "import kikuchipy, orix, pyebsdindex"],
                        capture_output=True, text=True, timeout=180)
    assert ok.returncode == 0, ok.stderr[-1500:]
    assert not (old_home / ".packages_sync.json").exists()
    assert _record(old_home, ".packages_lock.json")["how"] == "repair"
