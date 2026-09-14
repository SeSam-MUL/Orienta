"""The EMsoft install wizard must never put the sudo password where it does
not belong.

Three defects (tasks/install-security-audit-2026-09-09.md, M1 + O1):

1. The wrapper defined ``sudo() { echo "$PW" | command sudo -S "$@"; }``. The
   pipe REPLACES stdin, so every ``echo <path> | sudo tee <file>`` in
   install_emsoft.sh wrote either nothing or — once sudo's timestamp was warm,
   i.e. always after the pre-check — the password itself into /etc/fstab and
   the OpenCL ICD files. That is the "ICD files contained 1234" finding.
2. The wrapper file holding the password was chmod 0755 on Linux/macOS for the
   whole 30-90 minute build.
3. Passwords were spliced into ``bash -c`` command lines, visible in ``ps``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from backend.api.routes import install


PASSWORD = "s3cr3t!pw with space"


def _find_bash() -> str | None:
    """A POSIX bash to run the wrapper under. On Windows that is Git's bash —
    C:\\Windows\\System32\\bash.exe is WSL and would need the distro."""
    if env := os.environ.get("ORIENTA_TEST_BASH"):
        return env
    if sys.platform == "win32":
        git = shutil.which("git")
        if git:
            # git may resolve to <Git>/cmd/git.exe or <Git>/mingw64/bin/git.exe;
            # walk up until the Git root with its bash shows up.
            for root in Path(git).resolve().parents:
                for cand in (root / "bin" / "bash.exe", root / "usr" / "bin" / "bash.exe"):
                    if cand.exists():
                        return str(cand)
        return None
    return shutil.which("bash")


BASH = _find_bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="no POSIX bash available")


FAKE_SUDO = r"""#!/bin/sh
# Stand-in for sudo. Understands -A (ask via $SUDO_ASKPASS), -n, -v, -k, --.
# Refuses -S loudly: that is the option whose stdin side effect we are testing
# against. Compares the askpass answer with $FAKE_SUDO_EXPECTED.
ask=0; validate=0
while [ $# -gt 0 ]; do
  case "$1" in
    -A) ask=1 ;;
    -n|-k) ;;
    -v) validate=1 ;;
    -S) echo "fake sudo: -S must not be used" >&2; exit 99 ;;
    --) shift; break ;;
    -*) ;;
    *) break ;;
  esac
  shift
done
if [ "$ask" = 1 ]; then
  [ -n "$SUDO_ASKPASS" ] || { echo "fake sudo: SUDO_ASKPASS unset" >&2; exit 98; }
  pw="$("$SUDO_ASKPASS")"
  [ "$pw" = "$FAKE_SUDO_EXPECTED" ] || { echo "fake sudo: wrong password" >&2; exit 1; }
fi
[ "$validate" = 1 ] && exit 0
exec "$@"
"""


def _posix(p: Path) -> str:
    return str(p).replace("\\", "/")


def _run_wrapper(tmp_path: Path, script: str, password: str, expected: str):
    fake_dir = tmp_path / "fakebin"
    fake_dir.mkdir()
    (fake_dir / "sudo").write_text(FAKE_SUDO, newline="\n")
    os.chmod(fake_dir / "sudo", 0o755)
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()

    wrapper = install.build_install_wrapper(script, password)
    # The helper is pinned to /tmp (see build_install_wrapper); under Git's
    # bash on Windows /tmp is the user's temp dir. For the leftover check the
    # test looks there through the same bash.
    wrapper_path = install.write_wrapper_file(wrapper)
    try:
        env = dict(os.environ)
        env["PATH"] = _posix(fake_dir) + os.pathsep + env.get("PATH", "")
        env["FAKE_SUDO_EXPECTED"] = expected
        env["TMPDIR"] = _posix(tmpdir)
        proc = subprocess.run(
            [BASH, _posix(Path(wrapper_path))],
            capture_output=True, text=True, env=env, timeout=60,
        )
    finally:
        os.unlink(wrapper_path)
    return proc, tmpdir


@needs_bash
def test_sudo_tee_writes_the_path_not_the_password(tmp_path):
    out = tmp_path / "pocl.icd"
    fstab = tmp_path / "fstab"
    script = (
        f'echo "libpocl.so" | sudo tee {_posix(out)} >/dev/null\n'
        f"echo '/swapfile none swap sw 0 0' | sudo tee -a {_posix(fstab)} >/dev/null\n"
    )
    proc, tmpdir = _run_wrapper(tmp_path, script, PASSWORD, PASSWORD)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out.read_text().strip() == "libpocl.so"
    assert fstab.read_text().strip() == "/swapfile none swap sw 0 0"
    for f in (out, fstab):
        assert PASSWORD not in f.read_text()
    # The askpass helper must not outlive the run (it lives in bash's /tmp).
    left = subprocess.run([BASH, "-c", "ls /tmp/_orienta_askpass.* 2>/dev/null"],
                          capture_output=True, text=True, timeout=30).stdout.strip()
    assert left == "", f"askpass helper left behind: {left}"


@needs_bash
def test_wrong_password_aborts_before_the_script_runs(tmp_path):
    marker = tmp_path / "ran"
    script = f"touch {_posix(marker)}\n"
    proc, _ = _run_wrapper(tmp_path, script, "wrong", PASSWORD)
    assert proc.returncode != 0
    assert "sudo password is incorrect" in proc.stdout
    assert not marker.exists()


@needs_bash
def test_stdin_of_the_wrapped_script_is_untouched(tmp_path):
    """A `read` inside the script (or a `| sudo tee`) sees the script's own
    stdin, never a password stream."""
    out = tmp_path / "seen"
    script = f'printf "%s" "$(sudo cat)" > {_posix(out)} </dev/null\n'
    proc, _ = _run_wrapper(tmp_path, script, PASSWORD, PASSWORD)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out.read_text() == ""


def test_wrapper_never_uses_sudo_dash_s():
    wrapper = install.build_install_wrapper("true\n", PASSWORD)
    assert "sudo -S" not in wrapper
    assert "-S " not in wrapper


def test_password_with_newline_is_rejected():
    with pytest.raises(ValueError):
        install.build_install_wrapper("true\n", "a\nb")


def test_wrapper_file_keeps_mkstemp_permissions(monkeypatch):
    calls = []
    monkeypatch.setattr(install.os, "chmod", lambda *a, **k: calls.append(a))
    path = install.write_wrapper_file("#!/usr/bin/env bash\ntrue\n")
    try:
        assert os.path.exists(path)
        assert calls == [], "the wrapper holds the password — never widen its mode"
        if os.name == "posix":
            assert os.stat(path).st_mode & 0o777 == 0o600
    finally:
        os.unlink(path)


# --------------------------------------------------------------------------
# Passwords go through stdin, never through argv
# --------------------------------------------------------------------------

class _Recorder:
    def __init__(self, stdout="NOTFOUND", returncode=0):
        self.calls = []
        self.stdout = stdout
        self.returncode = returncode

    def __call__(self, args, **kw):
        self.calls.append((list(args), kw))
        return subprocess.CompletedProcess(args, self.returncode, stdout=self.stdout, stderr="")

    def argv_text(self):
        return "\n".join(" ".join(a) for a, _ in self.calls)

    def inputs(self):
        out = []
        for _, kw in self.calls:
            i = kw.get("input")
            if i:
                out.append(i.decode("utf-8") if isinstance(i, bytes) else i)
        return out


@pytest.fixture
def on_windows(monkeypatch):
    monkeypatch.setattr(install, "_detect_platform",
                        lambda: {"os": "windows", "arch": "amd64", "needs_wsl": True})


def test_create_user_passes_password_on_stdin(monkeypatch, on_windows):
    rec = _Recorder()
    monkeypatch.setattr(install.subprocess, "run", rec)
    res = install._create_user_sync("bob", PASSWORD, "Ubuntu-22.04")
    assert res["success"], res
    assert PASSWORD not in rec.argv_text()
    assert any("bob:" + PASSWORD in i for i in rec.inputs())


def test_reset_password_passes_password_on_stdin(monkeypatch, on_windows):
    rec = _Recorder()
    monkeypatch.setattr(install.subprocess, "run", rec)
    res = install._reset_password_sync("bob", PASSWORD, "")
    assert res["success"], res
    assert PASSWORD not in rec.argv_text()
    assert any("bob:" + PASSWORD in i for i in rec.inputs())


def test_validate_password_uses_stdin_and_a_fresh_check(monkeypatch, on_windows):
    rec = _Recorder(stdout="")
    monkeypatch.setattr(install.subprocess, "run", rec)
    assert install._validate_password_sync(PASSWORD) is True
    assert PASSWORD not in rec.argv_text()
    assert any(PASSWORD in i for i in rec.inputs())
    # -k: a warm sudo timestamp must not make a wrong password look right.
    assert "-k" in rec.argv_text()


def test_validate_password_targets_the_named_distro(monkeypatch, on_windows):
    rec = _Recorder(stdout="")
    monkeypatch.setattr(install.subprocess, "run", rec)
    assert install._validate_password_sync(PASSWORD, "orienta-test") is True
    args, _ = rec.calls[0]
    assert args[:3] == ["wsl", "-d", "orienta-test"]
    # A name that fails validation is refused before any command runs — and
    # as ValueError, not as "wrong password" (the route turns it into 400).
    rec2 = _Recorder(stdout="")
    monkeypatch.setattr(install.subprocess, "run", rec2)
    with pytest.raises(ValueError):
        install._validate_password_sync(PASSWORD, "bad name; rm")
    assert rec2.calls == []


def test_stdin_is_passed_as_bytes_not_text(monkeypatch, on_windows):
    """text=True would turn the trailing '\\n' into '\\r\\n' on Windows and
    chpasswd would store a password ending in a carriage return."""
    rec = _Recorder()
    monkeypatch.setattr(install.subprocess, "run", rec)
    install._create_user_sync("bob", PASSWORD, "Ubuntu-22.04")
    stdin_calls = [(a, kw) for a, kw in rec.calls if kw.get("input") is not None]
    assert stdin_calls, "no call passed data on stdin"
    for _, kw in stdin_calls:
        assert isinstance(kw["input"], bytes)
        assert not kw.get("text"), "text mode re-introduces the CRLF bug"
        assert not kw["input"].endswith(b"\r\n")


def test_websocket_handler_no_longer_widens_the_wrapper_mode():
    import inspect
    src = inspect.getsource(install.ws_install_emsoft)
    assert "os.chmod(" not in src, "the wrapper holds the password — mkstemp's 0600 must stand"
    assert "DEVNULL" in src, "the build must not inherit the backend's stdin"


def test_wrapper_precheck_and_teardown_shape():
    wrapper = install.build_install_wrapper("true\n", PASSWORD)
    # Pre-check with -k: a warm timestamp from an earlier terminal must not
    # let a typo through and fail 20 minutes into the build.
    assert "sudo -k -A -v" in wrapper
    # Cleanup on signals too, not only on a clean exit.
    assert "trap _orienta_cleanup EXIT INT TERM HUP" in wrapper
    # Non-interactive apt: with stdin at EOF a conffile prompt must not stall.
    assert "DEBIAN_FRONTEND=noninteractive" in wrapper
    # The helper lives in the distro's /tmp, never under a drvfs TMPDIR.
    assert "mktemp /tmp/_orienta_askpass" in wrapper


def test_validate_password_reports_failure(monkeypatch, on_windows):
    rec = _Recorder(stdout="", returncode=1)
    monkeypatch.setattr(install.subprocess, "run", rec)
    assert install._validate_password_sync("nope") is False
