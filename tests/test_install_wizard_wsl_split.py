"""WSL distributions are registered PER USER (HKCU + %LOCALAPPDATA%). Only the
Windows feature itself is machine-wide.

The wizard used to run ``wsl --install -d <distro>`` in ONE elevated window.
When a different admin account answers the UAC prompt (over-the-shoulder
elevation on a lab PC), that window runs as the admin, and Ubuntu lands in the
admin's profile — invisible to the actual user ("WSL installed cleanly, but no
Ubuntu"). So: elevate only for the feature; install the distro unelevated as
the user who will use it. When the feature is already present, no UAC at all.
"""

from __future__ import annotations

import subprocess

import pytest

from backend.api.routes import install


@pytest.fixture
def on_windows(monkeypatch):
    monkeypatch.setattr(install, "_detect_platform",
                        lambda: {"os": "windows", "arch": "amd64", "needs_wsl": True})


class _Popen:
    def __init__(self):
        self.calls = []

    def __call__(self, args, **kw):
        self.calls.append((list(args), kw))
        return object()


def _ok_run(args, **kw):
    return subprocess.CompletedProcess(args, 0, stdout="", stderr="")


def test_feature_missing_elevates_for_the_feature_only(monkeypatch, on_windows):
    elevated = []
    popen = _Popen()
    monkeypatch.setattr(install, "_wsl_feature_present", lambda: False)
    monkeypatch.setattr(install, "_run_elevated", lambda exe, args: elevated.append((exe, args)) or True)
    monkeypatch.setattr(install.subprocess, "Popen", popen)

    res = install._install_wsl_sync("Ubuntu-22.04", False, "")

    assert res["success"], res
    assert len(elevated) == 1
    _, args = elevated[0]
    assert "--no-distribution" in args
    assert "Ubuntu-22.04" not in args, "the distro must never be installed elevated"
    assert popen.calls == []
    assert "restart" in res["message"].lower()
    assert res.get("stage") == "feature"


def test_feature_present_installs_distro_unelevated(monkeypatch, on_windows):
    popen = _Popen()
    monkeypatch.setattr(install, "_wsl_feature_present", lambda: True)
    monkeypatch.setattr(install, "_run_elevated",
                        lambda *a: pytest.fail("must not elevate when the feature is present"))
    monkeypatch.setattr(install.subprocess, "Popen", popen)

    res = install._install_wsl_sync("Ubuntu-22.04", False, "")

    assert res["success"], res
    assert len(popen.calls) == 1
    args, _ = popen.calls[0]
    assert args[:2] == ["wsl", "--install"]
    # `-d <distro>`: the form both the inbox wsl.exe and WSL 2.x understand.
    assert args[args.index("-d") + 1] == "Ubuntu-22.04"
    assert "--no-launch" in args
    assert res.get("stage") == "distro"


def test_repair_unregisters_then_installs_unelevated(monkeypatch, on_windows):
    runs = []
    popen = _Popen()

    def run(args, **kw):
        runs.append(list(args))
        return _ok_run(args)

    monkeypatch.setattr(install, "_wsl_feature_present", lambda: True)
    monkeypatch.setattr(install.subprocess, "run", run)
    monkeypatch.setattr(install.subprocess, "Popen", popen)

    res = install._install_wsl_sync("Ubuntu-22.04", True, "Ubuntu-22.04")

    assert res["success"], res
    assert ["wsl", "--unregister", "Ubuntu-22.04"] in runs
    assert popen.calls and "Ubuntu-22.04" in popen.calls[0][0]


def test_declined_uac_is_reported(monkeypatch, on_windows):
    monkeypatch.setattr(install, "_wsl_feature_present", lambda: False)
    monkeypatch.setattr(install, "_run_elevated", lambda exe, args: False)
    res = install._install_wsl_sync("Ubuntu-22.04", False, "")
    assert not res["success"]
    assert "administrator" in res["message"].lower()


def test_invalid_distro_name_still_rejected(on_windows):
    res = install._install_wsl_sync("Ubuntu; rm -rf /", False, "")
    assert not res["success"]


def test_a_slow_first_answer_is_not_reported_as_corrupted(monkeypatch, on_windows):
    """A freshly installed distro boots its VM on the first command (measured
    41 s). The status probe used to call a timeout "corrupted" and offer a
    repair — which unregisters the distro."""
    def run(args, **kw):
        if args[:2] == ["wsl", "--list"] and "--quiet" in args:
            return subprocess.CompletedProcess(args, 0, stdout="Ubuntu-22.04\n", stderr="")
        if "whoami" in args:
            raise subprocess.TimeoutExpired(args, kw.get("timeout", 0))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(install.subprocess, "run", run)
    monkeypatch.setattr(install, "_check_nvidia_driver_windows",
                        lambda: {"available": False, "driver_version": "", "cuda_version": "",
                                 "gpu_name": "", "wsl2_ready": False})
    status = install._check_wsl_sync()
    assert status["installed"] is True
    assert status["corrupted"] is False
    assert status["has_user"] is False


def test_wizard_commands_get_the_cold_start_budget(monkeypatch, on_windows):
    seen = []

    def run(args, **kw):
        seen.append((list(args), kw.get("timeout")))
        return subprocess.CompletedProcess(args, 0, stdout="NOTFOUND", stderr="")

    monkeypatch.setattr(install.subprocess, "run", run)
    install._create_user_sync("bob", "s3cret!!", "Ubuntu-22.04")
    first = seen[0]
    assert "id bob" in " ".join(first[0])
    assert first[1] >= 60, "the first command to a new distro boots the VM"
    assert all(t is not None and t >= 30 for _, t in seen), seen


def test_feature_probe_reads_wsl_status(monkeypatch):
    seen = []

    def run(args, **kw):
        seen.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout="Default Version: 2\n", stderr="")

    monkeypatch.setattr(install.subprocess, "run", run)
    assert install._wsl_feature_present() is True
    assert seen and seen[0][:2] == ["wsl", "--status"]


def test_feature_probe_false_when_wsl_is_absent(monkeypatch):
    def run(args, **kw):
        raise FileNotFoundError("wsl")

    monkeypatch.setattr(install.subprocess, "run", run)
    assert install._wsl_feature_present() is False
