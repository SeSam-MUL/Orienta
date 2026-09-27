"""Tests for the EMsoft startup reaper.

The reaper kills EMsoft compute processes left running inside the WSL2 VM by
a previous backend session (a backend restart tears down only the Windows
wsl.exe relay; the VM-side process keeps running forever). These tests drive
``reap_orphaned_emsoft_processes`` with an injected fake runner so no real
WSL is required.
"""

import pytest

from simulation.simulation_controller import reap_orphaned_emsoft_processes


class _FakeProc:
    def __init__(self, stdout="", returncode=0):
        self.stdout = stdout
        self.returncode = returncode


class _FakeRunner:
    """Records commands; returns canned `ps` output, succeeds on `kill`."""

    def __init__(self, ps_output="", ps_returncode=0, ps_proc=...):
        self.ps_output = ps_output
        self.ps_returncode = ps_returncode
        self.ps_proc = ps_proc  # sentinel -> build from output; else use as-is (e.g. None)
        self.calls = []

    def __call__(self, cmd, timeout=15):
        self.calls.append(cmd)
        if cmd.startswith("ps "):
            if self.ps_proc is not ...:
                return self.ps_proc
            return _FakeProc(self.ps_output, self.ps_returncode)
        if cmd.startswith("kill "):
            return _FakeProc("", 0)
        return _FakeProc("", 0)

    @property
    def kill_calls(self):
        return [c for c in self.calls if c.startswith("kill ")]


# The binary directory these fake processes come from. Passed explicitly so
# the tests never read the developer's own emsphinx_config.ini -- that file is
# gitignored, so on a fresh checkout (and on the Linux runner) the reaper would
# find no directory, treat every process as somebody else's, and reap nothing.
BIN = "/home/operator/emsoft/builds/EMsoft-Release/Bin"

# A realistic-looking `ps -eo pid=,etimes=,args=` dump (pid, elapsed-seconds, args)
_PS_WITH_TWO_ORPHANS = (
    "   1240 36000 /home/operator/emsoft/builds/EMsoft-Release/Bin/EMEBSDmasterSHT EMEBSDmasterSHT_Al-Cu-Fe-Si.nml\n"
    "   3532  1460 /home/operator/emsoft/builds/EMsoft-Release/Bin/EMEBSDmasterSHT EMEBSDmasterSHT_Al-Fe-Cu-Si.nml\n"
    "      1 99999 /sbin/init\n"
    "    210    88 /usr/bin/dbus-daemon --system\n"
)

_PS_NO_EMSOFT = (
    "      1 99999 /sbin/init\n"
    "    210    88 /usr/bin/dbus-daemon --system\n"
    "    363    87 /lib/systemd/systemd --user\n"
)


def test_reaps_emsoft_and_issues_single_kill():
    runner = _FakeRunner(ps_output=_PS_WITH_TWO_ORPHANS)
    reaped = reap_orphaned_emsoft_processes(runner=runner, emsoft_bin_dir=BIN)

    pids = {r["pid"] for r in reaped}
    assert pids == {"1240", "3532"}
    assert all(r["name"] == "EMEBSDmasterSHT" for r in reaped)
    # elapsed seconds parsed
    assert {r["elapsed_sec"] for r in reaped} == {36000, 1460}
    # exactly one batched kill with both pids
    assert len(runner.kill_calls) == 1
    killed = runner.kill_calls[0].split()[2:]  # ["kill","-9","1240","3532"]
    assert set(killed) == {"1240", "3532"}


def test_no_emsoft_means_no_kill():
    runner = _FakeRunner(ps_output=_PS_NO_EMSOFT)
    reaped = reap_orphaned_emsoft_processes(runner=runner, emsoft_bin_dir=BIN)

    assert reaped == []
    assert runner.kill_calls == []


def test_runner_returning_none_is_safe():
    """WSL unavailable -> runner returns None -> no crash, nothing reaped."""
    runner = _FakeRunner(ps_proc=None)
    reaped = reap_orphaned_emsoft_processes(runner=runner, emsoft_bin_dir=BIN)

    assert reaped == []
    assert runner.kill_calls == []


def test_ps_nonzero_returncode_is_safe():
    runner = _FakeRunner(ps_output="garbage", ps_returncode=1)
    reaped = reap_orphaned_emsoft_processes(runner=runner, emsoft_bin_dir=BIN)

    assert reaped == []
    assert runner.kill_calls == []


def test_label_picks_most_specific_binary():
    ps = (
        "   100   10 /opt/EMsoft/Bin/EMEBSDmasterOpenCL EMEBSDmasterOpenCL_X.nml\n"
        "   101   20 /opt/EMsoft/Bin/EMMCOpenCL EMMCOpenCL_X.nml\n"
        "   102   30 /opt/EMsoft/Bin/EMMC EMMC_X.nml\n"
        "   103   40 /opt/EMsoft/Bin/EMEBSDmaster EMEBSDmaster_X.nml\n"
    )
    runner = _FakeRunner(ps_output=ps)
    reaped = reap_orphaned_emsoft_processes(runner=runner, emsoft_bin_dir=BIN)

    by_pid = {r["pid"]: r["name"] for r in reaped}
    assert by_pid == {
        "100": "EMEBSDmasterOpenCL",
        "101": "EMMCOpenCL",
        "102": "EMMC",
        "103": "EMEBSDmaster",
    }


def test_malformed_lines_skipped():
    ps = (
        "\n"
        "   abc 10 /opt/EMsoft/Bin/EMEBSDmasterSHT foo.nml\n"   # non-numeric pid
        "   200\n"                                              # too few fields
        "   201 50 /opt/EMsoft/Bin/EMEBSDmasterSHT bar.nml\n"   # valid
    )
    runner = _FakeRunner(ps_output=ps)
    reaped = reap_orphaned_emsoft_processes(runner=runner, emsoft_bin_dir=BIN)

    assert [r["pid"] for r in reaped] == ["201"]


# ---------------------------------------------------------------------------
# platform truth: the reaper and the cancel path off Windows
# ---------------------------------------------------------------------------

from simulation.simulation_controller import (  # noqa: E402
    _pkill_command, _ps_command, _parse_elapsed, _is_ours,
    _configured_emsoft_bin_dir,
)


def test_cancel_reaches_a_native_linux_run():
    # `wsl pkill` on a machine without WSL raises FileNotFoundError, which the
    # caller swallows: cancelling a native run stopped nothing at all.
    assert _pkill_command("EMMC", "win32") == [["wsl", "pkill", "-f", "EMMC"]]
    linux = _pkill_command("EMMC", "linux", own_bin_dir=BIN)
    assert linux and linux[0][0] == "pkill"
    # anchored on OUR binary's full path, not the bare name: `pkill -f EMMC`
    # would also match a colleague's build and "python x.py EMMC_run.h5"
    assert linux[0][-1] == f"^{BIN}/EMMC( |$)"


def test_cancel_stops_nothing_rather_than_guessing_on_linux():
    # Without a known directory there is no way to tell our processes from
    # anyone else's. A cancel that does nothing is recoverable; killing a
    # colleague's twelve-hour run is not.
    assert _pkill_command("EMMC", "linux", own_bin_dir="") == []


def test_cancel_does_nothing_on_macos_rather_than_failing():
    # No prebuilt EMsoft for macOS and no WSL, so there is nothing to kill.
    assert _pkill_command("EMMC", "darwin") == []


def test_ps_asks_macos_for_a_field_it_has():
    # BSD ps has no `etimes`; asking for it drops the column and every line
    # would then be too short for the parser, so nothing would ever be reaped.
    assert "etimes" in _ps_command("linux")
    assert "etime=" in _ps_command("darwin") and "etimes" not in _ps_command("darwin")


@pytest.mark.parametrize("text, seconds", [
    ("3600", 3600),          # procps etimes
    ("05:00", 300),          # BSD mm:ss
    ("01:00:00", 3600),      # BSD hh:mm:ss
    ("2-03:04:05", 183845),  # BSD dd-hh:mm:ss
    ("", -1),
    ("junk", -1),
])
def test_elapsed_is_read_in_both_formats(text, seconds):
    assert _parse_elapsed(text) == seconds


def test_a_colleagues_own_emsoft_build_is_left_alone_on_linux():
    # The reaper runs at startup and kills by binary NAME. On a shared Linux
    # workstation that would also kill an EMsoft run somebody started from
    # their own build in another terminal -- hours of foreign compute.
    assert _is_ours("/opt/orienta/emsoft/Bin/EMMC -x", "/opt/orienta/emsoft/Bin", "linux") is True
    assert _is_ours("/home/other/emsoft/Bin/EMMC -x", "/opt/orienta/emsoft/Bin", "linux") is False
    # unknown own directory: kill nothing, rather than everything
    assert _is_ours("/anywhere/EMMC", "", "linux") is False
    # Windows reaches only the WSL distribution Orienta set up
    assert _is_ours("/anywhere/EMMC", "", "win32") is True


def test_the_reaper_skips_a_foreign_process_but_still_reaps_ours(monkeypatch):
    lines = (
        "111 900 /opt/orienta/emsoft/Bin/EMEBSDmaster -nml x.nml\n"
        "222 900 /home/other/build/Bin/EMEBSDmaster -nml y.nml\n"
    )
    calls = []

    def runner(cmd, timeout=15):
        calls.append(cmd)
        if cmd.startswith("ps "):
            return _FakeProc(stdout=lines)
        return _FakeProc()

    import simulation.simulation_controller as sc
    monkeypatch.setattr(sc.sys, "platform", "linux")
    reaped = reap_orphaned_emsoft_processes(
        runner=runner, emsoft_bin_dir="/opt/orienta/emsoft/Bin")

    assert [r["pid"] for r in reaped] == ["111"]
    assert any(c == "kill -9 111" for c in calls), calls
    assert not any("222" in c for c in calls if c.startswith("kill")), calls


def test_an_unreadable_config_means_no_directory_not_a_crash(tmp_path):
    assert _configured_emsoft_bin_dir(tmp_path / "nope.ini") == ""
    broken = tmp_path / "broken.ini"
    broken.write_text("not an ini at all", encoding="utf-8")
    assert _configured_emsoft_bin_dir(broken) == ""
    good = tmp_path / "good.ini"
    good.write_text("[EMsoftPaths]\nemmc = /opt/o/Bin/EMMC\n", encoding="utf-8")
    assert _configured_emsoft_bin_dir(good) == "/opt/o/Bin"
