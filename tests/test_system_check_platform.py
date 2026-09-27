"""What the system check may say about EMsoft on each platform.

A Mac tester on 2026-09-25 opened Settings and was told, in red:

    Fehler: EMsoft executables not found. Install EMsoft in WSL.

followed by six warnings about EMMCOpenCL, EMEBSDmasterSHT, EMSphInx, a
missing config file and OpenCL-in-WSL. Seven messages, all of them saying the
same thing -- EMsoft is not installed -- one of them naming a Windows feature
that cannot exist on a Mac, on a machine where every part of the app he tried
worked, because Orienta's own simulation engine needs no EMsoft at all.

These tests pin the three platforms apart. They do not touch the discovery
logic: every probe is stubbed to "found nothing", so what is under test is the
reporting.
"""

import sys
from unittest import mock

import pytest

from simulation import system_check as sc


class _NoOpenCL:
    """What the detector reports on a machine with no OpenCL at all."""
    available = False
    devices: list = []
    gpu_devices: list = []
    has_gpu = False
    cpu_count = 8
    binaries: dict = {}


def _status_on(platform: str):
    """Run the check on `platform` with nothing installed and no shell.

    The OpenCL detector is stubbed as well, and that is not incidental: the
    first version of this test left it live, so it probed the RTX 4070 of the
    machine the test ran on and added "GPU has 11.9 GB -- using blocksize=16"
    to the output. The test was measuring this computer instead of the
    platform rule, and would have said something different on the macOS and
    Linux runners.
    """
    from simulation import opencl_detector

    with mock.patch.object(sc.sys, "platform", platform), \
         mock.patch.object(sc, "_run_wsl_cmd", lambda cmd, timeout=15: ("", False)), \
         mock.patch.object(sc, "check_executable", lambda name: (False, "")), \
         mock.patch.object(sc, "check_executable_at_path", lambda d, n: (False, "")), \
         mock.patch.object(sc, "check_wsl_installed", lambda: (True, "distro")), \
         mock.patch.object(sc, "check_wsl_responsive", lambda: True), \
         mock.patch.object(sc, "check_emsoft_config", lambda: (False, {})), \
         mock.patch.object(sc, "check_opencl", lambda: (False, [])), \
         mock.patch.object(opencl_detector, "detect_opencl",
                           lambda emsoft_bin_dir=None: _NoOpenCL()), \
         mock.patch.object(opencl_detector, "recommend_device",
                           lambda status: _RECOMMENDATION_WITH_NO_OPENCL()):
        return sc.check_system_status()


def _RECOMMENDATION_WITH_NO_OPENCL():
    """What recommend_device really returns when there is no OpenCL.

    It early-returns with this one warning (opencl_detector.py, `if not
    status.available`). The first version of this stub returned an EMPTY
    warning list, which quietly deleted the second message before the
    "says it once" assertion could see it -- so that test passed while a real
    Mac still showed two. A stub that is tidier than the thing it stands in
    for does not test the thing it stands in for.
    """
    return {
        "mode": "cpu",
        "warnings": ["No OpenCL detected — CPU-only mode"],
        "warning_items": [],
    }


def _emsoft_messages(status):
    """Everything the check said about EMsoft, errors and warnings alike."""
    said = list(status.errors) + list(status.warnings)
    return [m for m in said
            if any(w in m for w in ("EMsoft", "EMSphInx", "EMMCOpenCL", "EMEBSDmasterSHT", "OpenCL"))]


# ---------------------------------------------------------------------------
# macOS: a state, said once
# ---------------------------------------------------------------------------

def test_macos_says_it_once():
    status = _status_on("darwin")
    assert len(_emsoft_messages(status)) == 1, _emsoft_messages(status)


def test_macos_does_not_call_it_an_error():
    """There is nothing to act on: the app offers no EMsoft install on macOS."""
    status = _status_on("darwin")
    assert status.errors == []


def test_macos_never_mentions_wsl():
    status = _status_on("darwin")
    for message in list(status.errors) + list(status.warnings):
        assert "WSL" not in message, message


def test_macos_says_the_app_works_without_it():
    """The sentence that decides whether the user panics or carries on."""
    (message,) = _emsoft_messages(_status_on("darwin"))
    assert "optional" in message.lower()
    assert "simulation engine" in message.lower()


# ---------------------------------------------------------------------------
# Windows: unchanged -- the wizard can install EMsoft there
# ---------------------------------------------------------------------------

def test_windows_still_reports_an_error_and_names_wsl():
    status = _status_on("win32")
    assert any("Install EMsoft in WSL" in e for e in status.errors)


def test_windows_still_lists_the_individual_binaries():
    # Positive control: if the platform split ever swallowed these everywhere,
    # the macOS tests above would still pass while Windows lost its detail.
    status = _status_on("win32")
    assert any("EMMCOpenCL" in w for w in status.warnings)
    assert any("EMEBSDmasterSHT" in w for w in status.warnings)


# ---------------------------------------------------------------------------
# Linux: detail kept, but it is not told to use a Windows feature
# ---------------------------------------------------------------------------

def test_linux_is_treated_like_macos():
    """Linux joined macOS on 2026-09-25.

    The endpoint does run the install script natively there, but nobody has
    ever run it and the wizard no longer offers it -- so the same rule
    applies: one sentence, not an error, nothing to act on.
    """
    status = _status_on("linux")
    assert status.errors == []
    assert len(_emsoft_messages(status)) == 1, _emsoft_messages(status)
    for message in status.warnings:
        assert "WSL" not in message, message


def test_linux_says_it_in_a_platform_neutral_way():
    """One wording for both, so the two cannot drift apart."""
    (mac,) = _emsoft_messages(_status_on("darwin"))
    (linux,) = _emsoft_messages(_status_on("linux"))
    assert mac == linux
    assert "Mac" not in linux


@pytest.mark.parametrize("platform,expected", [
    ("win32", True),
    ("darwin", False),
    # Linux runs the install script on the machine itself; there is no WSL to
    # put EMsoft into, and saying so sent a Linux user after a Windows feature.
    ("linux", False),
])
def test_only_windows_is_told_about_wsl_for_opencl(platform, expected):
    with mock.patch.object(sc.sys, "platform", platform):
        assert ("WSL" in sc._opencl_missing_message()) is expected


def test_the_message_carries_a_code_so_the_page_can_translate_it():
    """Prose alone would have put an English sentence in a German interface.

    `translateSystemWarnings` matches a warning to its item BY MESSAGE, so the
    two must be identical strings.
    """
    status = _status_on("darwin")
    (message,) = _emsoft_messages(status)
    codes = [it for it in status.warning_items if it.get("code") == "emsoftNotOffered"]
    assert codes, status.warning_items
    assert codes[0]["message"] == message


def test_macos_does_not_repeat_itself_through_the_recommender():
    """The detector's own "No OpenCL detected" is the same news again.

    recommend_device early-returns with it whenever OpenCL is absent, which on
    a Mac it always is. Suppressing our sentence and then copying the
    recommender's list verbatim is how the tester still saw two.
    """
    status = _status_on("darwin")
    assert not any("OpenCL" in w for w in status.warnings), status.warnings


def test_windows_still_hears_the_recommender():
    # Positive control for the line above: the suppression must be macOS-only.
    status = _status_on("win32")
    assert any("OpenCL" in w for w in status.warnings), status.warnings
