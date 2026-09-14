"""
Install Wizard API Routes

Ports the PyQt5 gui/install_dialog.py into FastAPI endpoints:
- WSL status checking
- WSL installation via UAC elevation
- Linux user creation and password management
- EMsoft installation via WebSocket streaming
"""

import asyncio
import logging
import os
import platform
import re
import sys
import shlex
import subprocess
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


def _project_root() -> Path:
    """Return the project root directory (three levels above this file's package)."""
    # This file is at <root>/backend/api/routes/install.py
    return Path(__file__).resolve().parents[3]


def _install_script_path() -> Path:
    return _project_root() / "simulation" / "install_emsoft.sh"


# The first command sent to a freshly registered distro boots the WSL VM and
# initialises the distro: measured 41 s on a warm desktop (Stage 2 smoke test,
# tasks/install_smoke/wizard_wsl.ps1). The old 10 s made the wizard's very
# first action after "Install" fail with "WSL is not responding".
WSL_COLD_START_S = 90
WSL_CMD_S = 60


def _detect_platform() -> dict:
    """Detect host platform for the install wizard.

    Returns dict with: os ("windows", "linux", "macos"), arch, needs_wsl.
    """
    arch = platform.machine().lower()
    if sys.platform == "win32":
        return {"os": "windows", "arch": arch, "needs_wsl": True}
    elif sys.platform == "darwin":
        return {"os": "macos", "arch": arch, "needs_wsl": False}
    else:
        return {"os": "linux", "arch": arch, "needs_wsl": False}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CreateUserRequest(BaseModel):
    username: str
    password: str
    distro: str = ""


class ResetPasswordRequest(BaseModel):
    username: str
    password: str
    distro: str = ""


class ValidatePasswordRequest(BaseModel):
    password: str
    distro: str = ""  # which WSL distro; "" = the default one


# ---------------------------------------------------------------------------
# Sync helper functions (run in executor to avoid blocking the event loop)
# ---------------------------------------------------------------------------

def _check_wsl_version(distro: str) -> int:
    """Check if distro runs WSL 1 or WSL 2.  Returns 0 if unknown."""
    try:
        result = subprocess.run(
            ["wsl", "--list", "--verbose"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return 0
        # Output looks like:  "  NAME            STATE           VERSION"
        #                      "* Ubuntu-22.04    Running         2"
        for line in result.stdout.splitlines():
            cleaned = line.replace("\x00", "").strip()
            if not cleaned or cleaned.startswith("NAME"):
                continue
            # Remove leading * for default distro
            cleaned = cleaned.lstrip("* ")
            if distro.lower() in cleaned.lower():
                parts = cleaned.rsplit(None, 1)
                if parts and parts[-1].isdigit():
                    return int(parts[-1])
        return 0
    except Exception:
        return 0


def _check_nvidia_driver_windows() -> dict:
    """Check NVIDIA driver on the Windows host.

    Returns dict with: available, driver_version, cuda_version, gpu_name, wsl2_ready.
    """
    info = {
        "available": False,
        "driver_version": "",
        "cuda_version": "",
        "gpu_name": "",
        "wsl2_ready": False,
    }
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=driver_version,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",", 1)
            driver_ver = parts[0].strip()
            gpu_name = parts[1].strip() if len(parts) > 1 else ""
            info["available"] = True
            info["driver_version"] = driver_ver
            info["gpu_name"] = gpu_name
            # WSL2 GPU support requires driver >= 470.x
            try:
                major = int(driver_ver.split(".")[0])
                info["wsl2_ready"] = major >= 470
            except (ValueError, IndexError):
                pass
        # Also grab CUDA version
        result2 = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        # Parse CUDA version from the nvidia-smi header output
        header_result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=10,
        )
        if header_result.returncode == 0:
            for line in header_result.stdout.splitlines():
                if "CUDA Version" in line:
                    # e.g. "| NVIDIA-SMI 560.94   Driver Version: 560.94   CUDA Version: 12.6  |"
                    idx = line.index("CUDA Version:")
                    cuda_part = line[idx + len("CUDA Version:"):].strip().rstrip("|").strip()
                    info["cuda_version"] = cuda_part
                    break
    except FileNotFoundError:
        pass  # nvidia-smi not available
    except Exception as exc:
        logger.debug("NVIDIA driver check failed: %s", exc)
    return info


def _check_wsl_sync() -> dict:
    """Check WSL installation status.

    Returns a dict with keys: installed, distro, username, has_user, corrupted,
    wsl_version, nvidia.
    """
    plat = _detect_platform()

    # On Linux/macOS: no WSL needed, EMsoft installs directly
    if plat["os"] != "windows":
        nvidia_info = {"available": False, "driver_version": "", "cuda_version": "",
                       "gpu_name": "", "wsl2_ready": False}
        # Check nvidia-smi on native Linux
        if plat["os"] == "linux":
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version,name",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                )
                if r.returncode == 0 and r.stdout.strip():
                    parts = r.stdout.strip().split(",", 1)
                    nvidia_info["available"] = True
                    nvidia_info["driver_version"] = parts[0].strip()
                    nvidia_info["gpu_name"] = parts[1].strip() if len(parts) > 1 else ""
                    nvidia_info["wsl2_ready"] = True  # native Linux, no WSL needed
            except (FileNotFoundError, Exception):
                pass

        # Check if EMsoft is already installed natively
        emsoft_installed = False
        try:
            r = subprocess.run(
                ["bash", "-lc", "which EMMCOpenCL 2>/dev/null"],
                capture_output=True, text=True, timeout=10,
            )
            emsoft_installed = bool(r.stdout.strip())
        except Exception:
            pass

        return {
            "installed": True,  # "WSL" is always "ready" on native Linux/Mac
            "distro": plat["os"],
            "username": os.environ.get("USER", ""),
            "has_user": True,
            "corrupted": False,
            "wsl_version": 0,  # not applicable
            "nvidia": nvidia_info,
            "platform": plat,
            "emsoft_installed": emsoft_installed,
        }

    # Windows path: check WSL
    nvidia_info = _check_nvidia_driver_windows()

    try:
        result = subprocess.run(
            ["wsl", "--list", "--quiet"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            # wsl.exe prints UTF-16: drop the NULs and the BOM, then keep only
            # names the wizard can pass back to `wsl -d` (the page sends this
            # value to validate-password and the install socket; a name with a
            # BOM in it would read as "wrong password" there).
            lines = [
                line.replace("\x00", "").replace("﻿", "").strip()
                for line in result.stdout.splitlines()
            ]
            lines = [ln for ln in lines if ln and _validate_distro_name(ln)]
            if lines:
                # Prefer Ubuntu distros over docker-desktop etc.
                distro = lines[0]
                for line in lines:
                    if "ubuntu" in line.lower():
                        distro = line
                        break

                username = ""
                corrupted = False
                try:
                    user_result = subprocess.run(
                        ["wsl", "-d", distro, "bash", "-c", "whoami"],
                        capture_output=True, text=True, timeout=WSL_COLD_START_S
                    )
                    if user_result.returncode == 0:
                        username = user_result.stdout.strip().replace("\x00", "")
                    else:
                        combined = (
                            user_result.stdout.replace("\x00", "")
                            + user_result.stderr.replace("\x00", "")
                        )
                        if "ERROR_FILE_NOT_FOUND" in combined or "ext4.vhdx" in combined:
                            corrupted = True
                except subprocess.TimeoutExpired:
                    # Slow is not broken. A freshly installed distro boots its
                    # VM on the first command (measured 41 s); calling that
                    # "corrupted" used to offer a repair that UNREGISTERS it.
                    logger.warning("WSL distro %s did not answer within %d s — "
                                   "reporting 'no user yet', not 'corrupted'",
                                   distro, WSL_COLD_START_S)
                except Exception:
                    corrupted = True

                wsl_version = _check_wsl_version(distro)

                return {
                    "installed": True,
                    "distro": distro,
                    "username": username,
                    "has_user": bool(username and username != "root"),
                    "corrupted": corrupted,
                    "wsl_version": wsl_version,
                    "nvidia": nvidia_info,
                    "platform": plat,
                }

        return {
            "installed": False, "distro": "", "username": "", "has_user": False,
            "corrupted": False, "wsl_version": 0, "nvidia": nvidia_info,
            "platform": plat,
        }

    except Exception as exc:
        logger.warning("WSL check failed: %s", exc)
        return {
            "installed": False, "distro": "", "username": "", "has_user": False,
            "corrupted": False, "wsl_version": 0, "nvidia": nvidia_info,
            "platform": plat,
        }


def _validate_distro_name(name: str) -> bool:
    """Validate distro name to prevent command injection."""
    return bool(re.match(r'^[A-Za-z0-9._-]+$', name))


def _wsl_feature_present() -> bool:
    """Is the Windows Subsystem for Linux itself installed?

    ``wsl --status`` exits 0 once the optional component is there, with or
    without a distribution. Anything else (stub wsl.exe, missing binary,
    hang) counts as absent — the elevated feature install is harmless when
    the feature already exists, the reverse is not.
    """
    try:
        r = subprocess.run(["wsl", "--status"], capture_output=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def _run_elevated(exe: str, args: str) -> bool:
    """Run `exe args` through the UAC prompt. False when it was denied."""
    import ctypes
    ret = ctypes.windll.shell32.ShellExecuteW(None, "runas", exe, args, None, 1)  # SW_SHOWNORMAL
    return ret > 32  # ShellExecuteW returns > 32 on success


def _install_wsl_sync(distro: str, repair: bool, broken_distro: str) -> dict:
    """Install WSL (or repair a distro) in two stages with different rights.

    WSL distributions are registered PER USER (HKCU + %LOCALAPPDATA%); only
    the Windows feature is machine-wide. Installing the distro inside the
    elevated window put it into whichever ADMIN answered the UAC prompt —
    on a lab PC that is someone else, and the user then sees "WSL installed
    cleanly, but no Ubuntu". So:

      stage "feature": the feature only, elevated, then a restart;
      stage "distro":  the distribution, UNELEVATED, as the user who will
                       use it. No administrator at all when the feature is
                       already there.

    Returns dict with keys: success, stage, message.
    """
    if _detect_platform()["os"] != "windows":
        return {"success": False, "message": "WSL installation is only available on Windows."}
    if not _validate_distro_name(distro):
        return {"success": False, "message": "Invalid distro name."}
    if broken_distro and not _validate_distro_name(broken_distro):
        return {"success": False, "message": "Invalid broken distro name."}

    try:
        if repair and broken_distro:
            # Unregister the broken distro first
            unreg = subprocess.run(
                ["wsl", "--unregister", broken_distro],
                capture_output=True, text=True, timeout=30
            )
            if unreg.returncode != 0:
                stderr = unreg.stderr.replace("\x00", "").strip()
                return {
                    "success": False,
                    "message": f"Could not unregister '{broken_distro}': {stderr}",
                }

        if not _wsl_feature_present():
            cmd_args = (
                "/c wsl --install --no-distribution & wsl --update"
                " & echo."
                " & echo The Windows Subsystem for Linux is installed."
                " & echo Restart Windows, then click Install again to add the Linux distribution."
                " & pause"
            )
            if _run_elevated("cmd.exe", cmd_args):
                return {
                    "success": True,
                    "stage": "feature",
                    "message": (
                        "Installing the Windows Subsystem for Linux in an elevated window. "
                        "Restart Windows when it finishes, then click Install again to add "
                        "the Linux distribution — that step needs no administrator."
                    ),
                }
            return {
                "success": False,
                "stage": "feature",
                "message": (
                    "Could not start the installation — the administrator prompt was denied. "
                    "Ask an administrator to run once: wsl --install --no-distribution"
                ),
            }

        # Feature present: the distribution goes into THIS user's profile, so it
        # must run as this user. A console of its own shows wsl's download
        # progress and any error; the wizard's status refresh turns green when
        # the distro is registered.
        # `-d` rather than the positional form: the inbox wsl.exe of older
        # Windows 10/11 builds only knows `--install -d <distro>`.
        creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        subprocess.Popen(
            ["wsl", "--install", "-d", distro, "--no-launch"],
            creationflags=creationflags,
        )
        return {
            "success": True,
            "stage": "distro",
            "message": (
                f"Installing {distro} for your account — a console window shows the "
                "download. No administrator needed; the status above turns green "
                "when it is done."
            ),
        }

    except Exception as exc:
        return {"success": False, "message": f"Error: {exc}"}


def _text(raw) -> str:
    """Decode subprocess output that may be bytes (our stdin helper) or str
    (a test double), dropping the NULs wsl.exe's UTF-16 messages leave."""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return raw.replace("\x00", "")


def _run_with_stdin(args: list, data: str, timeout: float) -> subprocess.CompletedProcess:
    """Run `args` with `data` on stdin. Bytes on purpose: in text mode
    Python turns the trailing '\\n' into '\\r\\n' on Windows, and chpasswd
    would store a password ending in a carriage return."""
    raw = subprocess.run(args, input=data.encode("utf-8"), capture_output=True, timeout=timeout)
    return subprocess.CompletedProcess(raw.args, raw.returncode,
                                       stdout=_text(raw.stdout), stderr=_text(raw.stderr))


def _create_user_sync(username: str, password: str, distro: str) -> dict:
    """Create a Linux user in WSL.

    Returns dict with keys: success, message.
    """
    if _detect_platform()["os"] != "windows":
        return {"success": False, "message": "WSL user creation is only available on Windows. On Linux/macOS, use your system's user management."}
    # Validate username
    if not username:
        return {"success": False, "message": "Username is required."}
    if not username[0].isalpha():
        return {"success": False, "message": "Username must start with a letter."}
    if username != username.lower():
        return {"success": False, "message": "Username must be lowercase."}
    if not all(c.isalnum() or c == "_" for c in username):
        return {
            "success": False,
            "message": "Username may only contain lowercase letters, numbers, and underscores.",
        }

    # Validate password
    if not password:
        return {"success": False, "message": "Password is required."}
    if len(password) < 4:
        return {"success": False, "message": "Password must be at least 4 characters."}

    try:
        escaped_user = shlex.quote(username)
        wsl_cmd = _wsl_prefix(distro)

        # Check if user already exists. This is typically the first command a
        # brand-new distro ever runs, so it gets the cold-start budget.
        check = subprocess.run(
            wsl_cmd + ["-u", "root", "bash", "-c",
                        f"id {escaped_user} >/dev/null 2>&1 && echo EXISTS || echo NOTFOUND"],
            capture_output=True, text=True, timeout=WSL_COLD_START_S,
        )
        user_exists = "EXISTS" in check.stdout

        if not user_exists:
            create = subprocess.run(
                wsl_cmd + ["-u", "root", "bash", "-c",
                            f"useradd -m -s /bin/bash -G sudo {escaped_user}"],
                capture_output=True, text=True, timeout=WSL_CMD_S,
            )
            if create.returncode != 0:
                err = create.stderr.strip() or create.stdout.strip() or "Unknown error"
                return {"success": False, "message": f"Failed to create user: {err}"}

        # Set password — on stdin, never on the command line (visible in `ps`).
        pw_result = _run_with_stdin(
            wsl_cmd + ["-u", "root", "chpasswd"],
            f"{username}:{password}\n", timeout=WSL_CMD_S,
        )
        if pw_result.returncode != 0:
            err = pw_result.stderr.strip() or "Unknown error"
            return {"success": False, "message": f"Failed to set password: {err}"}

        # Ensure sudo group membership
        subprocess.run(
            wsl_cmd + ["-u", "root", "bash", "-c",
                        f"usermod -aG sudo {escaped_user}"],
            capture_output=True, text=True, timeout=WSL_CMD_S,
        )

        # Set default user via /etc/wsl.conf
        subprocess.run(
            wsl_cmd + ["-u", "root", "bash", "-c",
                        f'printf "[user]\\ndefault={escaped_user}\\n" > /etc/wsl.conf'],
            capture_output=True, text=True, timeout=WSL_CMD_S,
        )

        # Terminate WSL to apply default user change
        terminate_cmd = ["wsl", "--terminate", distro] if distro else ["wsl", "--shutdown"]
        subprocess.run(terminate_cmd, capture_output=True, text=True, timeout=WSL_CMD_S)

        action = "configured" if user_exists else "created"
        return {
            "success": True,
            "message": f"User '{username}' {action} successfully.",
            "username": username,
        }

    except subprocess.TimeoutExpired:
        return {"success": False, "message": "WSL is not responding (timeout)."}
    except Exception as exc:
        return {"success": False, "message": f"Error: {exc}"}


def _reset_password_sync(username: str, password: str, distro: str) -> dict:
    """Reset WSL user password via root access.

    Returns dict with keys: success, message.
    """
    if _detect_platform()["os"] != "windows":
        return {"success": False, "message": "WSL password reset is only available on Windows."}
    if not username:
        return {"success": False, "message": "Username is required."}
    if not password or len(password) < 4:
        return {"success": False, "message": "Password must be at least 4 characters."}

    try:
        wsl_cmd = _wsl_prefix(distro)

        result = _run_with_stdin(
            wsl_cmd + ["-u", "root", "chpasswd"],
            f"{username}:{password}\n", timeout=WSL_COLD_START_S,
        )
        if result.returncode == 0:
            return {"success": True, "message": f"Password for '{username}' has been reset."}
        err = result.stderr.strip() or "Unknown error"
        return {"success": False, "message": f"Failed to reset password: {err}"}

    except subprocess.TimeoutExpired:
        return {"success": False, "message": "WSL is not responding (timeout)."}
    except Exception as exc:
        return {"success": False, "message": f"Error: {exc}"}


def _wsl_prefix(distro: str) -> list:
    """``wsl`` or ``wsl -d <distro>``; an unvalidated name is refused, not passed on."""
    if not distro:
        return ["wsl"]
    if not _validate_distro_name(distro):
        raise ValueError(f"Invalid distro name: {distro!r}")
    return ["wsl", "-d", distro]


def _validate_password_sync(password: str, distro: str = "") -> bool:
    """Validate the sudo password. On Windows via WSL, on Linux/macOS directly.

    ``-k`` first: with a warm sudo timestamp any password "works", which is
    how a typo used to pass this check and fail 20 minutes into the build.
    The password travels on stdin (``-S``), not in the command line.

    Raises ValueError for an unusable distro name — that is not "wrong
    password" and must not be reported as one.
    """
    sudo_cmd = ["sudo", "-k", "-S", "-v"]
    cmd = _wsl_prefix(distro) + sudo_cmd if _detect_platform()["os"] == "windows" else sudo_cmd
    try:
        result = _run_with_stdin(cmd, password + "\n", timeout=WSL_COLD_START_S)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ---------------------------------------------------------------------------
# EMsoft install wrapper
# ---------------------------------------------------------------------------

_ASKPASS_EOF = "ORIENTA_ASKPASS_EOF"


def build_install_wrapper(script_content: str, password: str) -> str:
    """Bash text that runs `script_content` with sudo answered automatically.

    The password reaches sudo through ``SUDO_ASKPASS`` (a 0700 helper in the
    distro's own tmp, deleted on exit) and ``sudo -A``. The old wrapper piped
    it into ``sudo -S`` — and a pipe REPLACES stdin, so every
    ``echo <path> | sudo tee <file>`` in install_emsoft.sh wrote either
    nothing or, once sudo's timestamp was warm, the password itself into
    /etc/fstab and the OpenCL ICD files. With askpass the script's stdin is
    its own.

    A background ``sudo -n -v`` keeps the timestamp warm so child scripts
    that call plain ``sudo`` (sh, not bash — they never see the function
    below) do not prompt either.
    """
    if "\n" in password or "\r" in password:
        raise ValueError("password must not contain line breaks")
    if _ASKPASS_EOF in password:
        raise ValueError("password contains the wrapper delimiter")
    quoted_pw = shlex.quote(password)
    return (
        "#!/usr/bin/env bash\n"
        "# Generated by Orienta: runs simulation/install_emsoft.sh with sudo answered\n"
        "# through SUDO_ASKPASS, so the script's stdin stays its own.\n"
        "# /tmp on purpose, not $TMPDIR: a login profile may point TMPDIR at\n"
        "# /mnt/c, where chmod is a no-op and the helper would be world-readable.\n"
        "rm -f /tmp/_orienta_askpass.* 2>/dev/null  # leftovers of a killed run\n"
        '_ORIENTA_ASKPASS="$(mktemp /tmp/_orienta_askpass.XXXXXX)" || exit 1\n'
        f"cat > \"$_ORIENTA_ASKPASS\" <<'{_ASKPASS_EOF}'\n"
        "#!/bin/sh\n"
        f"printf '%s' {quoted_pw}\n"
        f"{_ASKPASS_EOF}\n"
        'chmod 700 "$_ORIENTA_ASKPASS"\n'
        'export SUDO_ASKPASS="$_ORIENTA_ASKPASS"\n'
        "export DEBIAN_FRONTEND=noninteractive\n"
        "_ORIENTA_KEEPALIVE=\n"
        "_orienta_cleanup() {\n"
        '  [ -n "$_ORIENTA_KEEPALIVE" ] && kill "$_ORIENTA_KEEPALIVE" 2>/dev/null\n'
        '  rm -f "$_ORIENTA_ASKPASS"\n'
        "}\n"
        "trap _orienta_cleanup EXIT INT TERM HUP\n"
        "\n"
        'echo "Verifying sudo access..."\n'
        "# -k: a warm timestamp from an earlier terminal session must not let a\n"
        "# typo through here and fail 20 minutes into the build.\n"
        "if ! command sudo -k -A -v 2>/dev/null; then\n"
        '  echo "ERROR: sudo password is incorrect. Aborting installation."\n'
        "  exit 1\n"
        "fi\n"
        'echo "Sudo access verified."\n'
        "\n"
        "# Keep the sudo timestamp warm for child scripts that call plain sudo.\n"
        "# Detached from our stdio so it can never hold the log pipe open; the\n"
        "# subshell kills its own sleep on TERM so nothing outlives the wrapper.\n"
        "( trap 'kill $_s 2>/dev/null; exit 0' TERM INT HUP\n"
        "  while :; do sleep 60 & _s=$!; wait $_s; command sudo -A -n -v || exit; done\n"
        ") >/dev/null 2>&1 </dev/null &\n"
        "_ORIENTA_KEEPALIVE=$!\n"
        "\n"
        "sudo() { command sudo -A \"$@\"; }\n"
        "export -f sudo\n"
        "\n"
        + script_content
        + "\n"
    )


def write_wrapper_file(wrapper: str) -> str:
    """Write the wrapper to a private temp file (mkstemp: 0600) and return its path.

    Never widen the mode: the file holds the password for the whole build.
    ``bash <file>`` does not need it to be executable.
    """
    fd, temp_path = tempfile.mkstemp(suffix=".sh", prefix="_emsoft_install_")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(wrapper)
    return temp_path


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@router.get("/wsl-status")
async def get_wsl_status():
    """Check WSL installation status.

    Returns installed, distro, username, and whether the install is corrupted.
    """
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _check_wsl_sync)
    return result


@router.post("/wsl-install")
async def trigger_wsl_install(
    distro: str = Query(default="Ubuntu-22.04", description="WSL distro to install"),
    repair: bool = Query(default=False, description="Repair mode (unregister broken distro first)"),
    broken_distro: str = Query(default="", description="Name of broken distro to unregister"),
):
    """Install WSL in two stages: the Windows feature elevated (UAC) if it is
    missing, otherwise the distribution unelevated as the current user.

    In repair mode, first unregisters the broken distro then installs fresh.
    Answers {success, stage, message}; a denied prompt is success=false.
    """
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, _install_wsl_sync, distro, repair, broken_distro
    )
    return result


@router.post("/wsl-create-user")
async def create_wsl_user(body: CreateUserRequest):
    """Create a Linux user in WSL.

    Validates the username (lowercase, starts with letter, alphanumeric + underscore)
    and password (min 4 chars), then creates the user, sets the password, ensures sudo
    group membership, sets default user via /etc/wsl.conf, and terminates WSL to apply.
    """
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, _create_user_sync, body.username, body.password, body.distro
    )
    return result


@router.post("/wsl-reset-password")
async def reset_wsl_password(body: ResetPasswordRequest):
    """Reset WSL user password via root access (no old password needed)."""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, _reset_password_sync, body.username, body.password, body.distro
    )
    return result


@router.post("/validate-password")
async def validate_password(body: ValidatePasswordRequest):
    """Validate WSL sudo password by running a test sudo command."""
    loop = asyncio.get_running_loop()
    try:
        valid = await loop.run_in_executor(None, _validate_password_sync, body.password, body.distro)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"valid": valid}


# ---------------------------------------------------------------------------
# WebSocket endpoint for streaming EMsoft installation
# ---------------------------------------------------------------------------

@router.websocket("/ws/install-emsoft")
async def ws_install_emsoft(websocket: WebSocket):
    """Stream EMsoft installation output line by line.

    Protocol:
    1. Client connects and sends JSON: {"password": "<sudo_password>", "distro": "<optional>"}
    2. Server streams: {"type": "log", "line": "<output_line>"}
    3. Server sends final: {"type": "done", "success": bool, "message": "<msg>"}
    """
    await websocket.accept()
    temp_path = None

    try:
        # Receive the first message containing the password
        try:
            data = await asyncio.wait_for(websocket.receive_json(), timeout=30)
        except asyncio.TimeoutError:
            await websocket.send_json({
                "type": "done", "success": False,
                "message": "Timeout waiting for password."
            })
            return

        password = data.get("password", "")
        distro = str(data.get("distro") or "")
        if distro and not _validate_distro_name(distro):
            await websocket.send_json({
                "type": "done", "success": False,
                "message": "Invalid distro name.",
            })
            return

        script_path = _install_script_path()
        if not script_path.exists():
            await websocket.send_json({
                "type": "done", "success": False,
                "message": f"Install script not found: {script_path}"
            })
            return

        # Read install script
        try:
            script_content = script_path.read_text(encoding="utf-8")
        except Exception as exc:
            await websocket.send_json({
                "type": "done", "success": False,
                "message": f"Failed to read install script: {exc}"
            })
            return

        try:
            wrapper = build_install_wrapper(script_content, password)
        except ValueError as exc:
            await websocket.send_json({
                "type": "done", "success": False,
                "message": f"Unusable password: {exc}",
            })
            return
        temp_path = write_wrapper_file(wrapper)

        # Platform-aware script execution
        plat = _detect_platform()

        if plat["os"] == "windows":
            # Convert Windows path to WSL path: C:\Users\... -> /mnt/c/Users/...
            script_exec_path = temp_path.replace("\\", "/")
            if len(script_exec_path) >= 2 and script_exec_path[1] == ":":
                drive = script_exec_path[0].lower()
                script_exec_path = f"/mnt/{drive}{script_exec_path[2:]}"
            exec_cmd = _wsl_prefix(distro) + ["bash", "-l", script_exec_path]
            not_found_msg = "WSL is not installed. Please install WSL first: wsl --install"
        else:
            # Linux/macOS: run directly. No chmod — the file holds the
            # password, and `bash <file>` needs no execute bit.
            script_exec_path = temp_path
            exec_cmd = ["bash", "-l", script_exec_path]
            not_found_msg = "bash is not available."

        await websocket.send_json({
            "type": "log",
            "line": f"Starting installation script: {script_exec_path}"
        })

        # Launch the process asynchronously
        try:
            process = await asyncio.create_subprocess_exec(
                *exec_cmd,
                # EOF on stdin: an apt/dpkg conffile prompt must fail fast, not
                # sit on the backend's stdin until the 30-minute silence timeout.
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError:
            await websocket.send_json({
                "type": "done", "success": False,
                "message": not_found_msg,
            })
            return

        # Stream stdout line by line.
        # Timeout per line is 30 minutes — EMsoft compilation steps can run
        # for 10-20 minutes without output (especially make -j on slow machines).
        # We also send keep-alive pings every 30s to prevent WebSocket idle drops.
        LINE_TIMEOUT = 1800  # 30 minutes max silence before giving up
        KEEPALIVE_INTERVAL = 30  # seconds between keep-alive pings

        if process.stdout is None:
            await websocket.send_json({
                "type": "done", "success": False,
                "message": "Failed to capture process output."
            })
            return
        while True:
            try:
                # Use a loop with short waits + keepalive instead of one long wait
                raw_line = None
                elapsed = 0.0
                while elapsed < LINE_TIMEOUT:
                    try:
                        raw_line = await asyncio.wait_for(
                            process.stdout.readline(), timeout=KEEPALIVE_INTERVAL
                        )
                        break  # Got a line (or EOF)
                    except asyncio.TimeoutError:
                        elapsed += KEEPALIVE_INTERVAL
                        # Send keep-alive ping to prevent WebSocket idle disconnect
                        try:
                            await websocket.send_json({
                                "type": "keepalive",
                                "elapsed": int(elapsed),
                            })
                        except WebSocketDisconnect:
                            process.kill()
                            return

                if raw_line is None:
                    # Timed out after LINE_TIMEOUT seconds of total silence
                    await websocket.send_json({
                        "type": "done", "success": False,
                        "message": f"Installation timed out ({LINE_TIMEOUT // 60} min without output)."
                    })
                    process.kill()
                    return

            except asyncio.CancelledError:
                process.kill()
                return

            if not raw_line:
                break  # EOF

            line = raw_line.decode("utf-8", errors="replace").rstrip()

            # Filter out sudo password prompts that would expose sensitive data
            if "[sudo]" in line or "password for" in line.lower():
                continue

            # Also into the persistent log — a WebSocket drop used to lose
            # the whole 10-20 minute build log with it.
            logger.info("[emsoft-install] %s", line)

            try:
                await websocket.send_json({"type": "log", "line": line})
            except WebSocketDisconnect:
                process.kill()
                return

        await process.wait()
        success = process.returncode == 0
        message = (
            "Installation completed successfully!"
            if success
            else f"Installation failed (exit code {process.returncode})"
        )
        logger.info("[emsoft-install] %s", message)
        await websocket.send_json({"type": "done", "success": success, "message": message})

    except WebSocketDisconnect:
        logger.info("Install WebSocket client disconnected")
    except Exception as exc:
        logger.exception("Unexpected error in ws_install_emsoft")
        try:
            await websocket.send_json({
                "type": "done", "success": False,
                "message": f"Unexpected error: {exc}"
            })
        except Exception:
            pass
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
