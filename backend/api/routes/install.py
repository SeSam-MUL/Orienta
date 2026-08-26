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

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


def _project_root() -> Path:
    """Return the project root directory (three levels above this file's package)."""
    # This file is at <root>/backend/api/routes/install.py
    return Path(__file__).resolve().parents[3]


def _install_script_path() -> Path:
    return _project_root() / "simulation" / "install_emsoft.sh"


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
            lines = [
                line.strip().replace("\x00", "")
                for line in result.stdout.splitlines()
                if line.strip().replace("\x00", "")
            ]
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
                        ["wsl", "bash", "-c", "whoami"],
                        capture_output=True, text=True, timeout=10
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


def _install_wsl_sync(distro: str, repair: bool, broken_distro: str) -> dict:
    """Trigger WSL installation (or repair) using UAC elevation.

    Returns dict with keys: success, message.
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

        # Use ShellExecuteW with "runas" to trigger UAC elevation (Windows-only)
        import ctypes
        cmd_args = (
            f"/c wsl --install -d {distro} --no-launch"
            " & echo."
            " & echo WSL installation complete. You can close this window."
            " & pause"
        )
        ret = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", "cmd.exe", cmd_args, None, 1  # SW_SHOWNORMAL
        )
        # ShellExecuteW returns > 32 on success
        if ret > 32:
            return {
                "success": True,
                "message": (
                    "WSL installation started in an elevated command window. "
                    "After it completes, you may need to restart your PC."
                ),
            }
        return {
            "success": False,
            "message": (
                "Could not start WSL installation — administrator prompt may have been denied. "
                f"You can install manually: wsl --install -d {distro}"
            ),
        }

    except Exception as exc:
        return {"success": False, "message": f"Error: {exc}"}


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
        escaped_combo = shlex.quote(f"{username}:{password}")

        wsl_cmd = ["wsl"]
        if distro:
            wsl_cmd += ["-d", distro]

        # Check if user already exists
        check = subprocess.run(
            wsl_cmd + ["-u", "root", "bash", "-c",
                        f"id {escaped_user} >/dev/null 2>&1 && echo EXISTS || echo NOTFOUND"],
            capture_output=True, text=True, timeout=10,
        )
        user_exists = "EXISTS" in check.stdout

        if not user_exists:
            create = subprocess.run(
                wsl_cmd + ["-u", "root", "bash", "-c",
                            f"useradd -m -s /bin/bash -G sudo {escaped_user}"],
                capture_output=True, text=True, timeout=15,
            )
            if create.returncode != 0:
                err = create.stderr.strip() or create.stdout.strip() or "Unknown error"
                return {"success": False, "message": f"Failed to create user: {err}"}

        # Set password
        pw_result = subprocess.run(
            wsl_cmd + ["-u", "root", "bash", "-c",
                        f"echo {escaped_combo} | chpasswd"],
            capture_output=True, text=True, timeout=10,
        )
        if pw_result.returncode != 0:
            err = pw_result.stderr.strip() or "Unknown error"
            return {"success": False, "message": f"Failed to set password: {err}"}

        # Ensure sudo group membership
        subprocess.run(
            wsl_cmd + ["-u", "root", "bash", "-c",
                        f"usermod -aG sudo {escaped_user}"],
            capture_output=True, text=True, timeout=5,
        )

        # Set default user via /etc/wsl.conf
        subprocess.run(
            wsl_cmd + ["-u", "root", "bash", "-c",
                        f'printf "[user]\\ndefault={escaped_user}\\n" > /etc/wsl.conf'],
            capture_output=True, text=True, timeout=5,
        )

        # Terminate WSL to apply default user change
        terminate_cmd = ["wsl", "--terminate", distro] if distro else ["wsl", "--shutdown"]
        subprocess.run(terminate_cmd, capture_output=True, text=True, timeout=10)

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
        escaped_combo = shlex.quote(f"{username}:{password}")

        wsl_cmd = ["wsl"]
        if distro:
            wsl_cmd += ["-d", distro]

        result = subprocess.run(
            wsl_cmd + ["-u", "root", "bash", "-c",
                        f"echo {escaped_combo} | chpasswd"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return {"success": True, "message": f"Password for '{username}' has been reset."}
        err = result.stderr.strip() or "Unknown error"
        return {"success": False, "message": f"Failed to reset password: {err}"}

    except subprocess.TimeoutExpired:
        return {"success": False, "message": "WSL is not responding (timeout)."}
    except Exception as exc:
        return {"success": False, "message": f"Error: {exc}"}


def _validate_password_sync(password: str) -> bool:
    """Validate sudo password. On Windows: via WSL. On Linux/Mac: directly."""
    try:
        escaped = shlex.quote(password)
        test_cmd = f'echo {escaped} | sudo -S echo "SUDO_OK" 2>/dev/null'
        plat = _detect_platform()
        if plat["os"] == "windows":
            cmd = ["wsl", "bash", "-c", test_cmd]
        else:
            cmd = ["bash", "-c", test_cmd]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return "SUDO_OK" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


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
    """Trigger WSL installation via UAC elevation.

    Uses ShellExecuteW with 'runas' to show the UAC prompt.
    In repair mode, first unregisters the broken distro then installs fresh.
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
    valid = await loop.run_in_executor(None, _validate_password_sync, body.password)
    return {"valid": valid}


# ---------------------------------------------------------------------------
# WebSocket endpoint for streaming EMsoft installation
# ---------------------------------------------------------------------------

@router.websocket("/ws/install-emsoft")
async def ws_install_emsoft(websocket: WebSocket):
    """Stream EMsoft installation output line by line.

    Protocol:
    1. Client connects and sends JSON: {"password": "<sudo_password>"}
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

        # Build wrapper script that overrides sudo to auto-pipe the password
        escaped_pw = shlex.quote(password)
        wrapper = (
            "#!/usr/bin/env bash\n"
            f"_SUDO_PASS={escaped_pw}\n"
            "\n"
            "# Verify sudo access before starting installation\n"
            'echo "Verifying sudo access..."\n'
            'echo "$_SUDO_PASS" | command sudo -S echo "SUDO_OK" 2>/dev/null\n'
            "if [ $? -ne 0 ]; then\n"
            '  echo "ERROR: sudo password is incorrect. Aborting installation."\n'
            "  exit 1\n"
            "fi\n"
            'echo "Sudo access verified."\n'
            "\n"
            "# Override sudo to automatically provide password\n"
            "sudo() {\n"
            '  echo "$_SUDO_PASS" | command sudo -S "$@"\n'
            "}\n"
            "export -f sudo\n"
            "\n"
            + script_content
            + "\n"
            "\n"
            "unset _SUDO_PASS\n"
        )

        # Write wrapper to temp file with Unix line endings and random name
        fd, temp_path = tempfile.mkstemp(suffix=".sh", prefix="_emsoft_install_")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(wrapper)

        # Platform-aware script execution
        plat = _detect_platform()

        if plat["os"] == "windows":
            # Convert Windows path to WSL path: C:\Users\... -> /mnt/c/Users/...
            script_exec_path = temp_path.replace("\\", "/")
            if len(script_exec_path) >= 2 and script_exec_path[1] == ":":
                drive = script_exec_path[0].lower()
                script_exec_path = f"/mnt/{drive}{script_exec_path[2:]}"
            exec_cmd = ["wsl", "bash", "-l", script_exec_path]
            not_found_msg = "WSL is not installed. Please install WSL first: wsl --install"
        else:
            # Linux/macOS: run directly
            script_exec_path = temp_path
            os.chmod(temp_path, 0o755)
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
