#!/usr/bin/env python
"""
Starts Orienta as a Desktop App (Electron).

Usage: Just run this file (double-click or Run in VS Code).

Shutting down
-------------
Closing the app must leave nothing behind. Three things can end this
launcher, and all three end the backend with it:

  * the Electron window is closed  -> npm exits -> we clean up and return
  * Ctrl+C in this console         -> KeyboardInterrupt -> same cleanup
  * this console window is closed  -> atexit still runs for a normal exit,
                                      and the backend's own parent watchdog
                                      catches the hard kill a few seconds later

`proc.terminate()` alone is not enough on Windows: it ends the npm shim and
leaves its children — concurrently, vite, electron, and the Python backend —
running, which is how a backend ends up holding port 8000 after the window is
gone. So the whole process TREE is killed (`taskkill /T`), and as a last resort
the process that owns the backend port is checked against the one we started
and killed if it is still ours.
"""

import atexit
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
BACKEND_PORT = int(os.environ.get("KIKUCHIPY_BACKEND_PORT", "8000"))

IS_WINDOWS = sys.platform == "win32"


def _port_owner(port: int):
    """(pid, create_time) of the process listening on `port`, or None.

    The creation time is part of the identity on purpose: PIDs are recycled,
    and killing "whatever holds 8000" minutes later could hit something else
    entirely.
    """
    try:
        import psutil
    except ImportError:
        return None
    try:
        conns = psutil.net_connections(kind="tcp")
    except (psutil.AccessDenied, RuntimeError, OSError):
        return None
    for conn in conns:
        # Per connection, not per scan: one socket we may not inspect must not
        # end the search. It did — the owner then read as "nobody", cleanup
        # concluded the backend was not ours, and left it running.
        try:
            if not conn.laddr or conn.laddr.port != port:
                continue
            if conn.status != psutil.CONN_LISTEN or not conn.pid:
                continue
            return (conn.pid, psutil.Process(conn.pid).create_time())
        except (psutil.Error, AttributeError, IndexError):
            continue
    return None


def _kill_tree(pid: int) -> None:
    """End a process and everything it spawned."""
    if IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True, check=False,
        )
        return
    try:
        import signal
        # POSIX only; the attributes do not exist on Windows, which is why the
        # branch above returns first.
        os.killpg(os.getpgid(pid), signal.SIGTERM)  # type: ignore[attr-defined]
    except (ProcessLookupError, PermissionError, AttributeError):
        pass


def _kill_backend_if_ours(identity, port: int = BACKEND_PORT) -> bool:
    """Kill the port's owner, but only if it is the one we started."""
    if not identity:
        return False
    now = _port_owner(port)
    if not now or now != identity:
        # Either gone already (the normal case — the tree kill got it), or a
        # different process took the port. Not ours to kill.
        return False
    _kill_tree(now[0])
    return True


def main():
    print("=" * 60)
    print("  Orienta - Desktop App")
    print("  Electron + React + FastAPI")
    print("=" * 60)
    print()

    npm_cmd = "npm.cmd" if IS_WINDOWS else "npm"

    print("[1/1] Starting Electron desktop app...")
    print("      (This starts React dev server + FastAPI backend + Electron window)")
    print()

    cmd = [npm_cmd, "run", "electron:dev"]
    try:
        if IS_WINDOWS:
            # Its own process group, so a Ctrl+C here reaches the whole tree
            # instead of only this script.
            proc = subprocess.Popen(
                cmd, cwd=str(FRONTEND_DIR),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        else:
            proc = subprocess.Popen(cmd, cwd=str(FRONTEND_DIR), start_new_session=True)
    except FileNotFoundError:
        print("ERROR: npm not found!")
        print("Make sure Node.js is installed: https://nodejs.org/")
        sys.exit(1)

    backend_identity = None

    def cleanup():
        """Runs on every ordinary way out of this script."""
        if proc.poll() is None:
            _kill_tree(proc.pid)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        if _kill_backend_if_ours(backend_identity):
            print(f"  Backend on port {BACKEND_PORT} was still up — stopped it.")

    atexit.register(cleanup)

    print("  Desktop app is starting...")
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    try:
        # Note who owns the backend port once it comes up, so cleanup can tell
        # our backend from someone else's. Give it a while: a cold start reads
        # the scientific stack before it binds.
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and proc.poll() is None:
            backend_identity = _port_owner(BACKEND_PORT)
            if backend_identity:
                break
            time.sleep(1.0)

        proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cleanup()
        atexit.unregister(cleanup)
        print("Done.")


if __name__ == "__main__":
    main()
