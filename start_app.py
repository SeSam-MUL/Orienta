#!/usr/bin/env python
"""
Starts Orienta (Electron + React + FastAPI).

Usage:
    python start_app.py              # Start backend + open browser (watchdog ON)
    python start_app.py --dev        # Start backend + React dev server
    python start_app.py --headless   # Start backend without watchdog so long-running
                                     # batches/simulations survive a browser/UI crash.
                                     # Browser still opened — close it freely;
                                     # only Ctrl+C in this terminal stops the backend.
"""

import atexit
import subprocess
import sys
import os
import time
import webbrowser
import signal
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
BACKEND_PORT = 8000
FRONTEND_PORT = 5173

# npm is "npm.cmd" on Windows, "npm" on macOS/Linux. Always invoke it with the
# resolved name and shell=False: combining a list with shell=True drops the
# run/dev/build arguments on POSIX (they become $0/$1 of the shell), so the dev
# server / build never actually start on macOS or Linux.
NPM = "npm.cmd" if sys.platform == "win32" else "npm"


def wait_for_server(port, timeout=15):
    """Wait for a server to respond on the given port."""
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health" if port == BACKEND_PORT else f"http://127.0.0.1:{port}")
            return True
        except Exception:
            time.sleep(0.5)
    return False


def _backend_env(headless: bool, base: dict | None = None) -> dict:
    """Environment for the backend process.

    KIKUCHIPY_PARENT_PID makes the backend watch THIS launcher and exit ~3 s
    after it disappears. It is the only thing that covers a hard kill: closing
    the console window (or Task Manager) terminates this script without running
    any handler, and the backend would otherwise keep the port — measured: it
    did, until the machine went down. Set in headless mode too; "headless"
    means the backend survives the BROWSER closing, not this launcher exiting.
    """
    env = dict(os.environ if base is None else base)
    env["KIKUCHIPY_WATCHDOG"] = "0" if headless else "1"
    env["KIKUCHIPY_PARENT_PID"] = str(os.getpid())
    return env


def _kill_tree(pid: int) -> None:
    """End a process and everything it spawned."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True, check=False)
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)  # type: ignore[attr-defined]
    except (ProcessLookupError, PermissionError, AttributeError):
        pass


def main():
    dev_mode = "--dev" in sys.argv
    headless = "--headless" in sys.argv

    print("=" * 60)
    print("  Orienta - EBSD Pattern Analysis")
    print("  Electron + React + FastAPI")
    if headless:
        print("  *** HEADLESS MODE — backend survives browser crashes ***")
    print("=" * 60)
    print()

    # Watchdog policy:
    #   Default (no flag)   → KIKUCHIPY_WATCHDOG=1: backend exits if no WS client
    #                         for 60s. Convenient for short interactive sessions:
    #                         close the tab and the server stops.
    #   --headless          → KIKUCHIPY_WATCHDOG=0: backend never auto-exits.
    #                         Required for multi-hour batches and simulations —
    #                         a renderer freeze used to kill the backend with it
    #                         and wipe an in-flight 12 h job.
    print("[1/3] Starting FastAPI backend...")
    backend_env = _backend_env(headless)
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.api.main:app",
         "--host", "127.0.0.1", "--port", str(BACKEND_PORT),
         "--log-level", "info"],
        cwd=str(PROJECT_ROOT),
        env=backend_env,
    )

    if not wait_for_server(BACKEND_PORT):
        print("ERROR: Backend failed to start!")
        backend_proc.kill()
        sys.exit(1)
    print(f"  Backend ready at http://127.0.0.1:{BACKEND_PORT}")
    print(f"  API docs at http://127.0.0.1:{BACKEND_PORT}/docs")

    frontend_proc = None

    if dev_mode:
        # Start React dev server
        print("[2/3] Starting React dev server...")
        frontend_proc = subprocess.Popen(
            [NPM, "run", "dev"],
            cwd=str(PROJECT_ROOT / "frontend"),
        )
        time.sleep(3)
        url = f"http://127.0.0.1:{FRONTEND_PORT}"
    else:
        # Serve built frontend via FastAPI static mount
        print("[2/3] Using pre-built frontend...")
        dist_path = PROJECT_ROOT / "frontend" / "dist"
        if not dist_path.exists():
            print("  Building frontend...")
            subprocess.run([NPM, "run", "build"], cwd=str(PROJECT_ROOT / "frontend"),
                         check=True)

        # FastAPI serves the React SPA at root
        url = f"http://127.0.0.1:{BACKEND_PORT}"

    print(f"[3/3] Opening browser...")
    webbrowser.open(url)
    print()
    print(f"  App running at: {url}")
    if headless:
        print(f"  Headless: backend WILL keep running if you close the browser.")
        print(f"  Stop with Ctrl+C in this terminal when the batch is done.")
    else:
        print(f"  Press Ctrl+C to stop")
    print("=" * 60)

    def shutdown():
        """Leave nothing behind, whichever way this script ends.

        `terminate()` ends the process we spawned and not what IT spawned,
        which on Windows can leave the actual server holding the port. And it
        only ran on Ctrl+C — closing the console left everything up.

        Headless mode keeps the backend alive when the BROWSER closes; that is
        not the same as this launcher exiting, which does stop it.
        """
        for proc in (frontend_proc, backend_proc):
            if proc is not None and proc.poll() is None:
                _kill_tree(proc.pid)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass

    atexit.register(shutdown)
    try:
        backend_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        shutdown()
        atexit.unregister(shutdown)
        print("Done.")


if __name__ == "__main__":
    main()
