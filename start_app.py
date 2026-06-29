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
    backend_env = {**os.environ, "KIKUCHIPY_WATCHDOG": "0" if headless else "1"}
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
            ["npm", "run", "dev"],
            cwd=str(PROJECT_ROOT / "frontend"),
            shell=True,
        )
        time.sleep(3)
        url = f"http://127.0.0.1:{FRONTEND_PORT}"
    else:
        # Serve built frontend via FastAPI static mount
        print("[2/3] Using pre-built frontend...")
        dist_path = PROJECT_ROOT / "frontend" / "dist"
        if not dist_path.exists():
            print("  Building frontend...")
            subprocess.run(["npm", "run", "build"], cwd=str(PROJECT_ROOT / "frontend"),
                         shell=True, check=True)

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

    try:
        backend_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend_proc.terminate()
        if frontend_proc:
            frontend_proc.terminate()
        print("Done.")


if __name__ == "__main__":
    main()
