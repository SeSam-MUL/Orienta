#!/usr/bin/env python
"""
Starts Orienta as a Desktop App (Electron).

Usage: Just run this file (double-click or Run in VS Code).
"""

import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def main():
    print("=" * 60)
    print("  Orienta - Desktop App")
    print("  Electron + React + FastAPI")
    print("=" * 60)
    print()

    # Check npm is available
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"

    # Start electron:dev (this runs Vite + Electron + Backend all together)
    print("[1/1] Starting Electron desktop app...")
    print("      (This starts React dev server + FastAPI backend + Electron window)")
    print()

    try:
        proc = subprocess.Popen(
            [npm_cmd, "run", "electron:dev"],
            cwd=str(FRONTEND_DIR),
        )
        print("  Desktop app is starting...")
        print("  Press Ctrl+C to stop")
        print("=" * 60)
        proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        proc.terminate()
        print("Done.")
    except FileNotFoundError:
        print("ERROR: npm not found!")
        print("Make sure Node.js is installed: https://nodejs.org/")
        sys.exit(1)


if __name__ == "__main__":
    main()
