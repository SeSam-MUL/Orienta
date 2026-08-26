"""App version identity, derived from git.

The project has no maintained release number — the meaningful identity of an
installed copy is the git commit it was checked out at.  This module resolves
that identity once per process:

    1. `git log -1` / `git rev-parse` against the project root (preferred —
       gives short hash, commit date and branch),
    2. manual parse of `.git/HEAD` + refs when the git binary is missing
       (hash + branch, no date),
    3. "unknown" when there is no `.git` at all (e.g. a zip download).

Used by `GET /api/system/version`, the About section, the log-file banner and
the diagnostics export.
"""

from __future__ import annotations

import logging
import subprocess
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

APP_NAME = "Orienta"

_GIT_TIMEOUT_S = 5


def _run_git(args: list[str], cwd: Path) -> str | None:
    """Run a git command; return stripped stdout or None on any failure."""
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    out = result.stdout.strip()
    return out or None


def _read_git_files(root: Path) -> tuple[str | None, str | None]:
    """Fallback without the git binary: (short_hash, branch) from .git files."""
    head_file = root / ".git" / "HEAD"
    try:
        head = head_file.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None, None

    if head.startswith("ref: "):
        ref = head[5:].strip()  # e.g. refs/heads/golive/unified-phase-identity
        branch = ref[len("refs/heads/"):] if ref.startswith("refs/heads/") else ref
        branch = branch or None
        ref_file = root / ".git" / Path(ref)
        try:
            commit = ref_file.read_text(encoding="utf-8").strip()
        except OSError:
            commit = _lookup_packed_ref(root, ref)
        return (commit[:9] if commit else None), branch

    # Detached HEAD: the file holds the hash itself.
    if len(head) >= 9 and all(c in "0123456789abcdef" for c in head[:9]):
        return head[:9], None
    return None, None


def _lookup_packed_ref(root: Path, ref: str) -> str | None:
    packed = root / ".git" / "packed-refs"
    try:
        for line in packed.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.endswith(" " + ref):
                return line.split(" ", 1)[0]
    except OSError:
        pass
    return None


@lru_cache(maxsize=1)
def get_version_info() -> dict:
    """Resolve the app's version identity. Cached for the process lifetime.

    Returns a dict with keys: app, version (display string), commit,
    commit_date, branch, source ("git" | "git-files" | "unknown").
    Never raises.
    """
    commit = None
    commit_date = None
    branch = None
    source = "unknown"

    out = _run_git(["log", "-1", "--format=%h|%cs"], PROJECT_ROOT)
    if out and "|" in out:
        commit, commit_date = out.split("|", 1)
        branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], PROJECT_ROOT)
        source = "git"
    else:
        commit, branch = _read_git_files(PROJECT_ROOT)
        if commit:
            source = "git-files"

    if commit and commit_date:
        version = f"{commit_date} ({commit})"
    elif commit:
        version = f"({commit})"
    else:
        version = "unknown"

    return {
        "app": APP_NAME,
        "version": version,
        "commit": commit,
        "commit_date": commit_date,
        "branch": branch,
        "source": source,
    }


def version_line() -> str:
    """One-line identity for log banners: 'Orienta 2026-08-26 (a1b2c3d) [branch]'."""
    info = get_version_info()
    line = f"{info['app']} {info['version']}"
    if info["branch"]:
        line += f" [{info['branch']}]"
    return line
