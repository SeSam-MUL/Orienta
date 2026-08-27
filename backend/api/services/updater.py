"""Self-update from the project's git remote.

Why git and not the GitHub API: the release repository is **private**, so an
unauthenticated HTTP check returns 404 for every tester. `git fetch` works for
anyone who cloned it, using the credentials they already needed to clone —
so git is both the more capable and the more portable channel here.

Policy (chosen by the maintainer):
- update only to **tagged releases**, never to the tip of a branch, so a
  work-in-progress push cannot reach testers;
- do the **whole** job — pull, dependencies, frontend build — because a source
  update without a rebuild leaves a broken app;
- an install without `.git` (a downloaded zip) cannot be updated in place and
  is told so, rather than offered a button that cannot work.

Any failure must leave a *working* app: the update runs against a saved
snapshot of HEAD and of `frontend/dist`, and both are restored together, so
source and built interface can never end up from different versions.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from backend.api.services.app_version import PROJECT_ROOT, get_version_info

logger = logging.getLogger(__name__)

FETCH_TIMEOUT_S = 30
STEP_TIMEOUT_S = 1800  # pip/npm on a cold cache can genuinely take this long
CHECK_CACHE_S = 600

_state_lock = threading.Lock()
_progress: dict = {"state": "idle"}
_check_cache: dict = {}


# --------------------------------------------------------------------------
# git helpers
# --------------------------------------------------------------------------

def _git_env() -> dict:
    """Environment for every git call we make.

    The backend is a non-interactive process: if git decides it needs a
    username it cannot ask, and depending on the credential helper it either
    hangs until the timeout or pops a window out of nowhere while the user is
    doing something else. Both are worse than failing fast with a message we
    can explain, so prompting is switched off and a missing credential is
    reported as such.
    """
    import os
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "never"
    return env


def _git(*args: str, timeout: int = 30, cwd: Path | None = None) -> subprocess.CompletedProcess:
    # encoding must be spelled out: with text=True alone Python decodes using
    # the console code page, which turns the em-dashes in the changelog into
    # mojibake on Windows.
    return subprocess.run(
        ["git", "-C", str(cwd or PROJECT_ROOT), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, env=_git_env(),
    )


def _git_out(*args: str, timeout: int = 30) -> str | None:
    try:
        r = _git(*args, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def install_kind() -> str:
    """'git' when this install can update itself, else 'zip'."""
    if not (PROJECT_ROOT / ".git").exists():
        return "zip"
    return "git" if _git_out("remote", "get-url", "origin") else "zip"


_TAG_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:[-+.](.*))?$")


def parse_version(tag: str) -> tuple | None:
    """Sortable key for a release tag, or None if it is not one.

    A pre-release (v1.0.0-rc1) sorts *before* its final release, per semver.
    """
    m = _TAG_RE.match((tag or "").strip())
    if not m:
        return None
    major, minor, patch, suffix = m.groups()
    # (…, 1, '') for a final release sorts above (…, 0, 'rc1') for a pre-release
    return (int(major), int(minor), int(patch), 0 if suffix else 1, suffix or "")


# A private repository answers "not found" rather than "forbidden" when the
# credentials are missing, so that phrase belongs here too.
_AUTH_MARKERS = (
    "could not read Username",
    "terminal prompts disabled",
    "Authentication failed",
    "Permission denied",
    "Repository not found",
    "repository not found",
)


def _parse_ls_remote(out: str) -> list[str]:
    tags = []
    for line in (out or "").splitlines():
        parts = line.split("refs/tags/")
        if len(parts) == 2 and parse_version(parts[1].strip()):
            tags.append(parts[1].strip())
    return sorted(tags, key=parse_version, reverse=True)


def probe_remote() -> tuple[list[str], str]:
    """(release tags newest first, reason). reason is '' on success.

    Distinguishes "cannot sign in" from "cannot reach the server": the first
    is fixed by authenticating once, the second by waiting — and telling the
    user the wrong one wastes their time.
    """
    try:
        r = _git("ls-remote", "--tags", "--refs", "origin", timeout=FETCH_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return [], "remote_unreachable"
    if r.returncode != 0:
        err = r.stderr or ""
        if any(m in err for m in _AUTH_MARKERS):
            logger.info("Update check needs credentials: %s", err.strip()[:200])
            return [], "auth_required"
        logger.info("Update check could not reach the remote: %s", err.strip()[:200])
        return [], "remote_unreachable"
    return _parse_ls_remote(r.stdout), ""


def _remote_release_tags() -> list[str]:
    """Release tags on the remote, newest first. Empty on any failure."""
    return probe_remote()[0]


def _changelog_for(tag: str) -> str:
    """The CHANGELOG section of `tag`, or '' when there is none."""
    text = _git_out("show", f"{tag}:CHANGELOG.md", timeout=FETCH_TIMEOUT_S)
    if not text:
        return ""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("## ") and tag.lstrip("v") in line:
            start = i
            break
    if start is None:
        return ""
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    return "\n".join(lines[start:end]).strip()


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------

def check_for_update(force: bool = False) -> dict:
    """Is a newer *released* version available? Never raises, never blocks long.

    Returns: available, current, latest, install_kind, notes, reason.
    `reason` explains a negative answer so the UI can say something useful
    instead of staying silent.
    """
    now = time.time()
    if not force and _check_cache and now - _check_cache.get("at", 0) < CHECK_CACHE_S:
        return _check_cache["result"]

    info = get_version_info()
    current = info.get("release")
    kind = install_kind()
    result = {
        "available": False,
        "current": info.get("version"),
        "current_release": current,
        "latest": None,
        "install_kind": kind,
        "notes": "",
        "reason": "",
    }

    if kind != "git":
        result["reason"] = "not_a_git_install"
        _check_cache.update(at=now, result=result)
        return result

    # Bring tags in so a freshly published release is visible locally too.
    try:
        _git("fetch", "--tags", "--quiet", "origin", timeout=FETCH_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        pass  # offline / no credentials — ls-remote below decides

    tags, reason = probe_remote()
    if not tags:
        result["reason"] = reason or "remote_unreachable"
        _check_cache.update(at=now, result=result)
        return result

    latest = tags[0]
    result["latest"] = latest
    if current is None:
        # Untagged checkout (a developer tree, or a clone predating tags):
        # offer the release but let the UI say the comparison is approximate.
        result["available"] = True
        result["reason"] = "no_local_release"
    elif parse_version(latest) > parse_version(current):
        result["available"] = True
    else:
        result["reason"] = "up_to_date"

    if result["available"]:
        result["notes"] = _changelog_for(latest)

    _check_cache.update(at=now, result=result)
    return result


# --------------------------------------------------------------------------
# update
# --------------------------------------------------------------------------

def get_progress() -> dict:
    with _state_lock:
        return dict(_progress)


def _set(**kw) -> None:
    with _state_lock:
        _progress.update(kw)


def _log_line(text: str) -> None:
    with _state_lock:
        _progress.setdefault("log", []).append(text)
        # The tail is what a failure report needs; the head is boilerplate.
        if len(_progress["log"]) > 400:
            del _progress["log"][:-400]
    logger.info("[update] %s", text)


def _run_step(name: str, cmd: list[str], cwd: Path) -> bool:
    """Run one update step, streaming its output into the progress log."""
    _set(step=name)
    _log_line(f"$ {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=STEP_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _log_line(f"!! {name} failed to run: {exc}")
        return False
    for line in (proc.stdout or "").splitlines()[-80:]:
        _log_line(line)
    for line in (proc.stderr or "").splitlines()[-80:]:
        _log_line(line)
    if proc.returncode != 0:
        _log_line(f"!! {name} exited with code {proc.returncode}")
        return False
    return True


def _changed_between(old: str, new: str, path: str) -> bool:
    """Did `path` change between two revisions? Assume yes when unsure —
    a needless reinstall is cheap, a missing one breaks the app."""
    out = _git_out("diff", "--name-only", f"{old}..{new}", "--", path)
    if out is None:
        return True
    return bool(out.strip())


def _npm_command() -> list[str]:
    import sys
    return ["npm.cmd" if sys.platform == "win32" else "npm"]


def perform_update(tag: str) -> dict:
    """Update to `tag` and rebuild. Blocking; run it in a thread.

    Restores both HEAD and frontend/dist on any failure, so the app that
    comes back is the one that worked.
    """
    frontend = PROJECT_ROOT / "frontend"
    dist = frontend / "dist"
    backup = frontend / "dist.update-backup"

    with _state_lock:
        _progress.clear()
        _progress.update(state="running", step="preflight", target=tag, log=[])

    previous = _git_out("rev-parse", "HEAD")
    if not previous:
        return _fail("Not a git checkout — cannot update in place.")

    # Only edits to TRACKED files can be lost by the checkout. Untracked files
    # (a dataset someone dropped in the folder, runtime output) are left alone
    # by git, and if an incoming file would overwrite one, the checkout step
    # itself refuses with git's own message.
    dirty = _git_out("status", "--porcelain", "--untracked-files=no")
    if dirty:
        n = len(dirty.splitlines())
        return _fail(
            f"{n} file(s) in the installation folder have local changes. "
            "Updating would overwrite them, so it was cancelled. Move or undo "
            "the changes and try again."
        )

    if not _git_out("rev-parse", "--verify", f"{tag}^{{commit}}"):
        if not _run_step("fetch", ["git", "-C", str(PROJECT_ROOT), "fetch", "--tags", "origin"], PROJECT_ROOT):
            return _fail("Could not reach the update server.")
    target = _git_out("rev-parse", f"{tag}^{{commit}}")
    if not target:
        return _fail(f"Release {tag} not found on the server.")

    # dist is restored together with HEAD; otherwise a failed build would
    # leave the old source serving a half-written interface.
    try:
        if dist.exists():
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
            shutil.copytree(dist, backup)
    except OSError as exc:
        logger.warning("Could not back up frontend/dist: %s", exc)

    def rollback(message: str) -> dict:
        _log_line("Rolling back to the previous version…")
        _git("checkout", "--force", "--detach", previous, timeout=120)
        try:
            if backup.exists():
                shutil.rmtree(dist, ignore_errors=True)
                shutil.move(str(backup), str(dist))
        except OSError as exc:
            logger.warning("Could not restore frontend/dist: %s", exc)
        return _fail(message)

    if not _run_step(
        "checkout",
        ["git", "-C", str(PROJECT_ROOT), "-c", "advice.detachedHead=false",
         "checkout", "--detach", tag],
        PROJECT_ROOT,
    ):
        return rollback("Could not switch to the new version.")

    if _changed_between(previous, target, "requirements.txt"):
        import sys
        if not _run_step(
            "dependencies",
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            PROJECT_ROOT,
        ):
            return rollback("Installing the Python packages failed.")
    else:
        _log_line("Python packages unchanged — skipped.")

    npm = _npm_command()
    if _changed_between(previous, target, "frontend/package.json") or \
       _changed_between(previous, target, "frontend/package-lock.json"):
        if not _run_step("node_modules", npm + ["install"], frontend):
            return rollback("Installing the interface packages failed.")
    else:
        _log_line("Interface packages unchanged — skipped.")

    if not _run_step("build", npm + ["run", "build"], frontend):
        return rollback("Building the interface failed.")

    shutil.rmtree(backup, ignore_errors=True)
    _check_cache.clear()
    get_version_info.cache_clear()
    _set(state="done", step="done", installed=tag)
    _log_line(f"Updated to {tag}.")
    return get_progress()


def _fail(message: str) -> dict:
    _set(state="failed", error=message)
    _log_line(f"!! {message}")
    return get_progress()


def start_update(tag: str) -> bool:
    """Kick off an update in the background. False when one already runs."""
    with _state_lock:
        if _progress.get("state") == "running":
            return False
        _progress.clear()
        _progress.update(state="running", step="starting", target=tag, log=[])
    threading.Thread(
        target=perform_update, args=(tag,), name="orienta-update", daemon=True
    ).start()
    return True
