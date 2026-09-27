"""Self-update from the project's git remote.

Why git for a CHECKOUT: `git fetch` works for anyone who cloned the repository,
using the credentials they already needed to clone it, and it sees every tag
rather than only what a release API chooses to call "latest".

(The repository is public as of 2026-09-14, so the original reason given here —
that an unauthenticated HTTP check would 404 — no longer holds. An installed
copy made by the setup has no git at all and uses `github_releases` instead;
see `install_kind`.)

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

from backend.api.services.app_version import (
    PROJECT_ROOT,
    _TAG_LINE_RE,
    get_version_info,
)

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
    """How this installation can update itself.

    'git'    -- a checkout with a remote: fetch + checkout, the original path
    'bundle' -- installed by the setup: no .git, but a VERSION file naming the
                release it was built from; updates by downloading a package
    'zip'    -- neither; tell the user rather than offering a dead button

    'absent' and 'cannot tell' are different answers here. Only an ABSENT
    VERSION file makes this a zip; unreadable (an ACL, a lock) and unparseable
    (a crashed update left it half written) both mean "this IS an installed
    copy, and something is wrong with it", which the update path can repair.
    Reporting 'zip' for those shows the user "this installation cannot update
    itself" — the one sentence this work exists to delete — with no way back.

    An earlier version guarded that with `Path.stat()`, which does not read the
    file: an ACL or an exclusive lock passes the stat and fails on the open, so
    the guard was a no-op for exactly its own scenario.
    """
    from backend.api.services.app_version import read_version_state

    # Ask git, do not stat `.git`. A `git worktree`, a submodule and
    # `clone --separate-git-dir` all give a `.git` FILE, so `.is_dir()` is False
    # and a perfectly updatable checkout was being told it cannot update itself
    # — and this project uses worktrees routinely.
    #
    # `--show-toplevel` anchors the answer: `git -C <dir>` SEARCHES UPWARD, so a
    # bundle unpacked anywhere inside someone else's repository would otherwise
    # be reported as a git install of THAT project.
    toplevel = _git_out("rev-parse", "--show-toplevel")
    if toplevel and Path(toplevel).resolve() == PROJECT_ROOT.resolve():
        return "git" if _git_out("remote", "get-url", "origin") else "zip"

    state, _tag = read_version_state(PROJECT_ROOT)
    if state == "absent":
        return "zip"
    if state == "tag":
        return "bundle"
    # "unreadable" (a lock, an ACL) or "junk" (a half-written file from a
    # crashed update). Both mean "this IS an installed copy, and something is
    # wrong with it" — which the update path can repair. Reporting "zip" would
    # show the one sentence this work exists to delete, with no way back.
    logger.warning("VERSION is present but %s; treating this as a bundle install", state)
    return "bundle"


# ONE pattern, imported rather than respelled. Keeping a second copy here with
# capture groups is how the two drifted: `(.*)` accepted "v1.2.3-" while the
# other rejected it, and tightening this copy alone silently began rejecting
# real pre-release tags like v1.2.3-rc-2 — which `_parse_ls_remote` then drops
# from the available-release list with no message at all.
_TAG_RE = _TAG_LINE_RE


def parse_version(tag: str) -> tuple | None:
    """Sortable key for a release tag, or None if it is not one.

    A pre-release (v1.0.0-rc1) sorts *before* its final release, per semver.
    """
    m = _TAG_RE.match((tag or "").strip())
    if not m:
        return None
    major, minor, patch = m.group("major"), m.group("minor"), m.group("patch")
    suffix = m.group("suffix")
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

def _fill_bundle_release(result: dict) -> None:
    """What an installed copy — dmg, AppImage, Setup.exe — can be told.

    It cannot pull and rebuild: there is no git and no Node. But it is not the
    dead end the check used to report. Until now every packaged installation
    got `not_a_git_install`, which the interface renders as "This installation
    cannot update itself." — the one sentence the module docstring above says
    this work exists to delete — and it got it WITHOUT asking anyone, so a
    tester was told nothing was available while a release sat on the page.

    `github_releases` was written for exactly this and, until now, called from
    nowhere; the docstring's claim that a bundle "uses github_releases instead"
    described an intention, not a code path.

    Nothing here raises: `latest_release` answers with a reason instead
    (`rate_limited`, `unreachable`, `not_found`), and the reason is what the
    user sees, so "no newer version" is never confused with "could not look".
    """
    from backend.api.services import github_releases

    result["download_url"] = github_releases.releases_page_url()

    release, why = github_releases.latest_release()
    if not release:
        result["reason"] = why or "remote_unreachable"
        return

    latest = str(release.get("tag_name") or "").strip()
    if not parse_version(latest):
        # A release named something we do not recognise as a version is not a
        # thing to offer; saying "up to date" would be a guess.
        result["reason"] = "no_release_tag"
        return

    result["latest"] = latest
    current_key = parse_version(result.get("current_release") or "")
    if current_key is None:
        result["available"] = True
        result["reason"] = "no_local_release"
    elif parse_version(latest) > current_key:
        result["available"] = True
        result["reason"] = "bundle_download"
    else:
        result["reason"] = "up_to_date"
        return

    # The notes of the version being OFFERED. `_changelog_for` reads the
    # CHANGELOG of the copy on disk, which is the OLD one — for a bundle the
    # only description of the new version is the release body.
    body = str(release.get("body") or "").strip()
    if body:
        result["notes"] = body[:4000]


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

    if kind == "bundle":
        _fill_bundle_release(result)
        _check_cache.update(at=now, result=result)
        return result

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
    # `current` is whatever `git describe --tags` found, which is the nearest
    # reachable tag of ANY shape — this project creates tags like
    # "wip-uncommitted-2026-09-14". Comparing a tuple with None raises
    # TypeError, which `routes/system.py` swallows into reason="check_failed":
    # a dead update dialog with no explanation. Resolve the key once, and treat
    # "not a release tag" exactly like "no tag at all".
    current_key = parse_version(current) if current else None
    if current_key is None:
        # Untagged checkout (a developer tree, or a clone predating tags):
        # offer the release but let the UI say the comparison is approximate.
        result["available"] = True
        result["reason"] = "no_local_release"
    elif parse_version(latest) > current_key:
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

    deps_cmd, deps_refusal = python_dependency_command(
        requirements_changed=_changed_between(previous, target, "requirements.txt"),
        macos_env_changed=_changed_between(previous, target, "environment-macos.yml"),
        target_tag=tag,
    )
    if deps_refusal:
        return rollback(deps_refusal)
    if deps_cmd:
        if not _run_step("dependencies", deps_cmd, PROJECT_ROOT):
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


def _find_conda(prefix: str, environ) -> str | None:
    """The conda executable that owns `prefix`.

    CONDA_EXE is set only in an activated shell; an app started from the
    Finder or the Dock has none, so look next to the environment too:
    `<base>/envs/<name>` -> `<base>/bin/conda`, or the base env itself.
    """
    import shutil
    if environ.get("CONDA_EXE"):
        return environ["CONDA_EXE"]
    p = Path(prefix)
    for base in (p.parent.parent, p):
        for rel in ("bin/conda", "condabin/conda"):
            if (base / rel).is_file():
                return str(base / rel)
    return shutil.which("conda")


def python_dependency_command(
    requirements_changed: bool,
    macos_env_changed: bool,
    platform: str | None = None,
    prefix: str | None = None,
    environ: dict | None = None,
    executable: str | None = None,
    target_tag: str | None = None,
) -> tuple[list[str] | None, str | None]:
    """How an update brings the Python packages in line: ``(argv, None)`` to
    run, ``(None, None)`` when nothing changed, ``(None, message)`` to refuse.

    A conda environment on macOS is never touched with pip. There the pip
    wheels of torch, scikit-learn and faiss-cpu each bundle their own OpenMP
    runtime, and the second one to start aborts the backend with
    "OMP: Error #15" — the environment is built from environment-macos.yml so
    that exactly one runtime exists, and it is updated from that file too.
    """
    import os
    import sys
    platform = sys.platform if platform is None else platform
    prefix = sys.prefix if prefix is None else prefix
    environ = os.environ if environ is None else environ
    executable = sys.executable if executable is None else executable

    if platform == "darwin" and (Path(prefix) / "conda-meta").is_dir():
        if not (requirements_changed or macos_env_changed):
            return None, None
        conda = _find_conda(prefix, environ)
        if not conda:
            # The refusal rolls the code back, so "run it in the install
            # folder" would apply the OLD file; name the target explicitly.
            return None, (
                "The Python packages changed, and this macOS environment must "
                "be updated with conda, not pip, but conda was not found. In "
                "the Orienta folder run:  git checkout "
                f"{target_tag or '<new version>'} && conda env update -p "
                f"{prefix} -f environment-macos.yml  — then restart Orienta."
            )
        return [conda, "env", "update", "-p", prefix,
                "-f", "environment-macos.yml"], None

    if requirements_changed:
        return [executable, "-m", "pip", "install", "-r", "requirements.txt"], None
    return None, None


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
