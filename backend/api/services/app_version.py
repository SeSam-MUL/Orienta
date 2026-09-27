"""App version identity, derived from git.

The project has no maintained release number — the meaningful identity of an
installed copy is the git commit it was checked out at.  This module resolves
that identity once per process:

    1. `git log -1` / `git rev-parse` against the project root (preferred —
       gives short hash, commit date and branch),
    2. manual parse of `.git/HEAD` + refs when the git binary is missing
       (hash + branch, no date),
    3. the `VERSION` file a runtime package ships, for an installation made by
       the setup — it has no `.git`, and without this it could never be told
       apart from a newer or older release,
    4. "unknown" when there is none of the above (e.g. a bare zip download).

Used by `GET /api/system/version`, the About section, the log-file banner and
the diagnostics export.
"""

from __future__ import annotations

import logging
import re
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


def _parse_describe(described: str) -> tuple[str | None, int | None]:
    """Split `git describe` output into (release tag, commits since it).

    'v0.2.0'              -> ('v0.2.0', 0)      exactly on the release
    'v0.1.0-164-g5faee31' -> ('v0.1.0', 164)    164 commits past it
    """
    if not described:
        return None, None
    m = re.match(r"^(?P<tag>.+?)-(?P<n>\d+)-g[0-9a-f]+$", described)
    if m:
        return m.group("tag"), int(m.group("n"))
    return described, 0


#: THE definition of "is this a release tag". `updater.parse_version` builds on
#: this pattern rather than keeping its own, so the check cannot offer a tag the
#: download step then refuses as unparseable.
#:
#: The suffix allows "-" so that real pre-release tags parse: v1.2.3-beta-1,
#: v1.2.3-rc-2, v1.0.0+build-5. A stricter class silently dropped them from the
#: available-release list — a failure with no message, because
#: `_parse_ls_remote` discards whatever `parse_version` cannot read.
_TAG_LINE_RE = re.compile(
    r"^v?(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?:[-+.](?P<suffix>[\w.-]+))?$"
)


def read_version_state(root: Path) -> tuple[str, str | None]:
    """(state, tag) for the VERSION file, distinguishing four different answers.

        ("tag", "v0.4.0")   a release tag
        ("absent", None)    there is no VERSION file
        ("unreadable", None) it is there and we may not read it
        ("junk", None)      it is there and is not a release tag

    The distinction is the whole point. Callers that collapse "unreadable" into
    "absent" conclude "this is not an installed copy" from a permission error or
    a lock — and then tell a user who has a perfectly updatable installation
    that it cannot update itself.

    `utf-8-sig` because PowerShell writes a BOM by default, and `str.strip()`
    does not remove one: a VERSION file regenerated on Windows would otherwise
    parse as junk and take the installation out of the update channel.
    """
    path = root / "VERSION"
    try:
        text = path.read_text(encoding="utf-8-sig").strip()
    except FileNotFoundError:
        return "absent", None
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("%s exists but cannot be read: %s", path, exc)
        return "unreadable", None
    return ("tag", text) if _TAG_LINE_RE.match(text) else ("junk", None)


def _read_version_file(root: Path) -> str | None:
    """The release tag a runtime package was built from, or None.

    One line, nothing else. An installation made by the setup has no `.git`, so
    without this it reports "unknown" and the update check has nothing to compare
    the newest release against — it can then never say "up to date" and never
    say "there is a newer one" either.

    Anything that is not a release tag is ignored rather than shown: a
    half-written file must not become the version the update check compares
    against. Use `read_version_state` where the REASON matters.
    """
    state, tag = read_version_state(root)
    return tag if state == "tag" else None


@lru_cache(maxsize=1)
def get_version_info() -> dict:
    """Resolve the app's version identity. Cached for the process lifetime.

    Prefers the release lineage from `git describe --tags`, which answers both
    "which release is this" and "how far past it" in one string. Falls back to
    date + hash when the checkout has no tags at all.

    Returns: app, version (display string), release (tag), commits_since_release,
    commit, commit_date, branch, source ("git" | "git-files" | "unknown").
    Never raises.
    """
    commit = None
    commit_date = None
    branch = None
    release = None
    commits_since = None
    source = "unknown"

    out = _run_git(["log", "-1", "--format=%h|%cs"], PROJECT_ROOT)
    if out and "|" in out:
        commit, commit_date = out.split("|", 1)
        branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], PROJECT_ROOT)
        source = "git"
        # --tags so lightweight tags count too; --always never fails on a
        # repo without tags (it just returns the hash, which we ignore here).
        release, commits_since = _parse_describe(
            _run_git(["describe", "--tags", "--abbrev=8"], PROJECT_ROOT) or ""
        )
    else:
        commit, branch = _read_git_files(PROJECT_ROOT)
        if commit:
            source = "git-files"
        if not commit:
            release = _read_version_file(PROJECT_ROOT)
            if release:
                # commits_since = 0 makes the FIRST arm of the version-string
                # chain below match, so the display is "v0.4.0" and not
                # "v0.4.0+None (None)". Do not add a new arm at the end of that
                # chain: `elif release:` catches it first and it is unreachable.
                commits_since = 0
                source = "version-file"

    if release and commits_since == 0:
        version = release                                  # v0.2.0
    elif release:
        version = f"{release}+{commits_since} ({commit})"  # v0.1.0+164 (5faee319)
    elif commit and commit_date:
        version = f"{commit_date} ({commit})"              # no tags in this repo
    elif commit:
        version = f"({commit})"
    else:
        version = "unknown"

    return {
        "app": APP_NAME,
        "version": version,
        "release": release,
        "commits_since_release": commits_since,
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
