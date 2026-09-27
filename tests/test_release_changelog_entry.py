"""The changelog that ships must have an entry for the version that ships.

`CHANGELOG.md` travels inside the runtime package -- the same `EXTRA_FILES`
line as `CITATION.cff` in `scripts/build_runtime_package.py` -- so it is not a
repository detail. It is the only file in an installation that answers "what
changed in the version I am running", and Settings -> About points at the
version it should answer for.

Writing the entry is a manual step on the release checklist, and Build 1 of
0.4.6 was discarded because a different manual step on that same checklist was
missed. Measured on 2026-09-27: the dev tree's CHANGELOG had no section for
v0.2.5 or v0.2.6 at all. Both entries were written in the public repository and
never came back, because the rule for these files ports one way by hand -- so
every installed copy carried a changelog with a hole where two released
versions should have been.

This file deliberately does NOT import `scripts/build_release.py`. That script
is not part of the published subset, and a module-level import of it makes the
whole suite fail collection in a clone with `ModuleNotFoundError` rather than
skipping -- which is why `test_release_citation_version.py` has to be held back
from the port. Parsing three lines of markdown here costs less than that, and
it has a second benefit: the parser under test in the build script and the
parser here are independent, so the two have to agree about the file for this
to pass.

**It can travel to the public repository, but not before two things there are
true.** Measured on 2026-09-27 against `origin/main`: 3 of these 4 tests fail
there, because the public `frontend/package.json` still says `"version":
"1.0.0"` (it was never synced) while the newest changelog entry is v0.3.0. So
the port has to bring the real version into that file and the release's section
into its changelog -- both already decided -- and only then does this file hold
it honest instead of failing on arrival. "Portable" without that precondition
was too strong a word, and a review was right to say so.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: `## v0.4.6 — 2026-09-27`. Level two only: a version named in prose, or under
#: a `###`, is not an entry, and accepting one would let a release pass on a
#: mention of itself.
_HEADING = re.compile(r"^## +v?(?P<version>\d+\.\d+\.\d+(?:[-+.][\w.-]+)?)\b")


def _changelog_versions() -> list[str]:
    text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    return [m.group("version") for m in map(_HEADING.match, text.splitlines()) if m]


def _package_version() -> str:
    pkg = json.loads((REPO_ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
    return pkg["version"]


def test_the_changelog_has_a_section_for_the_version_we_would_ship():
    entries = _changelog_versions()
    version = _package_version()
    assert entries, "CHANGELOG.md has no version headings at all"
    assert version in entries, (
        f"frontend/package.json says {version!r} and CHANGELOG.md's entries are "
        f"{entries[:3]}. CHANGELOG.md ships inside the runtime package, so this "
        f"build would install a changelog that never mentions {version}."
    )


def test_the_newest_entry_is_the_version_we_would_ship():
    """Not just present -- first.

    An entry that exists but sits below an older one means the sections were
    inserted in the wrong place, and a reader takes the top of the file as "what
    is new". Kept separate from the test above so a failure says which of the two
    things is wrong.
    """
    entries = _changelog_versions()
    # Without this, an empty changelog raises IndexError from the line below and
    # reports a crash instead of the defect -- the same hazard the production
    # check guards with its "no entries at all" branch. A review caught that the
    # guard existed there and not here.
    assert entries, "CHANGELOG.md has no version headings at all"
    assert entries[0] == _package_version(), (
        f"CHANGELOG.md's newest entry is v{entries[0]}, but this tree builds "
        f"{_package_version()}."
    )


def test_every_released_version_still_has_its_entry():
    """A section may never be dropped, only added to.

    This is the defect that was actually found: sections for two released
    versions were simply missing, and nothing noticed for three weeks, because
    every check looked at the newest entry only. Pinning the known history means
    a future edit that removes one fails here instead of shipping.

    New releases are added to this list as they are made -- the list is the
    record of what has been PUBLISHED, so it is meant to be edited upward, and
    only once a release exists.

    It therefore does NOT contain the version this tree is preparing. A review
    caught 0.4.6 in here while the release page was still a draft, and that was
    wrong twice over: the public repository's tags stop at v0.3.0, so the list
    claimed something untrue; and had this release come out as 0.5.0 instead
    (still an open question at the time), this test would have demanded a
    section for a version that never existed, while its own docstring says a
    section may never be dropped. The unreleased version is covered by the two
    tests above, which read it from package.json instead of a hardcoded list.
    """
    published = [
        "0.3.0", "0.2.6", "0.2.5", "0.2.4", "0.2.3", "0.2.2", "0.2.1", "0.2.0",
        "0.1.0",
    ]
    entries = _changelog_versions()
    missing = [v for v in published if v not in entries]
    assert not missing, f"CHANGELOG.md lost the section(s) for: {missing}"


def test_the_entries_run_newest_first():
    """The file says "newest first" in its own second line; hold it to that."""
    def key(version: str) -> tuple:
        # The parser accepts suffixes (`0.4.6-rc-1`, `0.4.6.post1`), so a naive
        # int() over every dot-separated part raises on versions the parser
        # happily returns -- a review found the two disagreeing about the
        # grammar. Compare on the leading numeric parts and ignore the rest:
        # ordering a pre-release against its release is not this test's job.
        head = re.split(r"[-+]", version, maxsplit=1)[0]
        parts = []
        for piece in head.split("."):
            if not piece.isdigit():
                break
            parts.append(int(piece))
        return tuple(parts)

    entries = _changelog_versions()
    ordered = sorted(entries, key=key, reverse=True)
    assert entries == ordered, f"out of order: {entries}"
