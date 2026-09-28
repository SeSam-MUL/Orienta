"""No answer of the phase-library endpoints may carry a machine-local path.

Found by f7 in the live run: `simulation.sht.provenance.source_xtal.path` came
back as `C:\\Users\\<name>\\...`. It was not one field — measured over the real
library, **29 of 36** phases carried two (`source_xtal` and `source_cif`), and
the same block is served by the Database browser's File Info panel, so the leak
was never only ours.

Why a guard and not just the fix: this project has already had a local path
reach a manuscript (`Path.stem` is platform-dependent -- 2026-09-24), and these
answers are pasted into bug reports and quoted in writing. The rule has to be
"no response carries one", checked over every field, because the next absolute
path will arrive in a field nobody thought about -- which is exactly how this
one did.

The test walks the WHOLE response rather than asserting on the two known keys:
an assertion on `source_xtal.path` would have been green on the day a third
field appeared.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.services import phase_library as pl        # noqa: E402
from backend.api.services import sht_provenance as sp       # noqa: E402

#: Windows drive letters, UNC shares, and every POSIX root that is machine-local.
#: Anchored: a bare `Database/CIF_Library/Al.cif` is exactly what we DO want.
#:
#: The list grew after the 2026-09-27 review probed it. `/media/<user>/` and
#: `/run/media/<user>/` are the removable-media mounts on Ubuntu and on
#: Fedora/Arch and they CONTAIN THE ACCOUNT NAME -- a library on a USB drive
#: mounts exactly there. They slipped through, and together with the flavour bug
#: in `_public_path` that made a fully silent leak: the path was passed through
#: verbatim AND nothing fired.
ABSOLUTE = re.compile(
    r"^[A-Za-z]:[\\/]"                       # C:\… or C:/…
    r"|^\\\\"                                # \\server\share
    r"|^/(?:home|Users|root)/"               # POSIX homes
    r"|^/(?:media|run/media|Volumes|mnt)/"   # removable and mounted volumes
    r"|^/(?:opt|srv|var/folders|tmp)/"       # other machine-local roots
)

#: Strings that are a path leak even in the middle of a sentence -- a message or
#: a reference that quotes one still ships the account name. Case-insensitive and
#: separator-agnostic: `c:/users/<name>/` is the same leak as `C:\Users\<name>\`,
#: and the review found the lowercase form passing.
EMBEDDED = re.compile(
    r"[A-Za-z]:[\\/]users[\\/]"
    r"|(?:^|[\\/\s])users[\\/][^\\/\s]+[\\/]"
    r"|/home/[^/\s]+/"
    r"|/users/[^/\s]+/"
    r"|/(?:media|run/media)/[^/\s]+/",
    re.IGNORECASE,
)


def _walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


def _offenders(doc):
    return [(p, v) for p, v in _walk(doc)
            if ABSOLUTE.match(v) or EMBEDDED.search(v)]


def _library_or_skip():
    if not (ROOT / "Database" / "CIF_Library").is_dir():
        pytest.skip("crystal library not in this checkout")


def test_the_index_answer_carries_no_local_path():
    _library_or_skip()
    assert _offenders(pl.build_index_document()) == []


def test_no_phase_profile_carries_a_local_path():
    """Every phase, not a sample: the leak was in 29 of 36, so a spot check of
    the first few would have found it -- and a spot check of the wrong few
    would not."""
    _library_or_skip()
    doc = pl.build_index_document()
    keys = [p["key"] for p in doc["phases"]]
    assert len(keys) >= 30, f"only {len(keys)} phases -- is the library complete?"
    bad = {}
    for k in keys:
        off = _offenders(pl.build_phase_detail(k))
        if off:
            bad[k] = off
    assert bad == {}, json.dumps(
        {k: v[:2] for k, v in list(bad.items())[:5]}, indent=2)


def test_a_resolved_file_is_reported_project_relative():
    """The positive half. "No absolute path" is also satisfied by emitting
    nothing at all, which would pass the two tests above while destroying the
    field."""
    _library_or_skip()
    d = pl.build_phase_detail("Al")
    prov = d["simulation"]["sht"]["provenance"]
    assert prov["source_xtal"]["path"] == "Database/XTAL_Library/Al.xtal"
    assert prov["source_cif"]["path"] == "Database/CIF_Library/Al.cif"
    assert "\\" not in prov["source_xtal"]["path"], "POSIX separators, one form"


def test_a_path_recorded_on_another_machine_is_reduced_to_its_name(tmp_path):
    """The case the shipped library cannot produce, and the one that matters
    most: a sidecar written elsewhere names a file that is not here, and its
    recorded path is BY DEFINITION somebody else's account.

    It is kept -- reduced -- rather than dropped, so "the sidecar named a file
    we could not find" stays visible.
    """
    entry = sp._file_entry(
        "NotHere.xtal", tmp_path,
        recorded_path=r"C:\Users\someone-else\Orienta\Database\XTAL_Library\NotHere.xtal")
    assert entry["found"] is False
    assert entry["path"] is None
    assert entry["recorded_path"] == "NotHere.xtal"
    assert _offenders(entry) == []


def test_an_already_relative_path_is_passed_through(tmp_path):
    """The first version of `_public_path` reduced these to a bare name.

    Sidecars written since this change record a project-relative path, so that
    bug threw away exactly the information the field exists for -- and it looked
    like a pass, because "no absolute path" was still true.
    """
    assert sp._public_path("Database/XTAL_Library/Ni.xtal") == \
        "Database/XTAL_Library/Ni.xtal"
    assert sp._public_path(r"Database\XTAL_Library\Ni.xtal") == \
        "Database/XTAL_Library/Ni.xtal", "one form, POSIX"
    assert sp._is_outside_project("Database/XTAL_Library/Ni.xtal") is False


def test_an_absolute_path_inside_the_project_is_not_elsewhere():
    """`elsewhere` must not be "the string changed": an absolute path pointing
    into our own tree changes too (it becomes relative), and flagging that would
    put a foreign-library warning on ordinary local files."""
    inside = ROOT / "Database" / "XTAL_Library" / "Al.xtal"
    assert sp._is_outside_project(inside) is False
    assert sp._public_path(inside) == "Database/XTAL_Library/Al.xtal"
    assert sp._is_outside_project(r"C:\Users\someone-else\Al.xtal") is True


def test_a_file_outside_the_project_reports_its_name_not_a_relative_guess(tmp_path):
    """A library on another drive has no relative form. Inventing one with
    `..` would point at something that is not there, and on another machine at
    something that is."""
    f = tmp_path / "Elsewhere.cif"
    f.write_text("data_x\n", encoding="utf-8")
    assert sp._public_path(f) == "Elsewhere.cif"
    assert _offenders({"p": sp._public_path(f)}) == []


def test_a_new_sidecar_is_written_with_project_relative_paths(tmp_path):
    """The leak at rest, which is the worse one: this file is written INTO
    `Database/`, and the library is copied and mailed between machines."""
    xtal = ROOT / "Database" / "XTAL_Library" / "Al.xtal"
    if not xtal.is_file():
        pytest.skip("crystal library not in this checkout")
    out = sp.write_provenance_sidecar(
        tmp_path / "Al.sht", xtal_path=xtal,
        cif_dir=ROOT / "Database" / "CIF_Library", params={"engine": "ours"})
    doc = json.loads(Path(out).read_text(encoding="utf-8"))
    assert doc["source_xtal"]["path"] == "Database/XTAL_Library/Al.xtal"
    assert doc["source_cif"]["path"] == "Database/CIF_Library/Al.cif"
    assert _offenders(doc) == []


def test_a_foreign_flavour_path_is_recognised_in_both_directions():
    """The 2026-09-27 review's HIGH 3, and the reason it is not theory: all 28
    recorded paths in this tree's 14 sidecars are absolute WINDOWS paths, and the
    .dmg and AppImage have existed since 2026-09-24.

    `Path.is_absolute()` answers for the HOST's flavour only. Measured before the
    fix, on this Windows host: a POSIX path came back VERBATIM with
    `recorded_elsewhere: False` -- "this is local" about the one kind of path that
    certainly is not. The mirror is the case that ships: on Linux and macOS every
    one of those 28 Windows paths would have passed through with the account name
    in it.
    """
    posix_foreign = "/Users/other-account/Orienta/Database/XTAL_Library/Al.xtal"
    win_foreign = r"C:\Users\other-account\Orienta\Database\XTAL_Library\Al.xtal"
    for s in (posix_foreign, win_foreign):
        assert sp._is_absolute_anywhere(s) is True, s
        assert sp._is_outside_project(s) is True, s
        assert sp._public_path(s) == "Al.xtal", s
        assert _offenders({"p": sp._public_path(s)}) == [], s

    # And the flavour asymmetry that caused it, stated so nobody "simplifies"
    # the helper back to `Path(...)`:
    assert PureWindowsPath(posix_foreign).is_absolute() is False
    assert PurePosixPath(posix_foreign).is_absolute() is True
    assert PureWindowsPath(win_foreign).is_absolute() is True
    assert PurePosixPath(win_foreign).is_absolute() is False


def test_the_basename_fallback_splits_on_both_separators():
    """`Path(s).name` is flavour-bound too -- the 2026-09-24 finding verbatim: on
    Linux a Windows path has no separators, so `.name` hands back the WHOLE
    string, account name included. A bare-name fallback that leaks the path it
    was meant to strip is worse than none.

    HONEST LIMIT: on Windows this test is weak, and I only know that because I
    mutated the helper back to `Path(s).name` and it SURVIVED. `WindowsPath`
    treats `/` as a separator too, so every input below gives the right answer
    either way here. The assertion that bites on every host is the source guard
    below; this one pins the behaviour and would bite on the Linux and macOS
    runners.
    """
    assert sp._basename_anyflavour(r"C:\Users\someone\lib\Al.xtal") == "Al.xtal"
    assert sp._basename_anyflavour("/home/someone/lib/Al.xtal") == "Al.xtal"
    assert sp._basename_anyflavour(r"\\server\share\Al.xtal") == "Al.xtal"
    assert sp._basename_anyflavour("Al.xtal") == "Al.xtal"

    # What the flavour-bound call would do to that first path on a POSIX host.
    assert PurePosixPath(r"C:\Users\someone\lib\Al.xtal").name != "Al.xtal", (
        "if this ever becomes equal, PurePosixPath learned about backslashes "
        "and the guard below can go")


def test_neither_helper_delegates_to_the_hosts_flavour():
    """A source guard, because the defect is invisible on Windows.

    Both helpers exist precisely so that one string gets the same answer on every
    platform. A `Path(...)` inside either of them is the bug, and on this host no
    input can demonstrate it -- the tests above pass with the bug reinstated. The
    project already uses this shape where behaviour cannot be observed locally
    (the Electron test that reads `main.js` and fails if the update slips behind
    the spawn).

    Parsed with `ast`, NOT grepped: my first version searched the file text and
    failed on its own docstrings, which quote `Path(s).name` while explaining why
    it is wrong. A guard that prose can satisfy -- or break -- is not a guard.
    That is the same mistake as the export guard that matched a word in a
    comment, two commits ago.
    """
    import ast

    tree = ast.parse(Path(sp.__file__).read_text(encoding="utf-8"))
    funcs = {n.name: n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef)}
    for want in ("_is_absolute_anywhere", "_basename_anyflavour", "_public_path"):
        assert want in funcs, f"{want} is gone -- did it get inlined?"

    def called(fn):
        """Every function-ish name called in `fn`, docstrings excluded by ast."""
        out = set()
        for node in ast.walk(funcs[fn]):
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Name):
                    out.add(f.id)
                elif isinstance(f, ast.Attribute):
                    out.add(f.attr)
        return out

    flavours = called("_is_absolute_anywhere")
    assert {"PureWindowsPath", "PurePosixPath"} <= flavours, (
        f"both flavours must be asked by name; got {sorted(flavours)}")

    base = called("_basename_anyflavour")
    assert "split" in base, "the basename must split on both separators"
    assert "Path" not in base, (
        "the basename must not come from the host's flavour -- that is the whole "
        "defect, and on Windows no input can demonstrate it")

    assert "_basename_anyflavour" in called("_public_path"), (
        "_public_path's fallback must go through the helper, not `.name`")


def test_the_guard_catches_the_forms_that_slipped_through():
    """Each of these was probed by the review and NOT caught. The first two carry
    the account name, so they are leaks, not untidiness."""
    for s in ("/media/someone/USB-4TB/Database/Al.xtal",
              "/run/media/someone/USB-4TB/Database/Al.xtal",
              "/Volumes/Data/Database/Al.xtal",
              "/mnt/e/Database/Al.xtal",
              "/root/Database/Al.xtal",
              "/opt/orienta/Database/Al.xtal",
              "/srv/share/Database/Al.xtal",
              "/var/folders/ab/cd/T/Al.xtal",
              "/tmp/Al.xtal",
              "c:/users/someone/lib/Al.cif",
              "see Users/someone/lib/Al.cif for the source"):
        assert _offenders({"p": s}), s


def test_the_guard_itself_recognises_a_local_path():
    """A detector that matches nothing passes everything.

    Each form is one this project has actually produced: the Windows install
    path, a UNC share (the network library), and the two POSIX homes from the
    macOS and Linux builds.

    The account names here are SYNTHETIC on purpose. They used to be the real
    ones, which made this a guard against leaking account names that shipped one
    itself -- in a repo whose public half is curated by hand and whose
    `port_guard` never reads a held-back file's contents. A synthetic name tests
    the pattern exactly as well.
    """
    for s in (r"C:\Users\someone\Kikuchipy_GUI\Database\CIF_Library\Al.cif",
              r"\\server\share\Database\Al.cif",
              "/home/someone/EMsoftXtal/Al.xtal",
              "/Users/someone/Orienta/Database/Al.cif"):
        assert _offenders({"p": s}), s
    # And that it does NOT fire on what we want to see.
    for s in ("Database/CIF_Library/Al.cif", "Al.xtal",
              "https://doi.org/10.1016/j.ultramic.2019.112841"):
        assert _offenders({"p": s}) == [], s
