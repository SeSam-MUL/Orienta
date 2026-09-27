"""A path becomes the same NAME on every platform.

The bug these pin, found by the first Linux run of the suite: `Path` is
`WindowsPath` on Windows and `PosixPath` elsewhere, and a backslash is an
ordinary filename character on POSIX. `Path(r"C:\\Users\\operator\\Al.cif").stem`
is "Al" on Windows and "C:\\Users\\operator\\Al" on Linux and macOS — an
operator's username and directory layout, printed into the methods paragraph
and into every exported .h5.

These tests run on Windows too, where the bug cannot reproduce. So they do not
ask "what does this platform do" — they pin the POSIX reading explicitly with
PurePosixPath and require the helper to beat it.
"""
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from display_names import display_basename, display_stem

WINDOWS_PATH = r"C:\Users\operator\Orienta\Database\CIF_Library\Al.cif"
POSIX_PATH = "/home/sesam/Orienta/Database/CIF_Library/Al.cif"
BARE_NAME = "Al.cif"


# ---------------------------------------------------------------------------
# The three rows of the finding's table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("given", [WINDOWS_PATH, POSIX_PATH, BARE_NAME])
def test_every_shape_of_path_gives_the_same_name(given):
    assert display_stem(given) == "Al"
    assert display_basename(given) == "Al.cif"


def test_the_windows_path_is_where_the_platform_disagrees_with_itself():
    """The regression, stated without asking which OS is running.

    PurePosixPath IS what `Path` does on Linux and macOS; PureWindowsPath is
    what it does on Windows. They disagree on this string, which is the whole
    bug — and the helper must agree with the Windows reading, because that is
    the one that yields a name rather than a path.
    """
    assert PureWindowsPath(WINDOWS_PATH).stem == "Al"
    leaked = PurePosixPath(WINDOWS_PATH).stem
    assert leaked == r"C:\Users\operator\Orienta\Database\CIF_Library\Al"
    assert "operator" in leaked                      # the username, in a name
    assert display_stem(WINDOWS_PATH) == "Al"
    assert "operator" not in display_stem(WINDOWS_PATH)


def test_a_posix_path_is_read_the_posix_way_on_windows_too():
    # The mirror image: PureWindowsPath happens to handle forward slashes, but
    # do not rely on that — pin the helper itself.
    assert display_stem(POSIX_PATH) == "Al"
    assert display_basename(POSIX_PATH) == "Al.cif"


# ---------------------------------------------------------------------------
# It must stay a stem, not become a different kind of string
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,stem", [
    ("Al.cif", "Al"),
    ("a.b.c", "a.b"),                                  # only the last suffix
    (".gitignore", ".gitignore"),                      # a dotfile keeps its dot
    ("no_suffix", "no_suffix"),
    ("Al (Al) [cF4] {20kV}.sht", "Al (Al) [cF4] {20kV}"),   # the library's names
])
def test_it_keeps_pathlib_s_own_rules_for_the_name_part(name, stem):
    assert display_stem(name) == stem
    assert display_stem("/tmp/dir/" + name) == stem
    assert display_stem("C:\\dir\\" + name) == stem


def test_an_empty_string_stays_empty():
    # Not folded into the table above: "/tmp/dir/" + "" is a trailing
    # separator, which legitimately reads as "dir" — see the test below.
    assert display_stem("") == ""
    assert display_basename("") == ""


def test_a_path_object_is_accepted_not_just_a_string():
    assert display_stem(Path("dir") / "Al.cif") == "Al"


def test_a_trailing_separator_does_not_produce_an_empty_name():
    # Path("x/y/").name is "y" — a directory handed in by mistake should still
    # read as a name, not as "".
    assert display_stem("/home/sesam/Database/") == "Database"
    assert display_stem("C:\\Database\\") == "Database"


# ---------------------------------------------------------------------------
# The callers that carry it into a user-visible record
# ---------------------------------------------------------------------------

def test_the_eds_strength_sentence_names_the_phase_not_the_operator(monkeypatch):
    """`_eds_strengths_by_phase_name` is the site the Linux run caught.

    Its output is recorded in the methods paragraph and in the exported .h5,
    keyed by the name a reader sees.

    `indexing.Path` is pinned to PurePosixPath for the duration: that IS what
    the module binds on Linux and macOS, and without it this test cannot fail
    on Windows — `Path.stem` would answer "Al" there whatever the code does.
    With it, the pre-fix line (`Path(p).stem`) returns the operator's whole
    path and this assertion catches it.
    """
    from backend.api.routes import indexing as idx

    monkeypatch.setattr(idx, "Path", PurePosixPath)

    class _Req:
        eds_phase_strengths = {WINDOWS_PATH: 0.75}
        method = "spherical"
        sht_paths = [WINDOWS_PATH]
        cif_paths = None
        dictionary_paths = None

    out = idx._eds_strengths_by_phase_name(_Req())
    assert out == {"Al": 0.75}, out
    assert not any("operator" in k for k in out)


@pytest.mark.parametrize("phase_file", [
    # The one Spherical always uses, and the one the first version of this
    # fix missed: the name comes back through `meta.formula`, which
    # phase_metadata builds from the stem, so fixing only the callers left
    # the leak in place for .sht.
    r"C:\Users\operator\Orienta\Database\EBSD_SHT_Database\Al\Al (Al) [cF4] {20kV}.sht",
    r"C:\Users\operator\Orienta\Database\Dictionary_Library\Al_master_E20kV.h5",
    # No .cif row: extract_metadata_from_cif RAISES on a missing file rather
    # than falling back to the name, so there the caller's except branch is
    # what derives the name — covered by the EDS-strength test below.
])
def test_phase_metadata_never_puts_a_path_in_a_name(phase_file):
    """`phase_metadata` is where a file name becomes a phase identity.

    A POSIX path object is passed deliberately: that is what `Path` resolves
    to on Linux and macOS, so this reproduces their reading on Windows too.
    The file does not exist on either platform, which is the point — every
    reader below then falls through to the name, which is the leaking step.
    """
    from phase_metadata import (
        extract_metadata_from_sht_filename, get_phase_metadata,
    )

    class _MissingPosixFile(PurePosixPath):
        """A POSIX path whose file is absent, without touching the disk.

        PurePosixPath alone has no I/O at all, so the readers raise
        AttributeError instead of the FileNotFoundError they handle.
        """
        def read_text(self, *a, **kw):
            raise FileNotFoundError(str(self))

        def exists(self):
            return False

        def is_file(self):
            return False

    meta = get_phase_metadata(_MissingPosixFile(phase_file))
    for field in ("formula", "display_label", "phase_name"):
        value = str(getattr(meta, field, "") or "")
        assert "operator" not in value, (field, value)
        assert "\\" not in value, (field, value)

    if phase_file.endswith(".sht"):
        sht = extract_metadata_from_sht_filename(PurePosixPath(phase_file))
        assert sht.formula == "Al", sht.formula
        assert "operator" not in str(sht.display_label)


def test_the_calibration_key_is_the_same_string_on_either_platform():
    """BatchPage sends a file name, CalibrationStore is keyed by the stem.

    If those two derivations disagree the /copy-pc lookup 404s — which is the
    bug `_resolve_dataset_name` was written for in the first place, and a
    platform-dependent stem brings it back for a path-shaped name.
    """
    from backend.api.routes import batch_v2

    assert batch_v2._resolve_dataset_name("Sample.h5oina") == "Sample"
    assert batch_v2._resolve_dataset_name(WINDOWS_PATH) == "Al"
    assert batch_v2._resolve_dataset_name(POSIX_PATH) == "Al"
    assert batch_v2._resolve_dataset_name("") == ""


def test_every_calibration_key_comes_from_one_derivation():
    """Writers and readers of the CalibrationStore key, held side by side.

    The key is written in ebsd_viewer (on load) and batch_manager (per batch
    file), and read in batch_v2 (/copy-pc), batch_manager and indexing. Before
    this change they all said `Path.stem`, which agreed by construction. They
    now have to agree by intent instead, so this pins it: one derivation, used
    by every one of them. A reader that drifts makes /copy-pc 404 — the bug
    `_resolve_dataset_name` was written for.
    """
    import inspect

    from backend.api.routes import batch_v2, ebsd_viewer
    from backend.api.services import batch_manager

    for module in (batch_v2, ebsd_viewer, batch_manager):
        src = inspect.getsource(module)
        for bad in ("Path(path).stem", "Path(file_path).stem", "fp.stem",
                    "_P(path).stem", "Path(_ebsd_file_path).stem"):
            assert bad not in src, (module.__name__, bad)

    # And the derivations really are the same function, not two lookalikes.
    from display_names import display_stem
    assert batch_v2.display_stem is display_stem
    assert ebsd_viewer.display_stem is display_stem
    assert batch_manager.display_stem is display_stem
