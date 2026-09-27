"""
Shared pytest fixtures and markers for Kikuchipy GUI tests.

Provides fixtures for real test data files in Test_data/.
Integration tests using these fixtures are auto-skipped when files are missing.
"""

import builtins
import functools
import io
import sys
import os
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Keep test runs out of the real logs/orienta.log (importing backend.api.main
# installs a rotating file handler at import time). Tests that exercise the
# file logging itself delete this variable and use a tmp_path log dir.
os.environ.setdefault("ORIENTA_NO_FILE_LOG", "1")
# Same idea for the per-phase Hough reflector limits: a test run must
# neither read nor overwrite the developer's own stored preferences.
os.environ.setdefault("ORIENTA_NO_PHASE_LIMIT_STORE", "1")
# Starlette's TestClient sends `Host: testserver` from client address
# ("testclient", 50000); the backend's local-only guard (backend/api/security.py)
# would otherwise refuse every request. Set unconditionally: a developer who
# exported ORIENTA_ALLOWED_HOSTS for LAN use must not see 400s across the
# suite. tests/test_api_origin_guard.py covers the guard itself.
os.environ["ORIENTA_ALLOWED_HOSTS"] = "testserver"

# Ai_Ml/ holds the ebsd_ai package (FEAT-22 ML module); put it on sys.path
# so `import ebsd_ai` resolves from the test suite.
_AI_ML_DIR = Path(__file__).resolve().parents[1] / "Ai_Ml"
if _AI_ML_DIR.is_dir() and str(_AI_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_ML_DIR))

# Project and test data paths
PROJECT_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = PROJECT_ROOT / "Test_data"

# Real test data filenames
H5OINA_FILENAME = (
    "EBSD_SampleB_extrusion_withPattern Sample_B "
    "Arbeitsbereich 1 Elementverteilungsdaten 1.h5oina"
)
EDAX_H5_FILENAME = "HiGainNi.h5"
AL_XTAL_FILENAME = "Al.xtal"
AL_CIF_FILENAME = "Al.cif"
NI_CIF_FILENAME = "Ni.cif"


def _data_file(name: str) -> Path:
    """Return path to a test data file, or None if missing."""
    path = TEST_DATA_DIR / name
    return path if path.exists() else None


def _require_data_file(name: str) -> Path:
    """Return path to a test data file, skip test if missing."""
    path = TEST_DATA_DIR / name
    if not path.exists():
        pytest.skip(f"Test data file not found: {name}")
    return path


# --- Fixtures for real test data (auto-skip when missing) ---

@pytest.fixture
def h5oina_path():
    """Path to the Oxford/Aztec h5oina file (521 MB, has EDS data).

    Structure:
        /1/EBSD/Data/Processed Patterns: (10800, 128, 156) uint8
        /1/EBSD/Data/Pattern Center X/Y/DD: (10800,) float32
        /1/EDS/Data/Window Integral/{Element}/Counts: (10800,) float32
            Elements: Al, Fe, Si, Mn, Cu, Zn, O, C
        /1/EDS/Data/Spectrum/Counts: (10800, 2048) int32
    """
    return _require_data_file(H5OINA_FILENAME)


@pytest.fixture
def edax_h5_path():
    """Path to the EDAX h5 file (101 MB, no EDS data)."""
    return _require_data_file(EDAX_H5_FILENAME)


@pytest.fixture
def al_xtal_path():
    """Path to Al.xtal crystal structure file (EMsoft format)."""
    return _require_data_file(AL_XTAL_FILENAME)


@pytest.fixture
def al_cif_path():
    """Path to Al.cif crystal structure file."""
    return _require_data_file(AL_CIF_FILENAME)


@pytest.fixture
def ni_cif_path():
    """Path to Ni.cif crystal structure file."""
    return _require_data_file(NI_CIF_FILENAME)


@pytest.fixture
def test_data_dir():
    """Path to the Test_data/ directory. Skips if directory missing."""
    if not TEST_DATA_DIR.is_dir():
        pytest.skip("Test_data/ directory not found")
    return TEST_DATA_DIR


# ---------------------------------------------------------------------------
# Database/ is the USER'S data, not a scratch directory
# ---------------------------------------------------------------------------
#
# Found 2026-09-25: ten tests across two files drove the real
# gpu_sim_runner.run_gpu_simulation with the heavy steps mocked but NOT
# write_provenance_sidecar, and the runner takes its output root from a module
# constant pointing at Database/EBSD_SHT_Database. Every run wrote a provenance
# record into the user's own simulated-master library — Database/ is a junction
# inside a worktree — each claiming a source .xtal that never existed. One sat
# beside a genuine EMsoft master from 2026-08-07 until it was found, and it
# also broke a second test, which reads that sidecar. Nobody noticed, because
# nothing said a file had been written.
#
# Two mechanisms, because neither alone is enough:
#   * an interceptor on Python-level opens, which FAILS THE TEST THAT DOES IT
#     and names the path — attribution, at the moment it happens;
#   * a snapshot of Database/ compared at the end of the session, which also
#     catches writes that bypass Python file objects (h5py and torch open
#     files themselves — verified) and directories created without a file.
#
# Measured: a walk of the library's 677 entries costs ~53 ms, so twice per
# session is free, while once per test would be ~330 s across the suite.
# (An earlier comment said 475 entries / 47 ms: that counted files only and
# was taken before this walk included directories.)
#
# Installed at IMPORT rather than from a fixture: a fixture first runs after
# collection, and a module that writes at import time is exactly the shape
# this exists to catch.
#
# Known blind spots, measured rather than assumed: a file written and deleted
# within the same session; a rewrite that restores mtime with os.utime at an
# identical size; anything written after the final snapshot (atexit).

_DB_ROOT = Path(__file__).resolve().parents[1] / "Database"
_DB_WRITES: list = []
_DB_SNAPSHOT: dict = {}
_DB_ALLOWED_NOW = False          # set per test, from the marker
_DB_MARKED_TESTS: list = []
_GUARD_OFF = os.environ.get("ORIENTA_ALLOW_DB_WRITES") == "1"


def _is_under_database(path) -> bool:
    """True when ``path`` lands inside the user's Database/ tree."""
    try:
        text = os.fsdecode(path)   # str, bytes, PathLike; an int fd -> TypeError
    except TypeError:
        return False
    # Fast path: almost every open in the suite ends here. Case-insensitive,
    # because on Windows ...\database\x is the same directory and the exact
    # spelling is not ours to assume.
    if "database" not in text.lower():
        return False
    try:
        Path(text).resolve().relative_to(_DB_ROOT.resolve())
        return True
    except (ValueError, OSError, TypeError):
        return False


def _snapshot_database() -> dict:
    """Every file AND directory under Database/, with mtime and size.

    Directories are included because the code under test does
    ``sht_dir.mkdir(parents=True, exist_ok=True)`` BEFORE it writes the
    sidecar: blocking the write alone still leaves an empty folder behind in
    the user's library.
    """
    out = {}
    if not _DB_ROOT.is_dir():
        return out
    for dirpath, dirnames, filenames in os.walk(_DB_ROOT):
        for name in dirnames:
            out[os.path.join(dirpath, name)] = ("dir", 0)
        for name in filenames:
            p = os.path.join(dirpath, name)
            try:
                st = os.stat(p)
            except OSError:
                continue
            out[p] = (st.st_mtime_ns, st.st_size)
    return out


_REAL_OPEN = builtins.open
_REAL_PATH_OPEN = Path.open
_WRITE_MODES = ("w", "a", "x", "+")


def _check_db_write(path, mode) -> None:
    if _DB_ALLOWED_NOW or not any(c in mode for c in _WRITE_MODES):
        return
    if not _is_under_database(path):
        return
    node = os.environ.get("PYTEST_CURRENT_TEST", "<import or collection>")
    _DB_WRITES.append((node, str(path), mode))
    # pytest.fail raises Failed, which derives from BaseException. That is
    # deliberate and was measured: the code under test wraps its sidecar write
    # in `except Exception` ("provenance is best-effort; never fail the sim"),
    # so an AssertionError here was swallowed — the guard fired, the write was
    # correctly prevented, and the test still passed green with nothing said.
    # A guard that an `except Exception` can swallow is not a guard.
    pytest.fail(
        f"a test opened {path} for writing (mode {mode!r}).\n"
        f"Database/ is the user's phase library, not scratch space — a "
        f"simulated master costs GPU hours, and a provenance sidecar written "
        f"here describes a run that never happened.\n"
        f"Point the code at tmp_path (patch the module's output-dir constant), "
        f"or mark the test @pytest.mark.writes_user_database if it truly must.\n"
        f"Test: {node}",
        pytrace=False,
    )


@functools.wraps(_REAL_OPEN)
def _guarded_open(file, mode="r", *args, **kwargs):
    _check_db_write(file, mode)
    return _REAL_OPEN(file, mode, *args, **kwargs)


@functools.wraps(_REAL_PATH_OPEN)
def _guarded_path_open(self, mode="r", *args, **kwargs):
    _check_db_write(self, mode)
    return _REAL_PATH_OPEN(self, mode, *args, **kwargs)


if not _GUARD_OFF:
    builtins.open = _guarded_open
    io.open = _guarded_open      # a separate name: `import io; io.open(...)`
    Path.open = _guarded_path_open


def pytest_runtest_setup(item):
    """Honour @pytest.mark.writes_user_database for the duration of one test."""
    global _DB_ALLOWED_NOW
    _DB_ALLOWED_NOW = item.get_closest_marker("writes_user_database") is not None
    if _DB_ALLOWED_NOW:
        _DB_MARKED_TESTS.append(item.nodeid)


def pytest_runtest_teardown(item, nextitem):
    global _DB_ALLOWED_NOW
    _DB_ALLOWED_NOW = False


def pytest_sessionfinish(session, exitstatus):
    """Restore the real open(), and report anything that changed in Database/."""
    if _GUARD_OFF:
        return
    builtins.open = _REAL_OPEN
    io.open = _REAL_OPEN
    Path.open = _REAL_PATH_OPEN

    if _DB_WRITES:
        print(f"\n[database guard] blocked {len(_DB_WRITES)} write(s) under "
              f"Database/; first: {_DB_WRITES[0][1]} by {_DB_WRITES[0][0]}")

    if not _DB_SNAPSHOT:
        return
    after = _snapshot_database()
    changed = sorted(
        [p for p, v in after.items() if _DB_SNAPSHOT.get(p) not in (None, v)]
        + [p for p in after if p not in _DB_SNAPSHOT]
        + [p for p in _DB_SNAPSHOT if p not in after]
    )
    if not changed:
        return
    if _DB_MARKED_TESTS:
        print(f"\n[database guard] {len(changed)} entry/entries under Database/ "
              f"changed, and {len(_DB_MARKED_TESTS)} test(s) were marked "
              f"writes_user_database, so this is not a failure: "
              f"{', '.join(_DB_MARKED_TESTS[:3])}")
        return
    listed = "\n  ".join(os.path.relpath(p, _DB_ROOT) for p in changed[:20])
    more = "" if len(changed) <= 20 else f"\n  ... and {len(changed) - 20} more"
    print(f"\n[database guard] THE SESSION CHANGED {len(changed)} ENTRY/ENTRIES "
          f"UNDER Database/ - the user's phase library:\n  {listed}{more}\n"
          f"The per-open guard did not catch it, so it went through a C "
          f"library (h5py, torch), os.replace, or a bare mkdir.")
    session.exitstatus = 1


# --- Register custom markers ---

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (uses real data files, slower)"
    )
    config.addinivalue_line(
        "markers",
        "writes_user_database: test legitimately writes under Database/ - the "
        "user's phase library. Use sparingly and say why; it also exempts the "
        "whole session from the end-of-run comparison."
    )
    # Taken here rather than in a fixture: pytest_configure runs before
    # collection, so a module that writes at import time is still caught.
    if not _GUARD_OFF:
        _DB_SNAPSHOT.update(_snapshot_database())
