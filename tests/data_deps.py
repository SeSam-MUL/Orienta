"""Where the suite's data lives, and how a test says it needs some.

Measurement files and the crystal library are NOT in the repository: no
``Database/``, no ``Test_data/``, no ``tests/test_spherical_gpu/data/*.h5``.
That is deliberate — they are gigabytes of the maintainer's own scans — and it
means a clone has them only on the machine that made them.

A test that FAILS for their absence is asserting that the machine holds those
files, which no CI runner and no contributor ever will. The first full Linux
run of the suite (GitHub Actions ``35991400714``) produced 55 such failures,
all of them saying nothing about the code. They must SKIP, and the skip has to
name the missing file so the reader knows what to install rather than guessing
whether the test is broken.

Two ways to say it:

    pytestmark = requires(CIF_LIBRARY)          # whole module / class / test
    path = need(CIF_LIBRARY / "Al.cif")         # inside a test or fixture

``requires`` decides at collection time and is right when the whole file is
data-bound; ``need`` decides at call time and is right when one test in a file
of unit tests reaches for a real file. Both name the path that is missing.
"""
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: The crystal library and the simulation caches (gitignored, machine-local).
DATABASE_DIR = PROJECT_ROOT / "Database"
CIF_LIBRARY = DATABASE_DIR / "CIF_Library"
XTAL_LIBRARY = DATABASE_DIR / "XTAL_Library"
SHT_DATABASE = DATABASE_DIR / "EBSD_SHT_Database"
H5_CACHE = DATABASE_DIR / "EBSD_H5_Cache"

#: Measured scans (h5oina, EDAX h5, up1/osc).
TEST_DATA_DIR = PROJECT_ROOT / "Test_data"

#: The spherical indexer's reference oracle. tests/test_spherical_gpu/conftest.py
#: has an ``oracle`` fixture that validates its sha256; prefer that fixture. This
#: constant is for the diagnostics that open the file themselves.
SPHERICAL_ORACLE = (PROJECT_ROOT / "tests" / "test_spherical_gpu" / "data"
                    / "reference_oracle.h5")

_WHY = ("measurement data and the crystal library are not in the repository; "
        "this checkout has no copy of it")


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def first_missing(*paths):
    """The first of ``paths`` that is not on disk, or None if all are."""
    for p in paths:
        if not Path(p).exists():
            return Path(p)
    return None


def need(*paths):
    """Skip the running test unless every path exists; return the last one.

    Use inside a test or fixture:

        cif = need(CIF_LIBRARY / "Al.cif")
    """
    gone = first_missing(*paths)
    if gone is not None:
        pytest.skip(f"needs {_rel(gone)} — {_WHY}")
    return Path(paths[-1]) if paths else None


def requires(*paths):
    """A skipif marker for a module, class or test that needs these paths.

        pytestmark = requires(CIF_LIBRARY)

    Evaluated once at collection, so the reason names the file that was
    missing at that moment.
    """
    gone = first_missing(*paths)
    return pytest.mark.skipif(
        gone is not None,
        reason=(f"needs {_rel(gone)} — {_WHY}" if gone is not None
                else "data present"),
    )
