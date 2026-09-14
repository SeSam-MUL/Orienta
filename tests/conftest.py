"""
Shared pytest fixtures and markers for Kikuchipy GUI tests.

Provides fixtures for real test data files in Test_data/.
Integration tests using these fixtures are auto-skipped when files are missing.
"""

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


# --- Register custom markers ---

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (uses real data files, slower)"
    )
