"""
FEAT-15 Task 6: ICD Roundtrip Test

Tests the OpenCL ICD file break/fix/verify cycle:
  1. Validate detects broken ICD files
  2. fix_icd_files() repairs them
  3. clinfo fails with broken ICD
  4. clinfo works after fix
  5. Full roundtrip: break → fix → simulate → verify

CRITICAL: autouse fixture ALWAYS restores original ICD files, even on crash.
Requires: sudo (NOPASSWD configured), /etc/OpenCL/vendors/*.icd.
"""
import os
import subprocess
from pathlib import Path

import pytest

# Import from our detector module
from simulation.opencl_detector import (
    detect_opencl,
    fix_icd_files,
    validate_icd_files,
)

# ---------------------------------------------------------------------------
# Constants — override via environment variables for portability
# ---------------------------------------------------------------------------
ICD_DIR = Path("/etc/OpenCL/vendors")
EMSOFT_BIN = Path(os.environ.get("EMSOFT_BIN", str(Path.home() / "emsoft/builds/EMsoft-Release/Bin")))
EMSOFT_DATA = Path(os.environ.get("EMSOFT_DATA", str(Path.home() / "EMsoftData")))
TEST_DIR_NAME = "TestICD"
TEST_DIR = EMSOFT_DATA / TEST_DIR_NAME

# Skip if ICD directory doesn't exist
if not ICD_DIR.exists():
    pytest.skip("OpenCL ICD directory not found", allow_module_level=True)

# Skip if no ICD files
_icd_files = list(ICD_DIR.glob("*.icd"))
if not _icd_files:
    pytest.skip("No ICD files found in /etc/OpenCL/vendors/", allow_module_level=True)


# ---------------------------------------------------------------------------
# Safety fixture: ALWAYS restore ICD files
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def restore_icd_files():
    """Backup and ALWAYS restore ICD files, even on test crash."""
    originals = {}
    for icd in ICD_DIR.glob("*.icd"):
        try:
            originals[str(icd)] = icd.read_text().strip()
        except PermissionError:
            pass

    if not originals:
        pytest.skip("Could not read ICD files")

    yield originals

    # ALWAYS restore - this runs even if test crashes
    for path, content in originals.items():
        subprocess.run(
            ["sudo", "tee", path],
            input=content.encode(),
            capture_output=True,
            timeout=10,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _break_icd_files():
    """Write garbage to all ICD files."""
    for icd in ICD_DIR.glob("*.icd"):
        subprocess.run(
            ["sudo", "tee", str(icd)],
            # Any non-path value breaks the file; deliberately NOT the value
            # the old sudo-wrapper bug wrote here, which was a real password.
            input=b"not-a-library-path",
            capture_output=True,
            timeout=10,
        )


def _get_clinfo_platform_count() -> int:
    """Run clinfo and count platforms found."""
    try:
        result = subprocess.run(
            ["clinfo"],
            capture_output=True, text=True, timeout=15,
        )
        output = result.stdout
        # Look for "Number of platforms" line
        for line in output.split("\n"):
            if "Number of platforms" in line:
                parts = line.strip().split()
                return int(parts[-1])
        # Fallback: count "Platform Name" occurrences
        return output.count("Platform Name")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return -1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestIcdValidation:
    """Tests for ICD file validation and repair."""

    def test_validate_detects_broken_icd(self, restore_icd_files):
        """Breaking ICD files should make validate_icd_files report invalid."""
        _break_icd_files()

        results = validate_icd_files()
        assert len(results) > 0, "No ICD files found by validator"

        for entry in results:
            assert entry["valid"] is False, (
                f"ICD file {entry['name']} should be invalid after breaking, "
                f"content: {entry['content']!r}"
            )

    def test_fix_repairs_broken_icd(self, restore_icd_files):
        """fix_icd_files should repair broken ICD files."""
        _break_icd_files()

        # Verify broken
        broken = validate_icd_files()
        assert all(not e["valid"] for e in broken), "ICD files should be broken"

        # Fix
        success, messages = fix_icd_files()
        assert success, f"fix_icd_files failed: {messages}"

        # Verify fixed
        fixed = validate_icd_files()
        valid_count = sum(1 for e in fixed if e["valid"])
        assert valid_count > 0, (
            f"No ICD files valid after fix. Messages: {messages}"
        )


class TestClinfoIntegration:
    """Tests that clinfo actually fails/works with broken/fixed ICD."""

    def test_clinfo_fails_with_broken_icd(self, restore_icd_files):
        """clinfo should find 0 platforms when ICD files are broken."""
        _break_icd_files()

        platforms = _get_clinfo_platform_count()
        assert platforms <= 0, (
            f"clinfo found {platforms} platforms with broken ICD files. "
            "ICD breakage did not take effect."
        )

    def test_clinfo_works_after_fix(self, restore_icd_files):
        """After fix, clinfo should find at least 1 platform."""
        _break_icd_files()

        # Verify broken
        platforms_broken = _get_clinfo_platform_count()
        assert platforms_broken <= 0, "ICD should be broken before fix"

        # Fix
        success, messages = fix_icd_files()
        assert success, f"fix_icd_files failed: {messages}"

        # Verify working
        platforms_fixed = _get_clinfo_platform_count()
        assert platforms_fixed >= 1, (
            f"clinfo found {platforms_fixed} platforms after fix. "
            f"Fix messages: {messages}"
        )


class TestFullRoundtrip:
    """Full roundtrip: break → fix → simulate → verify."""

    def test_full_roundtrip_with_simulation(self, restore_icd_files):
        """Break ICD, fix, then run a real EMMCOpenCL simulation."""
        emmcopencl = EMSOFT_BIN / "EMMCOpenCL"
        if not emmcopencl.exists():
            pytest.skip("EMMCOpenCL binary not found")

        xtal_file = Path(os.environ.get("EMSOFT_XTAL", str(Path.home() / "EMsoftXtal"))) / "Al.xtal"
        if not xtal_file.exists():
            pytest.skip("Al.xtal not found")

        # Setup output directory
        TEST_DIR.mkdir(parents=True, exist_ok=True)
        output_h5 = TEST_DIR / "Al_icd_test.h5"
        output_h5.unlink(missing_ok=True)

        # 1. Break ICD files
        _break_icd_files()
        assert _get_clinfo_platform_count() <= 0, "ICD should be broken"

        # 2. Fix ICD files
        success, messages = fix_icd_files()
        assert success, f"fix_icd_files failed: {messages}"
        assert _get_clinfo_platform_count() >= 1, "clinfo should work after fix"

        # 3. Run EMMCOpenCL with 1M electrons (very fast)
        nml_path = TEST_DIR / "EMMCOpenCL_icd_test.nml"
        nml_path.write_text(
            " &MCCLdata\n"
            "  mode = 'full',\n"
            "  xtalname = 'Al.xtal',\n"
            "  numsx = 101,\n"
            "  sig = 70.0,\n"
            "  omega = 0.0,\n"
            "  num_el = 10,\n"
            "  platid = 1,\n"
            "  devid = 1,\n"
            "  globalworkgrpsz = 150,\n"
            "  totnum_el = 1000000,\n"
            "  multiplier = 1,\n"
            "  EkeV = 20.0D0,\n"
            "  Ehistmin = 10.0D0,\n"
            "  Ebinsize = 1.0D0,\n"
            "  depthmax = 100.0D0,\n"
            "  depthstep = 1.0D0,\n"
            f"  dataname = '{TEST_DIR_NAME}/Al_icd_test.h5',\n"
            "  Notify = 'off',\n"
            " /\n"
        )

        result = subprocess.run(
            ["bash", "-lc", f"{emmcopencl} {nml_path}"],
            capture_output=True, text=True, timeout=60,
        )

        # 4. Verify simulation succeeded
        assert result.returncode == 0, (
            f"EMMCOpenCL failed after ICD fix (rc={result.returncode}):\n"
            f"stdout: {result.stdout[-300:]}\n"
            f"stderr: {result.stderr[-300:]}"
        )
        assert output_h5.exists(), f"Output file not created: {output_h5}"
        assert output_h5.stat().st_size > 500, "Output file suspiciously small"

        # Cleanup
        nml_path.unlink(missing_ok=True)
        output_h5.unlink(missing_ok=True)
