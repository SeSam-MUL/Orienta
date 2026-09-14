"""Shared fixtures for batch tests."""
import os
import tempfile
import pytest
import numpy as np
import h5py


@pytest.fixture
def write_h5oina():
    """Return a function that writes a minimal valid H5OINA file.

    The rewritten BatchManager (commit 5ca6ada) self-loads each file via
    safe_loader, so test fixtures must be real H5OINA, not empty placeholder
    files. Usage: ``path = write_h5oina(tmp_path / "a.h5oina")``.
    """
    def _write(path, rows=5, cols=5):
        path = str(path)
        with h5py.File(path, "w") as f:
            hdr = f.create_group("1/EBSD/Header")
            hdr.create_dataset("X Cells", data=cols)
            hdr.create_dataset("Y Cells", data=rows)
            hdr.create_dataset("X Step", data=0.5)
            hdr.create_dataset("Pattern Width", data=10)
            hdr.create_dataset("Pattern Height", data=8)
            data_grp = f.create_group("1/EBSD/Data")
            patterns = np.random.randint(
                0, 255, (rows * cols, 8, 10), dtype=np.uint8
            )
            data_grp.create_dataset("Processed Patterns", data=patterns)
        return str(path)
    return _write


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary SQLite database path."""
    return str(tmp_path / "test_batch.db")


@pytest.fixture
def small_ci_map():
    """Small CI map for checkpoint tests (10x10)."""
    return np.random.rand(10, 10).astype(np.float32)


@pytest.fixture
def small_orientation_map():
    """Small orientation map for checkpoint tests (10x10x3 Euler angles)."""
    return np.random.rand(10, 10, 3).astype(np.float32)


@pytest.fixture
def test_files_dir():
    """Path to batch test data (real H5OINA files)."""
    d = os.path.join(os.path.dirname(__file__), "..", "..", "Test_data", "batch_test")
    if os.path.isdir(d):
        return d
    pytest.skip("Test_data/batch_test not found")


@pytest.fixture
def sht_phases_dir():
    """Path to SHT phase database."""
    d = os.path.join(os.path.dirname(__file__), "..", "..", "Database", "EBSD_SHT_Database")
    if os.path.isdir(d):
        return d
    pytest.skip("Database/EBSD_SHT_Database not found")
