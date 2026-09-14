"""Pillar 3: Memory soak test — detect leaks over 10 batch cycles.

Runs 10 full batch cycles (create -> run -> complete -> cleanup).
Measures process RSS after each cycle. Linear regression slope must be < 2 MB/cycle.
"""
import gc
import os

import h5py
import numpy as np
import psutil
import pytest

from backend.api.services.batch_manager import BatchManager
from backend.api.services.checkpoint_writer import CheckpointWriter
from unittest.mock import patch


def _linear_regression_slope(y_values):
    """Simple linear regression slope."""
    x = np.arange(len(y_values), dtype=np.float64)
    y = np.array(y_values, dtype=np.float64)
    n = len(x)
    slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
    return float(slope)


def _generate_test_files(tmp_dir, n_files=2, rows=5, cols=5):
    files = []
    for i in range(n_files):
        path = os.path.join(tmp_dir, f"soak_file_{i}.h5oina")
        with h5py.File(path, "w") as f:
            hdr = f.create_group("1/EBSD/Header")
            hdr.create_dataset("X Cells", data=cols)
            hdr.create_dataset("Y Cells", data=rows)
            hdr.create_dataset("X Step", data=0.5)
            hdr.create_dataset("Pattern Width", data=10)
            hdr.create_dataset("Pattern Height", data=8)
            data_grp = f.create_group("1/EBSD/Data")
            patterns = np.random.randint(0, 255, (rows * cols, 8, 10), dtype=np.uint8)
            data_grp.create_dataset("Processed Patterns", data=patterns)
        files.append(path)
    return files


def _fake_indexing(*args, **kwargs):
    return (
        np.random.rand(5, 5).astype(np.float32),
        np.random.rand(5, 5, 3).astype(np.float32),
        {"ci_mean": 0.65, "ci_median": 0.67, "duration_sec": 0.05, "phase_file": "/fake"},
    )


def test_memory_soak(tmp_path):
    """10 batch cycles — memory growth slope must be < 2 MB/cycle."""
    soak_dir = str(tmp_path / "soak")
    os.makedirs(soak_dir, exist_ok=True)
    files = _generate_test_files(soak_dir, n_files=2)
    phases = [
        {"name": "PhaseA", "path": "/fake/a.sht", "method": "spherical"},
        {"name": "PhaseB", "path": "/fake/b.sht", "method": "spherical"},
        {"name": "PhaseC", "path": "/fake/c.sht", "method": "spherical"},
    ]

    baselines = []
    proc = psutil.Process(os.getpid())

    for cycle in range(10):
        db_path = os.path.join(soak_dir, f"soak_cycle_{cycle}.db")
        bm = BatchManager(db_path=db_path)
        batch_id, _ = bm.create_batch(files, phases, {})

        with patch("backend.api.services.batch_manager.run_single_indexing_job", side_effect=_fake_indexing):
            bm.run_batch_sync(batch_id)

        # Cleanup checkpoint files
        for f in files:
            cp = CheckpointWriter(f).checkpoint_path
            if os.path.isfile(cp):
                os.remove(cp)

        del bm
        gc.collect()
        gc.collect()

        rss_mb = proc.memory_info().rss / (1024 ** 2)
        baselines.append(rss_mb)

    slope = _linear_regression_slope(baselines)

    print(f"\nMemory soak results:")
    print(f"  Baselines (MB): {[f'{b:.0f}' for b in baselines]}")
    print(f"  Slope: {slope:.2f} MB/cycle")

    assert slope < 2.0, (
        f"Memory leak: slope = {slope:.2f} MB/cycle "
        f"(start={baselines[0]:.0f}, end={baselines[-1]:.0f}). "
        f"All: {[f'{b:.0f}' for b in baselines]}"
    )
